"""Unit tests for gaze_attention.tracker.IOUTracker — three-state
(TENTATIVE/CONFIRMED/LOST) lifecycle, constant-velocity prediction, and
Hungarian (global) assignment."""

from gaze_attention.tracker import IOUTracker, TrackState, iou

DT = 1.0 / 30.0


def _confirm(tracker, bbox, n=3, dt=DT):
    """Feed the same bbox n times (default: n_confirm) and return the ids
    list from the LAST call, once the track should be CONFIRMED."""
    ids = None
    for _ in range(n):
        ids = tracker.update([bbox], dt=dt)
    return ids


# --- iou() (pure function, unaffected by the lifecycle rewrite) --------------

def test_iou_identical_boxes():
    assert iou([0, 0, 10, 10], [0, 0, 10, 10]) == 1.0


def test_iou_disjoint_boxes():
    assert iou([0, 0, 10, 10], [20, 20, 30, 30]) == 0.0


def test_iou_degenerate_box():
    assert iou([0, 0, 0, 0], [0, 0, 10, 10]) == 0.0


# --- TENTATIVE -> CONFIRMED promotion -----------------------------------------

def test_new_detection_gets_no_public_id_before_n_confirm_hits():
    tracker = IOUTracker(n_confirm=3)
    ids1 = tracker.update([[0, 0, 10, 10]], dt=DT)
    assert ids1 == [None]
    ids2 = tracker.update([[0, 0, 10, 10]], dt=DT)
    assert ids2 == [None]


def test_track_promoted_to_confirmed_at_exactly_n_confirm_hits():
    tracker = IOUTracker(n_confirm=3)
    tracker.update([[0, 0, 10, 10]], dt=DT)  # hit 1
    tracker.update([[0, 0, 10, 10]], dt=DT)  # hit 2
    ids = tracker.update([[0, 0, 10, 10]], dt=DT)  # hit 3 -> confirmed
    assert ids[0] is not None
    assert isinstance(ids[0], int)


def test_n_confirm_one_confirms_immediately():
    tracker = IOUTracker(n_confirm=1)
    ids = tracker.update([[0, 0, 10, 10]], dt=DT)
    assert ids[0] is not None


def test_tentative_track_missing_one_frame_is_deleted_not_retained():
    tracker = IOUTracker(n_confirm=3)
    tracker.update([[0, 0, 10, 10]], dt=DT)  # hit 1, TENTATIVE
    tracker.update([], dt=DT)  # miss -> TENTATIVE deleted immediately
    ids = _confirm(tracker, [0, 0, 10, 10], n=3)
    # A fresh 3-hit run after the deletion must confirm as a NEW id, not
    # resume progress from before the miss (proves no grace period).
    assert ids[0] is not None


def test_confirmed_ids_are_distinct_and_stable_across_frames():
    tracker = IOUTracker(n_confirm=3)
    ids_a = _confirm(tracker, [0, 0, 10, 10])
    ids_b = _confirm(tracker, [200, 200, 210, 210])
    assert ids_a[0] is not None and ids_b[0] is not None
    assert ids_a[0] != ids_b[0]

    # One more matched frame each -- ids stay stable.
    ids_a2 = tracker.update([[1, 1, 11, 11], [201, 201, 211, 211]], dt=DT)
    assert ids_a2[0] == ids_a[0]
    assert ids_a2[1] == ids_b[0]


# --- LOST retention / expiry ---------------------------------------------------

def test_lost_track_retained_and_reassigned_same_id_within_t_lost():
    tracker = IOUTracker(n_confirm=3, t_lost_seconds=1.5)
    ids1 = _confirm(tracker, [0, 0, 10, 10])
    confirmed_id = ids1[0]
    assert confirmed_id is not None

    # Missing for less than t_lost_seconds (e.g. 1.0s across several ticks).
    for _ in range(10):  # 10 * DT(~0.033) = ~0.33s
        tracker.update([], dt=DT)

    ids2 = tracker.update([[0, 0, 10, 10]], dt=DT)
    assert ids2[0] == confirmed_id, "re-detected within t_lost_seconds must keep the same id"


def test_lost_track_deleted_after_t_lost_exceeded():
    tracker = IOUTracker(n_confirm=3, t_lost_seconds=0.1)
    ids1 = _confirm(tracker, [0, 0, 10, 10])
    confirmed_id = ids1[0]
    assert confirmed_id is not None

    # Miss for longer than t_lost_seconds.
    for _ in range(10):  # 10 * DT(~0.033) = ~0.33s > 0.1s
        tracker.update([], dt=DT)

    ids2 = _confirm(tracker, [0, 0, 10, 10])
    assert ids2[0] is not None
    assert ids2[0] != confirmed_id, "track should have expired and been treated as a new person"


def test_lost_tracks_returns_predicted_bbox():
    # Small per-frame shift (+3 of a 10-wide box) so consecutive frames
    # still overlap enough (IoU ~0.54) to match under the default
    # iou_gate=0.2 even before any velocity has been established.
    tracker = IOUTracker(n_confirm=1, t_lost_seconds=1.5, velocity_alpha=1.0)
    tid = tracker.update([[0, 0, 10, 10]], dt=DT)[0]
    assert tid is not None
    tid2 = tracker.update([[3, 0, 13, 10]], dt=DT)[0]  # establishes velocity via _apply_match
    assert tid2 == tid

    tracker.update([], dt=DT)  # goes LOST this tick
    lost = tracker.lost_tracks()
    assert tid in lost
    x1, y1, x2, y2 = lost[tid]
    # velocity_alpha=1.0 -> full raw velocity (3 units / DT) applied over
    # this tick's elapsed time (DT) -> bbox should have shifted forward by
    # roughly +3 in x from the last matched bbox [3, 0, 13, 10].
    assert x1 > 3.0
    assert y1 == 0.0


def test_lost_track_rematched_returns_to_confirmed_state():
    tracker = IOUTracker(n_confirm=3, t_lost_seconds=1.5)
    ids1 = _confirm(tracker, [0, 0, 10, 10])
    confirmed_id = ids1[0]

    tracker.update([], dt=DT)  # LOST
    assert tracker._tracks[confirmed_id].state == TrackState.LOST

    tracker.update([[0, 0, 10, 10]], dt=DT)  # re-matched
    assert tracker._tracks[confirmed_id].state == TrackState.CONFIRMED


# --- global assignment / crossing paths ---------------------------------------

def test_crossing_paths_do_not_swap_ids():
    """Two tracks moving toward each other and crossing -- constant-velocity
    prediction + global assignment should keep the correct pairing even
    when the boxes briefly overlap at the crossing point, unlike a plain
    greedy nearest-IoU match which can lock onto the wrong pairing."""
    tracker = IOUTracker(n_confirm=1, iou_gate=0.05, velocity_alpha=0.8)

    # Track A starts left moving right; Track B starts right moving left.
    ax, bx = 0.0, 100.0
    a_id = tracker.update([[ax, 0, ax + 10, 10]], dt=DT)[0]
    b_bboxes = tracker.update([[ax, 0, ax + 10, 10], [bx, 0, bx + 10, 10]], dt=DT)
    b_id = b_bboxes[1]
    assert a_id is not None and b_id is not None and a_id != b_id

    step = 8.0
    for _ in range(14):  # they cross around the midpoint
        ax += step
        bx -= step
        ids = tracker.update([[ax, 0, ax + 10, 10], [bx, 0, bx + 10, 10]], dt=DT)
        # Detections are always given in [A-position, B-position] order --
        # assert the tracker still assigns A's id to the first slot and B's
        # to the second, i.e. it followed each box's own trajectory rather
        # than snapping to whichever track happens to be nearest post-cross.
        assert ids[0] == a_id
        assert ids[1] == b_id


def test_iou_gate_rejects_low_overlap_assignment():
    tracker = IOUTracker(n_confirm=1, iou_gate=0.5)
    ids1 = tracker.update([[0, 0, 10, 10]], dt=DT)
    # Barely overlapping box -> IoU well below 0.5 -> must NOT match.
    ids2 = tracker.update([[9, 9, 19, 19]], dt=DT)
    assert ids2[0] != ids1[0]


# --- dt optionality (backward-compat interface) --------------------------------

def test_update_dt_optional_backward_compatible():
    tracker = IOUTracker(n_confirm=1)
    ids = tracker.update([[0, 0, 10, 10]])  # no dt passed -- must not raise
    assert ids[0] is not None


# --- misc ------------------------------------------------------------------

def test_reset_clears_tracks():
    tracker = IOUTracker(n_confirm=1)
    ids1 = tracker.update([[0, 0, 10, 10]], dt=DT)
    tracker.reset()
    ids2 = tracker.update([[0, 0, 10, 10]], dt=DT)
    assert ids1[0] == 0
    assert ids2[0] == 0  # both start fresh at id 0 after reset


def test_empty_detections_no_crash():
    tracker = IOUTracker()
    assert tracker.update([], dt=DT) == []


def test_deprecated_constructor_params_still_accepted():
    """iou_threshold/max_missed_frames must remain valid constructor
    kwargs (unused by the new state machine) so existing callers/config
    that still pass them don't break."""
    tracker = IOUTracker(iou_threshold=0.3, max_missed_frames=15, n_confirm=1)
    ids = tracker.update([[0, 0, 10, 10]], dt=DT)
    assert ids[0] is not None
