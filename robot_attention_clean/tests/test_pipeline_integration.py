"""End-to-end integration tests: perception-like observations -> tracker
output -> filter -> policy update -> target persistence -> target
selection, using the REAL IOUTracker, the REAL filters, and the REAL
policies (nothing mocked) — the exact chain main_webcam.py drives, minus
only the camera/MediaPipe/GazeEstimator call itself (which perception.py
owns and which is exercised separately in tests/test_perception.py).

`tick()` below mirrors main_webcam.py's per-frame orchestration exactly:
bbox detections -> tracker.update() for stable ids -> per-id filter.update()
-> policy_confidence = raw_confidence (Version 1, see DEBUG_ATTENTION.md
FM1) -> seeking_attention recomputed on the filtered signal -> policy.update().

Every test builds a full timeline of (timestamp, target, ids[, ...]) and,
on failure, dumps it via `_fmt()` so a failure reports the exact scenario
and the full expected-vs-actual target timeline, not just a bare assert.
"""

from __future__ import annotations

from typing import List, Optional

import pytest

import gaze_attention as ga
from gaze_attention.filters import make_filter
from gaze_attention.policies import make_policy
from gaze_attention.tracker import IOUTracker

DT = 0.033


# ---------------------------------------------------------------------------
# Real-pipeline harness (tracker + filter + policy, no mocks).
# ---------------------------------------------------------------------------
def make_pipeline(filter_name="moving_average", policy_name="confidence_debt", config_overrides=None):
    config = dict(ga.DEFAULT_CONFIG)
    if config_overrides:
        config.update(config_overrides)

    tracker = IOUTracker(iou_threshold=config["iou_threshold"], max_missed_frames=config["max_missed_frames"])
    filt = make_filter(filter_name)

    policy_kwargs = {
        "confidence_threshold": config["confidence_threshold"],
        "minimum_hold_seconds": config["minimum_hold_seconds"],
        "target_grace_seconds": config["target_grace_seconds"],
        "engagement_decay": config["engagement_decay"],
    }
    if policy_name == "confidence_debt":
        policy_kwargs["debt_weight"] = config["debt_weight"]
        policy_kwargs["engagement_max"] = config["engagement_max"]
        policy_kwargs["debt_max"] = config["debt_max"]
    policy = make_policy(policy_name, **policy_kwargs)

    return tracker, filt, policy, config


def tick(tracker, filt, policy, config, detections, timestamp, raw_dt, known_ids):
    """One tick of the real pipeline. `detections` is a list of
    {"bbox": [x1,y1,x2,y2], "raw_gaze_signal": float, "raw_confidence": float}
    -- perception-like raw observations, pre-tracker/pre-filter. Returns
    (target, ids, policy_observations, used_dt)."""
    used_dt, _raw_dt = ga.sanitize_dt(
        raw_dt, fallback=config["dt_fallback_seconds"], max_dt=config["dt_max_seconds"]
    )

    bboxes = [d["bbox"] for d in detections]
    ids = tracker.update(bboxes)

    current_ids = set(ids)
    for gone_id in known_ids - current_ids:
        filt.forget(gone_id)
    known_ids.clear()
    known_ids.update(current_ids)

    policy_observations = []
    for pid, det in zip(ids, detections):
        raw_confidence = det["raw_confidence"]
        filtered_signal, filter_uncertainty = filt.update(pid, det["raw_gaze_signal"], raw_confidence, used_dt)
        policy_confidence = raw_confidence  # Version 1: identity (DEBUG_ATTENTION.md FM1)
        seeking = filtered_signal >= config["seeking_threshold"]
        policy_observations.append({
            "id": pid,
            "bbox": det["bbox"],
            "gaze_signal": filtered_signal,
            "confidence": policy_confidence,
            "seeking_attention": seeking,
            "timestamp": timestamp,
        })

    target = policy.update(policy_observations, used_dt)
    return target, ids, policy_observations, used_dt


def _fmt(timeline: List[tuple], msg: str) -> str:
    lines = [msg, "timeline (t, target, ids[, extra]):"]
    for entry in timeline:
        t, target, ids, *rest = entry
        line = f"  t={t:.3f} target={target} ids={ids}"
        if rest:
            line += f" extra={rest}"
        lines.append(line)
    return "\n".join(lines)


def det(bbox, gaze_signal=0.9, confidence=0.9):
    return {"bbox": bbox, "raw_gaze_signal": gaze_signal, "raw_confidence": confidence}


BBOX_A = [0, 0, 50, 50]
BBOX_B = [200, 200, 250, 250]


# ---------------------------------------------------------------------------
# 1. One-frame missed detection.
# ---------------------------------------------------------------------------
def test_one_frame_missed_detection():
    tracker, filt, policy, config = make_pipeline()
    known_ids: set = set()
    t = 0.0
    timeline = []

    # Hold well past minimum_hold_seconds (0.75s default -> ~23 ticks).
    for _ in range(30):
        target, ids, _, _ = tick(tracker, filt, policy, config, [det(BBOX_A)], t, DT, known_ids)
        timeline.append((t, target, ids))
        t += DT
    held_target = timeline[-1][1]
    assert held_target is not None

    # One frame with zero detections -- a MediaPipe blip perception.py
    # would emit as an empty observation list for this person.
    target, ids, _, _ = tick(tracker, filt, policy, config, [], t, DT, known_ids)
    timeline.append((t, target, ids))
    t += DT
    assert target == held_target, _fmt(timeline, "expected target to survive a single missed-detection frame")

    # Person reappears in the same location -- tracker should re-match the
    # same id (within max_missed_frames), and the target should still hold.
    target, ids, _, _ = tick(tracker, filt, policy, config, [det(BBOX_A)], t, DT, known_ids)
    timeline.append((t, target, ids))
    assert ids[0] == held_target, _fmt(timeline, "expected the tracker to reassign the same id on reappearance")
    assert target == held_target, _fmt(timeline, "expected target to still hold after reappearance")


# ---------------------------------------------------------------------------
# 2. Low raw confidence under model-based (Bayesian) filtering.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("filter_name", ["moving_average", "bayesian"])
def test_low_raw_confidence_blocks_debt_gate(filter_name):
    # Pin A as target so B is unambiguously "overlooked" for the whole run.
    tracker, filt, policy, config = make_pipeline(
        filter_name=filter_name, policy_name="confidence_debt",
        config_overrides={"minimum_hold_seconds": 100.0, "target_grace_seconds": 100.0},
    )
    known_ids: set = set()
    t = 0.0
    timeline = []

    for _ in range(120):  # far past filter convergence
        target, ids, _, _ = tick(
            tracker, filt, policy, config,
            [det(BBOX_A, confidence=0.9), det(BBOX_B, confidence=0.10)],  # B: low raw confidence
            t, DT, known_ids,
        )
        timeline.append((t, target, ids))
        t += DT

    id_a, id_b = timeline[-1][2]
    assert target == id_a, _fmt(timeline, "expected the high-confidence person to hold the target")
    assert policy.get_metrics(id_b)["debt"] == 0.0, _fmt(
        timeline,
        f"expected low raw confidence to never clear the debt gate, even after the {filter_name} filter converges",
    )


# ---------------------------------------------------------------------------
# 3. A delayed frame causing large dt.
# ---------------------------------------------------------------------------
def test_delayed_frame_large_dt_does_not_spike_score():
    tracker, filt, policy, config = make_pipeline()
    known_ids: set = set()
    t = 0.0
    timeline = []

    for _ in range(10):
        target, ids, _, used_dt = tick(tracker, filt, policy, config, [det(BBOX_A)], t, DT, known_ids)
        timeline.append((t, target, ids, used_dt))
        t += DT
    pid = timeline[-1][2][0]
    engagement_before = policy.get_metrics(pid)["engagement"]

    # Simulate a paused window / delayed inference: a 9-second gap.
    target, ids, _, used_dt = tick(tracker, filt, policy, config, [det(BBOX_A)], t, 9.0, known_ids)
    timeline.append((t, target, ids, used_dt))
    engagement_after = policy.get_metrics(pid)["engagement"]

    assert used_dt <= config["dt_max_seconds"], _fmt(timeline, "expected the huge raw dt to be clamped")
    max_single_tick_gain = 1.0 * config["dt_max_seconds"]  # confidence<=1.0 * dt_max_seconds
    assert engagement_after - engagement_before <= max_single_tick_gain + 1e-9, _fmt(
        timeline, "expected a delayed frame to not create a disproportionate score jump"
    )
    assert target == pid, _fmt(timeline, "expected the pipeline to remain sane (no crash/None) after the delay")


# ---------------------------------------------------------------------------
# 4. A persistent target and intermittent challenger.
# ---------------------------------------------------------------------------
def test_persistent_target_beats_intermittent_challenger():
    tracker, filt, policy, config = make_pipeline()
    known_ids: set = set()
    t = 0.0
    timeline = []

    for i in range(300):
        intermittent_on = (i // 10) % 2 == 0
        detections = [
            det(BBOX_A, gaze_signal=0.9),  # persistent
            det(BBOX_B, gaze_signal=0.9 if intermittent_on else 0.1),  # intermittent
        ]
        target, ids, _, _ = tick(tracker, filt, policy, config, detections, t, DT, known_ids)
        timeline.append((t, target, ids))
        t += DT

    # Frame 0's tracker id can be None (TENTATIVE, not yet confirmed) --
    # take the id from the first tick it's actually been assigned.
    persistent_id = next(entry[2][0] for entry in timeline if entry[2][0] is not None)
    assert timeline[-1][1] == persistent_id, _fmt(
        timeline, "expected the persistent seeker to hold the target against an intermittent challenger"
    )


# ---------------------------------------------------------------------------
# 5. Temporary disappearance and reappearance of a person.
# ---------------------------------------------------------------------------
def test_temporary_disappearance_and_reappearance():
    tracker, filt, policy, config = make_pipeline()
    known_ids: set = set()
    t = 0.0
    timeline = []

    for _ in range(30):
        target, ids, _, _ = tick(tracker, filt, policy, config, [det(BBOX_A)], t, DT, known_ids)
        timeline.append((t, target, ids))
        t += DT
    original_id = timeline[-1][2][0]
    assert target == original_id

    # Disappear for several frames -- comfortably within both
    # max_missed_frames (15, tracker-side) and target_grace_seconds
    # (0.5s, policy-side).
    for _ in range(5):
        target, ids, _, _ = tick(tracker, filt, policy, config, [], t, DT, known_ids)
        timeline.append((t, target, ids))
        t += DT
    assert target == original_id, _fmt(
        timeline, "expected the target to be preserved during a brief disappearance within grace"
    )

    target, ids, _, _ = tick(tracker, filt, policy, config, [det(BBOX_A)], t, DT, known_ids)
    timeline.append((t, target, ids))
    assert ids[0] == original_id, _fmt(timeline, "expected the tracker to reassign the same id on reappearance")
    assert target == original_id, _fmt(timeline, "expected the target to still hold after reappearance")


# ---------------------------------------------------------------------------
# 6. A current target that should remain selected during its hold period.
# ---------------------------------------------------------------------------
def test_target_remains_selected_during_hold_period():
    tracker, filt, policy, config = make_pipeline(
        config_overrides={"minimum_hold_seconds": 5.0, "target_grace_seconds": 0.1}
    )
    known_ids: set = set()
    t = 0.0
    timeline = []

    for _ in range(5):
        target, ids, _, _ = tick(
            tracker, filt, policy, config,
            [det(BBOX_A, gaze_signal=0.9), det(BBOX_B, gaze_signal=0.1)],
            t, DT, known_ids,
        )
        timeline.append((t, target, ids))
        t += DT
    id_a, id_b = timeline[-1][2]
    assert target == id_a

    # B now outscores A on paper (A stops seeking, B seeks strongly), but
    # well under the 5s hold -- A must remain selected regardless.
    for _ in range(60):  # ~2s
        target, ids, _, _ = tick(
            tracker, filt, policy, config,
            [det(BBOX_A, gaze_signal=0.1), det(BBOX_B, gaze_signal=0.9)],
            t, DT, known_ids,
        )
        timeline.append((t, target, ids))
        t += DT

    assert target == id_a, _fmt(
        timeline, "expected the target to remain selected through its hold period despite a higher-scoring challenger"
    )


# ---------------------------------------------------------------------------
# 7. A genuinely sustained challenger that should eventually trigger a switch.
# ---------------------------------------------------------------------------
def test_sustained_challenger_eventually_triggers_switch():
    tracker, filt, policy, config = make_pipeline(
        config_overrides={"minimum_hold_seconds": 0.2, "target_grace_seconds": 0.1}
    )
    known_ids: set = set()
    t = 0.0
    timeline = []

    for _ in range(5):
        target, ids, _, _ = tick(
            tracker, filt, policy, config,
            [det(BBOX_A, gaze_signal=0.9), det(BBOX_B, gaze_signal=0.1)],
            t, DT, known_ids,
        )
        timeline.append((t, target, ids))
        t += DT
    id_a, id_b = timeline[-1][2]
    assert target == id_a

    # B seeks persistently and confidently while ignored, and A stops
    # seeking (decays) -- B's debt should legitimately overtake A's frozen
    # engagement eventually, and the switch should stick.
    for _ in range(500):
        target, ids, _, _ = tick(
            tracker, filt, policy, config,
            [det(BBOX_A, gaze_signal=0.1), det(BBOX_B, gaze_signal=0.9)],
            t, DT, known_ids,
        )
        timeline.append((t, target, ids))
        t += DT

    assert target == id_b, _fmt(
        timeline, "expected a genuinely sustained, confident challenger to eventually win the target"
    )
    # And it should stick, not flicker back, for the remaining tail.
    tail_targets = [entry[1] for entry in timeline[-20:]]
    assert all(tt == id_b for tt in tail_targets), _fmt(
        timeline, "expected the switch to B to stick, not flicker"
    )


# ---------------------------------------------------------------------------
# Cross-check: same scenarios hold for GreedyDwellPolicy too (no mocking of
# policy logic means both real policies get exercised through the same
# real tracker/filter chain).
# ---------------------------------------------------------------------------
def test_one_frame_missed_detection_greedy_policy():
    tracker, filt, policy, config = make_pipeline(policy_name="greedy")
    known_ids: set = set()
    t = 0.0
    timeline = []
    for _ in range(30):
        target, ids, _, _ = tick(tracker, filt, policy, config, [det(BBOX_A)], t, DT, known_ids)
        timeline.append((t, target, ids))
        t += DT
    held_target = timeline[-1][1]

    target, ids, _, _ = tick(tracker, filt, policy, config, [], t, DT, known_ids)
    timeline.append((t, target, ids))
    assert target == held_target, _fmt(timeline, "GreedyDwellPolicy: expected target to survive a single missed frame")


def test_persistent_target_beats_intermittent_challenger_greedy_policy():
    tracker, filt, policy, config = make_pipeline(policy_name="greedy")
    known_ids: set = set()
    t = 0.0
    timeline = []
    for i in range(300):
        intermittent_on = (i // 10) % 2 == 0
        detections = [
            det(BBOX_A, gaze_signal=0.9),
            det(BBOX_B, gaze_signal=0.9 if intermittent_on else 0.1),
        ]
        target, ids, _, _ = tick(tracker, filt, policy, config, detections, t, DT, known_ids)
        timeline.append((t, target, ids))
        t += DT
    persistent_id = next(entry[2][0] for entry in timeline if entry[2][0] is not None)
    assert timeline[-1][1] == persistent_id, _fmt(
        timeline, "GreedyDwellPolicy: expected the persistent seeker to hold the target"
    )
