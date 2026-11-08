"""Unit tests for perception.py's pure conversion logic — no camera, no
MediaPipe. Exercises the documented gaze_signal conversion and the fallback
heuristic directly, plus a couple of smoke tests through the real Perception
class on synthetic (blank) frames.
"""

import math
from types import SimpleNamespace

import numpy as np
import pytest

from gaze_attention.perception import (
    _face_bbox,
    _fallback_signal,
    _gaze_result_to_signal,
    _mouth_opening_from_landmarks,
)


def _landmark(x, y):
    return SimpleNamespace(x=x, y=y)


def _gaze_result(gaze_h=0.0, gaze_v=0.0, confidence=0.9, is_valid=True):
    return SimpleNamespace(gaze_h=gaze_h, gaze_v=gaze_v, confidence=confidence, is_valid=is_valid)


# --- gaze_signal conversion ---------------------------------------------------

def test_centered_gaze_gives_max_signal():
    signal, conf = _gaze_result_to_signal(_gaze_result(0.0, 0.0), last_signal=0.0)
    assert signal == pytest.approx(1.0)


def test_signal_crosses_seeking_threshold_at_eye_contact_margin():
    # By construction (see perception.py docstring): gaze_signal == 0.6
    # exactly at horizontal offset == 0.18, the GazeEstimator eye-contact
    # margin, for a purely horizontal look-away (gaze_v == 0).
    signal, _ = _gaze_result_to_signal(_gaze_result(gaze_h=0.18, gaze_v=0.0), last_signal=0.0)
    assert signal == pytest.approx(0.6, abs=1e-6)


def test_large_offset_clamps_to_zero():
    signal, _ = _gaze_result_to_signal(_gaze_result(gaze_h=0.5, gaze_v=0.5), last_signal=0.0)
    assert signal == 0.0


def test_confidence_passes_through_unchanged():
    _, conf = _gaze_result_to_signal(_gaze_result(confidence=0.73), last_signal=0.0)
    assert conf == pytest.approx(0.73)


def test_invalid_frame_holds_last_signal_not_zero_offset():
    # Blink: GazeEstimator stubs gaze_h=gaze_v=0.0, which would look like
    # perfect eye contact if converted naively. Must instead freeze at the
    # last valid signal.
    signal, conf = _gaze_result_to_signal(
        _gaze_result(gaze_h=0.0, gaze_v=0.0, confidence=0.10, is_valid=False),
        last_signal=0.2,
    )
    assert signal == 0.2
    assert conf == pytest.approx(0.10)


def test_never_nan_or_out_of_range():
    for gh in (-1.0, -0.1, 0.0, 0.1, 1.0, 5.0):
        for gv in (-1.0, 0.0, 1.0):
            signal, conf = _gaze_result_to_signal(_gaze_result(gh, gv), last_signal=0.0)
            assert not math.isnan(signal)
            assert 0.0 <= signal <= 1.0


# --- fallback heuristic --------------------------------------------------------

def test_fallback_signal_centered_face_is_high():
    w, h = 640, 480
    bbox = [w / 2 - 20, h / 2 - 20, w / 2 + 20, h / 2 + 20]
    landmarks = [_landmark(0.5, 0.5) for _ in range(478)]
    signal, conf = _fallback_signal(bbox, landmarks, w, h)
    assert signal > 0.9


def test_fallback_signal_corner_face_is_low():
    w, h = 640, 480
    bbox = [0, 0, 20, 20]
    landmarks = [_landmark(0.02, 0.02) for _ in range(478)]
    signal, _ = _fallback_signal(bbox, landmarks, w, h)
    assert signal < 0.5


def test_fallback_confidence_reflects_in_bounds_landmark_fraction():
    w, h = 640, 480
    bbox = [0, 0, 20, 20]
    # Half the tracked landmark indices out of [0,1] bounds.
    landmarks = [_landmark(2.0 if i % 2 == 0 else 0.5, 0.5) for i in range(500)]
    _, conf = _fallback_signal(bbox, landmarks, w, h)
    assert 0.0 <= conf <= 1.0


def test_fallback_never_nan():
    w, h = 640, 480
    bbox = [0, 0, 0, 0]  # degenerate
    landmarks = [_landmark(0.5, 0.5) for _ in range(478)]
    signal, conf = _fallback_signal(bbox, landmarks, w, h)
    assert not math.isnan(signal) and not math.isnan(conf)


# --- bbox ------------------------------------------------------------------

def test_face_bbox_clamped_to_frame():
    w, h = 100, 100
    landmarks = [_landmark(-0.5, -0.5), _landmark(1.5, 1.5)]
    bbox = _face_bbox(landmarks, w, h)
    assert bbox == [0.0, 0.0, 100.0, 100.0]


# --- mouth opening (visual_speaking_proxy input) -----------------------------

def _mouth_landmarks(upper_y=0.5, lower_y=0.5, left_x=0.4, right_x=0.6):
    """Landmarks with only the 4 mouth indices (13, 14, 61, 291) set
    meaningfully; enough entries to cover the highest index used."""
    landmarks = [_landmark(0.5, 0.5) for _ in range(300)]
    landmarks[13] = _landmark(0.5, upper_y)   # upper inner lip
    landmarks[14] = _landmark(0.5, lower_y)   # lower inner lip
    landmarks[61] = _landmark(left_x, 0.5)    # left corner
    landmarks[291] = _landmark(right_x, 0.5)  # right corner
    return landmarks


def test_mouth_opening_zero_when_lips_closed():
    w, h = 640, 480
    landmarks = _mouth_landmarks(upper_y=0.5, lower_y=0.5)  # no vertical gap
    opening = _mouth_opening_from_landmarks(landmarks, w, h)
    assert opening == pytest.approx(0.0, abs=1e-6)


def test_mouth_opening_positive_when_open():
    w, h = 640, 480
    landmarks = _mouth_landmarks(upper_y=0.45, lower_y=0.55)  # vertical gap
    opening = _mouth_opening_from_landmarks(landmarks, w, h)
    assert opening > 0.0


def test_mouth_opening_scales_with_gap():
    w, h = 640, 480
    small = _mouth_opening_from_landmarks(_mouth_landmarks(0.48, 0.52), w, h)
    large = _mouth_opening_from_landmarks(_mouth_landmarks(0.40, 0.60), w, h)
    assert large > small


def test_mouth_opening_never_nan_on_degenerate_width():
    w, h = 640, 480
    landmarks = _mouth_landmarks(left_x=0.5, right_x=0.5)  # zero-width mouth
    opening = _mouth_opening_from_landmarks(landmarks, w, h)
    assert not math.isnan(opening)
    assert opening == 0.0


# --- Perception class smoke tests (blank frame, no MediaPipe faces) -----------

def test_perception_smoke_on_blank_frame():
    from gaze_attention import DEFAULT_CONFIG
    from gaze_attention.perception import Perception

    p = Perception(DEFAULT_CONFIG)
    frame = np.zeros((240, 320, 3), dtype=np.uint8)
    obs = p.process_frame(frame, timestamp=0.0)
    assert obs == []
    assert p.last_frame_used_fallback is False
    p.close()


def test_perception_smoke_forced_fallback():
    from gaze_attention import DEFAULT_CONFIG
    from gaze_attention.perception import Perception

    p = Perception(DEFAULT_CONFIG, force_fallback=True)
    frame = np.zeros((240, 320, 3), dtype=np.uint8)
    obs = p.process_frame(frame, timestamp=0.0)
    assert obs == []
    p.close()


# --- id=None filtering (Step 2 hard requirement) --------------------------
#
# The real mediapipe package isn't installed in this environment (see the
# two skips above -- process_frame() needs it for face detection even in
# force_fallback mode, which only bypasses GazeEstimator). To actually
# exercise perception.py's real code path -- not a reimplementation of it
# -- rather than just assert-and-hope, this installs a minimal fake
# `mediapipe` package into sys.modules for the duration of one test, with
# just enough surface for Perception.__init__/process_frame to run:
# ImageFormat/Image, and a FaceLandmarker stand-in whose detect_for_video()
# returns a scripted sequence of synthetic per-frame face landmark lists.

def _face_landmarks(cx=0.5, cy=0.5, half_size=0.08, n=480):
    """n synthetic landmarks: two at the bbox corners (so _face_bbox gets a
    real, non-degenerate box) and the rest clustered at the center (so every
    _TRACKED_LANDMARK_INDICES index used by _fallback_signal/mouth-opening
    stays in-frame bounds)."""
    landmarks = [_landmark(cx, cy) for _ in range(n)]
    landmarks[0] = _landmark(cx - half_size, cy - half_size)
    landmarks[1] = _landmark(cx + half_size, cy + half_size)
    return landmarks


class _FakeFaceLandmarker:
    def __init__(self, per_frame_faces):
        self._per_frame_faces = per_frame_faces
        self._i = 0

    def detect_for_video(self, mp_image, timestamp_ms):
        i = min(self._i, len(self._per_frame_faces) - 1)
        self._i += 1
        return SimpleNamespace(face_landmarks=self._per_frame_faces[i])

    def close(self):
        pass


def _install_fake_mediapipe(per_frame_faces):
    """Returns a cleanup callable; restores whatever was in sys.modules
    (real or absent) for every key this touches."""
    import sys
    from unittest.mock import MagicMock

    keys = [
        "mediapipe", "mediapipe.tasks", "mediapipe.tasks.python",
        "mediapipe.tasks.python.vision", "mediapipe.tasks.python.core",
        "mediapipe.tasks.python.core.base_options",
    ]
    saved = {k: sys.modules.get(k) for k in keys}

    fake_mp = MagicMock()
    fake_mp.ImageFormat.SRGB = "SRGB"
    fake_mp.Image = lambda image_format, data: SimpleNamespace(image_format=image_format, data=data)

    fake_landmarker = _FakeFaceLandmarker(per_frame_faces)
    fake_vision = MagicMock()
    fake_vision.FaceLandmarker.create_from_options = MagicMock(return_value=fake_landmarker)
    fake_vision.RunningMode.VIDEO = "VIDEO"

    fake_core = MagicMock()
    fake_base_options_mod = MagicMock()
    fake_core.base_options = fake_base_options_mod

    # `from X import Y` prefers attribute access on the already-imported
    # package object over re-checking sys.modules["X.Y"] -- a bare
    # MagicMock() auto-creates a DIFFERENT child mock on that access rather
    # than falling through to the dotted sys.modules entry, so every
    # submodule attribute has to be wired explicitly here.
    fake_python_pkg = MagicMock()
    fake_python_pkg.vision = fake_vision
    fake_python_pkg.core = fake_core

    sys.modules["mediapipe"] = fake_mp
    sys.modules["mediapipe.tasks"] = MagicMock()
    sys.modules["mediapipe.tasks.python"] = fake_python_pkg
    sys.modules["mediapipe.tasks.python.vision"] = fake_vision
    sys.modules["mediapipe.tasks.python.core"] = fake_core
    sys.modules["mediapipe.tasks.python.core.base_options"] = fake_base_options_mod

    def cleanup():
        for k, v in saved.items():
            if v is None:
                sys.modules.pop(k, None)
            else:
                sys.modules[k] = v

    return cleanup


def test_unconfirmed_frames_produce_no_observations_confirmed_frames_do():
    """Explicit per-frame check (not just the aggregate id!=None check
    below): frames 0/1 (TENTATIVE, hits<n_confirm=3) must produce ZERO
    observations for that face; frame 2 onward (CONFIRMED) must produce
    exactly one, with a real id."""
    same_face = [_face_landmarks(cx=0.5, cy=0.5)]
    cleanup = _install_fake_mediapipe([same_face] * 5)
    try:
        from gaze_attention import DEFAULT_CONFIG
        from gaze_attention.perception import Perception

        config = dict(DEFAULT_CONFIG)
        config["n_confirm"] = 3
        p = Perception(config, force_fallback=True)
        frame = np.zeros((240, 320, 3), dtype=np.uint8)

        obs0 = p.process_frame(frame, timestamp=0.0)
        obs1 = p.process_frame(frame, timestamp=0.033)
        assert obs0 == [], "frame 0 (hit 1/3, still TENTATIVE) must skip -- no observation"
        assert obs1 == [], "frame 1 (hit 2/3, still TENTATIVE) must skip -- no observation"

        obs2 = p.process_frame(frame, timestamp=0.066)
        assert len(obs2) == 1, "frame 2 (hit 3/3, now CONFIRMED) must produce exactly one observation"
        assert obs2[0]["id"] is not None
        p.close()
    finally:
        cleanup()


def test_none_never_enters_present_ids_or_policy_state_dicts():
    """Drives real Perception output into a real ConversationalSimplePolicy
    tick by tick. During the TENTATIVE ramp-up (frames 0-1), present_ids
    must never contain None, and after every tick, None must never appear
    as a key in the policy's own per-person state dicts -- proving the
    guard holds all the way through the policy layer, not just at the
    Observation-construction boundary."""
    same_face = [_face_landmarks(cx=0.5, cy=0.5)]
    cleanup = _install_fake_mediapipe([same_face] * 6)
    try:
        from gaze_attention import DEFAULT_CONFIG
        from gaze_attention.perception import Perception
        from gaze_attention.policies import ConversationalSimplePolicy

        config = dict(DEFAULT_CONFIG)
        config["n_confirm"] = 3
        p = Perception(config, force_fallback=True)
        policy = ConversationalSimplePolicy()
        frame = np.zeros((240, 320, 3), dtype=np.uint8)

        for i in range(6):
            obs = p.process_frame(frame, timestamp=i * 0.033)
            present_ids = {o["id"] for o in obs}
            assert None not in present_ids, f"frame {i}: None leaked into present_ids"

            policy.update(obs, 0.033)
            assert None not in policy.short_term_score, f"frame {i}: None key in short_term_score"
            assert None not in policy.attention_debt, f"frame {i}: None key in attention_debt"
        p.close()
    finally:
        cleanup()


def _face_landmarks_with_pose(n=480):
    """Non-degenerate, roughly frontal placement of the 6 head-pose
    landmark indices (1,152,33,263,61,291) so cv2.solvePnP produces a valid
    solution -- for the 'valid observation populates yaw/pitch/phi/e/q'
    test. Other landmarks sit at the face centre (unused by head pose;
    kept in-frame so the fallback confidence proxy stays well-defined)."""
    landmarks = [_landmark(0.5, 0.55) for _ in range(n)]
    landmarks[1] = _landmark(0.50, 0.55)    # nose tip
    landmarks[152] = _landmark(0.50, 0.75)  # chin
    landmarks[33] = _landmark(0.40, 0.45)   # left eye, left corner
    landmarks[263] = _landmark(0.60, 0.45)  # right eye, right corner
    landmarks[61] = _landmark(0.45, 0.65)   # left mouth corner
    landmarks[291] = _landmark(0.55, 0.65)  # right mouth corner
    return landmarks


def test_valid_head_pose_populates_gaze_fields():
    """Change A: when head pose is recoverable, an emitted observation
    carries finite yaw/pitch/phi/e/q (radians), and gaze_signal is q
    (mouth_cue_weight defaults to 0)."""
    import math

    faces = _face_landmarks_with_pose()
    cleanup = _install_fake_mediapipe([[faces]] * 5)
    try:
        from gaze_attention import DEFAULT_CONFIG
        from gaze_attention.perception import Perception

        config = dict(DEFAULT_CONFIG)
        config["n_confirm"] = 1  # confirm immediately so we get an observation on frame 0
        p = Perception(config, force_fallback=True)
        frame = np.zeros((480, 640, 3), dtype=np.uint8)

        obs = None
        for i in range(3):
            out = p.process_frame(frame, timestamp=i * 0.033)
            if out:
                obs = out[0]
        assert obs is not None
        for key in ("yaw", "pitch", "phi", "e", "q"):
            assert obs[key] is not None, f"{key} should be populated when head pose is available"
            assert math.isfinite(obs[key])
        assert 0.0 <= obs["q"] <= 1.0
        # Default mouth_cue_weight == 0.0 -> interaction cue is q alone.
        assert obs["gaze_signal"] == pytest.approx(obs["q"], abs=1e-9)
        p.close()
    finally:
        cleanup()


def test_head_pose_unavailable_falls_back_without_fabricated_gaze():
    """Change A fallback: degenerate landmarks (solvePnP can't get a real
    pose) must NOT crash and must NOT fabricate gaze -- yaw/phi/e/q stay
    None, the observation is still marked fallback, and gaze_signal keeps
    the existing position-based value (a real number, but not labelled as
    calibrated gaze)."""
    # All landmarks at a single point -> the 6 head-pose points are fully
    # coincident -> solvePnP cannot recover a pose -> pose unavailable.
    degenerate = [_landmark(0.5, 0.5) for _ in range(480)]
    cleanup = _install_fake_mediapipe([[degenerate]] * 3)
    try:
        from gaze_attention import DEFAULT_CONFIG
        from gaze_attention.perception import Perception

        config = dict(DEFAULT_CONFIG)
        config["n_confirm"] = 1
        p = Perception(config, force_fallback=True)
        frame = np.zeros((480, 640, 3), dtype=np.uint8)

        obs = None
        for i in range(3):
            out = p.process_frame(frame, timestamp=i * 0.033)
            if out:
                obs = out[0]
        assert obs is not None
        assert obs["yaw"] is None and obs["phi"] is None and obs["e"] is None and obs["q"] is None
        assert obs.get("fallback") is True, "fallback observation must stay marked fallback"
        assert isinstance(obs["gaze_signal"], float)  # position-based fallback, still usable
        p.close()
    finally:
        cleanup()


def test_mouth_cue_available_with_zero_default_weight():
    """Change A: the existing mouth-motion cue stays available on the
    observation, and with the default mouth_cue_weight == 0.0 it does not
    contribute to the interaction cue (gaze_signal == q)."""
    from gaze_attention import DEFAULT_CONFIG

    assert DEFAULT_CONFIG["mouth_cue_weight"] == 0.0

    faces = _face_landmarks_with_pose()
    cleanup = _install_fake_mediapipe([[faces]] * 3)
    try:
        from gaze_attention.perception import Perception

        config = dict(DEFAULT_CONFIG)
        config["n_confirm"] = 1
        p = Perception(config, force_fallback=True)
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        obs = None
        for i in range(3):
            out = p.process_frame(frame, timestamp=i * 0.033)
            if out:
                obs = out[0]
        assert obs is not None
        assert "mouth_activity" in obs  # mouth cue still computed and carried
        assert obs["gaze_signal"] == pytest.approx(obs["q"], abs=1e-9)  # zero weight -> cue is q
        p.close()
    finally:
        cleanup()


def test_no_observation_ever_carries_id_none():
    """Step 2 hard requirement: an unconfirmed (TENTATIVE) tracker id must
    never reach an Observation. Runs the SAME bbox for several frames (so
    the tracker's TENTATIVE->CONFIRMED promotion actually happens mid-test)
    through the real Perception.process_frame(), and asserts every single
    observation across every frame has a real, non-None id."""
    same_face = [_face_landmarks(cx=0.5, cy=0.5)]  # one face, same position every frame
    cleanup = _install_fake_mediapipe([same_face] * 10)
    try:
        from gaze_attention import DEFAULT_CONFIG
        from gaze_attention.perception import Perception

        config = dict(DEFAULT_CONFIG)
        config["n_confirm"] = 3
        p = Perception(config, force_fallback=True)
        frame = np.zeros((240, 320, 3), dtype=np.uint8)

        saw_a_real_id = False
        for i in range(10):
            obs = p.process_frame(frame, timestamp=i * 0.033)
            for o in obs:
                assert o["id"] is not None, f"frame {i}: observation carried id=None"
                saw_a_real_id = True
        assert saw_a_real_id, "expected the persistent face to eventually confirm and produce a real id"
        p.close()
    finally:
        cleanup()


# --- Change B: stability confidence ------------------------------------------

def test_stability_identical_landmarks_gives_high_c_stable():
    """Identical landmarks across a full window -> zero jitter -> c_stable
    approaches 1/(1+0) = 1.0."""
    stable_face = [_face_landmarks_with_pose()]
    n_frames = 10
    cleanup = _install_fake_mediapipe([stable_face] * n_frames)
    try:
        from gaze_attention import DEFAULT_CONFIG
        from gaze_attention.perception import Perception

        config = dict(DEFAULT_CONFIG)
        config["n_confirm"] = 1
        config["stability_window_frames"] = 5
        p = Perception(config, force_fallback=True)
        frame = np.zeros((480, 640, 3), dtype=np.uint8)

        obs = None
        for i in range(n_frames):
            out = p.process_frame(frame, timestamp=i * 0.033)
            if out:
                obs = out[0]
        assert obs is not None
        assert obs.get("c_stable") is not None
        assert obs["c_stable"] > 0.9, f"identical landmarks should give high c_stable, got {obs['c_stable']}"
        p.close()
    finally:
        cleanup()


def _face_landmarks_jittery(frame_idx, n=480):
    """Landmarks whose BBOX stays constant (outermost landmarks fixed) but
    whose interior landmarks alternate position, creating high jitter in
    the stability metric without confusing the tracker."""
    offset = 0.08 if frame_idx % 2 == 0 else -0.08
    lms = [_landmark(0.5 + offset, 0.5 + offset) for _ in range(n)]
    # Fix the two bbox-corner landmarks so _face_bbox is constant.
    lms[0] = _landmark(0.35, 0.35)
    lms[1] = _landmark(0.65, 0.65)
    return lms


def test_stability_jittery_landmarks_gives_low_c_stable():
    """Interior landmarks alternating positions (bbox constant so the
    tracker keeps matching) -> high jitter -> low c_stable."""
    per_frame = []
    for i in range(12):
        per_frame.append([_face_landmarks_jittery(i)])
    cleanup = _install_fake_mediapipe(per_frame)
    try:
        from gaze_attention import DEFAULT_CONFIG
        from gaze_attention.perception import Perception

        config = dict(DEFAULT_CONFIG)
        config["n_confirm"] = 1
        config["stability_window_frames"] = 5
        config["stability_k"] = 50.0
        p = Perception(config, force_fallback=True)
        frame = np.zeros((480, 640, 3), dtype=np.uint8)

        obs = None
        for i in range(12):
            out = p.process_frame(frame, timestamp=i * 0.033)
            if out:
                obs = out[0]
        assert obs is not None
        assert obs.get("c_stable") is not None
        assert obs["c_stable"] < 0.3, f"jittery landmarks should give low c_stable, got {obs['c_stable']}"
        p.close()
    finally:
        cleanup()


def test_stability_per_person_history():
    """Two faces tracked simultaneously must have independent c_stable
    histories. One face is perfectly stable (identical landmarks every
    frame), the other has jittering interior landmarks (bbox-corner
    landmarks fixed so the tracker keeps matching both). After enough
    frames, stable face's c_stable > jittery face's c_stable."""
    # Stable face: left side, all landmarks identical every frame.
    stable_lms = [_landmark(0.2, 0.5) for _ in range(480)]
    stable_lms[0] = _landmark(0.12, 0.42)
    stable_lms[1] = _landmark(0.28, 0.58)

    per_frame = []
    for i in range(12):
        # Jittery face: right side, bbox-corner landmarks fixed, interiors jitter.
        offset = 0.08 if i % 2 == 0 else -0.08
        jittery_lms = [_landmark(0.7 + offset, 0.5 + offset) for _ in range(480)]
        jittery_lms[0] = _landmark(0.62, 0.42)  # fixed bbox corner
        jittery_lms[1] = _landmark(0.78, 0.58)  # fixed bbox corner
        per_frame.append([stable_lms, jittery_lms])

    cleanup = _install_fake_mediapipe(per_frame)
    try:
        from gaze_attention import DEFAULT_CONFIG
        from gaze_attention.perception import Perception

        config = dict(DEFAULT_CONFIG)
        config["n_confirm"] = 1
        config["stability_window_frames"] = 3
        config["stability_k"] = 50.0
        p = Perception(config, force_fallback=True)
        frame = np.zeros((480, 640, 3), dtype=np.uint8)

        last_obs = []
        for i in range(12):
            out = p.process_frame(frame, timestamp=i * 0.033)
            if len(out) == 2:
                last_obs = out
        assert len(last_obs) == 2
        # Identify which obs is the stable vs jittery face by bbox position.
        left = min(last_obs, key=lambda o: o["bbox"][0])
        right = max(last_obs, key=lambda o: o["bbox"][0])
        assert left["c_stable"] > right["c_stable"], \
            f"stable face should have higher c_stable than jittery: " \
            f"stable={left['c_stable']}, jittery={right['c_stable']}"
        p.close()
    finally:
        cleanup()


def test_stability_history_cleanup_on_track_disappearance():
    """When a tracked face disappears (past t_lost_seconds), its landmark
    history must be pruned so a future person at the same id slot doesn't
    inherit stale stability data."""
    face = [_face_landmarks_with_pose()]
    empty = []
    per_frame = [face] * 3 + [empty] * 80 + [face] * 5
    cleanup = _install_fake_mediapipe(per_frame)
    try:
        from gaze_attention import DEFAULT_CONFIG
        from gaze_attention.perception import Perception

        config = dict(DEFAULT_CONFIG)
        config["n_confirm"] = 1
        config["t_lost_seconds"] = 0.1
        config["stability_window_frames"] = 3
        p = Perception(config, force_fallback=True)
        frame = np.zeros((480, 640, 3), dtype=np.uint8)

        for i in range(83):
            p.process_frame(frame, timestamp=i * 0.033)
        # After 80 empty frames at 33ms each (~2.6s), far past t_lost=0.1s,
        # the old track is deleted and its landmark history pruned.
        assert len(p._landmark_history) == 0, \
            f"landmark history should be empty after track deletion, got {len(p._landmark_history)} entries"

        # New face appears: its FIRST observation should use
        # stability_init_confidence, not inherited history.
        first_obs = None
        for i in range(83, 88):
            out = p.process_frame(frame, timestamp=i * 0.033)
            if out and first_obs is None:
                first_obs = out[0]
        assert first_obs is not None
        assert first_obs["c_stable"] == pytest.approx(config["stability_init_confidence"]), \
            f"new track's first observation should use init confidence, got {first_obs['c_stable']}"
        p.close()
    finally:
        cleanup()


def test_stability_insufficient_history_uses_init_confidence():
    """Before accumulating a full stability_window_frames, c_stable must
    be the conservative stability_init_confidence."""
    face = [_face_landmarks_with_pose()]
    cleanup = _install_fake_mediapipe([face] * 5)
    try:
        from gaze_attention import DEFAULT_CONFIG
        from gaze_attention.perception import Perception

        config = dict(DEFAULT_CONFIG)
        config["n_confirm"] = 1
        config["stability_window_frames"] = 5
        config["stability_init_confidence"] = 0.5
        p = Perception(config, force_fallback=True)
        frame = np.zeros((480, 640, 3), dtype=np.uint8)

        out = p.process_frame(frame, timestamp=0.0)
        assert len(out) == 1
        assert out[0]["c_stable"] == pytest.approx(0.5), \
            f"first frame should use init confidence, got {out[0]['c_stable']}"
        p.close()
    finally:
        cleanup()


# --- Change B: confidence product and bounds ---------------------------------

def test_confidence_equals_clipped_product_of_components():
    """C must equal clip(c_size * c_pose * c_stable, confidence_min,
    confidence_max)."""
    face = [_face_landmarks_with_pose()]
    cleanup = _install_fake_mediapipe([face] * 8)
    try:
        from gaze_attention import DEFAULT_CONFIG
        from gaze_attention.perception import Perception

        config = dict(DEFAULT_CONFIG)
        config["n_confirm"] = 1
        config["stability_window_frames"] = 3
        config["confidence_min"] = 0.05
        config["confidence_max"] = 1.0
        p = Perception(config, force_fallback=True)
        frame = np.zeros((480, 640, 3), dtype=np.uint8)

        obs = None
        for i in range(8):
            out = p.process_frame(frame, timestamp=i * 0.033)
            if out:
                obs = out[0]
        assert obs is not None
        product = obs["c_size"] * obs["c_pose"] * obs["c_stable"]
        expected = max(config["confidence_min"], min(config["confidence_max"], product))
        assert obs["C"] == pytest.approx(expected, abs=1e-9)
        p.close()
    finally:
        cleanup()


def test_confidence_within_configured_bounds():
    """C must always be within [confidence_min, confidence_max]."""
    face = [_face_landmarks_with_pose()]
    cleanup = _install_fake_mediapipe([face] * 5)
    try:
        from gaze_attention import DEFAULT_CONFIG
        from gaze_attention.perception import Perception

        config = dict(DEFAULT_CONFIG)
        config["n_confirm"] = 1
        config["confidence_min"] = 0.10
        config["confidence_max"] = 0.90
        p = Perception(config, force_fallback=True)
        frame = np.zeros((480, 640, 3), dtype=np.uint8)

        for i in range(5):
            out = p.process_frame(frame, timestamp=i * 0.033)
            for o in out:
                if not o.get("lost", False):
                    assert config["confidence_min"] <= o["C"] <= config["confidence_max"], \
                        f"C={o['C']} outside [{config['confidence_min']}, {config['confidence_max']}]"
        p.close()
    finally:
        cleanup()


def test_lost_observation_has_zero_confidence_and_null_components():
    """A LOST observation must have C=0.0 and all components None."""
    face = [_face_landmarks_with_pose()]
    empty = []
    per_frame = [face] * 4 + [empty] * 2
    cleanup = _install_fake_mediapipe(per_frame)
    try:
        from gaze_attention import DEFAULT_CONFIG
        from gaze_attention.perception import Perception

        config = dict(DEFAULT_CONFIG)
        config["n_confirm"] = 1
        config["t_lost_seconds"] = 2.0
        p = Perception(config, force_fallback=True)
        frame = np.zeros((480, 640, 3), dtype=np.uint8)

        for i in range(4):
            p.process_frame(frame, timestamp=i * 0.033)
        # Frame 4: face disappears -> LOST observation.
        out = p.process_frame(frame, timestamp=4 * 0.033)
        lost_obs = [o for o in out if o.get("lost", False)]
        assert len(lost_obs) == 1, "expected one LOST observation"
        o = lost_obs[0]
        assert o["C"] == 0.0
        assert o["confidence"] == 0.0
        for k in ("c_size", "c_pose", "c_stable"):
            assert o[k] is None, f"{k} should be None for LOST, got {o[k]}"
        p.close()
    finally:
        cleanup()


def test_confidence_no_nan_or_inf():
    """No observation should ever have NaN or Inf in any confidence field."""
    face = [_face_landmarks_with_pose()]
    cleanup = _install_fake_mediapipe([face] * 8)
    try:
        from gaze_attention import DEFAULT_CONFIG
        from gaze_attention.perception import Perception

        config = dict(DEFAULT_CONFIG)
        config["n_confirm"] = 1
        p = Perception(config, force_fallback=True)
        frame = np.zeros((480, 640, 3), dtype=np.uint8)

        for i in range(8):
            out = p.process_frame(frame, timestamp=i * 0.033)
            for o in out:
                for k in ("C", "confidence"):
                    assert math.isfinite(o[k]), f"{k}={o[k]} is not finite"
                for k in ("c_size", "c_pose", "c_stable"):
                    if o[k] is not None:
                        assert math.isfinite(o[k]), f"{k}={o[k]} is not finite"
        p.close()
    finally:
        cleanup()


# --- Change B: variation test (confidence NOT constant 1.0) -------------------

def test_confidence_varies_across_different_conditions():
    """The whole point of Change B: confidence must NOT be constant ~1.0
    across all conditions. A small face and a large face at the same
    position must produce different C values."""
    small_face = _face_landmarks(cx=0.5, cy=0.5, half_size=0.02)
    large_face = _face_landmarks(cx=0.5, cy=0.5, half_size=0.15)
    # Run small face for several frames, record C.
    cleanup = _install_fake_mediapipe([[small_face]] * 6)
    try:
        from gaze_attention import DEFAULT_CONFIG
        from gaze_attention.perception import Perception

        config = dict(DEFAULT_CONFIG)
        config["n_confirm"] = 1
        config["stability_window_frames"] = 3
        p = Perception(config, force_fallback=True)
        frame = np.zeros((480, 640, 3), dtype=np.uint8)

        small_C = None
        for i in range(6):
            out = p.process_frame(frame, timestamp=i * 0.033)
            if out:
                small_C = out[0]["C"]
        p.close()
    finally:
        cleanup()

    cleanup = _install_fake_mediapipe([[large_face]] * 6)
    try:
        config2 = dict(DEFAULT_CONFIG)
        config2["n_confirm"] = 1
        config2["stability_window_frames"] = 3
        p2 = Perception(config2, force_fallback=True)
        large_C = None
        for i in range(6):
            out = p2.process_frame(frame, timestamp=i * 0.033)
            if out:
                large_C = out[0]["C"]
        p2.close()
    finally:
        cleanup()

    assert small_C is not None and large_C is not None
    assert small_C != pytest.approx(large_C, abs=0.05), \
        f"Confidence should vary with face size: small_C={small_C}, large_C={large_C}"
    assert large_C > small_C, \
        f"Larger face should have higher confidence: large_C={large_C}, small_C={small_C}"


# --- Change B: data-flow (components survive perception -> policy -> logger) --

def test_confidence_components_survive_to_policy_observations():
    """All Change B keys (c_size, c_pose, c_stable, C) must appear on
    observations that reach the policy layer."""
    face = [_face_landmarks_with_pose()]
    cleanup = _install_fake_mediapipe([face] * 5)
    try:
        from gaze_attention import DEFAULT_CONFIG
        from gaze_attention.perception import Perception

        config = dict(DEFAULT_CONFIG)
        config["n_confirm"] = 1
        p = Perception(config, force_fallback=True)
        frame = np.zeros((480, 640, 3), dtype=np.uint8)

        obs = None
        for i in range(5):
            out = p.process_frame(frame, timestamp=i * 0.033)
            if out:
                obs = out[0]
        assert obs is not None
        for key in ("c_size", "c_pose", "c_stable", "C"):
            assert key in obs, f"missing {key} on observation"
            assert obs[key] is not None, f"{key} should not be None for a valid observation"
            assert math.isfinite(obs[key]), f"{key}={obs[key]} is not finite"
        p.close()
    finally:
        cleanup()
