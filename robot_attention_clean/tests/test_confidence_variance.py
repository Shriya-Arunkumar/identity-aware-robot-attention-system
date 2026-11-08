"""Change B (real confidence variance) -- pure-function tests for the
face-size and pose confidence components. Deterministic, no camera, no
Perception instance (the stability component, _compute_confidence product,
and the perception->policy->logger data flow are exercised in
tests/test_perception.py, which already has the MediaPipe mock)."""

from __future__ import annotations

import json
import math

import pytest

from gaze_attention.perception import _face_size_confidence, _pose_confidence

H = 480
REF_RATIO = 0.18


def _bbox_diag(diag_px, cx=320.0, cy=240.0):
    """Axis-aligned bbox with the given diagonal length (square)."""
    half = diag_px / (2.0 * math.sqrt(2.0))
    return [cx - half, cy - half, cx + half, cy + half]


# --- 1. face size -------------------------------------------------------------

def test_larger_face_gives_greater_or_equal_c_size():
    small = _face_size_confidence(_bbox_diag(30), H, REF_RATIO)
    large = _face_size_confidence(_bbox_diag(80), H, REF_RATIO)
    assert large >= small


def test_small_face_is_low_confidence():
    assert _face_size_confidence(_bbox_diag(10), H, REF_RATIO) < 0.3


def test_c_size_clipped_to_one_for_huge_face():
    # A face far larger than the reference must clip at 1.0, not exceed it.
    assert _face_size_confidence(_bbox_diag(5000), H, REF_RATIO) == 1.0


def test_c_size_reaches_one_at_reference_size():
    # diagonal == ref_ratio * H -> exactly 1.0.
    assert _face_size_confidence(_bbox_diag(REF_RATIO * H), H, REF_RATIO) == pytest.approx(1.0)


def test_c_size_in_unit_range_and_finite():
    for diag in (0, 5, 50, 200, 100000):
        c = _face_size_confidence(_bbox_diag(diag), H, REF_RATIO)
        assert 0.0 <= c <= 1.0 and math.isfinite(c)


def test_c_size_guards_bad_frame_height():
    assert _face_size_confidence(_bbox_diag(50), 0, REF_RATIO) == 0.0
    assert _face_size_confidence(_bbox_diag(50), -10, REF_RATIO) == 0.0


def test_c_size_degenerate_bbox_is_zero():
    assert _face_size_confidence([100, 100, 100, 100], H, REF_RATIO) == 0.0


# --- 2. pose ------------------------------------------------------------------

def test_frontal_pose_is_high_confidence():
    assert _pose_confidence(0.0) == pytest.approx(1.0)


def test_large_yaw_lowers_pose_confidence():
    frontal = _pose_confidence(0.0)
    turned = _pose_confidence(math.radians(45))
    strongly = _pose_confidence(math.radians(70))
    assert frontal > turned > strongly
    assert 0.0 <= strongly < turned < frontal <= 1.0


def test_pose_confidence_clipped_to_zero_past_90deg():
    # cos(yaw) < 0 past +-90deg -> clipped to 0, then squared -> 0.
    assert _pose_confidence(math.radians(120)) == 0.0


def test_pose_confidence_symmetric_in_yaw_sign():
    assert _pose_confidence(math.radians(30)) == pytest.approx(_pose_confidence(math.radians(-30)))


def test_pose_confidence_nonfinite_yaw_is_zero():
    assert _pose_confidence(float("nan")) == 0.0
    assert _pose_confidence(float("inf")) == 0.0


# --- 3. Change C: use_confidence ablation flag --------------------------------
#
# Runs a short synthetic sequence with genuinely varying C (oscillating above
# and below confidence_threshold, so the gate in GreedyDwellPolicy actually
# flips) once with use_confidence=True and once with use_confidence=False.
# Perception isn't involved -- this drives GreedyDwellPolicy.update() and
# JsonlLogger.log_frame() directly with hand-built observation dicts, which
# is exactly the boundary Change C operates at (consumption sites in
# policies.py/baseline_policies.py/attention_manager.py, never perception.py
# or jsonl_logger.py).

def _synthetic_observations(tick: int):
    """One person, id=1, always seeking. Confidence (and its Change-B
    components) genuinely varies tick to tick, oscillating across
    confidence_threshold=0.45 -- high on even ticks, low on odd ticks."""
    c = 0.90 if tick % 2 == 0 else 0.10
    obs = {
        "id": 1,
        "bbox": [10.0, 10.0, 50.0, 50.0],
        "gaze_signal": 0.9,
        "confidence": c,
        "seeking_attention": True,
        "timestamp": tick * 0.1,
        "mouth_activity": 0.0,
        "speaking": False,
        "speech_confidence": 0.0,
        "lost": False,
        "yaw": 0.01 * tick,
        "pitch": 0.02 * tick,
        "phi": 0.03 * tick,
        "e": 0.04 * tick,
        "q": c,
        "c_size": c,
        "c_pose": c,
        "c_stable": c,
        "C": c,
    }
    return [obs]


def test_use_confidence_ablation_flag_bypasses_consumption_not_logging(tmp_path):
    from gaze_attention.baseline_policies import GreedyDwellPolicy
    from gaze_attention.jsonl_logger import JsonlLogger

    n_ticks = 10
    dt = 0.1
    threshold = 0.45

    results = {}
    for use_confidence in (True, False):
        policy = GreedyDwellPolicy(confidence_threshold=threshold, use_confidence=use_confidence)
        logger = JsonlLogger(logs_dir=tmp_path / f"run_{use_confidence}")

        score_trajectory = []
        for tick in range(n_ticks):
            observations = _synthetic_observations(tick)
            policy.update(observations, dt)
            metrics_by_id = {1: {"engagement": policy._score[1], "score": policy._score[1]}}
            logger.log_frame(tick * dt, observations, metrics_by_id)
            score_trajectory.append(policy._score[1])
        logger.close()

        logged_rows = [json.loads(line) for line in logger.path.read_text().splitlines()]
        results[use_confidence] = {
            "score_trajectory": score_trajectory,
            "logged_rows": logged_rows,
        }

    # -- logging must be IDENTICAL regardless of the flag (proves the flag
    # doesn't touch perception/logging) --------------------------------------
    rows_true = results[True]["logged_rows"]
    rows_false = results[False]["logged_rows"]
    assert len(rows_true) == len(rows_false) == n_ticks
    for r_true, r_false in zip(rows_true, rows_false):
        for key in ("c_size", "c_pose", "c_stable", "C", "yaw", "pitch", "phi", "e", "q"):
            assert r_true[key] == r_false[key], (
                f"logged {key} differs between use_confidence=True/False runs: "
                f"{r_true[key]!r} vs {r_false[key]!r} -- the ablation flag must "
                f"never affect what's logged"
            )

    # -- consumption (score trajectory) MUST differ between the two runs -----
    traj_true = results[True]["score_trajectory"]
    traj_false = results[False]["score_trajectory"]
    assert traj_true != traj_false, (
        "use_confidence=True vs False should produce different score "
        f"trajectories, got identical: {traj_true}"
    )
    # Specifically: with use_confidence=False, c_eff is always 1.0 >=
    # threshold, so the gate never fails and score only ever grows (monotonic
    # non-decreasing). With use_confidence=True, the real oscillating C drops
    # below threshold on odd ticks, so the gate fails and score decays there
    # -- proving the gate itself flipped, not just a magnitude difference.
    assert all(b >= a - 1e-9 for a, b in zip(traj_false, traj_false[1:])), \
        f"use_confidence=False should be monotonic non-decreasing, got {traj_false}"
    assert any(b < a for a, b in zip(traj_true, traj_true[1:])), \
        f"use_confidence=True should show at least one decay (gate failing on low-C ticks), got {traj_true}"
