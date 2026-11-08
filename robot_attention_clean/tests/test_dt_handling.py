"""Tests for FM4: dt computation, rejection, and clamping (see
DEBUG_ATTENTION.md FM4). Covers gaze_attention.sanitize_dt/raw_dt_from_timestamps
directly plus the real integration paths that use them (main_webcam.py's
per-tick dt, evaluation/replay.py's run_scenario and frames_from_csv), so a
regression at either the unit or the wiring level would be caught.
"""

import math

import pytest

from gaze_attention import DEFAULT_CONFIG, raw_dt_from_timestamps, sanitize_dt
from gaze_attention.policies import ConfidenceDebtPolicy

FALLBACK = DEFAULT_CONFIG["dt_fallback_seconds"]
MAX_DT = DEFAULT_CONFIG["dt_max_seconds"]


# --- sanitize_dt: the six required cases -------------------------------------

def test_dt_zero_replaced_with_fallback():
    used_dt, raw_dt = sanitize_dt(0.0)
    assert used_dt == FALLBACK
    assert raw_dt == 0.0


def test_dt_negative_rejected_and_replaced_with_fallback():
    used_dt, raw_dt = sanitize_dt(-0.5)
    assert used_dt == FALLBACK
    assert raw_dt == -0.5
    assert used_dt >= 0.0


def test_dt_one_second_clamped_to_max():
    used_dt, raw_dt = sanitize_dt(1.0)
    assert used_dt == MAX_DT
    assert raw_dt == 1.0  # raw value preserved for logging even though clamped


def test_dt_ten_seconds_clamped_to_max():
    used_dt, raw_dt = sanitize_dt(10.0)
    assert used_dt == MAX_DT
    assert raw_dt == 10.0


def test_dt_normal_value_passes_through_unchanged():
    used_dt, raw_dt = sanitize_dt(0.033)
    assert used_dt == 0.033
    assert raw_dt == 0.033


def test_dt_missing_timestamps_replaced_with_fallback():
    # Simulates the first frame: no previous timestamp recorded yet.
    raw_dt = raw_dt_from_timestamps(current_timestamp=100.0, previous_timestamp=None)
    assert raw_dt is None
    used_dt, returned_raw_dt = sanitize_dt(raw_dt)
    assert used_dt == FALLBACK
    assert returned_raw_dt is None


def test_dt_both_timestamps_missing():
    raw_dt = raw_dt_from_timestamps(None, None)
    assert raw_dt is None
    used_dt, _ = sanitize_dt(raw_dt)
    assert used_dt == FALLBACK


# --- raw_dt_from_timestamps: normal computation ------------------------------

def test_raw_dt_computed_from_consecutive_timestamps():
    assert raw_dt_from_timestamps(100.033, 100.0) == pytest.approx(0.033)


def test_raw_dt_negative_when_timestamps_go_backward():
    # e.g. a clock adjustment -- raw_dt_from_timestamps reports it honestly;
    # sanitize_dt is what rejects it.
    raw_dt = raw_dt_from_timestamps(99.0, 100.0)
    assert raw_dt == -1.0
    used_dt, _ = sanitize_dt(raw_dt)
    assert used_dt == FALLBACK


# --- Non-finite inputs never crash -------------------------------------------

def test_dt_nan_replaced_with_fallback():
    used_dt, _ = sanitize_dt(float("nan"))
    assert used_dt == FALLBACK


def test_dt_inf_replaced_with_fallback():
    used_dt, _ = sanitize_dt(float("inf"))
    assert used_dt == FALLBACK


def test_dt_negative_inf_replaced_with_fallback():
    used_dt, _ = sanitize_dt(float("-inf"))
    assert used_dt == FALLBACK


# --- Always within [0, max_dt] no matter what --------------------------------

@pytest.mark.parametrize("raw", [None, -100.0, -1e-9, 0.0, 1e-9, 0.033, 0.2, 0.2000001, 1.0, 10.0, 1e6, float("nan"), float("inf"), float("-inf")])
def test_used_dt_always_within_bounds(raw):
    used_dt, _ = sanitize_dt(raw)
    assert 0.0 <= used_dt <= MAX_DT
    assert not math.isnan(used_dt)


# --- Custom fallback/max_dt are honored --------------------------------------

def test_custom_fallback_and_max_dt():
    used_dt, _ = sanitize_dt(-1.0, fallback=0.05, max_dt=0.1)
    assert used_dt == 0.05
    used_dt, _ = sanitize_dt(5.0, fallback=0.05, max_dt=0.1)
    assert used_dt == 0.1


# --- Integration: a huge/invalid dt cannot create a huge score jump ---------

def test_huge_dt_does_not_create_huge_score_jump_in_policy():
    """A paused window / dropped frame / delayed inference (huge raw dt)
    must not blow up engagement in a single tick -- it must be clamped to
    at most dt_max_seconds worth of accumulation, same as any normal tick
    sequence covering that much wall-clock time would allow at minimum."""
    policy = ConfidenceDebtPolicy()

    def obs(t):
        return {
            "id": 0, "bbox": [0, 0, 1, 1], "gaze_signal": 0.9, "confidence": 0.9,
            "seeking_attention": True, "timestamp": t,
        }

    used_dt, _ = sanitize_dt(9999.0)  # e.g. laptop resumed from sleep
    policy.update([obs(0.0)], used_dt)
    engagement_after_huge_gap = policy.get_metrics(0)["engagement"]

    # Upper bound: confidence(<=1.0) * dt_max_seconds is the most a single
    # tick can legitimately contribute, regardless of how large the real
    # elapsed time was.
    assert engagement_after_huge_gap <= 1.0 * MAX_DT + 1e-9


def test_zero_dt_does_not_freeze_engagement_accumulation():
    """A zero/invalid raw dt must not silently stall the whole pipeline --
    it should fall back to a normal frame-time contribution instead of
    contributing exactly nothing forever."""
    policy = ConfidenceDebtPolicy()

    def obs(t):
        return {
            "id": 0, "bbox": [0, 0, 1, 1], "gaze_signal": 0.9, "confidence": 0.9,
            "seeking_attention": True, "timestamp": t,
        }

    t = 0.0
    for _ in range(30):
        used_dt, _ = sanitize_dt(0.0)  # e.g. a stalled/misbehaving clock source
        policy.update([obs(t)], used_dt)
        t += used_dt

    assert policy.get_metrics(0)["engagement"] > 0.0


# --- Debug logging ------------------------------------------------------------

def test_debug_mode_logs_raw_and_used_dt(capsys):
    sanitize_dt(10.0, debug=True)
    captured = capsys.readouterr()
    assert "raw_dt=10.0" in captured.out
    assert f"used_dt={MAX_DT}" in captured.out


def test_no_debug_output_when_disabled(capsys):
    sanitize_dt(10.0, debug=False)
    captured = capsys.readouterr()
    assert captured.out == ""
