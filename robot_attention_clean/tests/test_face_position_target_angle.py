"""Tests for gaze_attention/target_angle.py: the SELECTED person's CURRENT
face position drives /attention_target_angle, independent of the
identity-driven /attention_target_id (which still alone controls policy
state -- engagement/debt/speaking/target persistence/attention history;
policies.py is untouched by this feature, see policies.py's own test
suite for that coverage).
"""

from __future__ import annotations

import math

import pytest

from gaze_attention import target_angle as target_angle_module
from gaze_attention.target_angle import (
    FALLBACK_ANGLES,
    MAX_ANGLE,
    compute_target_angle,
    face_center_angle,
    target_angle_for,
)

FRAME_WIDTH = 640.0


def bbox_centered_at(x_center: float, half_width: float = 20.0):
    return [x_center - half_width, 100.0, x_center + half_width, 200.0]


# --- 1. left face -> negative angle -------------------------------------------

def test_left_face_produces_negative_angle():
    bbox = bbox_centered_at(0.0)  # far left of a 640px-wide frame
    angle = face_center_angle(bbox, FRAME_WIDTH)
    assert angle is not None
    assert angle < 0.0


# --- 2. center face -> approximately zero -------------------------------------

def test_center_face_produces_approximately_zero_angle():
    bbox = bbox_centered_at(FRAME_WIDTH / 2.0)
    angle = face_center_angle(bbox, FRAME_WIDTH)
    assert angle == pytest.approx(0.0, abs=1e-9)


# --- 3. right face -> positive angle ------------------------------------------

def test_right_face_produces_positive_angle():
    bbox = bbox_centered_at(FRAME_WIDTH)  # far right
    angle = face_center_angle(bbox, FRAME_WIDTH)
    assert angle is not None
    assert angle > 0.0


# --- 4. same ID moving from left to center changes angle ----------------------

def test_same_id_moving_left_to_center_changes_angle_id_unchanged():
    target_id = 1
    left_bbox = bbox_centered_at(50.0)
    center_bbox = bbox_centered_at(FRAME_WIDTH / 2.0)

    angle_left = target_angle_for(target_id, left_bbox, FRAME_WIDTH)
    angle_center = target_angle_for(target_id, center_bbox, FRAME_WIDTH)

    assert angle_left < 0.0
    assert angle_center == pytest.approx(0.0, abs=1e-9)
    assert angle_center > angle_left  # moved smoothly toward zero, not away from it
    # The id itself is a separate, untouched concept -- target_angle_for
    # doesn't even look at policy state, only at (id, bbox, frame_width).


# --- 5. angle clamping ---------------------------------------------------------

def test_angle_is_clamped_to_max_angle_range():
    # A bbox centered far outside the frame (e.g. a stale/erroneous
    # detection) must still clamp to [-MAX_ANGLE, +MAX_ANGLE], never exceed it.
    far_left = bbox_centered_at(-5000.0)
    far_right = bbox_centered_at(5000.0)

    angle_left = face_center_angle(far_left, FRAME_WIDTH)
    angle_right = face_center_angle(far_right, FRAME_WIDTH)

    assert angle_left == pytest.approx(-MAX_ANGLE)
    assert angle_right == pytest.approx(MAX_ANGLE)
    assert -MAX_ANGLE <= angle_left <= MAX_ANGLE
    assert -MAX_ANGLE <= angle_right <= MAX_ANGLE


def test_clamp_respects_a_custom_max_angle():
    angle = face_center_angle(bbox_centered_at(5000.0), FRAME_WIDTH, max_angle=0.3)
    assert angle == pytest.approx(0.3)


# --- 6. webcam angle is NOT exponentially smoothed ----------------------------
# (target_angle.py's raw angle is the value main_webcam.py now publishes
# directly on /attention_target_angle -- see requirement A/B. Smoothing is
# exactly one stage, downstream in eye_gaze_node.py -- see
# test_slew_rate_limiter.py. A prior version of this module had its own
# ExponentialAngleSmoother running here too, in series with eye_gaze_node's
# limiter -- removed; these tests prove it's gone and stays gone.)

def test_target_angle_module_has_no_smoothing_class():
    """Requirement D: don't leave a silently active second smoothing stage.
    This module must not define (or re-export) any smoothing/rate-limiting
    class -- smoothing lives exclusively in gaze_attention_ros.slew_rate."""
    assert not hasattr(target_angle_module, "ExponentialAngleSmoother")
    for name in dir(target_angle_module):
        assert "smooth" not in name.lower(), f"unexpected smoothing-related name in target_angle.py: {name}"


def test_raw_angle_computation_is_stateless_not_smoothed():
    """compute_target_angle/face_center_angle must depend ONLY on the
    (bbox, frame_width) passed in this call -- no hidden history that would
    make consecutive calls "ease into" a new value the way a smoother
    would. Jumping the bbox from far left to far right must jump the raw
    angle immediately, in a single call, with no intermediate call needed."""
    far_left_bbox = bbox_centered_at(0.0)
    far_right_bbox = bbox_centered_at(FRAME_WIDTH)

    first = face_center_angle(far_left_bbox, FRAME_WIDTH)
    second = face_center_angle(far_right_bbox, FRAME_WIDTH)  # same call sequence a smoother would ease between

    assert first == pytest.approx(-MAX_ANGLE)
    assert second == pytest.approx(MAX_ANGLE), (
        "a single call with a far-right bbox must produce the far-right angle "
        "immediately -- any lag here would mean a smoothing stage snuck back in"
    )

    # Calling with the exact same input twice must be perfectly repeatable
    # (no internal state accumulating between calls).
    assert face_center_angle(far_right_bbox, FRAME_WIDTH) == second


# --- 7. missing bbox uses fallback mapping -------------------------------------

def test_missing_bbox_uses_fallback_mapping():
    assert target_angle_for(0, None, FRAME_WIDTH) == FALLBACK_ANGLES[0] == -MAX_ANGLE
    assert target_angle_for(1, None, FRAME_WIDTH) == FALLBACK_ANGLES[1] == 0.0
    assert target_angle_for(2, None, FRAME_WIDTH) == FALLBACK_ANGLES[2] == MAX_ANGLE


def test_missing_frame_width_uses_fallback_mapping():
    bbox = bbox_centered_at(100.0)
    assert target_angle_for(0, bbox, None) == -MAX_ANGLE
    assert target_angle_for(0, bbox, 0) == -MAX_ANGLE


def test_degenerate_zero_width_bbox_uses_fallback_mapping():
    zero_width_bbox = [100.0, 0.0, 100.0, 50.0]  # x1 == x2
    assert target_angle_for(2, zero_width_bbox, FRAME_WIDTH) == MAX_ANGLE


def test_no_target_selected_returns_none_regardless_of_bbox():
    assert target_angle_for(None, bbox_centered_at(0.0), FRAME_WIDTH) is None


def test_compute_target_angle_reports_used_fallback_and_diagnostics():
    with_bbox = compute_target_angle(1, bbox_centered_at(FRAME_WIDTH / 2.0), FRAME_WIDTH)
    assert with_bbox.used_fallback is False
    assert with_bbox.face_center_x == pytest.approx(FRAME_WIDTH / 2.0)
    assert with_bbox.normalized_x == pytest.approx(0.5)
    assert with_bbox.raw_angle == pytest.approx(0.0, abs=1e-9)

    without_bbox = compute_target_angle(1, None, FRAME_WIDTH)
    assert without_bbox.used_fallback is True
    assert without_bbox.face_center_x is None
    assert without_bbox.normalized_x is None
    assert without_bbox.raw_angle == 0.0


def test_unmapped_id_without_bbox_defaults_to_center_and_warns_once():
    warned = set()
    result = compute_target_angle(99, None, FRAME_WIDTH, warned=warned)
    assert result.raw_angle == 0.0
    assert result.used_fallback is True
    assert 99 in warned


# --- scores/angles stay finite across a sweep of positions --------------------

def test_angle_finite_across_a_sweep_of_face_positions():
    for x in range(-2000, 3000, 37):
        angle = face_center_angle(bbox_centered_at(float(x)), FRAME_WIDTH)
        assert angle is not None
        assert math.isfinite(angle)
        assert -MAX_ANGLE - 1e-9 <= angle <= MAX_ANGLE + 1e-9
