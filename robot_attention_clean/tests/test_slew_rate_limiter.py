"""Tests for ros2_ws/src/gaze_attention_ros/gaze_attention_ros/slew_rate.py
-- the ONE smoothing/rate-limiting stage in the whole angle path (webcam/
policy node -> eye command node -> ros_gz_bridge -> Gazebo). Pure Python,
no rclpy dependency, so these tests run in the normal pytest suite without
sourcing ROS.

Covers requirement F's remaining cases not already covered by
test_face_position_target_angle.py (which proves main_webcam.py's side is
UNsmoothed): eye_gaze_node performs the only slew-rate limiting, manual
commands still move smoothly (this module doesn't care where a target
value came from -- webcam-driven or a manual `ros2 topic pub`, both go
through the exact same limiter), and clamping (eye_gaze_node's own
[angle_min, angle_max] clamp, via slew_rate.clamp()).
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

# gaze_attention_ros is a separate top-level package (its own ROS 2
# package directory), not part of the gaze_attention/ package tree --
# same pattern main_webcam.py/evaluation/replay.py use to make sure their
# own root is importable.
_ROS_PKG_DIR = Path(__file__).resolve().parent.parent / "ros2_ws" / "src" / "gaze_attention_ros"
if str(_ROS_PKG_DIR) not in sys.path:
    sys.path.insert(0, str(_ROS_PKG_DIR))

from gaze_attention_ros.slew_rate import SlewRateLimiter, clamp  # noqa: E402


# --- angle clamping ------------------------------------------------------------

def test_clamp_passes_through_in_range_values():
    assert clamp(0.3, -0.8, 0.8) == 0.3
    assert clamp(-0.8, -0.8, 0.8) == -0.8
    assert clamp(0.8, -0.8, 0.8) == 0.8


def test_clamp_bounds_out_of_range_values():
    assert clamp(1.5, -0.8, 0.8) == 0.8
    assert clamp(-1.5, -0.8, 0.8) == -0.8


# --- eye_gaze_node performs the only slew-rate limiting -------------------------

def test_limiter_moves_gradually_not_instantly_toward_target():
    limiter = SlewRateLimiter(max_rate=2.0, initial_value=0.0)  # matches eye_gaze_node's default max_angular_speed
    dt = 0.05  # matches eye_gaze_node's default control_rate_hz=20

    first = limiter.step(0.8, dt)
    assert 0.0 < first < 0.8, "must not snap instantly to the target"
    assert first == pytest.approx(2.0 * dt)  # max_rate * dt, the exact linear-ramp step size


def test_limiter_converges_to_a_sustained_target():
    limiter = SlewRateLimiter(max_rate=2.0, initial_value=0.0)
    dt = 0.05
    values = [limiter.step(0.65, dt) for _ in range(100)]
    assert values[-1] == pytest.approx(0.65, abs=1e-9)
    assert all(b >= a - 1e-12 for a, b in zip(values, values[1:])), "should approach monotonically, not overshoot/oscillate"


def test_limiter_convergence_time_scales_with_step_size_not_a_fixed_time_constant():
    """Unlike an exponential filter (fixed tau), a slew-rate limiter's
    settling time is proportional to the size of the step -- this is the
    behavioral signature that distinguishes it from the removed
    ExponentialAngleSmoother, and is exactly what "use max_angular_speed"
    (a literal rad/s bound) calls for."""
    dt = 0.05
    small_step_limiter = SlewRateLimiter(max_rate=2.0, initial_value=0.0)
    big_step_limiter = SlewRateLimiter(max_rate=2.0, initial_value=0.0)

    small_ticks = 0
    while abs(small_step_limiter.step(0.1, dt) - 0.1) > 1e-6:
        small_ticks += 1
        assert small_ticks < 1000

    big_ticks = 0
    while abs(big_step_limiter.step(0.8, dt) - 0.8) > 1e-6:
        big_ticks += 1
        assert big_ticks < 1000

    assert big_ticks > small_ticks, "a bigger step must take proportionally longer to converge"


def test_limiter_reset_clears_state():
    limiter = SlewRateLimiter(max_rate=2.0, initial_value=0.0)
    limiter.step(0.5, 0.05)
    limiter.reset(value=-0.3)
    assert limiter.value == -0.3


def test_limiter_holds_last_value_when_target_is_none():
    limiter = SlewRateLimiter(max_rate=2.0, initial_value=0.2)
    held = limiter.step(None, 0.05)
    assert held == 0.2


# --- manual commands still move smoothly ---------------------------------------

def test_manual_step_command_is_smoothed_same_as_webcam_driven_one():
    """The limiter has no notion of "webcam" vs "manual" -- it smooths
    whatever target value it's given, which is exactly what makes manual
    `ros2 topic pub /attention_target_angle ...` commands move smoothly:
    they arrive on the same subscription and go through this exact same
    limiter instance in eye_gaze_node.py."""
    limiter = SlewRateLimiter(max_rate=2.0, initial_value=0.0)
    dt = 0.05

    # Simulate a manual test script: three step commands, like
    # scripts/manual_eye_test.sh's -0.65 / 0.0 / 0.65 sequence.
    for manual_target in (-0.65, 0.0, 0.65):
        first_tick = limiter.step(manual_target, dt)
        assert first_tick != manual_target, "a manual step command must not snap instantly either"
        # Converge before moving to the next manual command.
        for _ in range(200):
            limiter.step(manual_target, dt)
        assert limiter.value == pytest.approx(manual_target, abs=1e-6)


# --- only one limiter in the whole angle path -----------------------------------

def test_no_second_rate_limiting_class_in_target_angle_module():
    from gaze_attention import target_angle as target_angle_module

    assert not hasattr(target_angle_module, "SlewRateLimiter")
    assert not hasattr(target_angle_module, "ExponentialAngleSmoother")
