"""
slew_rate.py — the ONE smoothing/rate-limiting stage in the whole angle
path (webcam/policy -> eye command node -> ros_gz_bridge -> Gazebo). Pure
Python, no rclpy dependency, so it's importable and testable on its own
(see multi_person/tests/test_slew_rate_limiter.py) without sourcing ROS.

Used exclusively by eye_gaze_node.py, which is deliberately the ONLY place
in the pipeline that smooths the angle -- main_webcam.py/target_angle.py
publish the RAW (clamped but unsmoothed) angle, and this slew-rate limiter
is what turns that into continuous physical motion, for BOTH the live
webcam path and manual `ros2 topic pub` commands (which never go through
main_webcam.py at all). Do not add a second smoothing stage anywhere else
in this path.
"""

from __future__ import annotations

from typing import Optional


def clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


class SlewRateLimiter:
    """Constant-rate linear ramp toward a target value -- moves by at most
    `max_rate * dt` per call, never an instant snap. Not exponential: unlike
    a low-pass filter, convergence time scales with the size of the step,
    not a fixed time constant, which is what "use max_angular_speed" (a
    literal rad/s bound) calls for."""

    def __init__(self, max_rate: float, initial_value: float = 0.0):
        self.max_rate = max_rate
        self._value = initial_value

    @property
    def value(self) -> float:
        return self._value

    def step(self, target: Optional[float], dt: float) -> float:
        if target is None:
            return self._value
        error = target - self._value
        max_step = self.max_rate * max(dt, 0.0)
        step = clamp(error, -max_step, max_step)
        self._value += step
        return self._value

    def reset(self, value: float = 0.0) -> None:
        self._value = value
