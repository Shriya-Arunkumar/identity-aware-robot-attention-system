"""
eye_gaze_node.py — ROS 2 side of the single-eye Gazebo CLASSIC 11 demo
(gazebo/models/eye/, gazebo/worlds/eye_test.world).

Architecture note: Gazebo Classic has no ros_gz_bridge-style mechanism to
let a plain ROS 2 topic drive a joint directly -- something running INSIDE
the gzserver process has to do that. That's ros2_ws/src/eye_yaw_plugin (a
custom C++ model plugin, see its own docstring): it subscribes directly to
/eye/target_angle (std_msgs/Float64, radians) and drives eye_yaw_joint's
position PID every physics step. This node is NOT required for the eye to
move -- `ros2 topic pub /eye/target_angle ...` alone is enough, which is
exactly what makes manual testing possible without the webcam pipeline (see
the REQUIRED TEST section of the task this was built against). This node's
job is purely the smoothing/clamping layer between the wider attention
system and that raw topic.

Subscribes:
    /attention_target_angle (std_msgs/Float32) — desired gaze angle, rad.
    This is the SAME topic the existing attention pipeline (main_webcam.py's
    RosPublisher / gazebo_response_node.py) already publishes to -- so
    nothing upstream needs to change. To command the eye directly (bypassing
    this node and its smoothing) publish std_msgs/Float64 straight to
    /eye/target_angle instead; see the plugin's docstring.

Publishes:
    /eye/target_angle (std_msgs/Float64) — smoothed, clamped angle, radians.
    Read directly by eye_yaw_plugin inside gzserver.

Behavior:
    - Clamp: every incoming angle is clamped to [angle_min, angle_max]
      (ROS params, default -0.7854/+0.7854 rad = +-45 deg) before anything
      else touches it -- an out-of-range command from upstream can never
      reach the joint. (The plugin also clamps independently, so a direct
      manual publish to /eye/target_angle is clamped too.)
    - Smooth: NOT an instant snap -- exactly ONE slew-rate limiter
      (slew_rate.SlewRateLimiter, a rate-limited linear approach toward the
      clamped target at up to max_angular_speed rad/s, recomputed every
      control tick at control_rate_hz). This is the ONLY smoothing stage in
      the whole angle path (main_webcam.py/target_angle.py publish the RAW,
      unsmoothed angle upstream -- see target_angle.py's module docstring)
      -- it applies equally to webcam-driven commands and to manual
      `ros2 topic pub /attention_target_angle ...` commands, since both
      arrive on the same subscription. Do not add smoothing anywhere else
      in this path.
    - No-Gazebo fallback: if nothing is subscribed downstream (Gazebo not
      running), publishing is a harmless no-op; this node detects that via
      publisher subscriber count and logs simulated movement instead of
      silently doing nothing, matching this project's "every component
      needs a simple fallback" rule (CLAUDE.md) -- it never blocks or
      errors just because Gazebo isn't up.

This node is intentionally independent of gazebo_response_node.py (the
existing single whole-head-yaw demo, gazebo/worlds/attention_demo.sdf) --
neither replaces the other.
"""

from __future__ import annotations

from typing import Optional

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32, Float64

from gaze_attention_ros.slew_rate import SlewRateLimiter, clamp

# Manual-test convenience mapping only -- NOT read by this node at runtime;
# the node only ever acts on the angle it's actually sent on
# /attention_target_angle. See scripts/manual_eye_test.sh, which uses this
# same mapping to drive `ros2 topic pub` directly.
TARGET_ANGLE_MAP = {0: -0.65, 1: 0.0, 2: 0.65}

# A clamp warning only means something if the difference is bigger than
# float32-round-trip noise: /attention_target_angle is std_msgs/Float32, so
# by the time msg.data reaches Python it's already been narrowed to 32-bit
# precision (e.g. -0.65 can decode as -0.6500000...119) -- comparing that
# against a 64-bit clamp bound with `!=` would spuriously "warn" on every
# single in-range message. A real clamp (e.g. 1.5 -> 0.8) differs by orders
# of magnitude more than this tolerance.
_CLAMP_WARN_TOLERANCE = 1e-4


class EyeGazeNode(Node):
    def __init__(self):
        super().__init__("eye_gaze_node")

        self.declare_parameter("angle_min", -0.7854)  # -45 deg
        self.declare_parameter("angle_max", 0.7854)  # +45 deg
        self.declare_parameter("max_angular_speed", 2.0)  # rad/s
        self.declare_parameter("control_rate_hz", 20.0)
        self.declare_parameter("log_every_n_ticks", 10)

        self._angle_min = float(self.get_parameter("angle_min").value)
        self._angle_max = float(self.get_parameter("angle_max").value)
        if self._angle_min > self._angle_max:
            raise ValueError(f"angle_min ({self._angle_min}) must be <= angle_max ({self._angle_max})")
        self._max_speed = float(self.get_parameter("max_angular_speed").value)
        self._control_rate = float(self.get_parameter("control_rate_hz").value)
        self._log_every_n = int(self.get_parameter("log_every_n_ticks").value)

        self._limiter = SlewRateLimiter(max_rate=self._max_speed, initial_value=0.0)
        self._target_angle: Optional[float] = None
        self._tick = 0
        self._logged_reached = False

        self._target_angle_pub = self.create_publisher(Float64, "/eye/target_angle", 10)
        self.create_subscription(Float32, "/attention_target_angle", self._on_target_angle, 10)

        self.create_timer(1.0 / self._control_rate, self._on_timer)

        self.get_logger().info(
            f"eye_gaze_node up: angle range=[{self._angle_min:+.3f}, {self._angle_max:+.3f}] rad, "
            f"max_angular_speed={self._max_speed} rad/s, control_rate={self._control_rate} Hz. "
            f"Waiting for /attention_target_angle."
        )

    def _on_target_angle(self, msg: Float32) -> None:
        clamped = clamp(float(msg.data), self._angle_min, self._angle_max)
        if abs(clamped - msg.data) > _CLAMP_WARN_TOLERANCE:
            self.get_logger().warn(f"target angle {msg.data:.3f} rad clamped to {clamped:+.3f} rad")
        if self._target_angle is None or abs(clamped - self._target_angle) > 1e-6:
            self.get_logger().info(f"target angle -> {clamped:+.3f} rad")
            self._logged_reached = False
        self._target_angle = clamped

    def _on_timer(self) -> None:
        if self._target_angle is None:
            return

        dt = 1.0 / self._control_rate
        error = self._target_angle - self._limiter.value
        current_angle = self._limiter.step(self._target_angle, dt)

        self._target_angle_pub.publish(Float64(data=current_angle))

        gazebo_connected = self._target_angle_pub.get_subscription_count() > 0
        self._tick += 1
        just_reached = abs(error) < 1e-3 and not self._logged_reached
        if just_reached:
            self._logged_reached = True
        if just_reached or self._tick % self._log_every_n == 0:
            status = "eye_yaw_plugin connected" if gazebo_connected else "no Gazebo subscriber detected, simulated only"
            self.get_logger().info(
                f"eye yaw: {current_angle:+.3f} rad -> target {self._target_angle:+.3f} rad ({status})"
            )


def main(args=None):
    rclpy.init(args=args)
    node = EyeGazeNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
