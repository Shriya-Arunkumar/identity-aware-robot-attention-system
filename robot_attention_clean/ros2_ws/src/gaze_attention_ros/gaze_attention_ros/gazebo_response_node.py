"""
gazebo_response_node.py — subscribes to the attention target, smoothly moves
a simulated robot head toward it, and keeps working (printing simulated
movement) if Gazebo isn't running.

Subscribes:
    /attention_target_id     (std_msgs/Int32)   — selected person id (WHO;
                              drives policy state -- engagement/debt/etc,
                              see policies.py), or absent if none.
    /attention_target_angle  (std_msgs/Float32) — that person's CURRENT
                              face-position-derived angle (WHERE; see
                              gaze_attention/target_angle.py), independent
                              of how long they've held the target id.

Publishes:
    /gaze_attention/head_yaw_cmd (std_msgs/Float64) — smoothed head yaw command.
    Bridged into Gazebo Sim's JointPositionController plugin on the
    simple_robot model's head_yaw joint by ros_gz_bridge (see
    launch/attention_gazebo.launch.py) — this node has no Gazebo-specific
    code at all, it only knows about a ROS 2 topic. If nothing is bridged
    (Gazebo not running, or --no-gazebo), publishing is a harmless no-op and
    this node detects that (via publisher subscriber count) to print
    simulated-movement text instead, per the project's fallback requirement.

Motion model: linear rate-limited approach toward the target angle
(`max_angular_speed` rad/s), not an instant snap — "smoothly move the
robot head/base toward the target" per spec.
"""

from __future__ import annotations

from typing import Optional

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32, Float64, Int32


class GazeboResponseNode(Node):
    def __init__(self):
        super().__init__("gazebo_response_node")

        self.declare_parameter("max_angular_speed", 2.0)  # rad/s
        self.declare_parameter("control_rate_hz", 20.0)
        self.declare_parameter("log_every_n_ticks", 10)

        self._max_speed = self.get_parameter("max_angular_speed").value
        control_rate = self.get_parameter("control_rate_hz").value
        self._log_every_n = int(self.get_parameter("log_every_n_ticks").value)

        self._current_angle = 0.0
        self._target_angle: Optional[float] = None
        self._current_target_id: Optional[int] = None
        self._tick = 0
        self._logged_reached = False

        self._angle_cmd_pub = self.create_publisher(Float64, "/gaze_attention/head_yaw_cmd", 10)
        self.create_subscription(Int32, "/attention_target_id", self._on_target_id, 10)
        self.create_subscription(Float32, "/attention_target_angle", self._on_target_angle, 10)

        self.create_timer(1.0 / control_rate, self._on_timer)

        self.get_logger().info(
            f"gazebo_response_node up: max_angular_speed={self._max_speed} rad/s, "
            f"control_rate={control_rate} Hz. Waiting for /attention_target_id, /attention_target_angle."
        )

    def _on_target_id(self, msg: Int32) -> None:
        if msg.data != self._current_target_id:
            self.get_logger().info(f"target changed: person {msg.data}")
            self._current_target_id = msg.data

    def _on_target_angle(self, msg: Float32) -> None:
        if self._target_angle is None or abs(msg.data - self._target_angle) > 1e-6:
            self.get_logger().info(f"target angle -> {msg.data:.3f} rad")
            self._logged_reached = False  # new target: allow a fresh "reached" log later
        self._target_angle = float(msg.data)

    def _on_timer(self) -> None:
        if self._target_angle is None:
            return

        dt = 1.0 / self.get_parameter("control_rate_hz").value
        error = self._target_angle - self._current_angle
        max_step = self._max_speed * dt
        step = max(-max_step, min(max_step, error))
        self._current_angle += step

        self._angle_cmd_pub.publish(Float64(data=self._current_angle))

        bridge_connected = self._angle_cmd_pub.get_subscription_count() > 0
        self._tick += 1
        just_reached = abs(error) < 1e-3 and not self._logged_reached
        if just_reached:
            self._logged_reached = True
        if just_reached or self._tick % self._log_every_n == 0:
            status = "Gazebo bridge connected" if bridge_connected else "no Gazebo bridge detected, simulated only"
            self.get_logger().info(
                f"head yaw: {self._current_angle:+.3f} rad -> target {self._target_angle:+.3f} rad ({status})"
            )


def main(args=None):
    rclpy.init(args=args)
    node = GazeboResponseNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
