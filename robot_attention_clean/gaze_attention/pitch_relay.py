import rclpy
from rclpy.node import Node

from std_msgs.msg import Float32, Float64


class PitchRelay(Node):

    def __init__(self):
        super().__init__("pitch_relay")

        self.publisher = self.create_publisher(
            Float64,
            "/eye/pitch_angle",
            10
        )

        self.subscription = self.create_subscription(
            Float32,
            "/attention_target_pitch",
            self.pitch_callback,
            10
        )

        self.get_logger().info(
            "Pitch relay: /attention_target_pitch -> /eye/pitch_angle"
        )

    def pitch_callback(self, msg):
        output = Float64()
        output.data = float(msg.data)

        self.publisher.publish(output)


def main(args=None):
    rclpy.init(args=args)

    node = PitchRelay()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass

    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()