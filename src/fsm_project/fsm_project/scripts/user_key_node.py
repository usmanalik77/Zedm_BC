#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from std_msgs.msg import Int32

class UserKeyNode(Node):
    def __init__(self):
        super().__init__('user_key_node')
        self.publisher_ = self.create_publisher(Int32, '/user_key', 10)
        self.get_logger().info("UserKeyNode started. Enter keys as follows:")
        self.get_logger().info("0: Reset/Action (Purple), 1: Monitor (Orange), 2: Square (Finished - Green), 3: Not-Square (Error - Red)")

def main(args=None):
    rclpy.init(args=args)
    node = UserKeyNode()
    try:
        while rclpy.ok():
            rclpy.spin_once(node, timeout_sec=0.01)
            user_input = input("Enter key [0=Action, 1=Monitor, 2=Square, 3=Not-Square], q=quit: ").strip()
            if user_input.lower() == 'q':
                break

            try:
                key_val = int(user_input)
                msg = Int32()
                msg.data = key_val
                node.publisher_.publish(msg)
                node.get_logger().info(f"Published user key: {key_val}")
            except ValueError:
                node.get_logger().warn(f"Invalid input: {user_input}")
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
