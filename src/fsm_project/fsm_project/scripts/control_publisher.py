# fsm_project/scripts/control_publisher.py
import rclpy
from rclpy.node import Node
from std_msgs.msg import Int32

class LEDControllerNode(Node):
    def __init__(self):
        super().__init__('led_controller_node')
        self.subscription = self.create_subscription(
            Int32,
            '/fsm_state',
            self.fsm_callback,
            10
        )

    def fsm_callback(self, msg):
        state = msg.data
        if state == 0:
            self.get_logger().info("🟣 Purple LED (Action State)")
        elif state == 1:
            self.get_logger().info("🟠 Orange LED (Monitoring State)")
        elif state == 2:
            self.get_logger().info("🔴 Red LED (Error State)")
        elif state == 3:
            self.get_logger().info("🟢 Green LED (Finished State)")

def main():
    rclpy.init()
    node = LEDControllerNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
