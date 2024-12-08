import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray

class ControlPublisher(Node):
    def __init__(self):
        super().__init__('control_publisher')
        self.publisher_ = self.create_publisher(Float32MultiArray, '/behavior_controls', 10)
        self.timer = self.create_timer(1.0, self.timer_callback)  # Adjust the frequency as needed

    def timer_callback(self):
        msg = Float32MultiArray()
        msg.data = [0.5, 0.3]  # Example control data
        self.publisher_.publish(msg)
        self.get_logger().info('Publishing: "%s"' % msg.data)

def main(args=None):
    rclpy.init(args=args)
    control_publisher = ControlPublisher()
    rclpy.spin(control_publisher)
    control_publisher.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
