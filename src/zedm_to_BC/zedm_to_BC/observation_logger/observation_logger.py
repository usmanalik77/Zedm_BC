import os
import csv
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Float32MultiArray

class ObservationLogger(Node):
    def __init__(self):
        super().__init__('observation_logger')

        self.log_dir = '/home/usman/behavior_logs'
        os.makedirs(self.log_dir, exist_ok=True)

        self.image_subscription = self.create_subscription(
            Image,
            '/zed/zed_node/rgb/image_rect_color',
            self.image_callback,
            10
        )
        
        self.controls_subscription = self.create_subscription(
            Float32MultiArray,
            '/behavior_controls',
            self.controls_callback,
            10
        )

        self.image_index = 0
        self.last_image = None
        self.last_controls = None

        self.control_log_file = open(os.path.join(self.log_dir, 'control_logs.csv'), 'w', newline='')
        self.control_log_writer = csv.writer(self.control_log_file)
        self.control_log_writer.writerow(['image_index', 'sec', 'nanosec', 'steering', 'throttle'])

        self.get_logger().info('Observation Logger Node Started.')

        # Timer to process observations every second
        self.timer = self.create_timer(1.0, self.process_observations)

    def image_callback(self, msg):
        self.last_image = msg
        self.get_logger().info('Image received')

    def controls_callback(self, msg):
        self.last_controls = msg
        self.get_logger().info('Controls received')

    def process_observations(self):
        self.get_logger().info('Processing observations')
        if self.last_image and self.last_controls:
            image = self.decode_image(self.last_image)
            image_filename = os.path.join(self.log_dir, f'image_{self.image_index}.npy')
            np.save(image_filename, image)
            self.get_logger().info(f'Saved image: {image_filename}')

            control_data = self.last_controls.data
            timestamp = self.get_clock().now().to_msg()
            log_entry = [self.image_index, timestamp.sec, timestamp.nanosec, *control_data]
            self.control_log_writer.writerow(log_entry)
            self.control_log_file.flush()

            self.get_logger().info(f"Control data written to CSV for image index {self.image_index}.")
            self.image_index += 1
            self.last_image = None
            self.last_controls = None

    def decode_image(self, msg):
        if msg.encoding == 'bgra8':
            image = np.frombuffer(msg.data, dtype=np.uint8).reshape((msg.height, msg.width, 4))
            image = image[:, :, :3][:, :, ::-1]
            return image
        else:
            raise ValueError(f"Unsupported encoding: {msg.encoding}")

    def destroy_node(self):
        self.control_log_file.close()
        self.get_logger().info('Observation Logger Node shutting down.')
        super().destroy_node()

def main(args=None):
    rclpy.init(args=args)
    node = ObservationLogger()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
