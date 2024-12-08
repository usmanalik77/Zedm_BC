import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
import numpy as np

class RGBDConverter(Node):
    def __init__(self):
        super().__init__('rgbd_converter')
        
        # Subscribing to RGB and Depth topics
        self.rgb_sub = self.create_subscription(
            Image,
            '/zed/zed_node/rgb/image_rect_color',
            self.rgb_callback,
            10
        )

        self.depth_sub = self.create_subscription(
            Image,
            '/zed/zed_node/depth/depth_registered',
            self.depth_callback,
            10
        )

        self.rgb_image = None
        self.depth_image = None

    def rgb_callback(self, msg):
        self.get_logger().info("RGB image received")
        # Convert ROS Image message to a NumPy array (RGB image)
        self.rgb_image = self.ros_img_to_numpy(msg, "rgb")

    def depth_callback(self, msg):
        self.get_logger().info("Depth image received")
        # Convert ROS Image message to a NumPy array (Depth image)
        self.depth_image = self.ros_img_to_numpy(msg, "depth")

        # Process the images when both are available
        if self.rgb_image is not None and self.depth_image is not None:
            self.process_rgbd()

    def ros_img_to_numpy(self, msg, img_type):
        if img_type == "rgb":
            channels = 4  # RGBA instead of RGB
            expected_size = msg.height * msg.width * channels
            if len(msg.data) != expected_size:
                self.get_logger().error(f"Buffer size ({len(msg.data)}) does not match expected size ({expected_size}).")
                return None
            # Convert to RGB (strip the alpha channel if needed)
            rgb_image = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, channels)
            rgb_image = rgb_image[:, :, :3]  # Keep only RGB channels
            return rgb_image
        elif img_type == "depth":
            return np.frombuffer(msg.data, dtype=np.float32).reshape(msg.height, msg.width)
    def process_rgbd(self):
        # Ensure the depth image and RGB image have the same size
        if self.depth_image.shape != self.rgb_image.shape[:2]:
            self.get_logger().error("RGB and Depth images do not have the same dimensions!")
            return

        # Expand the depth image to have a third dimension so it can be concatenated
        depth_image_expanded = np.expand_dims(self.depth_image, axis=2)

        # Concatenate the RGB image and depth image along the third dimension
        rgbd_image = np.concatenate((self.rgb_image, depth_image_expanded), axis=2)

        self.get_logger().info(f'RGB-D image shape: {rgbd_image.shape}')

        # Now you can perform further processing on the RGB-D image


def main(args=None):
    rclpy.init(args=args)
    node = RGBDConverter()
    rclpy.spin(node)
    rclpy.shutdown()


if __name__ == '__main__':
    main()
