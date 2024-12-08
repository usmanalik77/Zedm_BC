import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from matplotlib import pyplot as plt
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
import numpy as np
import os

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
            depth_image = np.frombuffer(msg.data, dtype=np.float32).reshape(msg.height, msg.width)
            # Replace NaNs and Infs in depth image
            depth_image = np.nan_to_num(depth_image, nan=0.0, posinf=0.0, neginf=0.0)
            return depth_image

    def process_rgbd(self):
        # Ensure the depth image and RGB image have the same size
        if self.depth_image.shape != self.rgb_image.shape[:2]:
            self.get_logger().error("RGB and Depth images do not have the same dimensions!")
            return

        # Expand the depth image to have a third dimension so it can be concatenated
        depth_image_expanded = np.expand_dims(self.depth_image, axis=2)

        # Concatenate the RGB image and depth image along the third dimension
        rgbd_image = np.concatenate((self.rgb_image, depth_image_expanded), axis=2)

        # Check for invalid values
        if np.any(np.isnan(rgbd_image)) or np.any(np.isinf(rgbd_image)):
            self.get_logger().error("RGB-D image contains NaNs or Infs after concatenation.")
            rgbd_image = np.nan_to_num(rgbd_image, nan=0.0, posinf=0.0, neginf=0.0)

        # Log statistics for debugging
        self.get_logger().info(f'Min value: {np.min(rgbd_image)}, Max value: {np.max(rgbd_image)}')

        # Save the RGB-D image to a PDF
        self.save_rgbd_to_pdf(rgbd_image)

    def save_rgbd_to_pdf(self, rgbd_image):
        pdf_path = '/tmp/rgbd_image.pdf'
        png_path = '/tmp/rgbd_image.png'
        
        # Handle invalid values
        if np.any(np.isnan(rgbd_image)) or np.any(np.isinf(rgbd_image)):
            self.get_logger().error("RGB-D image contains NaNs or Infs. Cleaning up data.")
            rgbd_image = np.nan_to_num(rgbd_image, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Normalize image to [0, 255] and handle RGB-D image format
        if rgbd_image.dtype != np.uint8:
            rgbd_image = np.clip(rgbd_image, 0, 255).astype(np.uint8)
        
        # Save RGB-D image to file
        fig, ax = plt.subplots()
        ax.imshow(rgbd_image[:, :, :3])  # Display only RGB part
        ax.axis('off')
        fig.savefig(png_path, bbox_inches='tight', pad_inches=0)
        plt.close(fig)

        # Create PDF
        c = canvas.Canvas(pdf_path, pagesize=letter)
        width, height = letter

        # Draw the image on the PDF
        c.drawImage(png_path, 0, 0, width, height)
        c.save()

        # Clean up
        os.remove(png_path)

        self.get_logger().info(f"PDF created and saved at {pdf_path}.")

def main(args=None):
    rclpy.init(args=args)
    node = RGBDConverter()
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == '__main__':
    main()
