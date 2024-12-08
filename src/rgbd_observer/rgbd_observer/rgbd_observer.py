import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from pdf2image import convert_from_path
from PIL import Image as PILImage
import numpy as np
import time
from matplotlib import pyplot as plt

class RGBDObserver(Node):
    def __init__(self):
        super().__init__('rgbd_observer')
        
        # Initialize the subscription to RGB and Depth topics
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

        # Initialize the PDF to image conversion
        self.pdf_path = '/home/usman/Downloads/example.pdf'
        self.image_matrix = self.convert_pdf_to_matrix()

        # Initialize storage for RGB-D data and observation vectors
        self.rgb_image = None
        self.depth_image = None
        self.observation_list = []

        # Create a timer to store observations every 500ms
        self.create_timer(0.5, self.store_observation)

    def rgb_callback(self, msg):
        self.rgb_image = self.ros_img_to_numpy(msg, img_type="rgb")
        self.get_logger().info("RGB image received")

    def depth_callback(self, msg):
        self.depth_image = self.ros_img_to_numpy(msg, img_type="depth")
        self.get_logger().info("Depth image received")

    def ros_img_to_numpy(self, msg, img_type):
        if img_type == "rgb":
            channels = 4
            rgb_image = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, channels)
            return rgb_image[:, :, :3]  # Keep only RGB channels
        elif img_type == "depth":
            depth_image = np.frombuffer(msg.data, dtype=np.float32).reshape(msg.height, msg.width)
            return np.nan_to_num(depth_image, nan=0.0, posinf=0.0, neginf=0.0)

    def convert_pdf_to_matrix(self):
        # Convert the PDF to an image (taking the first page)
        images = convert_from_path(self.pdf_path)
        if images:
            img = images[0]
            img = img.convert('L')  # Convert to grayscale
            image_matrix = np.array(img)
            return image_matrix
        else:
            self.get_logger().error("Failed to convert PDF to image.")
            return None

    def store_observation(self):
        if self.rgb_image is not None and self.depth_image is not None:
            # Normalize depth image to [0, 255] range for better visualization
            normalized_depth = np.clip(self.depth_image, 0, np.max(self.depth_image))
            normalized_depth = (255 * (normalized_depth - np.min(normalized_depth)) / np.ptp(normalized_depth)).astype(np.uint8)
            
            # Convert normalized depth to a 3-channel grayscale image
            depth_colored = np.stack([normalized_depth] * 3, axis=-1)

            # Combine RGB and Depth images into a 4D vector (with depth as a separate image)
            rgbd_image = self.rgb_image

            # Create the observation vector by combining RGB-D and PDF image
            observation_vector = [rgbd_image, depth_colored, self.image_matrix]

            # Store the observation
            self.observation_list.append(observation_vector)
            self.get_logger().info(f"Stored observation, total count: {len(self.observation_list)}")
        else:
            self.get_logger().info("Waiting for both RGB and Depth images to be available.")

    def visualize_observation(self, observation_index):
        if observation_index < len(self.observation_list):
            rgb_image, depth_image, pdf_image = self.observation_list[observation_index]

            # Display RGB-D image
            plt.figure(figsize=(15, 5))

            plt.subplot(1, 3, 1)
            plt.imshow(rgb_image)
            plt.title('RGB Image')

            plt.subplot(1, 3, 2)
            plt.imshow(depth_image, cmap='gray')
            plt.title('Depth Image')

            plt.subplot(1, 3, 3)
            plt.imshow(pdf_image, cmap='gray')
            plt.title('PDF Image')

            plt.show()
        else:
            self.get_logger().info("Observation index out of range.")

def main(args=None):
    rclpy.init(args=args)
    node = RGBDObserver()

    # Run the node in a separate thread to keep spinning
    import threading
    thread = threading.Thread(target=rclpy.spin, args=(node,))
    thread.start()

    # After storing some observations, visualize the first one
    time.sleep(5)  # Wait for some observations to be stored
    node.visualize_observation(0)  # Visualize the first stored observation

    # Shutdown
    rclpy.shutdown()
    thread.join()

if __name__ == '__main__':
    main()
