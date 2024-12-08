import os
import time
import numpy as np
import rclpy
from rclpy.node import Node
import tensorflow as tf
from sensor_msgs.msg import Image
from zedm_to_BC.models.behavior_cloning_model import ConvCfCModel

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TensorFlow warnings


class BehaviorCloningNode(Node):
    def __init__(self):
        super().__init__('behavior_cloning_node')

        # Subscribe to ZED Mini RGB image topic
        self.subscription = self.create_subscription(
            Image,
            '/zed/zed_node/rgb/image_rect_color',  # Replace with your ZED Mini topic
            self.image_callback,
            10
        )

        # Flag to determine if we are in training or inference mode
        self.is_collecting_data = True  # Change to `False` for inference mode
        self.collected_data_path = "collected_data.npy"  # Path for saving collected data

        if self.is_collecting_data:
            # Initialize data collection buffer
            self.data_buffer = []
            self.get_logger().info("Behavior Cloning Node started in DATA COLLECTION mode.")
        else:
            # Load pre-trained model for inference
            self.model = tf.keras.models.load_model("behavior_cloning_model.h5")
            self.get_logger().info("Behavior Cloning Node started in INFERENCE mode.")

    def image_callback(self, msg):
        try:
            # Decode the image
            image = self.decode_image(msg)

            # Preprocess image
            image_resized = self.resize_image(image, (224, 224))
            image_normalized = image_resized / 255.0  # Normalize
            input_image = np.expand_dims(image_normalized, axis=0)  # Add batch dimension

            if self.is_collecting_data:
                # Simulate label collection (replace with actual labels from your setup)
                steering = np.random.uniform(-1, 1)  # Example steering label
                throttle = np.random.uniform(0, 1)  # Example throttle label

                # Append image and labels to data buffer
                self.data_buffer.append((image_normalized, [steering, throttle]))

                # Save data every 100 samples
                if len(self.data_buffer) >= 100:
                    self.save_collected_data()
            else:
                # Use the model for inference
                logits = self.model.predict(input_image)
                steering, throttle = logits[0]
                self.get_logger().info(f"Predicted Actions: Steering={steering:.2f}, Throttle={throttle:.2f}")

        except Exception as e:
            self.get_logger().error(f"Error processing image: {e}")

    def decode_image(self, msg):
        """
        Decode the sensor_msgs/Image message to a NumPy array.
        """
        if msg.encoding == 'bgra8':
            # BGRA8 has 4 channels: Blue, Green, Red, Alpha
            image = np.frombuffer(msg.data, dtype=np.uint8).reshape((msg.height, msg.width, 4))
            # Convert BGRA to RGB
            image = image[:, :, :3][:, :, ::-1]  # Drop alpha channel and convert to RGB
        else:
            raise ValueError(f"Unsupported encoding: {msg.encoding}")
        return image

    def resize_image(self, image, size):
        """
        Resize the image using TensorFlow.
        """
        resized_image = tf.image.resize(image, size)
        return resized_image.numpy()

    def save_collected_data(self):
        """
        Save the collected data to a binary .npy file for fast reading during training.
        """
        if os.path.exists(self.collected_data_path):
            # Load existing data and append new data
            existing_data = np.load(self.collected_data_path, allow_pickle=True)
            combined_data = np.concatenate([existing_data, self.data_buffer])
        else:
            combined_data = self.data_buffer

        # Save combined data to file
        np.save(self.collected_data_path, combined_data)
        self.get_logger().info(f"Saved {len(self.data_buffer)} samples to {self.collected_data_path}")
        self.data_buffer = []  # Clear buffer

    def run(self):
        rclpy.spin(self)  # Keep the node spinning


# Debug GPU availability
print("TensorFlow Version: ", tf.__version__)
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
print("GPU Devices: ", tf.config.list_physical_devices('GPU'))


def main(args=None):
    rclpy.init(args=args)
    node = BehaviorCloningNode()
    try:
        node.run()
    except KeyboardInterrupt:
        pass
    finally:
        if node.is_collecting_data and node.data_buffer:
            # Save remaining data before shutting down
            node.save_collected_data()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
