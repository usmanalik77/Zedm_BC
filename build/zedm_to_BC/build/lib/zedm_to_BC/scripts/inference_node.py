import os
import csv
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Float32MultiArray
import tensorflow as tf

MODEL_PATH = '/home/usman/behavior_cloning_model'
LOG_DIR = '/home/usman/inference_logs'

class InferenceNode(Node):
    def __init__(self):
        super().__init__('inference_node')

        # Load the pre-trained model
        self.get_logger().info(f'Loading model from: {MODEL_PATH}')
        self.model = tf.keras.models.load_model(MODEL_PATH)
        
        # Ensure the log directory exists
        os.makedirs(LOG_DIR, exist_ok=True)
        self.log_file = open(os.path.join(LOG_DIR, 'inference_outputs.csv'), 'w', newline='')
        self.log_writer = csv.writer(self.log_file)
        self.log_writer.writerow(['timestamp_sec', 'timestamp_nanosec', 'steering', 'throttle'])

        # Subscribe to the RGB image topic
        self.image_subscription = self.create_subscription(
            Image,
            '/zed/zed_node/rgb/image_rect_color',
            self.image_callback,
            10
        )

        # Publish the predicted controls
        self.control_publisher = self.create_publisher(Float32MultiArray, '/behavior_controls', 10)

    def image_callback(self, msg):
        # Decode and preprocess the image
        try:
            image = self.decode_image(msg)
            processed_image = self.preprocess_image(image)

            # Predict control commands
            control_prediction = self.model.predict(processed_image)

            # Publish the predicted controls
            self.publish_controls(control_prediction)

            # Log the predictions
            self.log_predictions(msg, control_prediction)
        except Exception as e:
            self.get_logger().error(f"Error in image callback: {e}")

    def decode_image(self, msg):
        """
        Decode the sensor_msgs/Image message to a NumPy array.
        """
        if msg.encoding == 'bgra8':
            image = np.frombuffer(msg.data, dtype=np.uint8).reshape((msg.height, msg.width, 4))
            image = image[:, :, :3][:, :, ::-1]  # Convert BGRA to RGB
        else:
            raise ValueError(f"Unsupported encoding: {msg.encoding}")
        return image

    def preprocess_image(self, image):
        """
        Preprocess the image for the model.
        - Resize, normalize, and expand dimensions.
        """
        resized_image = tf.image.resize(image, (360, 640))  # Use the training input size
        normalized_image = resized_image / 255.0  # Normalize to [0, 1]
        return np.expand_dims(normalized_image, axis=0)  # Add batch dimension

    def publish_controls(self, prediction):
        """
        Publish the predicted steering and throttle.
        """
        msg = Float32MultiArray()
        msg.data = prediction[0].tolist()  # Convert prediction to list
        self.control_publisher.publish(msg)
        self.get_logger().info(f"Published controls: {msg.data}")

    def log_predictions(self, msg, prediction):
        """
        Log the predictions (steering, throttle) with the timestamp into a CSV file.
        """
        timestamp = msg.header.stamp  # Extract timestamp from the image message
        log_entry = [timestamp.sec, timestamp.nanosec, prediction[0][0], prediction[0][1]]
        self.log_writer.writerow(log_entry)
        self.get_logger().info(f"Logged predictions: {log_entry}")

    def destroy_node(self):
        # Close the log file upon shutdown
        self.log_file.close()
        self.get_logger().info('Inference Node shutting down.')
        super().destroy_node()
    
def main(args=None):
    rclpy.init(args=args)
    node = InferenceNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
