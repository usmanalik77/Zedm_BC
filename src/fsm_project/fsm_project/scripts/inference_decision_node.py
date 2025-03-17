#!/usr/bin/env python3
import os
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Int32
import tensorflow as tf

# Path to your trained model (SavedModel format)
MODEL_PATH = '/home/usman/bc_model'

# Confidence threshold for automatic acceptance as square
SQUARE_THRESHOLD = 0.90

# State definitions
STATE_ACTION    = 0  # Purple LED: Ready to process new image
STATE_MONITOR   = 1  # Orange LED: Monitoring state (waiting for user key input)
STATE_FINISHED  = 2  # Green LED: Correct square shape accepted
STATE_ERROR     = 3  # Red LED: Incorrect shape

class InferenceDecisionNode(Node):
    def __init__(self):
        super().__init__('inference_decision_node')
        self.get_logger().info(f'Loading model from: {MODEL_PATH}')
        self.model = tf.keras.models.load_model(MODEL_PATH)
        self.get_logger().info("Model loaded successfully.")

        self.create_subscription(
            Image,
            '/zed/zed_node/rgb/image_rect_color',
            self.image_callback,
            10
        )
        self.create_subscription(
            Int32,
            '/user_key',
            self.user_key_callback,
            10
        )

        self.state_publisher = self.create_publisher(Int32, '/fsm_state', 10)

        # Internal flags: node is ready to process new image.
        # waiting_for_decision indicates that an image has been processed 
        self.active = True
        self.waiting_for_decision = False

        self.get_logger().info("Inference Decision Node started in ACTION state (0 - Purple LED).")

    def image_callback(self, msg):
        # Process image only if the node is in active .
        if not self.active or self.waiting_for_decision:
            return

        self.waiting_for_decision = True  # Block further image processing until decision is made

        try:
            image = self.decode_image(msg)
            processed_image = self.preprocess_image(image)
            prediction = self.model.predict(processed_image)
            p_square, p_not_square = prediction[0]
            self.get_logger().info(f"Model prediction: square={p_square:.4f}, not-square={p_not_square:.4f}")
            
            # If the model is highly confident for square, auto-accept:
            if p_square >= SQUARE_THRESHOLD:
                self.publish_state(STATE_FINISHED)
                self.get_logger().info("High confidence for 'square'. Entering FINISHED state (2 - Green LED).")
                self.active = False  # Pause processing until resumed via key input
                self.waiting_for_decision = False
            else:
                self.get_logger().warn("Prediction not confident for 'square'. Awaiting user decision on /user_key...")
        except Exception as e:
            self.get_logger().error(f"Error in image callback: {e}")
            self.waiting_for_decision = False

    def user_key_callback(self, msg):
        key = msg.data
        self.get_logger().info(f"Received user key: {key}")

        # Process user key only if waiting for decision
        if self.waiting_for_decision:
            if key == 1:
                # User indicates to continue monitoring
                self.publish_state(STATE_MONITOR)
                self.get_logger().info("User set state to MONITOR (1 - Orange LED). Ready for a new image.")
                self.waiting_for_decision = False
                self.active = True
            elif key == 2:
                # User forces acceptance as square:  Finished state.
                self.publish_state(STATE_FINISHED)
                self.get_logger().info("User forced shape as 'square'. Entering FINISHED state (2 - Green LED).")
                self.waiting_for_decision = False
                self.active = False
            elif key == 3:
                # User forces rejection (not-square):  Error state.
                self.publish_state(STATE_ERROR)
                self.get_logger().info("User forced shape as 'not-square'. Entering ERROR state (3 - Red LED).")
                self.waiting_for_decision = False
                self.active = False
            elif key == 0:
                # User requests to reset/resume processing (Action state)
                self.publish_state(STATE_ACTION)
                self.get_logger().info("User reset to ACTION state (0 - Purple LED). Ready for a new image.")
                self.waiting_for_decision = False
                self.active = True
            else:
                self.get_logger().warn("User key not recognized. No state change.")
        else:
            # If not waiting for a decision, allow key 0 to resume processing.
            if key == 0:
                self.publish_state(STATE_ACTION)
                self.get_logger().info("User set to ACTION state (0 - Purple LED). Resuming image processing.")
                self.active = True

    def decode_image(self, msg):
        if msg.encoding != 'bgra8':
            raise ValueError(f"Unsupported encoding: {msg.encoding}")
        raw = np.frombuffer(msg.data, dtype=np.uint8)
        image_4ch = raw.reshape((msg.height, msg.width, 4))
        image_rgb = image_4ch[:, :, :3][:, :, ::-1]
        return image_rgb

    def preprocess_image(self, image):
        resized = tf.image.resize(image, (360, 640))
        normalized = resized / 255.0
        return np.expand_dims(normalized.numpy(), axis=0)

    def publish_state(self, state_value):
        msg = Int32()
        msg.data = state_value
        self.state_publisher.publish(msg)
        self.get_logger().info(f"Published FSM state: {state_value}")

def main(args=None):
    rclpy.init(args=args)
    node = InferenceDecisionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
