# fsm_project/observation_logger/observation_logger.py

import os
import csv
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Int32

class ObservationLogger(Node):
    def __init__(self):
        super().__init__('observation_logger')

        self.data_dir = "/home/usman/duplo_data"
        os.makedirs(self.data_dir, exist_ok=True)

        # Open CSV
        csv_path = os.path.join(self.data_dir, "data_logs.csv")
        self.csv_file = open(csv_path, 'w', newline='')
        self.csv_writer = csv.writer(self.csv_file)
        self.csv_writer.writerow(["image_index", "label", "state"])

        self.image_index = 0
        self.latest_image = None

        # monitor_active means “waiting for user to press 2 or 3”:
        self.monitor_active = False

        # Subscriptions
        self.image_sub = self.create_subscription(
            Image,
            '/zed/zed_node/rgb/image_rect_color',
            self.image_callback,
            10
        )
        self.key_sub = self.create_subscription(
            Int32,
            '/user_key',
            self.key_callback,
            10
        )

        self.get_logger().info(
            "Observation Logger Node started. Waiting for user keys + camera images."
        )

    def destroy_node(self):
        self.csv_file.close()
        self.get_logger().info("Observation Logger Node shutting down.")
        super().destroy_node()

    def image_callback(self, msg):
        """
        latest camera frame so that if the user
        presses 2 or 3 while monitor_active, we can save it.
        """
        self.latest_image = msg

    def key_callback(self, msg):
        """
        - 1 => go into “monitor mode” (monitor_active = True)
        - 2 => if in monitor mode, save image as label 'square'
        - 3 => if in monitor mode, save image as label 'not-square'
        """
        key = msg.data
        self.get_logger().info(f"Received user key: {key}")

        if key == 1:
            self.monitor_active = True
            self.get_logger().info("monitor_active = True (Press 2 or 3 to save image)")
        
        elif key in (2, 3):
            if self.monitor_active and self.latest_image is not None:
                label = "square" if key == 2 else "not-square"
                self.save_image_and_log(label, state=1)
                # After one save, we exit monitor mode (adjust if you want multiple saves)
                self.monitor_active = False
            else:
                # Either not in monitor mode or no image to save
                self.get_logger().info("Ignoring key=2 or 3, because monitor is off or no latest image.")
        
        else:
            # If key=4,5,... we do nothing here
            pass

    def save_image_and_log(self, label, state=1):
        if self.latest_image is None:
            self.get_logger().warn("No image to save (latest_image is None).")
            return

        # Decode BGRA8 to shape(H, W, 3) in RGB or BGR
        decoded = self.decode_bgra8(self.latest_image)

        filename = f"image_{self.image_index}.npy"
        path = os.path.join(self.data_dir, filename)
        np.save(path, decoded)

        self.get_logger().info(f"Saved {filename} with label={label}, state={state}")
        self.csv_writer.writerow([self.image_index, label, state])
        self.csv_file.flush()

        self.image_index += 1
        self.latest_image = None

    def decode_bgra8(self, msg):
        if msg.encoding != 'bgra8':
            raise ValueError(f"Unsupported encoding: {msg.encoding}")

        raw = np.frombuffer(msg.data, dtype=np.uint8)
        image_4ch = raw.reshape((msg.height, msg.width, 4))

        image_rgb = image_4ch[:, :, :3][:, :, ::-1]
        return image_rgb


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


if __name__ == "__main__":
    main()
