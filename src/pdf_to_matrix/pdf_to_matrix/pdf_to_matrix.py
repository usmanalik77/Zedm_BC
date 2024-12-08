import rclpy
from rclpy.node import Node
from pdf2image import convert_from_path
from PIL import Image
import numpy as np
import os
import matplotlib.pyplot as plt

class PDFToMatrix(Node):
    def __init__(self):
        super().__init__('pdf_to_matrix')
        
        # Path to the PDF file
        self.pdf_path = '/home/usman/Downloads/example.pdf'  # Replace with your PDF path
        self.jpg_path = '/home/usman/image_files/output_image.jpg'  # Temporary path to save the JPG

        # Convert PDF to JPG
        self.convert_pdf_to_jpg()

        # Convert JPG to a matrix
        self.image_matrix = self.convert_jpg_to_matrix()

        if self.image_matrix is not None:
            self.get_logger().info(f"Matrix shape: {self.image_matrix.shape}")
            # Visualize the matrix
            self.visualize_matrix(self.image_matrix)
        else:
            self.get_logger().error("Failed to convert the image to a matrix.")

    def convert_pdf_to_jpg(self):
        # Convert the PDF to a list of images (one image per page)
        images = convert_from_path(self.pdf_path)

        # Save the first page as a JPG image
        if images:
            images[0].save(self.jpg_path, 'JPEG')
            self.get_logger().info(f"PDF converted to JPG at: {self.jpg_path}")
        else:
            self.get_logger().error("Failed to convert PDF to image.")

    def convert_jpg_to_matrix(self):
        try:
            # Open the image using Pillow (PIL)
            with Image.open(self.jpg_path) as img:
                # Convert the image to grayscale (optional)
                img = img.convert('L')  # 'L' mode is for grayscale

                # Convert the image to a NumPy array (matrix)
                image_matrix = np.array(img)

            return image_matrix
        except Exception as e:
            self.get_logger().error(f"Error converting image to matrix: {e}")
            return None

    def visualize_matrix(self, matrix):
        plt.imshow(matrix, cmap='gray')
        plt.title('Floor Plan Matrix Visualization')
        plt.show()

def main(args=None):
    rclpy.init(args=args)
    node = PDFToMatrix()
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == '__main__':
    main()
