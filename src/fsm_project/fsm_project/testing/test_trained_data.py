#!/usr/bin/env python3

import os
import numpy as np
import tensorflow as tf

# Paths (adjust these if necessary)
MODEL_PATH = '/home/usman/bc_model'
SAMPLE_IMAGE_PATH = '/home/usman/duplo_data/image_6.npy'  # change image index as needed

def preprocess_image(image):
    """
    Preprocess the image for the model.
    This function assumes the image is stored as a numpy array.
    The training normalization was dividing by 255.0.
    Adjust resizing if your model expects a different size.
    """
    # If needed, you can add a resize step using tf.image.resize.
    # For example, if your model input shape is (360, 640, 3):
    desired_size = (360, 640)
    image_tensor = tf.convert_to_tensor(image, dtype=tf.float32)
    # If the image shape does not match, resize it:
    if image_tensor.shape[:2] != desired_size:
        image_tensor = tf.image.resize(image_tensor, desired_size)
    # Normalize to [0, 1]
    normalized_image = image_tensor / 255.0
    # Add batch dimension
    return tf.expand_dims(normalized_image, axis=0)

def main():
    # Load the trained model
    model = tf.keras.models.load_model(MODEL_PATH)
    print("Model loaded from:", MODEL_PATH)

    # Load a sample image from your collected data
    if not os.path.exists(SAMPLE_IMAGE_PATH):
        print(f"Sample image file not found: {SAMPLE_IMAGE_PATH}")
        return

    image = np.load(SAMPLE_IMAGE_PATH)
    print(f"Loaded sample image from {SAMPLE_IMAGE_PATH}, shape: {image.shape}")

    # Preprocess the image for the model
    input_image = preprocess_image(image)
    print("Preprocessed image shape:", input_image.shape)

    # Run inference
    predictions = model.predict(input_image)
    print("Predictions:", predictions)

if __name__ == '__main__':
    main()
