import os
import numpy as np
from keras.optimizers import Adam
from zedm_to_BC.models.behavior_cloning_model import ConvCfCModel

# Training configuration
COLLECTED_DATA_FILE = '/home/usman/collected_data.npy'  # Path to the `.npy` file containing the training data
BATCH_SIZE = 32
EPOCHS = 10
LEARNING_RATE = 0.001

def load_training_data(data_file):
    """
    Load training data from a combined `.npy` file.
    The `.npy` file contains a dictionary with keys 'images' and 'controls'.
    """
    if not os.path.exists(data_file):
        raise FileNotFoundError(f"Training data file not found: {data_file}")

    # Load data from the `.npy` file
    data = np.load(data_file, allow_pickle=True).item()  # Load as dictionary

    # Extract images and controls
    images = data['images']
    controls = data['controls']

    return images, controls

def train_behavior_cloning_model(images, controls):
    """
    Train the behavior cloning model using the provided images and controls.
    """
    # Create model instance
    input_shape = images.shape[1:]  # Assuming (height, width, channels)
    output_size = controls.shape[1]  # Should be 2 (steering, throttle)
    model = ConvCfCModel(input_shape=input_shape, output_size=output_size)

    # Compile the model
    model.compile(optimizer=Adam(learning_rate=LEARNING_RATE), loss='mse')

    print(f"Starting training with {len(images)} samples...")

    # Train the model
    history = model.fit(
        images,
        controls,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        validation_split=0.2,
        verbose=1
    )

    # Save the model in TensorFlow SavedModel format
    model_save_path = '/home/usman/behavior_cloning_model'
    model.save(model_save_path, save_format='tf')
    print(f"Model training complete and saved to '{model_save_path}'.")

def main():
    """
    Main function to load training data and train the behavior cloning model.
    """
    try:
        # Load training data
        images, controls = load_training_data(COLLECTED_DATA_FILE)

        if len(images) == 0 or len(controls) == 0:
            print("No training data found. Please check your collected data file.")
        else:
            # Train the behavior cloning model
            train_behavior_cloning_model(images, controls)
    except FileNotFoundError as e:
        print(e)

if __name__ == '__main__':
    main()
