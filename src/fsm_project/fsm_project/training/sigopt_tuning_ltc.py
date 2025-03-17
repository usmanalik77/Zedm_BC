#!/usr/bin/env python3
import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras.optimizers import Adam
from fsm_project.models.behavior_cloning_model import SimpleCNN
from sigoptlite import LocalDriver

# Configuration
DATA_FILE = '/home/usman/collected_data.npy' 
MODEL_SAVE_PATH = '/home/usman/bc_model_tuned'
EPOCHS = 10
BATCH_SIZE = 16

def load_training_data(data_file):

    if not os.path.exists(data_file):
        raise FileNotFoundError(f"Training data file not found: {data_file}")
    data = np.load(data_file, allow_pickle=True).item()
    images = data['images']  # shape (N, H, W, 3)
    raw_labels = data['labels']
    labels = np.array([1 if str(lbl).strip().lower() == "square" else 0 for lbl in raw_labels])
    return images, labels

def build_model(input_shape, num_classes, conv1_filters, conv2_filters, dense_units, learning_rate):

    model = keras.Sequential([
        keras.layers.Input(shape=input_shape),
        keras.layers.Conv2D(conv1_filters, (3,3), activation='relu', padding='same'),
        keras.layers.MaxPooling2D((2,2)),
        keras.layers.Conv2D(conv2_filters, (3,3), activation='relu', padding='same'),
        keras.layers.MaxPooling2D((2,2)),
        keras.layers.Flatten(),
        keras.layers.Dense(dense_units, activation='relu'),
        keras.layers.Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer=Adam(learning_rate=learning_rate),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model

def sigopt_experiment(driver):

    experiment_meta = {
        "name": "Behavior Cloning Hyperparameter Tuning",
      "parameters": [
            {"name": "conv1_filters", "type": "int", "bounds": {"min": 16, "max": 64}},
            {"name": "conv2_filters", "type": "int", "bounds": {"min": 32, "max": 128}},
            {"name": "dense_units", "type": "int", "bounds": {"min": 64, "max": 256}},
            {"name": "learning_rate", "type": "double", "bounds": {"min": 1e-4, "max": 1e-2}},
        ] ,

        "metrics": [{"name": "val_loss", "objective": "minimize"}],
        "observation_budget": 10,
    }
    experiment = driver.experiments().create(**experiment_meta)
    return experiment

def train_with_params(params, images, labels, input_shape, num_classes):
    conv1_filters = int(params["conv1_filters"])
    conv2_filters = int(params["conv2_filters"])
    dense_units = int(params["dense_units"])
    learning_rate = params["learning_rate"]
    
    model = build_model(input_shape, num_classes, conv1_filters, conv2_filters, dense_units, learning_rate)
    history = model.fit(images, labels, batch_size=BATCH_SIZE, epochs=EPOCHS, validation_split=0.2, verbose=0)
    best_val_loss = min(history.history["val_loss"])
    return best_val_loss

def main():
    print("Loading training data...")
    images, labels = load_training_data(DATA_FILE)
    print(f"Loaded {len(images)} samples, image shape: {images.shape}, labels shape: {labels.shape}")
    images = images.astype(np.float32) / 255.0  # Normalize
    input_shape = images.shape[1:]  # e.g., (H, W, 3)
    num_classes = 2

    driver = LocalDriver()
    experiment = sigopt_experiment(driver)
    best_loss = float("inf")
    best_params = None

    for _ in range(experiment.observation_budget):
        suggestion = driver.experiments(experiment.id).suggestions().create()
        params = suggestion.assignments
        print("Trying parameters:", params)
        val_loss = train_with_params(params, images, labels, input_shape, num_classes)
        print("Validation loss:", val_loss)
        driver.experiments(experiment.id).observations().create(
            suggestion=suggestion.id,
            values=[{"name": "val_loss", "value": val_loss}]
        )
        if val_loss < best_loss:
            best_loss = val_loss
            best_params = params

    print("Best parameters:", best_params, "with validation loss:", best_loss)
    print("Training final model with best parameters...")
    final_model = build_model(input_shape, num_classes,
                              int(best_params["conv1_filters"]),
                              int(best_params["conv2_filters"]),
                              int(best_params["dense_units"]),
                              best_params["learning_rate"])
    final_model.fit(images, labels, batch_size=BATCH_SIZE, epochs=EPOCHS, verbose=1)
    final_model.save(MODEL_SAVE_PATH, save_format='tf')
    print(f"Final model saved to {MODEL_SAVE_PATH}")

if __name__ == '__main__':
    main()
