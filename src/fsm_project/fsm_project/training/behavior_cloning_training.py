#!/usr/bin/env python3
import os
import numpy as np
import tensorflow as tf
from keras import layers, models

DATA_FILE = '/home/usman/duplo_combined/collected_data.npy'
MODEL_SAVE_PATH = '/home/usman/bc_model'
EPOCHS = 10
BATCH_SIZE = 16

def load_data():
    if not os.path.exists(DATA_FILE):
        raise FileNotFoundError(f"Data file not found: {DATA_FILE}")

    data = np.load(DATA_FILE, allow_pickle=True).item()
    images = data['images']   # shape (N,H,W,3)
    labels = data['labels']   # shape (N,)
    return images, labels

def build_model(input_shape=(224,224,3), num_classes=2):
    model = models.Sequential([
        layers.Input(shape=input_shape),
        layers.Conv2D(32, (3,3), activation='relu', padding='same'),
        layers.MaxPooling2D((2,2)),
        layers.Conv2D(64, (3,3), activation='relu', padding='same'),
        layers.MaxPooling2D((2,2)),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(num_classes, activation='softmax')
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

def main():
    print("Loading data...")
    images, labels = load_data()
    print(f"Loaded {len(images)} samples, shape={images.shape}, labels={labels.shape}")


    # images are (N,360,640,3)
    input_shape = images.shape[1:]  #
    model = build_model(input_shape=input_shape, num_classes=2)

    print("Fitting model...")
    history = model.fit(
        images, labels,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        validation_split=0.2,
        verbose=1
    )

    print("Saving model...")
    model.save(MODEL_SAVE_PATH)
    print(f"Model saved to: {MODEL_SAVE_PATH}")

if __name__ == "__main__":
    main()