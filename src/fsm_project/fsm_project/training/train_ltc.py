#!/usr/bin/env python3
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import torch.nn.functional as F
from fsm_project.models.ltc_model import LTCCell  # Use your LTC cell from your models folder

# Define the LTC-based model using PyTorch
class LTCModel(nn.Module):
    def __init__(self, input_shape=(3, 224, 224), num_classes=2, units=64):

        super(LTCModel, self).__init__()
        # Convolutional Feature Extractor
        self.conv1 = nn.Conv2d(in_channels=input_shape[0], out_channels=32, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.flatten = nn.Flatten()
        self._calculate_dense_input_size(input_shape)
        self.dense1 = nn.Linear(self.dense_input_size, 128)
        self.relu = nn.ReLU()
        # LTC layer using your LTCCell (assumes LTCCell has a property state_size)
        self.ltc_cell = LTCCell(num_units=units)
        # Final classification layer
        self.output_layer = nn.Linear(units, num_classes)

    def _calculate_dense_input_size(self, input_shape):
        with torch.no_grad():
            dummy_input = torch.zeros(1, *input_shape)  # e.g. (1, 3, 224, 224)
            x = self.pool1(torch.relu(self.conv1(dummy_input)))
            x = self.pool2(torch.relu(self.conv2(x)))
            self.dense_input_size = x.view(1, -1).shape[1]
            print("Dense input size calculated:", self.dense_input_size)

    def forward(self, x):
        x = self.pool1(torch.relu(self.conv1(x)))
        x = self.pool2(torch.relu(self.conv2(x)))
        x = self.flatten(x)
        x = self.relu(self.dense1(x))
        batch_size = x.size(0)
        h0 = torch.zeros(batch_size, self.ltc_cell.state_size, device=x.device)
        ltc_output, _ = self.ltc_cell(x, h0)
        return self.output_layer(ltc_output)

def load_training_data(data_file):

    if not os.path.exists(data_file):
        raise FileNotFoundError(f"Training data file not found: {data_file}")
    data = np.load(data_file, allow_pickle=True).item()
    images = data['images']  # shape (N, H, W, 3) e.g. (N,360,640,3)
    raw_labels = data['labels']  # should be an array of strings like "square" or "not-square"
    labels = [1 if isinstance(lbl, str) and lbl.lower() == "square" else 0 for lbl in raw_labels]
    return images, np.array(labels)

def main():
    DATA_FILE = '/home/usman/collected_data.npy'  # Adjust path if needed
    EPOCHS = 10
    BATCH_SIZE = 4
    LEARNING_RATE = 0.001

    # Use a fresh training run by removing any old saved model file.
    MODEL_SAVE_PATH = '/home/usman/bc_model_ltc.pth'
    if os.path.exists(MODEL_SAVE_PATH):
        os.remove(MODEL_SAVE_PATH)
        print(f"Old model at {MODEL_SAVE_PATH} removed.")

    print("Loading data...")
    images, labels = load_training_data(DATA_FILE)
    print(f"Loaded {len(images)} samples, image shape: {images.shape}, labels shape: {labels.shape}")

    # Convert images to torch tensor and permute from (N, H, W, 3) to (N, 3, H, W)
    images = torch.tensor(images, dtype=torch.float32)
    images = images.permute(0, 3, 1, 2)

    # Resize images to 224x224 to match the model's expected input size.
    images = F.interpolate(images, size=(224, 224), mode='bilinear', align_corners=False)

    labels = torch.tensor(labels, dtype=torch.long)

    dataset = TensorDataset(images, labels)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_shape = images.shape[1:]  # should now be (3,224,224)
    model = LTCModel(input_shape=input_shape, num_classes=2, units=64).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    model.train()
    print(f"Starting training on {len(dataset)} samples...")
    for epoch in range(EPOCHS):
        total_loss = 0.0
        for batch_images, batch_labels in dataloader:
            batch_images, batch_labels = batch_images.to(device), batch_labels.to(device)
            optimizer.zero_grad()
            outputs = model(batch_images)
            loss = criterion(outputs, batch_labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {avg_loss:.4f}")
    
    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    print(f"Model saved to: {MODEL_SAVE_PATH}")

if __name__ == '__main__':
    main()
