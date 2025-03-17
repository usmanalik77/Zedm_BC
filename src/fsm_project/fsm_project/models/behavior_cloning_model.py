import torch
import torch.nn as nn
import torch.nn.functional as F



class SimpleCNN(nn.Module):
    def __init__(self, input_channels=3, num_classes=2, dense_neuron =128):
        super(SimpleCNN, self).__init__()
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(in_channels=input_channels, out_channels=32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)

        self.fc1 = nn.Linear(64 * 56 * 56, dense_neuron)
        self.fc2 = nn.Linear(dense_neuron , num_classes)

    def forward(self, x):
        # x: (B, 3, 224, 224)
        x = self.pool(F.relu(self.conv1(x)))  # -> (B, 32, 112, 112)
        x = self.pool(F.relu(self.conv2(x)))  # -> (B, 64, 56, 56)
        x = x.view(x.size(0), -1)             # Flatten the tensor
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

def build_model():
    model = SimpleCNN()
    return model
