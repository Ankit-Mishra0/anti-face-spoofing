import torch
import torch.nn as nn
import torch.nn.functional as F

class FaceAntiSpoofCNN(nn.Module):
    def __init__(self):
        super(FaceAntiSpoofCNN, self).__init__()

        # Convolutional layers
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)

        # Pooling layer
        self.pool = nn.MaxPool2d(2, 2)

        # Fully connected layers
        self.fc1 = nn.Linear(32 * 32 * 32, 128)
        self.fc2 = nn.Linear(128, 1)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))   # 128 → 64
        x = self.pool(F.relu(self.conv2(x)))   # 64 → 32

        x = x.view(x.size(0), -1)               # Flatten
        x = F.relu(self.fc1(x))
        x = self.fc2(x)


        return x
