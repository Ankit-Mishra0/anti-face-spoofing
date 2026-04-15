import os
import cv2
import torch
import torch.nn as nn
import torch.optim as optim

from models.cnn_model import FaceAntiSpoofCNN

# Settings
input_size = 128
epochs = 3
learning_rate = 0.001

# Paths
real_dirs = ["dataset/casia/real", "dataset/msu/real"]
spoof_dirs = ["dataset/casia/spoof", "dataset/msu/spoof"]

# Load Data
data = []
labels = []

def load_images(paths, label):
    for path in paths:
        for img_name in os.listdir(path):
            img_path = os.path.join(path, img_name)
            img = cv2.imread(img_path)
            if img is None:
                continue
            img = cv2.resize(img, (input_size, input_size))
            img = img / 255.0
            data.append(img)
            labels.append(label)

load_images(real_dirs, 0)
load_images(spoof_dirs, 1)

data = data[:300]
labels = labels[:300]

# Convert to tensor
X = torch.tensor(data, dtype=torch.float32).permute(0, 3, 1, 2)
y = torch.tensor(labels, dtype=torch.float32).unsqueeze(1)

# Model
model = FaceAntiSpoofCNN()
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Training
for epoch in range(epochs):
    outputs = torch.sigmoid(model(X))
    loss = criterion(outputs, y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # Accuracy
    predicted = (outputs > 0.5).float()
    correct = (predicted == y).sum().item()
    accuracy = correct / y.size(0) * 100

    print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}, Accuracy: {accuracy:.2f}%")

# Save model
torch.save(model.state_dict(), "models/multi_dataset_model.pth")

print("Training complete. Model saved!")