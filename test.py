import torch
import numpy as np
from sklearn.metrics import accuracy_score

from dataset_loader import load_dataset
from models.cnn_model import FaceAntiSpoofCNN

# -----------------------------
# Load dataset
# -----------------------------
X, y = load_dataset()

X = torch.tensor(X, dtype=torch.float32).permute(0, 3, 1, 2)
y = torch.tensor(y, dtype=torch.float32)

# -----------------------------
# Load trained model
# -----------------------------
model = FaceAntiSpoofCNN()
model.load_state_dict(torch.load("models/face_antispoof_cnn.pth"))
model.eval()

# -----------------------------
# Prediction
# -----------------------------
with torch.no_grad():
    outputs = model(X).squeeze()
    probs = torch.sigmoid(outputs)
    preds = (probs > 0.5).float()

# -----------------------------
# Metrics
# -----------------------------
accuracy = accuracy_score(y, preds)

# FAR & FRR
spoof_idx = (y == 0)
real_idx = (y == 1)

false_accepts = ((preds == 1) & spoof_idx).sum().item()
false_rejects = ((preds == 0) & real_idx).sum().item()

FAR = false_accepts / spoof_idx.sum().item()
FRR = false_rejects / real_idx.sum().item()

print(f"Accuracy: {accuracy * 100:.2f}%")
print(f"FAR (False Acceptance Rate): {FAR * 100:.2f}%")
print(f"FRR (False Rejection Rate): {FRR * 100:.2f}%")
