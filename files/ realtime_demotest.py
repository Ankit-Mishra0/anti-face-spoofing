import cv2
import torch
import numpy as np

from models.cnn_model import FaceAntiSpoofCNN
face_cascade = cv2.CascadeClassifier(
    "haarcascade_frontalface_default.xml"
)

model = FaceAntiSpoofCNN()
model.load_state_dict(
    torch.load("models/face_antispoof_cnn.pth", map_location="cpu")
)
model.eval()

IMG_SIZE = 128

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.3,
        minNeighbors=5
    )

    for (x, y, w, h) in faces:
        face = frame[y:y+h, x:x+w]
        face = cv2.resize(face, (IMG_SIZE, IMG_SIZE))
        face = face / 255.0

        face_tensor = torch.tensor(face, dtype=torch.float32)
        face_tensor = face_tensor.permute(2, 0, 1).unsqueeze(0)

        with torch.no_grad():
            output = model(face_tensor)
            prob = torch.sigmoid(output).item()

        if prob > 0.9:
            label = "REAL"
            color = (0, 255, 0)
        else:
            label = "SPOOF"
            color = (0, 0, 255)

        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
        cv2.putText(
            frame,
            f"{label} ({prob:.2f})",
            (x, y-10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            color,
            2
        )

    cv2.imshow("Face Anti-Spoofing System", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
