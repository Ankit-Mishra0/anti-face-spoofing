import cv2
import torch
import time
import subprocess

from models.cnn_model import FaceAntiSpoofCNN

facedetector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

antispoofmodel = FaceAntiSpoofCNN()
antispoofmodel.load_state_dict(
    torch.load("models/face_antispoof_cnn.pth", map_location="cpu")
)
antispoofmodel.eval()

inputsize = 128
slidingwindow = 500
recentpredictions = []
camera = cv2.VideoCapture(0)
spoofdetected = False


def lock_screen():
    subprocess.run(
        ["osascript", "-e", "tell application \"loginwindow\" to «event aevtrlgo»"],
        capture_output=True, text=True
    )


while True:
    success, currentframe = camera.read()
    if not success:
        break

    grayscaleframe = cv2.cvtColor(currentframe, cv2.COLOR_BGR2GRAY)
    detectedfaces = facedetector.detectMultiScale(grayscaleframe, 1.3, 5)

    for (facex, facey, facewidth, faceheight) in detectedfaces:
        croppedface = currentframe[facey:facey+faceheight, facex:facex+facewidth]
        resizedface = cv2.resize(croppedface, (inputsize, inputsize))
        normalizedface = resizedface / 255.0

      
        facetensor = torch.tensor(normalizedface, dtype=torch.float32)
        facetensor = facetensor.permute(2, 0, 1).unsqueeze(0)
        with torch.no_grad():
            confidence = torch.sigmoid(antispoofmodel(facetensor)).item()

        if confidence > 0.8:
            recentpredictions.append(0)
            displaylabel, boxcolor = "REAL", (0, 255, 0)
        else:
            recentpredictions.append(1)
            displaylabel, boxcolor = "SPOOF", (0, 0, 255)

        if len(recentpredictions) > slidingwindow:
            recentpredictions.pop(0)

        if len(recentpredictions) == slidingwindow:
            spoof_ratio = sum(recentpredictions) / slidingwindow
            if spoof_ratio > 0.6:
                spoofdetected = True

        cv2.rectangle(currentframe, (facex, facey), (facex+facewidth, facey+faceheight), boxcolor, 2)
        cv2.putText(currentframe, f"{displaylabel} ({confidence:.2f})",
                    (facex, facey-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, boxcolor, 2)

    cv2.imshow("Face Anti-Spoofing System", currentframe)

    if spoofdetected:
        break

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

camera.release()
cv2.destroyAllWindows()

if spoofdetected:
    print("SPOOF DETECTED — LOCKING NOW...")
    time.sleep(1)
    lock_screen()
    time.sleep(3)