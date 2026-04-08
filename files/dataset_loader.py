import os
import cv2
import numpy as np

IMG_SIZE = 128

def load_images_from_folder(folder, label):
    images = []
    labels = []

    for filename in os.listdir(folder):
        if not filename.lower().endswith(".jpg"):
            continue

        img_path = os.path.join(folder, filename)
        img = cv2.imread(img_path)

        if img is None:
            continue

        # Resize image
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))

        # Normalize pixel values
        img = img / 255.0

        images.append(img)
        labels.append(label)

    return np.array(images), np.array(labels)

def load_dataset():
    real_images, real_labels = load_images_from_folder("dataset/real", 1)
    spoof_images, spoof_labels = load_images_from_folder("dataset/spoof", 0)

    X = np.concatenate((real_images, spoof_images), axis=0)
    y = np.concatenate((real_labels, spoof_labels), axis=0)

    return X, y

