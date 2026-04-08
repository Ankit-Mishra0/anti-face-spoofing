# Face Anti-Spoofing System

## Overview

This project implements a real-time face anti-spoofing system using a Convolutional Neural Network (CNN) to detect whether a face in a video stream is real or spoofed (e.g., from a photo, video, or mask). The system uses OpenCV for face detection and PyTorch for the deep learning model. If spoofing is detected consistently over a sliding window of frames, the system locks the computer screen for security.

This project was developed as a Computer Vision (CV) project by:

- **Ankit Mishra** (2023UCS1711)
- **Amit Kumar** (2023UCS1744)

## Features

- **Real-time Detection**: Processes live video feed from the camera to detect faces and classify them as real or spoof.
- **CNN Model**: A custom CNN architecture trained on color and depth images to distinguish real faces from spoof attempts.
- **Face Detection**: Utilizes Haar cascades for efficient face localization.
- **Security Integration**: Automatically locks the screen on macOS when spoofing is detected.
- **Sliding Window Analysis**: Uses a sliding window of predictions to reduce false positives and ensure consistent detection.

## Installation

### Prerequisites

- Python 3.8 or higher
- macOS (for screen locking functionality; modify `lock_screen()` function for other OS)
- Webcam

### Dependencies

Install the required Python packages using pip:

```bash
pip install -r files/requirements.txt
```

Additionally, install PyTorch (if not included):

```bash
pip install torch torchvision
```

### Dataset Preparation

1. Place your dataset in the `dataset/archive/` directory with the following structure:

   ```
   dataset/archive/
   ├── train_img/train_img/
   │   ├── color/
   │   └── depth/
   └── test_img/test_img/
       ├── color/
       └── depth/
   ```

2. Run the dataset preparation script:
   ```bash
   python files/prepare_dataset.py
   ```
   This will organize images into `dataset/real/` and `dataset/spoof/` folders.

### Model Training

To train the CNN model:

```bash
python files/train.py
```

The trained model will be saved as `models/face_antispoof_cnn.pth`.

## Usage

### Real-time Detection

Run the main application for real-time face anti-spoofing:

```bash
python main.py
```

- The application will open a camera feed.
- Detected faces will be labeled as "REAL" (green) or "SPOOF" (red) with confidence scores.
- If spoofing is detected in more than 60% of the last 500 frames, the screen will lock.

Press 'q' to quit manually.

### Other Scripts

- `files/realtime_demotest.py`: Demo script for testing detection.
- `files/dataset_loader.py`: Utility for loading the dataset.
- `files/liveness_detector.py`: Additional liveness detection logic.

## Model Architecture

The CNN model consists of:

- Two convolutional layers (16 and 32 filters, 3x3 kernels)
- Max pooling layers
- Two fully connected layers (128 and 1 output)
- Uses ReLU activation and sigmoid for binary classification.

Input size: 128x128 RGB images.

## Dataset

The system is trained on a dataset containing real and spoof face images from color and depth modalities. Images are resized to 128x128 pixels and normalized.

- **Real Images**: Genuine face captures.
- **Spoof Images**: Fake faces from photos, videos, etc.

## Requirements

See `files/requirements.txt` for a full list. Key libraries:

- OpenCV
- PyTorch
- NumPy
- scikit-learn
- dlib

## License

This project is for educational purposes. Please check individual library licenses for usage.

## Acknowledgments

- OpenCV for computer vision tasks.
- PyTorch for deep learning framework.
- Haar cascade classifier for face detection.
