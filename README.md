# Face Anti-Spoofing

A deep learning project for detecting face spoofing attacks (presentation attacks) using a Convolutional Neural Network (CNN). This system can distinguish between genuine live faces and spoofed faces (photos, videos, masks, etc.).

## Overview

Face anti-spoofing is a critical security component for facial recognition systems. This project implements a CNN-based solution to detect whether a face presented to a camera is a live person or a spoofing attempt.

## Features

- **CNN-based Detection**: Custom lightweight CNN architecture for binary classification (real vs. spoof)
- **Real-time Detection**: Webcam-based live inference with Haar Cascade face detection
- **Dataset Support**: Support for color and depth-based datasets
- **PyTorch Implementation**: Modern deep learning framework for flexibility and performance
- **Class Imbalance Handling**: Weighted loss function to handle imbalanced datasets

## Project Structure

```
face-anti-spoofing/
├── models/
│   ├── cnn_model.py              # CNN architecture definition
│   └── face_antispoof_cnn.pth     # Trained model weights
├── dataset/
│   ├── archive/                  # Archive dataset with color and depth
│   ├── real/                      # Real face samples
│   └── spoof/                     # Spoofed face samples
├── train.py                       # Model training script
├── prepare_dataset.py             # Dataset preparation utilities
├── dataset_loader.py              # Custom data loader
├── realtime_demo.py               # Live webcam inference
├── realtime_demotest.py           # Testing the real-time demo
├── test.py                        # Model testing/evaluation
├── test_camera.py                 # Camera functionality test
├── test_loader.py                 # Dataset loader test
├── requirements.txt               # Python dependencies
└── haarcascade_frontalface_default.xml  # Haar Cascade for face detection
```

## Requirements

- Python 3.7+
- PyTorch 2.9.1
- OpenCV 4.12.0
- NumPy, Scikit-learn, Matplotlib
- See `requirements.txt` for complete dependencies

## About This Project

This is a Computer Vision course project developed at **Netaji Subhas University of Technology (NSUT)**.

**Contributors:**
- Ankit Mishra
- Amit Kumar

## Installation

1. Clone the repository:
```bash
git clone https://github.com/Ankit-Mishra0/anti-face-spoofing.git
cd anti-face-spoofing
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### 1. Prepare Dataset

Organize your dataset with real and spoofed face images:

```bash
python prepare_dataset.py
```

### 2. Train the Model

Train the CNN model on your dataset:

```bash
python train.py
```

**Hyperparameters** (configurable in `train.py`):
- Batch Size: 32
- Epochs: 10
- Learning Rate: 0.001
- Input Size: 128×128 pixels

### 3. Test the Model

Evaluate the trained model on test data:

```bash
python test.py
```

### 4. Real-time Detection

Run live face anti-spoofing detection using your webcam:

```bash
python realtime_demo.py
```

**Controls:**
- Press `q` to quit the application
- The system displays:
  - Green box: Live face detected (confidence score)
  - Red box: Spoof detected (confidence score)

## Model Architecture

The `FaceAntiSpoofCNN` model consists of:

- **Input**: 3-channel RGB image (128×128)
- **Conv Layer 1**: 16 filters, 3×3 kernel, ReLU activation + Max Pooling
- **Conv Layer 2**: 32 filters, 3×3 kernel, ReLU activation + Max Pooling
- **Fully Connected Layer 1**: 128 units, ReLU activation
- **Output Layer**: Single neuron with sigmoid activation for binary classification

**Output**:
- Value close to 0: Real face
- Value close to 1: Spoofed face

## Data Format

The dataset expects:
- RGB images in 128×128 resolution
- Labels: 0 for real faces, 1 for spoofed faces
- Train/validation split: 80/20

Supported dataset types:
- Color images (RGB)
- Depth maps (for enhanced detection)

## Performance Considerations

- **Class Imbalance**: The training script automatically handles imbalanced datasets using weighted BCE loss
- **Real-time Speed**: Optimized for CPU inference with fast Haar Cascade preprocessing
- **Memory**: Designed to run on CPU or GPU with minimal memory requirements

## Testing

Test individual components:

```bash
# Test camera functionality
python test_camera.py

# Test dataset loading
python test_loader.py

# Test real-time demo
python realtime_demotest.py
```

## Troubleshooting

- **Camera not detected**: Ensure your webcam is connected and accessible. Test with `test_camera.py`
- **Model not loading**: Verify `face_antispoof_cnn.pth` exists in the `models/` directory
- **Poor detection accuracy**: Ensure dataset images are properly preprocessed and labeled
- **Out of memory**: Reduce `BATCH_SIZE` in `train.py`

## Future Improvements

- [ ] Support for depth-based detection (RGB-D)
- [ ] Multi-spectral analysis
- [ ] Domain adaptation for cross-dataset generalization
- [ ] Hardware acceleration (CUDA/GPU optimization)
- [ ] Web API for remote inference
- [ ] Mobile deployment

## License

This project is provided as-is for educational and research purposes.

## Contributors

- **Ankit Mishra** - Project Lead & Developer
- **Amit Kumar** - Developer

## Citation

If you use this project for research or educational purposes, please cite:

```
@misc{antispoof2026,
  title={Face Anti-Spoofing Detection using CNN},
  author={Mishra, Ankit and Kumar, Amit},
  year={2026},
  school={Netaji Subhas University of Technology (NSUT)},
  howpublished={\url{https://github.com/Ankit-Mishra0/anti-face-spoofing}}
}
```

## Contact

For questions or suggestions, please reach out to the contributors or open an issue on the GitHub repository.

---

For more information on face anti-spoofing techniques, refer to relevant research papers in biometric security and presentation attack detection (PAD).
