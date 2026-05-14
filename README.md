# Real-time Face Mask Detection

A lightweight CNN-based face mask detection system that runs in real-time on webcam feeds.

## Features

- ✅ Real-time mask detection via webcam
- ✅ Lightweight CNN architecture (fast inference)
- ✅ Face detection using Haar Cascade
- ✅ Confidence scores for predictions
- ✅ Image and video file support
- ✅ Easy to train on custom datasets

## Project Structure

```
nn_final_project/
├── datasets/
│   ├── train/
│   │   ├── with_mask/       # 3,725 images
│   │   └── without_mask/    # 3,828 images
│   └── test/
│       ├── Mask/
│       ├── No_Mask/
│       └── Incorrect_Mask/
├── train.py                 # Training script
├── realtime_detection.py    # Webcam real-time detection
├── detect_image.py          # Image/video detection
├── mask_model.pth           # Trained model weights
├── requirements.txt         # Dependencies
└── README.md               # This file
```

## Installation

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

Required packages:
- PyTorch >= 2.0.0
- OpenCV >= 4.8.0
- NumPy >= 1.24.0

### 2. Verify Installation

```bash
python -c "import torch; import cv2; print('PyTorch:', torch.__version__); print('OpenCV:', cv2.__version__)"
```

## Usage

### 1. Train the Model

Train on your dataset (7,553 images):

```bash
python train.py
```

**Training Output:**
- 10 epochs of training
- Loss and accuracy per epoch
- Test accuracy evaluation
- Inference speed benchmark
- Saves model as `mask_model.pth`

**Expected Results:**
- Training time: ~5-10 minutes (CPU) or ~1-2 minutes (GPU)
- Test accuracy: ~95-98%
- Inference speed: ~50-100 FPS

### 2. Real-time Webcam Detection

Run mask detection on your webcam:

```bash
python realtime_detection.py
```

**Controls:**
- Press `q` to quit
- Press `s` to save screenshot

**Features:**
- Green box = Wearing mask ✅
- Red box = Not wearing mask ❌
- Shows confidence percentage
- Displays number of faces detected

### 3. Detect on Images

Test on a single image:

```bash
# Display result
python detect_image.py path/to/image.jpg

# Save result
python detect_image.py path/to/image.jpg --output result.jpg
```

**Example:**

```bash
python detect_image.py datasets/test/Mask/test_001.jpg --output output.jpg
```

## Model Architecture

**RealTimeMaskNet** - Lightweight CNN for fast inference

```
Input: 128x128x3 RGB image

Block 1: Conv2d(3→16) → ReLU → MaxPool
Block 2: Conv2d(16→32) → ReLU → MaxPool  
Block 3: Conv2d(32→64) → ReLU → MaxPool

Classifier: AdaptiveAvgPool → Flatten → Linear(64→2)

Output: 2 classes (No Mask, With Mask)
```

**Model Stats:**
- Parameters: ~2,000
- Input size: 128x128x3
- Output: 2 classes
- Inference: ~10-20ms per image

## Dataset

**Current Dataset:**
- Total: 7,553 images
- With Mask: 3,725 images
- Without Mask: 3,828 images
- Split: 80% train / 20% test

**Data Preprocessing:**
1. BGR → RGB conversion
2. Resize to 128x128
3. Normalize to [0, 1]
4. Convert to PyTorch tensor

## Performance

**Accuracy:**
- Training: ~98-99%
- Testing: ~95-98%

**Speed:**
- Inference: 10-20ms per image
- FPS: 50-100 (depending on hardware)
- Real-time: ✅ Yes

**Hardware Requirements:**
- CPU: Any modern processor
- GPU: Optional (CUDA-enabled for faster training)
- RAM: 4GB minimum
- Webcam: Any USB/built-in camera

## Troubleshooting

### Webcam Not Opening

```python
# Try different camera indices
cap = cv2.VideoCapture(0)  # Try 0, 1, 2, etc.
```

### Model Not Found

Make sure you've trained the model first:

```bash
python train.py
```

This will create `mask_model.pth` in the project directory.

### Low Accuracy

Try these improvements:
1. Train for more epochs (change `EPOCHS = 10` to `EPOCHS = 20`)
2. Add data augmentation
3. Use a larger model
4. Collect more training data

### Slow Performance

Optimize for speed:
1. Reduce image size (128x128 → 96x96)
2. Use GPU if available
3. Reduce face detection frequency
4. Use a smaller model

## Customization

### Change Image Size

In `train.py` and detection scripts:

```python
image_size = (128, 128)  # Change to (96, 96) or (224, 224)
```

### Adjust Learning Rate

In `train.py`:

```python
optimizer = optim.Adam(model.parameters(), lr=0.001)  # Try 0.0001 or 0.01
```

### Change Batch Size

In `train.py`:

```python
train_loader = DataLoader(train_dataset, batch_size=32)  # Try 16 or 64
```

### Add More Classes

Modify the dataset class to support 3 classes (Mask, No Mask, Incorrect Mask):

```python
# In FaceMaskDataset.__init__
if 'with_mask' in str(p):
    label = 0  # Mask
elif 'without_mask' in str(p):
    label = 1  # No Mask
else:
    label = 2  # Incorrect Mask
```

Update model output:

```python
nn.Linear(64, 3)  # Change from 2 to 3 classes
```

## Future Improvements

- [ ] Add data augmentation (rotation, flip, brightness)
- [ ] Support for 3 classes (Mask, No Mask, Incorrect Mask)
- [ ] Mobile deployment (TensorFlow Lite, ONNX)
- [ ] Video file processing
- [ ] Batch processing for multiple images
- [ ] Web interface (Flask/FastAPI)
- [ ] Model quantization for edge devices
- [ ] Multi-face tracking with IDs

## License

MIT License - Feel free to use for educational and commercial purposes.

## Credits

- Dataset: Face Mask Detection Dataset
- Framework: PyTorch
- Face Detection: OpenCV Haar Cascade

## Contact

For questions or issues, please open an issue on GitHub.

---

**Happy Detecting! 😷✅**
