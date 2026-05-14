# Command Reference

Quick reference for all available commands in the Face Mask Detection project.

## 🚀 Setup Commands

### Install Dependencies
```bash
pip install -r requirements.txt
```

### Verify Installation
```bash
python test_setup.py
```

### Check Python Version
```bash
python --version  # Should be 3.8+
```

## 🎓 Training Commands

### Train Model (Default Settings)
```bash
python train.py
```

### Train with Custom Epochs (Edit train.py first)
```python
# In train.py, change:
EPOCHS = 20  # Instead of 10
```

## 📹 Real-time Detection Commands

### Webcam Detection (Main Use Case)
```bash
python realtime_detection.py
```

**Controls:**
- Press `q` to quit
- Press `s` to save screenshot

### Interactive Demo
```bash
python demo.py
```

**Menu Options:**
1. Webcam (real-time)
2. Test image from dataset
3. Custom image

## 🖼️ Image Detection Commands

### Detect and Display
```bash
python detect_image.py path/to/image.jpg
```

### Detect and Save Result
```bash
python detect_image.py path/to/image.jpg --output result.jpg
```

### Examples
```bash
# Test on dataset image
python detect_image.py datasets/train/with_mask/with_mask_1.jpg

# Save result
python detect_image.py photo.jpg --output detected.jpg

# Test multiple images (use a loop)
for img in datasets/train/with_mask/*.jpg; do
    python detect_image.py "$img" --output "output_$(basename $img)"
done
```

## 🎬 Video Processing Commands

### Process and Display
```bash
python detect_video.py path/to/video.mp4
```

### Process and Save
```bash
python detect_video.py path/to/video.mp4 --output result.mp4
```

### Process Without Display (Faster)
```bash
python detect_video.py path/to/video.mp4 --output result.mp4 --no-display
```

### Examples
```bash
# Process webcam recording
python detect_video.py recording.mp4 --output detected.mp4

# Process without preview (background processing)
python detect_video.py long_video.mp4 --output output.mp4 --no-display
```

## 🔧 Utility Commands

### List All Python Files
```bash
ls -lh *.py
```

### Check Model File
```bash
ls -lh mask_model.pth
```

### Count Dataset Images
```bash
# Count with_mask images
ls datasets/train/with_mask/*.jpg | wc -l

# Count without_mask images
ls datasets/train/without_mask/*.jpg | wc -l

# Total images
find datasets/train -name "*.jpg" | wc -l
```

### View Documentation
```bash
# Quick start
cat QUICKSTART.md

# Full README
cat README.md

# Project structure
cat PROJECT_STRUCTURE.md
```

## 🐍 Python API Usage

### Use in Your Own Script
```python
from demo import MaskDetector
import cv2

# Initialize
detector = MaskDetector()

# Load image
image = cv2.imread('photo.jpg')

# Detect
results = detector.detect_and_predict(image)

# Print results
for result in results:
    pred = result['prediction']
    print(f"{pred['class']}: {pred['confidence']*100:.1f}%")

# Annotate and save
annotated = detector.annotate_image(image, results)
cv2.imwrite('output.jpg', annotated)
```

## 🔄 Common Workflows

### First Time Setup
```bash
# 1. Install
pip install -r requirements.txt

# 2. Verify
python test_setup.py

# 3. Train (if model doesn't exist)
python train.py

# 4. Run
python realtime_detection.py
```

### Quick Test
```bash
# Test on sample image
python detect_image.py datasets/train/with_mask/with_mask_1.jpg
```

### Batch Processing Images
```bash
# Create output directory
mkdir -p output

# Process all images
for img in input_images/*.jpg; do
    python detect_image.py "$img" --output "output/$(basename $img)"
done
```

### Retrain Model
```bash
# Backup old model
mv mask_model.pth mask_model_backup.pth

# Train new model
python train.py

# Test new model
python realtime_detection.py
```

## 🐛 Troubleshooting Commands

### Check if PyTorch is Installed
```bash
python -c "import torch; print(torch.__version__)"
```

### Check if OpenCV is Installed
```bash
python -c "import cv2; print(cv2.__version__)"
```

### Check CUDA Availability
```bash
python -c "import torch; print('CUDA:', torch.cuda.is_available())"
```

### Test Webcam
```bash
python -c "import cv2; cap = cv2.VideoCapture(0); print('Webcam:', cap.isOpened())"
```

### Reinstall Dependencies
```bash
pip uninstall torch torchvision opencv-python numpy -y
pip install -r requirements.txt
```

## 📊 Performance Testing

### Measure Training Time
```bash
time python train.py
```

### Measure Inference Speed
```bash
python -c "
import torch
import time
from train import RealTimeMaskNet

model = RealTimeMaskNet()
model.eval()

dummy = torch.randn(1, 3, 128, 128)
start = time.time()
for _ in range(100):
    with torch.no_grad():
        _ = model(dummy)
print(f'Avg time: {(time.time()-start)/100*1000:.2f}ms')
"
```

## 🎯 Advanced Usage

### Change Image Size (Edit Scripts)
```python
# In train.py and detection scripts, change:
image_size = (224, 224)  # Instead of (128, 128)
```

### Change Batch Size (Edit train.py)
```python
# In train.py, change:
train_loader = DataLoader(train_dataset, batch_size=16)  # Instead of 32
```

### Change Learning Rate (Edit train.py)
```python
# In train.py, change:
optimizer = optim.Adam(model.parameters(), lr=0.0001)  # Instead of 0.001
```

### Train for More Epochs (Edit train.py)
```python
# In train.py, change:
EPOCHS = 20  # Instead of 10
```

## 📝 Git Commands (If Using Version Control)

### Initialize Git
```bash
git init
git add .
git commit -m "Initial commit: Face mask detection system"
```

### Create .gitignore
```bash
cat > .gitignore << EOF
__pycache__/
*.pyc
*.pyo
*.pyd
.Python
*.so
*.egg
*.egg-info/
dist/
build/
.DS_Store
*.jpg
*.png
*.mp4
*.avi
mask_model.pth
EOF
```

## 🚀 Quick Commands Summary

```bash
# Setup
pip install -r requirements.txt && python test_setup.py

# Train
python train.py

# Real-time (most common)
python realtime_detection.py

# Image
python detect_image.py image.jpg --output result.jpg

# Video
python detect_video.py video.mp4 --output result.mp4

# Demo
python demo.py
```

---

**Tip:** Bookmark this file for quick reference! 📌
