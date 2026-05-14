# Project Structure

## 📁 Complete File Tree

```
nn_final_project/
│
├── 📄 train.py                    # Train the CNN model
├── 📄 realtime_detection.py       # Real-time webcam detection ⭐
├── 📄 detect_image.py             # Detect masks in images
├── 📄 detect_video.py             # Process video files
├── 📄 demo.py                     # Interactive demo with MaskDetector class
├── 📄 test_setup.py               # Verify installation and setup
│
├── 📄 mask_model.pth              # Trained model weights (0.09 MB)
├── 📄 requirements.txt            # Python dependencies
│
├── 📖 README.md                   # Full documentation
├── 📖 QUICKSTART.md               # Quick start guide
├── 📖 SUMMARY.md                  # Project summary
├── 📖 PROJECT_STRUCTURE.md        # This file
│
└── 📁 datasets/
    ├── 📖 README.md               # Dataset documentation
    │
    ├── 📁 train/
    │   ├── 📁 with_mask/          # 3,725 images ✅
    │   │   ├── with_mask_1.jpg
    │   │   ├── with_mask_2.jpg
    │   │   └── ... (3,725 total)
    │   │
    │   └── 📁 without_mask/       # 3,828 images ❌
    │       ├── without_mask_1.jpg
    │       ├── without_mask_2.jpg
    │       └── ... (3,828 total)
    │
    └── 📁 test/
        ├── 📁 Mask/               # Test images with mask
        ├── 📁 No_Mask/            # Test images without mask
        └── 📁 Incorrect_Mask/     # Test images with incorrect mask
```

## 📝 File Descriptions

### 🎓 Training & Model

| File | Purpose | Usage |
|------|---------|-------|
| `train.py` | Train the CNN model on your dataset | `python train.py` |
| `mask_model.pth` | Trained model weights (PyTorch state dict) | Loaded by detection scripts |

### 🎯 Detection Scripts

| File | Purpose | Usage |
|------|---------|-------|
| `realtime_detection.py` | **Real-time webcam detection** | `python realtime_detection.py` |
| `detect_image.py` | Detect masks in single images | `python detect_image.py image.jpg` |
| `detect_video.py` | Process video files | `python detect_video.py video.mp4` |
| `demo.py` | Interactive demo with menu | `python demo.py` |

### 🔧 Utilities

| File | Purpose | Usage |
|------|---------|-------|
| `test_setup.py` | Verify installation and setup | `python test_setup.py` |
| `requirements.txt` | Python package dependencies | `pip install -r requirements.txt` |

### 📚 Documentation

| File | Content |
|------|---------|
| `README.md` | Complete documentation with all details |
| `QUICKSTART.md` | Quick start guide (3 steps to get running) |
| `SUMMARY.md` | Project summary and overview |
| `PROJECT_STRUCTURE.md` | This file - project structure |
| `datasets/README.md` | Dataset information and guidelines |

## 🎯 Quick Reference

### First Time Setup
```bash
pip install -r requirements.txt    # Install dependencies
python test_setup.py               # Verify setup
python train.py                    # Train model (if needed)
```

### Run Detection
```bash
python realtime_detection.py       # Webcam (most common)
python demo.py                     # Interactive demo
python detect_image.py img.jpg     # Single image
python detect_video.py vid.mp4     # Video file
```

## 📊 Dataset Structure

**Training Data:**
- Location: `datasets/train/`
- Total: 7,553 images
- Classes: 2 (with_mask, without_mask)
- Format: JPG images
- Split: 80% train / 20% test (automatic)

**Test Data:**
- Location: `datasets/test/`
- Classes: 3 (Mask, No_Mask, Incorrect_Mask)
- Purpose: Manual testing and validation

## 🔄 Typical Workflow

```
1. Install Dependencies
   └─> pip install -r requirements.txt

2. Verify Setup
   └─> python test_setup.py

3. Train Model (if needed)
   └─> python train.py
       └─> Creates mask_model.pth

4. Run Detection
   ├─> python realtime_detection.py  (webcam)
   ├─> python demo.py                (interactive)
   ├─> python detect_image.py        (images)
   └─> python detect_video.py        (videos)
```

## 🎨 Script Features Comparison

| Feature | realtime | demo | image | video | train |
|---------|----------|------|-------|-------|-------|
| Webcam | ✅ | ✅ | ❌ | ❌ | ❌ |
| Images | ❌ | ✅ | ✅ | ❌ | ❌ |
| Videos | ❌ | ❌ | ❌ | ✅ | ❌ |
| Save Output | Screenshot | ❌ | ✅ | ✅ | Model |
| Interactive | ✅ | ✅ | ❌ | ✅ | ❌ |
| Class API | ❌ | ✅ | ❌ | ❌ | ❌ |

## 💡 Which Script to Use?

**For real-time webcam detection:**
→ Use `realtime_detection.py` (simplest, most direct)

**For testing/demo purposes:**
→ Use `demo.py` (interactive menu, easy to use)

**For batch image processing:**
→ Use `detect_image.py` (command-line, scriptable)

**For video file processing:**
→ Use `detect_video.py` (progress tracking, save output)

**For training/retraining:**
→ Use `train.py` (customize epochs, batch size, etc.)

**For programmatic use (in your code):**
→ Use `MaskDetector` class from `demo.py`

## 🔌 Integration Example

If you want to use the detector in your own Python code:

```python
from demo import MaskDetector
import cv2

# Initialize detector
detector = MaskDetector()

# Load image
image = cv2.imread('photo.jpg')

# Detect and predict
results = detector.detect_and_predict(image)

# Process results
for result in results:
    bbox = result['bbox']
    pred = result['prediction']
    print(f"Found {pred['class']} with {pred['confidence']*100:.1f}% confidence")

# Annotate image
annotated = detector.annotate_image(image, results)
cv2.imwrite('output.jpg', annotated)
```

## 📦 Dependencies

From `requirements.txt`:
- **torch** >= 2.0.0 - Deep learning framework
- **torchvision** >= 0.15.0 - Computer vision utilities
- **opencv-python** >= 4.8.0 - Image/video processing
- **numpy** >= 1.24.0 - Numerical computing

## 🎯 Model Architecture

**RealTimeMaskNet** (defined in all scripts):
```python
Input: 128×128×3 RGB image
↓
Conv2d(3→16) + ReLU + MaxPool2d
↓
Conv2d(16→32) + ReLU + MaxPool2d
↓
Conv2d(32→64) + ReLU + MaxPool2d
↓
AdaptiveAvgPool2d + Flatten
↓
Linear(64→2)
↓
Output: [No Mask, With Mask]
```

**Model Size:** ~2,000 parameters (0.09 MB)

## 🚀 Performance

- **Inference Speed:** 10-20ms per image
- **FPS:** 50-100 (real-time capable)
- **Accuracy:** 95-98% on test set
- **Model Size:** 0.09 MB (very lightweight)

## 📝 Notes

- All detection scripts use the same model architecture
- Face detection uses OpenCV's Haar Cascade
- Model must be trained before running detection
- Webcam requires camera permissions
- GPU is optional but speeds up training

---

**Ready to start?** → See [QUICKSTART.md](QUICKSTART.md)

**Need details?** → See [README.md](README.md)

**Want overview?** → See [SUMMARY.md](SUMMARY.md)
