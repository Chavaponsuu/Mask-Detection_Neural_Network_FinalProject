# Troubleshooting Guide

## Issue: Cannot Detect "With Mask" in Real-time

### Possible Causes & Solutions

#### 1. Model Not Trained Properly

**Symptoms:**
- Always predicts "No Mask"
- Low confidence scores
- Inconsistent predictions

**Solutions:**

```bash
# Retrain the model
python train.py
```

**Check training output:**
- Training accuracy should reach >95%
- Test accuracy should be >90%
- If accuracy is low, train for more epochs

**Edit train.py to train longer:**
```python
EPOCHS = 20  # Change from 10 to 20
```

#### 2. Face Detection Issues

**Symptoms:**
- No faces detected
- Faces detected but predictions are wrong
- Works on images but not webcam

**Solutions:**

**A. Improve lighting:**
- Face the camera directly
- Ensure good lighting on your face
- Avoid backlighting

**B. Adjust face detection parameters:**

The updated `realtime_detection.py` now uses more sensitive parameters:
```python
faces = face_cascade.detectMultiScale(
    gray,
    scaleFactor=1.05,  # More sensitive (was 1.1)
    minNeighbors=3,    # Less strict (was 5)
    minSize=(40, 40),  # Smaller faces (was 60x60)
    flags=cv2.CASCADE_SCALE_IMAGE
)
```

**C. Test face detection separately:**
```bash
python -c "
import cv2
cap = cv2.VideoCapture(0)
cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
while True:
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = cascade.detectMultiScale(gray, 1.05, 3, minSize=(40,40))
    for (x,y,w,h) in faces:
        cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), 2)
    cv2.putText(frame, f'Faces: {len(faces)}', (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
    cv2.imshow('Face Detection Test', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
"
```

#### 3. Label Mapping Issue

**Check the label mapping:**

In training (`train.py`):
```python
# with_mask = 1
# without_mask = 0
```

In detection scripts:
```python
class_names = ['No Mask', 'With Mask']
# Index 0 = No Mask (without_mask)
# Index 1 = With Mask (with_mask)
```

This is **correct**. If you see reversed predictions, the model needs retraining.

#### 4. Model File Corrupted

**Symptoms:**
- Random predictions
- Errors when loading model
- Very low accuracy

**Solutions:**

```bash
# Delete old model
rm mask_model.pth

# Retrain from scratch
python train.py
```

#### 5. Webcam Image Quality

**Symptoms:**
- Works on dataset images but not webcam
- Predictions are inconsistent

**Solutions:**

**A. Check webcam resolution:**
```bash
python -c "
import cv2
cap = cv2.VideoCapture(0)
ret, frame = cap.read()
print(f'Webcam resolution: {frame.shape[1]}x{frame.shape[0]}')
cap.release()
"
```

**B. Improve webcam quality:**
- Clean your webcam lens
- Increase room lighting
- Use a better quality webcam if available

**C. Adjust preprocessing:**
The model expects 128x128 images. Make sure face extraction is clear.

#### 6. Mask Type Issues

**Symptoms:**
- Detects some masks but not others
- Works better with certain mask colors/types

**Solutions:**

**A. Training data diversity:**
- Check if your training data includes various mask types
- Retrain with more diverse mask images

**B. Test with different masks:**
- Try different colored masks
- Try surgical vs cloth masks
- Ensure mask covers nose and mouth

#### 7. Distance from Camera

**Symptoms:**
- Works when close but not far
- Works when far but not close

**Solutions:**

**A. Optimal distance:**
- Stay 1-3 feet from camera
- Ensure face fills 20-40% of frame

**B. Adjust minSize in face detection:**
```python
# For closer faces (larger in frame)
minSize=(80, 80)

# For farther faces (smaller in frame)
minSize=(30, 30)
```

## Diagnostic Steps

### Step 1: Test Model on Dataset Images

```bash
python test_model.py
```

This will test the model on sample images from your dataset.

**Expected output:**
- With Mask: 4-5/5 correct
- Without Mask: 4-5/5 correct

**If model test fails:**
→ Retrain the model: `python train.py`

### Step 2: Test on Single Image

```bash
# Test with a mask image
python detect_image.py datasets/train/with_mask/with_mask_1.jpg

# Test without mask image
python detect_image.py datasets/train/without_mask/without_mask_1.jpg
```

**Expected output:**
- Should correctly identify mask status
- Confidence should be >80%

**If image test fails:**
→ Model needs retraining

### Step 3: Test Face Detection

Run the face detection test code above to verify faces are being detected.

**Expected output:**
- Should detect your face
- Green box around face
- "Faces: 1" displayed

**If face detection fails:**
→ Check lighting and camera position

### Step 4: Test Real-time with Debug

The updated `realtime_detection.py` now prints debug info every 30 frames.

```bash
python realtime_detection.py
```

**Watch the terminal output:**
```
Face detected: With Mask (95.3%) - Raw output: [-2.1  3.4]
Face detected: No Mask (87.2%) - Raw output: [1.8  -1.2]
```

**Interpretation:**
- Raw output shows model's confidence for each class
- Higher value = higher confidence
- Index 0 = No Mask, Index 1 = With Mask

## Quick Fixes

### Fix 1: Retrain Model (Most Common)

```bash
# Backup old model
mv mask_model.pth mask_model_old.pth

# Train new model
python train.py

# Test new model
python realtime_detection.py
```

### Fix 2: Better Face Detection

Edit `realtime_detection.py`:

```python
# More sensitive (detects more faces, may have false positives)
faces = face_cascade.detectMultiScale(
    gray,
    scaleFactor=1.03,
    minNeighbors=2,
    minSize=(30, 30)
)

# Less sensitive (fewer false positives, may miss some faces)
faces = face_cascade.detectMultiScale(
    gray,
    scaleFactor=1.2,
    minNeighbors=7,
    minSize=(80, 80)
)
```

### Fix 3: Train Longer

Edit `train.py`:

```python
EPOCHS = 30  # Increase from 10
```

Then retrain:
```bash
python train.py
```

### Fix 4: Larger Model

Edit `train.py` to use a larger model:

```python
class RealTimeMaskNet(nn.Module):
    def __init__(self):
        super(RealTimeMaskNet, self).__init__()
        
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(3, 32, 3, padding=1),  # 16 → 32
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            
            # Block 2
            nn.Conv2d(32, 64, 3, padding=1),  # 32 → 64
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            
            # Block 3
            nn.Conv2d(64, 128, 3, padding=1),  # 64 → 128
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            
            # Block 4 (NEW)
            nn.Conv2d(128, 256, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(0.5),  # Add dropout
            nn.Linear(256, 128),  # Add hidden layer
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 2)
        )
```

## Still Having Issues?

### Collect Debug Information

Run these commands and share the output:

```bash
# 1. Test model
python test_model.py > model_test.txt 2>&1

# 2. Check dataset
echo "With mask images:" && ls datasets/train/with_mask/*.jpg | wc -l
echo "Without mask images:" && ls datasets/train/without_mask/*.jpg | wc -l

# 3. Check Python packages
python -c "import torch; import cv2; print('PyTorch:', torch.__version__); print('OpenCV:', cv2.__version__)"

# 4. Test on sample image
python detect_image.py datasets/train/with_mask/with_mask_1.jpg
```

### Common Error Messages

**"Model not found"**
→ Run `python train.py` first

**"Could not open webcam"**
→ Check camera permissions, close other apps using camera

**"No module named 'torch'"**
→ Run `pip install -r requirements.txt`

**"CUDA out of memory"**
→ Reduce batch size in train.py or use CPU

**All predictions are "No Mask"**
→ Model needs retraining: `python train.py`

**All predictions are "With Mask"**
→ Model needs retraining: `python train.py`

## Best Practices

1. **Always retrain after making changes** to model architecture
2. **Test on dataset images first** before testing real-time
3. **Ensure good lighting** for webcam detection
4. **Face the camera directly** for best results
5. **Keep face 1-3 feet from camera**
6. **Train for at least 10 epochs** (20-30 for better accuracy)
7. **Check training accuracy** - should be >95%

## Contact & Support

If you're still having issues after trying these solutions:

1. Run `python test_model.py` and note the results
2. Check if face detection works (Step 3 above)
3. Verify training accuracy was >90%
4. Try retraining with more epochs

---

**Most common solution: Retrain the model with `python train.py`**
