# Fix: Cannot Detect "With Mask" in Real-time

## 🔍 Problem
The real-time detection is not correctly identifying when you're wearing a mask.

## ✅ Solution Steps

### Step 1: Install Dependencies (If Not Done)

```bash
pip install -r requirements.txt
```

### Step 2: Test the Model

```bash
python quick_test.py
```

**What to look for:**
- Should show 75-100% accuracy on test images
- "With Mask" images should predict "With Mask"
- "Without Mask" images should predict "No Mask"

**If test fails or accuracy is low:**
→ Go to Step 3 (Retrain)

**If test passes:**
→ Go to Step 4 (Face Detection)

### Step 3: Retrain the Model

The existing model might not be trained properly. Retrain it:

```bash
# Backup old model (optional)
mv mask_model.pth mask_model_old.pth

# Train new model
python train.py
```

**What to expect:**
- Training will take 5-10 minutes on CPU
- Accuracy should reach >95% by epoch 10
- Test accuracy should be >90%

**Example output:**
```
Epoch [1/10] Loss: 0.3245 Accuracy: 87.23%
Epoch [2/10] Loss: 0.1892 Accuracy: 92.45%
...
Epoch [10/10] Loss: 0.0523 Accuracy: 98.12%

Test Accuracy: 96.45%
```

**After training, test again:**
```bash
python quick_test.py
```

### Step 4: Improve Face Detection

The updated `realtime_detection.py` now has better face detection parameters.

**Run real-time detection:**
```bash
python realtime_detection.py
```

**Tips for better detection:**
1. **Lighting**: Face the light source, avoid backlighting
2. **Distance**: Stay 1-3 feet from camera
3. **Position**: Face the camera directly
4. **Mask**: Ensure mask covers nose and mouth properly

**Watch the terminal output:**
- Every 30 frames, it will print predictions
- Check if predictions match what you're wearing

```
Face detected: With Mask (95.3%) - Raw output: [-2.1  3.4]
Face detected: No Mask (87.2%) - Raw output: [1.8  -1.2]
```

### Step 5: If Still Not Working

**Option A: Train Longer**

Edit `train.py`, change:
```python
EPOCHS = 20  # Change from 10 to 20
```

Then retrain:
```bash
python train.py
```

**Option B: Adjust Face Detection**

Edit `realtime_detection.py`, find this section and adjust:

```python
# More sensitive (detects more faces)
faces = face_cascade.detectMultiScale(
    gray,
    scaleFactor=1.03,  # Lower = more sensitive
    minNeighbors=2,    # Lower = less strict
    minSize=(30, 30),  # Smaller = detect smaller faces
    flags=cv2.CASCADE_SCALE_IMAGE
)
```

**Option C: Test on Images First**

Before using webcam, test on static images:

```bash
# Take a photo of yourself with mask
# Save as test_with_mask.jpg

python detect_image.py test_with_mask.jpg

# Take a photo without mask
# Save as test_without_mask.jpg

python detect_image.py test_without_mask.jpg
```

## 🎯 Quick Checklist

- [ ] Dependencies installed (`pip install -r requirements.txt`)
- [ ] Model tested (`python quick_test.py`) - should show >75% accuracy
- [ ] Model retrained if needed (`python train.py`)
- [ ] Good lighting when using webcam
- [ ] Face clearly visible to camera
- [ ] Mask properly worn (covers nose and mouth)

## 🔧 Most Common Issues & Fixes

| Issue | Fix |
|-------|-----|
| Model always predicts "No Mask" | Retrain: `python train.py` |
| Model always predicts "With Mask" | Retrain: `python train.py` |
| No face detected | Improve lighting, adjust face detection |
| Low confidence scores | Train for more epochs (20-30) |
| Works on images but not webcam | Check webcam quality and lighting |

## 📊 Expected Performance

After proper training:
- **Training accuracy**: >95%
- **Test accuracy**: >90%
- **Real-time confidence**: >80%
- **Detection speed**: 50-100 FPS

## 🎬 Complete Workflow

```bash
# 1. Install (first time only)
pip install -r requirements.txt

# 2. Test model
python quick_test.py

# 3. If test fails, retrain
python train.py

# 4. Test again
python quick_test.py

# 5. Run real-time detection
python realtime_detection.py
```

## 💡 Pro Tips

1. **Always test the model first** with `quick_test.py` before using real-time
2. **Retrain if accuracy is low** - this fixes 90% of issues
3. **Good lighting is crucial** for webcam detection
4. **Train for 20-30 epochs** for best results
5. **Test on images first** before testing real-time

## 📝 Debug Information

If you need to report an issue, run these commands:

```bash
# Test model
python quick_test.py > debug_model.txt 2>&1

# Check dataset
echo "Dataset info:" > debug_dataset.txt
echo "With mask: $(ls datasets/train/with_mask/*.jpg | wc -l)" >> debug_dataset.txt
echo "Without mask: $(ls datasets/train/without_mask/*.jpg | wc -l)" >> debug_dataset.txt

# Check packages
python -c "import torch, cv2; print('PyTorch:', torch.__version__); print('OpenCV:', cv2.__version__)" > debug_packages.txt 2>&1
```

---

## 🚀 TL;DR (Too Long; Didn't Read)

**Most likely fix:**

```bash
# Retrain the model
python train.py

# Test it
python quick_test.py

# Run real-time
python realtime_detection.py
```

**If that doesn't work:**
- Check lighting
- Face the camera directly
- Ensure mask covers nose and mouth
- See TROUBLESHOOTING.md for more details

---

**Need more help?** See `TROUBLESHOOTING.md` for detailed solutions.
