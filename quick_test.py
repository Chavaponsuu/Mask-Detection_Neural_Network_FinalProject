"""
Quick test to check if model predictions are working
Run this BEFORE real-time detection to verify the model
"""

import torch
import torch.nn as nn
import cv2
import numpy as np
from pathlib import Path

class RealTimeMaskNet(nn.Module):
    def __init__(self):
        super(RealTimeMaskNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2, 2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2, 2)
        )
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(64, 2)
        )
    def forward(self, x):
        return self.classifier(self.features(x))

print("=" * 60)
print("QUICK MODEL TEST")
print("=" * 60)

# Check if model exists
if not Path("mask_model.pth").exists():
    print("\n❌ ERROR: mask_model.pth not found!")
    print("   Please run: python train.py")
    exit(1)

# Load model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = RealTimeMaskNet().to(device)
model.load_state_dict(torch.load("mask_model.pth", map_location=device))
model.eval()
print(f"\n✅ Model loaded successfully on {device}")

# Test on 2 images
class_names = ['No Mask', 'With Mask']

def test_image(img_path, expected_class):
    img = cv2.imread(str(img_path))
    if img is None:
        return None
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (128, 128))
    img = img.astype(np.float32) / 255.0
    img = np.transpose(img, (2, 0, 1))
    img_tensor = torch.from_numpy(img).unsqueeze(0).float().to(device)
    
    with torch.no_grad():
        output = model(img_tensor)
        probs = torch.softmax(output, dim=1)
        conf, pred = torch.max(probs, 1)
    
    pred_class = pred.item()
    conf_score = conf.item()
    
    is_correct = (pred_class == expected_class)
    status = "✅" if is_correct else "❌"
    
    print(f"\n{status} {img_path.name[:40]}")
    print(f"   Predicted: {class_names[pred_class]} ({conf_score*100:.1f}%)")
    print(f"   Expected:  {class_names[expected_class]}")
    print(f"   Raw scores: No Mask={probs[0][0].item():.3f}, With Mask={probs[0][1].item():.3f}")
    
    return is_correct

print("\n" + "=" * 60)
print("Testing on sample images...")
print("=" * 60)

# Test with mask
with_mask_imgs = list(Path("datasets/train/with_mask").glob("*.jpg"))[:2]
without_mask_imgs = list(Path("datasets/train/without_mask").glob("*.jpg"))[:2]

results = []

if with_mask_imgs:
    print("\n📸 WITH MASK images:")
    for img in with_mask_imgs:
        result = test_image(img, expected_class=1)
        if result is not None:
            results.append(result)

if without_mask_imgs:
    print("\n📸 WITHOUT MASK images:")
    for img in without_mask_imgs:
        result = test_image(img, expected_class=0)
        if result is not None:
            results.append(result)

# Summary
print("\n" + "=" * 60)
print("RESULTS")
print("=" * 60)

if results:
    correct = sum(results)
    total = len(results)
    accuracy = (correct / total) * 100
    
    print(f"\n✅ Correct: {correct}/{total} ({accuracy:.0f}%)")
    
    if accuracy >= 75:
        print("\n🎉 Model is working well!")
        print("   You can now run: python realtime_detection.py")
    else:
        print("\n⚠️  Model accuracy is low!")
        print("   Please retrain: python train.py")
else:
    print("\n❌ No test images found!")
    print("   Check if datasets/train/ has images")

print("\n" + "=" * 60)
