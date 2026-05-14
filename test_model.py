"""
Test the trained model to verify it's working correctly
"""

import torch
import torch.nn as nn
import cv2
import numpy as np
from pathlib import Path

# =========================
# Model Definition
# =========================
class RealTimeMaskNet(nn.Module):
    def __init__(self):
        super(RealTimeMaskNet, self).__init__()
        
        self.features = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(64, 2)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

# =========================
# Test Model
# =========================
def test_model():
    print("=" * 60)
    print("Model Testing & Diagnostics")
    print("=" * 60)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n✅ Device: {device}")
    
    # Load model
    model = RealTimeMaskNet().to(device)
    
    model_path = "mask_model.pth"
    if not Path(model_path).exists():
        print(f"\n❌ Model not found: {model_path}")
        print("Please run: python train.py")
        return
    
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    print(f"✅ Model loaded: {model_path}")
    
    # Test on sample images
    print("\n" + "=" * 60)
    print("Testing on Sample Images")
    print("=" * 60)
    
    # Get sample images
    with_mask_dir = Path("datasets/train/with_mask")
    without_mask_dir = Path("datasets/train/without_mask")
    
    with_mask_images = list(with_mask_dir.glob("*.jpg"))[:5]
    without_mask_images = list(without_mask_dir.glob("*.jpg"))[:5]
    
    class_names = ['No Mask', 'With Mask']
    
    def preprocess(img_path):
        img = cv2.imread(str(img_path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (128, 128))
        img = img.astype(np.float32) / 255.0
        img = np.transpose(img, (2, 0, 1))
        img_tensor = torch.from_numpy(img).unsqueeze(0).float()
        return img_tensor.to(device)
    
    # Test WITH MASK images
    print("\n📸 Testing WITH MASK images:")
    correct_with = 0
    for img_path in with_mask_images:
        img_tensor = preprocess(img_path)
        
        with torch.no_grad():
            output = model(img_tensor)
            probabilities = torch.softmax(output, dim=1)
            confidence, predicted = torch.max(probabilities, 1)
            
            pred_class = predicted.item()
            conf_score = confidence.item()
        
        result = "✅" if pred_class == 1 else "❌"
        if pred_class == 1:
            correct_with += 1
        
        print(f"  {result} {img_path.name[:30]:30s} → {class_names[pred_class]:10s} ({conf_score*100:.1f}%)")
    
    # Test WITHOUT MASK images
    print("\n📸 Testing WITHOUT MASK images:")
    correct_without = 0
    for img_path in without_mask_images:
        img_tensor = preprocess(img_path)
        
        with torch.no_grad():
            output = model(img_tensor)
            probabilities = torch.softmax(output, dim=1)
            confidence, predicted = torch.max(probabilities, 1)
            
            pred_class = predicted.item()
            conf_score = confidence.item()
        
        result = "✅" if pred_class == 0 else "❌"
        if pred_class == 0:
            correct_without += 1
        
        print(f"  {result} {img_path.name[:30]:30s} → {class_names[pred_class]:10s} ({conf_score*100:.1f}%)")
    
    # Summary
    print("\n" + "=" * 60)
    print("Test Results Summary")
    print("=" * 60)
    print(f"With Mask:    {correct_with}/5 correct ({correct_with*20}%)")
    print(f"Without Mask: {correct_without}/5 correct ({correct_without*20}%)")
    print(f"Overall:      {correct_with + correct_without}/10 correct ({(correct_with + correct_without)*10}%)")
    
    # Diagnosis
    print("\n" + "=" * 60)
    print("Diagnosis")
    print("=" * 60)
    
    if correct_with >= 4 and correct_without >= 4:
        print("✅ Model is working correctly!")
        print("   The issue might be with face detection in real-time.")
        print("\n💡 Suggestions:")
        print("   1. Make sure your face is well-lit")
        print("   2. Face the camera directly")
        print("   3. Ensure the mask covers your face properly")
        print("   4. Try adjusting the face detection parameters")
    elif correct_with < 3:
        print("❌ Model has trouble detecting WITH MASK")
        print("\n💡 Suggestions:")
        print("   1. Retrain the model: python train.py")
        print("   2. Check if training data is correct")
        print("   3. Train for more epochs (change EPOCHS in train.py)")
    elif correct_without < 3:
        print("❌ Model has trouble detecting WITHOUT MASK")
        print("\n💡 Suggestions:")
        print("   1. Retrain the model: python train.py")
        print("   2. Check if training data is correct")
        print("   3. Train for more epochs (change EPOCHS in train.py)")
    else:
        print("⚠️  Model accuracy is moderate")
        print("\n💡 Suggestions:")
        print("   1. Retrain for more epochs")
        print("   2. Add data augmentation")
        print("   3. Use a larger model")
    
    print("\n" + "=" * 60)

if __name__ == "__main__":
    test_model()
