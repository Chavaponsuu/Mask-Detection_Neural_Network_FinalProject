"""
Quick test script to verify installation and setup
"""

import sys

def test_imports():
    """Test if all required packages are installed"""
    print("Testing imports...")
    
    try:
        import torch
        print(f"✅ PyTorch {torch.__version__}")
        print(f"   CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"   CUDA device: {torch.cuda.get_device_name(0)}")
    except ImportError:
        print("❌ PyTorch not installed")
        return False
    
    try:
        import cv2
        print(f"✅ OpenCV {cv2.__version__}")
    except ImportError:
        print("❌ OpenCV not installed")
        return False
    
    try:
        import numpy as np
        print(f"✅ NumPy {np.__version__}")
    except ImportError:
        print("❌ NumPy not installed")
        return False
    
    return True

def test_dataset():
    """Test if dataset is accessible"""
    print("\nTesting dataset...")
    
    from pathlib import Path
    
    base_path = Path("datasets/train")
    
    if not base_path.exists():
        print("❌ Dataset directory not found")
        return False
    
    with_mask = list(base_path.glob("with_mask/*.jpg"))
    without_mask = list(base_path.glob("without_mask/*.jpg"))
    
    print(f"✅ Dataset found")
    print(f"   With mask: {len(with_mask)} images")
    print(f"   Without mask: {len(without_mask)} images")
    print(f"   Total: {len(with_mask) + len(without_mask)} images")
    
    if len(with_mask) == 0 or len(without_mask) == 0:
        print("⚠️  Warning: One or more classes have no images")
        return False
    
    return True

def test_model():
    """Test if model file exists"""
    print("\nTesting model...")
    
    from pathlib import Path
    
    model_path = Path("mask_model.pth")
    
    if model_path.exists():
        print(f"✅ Model found: {model_path}")
        
        # Check model size
        size_mb = model_path.stat().st_size / (1024 * 1024)
        print(f"   Size: {size_mb:.2f} MB")
        
        return True
    else:
        print("⚠️  Model not found (run train.py first)")
        return False

def test_webcam():
    """Test if webcam is accessible"""
    print("\nTesting webcam...")
    
    try:
        import cv2
        
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            print("❌ Could not open webcam")
            return False
        
        ret, frame = cap.read()
        cap.release()
        
        if not ret:
            print("❌ Could not read from webcam")
            return False
        
        print(f"✅ Webcam accessible")
        print(f"   Resolution: {frame.shape[1]}x{frame.shape[0]}")
        
        return True
        
    except Exception as e:
        print(f"❌ Webcam test failed: {e}")
        return False

def test_face_detection():
    """Test if face detection works"""
    print("\nTesting face detection...")
    
    try:
        import cv2
        
        cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        face_cascade = cv2.CascadeClassifier(cascade_path)
        
        if face_cascade.empty():
            print("❌ Could not load face cascade")
            return False
        
        print("✅ Face detection ready")
        return True
        
    except Exception as e:
        print(f"❌ Face detection test failed: {e}")
        return False

def main():
    """Run all tests"""
    print("=" * 50)
    print("Face Mask Detection - Setup Test")
    print("=" * 50)
    
    results = []
    
    results.append(("Imports", test_imports()))
    results.append(("Dataset", test_dataset()))
    results.append(("Model", test_model()))
    results.append(("Webcam", test_webcam()))
    results.append(("Face Detection", test_face_detection()))
    
    print("\n" + "=" * 50)
    print("Test Summary")
    print("=" * 50)
    
    for name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{name:20s} {status}")
    
    all_passed = all(result[1] for result in results[:2])  # Only check critical tests
    
    print("\n" + "=" * 50)
    
    if all_passed:
        print("✅ Setup complete! You can now:")
        print("   1. Train model: python train.py")
        print("   2. Run real-time: python realtime_detection.py")
        print("   3. Test on image: python detect_image.py <image_path>")
    else:
        print("❌ Setup incomplete. Please fix the issues above.")
        print("\nInstall missing packages:")
        print("   pip install -r requirements.txt")
    
    print("=" * 50)

if __name__ == "__main__":
    main()
