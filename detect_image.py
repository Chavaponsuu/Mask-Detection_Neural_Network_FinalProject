import torch
import torch.nn as nn
import cv2
import numpy as np
from pathlib import Path
import argparse

# =========================
# CNN Model (same as training)
# =========================
class RealTimeMaskNet(nn.Module):
    def __init__(self):
        super(RealTimeMaskNet, self).__init__()
        
        # Feature Extractor
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(in_channels=3, 
                     out_channels=16, 
                     kernel_size=3, 
                     padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            
            # Block 2
            nn.Conv2d(in_channels=16, 
                     out_channels=32, 
                     kernel_size=3, 
                     padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            
            # Block 3
            nn.Conv2d(in_channels=32, 
                     out_channels=64, 
                     kernel_size=3, 
                     padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        
        # Classifier
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
# Load Model
# =========================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = RealTimeMaskNet().to(device)

# Load trained weights
model_path = "mask_model.pth"
if Path(model_path).exists():
    model.load_state_dict(torch.load(model_path, map_location=device))
    print(f"Model loaded from {model_path}")
else:
    print(f"Error: {model_path} not found. Please train the model first.")
    exit()

model.eval()

# =========================
# Face Detection
# =========================
cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
face_cascade = cv2.CascadeClassifier(cascade_path)

# =========================
# Preprocessing Function
# =========================
def preprocess_face(face_img, image_size=(128, 128)):
    """Preprocess face image for model input"""
    face_resized = cv2.resize(face_img, image_size)
    face_rgb = cv2.cvtColor(face_resized, cv2.COLOR_BGR2RGB)
    face_normalized = face_rgb.astype(np.float32) / 255.0
    face_transposed = np.transpose(face_normalized, (2, 0, 1))
    face_tensor = torch.from_numpy(face_transposed).unsqueeze(0).float()
    return face_tensor

# =========================
# Detection Function
# =========================
def detect_mask_in_image(image_path, output_path=None):
    """Detect masks in a single image"""
    
    # Read image
    frame = cv2.imread(image_path)
    
    if frame is None:
        print(f"Error: Could not read image {image_path}")
        return
    
    # Convert to grayscale for face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detect faces
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(60, 60)
    )
    
    print(f"Detected {len(faces)} face(s)")
    
    # Class labels
    class_names = ['No Mask', 'With Mask']
    colors = [(0, 0, 255), (0, 255, 0)]
    
    # Process each detected face
    for i, (x, y, w, h) in enumerate(faces):
        # Extract face ROI
        face_roi = frame[y:y+h, x:x+w]
        
        # Preprocess face
        face_tensor = preprocess_face(face_roi).to(device)
        
        # Predict
        with torch.no_grad():
            output = model(face_tensor)
            probabilities = torch.softmax(output, dim=1)
            confidence, predicted = torch.max(probabilities, 1)
            
            pred_class = predicted.item()
            conf_score = confidence.item()
        
        print(f"Face {i+1}: {class_names[pred_class]} ({conf_score*100:.1f}%)")
        
        # Draw bounding box
        color = colors[pred_class]
        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 3)
        
        # Draw label
        label = f"{class_names[pred_class]}: {conf_score*100:.1f}%"
        
        (text_width, text_height), _ = cv2.getTextSize(
            label, 
            cv2.FONT_HERSHEY_SIMPLEX, 
            0.8, 
            2
        )
        cv2.rectangle(
            frame, 
            (x, y - text_height - 15), 
            (x + text_width, y), 
            color, 
            -1
        )
        cv2.putText(
            frame, 
            label, 
            (x, y - 8), 
            cv2.FONT_HERSHEY_SIMPLEX, 
            0.8, 
            (255, 255, 255), 
            2
        )
    
    # Save or display
    if output_path:
        cv2.imwrite(output_path, frame)
        print(f"Result saved to {output_path}")
    else:
        cv2.imshow('Mask Detection Result', frame)
        print("Press any key to close...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()

# =========================
# Main
# =========================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Detect face masks in images')
    parser.add_argument('image', type=str, help='Path to input image')
    parser.add_argument('--output', '-o', type=str, help='Path to save output image')
    
    args = parser.parse_args()
    
    detect_mask_in_image(args.image, args.output)
