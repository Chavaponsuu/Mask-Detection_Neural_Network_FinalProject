import torch
import torch.nn as nn
import cv2
import numpy as np
from pathlib import Path

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
print("Using device:", device)

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
# Face Detection (Haar Cascade)
# =========================
# Download haarcascade if needed
cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
face_cascade = cv2.CascadeClassifier(cascade_path)

if face_cascade.empty():
    print("Error: Could not load face cascade classifier")
    exit()

# =========================
# Preprocessing Function
# =========================
def preprocess_face(face_img, image_size=(128, 128)):
    """Preprocess face image for model input"""
    # Resize
    face_resized = cv2.resize(face_img, image_size)
    
    # BGR -> RGB
    face_rgb = cv2.cvtColor(face_resized, cv2.COLOR_BGR2RGB)
    
    # Normalize
    face_normalized = face_rgb.astype(np.float32) / 255.0
    
    # HWC -> CHW
    face_transposed = np.transpose(face_normalized, (2, 0, 1))
    
    # Add batch dimension
    face_tensor = torch.from_numpy(face_transposed).unsqueeze(0).float()
    
    return face_tensor

# =========================
# Real-time Detection
# =========================
def run_realtime_detection():
    """Run real-time mask detection on webcam feed"""
    
    # Open webcam
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Could not open webcam")
        return
    
    print("\n=== Real-time Mask Detection ===")
    print("Press 'q' to quit")
    print("Press 's' to save screenshot")
    
    # Class labels
    class_names = ['No Mask', 'With Mask']
    colors = [(0, 0, 255), (0, 255, 0)]  # Red for No Mask, Green for With Mask
    
    frame_count = 0
    
    while True:
        # Read frame
        ret, frame = cap.read()
        
        if not ret:
            print("Error: Could not read frame")
            break
        
        frame_count += 1
        
        # Convert to grayscale for face detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect faces (adjusted parameters for better detection)
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.05,  # More sensitive
            minNeighbors=3,     # Less strict
            minSize=(40, 40),   # Smaller minimum size
            flags=cv2.CASCADE_SCALE_IMAGE
        )
        
        # Process each detected face
        for (x, y, w, h) in faces:
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
            
            # Debug: Print predictions
            if frame_count % 30 == 0:  # Print every 30 frames
                print(f"Face detected: {class_names[pred_class]} ({conf_score*100:.1f}%) - Raw output: {output[0].cpu().numpy()}")
            
            # Draw bounding box
            color = colors[pred_class]
            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
            
            # Draw label with confidence
            label = f"{class_names[pred_class]}: {conf_score*100:.1f}%"
            
            # Background for text
            (text_width, text_height), _ = cv2.getTextSize(
                label, 
                cv2.FONT_HERSHEY_SIMPLEX, 
                0.6, 
                2
            )
            cv2.rectangle(
                frame, 
                (x, y - text_height - 10), 
                (x + text_width, y), 
                color, 
                -1
            )
            
            # Draw text
            cv2.putText(
                frame, 
                label, 
                (x, y - 5), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                0.6, 
                (255, 255, 255), 
                2
            )
        
        # Display info
        info_text = f"Faces: {len(faces)} | Frame: {frame_count}"
        cv2.putText(
            frame, 
            info_text, 
            (10, 30), 
            cv2.FONT_HERSHEY_SIMPLEX, 
            0.7, 
            (0, 255, 0), 
            2
        )
        
        # Display instructions
        cv2.putText(
            frame,
            "Press 'q' to quit | 's' to screenshot",
            (10, frame.shape[0] - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1
        )
        
        # Display frame
        cv2.imshow('Real-time Mask Detection', frame)
        
        # Handle key presses
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q'):
            print("\nQuitting...")
            break
        elif key == ord('s'):
            screenshot_path = f"screenshot_{frame_count}.jpg"
            cv2.imwrite(screenshot_path, frame)
            print(f"Screenshot saved: {screenshot_path}")
    
    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    print("Webcam released")

# =========================
# Main
# =========================
if __name__ == "__main__":
    run_realtime_detection()
