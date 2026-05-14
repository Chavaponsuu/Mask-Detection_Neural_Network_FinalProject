import torch
import torch.nn as nn
import cv2
import numpy as np
from pathlib import Path
import argparse
import time

# =========================
# CNN Model
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
# Load Model
# =========================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = RealTimeMaskNet().to(device)
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
# Preprocessing
# =========================
def preprocess_face(face_img, image_size=(128, 128)):
    face_resized = cv2.resize(face_img, image_size)
    face_rgb = cv2.cvtColor(face_resized, cv2.COLOR_BGR2RGB)
    face_normalized = face_rgb.astype(np.float32) / 255.0
    face_transposed = np.transpose(face_normalized, (2, 0, 1))
    face_tensor = torch.from_numpy(face_transposed).unsqueeze(0).float()
    return face_tensor

# =========================
# Video Processing
# =========================
def process_video(input_path, output_path=None, display=True):
    """Process video file for mask detection"""
    
    # Open video
    cap = cv2.VideoCapture(input_path)
    
    if not cap.isOpened():
        print(f"Error: Could not open video {input_path}")
        return
    
    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"\nVideo Info:")
    print(f"  Resolution: {width}x{height}")
    print(f"  FPS: {fps}")
    print(f"  Total Frames: {total_frames}")
    print(f"  Duration: {total_frames/fps:.2f}s")
    
    # Setup video writer if output specified
    writer = None
    if output_path:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        print(f"\nSaving output to: {output_path}")
    
    # Class labels
    class_names = ['No Mask', 'With Mask']
    colors = [(0, 0, 255), (0, 255, 0)]
    
    frame_count = 0
    start_time = time.time()
    
    print("\nProcessing video...")
    print("Press 'q' to quit early")
    
    while True:
        ret, frame = cap.read()
        
        if not ret:
            break
        
        frame_count += 1
        
        # Convert to grayscale for face detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect faces
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(60, 60)
        )
        
        # Process each face
        for (x, y, w, h) in faces:
            face_roi = frame[y:y+h, x:x+w]
            face_tensor = preprocess_face(face_roi).to(device)
            
            with torch.no_grad():
                output = model(face_tensor)
                probabilities = torch.softmax(output, dim=1)
                confidence, predicted = torch.max(probabilities, 1)
                
                pred_class = predicted.item()
                conf_score = confidence.item()
            
            # Draw bounding box
            color = colors[pred_class]
            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
            
            # Draw label
            label = f"{class_names[pred_class]}: {conf_score*100:.1f}%"
            
            (text_width, text_height), _ = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
            )
            cv2.rectangle(
                frame, 
                (x, y - text_height - 10), 
                (x + text_width, y), 
                color, 
                -1
            )
            cv2.putText(
                frame, label, (x, y - 5), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, 
                (255, 255, 255), 2
            )
        
        # Add progress info
        progress = (frame_count / total_frames) * 100
        info_text = f"Frame: {frame_count}/{total_frames} ({progress:.1f}%) | Faces: {len(faces)}"
        cv2.putText(
            frame, info_text, (10, 30), 
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, 
            (255, 255, 255), 2
        )
        
        # Write frame
        if writer:
            writer.write(frame)
        
        # Display frame
        if display:
            cv2.imshow('Video Mask Detection', frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("\nStopped by user")
                break
        
        # Progress update
        if frame_count % 30 == 0:
            elapsed = time.time() - start_time
            fps_actual = frame_count / elapsed
            eta = (total_frames - frame_count) / fps_actual
            print(f"Progress: {progress:.1f}% | FPS: {fps_actual:.1f} | ETA: {eta:.1f}s")
    
    # Cleanup
    cap.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()
    
    # Final stats
    elapsed = time.time() - start_time
    print(f"\n✅ Processing complete!")
    print(f"  Processed: {frame_count} frames")
    print(f"  Time: {elapsed:.2f}s")
    print(f"  Average FPS: {frame_count/elapsed:.2f}")
    
    if output_path:
        print(f"  Output saved: {output_path}")

# =========================
# Main
# =========================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Detect face masks in video files')
    parser.add_argument('video', type=str, help='Path to input video')
    parser.add_argument('--output', '-o', type=str, help='Path to save output video')
    parser.add_argument('--no-display', action='store_true', help='Do not display video while processing')
    
    args = parser.parse_args()
    
    process_video(args.video, args.output, display=not args.no_display)
