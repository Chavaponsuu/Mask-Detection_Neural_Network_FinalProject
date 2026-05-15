"""
Real-time Face Mask Detection with DNN Face Detection + Temporal Smoothing
===========================================================================
Uses OpenCV's DNN module with Caffe face detector
Includes prediction stabilization to prevent flickering

Stabilization Features:
1. Temporal smoothing with moving average
2. Confidence thresholding
3. Prediction history tracking
4. Majority voting
5. Hysteresis for class changes
"""

import torch
import torch.nn as nn
import cv2
import numpy as np
from pathlib import Path
import time
import urllib.request
from collections import deque, Counter

# =========================
# CNN Model
# =========================
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

# =========================
# Prediction Stabilizer
# =========================
class PredictionStabilizer:
    """
    Stabilizes predictions across frames to prevent flickering
    
    Methods:
    1. Moving Average: Smooth probabilities over time
    2. Majority Voting: Use most common prediction
    3. Confidence Threshold: Only change if confident
    4. Hysteresis: Require stronger evidence to change class
    """
    
    def __init__(self, 
                 history_size=10,
                 confidence_threshold=0.65,
                 hysteresis_margin=0.15,
                 smoothing_alpha=0.7):
        """
        Args:
            history_size: Number of frames to keep in history
            confidence_threshold: Minimum confidence to accept prediction
            hysteresis_margin: Extra confidence needed to change class
            smoothing_alpha: Weight for exponential moving average (0-1)
                           Higher = more weight to current frame
        """
        self.history_size = history_size
        self.confidence_threshold = confidence_threshold
        self.hysteresis_margin = hysteresis_margin
        self.smoothing_alpha = smoothing_alpha
        
        # Prediction history
        self.prediction_history = deque(maxlen=history_size)
        self.probability_history = deque(maxlen=history_size)
        
        # Current stable state
        self.current_class = None
        self.current_confidence = 0.0
        self.smoothed_probs = None
        
        # Statistics
        self.frame_count = 0
        self.class_changes = 0
    
    def update(self, probabilities):
        """
        Update with new prediction and return stabilized result
        
        Args:
            probabilities: numpy array [prob_no_mask, prob_with_mask]
            
        Returns:
            (predicted_class, confidence, is_stable)
        """
        self.frame_count += 1
        
        # Get raw prediction
        raw_class = np.argmax(probabilities)
        raw_confidence = probabilities[raw_class]
        
        # Add to history
        self.prediction_history.append(raw_class)
        self.probability_history.append(probabilities.copy())
        
        # Initialize smoothed probabilities
        if self.smoothed_probs is None:
            self.smoothed_probs = probabilities.copy()
        else:
            # Exponential moving average
            self.smoothed_probs = (
                self.smoothing_alpha * probabilities + 
                (1 - self.smoothing_alpha) * self.smoothed_probs
            )
        
        # Get smoothed prediction
        smoothed_class = np.argmax(self.smoothed_probs)
        smoothed_confidence = self.smoothed_probs[smoothed_class]
        
        # Majority voting from history
        if len(self.prediction_history) >= 3:
            vote_counts = Counter(self.prediction_history)
            majority_class = vote_counts.most_common(1)[0][0]
            majority_ratio = vote_counts[majority_class] / len(self.prediction_history)
        else:
            majority_class = smoothed_class
            majority_ratio = 1.0
        
        # Decision logic with hysteresis
        if self.current_class is None:
            # First prediction
            if smoothed_confidence >= self.confidence_threshold:
                self.current_class = smoothed_class
                self.current_confidence = smoothed_confidence
                is_stable = True
            else:
                # Not confident enough, use majority
                self.current_class = majority_class
                self.current_confidence = smoothed_confidence
                is_stable = False
        else:
            # Check if we should change class
            if smoothed_class != self.current_class:
                # Require higher confidence to change (hysteresis)
                required_confidence = self.confidence_threshold + self.hysteresis_margin
                
                if smoothed_confidence >= required_confidence and majority_ratio >= 0.6:
                    # Strong evidence to change
                    self.current_class = smoothed_class
                    self.current_confidence = smoothed_confidence
                    self.class_changes += 1
                    is_stable = True
                else:
                    # Not confident enough, keep current class
                    is_stable = False
            else:
                # Same class, update confidence
                self.current_confidence = smoothed_confidence
                is_stable = True
        
        return self.current_class, self.current_confidence, is_stable
    
    def get_stats(self):
        """Get stabilizer statistics"""
        return {
            'frames': self.frame_count,
            'class_changes': self.class_changes,
            'stability_ratio': 1.0 - (self.class_changes / max(1, self.frame_count)),
            'avg_confidence': np.mean([p[self.current_class] for p in self.probability_history]) if self.probability_history else 0.0
        }
    
    def reset(self):
        """Reset stabilizer state"""
        self.prediction_history.clear()
        self.probability_history.clear()
        self.current_class = None
        self.current_confidence = 0.0
        self.smoothed_probs = None

# =========================
# Face Tracker
# =========================
class FaceTracker:
    """
    Track individual faces across frames
    Each face gets its own stabilizer
    """
    
    def __init__(self, max_faces=5):
        self.max_faces = max_faces
        self.trackers = {}  # face_id -> stabilizer
        self.face_positions = {}  # face_id -> (x, y, w, h)
        self.next_id = 0
    
    def match_face(self, bbox):
        """
        Match detected face to existing tracked face
        Returns face_id or None if new face
        """
        x, y, w, h = bbox
        center = (x + w/2, y + h/2)
        
        # Find closest existing face
        min_dist = float('inf')
        matched_id = None
        
        for face_id, (px, py, pw, ph) in self.face_positions.items():
            pcenter = (px + pw/2, py + ph/2)
            dist = np.sqrt((center[0] - pcenter[0])**2 + (center[1] - pcenter[1])**2)
            
            # If within reasonable distance (50 pixels)
            if dist < 50 and dist < min_dist:
                min_dist = dist
                matched_id = face_id
        
        return matched_id
    
    def update_face(self, bbox, probabilities):
        """
        Update or create face tracker
        Returns (face_id, predicted_class, confidence, is_stable)
        """
        # Try to match to existing face
        face_id = self.match_face(bbox)
        
        if face_id is None:
            # New face
            face_id = self.next_id
            self.next_id += 1
            self.trackers[face_id] = PredictionStabilizer(
                history_size=10,
                confidence_threshold=0.65,
                hysteresis_margin=0.15,
                smoothing_alpha=0.7
            )
        
        # Update position
        self.face_positions[face_id] = bbox
        
        # Update stabilizer
        pred_class, confidence, is_stable = self.trackers[face_id].update(probabilities)
        
        return face_id, pred_class, confidence, is_stable
    
    def cleanup_old_faces(self, current_bboxes):
        """Remove faces that are no longer detected"""
        current_ids = set()
        for bbox in current_bboxes:
            face_id = self.match_face(bbox)
            if face_id is not None:
                current_ids.add(face_id)
        
        # Remove old faces
        all_ids = set(self.trackers.keys())
        to_remove = all_ids - current_ids
        
        for face_id in to_remove:
            del self.trackers[face_id]
            del self.face_positions[face_id]

# =========================
# CNN Model
# =========================
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

# =========================
# Initialize
# =========================
print("=" * 60)
print("Real-time Face Mask Detection with DNN")
print("=" * 60)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"\n✅ Device: {device}")

# Load CNN model
model = RealTimeMaskNet().to(device)
model_path = "mask_model.pth"

if Path(model_path).exists():
    model.load_state_dict(torch.load(model_path, map_location=device))
    print(f"✅ CNN model loaded: {model_path}")
else:
    print(f"❌ Error: {model_path} not found. Run: python train.py")
    exit()

model.eval()

# Download DNN face detector models if needed
prototxt_url = "https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt"
caffemodel_url = "https://raw.githubusercontent.com/opencv/opencv_3rdparty/dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel"

prototxt_path = "deploy.prototxt"
caffemodel_path = "res10_300x300_ssd_iter_140000.caffemodel"

print("\n📥 Checking face detector models...")

if not Path(prototxt_path).exists():
    print(f"   Downloading {prototxt_path}...")
    urllib.request.urlretrieve(prototxt_url, prototxt_path)
    print("   ✅ Downloaded")

if not Path(caffemodel_path).exists():
    print(f"   Downloading {caffemodel_path} (10MB)...")
    urllib.request.urlretrieve(caffemodel_url, caffemodel_path)
    print("   ✅ Downloaded")

# Load DNN face detector
print("\n📦 Loading DNN face detector...")
net = cv2.dnn.readNetFromCaffe(prototxt_path, caffemodel_path)
print("✅ DNN face detector loaded")

print("\n" + "=" * 60)

# =========================
# Helper Functions
# =========================
def preprocess_face(face_img, image_size=(128, 128)):
    face_resized = cv2.resize(face_img, image_size)
    face_rgb = cv2.cvtColor(face_resized, cv2.COLOR_BGR2RGB)
    face_normalized = face_rgb.astype(np.float32) / 255.0
    face_transposed = np.transpose(face_normalized, (2, 0, 1))
    return torch.from_numpy(face_transposed).unsqueeze(0).float()

class FPSCalculator:
    def __init__(self, buffer_size=30):
        self.frame_times = []
        self.last_time = time.time()
        self.buffer_size = buffer_size
    
    def update(self):
        current_time = time.time()
        self.frame_times.append(current_time - self.last_time)
        self.last_time = current_time
        if len(self.frame_times) > self.buffer_size:
            self.frame_times.pop(0)
    
    def get_fps(self):
        if not self.frame_times:
            return 0.0
        return 1.0 / (sum(self.frame_times) / len(self.frame_times))

# =========================
# Detection
# =========================
def run_realtime_detection():
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("❌ Error: Could not open webcam")
        return
    
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"\n📹 Webcam: {frame_width}x{frame_height}")
    
    print("\n=== Stabilization Settings ===")
    print("• History size: 10 frames")
    print("• Confidence threshold: 65%")
    print("• Hysteresis margin: 15%")
    print("• Smoothing alpha: 0.7")
    
    print("\n=== Controls ===")
    print("Press 'q' to quit")
    print("Press 's' to save screenshot")
    print("Press 'r' to reset stabilizers")
    print("Press 'd' to toggle debug info")
    print("\n=== Starting ===\n")
    
    class_names = ['No Mask', 'With Mask']
    colors = [(0, 0, 255), (0, 255, 0)]
    
    fps_calc = FPSCalculator()
    frame_count = 0
    
    # Initialize face tracker
    face_tracker = FaceTracker(max_faces=5)
    
    # Debug mode
    show_debug = False
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        fps_calc.update()
        
        h, w = frame.shape[:2]
        
        # Prepare blob for DNN
        blob = cv2.dnn.blobFromImage(
            cv2.resize(frame, (300, 300)), 1.0,
            (300, 300), (104.0, 177.0, 123.0)
        )
        
        # Detect faces
        net.setInput(blob)
        detections = net.forward()
        
        num_faces = 0
        current_bboxes = []
        
        # Process detections
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            
            # Filter by confidence
            if confidence > 0.5:
                num_faces += 1
                
                # Get bounding box
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (x, y, x2, y2) = box.astype("int")
                
                # Ensure within frame
                x, y = max(0, x), max(0, y)
                x2, y2 = min(w, x2), min(h, y2)
                
                if x2 - x < 20 or y2 - y < 20:
                    continue
                
                bbox = (x, y, x2 - x, y2 - y)
                current_bboxes.append(bbox)
                
                # Extract face
                face_roi = frame[y:y2, x:x2]
                
                # Classify
                face_tensor = preprocess_face(face_roi).to(device)
                
                with torch.no_grad():
                    output = model(face_tensor)
                    probs = torch.softmax(output, dim=1)
                    probs_np = probs[0].cpu().numpy()
                
                # Update face tracker with stabilization
                face_id, pred_class, conf_score, is_stable = face_tracker.update_face(
                    bbox, probs_np
                )
                
                # Get stabilizer stats
                stabilizer = face_tracker.trackers[face_id]
                stats = stabilizer.get_stats()
                
                # Debug output
                if frame_count % 30 == 0:
                    print(f"Face {face_id}: {class_names[pred_class]} ({conf_score*100:.1f}%) | "
                          f"Raw: [No: {probs_np[0]:.3f}, With: {probs_np[1]:.3f}] | "
                          f"Stable: {is_stable} | Changes: {stats['class_changes']}")
                
                # Choose color based on stability
                color = colors[pred_class]
                if not is_stable:
                    # Dim color if unstable
                    color = tuple(int(c * 0.6) for c in color)
                
                # Draw bounding box (thicker if stable)
                thickness = 3 if is_stable else 2
                cv2.rectangle(frame, (x, y), (x2, y2), color, thickness)
                
                # Draw label
                label = f"{class_names[pred_class]}: {conf_score*100:.1f}%"
                if not is_stable:
                    label += " (?)"
                
                (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
                cv2.rectangle(frame, (x, y - th - 15), (x + tw + 10, y), color, -1)
                cv2.putText(frame, label, (x + 5, y - 8),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
                # Show debug info if enabled
                if show_debug:
                    # Detection confidence
                    cv2.putText(frame, f"Det: {confidence*100:.0f}%", (x, y2 + 20),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
                    
                    # Stability info
                    stability_text = f"Stability: {stats['stability_ratio']*100:.0f}%"
                    cv2.putText(frame, stability_text, (x, y2 + 35),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
                    
                    # Raw probabilities
                    raw_text = f"Raw: {probs_np[0]:.2f}/{probs_np[1]:.2f}"
                    cv2.putText(frame, raw_text, (x, y2 + 50),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        
        # Cleanup old faces
        face_tracker.cleanup_old_faces(current_bboxes)
        
        # Info
        fps = fps_calc.get_fps()
        info_text = f"FPS: {fps:.1f} | Faces: {num_faces} | Frame: {frame_count}"
        cv2.putText(frame, info_text,
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Controls
        controls = "q:quit | s:screenshot | r:reset | d:debug"
        cv2.putText(frame, controls,
                   (10, h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Method label
        method_text = "DNN Face Detection + Stabilized CNN"
        if show_debug:
            method_text += " [DEBUG MODE]"
        cv2.putText(frame, method_text,
                   (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        # Show stabilization status
        if len(face_tracker.trackers) > 0:
            avg_stability = np.mean([
                t.get_stats()['stability_ratio'] 
                for t in face_tracker.trackers.values()
            ])
            stability_text = f"Avg Stability: {avg_stability*100:.0f}%"
            cv2.putText(frame, stability_text,
                       (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        cv2.imshow('Real-time Mask Detection (Stabilized)', frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            print("\n✅ Quitting...")
            break
        elif key == ord('s'):
            path = f"screenshot_{frame_count}.jpg"
            cv2.imwrite(path, frame)
            print(f"📸 Saved: {path}")
        elif key == ord('r'):
            # Reset all stabilizers
            for stabilizer in face_tracker.trackers.values():
                stabilizer.reset()
            print("🔄 Stabilizers reset")
        elif key == ord('d'):
            # Toggle debug mode
            show_debug = not show_debug
            print(f"🐛 Debug mode: {'ON' if show_debug else 'OFF'}")
    
    cap.release()
    cv2.destroyAllWindows()
    
    # Final statistics
    print("\n" + "=" * 60)
    print("Session Statistics")
    print("=" * 60)
    print(f"Total frames: {frame_count}")
    print(f"Average FPS: {fps_calc.get_fps():.1f}")
    
    if face_tracker.trackers:
        print(f"\nTracked faces: {len(face_tracker.trackers)}")
        for face_id, stabilizer in face_tracker.trackers.items():
            stats = stabilizer.get_stats()
            print(f"\nFace {face_id}:")
            print(f"  Frames: {stats['frames']}")
            print(f"  Class changes: {stats['class_changes']}")
            print(f"  Stability: {stats['stability_ratio']*100:.1f}%")
            print(f"  Avg confidence: {stats['avg_confidence']*100:.1f}%")
    
    print("=" * 60)

if __name__ == "__main__":
    try:
        run_realtime_detection()
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
