import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import cv2
import numpy as np
from pathlib import Path
import time

# =========================
# Device
# =========================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# =========================
# Dataset Class
# =========================
class FaceMaskDataset(Dataset):
    def __init__(self, image_paths, image_size=(128, 128)):
        self.image_paths = image_paths
        self.image_size = image_size
        
        # with_mask = 1
        # without_mask = 0
        self.labels = [1 if 'with_mask' in str(p) else 0 
                       for p in image_paths]
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        # Read image
        img = cv2.imread(self.image_paths[idx])
        
        # BGR -> RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Resize
        img = cv2.resize(img, self.image_size)
        
        # Normalize
        img = img.astype(np.float32) / 255.0
        
        # HWC -> CHW
        img = np.transpose(img, (2, 0, 1))
        
        # Convert to tensor
        img_tensor = torch.from_numpy(img).float()
        label_tensor = torch.tensor(self.labels[idx]).long()
        
        return img_tensor, label_tensor

# =========================
# Load Dataset
# =========================
base_path = Path("datasets/train")

# Find all jpg images
all_paths = [str(p) for p in base_path.rglob("*.jpg")]

print("Total Images:", len(all_paths))

# Create dataset
dataset = FaceMaskDataset(all_paths)

# =========================
# Train / Test Split
# =========================
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size

train_dataset, test_dataset = random_split(dataset, 
                                           [train_size, test_size])

# =========================
# DataLoader
# =========================
train_loader = DataLoader(train_dataset, 
                          batch_size=32, 
                          shuffle=True)

test_loader = DataLoader(test_dataset, 
                         batch_size=32, 
                         shuffle=False)

# =========================
# CNN Model
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
# Create Model
# =========================
model = RealTimeMaskNet().to(device)

# =========================
# Loss Function & Optimizer
# =========================
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), 
                       lr=0.001)

# =========================
# Training
# =========================
EPOCHS = 30

for epoch in range(EPOCHS):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for images, labels in train_loader:
        # Move to device
        images = images.to(device)
        labels = labels.to(device)
        
        # Clear gradients
        optimizer.zero_grad()
        
        # Forward
        outputs = model(images)
        
        # Loss
        loss = criterion(outputs, labels)
        
        # Backpropagation
        loss.backward()
        
        # Update weights
        optimizer.step()
        
        running_loss += loss.item()
        
        # Accuracy
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    
    epoch_loss = running_loss / len(train_loader)
    epoch_acc = 100 * correct / total
    
    print(f"Epoch [{epoch+1}/{EPOCHS}] "
          f"Loss: {epoch_loss:.4f} "
          f"Accuracy: {epoch_acc:.2f}%")

# =========================
# Testing
# =========================
model.eval()
correct = 0
total = 0

with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)
        
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

test_acc = 100 * correct / total
print(f"\nTest Accuracy: {test_acc:.2f}%")

# =========================
# Speed Benchmark
# =========================
dummy_input = torch.randn(1, 3, 128, 128).to(device)

start = time.time()
with torch.no_grad():
    for _ in range(100):
        _ = model(dummy_input)
latency = (time.time() - start) / 100

print(f"\nInference Latency: {latency*1000:.2f} ms")
print(f"Estimated FPS: {1/latency:.2f}")

# =========================
# Save Model
# =========================
torch.save(model.state_dict(), "mask_model.pth")
print("\nModel saved as mask_model.pth")
