# -*- coding: utf-8 -*-
"""
Created on Wed Feb  4 13:37:37 2026

@author: Ali
"""

# ============================================================================
# CNN EXPLORATION FOR IMAGE CLASSIFICATION
# Uses the SAME data as the main project for a fair comparison.
# ============================================================================

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import matplotlib.pyplot as plt

# Import the data loading/feature extraction from your main project
# We will re-use the same splits for a fair comparison
from microclimate_classifier import X_train_scaled, X_test_scaled, y_train, y_test

print("="*60)
print("CNN EXPLORATION: 1D Neural Network on Engineered Features")
print("="*60)

# ============================================================================
# 1. PREPARE DATA FOR PYTORCH
# ============================================================================

print("\n1. Preparing PyTorch datasets...")

# Convert numpy arrays to PyTorch tensors
X_train_tensor = torch.FloatTensor(X_train_scaled)
y_train_tensor = torch.LongTensor(y_train)
X_test_tensor = torch.FloatTensor(X_test_scaled)
y_test_tensor = torch.LongTensor(y_test)

# Create TensorDatasets and DataLoaders (handles batching)
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

print(f"   Training samples: {len(train_dataset)}")
print(f"   Test samples: {len(test_dataset)}")
print(f"   Feature dimension: {X_train_tensor.shape[1]}")

# ============================================================================
# 2. DEFINE A SIMPLE 1D NEURAL NETWORK
# ============================================================================

class SimpleNN(nn.Module):
    """A simple feedforward neural network for 1D feature vectors."""
    def __init__(self, input_size=1860, num_classes=3):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_size, 256)   # Input layer
        self.fc2 = nn.Linear(256, 128)          # Hidden layer 1
        self.fc3 = nn.Linear(128, 64)           # Hidden layer 2
        self.fc4 = nn.Linear(64, num_classes)   # Output layer
        self.dropout = nn.Dropout(p=0.3)        # Regularization
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = F.relu(self.fc3(x))
        x = self.fc4(x)  # No activation here (used with CrossEntropyLoss)
        return x

# Initialize model, loss function, and optimizer
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"\n2. Initializing model on device: {device}")

model = SimpleNN(input_size=X_train_tensor.shape[1], num_classes=3).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

print(f"   Model architecture: {model}")
print(f"   Total parameters: {sum(p.numel() for p in model.parameters()):,}")

# ============================================================================
# 3. TRAIN THE MODEL
# ============================================================================

print("\n3. Training the neural network...")
num_epochs = 30
train_losses = []
train_accuracies = []

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for batch_x, batch_y in train_loader:
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)
        
        # Zero the parameter gradients
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(batch_x)
        loss = criterion(outputs, batch_y)
        
        # Backward pass and optimize
        loss.backward()
        optimizer.step()
        
        # Statistics
        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += batch_y.size(0)
        correct += (predicted == batch_y).sum().item()
    
    epoch_loss = running_loss / len(train_loader)
    epoch_acc = 100 * correct / total
    train_losses.append(epoch_loss)
    train_accuracies.append(epoch_acc)
    
    if (epoch + 1) % 5 == 0:
        print(f"   Epoch [{epoch+1:3d}/{num_epochs}] | Loss: {epoch_loss:.4f} | Acc: {epoch_acc:.2f}%")

# ============================================================================
# 4. EVALUATE ON TEST SET
# ============================================================================

print("\n4. Evaluating on test set...")
model.eval()
test_correct = 0
test_total = 0
all_predictions = []
all_labels = []

with torch.no_grad():
    for batch_x, batch_y in test_loader:
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)
        outputs = model(batch_x)
        _, predicted = torch.max(outputs.data, 1)
        
        test_total += batch_y.size(0)
        test_correct += (predicted == batch_y).sum().item()
        
        all_predictions.extend(predicted.cpu().numpy())
        all_labels.extend(batch_y.cpu().numpy())

test_accuracy = 100 * test_correct / test_total
print(f"   Test Accuracy: {test_accuracy:.2f}%")
print(f"   Correct: {test_correct}/{test_total}")

# ============================================================================
# 5. VISUALIZE TRAINING PROCESS
# ============================================================================

print("\n5. Generating training plots...")
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

# Plot loss
ax1.plot(train_losses, 'b-', linewidth=2)
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Loss')
ax1.set_title('Training Loss Over Time')
ax1.grid(True, alpha=0.3)

# Plot accuracy
ax2.plot(train_accuracies, 'g-', linewidth=2)
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Accuracy (%)')
ax2.set_title('Training Accuracy Over Time')
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('cnn_training_history.png', dpi=150, bbox_inches='tight')
plt.show()

# ============================================================================
# 6. COMPARISON WITH TRADITIONAL MODELS
# ============================================================================

print("\n" + "="*60)
print("COMPARISON: Neural Network vs. Traditional ML")
print("="*60)

# You'll need to manually add your best traditional model accuracy here
# From your main project: Logistic Regression = 93.3%
traditional_accuracy = 93.3  # <-- UPDATE THIS with your actual best score

print(f"\nTraditional ML (Logistic Regression): {traditional_accuracy:.1f}%")
print(f"Neural Network (1D Feedforward):      {test_accuracy:.2f}%")

difference = test_accuracy - traditional_accuracy
if difference > 0:
    print(f"\n✓ Neural Network performed BETTER by {abs(difference):.2f} percentage points")
elif difference < 0:
    print(f"\n✓ Traditional ML performed BETTER by {abs(difference):.2f} percentage points")
else:
    print(f"\n✓ Both methods performed equally well")

print("\n" + "="*60)
print("KEY INSIGHT FOR YOUR REPORT:")
print("="*60)
print("""
This 1D Neural Network learned directly from the same engineered features
(HOG + Color Histograms) as your traditional models. The comparison shows:

1. If NN accuracy is similar (~93%): "A neural network can achieve comparable
   performance to optimized traditional models, validating our feature engineering."

2. If NN accuracy is lower: "The simpler linear model was sufficient for this
   nearly linearly separable problem, demonstrating that complexity doesn't
   always guarantee better performance."

3. If NN accuracy is higher: "The neural network's ability to learn non-linear
   combinations of features provided an edge over traditional methods."
""")

# ============================================================================
# BONUS: How a TRUE 2D CNN would differ (for your report discussion)
# ============================================================================

print("\n" + "="*60)
print("FOR YOUR REPORT: Advanced CNN Concept")
print("="*60)
print("""
A *true* Convolutional Neural Network (CNN) would work with RAW IMAGES (not our
1D features). Its architecture would look different:

class TrueCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3)  # 2D convolutions
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
        self.pool = nn.MaxPool2d(2, 2)
        # ... more layers
        
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))  # Processes spatial patterns
        x = self.pool(F.relu(self.conv2(x)))
        # ... rest

Key Advantage: A true CNN learns spatial hierarchies automatically - edges → 
textures → patterns → objects - directly from pixels, eliminating the need for
manual feature engineering (HOG/Histograms).

For this project extension, we used the 1D NN for a fair comparison using the
same features, but future work could implement a 2D CNN on raw images.
""")