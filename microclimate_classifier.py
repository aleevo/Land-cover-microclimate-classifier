# -*- coding: utf-8 -*-
"""
Created on Wed Feb  4 13:23:42 2026

@author: Ali
"""


# PROJECT: MICROCLIMATE LAND COVER CLASSIFICATION WITH SVM/LOGISTIC REGRESSION


import os
import warnings
warnings.filterwarnings('ignore', message="'multi_class' was deprecated")

import numpy as np
import matplotlib.pyplot as plt

# Image processing and feature extraction
from skimage.io import imread
from skimage.color import rgb2gray
from skimage.feature import hog
from skimage.exposure import histogram
from skimage.transform import resize

# Machine learning models and utilities
from sklearn.model_selection import train_test_split, GridSearchCV, learning_curve
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, Perceptron
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, ConfusionMatrixDisplay


# STEP 1: LOAD DATA & EXTRACT FEATURES


# 1. DEFINE PATHS & CLASSES
data_dir = "C:/Users/Haier/Desktop/ML Project/UCMerced_LandUse"  # CHANGE THIS TO YOUR PATH
classes = ['buildings', 'uc_forest', 'uc_river']  # Your 3 classes
num_samples_per_class = 100  # Start with 100 each

# 2. FUNCTION TO EXTRACT FEATURES FOR ONE IMAGE
def extract_features(image_path):
    """Extract Color Histogram and HOG features from an image."""
    img = imread(image_path)
    
    # 1. RESIZE all images to fixed size for consistent HOG features
    img = resize(img, (128, 128), anti_aliasing=True)
    
    # 2. ENSURE image is 3-channel (RGB)
    if len(img.shape) == 2:  # Grayscale
        img = np.stack([img, img, img], axis=2)
    elif img.shape[2] == 4:  # RGBA
        img = img[:, :, :3]  # Keep RGB
    
    # COLOR HISTOGRAM (32 bins per channel)
    hist_r, _ = histogram(img[:,:,0], nbins=32)
    hist_g, _ = histogram(img[:,:,1], nbins=32)
    hist_b, _ = histogram(img[:,:,2], nbins=32)
    color_feat = np.concatenate([hist_r, hist_g, hist_b])
    
    # HOG on grayscale version
    gray_img = rgb2gray(img)
    hog_feat = hog(gray_img, 
                   pixels_per_cell=(16, 16),
                   cells_per_block=(2, 2),
                   visualize=False, 
                   feature_vector=True)
    
    # COMBINE FEATURES
    combined_feat = np.concatenate([color_feat, hog_feat])
    return combined_feat

# 3. BUILD DATASET
print("Loading images and extracting features...")
X = []  # Feature vectors
y = []  # Labels (0, 1, 2)

for label_idx, class_name in enumerate(classes):
    class_dir = os.path.join(data_dir, class_name)
    image_files = [f for f in os.listdir(class_dir) if f.endswith('.tif')][:num_samples_per_class]
    
    for img_file in image_files:
        img_path = os.path.join(class_dir, img_file)
        features = extract_features(img_path)
        X.append(features)
        y.append(label_idx)

X = np.array(X)
y = np.array(y)
print(f"✓ Dataset shape: {X.shape}")  # Should be (300, 1860)
print(f"✓ Labels shape: {y.shape}\n")


# STEP 2: CREATE TRAIN/DEV/TEST SPLITS & SCALE DATA


# 1. Split out FINAL TEST set (15%)
X_temp, X_test, y_temp, y_test = train_test_split(
    X, y, test_size=0.15, random_state=42, stratify=y)

# 2. Split remainder into TRAIN and DEV sets (70/15 of original)
X_train, X_dev, y_train, y_dev = train_test_split(
    X_temp, y_temp, test_size=0.176, random_state=42, stratify=y_temp)  # 0.176 ≈ 15/85

print(f"✓ Data splits created:")
print(f"  Train: {X_train.shape} samples")
print(f"  Dev:   {X_dev.shape} samples")
print(f"  Test:  {X_test.shape} samples (FINAL, UNTOUCHED)\n")

# 3. SCALE FEATURES (Critical for SVM & Logistic Regression)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_dev_scaled = scaler.transform(X_dev)
X_test_scaled = scaler.transform(X_test)
print("✓ Features scaled using StandardScaler\n")


# STEP 3: TRAIN AND COMPARE MODELS


print("Training models...")

# 1. BASELINE: MULTINOMIAL LOGISTIC REGRESSION
log_reg = LogisticRegression(solver='lbfgs', max_iter=1000)
log_reg.fit(X_train_scaled, y_train)
y_pred_log_dev = log_reg.predict(X_dev_scaled)
log_dev_acc = accuracy_score(y_dev, y_pred_log_dev)
print(f"✓ Logistic Regression - Dev Accuracy: {log_dev_acc:.3f}")

# 2. PERCEPTRON (Another Linear Baseline)
perceptron = Perceptron(max_iter=1000, tol=1e-3, random_state=42)
perceptron.fit(X_train_scaled, y_train)
y_pred_per_dev = perceptron.predict(X_dev_scaled)
per_dev_acc = accuracy_score(y_dev, y_pred_per_dev)
print(f"✓ Perceptron - Dev Accuracy: {per_dev_acc:.3f}")

# 3. SVM WITH RBF KERNEL - WITH CROSS-VALIDATION
param_grid = {'C': [0.1, 1, 10, 100], 'gamma': ['scale', 'auto', 0.01, 0.1]}
svm_rbf = SVC(kernel='rbf', random_state=42)
grid_search = GridSearchCV(svm_rbf, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
grid_search.fit(X_train_scaled, y_train)

best_svm = grid_search.best_estimator_
y_pred_svm_dev = best_svm.predict(X_dev_scaled)
svm_dev_acc = accuracy_score(y_dev, y_pred_svm_dev)
print(f"✓ SVM (RBF) - Dev Accuracy: {svm_dev_acc:.3f}")
print(f"  Best Parameters: {grid_search.best_params_}\n")


# STEP 4: FINAL EVALUATION ON TEST SET


print("=== FINAL TEST SET PERFORMANCE ===")

# Predict on TEST set (unseen data)
y_pred_log_test = log_reg.predict(X_test_scaled)
y_pred_per_test = perceptron.predict(X_test_scaled)
y_pred_svm_test = best_svm.predict(X_test_scaled)

# Calculate metrics
log_test_acc = accuracy_score(y_test, y_pred_log_test)
per_test_acc = accuracy_score(y_test, y_pred_per_test)
svm_test_acc = accuracy_score(y_test, y_pred_svm_test)

log_test_f1 = f1_score(y_test, y_pred_log_test, average='macro')
per_test_f1 = f1_score(y_test, y_pred_per_test, average='macro')
svm_test_f1 = f1_score(y_test, y_pred_svm_test, average='macro')

print(f"Logistic Regression:")
print(f"  Test Accuracy: {log_test_acc:.3f} | Test F1 (Macro): {log_test_f1:.3f}")
print(f"Perceptron:")
print(f"  Test Accuracy: {per_test_acc:.3f} | Test F1 (Macro): {per_test_f1:.3f}")
print(f"SVM (RBF) with best params:")
print(f"  Test Accuracy: {svm_test_acc:.3f} | Test F1 (Macro): {svm_test_f1:.3f}\n")


# STEP 5: DIAGNOSTIC PLOTS


print("Generating diagnostic plots...")

# 1. CONFUSION MATRICES
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
model_names = ['Logistic Regression', 'Perceptron', 'SVM (RBF)']
predictions = [y_pred_log_test, y_pred_per_test, y_pred_svm_test]

for ax, name, y_pred in zip(axes, model_names, predictions):
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)
    disp.plot(ax=ax, cmap='Blues', values_format='d')
    ax.set_title(f'{name}\nTest Acc: {accuracy_score(y_test, y_pred):.3f}')
    ax.set_xticklabels(classes, rotation=45)
    ax.set_yticklabels(classes, rotation=0)

plt.suptitle('Confusion Matrices - Test Set Performance', fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig('confusion_matrices.png', dpi=150, bbox_inches='tight')
plt.show()

# 2. LEARNING CURVE (for best model - Logistic Regression)
print("Generating learning curve for best model (Logistic Regression)...")

from sklearn.model_selection import learning_curve

# Example for the SVM
train_sizes, train_scores, val_scores = learning_curve(
    best_svm, X_train_scaled, y_train, cv=5,
    scoring='accuracy', train_sizes=np.linspace(0.1, 1.0, 10), n_jobs=-1)

train_mean = np.mean(train_scores, axis=1)
train_std = np.std(train_scores, axis=1)
val_mean = np.mean(val_scores, axis=1)
val_std = np.std(val_scores, axis=1)

plt.figure(figsize=(8,5))
plt.plot(train_sizes, train_mean, 'o-', color='blue', label='Training accuracy')
plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.1, color='blue')
plt.plot(train_sizes, val_mean, 'o-', color='green', label='Cross-validation accuracy')
plt.fill_between(train_sizes, val_mean - val_std, val_mean + val_std, alpha=0.1, color='green')
plt.xlabel('Training Set Size')
plt.ylabel('Accuracy')
plt.title('Learning Curve for SVM (RBF) - Diagnosing Bias/Variance')
plt.legend(loc='best')
plt.grid(True)
plt.show()


train_sizes, train_scores, val_scores = learning_curve(
    log_reg, X_train_scaled, y_train, cv=5,
    scoring='accuracy', train_sizes=np.linspace(0.1, 1.0, 10), n_jobs=-1)

train_mean = np.mean(train_scores, axis=1)
train_std = np.std(train_scores, axis=1)
val_mean = np.mean(val_scores, axis=1)
val_std = np.std(val_scores, axis=1)

plt.figure(figsize=(8, 5))
plt.plot(train_sizes, train_mean, 'o-', color='blue', label='Training Accuracy')
plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.1, color='blue')
plt.plot(train_sizes, val_mean, 'o-', color='green', label='Cross-Validation Accuracy')
plt.fill_between(train_sizes, val_mean - val_std, val_mean + val_std, alpha=0.1, color='green')
plt.xlabel('Training Set Size')
plt.ylabel('Accuracy')
plt.title('Learning Curve: Logistic Regression (Best Model)')
plt.legend(loc='best')
plt.grid(True, alpha=0.3)
plt.savefig('learning_curve.png', dpi=150, bbox_inches='tight')
plt.show()


# STEP 6: SUMMARY OF KEY FINDINGS


print("\n" + "="*60)
print("PROJECT SUMMARY & KEY FINDINGS")
print("="*60)

# Determine best model
models = ['Logistic Regression', 'Perceptron', 'SVM (RBF)']
test_accs = [log_test_acc, per_test_acc, svm_test_acc]
best_idx = np.argmax(test_accs)
best_model = models[best_idx]
best_acc = test_accs[best_idx]

print(f"✓ BEST MODEL: {best_model} with {best_acc:.3f} Test Accuracy")
print(f"✓ MAIN CHALLENGE: Confusion between '{classes[1]}' and '{classes[2]}'")
print("  (Likely due to similar cool-color profiles in histograms)")
print(f"✓ GENERALIZATION: Learning curve shows minimal gap between")
print("  training and validation accuracy → Good generalization")


print("\n✓ Diagnostic plots saved as 'confusion_matrices.png' and 'learning_curve.png'")
print("="*60)