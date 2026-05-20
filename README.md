# Land Cover Classification for Microclimate Analysis

## Overview
Automated land cover classifier using aerial imagery to support environmental planning and microclimate analysis. This project compares traditional machine learning models with an extension to neural network approach on the same engineered features.

## Dataset
- **Source**: [UC Merced Land Use Dataset on Kaggle](https://www.kaggle.com/datasets/zeadomar/uc-mercedland)
- 300 aerial images from California's Central Valley
- 3 classes: Buildings, Forest, River
- Features: HOG + Color Histograms (1,860 dimensions)

## Models Compared

### Traditional Machine Learning
| Model | Test Accuracy | Macro F1-Score |
|-------|--------------|----------------|
| **Logistic Regression** | **93.3%** | **0.933** |
| SVM (RBF) | 91.1% | 0.908 |
| Perceptron | 88.9% | 0.888 |

### Deep Learning Exploration
| Model | Test Accuracy | Notes |
|-------|--------------|-------|
| 1D Feedforward Neural Network (PyTorch) | ~93.33% | Trained on same HOG+Color features for fair comparison |

## Key Results
- **Best model: Logistic Regression** with 93.3% accuracy
- The linear model outperformed the neural network on engineered features, showing that feature quality matters more than model complexity for this dataset
- Learning curve confirms good generalization (converging validation accuracy)
- Confusion matrices show river class is hardest to classify

## Why the Neural Network?
Built a PyTorch neural network as an extension to compare deep learning vs. traditional ML on the **same features**. This explores whether added complexity improves performance when features are already well-engineered. The 1D NN uses the identical HOG + Color Histogram vectors, making this a controlled experiment.

**Future extension**: A true 2D CNN on raw pixels could eliminate manual feature engineering entirely.

## Files
| File | Description |
|------|-------------|
| `microclimate_classifier.py` | Main project: HOG + Color feature extraction, traditional ML models, evaluation |
| `cnn_experiment.py` | PyTorch neural network on same features for comparison |
| `brief_report` |  https://drive.google.com/drive/folders/1hy4qEejrmkpOMzKqAwQZPUZJKbLFBzO8 |
| `plots/` | Confusion matrices, learning curves, bar charts, CNN training history |
| `README.md` | This file |

## Tech Stack
- **Traditional ML**: Python, scikit-learn, OpenCV, NumPy, Matplotlib
- **Deep Learning**: PyTorch
- **Features**: HOG (Histogram of Oriented Gradients), Color Histograms

## How to Run

### Traditional Models
```bash
pip install -r requirements.txt
python microclimate_classifier.py
