# 🧑 Face Recognition using VGG-Face + SVM

## Overview
A face recognition pipeline that identifies individuals from images using deep learning embeddings and classical machine learning classification. The system uses a pretrained VGG-Face model to generate rich face embeddings, reduces their dimensionality using PCA, and classifies them using a Support Vector Machine (SVM).

## Problem Statement
Given a dataset of labelled face images, can we build a system that correctly identifies which person appears in a new, unseen image?

## Approach

| Step | Technique |
|------|-----------|
| Face Embedding | Pretrained VGG-Face model (2048-dim vectors) |
| Similarity Metric | Cosine distance |
| Dimensionality Reduction | PCA (2048 → 100 dimensions) |
| Classification | SVM (One-vs-One multiclass strategy) |
| Train/Test Split | 80% / 20% |

## Key Concepts
- **VGG-Face**: A deep CNN pretrained on millions of face images, used here as a feature extractor
- **Cosine Distance**: Measures similarity between face embedding vectors - lower value = more similar faces
- **PCA**: Reduces the 2048-dimension embedding to 100 dimensions, retaining key variance while improving classifier efficiency
- **SVM (One-vs-One)**: Multiclass classification strategy that trains a binary classifier for each pair of classes

## Dataset
- 20 face classes (individuals)
- Images resized to 224x224 pixels for VGG-Face compatibility

## Results
- Successfully distinguishes between same-class (similar) and different-class (dissimilar) face pairs using cosine distance
- SVM classifier trained on PCA-reduced embeddings achieves strong multiclass identification performance

## Tech Stack
```
Python | TensorFlow/Keras | OpenCV | scikit-learn | NumPy | Matplotlib
```

## How to Run
1. Clone the repo
2. Install dependencies: `pip install tensorflow keras scikit-learn opencv-python matplotlib numpy`
3. Open the notebook in Jupyter or Google Colab
4. Run all cells sequentially

---
*Completed as part of the Post-Graduate Program in AI & Machine Learning - University of Texas at Austin (2021)*
