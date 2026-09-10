## Table 1. Comparison of Transfer Learning Models

| Model | Accuracy (%) | Precision (%) | Recall (%) | F1-Score (%) | AUC (%) |
|---|---|---|---|---|---|
| AlexNet | 65.62 | 66.98 | 65.62 | 62.23 | 89.42 |
| VGG16 | 73.44 | 78.45 | 73.44 | 71.48 | 90.85 |
| VGG19 | 70.31 | 56.60 | 70.31 | 61.71 | 85.40 |
| ResNet18 | 65.62 | 71.21 | 65.62 | 62.45 | 88.96 |
| ResNet50 | 75.00 | 80.43 | 75.00 | 73.19 | 88.93 |
| ResNet101 | 65.62 | 75.55 | 65.62 | 62.35 | 88.51 |
| DenseNet121 | 70.31 | 78.76 | 70.31 | 66.44 | 91.50 |
| EfficientNet-B0 | 70.31 | 78.89 | 70.31 | 67.67 | 92.74 |

**Best model: ResNet50** (highest accuracy, 75.00%) — used as the feature extractor for Table 2.

## Table 2. Comparison of Different Classifiers

| Feature Extractor | Classifier | Accuracy (%) | Precision (%) | Recall (%) | F1-Score (%) | AUC (%) |
|---|---|---|---|---|---|---|
| Deep Features | Logistic Regression | 73.44 | 78.08 | 73.44 | 71.98 | 85.97 |
| Deep Features | Decision Tree | 56.25 | 55.76 | 56.25 | 53.42 | 70.83 |
| Deep Features | Random Forest | 76.56 | 83.09 | 76.56 | 72.98 | 90.32 |
| Deep Features | K-Nearest Neighbors (KNN) | 67.19 | 70.37 | 67.19 | 65.57 | 85.79 |
| Deep Features | Linear SVM | 73.44 | 78.08 | 73.44 | 71.98 | 86.52 |
| Deep Features | RBF-SVM | 73.44 | 79.24 | 73.44 | 71.66 | 91.70 |
| Deep Features | XGBoost | 76.56 | 82.11 | 76.56 | 74.70 | 89.42 |

**Best classifier: Random Forest / XGBoost** (tied at 76.56% accuracy, both beating the ResNet50 end-to-end fine-tuned result of 75.00%).

## Table 3. Computational Efficiency Comparison

| Model | Parameters (M) | Model Size (MB) | FLOPs (G) | Inference Time (ms) | Accuracy (%) |
|---|---|---|---|---|---|
| AlexNet | 57.020 | 228.087 | 0.710 | 2.111 | 65.62 |
| VGG16 | 134.277 | 537.120 | 15.466 | 9.013 | 73.44 |
| VGG19 | 139.587 | 558.361 | 19.628 | 10.741 | 70.31 |
| ResNet18 | 11.179 | 44.794 | 1.824 | 2.502 | 65.62 |
| ResNet50 | 23.516 | 94.385 | 4.132 | 6.023 | 75.00 |
| DenseNet121 | 6.958 | 28.443 | 2.896 | 17.289 | 70.31 |
| EfficientNet-B0 | 4.013 | 16.353 | 0.414 | 10.324 | 70.31 |

**Best accuracy/efficiency trade-off: ResNet50** — highest accuracy of all 7 profiled models while staying mid-pack on parameters (23.5M), size (94 MB), and latency (6 ms) — far cheaper than VGG16/VGG19 despite beating their accuracy.

---
*Results from a single run: 4 classes (melanoma, nevus, basal cell carcinoma, pigmented benign keratosis), test set n=64, `FT_EPOCHS=15`.*
