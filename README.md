# Suicide Detection using NLP and Deep Learning

This project aims to detect suicide-related content in text data using Natural Language Processing (NLP) and Deep Learning techniques. The model is trained on a dataset of labeled text samples and uses pre-trained GloVe embeddings for text representation. It achieves an accuracy of **92.86%** and an ROC-AUC score of **98.02%** on the test set.

---

## Project Overview

The goal of this project is to classify text data into two categories:
- **Suicide-related posts**
- **Non-suicide-related posts**

The model is built using:
- **Pre-trained GloVe embeddings** for word representation.
- **LSTM (Long Short-Term Memory)** layers for sequence modeling.
- **Dropout layers** for regularization.
- **Sigmoid activation** for binary classification.

---

## Dataset

The dataset used for this project is the [Suicide Detection Dataset](https://www.kaggle.com/datasets/nikhileswarkomati/suicide-watch/data) sourced from Kaggle.

### Preprocessing
- Text cleaning (lowercasing, removing special characters, stopwords, etc.).
- Tokenization and padding to a fixed sequence length (50).
- Label encoding for binary classification.
- The dataset is split into training (0.8), validation (0.1), and testing (0.1) sets.

---

## Model Architecture

The model is a **Sequential Neural Network** with the following layers:
1. **Input Layer**: Accepts sequences of length 50.
2. **Embedding Layer**: Uses pre-trained GloVe embeddings (300 dimensions).
3. **LSTM Layer**: 20 units.
4. **Global Max Pooling Layer**: Reduces sequence dimension.
5. **Dropout Layer**: Dropout rate of 0.3.
6. **Dense Layer**: 256 units with ReLU activation.
7. **Dropout Layer**: Dropout rate of 0.2.
8. **Output Layer**: 1 unit with sigmoid activation.

### Training
- **Optimizer**: SGD with momentum (learning rate = 0.1, momentum = 0.09).
- **Loss Function**: Binary cross-entropy.
- **Metrics**: Accuracy.
- **Callbacks**: Early stopping and learning rate reduction.

## Results

The model achieves the following performance on the test set:

### Key Metrics
- **Test Loss**: 0.1853
- **Test Accuracy**: 92.86%
- **Precision**: 94.44%
- **Recall**: 90.98%
- **F1-Score**: 92.68%
- **ROC-AUC Score**: 98.02%

### Confusion Matrix
The confusion matrix provides a detailed breakdown of the model's predictions:

|                     | Predicted: 0 (Non-Suicide) | Predicted: 1 (Suicide) |
|---------------------|---------------------------|------------------------|
| **Actual: 0 (Non-Suicide)** | 11,070                    | 617                    |
| **Actual: 1 (Suicide)**     | 1,039                     | 10,482                 |

- **True Positives (TP)**: 10,482 (correctly predicted suicide-related posts).
- **True Negatives (TN)**: 11,070 (correctly predicted non-suicide-related posts).
- **False Positives (FP)**: 617 (non-suicide posts incorrectly classified as suicide-related).
- **False Negatives (FN)**: 1,039 (suicide-related posts incorrectly classified as non-suicide).
 
### Interpretations
- **High Accuracy**: The model achieves **92.86% accuracy** on the test set, demonstrating its ability to reliably classify text data.
- **Precision and Recall**: With a precision of **94.44%** and recall of **90.98%**, the model effectively identifies suicide-related posts while minimizing false positives and false negatives.
- **Robust Generalization**: The high **ROC-AUC score (98.02%)** indicates excellent discrimination between suicide and non-suicide posts, even on unseen data.
