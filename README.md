# fault-detection
# 🔍 ML Model Evaluator from CSV

This repository provides a **plug-and-play Python script** to automatically train, evaluate, and compare multiple machine learning classifiers using a labeled CSV dataset. It outputs performance metrics, confusion matrices, and trained models – all saved for reproducibility.

---

## ✅ What This Repo Does

Given any classification dataset in CSV format, this tool:

- Loads and splits the dataset
- Trains the following models:
  - Random Forest
  - XGBoost
  - LightGBM
  - Support Vector Machine (SVM)
  - Multi-Layer Perceptron (MLP)
- Evaluates using:
  - Accuracy
  - Precision
  - Recall
  - F1-Score
- Saves:
  - Performance scores to CSV
  - Confusion matrices as PNGs
  - Trained models (`.pkl` files)

---

## 💡 Why Use This?

This project is ideal for:

- Rapid benchmarking of classification models
- Academic or industry research pipelines
- Fault diagnosis, signal classification, or any supervised ML task with labeled CSV data
- Users who want **automation** without writing boilerplate code

---

## 🚀 Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/your-username/ml-model-evaluator.git
cd ml-model-evaluator
