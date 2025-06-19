# Compressor Fault Diagnosis (OF550W)

This repository contains the dataset and implementation code for the research paper:

*"Comparative Evaluation of Machine Learning Algorithms for Multiclass Fault Diagnosis of Rotary Compressors Using Triaxial Vibration Signals"* 
📍 IEEE QPAIN 2025

---

## 📦 Contents
- `com_dataset.csv` — Vibration dataset from an **OF550W rotary compressor** using an ADXL345 triaxial accelerometer.
- `minimal_xgboost_demo.py` — Optimized XGBoost demonstration script.
- `evaluate_all_models.py` — Full benchmarking script for:
  - Random Forest
  - SVM
  - MLP
  - LightGBM
  - XGBoost

---

## 🔧 How to Use

### 1. Clone This Repository
```bash
git clone https://github.com/iftitalukder/fault-detection.git
cd fault-detection
