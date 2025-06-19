# Compressor Fault Diagnosis using Machine Learning

This repository provides a reproducible machine learning pipeline for multiclass fault diagnosis of rotary compressors using triaxial vibration signals. Developed as part of a research project accepted at IEEE QPAIN 2025, the goal is to evaluate and compare the performance of several machine learning algorithms for detecting common faults in rotary compressors, such as tilt, external vibrations, and loose sensor conditions.

## 🔬 About the Research
Rotary compressors are widely used in industrial environments including HVAC, refrigeration, and process automation systems. Detecting faults early is crucial to prevent equipment failure and reduce downtime. This work investigates five machine learning classifiers on vibration-derived time-domain features and evaluates their effectiveness for real-time edge deployment.

The original dataset was collected using an ADXL345 accelerometer and processed into 15 statistical features (mean, RMS, std, skewness, kurtosis) per sample.

## 🚀 Features
- Supports Random Forest, XGBoost, LightGBM, SVM, and MLP
- Outputs classification metrics: Accuracy, Precision, Recall, F1-Score
- Saves confusion matrices and trained model files (.pkl)
- Accepts any CSV containing numerical features and a `Fault_Type` label column

## 📂 Directory Structure
```
compressor-fault-diagnosis/
├── main.py                  # Main script (runnable)
├── outputs/                 # Folder where results will be saved
├── requirements.txt         # Python dependencies
└── compressor_data.csv               # Example input file format
```

## 📋 Installation
Create a virtual environment and install dependencies:
```bash
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows
pip install -r requirements.txt
```

### Required Packages (included in requirements.txt)
```
pandas
numpy
matplotlib
seaborn
scikit-learn
xgboost
lightgbm
joblib
```

## 🧠 How to Use
1. Clone the repo:
```bash
git clone https://github.com/yourname/compressor-fault-diagnosis.git
cd compressor-fault-diagnosis
```
2. Place your dataset CSV file in the root folder.
3. Update the path in `main.py`:
```python
CSV_PATH = "your_file.csv"
```
4. Run the script:
```bash
python main.py
```

## 📈 Outputs
After running, the following will be saved in the `outputs/` folder:
- `model_performance.csv` — evaluation metrics
- `*.pkl` — trained model files
- `*_confusion.png` — confusion matrix heatmaps

## 📊 Input CSV Format
The input dataset must include 15 numerical features per sample and a `Fault_Type` column indicating the class (e.g., 0=Healthy, 1=Tilted, 2=Vibration-Added, 3=Loose Sensor). Example:
```
X_mean,Y_mean,Z_mean,X_std,Y_std,Z_std,X_skew,Y_skew,Z_skew,X_kurt,Y_kurt,Z_kurt,RMS_mean,RMS_std,RMS_kurt,Fault_Type
```

## 📜 License
MIT License

## 🤝 Citation
If you use this repository in your academic work or industrial projects, please cite:
```
K. Hossen et al., "Comparative Evaluation of Machine Learning Algorithms for Multiclass Fault Diagnosis of Rotary Compressors Using Triaxial Vibration Signals," in Proc. IEEE QPAIN 2025.
`
