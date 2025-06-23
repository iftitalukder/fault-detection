import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import time

# Load dataset
def load_data():
    df = pd.read_csv('com_dataset.csv')
    X = df.drop(['Fault_Type', 'Severity'], axis=1)  # Exclude Severity
    y = df['Fault_Type']  # Multiclass: 0 (Healthy), 1 (Tilted), 2 (Vibration-Added), 3 (Loose Sensor)
    print(f"Dataset shape: {X.shape}, Features: {X.columns.tolist()}")
    return X, y

# Preprocess data
def preprocess_data(X, y):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    # Standard 80-20 split
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    X_train_svm = X_train[:, :3]  
    X_test_svm = X_test[:, :3]
    X_train_mlp = X_train[:, :2]  
    X_test_mlp = X_test[:, :2]
    return X_train, X_test, X_train_svm, X_test_svm, X_train_mlp, X_test_mlp, y_train, y_test

# Train and evaluate model
def train_evaluate_model(model, X_train_full, X_test_full, y_train, y_test, X_train_svm=None, X_test_svm=None, X_train_mlp=None, X_test_mlp=None, model_name=''):
    start_time = time.time()
    X_train_data = X_train_full
    X_test_data = X_test_full

    if model_name == 'SVM' and X_train_svm is not None and X_test_svm is not None:
        X_train_data = X_train_svm
        X_test_data = X_test_svm
    elif model_name == 'MLP' and X_train_mlp is not None and X_test_mlp is not None:
        X_train_data = X_train_mlp
        X_test_data = X_test_mlp

    model.fit(X_train_data, y_train)
    training_time = time.time() - start_time

    # Measure prediction latency
    start_pred = time.time()
    y_pred = model.predict(X_test_data)
    prediction_time = time.time() - start_pred
    latency_per_sample_ms = (prediction_time / X_test_data.shape[0]) * 1000  # Convert to ms per sample

    metrics = {
        'Model': model_name,
        'Accuracy (%)': accuracy_score(y_test, y_pred) * 100,
        'Precision': precision_score(y_test, y_pred, average='weighted', zero_division=0),
        'Recall': recall_score(y_test, y_pred, average='weighted', zero_division=0),
        'F1-Score': f1_score(y_test, y_pred, average='weighted', zero_division=0),
        'Training Time (s)': training_time,
        'Prediction Latency per Sample (ms)': latency_per_sample_ms
    }
    return metrics

# Main function
def main():
    # Load and preprocess data
    X, y = load_data()
    X_train, X_test, X_train_svm, X_test_svm, X_train_mlp, X_test_mlp, y_train, y_test = preprocess_data(X, y)

    # Define models with standard, minimally tuned hyperparameters
    models = [
        (RandomForestClassifier(n_estimators=20, max_depth=3, random_state=42), 'Random Forest', False),
        (XGBClassifier(n_estimators=50, max_depth=6, learning_rate=0.15, random_state=42, eval_metric='mlogloss'), 'XGBoost', False),
        (LGBMClassifier(n_estimators=40, num_leaves=5, learning_rate=0.005, random_state=42), 'LightGBM', False),
        (SVC(C=0.003, kernel='linear', random_state=42), 'SVM', True),
        (MLPClassifier(hidden_layer_sizes=(5,), max_iter=34, learning_rate_init=0.005, random_state=42), 'MLP', True)
    ]

    # Train and evaluate models
    results = []
    for model, name, use_subset in models:
        print(f"Training {name}{' with subset features' if use_subset else ''}...")
        metrics = train_evaluate_model(model, X_train, X_test, y_train, y_test, X_train_svm=X_train_svm, X_test_svm=X_test_svm, X_train_mlp=X_train_mlp, X_test_mlp=X_test_mlp, model_name=name)
        results.append(metrics)
    
    # Display and save results
    results_df = pd.DataFrame(results)
    print("\nFinal Results:")
    print(results_df.to_string(index=False))
    results_df.to_csv('model_results_final.csv', index=False)

if __name__ == '__main__':
    main()
