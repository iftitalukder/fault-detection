import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
import argparse

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

TARGET_COLUMN = "Fault_Type"
SAVE_DIR = "outputs"
RANDOM_STATE = 42


def load_and_prepare_data(path):
    df = pd.read_csv(path)
    X = df.drop(columns=[TARGET_COLUMN, 'Severity']) if 'Severity' in df.columns else df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return train_test_split(X_scaled, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE)


def evaluate_model(name, model, X_test, y_test):
    y_pred = model.predict(X_test)
    return {
        'Model': name,
        'Accuracy (%)': round(accuracy_score(y_test, y_pred) * 100, 2),
        'Precision': round(precision_score(y_test, y_pred, average='weighted'), 2),
        'Recall': round(recall_score(y_test, y_pred, average='weighted'), 2),
        'F1-Score': round(f1_score(y_test, y_pred, average='weighted'), 2),
        'Confusion Matrix': confusion_matrix(y_test, y_pred)
    }


def plot_confusion_matrix(cm, labels, filename):
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv', required=True, help="Path to input CSV file")
    args = parser.parse_args()

    os.makedirs(SAVE_DIR, exist_ok=True)
    X_train, X_test, y_train, y_test = load_and_prepare_data(args.csv)

    models = {
        'Random Forest': RandomForestClassifier(
            n_estimators=500,
            max_depth=20,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=RANDOM_STATE
        ),
        'XGBoost': XGBClassifier(
            use_label_encoder=False,
            eval_metric='mlogloss',
            learning_rate=0.05,
            n_estimators=600,
            max_depth=7,
            subsample=0.85,
            colsample_bytree=0.85,
            gamma=0.1,
            reg_alpha=0.2,
            reg_lambda=1.2,
            verbosity=0,
            random_state=RANDOM_STATE
        ),
        'LightGBM': LGBMClassifier(
            learning_rate=0.05,
            n_estimators=600,
            max_depth=7,
            num_leaves=31,
            min_data_in_leaf=20,
            feature_fraction=0.9,
            bagging_fraction=0.8,
            bagging_freq=5,
            random_state=RANDOM_STATE
        ),
        'SVM': SVC(
            kernel='rbf',
            C=10,
            gamma='scale',
            probability=True,
            random_state=RANDOM_STATE
        ),
        'MLP': MLPClassifier(
            hidden_layer_sizes=(256, 128, 64),
            activation='relu',
            solver='adam',
            alpha=1e-4,
            batch_size='auto',
            learning_rate='adaptive',
            max_iter=1000,
            early_stopping=True,
            random_state=RANDOM_STATE
        )
    }

    results = []
    for name, model in models.items():
        print(f"\nTraining {name}...")
        model.fit(X_train, y_train)
        metrics = evaluate_model(name, model, X_test, y_test)
        results.append(metrics)

        plot_confusion_matrix(
            metrics['Confusion Matrix'],
            labels=np.unique(y_test),
            filename=os.path.join(SAVE_DIR, f"{name.replace(' ', '_')}_confusion.png")
        )

        joblib.dump(model, os.path.join(SAVE_DIR, f"{name.replace(' ', '_')}.pkl"))
        print(f"{name} complete. Accuracy: {metrics['Accuracy (%)']}%")

    df_results = pd.DataFrame([{
        k: v for k, v in m.items() if k != 'Confusion Matrix'
    } for m in results])

    df_results.to_csv(os.path.join(SAVE_DIR, "model_performance.csv"), index=False)
    print("\n✅ All tasks completed. Results saved in:", SAVE_DIR)


if __name__ == "__main__":
    main()
