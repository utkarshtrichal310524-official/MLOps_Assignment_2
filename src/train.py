"""
train.py
--------
MLOps Assignment 2: Track Machine Learning Experiments Using MLflow
Dataset: Wine Dataset (Kaggle / UCI - 178 samples, 13 features, 3 classes)

This script:
1. Loads and preprocesses the Wine dataset
2. Splits data into train (80%) and test (20%) sets
3. Trains three classifiers: Decision Tree, Random Forest, Logistic Regression
4. Logs parameters, metrics, confusion matrix, classification report,
   and the trained model to MLflow for every run
5. Compares all runs and identifies the best-performing model
6. Registers the best model in the MLflow Model Registry
"""

import os
import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "wine.csv")
MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
EXPERIMENT_NAME = "Wine_Classification_Experiment"
REGISTERED_MODEL_NAME = "wine-best-model"

# Use a local SQLite database as the MLflow tracking store.
# A database-backed store (instead of the plain file store) is required
# to use the MLflow Model Registry locally.
mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment(EXPERIMENT_NAME)

os.makedirs(MODELS_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# Step 1: Load dataset
# ---------------------------------------------------------------------------
def load_data():
    df = pd.read_csv(DATA_PATH)
    df.columns = [c.strip() for c in df.columns]  # clean stray whitespace
    print("Dataset shape:", df.shape)
    print("Class distribution:\n", df["class"].value_counts())
    return df


# ---------------------------------------------------------------------------
# Step 2: Preprocessing
# ---------------------------------------------------------------------------
def preprocess(df):
    # Handle missing values (none present in this dataset, but check anyway)
    missing = df.isnull().sum().sum()
    if missing > 0:
        df = df.fillna(df.mean(numeric_only=True))
    print("Missing values after cleaning:", df.isnull().sum().sum())

    X = df.drop(columns=["class"])
    y = df["class"]

    # Labels are already numeric (1, 2, 3), so no label encoding is required.
    # Feature scaling: needed for Logistic Regression, harmless for tree models.
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled = pd.DataFrame(X_scaled, columns=X.columns)

    return X, X_scaled, y


# ---------------------------------------------------------------------------
# Helper: log confusion matrix + classification report as artifacts
# ---------------------------------------------------------------------------
def log_confusion_matrix(y_test, y_pred, run_name):
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(f"Confusion Matrix - {run_name}")
    fig_path = f"confusion_matrix_{run_name}.png"
    fig.savefig(fig_path, bbox_inches="tight")
    plt.close(fig)
    mlflow.log_artifact(fig_path)
    os.remove(fig_path)


def log_classification_report(y_test, y_pred, run_name):
    report = classification_report(y_test, y_pred)
    report_path = f"classification_report_{run_name}.txt"
    with open(report_path, "w") as f:
        f.write(report)
    mlflow.log_artifact(report_path)
    os.remove(report_path)
    print(f"\nClassification Report - {run_name}\n{report}")


# ---------------------------------------------------------------------------
# Step 4-5: Train models + track experiments
# ---------------------------------------------------------------------------
def train_and_log(model, model_name, params, X_train, X_test, y_train, y_test):
    with mlflow.start_run(run_name=model_name) as run:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, average="macro")
        rec = recall_score(y_test, y_pred, average="macro")
        f1 = f1_score(y_test, y_pred, average="macro")

        # Log parameters
        mlflow.log_params(params)

        # Log metrics
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("precision", prec)
        mlflow.log_metric("recall", rec)
        mlflow.log_metric("f1_score", f1)

        # Log confusion matrix + classification report as artifacts
        log_confusion_matrix(y_test, y_pred, model_name)
        log_classification_report(y_test, y_pred, model_name)

        # Log the trained model itself
        mlflow.sklearn.log_model(
            model,
            name="model",
            skops_trusted_types=["sklearn.tree._tree.Tree"]
        )
        print(f"[{model_name}] accuracy={acc:.4f} precision={prec:.4f} "
              f"recall={rec:.4f} f1={f1:.4f} (run_id={run.info.run_id})")

        return {
            "run_id": run.info.run_id,
            "model_name": model_name,
            "accuracy": acc,
            "precision": prec,
            "recall": rec,
            "f1_score": f1,
        }


# ---------------------------------------------------------------------------
# Step 6-8: Compare experiments, select + register best model
# ---------------------------------------------------------------------------
def select_and_register_best(results):
    results_df = pd.DataFrame(results).sort_values("accuracy", ascending=False)
    print("\n===== Experiment Comparison =====")
    print(results_df.to_string(index=False))

    comparison_path = "model_comparison.csv"
    results_df.to_csv(comparison_path, index=False)

    best = results_df.iloc[0]
    print(f"\nBest model: {best['model_name']} "
          f"(accuracy={best['accuracy']:.4f}, f1={best['f1_score']:.4f})")

    # Register the best model's run under a single registered model name
    model_uri = f"runs:/{best['run_id']}/model"
    registered = mlflow.register_model(model_uri=model_uri, name=REGISTERED_MODEL_NAME)
    print(f"Registered '{REGISTERED_MODEL_NAME}' version {registered.version} "
          f"from run {best['run_id']}")

    # Log the comparison table as an artifact under a small summary run
    with mlflow.start_run(run_name="model_comparison_summary"):
        mlflow.log_artifact(comparison_path)
        mlflow.log_param("best_model", best["model_name"])
        mlflow.log_metric("best_accuracy", best["accuracy"])
    os.remove(comparison_path)

    return results_df


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    df = load_data()
    X, X_scaled, y = preprocess(df)

    # Split: 80% train / 20% test (stratified to preserve class balance)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    X_train_s, X_test_s, y_train_s, y_test_s = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )

    results = []

    # Model 1: Decision Tree
    dt_params = {"max_depth": 5, "criterion": "gini", "random_state": 42}
    dt_model = DecisionTreeClassifier(**dt_params)
    results.append(train_and_log(
        dt_model, "Decision_Tree", dt_params, X_train, X_test, y_train, y_test
    ))

    # Model 2: Random Forest
    rf_params = {"n_estimators": 100, "max_depth": 6, "random_state": 42}
    rf_model = RandomForestClassifier(**rf_params)
    results.append(train_and_log(
        rf_model, "Random_Forest", rf_params, X_train, X_test, y_train, y_test
    ))

    # Model 3: Logistic Regression (uses scaled features)
    lr_params = {"max_iter": 1000, "C": 1.0, "solver": "lbfgs", "random_state": 42}
    lr_model = LogisticRegression(**lr_params)
    results.append(train_and_log(
        lr_model, "Logistic_Regression", lr_params,
        X_train_s, X_test_s, y_train_s, y_test_s
    ))

    # Compare all experiments and register the best one
    select_and_register_best(results)

    print("\nAll done. Run 'mlflow ui --backend-store-uri sqlite:///mlflow.db' "
          "to view results in the browser.")


if __name__ == "__main__":
    main()
