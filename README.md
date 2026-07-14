# Track Machine Learning Experiments Using MLflow

**Course:** MLOps — Assignment 2
**Dataset:** Wine Dataset (Kaggle, 178 samples, 13 features, 3 classes)

## Aim
Train multiple ML models on the Wine dataset and use MLflow to track experiments —
logging parameters, metrics, artifacts, and trained models for reproducibility and comparison.

## Folder Structure
```
MLflow_Assignment/
├── data/
│   └── wine.csv              # Wine dataset
├── notebooks/
│   └── experiment.ipynb      # EDA (exploratory data analysis)
├── src/
│   └── train.py              # Main training + MLflow tracking script
├── models/                   # (reserved; MLflow logs models under mlruns/)
├── screenshots/              # MLflow UI screenshots go here
├── mlruns/                   # MLflow experiment tracking data (auto-generated)
├── mlflow.db                 # SQLite backend store (auto-generated, enables Model Registry)
├── requirements.txt
└── README.md
```

## What the script does
1. **Load data** — reads `data/wine.csv`.
2. **Preprocess** — checks for missing values, standard-scales features for
   Logistic Regression (tree models use raw features).
3. **Split** — 80% train / 20% test, stratified by class.
4. **Train 3 models** — Decision Tree, Random Forest, Logistic Regression.
5. **Track with MLflow** — for every run, logs:
   - Parameters (model hyperparameters)
   - Metrics: Accuracy, Precision, Recall, F1 Score (macro-averaged)
   - Confusion matrix (as a PNG artifact)
   - Classification report (as a text artifact)
   - The trained model itself
6. **Compare** — builds a comparison table across all 3 runs.
7. **Select the best model** — highest accuracy.
8. **Register** — registers the best model in the MLflow Model Registry
   under the name `wine-best-model`.

## How to Run (VS Code)

1. Open the `MLflow_Assignment` folder in VS Code.
2. Open a terminal (`` Ctrl+` ``) and create a virtual environment:
   ```
   python -m venv venv
   venv\Scripts\activate        # Windows
   source venv/bin/activate     # macOS/Linux
   ```
3. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
4. Run the training script from the project root:
   ```
   python src/train.py
   ```
   This creates `mlflow.db` and `mlruns/` in the project root and prints
   a comparison table + the best model in the terminal.
5. Launch the MLflow UI:
   ```
   mlflow ui --backend-store-uri sqlite:///mlflow.db
   ```
6. Open **http://127.0.0.1:5000** in your browser to view and compare runs.

## Results (from the reference run included in this submission)

| Model               | Accuracy | Precision | Recall | F1 Score |
|---------------------|----------|-----------|--------|----------|
| Random Forest        | 1.00     | 1.00      | 1.00   | 1.00     |
| Logistic Regression  | 0.97     | 0.98      | 0.97   | 0.97     |
| Decision Tree         | 0.94     | 0.96      | 0.94   | 0.95     |

**Best model:** Random Forest — registered as `wine-best-model` (version 1) in the
MLflow Model Registry.

## Notes
- The Wine dataset has no missing values, so no imputation was required.
- Labels (`class` column) are already numeric (1/2/3), so no label encoding was needed.
- A SQLite tracking store (`sqlite:///mlflow.db`) is used instead of the default
  local file store because the **Model Registry** requires a database-backed store.
