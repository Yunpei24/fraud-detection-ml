## training/src/README.md

# 🧠 Fraud Detection Training Module
# 1. Training Overview

This directory contains the **source code** for the fraud detection model training component.  
It includes modules for data loading, preprocessing, feature engineering, model training, evaluation, and MLflow integration.


- Data ingestion and preprocessing

- Feature engineering and selection

- Model training and hyperparameter tuning

- Model evaluation, validation, and explainability

- Experiment tracking and model registry via MLflow

The module follows a modular, production-oriented design, enabling easy scaling, reproducibility, and CI/CD integration.


# 2. Dataset Description

The dataset used for this project is the well-known Kaggle Credit Card Fraud Detection Dataset (creditcard.csv), which contains transactions made by European cardholders in September 2013.

Key details:

Rows: 284,807

Features: 30 (28 PCA-transformed features, Time, and Amount)

Target: Class (1 = Fraud, 0 = Legitimate)

Fraud ratio: ~0.17% (highly imbalanced)

Data Pipeline Steps:

Loading: The dataset is loaded from local storage or Kaggle API via data/loader.py.

Preprocessing: Missing values, schema validation, and outlier detection handled in data/preprocessor.py.

Splitting: Stratified and time-aware splits are created using data/splitter.py.

Feature Engineering: Temporal and scaling transformations are performed in features/engineer.py.

Balancing: SMOTE is applied to minority (fraud) cases for supervised models.

# 3. Running Training Locally
# Step 1: Set Up Environment

cd training
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# Step 2: Prepare Data

- Ensure the dataset is located in:

data/raw/creditcard.csv

If not, download it from Kaggle:
- kaggle datasets download -d mlg-ulb/creditcardfraud


# Step 3: Run the Training Pipeline
python -m training.src.pipelines.training_pipeline --config configs/model_xgb.yaml

# Step 4: Run Comparison (A/B Testing)
from training.src.pipelines.comparison_pipeline import run_comparison

cfg = {
    "xgboost": {"params": {"n_estimators": 300}, "threshold": 0.35, "smote": True},
    "random_forest": {"params": {"n_estimators": 500}, "threshold": 0.5, "smote": True},
    "mlp": {"params": {"hidden_layer_sizes": (128, 64)}, "threshold": 0.5, "smote": True},
    "isolation_forest": {"params": {"n_estimators": 300}},
}

report = run_comparison(cfg)
print(report.head())


Artifacts (trained models, reports, plots) will be stored under:

training/artifacts/

# 4. Hyperparameter Tuning

Hyperparameter optimization is managed through Optuna within
training/src/hyperparameter/tuning.py.

You can start tuning with:

python -m training.src.hyperparameter.tuning


The search spaces for each model are defined in:

training/src/hyperparameter/spaces.py


Each trial’s best parameters and metrics are automatically logged to MLflow.

# 5. MLflow Tracking

All training runs are logged to MLflow for:

- Parameter tracking

- Metric evaluation (AUC, F1, Recall, FPR)

- Model versioning

Artifact storage

Launch MLflow UI
mlflow ui --backend-store-uri mlruns/


Then open your browser at:

http://127.0.0.1:5000


# Example tracked metrics:

* Metric	Description
- AUC	Area Under ROC Curve
- PR_AUC	Precision-Recall AUC
- Recall	Sensitivity to fraud cases
- FPR	False Positive Rate
- F1	Balance between precision and recall

* Business Constraints

- Minimum Recall: 95% (detect 95% of fraud cases)

- Maximum FPR: 2% (≤ 2% false positives)

- Imbalance Handling: SMOTE oversampling for minority class

---

### 📚 Related Documentation
- [Main Training README](../README.md)
- [Pipeline Code](./pipelines/)
- [Hyperparameter Tuning](./hyperparameter/)
- [Evaluation & Explainability](./evaluation/)

* Then push with:
- git add training/src/README.md
- git commit -m "Add detailed README for training/src module"
- git push origin feature/training
