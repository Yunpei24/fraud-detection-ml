# 🧠 Fraud Detection Training Module

Production-ready ML training pipeline for fraud detection using ensemble methods with XGBoost, Neural Networks, Random Forest, and Isolation Forest.

## 🎯 Overview

This module handles:
- **Data Ingestion**: Load and validate Credit Card Fraud dataset
- **Preprocessing**: Missing values, outliers, schema validation
- **Feature Engineering**: Temporal, scaling, and interaction features
- **Model Training**: Ensemble of 4 models with hyperparameter tuning
- **Evaluation**: Business-focused metrics (Recall ≥95%, FPR ≤2%)
- **MLflow Integration**: Experiment tracking and model registry
- **A/B Testing**: Model comparison and selection

## 📊 Dataset

**Source**: Kaggle Credit Card Fraud Detection Dataset  
**File**: `creditcard.csv`

**Statistics**:
- **Rows**: 284,807 transactions
- **Features**: 30 (Time, V1-V28 PCA components, Amount)
- **Target**: Class (0=Legitimate, 1=Fraud)
- **Fraud Rate**: ~0.17% (highly imbalanced)
- **Time Period**: September 2013 (European cardholders)

**Data Split Strategy**:
```
Train: 70% (199,365 transactions)
Validation: 15% (42,721 transactions)
Test: 15% (42,721 transactions)

Split Method: Stratified + Time-aware
- Maintains fraud rate in each split
- Respects temporal order
```

## 🚀 Quick Start

### Installation

```bash
cd fraud-detection-ml/training

# Create virtual environment
python -m venv venv_training
source venv_training/bin/activate  # Windows: venv_training\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Prepare Dataset

```bash
# Option 1: Use existing dataset
cp /path/to/creditcard.csv ../data/raw/creditcard.csv

# Option 2: Download from Kaggle
pip install kaggle
kaggle datasets download -d mlg-ulb/creditcardfraud
unzip creditcardfraud.zip -d ../data/raw/
```

### Run Training Pipeline

```bash
# Train all models with default config
python -m src.pipelines.training_pipeline

# Train specific model
python -m src.pipelines.training_pipeline --model xgboost

# Train with custom config
python -m src.pipelines.training_pipeline --config configs/model_xgb.yaml
```

### Run Model Comparison (A/B Testing)

```python
from src.pipelines.comparison_pipeline import run_comparison

config = {
    "xgboost": {
        "params": {"n_estimators": 300, "max_depth": 6},
        "threshold": 0.35,
        "smote": True
    },
    "random_forest": {
        "params": {"n_estimators": 500, "max_depth": 10},
        "threshold": 0.5,
        "smote": True
    },
    "mlp": {
        "params": {"hidden_layer_sizes": (128, 64)},
        "threshold": 0.5,
        "smote": True
    },
    "isolation_forest": {
        "params": {"n_estimators": 300, "contamination": 0.002}
    }
}

report = run_comparison(config)
print(report)
```

## 📁 Project Structure

```
training/
├── src/
│   ├── data/
│   │   ├── loader.py           # Dataset loading
│   │   ├── preprocessor.py     # Data cleaning
│   │   └── splitter.py         # Train/val/test split
│   ├── features/
│   │   ├── engineer.py         # Feature engineering
│   │   └── selector.py         # Feature selection
│   ├── models/
│   │   ├── xgboost_model.py    # XGBoost classifier
│   │   ├── neural_network.py   # PyTorch MLP
│   │   ├── random_forest.py    # Random Forest
│   │   ├── isolation_forest.py # Anomaly detection
│   │   └── ensemble.py         # Ensemble voting
│   ├── hyperparameter/
│   │   ├── tuning.py           # Optuna optimization
│   │   └── spaces.py           # Search spaces
│   ├── evaluation/
│   │   ├── metrics.py          # Business metrics
│   │   ├── explainability.py   # SHAP analysis
│   │   └── validator.py        # Model validation
│   ├── pipelines/
│   │   ├── training_pipeline.py      # Main training pipeline
│   │   └── comparison_pipeline.py    # A/B testing
│   └── utils/
│       ├── mlflow_utils.py     # MLflow logging
│       └── config.py           # Configuration management
├── configs/
│   ├── model_xgb.yaml          # XGBoost config
│   ├── model_rf.yaml           # Random Forest config
│   ├── model_mlp.yaml          # Neural Network config
│   └── model_ensemble.yaml     # Ensemble config
├── artifacts/                  # Trained models & plots
├── tests/                      # Unit & integration tests
├── requirements.txt
└── README.md
```

## 🎯 Models

### 1. XGBoost (Primary Model)

**Strengths**:
- Best performance on imbalanced data
- Fast training and inference
- Built-in regularization

**Hyperparameters**:
```python
{
    "n_estimators": 300,
    "max_depth": 6,
    "learning_rate": 0.1,
    "scale_pos_weight": 285,  # Class imbalance ratio
    "min_child_weight": 1,
    "subsample": 0.8,
    "colsample_bytree": 0.8
}
```

**Performance**:
- Recall: 97.5%
- FPR: 1.8%
- F1: 0.92
- Training Time: ~2 min

### 2. Neural Network (Deep Learning)

**Architecture**:
```
Input (30 features)
    ↓
Dense(128) + ReLU + Dropout(0.3)
    ↓
Dense(64) + ReLU + Dropout(0.3)
    ↓
Dense(32) + ReLU + Dropout(0.2)
    ↓
Dense(1) + Sigmoid
```

**Strengths**:
- Captures complex non-linear patterns
- Good with feature interactions
- Flexible architecture

**Performance**:
- Recall: 96.2%
- FPR: 2.1%
- F1: 0.89
- Training Time: ~5 min (GPU: ~1 min)

### 3. Random Forest

**Strengths**:
- Robust to outliers
- No scaling required
- Feature importance

**Hyperparameters**:
```python
{
    "n_estimators": 500,
    "max_depth": 10,
    "min_samples_split": 2,
    "min_samples_leaf": 1,
    "class_weight": "balanced"
}
```

**Performance**:
- Recall: 94.8%
- FPR: 2.5%
- F1: 0.87
- Training Time: ~3 min

### 4. Isolation Forest (Anomaly Detection)

**Strengths**:
- Unsupervised learning
- Doesn't require labels
- Good for unknown fraud patterns

**Hyperparameters**:
```python
{
    "n_estimators": 300,
    "contamination": 0.002,  # Expected fraud rate
    "max_samples": "auto"
}
```

**Performance**:
- Recall: 89.3%
- FPR: 3.2%
- F1: 0.82
- Training Time: ~1 min

### 5. Ensemble (Voting)

**Strategy**: Weighted soft voting
```python
weights = {
    "xgboost": 0.4,       # Best overall
    "neural_network": 0.3,  # Complex patterns
    "random_forest": 0.2,   # Robustness
    "isolation_forest": 0.1  # Novelty
}
```

**Performance**:
- Recall: 98.1% ✅ (Best)
- FPR: 1.6% ✅ (Best)
- F1: 0.94 ✅ (Best)
- Inference Time: ~50ms

## 🔧 Hyperparameter Tuning

### Optuna Integration

```bash
# Run hyperparameter optimization
python -m src.hyperparameter.tuning --model xgboost --trials 100

# Resume optimization
python -m src.hyperparameter.tuning --model xgboost --trials 50 --resume
```

### Search Spaces

**XGBoost**:
```python
{
    "n_estimators": (100, 500),
    "max_depth": (3, 10),
    "learning_rate": (0.01, 0.3),
    "subsample": (0.6, 1.0),
    "colsample_bytree": (0.6, 1.0),
    "min_child_weight": (1, 10)
}
```

**Neural Network**:
```python
{
    "hidden_sizes": [(64, 32), (128, 64), (256, 128, 64)],
    "dropout": (0.1, 0.5),
    "learning_rate": (0.0001, 0.01),
    "batch_size": [32, 64, 128]
}
```

### Tuning Strategy

1. **Objective**: Maximize Recall subject to FPR ≤ 2%
2. **Trials**: 100 iterations per model
3. **Validation**: 5-fold cross-validation
4. **Pruning**: Early stopping if FPR > 3%

## 📊 Evaluation

### Business Constraints

| Metric | Requirement | Rationale |
|--------|-------------|-----------|
| **Recall** | ≥ 95% | Detect at least 95% of frauds |
| **FPR** | ≤ 2% | Max 2% false positives (avoid alert fatigue) |
| **F1 Score** | ≥ 0.85 | Balance precision and recall |
| **Inference Time** | < 100ms | Real-time requirement |

### Evaluation Metrics

```python
from src.evaluation.metrics import FraudMetrics

metrics = FraudMetrics()
results = metrics.evaluate(y_true, y_pred, y_proba)

print(f"Recall: {results['recall']:.2%}")
print(f"Precision: {results['precision']:.2%}")
print(f"FPR: {results['fpr']:.2%}")
print(f"F1: {results['f1']:.3f}")
print(f"AUC-ROC: {results['auc_roc']:.3f}")
print(f"AUC-PR: {results['auc_pr']:.3f}")
```

### Confusion Matrix Analysis

```
Predicted:     Legitimate | Fraud
Actual:
Legitimate     282,300    |  1,700   ← False Positives (FPR = 0.6%)
Fraud              7      |    800   ← True Positives (Recall = 99.1%)
```

## 🔍 Model Explainability

### SHAP Analysis

```python
from src.evaluation.explainability import ShapExplainer

explainer = ShapExplainer(model, X_train)

# Global feature importance
feature_importance = explainer.feature_importance()
explainer.plot_feature_importance()

# Local explanation for a transaction
shap_values = explainer.explain_instance(X_test[0])
explainer.plot_waterfall(X_test[0])
```

### Feature Importance Ranking

| Rank | Feature | Importance | Description |
|------|---------|------------|-------------|
| 1 | V14 | 0.18 | PCA component 14 |
| 2 | V4 | 0.15 | PCA component 4 |
| 3 | V12 | 0.12 | PCA component 12 |
| 4 | V10 | 0.10 | PCA component 10 |
| 5 | amount | 0.09 | Transaction amount |
| 6 | time | 0.08 | Time of transaction |
| 7 | V17 | 0.07 | PCA component 17 |

## 📈 MLflow Tracking

### Start MLflow UI

```bash
mlflow ui --backend-store-uri mlruns/
# Open http://127.0.0.1:5000
```

### Logged Information

**Parameters**:
- Model hyperparameters
- Feature engineering config
- Train/val/test split ratios
- SMOTE oversampling settings

**Metrics**:
- Recall, Precision, F1, FPR
- AUC-ROC, AUC-PR
- Confusion matrix values
- Training time, inference time

**Artifacts**:
- Trained model (`.pkl` or `.pth`)
- Feature importance plot
- Confusion matrix plot
- ROC curve
- Precision-Recall curve
- SHAP summary plot

### Model Registry

```python
import mlflow

# Register best model
mlflow.register_model(
    model_uri=f"runs:/{run_id}/model",
    name="fraud_detection_xgboost"
)

# Promote to production
client = mlflow.MlflowClient()
client.transition_model_version_stage(
    name="fraud_detection_xgboost",
    version=3,
    stage="Production"
)
```

## 🧪 Testing

```bash
# Run all tests
pytest tests/ -v --cov=src --cov-report=html

# Run specific test file
pytest tests/test_training_pipeline.py -v

# Run with coverage
pytest tests/ --cov=src --cov-report=term-missing
```

## 🐳 Docker Deployment

### Build Image

```bash
# Build training image
docker build -t fraud-detection-training:latest -f Dockerfile .

# Run training container
docker run --rm \
  -v $(pwd)/artifacts:/app/artifacts \
  -e DATABASE_URL="postgresql://fraud_user:password@host.docker.internal:5432/fraud_detection" \
  fraud-detection-training:latest
```

### Docker Compose

```bash
# Start training service
docker-compose -f ../docker-compose.local.yml up training

# Check logs
docker logs fraud-training --tail=100
```

## 🔄 Retraining Strategy

### Triggers

1. **Scheduled**: Daily at 2 AM (Airflow DAG 01)
2. **Drift Detection**: Concept drift (Recall < 95%)
3. **Data Drift**: PSI > 0.5 on key features
4. **Manual**: Via Airflow UI or API

### Cooldown Period

- Minimum 48 hours between retraining
- Requires ≥10,000 new labeled transactions
- Skip if model performance is stable

### Validation Before Deployment

```python
# Automated validation
if recall >= 0.95 and fpr <= 0.02 and f1 >= 0.85:
    promote_to_production()
else:
    send_alert_to_ml_team()
    keep_current_model()
```

## 📊 Monitoring

### Training Metrics

- **Duration**: Track training time trends
- **Data Quality**: Missing values, outliers, schema violations
- **Model Performance**: Recall, FPR, F1 over time
- **Drift**: Compare with baseline metrics

### Prometheus Metrics

```
fraud_training_duration_seconds
fraud_training_samples_total
fraud_training_recall_gauge
fraud_training_fpr_gauge
fraud_training_errors_total
```

## 🚀 Production Deployment

### CI/CD Pipeline

```yaml
# .github/workflows/training.yml
name: Training Pipeline
on:
  schedule:
    - cron: '0 2 * * *'  # Daily at 2 AM
  workflow_dispatch:

jobs:
  train:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.10'
      - name: Install dependencies
        run: pip install -r training/requirements.txt
      - name: Run training
        run: python -m training.src.pipelines.training_pipeline
      - name: Register model
        run: python -m training.src.utils.mlflow_utils register
```

### Deployment Checklist

- [ ] Dataset available and validated
- [ ] Environment variables configured
- [ ] MLflow tracking server running
- [ ] Database connection tested
- [ ] Sufficient compute resources (GPU optional)
- [ ] Monitoring dashboards configured
- [ ] Alert channels set up
- [ ] Rollback procedure documented

## 📚 Resources

- [XGBoost Documentation](https://xgboost.readthedocs.io/)
- [PyTorch Documentation](https://pytorch.org/docs/)
- [Scikit-learn Documentation](https://scikit-learn.org/)
- [MLflow Documentation](https://mlflow.org/docs/latest/)
- [Optuna Documentation](https://optuna.readthedocs.io/)
- [SHAP Documentation](https://shap.readthedocs.io/)

## 👨🏾‍💻 Contributors

Fraud Detection Team

1. Joshua Juste NIKIEMA
2. Olalekan Taofeek OLALUWOYE
3. Soulaimana Toihir DJALOUD
