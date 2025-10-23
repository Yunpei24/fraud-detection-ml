# SHAP Explainers Configuration

## Overview

This document explains how SHAP explainers are configured and used in the Fraud Detection API. Each ML model in the ensemble has its own dedicated SHAP explainer for interpretability.

## Configuration Structure

### Settings (`src/config/settings.py`)

```python
# SHAP Explainers (one per model)
shap_explainer_xgb_name: str = Field(default="shap_explainer_xgb.pkl")
shap_explainer_nn_name: str = Field(default="shap_explainer_nn.pkl")
shap_explainer_iforest_name: str = Field(default="shap_explainer_iforest.pkl")
```

### Constants (`src/config/constants.py`)

```python
MODEL_PATHS = {
    "xgboost": "/app/models/xgboost_fraud_model.pkl",
    "neural_network": "/app/models/nn_fraud_model.pth",
    "isolation_forest": "/app/models/isolation_forest_model.pkl",
    "shap_explainer_xgb": "/app/models/shap_explainer_xgb.pkl",
    "shap_explainer_nn": "/app/models/shap_explainer_nn.pkl",
    "shap_explainer_iforest": "/app/models/shap_explainer_iforest.pkl"
}
```

## Model Directory Structure

```
/app/models/
├─ xgboost_fraud_model.pkl          # XGBoost model
├─ nn_fraud_model.pth               # Neural Network model
├─ isolation_forest_model.pkl       # Isolation Forest model
│
├─ shap_explainer_xgb.pkl           # SHAP for XGBoost
├─ shap_explainer_nn.pkl            # SHAP for Neural Network
└─ shap_explainer_iforest.pkl       # SHAP for Isolation Forest
```

## How to Create SHAP Explainers (Training Module)

### Training Flow

```python
import shap
import pickle
import xgboost as xgb
from sklearn.ensemble import IsolationForest

# 1. Train XGBoost model
xgb_model = xgb.XGBClassifier()
xgb_model.fit(X_train, y_train)

# 2. Create SHAP explainer for XGBoost
xgb_explainer = shap.TreeExplainer(xgb_model)

# Save both
with open('/app/models/xgboost_fraud_model.pkl', 'wb') as f:
    pickle.dump(xgb_model, f)

with open('/app/models/shap_explainer_xgb.pkl', 'wb') as f:
    pickle.dump(xgb_explainer, f)

# 3. Train Isolation Forest
iforest_model = IsolationForest(n_estimators=100)
iforest_model.fit(X_train)

# 4. Create SHAP explainer for Isolation Forest
iforest_explainer = shap.TreeExplainer(iforest_model)

# Save both
with open('/app/models/isolation_forest_model.pkl', 'wb') as f:
    pickle.dump(iforest_model, f)

with open('/app/models/shap_explainer_iforest.pkl', 'wb') as f:
    pickle.dump(iforest_explainer, f)

# 5. For Neural Network (use DeepExplainer - more expensive)
# NOTE: DeepExplainer is slower. Consider background data sampling.
nn_explainer = shap.DeepExplainer(nn_model, X_background)

with open('/app/models/shap_explainer_nn.pkl', 'wb') as f:
    pickle.dump(nn_explainer, f)
```

## API Implementation

### Loading SHAP Explainers (`src/models/ml_models/ensemble.py`)

```python
class EnsembleModel:
    def __init__(self):
        self.shap_explainer_xgb = None
        self.shap_explainer_nn = None
        self.shap_explainer_iforest = None
    
    def load_models(self):
        # Load XGBoost SHAP
        with open(f"{self.models_path}/shap_explainer_xgb.pkl", "rb") as f:
            self.shap_explainer_xgb = pickle.load(f)
        
        # Load Neural Network SHAP
        with open(f"{self.models_path}/shap_explainer_nn.pkl", "rb") as f:
            self.shap_explainer_nn = pickle.load(f)
        
        # Load Isolation Forest SHAP
        with open(f"{self.models_path}/shap_explainer_iforest.pkl", "rb") as f:
            self.shap_explainer_iforest = pickle.load(f)
```

### Generating Explanations

```python
def explain_prediction(
    self,
    features: np.ndarray,
    prediction_result: Dict[str, Any],
    model_type: str = "xgboost"  # Can be "xgboost", "neural_network", "isolation_forest"
) -> Dict[str, Any]:
    """Generate SHAP explanation for a specific model."""
    
    # Select appropriate explainer
    if model_type == "xgboost":
        explainer = self.shap_explainer_xgb
    elif model_type == "neural_network":
        explainer = self.shap_explainer_nn
    elif model_type == "isolation_forest":
        explainer = self.shap_explainer_iforest
    
    # Calculate SHAP values
    shap_values = explainer.shap_values(features)
    
    # Process and return explanations
    return {
        "model": model_type,
        "top_features": sorted_features[:10],
        "method": "SHAP"
    }
```

## Usage in Prediction Service

```python
from src.models.ml_models.ensemble import EnsembleModel

model = EnsembleModel()
model.load_models()

# Make prediction
features = np.array([[...30 features...]])
prediction = model.predict(features)

# Get explanation for XGBoost model
explanation = model.explain_prediction(
    features,
    prediction,
    model_type="xgboost"
)

# Result:
{
    "model": "xgboost",
    "top_features": {
        "feature_10": 0.45,
        "feature_5": -0.23,
        ...
    },
    "method": "SHAP"
}
```

## API Endpoint Example

```bash
POST /api/v1/predict
{
    "transaction_id": "TXN-001",
    "features": [0.5, -1.36, 2.54, ...28 more features],
    "metadata": {
        "explain": true,
        "explanation_model": "xgboost"
    }
}

Response:
{
    "transaction_id": "TXN-001",
    "prediction": 1,
    "fraud_score": 0.85,
    "confidence": 0.92,
    "explanation": {
        "model": "xgboost",
        "top_features": {
            "feature_10": 0.45,
            "feature_5": -0.23
        },
        "method": "SHAP"
    }
}
```

## Key Points

1. **One Explainer Per Model**: Each ML model has its own SHAP explainer
   - More accurate explanations
   - Can explain ensemble voting

2. **Not Trained**: SHAP explainers are CREATED from trained models
   - No hyperparameters to tune
   - Instantaneous to create
   - Calculation happens at prediction time

3. **Cost Trade-off**:
   - **XGBoost/IForest**: TreeExplainer (fast)
   - **Neural Network**: DeepExplainer (slower, background data sampling recommended)

4. **Enable/Disable**: SHAP can be disabled via settings:
   ```python
   enable_shap_explanation: bool = Field(default=True)
   ```

5. **Configuration**: Settings are externalized for flexibility:
   ```python
   shap_explainer_xgb_name: str = Field(default="shap_explainer_xgb.pkl")
   ```

## Training Module Checklist

- [ ] Train XGBoost, NN, and IForest models
- [ ] Create SHAP TreeExplainer for XGBoost
- [ ] Create SHAP TreeExplainer for Isolation Forest
- [ ] Create SHAP DeepExplainer for Neural Network (with background data)
- [ ] Save all 6 files to `/app/models/`:
  - `xgboost_fraud_model.pkl`
  - `shap_explainer_xgb.pkl`
  - `nn_fraud_model.pth`
  - `shap_explainer_nn.pkl`
  - `isolation_forest_model.pkl`
  - `shap_explainer_iforest.pkl`
- [ ] Verify all files load correctly in API

## References

- SHAP Documentation: https://shap.readthedocs.io/
- TreeExplainer: For tree-based models (XGBoost, RandomForest, IsolationForest)
- DeepExplainer: For neural networks
- SHAP Values: Game theory approach to explain model predictions
