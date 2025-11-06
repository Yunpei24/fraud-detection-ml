# 📦 Where are ML Models Stored?
**Date:** November 4, 2025
**Container:** `fraud-api`
---
## 🎯 Current Situation Summary
### ✅ **Configuration:**
- **Configured Path:** `/mnt/fraud-models/champion/`
- **Current State:** ❌ No real models exist
- **Fallback Activated:** ✅ The API uses **mock models** (dummy models)
---
## 📂 Model Storage Structure
### **1. Path Configuration in the API**
```python
# api/src/config/settings.py
model_path = os.getenv(
    "ML_MODEL_PATH",
    os.getenv("MODEL_PATH", os.getenv("AZURE_STORAGE_MOUNT_PATH", "/mnt/fraud-models")),
)
```
**Resolution Priority:**
1. `ML_MODEL_PATH` (environment variable)
2. `MODEL_PATH` (environment variable)
3. `AZURE_STORAGE_MOUNT_PATH` (environment variable)
4. `/mnt/fraud-models` (default)
### **2. Current Path in the Container**
```bash
$ docker exec fraud-api python -c "from src.config.settings import settings; print(settings.model_path)"
/mnt/fraud-models
```
### **3. Complete Directory with Traffic Routing**
The system uses a **champion/canary** approach:
```
/mnt/fraud-models/
├── champion/              # Production models (100% traffic)
│   ├── xgboost_fraud_model.pkl
│   ├── random_forest_fraud_model.pkl
│   ├── nn_fraud_model.pth
│   ├── isolation_forest_model.pkl
│   ├── shap_explainer_xgb.pkl
│   ├── shap_explainer_rf.pkl
│   ├── shap_explainer_nn.pkl
│   └── shap_explainer_iforest.pkl
│
└── canary/                # Test models (0-25% traffic)
    ├── xgboost_fraud_model.pkl
    ├── random_forest_fraud_model.pkl
    ├── nn_fraud_model.pth
    └── isolation_forest_model.pkl
```
---
## 🔍 Current State in the Container
### **Manual Verification:**
```bash
# 1. Check if the directory exists
$ docker exec fraud-api ls -lah /mnt/fraud-models/
ls: cannot access '/mnt/fraud-models/': No such file or directory
```
**❌ The directory does not exist!**
### **Container Logs:**
```json
{
    "level": "WARNING",
    "message": "Isolation Forest not found at /mnt/fraud-models/champion/isolation_forest_model.pkl, using mock"
}
{
    "level": "WARNING",
    "message": "SHAP explainer (XGBoost) not found at /mnt/fraud-models/champion/shap_explainer_xgb.pkl"
}
{
    "level": "INFO",
    "message": "All models loaded successfully"
}
{
    "level": "INFO",
    "message": "Available models: ['xgboost', 'random_forest', 'neural_network', 'isolation_forest', 'ensemble']"
}
```
**✅ The API is running with mock models (dummy models)!**
---
## 🤖 Mock Models System
The API has a fallback mechanism that creates **dummy models** when real models don't exist:
### **Fallback Code:**
```python
# api/src/models/ml_models/ensemble.py
def load_models(self) -> None:
    """Load all models from disk."""
    # Try to load XGBoost
    xgboost_path = os.path.join(self.models_path, settings.xgboost_model_name)
    if os.path.exists(xgboost_path):
        with open(xgboost_path, "rb") as f:
            self.xgboost_model = pickle.load(f)
        logger.info("✅ XGBoost model loaded")
    else:
        logger.warning(f"XGBoost model not found at {xgboost_path}, using mock")
        self.xgboost_model = self._create_mock_model("xgboost")  # ← Mock!
```
### **Currently Active Mock Models:**
| Model | Expected File | Status | Type Used |
|-------|--------------|--------|-----------|
| XGBoost | `xgboost_fraud_model.pkl` | ❌ Not Found | 🤖 Mock |
| Random Forest | `random_forest_fraud_model.pkl` | ❌ Not Found | 🤖 Mock |
| Neural Network | `nn_fraud_model.pth` | ❌ Not Found | 🤖 Mock |
| Isolation Forest | `isolation_forest_model.pkl` | ❌ Not Found | 🤖 Mock |
| SHAP Explainers | `shap_explainer_*.pkl` | ❌ Not Found | ❌ Disabled |
---
## 📍 Where are Real Models Created?
### **1. Training Container (`fraud-training`)**
Models are created by the **Airflow DAG `01_training_pipeline`**:
```bash
# Inside the training container
/app/models/           # Models saved locally
/mlflow/artifacts/     # Models registered in MLflow
```
**Command to verify:**
```bash
docker exec fraud-training ls -lah /app/models/
```
### **2. MLflow Model Registry**
Trained models are **registered in MLflow**:
- **MLflow URL:** http://localhost:5001
- **Registry path:** `/mlflow/artifacts/`
- **Stages:** None → Staging → Production
**View models in MLflow:**
```bash
curl http://localhost:5001/api/2.0/mlflow/registered-models/list | jq .
```
### **3. Azure File Share (Production Only)**
In production on Azure, models are stored in **Azure File Share**:
- **Storage Account:** `joshfraudstorageaccount`
- **File Share:** `fraud-models`
- **Mount point:** `/mnt/fraud-models`
---
## 🔄 How do Models Get to the API?
### **Complete Flow:**
```
┌─────────────────────────────────────────────────────────────────┐
│              TRAINING → API DEPLOYMENT                          │
└─────────────────────────────────────────────────────────────────┘

STEP 1: TRAINING
├─ Airflow DAG 01_training_pipeline
├─ Container fraud-training
├─ Trains XGBoost, RF, NN, Isolation Forest
├─ Saves to /app/models/
└─ Registers in MLflow Registry → Stage: None

STEP 2: STAGING PROMOTION
├─ Airflow DAG 05_model_deployment_canary_http
├─ Promotes models: None → Staging in MLflow
├─ Script deploy_canary.py
│   ├─ Downloads models from MLflow
│   ├─ Saves to /mnt/fraud-models/canary/
│   └─ Updates traffic_routing.json (5% canary)
└─ API auto-reload detects new files

STEP 3: CANARY 25%
├─ Airflow DAG 05_model_deployment_canary_http
├─ Updates traffic_routing.json (25% canary)
└─ API auto-reload detects the change

STEP 4: PRODUCTION PROMOTION
├─ Airflow DAG 05_model_deployment_canary_http
├─ Promotes models: Staging → Production in MLflow
├─ Script promote_to_production.py
│   ├─ Copies /mnt/fraud-models/canary/ → /mnt/fraud-models/champion/
│   └─ Updates traffic_routing.json (canary disabled)
└─ API auto-reload detects new files
```
---
## 🛠️ How to Create Models Manually?
### **Method 1: Trigger the Training DAG**
```bash
# 1. Go to Airflow UI
http://localhost:8080

# 2. Find the DAG "01_training_pipeline"
# 3. Click "Trigger DAG"
# 4. Wait for training to complete (~30-60 minutes)
# 5. Verify models in MLflow
http://localhost:5001
```
### **Method 2: Manual Training in Container**
```bash
# 1. Enter the training container
docker exec -it fraud-training bash

# 2. Run the training script
python -m src.pipelines.training_pipeline

# 3. Verify created models
ls -lah /app/models/

# 4. Copy to API (temporary for dev)
docker cp fraud-training:/app/models/xgboost_fraud_model.pkl /tmp/
docker exec fraud-api mkdir -p /mnt/fraud-models/champion
docker cp /tmp/xgboost_fraud_model.pkl fraud-api:/mnt/fraud-models/champion/
```
### **Method 3: Use Test Models**
For local development, you can create simple models:
```python
# Inside the API container
docker exec -it fraud-api python

>>> import pickle
>>> from sklearn.ensemble import RandomForestClassifier
>>> import os
>>>
>>> # Create the directory
>>> os.makedirs("/mnt/fraud-models/champion", exist_ok=True)
>>>
>>> # Create a simple model
>>> model = RandomForestClassifier(n_estimators=10)
>>>
>>> # Save
>>> with open("/mnt/fraud-models/champion/xgboost_fraud_model.pkl", "wb") as f:
...     pickle.dump(model, f)
>>>
>>> print("✅ Test model created!")
```
---
## 🔍 Diagnostic Commands
### **1. Check Configured Path**
```bash
docker exec fraud-api python -c "from src.config.settings import settings; print('Model Path:', settings.model_path)"
```
### **2. List Available Models**
```bash
docker exec fraud-api find /mnt/fraud-models -name "*.pkl" -o -name "*.pth"
```
### **3. Check Loading Logs**
```bash
docker logs fraud-api 2>&1 | grep -i "model\|loading"
```
### **4. Test API with Mock Models**
```bash
# Get a token
TOKEN=$(curl -s -X POST "http://localhost:8000/auth/login" \
  -H "Content-Type: application/x-www-form-urlencoded" \
  -d "username=admin&password=admin123" | jq -r '.access_token')

# List models
curl -X GET "http://localhost:8000/api/v1/explain/models" \
  -H "Authorization: Bearer $TOKEN"

# Result with mock models:
["xgboost", "random_forest", "neural_network", "isolation_forest", "ensemble"]
```
### **5. Check Docker Volumes Status**
```bash
docker volume ls | grep fraud
docker volume inspect fraud-detection-ml_mlflow_artifacts
```
---
## 📋 Summary
| Question | Answer |
|----------|--------|
| **Where are models stored?** | `/mnt/fraud-models/champion/` (configured) |
| **Do models currently exist?** | ❌ No, the directory doesn't exist |
| **Does the API still work?** | ✅ Yes, with mock models (dummy models) |
| **How to create real models?** | Trigger Airflow DAG `01_training_pipeline` |
| **Where are models after training?** | `/mlflow/artifacts/` in MLflow Registry |
| **How to deploy them to API?** | Via DAG `05_model_deployment_canary_http` |
| **Can we test without real models?** | ✅ Yes, mock models allow API testing |
---
## 🚀 Next Steps
1. **Launch training** to create real models
2. **Register in MLflow** for versioning
3. **Deploy via canary DAG** for production-ready setup
4. **Test with real models** for complete validation
---
**Need help?** Check out:
- [AUTO_RELOAD_GUIDE.md](AUTO_RELOAD_GUIDE.md) - Model auto-reload
- [DEPLOYMENT_API.md](DEPLOYMENT_API.md) - Canary deployment
- [README.md](../README.md) - General documentation