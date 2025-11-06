# 📦 Where are Models Stored After Training?

**Date:** November 4, 2025  
**After fixing the MLflow registration bug**

---

## 🎯 Overview

After executing the DAG `01_training_pipeline`, models are stored in **3 different locations**:

```
┌─────────────────────────────────────────────────────────────────┐
│              STORAGE LOCATIONS AFTER TRAINING                    │
└─────────────────────────────────────────────────────────────────┘

1️⃣  MLflow Tracking Server (Runs)
    📍 Location: MLflow container - http://localhost:5001
    📂 Path: /mlflow/artifacts/<experiment_id>/<run_id>/
    📊 Content: Model artifacts, metrics, parameters, plots
    🔄 Lifecycle: Permanent (until manual deletion)

2️⃣  MLflow Model Registry
    📍 Location: MLflow container - Model Registry
    📂 Path: models:/fraud_detection_<model_name>/Staging
    📊 Content: Model versions with stage (None → Staging → Production)
    🔄 Lifecycle: Versioning with stage promotion

3️⃣  Training Container (temporary)
    📍 Location: fraud-training container
    📂 Path: /app/models/ (empty after training)
    📊 Content: None (models not saved locally)
    🔄 Lifecycle: Ephemeral (lost on restart)
```

---

## 🔍 1. MLflow Tracking Server (Runs)

### **Where are the artifacts?**

Models are logged in **MLflow Tracking** during training:

```bash
# MLflow Container
docker exec fraud-mlflow ls -lah /mlflow/artifacts/
```

**Artifacts structure:**
```
/mlflow/artifacts/
├── 1/                          # Experiment ID (fraud_detection_training)
│   ├── <run_id_1>/            # Run: register_xgboost
│   │   ├── artifacts/
│   │   │   ├── model/         # XGBoost model
│   │   │   │   ├── MLmodel
│   │   │   │   ├── model.pkl
│   │   │   │   └── requirements.txt
│   │   │   └── xgboost_metadata.json
│   │   ├── metrics/
│   │   └── params/
│   │
│   ├── <run_id_2>/            # Run: register_random_forest
│   │   └── artifacts/
│   │       └── model/         # Random Forest model
│   │
│   ├── <run_id_3>/            # Run: register_neural_network
│   │   └── artifacts/
│   │       └── model/         # Neural Network model
│   │
│   └── <run_id_4>/            # Run: register_isolation_forest
│       └── artifacts/
│           └── model/         # Isolation Forest model
```

### **How to view them?**

**Method 1: MLflow Web Interface**
```bash
# Open in browser
http://localhost:5001
```

**Method 2: CLI in container**
```bash
# List experiments
docker exec fraud-training python -c "
import mlflow
mlflow.set_tracking_uri('http://mlflow:5000')
client = mlflow.MlflowClient()

experiments = client.search_experiments()
for exp in experiments:
    print(f'Experiment: {exp.name} (ID: {exp.experiment_id})')
    runs = client.search_runs([exp.experiment_id], max_results=10)
    print(f'  Total runs: {len(runs)}')
    for run in runs:
        print(f'    - {run.info.run_name} ({run.info.run_id[:8]}...)')
"
```

**Method 3: Check artifacts directly**
```bash
# Find experiment ID
docker exec fraud-mlflow ls -lah /mlflow/artifacts/

# List runs in experiment
docker exec fraud-mlflow ls -lah /mlflow/artifacts/1/

# View artifacts of a specific run
docker exec fraud-mlflow ls -lah /mlflow/artifacts/1/<run_id>/artifacts/model/
```

---

## 🏷️ 2. MLflow Model Registry

### **What is the Model Registry?**

The **Model Registry** is a **versioning database** for ML models:
- Stores model **metadata** (name, version, stage, description)
- Points to **artifacts** in the Tracking Server
- Enables **promotion** between stages (None → Staging → Production)

### **Where are models in the Registry?**

Models are **registered under names**:

```
📦 MLflow Model Registry
├── fraud_detection_xgboost
│   └── Version 1 (Stage: Staging)
│       ├── Run ID: <run_id>
│       └── Source: runs:/<run_id>/model
│
├── fraud_detection_random_forest
│   └── Version 1 (Stage: Staging)
│
├── fraud_detection_neural_network
│   └── Version 1 (Stage: Staging)
│
└── fraud_detection_isolation_forest
    └── Version 1 (Stage: Staging)
```

### **How to view them?**

**Method 1: MLflow Web Interface**
```bash
http://localhost:5001/#/models
```

**Method 2: Python API**
```bash
docker exec fraud-training python -c "
import mlflow
mlflow.set_tracking_uri('http://mlflow:5000')
client = mlflow.MlflowClient()

# List all registered models
models = client.search_registered_models()

print(f'📦 {len(models)} models registered in MLflow:\n')
for model in models:
    print(f'Model: {model.name}')
    
    # List versions
    versions = client.search_model_versions(f\"name='{model.name}'\")
    for v in versions:
        print(f'  - Version {v.version}')
        print(f'    Stage: {v.current_stage}')
        print(f'    Run ID: {v.run_id}')
        print(f'    Source: {v.source}')
    print()
"
```

**Method 3: Load a model from Registry**
```python
import mlflow

mlflow.set_tracking_uri('http://mlflow:5000')

# Load a model from Registry
model_uri = "models:/fraud_detection_xgboost/Staging"
model = mlflow.pyfunc.load_model(model_uri)

print(f"Model loaded: {model}")
```


### **How to verify?**

```bash
# Check models directory
docker exec fraud-training ls -lah /app/models/
# Output: total 8.0K (empty)

# Models are in MLflow, not locally
docker exec fraud-training python -c "
import mlflow
mlflow.set_tracking_uri('http://mlflow:5000')
client = mlflow.MlflowClient()
models = client.search_registered_models()
print(f'{len(models)} models in MLflow Registry')
"
```

---

## 🔄 Complete Flow: Training → MLflow → API

```
┌─────────────────────────────────────────────────────────────────┐
│                  MODEL LIFECYCLE FLOW                            │
└─────────────────────────────────────────────────────────────────┘

STEP 1: TRAINING (DAG 01_training_pipeline)
├─ Container: fraud-training
├─ Script: src/pipelines/training_pipeline.py
├─ Actions:
│  ├─ Train XGBoost, Random Forest, Neural Network, Isolation Forest
│  ├─ Evaluate models on test set
│  ├─ For each model:
│  │   ├─ Create MLflow run: register_<model_name>
│  │   ├─ Log model: mlflow.sklearn.log_model(model, "model")
│  │   ├─ Register: mlflow.register_model("runs:/<run_id>/model", name)
│  │   └─ Transition stage: Staging
│  └─ Log metadata and plots
└─ Result: 4 models in MLflow Registry (Stage: Staging)

STEP 2: STORAGE (MLflow)
├─ Location 1: Tracking Server
│  ├─ Path: /mlflow/artifacts/1/<run_id>/artifacts/model/
│  └─ Content: model.pkl, MLmodel, requirements.txt
│
└─ Location 2: Model Registry
   ├─ Name: fraud_detection_<model_name>
   ├─ Version: 1
   ├─ Stage: Staging
   └─ Source: runs:/<run_id>/model

STEP 3: DEPLOYMENT (DAG 05_model_deployment_canary_http)
├─ Container: airflow-worker (calls API)
├─ Script: api/scripts/deploy_canary.py
├─ Actions:
│  ├─ Load models from MLflow: models:/fraud_detection_*/Staging
│  ├─ Save to Azure File Share: /mnt/fraud-models/canary/
│  ├─ Update traffic_routing.json (5% canary)
│  └─ API auto-reloads models
└─ Result: Models deployed in API container

STEP 4: SERVING (API)
├─ Container: fraud-api
├─ Path: /mnt/fraud-models/champion/ (or canary/)
├─ Auto-reload: Timestamp-based detection
└─ Endpoints:
   ├─ POST /api/v1/predict (predictions)
   ├─ GET /api/v1/explain/models (list models)
   └─ POST /api/v1/explain/shap (SHAP explanations)
```

---

## 🛠️ Diagnostic Commands

### **1. Check that models are in MLflow Registry**

```bash
docker exec fraud-training python -c "
import mlflow
mlflow.set_tracking_uri('http://mlflow:5000')
client = mlflow.MlflowClient()

models = client.search_registered_models()
if not models:
    print('❌ No models in Registry - Training failed or bug in registration')
else:
    print(f'✅ {len(models)} models in Registry')
    for model in models:
        versions = client.search_model_versions(f\"name='{model.name}'\")
        print(f'  - {model.name}: {len(versions)} version(s)')
        for v in versions:
            print(f'      Version {v.version} ({v.current_stage})')
"
```

### **2. Check artifacts in MLflow**

```bash
# Find experiment ID
docker exec fraud-mlflow ls /mlflow/artifacts/

# List runs
docker exec fraud-mlflow ls /mlflow/artifacts/1/

# View artifacts of a run
docker exec fraud-mlflow find /mlflow/artifacts/1/ -name "model.pkl" | head -5
```

### **3. Load a model from MLflow**

```bash
docker exec fraud-training python -c "
import mlflow

mlflow.set_tracking_uri('http://mlflow:5000')

# Load XGBoost model in Staging
model_uri = 'models:/fraud_detection_xgboost/Staging'
try:
    model = mlflow.pyfunc.load_model(model_uri)
    print(f'✅ Model loaded successfully from {model_uri}')
    print(f'   Type: {type(model)}')
except Exception as e:
    print(f'❌ Failed to load model: {e}')
"
```

### **4. Check MLflow Docker volume**

```bash
# Inspect mlflow_artifacts volume
docker volume inspect fraud-detection-ml_mlflow_artifacts

# View volume size
docker system df -v | grep mlflow_artifacts
```

---

## 📋 Location Summary

| Where? | Path | Content | When? |
|--------|------|---------|-------|
| **MLflow Tracking** | `/mlflow/artifacts/1/<run_id>/` | Model artifacts (.pkl, MLmodel) | After `log_model()` |
| **MLflow Registry** | `models:/fraud_detection_*/Staging` | Metadata + pointer to artifacts | After `register_model()` |
| **Training Container** | `/app/models/` | ❌ Empty (not used) | Never |
| **API Container** | `/mnt/fraud-models/champion/` | Deployed models | After `deploy_canary.py` |

---

## 🚀 Next Steps

Now that the bug is fixed, **re-run the training DAG**:

1. **Trigger the DAG in Airflow UI**
   ```bash
   # Open Airflow
   http://localhost:8080
   
   # DAG: 01_training_pipeline
   # Click: Trigger DAG
   ```

2. **Wait for training to complete** (~15-30 minutes)

3. **Verify that models are registered**
   ```bash
   docker exec fraud-training python -c "
   import mlflow
   mlflow.set_tracking_uri('http://mlflow:5000')
   client = mlflow.MlflowClient()
   models = client.search_registered_models()
   print(f'{len(models)} models registered')
   "
   ```

4. **Deploy via canary DAG**
   ```bash
   # DAG: 05_model_deployment_canary_http
   # Click: Trigger DAG
   ```

5. **Verify in API**
   ```bash
   curl -X GET "http://localhost:8000/api/v1/explain/models" \
     -H "Authorization: Bearer $TOKEN"
   ```

---

**Questions?** Check out:
- [MODEL_STORAGE_EXPLAINED.md](../api/MODEL_STORAGE_EXPLAINED.md) - API-side storage
- [DEPLOYMENT_API.md](../api/DEPLOYMENT_API.md) - Canary deployment
- [README.md](README.md) - Training documentation