# Deployment API Documentation

## 📋 Overview

The Deployment API provides admin endpoints for managing canary deployments, promotions, and rollbacks of ML models. These endpoints are designed to be called by Airflow DAGs orchestrating the deployment pipeline.

## 🏗️ Architecture

```
┌─────────────────┐         ┌──────────────────┐
│  Azure VM       │         │ Azure Web App    │
│  ┌───────────┐  │  HTTP   │  ┌────────────┐  │
│  │ Airflow   │──┼────────►│  │ API Service│  │
│  │   DAG     │  │  API    │  │ (FastAPI)  │  │
│  └───────────┘  │  Calls  │  └────────────┘  │
│                 │         │   Models loaded   │
└─────────────────┘         └──────────────────┘
```

**Key Points:**
- Airflow runs on Azure VM
- API runs on Azure Web App
- Deployment orchestration via HTTP API calls
- No `Docker Operator` needed for API deployments

## 🔒 Authentication

All deployment endpoints require **admin authentication** via JWT tokens.

### Get Admin Token

```bash
# Login as admin
TOKEN=$(curl -s -X POST https://your-api.azurewebsites.net/auth/login \
  -H "Content-Type: application/x-www-form-urlencoded" \
  -d "username=admin&password=your-admin-password" | jq -r '.access_token')

echo "Admin Token: $TOKEN"
```

### Use Token in Requests

```bash
curl -X POST https://your-api.azurewebsites.net/admin/deployment/deploy-canary \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"model_uris": [...], "traffic_percentage": 5}'
```

## 🚀 Endpoints

### 1. Deploy Canary Models

**Endpoint:** `POST /admin/deployment/deploy-canary`

Deploy challenger models in canary mode with specified traffic percentage.

**Request:**
```json
{
  "model_uris": [
    "models:/fraud_detection_xgboost/Staging",
    "models:/fraud_detection_random_forest/Staging",
    "models:/fraud_detection_neural_network/Staging",
    "models:/fraud_detection_isolation_forest/Staging"
  ],
  "traffic_percentage": 5
}
```

**Response (200 OK):**
```json
{
  "status": "success",
  "message": "Canary deployed with 5% traffic",
  "details": {
    "model_uris": ["models:/fraud_detection_xgboost/Staging", ...],
    "traffic_percentage": 5,
    "models_loaded": 4
  },
  "timestamp": "2025-11-01T12:00:00.000Z"
}
```

**cURL Example:**
```bash
curl -X POST https://your-api.azurewebsites.net/admin/deployment/deploy-canary \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "model_uris": [
      "models:/fraud_detection_xgboost/Staging",
      "models:/fraud_detection_random_forest/Staging",
      "models:/fraud_detection_neural_network/Staging",
      "models:/fraud_detection_isolation_forest/Staging"
    ],
    "traffic_percentage": 5
  }'
```

---

### 2. Get Deployment Status

**Endpoint:** `GET /admin/deployment/deployment-status`

Get current deployment configuration and status.

**Request:** No body required

**Response (200 OK):**
```json
{
  "deployment_mode": "canary_active",
  "canary_percentage": 5,
  "champion_models": [
    "models:/fraud_detection_xgboost/Production",
    "models:/fraud_detection_random_forest/Production"
  ],
  "challenger_models": [
    "models:/fraud_detection_xgboost/Staging",
    "models:/fraud_detection_random_forest/Staging"
  ],
  "last_update": "2025-11-01T12:00:00.000Z",
  "config": {
    "canary_percentage": 5,
    "champion_models": [...],
    "challenger_models": [...],
    "last_update": "2025-11-01T12:00:00.000Z"
  }
}
```

**cURL Example:**
```bash
curl -X GET https://your-api.azurewebsites.net/admin/deployment/deployment-status \
  -H "Authorization: Bearer $TOKEN"
```

---

### 3. Promote to Production

**Endpoint:** `POST /admin/deployment/promote-to-production`

Promote challenger models to production (100% traffic).

**Request:**
```json
{
  "model_uris": [
    "models:/fraud_detection_xgboost/Staging",
    "models:/fraud_detection_random_forest/Staging",
    "models:/fraud_detection_neural_network/Staging",
    "models:/fraud_detection_isolation_forest/Staging"
  ]
}
```

**Response (200 OK):**
```json
{
  "status": "success",
  "message": "Models promoted to production",
  "details": {
    "model_uris": ["models:/fraud_detection_xgboost/Staging", ...],
    "models_promoted": 4,
    "production_versions": {
      "fraud_detection_xgboost": 5,
      "fraud_detection_random_forest": 3,
      "fraud_detection_neural_network": 2,
      "fraud_detection_isolation_forest": 4
    }
  },
  "timestamp": "2025-11-01T14:00:00.000Z"
}
```

**cURL Example:**
```bash
curl -X POST https://your-api.azurewebsites.net/admin/deployment/promote-to-production \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "model_uris": [
      "models:/fraud_detection_xgboost/Staging",
      "models:/fraud_detection_random_forest/Staging",
      "models:/fraud_detection_neural_network/Staging",
      "models:/fraud_detection_isolation_forest/Staging"
    ]
  }'
```

---

### 4. Rollback Deployment

**Endpoint:** `POST /admin/deployment/rollback-deployment`

Rollback to champion models (100% traffic).

**Request:** No body required

**Response (200 OK):**
```json
{
  "status": "success",
  "message": "Rolled back to champion models",
  "details": {
    "champion_models": [
      "models:/fraud_detection_xgboost/Production",
      "models:/fraud_detection_random_forest/Production"
    ],
    "traffic_percentage": 100
  },
  "timestamp": "2025-11-01T13:30:00.000Z"
}
```

**cURL Example:**
```bash
curl -X POST https://your-api.azurewebsites.net/admin/deployment/rollback-deployment \
  -H "Authorization: Bearer $TOKEN"
```

---

## 🔄 Deployment Flow

### Progressive Canary Deployment (5% → 25% → 100%)

```
1. Deploy Canary 5%
   ├─ POST /admin/deployment/deploy-canary (traffic_percentage: 5)
   ├─ Monitor metrics for 30 minutes
   └─ Decision: Promote or Rollback?

2. Deploy Canary 25%
   ├─ POST /admin/deployment/deploy-canary (traffic_percentage: 25)
   ├─ Monitor metrics for 1 hour
   └─ Decision: Promote or Rollback?

3. Promote to Production 100%
   ├─ POST /admin/deployment/promote-to-production
   ├─ Transitions models from Staging → Production in MLflow
   └─ 100% traffic to new models

OR

Rollback
   ├─ POST /admin/deployment/rollback-deployment
   └─ Restore 100% traffic to champion models
```

## 📊 Monitoring & Metrics

The API tracks deployment metrics via Prometheus:

- **Error Rate**: `rate(http_requests_total{version="canary",status=~"5.."}[5m])`
- **Latency P95**: `histogram_quantile(0.95, rate(http_request_duration_seconds_bucket{version="canary"}[5m]))`
- **Prediction Accuracy**: Custom metrics from model predictions

### Thresholds for Rollback

| Metric | Threshold | Action |
|--------|-----------|--------|
| Error Rate | > 5% | Rollback |
| Latency P95 | > 100ms | Rollback |
| Prediction Accuracy | < 90% | Rollback |

## 🔧 Airflow Integration

### Setup Connection

```bash
cd airflow

# Setup API connection and admin token
python scripts/setup_api_connection.py \
  --api-url "https://your-api.azurewebsites.net" \
  --admin-token "your-jwt-token-here" \
  --verify
```

### DAG Configuration

The DAG `05_model_deployment_canary_http` uses `SimpleHttpOperator`:

```python
deploy_canary_5_percent = SimpleHttpOperator(
    task_id="deploy_canary_5_percent",
    http_conn_id="fraud_api_connection",  # Connection ID
    endpoint="/admin/deployment/deploy-canary",
    method="POST",
    headers={
        "Content-Type": "application/json",
        "Authorization": "Bearer {{ var.value.API_ADMIN_TOKEN }}",  # From Airflow Variables
    },
    data=json.dumps({
        "model_uris": [...],
        "traffic_percentage": 5
    }),
    response_check=check_http_response,
    log_response=True,
)
```

### Trigger DAG

```bash
# Manually trigger deployment
airflow dags trigger 05_model_deployment_canary_http

# Or via Airflow UI
# Go to DAGs → 05_model_deployment_canary_http → Trigger DAG
```

## 🐛 Troubleshooting

### Issue 1: 401 Unauthorized

**Problem:** API returns 401 Unauthorized.

**Causes:**
- Expired JWT token
- Invalid admin credentials
- Token not in Authorization header

**Solution:**
```bash
# Get new token
TOKEN=$(curl -s -X POST https://your-api.azurewebsites.net/auth/login \
  -H "Content-Type: application/x-www-form-urlencoded" \
  -d "username=admin&password=your-password" | jq -r '.access_token')

# Update Airflow variable
airflow variables set API_ADMIN_TOKEN "$TOKEN"
```

### Issue 2: 403 Forbidden

**Problem:** API returns 403 Forbidden.

**Causes:**
- User is not admin
- Incorrect role permissions

**Solution:** Ensure user has admin role in database.

### Issue 3: Connection Refused

**Problem:** Airflow cannot connect to API.

**Causes:**
- Incorrect API URL
- Network connectivity issues
- API not running

**Solution:**
```bash
# Test connection
curl -v https://your-api.azurewebsites.net/health

# Check Airflow connection
airflow connections get fraud_api_connection
```

### Issue 4: Models Not Loading

**Problem:** Deployment succeeds but models not loaded.

**Causes:**
- MLflow URI incorrect
- Models not in Staging stage
- Azure File Share not mounted

**Solution:**
```bash
# Check MLflow models
curl https://your-mlflow.azurewebsites.net/api/2.0/mlflow/registered-models/get?name=fraud_detection_xgboost

# Verify model stage
mlflow models list --name fraud_detection_xgboost
```

## 📚 Related Documentation

- [User Management API](USER_MANAGEMENT_API.md)
- [CI/CD Build Strategy](../Guide/CI_CD_BUILD_STRATEGY.md)
- [Build Attestations](../Guide/BUILD_ATTESTATIONS_GUIDE.md)
- [Airflow DAG05 Guide](../airflow/DAG05_CANARY_DEPLOYMENT_GUIDE.md)

## 🎓 Best Practices

### 1. Token Management
- Rotate admin tokens regularly
- Store tokens securely in Airflow Variables
- Never commit tokens to Git

### 2. Deployment Strategy
- Always monitor metrics after each canary stage
- Set appropriate thresholds for rollback
- Test deployments in staging environment first

### 3. Error Handling
- Implement retry logic in Airflow DAGs
- Log all deployment events
- Set up alerting for failed deployments

### 4. Model Versioning
- Use semantic versioning for models
- Tag models with deployment timestamps
- Keep production models in MLflow for rollback

## 📞 Support

For issues with the Deployment API:
1. Check this documentation
2. Review API logs in Azure App Service
3. Check Airflow DAG logs
4. Contact the ML Platform team

---

**Last Updated:** November 1, 2025  
**API Version:** 1.0  
**Maintainer:** ML Platform Team
