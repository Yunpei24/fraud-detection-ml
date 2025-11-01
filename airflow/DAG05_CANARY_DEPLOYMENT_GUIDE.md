# DAG 05 - Canary Deployment Implementation

## 📋 Overview

DAG 05 implements a **progressive canary deployment strategy** with automatic monitoring and rollback in case of issues.

### Deployment Strategy

```
Champion (Production) vs Challenger (Staging)
                ↓
        Comparison Pipeline
                ↓
        ┌───────────────┐
        │  Approve?     │
        └───────┬───────┘
                │
        ├─ NO ──→ Keep Champion (end)
        │
        └─ YES ─→ Deploy 5% canary
                      ↓
                  Monitor 30min
                      ↓
                  ┌─────────┐
                  │ Healthy?│
                  └────┬────┘
                       │
                  ├─ NO ──→ Rollback
                  │
                  └─ YES ─→ Deploy 25% canary
                               ↓
                           Monitor 1h
                               ↓
                           ┌─────────┐
                           │ Healthy?│
                           └────┬────┘
                                │
                           ├─ NO ──→ Rollback
                           │
                           └─ YES ─→ Deploy 100%
                                        ↓
                                    Promote to Production
```

## 🎯 Components

### 1. Airflow DAG (`05_model_deployment_canary_refactored.py`)

**Trigger:** Manual only (after successful training)

**Tasks:**

1. **compare_models** (DockerOperator)
   - Runs `comparison_pipeline.py`
   - Compares Champion vs Challenger
   - Exit code: 0 = promote, 1 = reject

2. **decide_deployment** (BranchPythonOperator)
   - Parses comparison result
   - Routes to `deploy_canary_5_percent` or `keep_champion`

3. **deploy_canary_5_percent** (DockerOperator)
   - Runs `scripts/deploy_canary.py --traffic 5`
   - Configures routing: 5% → canary, 95% → champion

4. **monitor_5_percent** (PythonOperator)
   - Monitors Prometheus for 30 minutes
   - Metrics: error_rate, latency_p95
   - Thresholds: error_rate < 5%, latency < 100ms

5. **decide_25_percent** (BranchPythonOperator)
   - Routes to `deploy_canary_25_percent` or `rollback_deployment`

6. **deploy_canary_25_percent** (DockerOperator)
   - Configures routing: 25% → canary, 75% → champion

7. **monitor_25_percent** (PythonOperator)
   - Monitors for 1 hour

8. **decide_100_percent** (BranchPythonOperator)
   - Routes to `deploy_canary_100_percent` or `rollback_deployment`

9. **deploy_canary_100_percent** (DockerOperator)
   - Runs `scripts/promote_to_production.py`
   - Staging → Production in MLflow Registry
   - Production → Archived

10. **rollback_deployment** (DockerOperator)
    - Runs `scripts/rollback_deployment.py`
    - Restores 100% traffic to champion

11. **Notifications** (PythonOperator)
    - `notify_success`: Promotion successful
    - `notify_rollback`: Rollback performed
    - `notify_rejected`: Challenger rejected

### 2. Deployment Scripts

#### `api/scripts/deploy_canary.py`

**Usage:**
```bash
python deploy_canary.py --traffic 5 \
    --model-uri "models:/fraud_detection_xgboost/Staging"
```

**Features:**
- Loads model from MLflow Staging
- Creates/updates `config/traffic_routing.json`:
  ```json
  {
    "canary_enabled": true,
    "canary_traffic_pct": 5,
    "canary_model_uri": "models:/fraud_detection_xgboost/Staging",
    "champion_traffic_pct": 95
  }
  ```
- Validates deployment
- Exit code: 0 = success, 1 = failure

#### `api/scripts/rollback_deployment.py`

**Usage:**
```bash
python rollback_deployment.py
```

**Features:**
- Disables canary routing
- Updates `config/traffic_routing.json`:
  ```json
  {
    "canary_enabled": false,
    "canary_traffic_pct": 0,
    "champion_traffic_pct": 100,
    "rollback_timestamp": "2025-01-20T10:30:00"
  }
  ```
- Logs rollback event
- Exit code: 0 = success, 1 = failure

#### `api/scripts/promote_to_production.py`

**Usage:**
```bash
python promote_to_production.py \
    --model-uri "models:/fraud_detection_xgboost/Staging"
```

**Features:**
- Loads model from Staging
- Archives old Production model
- Transitions Staging → Production in MLflow Registry
- Updates traffic routing to 100% new model
- Logs promotion event
- Exit code: 0 = success, 1 = failure

## 🔍 Monitoring

### Monitored Metrics (Prometheus)

1. **Error Rate**
   ```promql
   rate(http_requests_total{version="canary",status=~"5.."}[30m])
   ```
   - Threshold: < 5%
   - If exceeded → Rollback

2. **Latency P95**
   ```promql
   histogram_quantile(0.95,
     rate(http_request_duration_seconds_bucket{version="canary"}[30m])
   )
   ```
   - Threshold: < 100ms
   - If exceeded → Rollback

### Monitoring Durations

- **5% canary**: 30 minutes
  - Low traffic → limited risk
  - Quick validation of basic metrics

- **25% canary**: 1 hour
  - Significant traffic → need more data
  - Thorough validation before 100%

- **100% rollout**: Immediate if 25% healthy
  - All tests passed
  - Direct promotion to Production

## 🔄 Decision Flow

### Challenger Promotion

```python
# Promotion criteria (comparison_pipeline.py)
if (challenger_recall >= 0.95 and
    challenger_fpr <= 0.02 and
    challenger_latency <= 100 and
    challenger_f1 >= champion_f1 * 1.01):  # +1% improvement
    return "promote"
else:
    return "reject"
```

### Rollback Decision

```python
# Monitoring at each stage
if error_rate > 0.05:
    return "rollback"

if latency_p95 > 100:
    return "rollback"

return "promote"  # Healthy → next stage
```

## 📊 MLflow Registry Stages

### Complete Workflow

```
1. Training completed
   ↓
2. New model → Staging
   ↓
3. DAG 05 triggered (manual)
   ↓
4. Comparison: Production vs Staging
   ↓
5. If approved → Canary 5%
   ↓
6. If healthy → Canary 25%
   ↓
7. If healthy → Staging → Production
   ↓
8. Old Production → Archived
```

### MLflow Stages

- **None**: Model trained but not yet evaluated
- **Staging**: Challenger ready for canary deployment
- **Production**: Champion currently in production
- **Archived**: Old models (history)

## 🚀 Usage

### 1. Trigger DAG after training

```bash
# In Airflow UI or via CLI
airflow dags trigger 05_model_deployment_canary
```

### 2. Deployment monitoring

```bash
# Follow DAG logs
airflow tasks logs 05_model_deployment_canary monitor_5_percent <execution_date>

# Check status
airflow tasks state 05_model_deployment_canary deploy_canary_5_percent <execution_date>
```

### 3. Manual rollback if necessary

```bash
# Directly via script
docker run --rm \
  -v $(pwd)/config:/app/config \
  frauddetectionacr.azurecr.io/api:latest \
  python scripts/rollback_deployment.py
```

## 🔧 Configuration

### Environment Variables

```bash
# MLflow
MLFLOW_TRACKING_URI=http://mlflow:5000

# PostgreSQL (for metrics)
POSTGRES_HOST=postgres
POSTGRES_PORT=5432
POSTGRES_DB=fraud_detection
POSTGRES_USER=fraud_user
POSTGRES_PASSWORD=***

# Prometheus (for monitoring)
PROMETHEUS_URL=http://prometheus:9090
```

### Configuration Files

#### `config/traffic_routing.json`

Created automatically by deployment scripts:

```json
{
  "canary_enabled": true,
  "canary_traffic_pct": 25,
  "canary_model_uri": "models:/fraud_detection_xgboost/Staging",
  "champion_traffic_pct": 75,
  "production_model_uri": "models:/fraud_detection_xgboost/Production"
}
```

## ⚠️ Important Points

### Security

1. **Automatic rollback**: If metrics degrade → immediate rollback
2. **Strict validation**: Business constraints (recall >= 0.95, FPR <= 0.02)
3. **Continuous monitoring**: Prometheus scrape every 15s

### Performance

1. **Latency**: Target < 100ms P95 in production
2. **Throughput**: Canary 5% = ~500 req/min (if 10K req/min total)
3. **Resource usage**: Canary + champion running in parallel during deployment

### Data

1. **Test set**: Use representative data for comparison
2. **Class imbalance**: Check fraud/non-fraud distribution
3. **Drift**: If drift detected → retraining before deployment

## 📈 Success Metrics

### Deployment

- **Success rate**: > 80% of deployments successful
- **Rollback rate**: < 20%
- **Time to production**: < 2h (5% + 25% + 100%)

### Business

- **Recall**: Maintained >= 0.95
- **Precision**: Improved >= 1%
- **False positive rate**: <= 2%
- **Latency**: < 100ms P95

## 🔗 Integration with Other DAGs

### DAG 01 (Training)

```python
# At the end of training
if training_successful:
    # New model → Staging
    mlflow_client.transition_model_version_stage(
        name="fraud_detection_xgboost",
        version=new_version,
        stage="Staging"
    )

    # Trigger DAG 05 (optional - can be manual)
    trigger_dag("05_model_deployment_canary")
```

### DAG 02 (Drift Monitoring)

```python
# If CRITICAL drift detected
if drift_severity == "CRITICAL":
    # Trigger retraining
    trigger_dag("01_training_pipeline")

    # After retraining → trigger deployment
    # (automatic or manual according to policy)
```

## 📝 Logs and Debugging

### Important Logs

```python
# Canary deployment
[INFO] 🚀 Deploying canary with 5% traffic
[INFO] ✅ Model loaded successfully
[INFO] ✅ Traffic routing updated: 5% → canary, 95% → champion

# Monitoring
[INFO] 📊 Monitoring canary 5% for 30 minutes...
[INFO] 📈 Canary 5% metrics:
[INFO]    - Error rate: 1.23%
[INFO]    - Latency P95: 87.5ms
[INFO] ✅ Canary 5% healthy - ready to promote

# Rollback
[ERROR] ❌ Error rate too high: 6.7% > 5%
[INFO] 🔄 Rolling back to champion model...
[INFO] ✅ 100% traffic restored to champion
```

## 🎓 Best Practices

1. **Testing**: Always test in staging before production
2. **Monitoring**: Monitor Grafana dashboards during deployment
3. **Communication**: Notify team before/after deployment
4. **Documentation**: Log all model changes
5. **Rollback plan**: Always have a rollback plan ready

## 📚 References

- MLflow Model Registry: https://mlflow.org/docs/latest/model-registry.html
- Canary Deployment Pattern: https://martinfowler.com/bliki/CanaryRelease.html
- Prometheus Queries: https://prometheus.io/docs/prometheus/latest/querying/basics/
- Airflow BranchOperator: https://airflow.apache.org/docs/apache-airflow/stable/howto/operator/branch.html
