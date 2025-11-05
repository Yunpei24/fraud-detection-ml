# Fraud Detection - Airflow Orchestration

ML Pipeline orchestration with Apache Airflow 2.7.0

## 🎯 Overview

Airflow orchestrates 11 production-ready DAGs covering the complete ML lifecycle:

**Real-Time Operations**:
1. **00_transaction_producer**: Generates synthetic transactions (every 5s)
2. **00_realtime_streaming**: Real-time fraud detection pipeline (every 10s)
3. **00_token_renewal**: JWT token management (daily)
4. **00_batch_prediction**: Batch fraud prediction jobs (on-demand)

**ML Operations**:
5. **01_training_pipeline**: Model training and validation (daily 2 AM)
6. **02_drift_monitoring**: Data/target/concept drift detection (hourly)
7. **06_model_performance_tracking**: Model metrics monitoring (hourly)

**Data Quality**:
8. **03_feedback_collection**: Collect prediction feedback (daily)
9. **04_data_quality**: Data validation and quality checks (hourly)

**Deployment**:
10. **05_model_deployment_canary**: Canary deployment strategy (manual)
11. **05_model_deployment_canary_http**: HTTP-based canary deployment (manual)

## 📋 Prerequisites

- Docker & Docker Compose
- PostgreSQL (port 5432 for fraud_detection)
- Redis (port 6379 for Celery backend)
- 4GB RAM minimum for Airflow

## 🚀 Quick Start

### 1. Configuration

```bash
cd fraud-detection-ml

# Copy environment file
cp .env.example .env

# Edit variables (if necessary)
nano .env
```

### 2. Start Airflow

```bash
# Start all services (PostgreSQL, Redis, Kafka, Airflow)
docker-compose -f docker-compose.local.yml up -d

# Check logs
docker-compose -f docker-compose.local.yml logs -f airflow-scheduler
```

### 3. Access Web UI

- URL: http://localhost:8080
- Username: `airflow`
- Password: `airflow`

### 4. Activate DAGs

In Airflow UI:
1. Go to **DAGs**
2. Activate critical real-time DAGs:
   - `00_transaction_producer` (generates transactions every 5s)
   - `00_realtime_streaming` (consumes and predicts every 10s)
   - `00_token_renewal` (renews JWT tokens daily)
3. Activate monitoring DAGs:
   - `02_drift_monitoring` (critical - monitors model health hourly)
   - `04_data_quality` (validates data quality hourly)
   - `06_model_performance_tracking` (tracks model metrics hourly)
4. Training and feedback:
   - `01_training_pipeline` (daily model retraining)
   - `03_feedback_collection` (daily feedback collection)

## 📊 DAG Architecture

### 🔄 Real-Time Operations

#### DAG 00: Transaction Producer

**Schedule**: Every 5 seconds (`*/5 * * * * *`)  
**File**: `00_transaction_producer.py`

**Flow**:
```
validate_producer_config
    ↓
generate_transactions (DockerOperator)
    ↓
log_producer_completion
```

**Purpose**:
- Generate 50 simulated transactions per run
- Publish to Kafka topic `fraud-detection-transactions`
- Simulate real-world transaction patterns with ~5% fraud rate
- Uses transaction simulator with Kaggle Credit Card dataset format

#### DAG 00: Realtime Streaming

**Schedule**: Every 10 seconds (`*/10 * * * * *`)  
**File**: `00_realtime_streaming.py`

**Flow**:
```
validate_streaming_config
    ↓
run_streaming_prediction (DockerOperator)
    ↓
log_streaming_completion
```

**Purpose**:
- Consume up to 100 transactions from Kafka
- Get fraud predictions from API (JWT authenticated)
- Save raw transactions + predictions to PostgreSQL
- Dual-table strategy: `transactions` + `predictions`

**Database Schema**:
```sql
-- Raw transaction data (Kaggle format)
transactions (
  transaction_id VARCHAR(50) UNIQUE,
  time DECIMAL(10, 2),
  v1-v28 DECIMAL(10, 6),  -- PCA features
  amount DECIMAL(15, 2),
  class INTEGER,           -- 0=legitimate, 1=fraud
  source VARCHAR(50),
  ingestion_timestamp TIMESTAMPTZ
)

-- ML predictions
predictions (
  transaction_id VARCHAR(50) REFERENCES transactions(transaction_id),
  fraud_score DECIMAL(5, 4),
  is_fraud_predicted BOOLEAN,
  model_version VARCHAR(20),
  prediction_time TIMESTAMPTZ
)
```

#### DAG 00: Token Renewal

**Schedule**: Daily at midnight (`0 0 * * *`)  
**File**: `00_token_renewal.py`

**Purpose**:
- Renew JWT authentication tokens for API access
- Prevent token expiration during long-running operations
- Store refreshed tokens securely in Airflow Variables/Connections

#### DAG 00: Batch Prediction

**Schedule**: On-demand (manual trigger)  
**File**: `00_batch_prediction.py`

**Purpose**:
- Process large batches of transactions for fraud detection
- Useful for historical data analysis
- Batch scoring for compliance/audit requirements

### 🧠 ML Operations

#### DAG 01: Training Pipeline

**Schedule**: Daily at 2 AM (`0 2 * * *`)  
**File**: `01_training_pipeline.py`

**Intelligent Decision**:
```python
# Don't retrain if:
- Last training < 48h (cooldown)
- New transactions < 10,000
- Model performance still acceptable
```

**Flow**:
```
check_should_retrain (intelligent decision)
    ↓
decide_training_branch
    ├→ load_training_data → feature_engineering → train_models → validate → register_mlflow
    └→ skip_training
```

**Validation Requirements**:
- Recall minimum: 95%
- FPR maximum: 2%
- F1 Score minimum: 0.85
- If fails → Alert + Keep current model

**Models Trained**:
- XGBoost (primary)
- Neural Network
- Random Forest
- Isolation Forest
- Ensemble (weighted voting)

#### DAG 02: Drift Monitoring

**Schedule**: Every hour (`0 * * * *`)  
**File**: `02_drift_monitoring.py`

**Flow**:
```
run_drift_monitoring
    ↓
parse_drift_results
    ↓
decide_next_step (BranchOperator)
    ├→ trigger_retraining_dag (if critical drift)
    ├→ send_drift_alert (if medium drift)
    └→ no_action (if no drift)
    ↓
save_drift_metrics
```

**Drift Types Detected**:
1. **Data Drift**: Feature distribution changes (PSI, KS Test)
2. **Target Drift**: Fraud rate changes
3. **Concept Drift**: Model performance degradation

**Retraining Trigger Conditions**:
- Concept drift: Recall < 95% or FPR > 2% → IMMEDIATE
- Data drift: PSI > 0.5 → HIGH priority
- Data drift: PSI > 0.3 → MEDIUM priority

#### DAG 06: Model Performance Tracking

**Schedule**: Every hour (`0 * * * *`)  
**File**: `06_model_performance_tracking.py`

**Purpose**:
- Track model metrics over time (Recall, Precision, F1, FPR)
- Monitor inference latency and throughput
- Detect performance degradation
- Store metrics in database for trend analysis
- Generate performance reports

**Metrics Tracked**:
- Prediction accuracy
- False positive/negative rates
- Model confidence scores
- Inference time (p50, p95, p99)
- API response times

### 📊 Data Quality

#### DAG 03: Feedback Collection

**Schedule**: Daily at 3 AM (`0 3 * * *`)  
**File**: `03_feedback_collection.py`

**Purpose**:
- Collect actual fraud labels from fraud investigation team
- Update predictions with ground truth
- Build labeled dataset for model retraining
- Track model accuracy with real outcomes

**Flow**:
```
fetch_pending_cases
    ↓
collect_investigator_feedback
    ↓
update_ground_truth_labels
    ↓
store_feedback_metrics
```

#### DAG 04: Data Quality

**Schedule**: Every hour (`0 * * * *`)  
**File**: `04_data_quality.py`

**Purpose**:
- Validate transaction data schema
- Detect missing values, outliers, anomalies
- Check data freshness and completeness
- Monitor Kafka lag and data pipeline health
- Alert on data quality issues

**Quality Checks**:
- Schema validation (30 features expected)
- Missing value detection
- Outlier detection (IQR method)
- Distribution shifts
- Data volume monitoring

### 🚀 Deployment

#### DAG 05: Model Deployment (Canary)

**Schedule**: Manual trigger  
**File**: `05_model_deployment_canary.py`

**Purpose**:
- Deploy new model version with canary strategy
- Gradually increase traffic to new model (10% → 50% → 100%)
- Monitor performance during rollout
- Automatic rollback if performance degrades

**Canary Steps**:
```
validate_new_model
    ↓
deploy_canary (10% traffic)
    ↓
monitor_canary_metrics (15 min)
    ↓
evaluate_canary_performance
    ├→ increase_traffic (50%)
    │   ↓
    │   monitor_canary_metrics (15 min)
    │   ↓
    │   promote_to_production (100%)
    └→ rollback_to_previous (if issues)
```

**Rollback Triggers**:
- Recall drops > 2%
- FPR increases > 0.5%
- Error rate > 1%
- Latency > 200ms (p95)

#### DAG 05: Model Deployment (Canary HTTP)

**Schedule**: Manual trigger  
**File**: `05_model_deployment_canary_http.py`

**Purpose**:
- HTTP-based canary deployment using API endpoints
- Test new model via HTTP requests
- A/B testing with real traffic
- Shadow mode testing (parallel predictions)

**Features**:
- Health check validation
- HTTP endpoint testing
- Traffic splitting via load balancer
- Real-time metrics comparison

## 🔧 Configuration

### Critical Environment Variables

```bash
# Database - fraud_detection
DATABASE_URL=postgresql://fraud_user:fraud_pass_dev_2024@postgres:5432/fraud_detection
DB_HOST=postgres
DB_PORT=5432
DB_NAME=fraud_detection
DB_USER=fraud_user
DB_PASSWORD=fraud_pass_dev_2024

# Airflow Database - airflow
AIRFLOW__CORE__SQL_ALCHEMY_CONN=postgresql+psycopg2://airflow:fraud_pass_dev_2024@postgres:5432/airflow

# Celery (for CeleryExecutor)
AIRFLOW__CORE__EXECUTOR=CeleryExecutor
AIRFLOW__CELERY__BROKER_URL=redis://:redis_pass_change_me@redis:6379/0
AIRFLOW__CELERY__RESULT_BACKEND=db+postgresql://airflow:fraud_pass_dev_2024@postgres:5432/airflow

# MLflow tracking
MLFLOW_TRACKING_URI=http://mlflow:5000

# Kafka
KAFKA_BOOTSTRAP_SERVERS=kafka:9092
KAFKA_TOPIC=fraud-detection-transactions

# API
API_URL=http://api:8000

# Drift Thresholds
DATA_DRIFT_THRESHOLD=0.3
CONCEPT_DRIFT_THRESHOLD=0.05
MIN_RECALL_THRESHOLD=0.80
MIN_PRECISION_THRESHOLD=0.75
```

**⚠️ CRITICAL**: The `DATABASE_URL` environment variable takes priority over individual `DB_*` variables. Always set both for compatibility with different database clients (psycopg2 vs SQLAlchemy).

### Airflow Import Path

**Important**: Airflow DAGs import configuration from `/config/constants.py` (root), NOT `/airflow/config/constants.py`.

```python
# In DAG files
from config import constants  # ← Loads /config/constants.py

# Access environment variables
ENV_VARS = constants.ENV_VARS  # Dict with all env vars
```

**Service Names**: Must match `docker-compose.local.yml`:
- `postgres` (not `fraud-postgres`)
- `kafka` (not `fraud-kafka`)
- `api` (not `fraud-api`)
- `mlflow` (not `fraud-mlflow`)

## 📁 Structure

```
airflow/
├── dags/
│   ├── 00_transaction_producer.py         # Transaction generation (every 5s)
│   ├── 00_realtime_streaming.py           # Real-time predictions (every 10s)
│   ├── 00_token_renewal.py                # JWT token management (daily)
│   ├── 00_batch_prediction.py             # Batch fraud predictions (on-demand)
│   ├── 01_training_pipeline.py            # Model training (daily 2 AM)
│   ├── 02_drift_monitoring.py             # Drift detection (hourly)
│   ├── 03_feedback_collection.py          # Feedback collection (daily)
│   ├── 04_data_quality.py                 # Data validation (hourly)
│   ├── 05_model_deployment_canary.py      # Canary deployment (manual)
│   ├── 05_model_deployment_canary_http.py # HTTP canary deployment (manual)
│   └── 06_model_performance_tracking.py   # Performance monitoring (hourly)
├── plugins/
│   ├── operators/
│   │   ├── mlflow_operator.py             # MLflow integration
│   │   ├── alert_operator.py              # Alert notifications
│   │   └── docker_operator_custom.py      # Custom Docker operators
│   └── hooks/
│       ├── api_hook.py                    # API connection hook
│       └── mlflow_hook.py                 # MLflow connection hook
├── config/
│   ├── constants.py                       # Configuration constants
│   ├── settings.py                        # Pydantic settings
│   └── airflow.cfg                        # Airflow configuration
├── scripts/
│   ├── init-airflow.sh                    # Airflow initialization script
│   ├── setup_connections.py               # Setup Airflow connections
│   ├── setup_api_connection.py            # Setup API connection
│   └── setup_smtp_connection.py           # Setup SMTP for alerts
├── tests/
│   └── test_airflow_dags.py               # DAG validation tests
├── logs/                                  # Airflow logs
├── Dockerfile                             # Custom Airflow image
├── requirements.txt                       # Python dependencies
├── Makefile                               # Build automation
└── README.md                              # This file
```

## 🔍 Monitoring

### Check DAGs Status

```bash
# List all DAGs
docker exec fraud-airflow-scheduler airflow dags list

# Check specific DAG status
docker exec fraud-airflow-scheduler airflow dags list-runs -d 00_transaction_producer --state success | head -10

# Test a DAG manually
docker exec fraud-airflow-scheduler airflow dags test 00_transaction_producer 2025-11-04
```

### Unpause DAGs After Restart

```bash
# Airflow pauses DAGs after restart by default
# Real-time operations
docker exec fraud-airflow-scheduler airflow dags unpause 00_transaction_producer
docker exec fraud-airflow-scheduler airflow dags unpause 00_realtime_streaming
docker exec fraud-airflow-scheduler airflow dags unpause 00_token_renewal

# Monitoring
docker exec fraud-airflow-scheduler airflow dags unpause 02_drift_monitoring
docker exec fraud-airflow-scheduler airflow dags unpause 04_data_quality
docker exec fraud-airflow-scheduler airflow dags unpause 06_model_performance_tracking

# Training and feedback
docker exec fraud-airflow-scheduler airflow dags unpause 01_training_pipeline
docker exec fraud-airflow-scheduler airflow dags unpause 03_feedback_collection
```

### Real-Time Logs

```bash
# Scheduler logs
docker-compose -f docker-compose.local.yml logs -f airflow-scheduler

# Worker logs (CeleryExecutor)
docker-compose -f docker-compose.local.yml logs -f airflow-worker

# Webserver logs
docker-compose -f docker-compose.local.yml logs -f airflow-webserver
```

### Database Verification

```bash
# Check transactions saved by pipeline
docker exec fraud-postgres psql -U fraud_user -d fraud_detection -c "
  SELECT COUNT(*) as total, 
         MAX(ingestion_timestamp) as last_insert 
  FROM transactions;
"

# Check predictions
docker exec fraud-postgres psql -U fraud_user -d fraud_detection -c "
  SELECT COUNT(*) as total_predictions,
         SUM(CASE WHEN is_fraud_predicted THEN 1 ELSE 0 END) as frauds_detected
  FROM predictions;
"

# Check drift metrics
docker exec fraud-postgres psql -U fraud_user -d fraud_detection -c "
  SELECT * FROM drift_metrics ORDER BY detected_at DESC LIMIT 10;
"

# Check retraining triggers
docker exec fraud-postgres psql -U fraud_user -d fraud_detection -c "
  SELECT * FROM retraining_triggers ORDER BY triggered_at DESC LIMIT 5;
"
```

## 🐛 Troubleshooting

### DAG Doesn't Appear in UI

```bash
# Check parsing errors
docker exec fraud-airflow-scheduler airflow dags list-import-errors
```

### PostgreSQL Connection Failed

```bash
# Test connection from scheduler
docker exec fraud-airflow-scheduler python3 -c "
import os
print('DATABASE_URL:', os.getenv('DATABASE_URL'))
print('DB_HOST:', os.getenv('DB_HOST'))

from config import constants
print('Constants DATABASE_URL:', constants.ENV_VARS.get('DATABASE_URL'))
"

# Expected output should show "postgres" not "localhost"
```

### Airflow Tasks Connect to `localhost` Instead of `postgres`

**Root Cause**: Missing `DATABASE_URL` environment variable in DockerOperator tasks.

**Solution**: Ensure `/config/constants.py` (root) contains:
```python
ENV_VARS = {
    "DATABASE_URL": "postgresql://fraud_user:fraud_pass_dev_2024@postgres:5432/fraud_detection",
    "DB_HOST": "postgres",
    "DB_PORT": "5432",
    "DB_NAME": "fraud_detection",
    "DB_USER": "fraud_user",
    "DB_PASSWORD": "fraud_pass_dev_2024",
}
```

**Verification**:
```bash
# Check if config is correct
docker exec fraud-airflow-scheduler python3 -c "from config import constants; print(constants.ENV_VARS.get('DB_HOST'))"
# Should output: postgres (not fraud-postgres or localhost)
```

### Drift Monitoring Doesn't Send Alerts

```bash
# Check logs for drift detection
docker-compose -f docker-compose.local.yml logs airflow-scheduler | grep drift_monitoring
```

### Worker Not Picking Up Tasks (CeleryExecutor)

```bash
# Check worker status
docker logs fraud-airflow-worker --tail=50

# Check Redis connection
docker exec fraud-airflow-worker python3 -c "
import os
print('BROKER_URL:', os.getenv('AIRFLOW__CELERY__BROKER_URL'))
"

# Restart worker if needed
docker-compose -f docker-compose.local.yml restart airflow-worker
```

### DAGs Paused After Restart

**Cause**: Airflow resets DAG pause state on restart (default behavior).

**Solution**: Unpause DAGs manually or set `is_paused_upon_creation = False` in DAG definition:

```python
# In DAG file
dag = DAG(
    dag_id="00_transaction_producer",
    schedule_interval="*/5 * * * * *",
    is_paused_upon_creation=False,  # ← Auto-unpause on creation
    ...
)
```

**Quick Fix**:
```bash
# Unpause all critical DAGs
docker exec fraud-airflow-scheduler airflow dags unpause 00_transaction_producer
docker exec fraud-airflow-scheduler airflow dags unpause 00_realtime_streaming
docker exec fraud-airflow-scheduler airflow dags unpause 02_drift_monitoring
```

## 📈 Key Metrics

### PostgreSQL Tables

1. **transactions**: Raw transaction data (Kaggle format: Time, V1-V28, amount, Class)
2. **predictions**: ML predictions (fraud_score, is_fraud_predicted, model_version)
3. **drift_metrics**: Drift detection results (PSI, recall, FPR, etc.)
4. **retraining_triggers**: Retraining history
5. **model_versions**: MLflow model versions
6. **airflow_task_metrics**: Airflow task performance

### Recommended Dashboards

- **Grafana**: Airflow metrics (task duration, success rate, queue depth)
- **MLflow UI**: Experiments and model registry
- **Airflow UI**: DAG runs, task logs, task instances
- **Prometheus**: Real-time metrics from data/drift/training services

### Performance Metrics

| DAG | Schedule | Duration | Throughput | Priority |
|-----|----------|----------|------------|----------|
| 00_transaction_producer | Every 5s | ~4s | 50 txn/run = 600 txn/min | ⚡ Real-time |
| 00_realtime_streaming | Every 10s | ~2-3s | 100 txn/run = 600 txn/min | ⚡ Real-time |
| 00_token_renewal | Daily | ~2s | 1 token/day | 🔐 Security |
| 00_batch_prediction | On-demand | Varies | Batch-dependent | 📦 Batch |
| 01_training_pipeline | Daily 2 AM | ~15-30min | 1 model/day | 🧠 Training |
| 02_drift_monitoring | Every hour | ~30s | 1 analysis/hour | 🔍 Critical |
| 03_feedback_collection | Daily 3 AM | ~5min | All pending cases | 📊 Data |
| 04_data_quality | Every hour | ~1min | Full validation | ✅ Quality |
| 05_model_deployment_canary | Manual | ~30-60min | Gradual rollout | 🚀 Deploy |
| 05_model_deployment_canary_http | Manual | ~20-40min | HTTP-based rollout | 🚀 Deploy |
| 06_model_performance_tracking | Every hour | ~2min | All metrics | 📈 Monitoring |

## 🔒 Security

- Change default passwords in `.env`
- Use Azure Key Vault for `DATABRICKS_TOKEN` and `API_KEYS`
- Enable RBAC authentication in Airflow
- Rotate API keys regularly
- Use secrets backend for sensitive values

## 🚨 Alerting

Alerts are sent by `FraudDetectionAlertOperator`:
- **Email**: `ALERT_EMAIL_RECIPIENTS` in `.env`
- **Slack**: Configure webhook in `config/settings.py`

Alert Triggers:
- Concept drift detected (Recall < 95% or FPR > 2%)
- Data drift PSI > 0.5
- Target drift (Fraud rate change > 50%)
- Training failure
- DAG failure (critical tasks)

## 📚 Resources

- [Airflow Documentation](https://airflow.apache.org/docs/)
- [Databricks Provider](https://airflow.apache.org/docs/apache-airflow-providers-databricks/)
- [MLflow Model Registry](https://mlflow.org/docs/latest/model-registry.html)
- [Celery Executor](https://airflow.apache.org/docs/apache-airflow/stable/executor/celery.html)

## Production Deployment

### Docker Compose Files

- **Local Development**: `docker-compose.local.yml`
  - Uses local builds (`fraud-detection/*:local`)
  - CeleryExecutor with Redis backend
  - All services included (API, monitoring, etc.)

- **Azure VM1 Production**: `docker-compose.vm1.yml`
  - Uses Docker Hub images (`yoshua24/*:latest`)
  - CeleryExecutor with Redis backend
  - API deployed separately to Azure Web App
  - Prometheus/Grafana on VM2

### Pre-Deployment Checklist

- [ ] Update `.env` with production values
- [ ] Push all images to Docker Hub
- [ ] Test `docker-compose.vm1.yml` locally
- [ ] Verify DATABASE_URL in all services
- [ ] Unpause critical DAGs
- [ ] Configure monitoring dashboards
- [ ] Set up alert channels (email, Slack)
- [ ] Test end-to-end pipeline
- [ ] Document rollback procedure

### Deployment Commands

```bash
# On Azure VM
cd fraud-detection-ml

# Pull latest images
docker-compose -f docker-compose.vm1.yml pull

# Start services
docker-compose -f docker-compose.vm1.yml up -d

# Unpause critical DAGs
docker exec fraud-airflow-scheduler airflow dags unpause 00_transaction_producer
docker exec fraud-airflow-scheduler airflow dags unpause 00_realtime_streaming
docker exec fraud-airflow-scheduler airflow dags unpause 00_token_renewal
docker exec fraud-airflow-scheduler airflow dags unpause 02_drift_monitoring
docker exec fraud-airflow-scheduler airflow dags unpause 04_data_quality
docker exec fraud-airflow-scheduler airflow dags unpause 06_model_performance_tracking

# Monitor
docker-compose -f docker-compose.vm1.yml logs -f
```

## 📚 Additional Documentation

- **[PRODUCER_CONSUMER_SETUP.md](PRODUCER_CONSUMER_SETUP.md)**: Detailed setup for producer/consumer DAGs
- **[DAG05_CANARY_DEPLOYMENT_GUIDE.md](DAG05_CANARY_DEPLOYMENT_GUIDE.md)**: Canary deployment guide
- **[TOKEN_MANAGEMENT.md](TOKEN_MANAGEMENT.md)**: JWT token management documentation
- **[ENVIRONMENT_CONFIG.md](ENVIRONMENT_CONFIG.md)**: Environment configuration guide
- **[AIRFLOW_AZURE_CONFIG.md](AIRFLOW_AZURE_CONFIG.md)**: Azure-specific configurations

## 👨🏾‍💻 Contributors

Fraud Detection Team

1. Joshua Juste NIKIEMA
2. Olalekan Taofeek OLALUWOYE
3. Soulaimana Toihir DJALOUD