# Data Module - Fraud Detection System

Composant d'ingestion, validation, transformation et stockage de données pour le système de détection de fraude.

## 📂 Structure

```
data/
├── src/
│   ├── config/              # Configuration settings
│   │   ├── settings.py      # Settings class with environment variables
│   │   └── constants.py     # Global constants
│   ├── ingestion/           # Data intake from Event Hub/Kafka
│   │   ├── event_hub.py     # Azure Event Hub consumer
│   │   └── kafka.py         # Kafka consumer (alternative)
│   ├── validation/          # Schema & data quality validation
│   │   ├── schema.py        # Transaction schema validator
│   │   ├── quality.py       # Data quality checks
│   │   └── anomalies.py     # Anomaly detection
│   ├── transformation/      # Data cleaning & feature engineering
│   │   ├── cleaner.py       # Data cleaning pipeline
│   │   ├── features.py      # Feature engineering
│   │   └── aggregator.py    # Batch aggregations
│   ├── storage/             # Data persistence
│   │   ├── database.py      # SQL database operations
│   │   ├── data_lake.py     # Azure Data Lake storage
│   │   └── feature_store.py # Feature store (Redis/Feast)
│   ├── monitoring/          # Health & metrics
│   │   ├── metrics.py       # Prometheus metrics
│   │   └── health.py        # System health monitoring
│   └── pipelines/           # Data orchestration
│       ├── realtime_pipeline.py   # Streaming pipeline
│       └── batch_pipeline.py      # Batch processing pipeline
├── tests/
│   ├── unit/                # Unit tests
│   ├── integration/         # Integration tests
│   └── conftest.py          # Pytest fixtures
├── requirements.txt
└── README.md
```

## 🚀 Quick Start

### Installation

```bash
# 1. Install dependencies
pip install -r data/requirements.txt

# 2. Configure environment variables
cp .env.example .env
# Edit .env with your Azure/DB credentials
```

### Configuration

Créer un fichier `.env` à la racine du projet:

```bash
# Azure
ENV=development
DEBUG=true
AZURE_STORAGE_CONNECTION_STRING=DefaultEndpointsProtocol=https;...
AZURE_STORAGE_ACCOUNT=frauddetectiondl
AZURE_STORAGE_KEY=your_key
EVENT_HUB_NAME=fraud-transactions
EVENT_HUB_CONNECTION_STRING=Endpoint=sb://...

# Database
DB_DRIVER=ODBC Driver 17 for SQL Server
DB_SERVER=localhost
DB_NAME=frauddb
DB_USER=sa
DB_PASSWORD=YourPassword123!
DB_PORT=1433

# Redis/Cache
REDIS_HOST=localhost
REDIS_PORT=6379
REDIS_DB=0
REDIS_PASSWORD=

# Kafka (optional)
KAFKA_BROKERS=localhost:9092
KAFKA_TOPIC=fraud-transactions
KAFKA_GROUP_ID=fraud-detection-group

# Monitoring
LOG_LEVEL=INFO
PROMETHEUS_PORT=8000
```

## 📊 Usage Examples

### 1. Validation - Valider les transactions

```python
from data.src.validation.schema import SchemaValidator

validator = SchemaValidator()

transaction = {
    "transaction_id": "TXN123456",
    "customer_id": "CUST001",
    "merchant_id": "MRCH001",
    "amount": 150.50,
    "currency": "USD",
    "transaction_time": "2025-10-18T10:30:00"
}

is_valid, report = validator.validate(transaction)
print(f"Valid: {is_valid}")
print(f"Errors: {report['errors']}")
print(f"Warnings: {report['warnings']}")
```

### 2. Data Cleaning - Nettoyer les données

```python
import pandas as pd
from data.src.transformation.cleaner import DataCleaner

df = pd.read_csv("transactions.csv")

cleaner = DataCleaner()
df_cleaned = cleaner.clean_pipeline(
    df,
    remove_dups=True,
    handle_missing=True,
    remove_outliers_flag=False,
    standardize_names=True
)

print(f"Original: {len(df)} rows")
print(f"Cleaned: {len(df_cleaned)} rows")
```

### 3. Feature Engineering - Créer des features

```python
from data.src.transformation.features import FeatureEngineer

engineer = FeatureEngineer()
df_features = engineer.engineer_features(
    df_cleaned,
    create_temporal=True,
    create_amount=True,
    create_customer=True,
    create_merchant=True,
    create_interaction=True
)

print(f"New features created: {len(engineer.features_created)}")
print(f"Total columns: {len(df_features.columns)}")
```

### 4. Stockage - Persister les données

```python
from data.src.storage.database import DatabaseService

db_service = DatabaseService()
db_service.connect()

# Insert transactions
rows_inserted = db_service.insert_transactions(df_features.to_dict('records'))
print(f"Inserted {rows_inserted} transactions")

# Query stats
stats = db_service.get_statistics()
print(f"Total transactions: {stats['total_transactions']}")
print(f"Fraud rate: {stats['fraud_rate']:.2%}")
```

### 5. Pipeline Batch - Traitement complet

```python
from data.src.pipelines.batch_pipeline import BatchPipeline
from data.src.validation.schema import SchemaValidator
from data.src.transformation.cleaner import DataCleaner
from data.src.transformation.features import FeatureEngineer
from data.src.storage.database import DatabaseService

pipeline = BatchPipeline()
validator = SchemaValidator()
cleaner = DataCleaner()
engineer = FeatureEngineer()
db_service = DatabaseService()

# Execute pipeline
stats = pipeline.execute(
    input_source="transactions.csv",
    validator=validator,
    cleaner=cleaner,
    feature_engineer=engineer,
    storage_service=db_service
)

print(f"Pipeline execution: {stats['status']}")
print(f"Rows processed: {stats['total_rows_processed']}")
print(f"Rows stored: {stats['total_rows_stored']}")
print(f"Duration: {stats.get('duration_seconds', 0):.2f}s")
```

### 6. Real-time Pipeline - Streaming

```python
from data.src.ingestion.event_hub import EventHubConsumer
from data.src.pipelines.realtime_pipeline import RealtimePipeline
from data.src.validation.schema import SchemaValidator
from data.src.transformation.cleaner import DataCleaner
from data.src.storage.database import DatabaseService

consumer = EventHubConsumer()
pipeline = RealtimePipeline(batch_size=100, flush_interval_seconds=60)
validator = SchemaValidator()
cleaner = DataCleaner()
db_service = DatabaseService()

def process_event(event):
    pipeline.process_event(event, validator, cleaner, db_service)

# Start consuming
consumer.start(on_event_received=process_event)
```

## 🧪 Testing

```bash
# Run all tests
pytest data/tests/ -v

# Run specific test file
pytest data/tests/unit/test_schema.py -v

# Run with coverage
pytest data/tests/ --cov=data/src --cov-report=html

# Run only unit tests
pytest data/tests/unit/ -v

# Run only integration tests
pytest data/tests/integration/ -v
```

## 📈 Monitoring

### Métriques Prometheus

Les métriques sont exposées sur `http://localhost:8000/metrics`:

```
fraud_detection_data_transactions_processed_total
fraud_detection_data_transactions_ingested_total
fraud_detection_data_validation_errors_total
fraud_detection_data_ingestion_latency_seconds
fraud_detection_data_processing_latency_seconds
```

### Health Check

```python
from data.src.monitoring.health import HealthMonitor
from data.src.storage.database import DatabaseService

monitor = HealthMonitor()
db_service = DatabaseService()

health = monitor.check_database_connection(db_service)
print(f"Database status: {health['status']}")

overall = monitor.get_overall_health()
print(f"Overall system health: {overall['overall_status']}")
```

## 🔌 Intégration avec d'autres composants

### → API (Inference)
Le module data prépare les données pour que l'API puisse faire des prédictions en temps réel.

### → Training
Le module data fournit les données nettoyées et features pour l'entraînement des modèles.

### → Drift Detection
Le module data alimente le système de détection de dérive avec des statistiques de données.

### → Airflow
Les pipelines peuvent être orchestrés via Airflow DAGs.

## 📝 Notes Importantes

1. **Validation**: Toujours valider avant de transformer
2. **Nettoyage**: Adapter les seuils selon le domaine
3. **Features**: Les features temporelles nécessitent un tri par date
4. **Stockage**: Supporter les modes offline (batch) et online (cache)
5. **Monitoring**: Collecter les métriques pour détecter les anomalies

## 🔧 Dépannage

## 🐳 Docker Deployment

### Quick Start with Docker Compose

The easiest way to run the data module locally with all dependencies:

```bash
# 1. Clone environment file
cp .env.example .env

# 2. Start all services (PostgreSQL, Redis, Data Pipeline)
make up

# 3. View logs
make logs

# 4. Run tests
make test

# 5. Stop services
make down
```

### Build Docker Image

```bash
# Development build
make build

# Production build
make build-prod TAG=v1.0.0

# Or use the build script directly
./build.sh latest localhost:5000
```

### Services Included in docker-compose.yml

1. **PostgreSQL 15** - Data storage
   - Port: 5432
   - Username: fraud_user
   - Database: fraud_detection
   - Initialized with schema.sql

2. **Redis 7** - Feature cache & temporary storage
   - Port: 6379
   - Password-protected

3. **Data Pipeline** - Real-time streaming processor
   - Port: 8000
   - Healthcheck enabled
   - Auto-restart on failure

### Docker Commands with Make

```bash
# View help
make help

# Build image
make build TAG=v1.0.0

# Start services
make up

# View logs
make logs

# Stop services
make down

# Open shell in container
make shell

# Run tests
make test

# Run verification
make verify

# Clean up containers and volumes
make clean

# Check Docker status
make status
```

### Manual Docker Commands

```bash
# Build image
docker build -t fraud-detection-data:latest -f Dockerfile .

# Run container
docker run -it --rm \
  -e DATABASE_URL="postgresql://fraud_user:password@localhost:5432/fraud_detection" \
  -e REDIS_URL="redis://:password@localhost:6379" \
  fraud-detection-data:latest

# Run tests
docker-compose exec data_pipeline pytest tests/ -v

# Execute command in container
docker-compose exec data_pipeline python -m src.pipelines.realtime_pipeline
```

### Production Deployment

#### To Kubernetes

```bash
# Build and push image
make build TAG=v1.0.0
make push TAG=v1.0.0 REGISTRY=your-registry.com

# Deploy
kubectl apply -f k8s-deployment.yaml
```

#### To Cloud (Azure Container Registry)

```bash
# Login to ACR
az acr login --name your_registry

# Build and push
docker build -t your_registry.azurecr.io/fraud-detection-data:v1.0.0 .
docker push your_registry.azurecr.io/fraud-detection-data:v1.0.0

# Deploy to Azure Container Instances
az container create \
  --resource-group myResourceGroup \
  --name fraud-detection-data \
  --image your_registry.azurecr.io/fraud-detection-data:v1.0.0 \
  --environment-variables DATABASE_URL=... REDIS_URL=...
```

### Docker Image Details

- **Base Image**: python:3.10-slim (optimized)
- **Multi-stage Build**: Separates build and runtime (smaller final image)
- **Size**: ~300MB (production)
- **Security**: Non-root user (appuser)
- **Health Check**: Enabled

### Environment Variables

See `.env.example` for all available variables:

```bash
# Database
POSTGRES_PASSWORD=fraud_password_dev
DB_HOST=postgres
DB_PORT=5432

# Redis
REDIS_PASSWORD=redis_password_dev
REDIS_HOST=redis

# Logging
LOG_LEVEL=INFO

# Optional: Azure Event Hub
EVENT_HUB_CONNECTION_STRING=

# Optional: Kafka
KAFKA_BOOTSTRAP_SERVERS=kafka:9092
```

## 🔧 Troubleshooting

### Erreur: "Import azure.eventhub not found"
```bash
pip install azure-eventhub
```

### Erreur: "Connection refused" pour la base de données
- Vérifier que SQL Server est running
- Vérifier les credentials dans .env
- Vérifier la connectivité réseau

### Erreur: "Redis connection refused"
```bash
# Start Redis locally
redis-server

# Or use Docker
docker run -d -p 6379:6379 redis
```

## 📚 Ressources

- [Pandas Documentation](https://pandas.pydata.org/)
- [Azure Event Hub Python SDK](https://github.com/Azure/azure-sdk-for-python/tree/main/sdk/eventhub/azure-eventhub)
- [SQLAlchemy Documentation](https://docs.sqlalchemy.org/)
- [Pytest Documentation](https://docs.pytest.org/)

---

**Module créé**: October 2025  
**Version**: 1.0.0  
**Statut**: Production-ready ✅
