# 📊 DATA MODULE - Détails d'Implémentation

## 🎯 Vue d'ensemble

Le module **d---

## 📊 Flux de Données - PRODUCTION (v1.1.0)

### Real-time Pipeline

```
Azure Event Hub / Kafka Topic
    ↓ (JSON transaction events)
[EventHubConsumer / KafkaConsumer]
    ↓
[RealtimePipeline.process_event()]
    ↓
[ProductionSchemaValidator.validate_batch()]
    ├─ Check required fields (10+)
    ├─ Validate types
    ├─ Validate business rules
    └─ ✅ Valid / ❌ Invalid → Log & Skip
    ↓
[Buffer (batch_size=100 OR flush_interval_seconds=60)]
    ↓
[Transformation]
    ├─ DataCleaner.clean_pipeline()
    ├─ FeatureEngineer.engineer_features()
    └─ Create 28+ features for ML
    ↓
[Storage]
    ├─ DatabaseService (insert_transactions)
    ├─ FeatureStoreService (save_features)
    └─ DataLakeService (save_parquet)
    ↓
[Metrics]
    └─ Prometheus (transactions_processed_total, etc)
```

### Batch Pipeline

```
Data Source (Event Hub batch export, Parquet, SQL Query)
    ↓
[BatchPipeline.load_data()]
    ↓
[ProductionSchemaValidator.validate_batch()]
    └─ Rows invalid → Log & Skip
    ↓
[DataCleaner.clean_pipeline()]
    ├─ Remove duplicates
    ├─ Handle missing values
    ├─ Remove outliers
    └─ Standardize column names
    ↓
[FeatureEngineer.engineer_features()]
    ├─ Temporal features (7)
    ├─ Amount features (3)
    ├─ Customer aggregations (7)
    ├─ Merchant aggregations (6)
    └─ Interaction features (5)
    ↓
[DatabaseService.insert_transactions()]
    └─ Store with predictions in database
    ↓
[Metrics]
    └─ Report statistics to Prometheus
```

---

## ✅ Production Schema (v1.1.0)

**Requiert**: Données depuis Event Hub/Kafka avec champs PRODUCTION

```json
{
  "transaction_id": "TXN123456",
  "customer_id": "CUST001", 
  "merchant_id": "MRCH001",
  "amount": 125.50,
  "currency": "USD",
  "transaction_time": "2025-10-19T14:30:00Z",
  "customer_zip": "12345",
  "merchant_zip": "54321",
  "customer_country": "US",
  "merchant_country": "US",
  "device_id": "DEV789",
  "session_id": "SES456",
  "ip_address": "192.168.1.1",
  "mcc": 4111,
  "transaction_type": "PURCHASE",
  "is_disputed": false,
  "source_system": "mobile"
}
```

**REMARQUE IMPORTANTE**: 
- Kaggle CSV format (Time, V1-V28, Amount, Class) a été utilisé UNIQUEMENT en développement
- Tout le code Kaggle-specific a été supprimé (v1.1.0)
- Le système en PRODUCTION utilise EXCLUSIVEMENT cette structure Event Hub/** est le cœur du système de détection de fraude. Il gère:
- ✅ **Ingestion** - Réception de données depuis Event Hub/Kafka
- ✅ **Validation** - Vérification PRODUCTION du schéma Event Hub/Kafka et qualité des données
- ✅ **Transformation** - Nettoyage et ingénierie des features
- ✅ **Stockage** - Persistance en base de données et Data Lake
- ✅ **Monitoring** - Métriques et santé du système

**Important**: Le système fonctionne **EXCLUSIVEMENT** avec le schéma PRODUCTION (Event Hub/Kafka).
Le CSV Kaggle était utilisé uniquement pour comprendre la structure des données en phase de développement.
Tout le code Kaggle-specific a été supprimé (v1.1.0, October 2025).

---

## 📁 Arborescence Détaillée

```
data/
├── src/
│   ├── __init__.py
│   ├── config/
│   │   ├── __init__.py
│   │   ├── settings.py          ⚙️ Configuration centralisée
│   │   └── constants.py         📋 Constantes globales
│   │
│   ├── ingestion/              🔌 COUCHE D'INGESTION
│   │   ├── __init__.py
│   │   ├── event_hub.py        → Azure Event Hub (streaming)
│   │   └── kafka.py            → Kafka (alternative)
│   │
│   ├── validation/             ✔️ COUCHE DE VALIDATION
│   │   ├── __init__.py
│   │   ├── schema.py           → Validation du schéma transactionnel
│   │   ├── quality.py          → Contrôle qualité des données
│   │   └── anomalies.py        → Détection d'anomalies statistiques
│   │
│   ├── transformation/         🔄 COUCHE DE TRANSFORMATION
│   │   ├── __init__.py
│   │   ├── cleaner.py          → Nettoyage et prétraitement
│   │   ├── features.py         → Ingénierie des features
│   │   └── aggregator.py       → Agrégations batch
│   │
│   ├── storage/               💾 COUCHE DE STOCKAGE
│   │   ├── __init__.py
│   │   ├── database.py         → SQL Server/PostgreSQL
│   │   ├── data_lake.py        → Azure Data Lake (big data)
│   │   └── feature_store.py    → Cache Redis/Feature Store
│   │
│   ├── monitoring/            📈 COUCHE DE MONITORING
│   │   ├── __init__.py
│   │   ├── metrics.py          → Prometheus metrics
│   │   └── health.py           → Health checks
│   │
│   └── pipelines/             🚀 PIPELINES D'ORCHESTRATION
│       ├── __init__.py
│       ├── realtime_pipeline.py   → Streaming en temps réel
│       └── batch_pipeline.py      → Traitement batch
│
├── tests/
│   ├── __init__.py
│   ├── conftest.py            → Pytest fixtures
│   ├── unit/
│   │   ├── __init__.py
│   │   ├── test_schema.py      → Tests validation
│   │   ├── test_quality.py     → Tests qualité
│   │   ├── test_cleaner.py     → Tests nettoyage
│   │   └── test_features.py    → Tests features
│   └── integration/
│       ├── __init__.py
│       └── test_data_pipeline.py → Tests e2e
│
├── examples.py                📚 Exemples d'utilisation
├── requirements.txt           📦 Dépendances Python
├── schema.sql                 🗄️ Schéma base de données
└── README.md                  📖 Documentation
```

---

## 🔌 Flux de Données (Data Flow)

### Real-time Pipeline

```
Event Hub/Kafka
    ↓
  [EventHubConsumer]
    ↓
[RealtimePipeline]
    ↓
[Validation] → ✅ Valid / ❌ Invalid
    ↓
[Buffer]
    ↓ (batch_size OR flush_interval)
[Transformation]
    ├─ Clean
    └─ Features
    ↓
[Database] + [Feature Store] + [Data Lake]
```

### Batch Pipeline

```
CSV / Parquet / SQL Query
    ↓
[BatchPipeline.load_data()]
    ↓
[SchemaValidator] → Rows invalid rejetés
    ↓
[DataCleaner]
    ├─ Remove duplicates
    ├─ Handle missing values
    ├─ Remove outliers
    └─ Standardize names
    ↓
[FeatureEngineer]
    ├─ Temporal features
    ├─ Amount features
    ├─ Customer aggregations
    ├─ Merchant aggregations
    └─ Interaction features
    ↓
[DatabaseService] → Insert transactions + predictions
    ↓
[Metrics] → Prometheus
```

---

## 📦 Composants Principaux

### 1️⃣ Configuration (`config/`)

**`settings.py`**
```python
settings = Settings()
# Charge automatiquement depuis .env
# settings.azure.connection_string
# settings.database.server
# settings.cache.host
```

**`constants.py`**
```python
BATCH_SIZE = 100
MAX_RETRIES = 3
MAX_MISSING_PERCENTAGE = 0.05
VALID_CURRENCIES = ["USD", "EUR", "GBP", ...]
```

### 2️⃣ Ingestion (`ingestion/`)

**`event_hub.py`**
```python
consumer = EventHubConsumer()
consumer.connect()
consumer.start(on_event_received=process_transaction)
```

**`kafka.py`**
```python
consumer = KafkaTransactionConsumer()
consumer.start(on_message=process_transaction)
```

### 3️⃣ Validation (`validation/`)

**`schema.py`**
```python
# PRODUCTION SCHEMA ONLY
validator = SchemaValidator()
df_validated = validator.validate_batch(df, schema_type='production')

# Valide les données Event Hub/Kafka:
# - 10+ required fields (transaction_id, customer_id, merchant_id, amount, etc.)
# - Types corrects
# - Règles métier (montant >= 0, devise 3-lettres, pas d'IDs vides)
```

**`base_schema.py`**
```python
# Abstract base class for custom schemas
class BaseSchema(ABC):
    @property
    def required_fields(self) -> list: ...
    
    def validate_fields(self, df: pd.DataFrame) -> tuple[bool, List[str]]: ...
    def validate_types(self, df: pd.DataFrame) -> tuple[bool, Dict[str, str]]: ...
    def validate_business_rules(self, df: pd.DataFrame) -> tuple[bool, List[str]]: ...

# ProductionSchemaValidator extends BaseSchema
```

**`quality.py`**
```python
quality_checker = QualityValidator()
report = quality_checker.validate_batch(df)
# Vérifie: nulls, doublons, outliers, types
```

**`anomalies.py`**
```python
anomaly_detector = AnomalyDetector()
report = anomaly_detector.run_full_analysis(df)
# Détecte: colonnes manquantes, distributions anormales, cardinality haute
```

### 4️⃣ Transformation (`transformation/`)

**`cleaner.py`**
```python
cleaner = DataCleaner()
df_clean = cleaner.clean_pipeline(df,
    remove_dups=True,
    handle_missing=True,
    remove_outliers_flag=False
)
```

**`features.py`**
```python
engineer = FeatureEngineer()
df_features = engineer.engineer_features(df)
# Crée 28+ features:
# - Temporelles (7): hour, day_of_week, is_weekend, etc
# - Montant (3): log, squared, buckets
# - Client (7): count, avg, std, min, max, sum
# - Marchand (6): count, avg, std, min, max
# - Interaction (5): count, avg, std, max customer-merchant
```

**`aggregator.py`**
```python
aggregator = TransactionAggregator()
daily_agg = aggregator.aggregate_by_time(df, period="D")
customer_agg = aggregator.aggregate_by_customer(df)
merchant_agg = aggregator.aggregate_by_merchant(df)
```

### 5️⃣ Stockage (`storage/`)

**`database.py`**
```python
db = DatabaseService()
db.connect()
rows_inserted = db.insert_transactions(transactions)
stats = db.get_statistics()
```

**`data_lake.py`**
```python
datalake = DataLakeService()
datalake.save_parquet(df, "transactions/2025-10-18.parquet")
df_loaded = datalake.read_parquet("transactions/2025-10-18.parquet")
datalake.save_json_lines(records, "raw/2025-10-18.jsonl")
```

**`feature_store.py`**
```python
feature_store = FeatureStoreService(backend="redis")
feature_store.save_features("CUST001", {"total_transactions": 42})
features = feature_store.get_features("CUST001")
```

### 6️⃣ Monitoring (`monitoring/`)

**`metrics.py`**
```python
metrics = MetricsCollector()
metrics.record_transaction_processed(100)
metrics.record_ingestion_latency(0.5)
metrics.record_validation_error()
# Expose via Prometheus sur :8000/metrics
```

**`health.py`**
```python
monitor = HealthMonitor()
monitor.check_database_connection(db_service)
monitor.check_data_lake_connection(datalake)
health = monitor.get_overall_health()
```

### 7️⃣ Pipelines (`pipelines/`)

**`realtime_pipeline.py`**
```python
pipeline = RealtimePipeline(batch_size=100, flush_interval_seconds=60)
pipeline.process_event(event, validator, cleaner, db_service)
pipeline.shutdown(cleaner, db_service)
```

**`batch_pipeline.py`**
```python
pipeline = BatchPipeline()
stats = pipeline.execute(
    input_source="transactions.csv",
    validator=validator,
    cleaner=cleaner,
    feature_engineer=engineer,
    storage_service=db_service
)
```

---

## 🧪 Tests

### Structure
```
tests/
├── conftest.py               # Fixtures partagées
├── unit/                     # Tests unitaires
│   ├── test_schema.py       # Validation schema
│   ├── test_quality.py      # Qualité données
│   ├── test_cleaner.py      # Nettoyage
│   └── test_features.py     # Features
└── integration/             # Tests d'intégration
    └── test_data_pipeline.py # Pipeline end-to-end
```

### Exécution
```bash
# Tous les tests
pytest data/tests/ -v

# Tests spécifiques
pytest data/tests/unit/test_schema.py -v

# Avec couverture
pytest data/tests/ --cov=data/src --cov-report=html

# Only integration
pytest data/tests/integration/ -v
```

---

## 📊 Métriques Prometheus

```
fraud_detection_data_transactions_processed_total
fraud_detection_data_transactions_ingested_total
fraud_detection_data_validation_errors_total
fraud_detection_data_data_quality_issues_total

fraud_detection_data_ingestion_latency_seconds
fraud_detection_data_processing_latency_seconds
fraud_detection_data_validation_latency_seconds

fraud_detection_data_active_connections
fraud_detection_data_queue_size
fraud_detection_data_last_processed_timestamp
```

---

## 🔧 Configuration (.env)

```bash
# Azure
AZURE_STORAGE_CONNECTION_STRING=...
EVENT_HUB_CONNECTION_STRING=...

# Database
DB_SERVER=localhost
DB_NAME=frauddb
DB_USER=sa
DB_PASSWORD=...

# Redis
REDIS_HOST=localhost
REDIS_PORT=6379

# Monitoring
LOG_LEVEL=INFO
PROMETHEUS_PORT=8000
```

---

## 📈 Points de Performance

| Composant | Latence Cible | Throughput |
|-----------|----------------|-----------|
| Validation | < 10ms | 10K tx/sec |
| Nettoyage | < 50ms/batch | 100K rows/sec |
| Features | < 100ms/batch | 50K rows/sec |
| Stockage DB | < 500ms/batch | 1K rows/sec |
| Data Lake | < 2s/batch | 100K rows/sec |

---

## 🚀 Prochaines Étapes

Après le module `data/`, implémenter:

1. **training/** - Entraînement des modèles
2. **api/** - Serveur FastAPI pour inférence
3. **drift/** - Détection de dérive conceptuelle
4. **airflow/** - Orchestration des workflows
5. **tests/** - Suite de tests globale
6. **CI/CD** - GitHub Actions workflows

---

## 📚 Ressources

- [Pandas Docs](https://pandas.pydata.org/)
- [Azure SDK Python](https://github.com/Azure/azure-sdk-for-python)
- [SQLAlchemy](https://docs.sqlalchemy.org/)
- [Pytest](https://docs.pytest.org/)
- [Prometheus Client](https://github.com/prometheus/client_python)

---

**Module**: Data Ingestion & Processing  
**Version**: 1.1.0 (Production-Ready, Kaggle-cleanup complete)  
**Status**: ✅ Production-Ready  
**Created**: October 2025
**Last Updated**: October 19, 2025

### Derniers Changes (v1.1.0)

✅ **Suppression Kaggle-Specific Code**
- Removed 10+ fichiers avec "kaggle" dans le nom (~1,500 lignes)
- Removed src/adapters/ directory
- Removed synthetic data generation code

✅ **Implémentation ProductionSchemaValidator**
- Validates Event Hub/Kafka messages exclusivement
- 10+ required fields support
- Business rules validation
- 14 comprehensive tests (all passing)

✅ **Refactoring Complet**
- Abstract base classes (BaseSchema, BasePipeline, BaseDataLoader)
- Production-only architecture
- All 36 tests passing (100%)
- verify.py updated to use new API
