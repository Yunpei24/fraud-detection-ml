# Fraud Detection - Airflow Orchestration

Orchestration des pipelines ML avec Apache Airflow 2.7.0

## 🎯 Vue d'ensemble

Airflow orchestre 3 pipelines critiques:
1. **DAG 01 - Training Pipeline**: Entraînement quotidien avec décision intelligente
2. **DAG 02 - Drift Monitoring**: Détection horaire de drift (Data/Target/Concept)

## 📋 Pré-requis

- Docker & Docker Compose
- PostgreSQL (port 5432 pour fraud_db)
- 4GB RAM minimum pour Airflow

## 🚀 Démarrage rapide

### 1. Configuration

```bash
cd fraud-detection-ml/airflow

# Copier le fichier .env
cp .env.example .env

# Modifier les variables (si nécessaire)
nano .env
```

### 2. Lancer Airflow

```bash
# Démarrer tous les services
docker-compose -f docker-compose.airflow.yml up -d

# Vérifier les logs
docker-compose -f docker-compose.airflow.yml logs -f airflow-scheduler
```

### 3. Accéder au Web UI

- URL: http://localhost:8080
- Username: `airflow`
- Password: `airflow`

### 4. Activer les DAGs

Dans l'UI Airflow:
1. Aller sur **DAGs**
2. Activer `02_drift_monitoring` (critique)
3. Activer `01_training_pipeline`

## 📊 Architecture des DAGs

### DAG 02: Drift Monitoring (PRIORITÉ #1)

**Schedule**: Toutes les heures (`0 * * * *`)

**Flow**:
```
run_drift_monitoring
    ↓
parse_drift_results
    ↓
decide_next_step (Branch)
    ├→ trigger_retraining_dag (si drift critique)
    ├→ send_drift_alert (si drift moyen)
    └→ no_action (si pas de drift)
    ↓
save_drift_metrics
```

**Déclenchement du retraining**:
- Concept drift détecté → Retraining IMMÉDIAT
- Data drift avec PSI > 0.5 → Retraining HIGH priority
- Data drift avec PSI > 0.3 → Retraining MEDIUM priority

### DAG 01: Training Pipeline

**Schedule**: Quotidien à 2h du matin (`0 2 * * *`)

**Décision intelligente**:
```python
# Ne retrain PAS si:
- Dernière training < 48h (cooldown)
- Nouvelles transactions < 10,000
```

**Flow**:
```
check_should_retrain (décision intelligente)
    ↓
decide_training_branch
    ├→ load_training_data → train_databricks → validate → register
    └→ skip_training
```

**Validation**:
- Recall minimum: 80%
- Precision minimum: 75%
- Si échec → Alerte + Pas de promotion

## 🔧 Configuration

### Variables d'environnement critiques

```bash
# Database fraud_db
FRAUD_DATABASE_URL=postgresql://postgres:postgres@postgres-fraud:5432/fraud_db

# MLflow tracking
MLFLOW_TRACKING_URI=http://mlflow:5000

# Databricks (training distribué)
DATABRICKS_HOST=https://your-workspace.cloud.databricks.com
DATABRICKS_TOKEN=your-token

# Thresholds
DATA_DRIFT_THRESHOLD=0.3
CONCEPT_DRIFT_THRESHOLD=0.05
MIN_RECALL_THRESHOLD=0.80
MIN_PRECISION_THRESHOLD=0.75
```

## 📁 Structure

```
airflow/
├── dags/
│   ├── 01_training_pipeline.py       # Training quotidien
│   └── 02_drift_monitoring.py        # Drift horaire (CRITIQUE)
├── plugins/
│   └── operators/
│       ├── mlflow_operator.py        # MLflow registration
│       └── alert_operator.py         # Fraud alerts
├── config/
│   └── settings.py                   # Pydantic settings
├── docker-compose.airflow.yml        # Services Airflow
├── Dockerfile                        # Image custom
└── requirements.txt                  # Dépendances
```

## 🔍 Monitoring

### Vérifier les DAGs

```bash
# Lister les DAGs
docker exec -it airflow-scheduler airflow dags list

# Tester un DAG
docker exec -it airflow-scheduler airflow dags test 02_drift_monitoring 2024-01-18
```

### Logs en temps réel

```bash
# Scheduler logs
docker-compose -f docker-compose.airflow.yml logs -f airflow-scheduler

# Webserver logs
docker-compose -f docker-compose.airflow.yml logs -f airflow-webserver
```

### Base de données

```bash
# Vérifier les métriques de drift
docker exec -it postgres-fraud psql -U postgres -d fraud_db -c "SELECT * FROM drift_metrics ORDER BY detected_at DESC LIMIT 10;"

# Vérifier les retraining triggers
docker exec -it postgres-fraud psql -U postgres -d fraud_db -c "SELECT * FROM retraining_triggers ORDER BY triggered_at DESC LIMIT 5;"
```

## 🐛 Troubleshooting

### DAG n'apparaît pas dans l'UI

```bash
# Vérifier les erreurs de parsing
docker exec -it airflow-scheduler airflow dags list-import-errors
```

### Connexion PostgreSQL échouée

```bash
# Tester la connexion depuis le scheduler
docker exec -it airflow-scheduler python -c "from airflow.config.settings import settings; print(settings.fraud_database_url)"
```

### Drift monitoring n'envoie pas d'alertes

```bash
# Vérifier les logs du task
docker-compose -f docker-compose.airflow.yml logs airflow-scheduler | grep drift_monitoring
```

## 📈 Métriques clés

### Tables PostgreSQL

1. **drift_metrics**: Métriques de drift (PSI, recall, etc.)
2. **retraining_triggers**: Historique des retranings
3. **model_versions**: Versions MLflow enregistrées
4. **airflow_task_metrics**: Performance Airflow

### Dashboards recommandés

- Grafana: Métriques Airflow (task duration, success rate)
- MLflow UI: Expériences et modèles
- Airflow UI: DAG runs, task logs

## 🔒 Sécurité

- Changer les passwords par défaut dans `.env`
- Utiliser Azure Key Vault pour DATABRICKS_TOKEN
- Activer l'authentification RBAC dans Airflow

## 🚨 Alertes

Les alertes sont envoyées par `FraudDetectionAlertOperator`:
- Email: `ALERT_EMAIL_RECIPIENTS` dans `.env`
- Slack: Configurer webhook dans `config/settings.py`

## 📚 Ressources

- [Airflow Docs](https://airflow.apache.org/docs/)
- [Databricks Provider](https://airflow.apache.org/docs/apache-airflow-providers-databricks/)
- [MLflow Model Registry](https://mlflow.org/docs/latest/model-registry.html)
