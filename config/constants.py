"""
Centralized Configuration Constants for Fraud Detection Airflow DAGs
All table names, thresholds, and environment variables are defined here
to eliminate hardcoded values across the codebase.
"""

import os

# ============================================================================
# DATABASE TABLE NAMES
# ============================================================================
TABLE_NAMES = {
    # Core transaction tables
    "USERS": "users",
    "TRAINING_TRANSACTIONS": "training_transactions",
    "TRANSACTIONS": "transactions",
    "PREDICTIONS": "predictions",
    # Feature tables
    "CUSTOMER_FEATURES": "customer_features",
    "MERCHANT_FEATURES": "merchant_features",
    # Monitoring and logging tables
    "DRIFT_METRICS": "drift_metrics",
    "MODEL_VERSIONS": "model_versions",
    "FEEDBACK_LABELS": "feedback_labels",
    "RETRAINING_TRIGGERS": "retraining_triggers",
    "PIPELINE_EXECUTION_LOG": "pipeline_execution_log",
    "DATA_QUALITY_LOG": "data_quality_log",
    "AIRFLOW_TASK_METRICS": "airflow_task_metrics",
}


# ============================================================================
# DOCKER AND ENVIRONMENT VARIABLES
# ============================================================================

# ============================================================================
# ENVIRONMENT CONFIGURATION
# ============================================================================

# Environment detection for Docker images
# Set ENVIRONMENT=production to use Docker Hub
# Set ENVIRONMENT=development (default) to use local Docker images
ENVIRONMENT = os.getenv(
    "ENVIRONMENT", "development"
)  # 'development', 'staging', 'production', or 'test'
DOCKERHUB_USERNAME = os.getenv("DOCKERHUB_USERNAME", "josh24")

# Docker images based on environment
if ENVIRONMENT == "production":
    DOCKER_IMAGES = {
        "TRAINING": f"{DOCKERHUB_USERNAME}/training:latest",
        "API": f"{DOCKERHUB_USERNAME}/api:latest",
        "DATA": f"{DOCKERHUB_USERNAME}/data:latest",
        "DRIFT": f"{DOCKERHUB_USERNAME}/drift:latest",
    }
elif (
    ENVIRONMENT == "development"
):  # local development (using Docker Hub images with develop tag)
    DOCKER_IMAGES = {
        "TRAINING": f"{DOCKERHUB_USERNAME}/training:develop",
        "API": f"{DOCKERHUB_USERNAME}/api:develop",
        "DATA": f"{DOCKERHUB_USERNAME}/data:develop",
        "DRIFT": f"{DOCKERHUB_USERNAME}/drift:develop",
    }
else:  # test or staging
    DOCKER_IMAGES = {
        "TRAINING": "fraud-detection/training:local",
        "API": "fraud-detection/api:local",
        "DATA": "fraud-detection/data:local",
        "DRIFT": "fraud-detection/drift:local",
    }

ENV_VARS = {
    # Docker configuration
    "DOCKER_NETWORK": "fraud-detection-network",
    "DOCKER_COMPOSE_PROJECT": "fraud-detection-ml",
    "DOCKER_IMAGE_TRAINING": DOCKER_IMAGES["TRAINING"],
    "DOCKER_IMAGE_API": DOCKER_IMAGES["API"],
    "DOCKER_IMAGE_DATA": DOCKER_IMAGES["DATA"],
    "DOCKER_IMAGE_DRIFT": DOCKER_IMAGES["DRIFT"],
    # Python configuration
    "PYTHONPATH": "/app",
    "PYTHONUNBUFFERED": "1",
    # PostgreSQL configuration
    "POSTGRES_HOST": "postgres",
    "POSTGRES_PORT": "5432",
    "POSTGRES_DB": "fraud_detection",
    "POSTGRES_USER": "fraud_user",
    "POSTGRES_PASSWORD": "fraud_pass_dev_2024",
    "POSTGRES_CONN_ID": "fraud_postgres",
    # MLflow configuration
    "MLFLOW_TRACKING_URI": "http://mlflow:5000",
    "MLFLOW_EXPERIMENT_NAME": "fraud-detection",
    # Prometheus metrics
    "PROMETHEUS_PORT_API": "9090",
    "PROMETHEUS_PORT_DRIFT": "9092",
    "PROMETHEUS_PORT_TRAINING": "9093",
    # Airflow API configuration (for drift trigger)
    "AIRFLOW_API_URL": "http://airflow-webserver:8080/api/v1",
    "AIRFLOW_USERNAME": "airflow",
    "AIRFLOW_PASSWORD": "airflow",
    # Model registry
    "MODEL_REGISTRY_PATH": "/app/models",
    "MODEL_ARTIFACT_PATH": "s3://fraud-models",
    # Kafka configuration
    "KAFKA_BOOTSTRAP_SERVERS": os.getenv("KAFKA_BOOTSTRAP_SERVERS", "kafka:9092"),
    "KAFKA_TOPIC": os.getenv("KAFKA_TOPIC", "fraud-detection-transactions"),
    "KAFKA_CONSUMER_GROUP": os.getenv("KAFKA_CONSUMER_GROUP", "fraud-detection-batch"),
    "KAFKA_AUTO_OFFSET_RESET": os.getenv("KAFKA_AUTO_OFFSET_RESET", "earliest"),
    "KAFKA_TIMEOUT_MS": os.getenv("KAFKA_TIMEOUT_MS", "60000"),
    "KAFKA_MAX_POLL_RECORDS": os.getenv("KAFKA_MAX_POLL_RECORDS", "500"),
    # API configuration
    "API_URL": os.getenv("API_URL", "http://api:8000"),
    "API_TIMEOUT_SECONDS": os.getenv("API_TIMEOUT_SECONDS", "60"),
    # JWT Authentication (from environment or Airflow Variables)
    "API_USERNAME": os.getenv("API_USERNAME", "admin"),
    "API_PASSWORD": os.getenv("API_PASSWORD", "admin123"),
    # Web Application for fraud alerts
    "WEBAPP_URL": os.getenv("WEBAPP_URL", "http://localhost:3001"),
    "WEBAPP_TIMEOUT_SECONDS": os.getenv("WEBAPP_TIMEOUT_SECONDS", "30"),
}


# ============================================================================
# MODEL PERFORMANCE THRESHOLDS
# ============================================================================
THRESHOLDS = {
    # Minimum acceptable model performance
    "MIN_RECALL": 0.80,
    "MIN_PRECISION": 0.75,
    "MIN_F1": 0.77,
    "MIN_AUC": 0.85,
    # Performance degradation alert threshold
    "PERFORMANCE_DEGRADATION_THRESHOLD": 0.05,  # 5% drop triggers alert
    # Drift detection thresholds
    "PSI_THRESHOLD": 0.2,  # Population Stability Index
    "KS_THRESHOLD": 0.1,  # Kolmogorov-Smirnov test
    "CONCEPT_DRIFT_THRESHOLD": 0.05,  # Concept drift p-value
    # Data quality thresholds
    "MAX_MISSING_PERCENTAGE": 0.05,  # 5% max missing values
    "MAX_OUTLIER_PERCENTAGE": 0.02,  # 2% max outliers
    "MIN_ROW_COUNT": 1000,  # Minimum rows per day
    # Training data requirements
    "MIN_TRAINING_SAMPLES": 10000,
    "MIN_FRAUD_SAMPLES": 1000,
    "MAX_CLASS_IMBALANCE_RATIO": 0.01,  # 1% minimum fraud rate
}


# ============================================================================
# DAG SCHEDULES (Cron expressions)
# ============================================================================
SCHEDULES = {
    "TRAINING_PIPELINE": None,  # Triggered manually or by drift
    "DRIFT_MONITORING": "0 * * * *",  # Every hour
    "FEEDBACK_COLLECTION": "0 4 * * *",  # Daily at 4 AM
    "DATA_QUALITY": "0 2 * * *",  # Daily at 2 AM
    "MODEL_DEPLOYMENT": None,  # Triggered after training
    "PERFORMANCE_TRACKING": "0 3 * * *",  # Daily at 3 AM
}


# ============================================================================
# DRIFT DETECTION CONFIGURATION
# ============================================================================
DRIFT_CONFIG = {
    # Drift severity levels
    "SEVERITY_LOW": "LOW",
    "SEVERITY_MEDIUM": "MEDIUM",
    "SEVERITY_HIGH": "HIGH",
    "SEVERITY_CRITICAL": "CRITICAL",
    # Drift metric types
    "METRIC_PSI": "psi",
    "METRIC_KS": "ks_test",
    "METRIC_CONCEPT": "concept_drift",
    # Retraining cooldown
    "RETRAINING_COOLDOWN_HOURS": 48,
    # Drift priority mapping
    "PRIORITY_LOW": 1,
    "PRIORITY_MEDIUM": 5,
    "PRIORITY_HIGH": 10,
}


# ============================================================================
# ALERT CONFIGURATION
# ============================================================================
ALERT_CONFIG = {
    # Alert severity levels
    "SEVERITY_INFO": "INFO",
    "SEVERITY_WARNING": "WARNING",
    "SEVERITY_CRITICAL": "CRITICAL",
    # Alert channels
    "CHANNEL_LOG": "log",
    "CHANNEL_EMAIL": "email",
    "CHANNEL_SLACK": "slack",
    "CHANNEL_PAGERDUTY": "pagerduty",
    # Default recipients
    "DEFAULT_EMAIL": "ml-alerts@frauddetection.com",
    "DEFAULT_SLACK_CHANNEL": "#ml-alerts",
}


# ============================================================================
# MODEL TRAINING CONFIGURATION
# ============================================================================
TRAINING_CONFIG = {
    # Model types
    "MODELS": ["xgboost", "random_forest", "neural_network", "isolation_forest"],
    # Training parameters
    "TEST_SIZE": 0.2,
    "RANDOM_STATE": 42,
    "CV_FOLDS": 5,
    # Parallel training
    "N_JOBS": -1,  # Use all available cores
    # Model selection criteria
    "PRIMARY_METRIC": "recall",
    "SECONDARY_METRIC": "f1",
}


# ============================================================================
# DEPLOYMENT CONFIGURATION
# ============================================================================
DEPLOYMENT_CONFIG = {
    # Deployment strategies
    "STRATEGY_CANARY": "canary",
    "STRATEGY_BLUE_GREEN": "blue_green",
    "STRATEGY_IMMEDIATE": "immediate",
    # Canary deployment
    "CANARY_PERCENTAGE": 0.1,  # 10% traffic to new model
    "CANARY_DURATION_HOURS": 24,
    # Deployment status
    "STATUS_PENDING": "pending",
    "STATUS_IN_PROGRESS": "in_progress",
    "STATUS_ACTIVE": "active",
    "STATUS_ROLLED_BACK": "rolled_back",
    "STATUS_FAILED": "failed",
}


# ============================================================================
# DOCKER NETWORK CONSTANT
# ============================================================================
DOCKER_NETWORK = ENV_VARS["DOCKER_NETWORK"]

# Docker images (environment-aware)
DOCKER_IMAGE_TRAINING = ENV_VARS["DOCKER_IMAGE_TRAINING"]
DOCKER_IMAGE_API = ENV_VARS["DOCKER_IMAGE_API"]
DOCKER_IMAGE_DATA = ENV_VARS["DOCKER_IMAGE_DATA"]
DOCKER_IMAGE_DRIFT = ENV_VARS["DOCKER_IMAGE_DRIFT"]

# Environment info
CURRENT_ENVIRONMENT = ENVIRONMENT
DOCKERHUB_REGISTRY = DOCKERHUB_USERNAME
