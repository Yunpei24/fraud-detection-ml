"""
Application settings for the Fraud Detection API.
Standalone configuration using environment variables.
"""
import os
from pathlib import Path

# Database settings
database_url = os.getenv("POSTGRES_HOST", "localhost")
database_port = int(os.getenv("POSTGRES_PORT", "5432"))
database_name = os.getenv("POSTGRES_DB", "fraud_detection")
database_user = os.getenv("POSTGRES_USER", "postgres")
database_password = os.getenv("POSTGRES_PASSWORD", "postgres")

# Construct database URL
database_url = f"postgresql://{database_user}:{database_password}@{database_url}:{database_port}/{database_name}"

# Redis settings
redis_host = os.getenv("REDIS_HOST", "localhost")
redis_port = int(os.getenv("REDIS_PORT", "6379"))
redis_password = os.getenv("REDIS_PASSWORD", "")
redis_url = f"redis://:{redis_password}@{redis_host}:{redis_port}/0" if redis_password else f"redis://{redis_host}:{redis_port}/0"

# Azure settings (optional)
azure_storage_connection_string = os.getenv("AZURE_STORAGE_CONNECTION_STRING", "")
azure_storage_container_name = os.getenv("AZURE_STORAGE_CONTAINER_NAME", "fraud-models")
azure_key_vault_url = os.getenv("AZURE_KEY_VAULT_URL", "")

# API-specific settings
api_host = os.getenv("API_HOST", "0.0.0.0")
api_port = int(os.getenv("API_PORT", "8000"))
workers = int(os.getenv("WORKERS", "4"))
reload = os.getenv("RELOAD", "false").lower() == "true"
api_version = os.getenv("API_VERSION", "1.0.0")
api_title = os.getenv("API_TITLE", "Fraud Detection API")
api_description = os.getenv("API_DESCRIPTION", "Real-time fraud detection API")

# Security settings
secret_key = os.getenv("SECRET_KEY", "change-me-in-production")
algorithm = os.getenv("ALGORITHM", "HS256")
access_token_expire_minutes = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "30"))
api_key_header = os.getenv("API_KEY_HEADER", "X-API-Key")
require_api_key = os.getenv("REQUIRE_API_KEY", "false").lower() == "true"
api_keys = os.getenv("API_KEYS", "")
admin_token = os.getenv("ADMIN_TOKEN", "change-me-in-production-admin-token")

# Prediction settings
fraud_threshold = float(os.getenv("FRAUD_THRESHOLD", "0.5"))
confidence_threshold = float(os.getenv("CONFIDENCE_THRESHOLD", "0.7"))

# Monitoring
log_level = os.getenv("LOG_LEVEL", "INFO")
log_format = os.getenv("LOG_FORMAT", "json")
metrics_enabled = os.getenv("METRICS_ENABLED", "true").lower() == "true"
prometheus_port = int(os.getenv("PROMETHEUS_PORT", "9090"))
enable_audit_log = os.getenv("ENABLE_AUDIT_LOG", "true").lower() == "true"

# Rate limiting
rate_limit_enabled = os.getenv("RATE_LIMIT_ENABLED", "true").lower() == "true"
rate_limit_requests = int(os.getenv("RATE_LIMIT_REQUESTS", "100"))
rate_limit_period = int(os.getenv("RATE_LIMIT_PERIOD", "60"))
enable_rate_limiting = rate_limit_enabled

# CORS
cors_origins = os.getenv("CORS_ORIGINS", "http://localhost:3000,http://localhost:8000").split(",")
cors_allow_credentials = os.getenv("CORS_ALLOW_CREDENTIALS", "true").lower() == "true"
cors_allow_methods = os.getenv("CORS_ALLOW_METHODS", "GET,POST,PUT,DELETE,OPTIONS").split(",")
cors_allow_headers = os.getenv("CORS_ALLOW_HEADERS", "*").split(",")

# Feature flags
enable_batch_prediction = os.getenv("ENABLE_BATCH_PREDICTION", "true").lower() == "true"
enable_shap_explanation = os.getenv("ENABLE_SHAP_EXPLANATION", "true").lower() == "true"
enable_model_reload = os.getenv("ENABLE_MODEL_RELOAD", "true").lower() == "true"
enable_cache = os.getenv("ENABLE_CACHE", "true").lower() == "true"

# Model configuration
model_path = os.getenv("MODEL_PATH", os.getenv("AZURE_STORAGE_MOUNT_PATH", "/mnt/fraud-models"))
model_version = os.getenv("MODEL_VERSION", "v1.0.0")
xgboost_model_name = os.getenv("XGBOOST_MODEL_NAME", "xgboost_fraud_model.pkl")
random_forest_model_name = os.getenv("RANDOM_FOREST_MODEL_NAME", "random_forest_fraud_model.pkl")
nn_model_name = os.getenv("NN_MODEL_NAME", "nn_fraud_model.pth")
isolation_forest_model_name = os.getenv("ISOLATION_FOREST_MODEL_NAME", "isolation_forest_model.pkl")

# SHAP explainers
shap_explainer_xgb_name = os.getenv("SHAP_EXPLAINER_XGB_NAME", "shap_explainer_xgb.pkl")
shap_explainer_rf_name = os.getenv("SHAP_EXPLAINER_RF_NAME", "shap_explainer_rf.pkl")
shap_explainer_nn_name = os.getenv("SHAP_EXPLAINER_NN_NAME", "shap_explainer_nn.pkl")
shap_explainer_iforest_name = os.getenv("SHAP_EXPLAINER_IFOREST_NAME", "shap_explainer_iforest.pkl")

# Environment
environment = os.getenv("ENVIRONMENT", "development")
debug = os.getenv("DEBUG", "false").lower() == "true"

# Traffic routing
traffic_routing_config = os.getenv("TRAFFIC_ROUTING_CONFIG", "/app/config/traffic_routing.json")


class Monitoring:
    def __init__(self):
        self.log_level = log_level
        self.log_format = log_format
        self.enable_audit_log = enable_audit_log


class API:
    def __init__(self):
        self.host = api_host
        self.port = api_port
        self.cors_origins = cors_origins
        self.cors_allow_credentials = cors_allow_credentials
        self.cors_allow_methods = cors_allow_methods
        self.cors_allow_headers = cors_allow_headers


class Auth:
    def __init__(self):
        self.secret_key = secret_key
        self.algorithm = algorithm
        self.access_token_expire_minutes = access_token_expire_minutes
        self.api_key_header = api_key_header
        self.require_api_key = require_api_key
        self.api_keys = api_keys
        self.admin_token = admin_token


class Settings:
    def __init__(self):
        self.model_path = model_path
        self.model_version = model_version
        self.xgboost_model_name = xgboost_model_name
        self.random_forest_model_name = random_forest_model_name
        self.nn_model_name = nn_model_name
        self.isolation_forest_model_name = isolation_forest_model_name
        self.shap_explainer_xgb_name = shap_explainer_xgb_name
        self.shap_explainer_rf_name = shap_explainer_rf_name
        self.shap_explainer_nn_name = shap_explainer_nn_name
        self.shap_explainer_iforest_name = shap_explainer_iforest_name
        self.fraud_threshold = fraud_threshold
        self.enable_shap_explanation = enable_shap_explanation
        self.monitoring = Monitoring()
        self.environment = environment
        self.api = API()
        self.auth = Auth()
        self.log_level = log_level
        
        # Database and Redis
        self.database_url = database_url
        self.redis_url = redis_url
        
        # Security
        self.require_api_key = require_api_key
        self.api_keys = api_keys
        self.secret_key = secret_key
        self.admin_token = admin_token
        
        # Rate limiting
        self.enable_rate_limiting = enable_rate_limiting
        self.rate_limit_requests = rate_limit_requests
        self.rate_limit_period = rate_limit_period
        
        # Caching
        self.enable_cache = enable_cache

        # Traffic routing
        self.traffic_routing_config = traffic_routing_config

        # CORS
        self.cors_origins = cors_origins


settings = Settings()
