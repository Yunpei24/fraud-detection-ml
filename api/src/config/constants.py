"""
Configuration constants for the Fraud Detection API.
"""

# API Metadata
API_VERSION = "1.0.0"
API_TITLE = "Fraud Detection API"
API_DESCRIPTION = "Real-time fraud detection with ensemble ML models"

# Model Configuration
MODEL_PATHS = {
    "xgboost": "/app/models/xgboost_fraud_model.pkl",
    "random_forest": "/app/models/random_forest_fraud_model.pkl",
    "neural_network": "/app/models/nn_fraud_model.pth",
    "isolation_forest": "/app/models/isolation_forest_model.pkl",
    "shap_explainer_xgb": "/app/models/shap_explainer_xgb.pkl",
    "shap_explainer_rf": "/app/models/shap_explainer_rf.pkl",
    "shap_explainer_nn": "/app/models/shap_explainer_nn.pkl",
    "shap_explainer_iforest": "/app/models/shap_explainer_iforest.pkl",
}

# Prediction Thresholds
FRAUD_THRESHOLD = 0.5
CONFIDENCE_THRESHOLD = 0.7

# Request Limits
MAX_REQUEST_SIZE = 1048576  # 1MB
TIMEOUT_SECONDS = 30
MAX_BATCH_SIZE = 100

# Ensemble Voting Configuration (4 models)
ENSEMBLE_WEIGHTS = {
    "xgboost": 0.50,  # Best overall performance
    "random_forest": 0.30,  # Second best (your test results)
    "neural_network": 0.15,  # Captures complex patterns
    "isolation_forest": 0.05,  # Anomaly detection fallback
}

# Cache Configuration
CACHE_TTL_SECONDS = 3600  # 1 hour

# Alert settings
ALERT_THRESHOLD = 0.9
MIN_ALERTS_PER_HOUR = 10

# Validation settings
EXPECTED_FEATURES = 30  # Expected number of features in transaction
MAX_BATCH_SIZE = 100  # Maximum transactions per batch request
CACHE_KEY_PREFIX = "fraud_prediction:"

# Monitoring
METRICS_LABELS = ["model_type", "prediction_class", "endpoint"]

# Error Codes
ERROR_CODES = {
    "INVALID_INPUT": "E001",
    "MODEL_NOT_LOADED": "E002",
    "PREDICTION_FAILED": "E003",
    "DATABASE_ERROR": "E004",
    "CACHE_ERROR": "E005",
    "TIMEOUT": "E006",
    "RATE_LIMIT_EXCEEDED": "E007",
    "UNAUTHORIZED": "E008",
}

# HTTP Status Codes
HTTP_200_OK = 200
HTTP_400_BAD_REQUEST = 400
HTTP_401_UNAUTHORIZED = 401
HTTP_403_FORBIDDEN = 403
HTTP_404_NOT_FOUND = 404
HTTP_429_TOO_MANY_REQUESTS = 429
HTTP_500_INTERNAL_SERVER_ERROR = 500
HTTP_503_SERVICE_UNAVAILABLE = 503

# Rate Limiting
DEFAULT_RATE_LIMIT = 100000  # requests per minute
RATE_LIMIT_PERIOD = 90  # seconds

# Health Check
HEALTH_CHECK_DEPENDENCIES = ["database", "redis", "models", "azure_storage"]
