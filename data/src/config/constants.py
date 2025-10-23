"""
Global constants for the data pipeline
"""

# Batch processing
BATCH_SIZE = 100
MAX_RETRIES = 3
TIMEOUT_SECONDS = 30

# Schema versioning
SCHEMA_VERSION = "1.0"

# Data quality thresholds
MAX_MISSING_PERCENTAGE = 0.05  # 5% max missing values
MAX_DUPLICATE_ROWS = 0.01     # 1% max duplicates
OUTLIER_STD_THRESHOLD = 5      # 5 standard deviations

# Feature engineering
TEMPORAL_WINDOW_HOURS = 24
CUSTOMER_AGGREGATION_DAYS = 30
MERCHANT_AGGREGATION_DAYS = 30

# Data validation
MIN_TRANSACTION_AMOUNT = 0.01
MAX_TRANSACTION_AMOUNT = 1000000.00
VALID_CURRENCIES = ["USD", "EUR", "GBP", "CAD", "AUD", "JPY"]

# Storage & Paths
DATA_RETENTION_DAYS = 365
CHECKPOINT_INTERVAL_HOURS = 1
LOG_LEVEL = "INFO"

# Azure configurations
AZURE_STORAGE_ACCOUNT = "frauddetectiondl"
AZURE_CONTAINER_NAME = "transactions"
FEATURE_STORE_CONTAINER = "features"

# Database
DB_POOL_SIZE = 20
DB_MAX_OVERFLOW = 40
DB_POOL_RECYCLE = 3600

# Monitoring
METRICS_FLUSH_INTERVAL_SECONDS = 60
ALERT_THRESHOLD_ERROR_RATE = 0.05  # 5% error rate

# Feature flags
ENABLE_CACHE = True
ENABLE_MONITORING = True
ENABLE_DATA_PROFILING = True
