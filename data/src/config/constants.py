"""
Global constants for the data pipeline

These constants provide default values that can be overridden by environment variables.
For production deployments, use environment variables for sensitive/environment-specific configs.
"""

import os

# ==============================================================================
# BATCH PROCESSING
# ==============================================================================
BATCH_SIZE = 100
MAX_RETRIES = 3
TIMEOUT_SECONDS = 30

# ==============================================================================
# SCHEMA & DATA QUALITY
# ==============================================================================
SCHEMA_VERSION = "1.0"

# Data quality thresholds
MAX_MISSING_PERCENTAGE = 0.05  # 5% max missing values
MAX_DUPLICATE_ROWS = 0.01      # 1% max duplicates
OUTLIER_STD_THRESHOLD = 5      # 5 standard deviations

# ==============================================================================
# KAFKA CONFIGURATION
# ==============================================================================
# Default Kafka settings (can be overridden by environment variables)
KAFKA_BOOTSTRAP_SERVERS = os.getenv('KAFKA_BOOTSTRAP_SERVERS', 'kafka:9092')
KAFKA_TOPIC = os.getenv('KAFKA_TOPIC', 'fraud-detection-transactions')
KAFKA_CONSUMER_GROUP = os.getenv('KAFKA_CONSUMER_GROUP', 'fraud-detection-pipeline')
KAFKA_AUTO_OFFSET_RESET = os.getenv('KAFKA_AUTO_OFFSET_RESET', 'earliest')
KAFKA_BATCH_SIZE = int(os.getenv('KAFKA_BATCH_SIZE', '1000'))
KAFKA_TIMEOUT_MS = int(os.getenv('KAFKA_TIMEOUT_MS', '60000'))
KAFKA_MAX_POLL_RECORDS = int(os.getenv('KAFKA_MAX_POLL_RECORDS', '500'))

# ==============================================================================
# API CONFIGURATION
# ==============================================================================
# Fraud Detection API endpoint
API_URL = os.getenv('API_URL', 'http://api:8000')
API_TIMEOUT_SECONDS = int(os.getenv('API_TIMEOUT_SECONDS', '60'))

# Web Application endpoint (for sending fraud alerts)
WEBAPP_URL = os.getenv('WEBAPP_URL', None)
WEBAPP_TIMEOUT_SECONDS = int(os.getenv('WEBAPP_TIMEOUT_SECONDS', '30'))
