"""Configuration module for data pipeline"""

from .settings import Settings
from .constants import (
    # Batch processing
    BATCH_SIZE,
    MAX_RETRIES,
    TIMEOUT_SECONDS,
    
    # Schema & quality
    SCHEMA_VERSION,
    MAX_MISSING_PERCENTAGE,
    MAX_DUPLICATE_ROWS,
    OUTLIER_STD_THRESHOLD,
    
    # Kafka
    KAFKA_BOOTSTRAP_SERVERS,
    KAFKA_TOPIC,
    KAFKA_CONSUMER_GROUP,
    KAFKA_AUTO_OFFSET_RESET,
    KAFKA_BATCH_SIZE,
    KAFKA_TIMEOUT_MS,
    KAFKA_MAX_POLL_RECORDS,
    
    # API
    API_URL,
    API_TIMEOUT_SECONDS,
    WEBAPP_URL,
    WEBAPP_TIMEOUT_SECONDS,
)

__all__ = [
    "Settings",
    # Batch processing
    "BATCH_SIZE",
    "MAX_RETRIES",
    "TIMEOUT_SECONDS",
    # Schema & quality
    "SCHEMA_VERSION",
    "MAX_MISSING_PERCENTAGE",
    "MAX_DUPLICATE_ROWS",
    "OUTLIER_STD_THRESHOLD",
    # Kafka
    "KAFKA_BOOTSTRAP_SERVERS",
    "KAFKA_TOPIC",
    "KAFKA_CONSUMER_GROUP",
    "KAFKA_AUTO_OFFSET_RESET",
    "KAFKA_BATCH_SIZE",
    "KAFKA_TIMEOUT_MS",
    "KAFKA_MAX_POLL_RECORDS",
    # API
    "API_URL",
    "API_TIMEOUT_SECONDS",
    "WEBAPP_URL",
    "WEBAPP_TIMEOUT_SECONDS",
]
