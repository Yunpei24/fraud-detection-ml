"""
Prometheus metrics collectors.
"""
from prometheus_client import Counter, Histogram, Gauge, Info

# Request metrics
request_counter = Counter(
    "fraud_detection_requests_total",
    "Total number of prediction requests",
    ["endpoint", "status"]
)

prediction_counter = Counter(
    "fraud_detection_predictions_total",
    "Total number of predictions made",
    ["prediction_label"]
)

fraud_detected_counter = Counter(
    "fraud_detection_fraud_detected_total",
    "Total number of fraud cases detected"
)

# Performance metrics
prediction_duration = Histogram(
    "fraud_detection_prediction_duration_seconds",
    "Time spent making predictions",
    buckets=[0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0]
)

batch_prediction_duration = Histogram(
    "fraud_detection_batch_prediction_duration_seconds",
    "Time spent making batch predictions",
    buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0]
)

# Model metrics
model_info = Info(
    "fraud_detection_model",
    "Information about the loaded model"
)

model_health = Gauge(
    "fraud_detection_model_health",
    "Model health status (1 = healthy, 0 = unhealthy)"
)

# Cache metrics
cache_hits = Counter(
    "fraud_detection_cache_hits_total",
    "Total number of cache hits"
)

cache_misses = Counter(
    "fraud_detection_cache_misses_total",
    "Total number of cache misses"
)

# Error metrics
error_counter = Counter(
    "fraud_detection_errors_total",
    "Total number of errors",
    ["error_type", "error_code"]
)

# Database metrics
database_queries = Counter(
    "fraud_detection_database_queries_total",
    "Total number of database queries",
    ["operation"]
)

database_query_duration = Histogram(
    "fraud_detection_database_query_duration_seconds",
    "Time spent on database queries",
    buckets=[0.001, 0.01, 0.05, 0.1, 0.5, 1.0]
)

# System metrics
active_connections = Gauge(
    "fraud_detection_active_connections",
    "Number of active connections"
)

memory_usage = Gauge(
    "fraud_detection_memory_usage_bytes",
    "Memory usage in bytes"
)
