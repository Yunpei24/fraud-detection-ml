"""
Prometheus metrics collectors for production monitoring.
"""
from prometheus_client import Counter, Gauge, Histogram, Info

# ============================================================================
# HTTP REQUEST METRICS
# ============================================================================

API_REQUESTS_TOTAL = Counter(
    "fraud_api_requests_total",
    "Total HTTP requests by endpoint, method and status",
    ["endpoint", "method", "status_code"],
)

API_REQUEST_DURATION_SECONDS = Histogram(
    "fraud_api_request_duration_seconds",
    "HTTP request latency in seconds",
    ["endpoint", "method"],
    buckets=[0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0],
)

API_ERRORS_TOTAL = Counter(
    "fraud_api_errors_total",
    "Total API errors by type and code",
    ["error_type", "error_code"],
)

INVALID_REQUESTS_TOTAL = Counter(
    "fraud_api_invalid_requests_total",
    "Total invalid/malformed requests",
    ["validation_error"],
)

# ============================================================================
# PREDICTION METRICS
# ============================================================================

PREDICTIONS_TOTAL = Counter(
    "fraud_predictions_total",
    "Total predictions made by model and result",
    ["model_version", "prediction_label"],
)

FRAUD_DETECTED_TOTAL = Counter(
    "fraud_detected_total",
    "Total fraud cases detected by model",
    ["model_version", "confidence_level"],
)

PREDICTION_LATENCY_SECONDS = Histogram(
    "fraud_prediction_latency_seconds",
    "Model prediction latency in seconds",
    ["model_version", "prediction_type"],
    buckets=[0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0],
)

PREDICTION_CONFIDENCE = Histogram(
    "fraud_prediction_confidence",
    "Distribution of prediction confidence scores",
    ["prediction_label"],
    buckets=[0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 0.98, 0.99, 1.0],
)

BATCH_PREDICTION_SIZE = Histogram(
    "fraud_batch_prediction_size",
    "Size of batch predictions",
    buckets=[1, 5, 10, 50, 100, 500, 1000, 5000],
)

# ============================================================================
# BUSINESS METRICS
# ============================================================================

TRANSACTION_AMOUNT_TOTAL = Counter(
    "fraud_transaction_amount_total",
    "Total transaction amount processed",
    ["prediction_label", "currency"],
)

HIGH_RISK_TRANSACTIONS_TOTAL = Counter(
    "fraud_high_risk_transactions_total",
    "High risk transactions (>0.8 fraud probability)",
    ["model_version"],
)

# ============================================================================
# MODEL METRICS
# ============================================================================

MODEL_INFO = Info("fraud_model_info", "Information about the loaded model")

MODEL_HEALTH = Gauge(
    "fraud_model_health",
    "Model health status (1=healthy, 0=unhealthy)",
    ["model_version"],
)

MODEL_LOAD_TIMESTAMP = Gauge(
    "fraud_model_load_timestamp", "Unix timestamp of model load", ["model_version"]
)

MODEL_PREDICTIONS_BY_HOUR = Gauge(
    "fraud_model_predictions_by_hour",
    "Predictions made in the last hour",
    ["model_version"],
)

# ============================================================================
# CACHE METRICS
# ============================================================================

CACHE_HITS_TOTAL = Counter("fraud_cache_hits_total", "Total cache hits", ["cache_type"])

CACHE_MISSES_TOTAL = Counter(
    "fraud_cache_misses_total", "Total cache misses", ["cache_type"]
)

CACHE_SIZE = Gauge(
    "fraud_cache_size", "Current cache size (number of items)", ["cache_type"]
)

# ============================================================================
# DATABASE METRICS
# ============================================================================

DB_QUERIES_TOTAL = Counter(
    "fraud_db_queries_total", "Total database queries", ["operation", "table"]
)

DB_QUERY_DURATION_SECONDS = Histogram(
    "fraud_db_query_duration_seconds",
    "Database query duration in seconds",
    ["operation", "table"],
    buckets=[0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0],
)

DB_ERRORS_TOTAL = Counter(
    "fraud_db_errors_total", "Total database errors", ["operation", "error_type"]
)

DB_POOL_SIZE = Gauge("fraud_db_pool_size", "Database connection pool size", ["state"])

# ============================================================================
# SYSTEM METRICS
# ============================================================================

ACTIVE_CONNECTIONS = Gauge(
    "fraud_active_connections", "Number of active HTTP connections"
)

MEMORY_USAGE_BYTES = Gauge(
    "fraud_memory_usage_bytes", "Memory usage in bytes", ["type"]
)

CPU_USAGE_PERCENT = Gauge("fraud_cpu_usage_percent", "CPU usage percentage")

# ============================================================================
# DEPENDENCY HEALTH METRICS
# ============================================================================

DEPENDENCY_HEALTH = Gauge(
    "fraud_dependency_health",
    "Dependency health status (1=healthy, 0=unhealthy)",
    ["dependency"],
)

DEPENDENCY_RESPONSE_TIME_SECONDS = Histogram(
    "fraud_dependency_response_time_seconds",
    "Dependency response time in seconds",
    ["dependency"],
    buckets=[0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0],
)

# ============================================================================
# LEGACY COMPATIBILITY (deprecated, kept for backward compatibility)
# ============================================================================

request_counter = API_REQUESTS_TOTAL
prediction_counter = PREDICTIONS_TOTAL
fraud_detected_counter = FRAUD_DETECTED_TOTAL
prediction_duration = PREDICTION_LATENCY_SECONDS
batch_prediction_duration = PREDICTION_LATENCY_SECONDS
model_info = MODEL_INFO
model_health = MODEL_HEALTH
cache_hits = CACHE_HITS_TOTAL
cache_misses = CACHE_MISSES_TOTAL
error_counter = API_ERRORS_TOTAL
database_queries = DB_QUERIES_TOTAL
database_query_duration = DB_QUERY_DURATION_SECONDS
active_connections = ACTIVE_CONNECTIONS
memory_usage = MEMORY_USAGE_BYTES
