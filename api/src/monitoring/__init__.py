"""
Monitoring package for the Fraud Detection API.
"""
from .health import (
    check_database_connection,
    check_model_loaded,
    check_redis_connection,
    get_system_health,
)
from .prometheus import (
    active_connections,
    batch_prediction_duration,
    cache_hits,
    cache_misses,
    database_queries,
    database_query_duration,
    error_counter,
    fraud_detected_counter,
    memory_usage,
    model_health,
    model_info,
    prediction_counter,
    prediction_duration,
    request_counter,
)

__all__ = [
    # Prometheus metrics
    "request_counter",
    "prediction_counter",
    "fraud_detected_counter",
    "prediction_duration",
    "batch_prediction_duration",
    "model_info",
    "model_health",
    "cache_hits",
    "cache_misses",
    "error_counter",
    "database_queries",
    "database_query_duration",
    "active_connections",
    "memory_usage",
    # Health checks
    "get_system_health",
    "check_model_loaded",
    "check_database_connection",
    "check_redis_connection",
]
