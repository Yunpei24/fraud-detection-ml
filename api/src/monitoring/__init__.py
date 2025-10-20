"""
Monitoring package for the Fraud Detection API.
"""
from .prometheus import (
    request_counter,
    prediction_counter,
    fraud_detected_counter,
    prediction_duration,
    batch_prediction_duration,
    model_info,
    model_health,
    cache_hits,
    cache_misses,
    error_counter,
    database_queries,
    database_query_duration,
    active_connections,
    memory_usage
)
from .health import (
    get_system_health,
    check_model_loaded,
    check_database_connection,
    check_redis_connection
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
    "check_redis_connection"
]
