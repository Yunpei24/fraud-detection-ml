"""
API package for fraud detection routes and dependencies.
"""
from .dependencies import (
    get_cache_service,
    get_database_service,
    get_model,
    get_prediction_service,
    verify_api_key,
)

__all__ = [
    "get_model",
    "get_prediction_service",
    "get_cache_service",
    "get_database_service",
    "verify_api_key",
]
