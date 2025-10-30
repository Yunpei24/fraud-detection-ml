"""
FastAPI dependencies for dependency injection.
"""
from typing import Any, Optional

import redis
from fastapi import Depends, Header, HTTPException, status

from ..config import get_logger, settings
from ..models import EnsembleModel
from ..services import (CacheService, DatabaseService, EvidentlyDriftService,
                        PredictionService, TrafficRouter)

logger = get_logger(__name__)

# Global instances (singleton pattern)
_model_instance: Optional[Any] = None
_prediction_service_instance: Optional[PredictionService] = None
_cache_service_instance: Optional[CacheService] = None
_database_service_instance: Optional[DatabaseService] = None
_drift_service_instance: Optional[EvidentlyDriftService] = None
_traffic_router_instance: Optional[TrafficRouter] = None


def get_traffic_router() -> Any:
    """
    Get or create the traffic router instance.

    Returns:
        Traffic router instance
    """
    global _traffic_router_instance

    if _traffic_router_instance is None:
        logger.info("Initializing traffic router")
        _traffic_router_instance = TrafficRouter()

    return _traffic_router_instance


def get_model(traffic_router: Any = Depends(get_traffic_router)) -> Any:
    """
    Get or create the ensemble model instance.

    Returns:
        Ensemble model instance
    """
    global _model_instance

    if _model_instance is None:
        logger.info("Initializing ensemble model")
        # Use champion model path if traffic routing is configured, otherwise use default
        model_path = traffic_router.champion_model_path or settings.model_path
        _model_instance = EnsembleModel(models_path=model_path)
        _model_instance.load_models()

    return _model_instance


def get_prediction_service(
    model: Any = Depends(get_model), traffic_router: Any = Depends(get_traffic_router)
) -> PredictionService:
    """
    Get or create the prediction service instance.

    Args:
        model: Ensemble model dependency
        traffic_router: Traffic router dependency

    Returns:
        Prediction service instance
    """
    global _prediction_service_instance

    if _prediction_service_instance is None:
        logger.info("Initializing prediction service")
        _prediction_service_instance = PredictionService(model, traffic_router)

    return _prediction_service_instance


def get_redis_client() -> Optional[redis.Redis]:
    """
    Get Redis client instance.

    Returns:
        Redis client or None if disabled
    """
    if not settings.enable_cache:
        return None

    try:
        client = redis.from_url(
            settings.redis_url, decode_responses=True, socket_connect_timeout=5
        )
        # Test connection
        client.ping()
        return client
    except Exception as e:
        logger.warning(f"Failed to connect to Redis: {e}")
        return None


def get_cache_service(
    redis_client: Optional[redis.Redis] = Depends(get_redis_client),
) -> CacheService:
    """
    Get or create the cache service instance.

    Args:
        redis_client: Redis client dependency

    Returns:
        Cache service instance
    """
    global _cache_service_instance

    if _cache_service_instance is None:
        logger.info("Initializing cache service")
        _cache_service_instance = CacheService(redis_client)

    return _cache_service_instance


def get_database_service() -> DatabaseService:
    """
    Get or create the database service instance.

    Returns:
        Database service instance
    """
    global _database_service_instance

    if _database_service_instance is None:
        logger.info("Initializing database service")
        _database_service_instance = DatabaseService()

        # Create tables if they don't exist
        try:
            _database_service_instance.create_tables()
        except Exception as e:
            logger.error(f"Failed to create database tables: {e}")

    return _database_service_instance


def get_drift_service(
    database_service: DatabaseService = Depends(get_database_service),
) -> EvidentlyDriftService:
    """
    Get or create the drift service instance.

    Args:
        database_service: Database service dependency

    Returns:
        Drift service instance
    """
    global _drift_service_instance

    if _drift_service_instance is None:
        logger.info("Initializing drift service")
        _drift_service_instance = EvidentlyDriftService(database_service)

    return _drift_service_instance


def verify_api_key(x_api_key: Optional[str] = Header(None)) -> str:
    """
    Verify API key from header.

    Args:
        x_api_key: API key from header

    Returns:
        Validated API key

    Raises:
        HTTPException: If API key is invalid
    """
    if not settings.require_api_key:
        return "not_required"

    if not x_api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="API key required"
        )

    # In production, validate against database or secure storage
    valid_keys = settings.api_keys.split(",") if settings.api_keys else []

    if x_api_key not in valid_keys:
        logger.warning(f"Invalid API key attempt")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid API key"
        )

    return x_api_key


async def verify_rate_limit(
    x_api_key: str = Depends(verify_api_key),
    cache_service: CacheService = Depends(get_cache_service),
) -> bool:
    """
    Verify rate limit for API key.

    Args:
        x_api_key: Validated API key
        cache_service: Cache service dependency

    Returns:
        True if within rate limit

    Raises:
        HTTPException: If rate limit exceeded
    """
    if not settings.enable_rate_limiting:
        return True

    # Rate limiting logic would go here
    # For now, just return True
    return True
