"""
Health check API routes.
"""

import time
from datetime import datetime

from fastapi import APIRouter, Depends, status

from ...api.dependencies import (
    get_cache_service,
    get_database_service,
    get_prediction_service,
)
from ...config import constants, get_logger, settings
from ...models import DetailedHealthCheckResponse, HealthCheckResponse
from ...services import CacheService, DatabaseService, PredictionService

logger = get_logger(__name__)

router = APIRouter(tags=["health"])

# Store startup time
startup_time = time.time()


@router.get(
    "/health",
    response_model=HealthCheckResponse,
    status_code=status.HTTP_200_OK,
    summary="Basic health check",
    description="Check if the API is running",
)
async def health_check():
    """
    Basic health check endpoint.

    Returns:
        Health status
    """
    uptime = time.time() - startup_time

    return HealthCheckResponse(
        status="healthy",
        timestamp=datetime.utcnow(),
        version=constants.API_VERSION,
        uptime_seconds=uptime,
    )


@router.get(
    "/health/detailed",
    response_model=DetailedHealthCheckResponse,
    status_code=status.HTTP_200_OK,
    summary="Detailed health check",
    description="Check health of all components",
)
async def detailed_health_check(
    prediction_service: PredictionService = Depends(get_prediction_service),
    cache_service: CacheService = Depends(get_cache_service),
    database_service: DatabaseService = Depends(get_database_service),
):
    """
    Detailed health check for all components.

    Args:
        prediction_service: Prediction service dependency
        cache_service: Cache service dependency
        database_service: Database service dependency

    Returns:
        Detailed health status
    """
    uptime = time.time() - startup_time

    # Check model health
    model_healthy = False
    model_info = {}
    try:
        model_healthy = prediction_service.check_model_health()
        model_info = prediction_service.get_model_info()
    except Exception as e:
        logger.error(f"Model health check failed: {e}")

    # Check cache health
    cache_healthy = False
    cache_stats = {}
    try:
        cache_healthy = cache_service.check_health()
        if cache_healthy:
            cache_stats = cache_service.get_stats()
    except Exception as e:
        logger.error(f"Cache health check failed: {e}")

    # Check database health
    database_healthy = False
    try:
        database_healthy = database_service.check_health()
    except Exception as e:
        logger.error(f"Database health check failed: {e}")

    # Determine overall status
    all_healthy = model_healthy and cache_healthy and database_healthy
    any_healthy = model_healthy or cache_healthy or database_healthy

    if all_healthy:
        overall_status = "healthy"
    elif any_healthy:
        overall_status = "degraded"
    else:
        overall_status = "unhealthy"

    return DetailedHealthCheckResponse(
        status=overall_status,
        timestamp=datetime.utcnow(),
        version=constants.API_VERSION,
        uptime_seconds=uptime,
        components={
            "model": {
                "status": "healthy" if model_healthy else "unhealthy",
                "details": model_info,
            },
            "cache": {
                "status": "healthy" if cache_healthy else "unhealthy",
                "details": cache_stats,
            },
            "database": {"status": "healthy" if database_healthy else "unhealthy"},
        },
        environment=settings.environment,
    )


@router.get(
    "/ready",
    status_code=status.HTTP_200_OK,
    summary="Readiness check",
    description="Check if the API is ready to serve requests",
)
async def readiness_check(
    prediction_service: PredictionService = Depends(get_prediction_service),
):
    """
    Kubernetes readiness probe endpoint.

    Args:
        prediction_service: Prediction service dependency

    Returns:
        Ready status
    """
    try:
        # Check if model is loaded
        model_healthy = prediction_service.check_model_health()

        if model_healthy:
            return {"status": "ready"}
        else:
            return {"status": "not_ready", "reason": "model_not_loaded"}

    except Exception as e:
        logger.error(f"Readiness check failed: {e}")
        return {"status": "not_ready", "reason": str(e)}


@router.get(
    "/live",
    status_code=status.HTTP_200_OK,
    summary="Liveness check",
    description="Check if the API is alive",
)
async def liveness_check():
    """
    Kubernetes liveness probe endpoint.

    Returns:
        Alive status
    """
    return {"status": "alive", "timestamp": datetime.utcnow().isoformat()}
