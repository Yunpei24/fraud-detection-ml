"""
Health monitoring utilities.
"""

import os
import time
from typing import Any, Dict

from ..config import get_logger

logger = get_logger(__name__)


def get_system_health() -> Dict[str, Any]:
    """
    Get system health metrics (simplified version).

    Returns:
        System health dictionary
    """
    try:
        # Basic system info without psutil
        return {
            "cpu": {"count": os.cpu_count() or 1},
            "memory": {"note": "Install psutil for detailed memory metrics"},
            "disk": {"note": "Install psutil for detailed disk metrics"},
        }
    except Exception as e:
        logger.error(f"Failed to get system health: {e}")
        return {}


def check_model_loaded(model) -> Dict[str, Any]:
    """
    Check if model is loaded and healthy.

    Args:
        model: Model instance

    Returns:
        Model health status
    """
    try:
        is_healthy = model.health_check()
        model_info = model.get_info()

        return {
            "status": "healthy" if is_healthy else "unhealthy",
            "loaded": is_healthy,
            "info": model_info,
        }
    except Exception as e:
        logger.error(f"Model health check failed: {e}")
        return {"status": "unhealthy", "loaded": False, "error": str(e)}


def check_database_connection(database_service) -> Dict[str, Any]:
    """
    Check database connection.

    Args:
        database_service: Database service instance

    Returns:
        Database health status
    """
    try:
        is_healthy = database_service.check_health()

        return {
            "status": "healthy" if is_healthy else "unhealthy",
            "connected": is_healthy,
        }
    except Exception as e:
        logger.error(f"Database health check failed: {e}")
        return {"status": "unhealthy", "connected": False, "error": str(e)}


def check_redis_connection(cache_service) -> Dict[str, Any]:
    """
    Check Redis connection.

    Args:
        cache_service: Cache service instance

    Returns:
        Redis health status
    """
    try:
        is_healthy = cache_service.check_health()

        stats = {}
        if is_healthy:
            stats = cache_service.get_stats()

        return {
            "status": "healthy" if is_healthy else "unhealthy",
            "connected": is_healthy,
            "stats": stats,
        }
    except Exception as e:
        logger.error(f"Redis health check failed: {e}")
        return {"status": "unhealthy", "connected": False, "error": str(e)}
