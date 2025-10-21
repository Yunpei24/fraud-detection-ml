"""
Metrics API routes for Prometheus scraping.
"""
import psutil
from fastapi import APIRouter, Response
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST

from ...config import get_logger
from ...monitoring.prometheus import (
    MEMORY_USAGE_BYTES,
    CPU_USAGE_PERCENT
)

logger = get_logger(__name__)

router = APIRouter(tags=["metrics"])


@router.get(
    "/metrics",
    summary="Prometheus metrics endpoint",
    description="Expose all Prometheus metrics for scraping",
    response_class=Response
)
async def get_metrics():
    """
    Expose Prometheus metrics in text format for Prometheus server scraping.
    
    This endpoint updates system metrics before returning all metrics:
    - Memory usage (RSS, VMS)
    - CPU usage percentage
    - Active connections
    
    Returns:
        Response: Prometheus text format metrics
    """
    try:
        # Update system metrics before exposing
        process = psutil.Process()
        memory_info = process.memory_info()
        
        # Memory metrics
        MEMORY_USAGE_BYTES.labels(type="rss").set(memory_info.rss)
        MEMORY_USAGE_BYTES.labels(type="vms").set(memory_info.vms)
        
        # CPU metrics
        cpu_percent = process.cpu_percent(interval=0.1)
        CPU_USAGE_PERCENT.set(cpu_percent)
        
        # Generate metrics
        metrics_data = generate_latest()
        return Response(
            content=metrics_data,
            media_type=CONTENT_TYPE_LATEST
        )
    except Exception as e:
        logger.error(f"Failed to generate metrics: {e}", exc_info=True)
        return Response(
            content=f"# Error generating metrics: {str(e)}",
            media_type="text/plain",
            status_code=500
        )
