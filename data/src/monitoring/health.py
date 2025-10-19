"""
Health monitoring for data pipeline
"""

import logging
from typing import Dict, Any, Optional
from datetime import datetime

logger = logging.getLogger(__name__)


class HealthMonitor:
    """
    Monitors health status of data pipeline components
    Tracks connectivity, performance, and error rates
    """

    def __init__(self):
        """Initialize health monitor"""
        self.component_health = {}
        self.last_check = None
        self.startup_time = datetime.utcnow()

    def check_database_connection(self, db_service) -> Dict[str, Any]:
        """
        Check database connectivity
        
        Args:
            db_service: DatabaseService instance
        
        Returns:
            Health status dictionary
        """
        try:
            stats = db_service.get_statistics()
            status = "healthy" if stats else "degraded"
            
            self.component_health["database"] = {
                "status": status,
                "last_check": datetime.utcnow().isoformat(),
                "details": stats
            }
            
            return self.component_health["database"]

        except Exception as e:
            self.component_health["database"] = {
                "status": "unhealthy",
                "last_check": datetime.utcnow().isoformat(),
                "error": str(e)
            }
            logger.error(f"Database health check failed: {str(e)}")
            return self.component_health["database"]

    def check_data_lake_connection(self, data_lake_service) -> Dict[str, Any]:
        """
        Check data lake connectivity
        
        Args:
            data_lake_service: DataLakeService instance
        
        Returns:
            Health status dictionary
        """
        try:
            files = data_lake_service.list_files()
            status = "healthy"
            
            self.component_health["data_lake"] = {
                "status": status,
                "last_check": datetime.utcnow().isoformat(),
                "files_count": len(files)
            }
            
            return self.component_health["data_lake"]

        except Exception as e:
            self.component_health["data_lake"] = {
                "status": "unhealthy",
                "last_check": datetime.utcnow().isoformat(),
                "error": str(e)
            }
            logger.error(f"Data Lake health check failed: {str(e)}")
            return self.component_health["data_lake"]

    def check_event_hub_connection(self, event_hub_service) -> Dict[str, Any]:
        """
        Check Event Hub connectivity
        
        Args:
            event_hub_service: EventHubConsumer instance
        
        Returns:
            Health status dictionary
        """
        try:
            partitions = event_hub_service.get_partition_ids()
            status = "healthy" if partitions else "degraded"
            
            self.component_health["event_hub"] = {
                "status": status,
                "last_check": datetime.utcnow().isoformat(),
                "partitions": partitions,
                "partition_count": len(partitions)
            }
            
            return self.component_health["event_hub"]

        except Exception as e:
            self.component_health["event_hub"] = {
                "status": "unhealthy",
                "last_check": datetime.utcnow().isoformat(),
                "error": str(e)
            }
            logger.error(f"Event Hub health check failed: {str(e)}")
            return self.component_health["event_hub"]

    def check_feature_store_connection(self, feature_store_service) -> Dict[str, Any]:
        """
        Check feature store connectivity
        
        Args:
            feature_store_service: FeatureStoreService instance
        
        Returns:
            Health status dictionary
        """
        try:
            stats = feature_store_service.get_statistics()
            status = "healthy" if stats else "degraded"
            
            self.component_health["feature_store"] = {
                "status": status,
                "last_check": datetime.utcnow().isoformat(),
                "stats": stats
            }
            
            return self.component_health["feature_store"]

        except Exception as e:
            self.component_health["feature_store"] = {
                "status": "unhealthy",
                "last_check": datetime.utcnow().isoformat(),
                "error": str(e)
            }
            logger.error(f"Feature Store health check failed: {str(e)}")
            return self.component_health["feature_store"]

    def get_overall_health(self) -> Dict[str, Any]:
        """
        Get overall system health
        
        Returns:
            Overall health report
        """
        if not self.component_health:
            return {
                "status": "unknown",
                "message": "No health checks performed yet"
            }

        statuses = [comp["status"] for comp in self.component_health.values()]
        
        if all(s == "healthy" for s in statuses):
            overall_status = "healthy"
        elif all(s in ["healthy", "degraded"] for s in statuses):
            overall_status = "degraded"
        else:
            overall_status = "unhealthy"

        uptime_seconds = (datetime.utcnow() - self.startup_time).total_seconds()

        return {
            "overall_status": overall_status,
            "uptime_seconds": uptime_seconds,
            "components": self.component_health,
            "last_check": datetime.utcnow().isoformat()
        }

    def is_healthy(self) -> bool:
        """Check if system is considered healthy"""
        health = self.get_overall_health()
        return health.get("overall_status") == "healthy"

    def get_degraded_components(self) -> list:
        """Get list of degraded or unhealthy components"""
        degraded = []
        for component, status in self.component_health.items():
            if status.get("status") != "healthy":
                degraded.append(component)
        return degraded
