"""
Time-series storage module for drift detection.

This module handles storing and querying time-series drift data.
"""

import pandas as pd
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
import structlog

logger = structlog.get_logger(__name__)


class TimeSeriesStorage:
    """
    Manages time-series storage for drift metrics.
    """
    
    def __init__(self):
        """Initialize time-series storage."""
        self.timeline_cache: List[Dict[str, Any]] = []
        logger.info("timeseries_storage_initialized")
    
    def store_drift_timeline(self, timeline_data: Dict[str, Any]) -> bool:
        """
        Store drift timeline data.
        
        Args:
            timeline_data: Dictionary containing timeline information
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Add timestamp if not present
            if "timestamp" not in timeline_data:
                timeline_data["timestamp"] = datetime.utcnow().isoformat()
            
            self.timeline_cache.append(timeline_data)
            
            logger.debug("timeline_data_stored", cache_size=len(self.timeline_cache))
            return True
        
        except Exception as e:
            logger.error("failed_to_store_timeline", error=str(e))
            return False
    
    def query_time_range(
        self,
        start_date: datetime,
        end_date: datetime
    ) -> pd.DataFrame:
        """
        Query time-series data for a specific time range.
        
        Args:
            start_date: Start of time range
            end_date: End of time range
            
        Returns:
            DataFrame with filtered time-series data
        """
        try:
            # Filter cached data by date range
            filtered_data = [
                data for data in self.timeline_cache
                if start_date <= datetime.fromisoformat(data["timestamp"]) <= end_date
            ]
            
            df = pd.DataFrame(filtered_data)
            
            logger.info(
                "time_range_queried",
                start=start_date.isoformat(),
                end=end_date.isoformat(),
                count=len(df)
            )
            
            return df
        
        except Exception as e:
            logger.error("failed_to_query_time_range", error=str(e))
            return pd.DataFrame()


# Convenience functions
_storage = TimeSeriesStorage()


def store_drift_timeline(timeline_data: Dict[str, Any]) -> bool:
    """Store drift timeline data."""
    return _storage.store_drift_timeline(timeline_data)


def query_time_range(start_date: datetime, end_date: datetime) -> pd.DataFrame:
    """Query time-series data."""
    return _storage.query_time_range(start_date, end_date)
