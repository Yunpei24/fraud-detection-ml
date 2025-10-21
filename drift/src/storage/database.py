"""
Database storage module for drift detection.

This module handles persisting drift metrics and retrieving historical data
from PostgreSQL database.
"""

import pandas as pd
from sqlalchemy import create_engine, Column, Integer, Float, String, DateTime, Boolean, JSON
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
import structlog

from ..config.settings import Settings

logger = structlog.get_logger(__name__)

Base = declarative_base()


class DriftMetric(Base):
    """SQLAlchemy model for drift metrics."""
    __tablename__ = "drift_metrics"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    timestamp = Column(DateTime, nullable=False)
    drift_type = Column(String(50), nullable=False)
    metric_name = Column(String(100), nullable=False)
    metric_value = Column(Float, nullable=False)
    threshold = Column(Float)
    threshold_exceeded = Column(Boolean, default=False)
    alert_triggered = Column(Boolean, default=False)
    details = Column(JSON)


class BaselineMetric(Base):
    """SQLAlchemy model for baseline metrics."""
    __tablename__ = "baseline_metrics"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    model_version = Column(String(50), nullable=False)
    metric_name = Column(String(100), nullable=False)
    metric_value = Column(Float, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow)


class DriftDatabaseService:
    """
    Service for managing drift metrics in database.
    """
    
    def __init__(self, settings: Optional[Settings] = None):
        """
        Initialize database service.
        
        Args:
            settings: Configuration settings
        """
        self.settings = settings or Settings()
        self.engine = create_engine(self.settings.database_url)
        self.SessionLocal = sessionmaker(bind=self.engine)
        
        # Create tables if they don't exist
        Base.metadata.create_all(self.engine)
        
        logger.info("drift_database_service_initialized")
    
    def save_drift_metrics(self, metrics: Dict[str, Any]) -> bool:
        """
        Save drift metrics to database.
        
        Args:
            metrics: Dictionary containing drift metrics
            
        Returns:
            True if successful, False otherwise
        """
        try:
            session: Session = self.SessionLocal()
            
            # Extract and save data drift metrics
            if "data_drift" in metrics:
                data_drift = metrics["data_drift"]
                drift_metric = DriftMetric(
                    timestamp=datetime.utcnow(),
                    drift_type="data",
                    metric_name="psi_avg",
                    metric_value=data_drift.get("avg_psi", 0),
                    threshold=data_drift.get("threshold", 0.3),
                    threshold_exceeded=data_drift.get("drift_detected", False),
                    alert_triggered=data_drift.get("drift_detected", False),
                    details=data_drift
                )
                session.add(drift_metric)
            
            # Save target drift metrics
            if "target_drift" in metrics:
                target_drift = metrics["target_drift"]
                drift_metric = DriftMetric(
                    timestamp=datetime.utcnow(),
                    drift_type="target",
                    metric_name="fraud_rate",
                    metric_value=target_drift.get("current_fraud_rate", 0),
                    threshold=target_drift.get("baseline_fraud_rate", 0),
                    threshold_exceeded=target_drift.get("drift_detected", False),
                    alert_triggered=target_drift.get("drift_detected", False),
                    details=target_drift
                )
                session.add(drift_metric)
            
            # Save concept drift metrics
            if "concept_drift" in metrics:
                concept_drift = metrics["concept_drift"]
                concept_metrics = concept_drift.get("metrics", {})
                
                # Recall
                drift_metric = DriftMetric(
                    timestamp=datetime.utcnow(),
                    drift_type="concept",
                    metric_name="recall",
                    metric_value=concept_metrics.get("recall", 0),
                    threshold=concept_metrics.get("baseline_recall", 0.98),
                    threshold_exceeded=concept_drift.get("drift_detected", False),
                    alert_triggered=concept_drift.get("drift_detected", False),
                    details=concept_drift
                )
                session.add(drift_metric)
            
            session.commit()
            session.close()
            
            logger.info("drift_metrics_saved")
            return True
        
        except Exception as e:
            logger.error("failed_to_save_drift_metrics", error=str(e))
            return False
    
    def get_baseline_metrics(self, model_version: Optional[str] = None) -> Dict[str, float]:
        """
        Retrieve baseline metrics from database.
        
        Args:
            model_version: Specific model version (if None, get latest)
            
        Returns:
            Dictionary of baseline metrics
        """
        try:
            session: Session = self.SessionLocal()
            
            query = session.query(BaselineMetric)
            if model_version:
                query = query.filter(BaselineMetric.model_version == model_version)
            
            baselines = query.all()
            session.close()
            
            baseline_dict = {
                baseline.metric_name: baseline.metric_value
                for baseline in baselines
            }
            
            logger.info("baseline_metrics_retrieved", count=len(baseline_dict))
            return baseline_dict
        
        except Exception as e:
            logger.error("failed_to_get_baseline_metrics", error=str(e))
            return {}
    
    def query_historical_drift(
        self,
        time_range: timedelta = timedelta(days=7),
        drift_type: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Query historical drift metrics.
        
        Args:
            time_range: Time range to query
            drift_type: Filter by drift type (optional)
            
        Returns:
            DataFrame with historical drift data
        """
        try:
            session: Session = self.SessionLocal()
            
            cutoff_time = datetime.utcnow() - time_range
            
            query = session.query(DriftMetric).filter(
                DriftMetric.timestamp >= cutoff_time
            )
            
            if drift_type:
                query = query.filter(DriftMetric.drift_type == drift_type)
            
            results = query.all()
            session.close()
            
            # Convert to DataFrame
            data = [{
                "timestamp": r.timestamp,
                "drift_type": r.drift_type,
                "metric_name": r.metric_name,
                "metric_value": r.metric_value,
                "threshold": r.threshold,
                "threshold_exceeded": r.threshold_exceeded
            } for r in results]
            
            df = pd.DataFrame(data)
            
            logger.info("historical_drift_queried", count=len(df))
            return df
        
        except Exception as e:
            logger.error("failed_to_query_historical_drift", error=str(e))
            return pd.DataFrame()


# Convenience functions
def save_drift_metrics(metrics: Dict[str, Any], settings: Optional[Settings] = None) -> bool:
    """Save drift metrics to database."""
    service = DriftDatabaseService(settings)
    return service.save_drift_metrics(metrics)


def get_baseline_metrics(
    model_version: Optional[str] = None,
    settings: Optional[Settings] = None
) -> Dict[str, float]:
    """Get baseline metrics from database."""
    service = DriftDatabaseService(settings)
    return service.get_baseline_metrics(model_version)


def query_historical_drift(
    time_range: timedelta = timedelta(days=7),
    drift_type: Optional[str] = None,
    settings: Optional[Settings] = None
) -> pd.DataFrame:
    """Query historical drift data."""
    service = DriftDatabaseService(settings)
    return service.query_historical_drift(time_range, drift_type)
