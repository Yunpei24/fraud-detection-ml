"""
Retraining trigger module.

This module decides when to trigger model retraining based on drift detection results.
"""

import requests
from typing import Dict, Any, Optional
from datetime import datetime, timedelta
import structlog

from ..config.settings import Settings

logger = structlog.get_logger(__name__)


class RetrainingTrigger:
    """
    Manages retraining triggers based on drift detection.
    
    Implements cooldown periods, priority assessment, and Airflow integration.
    """
    
    def __init__(self, settings: Optional[Settings] = None):
        """
        Initialize RetrainingTrigger.
        
        Args:
            settings: Configuration settings
        """
        self.settings = settings or Settings()
        self.last_retrain_time: Optional[datetime] = None
        self.consecutive_drift_count = 0
        
        logger.info("retraining_trigger_initialized")
    
    def should_retrain(
        self,
        drift_results: Dict[str, Any],
        force: bool = False
    ) -> tuple[bool, str]:
        """
        Determine if model retraining should be triggered.
        
        Args:
            drift_results: Results from drift detection
            force: Force retraining regardless of cooldown
            
        Returns:
            Tuple of (should_retrain: bool, reason: str)
        """
        logger.info("evaluating_retraining_decision")
        
        # Check for CRITICAL severity to bypass cooldown
        data_drift = drift_results.get("data_drift", {})
        target_drift = drift_results.get("target_drift", {})
        concept_drift = drift_results.get("concept_drift", {})
        
        is_critical = (
            data_drift.get("severity") == "CRITICAL" or
            target_drift.get("severity") == "CRITICAL" or
            concept_drift.get("severity") == "CRITICAL"
        )
        
        # Check cooldown period (skip if CRITICAL or force)
        if not force and not is_critical and not self._check_cooldown():
            reason = "In cooldown period since last retraining"
            logger.info("retraining_in_cooldown_period")
            return False, reason
        
        # Check drift conditions
        data_drift_detected = data_drift.get("drift_detected", False)
        target_drift_detected = target_drift.get("drift_detected", False)
        concept_drift_detected = concept_drift.get("drift_detected", False)
        
        # Critical condition: concept drift (model performance degradation)
        if concept_drift_detected:
            concept_severity = concept_drift.get("severity", "LOW")
            if concept_severity in ["HIGH", "CRITICAL", "ERROR"]:
                reason = f"Concept drift detected with severity: {concept_severity}"
                logger.warning("retraining_triggered_concept_drift", severity=concept_severity)
                return True, reason
        
        # High priority: both data and target drift
        if data_drift_detected and target_drift_detected:
            reason = "Both data drift and target drift detected"
            logger.warning("retraining_triggered_data_and_target_drift")
            return True, reason
        
        # Medium priority: any single drift type detected
        if data_drift_detected or target_drift_detected or concept_drift_detected:
            reason = "Drift detected and cooldown period has passed"
            logger.warning("retraining_triggered_drift_detected")
            return True, reason
        
        reason = "No retraining required"
        logger.info("retraining_not_required")
        return False, reason
    
    def get_retrain_priority(
        self,
        drift_results: Dict[str, Any]
    ) -> str:
        """
        Assess the priority level for retraining.
        
        Args:
            drift_results: Results from drift detection
            
        Returns:
            Priority level: "CRITICAL", "HIGH", "MEDIUM", or "LOW"
        """
        data_drift = drift_results.get("data_drift", {})
        target_drift = drift_results.get("target_drift", {})
        concept_drift = drift_results.get("concept_drift", {})
        
        # CRITICAL: Model performance severely degraded
        if concept_drift.get("severity") == "CRITICAL":
            return "CRITICAL"
        
        # HIGH: Both concept drift and data drift
        if concept_drift.get("drift_detected") and data_drift.get("drift_detected"):
            return "HIGH"
        
        # HIGH: High severity target drift
        if target_drift.get("severity") in ["HIGH", "CRITICAL"]:
            return "HIGH"
        
        # MEDIUM: Single drift type detected
        if any([
            data_drift.get("drift_detected"),
            target_drift.get("drift_detected"),
            concept_drift.get("drift_detected")
        ]):
            return "MEDIUM"
        
        return "LOW"
    
    def trigger_airflow_dag(
        self,
        dag_id: str = "fraud_detection_training",
        priority: str = "MEDIUM",
        conf: Optional[Dict[str, Any]] = None,
        drift_results: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Trigger Airflow DAG for model retraining.
        
        Args:
            dag_id: Airflow DAG ID to trigger
            priority: Priority level
            conf: Configuration dictionary to pass to DAG
            drift_results: Drift detection results to pass to DAG (deprecated, use conf)
            
        Returns:
            True if DAG triggered successfully, False otherwise
        """
        try:
            airflow_url = f"{self.settings.airflow_api_url}/api/v1/dags/{dag_id}/dagRuns"
            
            # Prepare configuration
            if conf is None:
                conf = {}
            
            # Merge drift_results into conf if provided (for backward compatibility)
            if drift_results:
                conf.setdefault("drift_summary", {})
                conf["drift_summary"].update({
                    "data_drift_detected": drift_results.get("data_drift", {}).get("drift_detected", False),
                    "target_drift_detected": drift_results.get("target_drift", {}).get("drift_detected", False),
                    "concept_drift_detected": drift_results.get("concept_drift", {}).get("drift_detected", False)
                })
            
            # Ensure priority and metadata are set
            conf.setdefault("priority", priority)
            conf.setdefault("triggered_by", "drift_detector")
            conf.setdefault("timestamp", datetime.utcnow().isoformat())
            
            # Trigger DAG
            response = requests.post(
                airflow_url,
                json={"conf": conf},
                auth=(self.settings.airflow_username, self.settings.airflow_password),
                headers={"Content-Type": "application/json"}
            )
            
            # Check if request was successful
            if response.status_code >= 400:
                logger.error(
                    "airflow_dag_trigger_failed",
                    dag_id=dag_id,
                    status_code=response.status_code,
                    response=response.text
                )
                return False
            
            response.raise_for_status()
            
            # Update last retrain time
            self.last_retrain_time = datetime.utcnow()
            self.consecutive_drift_count = 0
            
            logger.info(
                "airflow_dag_triggered",
                dag_id=dag_id,
                priority=priority,
                response_status=response.status_code
            )
            
            return True
        
        except Exception as e:
            logger.error("failed_to_trigger_airflow_dag", error=str(e), dag_id=dag_id)
            return False
    
    def _check_cooldown(self) -> bool:
        """
        Check if cooldown period has passed since last retraining.
        
        Returns:
            True if cooldown period passed, False otherwise
        """
        if self.last_retrain_time is None:
            return True
        
        # Support both retraining_cooldown_hours and retrain_cooldown_hours (for test compatibility)
        cooldown_hours = getattr(
            self.settings,
            'retrain_cooldown_hours',  # Check for alias first (test might use this)
            getattr(self.settings, 'retraining_cooldown_hours', 48)  # Fallback to main field
        )
        
        cooldown_period = timedelta(hours=cooldown_hours)
        time_since_retrain = datetime.utcnow() - self.last_retrain_time
        
        return time_since_retrain >= cooldown_period
    
    def update_last_retrain_time(self) -> None:
        """Update the last retraining timestamp to current time."""
        self.last_retrain_time = datetime.utcnow()
        logger.info("last_retrain_time_updated", timestamp=self.last_retrain_time.isoformat())


# Convenience functions
def should_retrain(drift_results: Dict[str, Any], settings: Optional[Settings] = None) -> bool:
    """
    Determine if retraining should be triggered.
    
    Args:
        drift_results: Results from drift detection
        settings: Configuration settings
        
    Returns:
        True if retraining should be triggered, False otherwise
    """
    trigger = RetrainingTrigger(settings)
    return trigger.should_retrain(drift_results)


def get_retrain_priority(drift_results: Dict[str, Any]) -> str:
    """
    Get retraining priority level.
    
    Args:
        drift_results: Results from drift detection
        
    Returns:
        Priority level string
    """
    trigger = RetrainingTrigger()
    return trigger.get_retrain_priority(drift_results)


def trigger_airflow_dag(
    dag_id: str = "fraud_detection_training",
    priority: str = "MEDIUM",
    drift_results: Optional[Dict[str, Any]] = None,
    settings: Optional[Settings] = None
) -> bool:
    """
    Trigger Airflow DAG for retraining.
    
    Args:
        dag_id: Airflow DAG ID
        priority: Priority level
        drift_results: Drift detection results
        settings: Configuration settings
        
    Returns:
        True if successful, False otherwise
    """
    trigger = RetrainingTrigger(settings)
    return trigger.trigger_airflow_dag(dag_id, priority, drift_results)
