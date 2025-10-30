"""
Retraining trigger module.

This module decides when to trigger model retraining based on drift detection results.
"""

import requests
from typing import Dict, Any, Optional, List
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
        self.last_trigger_time: Optional[datetime] = None
        self.trigger_history: List[Dict[str, Any]] = []
        self.cooldown_hours = self.settings.retraining_cooldown_hours
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
            drift_results: Results from API drift detection
            force: Force retraining regardless of cooldown
            
        Returns:
            Tuple of (should_retrain: bool, reason: str)
        """
        logger.info("evaluating_retraining_decision")
        
        # Check for CRITICAL severity from API drift summary
        drift_summary = drift_results.get("drift_summary", {})
        severity_score = drift_summary.get("severity_score", 0)
        
        # CRITICAL: High severity score from API
        if severity_score >= 3:
            reason = f"Critical drift detected with severity score: {severity_score}"
            logger.warning("retraining_triggered_critical_drift", severity_score=severity_score)
            return True, reason
        
        # Check cooldown period (skip if CRITICAL or force)
        if not force and severity_score < 3 and not self._check_cooldown():
            reason = "In cooldown period since last retraining"
            logger.info("retraining_in_cooldown_period")
            return False, reason
        
        # Check drift conditions from API results or direct severity results
        data_drift = drift_results.get("data_drift", {})
        target_drift = drift_results.get("target_drift", {})
        concept_drift = drift_results.get("concept_drift", {})
        
        # Check for drift detection (API format)
        data_drift_detected = data_drift.get("drift_detected", False)
        target_drift_detected = target_drift.get("drift_detected", False)
        concept_drift_detected = concept_drift.get("drift_detected", False)
        multivariate_drift_detected = drift_results.get("multivariate_drift", {}).get("drift_detected", False)
        
        # Check for severity levels (direct format used in tests)
        data_drift_severity = data_drift.get("severity", "").upper()
        target_drift_severity = target_drift.get("severity", "").upper()
        concept_drift_severity = concept_drift.get("severity", "").upper()
        
        # Overall drift detection from API summary
        overall_drift_detected = drift_summary.get("overall_drift_detected", False)
        
        # CRITICAL: Any CRITICAL severity drift
        if any(sev == "CRITICAL" for sev in [data_drift_severity, target_drift_severity, concept_drift_severity]):
            reason = "Critical severity drift detected"
            logger.warning("retraining_triggered_critical_severity")
            return True, reason
        
        # Critical condition: concept drift (model performance degradation)
        if concept_drift_detected or concept_drift_severity in ["HIGH", "CRITICAL"]:
            reason = "Concept drift detected (model performance degradation)"
            logger.warning("retraining_triggered_concept_drift")
            return True, reason
        
        # High priority: both data and target drift
        if (data_drift_detected or data_drift_severity in ["HIGH", "CRITICAL"]) and \
           (target_drift_detected or target_drift_severity in ["HIGH", "CRITICAL"]):
            reason = "Both data drift and target drift detected"
            logger.warning("retraining_triggered_data_and_target_drift")
            return True, reason
        
        # Medium priority: any single drift type detected or overall drift
        if overall_drift_detected or \
           data_drift_detected or data_drift_severity in ["HIGH", "CRITICAL"] or \
           target_drift_detected or target_drift_severity in ["HIGH", "CRITICAL"] or \
           multivariate_drift_detected:
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
            drift_results: Results from API drift detection
            
        Returns:
            Priority level: "CRITICAL", "HIGH", "MEDIUM", or "LOW"
        """
        # Check API drift summary first
        drift_summary = drift_results.get("drift_summary", {})
        severity_score = drift_summary.get("severity_score", 0)
        
        # CRITICAL: High severity score from API
        if severity_score >= 3:
            return "CRITICAL"
        
        data_drift = drift_results.get("data_drift", {})
        target_drift = drift_results.get("target_drift", {})
        concept_drift = drift_results.get("concept_drift", {})
        
        # CRITICAL: Concept drift detected
        if concept_drift.get("drift_detected", False):
            return "CRITICAL"
        
        # HIGH: Both concept drift and data drift
        if concept_drift.get("drift_detected", False) and data_drift.get("drift_detected", False):
            return "HIGH"
        
        # HIGH: High severity target drift
        if target_drift.get("severity") in ["HIGH", "CRITICAL"]:
            return "HIGH"
        
        # MEDIUM: Single drift type detected
        if any([
            data_drift.get("drift_detected", False),
            target_drift.get("drift_detected", False),
            concept_drift.get("drift_detected", False)
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
            # Use airflow_webserver_url if available (for test compatibility), otherwise use airflow_api_url
            base_url = getattr(self.settings, 'airflow_webserver_url', None) or self.settings.airflow_api_url
            airflow_url = f"{base_url}/api/v1/dags/{dag_id}/dagRuns"
            
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
            self.last_trigger_time = datetime.utcnow()
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
        if self.last_trigger_time is None:
            return True
        
        # Support both retraining_cooldown_hours and retrain_cooldown_hours (for test compatibility)
        cooldown_hours = getattr(
            self.settings,
            'retrain_cooldown_hours',  # Check for alias first (test might use this)
            getattr(self.settings, 'retraining_cooldown_hours', 48)  # Fallback to main field
        )
        
        cooldown_period = timedelta(hours=cooldown_hours)
        time_since_retrain = datetime.utcnow() - self.last_trigger_time
        
        return time_since_retrain >= cooldown_period
    
    def update_last_retrain_time(self) -> None:
        """Update the last retraining timestamp to current time."""
        self.last_trigger_time = datetime.utcnow()
        logger.info("last_retrain_time_updated", timestamp=self.last_trigger_time.isoformat())
    
    # Test compatibility methods
    def should_trigger_retraining(self, drift_results: Dict[str, Any]) -> bool:
        """
        Determine if retraining should be triggered (test compatibility).
        
        Args:
            drift_results: Results from drift detection
            
        Returns:
            True if retraining should be triggered
        """
        should_retrain, _ = self.should_retrain(drift_results)
        return should_retrain
    
    def trigger_retraining(
        self, 
        drift_results: Dict[str, Any], 
        dag_id: str = "01_training_pipeline",
        additional_conf: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Trigger retraining workflow (test compatibility).
        
        Args:
            drift_results: Results from drift detection
            dag_id: DAG ID to trigger
            additional_conf: Additional configuration
            
        Returns:
            True if successful
        """
        # Check if retraining should be triggered
        if not self.should_trigger_retraining(drift_results):
            return False
        
        # Prepare configuration
        conf = {
            "triggered_by": "drift_detection",
            "drift_results": drift_results,
            "timestamp": datetime.utcnow().isoformat()
        }
        
        if additional_conf:
            conf.update(additional_conf)
        
        # Trigger DAG
        success = self.trigger_airflow_dag(dag_id=dag_id, conf=conf)
        
        # Record in history
        self.trigger_history.append({
            "timestamp": datetime.utcnow().isoformat(),
            "drift_results": drift_results,
            "success": success
        })
        
        if success:
            self.last_trigger_time = datetime.utcnow()
        else:
            logger.error(
                "retraining_trigger_failed",
                dag_id=dag_id,
                drift_results=drift_results
            )
        
        return success
    
    def get_last_trigger_time(self) -> Optional[datetime]:
        """
        Get the last trigger time.
        
        Returns:
            Last trigger timestamp or None
        """
        return self.last_trigger_time
    
    def is_in_cooldown(self) -> bool:
        """
        Check if currently in cooldown period.
        
        Returns:
            True if in cooldown
        """
        if self.last_trigger_time is None:
            return False
        
        cooldown_period = timedelta(hours=self.cooldown_hours)
        time_since_trigger = datetime.utcnow() - self.last_trigger_time
        
        return time_since_trigger < cooldown_period
    
    def clear_old_history(self, days: int = 30) -> None:
        """
        Clear old trigger history.
        
        Args:
            days: Remove entries older than this many days
        """
        cutoff_time = datetime.utcnow() - timedelta(days=days)
        
        self.trigger_history = [
            entry for entry in self.trigger_history
            if datetime.fromisoformat(entry["timestamp"]) > cutoff_time
        ]
    
    def get_recent_triggers(self, hours: int = 24) -> List[Dict[str, Any]]:
        """
        Get recent trigger history.
        
        Args:
            hours: Look back this many hours
            
        Returns:
            List of recent trigger entries
        """
        cutoff_time = datetime.utcnow() - timedelta(hours=hours)
        
        return [
            entry for entry in self.trigger_history
            if datetime.fromisoformat(entry["timestamp"]) > cutoff_time
        ]


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
