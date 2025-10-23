"""
Alert custom operator for Airflow
"""
from typing import Any, Dict, Optional
from airflow.models import BaseOperator
from airflow.utils.decorators import apply_defaults
import sys
import os

# Add drift module to path
sys.path.append('/opt/airflow/fraud-detection-ml/drift/src')


class FraudDetectionAlertOperator(BaseOperator):
    """
    Send alerts via existing AlertManager from drift module.
    
    :param alert_type: Type of alert (e.g., 'data_drift', 'model_degradation')
    :param severity: Severity level ('LOW', 'MEDIUM', 'HIGH', 'CRITICAL')
    :param message: Alert message
    :param details: Additional details dictionary
    """
    
    @apply_defaults
    def __init__(
        self,
        alert_type: str,
        severity: str,
        message: str,
        details: Optional[Dict[str, Any]] = None,
        *args,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.alert_type = alert_type
        self.severity = severity
        self.message = message
        self.details = details or {}
    
    def execute(self, context):
        """Send alert"""
        self.log.info(f"Sending {self.severity} alert: {self.alert_type}")
        
        try:
            from drift.src.alerting.alert_manager import AlertManager
            from drift.src.config.settings import Settings
            
            settings = Settings()
            alert_manager = AlertManager(settings)
            
            success = alert_manager.trigger_alert(
                alert_type=self.alert_type,
                severity=self.severity,
                message=self.message,
                details=self.details
            )
            
            if success:
                self.log.info(f"✅ Alert sent successfully")
            else:
                self.log.warning(f"⚠️ Alert rate limited")
                
            return {"alert_sent": success, "alert_type": self.alert_type}
            
        except Exception as e:
            self.log.error(f"❌ Failed to send alert: {e}")
            raise
