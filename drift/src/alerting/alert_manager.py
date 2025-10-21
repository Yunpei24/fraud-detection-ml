"""
Alert Manager for drift detection system.

This module handles sending alerts via multiple channels (email, Slack, SMS)
when drift is detected or thresholds are exceeded.
"""

import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
import json
import structlog

from ..config.settings import Settings

logger = structlog.get_logger(__name__)


class AlertManager:
    """
    Manages alert dispatching for drift detection events.
    
    Supports multiple alert channels:
    - Email (SMTP)
    - Slack (webhook)
    - SMS (optional - via Twilio or similar)
    
    Includes rate limiting to prevent alert fatigue.
    """
    
    def __init__(self, settings: Optional[Settings] = None):
        """
        Initialize AlertManager.
        
        Args:
            settings: Configuration settings
        """
        self.settings = settings or Settings()
        self.alert_history: List[Dict[str, Any]] = []
        self.rate_limit_window = timedelta(hours=1)
        self.max_alerts_per_hour = self.settings.alert_max_per_hour
        
        logger.info("alert_manager_initialized", max_alerts_per_hour=self.max_alerts_per_hour)
    
    def trigger_alert(
        self,
        alert_type: str,
        severity: str,
        message: str,
        details: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Trigger an alert through configured channels.
        
        Args:
            alert_type: Type of alert (e.g., "data_drift", "model_degradation")
            severity: Severity level (AlertSeverity enum or string)
            message: Alert message
            details: Additional details dictionary
            
        Returns:
            True if alert was sent, False if rate limited
        """
        # Convert AlertSeverity enum to string if needed
        severity_str = severity.value if hasattr(severity, 'value') else str(severity)
        
        # Check rate limiting
        if not self._check_rate_limit():
            logger.warning(
                "alert_rate_limited",
                alert_type=alert_type,
                severity=severity_str
            )
            return False
        
        # Create alert record
        alert = {
            "timestamp": datetime.utcnow().isoformat(),
            "alert_type": alert_type,
            "severity": severity_str,
            "message": message,
            "details": details or {}
        }
        
        # Store in history
        self.alert_history.append(alert)
        
        # Send through configured channels
        success = True
        
        if self.settings.alert_email_enabled:
            success &= self.send_email(
                recipients=self.settings.alert_email_recipients,
                subject=f"[{severity_str}] Fraud Detection Alert: {alert_type}",
                message=message,
                alert_type=alert_type,
                severity=severity_str
            )
        
        log_level = "error" if severity_str in ["ERROR", "CRITICAL"] else "warning"
        getattr(logger, log_level)(
            "alert_triggered",
            alert_type=alert_type,
            severity=severity_str,
            success=success
        )
        
        return success
    
    def send_email(
        self,
        recipients: List[str],
        subject: str,
        message: str,
        alert_type: Optional[str] = None,
        severity: Optional[str] = None
    ) -> bool:
        """
        Send alert via email.
        
        Args:
            recipients: List of email addresses
            subject: Email subject
            message: Alert message
            alert_type: Type of alert
            severity: Severity level
            
        Returns:
            True if email sent successfully, False otherwise
        """
        try:
            # Create message
            msg = MIMEMultipart()
            msg['From'] = self.settings.alert_email_from
            msg['To'] = ', '.join(recipients)
            msg['Subject'] = subject
            
            # Email body
            body = f"""
Drift Detection Alert

Alert Type: {alert_type or 'UNKNOWN'}
Severity: {severity or 'UNKNOWN'}
Timestamp: {datetime.utcnow().isoformat()}

Message:
{message}

---
This is an automated alert from the Fraud Detection Drift Monitoring System.
"""
            msg.attach(MIMEText(body, 'plain'))
            
            # Send email
            with smtplib.SMTP(self.settings.alert_email_smtp_host, self.settings.alert_email_smtp_port) as server:
                if self.settings.alert_email_username:
                    server.starttls()
                    server.login(
                        self.settings.alert_email_username,
                        self.settings.alert_email_password
                    )
                server.send_message(msg)
            
            logger.info("email_alert_sent", recipients=recipients)
            return True
        
        except Exception as e:
            logger.error("email_alert_failed", error=str(e))
            return False
    

    def _check_rate_limit(self) -> bool:
        """
        Check if we're within rate limits for sending alerts.
        
        Returns:
            True if under rate limit, False otherwise
        """
        cutoff_time = datetime.utcnow() - self.rate_limit_window
        
        # Count recent alerts
        recent_alerts = [
            alert for alert in self.alert_history
            if datetime.fromisoformat(alert['timestamp']) >= cutoff_time
        ]
        
        return len(recent_alerts) < self.max_alerts_per_hour
    
    def get_alert_history(
        self,
        hours: int = 24,
        severity: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Get alert history for specified time period.
        
        Args:
            hours: Number of hours to look back
            severity: Filter by severity level (optional)
            
        Returns:
            List of alerts
        """
        cutoff_time = datetime.utcnow() - timedelta(hours=hours)
        
        alerts = [
            alert for alert in self.alert_history
            if datetime.fromisoformat(alert['timestamp']) >= cutoff_time
        ]
        
        if severity:
            alerts = [a for a in alerts if a['severity'] == severity]
        
        return alerts
    
    def get_recent_alerts(
        self,
        minutes: int = 60,
        severity: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Get recent alerts from the last N minutes.
        
        Args:
            minutes: Number of minutes to look back
            severity: Filter by severity level (optional)
            
        Returns:
            List of recent alerts
        """
        cutoff_time = datetime.utcnow() - timedelta(minutes=minutes)
        
        alerts = [
            alert for alert in self.alert_history
            if datetime.fromisoformat(alert['timestamp']) >= cutoff_time
        ]
        
        if severity:
            alerts = [a for a in alerts if a['severity'] == severity]
        
        return alerts
    
    def clear_old_alerts(self, days: int = 7) -> None:
        """
        Clear alerts older than specified number of days.
        
        Args:
            days: Number of days to keep
        """
        cutoff_time = datetime.utcnow() - timedelta(days=days)
        
        original_count = len(self.alert_history)
        self.alert_history = [
            alert for alert in self.alert_history
            if datetime.fromisoformat(alert['timestamp']) >= cutoff_time
        ]
        
        removed_count = original_count - len(self.alert_history)
        logger.info("old_alerts_cleared", removed_count=removed_count, days=days)
    
    def clear_history(self) -> None:
        """Clear alert history."""
        self.alert_history = []
        logger.info("alert_history_cleared")
