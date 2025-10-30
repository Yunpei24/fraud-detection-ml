"""
Custom Alert Operator for Fraud Detection
==========================================
Sends alerts for fraud detection events via multiple channels.
"""

import json
import logging
from datetime import datetime
from typing import Any, Dict, Optional

from airflow.models import BaseOperator
from airflow.utils.decorators import apply_defaults


class FraudDetectionAlertOperator(BaseOperator):
    """
    Operator to send alerts for fraud detection events.

    Supports multiple alert channels:
    - Logging (always enabled)
    - Email (via Airflow email backend)
    - Slack (via webhook - optional)
    - PagerDuty (via API - optional)

    Usage:
        alert = FraudDetectionAlertOperator(
            task_id='send_alert',
            alert_type='data_quality',
            severity='HIGH',
            message='Data quality issues detected',
            details={'missing_values': 10}
        )
    """

    ui_color = "#ff6b6b"  # Red for alerts
    ui_fgcolor = "#ffffff"

    @apply_defaults
    def __init__(
        self,
        alert_type: str,
        severity: str,
        message: str,
        details: Optional[Dict[str, Any]] = None,
        channels: Optional[list] = None,
        *args,
        **kwargs,
    ):
        """
        Initialize FraudDetectionAlertOperator.

        Args:
            alert_type: Type of alert (e.g., 'data_quality', 'drift_detection', 'model_performance')
            severity: Severity level ('INFO', 'WARNING', 'CRITICAL')
            message: Alert message
            details: Additional details (dictionary)
            channels: List of alert channels ('log', 'email', 'slack', 'pagerduty')
            *args: Additional BaseOperator arguments
            **kwargs: Additional BaseOperator keyword arguments
        """
        super().__init__(*args, **kwargs)
        self.alert_type = alert_type
        self.severity = severity
        self.message = message
        self.details = details or {}
        self.channels = channels or ["log"]

        self.logger = logging.getLogger(__name__)

    def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute the alert operator.

        Args:
            context: Airflow task context

        Returns:
            Dictionary with alert status and details
        """
        self.logger.info(f"Executing FraudDetectionAlertOperator for {self.alert_type}")

        # Prepare alert payload
        alert_payload = self._prepare_alert_payload(context)

        # Send alerts to configured channels
        results = {}

        if "log" in self.channels:
            results["log"] = self._send_log_alert(alert_payload)

        if "email" in self.channels:
            results["email"] = self._send_email_alert(alert_payload, context)

        if "slack" in self.channels:
            results["slack"] = self._send_slack_alert(alert_payload)

        if "pagerduty" in self.channels:
            results["pagerduty"] = self._send_pagerduty_alert(alert_payload)

        return {
            "status": "success",
            "alert_type": self.alert_type,
            "severity": self.severity,
            "channels": list(results.keys()),
            "results": results,
            "timestamp": alert_payload["timestamp"],
        }

    def _prepare_alert_payload(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Prepare alert payload with all relevant information.

        Args:
            context: Airflow task context

        Returns:
            Alert payload dictionary
        """
        return {
            "alert_type": self.alert_type,
            "severity": self.severity,
            "message": self.message,
            "details": self.details,
            "dag_id": context.get("dag").dag_id if "dag" in context else "unknown",
            "task_id": (
                context.get("task_instance").task_id
                if "task_instance" in context
                else "unknown"
            ),
            "execution_date": (
                context.get("execution_date").isoformat()
                if "execution_date" in context
                else None
            ),
            "timestamp": datetime.utcnow().isoformat(),
        }

    def _send_log_alert(self, alert_payload: Dict[str, Any]) -> Dict[str, str]:
        """
        Send alert to logging system.

        Args:
            alert_payload: Alert payload

        Returns:
            Status dictionary
        """
        severity = alert_payload["severity"]
        alert_type = alert_payload["alert_type"]
        message = alert_payload["message"]

        # Format alert message
        alert_msg = f"[{severity}] {alert_type}: {message}"

        # Log with appropriate level
        if severity == "CRITICAL":
            self.logger.error(alert_msg)
            self.logger.error(
                f"Details: {json.dumps(alert_payload['details'], indent=2)}"
            )
        elif severity in ["WARNING", "HIGH"]:
            self.logger.warning(alert_msg)
            self.logger.warning(
                f"Details: {json.dumps(alert_payload['details'], indent=2)}"
            )
        else:
            self.logger.info(alert_msg)
            self.logger.info(
                f"Details: {json.dumps(alert_payload['details'], indent=2)}"
            )

        # Also print for visibility
        print(f"ALERT: {alert_msg}")
        print(f"   Details: {alert_payload['details']}")

        return {"status": "sent", "channel": "log"}

    def _send_email_alert(
        self, alert_payload: Dict[str, Any], context: Dict[str, Any]
    ) -> Dict[str, str]:
        """
        Send alert via email using Airflow email backend.

        Args:
            alert_payload: Alert payload
            context: Airflow task context

        Returns:
            Status dictionary
        """
        try:
            from airflow.utils.email import send_email

            subject = f"[{alert_payload['severity']}] Fraud Detection Alert: {alert_payload['alert_type']}"

            html_content = f"""
            <html>
            <body>
                <h2>{alert_payload['message']}</h2>
                <h3>Alert Details</h3>
                <ul>
                    <li><strong>Type:</strong> {alert_payload['alert_type']}</li>
                    <li><strong>Severity:</strong> {alert_payload['severity']}</li>
                    <li><strong>DAG:</strong> {alert_payload['dag_id']}</li>
                    <li><strong>Task:</strong> {alert_payload['task_id']}</li>
                    <li><strong>Execution Date:</strong> {alert_payload['execution_date']}</li>
                    <li><strong>Timestamp:</strong> {alert_payload['timestamp']}</li>
                </ul>
                <h3>Additional Details</h3>
                <pre>{json.dumps(alert_payload['details'], indent=2)}</pre>
            </body>
            </html>
            """

            # Get email recipients from context or use default
            to = context.get("dag").default_args.get(
                "email", ["ml-alerts@frauddetection.com"]
            )

            send_email(to, subject, html_content)

            return {"status": "sent", "channel": "email", "recipients": to}

        except Exception as e:
            self.logger.error(f"Failed to send email alert: {e}")
            return {"status": "failed", "channel": "email", "error": str(e)}

    def _send_slack_alert(self, alert_payload: Dict[str, Any]) -> Dict[str, str]:
        """
        Send alert to Slack via webhook.

        Args:
            alert_payload: Alert payload

        Returns:
            Status dictionary
        """
        # Placeholder for Slack integration
        # In production, this would use Slack webhook API
        self.logger.info("Slack alert not configured - skipping")

        return {"status": "skipped", "channel": "slack", "reason": "not_configured"}

    def _send_pagerduty_alert(self, alert_payload: Dict[str, Any]) -> Dict[str, str]:
        """
        Send alert to PagerDuty via API.

        Args:
            alert_payload: Alert payload

        Returns:
            Status dictionary
        """
        # Placeholder for PagerDuty integration
        # In production, this would use PagerDuty Events API
        self.logger.info("PagerDuty alert not configured - skipping")

        return {"status": "skipped", "channel": "pagerduty", "reason": "not_configured"}
