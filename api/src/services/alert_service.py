import asyncio
import logging
import os
import smtplib
from datetime import datetime
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from enum import Enum
from typing import List, Optional

from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)


class AlertSeverity(str, Enum):
    INFO = "INFO"
    WARNING = "WARNING"
    CRITICAL = "CRITICAL"


class AlertService:
    def __init__(
        self,
        smtp_host: Optional[str] = None,
        smtp_port: Optional[int] = None,
        smtp_user: Optional[str] = None,
        smtp_password: Optional[str] = None,
    ):
        self.smtp_host = smtp_host or os.getenv("SMTP_HOST")
        self.smtp_port = smtp_port or int(os.getenv("SMTP_PORT", "587"))
        self.smtp_user = smtp_user or os.getenv("SMTP_USER")
        self.smtp_password = smtp_password or os.getenv("SMTP_PASSWORD")

    async def send_email_alert(
        self,
        recipients: List[str],
        subject: str,
        message: str,
        severity: AlertSeverity = AlertSeverity.INFO,
        retry_count: int = 3,
    ) -> bool:
        if not self.smtp_host or not self.smtp_user:
            logger.warning("SMTP not configured")
            return False

        for attempt in range(retry_count):
            try:
                msg = MIMEMultipart("alternative")
                msg["Subject"] = f"[{severity.value}] {subject}"
                msg["From"] = self.smtp_user
                msg["To"] = ", ".join(recipients)

                color_map = {
                    AlertSeverity.CRITICAL: "red",
                    AlertSeverity.WARNING: "orange",
                    AlertSeverity.INFO: "green",
                }
                color = color_map.get(severity, "green")
                html = f"""<html><body style="font-family: Arial; color: #333;">
<div style="border-left: 4px solid {color}; padding: 10px;">
<h2 style="color: {color}; margin: 0;">[{severity.value}] {subject}</h2>
<div style="margin: 15px 0;">{message}</div>
<div style="color: #999; font-size: 12px; margin-top: 15px;">{datetime.utcnow().isoformat()}</div>
</div></body></html>"""

                msg.attach(MIMEText(html, "html"))

                with smtplib.SMTP(self.smtp_host, self.smtp_port, timeout=10) as server:
                    server.starttls()
                    server.login(self.smtp_user, self.smtp_password)
                    server.sendmail(self.smtp_user, recipients, msg.as_string())

                logger.info(f"Email sent to {len(recipients)} recipients")
                return True

            except Exception as e:
                logger.error(
                    f"Email alert failed (attempt {attempt + 1}/{retry_count}): {e}"
                )
                if attempt < retry_count - 1:
                    await asyncio.sleep(2**attempt)

        return False

    async def send_fraud_alert(
        self,
        transaction_id: str,
        fraud_probability: float,
        recipients: Optional[List[str]] = None,
        severity: AlertSeverity = AlertSeverity.CRITICAL,
    ) -> bool:
        if not recipients:
            recipients_str = os.getenv("ALERT_EMAIL_RECIPIENTS", "")
            recipients = [e.strip() for e in recipients_str.split(",") if e.strip()]

        if not recipients:
            logger.warning("No alert recipients configured")
            return False

        message = f"Transaction ID: {transaction_id}<br>Fraud Probability: {fraud_probability:.2%}<br>Time: {datetime.utcnow().isoformat()}"

        return await self.send_email_alert(
            recipients=recipients,
            subject=f"Fraud Alert: {transaction_id}",
            message=message,
            severity=severity,
        )


_alert_service: Optional[AlertService] = None


def get_alert_service() -> AlertService:
    global _alert_service
    if _alert_service is None:
        _alert_service = AlertService()
    return _alert_service


def init_alert_service(
    smtp_host: Optional[str] = None,
    smtp_port: Optional[int] = None,
    smtp_user: Optional[str] = None,
    smtp_password: Optional[str] = None,
) -> AlertService:
    global _alert_service
    _alert_service = AlertService(
        smtp_host=smtp_host,
        smtp_port=smtp_port,
        smtp_user=smtp_user,
        smtp_password=smtp_password,
    )
    return _alert_service
