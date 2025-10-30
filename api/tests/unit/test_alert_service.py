import asyncio
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from src.services.alert_service import (
    AlertService,
    AlertSeverity,
    get_alert_service,
    init_alert_service,
)


class TestAlertSeverity:
    def test_severity_enum_values(self):
        assert AlertSeverity.INFO.value == "INFO"
        assert AlertSeverity.WARNING.value == "WARNING"
        assert AlertSeverity.CRITICAL.value == "CRITICAL"


class TestAlertService:
    @pytest.fixture
    def alert_service(self):
        return AlertService(
            smtp_host="smtp.test.com",
            smtp_port=587,
            smtp_user="test@test.com",
            smtp_password="testpass",
        )

    def test_initialization_with_params(self, alert_service):
        assert alert_service.smtp_host == "smtp.test.com"
        assert alert_service.smtp_port == 587
        assert alert_service.smtp_user == "test@test.com"
        assert alert_service.smtp_password == "testpass"

    def test_initialization_from_env(self, monkeypatch):
        monkeypatch.setenv("SMTP_HOST", "smtp.env.com")
        monkeypatch.setenv("SMTP_USER", "env@test.com")
        monkeypatch.setenv("SMTP_PASSWORD", "envpass")
        monkeypatch.setenv("SMTP_PORT", "25")

        service = AlertService()
        assert service.smtp_host == "smtp.env.com"
        assert service.smtp_user == "env@test.com"
        assert service.smtp_password == "envpass"
        assert service.smtp_port == 25

    @pytest.mark.asyncio
    async def test_send_email_alert_success(self, alert_service):
        with patch("smtplib.SMTP") as mock_smtp:
            mock_server = MagicMock()
            mock_smtp.return_value.__enter__.return_value = mock_server

            result = await alert_service.send_email_alert(
                recipients=["test@example.com"],
                subject="Test Subject",
                message="Test message",
                severity=AlertSeverity.INFO,
            )

            assert result is True
            mock_server.starttls.assert_called_once()
            mock_server.login.assert_called_once_with("test@test.com", "testpass")
            mock_server.sendmail.assert_called_once()

    @pytest.mark.asyncio
    async def test_send_email_alert_no_smtp_config(self):
        service = AlertService(smtp_host=None, smtp_user=None, smtp_password=None)

        result = await service.send_email_alert(
            recipients=["test@example.com"], subject="Test", message="Test"
        )

        assert result is False

    @pytest.mark.asyncio
    async def test_send_email_alert_retry_on_failure(self, alert_service):
        with patch("smtplib.SMTP") as mock_smtp:
            mock_smtp.side_effect = Exception("SMTP Error")

            result = await alert_service.send_email_alert(
                recipients=["test@example.com"],
                subject="Test",
                message="Test",
                retry_count=2,
            )

            assert result is False
            assert mock_smtp.call_count >= 2

    @pytest.mark.asyncio
    async def test_send_fraud_alert_with_recipients(self, alert_service):
        with patch("smtplib.SMTP") as mock_smtp:
            mock_server = MagicMock()
            mock_smtp.return_value.__enter__.return_value = mock_server

            result = await alert_service.send_fraud_alert(
                transaction_id="TXN-123",
                fraud_probability=0.95,
                recipients=["alert@example.com"],
                severity=AlertSeverity.CRITICAL,
            )

            assert result is True
            mock_server.sendmail.assert_called_once()

    @pytest.mark.asyncio
    async def test_send_fraud_alert_from_env(self, alert_service, monkeypatch):
        monkeypatch.setenv("ALERT_EMAIL_RECIPIENTS", "alert@example.com")

        with patch("smtplib.SMTP") as mock_smtp:
            mock_server = MagicMock()
            mock_smtp.return_value.__enter__.return_value = mock_server

            result = await alert_service.send_fraud_alert(
                transaction_id="TXN-123", fraud_probability=0.95
            )

            assert result is True


class TestAlertServiceGlobals:
    def test_get_alert_service_singleton(self):
        service1 = get_alert_service()
        service2 = get_alert_service()
        assert service1 is service2

    def test_init_alert_service(self):
        init_alert_service(smtp_host="smtp.init.com", smtp_user="init@test.com")

        service = get_alert_service()
        assert service.smtp_host == "smtp.init.com"
        assert service.smtp_user == "init@test.com"
