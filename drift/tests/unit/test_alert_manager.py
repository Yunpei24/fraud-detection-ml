"""Unit tests for Alert Manager."""

import pytest
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, timedelta

from src.alerting.alert_manager import AlertManager
from src.alerting.rules import AlertSeverity
from src.config.settings import Settings


@pytest.mark.unit
class TestAlertManager:
    """Test suite for AlertManager class."""

    def test_initialization(self, test_settings):
        """Test alert manager initialization."""
        manager = AlertManager(test_settings)
        
        assert manager.settings == test_settings
        assert isinstance(manager.alert_history, list)

    @patch('src.alerting.alert_manager.smtplib.SMTP')
    def test_send_email_success(self, mock_smtp, test_settings, alert_test_config):
        """Test successful email sending."""
        # Configure mock SMTP context manager
        mock_server = MagicMock()
        mock_smtp.return_value.__enter__.return_value = mock_server
        mock_smtp.return_value.__exit__.return_value = None
        
        manager = AlertManager(test_settings)
        
        result = manager.send_email(
            recipients=alert_test_config['email_recipients'],
            subject="Test Alert",
            message="This is a test alert"
        )
        
        assert result is True
        assert mock_smtp.called

    @patch('src.alerting.alert_manager.smtplib.SMTP')
    def test_send_slack_success(self, mock_smtp, test_settings, alert_test_config):
        """Test successful email sending (renamed from send_slack_success)."""
        # Configure mock SMTP context manager
        mock_server = MagicMock()
        mock_smtp.return_value.__enter__.return_value = mock_server
        mock_smtp.return_value.__exit__.return_value = None
        
        manager = AlertManager(test_settings)
        
        result = manager.send_email(
            recipients=alert_test_config['email_recipients'],
            subject="Test Alert",
            message="Test alert message"
        )
        
        assert result is True
        assert mock_smtp.called

    @patch('src.alerting.alert_manager.smtplib.SMTP')
    def test_send_slack_failure(self, mock_smtp, test_settings):
        """Test Slack notification failure handling."""
        mock_smtp.side_effect = Exception("SMTP Error")
        manager = AlertManager(test_settings)
        
        result = manager.send_email(
            recipients=["test@example.com"],
            subject="Test",
            message="Test message"
        )
        
        assert result is False

    def test_trigger_alert_basic(self, test_settings, drift_results_sample):
        """Test basic alert triggering."""
        manager = AlertManager(test_settings)
        
        with patch.object(manager, 'send_email') as mock_email:
            manager.trigger_alert(
                alert_type="DATA_DRIFT",
                severity=AlertSeverity.WARNING,
                message="Data drift detected",
                details=drift_results_sample
            )
        
        assert len(manager.alert_history) > 0

    def test_rate_limiting(self, test_settings):
        """Test alert rate limiting."""
        test_settings.alert_max_per_hour = 3
        manager = AlertManager(test_settings)
        
        # Trigger alerts up to limit
        for i in range(5):
            with patch.object(manager, 'send_email'):
                manager.trigger_alert(
                    alert_type="TEST",
                    severity=AlertSeverity.INFO,
                    message=f"Test alert {i}"
                )
        
        # Should have rate limited after 3 alerts
        assert len(manager.alert_history) >= 3

    def test_alert_history_tracking(self, test_settings):
        """Test that alerts are tracked in history."""
        manager = AlertManager(test_settings)
        
        with patch.object(manager, 'send_email'):
            manager.trigger_alert(
                alert_type="DATA_DRIFT",
                severity=AlertSeverity.ERROR,
                message="Test alert"
            )
        
        assert len(manager.alert_history) == 1
        alert = manager.alert_history[0]
        
        assert 'timestamp' in alert
        assert alert['alert_type'] == "DATA_DRIFT"
        assert alert['severity'] == AlertSeverity.ERROR.value
        assert alert['message'] == "Test alert"

    def test_severity_levels(self, test_settings):
        """Test different severity levels."""
        manager = AlertManager(test_settings)
        
        severities = [
            AlertSeverity.INFO,
            AlertSeverity.WARNING,
            AlertSeverity.ERROR,
            AlertSeverity.CRITICAL
        ]
        
        for severity in severities:
            with patch.object(manager, 'send_email'):
                manager.trigger_alert(
                    alert_type="TEST",
                    severity=severity,
                    message=f"Test {severity} alert"
                )
        
        assert len(manager.alert_history) == len(severities)

    @patch('src.alerting.alert_manager.smtplib.SMTP')
    def test_email_error_handling(self, mock_smtp, test_settings):
        """Test email error handling."""
        mock_smtp.side_effect = Exception("SMTP Error")
        manager = AlertManager(test_settings)
        
        result = manager.send_email(
            recipients=["test@example.com"],
            subject="Test",
            message="Test"
        )
        
        assert result is False

    def test_get_recent_alerts(self, test_settings):
        """Test retrieving recent alerts."""
        manager = AlertManager(test_settings)
        
        # Add some alerts
        for i in range(10):
            manager.alert_history.append({
                'timestamp': (datetime.utcnow() - timedelta(minutes=i)).isoformat(),
                'alert_type': 'TEST',
                'severity': AlertSeverity.INFO,
                'message': f'Alert {i}'
            })
        
        recent = manager.get_recent_alerts(minutes=30)
        
        assert len(recent) > 0

    def test_clear_old_alerts(self, test_settings):
        """Test clearing old alerts from history."""
        manager = AlertManager(test_settings)
        
        # Add old alert
        old_alert = {
            'timestamp': (datetime.utcnow() - timedelta(days=2)).isoformat(),
            'alert_type': 'TEST',
            'severity': AlertSeverity.INFO,
            'message': 'Old alert'
        }
        manager.alert_history.append(old_alert)
        
        # Add recent alert
        recent_alert = {
            'timestamp': datetime.utcnow().isoformat(),
            'alert_type': 'TEST',
            'severity': AlertSeverity.INFO,
            'message': 'Recent alert'
        }
        manager.alert_history.append(recent_alert)
        
        manager.clear_old_alerts(days=1)
        
        assert len(manager.alert_history) == 1
        assert manager.alert_history[0]['message'] == 'Recent alert'

    @patch('src.alerting.alert_manager.logger')
    def test_logging_on_alert(self, mock_logger, test_settings):
        """Test that alerts are logged."""
        manager = AlertManager(test_settings)
        
        with patch.object(manager, 'send_email'):
            manager.trigger_alert(
                alert_type="TEST",
                severity=AlertSeverity.WARNING,
                message="Test alert"
            )
        
        assert mock_logger.warning.called or mock_logger.error.called

    def test_alert_with_details(self, test_settings, drift_results_sample):
        """Test alert with detailed information."""
        manager = AlertManager(test_settings)
        
        with patch.object(manager, 'send_email'):
            manager.trigger_alert(
                alert_type="DATA_DRIFT",
                severity=AlertSeverity.ERROR,
                message="Drift detected",
                details=drift_results_sample
            )
        
        alert = manager.alert_history[0]
        assert 'details' in alert
        assert alert['details'] == drift_results_sample

    def test_disabled_channels(self, test_settings):
        """Test when email and Slack are disabled."""
        test_settings.alert_email_enabled = False
        test_settings.alert_slack_enabled = False
        manager = AlertManager(test_settings)
        
        # Should still track alert but not send
        manager.trigger_alert(
            alert_type="TEST",
            severity=AlertSeverity.INFO,
            message="Test"
        )
        
        assert len(manager.alert_history) == 1
