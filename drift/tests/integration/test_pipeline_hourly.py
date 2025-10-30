"""Integration tests for hourly monitoring pipeline."""

from datetime import datetime, timedelta
from unittest.mock import MagicMock, Mock, patch

import numpy as np
import pandas as pd
import pytest
from src.config.settings import Settings
from src.pipelines.hourly_monitoring import (call_api_drift_detection,
                                             check_thresholds,
                                             fetch_recent_predictions,
                                             run_hourly_monitoring,
                                             trigger_alerts, update_dashboards)


@pytest.mark.integration
class TestHourlyMonitoring:
    """Integration tests for hourly monitoring pipeline."""

    @pytest.fixture
    def mock_predictions_df(self):
        """Create mock predictions DataFrame."""
        np.random.seed(42)
        n_samples = 1000

        return pd.DataFrame(
            {
                "prediction_id": range(n_samples),
                "timestamp": [
                    datetime.utcnow() - timedelta(minutes=i) for i in range(n_samples)
                ],
                "V1": np.random.normal(0, 1, n_samples),
                "V2": np.random.normal(0, 1, n_samples),
                "V3": np.random.normal(0, 1, n_samples),
                "amount": np.random.exponential(88, n_samples),
                "prediction": np.random.choice([0, 1], n_samples, p=[0.998, 0.002]),
                "true_label": np.random.choice([0, 1], n_samples, p=[0.998, 0.002]),
            }
        )

    @pytest.fixture
    def mock_api_response(self):
        """Create mock API drift detection response."""
        return {
            "timestamp": datetime.utcnow().isoformat(),
            "data_drift": {
                "drift_detected": False,
                "avg_psi": 0.03,
                "dataset_drift_detected": False,
            },
            "target_drift": {"drift_detected": False, "drift_score": 0.05},
            "concept_drift": {"drift_detected": False, "drift_score": 0.1},
            "multivariate_drift": {"overall_drift_detected": False},
            "drift_summary": {
                "overall_drift_detected": False,
                "drift_types_detected": [],
                "severity_score": 0,
            },
        }

    @pytest.fixture
    def mock_api_response_with_drift(self):
        """Create mock API response with drift detected."""
        return {
            "timestamp": datetime.utcnow().isoformat(),
            "data_drift": {
                "drift_detected": True,
                "avg_psi": 0.25,
                "dataset_drift_detected": True,
            },
            "target_drift": {"drift_detected": True, "drift_score": 0.35},
            "concept_drift": {"drift_detected": False, "drift_score": 0.1},
            "multivariate_drift": {"overall_drift_detected": True},
            "drift_summary": {
                "overall_drift_detected": True,
                "drift_types_detected": [
                    "data_drift",
                    "target_drift",
                    "multivariate_drift",
                ],
                "severity_score": 3,
            },
        }

    def test_fetch_recent_predictions(self, mock_predictions_df, test_settings):
        """Test fetching recent predictions from database."""
        # This is now a placeholder function, so we just test it returns empty DataFrame
        df = fetch_recent_predictions(hours=1, settings=test_settings)

        assert isinstance(df, pd.DataFrame)
        # Function now returns empty DataFrame as placeholder
        assert len(df) == 0

    @patch("src.pipelines.hourly_monitoring.FraudDetectionAPIClient")
    def test_call_api_drift_detection_success(
        self, mock_api_client, mock_api_response, test_settings
    ):
        """Test successful API drift detection call."""
        mock_client_instance = MagicMock()
        mock_client_instance.detect_comprehensive_drift.return_value = mock_api_response
        mock_api_client.return_value = mock_client_instance

        result = call_api_drift_detection(
            window_hours=1, reference_window_days=30, settings=test_settings
        )

        assert "timestamp" in result
        assert "data_drift" in result
        assert "target_drift" in result
        assert "concept_drift" in result
        assert "drift_summary" in result
        assert "api_response" in result

        # Verify API client was called correctly
        mock_api_client.assert_called_once()
        mock_client_instance.detect_comprehensive_drift.assert_called_once_with(
            window_hours=1, reference_window_days=30, auth_token=None
        )

    @patch("src.pipelines.hourly_monitoring.FraudDetectionAPIClient")
    def test_call_api_drift_detection_error(self, mock_api_client, test_settings):
        """Test API drift detection with error response."""
        mock_client_instance = MagicMock()
        mock_client_instance.detect_comprehensive_drift.return_value = {
            "error": "API unavailable"
        }
        mock_api_client.return_value = mock_client_instance

        result = call_api_drift_detection(
            window_hours=1, reference_window_days=30, settings=test_settings
        )

        assert "error" in result
        assert result["error"] == "API unavailable"

    def test_check_thresholds_no_drift(self, mock_api_response, test_settings):
        """Test threshold checking with no drift detected."""
        exceeded = check_thresholds(mock_api_response)

        assert exceeded is False

    def test_check_thresholds_with_drift(
        self, mock_api_response_with_drift, test_settings
    ):
        """Test threshold checking with drift detected."""
        exceeded = check_thresholds(mock_api_response_with_drift)

        assert exceeded is True

    @patch("src.pipelines.hourly_monitoring.AlertManager")
    def test_trigger_alerts_no_drift(
        self, mock_alert_manager, mock_api_response, test_settings
    ):
        """Test alert triggering when no drift detected."""
        mock_manager = MagicMock()
        mock_alert_manager.return_value = mock_manager

        trigger_alerts(mock_api_response, test_settings)

        # No alerts should be triggered
        assert not mock_manager.trigger_alert.called

    @patch("src.pipelines.hourly_monitoring.AlertManager")
    def test_trigger_alerts_with_drift(
        self, mock_alert_manager, mock_api_response_with_drift, test_settings
    ):
        """Test alert triggering when drift is detected."""
        mock_manager = MagicMock()
        mock_alert_manager.return_value = mock_manager

        trigger_alerts(mock_api_response_with_drift, test_settings)

        # Alert should be triggered for overall drift
        assert mock_manager.trigger_alert.called

    @patch("src.pipelines.hourly_monitoring.update_drift_metrics")
    def test_update_dashboards(
        self, mock_update_metrics, mock_api_response, test_settings
    ):
        """Test dashboard update."""
        update_dashboards(mock_api_response)

        # Verify drift metrics were updated
        assert mock_update_metrics.called

    @patch("src.pipelines.hourly_monitoring.call_api_drift_detection")
    @patch("src.pipelines.hourly_monitoring.RetrainingTrigger")
    @patch("src.pipelines.hourly_monitoring.AlertManager")
    def test_run_hourly_monitoring_success(
        self,
        mock_alert_manager,
        mock_retraining_trigger,
        mock_api_call,
        mock_api_response,
        test_settings,
    ):
        """Test complete hourly monitoring pipeline success."""
        mock_api_call.return_value = mock_api_response

        mock_alert_instance = MagicMock()
        mock_alert_manager.return_value = mock_alert_instance

        mock_retraining_instance = MagicMock()
        mock_retraining_instance.should_retrain.return_value = (
            False,
            "No drift detected",
        )
        mock_retraining_trigger.return_value = mock_retraining_instance

        result = run_hourly_monitoring(test_settings)

        assert result["status"] == "success"
        assert "drift_results" in result
        assert "threshold_exceeded" in result
        assert "retraining_triggered" in result
        assert result["retraining_triggered"] is False

    @patch("src.pipelines.hourly_monitoring.call_api_drift_detection")
    def test_run_hourly_monitoring_api_error(self, mock_api_call, test_settings):
        """Test pipeline with API error."""
        mock_api_call.return_value = {"error": "API connection failed"}

        result = run_hourly_monitoring(test_settings)

        assert result["status"] == "api_error"
        assert "error" in result
        assert result["error"] == "API connection failed"

    @patch("src.pipelines.hourly_monitoring.call_api_drift_detection")
    @patch("src.pipelines.hourly_monitoring.RetrainingTrigger")
    @patch("src.pipelines.hourly_monitoring.AlertManager")
    def test_run_hourly_monitoring_with_retraining(
        self,
        mock_alert_manager,
        mock_retraining_trigger,
        mock_api_call,
        mock_api_response_with_drift,
        test_settings,
    ):
        """Test pipeline that triggers retraining."""
        mock_api_call.return_value = mock_api_response_with_drift

        mock_alert_instance = MagicMock()
        mock_alert_manager.return_value = mock_alert_instance

        mock_retraining_instance = MagicMock()
        mock_retraining_instance.should_retrain.return_value = (
            True,
            "High drift detected",
        )
        mock_retraining_instance.get_retrain_priority.return_value = "HIGH"
        mock_retraining_instance.trigger_airflow_dag.return_value = True
        mock_retraining_trigger.return_value = mock_retraining_instance

        result = run_hourly_monitoring(test_settings)

        assert result["status"] == "success"
        assert result["threshold_exceeded"] is True
        assert result["retraining_triggered"] is True
        assert result["retraining_reason"] == "High drift detected"

        # Verify retraining trigger was called
        mock_retraining_instance.trigger_airflow_dag.assert_called_once()

    @patch("src.pipelines.hourly_monitoring.call_api_drift_detection")
    def test_run_hourly_monitoring_exception_handling(
        self, mock_api_call, test_settings
    ):
        """Test pipeline exception handling."""
        mock_api_call.side_effect = Exception("Unexpected error")

        result = run_hourly_monitoring(test_settings)

        assert result["status"] == "failed"
        assert "error" in result
        assert "Unexpected error" in result["error"]
