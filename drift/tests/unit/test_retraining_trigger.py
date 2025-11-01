"""
Unit tests for Retraining Trigger module.
Tests trigger logic, Airflow API calls, cooldown management, and error handling.
"""

from datetime import datetime, timedelta
from unittest.mock import ANY, MagicMock, Mock, patch

import pytest
import requests
from src.alerting.rules import AlertSeverity
from src.config.settings import Settings
from src.retraining.trigger import RetrainingTrigger


@pytest.mark.unit
class TestRetrainingTrigger:
    """Test suite for RetrainingTrigger class."""

    def test_initialization(self, test_settings):
        """Test trigger initialization."""
        trigger = RetrainingTrigger(test_settings)

        assert trigger.settings == test_settings
        assert trigger.cooldown_hours == test_settings.retraining_cooldown_hours
        assert isinstance(trigger.trigger_history, list)
        assert trigger.last_trigger_time is None

    def test_should_trigger_retraining_high_severity(self, test_settings):
        """Test that high severity drift triggers retraining."""
        trigger = RetrainingTrigger(test_settings)

        drift_results = {
            "data_drift": {"severity": "HIGH", "psi": 0.25},
            "concept_drift": {"severity": "HIGH", "recall_drop": 0.15},
            "target_drift": {"severity": "MEDIUM", "relative_change": 0.3},
        }

        should_trigger = trigger.should_trigger_retraining(drift_results)

        assert should_trigger is True

    def test_should_trigger_retraining_critical_severity(self, test_settings):
        """Test that critical severity always triggers retraining."""
        trigger = RetrainingTrigger(test_settings)

        drift_results = {
            "data_drift": {"severity": "CRITICAL", "psi": 0.4},
            "concept_drift": {"severity": "LOW", "recall_drop": 0.05},
            "target_drift": {"severity": "LOW", "relative_change": 0.1},
        }

        should_trigger = trigger.should_trigger_retraining(drift_results)

        assert should_trigger is True

    def test_should_not_trigger_retraining_low_severity(self, test_settings):
        """Test that low severity drift doesn't trigger retraining."""
        trigger = RetrainingTrigger(test_settings)

        drift_results = {
            "data_drift": {"severity": "LOW", "psi": 0.05},
            "concept_drift": {"severity": "LOW", "recall_drop": 0.03},
            "target_drift": {"severity": "LOW", "relative_change": 0.05},
        }

        should_trigger = trigger.should_trigger_retraining(drift_results)

        assert should_trigger is False

    def test_should_not_trigger_retraining_cooldown_active(self, test_settings):
        """Test that retraining is not triggered during cooldown period."""
        trigger = RetrainingTrigger(test_settings)

        # Set last trigger time to within cooldown period
        trigger.last_trigger_time = datetime.utcnow() - timedelta(hours=1)
        trigger.cooldown_hours = 2  # 2 hour cooldown

        drift_results = {
            "data_drift": {"severity": "HIGH", "psi": 0.25},
            "concept_drift": {"severity": "HIGH", "recall_drop": 0.15},
            "target_drift": {"severity": "HIGH", "relative_change": 0.5},
        }

        should_trigger = trigger.should_trigger_retraining(drift_results)

        assert should_trigger is False

    def test_should_trigger_retraining_cooldown_expired(self, test_settings):
        """Test that retraining is triggered after cooldown expires."""
        trigger = RetrainingTrigger(test_settings)

        # Set last trigger time to outside cooldown period
        trigger.last_trigger_time = datetime.utcnow() - timedelta(hours=3)
        trigger.cooldown_hours = 2  # 2 hour cooldown

        drift_results = {
            "data_drift": {"severity": "HIGH", "psi": 0.25},
            "concept_drift": {"severity": "HIGH", "recall_drop": 0.15},
            "target_drift": {"severity": "HIGH", "relative_change": 0.5},
        }

        should_trigger = trigger.should_trigger_retraining(drift_results)

        assert should_trigger is True

    @patch("src.retraining.trigger.requests.post")
    def test_trigger_airflow_dag_success(self, mock_post, test_settings):
        """Test successful Airflow DAG trigger."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"dag_run_id": "test_run_123"}
        mock_post.return_value = mock_response

        trigger = RetrainingTrigger(test_settings)

        result = trigger.trigger_airflow_dag(
            dag_id="01_training_pipeline", conf={"triggered_by": "drift_detection"}
        )

        assert result is True
        mock_post.assert_called_once()
        assert "01_training_pipeline" in mock_post.call_args[0][0]

    @patch("src.retraining.trigger.requests.post")
    def test_trigger_airflow_dag_failure(self, mock_post, test_settings):
        """Test Airflow DAG trigger failure handling."""
        mock_post.side_effect = requests.exceptions.RequestException(
            "Connection failed"
        )

        trigger = RetrainingTrigger(test_settings)

        result = trigger.trigger_airflow_dag(
            dag_id="01_training_pipeline", conf={"triggered_by": "drift_detection"}
        )

        assert result is False

    @patch("src.retraining.trigger.requests.post")
    def test_trigger_airflow_dag_http_error(self, mock_post, test_settings):
        """Test Airflow DAG trigger HTTP error handling."""
        mock_response = MagicMock()
        mock_response.status_code = 400
        mock_response.text = "Bad Request"
        mock_post.return_value = mock_response

        trigger = RetrainingTrigger(test_settings)

        result = trigger.trigger_airflow_dag(
            dag_id="01_training_pipeline", conf={"triggered_by": "drift_detection"}
        )

        assert result is False

    def test_trigger_retraining_full_flow(self, test_settings):
        """Test complete retraining trigger flow."""
        trigger = RetrainingTrigger(test_settings)

        drift_results = {
            "data_drift": {"severity": "HIGH", "psi": 0.25},
            "concept_drift": {"severity": "HIGH", "recall_drop": 0.15},
            "target_drift": {"severity": "HIGH", "relative_change": 0.5},
        }

        with patch.object(
            trigger, "trigger_airflow_dag", return_value=True
        ) as mock_trigger:
            result = trigger.trigger_retraining(drift_results)

            assert result is True
            mock_trigger.assert_called_once_with(
                dag_id="01_training_pipeline",
                conf={
                    "triggered_by": "drift_detection",
                    "drift_results": drift_results,
                    "timestamp": ANY,  # Accept any timestamp string
                },
            )

            # Check that last_trigger_time was updated
            assert trigger.last_trigger_time is not None
            assert len(trigger.trigger_history) == 1

    def test_trigger_retraining_no_trigger_needed(self, test_settings):
        """Test retraining trigger when no trigger is needed."""
        trigger = RetrainingTrigger(test_settings)

        drift_results = {
            "data_drift": {"severity": "LOW", "psi": 0.05},
            "concept_drift": {"severity": "LOW", "recall_drop": 0.03},
            "target_drift": {"severity": "LOW", "relative_change": 0.05},
        }

        with patch.object(trigger, "trigger_airflow_dag") as mock_trigger:
            result = trigger.trigger_retraining(drift_results)

            assert result is False
            mock_trigger.assert_not_called()
            assert len(trigger.trigger_history) == 0

    def test_trigger_retraining_airflow_failure(self, test_settings):
        """Test retraining trigger when Airflow call fails."""
        trigger = RetrainingTrigger(test_settings)

        drift_results = {
            "data_drift": {"severity": "HIGH", "psi": 0.25},
            "concept_drift": {"severity": "HIGH", "recall_drop": 0.15},
            "target_drift": {"severity": "HIGH", "relative_change": 0.5},
        }

        with patch.object(
            trigger, "trigger_airflow_dag", return_value=False
        ) as mock_trigger:
            result = trigger.trigger_retraining(drift_results)

            assert result is False
            mock_trigger.assert_called_once()
            # Should still record the attempt in history
            assert len(trigger.trigger_history) == 1
            assert trigger.trigger_history[0]["success"] is False

    def test_get_last_trigger_time(self, test_settings):
        """Test getting last trigger time."""
        trigger = RetrainingTrigger(test_settings)

        # Initially None
        assert trigger.get_last_trigger_time() is None

        # After trigger
        trigger.last_trigger_time = datetime.utcnow()
        assert trigger.get_last_trigger_time() is not None

    def test_is_in_cooldown(self, test_settings):
        """Test cooldown checking."""
        trigger = RetrainingTrigger(test_settings)
        trigger.cooldown_hours = 2

        # No previous trigger
        assert trigger.is_in_cooldown() is False

        # Recent trigger (in cooldown)
        trigger.last_trigger_time = datetime.utcnow() - timedelta(hours=1)
        assert trigger.is_in_cooldown() is True

        # Old trigger (cooldown expired)
        trigger.last_trigger_time = datetime.utcnow() - timedelta(hours=3)
        assert trigger.is_in_cooldown() is False

    def test_trigger_history_tracking(self, test_settings):
        """Test trigger history tracking."""
        trigger = RetrainingTrigger(test_settings)

        drift_results = {
            "data_drift": {"severity": "HIGH", "psi": 0.25},
            "concept_drift": {"severity": "HIGH", "recall_drop": 0.15},
            "target_drift": {"severity": "HIGH", "relative_change": 0.5},
        }

        with patch.object(trigger, "trigger_airflow_dag", return_value=True):
            trigger.trigger_retraining(drift_results)

        assert len(trigger.trigger_history) == 1
        history_entry = trigger.trigger_history[0]

        assert "timestamp" in history_entry
        assert "drift_results" in history_entry
        assert "success" in history_entry
        assert history_entry["success"] is True
        assert history_entry["drift_results"] == drift_results

    def test_trigger_history_failure_tracking(self, test_settings):
        """Test trigger history tracking for failures."""
        trigger = RetrainingTrigger(test_settings)

        drift_results = {
            "data_drift": {"severity": "HIGH", "psi": 0.25},
            "concept_drift": {"severity": "HIGH", "recall_drop": 0.15},
            "target_drift": {"severity": "HIGH", "relative_change": 0.5},
        }

        with patch.object(trigger, "trigger_airflow_dag", return_value=False):
            trigger.trigger_retraining(drift_results)

        assert len(trigger.trigger_history) == 1
        history_entry = trigger.trigger_history[0]

        assert history_entry["success"] is False

    def test_get_recent_triggers(self, test_settings):
        """Test retrieving recent triggers."""
        trigger = RetrainingTrigger(test_settings)

        # Add some trigger history
        for i in range(5):
            trigger.trigger_history.append(
                {
                    "timestamp": (datetime.utcnow() - timedelta(hours=i)).isoformat(),
                    "drift_results": {"test": f"result_{i}"},
                    "success": i % 2 == 0,  # Alternate success/failure
                }
            )

        recent = trigger.get_recent_triggers(hours=2)

        assert len(recent) > 0
        # Should only include triggers within last 2 hours
        for entry in recent:
            entry_time = datetime.fromisoformat(entry["timestamp"])
            assert (datetime.utcnow() - entry_time).total_seconds() < 7200  # 2 hours

    def test_clear_old_history(self, test_settings):
        """Test clearing old trigger history."""
        trigger = RetrainingTrigger(test_settings)

        # Add old and recent history
        trigger.trigger_history = [
            {
                "timestamp": (datetime.utcnow() - timedelta(days=2)).isoformat(),
                "drift_results": {"old": True},
                "success": True,
            },
            {
                "timestamp": datetime.utcnow().isoformat(),
                "drift_results": {"recent": True},
                "success": True,
            },
        ]

        trigger.clear_old_history(days=1)

        assert len(trigger.trigger_history) == 1
        assert trigger.trigger_history[0]["drift_results"]["recent"] is True

    def test_max_triggers_per_day_limit(self, test_settings):
        """Test maximum triggers per day limit."""
        test_settings.max_triggers_per_day = 2
        trigger = RetrainingTrigger(test_settings)

        drift_results = {
            "data_drift": {"severity": "HIGH", "psi": 0.25},
            "concept_drift": {"severity": "HIGH", "recall_drop": 0.15},
            "target_drift": {"severity": "HIGH", "relative_change": 0.5},
        }

        # Trigger multiple times
        with patch.object(trigger, "trigger_airflow_dag", return_value=True):
            for i in range(5):
                trigger.trigger_retraining(drift_results)

        # Should be limited to max_triggers_per_day
        successful_triggers = [h for h in trigger.trigger_history if h["success"]]
        assert len(successful_triggers) <= test_settings.max_triggers_per_day

    @patch("src.retraining.trigger.logger")
    def test_logging_on_trigger(self, mock_logger, test_settings):
        """Test logging behavior on retraining trigger."""
        trigger = RetrainingTrigger(test_settings)

        drift_results = {
            "data_drift": {"severity": "HIGH", "psi": 0.25},
            "concept_drift": {"severity": "HIGH", "recall_drop": 0.15},
            "target_drift": {"severity": "HIGH", "relative_change": 0.5},
        }

        with patch.object(trigger, "trigger_airflow_dag", return_value=True):
            trigger.trigger_retraining(drift_results)

        assert mock_logger.info.called

    @patch("src.retraining.trigger.logger")
    def test_logging_on_trigger_failure(self, mock_logger, test_settings):
        """Test logging behavior on retraining trigger failure."""
        trigger = RetrainingTrigger(test_settings)

        drift_results = {
            "data_drift": {"severity": "HIGH", "psi": 0.25},
            "concept_drift": {"severity": "HIGH", "recall_drop": 0.15},
            "target_drift": {"severity": "HIGH", "relative_change": 0.5},
        }

        with patch.object(trigger, "trigger_airflow_dag", return_value=False):
            trigger.trigger_retraining(drift_results)

        assert mock_logger.error.called

    def test_trigger_with_custom_dag_id(self, test_settings):
        """Test triggering with custom DAG ID."""
        trigger = RetrainingTrigger(test_settings)

        with patch.object(
            trigger, "trigger_airflow_dag", return_value=True
        ) as mock_trigger:
            trigger.trigger_retraining(
                drift_results={"data_drift": {"severity": "HIGH"}},
                dag_id="custom_training_dag",
            )

            mock_trigger.assert_called_once()
            assert mock_trigger.call_args[1]["dag_id"] == "custom_training_dag"

    def test_trigger_with_additional_conf(self, test_settings):
        """Test triggering with additional configuration."""
        trigger = RetrainingTrigger(test_settings)

        additional_conf = {
            "model_version": "v2.1",
            "priority": "high",
            "custom_param": "value",
        }

        with patch.object(
            trigger, "trigger_airflow_dag", return_value=True
        ) as mock_trigger:
            trigger.trigger_retraining(
                drift_results={"data_drift": {"severity": "HIGH"}},
                additional_conf=additional_conf,
            )

            call_args = mock_trigger.call_args[1]
            conf = call_args["conf"]

            assert conf["model_version"] == "v2.1"
            assert conf["priority"] == "high"
            assert conf["custom_param"] == "value"
            assert "triggered_by" in conf
            assert "drift_results" in conf
            assert "timestamp" in conf

    def test_evaluate_drift_severity_combined(self, test_settings):
        """Test evaluation of combined drift severity."""
        trigger = RetrainingTrigger(test_settings)

        # Test various combinations
        test_cases = [
            # All low - should not trigger
            (
                {
                    "data_drift": {"severity": "LOW"},
                    "concept_drift": {"severity": "LOW"},
                    "target_drift": {"severity": "LOW"},
                },
                False,
            ),
            # One high - should trigger
            (
                {
                    "data_drift": {"severity": "HIGH"},
                    "concept_drift": {"severity": "LOW"},
                    "target_drift": {"severity": "LOW"},
                },
                True,
            ),
            # Multiple high - should trigger
            (
                {
                    "data_drift": {"severity": "HIGH"},
                    "concept_drift": {"severity": "HIGH"},
                    "target_drift": {"severity": "LOW"},
                },
                True,
            ),
            # Critical - should always trigger
            (
                {
                    "data_drift": {"severity": "CRITICAL"},
                    "concept_drift": {"severity": "LOW"},
                    "target_drift": {"severity": "LOW"},
                },
                True,
            ),
        ]

        for drift_results, expected in test_cases:
            result = trigger.should_trigger_retraining(drift_results)
            assert result == expected, f"Failed for {drift_results}"

    def test_airflow_api_url_construction(self, test_settings):
        """Test Airflow API URL construction."""
        trigger = RetrainingTrigger(test_settings)

        with patch("src.retraining.trigger.requests.post") as mock_post:
            mock_post.return_value.status_code = 200

            trigger.trigger_airflow_dag("test_dag", {})

            # Verify the URL was constructed correctly
            call_args = mock_post.call_args
            url = call_args[0][0]
            assert "airflow-webserver" in url
            assert "api/v1/dags/test_dag/dagRuns" in url

    def test_airflow_authentication(self, test_settings):
        """Test Airflow API authentication."""
        trigger = RetrainingTrigger(test_settings)

        with patch("src.retraining.trigger.requests.post") as mock_post:
            mock_post.return_value.status_code = 200

            trigger.trigger_airflow_dag("test_dag", {})

            # Verify authentication was included
            call_args = mock_post.call_args
            auth = call_args[1]["auth"]
            assert auth is not None
            assert len(auth) == 2  # (username, password)

    def test_trigger_payload_structure(self, test_settings):
        """Test Airflow trigger payload structure."""
        trigger = RetrainingTrigger(test_settings)

        drift_results = {"test": "data"}
        conf = {"custom": "value"}

        with patch("src.retraining.trigger.requests.post") as mock_post:
            mock_post.return_value.status_code = 200

            trigger.trigger_airflow_dag("test_dag", conf=conf)

            # Verify payload structure
            call_args = mock_post.call_args
            payload = call_args[1]["json"]

            assert "conf" in payload
            assert payload["conf"]["custom"] == "value"
            assert "triggered_by" in payload["conf"]
            assert "timestamp" in payload["conf"]


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def test_settings():
    """Test settings fixture."""
    settings = Settings()
    settings.retraining_cooldown_hours = 2
    settings.max_triggers_per_day = 5
    settings.airflow_webserver_url = "http://airflow-webserver:8080"
    settings.airflow_username = "airflow"
    settings.airflow_password = "airflow"
    return settings


@pytest.fixture
def drift_results_sample():
    """Sample drift results for testing."""
    return {
        "data_drift": {
            "severity": "HIGH",
            "psi": 0.25,
            "drifted_features": ["V1", "V2"],
        },
        "concept_drift": {
            "severity": "HIGH",
            "recall_drop": 0.15,
            "fpr_increase": 0.08,
        },
        "target_drift": {
            "severity": "MEDIUM",
            "relative_change": 0.3,
            "current_fraud_rate": 0.025,
        },
    }
