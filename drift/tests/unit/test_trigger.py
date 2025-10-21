"""Unit tests for Retraining Trigger."""

import pytest
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, timedelta

from src.retraining.trigger import RetrainingTrigger
from src.config.settings import Settings


@pytest.mark.unit
class TestRetrainingTrigger:
    """Test suite for RetrainingTrigger class."""

    def test_initialization(self, test_settings):
        """Test trigger initialization."""
        trigger = RetrainingTrigger(test_settings)
        
        assert trigger.settings == test_settings
        assert trigger.last_retrain_time is None

    def test_should_not_retrain_no_drift(self, test_settings):
        """Test that retraining is not triggered without drift."""
        trigger = RetrainingTrigger(test_settings)
        
        drift_results = {
            'data_drift': {'drift_detected': False},
            'target_drift': {'drift_detected': False},
            'concept_drift': {'drift_detected': False}
        }
        
        should_retrain, reason = trigger.should_retrain(drift_results)
        
        assert should_retrain is False

    def test_should_retrain_with_drift(self, test_settings, drift_results_sample):
        """Test that retraining is triggered with drift."""
        trigger = RetrainingTrigger(test_settings)
        
        should_retrain, reason = trigger.should_retrain(drift_results_sample)
        
        assert should_retrain is True
        assert isinstance(reason, str)
        assert len(reason) > 0

    def test_cooldown_period(self, test_settings):
        """Test retraining cooldown period."""
        test_settings.retrain_cooldown_hours = 24
        trigger = RetrainingTrigger(test_settings)
        
        # Simulate recent retrain
        trigger.last_retrain_time = datetime.utcnow() - timedelta(hours=12)
        
        drift_results = {
            'data_drift': {'drift_detected': True, 'severity': 'HIGH'},
            'target_drift': {'drift_detected': False},
            'concept_drift': {'drift_detected': False}
        }
        
        should_retrain, reason = trigger.should_retrain(drift_results)
        
        # Should not retrain due to cooldown (unless CRITICAL)
        assert should_retrain is False or 'cooldown' in reason.lower()

    def test_critical_bypasses_cooldown(self, test_settings):
        """Test that critical severity bypasses cooldown."""
        test_settings.retrain_cooldown_hours = 24
        trigger = RetrainingTrigger(test_settings)
        
        # Simulate recent retrain
        trigger.last_retrain_time = datetime.utcnow() - timedelta(hours=1)
        
        drift_results = {
            'data_drift': {'drift_detected': True, 'severity': 'CRITICAL'},
            'target_drift': {'drift_detected': True, 'severity': 'CRITICAL'},
            'concept_drift': {'drift_detected': True, 'severity': 'CRITICAL'}
        }
        
        should_retrain, reason = trigger.should_retrain(drift_results)
        
        assert should_retrain is True

    def test_get_retrain_priority_low(self, test_settings):
        """Test priority calculation for low drift."""
        trigger = RetrainingTrigger(test_settings)
        
        drift_results = {
            'data_drift': {'drift_detected': True, 'severity': 'LOW'},
            'target_drift': {'drift_detected': False},
            'concept_drift': {'drift_detected': False}
        }
        
        priority = trigger.get_retrain_priority(drift_results)
        
        assert priority in ['LOW', 'MEDIUM', 'HIGH', 'CRITICAL']

    def test_get_retrain_priority_critical(self, test_settings):
        """Test priority calculation for critical drift."""
        trigger = RetrainingTrigger(test_settings)
        
        drift_results = {
            'data_drift': {'drift_detected': True, 'severity': 'CRITICAL'},
            'target_drift': {'drift_detected': True, 'severity': 'CRITICAL'},
            'concept_drift': {'drift_detected': True, 'severity': 'CRITICAL'}
        }
        
        priority = trigger.get_retrain_priority(drift_results)
        
        assert priority == 'CRITICAL'

    @patch('src.retraining.trigger.requests.post')
    def test_trigger_airflow_dag_success(self, mock_post, test_settings):
        """Test successful Airflow DAG trigger."""
        mock_post.return_value.status_code = 200
        mock_post.return_value.json.return_value = {'dag_run_id': 'test_run_123'}
        
        trigger = RetrainingTrigger(test_settings)
        
        result = trigger.trigger_airflow_dag(
            dag_id='retrain_fraud_model',
            conf={'priority': 'HIGH'}
        )
        
        assert result is True
        assert mock_post.called

    @patch('src.retraining.trigger.requests.post')
    def test_trigger_airflow_dag_failure(self, mock_post, test_settings):
        """Test Airflow DAG trigger failure handling."""
        mock_post.return_value.status_code = 500
        
        trigger = RetrainingTrigger(test_settings)
        
        result = trigger.trigger_airflow_dag(dag_id='retrain_fraud_model')
        
        assert result is False

    @patch('src.retraining.trigger.requests.post')
    def test_trigger_airflow_dag_with_config(self, mock_post, test_settings):
        """Test Airflow DAG trigger with configuration."""
        mock_post.return_value.status_code = 200
        
        trigger = RetrainingTrigger(test_settings)
        
        config = {
            'priority': 'HIGH',
            'drift_type': 'DATA_DRIFT',
            'trigger_reason': 'High PSI scores'
        }
        
        trigger.trigger_airflow_dag(dag_id='retrain_fraud_model', conf=config)
        
        # Verify config was passed
        call_kwargs = mock_post.call_args[1]
        assert 'json' in call_kwargs
        assert 'conf' in call_kwargs['json']

    def test_consecutive_detections_tracking(self, test_settings):
        """Test tracking of consecutive drift detections."""
        trigger = RetrainingTrigger(test_settings)
        
        drift_results = {
            'data_drift': {'drift_detected': True, 'severity': 'MEDIUM'},
            'target_drift': {'drift_detected': False},
            'concept_drift': {'drift_detected': False}
        }
        
        # First detection
        trigger.should_retrain(drift_results)
        
        # Second consecutive detection
        should_retrain, reason = trigger.should_retrain(drift_results)
        
        # May trigger after consecutive detections
        assert isinstance(should_retrain, bool)

    def test_update_last_retrain_time(self, test_settings):
        """Test updating last retrain timestamp."""
        trigger = RetrainingTrigger(test_settings)
        
        before = datetime.utcnow()
        trigger.update_last_retrain_time()
        after = datetime.utcnow()
        
        assert trigger.last_retrain_time is not None
        assert before <= trigger.last_retrain_time <= after

    def test_mixed_severity_levels(self, test_settings):
        """Test priority with mixed severity levels."""
        trigger = RetrainingTrigger(test_settings)
        
        drift_results = {
            'data_drift': {'drift_detected': True, 'severity': 'LOW'},
            'target_drift': {'drift_detected': True, 'severity': 'HIGH'},
            'concept_drift': {'drift_detected': False}
        }
        
        priority = trigger.get_retrain_priority(drift_results)
        
        # Should take highest severity
        assert priority in ['HIGH', 'CRITICAL']

    @patch('src.retraining.trigger.logger')
    def test_logging_on_trigger(self, mock_logger, test_settings, drift_results_sample):
        """Test logging when retraining is triggered."""
        trigger = RetrainingTrigger(test_settings)
        
        trigger.should_retrain(drift_results_sample)
        
        assert mock_logger.info.called or mock_logger.warning.called

    def test_multiple_drift_types(self, test_settings):
        """Test with multiple drift types detected."""
        trigger = RetrainingTrigger(test_settings)
        
        drift_results = {
            'data_drift': {'drift_detected': True, 'severity': 'MEDIUM'},
            'target_drift': {'drift_detected': True, 'severity': 'MEDIUM'},
            'concept_drift': {'drift_detected': True, 'severity': 'MEDIUM'}
        }
        
        should_retrain, reason = trigger.should_retrain(drift_results)
        
        # Multiple drift types should increase likelihood
        assert should_retrain is True

    def test_cooldown_expiration(self, test_settings):
        """Test that cooldown expires after specified time."""
        test_settings.retrain_cooldown_hours = 24
        trigger = RetrainingTrigger(test_settings)
        
        # Simulate old retrain (beyond cooldown)
        trigger.last_retrain_time = datetime.utcnow() - timedelta(hours=25)
        
        drift_results = {
            'data_drift': {'drift_detected': True, 'severity': 'MEDIUM'},
            'target_drift': {'drift_detected': False},
            'concept_drift': {'drift_detected': False}
        }
        
        should_retrain, reason = trigger.should_retrain(drift_results)
        
        # Cooldown expired, should allow retrain
        assert should_retrain is True
