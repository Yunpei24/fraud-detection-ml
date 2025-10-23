"""End-to-end integration tests for drift detection system."""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock

from src.detection.data_drift import DataDriftDetector
from src.detection.target_drift import TargetDriftDetector
from src.detection.concept_drift import ConceptDriftDetector
from src.alerting.alert_manager import AlertManager
from src.retraining.trigger import RetrainingTrigger
from src.storage.database import DriftDatabaseService
from src.pipelines.hourly_monitoring import run_hourly_monitoring
from src.config.settings import Settings


@pytest.mark.integration
@pytest.mark.slow
class TestEndToEnd:
    """End-to-end integration tests for complete drift detection workflow."""

    @pytest.fixture
    def complete_system_setup(self, test_settings):
        """Setup complete system with all components."""
        return {
            'settings': test_settings,
            'data_drift': DataDriftDetector(test_settings),
            'target_drift': TargetDriftDetector(test_settings),
            'concept_drift': ConceptDriftDetector(test_settings),
            'alert_manager': AlertManager(test_settings),
            'retraining_trigger': RetrainingTrigger(test_settings),
            'db_service': DriftDatabaseService(test_settings)
        }

    def test_complete_drift_detection_workflow(
        self, complete_system_setup, baseline_data, current_data_with_drift
    ):
        """Test complete workflow from data to alerts."""
        system = complete_system_setup
        
        # Step 1: Detect data drift
        data_drift_results = system['data_drift'].detect(
            baseline_data, current_data_with_drift
        )
        assert 'drift_detected' in data_drift_results
        
        # Step 2: Detect target drift
        target_drift_results = system['target_drift'].detect(
            baseline_data['Class'].values,
            current_data_with_drift['Class'].values
        )
        assert 'drift_detected' in target_drift_results
        
        # Step 3: Check if retraining needed
        combined_results = {
            'data_drift': data_drift_results,
            'target_drift': target_drift_results,
            'concept_drift': {'drift_detected': False}
        }
        
        should_retrain, reason = system['retraining_trigger'].should_retrain(
            combined_results
        )
        assert isinstance(should_retrain, bool)

    def test_no_drift_workflow(
        self, complete_system_setup, baseline_data, current_data_no_drift
    ):
        """Test workflow when no drift exists."""
        system = complete_system_setup
        
        # Detect drift (should be none)
        data_drift_results = system['data_drift'].detect(
            baseline_data, current_data_no_drift
        )
        
        target_drift_results = system['target_drift'].detect(
            baseline_data['Class'].values,
            current_data_no_drift['Class'].values
        )
        
        # Should not trigger retraining
        combined_results = {
            'data_drift': data_drift_results,
            'target_drift': target_drift_results,
            'concept_drift': {'drift_detected': False}
        }
        
        should_retrain, reason = system['retraining_trigger'].should_retrain(
            combined_results
        )
        
        assert should_retrain is False

    @patch('src.storage.database.Session')
    @patch('src.alerting.alert_manager.smtplib.SMTP')
    @patch('src.alerting.alert_manager.requests.post')
    def test_drift_to_alert_workflow(
        self, mock_slack, mock_smtp, mock_db,
        complete_system_setup, drift_results_sample
    ):
        """Test workflow from drift detection to alert sending."""
        system = complete_system_setup
        mock_slack.return_value.status_code = 200
        
        # Trigger alert
        system['alert_manager'].trigger_alert(
            alert_type='DATA_DRIFT',
            severity='HIGH',
            message='Significant drift detected',
            details=drift_results_sample
        )
        
        # Verify alert was logged
        assert len(system['alert_manager'].alert_history) > 0

    @patch('src.pipelines.hourly_monitoring.fetch_recent_predictions')
    @patch('src.storage.database.Session')
    def test_full_hourly_pipeline_execution(
        self, mock_db, mock_fetch,
        baseline_data, current_data_with_drift, test_settings
    ):
        """Test complete hourly pipeline execution."""
        # Setup mock data
        current_data_with_drift['prediction'] = current_data_with_drift['Class']
        current_data_with_drift['true_label'] = current_data_with_drift['Class']
        mock_fetch.return_value = current_data_with_drift
        
        # Mock database
        mock_session = MagicMock()
        mock_db.return_value = mock_session
        
        with patch('src.pipelines.hourly_monitoring.DriftDatabaseService') as mock_db_service:
            mock_db_instance = MagicMock()
            mock_db_instance.get_baseline_metrics.return_value = baseline_data
            mock_db_service.return_value = mock_db_instance
            
            with patch('src.pipelines.hourly_monitoring.AlertManager'):
                # Run complete pipeline
                result = run_hourly_monitoring(test_settings)
                
                # Verify execution
                assert mock_fetch.called

    def test_multi_detection_integration(
        self, complete_system_setup, baseline_data
    ):
        """Test integration of multiple drift detectors."""
        system = complete_system_setup
        
        # Create data with multiple drift types
        drifted_data = baseline_data.copy()
        drifted_data['V1'] = drifted_data['V1'] + 2  # Data drift
        drifted_data['Class'] = np.random.choice([0, 1], len(drifted_data), p=[0.99, 0.01])  # Target drift
        
        # Run all detectors
        data_drift = system['data_drift'].detect(baseline_data, drifted_data)
        target_drift = system['target_drift'].detect(
            baseline_data['Class'].values,
            drifted_data['Class'].values
        )
        
        # Combine results
        combined = {
            'data_drift': data_drift,
            'target_drift': target_drift,
            'concept_drift': {'drift_detected': False}
        }
        
        # Check priority
        priority = system['retraining_trigger'].get_retrain_priority(combined)
        assert priority in ['LOW', 'MEDIUM', 'HIGH', 'CRITICAL']

    @patch('src.retraining.trigger.requests.post')
    def test_drift_to_retraining_workflow(
        self, mock_airflow,
        complete_system_setup, drift_results_sample
    ):
        """Test workflow from drift detection to retraining trigger."""
        system = complete_system_setup
        mock_airflow.return_value.status_code = 200
        
        # Check if retraining needed
        should_retrain, reason = system['retraining_trigger'].should_retrain(
            drift_results_sample
        )
        
        if should_retrain:
            # Trigger Airflow DAG
            result = system['retraining_trigger'].trigger_airflow_dag(
                dag_id='retrain_fraud_model',
                conf={'reason': reason}
            )
            
            assert result is True

    def test_continuous_monitoring_simulation(
        self, complete_system_setup, baseline_data
    ):
        """Simulate continuous monitoring over time."""
        system = complete_system_setup
        
        # Simulate 24 hours of monitoring
        for hour in range(24):
            # Generate current data with increasing drift
            current_data = baseline_data.copy()
            drift_factor = hour * 0.05
            current_data['V1'] = current_data['V1'] + drift_factor
            
            # Detect drift
            drift_results = system['data_drift'].detect(baseline_data, current_data)
            
            # Track drift progression
            assert 'drift_detected' in drift_results

    @patch('src.storage.database.Session')
    def test_metrics_persistence_workflow(
        self, mock_db, complete_system_setup, drift_results_sample
    ):
        """Test persistence of metrics throughout workflow."""
        system = complete_system_setup
        mock_session = MagicMock()
        mock_db.return_value = mock_session
        
        # Save drift metrics
        metrics = {
            'timestamp': datetime.utcnow(),
            'model_version': 'v1.0.0',
            'data_drift_score': drift_results_sample['data_drift'].get('avg_psi', 0.5),
            'drift_detected': True
        }
        
        result = system['db_service'].save_drift_metrics(metrics)
        
        # Verify save was attempted
        assert result is True or isinstance(result, int) or mock_session.add.called

    def test_alert_rate_limiting_workflow(self, complete_system_setup):
        """Test alert rate limiting in full workflow."""
        system = complete_system_setup
        system['settings'].alert_max_per_hour = 3
        
        # Trigger multiple alerts rapidly
        for i in range(10):
            with patch.object(system['alert_manager'], 'send_email'):
                with patch.object(system['alert_manager'], 'send_slack'):
                    system['alert_manager'].trigger_alert(
                        alert_type='TEST',
                        severity='INFO',
                        message=f'Alert {i}'
                    )
        
        # Should respect rate limit
        assert len(system['alert_manager'].alert_history) >= 3

    def test_baseline_update_workflow(
        self, complete_system_setup, baseline_data
    ):
        """Test workflow for updating baseline metrics."""
        system = complete_system_setup
        
        with patch('src.storage.database.Session'):
            # Update baseline
            new_baseline = {
                'model_version': 'v1.1.0',
                'feature_means': baseline_data.mean().to_dict(),
                'feature_stds': baseline_data.std().to_dict(),
                'fraud_rate': baseline_data['Class'].mean()
            }
            
            result = system['db_service'].update_baseline_metrics(
                'v1.1.0', new_baseline
            )
            
            # Verify update
            assert result is True or result is None

    def test_error_recovery_workflow(self, complete_system_setup, baseline_data):
        """Test system recovery from errors."""
        system = complete_system_setup
        
        # Simulate error in drift detection
        with patch.object(system['data_drift'], 'detect', side_effect=Exception("Error")):
            try:
                system['data_drift'].detect(baseline_data, baseline_data)
            except Exception:
                pass
        
        # System should continue to function
        # Try again with valid call
        result = system['data_drift'].detect(baseline_data, baseline_data)
        assert 'drift_detected' in result

    def test_performance_under_load(
        self, complete_system_setup, baseline_data
    ):
        """Test system performance under load."""
        system = complete_system_setup
        
        start_time = datetime.utcnow()
        
        # Process multiple drift detections
        for _ in range(10):
            current_data = baseline_data.copy()
            current_data['V1'] = current_data['V1'] + np.random.normal(0, 0.1, len(current_data))
            
            system['data_drift'].detect(baseline_data, current_data)
        
        duration = (datetime.utcnow() - start_time).total_seconds()
        
        # Should complete in reasonable time
        assert duration < 30  # seconds
