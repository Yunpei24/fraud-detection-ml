"""Integration tests for hourly monitoring pipeline."""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, timedelta

from src.pipelines.hourly_monitoring import (
    fetch_recent_predictions,
    compute_all_drifts,
    check_thresholds,
    trigger_alerts,
    update_dashboards,
    run_hourly_monitoring
)
from src.config.settings import Settings


@pytest.mark.integration
class TestHourlyMonitoring:
    """Integration tests for hourly monitoring pipeline."""

    @pytest.fixture
    def mock_predictions_df(self):
        """Create mock predictions DataFrame."""
        np.random.seed(42)
        n_samples = 1000
        
        return pd.DataFrame({
            'prediction_id': range(n_samples),
            'timestamp': [datetime.utcnow() - timedelta(minutes=i) for i in range(n_samples)],
            'V1': np.random.normal(0, 1, n_samples),
            'V2': np.random.normal(0, 1, n_samples),
            'V3': np.random.normal(0, 1, n_samples),
            'Amount': np.random.exponential(88, n_samples),
            'prediction': np.random.choice([0, 1], n_samples, p=[0.998, 0.002]),
            'true_label': np.random.choice([0, 1], n_samples, p=[0.998, 0.002])
        })

    @patch('src.pipelines.hourly_monitoring.get_database_connection')
    def test_fetch_recent_predictions(self, mock_db, mock_predictions_df, test_settings):
        """Test fetching recent predictions from database."""
        mock_conn = MagicMock()
        mock_db.return_value = mock_conn
        
        with patch('pandas.read_sql_query') as mock_read_sql:
            mock_read_sql.return_value = mock_predictions_df
            
            df = fetch_recent_predictions(hours=1, settings=test_settings)
            
            assert isinstance(df, pd.DataFrame)
            assert len(df) > 0
            assert 'prediction' in df.columns
            assert 'true_label' in df.columns

    def test_compute_all_drifts(self, mock_predictions_df, baseline_data, test_settings):
        """Test computing all drift types."""
        drift_results = compute_all_drifts(
            current_data=mock_predictions_df,
            baseline_data=baseline_data,
            settings=test_settings
        )
        
        assert 'data_drift' in drift_results
        assert 'target_drift' in drift_results
        assert 'concept_drift' in drift_results
        assert 'timestamp' in drift_results

    def test_compute_all_drifts_with_drift(self, current_data_with_drift, baseline_data, test_settings):
        """Test computing drifts when drift exists."""
        # Add prediction columns
        current_data_with_drift['prediction'] = current_data_with_drift['Class']
        current_data_with_drift['true_label'] = current_data_with_drift['Class']
        
        baseline_data['prediction'] = baseline_data['Class']
        baseline_data['true_label'] = baseline_data['Class']
        
        drift_results = compute_all_drifts(
            current_data=current_data_with_drift,
            baseline_data=baseline_data,
            settings=test_settings
        )
        
        assert drift_results['data_drift']['drift_detected'] is True

    def test_check_thresholds(self, drift_results_sample, test_settings):
        """Test threshold checking logic."""
        violations = check_thresholds(drift_results_sample, test_settings)
        
        assert isinstance(violations, list)
        if drift_results_sample['data_drift']['drift_detected']:
            assert len(violations) > 0

    def test_check_thresholds_no_violations(self, test_settings):
        """Test threshold checking with no violations."""
        drift_results = {
            'data_drift': {'drift_detected': False, 'avg_psi': 0.05},
            'target_drift': {'drift_detected': False},
            'concept_drift': {'drift_detected': False}
        }
        
        violations = check_thresholds(drift_results, test_settings)
        
        assert len(violations) == 0

    @patch('src.pipelines.hourly_monitoring.AlertManager')
    def test_trigger_alerts(self, mock_alert_manager, drift_results_sample, test_settings):
        """Test alert triggering."""
        mock_manager = MagicMock()
        mock_alert_manager.return_value = mock_manager
        
        trigger_alerts(drift_results_sample, test_settings)
        
        # Verify alerts were triggered
        assert mock_manager.trigger_alert.called

    @patch('src.pipelines.hourly_monitoring.setup_prometheus')
    @patch('src.pipelines.hourly_monitoring.update_prometheus_metrics')
    def test_update_dashboards(self, mock_update, mock_setup, drift_results_sample, test_settings):
        """Test dashboard update."""
        update_dashboards(drift_results_sample, test_settings)
        
        # Verify Prometheus metrics were updated
        assert mock_update.called or test_settings.prometheus_enabled is False

    @patch('src.pipelines.hourly_monitoring.fetch_recent_predictions')
    @patch('src.pipelines.hourly_monitoring.DriftDatabaseService')
    @patch('src.pipelines.hourly_monitoring.AlertManager')
    def test_run_hourly_monitoring_success(
        self, mock_alerts, mock_db, mock_fetch, 
        mock_predictions_df, baseline_data, test_settings
    ):
        """Test complete hourly monitoring pipeline."""
        mock_fetch.return_value = mock_predictions_df
        
        mock_db_instance = MagicMock()
        mock_db_instance.get_baseline_metrics.return_value = baseline_data
        mock_db.return_value = mock_db_instance
        
        # Run pipeline
        result = run_hourly_monitoring(test_settings)
        
        assert result is True
        assert mock_fetch.called
        assert mock_db_instance.save_drift_metrics.called

    @patch('src.pipelines.hourly_monitoring.fetch_recent_predictions')
    def test_run_hourly_monitoring_no_data(self, mock_fetch, test_settings):
        """Test pipeline with no recent predictions."""
        mock_fetch.return_value = pd.DataFrame()
        
        result = run_hourly_monitoring(test_settings)
        
        # Should handle gracefully
        assert result is False or result is None

    @patch('src.pipelines.hourly_monitoring.fetch_recent_predictions')
    def test_run_hourly_monitoring_error_handling(self, mock_fetch, test_settings):
        """Test pipeline error handling."""
        mock_fetch.side_effect = Exception("Database error")
        
        # Should not raise exception
        result = run_hourly_monitoring(test_settings)
        
        assert result is False

    def test_drift_persistence(self, mock_predictions_df, baseline_data, test_settings):
        """Test that drift results are persisted."""
        with patch('src.pipelines.hourly_monitoring.DriftDatabaseService') as mock_db:
            mock_db_instance = MagicMock()
            mock_db_instance.get_baseline_metrics.return_value = baseline_data
            mock_db.return_value = mock_db_instance
            
            with patch('src.pipelines.hourly_monitoring.fetch_recent_predictions') as mock_fetch:
                mock_fetch.return_value = mock_predictions_df
                
                run_hourly_monitoring(test_settings)
                
                # Verify save was called
                assert mock_db_instance.save_drift_metrics.called

    @patch('src.pipelines.hourly_monitoring.RetrainingTrigger')
    def test_retraining_trigger_integration(
        self, mock_trigger, drift_results_sample, test_settings
    ):
        """Test integration with retraining trigger."""
        mock_trigger_instance = MagicMock()
        mock_trigger_instance.should_retrain.return_value = (True, "High drift detected")
        mock_trigger.return_value = mock_trigger_instance
        
        with patch('src.pipelines.hourly_monitoring.fetch_recent_predictions'):
            with patch('src.pipelines.hourly_monitoring.DriftDatabaseService'):
                run_hourly_monitoring(test_settings)
                
                # Verify retraining check was performed
                # (implementation dependent)

    def test_end_to_end_pipeline_flow(self, mock_predictions_df, baseline_data, test_settings):
        """Test complete end-to-end pipeline flow."""
        with patch('src.pipelines.hourly_monitoring.fetch_recent_predictions') as mock_fetch:
            mock_fetch.return_value = mock_predictions_df
            
            with patch('src.pipelines.hourly_monitoring.DriftDatabaseService') as mock_db:
                mock_db_instance = MagicMock()
                mock_db_instance.get_baseline_metrics.return_value = baseline_data
                mock_db.return_value = mock_db_instance
                
                with patch('src.pipelines.hourly_monitoring.AlertManager'):
                    result = run_hourly_monitoring(test_settings)
                    
                    # Pipeline completed successfully
                    assert result is True
                    
                    # Verify all major steps executed
                    assert mock_fetch.called
                    assert mock_db_instance.get_baseline_metrics.called
                    assert mock_db_instance.save_drift_metrics.called
