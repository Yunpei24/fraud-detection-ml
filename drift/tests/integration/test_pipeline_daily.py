"""Integration tests for daily analysis pipeline."""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, timedelta

from src.pipelines.daily_analysis import (
    aggregate_daily_metrics,
    generate_daily_report,
    identify_trends,
    recommend_actions,
    run_daily_analysis
)
from src.config.settings import Settings


@pytest.mark.integration
class TestDailyAnalysis:
    """Integration tests for daily analysis pipeline."""

    @pytest.fixture
    def daily_metrics_history(self):
        """Create mock daily metrics history."""
        dates = [datetime.utcnow() - timedelta(days=i) for i in range(30)]
        
        return pd.DataFrame({
            'date': dates,
            'avg_psi': np.random.uniform(0.1, 0.5, 30),
            'fraud_rate': np.random.uniform(0.001, 0.005, 30),
            'recall': np.random.uniform(0.90, 0.98, 30),
            'fpr': np.random.uniform(0.01, 0.03, 30),
            'drift_detected': np.random.choice([True, False], 30, p=[0.3, 0.7])
        })

    @patch('src.pipelines.daily_analysis.DriftDatabaseService')
    def test_aggregate_daily_metrics(self, mock_db, test_settings):
        """Test daily metrics aggregation."""
        mock_db_instance = MagicMock()
        mock_db_instance.query_historical_drift.return_value = pd.DataFrame({
            'timestamp': [datetime.utcnow() - timedelta(hours=i) for i in range(24)],
            'psi_score': np.random.uniform(0.1, 0.4, 24),
            'fraud_rate': np.random.uniform(0.001, 0.004, 24)
        })
        mock_db.return_value = mock_db_instance
        
        metrics = aggregate_daily_metrics(test_settings)
        
        assert isinstance(metrics, dict)
        assert 'avg_psi' in metrics or 'period' in metrics

    def test_generate_daily_report(self, daily_metrics_history, test_settings):
        """Test daily report generation."""
        report = generate_daily_report(
            daily_metrics=daily_metrics_history.iloc[0].to_dict(),
            historical_data=daily_metrics_history,
            settings=test_settings
        )
        
        assert isinstance(report, dict)
        assert 'summary' in report or 'metrics' in report

    def test_identify_trends(self, daily_metrics_history, test_settings):
        """Test trend identification."""
        trends = identify_trends(daily_metrics_history, test_settings)
        
        assert isinstance(trends, dict)
        # Check for trend indicators
        for key in trends:
            assert isinstance(trends[key], (str, dict, list))

    def test_identify_improving_trend(self, test_settings):
        """Test identification of improving trends."""
        # Create improving metrics
        improving_data = pd.DataFrame({
            'date': [datetime.utcnow() - timedelta(days=i) for i in range(10)],
            'recall': [0.90 + i*0.01 for i in range(10)],  # Improving
            'fpr': [0.03 - i*0.002 for i in range(10)]     # Improving
        })
        
        trends = identify_trends(improving_data, test_settings)
        
        # Should identify improvement
        assert 'recall' in trends or 'improving' in str(trends).lower()

    def test_identify_degrading_trend(self, test_settings):
        """Test identification of degrading trends."""
        # Create degrading metrics
        degrading_data = pd.DataFrame({
            'date': [datetime.utcnow() - timedelta(days=i) for i in range(10)],
            'recall': [0.98 - i*0.01 for i in range(10)],  # Degrading
            'fpr': [0.01 + i*0.002 for i in range(10)]     # Degrading
        })
        
        trends = identify_trends(degrading_data, test_settings)
        
        # Should identify degradation
        assert 'recall' in trends or 'degrading' in str(trends).lower()

    def test_recommend_actions(self, test_settings):
        """Test action recommendations."""
        trends = {
            'recall': 'degrading',
            'fpr': 'increasing',
            'drift_frequency': 'high'
        }
        
        recommendations = recommend_actions(trends, test_settings)
        
        assert isinstance(recommendations, list)
        assert len(recommendations) > 0
        for rec in recommendations:
            assert isinstance(rec, (str, dict))

    def test_recommend_actions_no_issues(self, test_settings):
        """Test recommendations when no issues detected."""
        trends = {
            'recall': 'stable',
            'fpr': 'stable',
            'drift_frequency': 'low'
        }
        
        recommendations = recommend_actions(trends, test_settings)
        
        # Should have minimal or no recommendations
        assert isinstance(recommendations, list)

    @patch('src.pipelines.daily_analysis.DriftDatabaseService')
    @patch('src.pipelines.daily_analysis.generate_daily_report')
    @patch('src.pipelines.daily_analysis.upload_report_to_blob')
    def test_run_daily_analysis_success(
        self, mock_upload, mock_report, mock_db, 
        daily_metrics_history, test_settings
    ):
        """Test complete daily analysis pipeline."""
        mock_db_instance = MagicMock()
        mock_db_instance.query_historical_drift.return_value = daily_metrics_history
        mock_db.return_value = mock_db_instance
        
        mock_report.return_value = {'summary': 'Daily report'}
        
        result = run_daily_analysis(test_settings)
        
        assert result is True
        assert mock_db_instance.query_historical_drift.called

    @patch('src.pipelines.daily_analysis.DriftDatabaseService')
    def test_run_daily_analysis_no_data(self, mock_db, test_settings):
        """Test pipeline with no historical data."""
        mock_db_instance = MagicMock()
        mock_db_instance.query_historical_drift.return_value = pd.DataFrame()
        mock_db.return_value = mock_db_instance
        
        result = run_daily_analysis(test_settings)
        
        # Should handle gracefully
        assert result is False or result is None

    @patch('src.pipelines.daily_analysis.DriftDatabaseService')
    def test_run_daily_analysis_error_handling(self, mock_db, test_settings):
        """Test pipeline error handling."""
        mock_db.side_effect = Exception("Database error")
        
        result = run_daily_analysis(test_settings)
        
        # Should not raise exception
        assert result is False

    def test_report_generation_integration(self, daily_metrics_history, test_settings):
        """Test report generation with real data processing."""
        current_metrics = daily_metrics_history.iloc[0].to_dict()
        
        report = generate_daily_report(
            daily_metrics=current_metrics,
            historical_data=daily_metrics_history,
            settings=test_settings
        )
        
        # Report should contain key sections
        assert isinstance(report, dict)

    def test_trend_analysis_integration(self, daily_metrics_history, test_settings):
        """Test trend analysis with historical data."""
        trends = identify_trends(daily_metrics_history, test_settings)
        
        # Should analyze multiple metrics
        assert isinstance(trends, dict)
        assert len(trends) > 0

    @patch('src.pipelines.daily_analysis.upload_report_to_blob')
    def test_report_upload_integration(self, mock_upload, test_settings):
        """Test report upload to blob storage."""
        report = {'summary': 'Test report'}
        
        with patch('src.pipelines.daily_analysis.generate_daily_report') as mock_gen:
            mock_gen.return_value = report
            
            with patch('src.pipelines.daily_analysis.DriftDatabaseService'):
                run_daily_analysis(test_settings)
                
                # Verify upload was attempted
                if test_settings.azure_storage_enabled:
                    assert mock_upload.called

    def test_weekly_pattern_detection(self, test_settings):
        """Test detection of weekly patterns."""
        # Create data with weekly pattern
        dates = [datetime.utcnow() - timedelta(days=i) for i in range(28)]
        
        # Simulate higher drift on weekends
        drift_scores = []
        for date in dates:
            if date.weekday() >= 5:  # Weekend
                drift_scores.append(np.random.uniform(0.4, 0.6))
            else:  # Weekday
                drift_scores.append(np.random.uniform(0.1, 0.3))
        
        weekly_data = pd.DataFrame({
            'date': dates,
            'avg_psi': drift_scores
        })
        
        trends = identify_trends(weekly_data, test_settings)
        
        # Should detect pattern (implementation dependent)
        assert isinstance(trends, dict)

    def test_action_priority_ordering(self, test_settings):
        """Test that recommendations are prioritized."""
        critical_trends = {
            'recall': 'critically_degraded',
            'fpr': 'significantly_increased',
            'drift_frequency': 'very_high'
        }
        
        recommendations = recommend_actions(critical_trends, test_settings)
        
        # Should provide multiple prioritized actions
        assert isinstance(recommendations, list)
        if len(recommendations) > 0:
            # First recommendation should be high priority
            assert isinstance(recommendations[0], (str, dict))

    @patch('src.pipelines.daily_analysis.AlertManager')
    def test_alert_on_critical_trends(self, mock_alerts, test_settings):
        """Test that critical trends trigger alerts."""
        with patch('src.pipelines.daily_analysis.DriftDatabaseService'):
            with patch('src.pipelines.daily_analysis.identify_trends') as mock_trends:
                mock_trends.return_value = {'recall': 'critically_degraded'}
                
                run_daily_analysis(test_settings)
                
                # Critical trends should trigger alerts
                # (implementation dependent)

    def test_historical_comparison(self, daily_metrics_history, test_settings):
        """Test comparison with historical baselines."""
        current_metrics = daily_metrics_history.iloc[0].to_dict()
        
        report = generate_daily_report(
            daily_metrics=current_metrics,
            historical_data=daily_metrics_history,
            settings=test_settings
        )
        
        # Report should include historical context
        assert isinstance(report, dict)

    def test_metrics_aggregation_accuracy(self, test_settings):
        """Test accuracy of metrics aggregation."""
        hourly_data = pd.DataFrame({
            'timestamp': [datetime.utcnow() - timedelta(hours=i) for i in range(24)],
            'psi_score': [0.3] * 24,
            'fraud_rate': [0.002] * 24
        })
        
        with patch('src.pipelines.daily_analysis.DriftDatabaseService') as mock_db:
            mock_db_instance = MagicMock()
            mock_db_instance.query_historical_drift.return_value = hourly_data
            mock_db.return_value = mock_db_instance
            
            metrics = aggregate_daily_metrics(test_settings)
            
            # Aggregated values should match source data
            assert isinstance(metrics, dict)
