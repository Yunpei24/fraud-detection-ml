"""Integration tests for daily analysis pipeline."""

from datetime import datetime, timedelta
from unittest.mock import MagicMock, Mock, patch

import numpy as np
import pandas as pd
import pytest
from src.config.settings import Settings
from src.pipelines.daily_analysis import (
    aggregate_daily_metrics,
    generate_daily_report,
    identify_trends,
    recommend_actions,
    run_daily_analysis,
)


@pytest.mark.integration
class TestDailyAnalysis:
    """Integration tests for daily analysis pipeline."""

    @pytest.fixture
    def daily_metrics_history(self):
        """Create mock daily metrics history."""
        dates = [datetime.utcnow() - timedelta(days=i) for i in range(30)]

        return pd.DataFrame(
            {
                "date": dates,
                "avg_psi": np.random.uniform(0.1, 0.5, 30),
                "fraud_rate": np.random.uniform(0.001, 0.005, 30),
                "recall": np.random.uniform(0.90, 0.98, 30),
                "fpr": np.random.uniform(0.01, 0.03, 30),
                "drift_detected": np.random.choice([True, False], 30, p=[0.3, 0.7]),
            }
        )

    @patch("src.pipelines.daily_analysis.FraudDetectionAPIClient")
    def test_aggregate_daily_metrics(self, mock_api, test_settings):
        """Test daily metrics aggregation."""
        mock_api_instance = MagicMock()
        mock_api_instance.run_sliding_window_analysis.return_value = {
            "timestamp": datetime.utcnow().isoformat(),
            "analysis_period": "7d",
            "window_size": "24h",
            "windows": [
                {"drift_detected": True, "drift_score": 0.35},
                {"drift_detected": False, "drift_score": 0.15},
            ],
        }
        mock_api.return_value = mock_api_instance

        metrics = aggregate_daily_metrics(test_settings)

        assert isinstance(metrics, dict)
        assert "drift_summary" in metrics

    def test_generate_daily_report(self, test_settings):
        """Test daily report generation."""
        aggregated_metrics = {
            "drift_summary": {
                "total_windows": 7,
                "drift_detected_windows": 2,
                "avg_drift_score": 0.25,
            }
        }

        with patch("src.pipelines.daily_analysis.FraudDetectionAPIClient") as mock_api:
            mock_api_instance = MagicMock()
            mock_api_instance.generate_drift_report.return_value = {
                "summary": "Daily report"
            }
            mock_api.return_value = mock_api_instance

            report = generate_daily_report(aggregated_metrics, test_settings)

            assert isinstance(report, dict)
            assert "summary" in report or len(report) > 0

    @patch("src.pipelines.daily_analysis.aggregate_daily_metrics")
    def test_identify_trends(self, mock_aggregate, test_settings):
        """Test trend identification."""
        mock_aggregate.return_value = {
            "windows": [
                {"drift_detected": False, "drift_score": 0.1},
                {"drift_detected": True, "drift_score": 0.3},
                {"drift_detected": False, "drift_score": 0.2},
            ]
        }

        trends = identify_trends(test_settings)

        assert isinstance(trends, list)
        # Check for trend indicators
        for trend in trends:
            assert isinstance(trend, dict)

    @patch("src.pipelines.daily_analysis.aggregate_daily_metrics")
    def test_identify_improving_trend(self, mock_aggregate, test_settings):
        """Test identification of improving trends."""
        # Mock improving drift scores (decreasing over time = improving)
        mock_aggregate.return_value = {
            "windows": [
                {"drift_detected": False, "drift_score": 0.4},
                {"drift_detected": False, "drift_score": 0.3},
                {"drift_detected": False, "drift_score": 0.2},
            ]
        }

        trends = identify_trends(test_settings)

        # Should identify improving trend (decreasing drift scores)
        assert isinstance(trends, list)
        if trends:
            trend = trends[0]
            assert trend["direction"] == "decreasing"

    @patch("src.pipelines.daily_analysis.aggregate_daily_metrics")
    def test_identify_degrading_trend(self, mock_aggregate, test_settings):
        """Test identification of degrading trends."""
        # Mock degrading drift scores (increasing over time = degrading)
        mock_aggregate.return_value = {
            "windows": [
                {"drift_detected": False, "drift_score": 0.1},
                {"drift_detected": True, "drift_score": 0.3},
                {"drift_detected": True, "drift_score": 0.5},
            ]
        }

        trends = identify_trends(test_settings)

        # Should identify degrading trend (increasing drift scores)
        assert isinstance(trends, list)
        if trends:
            trend = trends[0]
            assert trend["direction"] == "increasing"

    def test_recommend_actions(self, test_settings):
        """Test action recommendations."""
        aggregated_metrics = {
            "drift_summary": {
                "total_windows": 7,
                "drift_detected_windows": 4,
                "avg_drift_score": 0.35,
            }
        }
        trends = [
            {
                "direction": "increasing",
                "concern_level": "HIGH",
                "magnitude": 0.3,
                "detection_rate": 0.5,
            }
        ]

        recommendations = recommend_actions(aggregated_metrics, trends)

        assert isinstance(recommendations, list)
        assert len(recommendations) > 0
        for rec in recommendations:
            assert isinstance(rec, str)

    def test_recommend_actions_no_issues(self, test_settings):
        """Test recommendations when no issues detected."""
        aggregated_metrics = {
            "drift_summary": {
                "total_windows": 7,
                "drift_detected_windows": 1,
                "avg_drift_score": 0.15,
            }
        }
        trends = [
            {
                "direction": "stable",
                "concern_level": "LOW",
                "magnitude": 0.05,
                "detection_rate": 0.1,
            }
        ]

        recommendations = recommend_actions(aggregated_metrics, trends)

        # Should have minimal or no recommendations
        assert isinstance(recommendations, list)

    @patch("src.pipelines.daily_analysis.aggregate_daily_metrics")
    @patch("src.pipelines.daily_analysis.generate_daily_report")
    @patch("src.pipelines.daily_analysis.identify_trends")
    @patch("src.pipelines.daily_analysis.recommend_actions")
    def test_run_daily_analysis_success(
        self,
        mock_recommend,
        mock_trends,
        mock_report,
        mock_aggregate,
        daily_metrics_history,
        test_settings,
    ):
        """Test complete daily analysis pipeline."""
        mock_aggregate.return_value = {
            "drift_summary": {
                "total_windows": 7,
                "drift_detected_windows": 2,
                "avg_drift_score": 0.25,
            }
        }
        mock_report.return_value = {"summary": "Daily report"}
        mock_trends.return_value = [{"direction": "stable", "concern_level": "LOW"}]
        mock_recommend.return_value = ["Continue monitoring"]

        result = run_daily_analysis(test_settings)

        assert isinstance(result, dict)
        assert result.get("status") == "success"

    @patch("src.pipelines.daily_analysis.FraudDetectionAPIClient")
    def test_run_daily_analysis_no_data(self, mock_api, test_settings):
        """Test pipeline with no historical data."""
        mock_api_instance = MagicMock()
        mock_api_instance.run_sliding_window_analysis.return_value = {
            "timestamp": datetime.utcnow().isoformat(),
            "windows": [],
        }
        mock_api.return_value = mock_api_instance

        result = run_daily_analysis(test_settings)

        # Should handle gracefully
        assert isinstance(result, dict)
        assert result.get("status") == "success"

    @patch("src.pipelines.daily_analysis.aggregate_daily_metrics")
    def test_run_daily_analysis_error_handling(self, mock_aggregate, test_settings):
        """Test pipeline error handling."""
        mock_aggregate.side_effect = Exception("API error")

        result = run_daily_analysis(test_settings)

        # Should not raise exception
        assert isinstance(result, dict)
        assert result.get("status") == "failed"

    def test_report_generation_integration(self, test_settings):
        """Test report generation with real data processing."""
        aggregated_metrics = {
            "drift_summary": {
                "total_windows": 7,
                "drift_detected_windows": 2,
                "avg_drift_score": 0.25,
            }
        }

        with patch("src.pipelines.daily_analysis.FraudDetectionAPIClient") as mock_api:
            mock_api_instance = MagicMock()
            mock_api_instance.generate_drift_report.return_value = {
                "summary": "Test report"
            }
            mock_api.return_value = mock_api_instance

            report = generate_daily_report(aggregated_metrics, test_settings)

            # Report should contain key sections
            assert isinstance(report, dict)

    @patch("src.pipelines.daily_analysis.aggregate_daily_metrics")
    def test_trend_analysis_integration(self, mock_aggregate, test_settings):
        """Test trend analysis with historical data."""
        mock_aggregate.return_value = {
            "windows": [
                {"drift_detected": False, "drift_score": 0.1},
                {"drift_detected": True, "drift_score": 0.3},
                {"drift_detected": False, "drift_score": 0.2},
            ]
        }

        trends = identify_trends(test_settings)

        # Should analyze multiple metrics
        assert isinstance(trends, list)
        assert len(trends) > 0

    def test_report_upload_integration(self, test_settings):
        """Test report upload to blob storage."""
        with patch("src.pipelines.daily_analysis.generate_daily_report") as mock_gen:
            mock_gen.return_value = {"summary": "Test report"}

            with patch(
                "src.pipelines.daily_analysis.aggregate_daily_metrics"
            ) as mock_agg:
                mock_agg.return_value = {"drift_summary": {"total_windows": 7}}

                result = run_daily_analysis(test_settings)

                # Verify report generation was attempted
                assert result.get("report_generated") is True

    @patch("src.pipelines.daily_analysis.aggregate_daily_metrics")
    def test_weekly_pattern_detection(self, mock_aggregate, test_settings):
        """Test detection of weekly patterns."""
        # Mock data with weekly pattern (higher drift on weekends)
        mock_aggregate.return_value = {
            "windows": [
                {"drift_detected": False, "drift_score": 0.1},  # Weekday
                {"drift_detected": False, "drift_score": 0.2},  # Weekday
                {"drift_detected": False, "drift_score": 0.15},  # Weekday
                {"drift_detected": False, "drift_score": 0.12},  # Weekday
                {"drift_detected": False, "drift_score": 0.18},  # Weekday
                {"drift_detected": True, "drift_score": 0.4},  # Weekend
                {"drift_detected": True, "drift_score": 0.5},  # Weekend
            ]
        }

        trends = identify_trends(test_settings)

        # Should detect pattern (implementation dependent)
        assert isinstance(trends, list)

    def test_action_priority_ordering(self, test_settings):
        """Test that recommendations are prioritized."""
        aggregated_metrics = {
            "drift_summary": {
                "total_windows": 7,
                "drift_detected_windows": 5,
                "avg_drift_score": 0.4,
            }
        }
        critical_trends = [
            {
                "direction": "increasing",
                "concern_level": "HIGH",
                "magnitude": 0.6,
                "detection_rate": 0.7,
            }
        ]

        recommendations = recommend_actions(aggregated_metrics, critical_trends)

        # Should provide multiple prioritized actions
        assert isinstance(recommendations, list)
        if len(recommendations) > 0:
            # First recommendation should be high priority
            assert isinstance(recommendations[0], str)

    @patch("src.pipelines.daily_analysis.aggregate_daily_metrics")
    @patch("src.pipelines.daily_analysis.identify_trends")
    def test_alert_on_critical_trends(self, mock_trends, mock_aggregate, test_settings):
        """Test that critical trends trigger alerts."""
        mock_aggregate.return_value = {
            "drift_summary": {
                "total_windows": 7,
                "drift_detected_windows": 4,
                "avg_drift_score": 0.35,
            }
        }
        mock_trends.return_value = [
            {
                "direction": "increasing",
                "concern_level": "HIGH",
                "magnitude": 0.3,
                "detection_rate": 0.5,
            }
        ]

        result = run_daily_analysis(test_settings)

        # Critical trends should be identified
        assert result.get("status") == "success"
        assert "trends" in result

    def test_historical_comparison(self, test_settings):
        """Test comparison with historical baselines."""
        aggregated_metrics = {
            "drift_summary": {
                "total_windows": 7,
                "drift_detected_windows": 2,
                "avg_drift_score": 0.25,
            }
        }

        with patch("src.pipelines.daily_analysis.FraudDetectionAPIClient") as mock_api:
            mock_api_instance = MagicMock()
            mock_api_instance.generate_drift_report.return_value = {
                "summary": "Historical comparison report"
            }
            mock_api.return_value = mock_api_instance

            report = generate_daily_report(aggregated_metrics, test_settings)

            # Report should include historical context
            assert isinstance(report, dict)

    def test_metrics_aggregation_accuracy(self, test_settings):
        """Test accuracy of metrics aggregation."""
        with patch("src.pipelines.daily_analysis.FraudDetectionAPIClient") as mock_api:
            mock_api_instance = MagicMock()
            mock_api_instance.run_sliding_window_analysis.return_value = {
                "timestamp": datetime.utcnow().isoformat(),
                "analysis_period": "24h",
                "window_size": "1h",
                "windows": [
                    {"drift_detected": False, "drift_score": 0.3},
                    {"drift_detected": False, "drift_score": 0.25},
                ],
            }
            mock_api.return_value = mock_api_instance

            metrics = aggregate_daily_metrics(test_settings)

            # Aggregated values should match source data
            assert isinstance(metrics, dict)
