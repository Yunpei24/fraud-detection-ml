"""
Unit tests for EvidentlyDriftService

Tests the core drift detection functionality using Evidently AI.
"""

import asyncio
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, Mock, patch

import numpy as np
import pandas as pd
import pytest
from src.services.evidently_drift_service import EvidentlyDriftService


@pytest.mark.unit
class TestEvidentlyDriftService:
    """Unit tests for EvidentlyDriftService methods."""

    @pytest.fixture
    def mock_db_service(self):
        """Mock database service."""
        return AsyncMock()

    @pytest.fixture
    def drift_service(self, mock_db_service):
        """Create EvidentlyDriftService instance with mocked database."""
        return EvidentlyDriftService(mock_db_service)

    @pytest.fixture
    def sample_current_data(self):
        """Sample current window data for testing."""
        np.random.seed(42)
        n_samples = 1000

        data = {
            "transaction_id": [f"TXN_{i:04d}" for i in range(n_samples)],
            "amount": np.random.normal(100, 20, n_samples),
            "is_fraud": np.random.choice([0, 1], n_samples, p=[0.95, 0.05]),
            "v1": np.random.normal(0, 1, n_samples),
            "v2": np.random.normal(0, 1, n_samples),
            "v3": np.random.normal(0, 1, n_samples),
            "transaction_type": np.random.choice(["online", "pos", "atm"], n_samples),
            "customer_country": np.random.choice(["US", "UK", "DE", "FR"], n_samples),
            "merchant_country": np.random.choice(["US", "UK", "DE", "FR"], n_samples),
            "currency": np.random.choice(["USD", "EUR", "GBP"], n_samples),
            "device_id": [f"DEV_{i%100:03d}" for i in range(n_samples)],
            "ip_address": [f"192.168.1.{i%255}" for i in range(n_samples)],
            "time": [datetime.utcnow() - timedelta(hours=i) for i in range(n_samples)],
        }

        # Add fraud_score column
        data["fraud_score"] = np.random.uniform(0, 1, n_samples)

        return pd.DataFrame(data)

    @pytest.fixture
    def sample_reference_data(self):
        """Sample reference window data for testing."""
        np.random.seed(123)
        n_samples = 5000

        data = {
            "transaction_id": [f"REF_{i:04d}" for i in range(n_samples)],
            "amount": np.random.normal(
                95, 15, n_samples
            ),  # Slightly different distribution
            "is_fraud": np.random.choice(
                [0, 1], n_samples, p=[0.97, 0.03]
            ),  # Lower fraud rate
            "v1": np.random.normal(0, 0.8, n_samples),  # Different variance
            "v2": np.random.normal(0, 0.8, n_samples),
            "v3": np.random.normal(0, 0.8, n_samples),
            "transaction_type": np.random.choice(
                ["online", "pos", "atm"], n_samples, p=[0.6, 0.3, 0.1]
            ),
            "customer_country": np.random.choice(
                ["US", "UK", "DE", "FR"], n_samples, p=[0.5, 0.3, 0.1, 0.1]
            ),
            "merchant_country": np.random.choice(
                ["US", "UK", "DE", "FR"], n_samples, p=[0.4, 0.4, 0.1, 0.1]
            ),
            "currency": np.random.choice(
                ["USD", "EUR", "GBP"], n_samples, p=[0.6, 0.3, 0.1]
            ),
            "device_id": [f"DEV_{i%200:03d}" for i in range(n_samples)],
            "ip_address": [f"10.0.0.{i%255}" for i in range(n_samples)],
            "time": [
                datetime.utcnow() - timedelta(days=i) for i in range(30, 30 + n_samples)
            ],
        }

        # Add fraud_score column
        data["fraud_score"] = np.random.uniform(0, 1, n_samples)

        return pd.DataFrame(data)

    def test_service_initialization(self, drift_service):
        """Test service initializes correctly."""
        assert drift_service.db is not None
        assert drift_service.column_mapping is not None
        assert hasattr(drift_service.column_mapping, "target")
        assert drift_service.column_mapping.target == "is_fraud"

    def test_calculate_psi_normal_case(self, drift_service):
        """Test PSI calculation with normal values."""
        expected = 0.05  # 5% fraud rate
        actual = 0.08  # 8% fraud rate

        psi = drift_service._calculate_psi(expected, actual)

        assert isinstance(psi, float)
        assert psi > 0  # Should be positive for drift

    def test_calculate_psi_edge_cases(self, drift_service):
        """Test PSI calculation with edge cases."""
        # Same values should give PSI = 0
        psi = drift_service._calculate_psi(0.05, 0.05)
        assert abs(psi) < 0.001

        # Zero values should be handled
        psi = drift_service._calculate_psi(0.0, 0.1)
        assert isinstance(psi, float)

    def test_generate_drift_summary_no_drift(self, drift_service):
        """Test drift summary generation when no drift is detected."""
        results = {
            "data_drift": {"drift_detected": False},
            "target_drift": {"drift_detected": False},
            "concept_drift": {"drift_detected": False},
            "multivariate_drift": {"drift_detected": False},
        }

        summary = drift_service._generate_drift_summary(results)

        assert summary["overall_drift_detected"] is False
        assert summary["severity_score"] == 0
        assert len(summary["drift_types_detected"]) == 0
        assert "LOW" in summary["recommendations"][0]

    def test_generate_drift_summary_with_drift(self, drift_service):
        """Test drift summary generation when drift is detected."""
        results = {
            "data_drift": {"drift_detected": True},
            "target_drift": {"drift_detected": True},
            "concept_drift": {"drift_detected": False},
            "multivariate_drift": {"drift_detected": True},
        }

        summary = drift_service._generate_drift_summary(results)

        assert summary["overall_drift_detected"] is True
        assert summary["severity_score"] == 3
        assert len(summary["drift_types_detected"]) == 3
        assert "CRITICAL" in summary["recommendations"][0]

    @pytest.mark.asyncio
    async def test_detect_data_drift_success(
        self, drift_service, sample_current_data, sample_reference_data
    ):
        """Test successful data drift detection."""
        result = await drift_service._detect_data_drift(
            sample_current_data, sample_reference_data
        )

        assert isinstance(result, dict)
        assert "dataset_drift_detected" in result
        assert "drift_share" in result
        assert "drifted_columns" in result
        assert isinstance(result["drifted_columns"], list)

    @pytest.mark.asyncio
    async def test_detect_target_drift_success(
        self, drift_service, sample_current_data, sample_reference_data
    ):
        """Test successful target drift detection."""
        result = await drift_service._detect_target_drift(
            sample_current_data, sample_reference_data
        )

        assert isinstance(result, dict)
        assert "drift_detected" in result
        assert "drift_score" in result
        assert "current_fraud_rate" in result
        assert "reference_fraud_rate" in result
        assert "stattest" in result

    @pytest.mark.asyncio
    async def test_detect_concept_drift_success(
        self, drift_service, sample_current_data, sample_reference_data
    ):
        """Test successful concept drift detection."""
        result = await drift_service._detect_concept_drift(
            sample_current_data, sample_reference_data
        )

        assert isinstance(result, dict)
        assert "drift_detected" in result
        assert "drift_score" in result
        assert "stattest_name" in result
        assert "features_analyzed" in result

    @pytest.mark.asyncio
    async def test_detect_multivariate_drift_success(
        self, drift_service, sample_current_data, sample_reference_data
    ):
        """Test successful multivariate drift detection."""
        result = await drift_service._detect_multivariate_drift(
            sample_current_data, sample_reference_data
        )

        assert isinstance(result, dict)
        assert "tests" in result
        assert "overall_drift_detected" in result
        assert "drift_columns_count" in result
        assert isinstance(result["tests"], list)

    @pytest.mark.asyncio
    async def test_comprehensive_drift_insufficient_data(
        self, drift_service, mock_db_service
    ):
        """Test comprehensive drift detection with insufficient data."""
        # Mock empty dataframes
        mock_db_service.fetch_all.return_value = []

        result = await drift_service.detect_comprehensive_drift(
            window_hours=24, reference_window_days=30
        )

        assert "error" in result
        assert "Insufficient data" in result["error"]

    @pytest.mark.asyncio
    async def test_comprehensive_drift_success(
        self, drift_service, mock_db_service, sample_current_data, sample_reference_data
    ):
        """Test successful comprehensive drift detection."""
        # Mock database calls
        mock_db_service.fetch_all.side_effect = [
            sample_current_data.to_dict("records"),  # Current window
            sample_reference_data.to_dict("records"),  # Reference window
        ]
        mock_db_service.execute = AsyncMock()

        result = await drift_service.detect_comprehensive_drift(
            window_hours=24, reference_window_days=30
        )

        assert "timestamp" in result
        assert "analysis_window" in result
        assert "reference_window" in result
        assert "data_drift" in result
        assert "target_drift" in result
        assert "concept_drift" in result
        assert "multivariate_drift" in result
        assert "drift_summary" in result

    @pytest.mark.asyncio
    async def test_sliding_window_analysis_success(
        self, drift_service, mock_db_service, sample_current_data, sample_reference_data
    ):
        """Test successful sliding window analysis."""
        # Mock database calls
        mock_db_service.fetch_all.side_effect = [
            sample_current_data.to_dict("records"),  # Window data
            sample_reference_data.to_dict("records"),  # Reference data
        ] * 10  # Multiple calls for sliding windows

        result = await drift_service.run_sliding_window_analysis(
            window_size_hours=24,
            step_hours=6,
            analysis_period_days=2,  # Small period for testing
        )

        assert "timestamp" in result
        assert "window_size" in result
        assert "step_size" in result
        assert "analysis_period" in result
        assert "windows" in result
        assert isinstance(result["windows"], list)

    @pytest.mark.asyncio
    async def test_generate_drift_report_no_alerts(
        self, drift_service, mock_db_service
    ):
        """Test drift report generation with no alerts."""
        analysis_results = {
            "drift_summary": {"overall_drift_detected": False, "severity_score": 0}
        }

        mock_db_service.execute = AsyncMock()

        report = await drift_service.generate_drift_report(analysis_results)

        assert "timestamp" in report
        assert "summary" in report
        assert "recommendations" in report
        assert "alerts" in report
        assert "severity" in report
        assert report["severity"] == "LOW"
        assert len(report["alerts"]) == 0

    @pytest.mark.asyncio
    async def test_generate_drift_report_with_alerts(
        self, drift_service, mock_db_service
    ):
        """Test drift report generation with alerts."""
        analysis_results = {
            "data_drift": {"dataset_drift_detected": True},
            "target_drift": {"drift_detected": True},
            "concept_drift": {"drift_detected": True},
            "drift_summary": {"overall_drift_detected": True, "severity_score": 3},
        }

        mock_db_service.execute = AsyncMock()

        report = await drift_service.generate_drift_report(analysis_results)

        assert report["severity"] == "CRITICAL"
        assert len(report["alerts"]) >= 2  # Should have multiple alerts
        assert len(report["recommendations"]) >= 1

    @pytest.mark.asyncio
    async def test_error_handling_in_comprehensive_drift(
        self, drift_service, mock_db_service
    ):
        """Test error handling in comprehensive drift detection."""
        # Mock database to raise exception
        mock_db_service.fetch_all.side_effect = Exception("Database connection failed")

        result = await drift_service.detect_comprehensive_drift()

        assert "error" in result
        assert "Database connection failed" in result["error"]

    @pytest.mark.asyncio
    async def test_quick_drift_check(
        self, drift_service, sample_current_data, sample_reference_data
    ):
        """Test quick drift check for sliding windows."""
        result = await drift_service._quick_drift_check(
            sample_current_data, sample_reference_data
        )

        assert isinstance(result, dict)
        assert "drift_detected" in result
        assert "drift_score" in result
        assert isinstance(result["drift_detected"], bool)
        assert isinstance(result["drift_score"], (int, float))
