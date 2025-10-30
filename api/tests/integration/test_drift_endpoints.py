"""
Integration tests for Evidently Drift Detection API endpoints

Tests the complete flow from API endpoints to database operations.
"""

from datetime import datetime, timedelta
from unittest.mock import AsyncMock, Mock, patch

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from src.api.dependencies import get_database_service
from src.main import app


@pytest.fixture
def client():
    """Test client with mocked dependencies."""
    # Mock the database service
    mock_db_service = AsyncMock()

    # Mock successful data retrieval
    mock_current_data = create_mock_transaction_data(1000, fraud_rate=0.05)
    mock_reference_data = create_mock_transaction_data(5000, fraud_rate=0.03)

    mock_db_service.fetch_all.side_effect = [
        mock_current_data,  # Current window data
        mock_reference_data,  # Reference window data
        mock_current_data,  # Window data for sliding analysis
        mock_reference_data,  # Reference for sliding analysis
    ] * 10  # Repeat for multiple calls

    mock_db_service.execute = AsyncMock()

    app.dependency_overrides[get_database_service] = lambda: mock_db_service

    with TestClient(app) as test_client:
        yield test_client

    # Clean up
    app.dependency_overrides.clear()


def create_mock_transaction_data(n_samples: int, fraud_rate: float = 0.05):
    """Create mock transaction data for testing."""
    np.random.seed(42)

    data = []
    for i in range(n_samples):
        transaction = {
            "transaction_id": f"TXN_{i:06d}",
            "amount": float(np.random.normal(100, 20)),
            "currency": "USD",
            "time": datetime.utcnow() - timedelta(hours=i),
            "is_fraud": int(np.random.random() < fraud_rate),
            "v1": float(np.random.normal(0, 1)),
            "v2": float(np.random.normal(0, 1)),
            "v3": float(np.random.normal(0, 1)),
            "v4": float(np.random.normal(0, 1)),
            "v5": float(np.random.normal(0, 1)),
            "v6": float(np.random.normal(0, 1)),
            "v7": float(np.random.normal(0, 1)),
            "v8": float(np.random.normal(0, 1)),
            "v9": float(np.random.normal(0, 1)),
            "v10": float(np.random.normal(0, 1)),
            "v11": float(np.random.normal(0, 1)),
            "v12": float(np.random.normal(0, 1)),
            "v13": float(np.random.normal(0, 1)),
            "v14": float(np.random.normal(0, 1)),
            "v15": float(np.random.normal(0, 1)),
            "v16": float(np.random.normal(0, 1)),
            "v17": float(np.random.normal(0, 1)),
            "v18": float(np.random.normal(0, 1)),
            "v19": float(np.random.normal(0, 1)),
            "v20": float(np.random.normal(0, 1)),
            "v21": float(np.random.normal(0, 1)),
            "v22": float(np.random.normal(0, 1)),
            "v23": float(np.random.normal(0, 1)),
            "v24": float(np.random.normal(0, 1)),
            "v25": float(np.random.normal(0, 1)),
            "v26": float(np.random.normal(0, 1)),
            "v27": float(np.random.normal(0, 1)),
            "v28": float(np.random.normal(0, 1)),
            "transaction_type": np.random.choice(["online", "pos", "atm"]),
            "customer_country": "US",
            "merchant_country": "US",
            "device_id": f"DEV_{i%100:03d}",
            "ip_address": f"192.168.1.{i%255}",
            "fraud_score": float(np.random.uniform(0, 1)),
        }
        data.append(transaction)

    return data


@pytest.mark.integration
class TestComprehensiveDriftEndpoint:
    """Integration tests for comprehensive drift detection endpoint."""

    def test_comprehensive_drift_success(self, client):
        """Test successful comprehensive drift detection."""
        response = client.post("/api/v1/drift/comprehensive-detect")

        assert response.status_code == 200
        data = response.json()

        # Validate response structure
        assert "timestamp" in data
        assert "analysis_window" in data
        assert "reference_window" in data
        assert "data_drift" in data
        assert "target_drift" in data
        assert "concept_drift" in data
        assert "multivariate_drift" in data
        assert "drift_summary" in data
        assert "processing_time" in data

        # Validate drift summary structure
        summary = data["drift_summary"]
        assert "overall_drift_detected" in summary
        assert "drift_types_detected" in summary
        assert "severity_score" in summary
        assert "recommendations" in summary

    def test_comprehensive_drift_custom_parameters(self, client):
        """Test comprehensive drift detection with custom parameters."""
        params = {"window_hours": 12, "reference_window_days": 15}

        response = client.post("/api/v1/drift/comprehensive-detect", params=params)

        assert response.status_code == 200
        data = response.json()

        assert data["analysis_window"] == "12h"
        assert data["reference_window"] == "15d"

    def test_comprehensive_drift_data_drift_structure(self, client):
        """Test data drift detection structure in comprehensive results."""
        response = client.post("/api/v1/drift/comprehensive-detect")

        assert response.status_code == 200
        data = response.json()

        data_drift = data["data_drift"]
        assert "dataset_drift_detected" in data_drift
        assert "drift_share" in data_drift
        assert "drifted_columns" in data_drift
        assert isinstance(data_drift["drifted_columns"], list)

    def test_comprehensive_drift_target_drift_structure(self, client):
        """Test target drift detection structure in comprehensive results."""
        response = client.post("/api/v1/drift/comprehensive-detect")

        assert response.status_code == 200
        data = response.json()

        target_drift = data["target_drift"]
        assert "drift_detected" in target_drift
        assert "drift_score" in target_drift
        assert "current_fraud_rate" in target_drift
        assert "reference_fraud_rate" in target_drift
        assert "stattest" in target_drift

    def test_comprehensive_drift_concept_drift_structure(self, client):
        """Test concept drift detection structure in comprehensive results."""
        response = client.post("/api/v1/drift/comprehensive-detect")

        assert response.status_code == 200
        data = response.json()

        concept_drift = data["concept_drift"]
        assert "drift_detected" in concept_drift
        assert "drift_score" in concept_drift
        assert "stattest_name" in concept_drift
        assert "features_analyzed" in concept_drift


@pytest.mark.integration
class TestSlidingWindowAnalysisEndpoint:
    """Integration tests for sliding window analysis endpoint."""

    def test_sliding_window_analysis_success(self, client):
        """Test successful sliding window analysis."""
        response = client.post("/api/v1/drift/sliding-window-analysis")

        assert response.status_code == 200
        data = response.json()

        # Validate response structure
        assert "timestamp" in data
        assert "window_size" in data
        assert "step_size" in data
        assert "analysis_period" in data
        assert "windows" in data
        assert "processing_time" in data

        # Validate windows structure
        assert isinstance(data["windows"], list)
        if len(data["windows"]) > 0:
            window = data["windows"][0]
            assert "window_id" in window
            assert "start_time" in window
            assert "end_time" in window
            assert "record_count" in window
            assert "drift_detected" in window
            assert "drift_score" in window

    def test_sliding_window_analysis_custom_parameters(self, client):
        """Test sliding window analysis with custom parameters."""
        params = {"window_size_hours": 8, "step_hours": 4, "analysis_period_days": 3}

        response = client.post("/api/v1/drift/sliding-window-analysis", params=params)

        assert response.status_code == 200
        data = response.json()

        assert data["window_size"] == "8h"
        assert data["step_size"] == "4h"
        assert data["analysis_period"] == "3d"

    def test_sliding_window_analysis_window_structure(self, client):
        """Test individual window structure in sliding analysis."""
        response = client.post("/api/v1/drift/sliding-window-analysis")

        assert response.status_code == 200
        data = response.json()

        for window in data["windows"]:
            assert isinstance(window["window_id"], int)
            assert isinstance(window["record_count"], int)
            assert isinstance(window["drift_detected"], bool)
            assert isinstance(window["drift_score"], (int, float))


@pytest.mark.integration
class TestDriftReportEndpoint:
    """Integration tests for drift report generation endpoint."""

    def test_generate_drift_report_success(self, client):
        """Test successful drift report generation."""
        # First get comprehensive drift results to use as input
        drift_response = client.post("/api/v1/drift/comprehensive-detect")
        assert drift_response.status_code == 200
        drift_results = drift_response.json()

        # Generate report using the drift results
        report_response = client.post(
            "/api/v1/drift/generate-report", json=drift_results
        )

        assert report_response.status_code == 200
        report = report_response.json()

        # Validate report structure
        assert "timestamp" in report
        assert "summary" in report
        assert "recommendations" in report
        assert "alerts" in report
        assert "severity" in report

        # Validate severity is valid
        assert report["severity"] in ["LOW", "MEDIUM", "HIGH", "CRITICAL"]

        # Validate recommendations and alerts are lists
        assert isinstance(report["recommendations"], list)
        assert isinstance(report["alerts"], list)

    def test_generate_drift_report_with_drift_alerts(self, client):
        """Test drift report generation includes appropriate alerts."""
        # Create mock drift results with detected drift
        drift_results = {
            "data_drift": {"dataset_drift_detected": True},
            "target_drift": {"drift_detected": True},
            "concept_drift": {"drift_detected": False},
            "drift_summary": {"overall_drift_detected": True, "severity_score": 2},
        }

        response = client.post("/api/v1/drift/generate-report", json=drift_results)

        assert response.status_code == 200
        report = response.json()

        # Should have alerts for detected drift types
        assert len(report["alerts"]) >= 2  # Data and target drift alerts
        assert report["severity"] in ["HIGH", "CRITICAL"]

        # Check alert structure
        for alert in report["alerts"]:
            assert "type" in alert
            assert "severity" in alert
            assert "message" in alert

    def test_generate_drift_report_no_drift(self, client):
        """Test drift report generation when no drift is detected."""
        drift_results = {
            "data_drift": {"dataset_drift_detected": False},
            "target_drift": {"drift_detected": False},
            "concept_drift": {"drift_detected": False},
            "drift_summary": {"overall_drift_detected": False, "severity_score": 0},
        }

        response = client.post("/api/v1/drift/generate-report", json=drift_results)

        assert response.status_code == 200
        report = response.json()

        assert report["severity"] == "LOW"
        assert len(report["alerts"]) == 0
        assert len(report["recommendations"]) >= 1


@pytest.mark.integration
class TestDriftEndpointErrorHandling:
    """Integration tests for error handling in drift endpoints."""

    def test_comprehensive_drift_database_error(self, client):
        """Test comprehensive drift detection with database error."""
        # Override with failing database service
        failing_db = AsyncMock()
        failing_db.fetch_all.side_effect = Exception("Database connection failed")

        app.dependency_overrides[get_database_service] = lambda: failing_db

        try:
            response = client.post("/api/v1/drift/comprehensive-detect")

            assert response.status_code == 500
            data = response.json()
            assert "detail" in data
            assert "error_code" in data["detail"]
            assert "E704" in data["detail"]["error_code"]
        finally:
            app.dependency_overrides.clear()

    def test_sliding_window_analysis_database_error(self, client):
        """Test sliding window analysis with database error."""
        # Override with failing database service
        failing_db = AsyncMock()
        failing_db.fetch_all.side_effect = Exception("Database connection failed")

        app.dependency_overrides[get_database_service] = lambda: failing_db

        try:
            response = client.post("/api/v1/drift/sliding-window-analysis")

            assert response.status_code == 500
            data = response.json()
            assert "detail" in data
            assert "error_code" in data["detail"]
            assert "E705" in data["detail"]["error_code"]
        finally:
            app.dependency_overrides.clear()

    def test_generate_report_invalid_input(self, client):
        """Test drift report generation with invalid input."""
        invalid_input = {"invalid": "data"}

        response = client.post("/api/v1/drift/generate-report", json=invalid_input)

        assert response.status_code == 500
        data = response.json()
        assert "detail" in data
        assert "error_code" in data["detail"]
        assert "E706" in data["detail"]["error_code"]


@pytest.mark.integration
class TestDriftEndpointAuthentication:
    """Integration tests for authentication on drift endpoints."""

    def test_comprehensive_drift_requires_auth(self, client):
        """Test that comprehensive drift endpoint requires authentication."""
        # Remove any existing overrides to test auth
        app.dependency_overrides.clear()

        response = client.post("/api/v1/drift/comprehensive-detect")

        # Should fail with 401 or 403 due to missing auth
        assert response.status_code in [401, 403, 422]  # 422 for missing auth header

    def test_sliding_window_requires_auth(self, client):
        """Test that sliding window endpoint requires authentication."""
        app.dependency_overrides.clear()

        response = client.post("/api/v1/drift/sliding-window-analysis")

        assert response.status_code in [401, 403, 422]

    def test_generate_report_requires_auth(self, client):
        """Test that generate report endpoint requires authentication."""
        app.dependency_overrides.clear()

        response = client.post("/api/v1/drift/generate-report", json={})

        assert response.status_code in [401, 403, 422]
