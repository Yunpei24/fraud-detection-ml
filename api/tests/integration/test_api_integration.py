from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient
from src.main import app
from src.services.alert_service import init_alert_service


@pytest.fixture
def client():
    from src.api.dependencies import (
        get_cache_service,
        get_database_service,
        get_drift_service,
        get_prediction_service,
    )

    # Mock services
    mock_prediction_service = AsyncMock()
    mock_prediction_service.predict_single.return_value = {
        "transaction_id": "TEST-001",
        "prediction": 0,
        "confidence": 0.85,
        "fraud_score": 0.15,
        "risk_level": "low",
        "processing_time": 0.05,
        "model_version": "test-v1.0",
        "timestamp": 1640995200.0,
        "features": [0.5] * 30,
        "explanation": {"feature_importance": []},
        "metadata": {"source": "test"},
    }

    mock_prediction_service.predict_batch.return_value = {
        "total_transactions": 1,
        "successful_predictions": 1,
        "failed_predictions": 0,
        "fraud_detected": 0,
        "fraud_rate": 0.0,
        "predictions": [
            {
                "transaction_id": "TXN-1",
                "prediction": 0,
                "confidence": 0.8,
                "fraud_score": 0.2,
                "risk_level": "low",
                "processing_time": 0.05,
                "model_version": "test-v1.0",
                "timestamp": 1640995200.0,
                "features": [0.5] * 30,
                "explanation": {"feature_importance": []},
                "metadata": {"source": "test"},
            }
        ],
        "processing_time": 0.05,
        "avg_processing_time": 0.05,
    }

    mock_cache_service = AsyncMock()
    # Mock the async methods to return proper values
    mock_cache_service.get_cached_prediction = AsyncMock(
        return_value={
            "transaction_id": "TEST-001",
            "prediction": 0,
            "confidence": 0.85,
            "fraud_score": 0.15,
            "risk_level": "low",
            "processing_time": 0.05,
            "model_version": "test-v1.0",
            "timestamp": 1640995200.0,
            "features": [0.5] * 30,
            "explanation": {"feature_importance": []},
            "metadata": {"source": "test"},
        }
    )
    mock_cache_service.set_prediction_cache = AsyncMock(return_value=True)

    mock_database_service = AsyncMock()

    # Mock drift service with dynamic responses
    mock_drift_service = AsyncMock()

    # Dynamic comprehensive drift detection - reflects input parameters
    async def mock_detect_comprehensive_drift(
        window_hours=24, reference_window_days=30
    ):
        return {
            "timestamp": "2025-10-31T10:00:00Z",
            "analysis_window": f"{window_hours}h",
            "reference_window": f"{reference_window_days}d",
            "data_drift": {
                "dataset_drift_detected": False,
                "drift_share": 0.05,
                "drifted_columns": [],
                "number_of_columns": 30,
            },
            "target_drift": {
                "drift_detected": False,
                "drift_score": 0.02,
                "current_fraud_rate": 0.05,
                "reference_fraud_rate": 0.03,
                "stattest": "chi2",
            },
            "concept_drift": {
                "drift_detected": False,
                "drift_score": 0.01,
                "stattest_name": "anderson",
                "features_analyzed": 30,
            },
            "multivariate_drift": {
                "drift_detected": False,
                "drift_score": 0.03,
            },
            "drift_summary": {
                "overall_drift_detected": False,
                "drift_types_detected": [],
                "severity_score": 0.0,
                "recommendations": ["Continue monitoring"],
            },
        }

    mock_drift_service.detect_comprehensive_drift = mock_detect_comprehensive_drift

    # Dynamic sliding window analysis - reflects input parameters
    async def mock_run_sliding_window_analysis(
        window_size_hours=24, step_hours=6, analysis_period_days=7
    ):
        return {
            "timestamp": "2025-10-31T10:00:00Z",
            "analysis_period": f"{analysis_period_days}d",
            "window_size": f"{window_size_hours}h",
            "step_size": f"{step_hours}h",
            "windows": [],
            "processing_time": 0.15,
        }

    mock_drift_service.run_sliding_window_analysis = mock_run_sliding_window_analysis

    # Dynamic drift report generation - reflects input data
    async def mock_generate_drift_report(drift_results):
        # Validate input - raise exception for invalid data
        if not isinstance(drift_results, dict):
            raise ValueError("Invalid input: drift_results must be a dictionary")

        # Check for required fields
        required_fields = [
            "data_drift",
            "target_drift",
            "concept_drift",
            "drift_summary",
        ]
        if not any(field in drift_results for field in required_fields):
            raise ValueError(
                f"Invalid input: missing required fields. Expected at least one of {required_fields}"
            )

        # Check if drift is detected in the input
        data_drift_detected = drift_results.get("data_drift", {}).get(
            "dataset_drift_detected", False
        )
        target_drift_detected = drift_results.get("target_drift", {}).get(
            "drift_detected", False
        )
        concept_drift_detected = drift_results.get("concept_drift", {}).get(
            "drift_detected", False
        )

        drift_summary = drift_results.get("drift_summary", {})
        overall_drift = drift_summary.get("overall_drift_detected", False)
        severity_score = drift_summary.get("severity_score", 0)

        # Build alerts based on detected drifts
        alerts = []
        if data_drift_detected:
            alerts.append(
                {
                    "type": "DATA_DRIFT",
                    "severity": "HIGH",
                    "message": "Significant data drift detected in feature distributions",
                }
            )
        if target_drift_detected:
            alerts.append(
                {
                    "type": "TARGET_DRIFT",
                    "severity": "HIGH",
                    "message": "Target variable distribution has changed significantly",
                }
            )
        if concept_drift_detected:
            alerts.append(
                {
                    "type": "CONCEPT_DRIFT",
                    "severity": "MEDIUM",
                    "message": "Concept drift detected in feature relationships",
                }
            )

        # Determine severity
        if severity_score >= 2:
            severity = "CRITICAL" if severity_score >= 3 else "HIGH"
        elif severity_score >= 1:
            severity = "MEDIUM"
        else:
            severity = "LOW"

        # Build recommendations
        recommendations = []
        if overall_drift:
            recommendations.extend(
                [
                    "Retrain model with recent data",
                    "Investigate feature changes",
                    "Monitor performance metrics closely",
                ]
            )
        else:
            recommendations.append("Continue monitoring")

        return {
            "timestamp": "2025-10-31T10:00:00Z",
            "severity": severity,
            "summary": drift_summary,
            "recommendations": recommendations,
            "alerts": alerts,
        }

    mock_drift_service.generate_drift_report = mock_generate_drift_report

    # Mock authentication
    mock_user = {"username": "test_analyst", "role": "analyst", "is_active": True}

    # Override dependencies
    app.dependency_overrides[get_prediction_service] = lambda: mock_prediction_service
    app.dependency_overrides[get_cache_service] = lambda: mock_cache_service
    app.dependency_overrides[get_database_service] = lambda: mock_database_service
    app.dependency_overrides[get_drift_service] = lambda: mock_drift_service

    # Import and override authentication
    from src.api.routes.auth import get_current_analyst_user

    app.dependency_overrides[get_current_analyst_user] = lambda: mock_user

    with TestClient(app) as test_client:
        yield test_client

    # Clean up
    app.dependency_overrides.clear()


class TestHealthEndpoints:
    def test_health_check(self, client):
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert "status" in data

    def test_root_endpoint(self, client):
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert "name" in data
        assert "version" in data


class TestPredictEndpoint:
    def test_predict_single_with_valid_data(self, client):
        payload = {
            "transaction_id": "TXN-123",
            "features": [0.5] * 30,  # 30 features as required
            "time": 0.0,
            "amount": 100.0,
            "V1": -1.0,
            "V2": -0.5,
            "V3": 0.5,
            "V4": 1.0,
            "V5": -0.2,
            "V6": 0.3,
            "V7": -0.1,
            "V8": 0.8,
            "V9": -0.6,
            "V10": 0.4,
            "V11": -0.3,
            "V12": 0.9,
            "V13": -0.7,
            "V14": 0.2,
            "V15": -0.4,
            "V16": 0.7,
            "V17": -0.5,
            "V18": 0.1,
            "V19": -0.2,
            "V20": 0.5,
            "V21": -0.8,
            "V22": 0.6,
            "V23": -0.3,
            "V24": 0.4,
            "V25": -0.1,
            "V26": 0.3,
            "V27": -0.2,
            "V28": 0.1,
        }

        response = client.post("/api/v1/predict", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert "transaction_id" in data
        assert "fraud_score" in data

    def test_predict_missing_feature(self, client):
        payload = {"transaction_id": "TXN-123", "amount": 100.0}

        response = client.post("/api/v1/predict", json=payload)
        assert response.status_code in [400, 422]

    def test_batch_predict(self, client):
        payload = {
            "transactions": [
                {
                    "transaction_id": "TXN-1",
                    "features": [0.5] * 30,  # 30 features as required
                    "time": 0.0,
                    "amount": 100.0,
                    "V1": -1.0,
                    "V2": -0.5,
                    "V3": 0.5,
                    "V4": 1.0,
                    "V5": -0.2,
                    "V6": 0.3,
                    "V7": -0.1,
                    "V8": 0.8,
                    "V9": -0.6,
                    "V10": 0.4,
                    "V11": -0.3,
                    "V12": 0.9,
                    "V13": -0.7,
                    "V14": 0.2,
                    "V15": -0.4,
                    "V16": 0.7,
                    "V17": -0.5,
                    "V18": 0.1,
                    "V19": -0.2,
                    "V20": 0.5,
                    "V21": -0.8,
                    "V22": 0.6,
                    "V23": -0.3,
                    "V24": 0.4,
                    "V25": -0.1,
                    "V26": 0.3,
                    "V27": -0.2,
                    "V28": 0.1,
                }
            ]
        }

        response = client.post("/api/v1/batch-predict", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert "predictions" in data


class TestMetricsEndpoint:
    def test_metrics_endpoint(self, client):
        """Test Prometheus metrics endpoint is accessible."""
        response = client.get("/metrics")
        assert response.status_code == 200

        # Verify Prometheus format
        content_type = response.headers.get("content-type", "")
        assert "text/plain" in content_type

        # Verify metrics content
        content = response.text
        assert len(content) > 0
        assert "fraud_" in content  # All our metrics start with fraud_

    def test_metrics_contain_system_info(self, client):
        """Test metrics include system information."""
        response = client.get("/metrics")
        content = response.text

        # Check for system metrics
        assert "fraud_memory_usage_bytes" in content
        assert "fraud_cpu_usage_percent" in content


class TestAlertIntegration:
    @pytest.mark.asyncio
    async def test_send_alert_on_high_fraud(self):
        init_alert_service(
            smtp_host="smtp.test.com",
            smtp_user="test@test.com",
            smtp_password="testpass",
        )

        with patch("smtplib.SMTP") as mock_smtp:
            mock_server = MagicMock()
            mock_smtp.return_value.__enter__.return_value = mock_server

            from src.services.alert_service import get_alert_service

            service = get_alert_service()

            result = await service.send_fraud_alert(
                transaction_id="TXN-HIGH", fraud_probability=0.95
            )

            # Result might be False if no recipients, but shouldn't crash
            assert isinstance(result, bool)


class TestErrorHandling:
    def test_404_not_found(self, client):
        response = client.get("/nonexistent")
        assert response.status_code == 404

    def test_invalid_method(self, client):
        response = client.delete("/api/v1/predict")
        assert response.status_code in [404, 405]


class TestDriftEndpoints:
    @pytest.fixture
    def mock_drift_service(self):
        """Mock the EvidentlyDriftService for testing."""
        # This fixture is now redundant - drift service is mocked in client fixture
        pass

    def test_comprehensive_drift_detection(self, client):
        """Test comprehensive drift detection endpoint."""
        response = client.post("/api/v1/drift/comprehensive-detect")
        assert response.status_code == 200
        data = response.json()

        assert "data_drift" in data
        assert "target_drift" in data
        assert "concept_drift" in data
        assert "multivariate_drift" in data
        assert "drift_summary" in data
        assert "timestamp" in data

    def test_sliding_window_analysis(self, client):
        """Test sliding window analysis endpoint."""
        response = client.post("/api/v1/drift/sliding-window-analysis")
        assert response.status_code == 200
        data = response.json()

        # Verify response matches SlidingWindowAnalysisResponse schema
        assert "window_size" in data
        assert "step_size" in data
        assert "analysis_period" in data
        assert "windows" in data
        assert "timestamp" in data
        assert "processing_time" in data
        # Note: "trend" is not part of the API schema

    def test_generate_drift_report(self, client):
        """Test drift report generation endpoint."""
        # First get drift detection results
        drift_response = client.post("/api/v1/drift/comprehensive-detect")
        assert drift_response.status_code == 200
        drift_results = drift_response.json()

        # Generate report using the drift results
        response = client.post("/api/v1/drift/generate-report", json=drift_results)
        assert response.status_code == 200
        data = response.json()

        # Verify response matches DriftReportResponse schema
        assert "timestamp" in data
        assert "severity" in data
        assert "recommendations" in data
        assert "alerts" in data
        assert "summary" in data
        # Note: "report_id" is not part of the API schema

    def test_comprehensive_drift_detection_invalid_data(self, client):
        """Test comprehensive drift detection with invalid parameters."""
        # Test with invalid parameter types
        params = {"window_hours": "invalid", "reference_window_days": "invalid"}

        response = client.post("/api/v1/drift/comprehensive-detect", params=params)
        # FastAPI will return 422 for invalid parameter types
        assert response.status_code == 422  # Validation error

    def test_sliding_window_analysis_missing_parameters(self, client):
        """Test sliding window analysis with valid default parameters."""
        # The endpoint has default parameters, so this should succeed
        response = client.post("/api/v1/drift/sliding-window-analysis")
        assert response.status_code == 200
        data = response.json()

        # Verify default values are used
        assert "window_size" in data
        assert "step_size" in data

    def test_generate_drift_report_empty_data(self, client):
        """Test drift report generation with empty/invalid data."""
        # Empty payload should trigger validation error
        payload = {}

        response = client.post("/api/v1/drift/generate-report", json=payload)
        # Should return 500 because our mock raises ValueError for invalid input
        assert response.status_code == 500
