from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient
from src.main import app
from src.services.alert_service import init_alert_service


@pytest.fixture
def client():
    from src.api.dependencies import (get_cache_service, get_database_service,
                                      get_prediction_service)

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

    # Override dependencies
    app.dependency_overrides[get_prediction_service] = lambda: mock_prediction_service
    app.dependency_overrides[get_cache_service] = lambda: mock_cache_service
    app.dependency_overrides[get_database_service] = lambda: mock_database_service

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
        with patch("src.api.routes.drift.EvidentlyDriftService") as mock_service:
            mock_instance = MagicMock()
            mock_service.return_value = mock_instance
            yield mock_instance

    def test_comprehensive_drift_detection(self, client, mock_drift_service):
        """Test comprehensive drift detection endpoint."""
        # Mock the service response
        mock_drift_service.detect_comprehensive_drift.return_value = {
            "data_drift": {"detected": False, "drift_score": 0.05},
            "target_drift": {"detected": False, "drift_score": 0.02},
            "concept_drift": {"detected": False, "drift_score": 0.01},
            "multivariate_drift": {"detected": False, "drift_score": 0.03},
        }

        payload = {
            "reference_data": [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
            "current_data": [[1.1, 2.1, 3.1], [4.1, 5.1, 6.1]],
            "target_reference": [0, 1],
            "target_current": [0, 1],
        }

        response = client.post("/api/v1/drift/comprehensive-detect", json=payload)
        assert response.status_code == 200
        data = response.json()

        assert "data_drift" in data
        assert "target_drift" in data
        assert "concept_drift" in data
        assert "multivariate_drift" in data

        mock_drift_service.detect_comprehensive_drift.assert_called_once()

    def test_sliding_window_analysis(self, client, mock_drift_service):
        """Test sliding window analysis endpoint."""
        # Mock the service response
        mock_drift_service.run_sliding_window_analysis.return_value = {
            "window_size": 1000,
            "step_size": 100,
            "drift_scores": [0.05, 0.08, 0.12, 0.15],
            "drift_detected": True,
            "drift_timestamps": ["2024-01-01T00:00:00", "2024-01-01T01:00:00"],
        }

        payload = {
            "data": [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]],
            "window_size": 1000,
            "step_size": 100,
        }

        response = client.post("/api/v1/drift/sliding-window-analysis", json=payload)
        assert response.status_code == 200
        data = response.json()

        assert "window_size" in data
        assert "drift_scores" in data
        assert "drift_detected" in data

        mock_drift_service.run_sliding_window_analysis.assert_called_once()

    def test_generate_drift_report(self, client, mock_drift_service):
        """Test drift report generation endpoint."""
        # Mock the service response
        mock_drift_service.generate_drift_report.return_value = {
            "report_id": "report-123",
            "timestamp": "2024-01-01T00:00:00",
            "summary": {
                "total_features": 10,
                "drifted_features": 2,
                "drift_percentage": 20.0,
            },
            "details": {
                "feature_1": {"drift_detected": True, "p_value": 0.01},
                "feature_2": {"drift_detected": False, "p_value": 0.15},
            },
        }

        payload = {
            "reference_data": [[1.0, 2.0], [3.0, 4.0]],
            "current_data": [[1.5, 2.5], [3.5, 4.5]],
            "report_name": "test_report",
        }

        response = client.post("/api/v1/drift/generate-report", json=payload)
        assert response.status_code == 200
        data = response.json()

        assert "report_id" in data
        assert "summary" in data
        assert "details" in data

        mock_drift_service.generate_drift_report.assert_called_once()

    def test_comprehensive_drift_detection_invalid_data(
        self, client, mock_drift_service
    ):
        """Test comprehensive drift detection with invalid data."""
        payload = {
            "reference_data": "invalid",  # Should be array
            "current_data": [[1.0, 2.0]],
            "target_reference": [0, 1],
            "target_current": [0, 1],
        }

        response = client.post("/api/v1/drift/comprehensive-detect", json=payload)
        assert response.status_code == 422  # Validation error

    def test_sliding_window_analysis_missing_parameters(
        self, client, mock_drift_service
    ):
        """Test sliding window analysis with missing parameters."""
        payload = {
            "data": [[1.0, 2.0], [3.0, 4.0]]
            # Missing window_size and step_size
        }

        response = client.post("/api/v1/drift/sliding-window-analysis", json=payload)
        assert response.status_code == 422  # Validation error

    def test_generate_drift_report_empty_data(self, client, mock_drift_service):
        """Test drift report generation with empty data."""
        payload = {
            "reference_data": [],
            "current_data": [],
            "report_name": "empty_report",
        }

        response = client.post("/api/v1/drift/generate-report", json=payload)
        assert response.status_code == 400  # Bad request for empty data
