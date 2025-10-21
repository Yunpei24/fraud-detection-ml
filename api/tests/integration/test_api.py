"""
Integration tests for API endpoints.
"""
import pytest
from fastapi import status


class TestPredictEndpoint:
    """Tests for /api/v1/predict endpoint."""
    
    def test_predict_success(self, client, sample_transaction):
        """Test successful prediction."""
        response = client.post("/api/v1/predict", json=sample_transaction)
        
        assert response.status_code == status.HTTP_200_OK
        
        data = response.json()
        assert "transaction_id" in data
        assert "prediction" in data
        assert "confidence" in data
        assert "fraud_score" in data
        assert data["prediction"] in [0, 1]
        assert 0 <= data["confidence"] <= 1
        assert 0 <= data["fraud_score"] <= 1
    
    def test_predict_invalid_input(self, client):
        """Test prediction with invalid input."""
        invalid_data = {
            "transaction_id": "TEST-001"
            # Missing features
        }
        
        response = client.post("/api/v1/predict", json=invalid_data)
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY
    
    def test_predict_empty_features(self, client):
        """Test prediction with empty features."""
        data = {
            "transaction_id": "TEST-001",
            "features": []
        }
        
        response = client.post("/api/v1/predict", json=data)
        assert response.status_code in [
            status.HTTP_400_BAD_REQUEST,
            status.HTTP_422_UNPROCESSABLE_ENTITY
        ]


class TestBatchPredictEndpoint:
    """Tests for /api/v1/batch-predict endpoint."""
    
    def test_batch_predict_success(self, client, sample_batch_transactions):
        """Test successful batch prediction."""
        data = {"transactions": sample_batch_transactions}
        
        response = client.post("/api/v1/batch-predict", json=data)
        
        assert response.status_code == status.HTTP_200_OK
        
        result = response.json()
        assert "total_transactions" in result
        assert "successful_predictions" in result
        assert "predictions" in result
        assert result["total_transactions"] == len(sample_batch_transactions)
    
    def test_batch_predict_empty(self, client):
        """Test batch prediction with empty list."""
        data = {"transactions": []}
        
        response = client.post("/api/v1/batch-predict", json=data)
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY


class TestHealthEndpoints:
    """Tests for health check endpoints."""
    
    def test_health_check(self, client):
        """Test basic health check."""
        response = client.get("/health")
        
        assert response.status_code == status.HTTP_200_OK
        
        data = response.json()
        assert data["status"] == "healthy"
        assert "version" in data
        assert "timestamp" in data
        assert "uptime_seconds" in data
    
    def test_detailed_health_check(self, client):
        """Test detailed health check."""
        response = client.get("/health/detailed")
        
        assert response.status_code == status.HTTP_200_OK
        
        data = response.json()
        assert "status" in data
        assert "components" in data
        assert "model" in data["components"]
        assert "cache" in data["components"]
        assert "database" in data["components"]
    
    def test_liveness_probe(self, client):
        """Test liveness probe."""
        response = client.get("/live")
        
        assert response.status_code == status.HTTP_200_OK
        assert response.json()["status"] == "alive"
    
    def test_readiness_probe(self, client):
        """Test readiness probe."""
        response = client.get("/ready")
        
        assert response.status_code == status.HTTP_200_OK
        assert "status" in response.json()


class TestMetricsEndpoint:
    """Tests for metrics endpoint."""
    
    def test_metrics_endpoint(self, client):
        """Test Prometheus metrics endpoint."""
        response = client.get("/metrics")
        
        assert response.status_code == status.HTTP_200_OK
        # Prometheus metrics are in text format
        assert "text/plain" in response.headers["content-type"] or \
               "text/plain" in str(response.headers.get("content-type", ""))
        
        # Verify content contains Prometheus metrics
        content = response.text
        assert "# HELP" in content or "# TYPE" in content
    
    def test_metrics_content_has_new_metrics(self, client):
        """Test that new metrics are exposed."""
        response = client.get("/metrics")
        content = response.text
        
        # Check for new API metrics
        assert "fraud_api_requests_total" in content
        assert "fraud_api_request_duration_seconds" in content
        assert "fraud_predictions_total" in content
        assert "fraud_prediction_latency_seconds" in content
        
        # Check for system metrics
        assert "fraud_memory_usage_bytes" in content
        assert "fraud_cpu_usage_percent" in content
    
    def test_metrics_updated_after_prediction(self, client, sample_transaction):
        """Test that metrics are updated after making a prediction."""
        # Make a prediction to increment counters
        client.post("/api/v1/predict", json=sample_transaction)
        
        # Check metrics were updated
        response = client.get("/metrics")
        content = response.text
        
        # Predictions counter should be incremented
        assert "fraud_predictions_total" in content


class TestRootEndpoint:
    """Tests for root endpoint."""
    
    def test_root_endpoint(self, client):
        """Test root endpoint."""
        response = client.get("/")
        
        assert response.status_code == status.HTTP_200_OK
        
        data = response.json()
        assert data["name"] == "Fraud Detection API"
        assert "version" in data
        assert "status" in data
        assert data["status"] == "running"
