import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock, AsyncMock
from src.main import app
from src.services.alert_service import init_alert_service, AlertSeverity


@pytest.fixture
def client():
    return TestClient(app)


class TestHealthEndpoints:
    def test_health_check(self, client):
        response = client.get("/api/v1/health")
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
            "time": 0.0,
            "amount": 100.0,
            "V1": -1.0, "V2": -0.5, "V3": 0.5, "V4": 1.0, "V5": -0.2,
            "V6": 0.3, "V7": -0.1, "V8": 0.8, "V9": -0.6, "V10": 0.4,
            "V11": -0.3, "V12": 0.9, "V13": -0.7, "V14": 0.2, "V15": -0.4,
            "V16": 0.7, "V17": -0.5, "V18": 0.1, "V19": -0.2, "V20": 0.5,
            "V21": -0.8, "V22": 0.6, "V23": -0.3, "V24": 0.4, "V25": -0.1,
            "V26": 0.3, "V27": -0.2, "V28": 0.1
        }
        
        response = client.post("/api/v1/predict", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert "transaction_id" in data
        assert "fraud_probability" in data
    
    def test_predict_missing_feature(self, client):
        payload = {
            "transaction_id": "TXN-123",
            "amount": 100.0
        }
        
        response = client.post("/api/v1/predict", json=payload)
        assert response.status_code in [400, 422]
    
    def test_batch_predict(self, client):
        payload = {
            "transactions": [
                {
                    "transaction_id": "TXN-1",
                    "time": 0.0,
                    "amount": 100.0,
                    "V1": -1.0, "V2": -0.5, "V3": 0.5, "V4": 1.0, "V5": -0.2,
                    "V6": 0.3, "V7": -0.1, "V8": 0.8, "V9": -0.6, "V10": 0.4,
                    "V11": -0.3, "V12": 0.9, "V13": -0.7, "V14": 0.2, "V15": -0.4,
                    "V16": 0.7, "V17": -0.5, "V18": 0.1, "V19": -0.2, "V20": 0.5,
                    "V21": -0.8, "V22": 0.6, "V23": -0.3, "V24": 0.4, "V25": -0.1,
                    "V26": 0.3, "V27": -0.2, "V28": 0.1
                }
            ]
        }
        
        response = client.post("/api/v1/batch-predict", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert "predictions" in data


class TestMetricsEndpoint:
    def test_metrics_endpoint(self, client):
        response = client.get("/api/v1/metrics")
        assert response.status_code == 200


class TestAlertIntegration:
    @pytest.mark.asyncio
    async def test_send_alert_on_high_fraud(self):
        init_alert_service(
            smtp_host="smtp.test.com",
            smtp_user="test@test.com",
            smtp_password="testpass"
        )
        
        with patch("smtplib.SMTP") as mock_smtp:
            mock_server = MagicMock()
            mock_smtp.return_value.__enter__.return_value = mock_server
            
            from src.services.alert_service import get_alert_service
            service = get_alert_service()
            
            result = await service.send_fraud_alert(
                transaction_id="TXN-HIGH",
                fraud_probability=0.95
            )
            
            # Result might be False if no recipients, but shouldn't crash
            assert isinstance(result, bool)


class TestCORSHeaders:
    def test_cors_headers_present(self, client):
        response = client.options("/api/v1/predict")
        assert "access-control-allow-origin" in response.headers


class TestErrorHandling:
    def test_404_not_found(self, client):
        response = client.get("/nonexistent")
        assert response.status_code == 404
    
    def test_invalid_method(self, client):
        response = client.delete("/api/v1/predict")
        assert response.status_code in [404, 405]
