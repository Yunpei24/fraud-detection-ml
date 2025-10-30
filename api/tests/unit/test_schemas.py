"""
Unit tests for Pydantic schemas.
"""
import pytest
from pydantic import ValidationError
from src.models import (BatchPredictionResponse, BatchTransactionRequest,
                        ErrorResponse, HealthCheckResponse, PredictionResponse,
                        TransactionRequest)


class TestTransactionRequest:
    """Tests for TransactionRequest schema."""

    def test_valid_transaction_request(self):
        """Test valid transaction request."""
        data = {
            "transaction_id": "TEST-001",
            "features": [0.5] * 30,
            "metadata": {"source": "test"},
        }

        request = TransactionRequest(**data)

        assert request.transaction_id == "TEST-001"
        assert len(request.features) == 30
        assert request.metadata["source"] == "test"

    def test_missing_transaction_id(self):
        """Test transaction request without transaction_id."""
        data = {"features": [0.5] * 30}

        with pytest.raises(ValidationError):
            TransactionRequest(**data)

    def test_missing_features(self):
        """Test transaction request without features."""
        data = {"transaction_id": "TEST-001"}

        with pytest.raises(ValidationError):
            TransactionRequest(**data)

    def test_invalid_features_type(self):
        """Test transaction request with invalid features type."""
        data = {"transaction_id": "TEST-001", "features": "not_a_list"}

        with pytest.raises(ValidationError):
            TransactionRequest(**data)


class TestPredictionResponse:
    """Tests for PredictionResponse schema."""

    def test_valid_prediction_response(self):
        """Test valid prediction response."""
        data = {
            "transaction_id": "TEST-001",
            "prediction": 1,
            "confidence": 0.85,
            "fraud_score": 0.92,
            "risk_level": "HIGH",
            "processing_time": 0.15,
            "model_version": "1.0.0",
            "timestamp": 1234567890.0,
        }

        response = PredictionResponse(**data)

        assert response.prediction == 1
        assert response.confidence == 0.85
        assert response.fraud_score == 0.92

    def test_prediction_bounds(self):
        """Test prediction value validation."""
        # Valid predictions (0 or 1)
        valid_data = {
            "transaction_id": "TEST-001",
            "prediction": 0,
            "confidence": 0.5,
            "fraud_score": 0.3,
            "risk_level": "LOW",
            "processing_time": 0.1,
            "model_version": "1.0.0",
            "timestamp": 1234567890.0,
        }

        response = PredictionResponse(**valid_data)
        assert response.prediction in [0, 1]


class TestBatchTransactionRequest:
    """Tests for BatchTransactionRequest schema."""

    def test_valid_batch_request(self):
        """Test valid batch transaction request."""
        transactions = [
            {"transaction_id": f"TEST-{i:03d}", "features": [0.5] * 30}
            for i in range(5)
        ]

        data = {"transactions": transactions}
        request = BatchTransactionRequest(**data)

        assert len(request.transactions) == 5

    def test_empty_batch_request(self):
        """Test empty batch request."""
        data = {"transactions": []}

        with pytest.raises(ValidationError):
            BatchTransactionRequest(**data)

    def test_batch_size_limit(self):
        """Test batch size exceeds maximum."""
        # Create 101 transactions (max is 100)
        transactions = [
            {"transaction_id": f"TEST-{i:03d}", "features": [0.5] * 30}
            for i in range(101)
        ]

        data = {"transactions": transactions}

        with pytest.raises(ValidationError):
            BatchTransactionRequest(**data)


class TestHealthCheckResponse:
    """Tests for HealthCheckResponse schema."""

    def test_valid_health_response(self):
        """Test valid health check response."""
        from datetime import datetime

        data = {
            "status": "healthy",
            "timestamp": datetime.utcnow(),
            "version": "1.0.0",
            "uptime_seconds": 3600.5,
        }

        response = HealthCheckResponse(**data)

        assert response.status == "healthy"
        assert response.version == "1.0.0"
        assert response.uptime_seconds > 0


class TestErrorResponse:
    """Tests for ErrorResponse schema."""

    def test_valid_error_response(self):
        """Test valid error response."""
        data = {
            "error_code": "E001",
            "message": "Invalid input",
            "details": {"field": "features"},
        }

        response = ErrorResponse(**data)

        assert response.error_code == "E001"
        assert response.message == "Invalid input"
        assert "field" in response.details
