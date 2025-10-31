"""
Unit tests for Prediction Service
Tests model prediction functionality and error handling
"""

from unittest.mock import MagicMock, Mock, patch

import pandas as pd
import pytest
from src.services.prediction_service import PredictionService


@pytest.mark.unit
class TestPredictionService:
    """Test suite for PredictionService class."""

    @pytest.fixture(autouse=True)
    def patch_traffic_router(self, mock_traffic_router):
        """Automatically patch TrafficRouter for all tests in this class."""
        with patch(
            "src.services.prediction_service.TrafficRouter",
            return_value=mock_traffic_router,
        ):
            yield

    def test_initialization(self, test_model):
        """Test prediction service initialization."""
        service = PredictionService(test_model)

        assert service.model == test_model

    @pytest.mark.asyncio
    async def test_predict_single_success(self, test_model, sample_features):
        """Test successful single prediction."""
        service = PredictionService(test_model)

        # Mock the model prediction
        test_model.predict.return_value = {
            "prediction": 0,
            "confidence": 0.85,
            "fraud_score": 0.15,
        }

        # Mock the _prepare_features method
        with patch.object(service, "_prepare_features", return_value=[[1.0, 2.0, 3.0]]):
            result = await service.predict_single(
                transaction_id="tx_123", features=sample_features
            )

        assert result["transaction_id"] == "tx_123"
        assert result["prediction"] == 0
        assert result["fraud_score"] == 0.15
        assert result["confidence"] == 0.85
        assert "processing_time" in result

    @pytest.mark.asyncio
    async def test_predict_single_fraud(self, test_model, sample_features):
        """Test prediction detecting fraud."""
        service = PredictionService(test_model)

        test_model.predict.return_value = {
            "prediction": 1,
            "confidence": 0.92,
            "fraud_score": 0.78,
        }

        with patch.object(service, "_prepare_features", return_value=[[1.0, 2.0, 3.0]]):
            result = await service.predict_single(
                transaction_id="tx_456", features=sample_features
            )

        assert result["prediction"] == 1
        assert result["fraud_score"] == 0.78

    @pytest.mark.asyncio
    async def test_predict_without_model_loaded(self, sample_features):
        """Test prediction without model loaded."""
        service = PredictionService(None)

        # The service checks for preprocessor before checking model, so it raises PredictionFailedException
        with pytest.raises(Exception):  # PredictionFailedException
            await service.predict_single(
                transaction_id="tx_123", features=sample_features
            )

    @pytest.mark.asyncio
    async def test_predict_batch_success(self, test_model, sample_batch_data):
        """Test successful batch prediction."""
        service = PredictionService(test_model)

        # Mock successful predictions for all transactions
        test_model.predict.return_value = {
            "prediction": 0,
            "confidence": 0.8,
            "fraud_score": 0.2,
        }

        with patch.object(service, "_prepare_features", return_value=[[1.0, 2.0, 3.0]]):
            result = await service.predict_batch(sample_batch_data)

        assert result["total_transactions"] == 2
        assert result["successful_predictions"] == 2
        assert result["failed_predictions"] == 0
        assert result["fraud_detected"] == 0
        assert len(result["predictions"]) == 2

    @pytest.mark.asyncio
    async def test_predict_batch_with_fraud(self, test_model, sample_batch_data):
        """Test batch prediction with fraud detection."""
        service = PredictionService(test_model)

        # Mock alternating predictions (one fraud, one legitimate)
        call_count = 0

        def mock_predict(features_array):
            nonlocal call_count
            call_count += 1
            if call_count % 2 == 1:
                return {"prediction": 1, "confidence": 0.9, "fraud_score": 0.8}
            else:
                return {"prediction": 0, "confidence": 0.7, "fraud_score": 0.3}

        test_model.predict.side_effect = mock_predict

        with patch.object(service, "_prepare_features", return_value=[[1.0, 2.0, 3.0]]):
            result = await service.predict_batch(sample_batch_data)

        assert result["fraud_detected"] == 1
        assert result["fraud_rate"] == 0.5  # 1 out of 2    @pytest.mark.asyncio

    async def test_predict_batch_empty(self, test_model):
        """Test batch prediction with empty input."""
        service = PredictionService(test_model)

        result = await service.predict_batch([])

        assert result["total_transactions"] == 0
        assert result["successful_predictions"] == 0
        assert result["failed_predictions"] == 0

    @pytest.mark.asyncio
    async def test_predict_batch_without_model(self, sample_batch_data):
        """Test batch prediction without model loaded."""
        service = PredictionService(None)

        # Should not raise AttributeError, but process with failures
        result = await service.predict_batch(sample_batch_data)

        # All predictions should fail
        assert result["total_transactions"] == 2
        assert result["successful_predictions"] == 0
        assert result["failed_predictions"] == 2
        assert len(result["errors"]) == 2

    def test_get_model_info(self, test_model):
        """Test getting model information."""
        service = PredictionService(test_model)

        test_model.get_info.return_value = {"version": "1.0.0", "type": "ensemble"}

        info = service.get_model_info()
        assert info["version"] == "1.0.0"
        assert info["type"] == "ensemble"

    def test_check_model_health(self, test_model):
        """Test model health check."""
        service = PredictionService(test_model)

        test_model.health_check.return_value = True

        health = service.check_model_health()
        assert health is True

    @pytest.mark.asyncio
    async def test_predict_single_with_metadata(self, test_model, sample_features):
        """Test prediction with metadata."""
        service = PredictionService(test_model)

        test_model.predict.return_value = {
            "prediction": 0,
            "confidence": 0.8,
            "fraud_score": 0.2,
        }

        metadata = {"source": "api", "version": "1.0"}

        with patch.object(service, "_prepare_features", return_value=[[1.0, 2.0, 3.0]]):
            result = await service.predict_single(
                transaction_id="tx_123", features=sample_features, metadata=metadata
            )

        assert result["metadata"] == metadata

    @pytest.mark.asyncio
    async def test_predict_batch_partial_failure(self, test_model, sample_features):
        """Test batch prediction with partial failures."""
        service = PredictionService(test_model)

        # Mock first prediction success, second failure
        call_count = 0

        def mock_predict(features_array):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return {"prediction": 0, "confidence": 0.8, "fraud_score": 0.2}
            else:
                raise Exception("Model prediction failed")

        test_model.predict.side_effect = mock_predict

        batch_data = [
            {"transaction_id": "tx_1", "features": sample_features},
            {"transaction_id": "tx_2", "features": sample_features},
        ]

        with patch.object(service, "_prepare_features", return_value=[[1.0, 2.0, 3.0]]):
            result = await service.predict_batch(batch_data)

        assert result["total_transactions"] == 2
        assert result["successful_predictions"] == 1
        assert result["failed_predictions"] == 1
        assert len(result["errors"]) == 1

    @pytest.mark.asyncio
    async def test_predict_single_error_handling(self, test_model, sample_features):
        """Test prediction error handling."""
        service = PredictionService(test_model)

        test_model.predict.side_effect = Exception("Model prediction failed")

        with patch.object(service, "_prepare_features", return_value=[[1.0, 2.0, 3.0]]):
            with pytest.raises(Exception):  # Should raise PredictionFailedException
                await service.predict_single(
                    transaction_id="tx_123", features=sample_features
                )

    def test_risk_level_calculation(self, test_model):
        """Test risk level calculation from fraud score."""
        service = PredictionService(test_model)

        # Test different risk levels
        assert service._get_risk_level(0.95) == "CRITICAL"
        assert service._get_risk_level(0.75) == "HIGH"
        assert service._get_risk_level(0.55) == "MEDIUM"
        assert service._get_risk_level(0.35) == "LOW"
        assert service._get_risk_level(0.15) == "MINIMAL"

    @pytest.mark.asyncio
    async def test_predict_single_with_explanation(self, test_model, sample_features):
        """Test prediction with SHAP explanation."""
        service = PredictionService(test_model)

        test_model.predict.return_value = {
            "prediction": 1,
            "confidence": 0.9,
            "fraud_score": 0.8,
        }

        test_model.explain_prediction.return_value = {
            "shap_values": [0.1, -0.2, 0.3],
            "base_value": 0.5,
        }

        # Mock settings to enable explanations
        with patch(
            "src.services.prediction_service.settings"
        ) as mock_settings, patch.object(
            service, "_prepare_features", return_value=[[1.0, 2.0, 3.0]]
        ):
            mock_settings.enable_shap_explanation = True
            mock_settings.model_version = "test-v1.0"

            result = await service.predict_single(
                transaction_id="tx_123", features=sample_features
            )

            assert "explanation" in result
            assert result["explanation"]["shap_values"] == [0.1, -0.2, 0.3]

    @pytest.mark.asyncio
    async def test_predict_single_explanation_failure(
        self, test_model, sample_features
    ):
        """Test prediction when explanation fails."""
        service = PredictionService(test_model)

        test_model.predict.return_value = {
            "prediction": 0,
            "confidence": 0.8,
            "fraud_score": 0.2,
        }

        test_model.explain_prediction.side_effect = Exception("Explanation failed")

        # Mock settings to enable explanations
        with patch(
            "src.services.prediction_service.settings"
        ) as mock_settings, patch.object(
            service, "_prepare_features", return_value=[[1.0, 2.0, 3.0]]
        ):
            mock_settings.enable_shap_explanation = True
            mock_settings.model_version = "test-v1.0"

            result = await service.predict_single(
                transaction_id="tx_123", features=sample_features
            )

            # Should still succeed but without explanation
            assert "explanation" not in result
            assert result["prediction"] == 0


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def test_model():
    """Test model fixture."""
    model = Mock()
    model.get_info.return_value = {"version": "1.0.0", "type": "ensemble"}
    model.health_check.return_value = True

    # Mock preprocessor
    preprocessor = Mock()
    preprocessor.transform.return_value = Mock()  # Mock DataFrame
    model.preprocessor = preprocessor

    return model


@pytest.fixture
def mock_traffic_router():
    """Mock traffic router fixture."""
    traffic_router = Mock()
    traffic_router_config = Mock()
    traffic_router_config.canary_enabled = False
    traffic_router_config.canary_traffic_pct = 0
    traffic_router_config.champion_traffic_pct = 100
    traffic_router_config.canary_model_uris = {}
    traffic_router_config.champion_model_uris = {}
    traffic_router_config.ensemble_weights = {}
    traffic_router.config = traffic_router_config
    traffic_router.should_use_canary.return_value = False
    traffic_router.get_model_info.return_value = {
        "canary_enabled": False,
        "canary_traffic_pct": 0,
        "champion_traffic_pct": 100,
        "canary_model_uris": {},
        "champion_model_uris": {},
        "ensemble_weights": {},
    }
    return traffic_router


@pytest.fixture
def sample_features():
    """Sample features for testing."""
    # 29 features: Time + V1-V28 + amount
    return [123456.0] + [0.1 * i for i in range(1, 29)] + [150.0]


@pytest.fixture
def sample_batch_data():
    """Batch of sample transaction data."""
    return [
        {
            "transaction_id": "tx_001",
            "features": [123456.0] + [0.1 * i for i in range(1, 29)] + [100.0],
        },
        {
            "transaction_id": "tx_002",
            "features": [123457.0] + [0.2 * i for i in range(1, 29)] + [200.0],
        },
    ]
