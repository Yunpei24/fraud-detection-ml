"""
Pytest configuration and fixtures.
"""

from unittest.mock import AsyncMock, Mock

import pytest
from fastapi.testclient import TestClient
from src.main import app
from src.models import EnsembleModel
from src.services import (CacheService, DatabaseService, EvidentlyDriftService,
                          PredictionService)


@pytest.fixture
def client(
    mock_prediction_service,
    mock_cache_service,
    mock_database_service,
    mock_drift_service,
    mock_auth,
):
    """
    Create a test client for the FastAPI application with mocked dependencies.

    Args:
        mock_prediction_service: Mock prediction service
        mock_cache_service: Mock cache service
        mock_database_service: Mock database service
        mock_drift_service: Mock drift service
        mock_auth: Mock authentication

    Returns:
        TestClient instance
    """
    from src.api.dependencies import (get_cache_service, get_database_service,
                                      get_drift_service,
                                      get_prediction_service)
    from src.api.routes.auth import get_current_analyst_user

    # Override dependencies with mocks
    app.dependency_overrides[get_prediction_service] = lambda: mock_prediction_service
    app.dependency_overrides[get_cache_service] = lambda: mock_cache_service
    app.dependency_overrides[get_database_service] = lambda: mock_database_service
    app.dependency_overrides[get_drift_service] = lambda: mock_drift_service
    app.dependency_overrides[get_current_analyst_user] = lambda: mock_auth

    with TestClient(app) as test_client:
        yield test_client

    # Clean up overrides
    app.dependency_overrides.clear()


@pytest.fixture
def mock_model():
    """
    Create a mock ensemble model.

    Returns:
        EnsembleModel instance with mock models
    """
    model = EnsembleModel()
    # Don't load real models, use mocks
    return model


@pytest.fixture
def mock_prediction_service(mock_model):
    """
    Create a mock prediction service for testing.

    Args:
        mock_model: Mock ensemble model

    Returns:
        Mock PredictionService instance
    """
    service = Mock(spec=PredictionService)

    # Mock async methods
    service.predict_single = AsyncMock(
        return_value={
            "transaction_id": "TEST-001",
            "prediction": 0,
            "confidence": 0.85,
            "fraud_score": 0.15,
            "risk_level": "low",
            "processing_time": 0.05,
            "model_version": "test-v1.0",
            "timestamp": 1640995200.0,  # Unix timestamp
            "features": [0.5] * 30,
            "explanation": {"feature_importance": []},
            "metadata": {"source": "test"},
        }
    )

    service.predict_batch = AsyncMock(
        return_value={
            "total_transactions": 10,
            "successful_predictions": 10,
            "failed_predictions": 0,
            "fraud_detected": 2,
            "fraud_rate": 0.2,
            "predictions": [
                {
                    "transaction_id": f"TEST-{i:03d}",
                    "prediction": (
                        0 if i % 5 != 0 else 1
                    ),  # Every 5th transaction is fraud
                    "confidence": 0.8 + (i * 0.01),
                    "fraud_score": 0.2 - (i * 0.01),
                    "risk_level": "low" if i % 5 != 0 else "high",
                    "processing_time": 0.05,
                    "model_version": "test-v1.0",
                    "timestamp": 1640995200.0 + (i * 60),  # Unix timestamp with offset
                    "features": [0.5 + (i * 0.01)] * 30,
                    "explanation": {"feature_importance": []},
                    "metadata": {"source": "test", "index": i},
                }
                for i in range(10)
            ],
            "avg_processing_time": 0.05,
        }
    )

    service.check_model_health = Mock(return_value=True)
    service.get_model_info = Mock(
        return_value={"version": "test-v1.0", "models": ["rf", "xgb", "lgb"]}
    )

    return service


@pytest.fixture
def mock_cache_service():
    """
    Create a mock cache service for testing.

    Returns:
        Mock CacheService instance
    """
    service = Mock(spec=CacheService)

    # Mock async methods
    service.get_cached_prediction = AsyncMock(return_value=None)
    service.set_prediction_cache = AsyncMock(return_value=None)
    service.get_cache_stats = Mock(return_value={"hits": 0, "misses": 0, "total": 0})
    service.clear_cache = AsyncMock(return_value=None)
    service.check_health = Mock(return_value=True)
    service.health_check = AsyncMock(return_value={"status": "healthy", "redis": "ok"})

    return service


@pytest.fixture
def mock_database_service():
    """
    Create a mock database service for testing.

    Returns:
        Mock DatabaseService instance
    """
    service = Mock(spec=DatabaseService)

    # Mock async methods
    service.save_prediction = AsyncMock(return_value=True)
    service.save_audit_log = AsyncMock(return_value=True)
    service.get_prediction = AsyncMock(return_value=None)
    service.check_health = Mock(return_value=True)
    service.health_check = Mock(return_value={"status": "healthy", "connection": "ok"})

    # Add methods needed for drift detection
    service.fetch_all = AsyncMock(return_value=[])
    service.execute = AsyncMock(return_value=None)

    return service


@pytest.fixture
def sample_transaction():
    """
    Create a sample transaction for testing.

    Returns:
        Transaction dictionary
    """
    return {
        "transaction_id": "TEST-001",
        "features": [0.5] * 30,  # 30 features with value 0.5
        "metadata": {"source": "test", "timestamp": "2024-01-01T00:00:00Z"},
    }


@pytest.fixture
def sample_batch_transactions():
    """
    Create sample batch transactions for testing.

    Returns:
        List of transaction dictionaries
    """
    return [
        {
            "transaction_id": f"TEST-{i:03d}",
            "features": [0.5 + (i * 0.01)] * 30,
            "metadata": {"source": "test", "index": i},
        }
        for i in range(10)
    ]


@pytest.fixture(scope="session")
def test_database_url():
    """
    Get test database URL.

    Returns:
        Database URL string
    """
    return "sqlite:///:memory:"  # In-memory SQLite for tests


@pytest.fixture
def database_service(test_database_url):
    """
    Create a database service for testing.

    Args:
        test_database_url: Test database URL

    Returns:
        DatabaseService instance
    """
    service = DatabaseService(test_database_url)
    service.create_tables()
    return service


@pytest.fixture
def mock_drift_service():
    """
    Create a mock drift service for testing.

    Returns:
        Mock EvidentlyDriftService instance
    """
    service = Mock(spec=EvidentlyDriftService)

    # Mock async methods
    service.detect_comprehensive_drift = AsyncMock(
        return_value={
            "timestamp": "2025-01-15T10:30:00Z",
            "analysis_window": "24h",
            "reference_window": "30d",
            "data_drift": {
                "dataset_drift_detected": False,
                "drift_share": 0.02,
                "drifted_columns": [],
                "statistical_tests": [],
            },
            "target_drift": {
                "drift_detected": False,
                "drift_score": 0.05,
                "current_fraud_rate": 0.0058,
                "reference_fraud_rate": 0.0059,
                "rate_change_percent": -1.69,
                "stattest": "psi_stat_test",
            },
            "concept_drift": {
                "drift_detected": False,
                "drift_score": 0.02,
                "stattest_name": "correlation_difference",
                "features_analyzed": ["amount", "v1", "v2", "v3"],
            },
            "multivariate_drift": {
                "tests": [
                    {
                        "name": "TestAllFeaturesValueDrift",
                        "status": "SUCCESS",
                        "description": "Test if all features have drifted",
                        "parameters": {},
                    }
                ],
                "overall_drift_detected": False,
                "drift_columns_count": 0,
            },
            "drift_summary": {
                "overall_drift_detected": False,
                "drift_types_detected": [],
                "severity_score": 0,
                "recommendations": [
                    "LOW: No significant drift detected - continue monitoring"
                ],
            },
        }
    )

    service.run_sliding_window_analysis = AsyncMock(
        return_value={
            "timestamp": "2025-01-15T10:30:00Z",
            "window_size": "24h",
            "step_size": "6h",
            "analysis_period": "7d",
            "windows": [
                {
                    "window_id": 1,
                    "start_time": "2025-01-13T10:30:00Z",
                    "end_time": "2025-01-14T10:30:00Z",
                    "record_count": 1250,
                    "drift_detected": False,
                    "drift_score": 0.01,
                }
            ],
        }
    )

    service.generate_drift_report = AsyncMock(
        return_value={
            "timestamp": "2025-01-15T10:30:00Z",
            "summary": {"overall_drift_detected": False, "severity_score": 0},
            "recommendations": [
                "Continue regular monitoring",
                "Review model performance metrics monthly",
            ],
            "alerts": [],
            "severity": "LOW",
        }
    )

    return service


@pytest.fixture
def mock_auth():
    """
    Create a mock authenticated analyst user for testing.

    Returns:
        Mock user dictionary with analyst role
    """
    return {"username": "test_analyst", "role": "analyst", "is_active": True}
