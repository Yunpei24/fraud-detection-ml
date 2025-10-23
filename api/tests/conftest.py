"""
Pytest configuration and fixtures.
"""
import pytest
from fastapi.testclient import TestClient

from src.main import app
from src.models import EnsembleModel
from src.services import PredictionService, CacheService, DatabaseService


@pytest.fixture
def client():
    """
    Create a test client for the FastAPI application.
    
    Returns:
        TestClient instance
    """
    return TestClient(app)


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
def prediction_service(mock_model):
    """
    Create a prediction service with mock model.
    
    Args:
        mock_model: Mock ensemble model
        
    Returns:
        PredictionService instance
    """
    return PredictionService(mock_model)


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
        "metadata": {
            "source": "test",
            "timestamp": "2024-01-01T00:00:00Z"
        }
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
            "metadata": {"source": "test", "index": i}
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
