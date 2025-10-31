"""
Pytest configuration for data module tests
"""

import sys
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

# Mock Databricks SDK (not installed in test environment)
sys.modules["databricks"] = MagicMock()
sys.modules["databricks.sdk"] = MagicMock()
sys.modules["databricks.sdk.service"] = MagicMock()
sys.modules["databricks.sdk.service.jobs"] = MagicMock()

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


@pytest.fixture
def test_api_credentials():
    """Test API credentials for JWT authentication"""
    return {
        "api_username": "test-user",
        "api_password": "test-password",
        "api_token": "test-token-12345",
    }


@pytest.fixture(autouse=True)
def mock_jwt_authentication(monkeypatch):
    """
    Auto-mock JWT authentication for all tests.

    This fixture automatically mocks the _authenticate method in RealtimePipeline
    to prevent actual API authentication attempts during testing.
    """
    from unittest.mock import MagicMock

    # Set test environment variables
    monkeypatch.setenv("API_USERNAME", "test-user")
    monkeypatch.setenv("API_PASSWORD", "test-password")
    monkeypatch.setenv("API_TOKEN", "test-token-12345")

    # Mock the authenticate method to always succeed without making API calls
    def mock_authenticate(self):
        self.api_token = "test-token-12345"
        self._token_expiry = time.time() + 3600
        self._token_issued_at = time.time()
        return True

    # Mock the refresh method
    def mock_refresh_token(self):
        return True

    # Mock the get_auth_headers method
    def mock_get_auth_headers(self):
        return {"Authorization": f"Bearer test-token-12345"}

    # Patch the methods
    try:
        from src.pipelines.realtime_pipeline import RealtimePipeline

        monkeypatch.setattr(RealtimePipeline, "_authenticate", mock_authenticate)
        monkeypatch.setattr(
            RealtimePipeline, "_refresh_token_if_needed", mock_refresh_token
        )
        monkeypatch.setattr(
            RealtimePipeline, "_get_auth_headers", mock_get_auth_headers
        )
    except ImportError:
        pass  # Module not yet loaded

    yield


@pytest.fixture(autouse=True)
def mock_session_requests(monkeypatch):
    """
    Auto-mock requests.Session for all tests.

    This fixture creates a mock session that tests can configure.
    The mock session is automatically injected into RealtimePipeline instances.
    """
    mock_session_instance = MagicMock()

    # Default successful response
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {"predictions": []}
    mock_response.text = "OK"
    mock_session_instance.post.return_value = mock_response

    # Patch Session class to return our mock
    with patch(
        "src.pipelines.realtime_pipeline.requests.Session",
        return_value=mock_session_instance,
    ):
        yield mock_session_instance


@pytest.fixture
def sample_transaction():
    """Sample transaction for testing"""
    return {
        "transaction_id": "TXN123456",
        "customer_id": "CUST001",
        "merchant_id": "MRCH001",
        "amount": 150.50,
        "currency": "USD",
        "time": "2025-10-18T10:30:00",
        "customer_zip": "10001",
        "merchant_zip": "10002",
        "customer_country": "US",
        "merchant_country": "US",
        "device_id": "DEV001",
        "session_id": "SESS001",
        "ip_address": "192.168.1.1",
        "mcc": 5411,
        "transaction_type": "purchase",
        "is_fraud": 0,
        "is_disputed": False,
        "source_system": "mobile",
    }


@pytest.fixture
def sample_dataframe():
    """Sample DataFrame for testing"""
    return pd.DataFrame(
        {
            "transaction_id": ["TXN001", "TXN002", "TXN003", "TXN004", "TXN005"],
            "customer_id": ["CUST001", "CUST001", "CUST002", "CUST002", "CUST003"],
            "merchant_id": ["MRCH001", "MRCH002", "MRCH001", "MRCH003", "MRCH001"],
            "amount": [100.0, 50.0, 200.0, 75.0, 150.0],
            "currency": ["USD", "USD", "USD", "USD", "USD"],
            "time": pd.date_range("2025-10-18", periods=5, freq="H"),
            "customer_country": ["US", "US", "US", "US", "US"],
            "merchant_country": ["US", "US", "US", "US", "US"],
            "is_fraud": [0, 0, 1, 0, 0],
            "is_disputed": [False, False, True, False, False],
        }
    )


@pytest.fixture
def dataframe_with_nulls():
    """DataFrame with missing values"""
    return pd.DataFrame(
        {
            "transaction_id": ["TXN001", "TXN002", None, "TXN004"],
            "amount": [100.0, np.nan, 200.0, 75.0],
            "currency": ["USD", "USD", None, "USD"],
            "is_fraud": [0, 1, 0, None],
        }
    )


@pytest.fixture
def dataframe_with_duplicates():
    """DataFrame with duplicate rows"""
    return pd.DataFrame(
        {
            "transaction_id": ["TXN001", "TXN001", "TXN002", "TXN002", "TXN003"],
            "amount": [100.0, 100.0, 200.0, 200.0, 150.0],
            "customer_id": ["CUST001", "CUST001", "CUST002", "CUST002", "CUST003"],
        }
    )


@pytest.fixture
def dataframe_with_outliers():
    """DataFrame with outliers (using lower threshold for detection)"""
    return pd.DataFrame(
        {
            "amount": [
                100.0,
                105.0,
                99.0,
                102.0,
                1000.0,
            ],  # 1000 is more extreme outlier
            "customer_id": ["C1", "C1", "C2", "C2", "C3"],
        }
    )
