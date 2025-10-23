"""
Pytest configuration for data module tests
"""

import pytest
import pandas as pd
import numpy as np
import sys
from pathlib import Path
from unittest.mock import MagicMock

# Mock Databricks SDK (not installed in test environment)
sys.modules['databricks'] = MagicMock()
sys.modules['databricks.sdk'] = MagicMock()
sys.modules['databricks.sdk.service'] = MagicMock()
sys.modules['databricks.sdk.service.jobs'] = MagicMock()

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


@pytest.fixture
def sample_transaction():
    """Sample transaction for testing"""
    return {
        "transaction_id": "TXN123456",
        "customer_id": "CUST001",
        "merchant_id": "MRCH001",
        "amount": 150.50,
        "currency": "USD",
        "transaction_time": "2025-10-18T10:30:00",
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
        "source_system": "mobile"
    }


@pytest.fixture
def sample_dataframe():
    """Sample DataFrame for testing"""
    return pd.DataFrame({
        "transaction_id": ["TXN001", "TXN002", "TXN003", "TXN004", "TXN005"],
        "customer_id": ["CUST001", "CUST001", "CUST002", "CUST002", "CUST003"],
        "merchant_id": ["MRCH001", "MRCH002", "MRCH001", "MRCH003", "MRCH001"],
        "amount": [100.0, 50.0, 200.0, 75.0, 150.0],
        "currency": ["USD", "USD", "USD", "USD", "USD"],
        "transaction_time": pd.date_range("2025-10-18", periods=5, freq="H"),
        "customer_country": ["US", "US", "US", "US", "US"],
        "merchant_country": ["US", "US", "US", "US", "US"],
        "is_fraud": [0, 0, 1, 0, 0],
        "is_disputed": [False, False, True, False, False]
    })


@pytest.fixture
def dataframe_with_nulls():
    """DataFrame with missing values"""
    return pd.DataFrame({
        "transaction_id": ["TXN001", "TXN002", None, "TXN004"],
        "amount": [100.0, np.nan, 200.0, 75.0],
        "currency": ["USD", "USD", None, "USD"],
        "is_fraud": [0, 1, 0, None]
    })


@pytest.fixture
def dataframe_with_duplicates():
    """DataFrame with duplicate rows"""
    return pd.DataFrame({
        "transaction_id": ["TXN001", "TXN001", "TXN002", "TXN002", "TXN003"],
        "amount": [100.0, 100.0, 200.0, 200.0, 150.0],
        "customer_id": ["CUST001", "CUST001", "CUST002", "CUST002", "CUST003"]
    })


@pytest.fixture
def dataframe_with_outliers():
    """DataFrame with outliers (using lower threshold for detection)"""
    return pd.DataFrame({
        "amount": [100.0, 105.0, 99.0, 102.0, 1000.0],  # 1000 is more extreme outlier
        "customer_id": ["C1", "C1", "C2", "C2", "C3"]
    })
