"""
Test configuration and fixtures for drift detection tests.

This module provides pytest fixtures and configuration for all test modules.
"""

import os
import tempfile
from datetime import datetime, timedelta
from typing import Any, Dict

import numpy as np
import pandas as pd
import pytest

# Import modules to test
from src.config.settings import Settings


@pytest.fixture
def test_settings():
    """Create test settings with safe defaults."""
    settings = Settings()
    settings.environment = "test"
    settings.database_url = "postgresql://test:test@localhost:5432/test_db"
    settings.log_level = "DEBUG"
    settings.prometheus_enabled = False
    settings.alert_email_enabled = False
    settings.target_drift_threshold = 0.5
    settings.data_drift_threshold = 0.3
    settings.concept_drift_threshold = 0.05
    return settings


@pytest.fixture
def baseline_data():
    """Create synthetic baseline dataset for testing."""
    np.random.seed(42)
    n_samples = 1000

    data = pd.DataFrame(
        {
            "Time": np.random.uniform(0, 172792, n_samples),
            "V1": np.random.normal(0, 1, n_samples),
            "V2": np.random.normal(0, 1, n_samples),
            "V3": np.random.normal(0, 1, n_samples),
            "V4": np.random.normal(0, 1, n_samples),
            "V5": np.random.normal(0, 1, n_samples),
            "amount": np.random.exponential(88.35, n_samples),
            "Class": np.random.choice([0, 1], n_samples, p=[0.998, 0.002]),
        }
    )

    # Ensure at least 2 fraud cases
    fraud_count = data["Class"].sum()
    if fraud_count < 2:
        data.loc[0, "Class"] = 1
        data.loc[1, "Class"] = 1

    return data


@pytest.fixture
def current_data_no_drift(baseline_data):
    """Create current data with no drift (preserve exact fraud distribution from baseline)."""
    # Sample from baseline to preserve the exact fraud rate distribution
    # This ensures no target drift can be detected
    np.random.seed(43)
    # Sample indices without replacement but enough data
    indices = np.random.choice(baseline_data.index, size=500, replace=True)
    data = baseline_data.iloc[indices].reset_index(drop=True)
    return data


@pytest.fixture
def current_data_with_drift(baseline_data):
    """Create current data with significant drift."""
    np.random.seed(44)
    n_samples = 500

    # Shifted distributions to simulate drift
    data = pd.DataFrame(
        {
            "Time": np.random.uniform(0, 172792, n_samples),
            "V1": np.random.normal(0.5, 1.2, n_samples),  # Shifted mean and variance
            "V2": np.random.normal(-0.3, 0.8, n_samples),
            "V3": np.random.normal(0, 1, n_samples),
            "V4": np.random.normal(0.2, 1.5, n_samples),
            "V5": np.random.normal(0, 1, n_samples),
            "amount": np.random.exponential(120, n_samples),  # Different scale
            "Class": np.random.choice(
                [0, 1], n_samples, p=[0.99, 0.01]
            ),  # 1% fraud rate (5x increase)
        }
    )

    # Ensure at least 3 fraud cases
    fraud_count = data["Class"].sum()
    if fraud_count < 3:
        data.loc[0, "Class"] = 1
        data.loc[1, "Class"] = 1
        data.loc[2, "Class"] = 1

    return data


@pytest.fixture
def predictions_and_labels():
    """Create predictions and true labels for concept drift testing."""
    np.random.seed(45)
    n_samples = 500
    n_fraud = 2  # Guarantee at least 2 fraud cases

    # Simulate good model performance
    y_true = np.concatenate([np.ones(n_fraud), np.zeros(n_samples - n_fraud)])
    y_true = np.random.permutation(y_true)  # Shuffle fraud and legitimate

    # Generate predictions with some errors
    y_pred = y_true.copy()

    # Add false negatives (miss some frauds)
    fraud_indices = np.where(y_true == 1)[0]
    if len(fraud_indices) > 0:
        n_fn = max(1, int(len(fraud_indices) * 0.05))
        fn_indices = np.random.choice(fraud_indices, size=n_fn, replace=False)
        y_pred[fn_indices] = 0

    # Add false positives
    normal_indices = np.where(y_true == 0)[0]
    if len(normal_indices) > 0:
        n_fp = max(1, int(len(normal_indices) * 0.015))
        fp_indices = np.random.choice(normal_indices, size=n_fp, replace=False)
        y_pred[fp_indices] = 1

    return y_true, y_pred


@pytest.fixture
def degraded_predictions():
    """Create degraded model predictions for concept drift testing."""
    np.random.seed(46)
    n_samples = 500
    n_fraud = 3  # Guarantee at least 3 fraud cases

    y_true = np.concatenate([np.ones(n_fraud), np.zeros(n_samples - n_fraud)])
    y_true = np.random.permutation(y_true)
    y_pred = y_true.copy()

    # Significantly worse performance
    fraud_indices = np.where(y_true == 1)[0]
    if len(fraud_indices) > 0:
        n_fn = max(1, int(len(fraud_indices) * 0.15))
        fn_indices = np.random.choice(fraud_indices, size=n_fn, replace=False)
        y_pred[fn_indices] = 0

    normal_indices = np.where(y_true == 0)[0]
    if len(normal_indices) > 0:
        n_fp = max(1, int(len(normal_indices) * 0.03))
        fp_indices = np.random.choice(normal_indices, size=n_fp, replace=False)
        y_pred[fp_indices] = 1

    return y_true, y_pred


@pytest.fixture
def drift_results_sample():
    """Create sample drift results for testing."""
    return {
        "timestamp": datetime.utcnow().isoformat(),
        "data_drift": {
            "drift_detected": True,
            "avg_psi": 0.45,
            "threshold": 0.3,
            "drifted_features": ["V1", "V2", "amount"],
            "psi_scores": {
                "V1": 0.52,
                "V2": 0.38,
                "V3": 0.15,
                "V4": 0.22,
                "amount": 0.48,
            },
        },
        "target_drift": {
            "drift_detected": True,
            "current_fraud_rate": 0.005,
            "baseline_fraud_rate": 0.002,
            "relative_change": 1.5,
            "severity": "HIGH",
        },
        "concept_drift": {
            "drift_detected": True,
            "severity": "MEDIUM",
            "metrics": {
                "recall": 0.92,
                "precision": 0.93,
                "fpr": 0.018,
                "f1_score": 0.925,
                "baseline_recall": 0.98,
                "baseline_precision": 0.95,
                "baseline_fpr": 0.015,
                "baseline_f1": 0.965,
            },
        },
    }


@pytest.fixture
def temp_output_dir():
    """Create temporary directory for test outputs."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    # Cleanup after test
    import shutil

    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)


@pytest.fixture
def mock_database_url():
    """Provide mock database URL for testing."""
    return "postgresql://test_user:test_pass@localhost:5432/test_drift_db"


@pytest.fixture
def alert_test_config():
    """Configuration for alert testing."""
    return {
        "email_recipients": ["test@example.com"],
        "slack_channel": "#test-alerts",
        "max_alerts_per_hour": 10,
        "smtp_server": "smtp.test.com",
        "smtp_port": 587,
    }


@pytest.fixture
def sample_timeline_data():
    """Create sample timeline data for testing."""
    timestamps = [datetime.utcnow() - timedelta(hours=i) for i in range(24, 0, -1)]

    return pd.DataFrame(
        {
            "timestamp": timestamps,
            "drift_type": ["data"] * 8 + ["target"] * 8 + ["concept"] * 8,
            "drift_score": np.random.uniform(0, 0.8, 24),
            "threshold": [0.3] * 8 + [0.5] * 8 + [0.05] * 8,
            "threshold_exceeded": np.random.choice([True, False], 24, p=[0.3, 0.7]),
        }
    )


@pytest.fixture(autouse=True)
def reset_test_state():
    """Reset any global state before each test."""
    # This fixture runs automatically before each test
    yield
    # Cleanup after test if needed


# Pytest configuration
def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line("markers", "unit: mark test as a unit test")
    config.addinivalue_line("markers", "integration: mark test as an integration test")
    config.addinivalue_line("markers", "slow: mark test as slow running")
    config.addinivalue_line("markers", "database: mark test as requiring database")
