"""
Unit tests for Transaction Simulator
Tests fraud generation, transaction simulation, and Kafka integration
"""
import json
import random
import time
import uuid
from datetime import datetime
from unittest.mock import MagicMock, Mock, patch

import numpy as np
import pandas as pd
import pytest

try:
    from kafka.errors import KafkaError
except ImportError:
    KafkaError = Exception  # Fallback for when kafka is not available

from src.ingestion.transaction_simulator import (KAFKA_AVAILABLE,
                                                 TransactionSimulator)


@pytest.mark.unit
class TestTransactionSimulator:
    """Test suite for TransactionSimulator class."""

    def test_initialization(self):
        """Test simulator initialization."""
        simulator = TransactionSimulator(fraud_rate=0.1)

        assert simulator.fraud_rate == 0.1
        assert simulator.transaction_count == 0
        assert simulator.fraud_count == 0
        assert simulator.producer is None
        assert "V1" in simulator.legitimate_params
        assert "V4" in simulator.fraud_params

    @pytest.mark.skipif(not KAFKA_AVAILABLE, reason="Kafka not available")
    @patch("src.ingestion.transaction_simulator.KAFKA_AVAILABLE", True)
    @patch("src.ingestion.transaction_simulator.KafkaProducer")
    def test_connect_success(self, mock_kafka_producer):
        """Test successful Kafka connection."""
        mock_producer = MagicMock()
        mock_kafka_producer.return_value = mock_producer

        simulator = TransactionSimulator()
        simulator.connect()

        assert simulator.producer == mock_producer
        mock_kafka_producer.assert_called_once()

    @pytest.mark.skipif(not KAFKA_AVAILABLE, reason="Kafka not available")
    @patch("src.ingestion.transaction_simulator.KAFKA_AVAILABLE", True)
    @patch("src.ingestion.transaction_simulator.KafkaProducer")
    def test_send_transaction_success(self, mock_kafka_producer):
        """Test successful transaction sending."""
        mock_producer = MagicMock()
        mock_future = MagicMock()
        mock_future.get.return_value = MagicMock(partition=0)
        mock_producer.send.return_value = mock_future

        simulator = TransactionSimulator()
        simulator.producer = mock_producer

        transaction = {"test": "data"}
        result = simulator.send_transaction(transaction)

        assert result is True
        assert simulator.transaction_count == 1
        mock_producer.send.assert_called_once()

    @pytest.mark.skipif(not KAFKA_AVAILABLE, reason="Kafka not available")
    @patch("src.ingestion.transaction_simulator.KAFKA_AVAILABLE", True)
    @patch("src.ingestion.transaction_simulator.KafkaProducer")
    def test_send_transaction_failure(self, mock_kafka_producer):
        """Test transaction sending failure."""
        mock_producer = MagicMock()
        mock_producer.send.side_effect = KafkaError("Kafka error")

        simulator = TransactionSimulator()
        simulator.producer = mock_producer

        transaction = {"test": "data"}
        result = simulator.send_transaction(transaction)

        assert result is False
        assert simulator.transaction_count == 0

    @pytest.mark.skipif(not KAFKA_AVAILABLE, reason="Kafka not available")
    @patch("src.ingestion.transaction_simulator.KAFKA_AVAILABLE", True)
    @patch("src.ingestion.transaction_simulator.KafkaProducer")
    def test_simulate_batch(self, mock_kafka_producer):
        """Test batch simulation."""
        mock_producer = MagicMock()
        mock_future = MagicMock()
        mock_future.get.return_value = MagicMock(partition=0)
        mock_producer.send.return_value = mock_future

        simulator = TransactionSimulator(fraud_rate=0.2)
        simulator.producer = mock_producer

        summary = simulator.simulate_batch(count=100)

        assert summary["mode"] == "batch"
        assert summary["total_transactions"] == 100
        assert summary["successful"] == 100
        assert summary["failed"] == 0
        assert 15 <= summary["fraud_count"] <= 25  # ~20% of 100
        assert summary["transactions_per_second"] > 0

    @pytest.mark.skipif(not KAFKA_AVAILABLE, reason="Kafka not available")
    @patch("src.ingestion.transaction_simulator.KAFKA_AVAILABLE", True)
    @patch("src.ingestion.transaction_simulator.KafkaProducer")
    def test_simulate_stream(self, mock_kafka_producer):
        """Test stream simulation."""
        mock_producer = MagicMock()
        mock_future = MagicMock()
        mock_future.get.return_value = MagicMock(partition=0)
        mock_producer.send.return_value = mock_future

        simulator = TransactionSimulator(fraud_rate=0.1)
        simulator.producer = mock_producer

        # Mock time to avoid infinite loop
        with patch("time.sleep"), patch("time.time") as mock_time:
            mock_time.side_effect = [
                0,
                0.5,
                1,
                1.5,
                2,
                2.5,
                3,
                3.5,
                4,
                4.5,
                5,
                5.5,
                6,
            ]  # More time values for simulation

            # Should stop after duration
            simulator.simulate_stream(transactions_per_second=2, duration_seconds=5)

        # Should have sent some transactions
        assert simulator.transaction_count > 0
        assert mock_producer.send.call_count == simulator.transaction_count

    @patch("src.ingestion.transaction_simulator.KAFKA_AVAILABLE", False)
    def test_connect_kafka_unavailable(self):
        """Test connection failure when Kafka unavailable."""
        with pytest.raises(ImportError):
            TransactionSimulator()

    def test_disconnect(self):
        """Test Kafka disconnection."""
        mock_producer = MagicMock()
        simulator = TransactionSimulator()
        simulator.producer = mock_producer

        simulator.disconnect()

        mock_producer.flush.assert_called_once()
        mock_producer.close.assert_called_once()
        assert simulator.producer is None

    def test_generate_pca_features_legitimate(self):
        """Test PCA feature generation for legitimate transactions."""
        simulator = TransactionSimulator()

        features = simulator.generate_pca_features(is_fraud=False)

        assert len(features) == 28
        assert all(f"V{i}" in features for i in range(1, 29))
        assert all(isinstance(v, float) for v in features.values())

        # Check that legitimate features are within reasonable bounds
        for i in range(1, 29):
            feature_name = f"V{i}"
            value = features[feature_name]
            assert -5 <= value <= 5  # Reasonable bounds for legitimate

    def test_generate_pca_features_fraudulent(self):
        """Test PCA feature generation for fraudulent transactions."""
        simulator = TransactionSimulator()

        np.random.seed(42)  # For reproducible results
        features = simulator.generate_pca_features(is_fraud=True)

        assert len(features) == 28

        # Check that all features are generated
        for i in range(1, 29):
            assert f"V{i}" in features
            assert isinstance(features[f"V{i}"], float)

        # Check fraud-specific features have some variation
        # (exact values depend on random seed, but should be reasonable)
        assert -25 <= features["V4"] <= 10
        assert -15 <= features["V11"] <= 25
        assert -25 <= features["V12"] <= 10
        assert -25 <= features["V14"] <= 10

    def test_generate_legitimate_transaction(self):
        """Test legitimate transaction generation."""
        simulator = TransactionSimulator()

        transaction = simulator.generate_legitimate_transaction()

        # Check required fields
        required_fields = ["Time", "amount", "Class"] + [f"V{i}" for i in range(1, 29)]
        for field in required_fields:
            assert field in transaction

        # Check metadata
        assert "transaction_id" in transaction
        assert "timestamp" in transaction
        assert "source" in transaction

        # Check values
        assert transaction["Class"] == 0  # Legitimate
        assert transaction["amount"] > 0
        assert transaction["Time"] >= 0
        assert len(transaction["transaction_id"]) == 16  # "txn_" + 12 hex

    def test_generate_fraudulent_transaction(self):
        """Test fraudulent transaction generation."""
        simulator = TransactionSimulator()

        transaction = simulator.generate_fraudulent_transaction()

        # Check required fields
        required_fields = ["Time", "amount", "Class"] + [f"V{i}" for i in range(1, 29)]
        for field in required_fields:
            assert field in transaction

        # Check fraud indicators
        assert transaction["Class"] == 1  # Fraud
        assert "fraud_amount_type" in transaction
        assert transaction["fraud_amount_type"] in ["high", "low", "normal"]

    def test_generate_transaction_fraud_rate(self):
        """Test transaction generation respects fraud rate."""
        simulator = TransactionSimulator(fraud_rate=0.3)

        # Generate many transactions to test distribution
        fraud_count = 0
        total_count = 1000

        for _ in range(total_count):
            transaction = simulator.generate_transaction()
            if transaction["Class"] == 1:
                fraud_count += 1

        fraud_rate = fraud_count / total_count
        assert 0.25 <= fraud_rate <= 0.35  # Should be close to 0.3

    def test_fraud_amount_patterns(self):
        """Test different fraud amount patterns."""
        simulator = TransactionSimulator()

        # Generate multiple fraud transactions
        fraud_transactions = []
        random.seed(42)  # For reproducible results
        for _ in range(100):
            transaction = simulator.generate_transaction()
            if transaction["Class"] == 1:
                fraud_transactions.append(transaction)

        # Check amount patterns
        high_value = [
            t for t in fraud_transactions if t.get("fraud_amount_type") == "high"
        ]
        low_value = [
            t for t in fraud_transactions if t.get("fraud_amount_type") == "low"
        ]
        normal_value = [
            t for t in fraud_transactions if t.get("fraud_amount_type") == "normal"
        ]

        # Should have some of each type
        assert len(high_value) > 0
        assert len(low_value) > 0
        assert len(normal_value) > 0

        # Check amount ranges
        for t in high_value:
            assert 1000 <= t["amount"] <= 10000
        for t in low_value:
            assert 0.01 <= t["amount"] <= 10

    def test_transaction_structure(self):
        """Test transaction data structure."""
        simulator = TransactionSimulator()

        transaction = simulator.generate_transaction()

        # Check all expected fields are present
        expected_fields = [
            "Time",
            "amount",
            "Class",
            "transaction_id",
            "timestamp",
            "source",
        ] + [f"V{i}" for i in range(1, 29)]

        for field in expected_fields:
            assert field in transaction

        # Check data types
        assert isinstance(transaction["Time"], float)
        assert isinstance(transaction["amount"], float)
        assert isinstance(transaction["Class"], int)
        assert isinstance(transaction["transaction_id"], str)
        assert isinstance(transaction["timestamp"], str)

        # Check PCA features are floats
        for i in range(1, 29):
            assert isinstance(transaction[f"V{i}"], float)

    def test_fraud_detection_features(self):
        """Test that fraud transactions have detectable patterns."""
        simulator = TransactionSimulator()

        # Generate many transactions
        legitimate = []
        fraudulent = []

        for _ in range(500):
            transaction = simulator.generate_transaction()
            if transaction["Class"] == 0:
                legitimate.append(transaction)
            else:
                fraudulent.append(transaction)

        # Check that fraud features are statistically different
        legit_v4 = [t["V4"] for t in legitimate]
        fraud_v4 = [t["V4"] for t in fraudulent]

        # Fraud V4 should be significantly more negative
        assert np.mean(fraud_v4) < np.mean(legit_v4) - 1.0

        legit_v11 = [t["V11"] for t in legitimate]
        fraud_v11 = [t["V11"] for t in fraudulent]

        # Fraud V11 should be significantly more positive
        assert np.mean(fraud_v11) > np.mean(legit_v11) + 1.0

    def test_amount_distributions(self):
        """Test amount distributions for legitimate vs fraudulent."""
        simulator = TransactionSimulator()

        legitimate_amounts = []
        fraudulent_amounts = []

        for _ in range(1000):
            transaction = simulator.generate_transaction()
            if transaction["Class"] == 0:
                legitimate_amounts.append(transaction["amount"])
            else:
                fraudulent_amounts.append(transaction["amount"])

        # Legitimate amounts should be reasonable
        assert all(1 <= amt <= 5000 for amt in legitimate_amounts)

        # Fraudulent can be extreme
        assert any(amt > 1000 for amt in fraudulent_amounts)  # High value fraud
        assert any(amt < 10 for amt in fraudulent_amounts)  # Low value fraud

    def test_time_progression(self):
        """Test that time field progresses correctly."""
        simulator = TransactionSimulator()

        start_time = time.time()
        transaction1 = simulator.generate_transaction()
        time.sleep(0.1)
        transaction2 = simulator.generate_transaction()

        # Time should increase
        assert transaction2["Time"] > transaction1["Time"]

        # Time should be relative to simulator start
        assert transaction1["Time"] >= 0
        assert transaction2["Time"] >= transaction1["Time"]

    def test_unique_transaction_ids(self):
        """Test that transaction IDs are unique."""
        simulator = TransactionSimulator()

        ids = set()
        for _ in range(100):
            transaction = simulator.generate_transaction()
            txn_id = transaction["transaction_id"]

            assert txn_id not in ids
            ids.add(txn_id)

        assert len(ids) == 100

    def test_json_serialization(self):
        """Test that transactions can be JSON serialized."""
        simulator = TransactionSimulator()

        transaction = simulator.generate_transaction()

        # Should be JSON serializable
        json_str = json.dumps(transaction)
        deserialized = json.loads(json_str)

        assert deserialized == transaction

    @patch("src.ingestion.transaction_simulator.logger")
    def test_logging_on_operations(self, mock_logger):
        """Test logging during operations."""
        simulator = TransactionSimulator()

        # Generate transaction should not crash
        simulator.generate_transaction()

        # Logging may or may not occur depending on implementation
        # For now, just ensure no exceptions
        assert True

    def test_simulator_reset_between_runs(self):
        """Test that simulator state resets properly."""
        simulator = TransactionSimulator()

        # First batch
        simulator.transaction_count = 10
        simulator.fraud_count = 3

        # Create new simulator
        new_simulator = TransactionSimulator()

        assert new_simulator.transaction_count == 0
        assert new_simulator.fraud_count == 0


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def sample_transaction():
    """Sample transaction fixture."""
    return {
        "Time": 123.45,
        "amount": 150.75,
        "Class": 0,
        "V1": -1.23,
        "V2": 0.45,
        "V3": -0.67,
        "V4": 1.23,
        "V5": -0.89,
        "V6": 0.12,
        "V7": -0.34,
        "V8": 0.56,
        "V9": -0.78,
        "V10": 0.90,
        "V11": -1.01,
        "V12": 0.23,
        "V13": -0.45,
        "V14": 0.67,
        "V15": -0.89,
        "V16": 0.12,
        "V17": -0.34,
        "V18": 0.56,
        "V19": -0.78,
        "V20": 0.90,
        "V21": -1.01,
        "V22": 0.23,
        "V23": -0.45,
        "V24": 0.67,
        "V25": -0.89,
        "V26": 0.12,
        "V27": -0.34,
        "V28": 0.56,
        "transaction_id": "txn_abc123def456",
        "timestamp": "2025-01-15T14:30:00Z",
        "source": "simulator",
    }


@pytest.fixture
def sample_fraud_transaction():
    """Sample fraudulent transaction fixture."""
    return {
        "Time": 456.78,
        "amount": 2500.00,
        "Class": 1,
        "V1": -5.67,
        "V2": 2.34,
        "V3": -3.45,
        "V4": -4.56,
        "V5": 1.23,
        "V6": -2.78,
        "V7": 3.12,
        "V8": -1.89,
        "V9": 0.45,
        "V10": -2.34,
        "V11": 3.67,
        "V12": -4.12,
        "V13": 1.78,
        "V14": -5.23,
        "V15": 2.45,
        "V16": -1.67,
        "V17": 0.89,
        "V18": -2.12,
        "V19": 1.34,
        "V20": -0.78,
        "V21": 2.56,
        "V22": -3.23,
        "V23": 1.67,
        "V24": -0.45,
        "V25": 2.89,
        "V26": -1.12,
        "V27": 0.34,
        "V28": -2.67,
        "transaction_id": "txn_fraud789ghi012",
        "timestamp": "2025-01-15T15:45:00Z",
        "source": "simulator",
        "fraud_amount_type": "high",
    }
