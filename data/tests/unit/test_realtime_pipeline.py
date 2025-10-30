"""
Unit tests for Rea        with patch('src.pipelines.realtime_pipeline.DatabaseService') as mock_db_service:
            mock_db_instance = MagicMock()
            mock_db_service.return_value = mock_db_instance
            
            pipeline = RealtimePipeline(connect_db=False) Pipeline
Tests Kafka consumption, data cleaning, API prediction, and database saving
"""
from datetime import datetime
from unittest.mock import MagicMock, Mock, patch

import numpy as np
import pandas as pd
import pytest
import requests
from src.pipelines.realtime_pipeline import KAFKA_AVAILABLE, RealtimePipeline


@pytest.mark.unit
class TestRealtimePipeline:
    """Test suite for RealtimePipeline class."""

    def test_initialization(self):
        """Test pipeline initialization."""
        with patch("src.pipelines.realtime_pipeline.DatabaseService") as mock_db:
            mock_db_instance = MagicMock()
            mock_db.return_value = mock_db_instance

            pipeline = RealtimePipeline(
                kafka_bootstrap_servers="localhost:9092",
                kafka_topic="test-topic",
                api_url="http://localhost:8000",
                connect_db=False,
            )

            assert pipeline.kafka_bootstrap_servers == "localhost:9092"
            assert pipeline.kafka_topic == "test-topic"
            assert pipeline.api_url == "http://localhost:8000"
            assert pipeline.consumer is None
            assert "total_consumed" in pipeline.metrics

    @pytest.mark.skipif(not KAFKA_AVAILABLE, reason="Kafka not available")
    @patch("src.pipelines.realtime_pipeline.KAFKA_AVAILABLE", True)
    @patch("src.pipelines.realtime_pipeline.KafkaConsumer")
    def test_create_kafka_consumer(self, mock_kafka_consumer):
        """Test Kafka consumer creation."""
        mock_consumer = MagicMock()
        mock_kafka_consumer.return_value = mock_consumer

        pipeline = RealtimePipeline(connect_db=False)
        consumer = pipeline._create_kafka_consumer()

        assert consumer == mock_consumer
        mock_kafka_consumer.assert_called_once()

    @pytest.mark.skipif(not KAFKA_AVAILABLE, reason="Kafka not available")
    @patch("src.pipelines.realtime_pipeline.KAFKA_AVAILABLE", False)
    def test_create_kafka_consumer_unavailable(self):
        """Test Kafka consumer creation when unavailable."""
        pipeline = RealtimePipeline(connect_db=False)

        with pytest.raises(ImportError):
            pipeline._create_kafka_consumer()

    @pytest.mark.skipif(not KAFKA_AVAILABLE, reason="Kafka not available")
    @patch("src.pipelines.realtime_pipeline.KAFKA_AVAILABLE", True)
    @patch("src.pipelines.realtime_pipeline.KafkaConsumer")
    def test_load_from_kafka_batch(
        self, mock_kafka_consumer, sample_transactions_batch
    ):
        """Test batch loading from Kafka."""
        mock_consumer = MagicMock()
        mock_consumer.__iter__.return_value = [
            MagicMock(value=txn) for txn in sample_transactions_batch
        ]
        mock_kafka_consumer.return_value = mock_consumer

        pipeline = RealtimePipeline(connect_db=False)
        df = pipeline.load_from_kafka_batch(count=3, timeout_seconds=1)

        assert len(df) == 3
        assert list(df.columns) == list(sample_transactions_batch[0].keys())
        assert pipeline.metrics["total_consumed"] == 3

    @pytest.mark.skipif(not KAFKA_AVAILABLE, reason="Kafka not available")
    @patch("src.pipelines.realtime_pipeline.KAFKA_AVAILABLE", True)
    @patch("src.pipelines.realtime_pipeline.KafkaConsumer")
    def test_load_from_kafka_batch_timeout(self, mock_kafka_consumer):
        """Test batch loading timeout."""
        mock_consumer = MagicMock()
        mock_consumer.__iter__.return_value = []  # No messages
        mock_kafka_consumer.return_value = mock_consumer

        pipeline = RealtimePipeline(connect_db=False)
        df = pipeline.load_from_kafka_batch(count=10, timeout_seconds=0.1)

        assert len(df) == 0

    @patch("src.pipelines.realtime_pipeline.DatabaseService")
    def test_clean_and_preprocess(self, mock_db_service, sample_transactions_df):
        """Test data cleaning and preprocessing."""
        mock_db_instance = MagicMock()
        mock_db_service.return_value = mock_db_instance

        pipeline = RealtimePipeline(connect_db=False)

        # Add some problematic data
        dirty_df = sample_transactions_df.copy()
        dirty_df.loc[0, "amount"] = np.nan  # Missing value
        dirty_df.loc[1, "transaction_id"] = dirty_df.loc[
            0, "transaction_id"
        ]  # Duplicate
        dirty_df.loc[2, "amount"] = 10000  # Outlier

        clean_df = pipeline.clean_and_preprocess(dirty_df)

        # Should have removed problematic rows
        assert len(clean_df) < len(dirty_df)
        # Should not have NaN values
        assert not clean_df.isnull().any().any()
        # Should have correct dtypes
        assert pd.api.types.is_numeric_dtype(clean_df["amount"])

    def test_clean_and_preprocess_empty_df(self):
        """Test cleaning with empty dataframe."""
        pipeline = RealtimePipeline(connect_db=False)

        # Create empty DataFrame with expected columns
        columns = (
            ["Time", "amount"]
            + [f"V{i}" for i in range(1, 29)]
            + ["transaction_id", "timestamp"]
        )
        empty_df = pd.DataFrame(columns=columns)
        result_df = pipeline.clean_and_preprocess(empty_df)

        assert len(result_df) == 0

    @patch("src.pipelines.realtime_pipeline.requests.post")
    def test_predict_batch_success(
        self, mock_requests_post, sample_clean_transactions_df
    ):
        """Test successful batch prediction."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "predictions": [
                {"is_fraud": False, "fraud_probability": 0.1},
                {"is_fraud": True, "fraud_probability": 0.9},
            ]
        }
        mock_requests_post.return_value = mock_response

        pipeline = RealtimePipeline(connect_db=False)
        result_df = pipeline.predict_batch(sample_clean_transactions_df)

        assert "predicted_fraud" in result_df.columns
        assert "fraud_probability" in result_df.columns
        assert "prediction_timestamp" in result_df.columns
        assert pipeline.metrics["total_predicted"] == len(sample_clean_transactions_df)

    @patch("src.pipelines.realtime_pipeline.requests.post")
    def test_predict_batch_api_error(
        self, mock_requests_post, sample_clean_transactions_df
    ):
        """Test batch prediction with API error."""
        mock_response = MagicMock()
        mock_response.status_code = 500
        mock_response.text = "Internal Server Error"
        mock_requests_post.return_value = mock_response

        pipeline = RealtimePipeline(connect_db=False)
        result_df = pipeline.predict_batch(sample_clean_transactions_df)

        # Should have None values for predictions
        assert result_df["predicted_fraud"].isnull().all()
        assert result_df["fraud_probability"].isnull().all()
        assert pipeline.metrics["errors"] == 1

    @patch("src.pipelines.realtime_pipeline.requests.post")
    def test_predict_stream_success(self, mock_requests_post, sample_transaction):
        """Test successful stream prediction."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"is_fraud": True, "fraud_probability": 0.85}
        mock_requests_post.return_value = mock_response

        pipeline = RealtimePipeline(connect_db=False)
        result = pipeline.predict_stream(sample_transaction)

        assert result["predicted_fraud"] is True
        assert result["fraud_probability"] == 0.85
        assert "prediction_timestamp" in result
        assert pipeline.metrics["total_predicted"] == 1
        assert pipeline.metrics["total_fraud_detected"] == 1

    @patch("src.pipelines.realtime_pipeline.requests.post")
    def test_predict_stream_api_error(self, mock_requests_post, sample_transaction):
        """Test stream prediction with API error."""
        mock_requests_post.side_effect = requests.exceptions.RequestException(
            "Connection failed"
        )

        pipeline = RealtimePipeline(connect_db=False)
        result = pipeline.predict_stream(sample_transaction)

        assert result["predicted_fraud"] is None
        assert result["fraud_probability"] is None
        assert pipeline.metrics["errors"] == 1

    def test_save_to_database_success(self, sample_predictions_df):
        """Test successful database save."""
        mock_db = MagicMock()
        mock_db.insert_transactions.return_value = 5

        pipeline = RealtimePipeline(connect_db=False)
        pipeline.db_service = mock_db

        saved_count = pipeline.save_to_database(sample_predictions_df)

        assert saved_count == 5
        mock_db.insert_transactions.assert_called_once()
        assert pipeline.metrics["total_saved"] == 5

    def test_save_to_database_error(self, sample_predictions_df):
        """Test database save error."""
        mock_db = MagicMock()
        mock_db.insert_transactions.side_effect = Exception("DB Error")

        pipeline = RealtimePipeline(connect_db=False)
        pipeline.db_service = mock_db

        saved_count = pipeline.save_to_database(sample_predictions_df)

        assert saved_count == 0
        assert pipeline.metrics["errors"] == 1

    @patch("src.pipelines.realtime_pipeline.requests.post")
    def test_send_to_webapp_success(self, mock_requests_post, sample_fraud_predictions):
        """Test successful webapp notification."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_requests_post.return_value = mock_response

        pipeline = RealtimePipeline(
            webapp_url="http://localhost:3000", connect_db=False
        )
        success = pipeline.send_to_webapp(sample_fraud_predictions)

        assert success is True
        mock_requests_post.assert_called_once()

    @patch("src.pipelines.realtime_pipeline.requests.post")
    def test_send_to_webapp_no_url(self, mock_requests_post):
        """Test webapp notification when no URL configured."""
        pipeline = RealtimePipeline(webapp_url=None, connect_db=False)
        success = pipeline.send_to_webapp([])

        assert success is False
        mock_requests_post.assert_not_called()

    @patch("src.pipelines.realtime_pipeline.requests.post")
    def test_send_to_webapp_error(self, mock_requests_post, sample_fraud_predictions):
        """Test webapp notification error."""
        mock_requests_post.side_effect = requests.exceptions.RequestException(
            "Connection failed"
        )

        pipeline = RealtimePipeline(
            webapp_url="http://localhost:3000", connect_db=False
        )
        success = pipeline.send_to_webapp(sample_fraud_predictions)

        assert success is False

    @patch.object(RealtimePipeline, "load_from_kafka_batch")
    @patch.object(RealtimePipeline, "clean_and_preprocess")
    @patch.object(RealtimePipeline, "predict_batch")
    @patch.object(RealtimePipeline, "save_to_database")
    @patch.object(RealtimePipeline, "send_to_webapp")
    def test_execute_batch_success(
        self,
        mock_send_webapp,
        mock_save_db,
        mock_predict,
        mock_clean,
        mock_load_kafka,
        sample_clean_transactions_df,
        sample_predictions_df,
    ):
        """Test successful batch execution."""
        # Setup mocks
        mock_load_kafka.return_value = sample_clean_transactions_df
        mock_clean.return_value = sample_clean_transactions_df
        mock_predict.return_value = sample_predictions_df
        mock_save_db.return_value = 3
        mock_send_webapp.return_value = True

        pipeline = RealtimePipeline(connect_db=False)
        result = pipeline.execute_batch(count=3)

        assert result["status"] == "success"
        assert result["consumed"] == 2  # sample_clean_transactions_df has 2 rows
        assert result["predicted"] == 2
        assert result["saved"] == 3  # mock returns 3

        # Verify all steps called
        mock_load_kafka.assert_called_once_with(count=3)
        mock_clean.assert_called_once()
        mock_predict.assert_called_once()
        mock_save_db.assert_called_once()
        mock_send_webapp.assert_called_once()

    @patch.object(RealtimePipeline, "load_from_kafka_batch")
    def test_execute_batch_no_data(self, mock_load_kafka):
        """Test batch execution with no data."""
        mock_load_kafka.return_value = pd.DataFrame()

        pipeline = RealtimePipeline(connect_db=False)
        result = pipeline.execute_batch(count=10)

        assert result["status"] == "error"
        assert "No transactions" in result["message"]

    @patch.object(RealtimePipeline, "load_from_kafka_batch")
    @patch.object(RealtimePipeline, "clean_and_preprocess")
    def test_execute_batch_all_filtered(
        self, mock_clean, mock_load_kafka, sample_transactions_df
    ):
        """Test batch execution where all data is filtered."""
        mock_load_kafka.return_value = sample_transactions_df
        mock_clean.return_value = pd.DataFrame()  # All filtered out

        pipeline = RealtimePipeline(connect_db=False)
        result = pipeline.execute_batch(count=5)

        assert result["status"] == "error"
        assert "All transactions filtered" in result["message"]

    @patch.object(RealtimePipeline, "load_from_kafka_batch")
    @patch.object(RealtimePipeline, "clean_and_preprocess")
    @patch.object(RealtimePipeline, "predict_batch")
    def test_execute_batch_prediction_error(
        self, mock_predict, mock_clean, mock_load_kafka, sample_clean_transactions_df
    ):
        """Test batch execution with prediction error."""
        mock_load_kafka.return_value = sample_clean_transactions_df
        mock_clean.return_value = sample_clean_transactions_df
        mock_predict.side_effect = Exception("Prediction failed")

        pipeline = RealtimePipeline(connect_db=False)
        result = pipeline.execute_batch(count=3)

        assert result["status"] == "error"
        assert "Prediction failed" in result["message"]

    @pytest.mark.skipif(not KAFKA_AVAILABLE, reason="Kafka not available")
    @patch("src.pipelines.realtime_pipeline.KAFKA_AVAILABLE", True)
    @patch("src.pipelines.realtime_pipeline.KafkaConsumer")
    @patch.object(RealtimePipeline, "_process_stream_buffer")
    def test_execute_stream(self, mock_process_buffer, mock_kafka_consumer):
        """Test stream execution."""
        mock_consumer = MagicMock()
        messages = [MagicMock(value={"test": "data"})]
        mock_consumer.__iter__.return_value = messages
        mock_kafka_consumer.return_value = mock_consumer

        pipeline = RealtimePipeline(connect_db=False)

        # Mock to stop after first message
        call_count = 0

        def side_effect(transaction):
            nonlocal call_count
            call_count += 1
            if call_count >= 1:
                raise KeyboardInterrupt()

        mock_process_buffer.side_effect = side_effect

        # Should not raise exception
        pipeline.execute_stream(interval_seconds=1)

    def test_process_stream_buffer(self, sample_transactions_list):
        """Test stream buffer processing."""
        with patch.object(
            RealtimePipeline, "clean_and_preprocess"
        ) as mock_clean, patch.object(
            RealtimePipeline, "predict_batch"
        ) as mock_predict, patch.object(
            RealtimePipeline, "save_to_database"
        ) as mock_save, patch.object(
            RealtimePipeline, "send_to_webapp"
        ) as mock_send:
            mock_clean.return_value = pd.DataFrame(sample_transactions_list)
            # Mock predict to return DataFrame with fraud predictions
            fraud_df = pd.DataFrame(sample_transactions_list)
            fraud_df["predicted_fraud"] = [1, 0]  # One fraud, one not
            fraud_df["fraud_probability"] = [0.9, 0.1]
            mock_predict.return_value = fraud_df
            mock_save.return_value = 2
            mock_send.return_value = True

            pipeline = RealtimePipeline(connect_db=False)
            pipeline._process_stream_buffer(sample_transactions_list)

            mock_clean.assert_called_once()
            mock_predict.assert_called_once()
            mock_save.assert_called_once()
            mock_send.assert_called_once()

    def test_get_metrics(self):
        """Test metrics retrieval."""
        pipeline = RealtimePipeline(connect_db=False)

        # Modify some metrics
        pipeline.metrics["total_consumed"] = 100
        pipeline.metrics["total_fraud_detected"] = 15

        metrics = pipeline.get_metrics()

        assert metrics["total_consumed"] == 100
        assert metrics["total_fraud_detected"] == 15
        assert isinstance(metrics, dict)

    def test_pipeline_with_custom_db_service(self):
        """Test pipeline with custom database service."""
        custom_db = MagicMock()

        pipeline = RealtimePipeline(db_service=custom_db, connect_db=False)

        assert pipeline.db_service == custom_db

    @patch("src.pipelines.realtime_pipeline.logger")
    def test_logging_on_operations(self, mock_logger, sample_transactions_df):
        """Test logging during operations."""
        pipeline = RealtimePipeline(connect_db=False)

        pipeline.clean_and_preprocess(sample_transactions_df)

        assert mock_logger.info.called

    def test_batch_execution_timing(self):
        """Test that batch execution includes timing information."""
        with patch.object(
            RealtimePipeline, "load_from_kafka_batch"
        ) as mock_load, patch.object(
            RealtimePipeline, "clean_and_preprocess"
        ) as mock_clean, patch.object(
            RealtimePipeline, "predict_batch"
        ) as mock_predict, patch.object(
            RealtimePipeline, "save_to_database"
        ) as mock_save, patch.object(
            RealtimePipeline, "send_to_webapp"
        ) as mock_send:
            mock_load.return_value = pd.DataFrame({"test": [1, 2, 3]})
            mock_clean.return_value = pd.DataFrame({"test": [1, 2, 3]})
            mock_predict.return_value = pd.DataFrame(
                {"test": [1, 2, 3], "predicted_fraud": [0, 1, 0]}
            )
            mock_save.return_value = 3
            mock_send.return_value = True

            pipeline = RealtimePipeline(connect_db=False)
            result = pipeline.execute_batch(count=3)

            assert "elapsed_seconds" in result
            assert result["elapsed_seconds"] >= 0

    def test_fraud_alerts_only_sent_to_webapp(self, sample_mixed_predictions_df):
        """Test that only fraud predictions are sent to webapp during batch execution."""
        fraud_only = sample_mixed_predictions_df[
            sample_mixed_predictions_df["predicted_fraud"] == 1
        ]

        with patch.object(
            RealtimePipeline, "load_from_kafka_batch"
        ) as mock_load, patch.object(
            RealtimePipeline, "clean_and_preprocess"
        ) as mock_clean, patch.object(
            RealtimePipeline, "predict_batch"
        ) as mock_predict, patch.object(
            RealtimePipeline, "save_to_database"
        ) as mock_save, patch.object(
            RealtimePipeline, "send_to_webapp"
        ) as mock_send:
            mock_load.return_value = sample_mixed_predictions_df.drop(
                ["predicted_fraud", "fraud_probability"], axis=1
            )
            mock_clean.return_value = sample_mixed_predictions_df.drop(
                ["predicted_fraud", "fraud_probability"], axis=1
            )
            mock_predict.return_value = sample_mixed_predictions_df
            mock_save.return_value = len(sample_mixed_predictions_df)
            mock_send.return_value = True

            pipeline = RealtimePipeline(
                webapp_url="http://localhost:3000", connect_db=False
            )
            pipeline.execute_batch(count=3)

            # Should send only fraud transactions
            if len(fraud_only) > 0:
                mock_send.assert_called_once()
                call_args = mock_send.call_args[0][0]
                assert all(item["predicted_fraud"] == 1 for item in call_args)


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
        "V11": -0.4,
        "V12": 0.9,
        "V13": -0.6,
        "V14": 0.3,
        "V15": -0.2,
        "V16": 0.7,
        "V17": -0.5,
        "V18": 0.1,
        "V19": -0.8,
        "V20": 0.4,
        "V21": -0.3,
        "V22": 0.6,
        "V23": -0.9,
        "V24": 0.2,
        "V25": -0.1,
        "V26": 0.5,
        "V27": -0.7,
        "V28": 0.3,
        "transaction_id": "txn_abc123def456",
        "timestamp": "2025-01-15T14:30:00Z",
        "source": "simulator",
    }


@pytest.fixture
def sample_transactions_batch():
    """Sample batch of transactions fixture."""
    return [
        {
            "Time": 100.0,
            "amount": 100.0,
            "Class": 0,
            "V1": -1.0,
            "V2": 0.5,
            "V3": -0.5,
            "V4": 1.0,
            "V5": -0.8,
            "V6": 0.2,
            "V7": -0.3,
            "V8": 0.6,
            "V9": -0.7,
            "V10": 0.8,
            "V11": -0.4,
            "V12": 0.9,
            "V13": -0.6,
            "V14": 0.3,
            "V15": -0.2,
            "V16": 0.7,
            "V17": -0.5,
            "V18": 0.1,
            "V19": -0.8,
            "V20": 0.4,
            "V21": -0.3,
            "V22": 0.6,
            "V23": -0.9,
            "V24": 0.2,
            "V25": -0.1,
            "V26": 0.5,
            "V27": -0.7,
            "V28": 0.3,
            "transaction_id": "txn_001",
            "timestamp": "2025-01-15T10:00:00Z",
        },
        {
            "Time": 200.0,
            "amount": 200.0,
            "Class": 1,
            "V1": -2.0,
            "V2": 1.0,
            "V3": -1.0,
            "V4": 2.0,
            "V5": -1.6,
            "V6": 0.4,
            "V7": -0.6,
            "V8": 1.2,
            "V9": -1.4,
            "V10": 1.6,
            "V11": -0.8,
            "V12": 1.3,
            "V13": -1.2,
            "V14": 0.7,
            "V15": -0.5,
            "V16": 1.1,
            "V17": -0.9,
            "V18": 0.6,
            "V19": -1.5,
            "V20": 0.8,
            "V21": -0.7,
            "V22": 1.0,
            "V23": -1.3,
            "V24": 0.4,
            "V25": -0.6,
            "V26": 0.9,
            "V27": -1.1,
            "V28": 0.5,
            "transaction_id": "txn_002",
            "timestamp": "2025-01-15T11:00:00Z",
        },
        {
            "Time": 300.0,
            "amount": 300.0,
            "Class": 0,
            "V1": -0.5,
            "V2": 0.25,
            "V3": -0.25,
            "V4": 0.5,
            "V5": -0.4,
            "V6": 0.1,
            "V7": -0.15,
            "V8": 0.3,
            "V9": -0.35,
            "V10": 0.4,
            "V11": -0.2,
            "V12": 0.45,
            "V13": -0.3,
            "V14": 0.15,
            "V15": -0.1,
            "V16": 0.35,
            "V17": -0.25,
            "V18": 0.05,
            "V19": -0.4,
            "V20": 0.2,
            "V21": -0.15,
            "V22": 0.3,
            "V23": -0.45,
            "V24": 0.1,
            "V25": -0.05,
            "V26": 0.25,
            "V27": -0.35,
            "V28": 0.15,
            "transaction_id": "txn_003",
            "timestamp": "2025-01-15T12:00:00Z",
        },
    ]


@pytest.fixture
def sample_transactions_df(sample_transactions_batch):
    """Sample transactions DataFrame fixture."""
    return pd.DataFrame(sample_transactions_batch)


@pytest.fixture
def sample_clean_transactions_df():
    """Sample clean transactions DataFrame fixture."""
    return pd.DataFrame(
        {
            "Time": [100.0, 200.0],
            "amount": [150.0, 250.0],
            "V1": [-1.0, -2.0],
            "V2": [0.5, 1.0],
            "V3": [-0.5, -1.0],
            "V4": [1.0, 2.0],
            "V5": [-0.8, -1.6],
            "V6": [0.2, 0.4],
            "V7": [-0.3, -0.6],
            "V8": [0.6, 1.2],
            "V9": [-0.7, -1.4],
            "V10": [0.8, 1.6],
            "V11": [-0.4, -0.8],
            "V12": [0.9, 1.3],
            "V13": [-0.6, -1.2],
            "V14": [0.3, 0.7],
            "V15": [-0.2, -0.5],
            "V16": [0.7, 1.1],
            "V17": [-0.5, -0.9],
            "V18": [0.1, 0.6],
            "V19": [-0.8, -1.5],
            "V20": [0.4, 0.8],
            "V21": [-0.3, -0.7],
            "V22": [0.6, 1.0],
            "V23": [-0.9, -1.3],
            "V24": [0.2, 0.4],
            "V25": [-0.1, -0.6],
            "V26": [0.5, 0.9],
            "V27": [-0.7, -1.1],
            "V28": [0.3, 0.5],
            "transaction_id": ["txn_001", "txn_002"],
            "timestamp": ["2025-01-15T10:00:00Z", "2025-01-15T11:00:00Z"],
        }
    )


@pytest.fixture
def sample_predictions_df(sample_clean_transactions_df):
    """Sample predictions DataFrame fixture."""
    df = sample_clean_transactions_df.copy()
    df["predicted_fraud"] = [0, 1]
    df["fraud_probability"] = [0.1, 0.9]
    df["prediction_timestamp"] = ["2025-01-15T10:05:00Z", "2025-01-15T11:05:00Z"]
    return df


@pytest.fixture
def sample_fraud_predictions():
    """Sample fraud predictions fixture."""
    return [
        {
            "transaction_id": "txn_fraud001",
            "predicted_fraud": 1,
            "fraud_probability": 0.95,
            "amount": 5000.0,
        },
        {
            "transaction_id": "txn_fraud002",
            "predicted_fraud": 1,
            "fraud_probability": 0.88,
            "amount": 2500.0,
        },
    ]


@pytest.fixture
def sample_transactions_list():
    """Sample transactions list fixture."""
    return [
        {"Time": 100.0, "amount": 100.0, "transaction_id": "txn_001"},
        {"Time": 200.0, "amount": 200.0, "transaction_id": "txn_002"},
    ]


@pytest.fixture
def sample_mixed_predictions_df():
    """Sample DataFrame with mixed predictions fixture."""
    return pd.DataFrame(
        {
            "transaction_id": ["txn_001", "txn_002", "txn_003"],
            "predicted_fraud": [0, 1, 0],
            "fraud_probability": [0.1, 0.9, 0.2],
            "amount": [100.0, 5000.0, 200.0],
        }
    )
