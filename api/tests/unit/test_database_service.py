"""
Unit tests for Database Service
Tests database connections, queries, transactions, and error handling
"""

from datetime import datetime, timedelta
from unittest.mock import MagicMock, Mock, patch

import pandas as pd
import pytest
from src.services.database_service import (
    DatabaseConnectionError,
    DatabaseService,
    QueryExecutionError,
    TransactionError,
)


@pytest.mark.unit
class TestDatabaseService:
    """Test suite for DatabaseService class."""

    def test_initialization(self, test_db_settings):
        """Test database service initialization."""
        service = DatabaseService(test_db_settings)

        assert service.settings == test_db_settings
        assert service.connection_pool is None
        assert service.connection_string is not None

    @patch("src.services.database_service.psycopg2.connect")
    def test_connect_success(self, mock_connect, test_db_settings):
        """Test successful database connection."""
        mock_connection = MagicMock()
        mock_connect.return_value = mock_connection

        service = DatabaseService(test_db_settings)

        connection = service.connect()

        assert connection == mock_connection
        mock_connect.assert_called_once()

    @patch("src.services.database_service.psycopg2.connect")
    def test_connect_failure(self, mock_connect, test_db_settings):
        """Test database connection failure."""
        mock_connect.side_effect = Exception("Connection failed")

        service = DatabaseService(test_db_settings)

        with pytest.raises(DatabaseConnectionError):
            service.connect()

    def test_execute_query_success(self, test_db_settings, sample_query_result):
        """Test successful query execution."""
        service = DatabaseService(test_db_settings)
        mock_connection = Mock()
        mock_cursor = Mock()

        # Set up cursor as a context manager that returns mock_cursor
        mock_connection.cursor.return_value.__enter__ = Mock(return_value=mock_cursor)
        mock_connection.cursor.return_value.__exit__ = Mock(return_value=None)

        mock_cursor.fetchall.return_value = sample_query_result
        mock_cursor.description = [("id",), ("amount",), ("is_fraud",)]

        service.connection_pool = mock_connection

        result = service.execute_query("SELECT * FROM transactions")

        assert isinstance(result, pd.DataFrame)
        assert len(result) == 2
        assert "id" in result.columns
        assert "amount" in result.columns
        assert "is_fraud" in result.columns

    def test_execute_query_no_results(self, test_db_settings):
        """Test query execution with no results."""
        service = DatabaseService(test_db_settings)
        mock_connection = Mock()
        mock_cursor = Mock()

        # Set up cursor as a context manager
        mock_connection.cursor.return_value.__enter__ = Mock(return_value=mock_cursor)
        mock_connection.cursor.return_value.__exit__ = Mock(return_value=None)

        mock_cursor.fetchall.return_value = []
        mock_cursor.description = [("id",), ("amount",)]

        service.connection_pool = mock_connection

        result = service.execute_query("SELECT * FROM transactions WHERE 1=0")

        assert isinstance(result, pd.DataFrame)
        assert len(result) == 0

    def test_execute_query_with_params(self, test_db_settings, sample_query_result):
        """Test query execution with parameters."""
        service = DatabaseService(test_db_settings)
        mock_connection = Mock()
        mock_cursor = Mock()

        # Set up cursor as a context manager
        mock_connection.cursor.return_value.__enter__ = Mock(return_value=mock_cursor)
        mock_connection.cursor.return_value.__exit__ = Mock(return_value=None)

        mock_cursor.fetchall.return_value = sample_query_result
        mock_cursor.description = [("id",), ("amount",), ("is_fraud",)]

        service.connection_pool = mock_connection

        result = service.execute_query(
            "SELECT * FROM transactions WHERE amount > %s", params=(100.0,)
        )

        assert isinstance(result, pd.DataFrame)
        mock_cursor.execute.assert_called_with(
            "SELECT * FROM transactions WHERE amount > %s", (100.0,)
        )

    def test_execute_query_error(self, test_db_settings):
        """Test query execution error."""
        service = DatabaseService(test_db_settings)
        mock_connection = Mock()
        mock_cursor = Mock()

        # Set up cursor as a context manager
        mock_connection.cursor.return_value.__enter__ = Mock(return_value=mock_cursor)
        mock_connection.cursor.return_value.__exit__ = Mock(return_value=None)

        mock_cursor.execute.side_effect = Exception("Query failed")

        service.connection_pool = mock_connection

        with pytest.raises(QueryExecutionError):
            service.execute_query("SELECT * FROM invalid_table")

    def test_execute_non_select_query(self, test_db_settings):
        """Test execution of non-SELECT queries."""
        service = DatabaseService(test_db_settings)
        mock_connection = Mock()
        mock_cursor = Mock()

        # Set up cursor as a context manager
        mock_connection.cursor.return_value.__enter__ = Mock(return_value=mock_cursor)
        mock_connection.cursor.return_value.__exit__ = Mock(return_value=None)

        mock_cursor.rowcount = 5  # Rows affected

        service.connection_pool = mock_connection

        result = service.execute_query("UPDATE transactions SET status = 'processed'")

        assert result == 5  # Rows affected

    def test_insert_transaction_success(
        self, test_db_settings, sample_transaction_data
    ):
        """Test successful transaction insertion."""
        service = DatabaseService(test_db_settings)
        mock_connection = Mock()
        mock_cursor = Mock()

        # Set up cursor as a context manager
        mock_connection.cursor.return_value.__enter__ = Mock(return_value=mock_cursor)
        mock_connection.cursor.return_value.__exit__ = Mock(return_value=None)

        mock_cursor.fetchone.return_value = (123,)  # Returned transaction ID

        service.connection_pool = mock_connection

        transaction_id = service.insert_transaction(sample_transaction_data)

        assert transaction_id == 123
        assert mock_cursor.execute.called

    def test_insert_transaction_error(self, test_db_settings, sample_transaction_data):
        """Test transaction insertion error."""
        service = DatabaseService(test_db_settings)
        mock_connection = Mock()
        mock_cursor = Mock()

        # Set up cursor as a context manager
        mock_connection.cursor.return_value.__enter__ = Mock(return_value=mock_cursor)
        mock_connection.cursor.return_value.__exit__ = Mock(return_value=None)

        mock_cursor.execute.side_effect = Exception("Insert failed")

        service.connection_pool = mock_connection

        with pytest.raises(TransactionError):
            service.insert_transaction(sample_transaction_data)

    def test_bulk_insert_transactions(
        self, test_db_settings, sample_transactions_batch
    ):
        """Test bulk transaction insertion."""
        service = DatabaseService(test_db_settings)
        mock_connection = Mock()
        mock_cursor = Mock()

        # Set up cursor as a context manager
        mock_connection.cursor.return_value.__enter__ = Mock(return_value=mock_cursor)
        mock_connection.cursor.return_value.__exit__ = Mock(return_value=None)

        mock_cursor.rowcount = 3  # Rows inserted

        service.connection_pool = mock_connection

        result = service.bulk_insert_transactions(sample_transactions_batch)

        assert result == 3
        assert mock_cursor.executemany.called

    def test_bulk_insert_empty_batch(self, test_db_settings):
        """Test bulk insert with empty batch."""
        service = DatabaseService(test_db_settings)

        # For empty batch, should return 0 without connecting
        result = service.bulk_insert_transactions([])

        assert result == 0

    def test_get_transaction_by_id(self, test_db_settings, sample_query_result):
        """Test retrieving transaction by ID."""
        service = DatabaseService(test_db_settings)
        mock_connection = Mock()
        mock_cursor = Mock()

        # Set up cursor as a context manager
        mock_connection.cursor.return_value.__enter__ = Mock(return_value=mock_cursor)
        mock_connection.cursor.return_value.__exit__ = Mock(return_value=None)

        mock_cursor.fetchall.return_value = [sample_query_result[0]]
        mock_cursor.description = [("id",), ("amount",), ("is_fraud",)]

        service.connection_pool = mock_connection

        result = service.get_transaction_by_id(123)

        assert isinstance(result, pd.Series)
        assert result["id"] == sample_query_result[0][0]

    def test_get_transaction_not_found(self, test_db_settings):
        """Test retrieving non-existent transaction."""
        service = DatabaseService(test_db_settings)
        mock_connection = Mock()
        mock_cursor = Mock()

        # Set up cursor as a context manager
        mock_connection.cursor.return_value.__enter__ = Mock(return_value=mock_cursor)
        mock_connection.cursor.return_value.__exit__ = Mock(return_value=None)

        mock_cursor.fetchall.return_value = []
        mock_cursor.description = [("id",), ("amount",), ("is_fraud",)]

        service.connection_pool = mock_connection

        result = service.get_transaction_by_id(999)

        assert result is None

    def test_update_transaction_status(self, test_db_settings):
        """Test transaction status update."""
        service = DatabaseService(test_db_settings)
        mock_connection = Mock()
        mock_cursor = Mock()

        # Set up cursor as a context manager
        mock_connection.cursor.return_value.__enter__ = Mock(return_value=mock_cursor)
        mock_connection.cursor.return_value.__exit__ = Mock(return_value=None)

        mock_cursor.rowcount = 1

        service.connection_pool = mock_connection

        result = service.update_transaction_status(123, "processed")

        assert result is True
        assert mock_cursor.execute.called

    def test_update_transaction_status_not_found(self, test_db_settings):
        """Test transaction status update for non-existent transaction."""
        service = DatabaseService(test_db_settings)
        mock_connection = Mock()
        mock_cursor = Mock()

        # Set up cursor as a context manager
        mock_connection.cursor.return_value.__enter__ = Mock(return_value=mock_cursor)
        mock_connection.cursor.return_value.__exit__ = Mock(return_value=None)

        mock_cursor.rowcount = 0

        service.connection_pool = mock_connection

        result = service.update_transaction_status(999, "processed")

        assert result is False

    def test_get_recent_transactions(self, test_db_settings, sample_query_result):
        """Test retrieving recent transactions."""
        service = DatabaseService(test_db_settings)
        mock_connection = Mock()
        mock_cursor = Mock()

        # Set up cursor as a context manager
        mock_connection.cursor.return_value.__enter__ = Mock(return_value=mock_cursor)
        mock_connection.cursor.return_value.__exit__ = Mock(return_value=None)

        mock_cursor.fetchall.return_value = sample_query_result
        mock_cursor.description = [("id",), ("amount",), ("is_fraud",)]

        service.connection_pool = mock_connection

        result = service.get_recent_transactions(limit=10)

        assert isinstance(result, pd.DataFrame)
        assert len(result) == 2

    def test_get_fraud_statistics(self, test_db_settings):
        """Test fraud statistics retrieval."""
        service = DatabaseService(test_db_settings)
        mock_connection = Mock()
        mock_cursor = Mock()

        # Set up cursor as a context manager
        mock_connection.cursor.return_value.__enter__ = Mock(return_value=mock_cursor)
        mock_connection.cursor.return_value.__exit__ = Mock(return_value=None)

        mock_cursor.fetchall.return_value = [
            (
                1000,
                50,
                0.05,
                250.0,
                1500.0,
            )  # total, fraud_count, rate, avg_amount, max_amount
        ]
        mock_cursor.description = [
            ("total_transactions",),
            ("fraud_count",),
            ("fraud_rate",),
            ("avg_fraud_amount",),
            ("max_fraud_amount",),
        ]

        service.connection_pool = mock_connection

        result = service.get_fraud_statistics(hours=24)

        assert isinstance(result, dict)
        assert "total_transactions" in result
        assert "fraud_count" in result
        assert "fraud_rate" in result
        assert result["fraud_rate"] == 0.05

    def test_transaction_context_manager(self, test_db_settings):
        """Test transaction context manager."""
        service = DatabaseService(test_db_settings)
        mock_connection = Mock()

        service.connection_pool = mock_connection

        with service.transaction() as conn:
            assert conn == mock_connection

        # Should commit on success
        mock_connection.commit.assert_called_once()

    def test_transaction_context_manager_rollback(self, test_db_settings):
        """Test transaction rollback on exception."""
        service = DatabaseService(test_db_settings)
        mock_connection = Mock()

        service.connection_pool = mock_connection

        with pytest.raises(Exception):
            with service.transaction() as conn:
                raise Exception("Test error")

        # Should rollback on error
        mock_connection.rollback.assert_called_once()
        mock_connection.commit.assert_not_called()

    def test_connection_pool_management(self, test_db_settings):
        """Test connection pool management."""
        service = DatabaseService(test_db_settings)

        # Initially no pool
        assert service.connection_pool is None

        # Get connection creates pool
        with patch("src.services.database_service.psycopg2.connect") as mock_connect:
            mock_connection = Mock()
            mock_connect.return_value = mock_connection

            conn1 = service.connect()
            conn2 = service.connect()

            # Should reuse connection
            assert conn1 == mock_connection
            assert conn2 == mock_connection

    def test_health_check_healthy(self, test_db_settings):
        """Test database health check when healthy."""
        service = DatabaseService(test_db_settings)

        with patch.object(service, "connect") as mock_connect:
            mock_connection = Mock()
            mock_connect.return_value.__enter__.return_value = mock_connection

            health = service.health_check()

            assert health["status"] == "healthy"
            assert health["connection"] == "ok"

    def test_health_check_unhealthy(self, test_db_settings):
        """Test database health check when unhealthy."""
        service = DatabaseService(test_db_settings)

        with patch.object(service, "connect") as mock_connect:
            mock_connect.side_effect = DatabaseConnectionError("Connection failed")

            health = service.health_check()

            assert health["status"] == "unhealthy"
            assert "connection_error" in health

    def test_query_with_timeout(self, test_db_settings, sample_query_result):
        """Test query execution with timeout."""
        service = DatabaseService(test_db_settings)
        mock_connection = Mock()
        mock_cursor = Mock()

        # Set up cursor as a context manager
        mock_connection.cursor.return_value.__enter__ = Mock(return_value=mock_cursor)
        mock_connection.cursor.return_value.__exit__ = Mock(return_value=None)

        mock_cursor.fetchall.return_value = sample_query_result
        mock_cursor.description = [("id",), ("amount",), ("is_fraud",)]

        service.connection_pool = mock_connection

        result = service.execute_query("SELECT * FROM transactions", timeout=30)

        assert isinstance(result, pd.DataFrame)
        # In real implementation, timeout would be set on cursor/connection

    def test_large_result_set_handling(self, test_db_settings):
        """Test handling of large result sets."""
        service = DatabaseService(test_db_settings)
        mock_connection = Mock()
        mock_cursor = Mock()

        # Set up cursor as a context manager
        mock_connection.cursor.return_value.__enter__ = Mock(return_value=mock_cursor)
        mock_connection.cursor.return_value.__exit__ = Mock(return_value=None)

        # Simulate large result set
        large_results = [[f"value_{i}", i, i % 2] for i in range(10000)]
        mock_cursor.fetchall.return_value = large_results
        mock_cursor.description = [("id",), ("amount",), ("is_fraud",)]

        service.connection_pool = mock_connection

        result = service.execute_query("SELECT * FROM large_table")

        assert isinstance(result, pd.DataFrame)
        assert len(result) == 10000

    @patch("src.services.database_service.logger")
    def test_logging_on_query_execution(
        self, mock_logger, test_db_settings, sample_query_result
    ):
        """Test logging on query execution."""
        service = DatabaseService(test_db_settings)
        mock_connection = Mock()
        mock_cursor = Mock()

        # Set up cursor as a context manager
        mock_connection.cursor.return_value.__enter__ = Mock(return_value=mock_cursor)
        mock_connection.cursor.return_value.__exit__ = Mock(return_value=None)

        mock_cursor.fetchall.return_value = sample_query_result
        mock_cursor.description = [("id",), ("amount",), ("is_fraud",)]

        service.connection_pool = mock_connection

        service.execute_query("SELECT * FROM transactions")

        assert mock_logger.debug.called

    def test_connection_string_construction(self, test_db_settings):
        """Test database connection string construction."""
        service = DatabaseService(test_db_settings)

        expected_parts = [
            f"host={test_db_settings.db_host}",
            f"port={test_db_settings.db_port}",
            f"dbname={test_db_settings.db_name}",
            f"user={test_db_settings.db_user}",
            f"password={test_db_settings.db_password}",
        ]

        for part in expected_parts:
            assert part in service.connection_string

    @patch("src.services.database_service.logger")
    def test_logging_on_connection_error(self, mock_logger, test_db_settings):
        """Test logging on connection errors."""
        service = DatabaseService(test_db_settings)

        # Mock psycopg2.connect to raise an exception
        with patch("src.services.database_service.psycopg2.connect") as mock_connect:
            mock_connect.side_effect = Exception("Connection failed")

            try:
                service.connect()
            except DatabaseConnectionError:
                pass

            assert mock_logger.error.called


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def test_db_settings():
    """Test database settings fixture."""

    # Create a mock settings object for tests
    class MockSettings:
        def __init__(self):
            self.db_host = "localhost"
            self.db_port = 5432
            self.db_name = "fraud_detection"
            self.db_user = "fraud_user"
            self.db_password = "fraud_pass"
            self.connection_pool_min = 1
            self.connection_pool_max = 10

    return MockSettings()


@pytest.fixture
def sample_query_result():
    """Sample query result fixture."""
    return [(1, 100.50, 0), (2, 250.75, 1)]


@pytest.fixture
def sample_transaction_data():
    """Sample transaction data fixture."""
    return {
        "time": 123456.789,
        "v1": -1.3598071336738,
        "v2": -0.0727811733098497,
        "v3": 2.53634673796914,
        "v4": 1.37815522427443,
        "v5": -0.338320769942518,
        "v6": 0.462387777762292,
        "v7": 0.239598554061257,
        "v8": 0.0986979012610507,
        "v9": 0.363786969611213,
        "v10": 0.0907941719789316,
        "v11": -0.551599533260813,
        "v12": -0.617800855762348,
        "v13": -0.991389847235408,
        "v14": -0.311169353699879,
        "v15": 1.46817697209427,
        "v16": -0.470400525259478,
        "v17": 0.207971241929242,
        "v18": 0.0257905801985591,
        "v19": 0.403992960255733,
        "v20": 0.251412098239705,
        "v21": -0.018306777944153,
        "v22": 0.277837575558899,
        "v23": -0.110473910188767,
        "v24": 0.0669280749146731,
        "v25": 0.128539358273528,
        "v26": -0.189114843888824,
        "v27": 0.133558376740387,
        "v28": -0.0210530534538215,
        "amount": 149.62,
        "Class": 0,
    }


@pytest.fixture
def sample_transactions_batch():
    """Batch of sample transactions fixture."""
    return [
        {
            "time": 123456.789,
            "v1": -1.3598071336738,
            "v2": -0.0727811733098497,
            "v3": 2.53634673796914,
            "v4": 1.37815522427443,
            "v5": -0.338320769942518,
            "v6": 0.462387777762292,
            "v7": 0.239598554061257,
            "v8": 0.0986979012610507,
            "v9": 0.363786969611213,
            "v10": 0.0907941719789316,
            "v11": -0.551599533260813,
            "v12": -0.617800855762348,
            "v13": -0.991389847235408,
            "v14": -0.311169353699879,
            "v15": 1.46817697209427,
            "v16": -0.470400525259478,
            "v17": 0.207971241929242,
            "v18": 0.0257905801985591,
            "v19": 0.403992960255733,
            "v20": 0.251412098239705,
            "v21": -0.018306777944153,
            "v22": 0.277837575558899,
            "v23": -0.110473910188767,
            "v24": 0.0669280749146731,
            "v25": 0.128539358273528,
            "v26": -0.189114843888824,
            "v27": 0.133558376740387,
            "v28": -0.0210530534538215,
            "amount": 149.62,
            "Class": 0,
        },
        {
            "time": 123457.123,
            "v1": 1.19185711131486,
            "v2": 0.26615071205963,
            "v3": 0.16648011335321,
            "v4": 0.448154078460911,
            "v5": 0.0600176492822243,
            "v6": -0.0823608088155687,
            "v7": -0.0788029833323113,
            "v8": 0.0851016549148104,
            "v9": -0.255425128109186,
            "v10": -0.166974414004614,
            "v11": 1.61272666105479,
            "v12": 1.06523531137287,
            "v13": 0.48909501589608,
            "v14": -0.143772296441519,
            "v15": 0.635558093258208,
            "v16": 0.463917041022171,
            "v17": -0.114804663102346,
            "v18": -0.183361270123994,
            "v19": -0.145783041325259,
            "v20": -0.0690831352230203,
            "v21": -0.225775248033138,
            "v22": -0.638671952771851,
            "v23": 0.101288021253234,
            "v24": -0.339846475529127,
            "v25": 0.167170404418143,
            "v26": 0.125894532368176,
            "v27": -0.00898309914322813,
            "v28": 0.0147241697264928,
            "amount": 2.69,
            "Class": 1,
        },
        {
            "time": 123458.456,
            "v1": -1.35835406159823,
            "v2": -1.34016307473609,
            "v3": 1.77320934263119,
            "v4": 0.379779593034328,
            "v5": -0.503198133318973,
            "v6": 1.80049938079263,
            "v7": 0.791460956450422,
            "v8": 0.247675786588991,
            "v9": -1.51465432260583,
            "v10": 0.207642865216696,
            "v11": 0.624501459424895,
            "v12": 0.066083685268831,
            "v13": 0.717292731410831,
            "v14": -0.165945922763554,
            "v15": 2.34586494901581,
            "v16": -2.89008319444231,
            "v17": 1.10996937869599,
            "v18": -0.121359313195888,
            "v19": -2.26185709530414,
            "v20": 0.524979725224404,
            "v21": 0.247998153469754,
            "v22": 0.771679401917229,
            "v23": 0.909412262347719,
            "v24": -0.689280956490685,
            "v25": -0.327641833735251,
            "v26": -0.139096571514147,
            "v27": -0.0553527940384261,
            "v28": -0.0597518405929204,
            "amount": 378.66,
            "Class": 0,
        },
    ]
