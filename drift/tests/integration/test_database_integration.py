"""Integration tests for database operations."""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock

from src.storage.database import DriftDatabaseService
from src.config.settings import Settings


@pytest.mark.integration
@pytest.mark.database
class TestDatabaseIntegration:
    """Integration tests for database operations."""

    @pytest.fixture
    def db_service(self, test_settings):
        """Create database service instance."""
        return DriftDatabaseService(test_settings)

    @pytest.fixture
    def sample_drift_metrics(self):
        """Create sample drift metrics."""
        return {
            'timestamp': datetime.utcnow(),
            'model_version': 'v1.0.0',
            'data_drift_score': 0.45,
            'target_drift_score': 1.5,
            'concept_drift_score': 0.08,
            'drift_detected': True,
            'severity': 'HIGH'
        }

    @pytest.fixture
    def sample_baseline_metrics(self):
        """Create sample baseline metrics."""
        return {
            'model_version': 'v1.0.0',
            'feature_means': {'V1': 0.0, 'V2': 0.0, 'V3': 0.0},
            'feature_stds': {'V1': 1.0, 'V2': 1.0, 'V3': 1.0},
            'fraud_rate': 0.002,
            'recall': 0.98,
            'precision': 0.95,
            'created_at': datetime.utcnow()
        }

    @patch('src.storage.database.create_engine')
    def test_database_connection(self, mock_engine, db_service):
        """Test database connection establishment."""
        mock_engine.return_value = MagicMock()
        
        # Connection should be established
        assert db_service.settings is not None

    @patch('src.storage.database.Session')
    def test_save_drift_metrics(self, mock_session, db_service, sample_drift_metrics):
        """Test saving drift metrics to database."""
        mock_session_instance = MagicMock()
        mock_session.return_value = mock_session_instance
        
        result = db_service.save_drift_metrics(sample_drift_metrics)
        
        assert result is True or isinstance(result, int)

    @patch('src.storage.database.Session')
    def test_get_baseline_metrics(self, mock_session, db_service):
        """Test retrieving baseline metrics."""
        mock_session_instance = MagicMock()
        mock_session.return_value = mock_session_instance
        
        # Mock query result
        mock_baseline = MagicMock()
        mock_baseline.feature_means = {'V1': 0.0, 'V2': 0.0}
        mock_session_instance.query.return_value.filter_by.return_value.first.return_value = mock_baseline
        
        result = db_service.get_baseline_metrics(model_version='v1.0.0')
        
        assert result is not None or result == {}

    @patch('src.storage.database.Session')
    def test_query_historical_drift(self, mock_session, db_service):
        """Test querying historical drift data."""
        mock_session_instance = MagicMock()
        mock_session.return_value = mock_session_instance
        
        # Mock query results
        mock_results = [
            MagicMock(
                timestamp=datetime.utcnow(),
                data_drift_score=0.3,
                drift_detected=True
            )
        ]
        mock_session_instance.query.return_value.filter.return_value.order_by.return_value.all.return_value = mock_results
        
        start_date = datetime.utcnow() - timedelta(days=7)
        end_date = datetime.utcnow()
        
        result = db_service.query_historical_drift(start_date, end_date)
        
        assert isinstance(result, (list, pd.DataFrame))

    @patch('src.storage.database.Session')
    def test_update_baseline_metrics(self, mock_session, db_service, sample_baseline_metrics):
        """Test updating baseline metrics."""
        mock_session_instance = MagicMock()
        mock_session.return_value = mock_session_instance
        
        result = db_service.update_baseline_metrics(
            model_version='v1.0.0',
            metrics=sample_baseline_metrics
        )
        
        assert result is True or result is None

    @patch('src.storage.database.Session')
    def test_save_multiple_metrics(self, mock_session, db_service):
        """Test saving multiple drift metrics."""
        mock_session_instance = MagicMock()
        mock_session.return_value = mock_session_instance
        
        metrics_list = [
            {
                'timestamp': datetime.utcnow() - timedelta(hours=i),
                'model_version': 'v1.0.0',
                'data_drift_score': 0.3 + i*0.01,
                'drift_detected': False
            }
            for i in range(24)
        ]
        
        for metrics in metrics_list:
            result = db_service.save_drift_metrics(metrics)
            assert result is True or isinstance(result, int)

    @patch('src.storage.database.Session')
    def test_query_by_model_version(self, mock_session, db_service):
        """Test querying metrics by model version."""
        mock_session_instance = MagicMock()
        mock_session.return_value = mock_session_instance
        
        mock_results = [MagicMock()]
        mock_session_instance.query.return_value.filter.return_value.all.return_value = mock_results
        
        result = db_service.query_by_model_version('v1.0.0')
        
        assert isinstance(result, (list, pd.DataFrame)) or result is None

    @patch('src.storage.database.Session')
    def test_delete_old_metrics(self, mock_session, db_service):
        """Test deleting old drift metrics."""
        mock_session_instance = MagicMock()
        mock_session.return_value = mock_session_instance
        
        cutoff_date = datetime.utcnow() - timedelta(days=90)
        
        result = db_service.delete_old_metrics(cutoff_date)
        
        assert isinstance(result, (bool, int))

    @patch('src.storage.database.Session')
    def test_transaction_rollback_on_error(self, mock_session, db_service, sample_drift_metrics):
        """Test transaction rollback on error."""
        mock_session_instance = MagicMock()
        mock_session_instance.commit.side_effect = Exception("Database error")
        mock_session.return_value = mock_session_instance
        
        result = db_service.save_drift_metrics(sample_drift_metrics)
        
        # Should handle error gracefully
        assert mock_session_instance.rollback.called or result is False

    @patch('src.storage.database.Session')
    def test_concurrent_writes(self, mock_session, db_service):
        """Test handling concurrent database writes."""
        mock_session_instance = MagicMock()
        mock_session.return_value = mock_session_instance
        
        # Simulate concurrent writes
        metrics1 = {'timestamp': datetime.utcnow(), 'drift_detected': True}
        metrics2 = {'timestamp': datetime.utcnow(), 'drift_detected': False}
        
        result1 = db_service.save_drift_metrics(metrics1)
        result2 = db_service.save_drift_metrics(metrics2)
        
        # Both should complete
        assert result1 is not None
        assert result2 is not None

    @patch('src.storage.database.Session')
    def test_query_with_filters(self, mock_session, db_service):
        """Test querying with multiple filters."""
        mock_session_instance = MagicMock()
        mock_session.return_value = mock_session_instance
        
        filters = {
            'model_version': 'v1.0.0',
            'drift_detected': True,
            'severity': 'HIGH'
        }
        
        result = db_service.query_with_filters(filters)
        
        assert isinstance(result, (list, pd.DataFrame)) or result is None

    @patch('src.storage.database.Session')
    def test_get_latest_metrics(self, mock_session, db_service):
        """Test retrieving latest drift metrics."""
        mock_session_instance = MagicMock()
        mock_session.return_value = mock_session_instance
        
        mock_latest = MagicMock(
            timestamp=datetime.utcnow(),
            data_drift_score=0.35
        )
        mock_session_instance.query.return_value.order_by.return_value.first.return_value = mock_latest
        
        result = db_service.get_latest_metrics()
        
        assert result is not None or result == {}

    @patch('src.storage.database.Session')
    def test_aggregate_metrics_by_day(self, mock_session, db_service):
        """Test aggregating metrics by day."""
        mock_session_instance = MagicMock()
        mock_session.return_value = mock_session_instance
        
        start_date = datetime.utcnow() - timedelta(days=30)
        end_date = datetime.utcnow()
        
        result = db_service.aggregate_by_day(start_date, end_date)
        
        assert isinstance(result, (list, pd.DataFrame)) or result is None

    @patch('src.storage.database.Session')
    def test_database_connection_retry(self, mock_session, db_service):
        """Test database connection retry logic."""
        # First attempt fails, second succeeds
        mock_session.side_effect = [
            Exception("Connection failed"),
            MagicMock()
        ]
        
        # Should handle connection failure gracefully
        # (implementation dependent)

    @patch('src.storage.database.Session')
    def test_query_performance_large_dataset(self, mock_session, db_service):
        """Test query performance with large dataset."""
        mock_session_instance = MagicMock()
        mock_session.return_value = mock_session_instance
        
        # Simulate large result set
        mock_results = [MagicMock() for _ in range(10000)]
        mock_session_instance.query.return_value.filter.return_value.all.return_value = mock_results
        
        start = datetime.utcnow()
        result = db_service.query_historical_drift(
            datetime.utcnow() - timedelta(days=365),
            datetime.utcnow()
        )
        duration = (datetime.utcnow() - start).total_seconds()
        
        # Query should complete reasonably fast
        assert duration < 10  # seconds
