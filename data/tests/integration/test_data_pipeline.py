"""
Integration tests for data pipeline
"""

import pytest
import pandas as pd
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from src.pipelines.batch_pipeline import BatchPipeline
from src.validation.schema import SchemaValidator
from src.transformation.cleaner import DataCleaner
from src.transformation.features import FeatureEngineer


class MockStorage:
    """Mock storage service for testing"""
    def __init__(self):
        self.stored_data = []

    def insert_transactions(self, transactions):
        self.stored_data.extend(transactions)
        return len(transactions)


class TestBatchPipeline:
    """Tests for BatchPipeline integration"""

    def test_pipeline_with_dataframe(self, sample_dataframe):
        """Test batch pipeline with DataFrame input"""
        pipeline = BatchPipeline()
        validator = SchemaValidator()
        cleaner = DataCleaner()
        engineer = FeatureEngineer()
        storage = MockStorage()

        stats = pipeline.execute(
            sample_dataframe,
            validator,
            cleaner,
            engineer,
            storage
        )

        assert stats["status"] == "success"
        assert stats["total_rows_processed"] > 0
        assert stats["total_rows_stored"] > 0
        assert len(storage.stored_data) > 0

    def test_pipeline_statistics(self, sample_dataframe):
        """Test pipeline statistics calculation"""
        pipeline = BatchPipeline()
        validator = SchemaValidator()
        cleaner = DataCleaner()
        engineer = FeatureEngineer()
        storage = MockStorage()

        pipeline.execute(
            sample_dataframe,
            validator,
            cleaner,
            engineer,
            storage
        )

        stats = pipeline.get_statistics()
        
        assert "duration_seconds" in stats
        assert "rows_per_second" in stats
        assert stats["status"] == "success"

    def test_pipeline_with_invalid_data(self):
        """Test pipeline handles invalid data gracefully"""
        pipeline = BatchPipeline()
        validator = SchemaValidator()
        cleaner = DataCleaner()
        engineer = FeatureEngineer()
        storage = MockStorage()

        # Create invalid dataframe
        invalid_df = pd.DataFrame({
            "id": ["A", "B", "C"]
            # Missing required columns
        })

        stats = pipeline.execute(
            invalid_df,
            validator,
            cleaner,
            engineer,
            storage
        )

        # Should handle error gracefully
        assert "status" in stats


class TestDataPipelineIntegration:
    """End-to-end data pipeline tests"""

    def test_full_data_flow(self, sample_dataframe):
        """Test complete data flow"""
        # Validate using new batch API
        validator = SchemaValidator()
        is_valid, report = validator.validate_batch(sample_dataframe)
        
        df_valid = sample_dataframe
        assert len(df_valid) > 0

        # Clean
        cleaner = DataCleaner()
        df_cleaned = cleaner.clean_pipeline(df_valid)
        assert len(df_cleaned) > 0

        # Engineer features
        engineer = FeatureEngineer()
        df_features = engineer.engineer_features(df_cleaned)
        assert len(df_features.columns) > len(df_cleaned.columns)

        # Store
        storage = MockStorage()
        rows_stored = storage.insert_transactions(df_features.to_dict('records'))
        assert rows_stored == len(df_features)
