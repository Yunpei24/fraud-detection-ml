"""
Unit Tests for Training Data Loader
Tests the data loading functionality from PostgreSQL
"""

from unittest.mock import MagicMock, Mock, patch

import numpy as np
import pandas as pd
import pytest
from src.data.loader import (check_data_quality, load_training_data,
                             validate_data_schema)


class TestLoadTrainingData:
    """Test suite for load_training_data function"""

    @patch("sqlalchemy.create_engine")
    def test_load_training_data_success(self, mock_create_engine):
        """Test successful data loading from PostgreSQL"""
        mock_engine = MagicMock()
        mock_create_engine.return_value = mock_engine

        # Create mock data with all expected columns
        expected_data = pd.DataFrame(
            {"time": [1, 2, 3], "amount": [100.0, 250.0, 50.0], "class": [0, 1, 0]}
        )
        # Add v1-v28 columns
        for i in range(1, 29):
            expected_data[f"v{i}"] = [1.0, 2.0, 3.0]

        mock_conn = MagicMock()
        mock_create_engine.return_value.connect.return_value.__enter__.return_value = (
            mock_conn
        )

        # Mock pd.read_sql to return our data
        with patch("pandas.read_sql", return_value=expected_data):
            result = load_training_data()

        assert isinstance(result, pd.DataFrame)
        assert len(result) == 3
        assert "class" in result.columns
        assert "amount" in result.columns
        assert "v1" in result.columns
        assert "v28" in result.columns

    @patch("sqlalchemy.create_engine")
    def test_load_training_data_database_error(self, mock_create_engine):
        """Test error handling when database connection fails"""
        mock_create_engine.side_effect = Exception("Database unavailable")

        with pytest.raises(Exception):
            load_training_data()


class TestValidateDataSchema:
    """Test suite for validate_data_schema function"""

    def test_validate_schema_valid_data(self):
        """Test schema validation with valid data"""
        valid_df = pd.DataFrame(
            {"time": [1, 2], "amount": [100.0, 200.0], "class": [0, 1]}
        )
        # Add v1-v28 columns
        for i in range(1, 29):
            valid_df[f"v{i}"] = [1.0, 2.0]

        result = validate_data_schema(valid_df)
        assert result is True

    def test_validate_schema_missing_columns(self):
        """Test schema validation with missing required columns"""
        invalid_df = pd.DataFrame(
            {
                "time": [1, 2],
                "amount": [100.0, 200.0],
                # Missing: v1, class
            }
        )

        result = validate_data_schema(invalid_df)
        assert result is False


class TestCheckDataQuality:
    """Test suite for check_data_quality function"""

    def test_check_quality_clean_data(self):
        """Test data quality check on clean data"""
        clean_df = pd.DataFrame(
            {
                "time": [1, 2, 3],
                "v1": [1.0, 2.0, 3.0],
                "amount": [100.0, 200.0, 150.0],
                "class": [0, 1, 0],
            }
        )

        quality_report = check_data_quality(clean_df)

        assert quality_report["total_rows"] == 3
        assert quality_report["missing_values"] == 0
        assert quality_report["duplicate_rows"] == 0
        assert quality_report["status"] == "PASSED"

    def test_check_quality_missing_values(self):
        """Test data quality check with missing values"""
        dirty_df = pd.DataFrame(
            {
                "time": [1, 2, 3],
                "v1": [1.0, None, 3.0],
                "amount": [100.0, None, 150.0],
                "class": [0, 1, None],
            }
        )

        quality_report = check_data_quality(dirty_df)

        assert quality_report["missing_values"] > 0
        assert quality_report["total_rows"] == 3
