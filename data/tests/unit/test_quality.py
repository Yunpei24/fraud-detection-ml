"""
Unit tests for data quality validation
"""

import pytest
import pandas as pd
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from src.validation.quality import QualityValidator


class TestQualityValidator:
    """Tests for QualityValidator"""

    def test_check_missing_values_none(self, sample_dataframe):
        """Test with dataframe having no missing values"""
        validator = QualityValidator()
        stats = validator.check_missing_values(sample_dataframe)
        
        for col, col_stats in stats.items():
            assert col_stats["is_valid"] == True
            assert col_stats["percentage"] == 0.0

    def test_check_missing_values_threshold_exceeded(self, dataframe_with_nulls):
        """Test with missing values exceeding threshold"""
        validator = QualityValidator()
        stats = validator.check_missing_values(dataframe_with_nulls)
        
        # Column with 50% nulls should exceed threshold of 5%
        assert stats["transaction_id"]["percentage"] > 0.0
        assert len(validator.issues) > 0

    def test_check_duplicates_none(self, sample_dataframe):
        """Test with dataframe having no duplicates"""
        validator = QualityValidator()
        result = validator.check_duplicates(sample_dataframe)
        
        assert result["duplicate_percentage"] == 0.0
        assert result["is_valid"] == True

    def test_check_duplicates_found(self, dataframe_with_duplicates):
        """Test detection of duplicate rows"""
        validator = QualityValidator()
        result = validator.check_duplicates(dataframe_with_duplicates)
        
        assert result["duplicate_count"] > 0
        assert result["duplicate_percentage"] > 0.0

    def test_check_outliers(self, dataframe_with_outliers):
        """Test outlier detection"""
        validator = QualityValidator()
        # Use lower threshold to detect the 1000 value as outlier
        outliers = validator.check_outliers(dataframe_with_outliers, numeric_columns=["amount"], std_threshold=1.5)
        
        assert "amount" in outliers
        # With threshold of 1.5, the 1000 value should be detected as outlier
        assert outliers["amount"]["outlier_count"] > 0

    def test_validate_batch(self, sample_dataframe):
        """Test complete batch validation"""
        validator = QualityValidator()
        expected_types = {
            "transaction_id": "object",
            "amount": "float64",
            "is_fraud": "int64"
        }
        
        report = validator.validate_batch(sample_dataframe, expected_types)
        
        assert report["is_valid"] is True
        assert report["row_count"] == 5
        assert "missing_values" in report
        assert "duplicates" in report
        assert "outliers" in report
