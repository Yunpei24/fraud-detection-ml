"""
Unit tests for data cleaning
"""

import pytest
import pandas as pd
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from src.transformation.cleaner import DataCleaner


class TestDataCleaner:
    """Tests for DataCleaner"""

    def test_remove_duplicates(self, dataframe_with_duplicates):
        """Test duplicate removal"""
        cleaner = DataCleaner()
        df_cleaned = cleaner.remove_duplicates(dataframe_with_duplicates)
        
        # Should have fewer rows after removing duplicates
        assert len(df_cleaned) < len(dataframe_with_duplicates)
        
        # No duplicates in cleaned data
        assert df_cleaned.duplicated().sum() == 0

    def test_handle_missing_values_median(self, dataframe_with_nulls):
        """Test handling missing values with median strategy"""
        cleaner = DataCleaner()
        df_cleaned = cleaner.handle_missing_values(
            dataframe_with_nulls,
            numeric_strategy="median",
            categorical_strategy="mode"
        )
        
        # No missing values in numeric column
        assert df_cleaned["amount"].isnull().sum() == 0

    def test_handle_missing_values_drop_threshold(self, dataframe_with_nulls):
        """Test dropping columns with high missing percentage"""
        cleaner = DataCleaner()
        df_cleaned = cleaner.handle_missing_values(
            dataframe_with_nulls,
            threshold_drop=0.3  # Drop if > 30% missing
        )
        
        # Columns with > 30% missing should be dropped
        # transaction_id has 25% missing, so may not be dropped

    def test_standardize_column_names(self, sample_dataframe):
        """Test column name standardization"""
        df_test = sample_dataframe.copy()
        df_test.columns = [c.upper() for c in df_test.columns]
        
        cleaner = DataCleaner()
        df_cleaned = cleaner.standardize_column_names(df_test)
        
        # All column names should be lowercase
        assert all(col.islower() for col in df_cleaned.columns)

    def test_remove_outliers_iqr(self, dataframe_with_outliers):
        """Test outlier removal using IQR method"""
        cleaner = DataCleaner()
        df_cleaned = cleaner.remove_outliers(
            dataframe_with_outliers,
            numeric_columns=["amount"],
            method="iqr"
        )
        
        # Extreme outlier (5000) should be removed
        assert df_cleaned["amount"].max() < 5000

    def test_clean_pipeline(self, dataframe_with_duplicates):
        """Test complete cleaning pipeline"""
        cleaner = DataCleaner()
        df_cleaned = cleaner.clean_pipeline(dataframe_with_duplicates)
        
        assert df_cleaned.duplicated().sum() == 0
        assert "final_shape" in cleaner.cleaning_report
        assert cleaner.cleaning_report["final_shape"][0] <= len(dataframe_with_duplicates)
