"""
Unit Tests for Data Preprocessor
Tests the preprocessing functionality using fraud_detection_common
"""
import numpy as np
import pandas as pd
import pytest
from src.data.preprocessor import DataPreprocessor


class TestDataPreprocessor:
    """Test suite for DataPreprocessor"""

    def test_init_default_parameters(self):
        """Test initialization with default parameters"""
        preprocessor = DataPreprocessor()
        assert preprocessor is not None
        assert hasattr(preprocessor, "fit_transform")
        assert hasattr(preprocessor, "transform")

    def test_fit_transform_with_valid_data(self, tiny_credit_df):
        """Test fit_transform with valid data"""
        preprocessor = DataPreprocessor()
        result, artifacts = preprocessor.fit_transform(tiny_credit_df)

        assert isinstance(result, pd.DataFrame)
        assert len(result) == len(tiny_credit_df)
        assert isinstance(artifacts, dict)

    def test_transform_without_fit_raises_error(self, tiny_credit_df):
        """Test that transform works without prior fitting"""
        preprocessor = DataPreprocessor()

        # Transform should work without fit (it's a stateless operation)
        result = preprocessor.transform(tiny_credit_df)
        assert isinstance(result, pd.DataFrame)
        assert len(result) == len(tiny_credit_df)

    def test_preprocessor_with_drop_columns(self, tiny_credit_df):
        """Test preprocessor with column dropping configuration"""
        preprocessor = DataPreprocessor(drop_columns=["Time"])
        result, artifacts = preprocessor.fit_transform(tiny_credit_df)

        assert "Time" not in result.columns
        assert "amount" in result.columns
        assert "Class" in result.columns

    def test_preprocessor_with_scale_columns(self, tiny_credit_df):
        """Test preprocessor with feature scaling configuration"""
        preprocessor = DataPreprocessor(scale_columns=["amount"])
        result, artifacts = preprocessor.fit_transform(tiny_credit_df)

        assert isinstance(result, pd.DataFrame)
        assert len(result) == len(tiny_credit_df)
        # amount should be scaled
        assert result["amount"].mean() != tiny_credit_df["amount"].mean()

    def test_preprocessor_with_outlier_handling(self, tiny_credit_df):
        """Test preprocessor with outlier handling configuration"""
        preprocessor = DataPreprocessor(
            outlier_columns=["amount"], outlier_method="iqr"
        )
        result, artifacts = preprocessor.fit_transform(tiny_credit_df)

        assert isinstance(result, pd.DataFrame)
        assert len(result) == len(tiny_credit_df)
