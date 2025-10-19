"""
Unit tests for feature engineering
"""

import pytest
import pandas as pd
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from src.transformation.features import FeatureEngineer


class TestFeatureEngineer:
    """Tests for FeatureEngineer"""

    def test_create_temporal_features(self, sample_dataframe):
        """Test temporal feature creation"""
        engineer = FeatureEngineer()
        df_features = engineer.create_temporal_features(sample_dataframe)
        
        # Check new columns exist
        assert "hour" in df_features.columns
        assert "day_of_week" in df_features.columns
        assert "is_weekend" in df_features.columns
        assert "is_business_hours" in df_features.columns

    def test_create_amount_features(self, sample_dataframe):
        """Test amount-based feature creation"""
        engineer = FeatureEngineer()
        df_features = engineer.create_amount_features(sample_dataframe)
        
        assert "amount_log" in df_features.columns
        assert "amount_squared" in df_features.columns
        assert "amount_bucket" in df_features.columns

    def test_create_customer_features(self, sample_dataframe):
        """Test customer aggregation features"""
        engineer = FeatureEngineer()
        df_features = engineer.create_customer_features(sample_dataframe)
        
        assert "customer_total_transactions" in df_features.columns
        assert "customer_avg_amount" in df_features.columns
        assert "customer_std_amount" in df_features.columns

    def test_create_merchant_features(self, sample_dataframe):
        """Test merchant aggregation features"""
        engineer = FeatureEngineer()
        df_features = engineer.create_merchant_features(sample_dataframe)
        
        assert "merchant_total_transactions" in df_features.columns
        assert "merchant_avg_amount" in df_features.columns

    def test_create_interaction_features(self, sample_dataframe):
        """Test customer-merchant interaction features"""
        engineer = FeatureEngineer()
        df_features = engineer.create_interaction_features(sample_dataframe)
        
        assert "customer_merchant_interaction_count" in df_features.columns
        assert "customer_merchant_total_transactions" in df_features.columns

    def test_engineer_features_full_pipeline(self, sample_dataframe):
        """Test complete feature engineering pipeline"""
        engineer = FeatureEngineer()
        df_features = engineer.engineer_features(sample_dataframe)
        
        # Should have more columns than input
        assert len(df_features.columns) > len(sample_dataframe.columns)
        
        # Check that original columns still exist
        for col in sample_dataframe.columns:
            assert col in df_features.columns
