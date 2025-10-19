"""
Feature engineering - abstract base and utilities for feature creation.

This module provides abstract base class for feature engineering and concrete
implementations for different data sources (Kaggle, Production, etc.).
"""

import logging
import pandas as pd
import numpy as np
from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from typing import Optional, List, Dict

logger = logging.getLogger(__name__)


class BaseFeatureEngineer(ABC):
    """
    Abstract base class for feature engineering.
    
    All concrete feature engineers (Kaggle, Production, etc.) inherit from this
    and implement their own feature creation logic based on available fields.
    """
    
    def __init__(self, engineer_name: str = "BaseFeatureEngineer"):
        """
        Initialize base feature engineer.
        
        Args:
            engineer_name (str): Name of this engineer
        """
        self.engineer_name = engineer_name
        self.features_created = []
    
    @abstractmethod
    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create all available features from the data.
        
        Args:
            df (pd.DataFrame): Input data
            
        Returns:
            pd.DataFrame: Data with engineered features
        """
        pass
    
    def get_created_features(self) -> List[str]:
        """Get list of features created."""
        return self.features_created
    
    def get_info(self) -> Dict:
        """Get engineer information."""
        return {
            "engineer_name": self.engineer_name,
            "features_created": self.features_created,
            "num_features": len(self.features_created)
        }


class FeatureEngineer(BaseFeatureEngineer):
    """
    Legacy feature engineer for production schema data.
    
    DEPRECATED: Use source-specific engineers (KaggleFeatureEngineer, etc.)
    This class is kept for backward compatibility but assumes production schema
    with customer_id, merchant_id, transaction_time, amount fields.
    
    For new code, use KaggleFeatureEngineer from src.transformation.kaggle_features
    """

    def __init__(self):
        super().__init__(engineer_name="LegacyFeatureEngineer")

    def create_temporal_features(self, df: pd.DataFrame, datetime_col: str = "transaction_time") -> pd.DataFrame:
        """
        Extract temporal features from datetime column
        
        Args:
            df: Input dataframe
            datetime_col: Name of datetime column
        
        Returns:
            DataFrame with new temporal features
        """
        df_features = df.copy()

        if datetime_col not in df_features.columns:
            logger.warning(f"Column '{datetime_col}' not found")
            return df_features

        # Convert to datetime if needed
        if df_features[datetime_col].dtype != 'datetime64[ns]':
            df_features[datetime_col] = pd.to_datetime(df_features[datetime_col])

        # Extract temporal features
        df_features['hour'] = df_features[datetime_col].dt.hour
        df_features['day_of_week'] = df_features[datetime_col].dt.dayofweek
        df_features['day_of_month'] = df_features[datetime_col].dt.day
        df_features['month'] = df_features[datetime_col].dt.month
        df_features['quarter'] = df_features[datetime_col].dt.quarter
        df_features['is_weekend'] = df_features[datetime_col].dt.dayofweek.isin([5, 6]).astype(int)
        df_features['is_business_hours'] = df_features['hour'].isin(range(9, 18)).astype(int)

        self.features_created.extend([
            'hour', 'day_of_week', 'day_of_month', 'month',
            'quarter', 'is_weekend', 'is_business_hours'
        ])

        logger.info("Created 7 temporal features")
        return df_features

    def create_amount_features(
        self,
        df: pd.DataFrame,
        amount_col: str = "amount"
    ) -> pd.DataFrame:
        """
        Create features based on transaction amount
        
        Args:
            df: Input dataframe
            amount_col: Name of amount column
        
        Returns:
            DataFrame with amount-based features
        """
        df_features = df.copy()

        if amount_col not in df_features.columns:
            logger.warning(f"Column '{amount_col}' not found")
            return df_features

        # Amount-based features
        df_features['amount_log'] = np.log1p(df_features[amount_col])
        df_features['amount_squared'] = df_features[amount_col] ** 2
        df_features['amount_bucket'] = pd.cut(
            df_features[amount_col],
            bins=[0, 25, 100, 500, 2000, float('inf')],
            labels=['tiny', 'small', 'medium', 'large', 'huge']
        )

        self.features_created.extend(['amount_log', 'amount_squared', 'amount_bucket'])

        logger.info("Created 3 amount-based features")
        return df_features

    def create_customer_features(
        self,
        df: pd.DataFrame,
        customer_id_col: str = "customer_id",
        amount_col: str = "amount",
        days_window: int = 30
    ) -> pd.DataFrame:
        """
        Create customer-level aggregation features
        
        Args:
            df: Input dataframe
            customer_id_col: Customer ID column name
            amount_col: Transaction amount column
            days_window: Lookback window in days
        
        Returns:
            DataFrame with customer features
        """
        df_features = df.copy()

        if customer_id_col not in df_features.columns:
            logger.warning(f"Column '{customer_id_col}' not found")
            return df_features

        # Customer transaction count
        df_features['customer_transaction_count'] = df_features.groupby(customer_id_col).cumcount()
        
        # Customer amount statistics
        customer_stats = df_features.groupby(customer_id_col)[amount_col].agg([
            'count', 'mean', 'std', 'min', 'max', 'sum'
        ]).rename(columns={
            'count': 'customer_total_transactions',
            'mean': 'customer_avg_amount',
            'std': 'customer_std_amount',
            'min': 'customer_min_amount',
            'max': 'customer_max_amount',
            'sum': 'customer_total_amount'
        })

        df_features = df_features.join(customer_stats, on=customer_id_col)
        
        # Fill any NaN std with 0
        df_features['customer_std_amount'] = df_features['customer_std_amount'].fillna(0)

        self.features_created.extend([
            'customer_transaction_count', 'customer_total_transactions',
            'customer_avg_amount', 'customer_std_amount',
            'customer_min_amount', 'customer_max_amount', 'customer_total_amount'
        ])

        logger.info("Created 7 customer aggregation features")
        return df_features

    def create_merchant_features(
        self,
        df: pd.DataFrame,
        merchant_id_col: str = "merchant_id",
        amount_col: str = "amount"
    ) -> pd.DataFrame:
        """
        Create merchant-level aggregation features
        
        Args:
            df: Input dataframe
            merchant_id_col: Merchant ID column name
            amount_col: Transaction amount column
        
        Returns:
            DataFrame with merchant features
        """
        df_features = df.copy()

        if merchant_id_col not in df_features.columns:
            logger.warning(f"Column '{merchant_id_col}' not found")
            return df_features

        # Merchant transaction count
        df_features['merchant_transaction_count'] = df_features.groupby(merchant_id_col).cumcount()
        
        # Merchant amount statistics
        merchant_stats = df_features.groupby(merchant_id_col)[amount_col].agg([
            'count', 'mean', 'std', 'min', 'max'
        ]).rename(columns={
            'count': 'merchant_total_transactions',
            'mean': 'merchant_avg_amount',
            'std': 'merchant_std_amount',
            'min': 'merchant_min_amount',
            'max': 'merchant_max_amount'
        })

        df_features = df_features.join(merchant_stats, on=merchant_id_col)
        df_features['merchant_std_amount'] = df_features['merchant_std_amount'].fillna(0)

        self.features_created.extend([
            'merchant_transaction_count', 'merchant_total_transactions',
            'merchant_avg_amount', 'merchant_std_amount',
            'merchant_min_amount', 'merchant_max_amount'
        ])

        logger.info("Created 6 merchant aggregation features")
        return df_features

    def create_interaction_features(
        self,
        df: pd.DataFrame,
        customer_col: str = "customer_id",
        merchant_col: str = "merchant_id",
        amount_col: str = "amount"
    ) -> pd.DataFrame:
        """
        Create interaction features between customer and merchant
        
        Args:
            df: Input dataframe
            customer_col: Customer ID column
            merchant_col: Merchant ID column
            amount_col: Amount column
        
        Returns:
            DataFrame with interaction features
        """
        df_features = df.copy()

        if customer_col not in df_features.columns or merchant_col not in df_features.columns:
            logger.warning("Required columns for interaction features not found")
            return df_features

        # Interaction count (how many times customer used merchant)
        df_features['customer_merchant_interaction_count'] = df_features.groupby(
            [customer_col, merchant_col]
        ).cumcount()

        # Interaction amount statistics
        interaction_stats = df_features.groupby([customer_col, merchant_col])[amount_col].agg([
            'count', 'mean', 'std', 'max'
        ]).rename(columns={
            'count': 'customer_merchant_total_transactions',
            'mean': 'customer_merchant_avg_amount',
            'std': 'customer_merchant_std_amount',
            'max': 'customer_merchant_max_amount'
        })

        # Join interaction stats
        for col in interaction_stats.columns:
            df_features[col] = df_features.apply(
                lambda row: interaction_stats.loc[(row[customer_col], row[merchant_col]), col]
                if (row[customer_col], row[merchant_col]) in interaction_stats.index else 0,
                axis=1
            )

        df_features['customer_merchant_std_amount'] = df_features['customer_merchant_std_amount'].fillna(0)

        self.features_created.extend([
            'customer_merchant_interaction_count',
            'customer_merchant_total_transactions',
            'customer_merchant_avg_amount',
            'customer_merchant_std_amount',
            'customer_merchant_max_amount'
        ])

        logger.info("Created 5 interaction features")
        return df_features

    def engineer_features(
        self,
        df: pd.DataFrame,
        create_temporal: bool = True,
        create_amount: bool = True,
        create_customer: bool = True,
        create_merchant: bool = True,
        create_interaction: bool = True
    ) -> pd.DataFrame:
        """
        Execute full feature engineering pipeline
        
        Args:
            df: Input dataframe
            create_temporal: Create temporal features
            create_amount: Create amount features
            create_customer: Create customer aggregations
            create_merchant: Create merchant aggregations
            create_interaction: Create customer-merchant interactions
        
        Returns:
            DataFrame with engineered features
        """
        df_features = df.copy()

        if create_temporal:
            df_features = self.create_temporal_features(df_features)

        if create_amount:
            df_features = self.create_amount_features(df_features)

        if create_customer:
            df_features = self.create_customer_features(df_features)

        if create_merchant:
            df_features = self.create_merchant_features(df_features)

        if create_interaction:
            df_features = self.create_interaction_features(df_features)

        logger.info(f"Total features created: {len(self.features_created)}")
        return df_features
