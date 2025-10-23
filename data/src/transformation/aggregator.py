"""
Data aggregation for batch processing and statistical analysis
"""

import logging
import pandas as pd
from typing import Optional, List, Dict
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


class TransactionAggregator:
    """
    Aggregates transaction data for analytics and reporting
    Supports temporal, customer, merchant, and custom aggregations
    """

    def __init__(self):
        self.aggregation_report = {}

    def aggregate_by_time(
        self,
        df: pd.DataFrame,
        datetime_col: str = "transaction_time",
        period: str = "H",  # H=hourly, D=daily, W=weekly, M=monthly
        agg_functions: Optional[Dict] = None
    ) -> pd.DataFrame:
        """
        Aggregate transactions by time period
        
        Args:
            df: Input dataframe
            datetime_col: Datetime column name
            period: Resampling period (H/D/W/M)
            agg_functions: Dict of {column: function} for aggregation
        
        Returns:
            Time-aggregated dataframe
        """
        df_agg = df.copy()
        df_agg[datetime_col] = pd.to_datetime(df_agg[datetime_col])
        df_agg = df_agg.set_index(datetime_col)

        if agg_functions is None:
            agg_functions = {
                'amount': ['sum', 'mean', 'count', 'std', 'min', 'max']
            }

        df_aggregated = df_agg.resample(period).agg(agg_functions)

        logger.info(f"Aggregated {len(df)} transactions to {len(df_aggregated)} {period} periods")
        return df_aggregated

    def aggregate_by_customer(
        self,
        df: pd.DataFrame,
        customer_col: str = "customer_id",
        amount_col: str = "amount",
        agg_functions: Optional[Dict] = None
    ) -> pd.DataFrame:
        """
        Aggregate transactions by customer
        
        Args:
            df: Input dataframe
            customer_col: Customer ID column
            amount_col: Amount column
            agg_functions: Custom aggregation functions
        
        Returns:
            Customer-level aggregated dataframe
        """
        if agg_functions is None:
            agg_functions = {
                amount_col: ['sum', 'mean', 'count', 'std', 'min', 'max'],
                'transaction_id': 'count'
            }

        df_aggregated = df.groupby(customer_col).agg(agg_functions).reset_index()
        df_aggregated.columns = ['_'.join(col).strip('_') for col in df_aggregated.columns.values]

        logger.info(f"Aggregated to {len(df_aggregated)} unique customers")
        return df_aggregated

    def aggregate_by_merchant(
        self,
        df: pd.DataFrame,
        merchant_col: str = "merchant_id",
        amount_col: str = "amount",
        agg_functions: Optional[Dict] = None
    ) -> pd.DataFrame:
        """
        Aggregate transactions by merchant
        
        Args:
            df: Input dataframe
            merchant_col: Merchant ID column
            amount_col: Amount column
            agg_functions: Custom aggregation functions
        
        Returns:
            Merchant-level aggregated dataframe
        """
        if agg_functions is None:
            agg_functions = {
                amount_col: ['sum', 'mean', 'count', 'std', 'min', 'max'],
                'transaction_id': 'count'
            }

        df_aggregated = df.groupby(merchant_col).agg(agg_functions).reset_index()
        df_aggregated.columns = ['_'.join(col).strip('_') for col in df_aggregated.columns.values]

        logger.info(f"Aggregated to {len(df_aggregated)} unique merchants")
        return df_aggregated

    def aggregate_by_country(
        self,
        df: pd.DataFrame,
        country_col: str = "merchant_country",
        amount_col: str = "amount"
    ) -> pd.DataFrame:
        """
        Aggregate transactions by country
        
        Args:
            df: Input dataframe
            country_col: Country column name
            amount_col: Amount column
        
        Returns:
            Country-level aggregated dataframe
        """
        df_aggregated = df.groupby(country_col).agg({
            amount_col: ['sum', 'mean', 'count', 'std'],
            'transaction_id': 'count'
        }).reset_index()

        df_aggregated.columns = ['_'.join(col).strip('_') for col in df_aggregated.columns.values]

        logger.info(f"Aggregated to {len(df_aggregated)} unique countries")
        return df_aggregated

    def aggregate_fraud_statistics(
        self,
        df: pd.DataFrame,
        fraud_col: str = "is_fraud",
        amount_col: str = "amount",
        groupby_col: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Calculate fraud statistics
        
        Args:
            df: Input dataframe
            fraud_col: Fraud label column
            amount_col: Amount column
            groupby_col: Optional column to group by
        
        Returns:
            Fraud statistics dataframe
        """
        if groupby_col:
            df_agg = df.groupby(groupby_col).agg({
                fraud_col: ['sum', 'mean', 'count'],
                amount_col: 'sum'
            }).reset_index()
        else:
            df_agg = pd.DataFrame({
                'fraud_count': [df[fraud_col].sum()],
                'fraud_rate': [df[fraud_col].mean()],
                'total_transactions': [len(df)],
                'total_amount': [df[amount_col].sum()]
            })

        logger.info(f"Calculated fraud statistics - fraud rate: {df[fraud_col].mean():.2%}")
        return df_agg

    def rolling_aggregation(
        self,
        df: pd.DataFrame,
        datetime_col: str = "transaction_time",
        customer_col: str = "customer_id",
        amount_col: str = "amount",
        window_hours: int = 24
    ) -> pd.DataFrame:
        """
        Calculate rolling window aggregations for customer behavior
        
        Args:
            df: Input dataframe
            datetime_col: Datetime column
            customer_col: Customer ID column
            amount_col: Amount column
            window_hours: Rolling window in hours
        
        Returns:
            Dataframe with rolling features
        """
        df_rolling = df.copy()
        df_rolling[datetime_col] = pd.to_datetime(df_rolling[datetime_col])

        # Sort by customer and time
        df_rolling = df_rolling.sort_values([customer_col, datetime_col])

        # Rolling aggregations per customer
        df_rolling['rolling_transaction_count'] = df_rolling.groupby(customer_col).rolling(
            f'{window_hours}H',
            on=datetime_col
        )['transaction_id'].count().reset_index(drop=True)

        df_rolling['rolling_amount_sum'] = df_rolling.groupby(customer_col).rolling(
            f'{window_hours}H',
            on=datetime_col
        )[amount_col].sum().reset_index(drop=True)

        df_rolling['rolling_amount_mean'] = df_rolling.groupby(customer_col).rolling(
            f'{window_hours}H',
            on=datetime_col
        )[amount_col].mean().reset_index(drop=True)

        logger.info(f"Calculated rolling aggregations with {window_hours}h window")
        return df_rolling

    def generate_aggregation_report(self, df: pd.DataFrame) -> Dict:
        """
        Generate comprehensive aggregation report
        
        Args:
            df: Input dataframe
        
        Returns:
            Aggregation report dictionary
        """
        report = {
            "total_rows": len(df),
            "total_columns": len(df.columns),
            "unique_customers": df['customer_id'].nunique() if 'customer_id' in df.columns else None,
            "unique_merchants": df['merchant_id'].nunique() if 'merchant_id' in df.columns else None,
            "date_range": {
                "start": str(df['transaction_time'].min()) if 'transaction_time' in df.columns else None,
                "end": str(df['transaction_time'].max()) if 'transaction_time' in df.columns else None
            }
        }

        self.aggregation_report = report
        logger.info(f"Aggregation report: {report}")
        return report
