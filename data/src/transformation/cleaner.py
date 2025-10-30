"""
Data cleaning - handles missing values, duplicates, and outliers
"""

import logging
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class DataCleaner:
    """
    Cleans transaction data by handling missing values,
    duplicates, and inconsistencies
    """

    def __init__(self):
        self.cleaning_report = {}
        self.transformations_applied = []

    def remove_duplicates(
        self, df: pd.DataFrame, subset: Optional[List[str]] = None, keep: str = "first"
    ) -> pd.DataFrame:
        """
        Remove duplicate rows

        Args:
            df: Input dataframe
            subset: Columns to consider for duplicates
            keep: 'first', 'last', or False (remove all)

        Returns:
            DataFrame with duplicates removed
        """
        initial_rows = len(df)
        df_cleaned = df.drop_duplicates(subset=subset, keep=keep)
        removed_rows = initial_rows - len(df_cleaned)

        self.transformations_applied.append(
            {
                "operation": "remove_duplicates",
                "rows_removed": removed_rows,
                "rows_kept": len(df_cleaned),
            }
        )

        logger.info(f"Removed {removed_rows} duplicate rows")
        return df_cleaned

    def handle_missing_values(
        self,
        df: pd.DataFrame,
        numeric_strategy: str = "median",
        categorical_strategy: str = "mode",
        threshold_drop: float = 0.5,
    ) -> pd.DataFrame:
        """
        Handle missing values using various strategies

        Args:
            df: Input dataframe
            numeric_strategy: 'mean', 'median', 'forward_fill', 'drop'
            categorical_strategy: 'mode', 'forward_fill', 'drop'
            threshold_drop: If % missing > threshold, drop column

        Returns:
            DataFrame with missing values handled
        """
        df_cleaned = df.copy()
        imputation_report = {}

        # First pass: drop columns with too many missing values
        for col in df.columns:
            missing_pct = df[col].isnull().sum() / len(df)

            if missing_pct > threshold_drop:
                logger.info(f"Dropping column '{col}' - {missing_pct*100:.1f}% missing")
                df_cleaned = df_cleaned.drop(columns=[col])
                imputation_report[col] = {
                    "action": "dropped",
                    "missing_percentage": missing_pct,
                }
                continue

            if df[col].isnull().sum() == 0:
                imputation_report[col] = {"action": "no_missing", "missing_count": 0}
                continue

            # Numeric columns
            if df_cleaned[col].dtype in [np.float64, np.int64]:
                if numeric_strategy == "mean":
                    fill_value = df[col].mean()
                    df_cleaned[col] = df_cleaned[col].fillna(fill_value)
                elif numeric_strategy == "median":
                    fill_value = df[col].median()
                    df_cleaned[col] = df_cleaned[col].fillna(fill_value)
                elif numeric_strategy == "forward_fill":
                    df_cleaned[col] = (
                        df_cleaned[col].fillna(method="ffill").fillna(method="bfill")
                    )
                elif numeric_strategy == "drop":
                    df_cleaned = df_cleaned.dropna(subset=[col])

                imputation_report[col] = {
                    "action": numeric_strategy,
                    "original_missing": df[col].isnull().sum(),
                }

            # Categorical columns
            elif df_cleaned[col].dtype == "object":
                if categorical_strategy == "mode":
                    fill_value = (
                        df[col].mode()[0] if len(df[col].mode()) > 0 else "UNKNOWN"
                    )
                    df_cleaned[col] = df_cleaned[col].fillna(fill_value)
                elif categorical_strategy == "forward_fill":
                    df_cleaned[col] = (
                        df_cleaned[col].fillna(method="ffill").fillna(method="bfill")
                    )
                elif categorical_strategy == "drop":
                    df_cleaned = df_cleaned.dropna(subset=[col])

                imputation_report[col] = {
                    "action": categorical_strategy,
                    "original_missing": df[col].isnull().sum(),
                }

        self.transformations_applied.append(
            {
                "operation": "handle_missing_values",
                "imputation_report": imputation_report,
            }
        )

        logger.info(
            f"Handled missing values using {numeric_strategy}/{categorical_strategy}"
        )
        return df_cleaned

    def remove_outliers(
        self,
        df: pd.DataFrame,
        numeric_columns: Optional[List[str]] = None,
        method: str = "iqr",
        std_threshold: float = 3.0,
    ) -> pd.DataFrame:
        """
        Remove outliers from numeric columns

        Args:
            df: Input dataframe
            numeric_columns: Columns to check
            method: 'iqr' or 'zscore'
            std_threshold: For zscore method, how many stds

        Returns:
            DataFrame with outliers removed
        """
        df_cleaned = df.copy()

        if numeric_columns is None:
            numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()

        removed_rows = 0

        for col in numeric_columns:
            if col not in df_cleaned.columns:
                continue

            if method == "iqr":
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR

                mask = (df_cleaned[col] >= lower_bound) & (
                    df_cleaned[col] <= upper_bound
                )
                rows_removed = (~mask).sum()

            elif method == "zscore":
                z_scores = np.abs((df[col] - df[col].mean()) / df[col].std())
                mask = z_scores <= std_threshold
                rows_removed = (~mask).sum()

            df_cleaned = df_cleaned[mask]
            removed_rows += rows_removed
            logger.info(f"Removed {rows_removed} outliers from '{col}' using {method}")

        self.transformations_applied.append(
            {
                "operation": "remove_outliers",
                "method": method,
                "rows_removed": removed_rows,
            }
        )

        return df_cleaned

    def standardize_column_names(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Standardize column names (lowercase, remove special chars)

        Args:
            df: Input dataframe

        Returns:
            DataFrame with standardized column names
        """
        df_cleaned = df.copy()

        # Convert to lowercase and replace spaces/special chars with underscore
        df_cleaned.columns = (
            df_cleaned.columns.str.lower()
            .str.replace(" ", "_")
            .str.replace("[^a-z0-9_]", "", regex=True)
        )

        self.transformations_applied.append({"operation": "standardize_column_names"})

        logger.info("Standardized column names")
        return df_cleaned

    def clean_pipeline(
        self,
        df: pd.DataFrame,
        remove_dups: bool = True,
        handle_missing: bool = True,
        remove_outliers_flag: bool = False,
        standardize_names: bool = True,
    ) -> pd.DataFrame:
        """
        Execute full cleaning pipeline

        Args:
            df: Input dataframe
            remove_dups: Remove duplicates
            handle_missing: Handle missing values
            remove_outliers_flag: Remove outliers
            standardize_names: Standardize column names

        Returns:
            Cleaned dataframe
        """
        df_cleaned = df.copy()
        self.cleaning_report["initial_shape"] = df.shape

        if standardize_names:
            df_cleaned = self.standardize_column_names(df_cleaned)

        if remove_dups:
            df_cleaned = self.remove_duplicates(df_cleaned)

        if handle_missing:
            df_cleaned = self.handle_missing_values(df_cleaned)

        if remove_outliers_flag:
            df_cleaned = self.remove_outliers(df_cleaned)

        self.cleaning_report["final_shape"] = df_cleaned.shape
        self.cleaning_report["transformations"] = self.transformations_applied

        return df_cleaned
