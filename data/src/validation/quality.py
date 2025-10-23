"""
Data quality validation
Checks for nulls, duplicates, outliers, and statistical anomalies
"""

import logging
import numpy as np
import pandas as pd
from typing import Optional, Dict, List

from config.constants import (
    MAX_MISSING_PERCENTAGE,
    MAX_DUPLICATE_ROWS,
    OUTLIER_STD_THRESHOLD
)

logger = logging.getLogger(__name__)


class QualityValidator:
    """Validates overall data quality of transaction batches"""

    def __init__(self):
        self.quality_report = {}
        self.issues = []

    def check_missing_values(self, df: pd.DataFrame) -> Dict:
        """
        Check for missing values in dataframe
        
        Returns:
            Dict with missing value statistics
        """
        missing_stats = {}
        
        for col in df.columns:
            missing_count = df[col].isnull().sum()
            missing_pct = missing_count / len(df)
            
            missing_stats[col] = {
                "count": missing_count,
                "percentage": missing_pct,
                "is_valid": missing_pct <= MAX_MISSING_PERCENTAGE
            }
            
            if not missing_stats[col]["is_valid"]:
                issue = f"Column '{col}' has {missing_pct*100:.2f}% missing values (threshold: {MAX_MISSING_PERCENTAGE*100}%)"
                self.issues.append(issue)
                logger.warning(issue)

        return missing_stats

    def check_duplicates(self, df: pd.DataFrame, subset: Optional[List[str]] = None) -> Dict:
        """
        Check for duplicate rows
        
        Args:
            df: DataFrame to check
            subset: Columns to consider for duplicates (default: all)
        
        Returns:
            Dict with duplicate statistics
        """
        duplicates = df.duplicated(subset=subset, keep=False).sum()
        duplicate_pct = duplicates / len(df) if len(df) > 0 else 0
        
        is_valid = duplicate_pct <= MAX_DUPLICATE_ROWS
        
        if not is_valid:
            issue = f"Found {duplicates} duplicate rows ({duplicate_pct*100:.2f}%, threshold: {MAX_DUPLICATE_ROWS*100}%)"
            self.issues.append(issue)
            logger.warning(issue)

        return {
            "duplicate_count": duplicates,
            "duplicate_percentage": duplicate_pct,
            "is_valid": is_valid
        }

    def check_outliers(
        self,
        df: pd.DataFrame,
        numeric_columns: Optional[List[str]] = None,
        std_threshold: float = OUTLIER_STD_THRESHOLD
    ) -> Dict:
        """
        Detect outliers using statistical methods (z-score)
        
        Args:
            df: DataFrame to check
            numeric_columns: Columns to check for outliers
            std_threshold: Number of standard deviations for outlier detection
        
        Returns:
            Dict with outlier statistics
        """
        if numeric_columns is None:
            numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()

        outlier_stats = {}
        
        for col in numeric_columns:
            if col not in df.columns:
                continue

            # Calculate z-scores
            z_scores = np.abs((df[col] - df[col].mean()) / df[col].std())
            outliers = (z_scores > std_threshold).sum()
            outlier_pct = outliers / len(df) if len(df) > 0 else 0

            outlier_stats[col] = {
                "outlier_count": outliers,
                "outlier_percentage": outlier_pct,
                "mean": float(df[col].mean()),
                "std": float(df[col].std()),
                "min": float(df[col].min()),
                "max": float(df[col].max())
            }

        return outlier_stats

    def check_data_types(self, df: pd.DataFrame, expected_types: Dict[str, str]) -> Dict:
        """
        Validate data types match expectations
        
        Args:
            df: DataFrame to check
            expected_types: Dict of column_name -> expected_dtype
        
        Returns:
            Dict with type validation results
        """
        type_validation = {}
        
        for col, expected_type in expected_types.items():
            if col not in df.columns:
                type_validation[col] = {
                    "present": False,
                    "is_valid": False,
                    "message": f"Column '{col}' not found in dataframe"
                }
                continue

            actual_type = str(df[col].dtype)
            is_valid = expected_type in actual_type
            
            type_validation[col] = {
                "present": True,
                "expected": expected_type,
                "actual": actual_type,
                "is_valid": is_valid
            }
            
            if not is_valid:
                issue = f"Column '{col}' has type {actual_type}, expected {expected_type}"
                self.issues.append(issue)

        return type_validation

    def validate_batch(
        self,
        df: pd.DataFrame,
        expected_types: Optional[Dict[str, str]] = None
    ) -> Dict:
        """
        Perform comprehensive quality validation on a batch
        
        Args:
            df: DataFrame to validate
            expected_types: Optional dict of expected column types
        
        Returns:
            Comprehensive quality report
        """
        self.issues = []
        
        report = {
            "row_count": len(df),
            "column_count": len(df.columns),
            "columns": df.columns.tolist(),
            "missing_values": self.check_missing_values(df),
            "duplicates": self.check_duplicates(df),
            "outliers": self.check_outliers(df),
        }

        if expected_types:
            report["data_types"] = self.check_data_types(df, expected_types)

        report["issues"] = self.issues
        report["is_valid"] = len(self.issues) == 0

        self.quality_report = report
        return report
