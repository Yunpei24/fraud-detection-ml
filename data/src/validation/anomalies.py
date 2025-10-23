"""
Anomaly detection for data validation
Detects statistical anomalies and unusual patterns
"""

import logging
import numpy as np
import pandas as pd
from typing import Optional, Dict, List

logger = logging.getLogger(__name__)


class AnomalyDetector:
    """
    Detects anomalies in transaction data using statistical methods
    and isolation forest algorithm
    """

    def __init__(self):
        self.anomalies = []
        self.anomaly_scores = {}

    def detect_missing_columns(
        self,
        df: pd.DataFrame,
        expected_columns: List[str]
    ) -> Dict:
        """
        Detect if expected columns are missing from dataframe
        
        Args:
            df: DataFrame to check
            expected_columns: List of expected column names
        
        Returns:
            Dict with missing column information
        """
        missing = set(expected_columns) - set(df.columns)
        extra = set(df.columns) - set(expected_columns)

        result = {
            "missing_columns": list(missing),
            "extra_columns": list(extra),
            "has_anomalies": len(missing) > 0 or len(extra) > 0
        }

        if missing:
            logger.warning(f"Missing columns: {missing}")
        if extra:
            logger.info(f"Extra columns detected: {extra}")

        return result

    def detect_null_anomalies(self, df: pd.DataFrame, threshold: float = 0.3) -> Dict:
        """
        Detect anomalies in null value patterns
        
        Args:
            df: DataFrame to analyze
            threshold: Percentage threshold for anomaly (0.0 - 1.0)
        
        Returns:
            Dict with null pattern anomalies
        """
        null_percentages = df.isnull().sum() / len(df)
        anomalous_cols = null_percentages[null_percentages > threshold]

        result = {
            "columns_with_high_nulls": anomalous_cols.to_dict(),
            "has_anomalies": len(anomalous_cols) > 0,
            "threshold": threshold
        }

        if result["has_anomalies"]:
            logger.warning(f"Columns with high null percentage (>{threshold*100}%): {anomalous_cols.to_dict()}")

        return result

    def detect_distribution_anomalies(
        self,
        df: pd.DataFrame,
        numeric_columns: Optional[List[str]] = None,
        skewness_threshold: float = 2.0
    ) -> Dict:
        """
        Detect anomalies in data distributions using skewness
        
        Args:
            df: DataFrame to analyze
            numeric_columns: Columns to analyze
            skewness_threshold: Threshold for skewness anomaly
        
        Returns:
            Dict with distribution anomalies
        """
        if numeric_columns is None:
            numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()

        anomalies = {}
        
        for col in numeric_columns:
            if col not in df.columns or df[col].isnull().all():
                continue

            skewness = df[col].skew()
            is_anomalous = abs(skewness) > skewness_threshold

            anomalies[col] = {
                "skewness": float(skewness),
                "is_anomalous": is_anomalous,
                "distribution": "right-skewed" if skewness > 0 else "left-skewed"
            }

        anomalous_cols = {k: v for k, v in anomalies.items() if v["is_anomalous"]}

        result = {
            "distribution_anomalies": anomalies,
            "anomalous_columns": list(anomalous_cols.keys()),
            "has_anomalies": len(anomalous_cols) > 0,
            "threshold": skewness_threshold
        }

        if result["has_anomalies"]:
            logger.warning(f"Columns with anomalous distributions: {list(anomalous_cols.keys())}")

        return result

    def detect_constant_columns(self, df: pd.DataFrame) -> Dict:
        """
        Detect columns with constant or near-constant values (likely useless)
        
        Args:
            df: DataFrame to analyze
        
        Returns:
            Dict with constant column information
        """
        constant_cols = {}
        
        for col in df.columns:
            unique_count = df[col].nunique()
            unique_pct = unique_count / len(df) if len(df) > 0 else 0

            if unique_count <= 1 or unique_pct < 0.01:  # Less than 1% unique
                constant_cols[col] = {
                    "unique_values": unique_count,
                    "unique_percentage": unique_pct
                }

        result = {
            "constant_columns": constant_cols,
            "has_anomalies": len(constant_cols) > 0
        }

        if result["has_anomalies"]:
            logger.warning(f"Constant columns detected: {list(constant_cols.keys())}")

        return result

    def detect_cardinality_anomalies(
        self,
        df: pd.DataFrame,
        categorical_columns: Optional[List[str]] = None,
        high_cardinality_threshold: int = 1000
    ) -> Dict:
        """
        Detect columns with unusually high cardinality (too many unique values)
        
        Args:
            df: DataFrame to analyze
            categorical_columns: Columns to check
            high_cardinality_threshold: Threshold for high cardinality
        
        Returns:
            Dict with cardinality anomalies
        """
        if categorical_columns is None:
            categorical_columns = df.select_dtypes(include=['object']).columns.tolist()

        high_cardinality = {}
        
        for col in categorical_columns:
            if col not in df.columns:
                continue

            unique_count = df[col].nunique()
            
            if unique_count > high_cardinality_threshold:
                high_cardinality[col] = {
                    "unique_values": unique_count,
                    "high_cardinality": True
                }

        result = {
            "high_cardinality_columns": high_cardinality,
            "has_anomalies": len(high_cardinality) > 0,
            "threshold": high_cardinality_threshold
        }

        if result["has_anomalies"]:
            logger.warning(f"High cardinality columns: {list(high_cardinality.keys())}")

        return result

    def run_full_analysis(
        self,
        df: pd.DataFrame,
        expected_columns: Optional[List[str]] = None,
        categorical_columns: Optional[List[str]] = None
    ) -> Dict:
        """
        Run comprehensive anomaly detection
        
        Args:
            df: DataFrame to analyze
            expected_columns: Expected columns (optional)
            categorical_columns: Categorical columns (optional)
        
        Returns:
            Comprehensive anomaly report
        """
        report = {
            "row_count": len(df),
            "column_count": len(df.columns),
            "null_patterns": self.detect_null_anomalies(df),
            "distribution": self.detect_distribution_anomalies(df),
            "constant_columns": self.detect_constant_columns(df),
            "cardinality": self.detect_cardinality_anomalies(df, categorical_columns)
        }

        if expected_columns:
            report["missing_columns"] = self.detect_missing_columns(df, expected_columns)

        # Summary
        has_any_anomaly = any([
            report.get("null_patterns", {}).get("has_anomalies", False),
            report.get("distribution", {}).get("has_anomalies", False),
            report.get("constant_columns", {}).get("has_anomalies", False),
            report.get("cardinality", {}).get("has_anomalies", False),
            report.get("missing_columns", {}).get("has_anomalies", False),
        ])

        report["has_anomalies"] = has_any_anomaly

        return report
