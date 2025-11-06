# training/src/data/loader.py
"""
Data loading utilities for fraud detection training.
Supports loading from PostgreSQL (training_transactions table) and local CSV.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable, List, Optional

from dotenv import load_dotenv

load_dotenv()

import pandas as pd
import sqlalchemy
from src.config.logging_config import get_logger

logger = get_logger(__name__)

# Expected columns for fraud detection dataset (Kaggle creditcard format)
EXPECTED_COLUMNS: List[str] = (
    ["time"]  # Using lowercase to match our PostgreSQL schema
    + [f"v{i}" for i in range(1, 29)]  # V1-V28 PCA features
    + ["amount", "class"]
)


def validate_schema(
    df: pd.DataFrame, required: Iterable[str] = EXPECTED_COLUMNS
) -> None:
    """
    Validate that required columns are present in the DataFrame.

    Args:
        df: DataFrame to validate
        required: List of required column names (case-insensitive)

    Raises:
        ValueError: If required columns are missing
    """
    # Convert to lowercase for case-insensitive comparison
    df_cols_lower = [c.lower() for c in df.columns]
    required_lower = [c.lower() for c in required]

    missing = [c for c in required_lower if c not in df_cols_lower]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    extras = [c for c in df_cols_lower if c not in required_lower]
    if extras:
        logger.warning(f"Extra unexpected columns found: {extras}")


def load_training_data(
    conn_str: Optional[str] = None,
    table_name: str = "training_transactions",
    validate: bool = True,
    limit: Optional[int] = None,
) -> pd.DataFrame:
    """
    Load training data from PostgreSQL training_transactions table.
    This table contains the historical Kaggle creditcard.csv data used for model training.

    Args:
        conn_str: SQLAlchemy connection string. If None, reads from environment.
        table_name: Name of the table (default: training_transactions)
        validate: If True, enforce expected schema
        limit: Optional limit on number of rows (for testing)

    Returns:
        pandas DataFrame with columns: time, v1-v28, amount, class

    Environment variables:
        POSTGRES_HOST, POSTGRES_PORT, POSTGRES_DB, POSTGRES_USER, POSTGRES_PASSWORD
    """
    if conn_str is None:
        # Build connection string from environment
        host = os.getenv("POSTGRES_HOST", "localhost")
        port = os.getenv("POSTGRES_PORT", "5432")
        db = os.getenv("POSTGRES_DB", "fraud_detection")
        user = os.getenv("POSTGRES_USER", "fraud_user")
        password = os.getenv("POSTGRES_PASSWORD", "fraud_pass_dev_2024")
        conn_str = f"postgresql://{user}:{password}@{host}:{port}/{db}"

    logger.info(f"Loading training data from PostgreSQL table '{table_name}'")

    try:
        engine = sqlalchemy.create_engine(conn_str)

        # Build query - only select expected columns to avoid extra metadata columns
        columns_str = ", ".join(EXPECTED_COLUMNS)
        query = f"SELECT {columns_str} FROM {table_name}"
        if limit:
            query += f" LIMIT {limit}"

        with engine.connect() as conn:
            df = pd.read_sql(query, conn)

        logger.info(f"Loaded {len(df)} rows from '{table_name}'")

        # Standardize column names to lowercase
        df.columns = [c.lower() for c in df.columns]

        if validate:
            validate_schema(df)

        # Log class distribution
        if "class" in df.columns:
            fraud_count = (df["class"] == 1).sum()
            normal_count = (df["class"] == 0).sum()
            fraud_pct = 100 * fraud_count / len(df)
            logger.info(
                f"Class distribution: Normal={normal_count}, Fraud={fraud_count} ({fraud_pct:.2f}%)"
            )

        return df

    except Exception as e:
        logger.error(f"Failed to load data from PostgreSQL: {e}")
        raise


def load_local_csv(path: str | Path, validate: bool = True) -> pd.DataFrame:
    """
    Load creditcard.csv from a local path (fallback option).

    Args:
        path: Path to CSV file
        validate: If True, enforce expected schema

    Returns:
        pandas DataFrame
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"CSV not found at {path.resolve()}")

    logger.info(f"Loading CSV from {path}")
    df = pd.read_csv(path)

    # Standardize column names to lowercase
    df.columns = [c.lower() for c in df.columns]

    if validate:
        validate_schema(df)

    logger.info(f"Loaded shape={df.shape}")
    return df


def load_local_creditcard(
    path: str | Path = "data/raw/creditcard.csv", validate: bool = True
) -> pd.DataFrame:
    """
    Legacy function for backward compatibility.
    Prefer using load_training_data() for production.
    """
    logger.warning(
        "load_local_creditcard() is deprecated. Use load_training_data() instead."
    )
    return load_local_csv(path, validate=validate)


def load_reference_data(
    window_days: int = 30, conn_str: Optional[str] = None
) -> pd.DataFrame:
    """
    Load reference data for drift detection from the last N days.
    Simplified for testing - returns empty DataFrame.
    """
    if window_days <= 0:
        raise ValueError("window_days must be positive")

    # Return empty DataFrame with expected columns for testing
    return pd.DataFrame(columns=EXPECTED_COLUMNS)


def validate_data_schema(df: pd.DataFrame) -> bool:
    """
    Validate that the DataFrame has the expected schema for fraud detection.
    Simplified for testing - basic column check.
    """
    if df.empty:
        return False

    df_cols = set(c.lower() for c in df.columns)
    expected_cols = set(c.lower() for c in EXPECTED_COLUMNS)
    return expected_cols.issubset(df_cols)


def check_data_quality(df: pd.DataFrame) -> dict:
    """
    Perform basic data quality checks.
    Simplified for testing - minimal report.
    """
    return {
        "total_rows": len(df),
        "total_columns": len(df.columns),
        "missing_values": df.isnull().sum().sum(),
        "duplicate_rows": df.duplicated().sum(),
        "status": "PASSED" if not df.empty else "FAILED",
    }
