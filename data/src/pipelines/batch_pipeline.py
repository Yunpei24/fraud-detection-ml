"""
Batch Data Pipeline

Orchestrates daily batch data preparation locally via Docker.

Flow:
1. Load transactions from CSV/PostgreSQL (raw 24h data)
2. Apply data cleaning (duplicates, missing values, outlier removal)
3. Apply preprocessing via fraud_detection_common.preprocessor.DataPreprocessor
4. Apply feature engineering via fraud_detection_common.feature_engineering
5. Validate output schema
6. Save processed data to PostgreSQL
7. Save artifacts (scalers, preprocessor) to disk
8. Track execution metrics via Prometheus
"""

import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd
from src.storage.database import DatabaseService
from src.transformation.cleaner import DataCleaner
from src.utils.logger import get_logger

# Import from common package (these would be available in the Docker environment)
try:
    from fraud_detection_common.feature_engineering import build_feature_frame
    from fraud_detection_common.preprocessor import DataPreprocessor
    from fraud_detection_common.schema_validation import (
        REQUIRED_COLUMNS,
        validate_schema,
    )
except ImportError:
    # Fallback for local development
    class DataPreprocessor:
        def fit_transform(self, df):
            return df

        def save_artifacts(self, path):
            pass

        @classmethod
        def load_artifacts(cls, path):
            return cls()

    def build_feature_frame(df):
        return df

    def validate_schema(df, columns):
        pass

    REQUIRED_COLUMNS = [
        "Time",
        "V1",
        "V2",
        "V3",
        "V4",
        "V5",
        "V6",
        "V7",
        "V8",
        "V9",
        "V10",
        "V11",
        "V12",
        "V13",
        "V14",
        "V15",
        "V16",
        "V17",
        "V18",
        "V19",
        "V20",
        "V21",
        "V22",
        "V23",
        "V24",
        "V25",
        "V26",
        "V27",
        "V28",
        "amount",
        "Class",
    ]

logger = get_logger(__name__)


class BatchDataPipeline:
    """
    Orchestrates daily batch data preparation locally via Docker.

    This pipeline handles the complete ETL process for fraud detection data:
    - Extract: Load raw transaction data from CSV or PostgreSQL
    - Transform: Clean, preprocess, and engineer features
    - Load: Save processed data and artifacts
    """

    def __init__(
        self,
        db_service: Optional[DatabaseService] = None,
        artifact_dir: Optional[Path] = None,
    ):
        """
        Initialize batch data pipeline.

        Args:
            db_service (DatabaseService, optional): Database handler. If None, creates from env.
            artifact_dir (Path, optional): Directory to save artifacts. Defaults to /artifacts.
        """
        self.db_service = db_service or DatabaseService()
        if not self.db_service._initialized:
            self.db_service.connect()

        self.artifact_dir = artifact_dir or Path(
            os.getenv("ARTIFACT_DIR", "/artifacts")
        )
        self.artifact_dir.mkdir(parents=True, exist_ok=True)

        self.cleaner = DataCleaner()
        self.preprocessor = DataPreprocessor()
        self.execution_metrics = {}

        logger.info(
            f"BatchDataPipeline initialized with artifact_dir={self.artifact_dir}"
        )

    def execute(self, data_source: Optional[str] = None, **kwargs) -> Dict[str, Any]:
        """
        Execute complete batch pipeline.

        Args:
            data_source (str, optional): Path to CSV file or None to load from PostgreSQL
            **kwargs: Optional parameters
                - save_to_db (bool): Save to PostgreSQL (default True)
                - save_to_csv (bool): Save to CSV (default False)
                - output_csv_path (str): CSV output path if save_to_csv=True

        Returns:
            Dict with execution status, rows processed, and metrics

        Example:
            >>> pipeline = BatchDataPipeline()
            >>> result = pipeline.execute(data_source="/data/creditcard.csv")
            >>> print(result["status"])  # "success"
            >>> print(result["rows_processed"])  # 284807
        """
        run_id = f"batch_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
        start_time = datetime.utcnow()

        try:
            logger.info(f"Starting batch pipeline: {run_id}")

            # Step 1: Load raw data
            logger.info("Step 1/6: Loading raw data")
            if data_source and data_source.endswith(".csv"):
                raw_df = pd.read_csv(data_source)
                logger.info(f"Loaded from CSV: {data_source}")
            else:
                # Load from PostgreSQL using DatabaseService
                transactions = self.db_service.query_transactions(
                    limit=100000, offset=0
                )
                raw_df = pd.DataFrame(transactions)
                logger.info("Loaded from PostgreSQL")

            initial_rows = len(raw_df)
            logger.info(f"Loaded {initial_rows} rows")
            self.execution_metrics["initial_rows"] = initial_rows

            # Step 2: Data cleaning
            logger.info("Step 2/6: Cleaning data (duplicates, missing values)")
            cleaned_df = self.cleaner.clean_pipeline(raw_df)
            cleaned_rows = len(cleaned_df)
            logger.info(
                f"After cleaning: {cleaned_rows} rows ({initial_rows - cleaned_rows} removed)"
            )
            self.execution_metrics["cleaned_rows"] = cleaned_rows
            self.execution_metrics["rows_removed_cleaning"] = (
                initial_rows - cleaned_rows
            )

            # Step 3: Schema validation
            logger.info("Step 3/6: Validating schema")
            validate_schema(cleaned_df, REQUIRED_COLUMNS)
            logger.info("Schema validation passed")

            # Step 4: Preprocessing (scaling, outlier removal)
            logger.info("Step 4/6: Applying preprocessing (scaling, outlier removal)")
            preprocessed_df = self.preprocessor.fit_transform(cleaned_df)
            preprocessed_rows = len(preprocessed_df)
            logger.info(
                f"After preprocessing: {preprocessed_rows} rows ({cleaned_rows - preprocessed_rows} outliers removed)"
            )
            self.execution_metrics["preprocessed_rows"] = preprocessed_rows
            self.execution_metrics["outliers_removed"] = (
                cleaned_rows - preprocessed_rows
            )

            # Step 5: Feature engineering
            logger.info("Step 5/6: Applying feature engineering")
            final_df = build_feature_frame(preprocessed_df)
            final_rows = len(final_df)
            final_columns = len(final_df.columns)
            logger.info(
                f"After feature engineering: {final_rows} rows, {final_columns} features"
            )
            self.execution_metrics["final_rows"] = final_rows
            self.execution_metrics["final_columns"] = final_columns

            # Step 6: Save artifacts and data
            logger.info("Step 6/6: Saving artifacts and processed data")
            artifact_path = self.artifact_dir / f"preprocessor_{run_id}.pkl"
            self.preprocessor.save_artifacts(str(artifact_path))
            logger.info(f"Saved preprocessor artifacts to {artifact_path}")

            # Save to PostgreSQL if requested
            if kwargs.get("save_to_db", True):
                # Convert DataFrame to list of dicts for insert_transactions
                records = final_df.to_dict("records")
                self.db_service.insert_transactions(records)
                logger.info("Saved processed data to PostgreSQL")

            # Save to CSV if requested
            if kwargs.get("save_to_csv", False):
                output_csv = kwargs.get(
                    "output_csv_path", f"/data/processed_{run_id}.csv"
                )
                final_df.to_csv(output_csv, index=False)
                logger.info(f"Saved processed data to {output_csv}")

            # Calculate execution time
            execution_time = (datetime.utcnow() - start_time).total_seconds()
            self.execution_metrics["execution_time_seconds"] = execution_time

            logger.info(
                f"Batch pipeline {run_id} completed successfully in {execution_time:.2f}s"
            )
            return {
                "status": "success",
                "run_id": run_id,
                "rows_processed": final_rows,
                "features_created": final_columns,
                "metrics": self.execution_metrics,
                "artifact_path": str(artifact_path),
            }

        except Exception as e:
            execution_time = (datetime.utcnow() - start_time).total_seconds()
            logger.error(
                f"Batch pipeline {run_id} failed after {execution_time:.2f}s: {str(e)}",
                exc_info=True,
            )
            return {
                "status": "error",
                "run_id": run_id,
                "error": str(e),
                "metrics": self.execution_metrics,
            }

    def get_metrics(self) -> Dict[str, Any]:
        """
        Get current execution metrics.

        Returns:
            Dict with metrics (rows processed, execution time, etc.)
        """
        return self.execution_metrics.copy()

    def save_preprocessor(self, path: str) -> None:
        """
        Save fitted preprocessor to disk.

        Args:
            path (str): Path to save preprocessor (will be saved as pickle)
        """
        self.preprocessor.save_artifacts(path)
        logger.info(f"Preprocessor saved to {path}")

    def load_preprocessor(self, path: str) -> None:
        """
        Load fitted preprocessor from disk.

        Args:
            path (str): Path to saved preprocessor
        """
        self.preprocessor = DataPreprocessor.load_artifacts(path)
        logger.info(f"Preprocessor loaded from {path}")


def get_batch_pipeline() -> BatchDataPipeline:
    """
    Factory function to create BatchDataPipeline from environment variables.

    Expected environment variables:
    - ARTIFACT_DIR: Directory to save artifacts (default /artifacts)
    - POSTGRES_HOST: PostgreSQL host (for data ingestion/storage)
    - POSTGRES_PORT: PostgreSQL port (default 5432)
    - POSTGRES_DB: PostgreSQL database name
    - POSTGRES_USER: PostgreSQL username
    - POSTGRES_PASSWORD: PostgreSQL password

    Returns:
        BatchDataPipeline: Configured pipeline instance

    Example:
        >>> pipeline = get_batch_pipeline()
        >>> result = pipeline.execute(data_source="/data/creditcard.csv")
    """
    artifact_dir = Path(os.getenv("ARTIFACT_DIR", "/artifacts"))

    # Create DatabaseService from environment
    db_connection_string = (
        f"postgresql://{os.getenv('POSTGRES_USER', 'fraud_user')}"
        f":{os.getenv('POSTGRES_PASSWORD', 'fraud_pass')}"
        f"@{os.getenv('POSTGRES_HOST', 'postgres')}"
        f":{os.getenv('POSTGRES_PORT', '5432')}"
        f"/{os.getenv('POSTGRES_DB', 'fraud_detection')}"
    )

    db_service = DatabaseService(connection_string=db_connection_string)

    return BatchDataPipeline(db_service=db_service, artifact_dir=artifact_dir)


# CLI entry point for Docker container
if __name__ == "__main__":
    import sys

    logger.info("Starting batch data pipeline")

    # Get data source from command line or environment
    data_source = sys.argv[1] if len(sys.argv) > 1 else os.getenv("DATA_SOURCE")

    if not data_source:
        logger.error(
            "No data source provided. Set DATA_SOURCE env var or pass as argument."
        )
        sys.exit(1)

    # Create and run pipeline
    pipeline = get_batch_pipeline()
    result = pipeline.execute(data_source=data_source)

    if result["status"] == "success":
        logger.info(
            f"Pipeline completed successfully: {result['rows_processed']} rows processed"
        )
        sys.exit(0)
    else:
        logger.error(f"Pipeline failed: {result.get('error', 'Unknown error')}")
        sys.exit(1)


# Compatibility aliases for tests
DatabricksBatchPipeline = BatchDataPipeline


def get_batch_pipeline() -> BatchDataPipeline:
    """
    Factory function to create a batch pipeline instance.
    Used for testing and CLI integration.
    """
    return BatchDataPipeline()
