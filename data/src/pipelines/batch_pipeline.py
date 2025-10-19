"""
Batch data pipeline for periodic processing of transaction datasets.

Provides abstract base and concrete implementations for different data sources.
Flow: Load -> Validate -> Clean -> Engineer -> Store
"""

import logging
import time
import pandas as pd
from typing import Optional, Dict, Any
from datetime import datetime
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


class BaseBatchPipeline(ABC):
    """
    Abstract base class for batch processing pipelines.
    
    All concrete pipelines inherit from this and define their specific
    loading, validation, and transformation logic.
    """
    
    def __init__(self, pipeline_name: str = "BaseBatchPipeline"):
        """
        Initialize base batch pipeline.
        
        Args:
            pipeline_name (str): Name of this pipeline
        """
        self.pipeline_name = pipeline_name
        self.execution_stats = {
            "start_time": None,
            "end_time": None,
            "total_rows_processed": 0,
            "total_rows_stored": 0,
            "errors": [],
            "status": None
        }
    
    @abstractmethod
    def execute(self, input_source, **kwargs) -> Dict[str, Any]:
        """Execute the complete batch pipeline."""
        pass
    
    def get_statistics(self) -> dict:
        """Get execution statistics."""
        stats = self.execution_stats.copy()
        
        if stats["start_time"] and stats["end_time"]:
            elapsed = (stats["end_time"] - stats["start_time"]).total_seconds()
            stats["duration_seconds"] = elapsed
            stats["rows_per_second"] = (
                stats["total_rows_stored"] / elapsed if elapsed > 0 else 0
            )
        
        return stats


class BatchPipeline(BaseBatchPipeline):
    """
    Legacy batch pipeline for daily/hourly processing of historical data.
    
    DEPRECATED: Use source-specific pipelines (KaggleBatchPipeline, etc.)
    This class is kept for backward compatibility.
    
    Flow: Load -> Validate -> Clean -> Engineer -> Store
    """

    def __init__(self):
        """Initialize batch pipeline"""
        super().__init__(pipeline_name="LegacyBatchPipeline")
        self.execution_stats = {
            "start_time": None,
            "end_time": None,
            "total_rows_processed": 0,
            "total_rows_stored": 0,
            "errors": []
        }

    def execute(
        self,
        input_source,
        validator,
        cleaner,
        feature_engineer,
        storage_service,
        metrics_collector=None
    ) -> dict:
        """
        Execute complete batch pipeline
        
        Args:
            input_source: Data source (DataFrame, file path, or SQL query)
            validator: Schema validator instance
            cleaner: Data cleaner instance
            feature_engineer: Feature engineer instance
            storage_service: Storage service instance
            metrics_collector: Optional metrics collector
        
        Returns:
            Execution report dictionary
        """
        self.execution_stats["start_time"] = datetime.utcnow()

        try:
            # 1. Load data
            logger.info("Step 1: Loading data...")
            df = self._load_data(input_source)
            self.execution_stats["total_rows_processed"] = len(df)
            logger.info(f"Loaded {len(df)} rows")

            # 2. Validate schema
            logger.info("Step 2: Validating schema...")
            df_validated = self._validate_data(df, validator, metrics_collector)
            logger.info(f"Validation complete - {len(df_validated)} rows valid")

            # 3. Clean data
            logger.info("Step 3: Cleaning data...")
            df_cleaned = cleaner.clean_pipeline(df_validated)
            logger.info(f"Cleaned data - {len(df_cleaned)} rows remaining")

            # 4. Engineer features
            logger.info("Step 4: Engineering features...")
            df_features = feature_engineer.engineer_features(df_cleaned)
            logger.info(f"Feature engineering complete - {len(df_features.columns)} total columns")

            # 5. Store results
            logger.info("Step 5: Storing results...")
            rows_stored = storage_service.insert_transactions(df_features.to_dict('records'))
            self.execution_stats["total_rows_stored"] = rows_stored
            logger.info(f"Stored {rows_stored} rows")

            # 6. Record metrics
            if metrics_collector:
                elapsed = (datetime.utcnow() - self.execution_stats["start_time"]).total_seconds()
                metrics_collector.record_transaction_processed(rows_stored)
                metrics_collector.record_processing_latency(elapsed)

            self.execution_stats["end_time"] = datetime.utcnow()
            self.execution_stats["status"] = "success"
            logger.info("Batch pipeline execution completed successfully")

        except Exception as e:
            logger.error(f"Batch pipeline execution failed: {str(e)}")
            self.execution_stats["status"] = "failed"
            self.execution_stats["errors"].append(str(e))
            self.execution_stats["end_time"] = datetime.utcnow()

        return self.execution_stats

    def _load_data(self, source) -> pd.DataFrame:
        """
        Load data from various sources
        
        Args:
            source: DataFrame, file path (.csv, .parquet), or SQL query dict
        
        Returns:
            Pandas DataFrame
        """
        if isinstance(source, pd.DataFrame):
            return source

        elif isinstance(source, str):
            if source.endswith('.csv'):
                return pd.read_csv(source)
            elif source.endswith('.parquet'):
                return pd.read_parquet(source)
            else:
                raise ValueError(f"Unsupported file format: {source}")

        elif isinstance(source, dict) and 'query' in source:
            # Load from database
            from sqlalchemy import create_engine, text
            engine = create_engine(source.get('connection_string'))
            with engine.connect() as conn:
                result = conn.execute(text(source['query']))
                columns = result.keys()
                data = [dict(zip(columns, row)) for row in result]
            return pd.DataFrame(data)

        else:
            raise ValueError(f"Unsupported source type: {type(source)}")

    def _validate_data(
        self,
        df: pd.DataFrame,
        validator,
        metrics_collector=None
    ) -> pd.DataFrame:
        """
        Validate batch data using the appropriate schema.
        
        NOTE: This method is for legacy compatibility with old tests.
        New code should use KaggleBatchPipeline or schema-aware validators.
        
        Args:
            df: Input DataFrame
            validator: Schema validator (new API with validate_batch)
            metrics_collector: Optional metrics collector
        
        Returns:
            DataFrame with validated rows
        """
        # Use new validate_batch API if available
        if hasattr(validator, 'validate_batch'):
            is_valid, report = validator.validate_batch(df)
            invalid_count = len(df) - report.get('valid_rows', 0)
            logger.info(f"Validation report: {invalid_count} invalid rows")
            self.execution_stats["errors"].append(f"Invalid rows: {invalid_count}")
            
            if metrics_collector and invalid_count > 0:
                metrics_collector.record_validation_error()
            
            return df  # Return all rows (validation already happened)
        
        # Fallback for very old validator interface
        else:
            logger.warning("Using legacy row-by-row validation. Consider upgrading to new schema API.")
            valid_rows = []
            invalid_count = 0

            for idx, row in df.iterrows():
                if hasattr(validator, 'validate'):
                    is_valid, report = validator.validate(row.to_dict())
                    if is_valid:
                        valid_rows.append(row)
                    else:
                        invalid_count += 1
                        if metrics_collector:
                            metrics_collector.record_validation_error()
                else:
                    # No validate method, assume row is valid
                    valid_rows.append(row)

            logger.info(f"Removed {invalid_count} invalid rows during validation")
            self.execution_stats["errors"].append(f"Invalid rows: {invalid_count}")
            
            return pd.DataFrame(valid_rows)

    def get_statistics(self) -> dict:
        """Get execution statistics"""
        stats = self.execution_stats.copy()
        
        if stats["start_time"] and stats["end_time"]:
            elapsed = (stats["end_time"] - stats["start_time"]).total_seconds()
            stats["duration_seconds"] = elapsed
            stats["rows_per_second"] = (
                stats["total_rows_stored"] / elapsed if elapsed > 0 else 0
            )

        return stats
