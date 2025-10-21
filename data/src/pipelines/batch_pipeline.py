"""
Batch data pipeline for periodic Databricks-based feature engineering.

Orchestrates daily batch processing:
1. Extract transactions from SQL Database
2. Submit Spark job to Databricks for feature engineering
3. Handle SMOTE resampling and feature transformations
4. Store results to S3 and Feature Store
5. Track execution metrics

Designed for integration with Apache Airflow DAGs.
"""

import logging
import os
from typing import Optional, Dict, Any
from datetime import datetime, timedelta

from src.cloud.databricks import DatabricksJobManager, DatabricksNotebookExecutor
from src.utils.logger import get_logger

logger = get_logger(__name__)


class DatabricksBatchPipeline:
    """
    Orchestrates daily batch data preparation via Databricks Spark cluster.
    
    Flow:
    1. Load transactions from Azure SQL Database (raw 24h data)
    2. Submit feature engineering job to Databricks cluster
    3. Databricks executes PySpark transformations:
       - Data validation and quality checks
       - SMOTE resampling (balance classes)
       - Feature engineering (50+ features)
       - Output to S3 + Feature Store
    4. Poll job status with exponential backoff
    5. Log metrics and handle errors gracefully
    """
    
    def __init__(self, databricks_host: str, databricks_token: str, 
                 cluster_id: str, polling_interval: int = 5, 
                 max_job_wait_time: int = 3600):
        """
        Initialize Databricks batch pipeline.
        
        Args:
            databricks_host (str): Databricks workspace URL (e.g., https://xxx.cloud.databricks.com)
            databricks_token (str): Databricks API token
            cluster_id (str): Existing Databricks cluster ID
            polling_interval (int): Seconds between status checks (default 5)
            max_job_wait_time (int): Max job execution time in seconds (default 3600)
        """
        self.job_manager = DatabricksJobManager(
            host=databricks_host,
            token=databricks_token,
            polling_interval=polling_interval,
            max_wait_seconds=max_job_wait_time
        )
        self.notebook_executor = DatabricksNotebookExecutor(
            host=databricks_host,
            token=databricks_token,
            polling_interval=polling_interval,
            max_wait_seconds=max_job_wait_time
        )
        self.cluster_id = cluster_id
        self.execution_metrics = {}
    
    def execute(self, **kwargs) -> Dict[str, Any]:
        """
        Execute complete batch pipeline.
        
        Args:
            **kwargs: Optional parameters passed to Databricks job
                - date_range_days: Number of days to process (default 1)
                - min_transactions: Minimum transactions to process (default 1000)
                - enable_smote: Enable SMOTE resampling (default True)
                - output_path: S3 output location
        
        Returns:
            Dict with execution status, rows processed, and metrics
        """
        job_name = f"fraud_detection_batch_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
        
        try:
            logger.info(f"Starting batch pipeline: {job_name}")
            
            # Submit feature engineering job to Databricks
            run_id = self._submit_feature_engineering_job(job_name, kwargs)
            
            # Wait for job completion with polling
            status = self.job_manager.wait_for_job(run_id)
            
            if status == 'SUCCEEDED':
                logger.info(f"Batch pipeline {job_name} completed successfully")
                return {
                    "status": "success",
                    "job_name": job_name,
                    "run_id": run_id,
                    "metrics": self.execution_metrics
                }
            else:
                logger.error(f"Batch pipeline {job_name} failed with status: {status}")
                return {
                    "status": "failed",
                    "job_name": job_name,
                    "run_id": run_id,
                    "error": f"Job failed with status: {status}"
                }
        
        except TimeoutError as e:
            logger.error(f"Batch pipeline timeout: {str(e)}")
            return {
                "status": "timeout",
                "job_name": job_name,
                "error": str(e)
            }
        except Exception as e:
            logger.error(f"Batch pipeline error: {str(e)}", exc_info=True)
            return {
                "status": "error",
                "job_name": job_name,
                "error": str(e)
            }
    
    def execute_via_notebook(self, notebook_path: str, **kwargs) -> Dict[str, Any]:
        """
        Execute batch pipeline via Databricks notebook.
        
        Alternative method if feature engineering is implemented as a notebook
        instead of a Spark job.
        
        Args:
            notebook_path (str): Path to notebook in Databricks workspace
                (e.g., /Workspace/fraud-detection/feature_engineering)
            **kwargs: Parameters passed to notebook
        
        Returns:
            Dict with execution status
        """
        job_name = f"notebook_feature_eng_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
        
        try:
            logger.info(f"Starting notebook execution: {job_name} from {notebook_path}")
            
            # Execute notebook on cluster
            run_id = self.notebook_executor.run_notebook(
                notebook_path=notebook_path,
                cluster_id=self.cluster_id,
                parameters=kwargs
            )
            
            # Wait for completion
            status = self.notebook_executor.wait_for_notebook(run_id)
            
            if status == 'SUCCEEDED':
                logger.info(f"Notebook {job_name} completed successfully")
                return {
                    "status": "success",
                    "notebook_path": notebook_path,
                    "run_id": run_id
                }
            else:
                logger.error(f"Notebook {job_name} failed with status: {status}")
                return {
                    "status": "failed",
                    "notebook_path": notebook_path,
                    "run_id": run_id,
                    "error": f"Notebook failed with status: {status}"
                }
        
        except TimeoutError as e:
            logger.error(f"Notebook execution timeout: {str(e)}")
            return {
                "status": "timeout",
                "notebook_path": notebook_path,
                "error": str(e)
            }
        except Exception as e:
            logger.error(f"Notebook execution error: {str(e)}", exc_info=True)
            return {
                "status": "error",
                "notebook_path": notebook_path,
                "error": str(e)
            }
    
    def _submit_feature_engineering_job(self, job_name: str, params: Dict[str, Any]) -> int:
        """
        Submit feature engineering Spark job to Databricks.
        
        Args:
            job_name (str): Name for this job run
            params (Dict): Job parameters (date_range_days, enable_smote, etc.)
        
        Returns:
            int: Job run ID
        """
        # Build Spark SQL job parameters
        spark_job_params = {
            'sql_file': 'dbfs:/fraud-detection/feature_engineering.sql',
            'parameters': {
                'date_range_days': str(params.get('date_range_days', 1)),
                'min_transactions': str(params.get('min_transactions', 1000)),
                'enable_smote': str(params.get('enable_smote', True)),
                'output_path': params.get('output_path', 's3://fraud-detection/features/')
            }
        }
        
        logger.debug(f"Submitting job with params: {spark_job_params}")
        
        # Submit Spark SQL job
        run_id = self.job_manager.submit_spark_sql_job(
            job_name=job_name,
            cluster_id=self.cluster_id,
            sql_file=spark_job_params['sql_file'],
            parameters=spark_job_params['parameters']
        )
        
        logger.info(f"Feature engineering job submitted: run_id={run_id}")
        return run_id
    
    def get_job_status(self, run_id: int) -> str:
        """
        Get current job status.
        
        Args:
            run_id (int): Job run ID
        
        Returns:
            str: Current status (PENDING, RUNNING, SUCCEEDED, FAILED, etc.)
        """
        return self.job_manager.get_job_status(run_id)
    
    def cancel_job(self, run_id: int) -> None:
        """
        Cancel a running job.
        
        Args:
            run_id (int): Job run ID
        """
        self.job_manager.cancel_job(run_id)
        logger.info(f"Job {run_id} cancelled")


def get_batch_pipeline() -> DatabricksBatchPipeline:
    """
    Factory function to create DatabricksBatchPipeline from environment variables.
    
    Expected environment variables:
    - DATABRICKS_HOST: Databricks workspace URL
    - DATABRICKS_TOKEN: Databricks API token
    - DATABRICKS_CLUSTER_ID: Existing cluster ID
    - DATABRICKS_POLLING_INTERVAL: Status poll interval (default 5s)
    - DATABRICKS_MAX_JOB_WAIT: Max job execution time (default 3600s)
    
    Returns:
        DatabricksBatchPipeline: Configured pipeline instance
    
    Raises:
        ValueError: If required environment variables are missing
    """
    databricks_host = os.getenv('DATABRICKS_HOST')
    databricks_token = os.getenv('DATABRICKS_TOKEN')
    cluster_id = os.getenv('DATABRICKS_CLUSTER_ID')
    
    if not all([databricks_host, databricks_token, cluster_id]):
        raise ValueError(
            "Missing required Databricks environment variables: "
            "DATABRICKS_HOST, DATABRICKS_TOKEN, DATABRICKS_CLUSTER_ID"
        )
    
    polling_interval = int(os.getenv('DATABRICKS_POLLING_INTERVAL', 5))
    max_job_wait = int(os.getenv('DATABRICKS_MAX_JOB_WAIT', 3600))
    
    return DatabricksBatchPipeline(
        databricks_host=databricks_host,
        databricks_token=databricks_token,
        cluster_id=cluster_id,
        polling_interval=polling_interval,
        max_job_wait_time=max_job_wait
    )
