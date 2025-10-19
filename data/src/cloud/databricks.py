"""
Databricks integration for distributed computing.

Handles Spark job submission, notebook execution, and data transfer.
"""

import os
from typing import Optional, List, Dict, Any
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.jobs import RunNow, TaskSettingsSQL

from src.utils.logger import get_logger

logger = get_logger(__name__)


class DatabricksJobManager:
    """Manages Databricks jobs and Spark applications."""
    
    def __init__(self, host: str, token: str):
        """
        Initialize Databricks client.
        
        Args:
            host (str): Databricks workspace host URL
            token (str): Databricks API token
        """
        self.host = host
        self.token = token
        self.client = WorkspaceClient(host=host, token=token)
    
    def submit_job(self, job_name: str, cluster_id: str, spark_params: Dict[str, Any]) -> int:
        """
        Submit a Spark job to Databricks.
        
        Args:
            job_name (str): Job name
            cluster_id (str): Cluster ID
            spark_params (Dict): Spark job parameters
            
        Returns:
            int: Job run ID
        """
        try:
            run = self.client.jobs.submit(
                run_name=job_name,
                tasks=[{
                    'task_key': job_name,
                    'existing_cluster_id': cluster_id,
                    'spark_python_task': {
                        'python_file': spark_params.get('python_file'),
                        'parameters': spark_params.get('parameters', [])
                    }
                }]
            )
            logger.info(f"Job {job_name} submitted with run ID: {run.run_id}")
            return run.run_id
        except Exception as e:
            logger.error(f"Failed to submit job: {str(e)}")
            raise
    
    def get_job_status(self, run_id: int) -> str:
        """
        Get the status of a Spark job.
        
        Args:
            run_id (int): Job run ID
            
        Returns:
            str: Job status (PENDING, RUNNING, SUCCEEDED, FAILED, etc.)
        """
        try:
            run = self.client.jobs.get_run(run_id)
            return run.state.value
        except Exception as e:
            logger.error(f"Failed to get job status: {str(e)}")
            raise
    
    def cancel_job(self, run_id: int) -> None:
        """
        Cancel a running job.
        
        Args:
            run_id (int): Job run ID
        """
        try:
            self.client.jobs.cancel_run(run_id)
            logger.info(f"Job {run_id} cancelled")
        except Exception as e:
            logger.error(f"Failed to cancel job: {str(e)}")
            raise


class DatabricksNotebookExecutor:
    """Executes Databricks notebooks."""
    
    def __init__(self, host: str, token: str):
        """
        Initialize Databricks notebook executor.
        
        Args:
            host (str): Databricks workspace host URL
            token (str): Databricks API token
        """
        self.host = host
        self.token = token
        self.client = WorkspaceClient(host=host, token=token)
    
    def run_notebook(self, notebook_path: str, cluster_id: str, 
                    parameters: Optional[Dict[str, str]] = None) -> int:
        """
        Run a Databricks notebook.
        
        Args:
            notebook_path (str): Path to notebook
            cluster_id (str): Cluster ID
            parameters (Dict): Notebook parameters
            
        Returns:
            int: Run ID
        """
        try:
            run = self.client.jobs.submit(
                run_name=f"notebook_run_{notebook_path.split('/')[-1]}",
                tasks=[{
                    'task_key': 'notebook_task',
                    'existing_cluster_id': cluster_id,
                    'notebook_task': {
                        'notebook_path': notebook_path,
                        'base_parameters': parameters or {}
                    }
                }]
            )
            logger.info(f"Notebook {notebook_path} executed with run ID: {run.run_id}")
            return run.run_id
        except Exception as e:
            logger.error(f"Failed to run notebook: {str(e)}")
            raise
    
    def get_notebook_output(self, run_id: int) -> Optional[str]:
        """
        Get output from notebook execution.
        
        Args:
            run_id (int): Run ID
            
        Returns:
            str: Notebook output
        """
        try:
            run = self.client.jobs.get_run(run_id)
            if hasattr(run, 'output_values'):
                return run.output_values
            return None
        except Exception as e:
            logger.error(f"Failed to get notebook output: {str(e)}")
            raise


class DatabricksDataTransfer:
    """Handles data transfer to/from Databricks."""
    
    def __init__(self, host: str, token: str):
        """
        Initialize data transfer client.
        
        Args:
            host (str): Databricks workspace host URL
            token (str): Databricks API token
        """
        self.host = host
        self.token = token
        self.client = WorkspaceClient(host=host, token=token)
    
    def upload_file(self, local_path: str, dbfs_path: str) -> None:
        """
        Upload file to Databricks file system.
        
        Args:
            local_path (str): Local file path
            dbfs_path (str): DBFS destination path
        """
        try:
            with open(local_path, 'rb') as f:
                data = f.read()
            
            self.client.dbfs.put(dbfs_path, overwrite=True, data=data)
            logger.info(f"File {local_path} uploaded to {dbfs_path}")
        except Exception as e:
            logger.error(f"Failed to upload file: {str(e)}")
            raise
    
    def download_file(self, dbfs_path: str, local_path: str) -> None:
        """
        Download file from Databricks file system.
        
        Args:
            dbfs_path (str): DBFS source path
            local_path (str): Local destination path
        """
        try:
            response = self.client.dbfs.get_status(dbfs_path)
            data = self.client.dbfs.get_file(dbfs_path)
            
            with open(local_path, 'wb') as f:
                f.write(data.data)
            
            logger.info(f"File {dbfs_path} downloaded to {local_path}")
        except Exception as e:
            logger.error(f"Failed to download file: {str(e)}")
            raise
