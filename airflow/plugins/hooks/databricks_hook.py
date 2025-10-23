"""
Databricks Hook for distributed training jobs
Uses centralized configuration from airflow.src.config.settings
"""
from typing import Any, Dict, Optional
from airflow.hooks.base import BaseHook
import requests
import time
import sys
from pathlib import Path

# Add airflow/src to path
AIRFLOW_SRC = Path(__file__).parent.parent.parent / "src"
sys.path.insert(0, str(AIRFLOW_SRC))

from config.settings import settings


class DatabricksHook(BaseHook):
    """
    Hook for interacting with Databricks Workspace API.
    Supports job submission, cluster management, and run monitoring.
    """
    
    def __init__(
        self,
        databricks_host: Optional[str] = None,
        databricks_token: Optional[str] = None
    ):
        """
        Initialize Databricks hook.
        
        Args:
            databricks_host: Databricks workspace URL (defaults to settings)
            databricks_token: Databricks API token (defaults to settings)
        """
        super().__init__()
        self.host = databricks_host or settings.databricks_host
        self.token = databricks_token or settings.databricks_token
        
        if not self.host or not self.token:
            raise ValueError(
                "Databricks host and token must be provided either as arguments "
                "or via environment variables (DATABRICKS_HOST, DATABRICKS_TOKEN)"
            )
        
        # Remove trailing slash from host
        self.host = self.host.rstrip('/')
        
        self.api_version = "2.1"
        self.base_url = f"{self.host}/api/{self.api_version}"
        
        self.headers = {
            "Authorization": f"Bearer {self.token}",
            "Content-Type": "application/json"
        }
    
    def _make_request(
        self,
        method: str,
        endpoint: str,
        json_data: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """
        Make HTTP request to Databricks API.
        
        Args:
            method: HTTP method (GET, POST, etc.)
            endpoint: API endpoint (e.g., 'jobs/runs/submit')
            json_data: Request payload
        
        Returns:
            Response JSON
        """
        url = f"{self.base_url}/{endpoint}"
        
        self.log.debug(f"Making {method} request to {url}")
        
        response = requests.request(
            method=method,
            url=url,
            headers=self.headers,
            json=json_data,
            timeout=30
        )
        
        response.raise_for_status()
        return response.json()
    
    # ==================== CLUSTER OPERATIONS ====================
    
    def get_cluster(self, cluster_id: str) -> Dict[str, Any]:
        """
        Get cluster information.
        
        Args:
            cluster_id: Databricks cluster ID
        
        Returns:
            Cluster details
        """
        self.log.info(f"Getting cluster {cluster_id}")
        return self._make_request("GET", f"clusters/get?cluster_id={cluster_id}")
    
    def start_cluster(self, cluster_id: str) -> None:
        """
        Start a cluster.
        
        Args:
            cluster_id: Databricks cluster ID
        """
        self.log.info(f"Starting cluster {cluster_id}")
        self._make_request("POST", "clusters/start", {"cluster_id": cluster_id})
    
    def wait_for_cluster_ready(
        self,
        cluster_id: str,
        timeout_seconds: int = 600,
        check_interval: int = 30
    ) -> bool:
        """
        Wait for cluster to be in RUNNING state.
        
        Args:
            cluster_id: Databricks cluster ID
            timeout_seconds: Maximum wait time
            check_interval: Check interval in seconds
        
        Returns:
            True if cluster is running, False if timeout
        """
        self.log.info(f"Waiting for cluster {cluster_id} to be ready...")
        
        start_time = time.time()
        
        while time.time() - start_time < timeout_seconds:
            cluster_info = self.get_cluster(cluster_id)
            state = cluster_info.get('state', 'UNKNOWN')
            
            self.log.info(f"Cluster state: {state}")
            
            if state == 'RUNNING':
                self.log.info(f"✓ Cluster {cluster_id} is running")
                return True
            elif state in ['TERMINATED', 'ERROR', 'TERMINATING']:
                self.log.error(f"Cluster {cluster_id} in bad state: {state}")
                raise ValueError(f"Cluster in state {state}")
            
            time.sleep(check_interval)
        
        self.log.error(f"Timeout waiting for cluster {cluster_id}")
        return False
    
    # ==================== JOB OPERATIONS ====================
    
    def submit_run(
        self,
        cluster_id: str,
        notebook_path: str,
        notebook_params: Optional[Dict[str, str]] = None,
        libraries: Optional[list] = None,
        timeout_seconds: int = 3600
    ) -> Dict[str, Any]:
        """
        Submit a notebook run on existing cluster.
        
        Args:
            cluster_id: Databricks cluster ID
            notebook_path: Path to notebook in workspace
            notebook_params: Parameters to pass to notebook
            libraries: Libraries to install
            timeout_seconds: Notebook execution timeout
        
        Returns:
            Run information with run_id
        """
        self.log.info(f"Submitting run on cluster {cluster_id}")
        self.log.info(f"Notebook: {notebook_path}")
        
        run_config = {
            "existing_cluster_id": cluster_id,
            "notebook_task": {
                "notebook_path": notebook_path,
                "base_parameters": notebook_params or {}
            },
            "timeout_seconds": timeout_seconds
        }
        
        if libraries:
            run_config["libraries"] = libraries
        
        response = self._make_request("POST", "jobs/runs/submit", run_config)
        
        run_id = response.get('run_id')
        self.log.info(f"✓ Run submitted with ID: {run_id}")
        
        return response
    
    def get_run(self, run_id: int) -> Dict[str, Any]:
        """
        Get run information.
        
        Args:
            run_id: Databricks run ID
        
        Returns:
            Run details
        """
        return self._make_request("GET", f"jobs/runs/get?run_id={run_id}")
    
    def wait_for_run_completion(
        self,
        run_id: int,
        timeout_seconds: int = 3600,
        check_interval: int = 30
    ) -> Dict[str, Any]:
        """
        Wait for run to complete.
        
        Args:
            run_id: Databricks run ID
            timeout_seconds: Maximum wait time
            check_interval: Check interval in seconds
        
        Returns:
            Final run state
        """
        self.log.info(f"Waiting for run {run_id} to complete...")
        
        start_time = time.time()
        
        while time.time() - start_time < timeout_seconds:
            run_info = self.get_run(run_id)
            state = run_info.get('state', {})
            life_cycle_state = state.get('life_cycle_state')
            result_state = state.get('result_state')
            
            self.log.info(f"Run state: {life_cycle_state} - {result_state}")
            
            # Terminal states
            if life_cycle_state in ['TERMINATED', 'SKIPPED', 'INTERNAL_ERROR']:
                
                if result_state == 'SUCCESS':
                    self.log.info(f"✓ Run {run_id} completed successfully")
                    return run_info
                else:
                    error_msg = state.get('state_message', 'Unknown error')
                    self.log.error(f"Run {run_id} failed: {error_msg}")
                    raise ValueError(f"Run failed with state {result_state}: {error_msg}")
            
            time.sleep(check_interval)
        
        self.log.error(f"Timeout waiting for run {run_id}")
        raise TimeoutError(f"Run {run_id} did not complete within {timeout_seconds}s")
    
    def cancel_run(self, run_id: int) -> None:
        """
        Cancel a running job.
        
        Args:
            run_id: Databricks run ID
        """
        self.log.warning(f"Cancelling run {run_id}")
        self._make_request("POST", "jobs/runs/cancel", {"run_id": run_id})
    
    def get_run_output(self, run_id: int) -> Dict[str, Any]:
        """
        Get notebook run output.
        
        Args:
            run_id: Databricks run ID
        
        Returns:
            Run output including logs and result
        """
        return self._make_request("GET", f"jobs/runs/get-output?run_id={run_id}")
    
    # ==================== HELPER METHODS ====================
    
    def submit_training_job(
        self,
        notebook_path: str,
        model_params: Dict[str, Any],
        cluster_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Submit a training job with common configuration.
        
        Args:
            notebook_path: Path to training notebook
            model_params: Model hyperparameters
            cluster_id: Cluster ID (defaults to settings)
        
        Returns:
            Run information
        """
        cluster_id = cluster_id or settings.databricks_cluster_id
        
        if not cluster_id:
            raise ValueError("Databricks cluster ID not configured")
        
        # Ensure cluster is running
        self.start_cluster(cluster_id)
        self.wait_for_cluster_ready(cluster_id)
        
        # Submit run
        run_info = self.submit_run(
            cluster_id=cluster_id,
            notebook_path=notebook_path,
            notebook_params={
                'model_params': str(model_params),
                'mlflow_tracking_uri': settings.mlflow_tracking_uri,
                'mlflow_experiment_name': settings.mlflow_experiment_name
            },
            libraries=[
                {"pypi": {"package": "mlflow==2.10.2"}},
                {"pypi": {"package": "xgboost==2.0.3"}},
                {"pypi": {"package": "scikit-learn==1.3.2"}}
            ]
        )
        
        return run_info
    
    def run_and_wait(
        self,
        notebook_path: str,
        notebook_params: Optional[Dict[str, str]] = None,
        cluster_id: Optional[str] = None,
        timeout: int = 3600
    ) -> Dict[str, Any]:
        """
        Submit run and wait for completion (blocking).
        
        Args:
            notebook_path: Path to notebook
            notebook_params: Notebook parameters
            cluster_id: Cluster ID (defaults to settings)
            timeout: Maximum execution time
        
        Returns:
            Final run state
        """
        cluster_id = cluster_id or settings.databricks_cluster_id
        
        # Submit
        run_info = self.submit_run(
            cluster_id=cluster_id,
            notebook_path=notebook_path,
            notebook_params=notebook_params,
            timeout_seconds=timeout
        )
        
        run_id = run_info['run_id']
        
        # Wait
        final_state = self.wait_for_run_completion(run_id, timeout_seconds=timeout)
        
        # Get output
        output = self.get_run_output(run_id)
        
        return {
            'run_id': run_id,
            'state': final_state,
            'output': output
        }


# Export
__all__ = ['DatabricksHook']
