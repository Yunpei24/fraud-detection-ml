"""
Custom Databricks Operator for training jobs
Uses DatabricksHook and centralized settings
"""
from typing import Any, Dict, Optional
from airflow.models import BaseOperator
from airflow.utils.decorators import apply_defaults
import sys
from pathlib import Path

# Add airflow/src to path
AIRFLOW_SRC = Path(__file__).parent.parent.parent / "src"
sys.path.insert(0, str(AIRFLOW_SRC))

from config.settings import settings


class DatabricksSubmitRunOperator(BaseOperator):
    """
    Operator to submit and monitor Databricks notebook runs.
    Simplifies training job submission with sensible defaults.
    """
    
    template_fields = ['notebook_params']
    
    @apply_defaults
    def __init__(
        self,
        notebook_path: str,
        notebook_params: Optional[Dict[str, str]] = None,
        cluster_id: Optional[str] = None,
        databricks_host: Optional[str] = None,
        databricks_token: Optional[str] = None,
        libraries: Optional[list] = None,
        timeout_seconds: int = 3600,
        polling_interval: int = 30,
        *args,
        **kwargs
    ):
        """
        Initialize Databricks operator.
        
        Args:
            notebook_path: Path to notebook in Databricks workspace
            notebook_params: Parameters to pass to notebook
            cluster_id: Databricks cluster ID (defaults to settings)
            databricks_host: Databricks workspace URL (defaults to settings)
            databricks_token: Databricks API token (defaults to settings)
            libraries: Additional libraries to install
            timeout_seconds: Maximum execution time
            polling_interval: Check interval in seconds
        """
        super().__init__(*args, **kwargs)
        
        self.notebook_path = notebook_path
        self.notebook_params = notebook_params or {}
        self.cluster_id = cluster_id or settings.databricks_cluster_id
        self.databricks_host = databricks_host or settings.databricks_host
        self.databricks_token = databricks_token or settings.databricks_token
        self.libraries = libraries
        self.timeout_seconds = timeout_seconds
        self.polling_interval = polling_interval
        
        # Validate
        if not self.cluster_id:
            raise ValueError("cluster_id must be provided or set in settings")
        if not self.databricks_host:
            raise ValueError("databricks_host must be provided or set in settings")
        if not self.databricks_token:
            raise ValueError("databricks_token must be provided or set in settings")
    
    def execute(self, context: Any) -> Dict[str, Any]:
        """
        Execute Databricks notebook run.
        
        Returns:
            Run result with metrics
        """
        from plugins.hooks.databricks_hook import DatabricksHook
        
        # Initialize hook
        hook = DatabricksHook(
            databricks_host=self.databricks_host,
            databricks_token=self.databricks_token
        )
        
        self.log.info(f"Submitting Databricks run: {self.notebook_path}")
        self.log.info(f"Cluster: {self.cluster_id}")
        self.log.info(f"Parameters: {self.notebook_params}")
        
        # Ensure cluster is running
        self.log.info("Checking cluster state...")
        hook.start_cluster(self.cluster_id)
        hook.wait_for_cluster_ready(self.cluster_id)
        
        # Submit run
        run_info = hook.submit_run(
            cluster_id=self.cluster_id,
            notebook_path=self.notebook_path,
            notebook_params=self.notebook_params,
            libraries=self.libraries,
            timeout_seconds=self.timeout_seconds
        )
        
        run_id = run_info['run_id']
        self.log.info(f"Run submitted with ID: {run_id}")
        
        # Wait for completion
        try:
            final_state = hook.wait_for_run_completion(
                run_id=run_id,
                timeout_seconds=self.timeout_seconds,
                check_interval=self.polling_interval
            )
            
            # Get output
            output = hook.get_run_output(run_id)
            
            self.log.info(f"✓ Run {run_id} completed successfully")
            
            result = {
                'run_id': run_id,
                'state': final_state.get('state', {}),
                'output': output,
                'success': True
            }
            
            # Store run_id in XCom for downstream tasks
            context['task_instance'].xcom_push(key='databricks_run_id', value=run_id)
            
            return result
            
        except Exception as e:
            self.log.error(f"Run {run_id} failed: {e}")
            
            # Try to cancel run
            try:
                hook.cancel_run(run_id)
            except Exception:
                pass
            
            raise


class DatabricksTrainingOperator(DatabricksSubmitRunOperator):
    """
    Specialized operator for ML training jobs.
    Includes MLflow integration and common libraries.
    """
    
    @apply_defaults
    def __init__(
        self,
        notebook_path: str,
        model_params: Dict[str, Any],
        experiment_name: Optional[str] = None,
        *args,
        **kwargs
    ):
        """
        Initialize training operator.
        
        Args:
            notebook_path: Path to training notebook
            model_params: Model hyperparameters
            experiment_name: MLflow experiment name
        """
        
        # Prepare notebook parameters with MLflow config
        notebook_params = {
            'model_params': str(model_params),
            'mlflow_tracking_uri': settings.mlflow_tracking_uri,
            'mlflow_experiment_name': experiment_name or settings.mlflow_experiment_name,
            'mlflow_model_name': settings.mlflow_model_name
        }
        
        # Default libraries for training
        libraries = kwargs.pop('libraries', [])
        default_libraries = [
            {"pypi": {"package": "mlflow==2.10.2"}},
            {"pypi": {"package": "xgboost==2.0.3"}},
            {"pypi": {"package": "scikit-learn==1.3.2"}},
            {"pypi": {"package": "pandas==2.1.4"}},
            {"pypi": {"package": "numpy==1.24.3"}}
        ]
        libraries = default_libraries + libraries
        
        super().__init__(
            notebook_path=notebook_path,
            notebook_params=notebook_params,
            libraries=libraries,
            *args,
            **kwargs
        )
        
        self.model_params = model_params
    
    def execute(self, context: Any) -> Dict[str, Any]:
        """
        Execute training run and extract metrics.
        
        Returns:
            Training result with model metrics
        """
        self.log.info("🚀 Starting Databricks training job")
        self.log.info(f"Model parameters: {self.model_params}")
        
        result = super().execute(context)
        
        # Parse output for metrics
        output = result.get('output', {})
        notebook_output = output.get('notebook_output', {})
        
        # Extract metrics if available
        if 'result' in notebook_output:
            try:
                import json
                metrics = json.loads(notebook_output['result'])
                result['metrics'] = metrics
                
                self.log.info(f"Training metrics: {metrics}")
                
                # Store metrics in XCom
                context['task_instance'].xcom_push(key='training_metrics', value=metrics)
                
            except Exception as e:
                self.log.warning(f"Could not parse training metrics: {e}")
        
        return result


# Export
__all__ = ['DatabricksSubmitRunOperator', 'DatabricksTrainingOperator']
