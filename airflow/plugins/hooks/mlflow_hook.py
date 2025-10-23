"""
MLflow Hook for Model Registry Operations
Uses centralized configuration from airflow.src.config.settings
"""
from typing import Any, Dict, Optional
from airflow.hooks.base import BaseHook
import mlflow
from mlflow.tracking import MlflowClient
import sys
from pathlib import Path

# Add airflow/src to path
AIRFLOW_SRC = Path(__file__).parent.parent.parent / "src"
sys.path.insert(0, str(AIRFLOW_SRC))

from config.settings import settings


class MLflowHook(BaseHook):
    """Hook for interacting with MLflow Model Registry"""
    
    def __init__(self, tracking_uri: Optional[str] = None):
        """
        Initialize MLflow hook.
        
        Args:
            tracking_uri: MLflow tracking server URI (defaults to settings.mlflow_tracking_uri)
        """
        super().__init__()
        self.tracking_uri = tracking_uri or settings.mlflow_tracking_uri
        mlflow.set_tracking_uri(self.tracking_uri)
        self.client = MlflowClient()
    
    def _get_tracking_uri(self) -> str:
        """Get MLflow tracking URI from settings"""
        from airflow.config.settings import settings
        return settings.mlflow_tracking_uri
    
    def create_experiment(self, experiment_name: str, tags: Optional[Dict] = None) -> str:
        """Create MLflow experiment"""
        experiment = self.client.get_experiment_by_name(experiment_name)
        
        if experiment:
            return experiment.experiment_id
        
        return self.client.create_experiment(
            experiment_name,
            tags=tags
        )
    
    def log_params(self, run_id: str, params: Dict[str, Any]) -> None:
        """Log parameters to run"""
        for key, value in params.items():
            self.client.log_param(run_id, key, value)
    
    def log_metrics(self, run_id: str, metrics: Dict[str, float]) -> None:
        """Log metrics to run"""
        for key, value in metrics.items():
            self.client.log_metric(run_id, key, value)
    
    def log_model(
        self,
        model,
        artifact_path: str,
        registered_model_name: Optional[str] = None
    ) -> None:
        """Log model to MLflow"""
        mlflow.sklearn.log_model(
            model,
            artifact_path,
            registered_model_name=registered_model_name
        )
    
    def register_model(
        self,
        model_uri: str,
        model_name: str,
        tags: Optional[Dict] = None
    ) -> Any:
        """Register model to Model Registry"""
        model_version = mlflow.register_model(
            model_uri=model_uri,
            name=model_name,
            tags=tags
        )
        return model_version
    
    def transition_model_stage(
        self,
        model_name: str,
        version: int,
        stage: str,
        archive_existing: bool = True
    ) -> None:
        """Transition model to new stage"""
        self.client.transition_model_version_stage(
            name=model_name,
            version=version,
            stage=stage,
            archive_existing_versions=archive_existing
        )
    
    def get_latest_model_version(
        self,
        model_name: str,
        stage: Optional[str] = None
    ) -> Optional[Any]:
        """Get latest model version"""
        if stage:
            versions = self.client.get_latest_versions(model_name, stages=[stage])
        else:
            versions = self.client.search_model_versions(f"name='{model_name}'")
        
        if not versions:
            return None
        
        return versions[0] if isinstance(versions, list) else versions
    
    def load_model(self, model_uri: str) -> Any:
        """Load model from URI"""
        return mlflow.sklearn.load_model(model_uri)
    
    def get_model_metrics(self, run_id: str) -> Dict[str, float]:
        """Get metrics for a run"""
        run = self.client.get_run(run_id)
        return run.data.metrics
    
    def search_runs(
        self,
        experiment_ids: list,
        filter_string: Optional[str] = None,
        max_results: int = 100
    ) -> list:
        """Search runs in experiments"""
        return self.client.search_runs(
            experiment_ids=experiment_ids,
            filter_string=filter_string,
            max_results=max_results
        )
    
    def get_experiment_by_name(self, name: str) -> Optional[Any]:
        """Get experiment by name"""
        return self.client.get_experiment_by_name(name)
    
    def delete_model_version(self, model_name: str, version: int) -> None:
        """Delete model version"""
        self.client.delete_model_version(
            name=model_name,
            version=version
        )
    
    def set_model_version_tag(
        self,
        model_name: str,
        version: int,
        key: str,
        value: Any
    ) -> None:
        """Set tag on model version"""
        self.client.set_model_version_tag(
            name=model_name,
            version=version,
            key=key,
            value=str(value)
        )
