"""
MLflow custom operator for Airflow
"""
from typing import Any, Dict, Optional
from airflow.models import BaseOperator
from airflow.utils.decorators import apply_defaults
import mlflow


class MLflowRegisterModelOperator(BaseOperator):
    """
    Register trained model to MLflow Model Registry.
    
    :param model_uri: URI of the model to register (e.g., 'runs:/run_id/model')
    :param model_name: Name to register the model under
    :param mlflow_tracking_uri: MLflow tracking server URI
    """
    
    @apply_defaults
    def __init__(
        self,
        model_uri: str,
        model_name: str,
        mlflow_tracking_uri: str = "http://mlflow:5000",
        tags: Optional[Dict[str, Any]] = None,
        description: Optional[str] = None,
        *args,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.model_uri = model_uri
        self.model_name = model_name
        self.mlflow_tracking_uri = mlflow_tracking_uri
        self.tags = tags or {}
        self.description = description
    
    def execute(self, context):
        """Register model to MLflow"""
        self.log.info(f"Registering model: {self.model_name}")
        self.log.info(f"Model URI: {self.model_uri}")
        
        mlflow.set_tracking_uri(self.mlflow_tracking_uri)
        
        # Register model
        result = mlflow.register_model(
            model_uri=self.model_uri,
            name=self.model_name,
            tags=self.tags
        )
        
        self.log.info(f"✅ Model registered: {self.model_name} version {result.version}")
        
        # Return version for downstream tasks
        return {
            "model_name": self.model_name,
            "model_version": result.version,
            "model_uri": self.model_uri
        }
