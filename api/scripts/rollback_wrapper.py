"""
Wrapper function for rollback to be called from API.
"""

from typing import Dict, Any
import os


def rollback_to_champion_models() -> Dict[str, Any]:
    """
    Rollback to champion models.

    Returns:
        Result dictionary
    """
    try:
        from scripts.rollback_deployment import DeploymentRollback

        mlflow_uri = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
        rollback_handler = DeploymentRollback(mlflow_uri=mlflow_uri)

        return rollback_handler.rollback()

    except Exception as e:
        return {"success": False, "error": str(e)}
