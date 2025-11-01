"""
Wrapper functions for deployment scripts to be called from API endpoints.
"""

import logging
from typing import List, Dict, Any

logger = logging.getLogger(__name__)


def deploy_canary_models(
    model_uris: List[str], traffic_pct: int, mlflow_uri: str
) -> Dict[str, Any]:
    """
    Deploy canary models from API endpoint.

    Args:
        model_uris: List of MLflow model URIs
        traffic_pct: Traffic percentage (0-100)
        mlflow_uri: MLflow tracking URI

    Returns:
        Result dictionary
    """
    try:
        from scripts.deploy_canary import CanaryDeployer

        deployer = CanaryDeployer(mlflow_uri=mlflow_uri)
        return deployer.deploy_canary(model_uris=model_uris, traffic_pct=traffic_pct)

    except Exception as e:
        logger.error(f"Error in deploy_canary_models: {e}", exc_info=True)
        return {"success": False, "error": str(e)}


def promote_models(model_uris: List[str], mlflow_uri: str) -> Dict[str, Any]:
    """
    Promote models to production from API endpoint.

    Args:
        model_uris: List of MLflow model URIs
        mlflow_uri: MLflow tracking URI

    Returns:
        Result dictionary
    """
    try:
        from scripts.promote_to_production import ProductionPromoter

        promoter = ProductionPromoter(mlflow_uri=mlflow_uri)
        return promoter.promote_to_production(model_uris=model_uris)

    except Exception as e:
        logger.error(f"Error in promote_models: {e}", exc_info=True)
        return {"success": False, "error": str(e)}


def rollback_to_champion() -> Dict[str, Any]:
    """
    Rollback to champion models from API endpoint.

    Returns:
        Result dictionary
    """
    try:
        from scripts.rollback_deployment import rollback_to_champion_models

        return rollback_to_champion_models()

    except Exception as e:
        logger.error(f"Error in rollback_to_champion: {e}", exc_info=True)
        return {"success": False, "error": str(e)}
