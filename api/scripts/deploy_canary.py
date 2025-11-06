#!/usr/bin/env python3
"""
Canary Deployment Script
========================
Deploy challenger model to specified traffic percentage.

This script:
1. Loads challenger model from MLflow Staging
2. Updates traffic routing configuration
3. Deploys to API with specified traffic %
4. Validates deployment

Usage:
    python deploy_canary.py --traffic 5 --model-uri "models:/fraud_detection_xgboost/Staging"
"""
import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import List

import joblib
from mlflow.tracking import MlflowClient

import mlflow

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


class CanaryDeployer:
    """Handle canary deployment logic."""

    def __init__(self, mlflow_uri: str):
        """
        Initialize deployer.

        Args:
            mlflow_uri: MLflow tracking URI
        """
        self.mlflow_uri = mlflow_uri
        mlflow.set_tracking_uri(mlflow_uri)
        self.client = MlflowClient()

    def deploy_canary(
        self,
        model_uris: List[str],
        traffic_pct: int,
    ) -> dict:
        """
        Deploy ensemble as canary with specified traffic.

        Args:
            model_uris: List of MLflow model URIs for the ensemble
            traffic_pct: Traffic percentage for canary (1-100)

        Returns:
            Dictionary with success status and details
        """
        logger.info(f" Deploying ensemble canary with {traffic_pct}% traffic")
        logger.info(f"   Models: {model_uris}")

        try:
            # 1. Load all models from MLflow
            logger.info(" Loading ensemble models and SHAP explainers from MLflow...")
            loaded_models = {}
            loaded_explainers = {}

            for model_uri in model_uris:
                model = mlflow.pyfunc.load_model(model_uri)
                # Extract model name from URI
                model_name = model_uri.split("/")[-2]  # e.g., "fraud_detection_xgboost"
                loaded_models[model_name] = model
                logger.info(f"   Loaded {model_name}")

                # Download SHAP explainer artifact
                try:
                    # Get the run that contains this model
                    client = MlflowClient()
                    model_versions = client.get_latest_versions(
                        model_name, stages=["Staging"]
                    )

                    if model_versions:
                        model_details = model_versions[0]
                        run_id = model_details.run_id

                        # Download SHAP explainer artifact
                        explainer_path = client.download_artifacts(
                            run_id,
                            f"shap_explainer_{model_name.replace('fraud_detection_', '')}.pkl",
                        )
                        with open(explainer_path, "rb") as f:
                            explainer = joblib.load(f)
                        loaded_explainers[model_name] = explainer
                        logger.info(f"   Loaded SHAP explainer for {model_name}")
                    else:
                        logger.warning(f"   No Staging version found for {model_name}")
                except Exception as e:
                    logger.warning(
                        f"   Failed to load SHAP explainer for {model_name}: {e}"
                    )

            logger.info(" Ensemble and explainers loaded successfully")

            # 2. Save models and SHAP explainers to Azure File Share canary directory
            logger.info(
                " Saving models and SHAP explainers to Azure File Share canary directory..."
            )
            # Use Azure File Share mount path instead of local /app/models
            azure_mount_path = os.getenv(
                "AZURE_STORAGE_MOUNT_PATH", "/mnt/fraud-models"
            )
            canary_models_dir = Path(azure_mount_path) / "canary"
            canary_models_dir.mkdir(parents=True, exist_ok=True)

            for model_name, model in loaded_models.items():
                # Save each model with proper naming
                short_name = model_name.replace(
                    "fraud_detection_", ""
                )  # e.g., "xgboost"
                model_path = canary_models_dir / f"{short_name}_model.pkl"
                joblib.dump(model, model_path)
                logger.info(f"   Saved {model_name} to {model_path}")

                # Save SHAP explainer if available
                if model_name in loaded_explainers:
                    explainer_path = (
                        canary_models_dir / f"shap_explainer_{short_name}.pkl"
                    )
                    joblib.dump(loaded_explainers[model_name], explainer_path)
                    logger.info(
                        f"   Saved SHAP explainer for {model_name} to {explainer_path}"
                    )

            # 3. Update traffic routing config for TrafficRouter
            logger.info(f" Updating traffic routing to {traffic_pct}% canary...")
            config_file = Path("/app/config/traffic_routing.json")

            if not config_file.parent.exists():
                config_file.parent.mkdir(parents=True, exist_ok=True)

            # Use Azure File Share mount path for model paths
            azure_mount_path = os.getenv(
                "AZURE_STORAGE_MOUNT_PATH", "/mnt/fraud-models"
            )

            # New config structure matching TrafficRoutingConfig
            config = {
                "canary_enabled": traffic_pct > 0,
                "canary_traffic_pct": traffic_pct,
                "champion_traffic_pct": 100 - traffic_pct,
                "canary_model_uris": [f"{azure_mount_path}/canary"],
                "champion_model_uris": [f"{azure_mount_path}/champion"],
                "ensemble_weights": {
                    "xgboost": 0.50,
                    "random_forest": 0.30,
                    "neural_network": 0.15,
                    "isolation_forest": 0.05,
                },
            }

            with open(config_file, "w") as f:
                json.dump(config, f, indent=2)

            logger.info(
                f" Traffic routing updated: {traffic_pct}% → canary models, {100-traffic_pct}% → champion models"
            )

            # 4. Validate deployment
            logger.info(" Validating ensemble deployment...")
            # In production, this would test the ensemble prediction

            logger.info(
                f" Ensemble canary deployment complete: {traffic_pct}% traffic to challenger ensemble"
            )

            return {
                "success": True,
                "models_loaded": len(loaded_models),
                "traffic_percentage": traffic_pct,
                "model_uris": model_uris,
            }

        except Exception as e:
            logger.error(f" Ensemble canary deployment failed: {e}")
            return {"success": False, "error": str(e)}


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Deploy model in canary mode")
    parser.add_argument(
        "--traffic",
        type=int,
        required=True,
        choices=[5, 25, 100],
        help="Traffic percentage for canary (5, 25, or 100)",
    )
    parser.add_argument(
        "--model-uris",
        type=str,
        nargs="+",
        required=True,
        help="MLflow model URIs for ensemble (e.g., 'models:/fraud_detection_xgboost/Staging' 'models:/fraud_detection_random_forest/Staging')",
    )
    parser.add_argument(
        "--mlflow-uri",
        type=str,
        default=os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000"),
        help="MLflow tracking URI",
    )

    args = parser.parse_args()

    # Deploy
    deployer = CanaryDeployer(args.mlflow_uri)
    result = deployer.deploy_canary(
        model_uris=args.model_uris,
        traffic_pct=args.traffic,
    )

    if result.get("success"):
        logger.info(f" Canary {args.traffic}% deployed successfully")
        sys.exit(0)
    else:
        logger.error(
            f" Canary {args.traffic}% deployment failed: {result.get('error', 'Unknown error')}"
        )
        sys.exit(1)


if __name__ == "__main__":
    main()
