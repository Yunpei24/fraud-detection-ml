#!/usr/bin/env python3
"""
Promote to Production Script
=============================
Promote challenger from            # 4. Move canary models to champion directory and update traffic routing
            logger.info("� Promoting canary models to champion...")
            # Use Azure File Share mount path instead of local /app/models
            azure_mount_path = os.getenv("AZURE_STORAGE_MOUNT_PATH", "/mnt/fraud-models")
            champion_models_dir = Path(f"{azure_mount_path}/champion")
            canary_models_dir = Path(f"{azure_mount_path}/canary")ging to Production (100% rollout).

This script:
1. Transitions challenger from Staging → Production in MLflow Registry
2. Archives old production model to Archived
3. Updates traffic routing to 100% new model
4. Logs promotion event

Usage:
    python promote_to_production.py --model-uri "models:/fraud_detection_xgboost/Staging"
"""
import argparse
import os
import sys
from pathlib import Path
import logging
import json
from datetime import datetime
from typing import List

import mlflow
from mlflow.tracking import MlflowClient
from mlflow.entities.model_registry import ModelVersion

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


class ProductionPromoter:
    """Handle production promotion logic."""
    
    def __init__(self, mlflow_uri: str):
        """
        Initialize promoter.
        
        Args:
            mlflow_uri: MLflow tracking URI
        """
        self.mlflow_uri = mlflow_uri
        mlflow.set_tracking_uri(mlflow_uri)
        self.client = MlflowClient()
        
    def promote_to_production(self, model_uris: List[str]) -> bool:
        """
        Promote ensemble from Staging to Production.
        
        Args:
            model_uris: List of MLflow model URIs for the ensemble in Staging
        
        Returns:
            True if promotion successful
        """
        logger.info(" Promoting ensemble to Production...")
        logger.info(f"   Models: {model_uris}")
        
        try:
            promoted_versions = {}
            
            # Process each model in the ensemble
            for model_uri in model_uris:
                # Parse model URI
                parts = model_uri.replace("models:/", "").split("/")
                model_name = parts[0]
                # stage = parts[1] if len(parts) > 1 else "Staging"
                
                logger.info(f" Processing {model_name}...")
                
                # 1. Get current Staging model
                staging_versions = self.client.get_latest_versions(model_name, stages=["Staging"])
                
                if not staging_versions:
                    logger.error(f"No {model_name} in Staging stage")
                    return False
                
                staging_version = staging_versions[0]
                logger.info(f"  Found Staging version: {staging_version.version}")
                
                # 2. Archive old Production model
                production_versions = self.client.get_latest_versions(model_name, stages=["Production"])
                
                for version in production_versions:
                    logger.info(f"   Archiving {model_name} version {version.version}...")
                    self.client.transition_model_version_stage(
                        name=model_name,
                        version=version.version,
                        stage="Archived",
                        archive_existing_versions=False,
                    )
                
                # 3. Promote Staging → Production
                logger.info(f"  Promoting {model_name} version {staging_version.version} to Production...")
                self.client.transition_model_version_stage(
                    name=model_name,
                    version=staging_version.version,
                    stage="Production",
                    archive_existing_versions=True,
                )
                
                promoted_versions[model_name] = staging_version.version
                logger.info(f"   {model_name} version {staging_version.version} promoted to Production")
            
            # 4. Move canary models to champion directory and update traffic routing
            logger.info("� Promoting canary models to champion...")
            champion_models_dir = Path("/app/models/champion")
            canary_models_dir = Path("/app/models/canary")
            
            # Remove old champion models
            if champion_models_dir.exists():
                import shutil
                shutil.rmtree(champion_models_dir)
            
            # Move canary models to champion
            if canary_models_dir.exists():
                shutil.move(str(canary_models_dir), str(champion_models_dir))
                logger.info("   Moved canary models to champion directory")
            else:
                logger.warning("  No canary models directory found")
            
            # Update traffic routing to 0% canary (all traffic to champion)
            logger.info("🔧 Updating traffic routing to 100% champion...")
            config_file = Path("/app/config/traffic_routing.json")
            
            if not config_file.parent.exists():
                config_file.parent.mkdir(parents=True, exist_ok=True)
            
            # Use Azure File Share mount path for model paths
            azure_mount_path = os.getenv("AZURE_STORAGE_MOUNT_PATH", "/mnt/fraud-models")
            config = {
                "canary_percentage": 0,
                "canary_model_path": f"{azure_mount_path}/canary",
                "champion_model_path": f"{azure_mount_path}/champion"
            }
            
            with open(config_file, "w") as f:
                json.dump(config, f, indent=2)
            
            logger.info(" Traffic routing updated: 100% → champion models (previously canary)")
            
            # 5. Log promotion event
            logger.info("Logging ensemble promotion event...")
            # In production, this would log to monitoring/alerting system
            
            logger.info("Ensemble production promotion complete!")
            for model_name, version in promoted_versions.items():
                logger.info(f"   {model_name}: v{version} → Production")
            
            return True
            
        except Exception as e:
            logger.error(f" Ensemble production promotion failed: {e}")
            import traceback
            traceback.print_exc()
            return False


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Promote ensemble to Production")
    parser.add_argument(
        "--model-uris",
        type=str,
        nargs='+',
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
    
    # Promote
    promoter = ProductionPromoter(args.mlflow_uri)
    success = promoter.promote_to_production(model_uris=args.model_uris)
    
    if success:
        logger.info("Ensemble promoted to Production successfully")
        sys.exit(0)
    else:
        logger.error("Ensemble production promotion failed")
        sys.exit(1)


if __name__ == "__main__":
    main()
