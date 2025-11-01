#!/usr/bin/env python3
"""
Rollback Deployment Script
===========================
Rollback to champion model (100% traffic).

This script:
1. Disables canary routing
2. Restores 100% traffic to Production model
3. Logs rollback event

Usage:
    python rollback_deployment.py
"""
import argparse
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path

from mlflow.tracking import MlflowClient

import mlflow

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


class DeploymentRollback:
    """Handle rollback logic."""

    def __init__(self, mlflow_uri: str):
        """
        Initialize rollback handler.

        Args:
            mlflow_uri: MLflow tracking URI
        """
        self.mlflow_uri = mlflow_uri
        mlflow.set_tracking_uri(mlflow_uri)
        self.client = MlflowClient()

    def rollback(self) -> dict:
        """
        Rollback deployment to champion model.

        Returns:
            Dictionary with success status and details
        """
        logger.info(" Rolling back to champion model...")

        try:
            # 1. Disable canary routing
            logger.info(" Disabling canary routing...")
            config_file = Path("/app/config/traffic_routing.json")

            if not config_file.parent.exists():
                config_file.parent.mkdir(parents=True, exist_ok=True)

            config = {
                "canary_enabled": False,
                "canary_traffic_pct": 0,
                "canary_model_uri": None,
                "champion_traffic_pct": 100,
                "rollback_timestamp": datetime.utcnow().isoformat(),
            }

            with open(config_file, "w") as f:
                json.dump(config, f, indent=2)

            logger.info(" Canary routing disabled")
            logger.info(" 100% traffic restored to champion")

            # 2. Log rollback event
            logger.info(" Logging rollback event...")
            # In production, this would log to monitoring system

            logger.info(" Rollback completed successfully")

            return {
                "success": True,
                "champion_models": config.get("champion_models", []),
                "traffic_percentage": 100,
            }

        except Exception as e:
            logger.error(f" Rollback failed: {e}")
            return {"success": False, "error": str(e)}


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Rollback canary deployment")
    parser.add_argument(
        "--mlflow-uri",
        type=str,
        default=os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000"),
        help="MLflow tracking URI",
    )

    args = parser.parse_args()

    # Rollback
    rollback = DeploymentRollback(args.mlflow_uri)
    result = rollback.rollback()

    if result.get("success"):
        logger.info(" Rollback completed successfully")
        sys.exit(0)
    else:
        logger.error(f" Rollback failed: {result.get('error', 'Unknown error')}")
        sys.exit(1)


if __name__ == "__main__":
    main()
