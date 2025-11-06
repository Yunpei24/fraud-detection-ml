"""
Traffic Router for canary deployment.
Routes traffic between champion and canary models based on configuration.
"""

import json
import logging
import os
import random
from pathlib import Path
from typing import Any, Dict, Optional

from ..config import get_logger, settings

from dotenv import load_dotenv

load_dotenv()

logger = get_logger(__name__)


class TrafficRouter:
    """
    Routes traffic between champion and canary models for canary deployment.

    Configuration format:
    {
        "canary_percentage": 20,
        "canary_model_path": "/app/models/canary",
        "champion_model_path": "/app/models/champion"
    }
    """

    def __init__(self):
        self.config: Dict[str, Any] = {}
        self.canary_percentage: float = 0.0
        self.canary_model_path: Optional[str] = None
        self.champion_model_path: Optional[str] = None
        self._load_config()

    def _load_config(self) -> None:
        """
        Load traffic routing configuration from file.
        """
        try:
            config_path = Path(settings.traffic_routing_config)
            if config_path.exists():
                with open(config_path, "r") as f:
                    self.config = json.load(f)

                self.canary_percentage = self.config.get("canary_percentage", 0.0)
                self.canary_model_path = self.config.get("canary_model_path")
                self.champion_model_path = self.config.get("champion_model_path")

                logger.info(
                    f"Loaded traffic routing config: {self.canary_percentage}% to canary"
                )
            else:
                logger.info(
                    "No traffic routing config found, using Azure File Share champion path"
                )
                self.canary_percentage = 0.0
                # Default to Azure File Share champion path when no config exists
                azure_mount_path = os.getenv(
                    "AZURE_STORAGE_MOUNT_PATH", "/mnt/fraud-models"
                )
                self.canary_model_path = f"{azure_mount_path}/canary"
                self.champion_model_path = f"{azure_mount_path}/champion"

        except Exception as e:
            logger.error(f"Failed to load traffic routing config: {e}")
            self.canary_percentage = 0.0

    def should_use_canary(self) -> bool:
        """
        Determine if the current request should use the canary model.

        Returns:
            True if canary model should be used, False for champion
        """
        if self.canary_percentage <= 0:
            return False

        if self.canary_percentage >= 100:
            return True

        # Random routing based on percentage
        return random.random() * 100 < self.canary_percentage

    def get_model_path(self, use_canary: bool = None) -> str:
        """
        Get the model path based on routing decision.

        Args:
            use_canary: Override routing decision

        Returns:
            Path to the model directory
        """
        if use_canary is None:
            use_canary = self.should_use_canary()

        if use_canary and self.canary_model_path:
            return self.canary_model_path
        elif self.champion_model_path:
            return self.champion_model_path
        else:
            # Fallback to default model path
            return settings.model_path

    def get_model_type(self, use_canary: bool = None) -> str:
        """
        Get the model type (champion or canary) for the current routing decision.

        Args:
            use_canary: Override routing decision

        Returns:
            "canary" or "champion"
        """
        if use_canary is None:
            use_canary = self.should_use_canary()

        return "canary" if use_canary else "champion"

    def reload_config(self) -> None:
        """
        Reload the traffic routing configuration.
        """
        self._load_config()
