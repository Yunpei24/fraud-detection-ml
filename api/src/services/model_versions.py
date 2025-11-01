import logging
import os
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class ModelStage(str, Enum):
    STAGING = "Staging"
    PRODUCTION = "Production"
    ARCHIVED = "Archived"


class ModelFramework(str, Enum):
    XGBOOST = "xgboost"
    RANDOM_FOREST = "random_forest"
    NEURAL_NETWORK = "neural_network"
    ISOLATION_FOREST = "isolation_forest"


@dataclass
class ModelMetrics:
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    auc_roc: float

    def to_dict(self) -> Dict[str, float]:
        return {
            "accuracy": self.accuracy,
            "precision": self.precision,
            "recall": self.recall,
            "f1_score": self.f1_score,
            "auc_roc": self.auc_roc,
        }


@dataclass
class ModelVersion:
    version_id: str
    framework: ModelFramework
    stage: ModelStage
    metrics: ModelMetrics
    created_at: datetime
    deployed_at: Optional[datetime] = None
    description: Optional[str] = None
    run_id: Optional[str] = None
    artifact_uri: Optional[str] = None
    tags: Optional[Dict[str, str]] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "version_id": self.version_id,
            "framework": self.framework.value,
            "stage": self.stage.value,
            "metrics": self.metrics.to_dict(),
            "created_at": self.created_at.isoformat(),
            "deployed_at": self.deployed_at.isoformat() if self.deployed_at else None,
            "description": self.description,
            "run_id": self.run_id,
            "artifact_uri": self.artifact_uri,
            "tags": self.tags or {},
        }


class ModelRegistry:
    def __init__(self, registry_path: Optional[str] = None):
        self.registry_path = registry_path or os.getenv(
            "MODEL_REGISTRY_PATH", "./models"
        )
        self._versions: Dict[str, ModelVersion] = {}
        self._current_production: Optional[str] = None
        logger.info(f"Model registry initialized at {self.registry_path}")

    def register_model(
        self,
        version_id: str,
        framework: ModelFramework,
        metrics: ModelMetrics,
        description: Optional[str] = None,
        run_id: Optional[str] = None,
        artifact_uri: Optional[str] = None,
        tags: Optional[Dict[str, str]] = None,
    ) -> ModelVersion:
        if version_id in self._versions:
            logger.warning(f"Model version {version_id} already registered")

        version = ModelVersion(
            version_id=version_id,
            framework=framework,
            stage=ModelStage.STAGING,
            metrics=metrics,
            created_at=datetime.utcnow(),
            description=description,
            run_id=run_id,
            artifact_uri=artifact_uri,
            tags=tags or {},
        )

        self._versions[version_id] = version
        logger.info(f"Model {version_id} registered in {ModelStage.STAGING.value}")

        return version

    def get_version(self, version_id: str) -> Optional[ModelVersion]:
        return self._versions.get(version_id)

    def get_current_version(
        self, stage: ModelStage = ModelStage.PRODUCTION
    ) -> Optional[ModelVersion]:
        for version in self._versions.values():
            if version.stage == stage:
                return version
        return None

    def get_best_model(self, metric: str = "f1_score") -> Optional[ModelVersion]:
        if not self._versions:
            return None

        best_version = None
        best_value = -1

        for version in self._versions.values():
            if hasattr(version.metrics, metric):
                value = getattr(version.metrics, metric)
                if value > best_value:
                    best_value = value
                    best_version = version

        return best_version

    async def transition_stage(
        self,
        version_id: str,
        new_stage: ModelStage,
        archive_current: bool = True,
    ) -> bool:
        version = self.get_version(version_id)
        if not version:
            logger.error(f"Model {version_id} not found in registry")
            return False

        old_stage = version.stage

        if old_stage == ModelStage.ARCHIVED and new_stage != ModelStage.STAGING:
            logger.error(
                f"Cannot transition from {old_stage.value} to {new_stage.value}"
            )
            return False

        if archive_current and new_stage in [ModelStage.PRODUCTION, ModelStage.STAGING]:
            current = self.get_current_version(new_stage)
            if current and current.version_id != version_id:
                current.stage = ModelStage.ARCHIVED
                logger.info(f"Archived {current.version_id}")

        version.stage = new_stage
        version.deployed_at = datetime.utcnow()

        if new_stage == ModelStage.PRODUCTION:
            self._current_production = version_id

        logger.info(
            f"Model {version_id} transitioned from {old_stage.value} to {new_stage.value}"
        )

        return True

    async def canary_deployment(
        self,
        new_version_id: str,
        current_version_id: str,
        initial_traffic_percentage: float = 5.0,
        max_traffic_percentage: float = 100.0,
    ) -> Dict[str, Any]:
        new_version = self.get_version(new_version_id)
        current_version = self.get_version(current_version_id)

        if not new_version or not current_version:
            logger.error("One or both versions not found in registry")
            return {}

        if initial_traffic_percentage < 0 or initial_traffic_percentage > 100:
            logger.error(
                f"Invalid initial traffic percentage: {initial_traffic_percentage}"
            )
            return {}

        canary_config = {
            "new_version": new_version_id,
            "current_version": current_version_id,
            "new_version_metrics": new_version.to_dict(),
            "current_version_metrics": current_version.to_dict(),
            "canary": {
                "initial_traffic_percentage": initial_traffic_percentage,
                "max_traffic_percentage": max_traffic_percentage,
                "current_traffic_percentage": initial_traffic_percentage,
                "steps": [5, 10, 25, 50, 75, 90, 100],
                "started_at": datetime.utcnow().isoformat(),
                "status": "ACTIVE",
            },
            "metrics_comparison": {
                "new_vs_current": {
                    "accuracy_diff": round(
                        new_version.metrics.accuracy - current_version.metrics.accuracy,
                        4,
                    ),
                    "f1_score_diff": round(
                        new_version.metrics.f1_score - current_version.metrics.f1_score,
                        4,
                    ),
                    "auc_roc_diff": round(
                        new_version.metrics.auc_roc - current_version.metrics.auc_roc, 4
                    ),
                }
            },
        }

        logger.info(
            f"Canary deployment started: {new_version_id} ({initial_traffic_percentage}% "
            f"→ {max_traffic_percentage}%) replacing {current_version_id}"
        )

        return canary_config

    async def complete_canary_deployment(self, new_version_id: str) -> bool:
        new_version = self.get_version(new_version_id)
        if not new_version:
            logger.error(f"Model {new_version_id} not found")
            return False

        current_prod = self.get_current_version(ModelStage.PRODUCTION)
        if current_prod and current_prod.version_id != new_version_id:
            current_prod.stage = ModelStage.ARCHIVED
            logger.info(f"Archived {current_prod.version_id}")

        await self.transition_stage(
            new_version_id, ModelStage.PRODUCTION, archive_current=False
        )

        logger.info(
            f"Canary deployment completed, {new_version_id} is now in production"
        )

        return True

    def get_all_versions(
        self, stage: Optional[ModelStage] = None
    ) -> List[ModelVersion]:
        versions = list(self._versions.values())

        if stage:
            versions = [v for v in versions if v.stage == stage]

        versions.sort(key=lambda v: v.created_at, reverse=True)

        return versions

    def get_version_stats(self) -> Dict[str, Any]:
        versions_by_stage = {}
        for stage in ModelStage:
            count = len(self.get_all_versions(stage))
            versions_by_stage[stage.value] = count

        total_versions = len(self._versions)
        current_prod = self.get_current_version(ModelStage.PRODUCTION)

        return {
            "total_versions": total_versions,
            "versions_by_stage": versions_by_stage,
            "current_production": (
                {
                    "version_id": current_prod.version_id,
                    "framework": current_prod.framework.value,
                    "created_at": current_prod.created_at.isoformat(),
                    "deployed_at": (
                        current_prod.deployed_at.isoformat()
                        if current_prod.deployed_at
                        else None
                    ),
                }
                if current_prod
                else None
            ),
        }


_model_registry: Optional[ModelRegistry] = None


def get_model_registry() -> ModelRegistry:
    global _model_registry
    if _model_registry is None:
        _model_registry = ModelRegistry()
    return _model_registry


def init_model_registry(registry_path: Optional[str] = None) -> ModelRegistry:
    global _model_registry
    _model_registry = ModelRegistry(registry_path=registry_path)
    return _model_registry
