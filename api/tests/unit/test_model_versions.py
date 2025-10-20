import pytest
from src.services.model_versions import (
    ModelRegistry, ModelVersion, ModelMetrics,
    ModelFramework, ModelStage, get_model_registry, init_model_registry
)
from datetime import datetime


class TestModelMetrics:
    def test_metrics_to_dict(self):
        metrics = ModelMetrics(
            accuracy=0.95,
            precision=0.96,
            recall=0.94,
            f1_score=0.95,
            auc_roc=0.99
        )
        
        d = metrics.to_dict()
        assert d["accuracy"] == 0.95
        assert d["precision"] == 0.96
        assert d["recall"] == 0.94
        assert d["f1_score"] == 0.95
        assert d["auc_roc"] == 0.99


class TestModelVersion:
    def test_model_version_creation(self):
        metrics = ModelMetrics(0.95, 0.96, 0.94, 0.95, 0.99)
        version = ModelVersion(
            version_id="xgb-v1.0.0",
            framework=ModelFramework.XGBOOST,
            stage=ModelStage.STAGING,
            metrics=metrics,
            created_at=datetime.utcnow()
        )
        
        assert version.version_id == "xgb-v1.0.0"
        assert version.framework == ModelFramework.XGBOOST
        assert version.stage == ModelStage.STAGING
    
    def test_model_version_to_dict(self):
        metrics = ModelMetrics(0.95, 0.96, 0.94, 0.95, 0.99)
        version = ModelVersion(
            version_id="xgb-v1.0.0",
            framework=ModelFramework.XGBOOST,
            stage=ModelStage.STAGING,
            metrics=metrics,
            created_at=datetime.utcnow(),
            description="Test model"
        )
        
        d = version.to_dict()
        assert d["version_id"] == "xgb-v1.0.0"
        assert d["framework"] == "xgboost"
        assert d["stage"] == "Staging"
        assert d["description"] == "Test model"


class TestModelRegistry:
    @pytest.fixture
    def registry(self):
        return ModelRegistry()
    
    def test_register_model(self, registry):
        metrics = ModelMetrics(0.95, 0.96, 0.94, 0.95, 0.99)
        version = registry.register_model(
            version_id="xgb-v1.0.0",
            framework=ModelFramework.XGBOOST,
            metrics=metrics,
            description="Test model"
        )
        
        assert version.version_id == "xgb-v1.0.0"
        assert version.stage == ModelStage.STAGING
        assert registry.get_version("xgb-v1.0.0") is not None
    
    def test_get_version_not_found(self, registry):
        result = registry.get_version("nonexistent")
        assert result is None
    
    def test_get_current_version(self, registry):
        metrics = ModelMetrics(0.95, 0.96, 0.94, 0.95, 0.99)
        version = registry.register_model(
            version_id="xgb-v1.0.0",
            framework=ModelFramework.XGBOOST,
            metrics=metrics
        )
        
        current = registry.get_current_version(ModelStage.STAGING)
        assert current.version_id == "xgb-v1.0.0"
    
    def test_get_best_model(self, registry):
        metrics1 = ModelMetrics(0.95, 0.96, 0.94, 0.95, 0.99)
        metrics2 = ModelMetrics(0.98, 0.99, 0.97, 0.98, 0.999)
        
        registry.register_model(
            version_id="xgb-v1.0.0",
            framework=ModelFramework.XGBOOST,
            metrics=metrics1
        )
        registry.register_model(
            version_id="rf-v1.0.0",
            framework=ModelFramework.RANDOM_FOREST,
            metrics=metrics2
        )
        
        best = registry.get_best_model(metric="f1_score")
        assert best.version_id == "rf-v1.0.0"
    
    @pytest.mark.asyncio
    async def test_transition_stage(self, registry):
        metrics = ModelMetrics(0.95, 0.96, 0.94, 0.95, 0.99)
        registry.register_model(
            version_id="xgb-v1.0.0",
            framework=ModelFramework.XGBOOST,
            metrics=metrics
        )
        
        result = await registry.transition_stage(
            version_id="xgb-v1.0.0",
            new_stage=ModelStage.PRODUCTION
        )
        
        assert result is True
        version = registry.get_version("xgb-v1.0.0")
        assert version.stage == ModelStage.PRODUCTION
        assert version.deployed_at is not None
    
    @pytest.mark.asyncio
    async def test_transition_nonexistent_version(self, registry):
        result = await registry.transition_stage(
            version_id="nonexistent",
            new_stage=ModelStage.PRODUCTION
        )
        assert result is False
    
    @pytest.mark.asyncio
    async def test_canary_deployment(self, registry):
        metrics_old = ModelMetrics(0.95, 0.96, 0.94, 0.95, 0.99)
        metrics_new = ModelMetrics(0.98, 0.99, 0.97, 0.98, 0.999)
        
        old_version = registry.register_model(
            version_id="xgb-v1.0.0",
            framework=ModelFramework.XGBOOST,
            metrics=metrics_old
        )
        new_version = registry.register_model(
            version_id="xgb-v1.1.0",
            framework=ModelFramework.XGBOOST,
            metrics=metrics_new
        )
        
        config = await registry.canary_deployment(
            new_version_id="xgb-v1.1.0",
            current_version_id="xgb-v1.0.0",
            initial_traffic_percentage=5.0
        )
        
        assert config["new_version"] == "xgb-v1.1.0"
        assert config["current_version"] == "xgb-v1.0.0"
        assert config["canary"]["current_traffic_percentage"] == 5.0
        assert config["canary"]["status"] == "ACTIVE"
        assert config["metrics_comparison"]["new_vs_current"]["f1_score_diff"] > 0
    
    @pytest.mark.asyncio
    async def test_complete_canary_deployment(self, registry):
        metrics_old = ModelMetrics(0.95, 0.96, 0.94, 0.95, 0.99)
        metrics_new = ModelMetrics(0.98, 0.99, 0.97, 0.98, 0.999)
        
        registry.register_model(
            version_id="xgb-v1.0.0",
            framework=ModelFramework.XGBOOST,
            metrics=metrics_old
        )
        registry.register_model(
            version_id="xgb-v1.1.0",
            framework=ModelFramework.XGBOOST,
            metrics=metrics_new
        )
        
        await registry.transition_stage("xgb-v1.0.0", ModelStage.PRODUCTION)
        result = await registry.complete_canary_deployment("xgb-v1.1.0")
        
        assert result is True
        current = registry.get_current_version(ModelStage.PRODUCTION)
        assert current.version_id == "xgb-v1.1.0"
        
        old = registry.get_version("xgb-v1.0.0")
        assert old.stage == ModelStage.ARCHIVED
    
    def test_get_all_versions(self, registry):
        metrics1 = ModelMetrics(0.95, 0.96, 0.94, 0.95, 0.99)
        metrics2 = ModelMetrics(0.98, 0.99, 0.97, 0.98, 0.999)
        
        registry.register_model(
            version_id="xgb-v1.0.0",
            framework=ModelFramework.XGBOOST,
            metrics=metrics1
        )
        registry.register_model(
            version_id="rf-v1.0.0",
            framework=ModelFramework.RANDOM_FOREST,
            metrics=metrics2
        )
        
        all_versions = registry.get_all_versions()
        assert len(all_versions) == 2
    
    def test_get_versions_by_stage(self, registry):
        metrics = ModelMetrics(0.95, 0.96, 0.94, 0.95, 0.99)
        registry.register_model(
            version_id="xgb-v1.0.0",
            framework=ModelFramework.XGBOOST,
            metrics=metrics
        )
        
        staging = registry.get_all_versions(ModelStage.STAGING)
        assert len(staging) == 1
        
        production = registry.get_all_versions(ModelStage.PRODUCTION)
        assert len(production) == 0
    
    def test_get_version_stats(self, registry):
        metrics = ModelMetrics(0.95, 0.96, 0.94, 0.95, 0.99)
        registry.register_model(
            version_id="xgb-v1.0.0",
            framework=ModelFramework.XGBOOST,
            metrics=metrics
        )
        
        stats = registry.get_version_stats()
        assert stats["total_versions"] == 1
        assert stats["versions_by_stage"]["Staging"] == 1
        assert stats["versions_by_stage"]["Production"] == 0


class TestRegistryGlobals:
    def test_get_model_registry_singleton(self):
        reg1 = get_model_registry()
        reg2 = get_model_registry()
        assert reg1 is reg2
    
    def test_init_model_registry(self):
        init_model_registry(registry_path="/tmp/test")
        registry = get_model_registry()
        assert registry.registry_path == "/tmp/test"
