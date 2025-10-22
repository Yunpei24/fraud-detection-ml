"""
Module Loader for Airflow
Dynamically loads modules from API, Data, Drift, and Training components
"""
import sys
import importlib
from pathlib import Path
from typing import Any, Optional
import logging

logger = logging.getLogger(__name__)


class ModuleLoader:
    """
    Dynamic module loader for importing components from other fraud-detection modules.
    Handles sys.path management and provides convenient access to all modules.
    """
    
    def __init__(self):
        """Initialize module loader with paths to all modules."""
        self.airflow_root = Path(__file__).parent.parent  # airflow/
        self.project_root = self.airflow_root.parent  # fraud-detection-ml/
        
        # Module paths
        self.api_path = self.project_root / "api"
        self.data_path = self.project_root / "data"
        self.drift_path = self.project_root / "drift"
        self.training_path = self.project_root / "training"  # Future
        
        # Add all module paths to sys.path
        self._setup_paths()
    
    def _setup_paths(self):
        """Add all module paths to sys.path if they exist."""
        paths = [
            str(self.api_path),
            str(self.data_path),
            str(self.drift_path),
            str(self.training_path),
        ]
        
        for path in paths:
            if Path(path).exists() and path not in sys.path:
                sys.path.insert(0, path)
                logger.debug(f"Added to sys.path: {path}")
    
    def check_module_availability(self, module_name: str) -> bool:
        """
        Check if a module is available for import.
        
        Args:
            module_name: Name of the module to check
            
        Returns:
            True if module can be imported, False otherwise
        """
        try:
            importlib.import_module(module_name)
            return True
        except ImportError:
            logger.warning(f"Module {module_name} not available")
            return False
    
    # ========== API Module ==========
    
    def load_api_settings(self) -> Any:
        """Load API settings."""
        try:
            from src.config.settings import settings
            return settings
        except ImportError as e:
            logger.error(f"Failed to load API settings: {e}")
            raise
    
    def load_prediction_service(self) -> Any:
        """Load PredictionService from API."""
        try:
            from src.services.prediction_service import PredictionService
            return PredictionService
        except ImportError as e:
            logger.error(f"Failed to load PredictionService: {e}")
            raise
    
    def load_model_service(self) -> Any:
        """Load ModelService from API."""
        try:
            from src.services.model_versions import ModelVersionService
            return ModelVersionService
        except ImportError as e:
            logger.error(f"Failed to load ModelVersionService: {e}")
            raise
    
    def load_cache_service(self) -> Any:
        """Load CacheService from API."""
        try:
            from src.services.cache_service import CacheService
            return CacheService
        except ImportError as e:
            logger.error(f"Failed to load CacheService: {e}")
            raise
    
    # ========== Data Module ==========
    
    def load_data_settings(self) -> Any:
        """Load Data module settings."""
        try:
            from src.config.settings import settings
            return settings
        except ImportError as e:
            logger.error(f"Failed to load Data settings: {e}")
            raise
    
    def load_data_ingestion(self) -> Any:
        """Load DataIngestionPipeline from Data module."""
        try:
            from src.ingestion.data_ingestion_pipeline import DataIngestionPipeline
            return DataIngestionPipeline
        except ImportError as e:
            logger.error(f"Failed to load DataIngestionPipeline: {e}")
            raise
    
    def load_database_service(self) -> Any:
        """Load DatabaseService from Data module."""
        try:
            from src.storage.database import DatabaseService
            return DatabaseService
        except ImportError as e:
            logger.error(f"Failed to load DatabaseService: {e}")
            raise
    
    def load_data_validation(self) -> Any:
        """Load DataValidation from Data module."""
        try:
            from src.validation.data_validation import DataValidation
            return DataValidation
        except ImportError as e:
            logger.error(f"Failed to load DataValidation: {e}")
            raise
    
    def load_feature_engineering(self) -> Any:
        """Load FeatureEngineering from Data module."""
        try:
            from src.features.feature_engineering import FeatureEngineering
            return FeatureEngineering
        except ImportError as e:
            logger.error(f"Failed to load FeatureEngineering: {e}")
            raise
    
    # ========== Drift Module ==========
    
    def load_drift_settings(self) -> Any:
        """Load Drift module settings."""
        try:
            from src.config.settings import settings
            return settings
        except ImportError as e:
            logger.error(f"Failed to load Drift settings: {e}")
            raise
    
    def load_drift_detector(self) -> Any:
        """Load DriftDetector from Drift module."""
        try:
            from src.detection.drift_detector import DriftDetector
            return DriftDetector
        except ImportError as e:
            logger.error(f"Failed to load DriftDetector: {e}")
            raise
    
    def load_hourly_monitoring(self) -> Any:
        """Load run_hourly_monitoring from Drift module."""
        try:
            from src.monitoring.hourly_monitoring import run_hourly_monitoring
            return run_hourly_monitoring
        except ImportError as e:
            logger.error(f"Failed to load run_hourly_monitoring: {e}")
            raise
    
    def load_alert_manager(self) -> Any:
        """Load AlertManager from Drift module."""
        try:
            from src.alerting.alert_manager import AlertManager
            return AlertManager
        except ImportError as e:
            logger.error(f"Failed to load AlertManager: {e}")
            raise
    
    def load_drift_analysis(self) -> Any:
        """Load DriftAnalysis from Drift module."""
        try:
            from src.analysis.drift_analysis import DriftAnalysis
            return DriftAnalysis
        except ImportError as e:
            logger.error(f"Failed to load DriftAnalysis: {e}")
            raise
    
    # ========== Training Module (Future) ==========
    
    def load_training_settings(self) -> Optional[Any]:
        """Load Training module settings (when available)."""
        try:
            from src.config.settings import settings
            return settings
        except ImportError:
            logger.warning("Training module not yet available")
            return None
    
    def load_training_pipeline(self) -> Optional[Any]:
        """Load TrainingPipeline from Training module (when available)."""
        try:
            from src.training.training_pipeline import TrainingPipeline
            return TrainingPipeline
        except ImportError:
            logger.warning("Training module not yet available")
            return None
    
    def load_model_validator(self) -> Optional[Any]:
        """Load ModelValidator from Training module (when available)."""
        try:
            from src.validation.model_validator import ModelValidator
            return ModelValidator
        except ImportError:
            logger.warning("Training module not yet available")
            return None
    
    def load_hyperparameter_tuning(self) -> Optional[Any]:
        """Load HyperparameterTuning from Training module (when available)."""
        try:
            from src.tuning.hyperparameter_tuning import HyperparameterTuning
            return HyperparameterTuning
        except ImportError:
            logger.warning("Training module not yet available")
            return None


# Global instance
loader = ModuleLoader()
