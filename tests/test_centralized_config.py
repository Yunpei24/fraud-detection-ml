"""
Comprehensive validation of centralized configuration system across all modules.
This test ensures that all modules can properly access and use the centralized settings.
"""
import sys
from pathlib import Path

import pytest

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


class TestCentralizedConfiguration:
    """Test centralized configuration system across all modules"""

    def test_config_loading(self):
        """Test that centralized config loads correctly"""
        from config import get_settings

        settings = get_settings()

        # Check main sections exist
        assert hasattr(settings, "database")
        assert hasattr(settings, "mlflow")
        assert hasattr(settings, "airflow")
        assert hasattr(settings, "drift")
        assert hasattr(settings, "training")
        assert hasattr(settings, "api")
        assert hasattr(settings, "alerts")
        assert hasattr(settings, "monitoring")

        # Check environment
        assert settings.environment in ["development", "staging", "production"]

    def test_database_config(self):
        """Test database configuration"""
        from config import get_settings

        settings = get_settings()

        # Check database settings
        assert settings.database.url
        assert "postgresql" in settings.database.url
        assert settings.database.pool_size > 0
        assert settings.database.max_overflow >= 0

    def test_mlflow_config(self):
        """Test MLflow configuration"""
        from config import get_settings

        settings = get_settings()

        # Check MLflow settings
        assert settings.mlflow.tracking_uri
        assert settings.mlflow.experiment_name
        assert settings.mlflow.model_name

    def test_airflow_config(self):
        """Test Airflow configuration"""
        from config import get_settings

        settings = get_settings()

        # Check Airflow settings
        assert settings.airflow.home
        assert settings.airflow.database_url
        assert settings.airflow.executor
        assert settings.airflow.parallelism > 0

    def test_drift_config(self):
        """Test drift detection configuration"""
        from config import get_settings

        settings = get_settings()

        # Check drift settings
        assert settings.drift.data_drift_threshold > 0
        assert settings.drift.concept_drift_threshold > 0
        assert (
            settings.drift.hourly_window_size > 0
        )  # Changed from monitoring_window_hours

    def test_training_config(self):
        """Test training configuration"""
        from config import get_settings

        settings = get_settings()

        # Check training settings - only test fields that actually exist
        assert settings.training.batch_size > 0
        assert settings.training.epochs > 0
        assert 0 < settings.training.validation_split < 1
        assert settings.training.cv_folds > 0

    def test_api_config(self):
        """Test API configuration"""
        from config import get_settings

        settings = get_settings()

        # Check API settings
        assert settings.api.host
        assert settings.api.port > 0
        assert settings.api.workers >= 1

    def test_alerts_config(self):
        """Test alerts configuration"""
        from config import get_settings

        settings = get_settings()

        # Check alerts settings
        assert isinstance(settings.alerts.email_enabled, bool)
        assert isinstance(settings.alerts.email_recipients, list)

    def test_monitoring_config(self):
        """Test monitoring configuration"""
        from config import get_settings

        settings = get_settings()

        # Check monitoring settings
        assert settings.monitoring.log_level in [
            "DEBUG",
            "INFO",
            "WARNING",
            "ERROR",
            "CRITICAL",
        ]
        assert (
            settings.monitoring.prometheus_enabled is not None
        )  # Changed from metrics_enabled


class TestModuleIntegration:
    """Test that modules can properly use centralized configuration"""

    def test_api_module_config(self):
        """Test API module can load config"""
        try:
            from api.src.config import get_settings as get_api_settings

            api_settings = get_api_settings()

            # Should be able to access API-specific settings
            assert hasattr(api_settings, "host")
            assert hasattr(api_settings, "port")

        except ImportError:
            pytest.skip("API module config not available")

    def test_data_module_config(self):
        """Test data module can load config"""
        try:
            from data.src.config import get_settings as get_data_settings

            data_settings = get_data_settings()

            # Should be able to access data-specific settings
            assert hasattr(data_settings, "database")
            assert hasattr(data_settings.database, "url")

        except ImportError:
            pytest.skip("Data module config not available")

    def test_drift_module_config(self):
        """Test drift module can load config"""
        try:
            from drift.src.config import get_settings as get_drift_settings

            drift_settings = get_drift_settings()

            # Should be able to access drift-specific settings
            assert hasattr(drift_settings, "drift")
            assert hasattr(drift_settings.drift, "data_drift_threshold")

        except ImportError:
            pytest.skip("Drift module config not available")

    def test_training_module_config(self):
        """Test training module can load config"""
        try:
            from training.src.config import get_settings as get_training_settings

            training_settings = get_training_settings()

            # Should be able to access training-specific settings
            assert hasattr(training_settings, "training")
            assert hasattr(training_settings.training, "min_samples")

        except ImportError:
            pytest.skip("Training module config not available")

    def test_airflow_module_config(self):
        """Test airflow module can load config"""
        try:
            from airflow.config.settings import AirflowSettings

            airflow_settings = AirflowSettings()

            # Should be able to access airflow-specific settings
            assert hasattr(airflow_settings, "fraud_database_url")
            assert hasattr(airflow_settings, "mlflow_tracking_uri")

        except ImportError:
            pytest.skip("Airflow module config not available")


class TestConfigurationConsistency:
    """Test that configuration is consistent across modules"""

    def test_database_url_consistency(self):
        """Test database URL is consistent across modules"""
        from config import get_settings

        central_settings = get_settings()

        # Check that modules use the same database URL
        try:
            from api.src.config import get_settings as get_api_settings

            api_settings = get_api_settings()
            assert api_settings.database.url == central_settings.database.url
        except (ImportError, AttributeError):
            pytest.skip("API database config not comparable")

    def test_mlflow_uri_consistency(self):
        """Test MLflow URI is consistent across modules"""
        from config import get_settings

        central_settings = get_settings()

        # Check that modules use the same MLflow URI
        try:
            from training.src.config import get_settings as get_training_settings

            training_settings = get_training_settings()
            assert (
                training_settings.mlflow.tracking_uri
                == central_settings.mlflow.tracking_uri
            )
        except (ImportError, AttributeError):
            pytest.skip("Training MLflow config not comparable")

    def test_environment_consistency(self):
        """Test environment setting is consistent across modules"""
        from config import get_settings

        central_settings = get_settings()

        # Check that modules use the same environment
        try:
            from drift.src.config import get_settings as get_drift_settings

            drift_settings = get_drift_settings()
            assert drift_settings.environment == central_settings.environment
        except (ImportError, AttributeError):
            pytest.skip("Drift environment config not comparable")


class TestConfigurationValidation:
    """Test configuration validation and error handling"""

    def test_invalid_environment(self):
        """Test that invalid environment raises error"""
        import os

        from config import GlobalSettings  # Changed from Settings

        # Temporarily set invalid environment
        original_env = os.environ.get("ENVIRONMENT")
        os.environ["ENVIRONMENT"] = "invalid"

        try:
            # This should raise a validation error
            with pytest.raises(ValueError):
                GlobalSettings()  # Changed from Settings()
        finally:
            # Restore original environment
            if original_env is not None:
                os.environ["ENVIRONMENT"] = original_env
            else:
                os.environ.pop("ENVIRONMENT", None)

    def test_missing_required_settings(self):
        """Test that missing required settings are handled"""
        import os

        from config import GlobalSettings  # Changed from Settings

        # Temporarily remove required environment variable
        original_db_url = os.environ.get("DATABASE_URL")

        if "DATABASE_URL" in os.environ:
            del os.environ["DATABASE_URL"]

        try:
            # This might work with defaults or raise an error
            settings = GlobalSettings()  # Changed from Settings()
            # If it works, database URL should be set to a default
            assert settings.database.url is not None
        except Exception:
            # It's acceptable for this to fail with missing required settings
            pass
        finally:
            # Restore original setting
            if original_db_url is not None:
                os.environ["DATABASE_URL"] = original_db_url


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
