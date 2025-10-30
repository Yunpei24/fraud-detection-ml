"""
Unit tests for Airflow DAGs and configuration
"""

import sys
from pathlib import Path

import pytest

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))


class TestAirflowConfiguration:
    """Test airflow configuration and centralized settings integration"""

    def test_constants_import(self):
        """Test that constants can be imported"""
        from airflow.config.constants import ENV_VARS, TABLE_NAMES, THRESHOLDS

        assert isinstance(TABLE_NAMES, dict)
        assert "TRANSACTIONS" in TABLE_NAMES
        assert "PREDICTIONS" in TABLE_NAMES

        assert isinstance(ENV_VARS, dict)
        assert "POSTGRES_HOST" in ENV_VARS
        assert "MLFLOW_TRACKING_URI" in ENV_VARS

        assert isinstance(THRESHOLDS, dict)
        assert "MIN_RECALL" in THRESHOLDS
        assert THRESHOLDS["MIN_RECALL"] > 0

    def test_airflow_settings_import(self):
        """Test that airflow settings can be imported and use centralized config"""
        from airflow.config.settings import AirflowSettings

        settings = AirflowSettings()

        # Check that settings are loaded from centralized config
        assert hasattr(settings, "fraud_database_url")
        assert hasattr(settings, "mlflow_tracking_uri")
        assert hasattr(settings, "data_drift_threshold")

        # Check values are reasonable
        assert "postgresql" in settings.fraud_database_url
        assert (
            "localhost" in settings.fraud_database_url
            or "postgres" in settings.fraud_database_url
        )

    def test_dag_imports(self):
        """Test that DAG files exist and can be read"""
        import os

        # Check that key DAG files exist
        dag_files = [
            "01_training_pipeline.py",
            "02_drift_monitoring.py",
            "03_feedback_collection.py",
            "04_data_quality.py",
            "05_model_deployment_canary.py",
            "06_model_performance_tracking.py",
        ]

        dags_dir = os.path.join(os.path.dirname(__file__), "..", "dags")

        for dag_file in dag_files:
            dag_path = os.path.join(dags_dir, dag_file)
            assert os.path.exists(dag_path), f"DAG file {dag_file} not found"

            # Check that file contains expected content
            with open(dag_path, "r") as f:
                content = f.read()
                # Most DAGs should import centralized constants, but some may define their own
                has_centralized_config = "from config.constants import" in content
                has_some_config = (
                    "TABLE_NAMES" in content
                    or "ENV_VARS" in content
                    or "THRESHOLDS" in content
                    or "SCHEDULES" in content
                    or "DOCKER_NETWORK" in content
                )

                # Either uses centralized config OR has some configuration
                assert (
                    has_centralized_config or has_some_config
                ), f"{dag_file} has no configuration"


class TestDAGTrainingPipeline:
    """Test the training pipeline DAG specifically"""

    def test_training_dag_file_exists(self):
        """Test that training DAG file exists and uses centralized config"""
        import os

        dag_path = os.path.join(
            os.path.dirname(__file__), "..", "dags", "01_training_pipeline.py"
        )
        assert os.path.exists(dag_path), "Training pipeline DAG file not found"

        with open(dag_path, "r") as f:
            content = f.read()
            assert "from config.constants import" in content
            assert "TABLE_NAMES" in content
            assert "ENV_VARS" in content
            assert "DOCKER_NETWORK" in content
            assert 'dag_id="01_training_pipeline"' in content

    def test_training_dag_constants_usage(self):
        """Test that training DAG uses centralized constants"""
        import os

        dag_path = os.path.join(
            os.path.dirname(__file__), "..", "dags", "01_training_pipeline.py"
        )
        with open(dag_path, "r") as f:
            content = f.read()
            assert "TABLE_NAMES[" in content  # Uses table names
            assert "ENV_VARS[" in content  # Uses environment vars


class TestDAGDriftMonitoring:
    """Test the drift monitoring DAG"""

    def test_drift_dag_file_exists(self):
        """Test that drift monitoring DAG file exists"""
        import os

        dag_path = os.path.join(
            os.path.dirname(__file__), "..", "dags", "02_drift_monitoring.py"
        )
        assert os.path.exists(dag_path), "Drift monitoring DAG file not found"

        with open(dag_path, "r") as f:
            content = f.read()
            assert "from config.constants import" in content
            assert "drift" in content.lower() or "monitoring" in content.lower()


class TestDAGDataQuality:
    """Test the data quality DAG"""

    def test_data_quality_dag_file_exists(self):
        """Test that data quality DAG file exists"""
        import os

        dag_path = os.path.join(
            os.path.dirname(__file__), "..", "dags", "04_data_quality.py"
        )
        assert os.path.exists(dag_path), "Data quality DAG file not found"

        with open(dag_path, "r") as f:
            content = f.read()
            assert "from config.constants import" in content
            assert "data" in content.lower() or "quality" in content.lower()


class TestDAGModelDeployment:
    """Test the model deployment DAG"""

    def test_deployment_dag_file_exists(self):
        """Test that model deployment DAG file exists"""
        import os

        dag_path = os.path.join(
            os.path.dirname(__file__), "..", "dags", "05_model_deployment_canary.py"
        )
        assert os.path.exists(dag_path), "Model deployment DAG file not found"

        with open(dag_path, "r") as f:
            content = f.read()
            # This DAG defines its own config but still uses centralized patterns
            assert "deployment" in content.lower() or "model" in content.lower()
            assert "canary" in content.lower()  # Should mention canary deployment


class TestDAGPerformanceTracking:
    """Test the performance tracking DAG"""

    def test_performance_dag_file_exists(self):
        """Test that performance tracking DAG file exists"""
        import os

        dag_path = os.path.join(
            os.path.dirname(__file__), "..", "dags", "06_model_performance_tracking.py"
        )
        assert os.path.exists(dag_path), "Performance tracking DAG file not found"

        with open(dag_path, "r") as f:
            content = f.read()
            assert "from config.constants import" in content
            assert "performance" in content.lower() or "tracking" in content.lower()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
