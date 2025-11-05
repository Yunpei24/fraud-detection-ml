"""
DAG 01: Model Training Pipeline (Refactored)
==================================================
Unified training pipeline that:
1. Loads data from PostgreSQL (training_transactions)
2. Performs preprocessing and feature engineering
3. Trains multiple models (XGBoost, RF, NN, IF) with SMOTE
4. Evaluates and registers best model in MLflow Registry

Triggers:
- Manual trigger
- Schedule: Every Saturday at 2 AM
- Drift detected (via sensor)
- Performance degradation (via sensor)
"""

from __future__ import annotations

import json
from datetime import timedelta

import pendulum
from airflow.hooks.postgres_hook import PostgresHook
from airflow.models.dag import DAG
from airflow.operators.empty import EmptyOperator
from airflow.operators.python import BranchPythonOperator, PythonOperator
from airflow.providers.docker.operators.docker import DockerOperator
from airflow.utils.trigger_rule import TriggerRule

# Import centralized configuration
from config.constants import (
    DOCKER_IMAGE_TRAINING,
    DOCKER_NETWORK,
    ENV_VARS,
    TABLE_NAMES,
)

# Default args
default_args = {
    "owner": "ml-team",
    "depends_on_past": False,
    "email": ["emmanuelyunpei@gmail.com"],  # Updated to your email
    "email_on_failure": True,
    "email_on_retry": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=10),
}


def check_training_conditions(**context):
    """
    Check if training should be triggered based on drift or performance.

    Returns:
        "run_training" if conditions met, "skip_training" otherwise
    """

    pg_hook = PostgresHook(postgres_conn_id="fraud_postgres")

    # Check 1: Drift detected in last 7 days
    drift_query = f"""
        SELECT EXISTS(
            SELECT 1 FROM {TABLE_NAMES['DRIFT_METRICS']} 
            WHERE threshold_exceeded = TRUE 
            AND severity IN ('HIGH', 'CRITICAL')
            AND timestamp > NOW() - INTERVAL '7 days'
        )
    """
    drift_detected = pg_hook.get_first(drift_query)[0]

    # Check 2: Performance degradation in last 7 days
    perf_query = f"""
        SELECT EXISTS(
            SELECT 1 FROM {TABLE_NAMES['PREDICTIONS']} 
            WHERE prediction_time > NOW() - INTERVAL '7 days'
            GROUP BY DATE(prediction_time)
            HAVING AVG(fraud_score) < 0.85
        )
    """
    perf_degraded = pg_hook.get_first(perf_query)[0]

    # Check 3: Last training was more than 7 days ago: column training_date
    last_training_query = f"""
        SELECT EXISTS(
            SELECT 1 FROM {TABLE_NAMES['MODEL_VERSIONS']} 
            WHERE training_date < NOW() - INTERVAL '7 days'
        )
    """
    recently_trained = pg_hook.get_first(last_training_query)[0]

    # Log conditions
    print(f"Training conditions:")
    print(f"   - Drift detected: {drift_detected}")
    print(f"   - Performance degraded: {perf_degraded}")
    print(f"   - Recently trained (< 7 days): {recently_trained}")

    # Decision logic
    if drift_detected or perf_degraded:
        print("Training triggered: Drift or performance issue detected")
        return "run_training"

    if not recently_trained:
        print("Training triggered: No recent training found")
        return "run_training"

    print("⏭Skipping training: All conditions normal")
    return "skip_training"


def send_training_notification(status: str, **context):
    """
    Send notification about training completion.

    Args:
        status: "success" or "failure"
    """

    ti = context["task_instance"]
    execution_date = context["execution_date"]

    # In production, this would send to Slack/Email
    # For now, just log

    if status == "success":
        print(f"Training pipeline completed successfully at {execution_date}")
        print(f"   Check MLflow: {ENV_VARS['MLFLOW_TRACKING_URI']}")
    else:
        print(f"Training pipeline failed at {execution_date}")
        print(f"   Check logs for details")


# Define the DAG
with DAG(
    dag_id="01_training_pipeline",
    default_args=default_args,
    description="Unified model training pipeline with drift detection and performance monitoring",
    schedule="0 2 * * 6",  # Every Saturday at 2 AM
    start_date=pendulum.datetime(2025, 1, 1, tz="UTC"),
    catchup=False,
    tags=["training", "mlflow", "production"],
    doc_md=__doc__,
) as dag:
    # Task 1: Check if training should run
    check_conditions = BranchPythonOperator(
        task_id="check_training_conditions",
        python_callable=check_training_conditions,
        provide_context=True,
    )

    # Task 2a: Skip training
    skip_training = EmptyOperator(
        task_id="skip_training",
        trigger_rule=TriggerRule.NONE_FAILED,
    )

    # Task 2b: Run unified training pipeline
    run_training = DockerOperator(
        task_id="run_training",
        image=DOCKER_IMAGE_TRAINING,
        command="python -m src.pipelines.training_pipeline",
        environment=ENV_VARS,
        docker_url="unix://var/run/docker.sock",
        network_mode=DOCKER_NETWORK,
        auto_remove=True,
        mount_tmp_dir=False,
        working_dir="/app",  # Specify working directory explicitly
        mounts=[
            {
                "target": "/app/data/raw/creditcard.csv",
                "source": "/Users/joshuajusteyunpeinikiema/Documents/MLOps/fraud-detection-project/fraud-detection-ml/creditcard.csv",
                "type": "bind",
                "read_only": True,
            },
            {
                "target": "/app/training/artifacts",
                "source": "fraud-detection-ml_training_artifacts",  # Named volume with project prefix
                "type": "volume",
                "read_only": False,
            },
            {
                "target": "/mlflow/artifacts",
                "source": "fraud-detection-ml_mlflow_artifacts",  # MLflow artifacts (shared with MLflow server)
                "type": "volume",
                "read_only": False,
            },
        ],
        trigger_rule=TriggerRule.NONE_FAILED,
        doc_md="""
        Runs the complete training pipeline:
        1. Load data from training_transactions (PostgreSQL)
        2. Feature engineering and preprocessing
        3. Train 4 models in parallel (XGBoost, RF, NN, IF)
        4. Evaluate on test set
        5. Register best model in MLflow (Production stage)
        """,
    )

    # Task 3: Notify success
    notify_success = PythonOperator(
        task_id="notify_success",
        python_callable=send_training_notification,
        op_kwargs={"status": "success"},
        trigger_rule=TriggerRule.ALL_SUCCESS,
    )

    # Task 4: Notify failure
    notify_failure = PythonOperator(
        task_id="notify_failure",
        python_callable=send_training_notification,
        op_kwargs={"status": "failure"},
        trigger_rule=TriggerRule.ONE_FAILED,
    )

    # Task 5: End
    end = EmptyOperator(
        task_id="end",
        trigger_rule=TriggerRule.NONE_FAILED_MIN_ONE_SUCCESS,
    )

    # Define task dependencies
    check_conditions >> [skip_training, run_training]
    skip_training >> end
    run_training >> [notify_success, notify_failure]
    [notify_success, notify_failure] >> end
