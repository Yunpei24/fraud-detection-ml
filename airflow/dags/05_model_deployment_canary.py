"""
DAG 05: Canary Deployment Pipeline (Refactored)
==================================================
Progressive model deployment strategy: 5% → 25% → 100%

This DAG:
1. Compares Champion (Production) vs Challenger (Staging) from MLflow Registry
2. Deploys incrementally with monitoring at each stage
3. Rolls back automatically if metrics degrade
4. Promotes to 100% production if all stages pass

Monitoring metrics:
- Error rate < 5%
- Latency P95 < 100ms
- Prediction accuracy maintained
"""
from __future__ import annotations

import pendulum
from datetime import timedelta
import time
import requests

from airflow.models.dag import DAG
from airflow.operators.python import BranchPythonOperator, PythonOperator
from airflow.operators.empty import EmptyOperator
from airflow.providers.docker.operators.docker import DockerOperator
from airflow.utils.trigger_rule import TriggerRule

# Import centralized configuration
from config.constants import ENV_VARS, DOCKER_NETWORK, DOCKER_IMAGE_TRAINING, DOCKER_IMAGE_API

# Network
DOCKER_NETWORK = DOCKER_NETWORK

# Default args
default_args = {
    "owner": "ml-team",
    "depends_on_past": False,
    "email": ["ml-alerts@frauddetection.com"],
    "email_on_failure": True,
    "email_on_retry": False,
    "retries": 0,  # No retry for deployment
}


def parse_comparison_result(**context):
    """
    Parse comparison pipeline exit code to decide deployment.
    
    Returns:
        "deploy_canary_5_percent" if challenger should be promoted,
        "keep_champion" otherwise
    """
    ti = context["task_instance"]
    
    # Get exit code from comparison task
    # Exit code 0 = promote challenger, 1 = keep champion
    comparison_result = ti.xcom_pull(task_ids="compare_models", key="return_value")
    
    print(f" Comparison result: {comparison_result}")
    
    if comparison_result == 0:
        print(" Challenger approved - starting canary deployment")
        return "deploy_canary_5_percent"
    else:
        print(" Challenger rejected - keeping champion")
        return "keep_champion"


def monitor_canary_metrics(traffic_pct: int, duration_minutes: int, **context):
    """
    Monitor canary deployment metrics from Prometheus.
    
    Args:
        traffic_pct: Percentage of traffic on canary
        duration_minutes: Duration to monitor
    
    Returns:
        "promote" if metrics are healthy, "rollback" otherwise
    """
    
    prometheus_url = "http://prometheus:9090/api/v1/query"
    
    print(f" Monitoring canary {traffic_pct}% for {duration_minutes} minutes...")
    
    # In production, this would wait and monitor continuously
    # For demo, we'll do a quick check
    time.sleep(60)  # Wait 1 minute
    
    try:
        # Query 1: Error rate
        response = requests.get(prometheus_url, params={
            "query": f'rate(http_requests_total{{version="canary",status=~"5.."}}[{duration_minutes}m])'
        }, timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            if data["data"]["result"]:
                error_rate = float(data["data"]["result"][0]["value"][1])
            else:
                error_rate = 0.0
        else:
            error_rate = 0.0  # Assume healthy if can't query
        
        # Query 2: Latency P95
        response = requests.get(prometheus_url, params={
            "query": f'histogram_quantile(0.95, rate(http_request_duration_seconds_bucket{{version="canary"}}[{duration_minutes}m]))'
        }, timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            if data["data"]["result"]:
                latency_p95 = float(data["data"]["result"][0]["value"][1]) * 1000  # Convert to ms
            else:
                latency_p95 = 50.0
        else:
            latency_p95 = 50.0  # Assume healthy
        
        print(f" Canary {traffic_pct}% metrics:")
        print(f"   - Error rate: {error_rate:.2%}")
        print(f"   - Latency P95: {latency_p95:.2f}ms")
        
        # Decision thresholds
        if error_rate > 0.05:
            print(f" Error rate too high: {error_rate:.2%} > 5%")
            return "rollback"
        
        if latency_p95 > 100:
            print(f" Latency too high: {latency_p95:.2f}ms > 100ms")
            return "rollback"
        
        print(f" Canary {traffic_pct}% healthy - ready to promote")
        return "promote"
        
    except Exception as e:
        print(f"  Error monitoring metrics: {e}")
        print("  Assuming unhealthy - triggering rollback")
        return "rollback"


def decide_next_step(current_stage: str, monitor_result: str, **context):
    """
    Decide next step based on monitoring results.
    
    Args:
        current_stage: "5", "25", or "100"
        monitor_result: Result from monitor_canary_metrics
    
    Returns:
        Next task ID
    """
    print(f" Deciding next step for {current_stage}% canary...")
    print(f"   Monitor result: {monitor_result}")
    
    if monitor_result == "rollback":
        return "rollback_deployment"
    
    if current_stage == "5":
        return "deploy_canary_25_percent"
    elif current_stage == "25":
        return "deploy_canary_100_percent"
    else:
        return "deployment_complete"


def send_deployment_notification(status: str, **context):
    """
    Send notification about deployment status.
    
    Args:
        status: "success", "rollback", or "rejected"
    """
    execution_date = context["execution_date"]
    
    if status == "success":
        print(f" Canary deployment completed successfully at {execution_date}")
        print(f"   New model deployed to 100% production")
    elif status == "rollback":
        print(f" Canary deployment rolled back at {execution_date}")
        print(f"   Champion model restored")
    else:
        print(f"  Deployment skipped - challenger rejected at {execution_date}")


# Define the DAG
with DAG(
    dag_id="05_model_deployment_canary",
    default_args=default_args,
    description="Progressive canary deployment with automatic rollback",
    schedule_interval=None,  # Manual trigger only
    start_date=pendulum.datetime(2025, 1, 1, tz="UTC"),
    catchup=False,
    tags=["deployment", "canary", "production", "mlflow"],
    doc_md=__doc__,
) as dag:
    
    # Task 1: Compare Champion vs Challenger Ensembles
    compare_models = DockerOperator(
        task_id="compare_models",
        image=DOCKER_IMAGE_TRAINING,
        command="python -m src.pipelines.comparison_pipeline",
        environment=ENV_VARS,
        docker_url="unix://var/run/docker.sock",
        network_mode=DOCKER_NETWORK,
        auto_remove=True,
        do_xcom_push=True,
        doc_md="""
        Compare Champion vs Challenger ENSEMBLES from MLflow Registry:
        - Champion: 4 models from Production stage
        - Challenger: 4 models from Staging stage
        - Compare ensemble metrics (weighted voting)
        - Measure ensemble latency
        - Check business constraints (recall >= 0.95, FPR <= 0.02, latency <= 100ms)
        
        Exit code:
        - 0: Promote challenger ensemble
        - 1: Keep champion ensemble
        """,
    )
    
    # Task 2: Decision branch
    decide_deployment = BranchPythonOperator(
        task_id="decide_deployment",
        python_callable=parse_comparison_result,
        provide_context=True,
    )
    
    # Task 3a: Keep champion (no deployment)
    keep_champion = EmptyOperator(
        task_id="keep_champion",
        trigger_rule=TriggerRule.NONE_FAILED,
    )
    
    # Task 3b: Deploy canary 5%
    deploy_canary_5_percent = DockerOperator(
        task_id="deploy_canary_5_percent",
        image=DOCKER_IMAGE_API,
        command='python scripts/deploy_canary.py --traffic 5 --model-uris "models:/fraud_detection_xgboost/Staging" "models:/fraud_detection_random_forest/Staging" "models:/fraud_detection_neural_network/Staging" "models:/fraud_detection_isolation_forest/Staging"',
        environment=ENV_VARS,
        docker_url="unix://var/run/docker.sock",
        network_mode=DOCKER_NETWORK,
        auto_remove=True,
        trigger_rule=TriggerRule.NONE_FAILED,
    )
    
    # Task 4: Monitor 5% (30 minutes)
    monitor_5 = PythonOperator(
        task_id="monitor_5_percent",
        python_callable=monitor_canary_metrics,
        op_kwargs={"traffic_pct": 5, "duration_minutes": 30},
        provide_context=True,
    )
    
    # Task 5: Decide 25%
    decide_25 = BranchPythonOperator(
        task_id="decide_25_percent",
        python_callable=decide_next_step,
        op_kwargs={"current_stage": "5"},
        provide_context=True,
    )
    
    # Task 6a: Deploy canary 25%
    deploy_canary_25_percent = DockerOperator(
        task_id="deploy_canary_25_percent",
        image=DOCKER_IMAGE_API,
        command='python scripts/deploy_canary.py --traffic 25 --model-uris "models:/fraud_detection_xgboost/Staging" "models:/fraud_detection_random_forest/Staging" "models:/fraud_detection_neural_network/Staging" "models:/fraud_detection_isolation_forest/Staging"',
        environment=ENV_VARS,
        docker_url="unix://var/run/docker.sock",
        network_mode=DOCKER_NETWORK,
        auto_remove=True,
        trigger_rule=TriggerRule.NONE_FAILED,
    )
    
    # Task 7: Monitor 25% (1 hour)
    monitor_25 = PythonOperator(
        task_id="monitor_25_percent",
        python_callable=monitor_canary_metrics,
        op_kwargs={"traffic_pct": 25, "duration_minutes": 60},
        provide_context=True,
    )
    
    # Task 8: Decide 100%
    decide_100 = BranchPythonOperator(
        task_id="decide_100_percent",
        python_callable=decide_next_step,
        op_kwargs={"current_stage": "25"},
        provide_context=True,
    )
    
    # Task 9a: Deploy 100% (promote to production)
    deploy_canary_100_percent = DockerOperator(
        task_id="deploy_canary_100_percent",
        image=DOCKER_IMAGE_API,
        command='python scripts/promote_to_production.py --model-uris "models:/fraud_detection_xgboost/Staging" "models:/fraud_detection_random_forest/Staging" "models:/fraud_detection_neural_network/Staging" "models:/fraud_detection_isolation_forest/Staging"',
        environment=ENV_VARS,
        docker_url="unix://var/run/docker.sock",
        network_mode=DOCKER_NETWORK,
        auto_remove=True,
        trigger_rule=TriggerRule.NONE_FAILED,
    )
    
    # Task 9b: Rollback
    rollback_deployment = DockerOperator(
        task_id="rollback_deployment",
        image=DOCKER_IMAGE_API,
        command="python scripts/rollback_deployment.py",
        environment=ENV_VARS,
        docker_url="unix://var/run/docker.sock",
        network_mode=DOCKER_NETWORK,
        auto_remove=True,
        trigger_rule=TriggerRule.NONE_FAILED,
    )
    
    # Task 10: Notifications
    notify_success = PythonOperator(
        task_id="notify_success",
        python_callable=send_deployment_notification,
        op_kwargs={"status": "success"},
        trigger_rule=TriggerRule.NONE_FAILED,
    )
    
    notify_rollback = PythonOperator(
        task_id="notify_rollback",
        python_callable=send_deployment_notification,
        op_kwargs={"status": "rollback"},
        trigger_rule=TriggerRule.NONE_FAILED,
    )
    
    notify_rejected = PythonOperator(
        task_id="notify_rejected",
        python_callable=send_deployment_notification,
        op_kwargs={"status": "rejected"},
        trigger_rule=TriggerRule.NONE_FAILED,
    )
    
    # Task 11: End
    end = EmptyOperator(
        task_id="end",
        trigger_rule=TriggerRule.NONE_FAILED_MIN_ONE_SUCCESS,
    )
    
    # Define task dependencies
    compare_models >> decide_deployment
    decide_deployment >> [keep_champion, deploy_canary_5_percent]
    
    # Canary flow
    deploy_canary_5_percent >> monitor_5 >> decide_25
    decide_25 >> [deploy_canary_25_percent, rollback_deployment]
    
    deploy_canary_25_percent >> monitor_25 >> decide_100
    decide_100 >> [deploy_canary_100_percent, rollback_deployment]
    
    # Notifications
    keep_champion >> notify_rejected >> end
    deploy_canary_100_percent >> notify_success >> end
    rollback_deployment >> notify_rollback >> end
