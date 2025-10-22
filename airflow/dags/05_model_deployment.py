"""
DAG 05: Model Deployment Pipeline
Déploiement progressif (canary) du modèle en production
Utilise module_loader pour paths et config centralisée

Schedule: Manual trigger
"""
from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator, BranchPythonOperator
from airflow.operators.bash import BashOperator
from airflow.utils.dates import days_ago
from airflow.utils.trigger_rule import TriggerRule
import sys
from pathlib import Path

# Add config and plugins to path
AIRFLOW_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(AIRFLOW_ROOT / "config"))
sys.path.insert(0, str(AIRFLOW_ROOT / "plugins"))

from settings import settings
from module_loader import loader
from hooks.postgres_hook import FraudPostgresHook

default_args = {
    'owner': 'ml-team',
    'depends_on_past': False,
    'email': settings.alert_email_recipients,
    'email_on_failure': True,
    'retries': 1,
    'retry_delay': timedelta(minutes=10)
}


def validate_model_for_deployment(**context):
    """Valide que le modèle peut être déployé"""
    from plugins.hooks.mlflow_hook import MLflowHook
    
    hook = MLflowHook()
    
    # Get staging model
    model_version = hook.get_latest_model_version(
        settings.mlflow_model_name,
        stage='staging'
    )
    
    if not model_version:
        raise ValueError("No model in staging stage")
    
    # Récupérer métriques
    metrics = hook.get_model_metrics(model_version.run_id)
    
    # Validation
    min_recall = 0.80
    min_precision = 0.75
    
    recall = metrics.get('recall', 0)
    precision = metrics.get('precision', 0)
    
    if recall < min_recall:
        raise ValueError(f"Model recall {recall:.3f} below threshold {min_recall}")
    
    if precision < min_precision:
        raise ValueError(f"Model precision {precision:.3f} below threshold {min_precision}")
    
    return {
        'model_version': model_version.version,
        'model_uri': model_version.source,
        'recall': recall,
        'precision': precision,
        'validated': True
    }


def build_docker_image(**context):
    """Build Docker image avec nouveau modèle using API module path"""
    ti = context['task_instance']
    model_info = ti.xcom_pull(task_ids='validate_model')
    
    model_version = model_info['model_version']
    
    # Get API module path from loader
    api_path = loader.api_path
    
    # Build command using absolute path
    build_cmd = f"""
    cd {api_path} && \
    docker build \
        --build-arg MODEL_VERSION={model_version} \
        --build-arg MLFLOW_TRACKING_URI={settings.mlflow_tracking_uri} \
        -t {settings.azure_acr_login_server}/fraud-detection-api:{model_version} \
        -t {settings.azure_acr_login_server}/fraud-detection-api:latest \
        .
    """
    
    return {
        'image_tag': f"{settings.azure_acr_login_server}/fraud-detection-api:{model_version}",
        'build_command': build_cmd,
        'api_path': str(api_path)
    }


def push_to_acr(**context):
    """Push image to Azure Container Registry using centralized config"""
    ti = context['task_instance']
    build_info = ti.xcom_pull(task_ids='build_docker_image')
    
    image_tag = build_info['image_tag']
    
    # Login to ACR using centralized settings
    login_cmd = f"""
    az acr login --name {settings.azure_acr_name}
    """
    
    # Push image
    push_cmd = f"""
    docker push {image_tag} && \
    docker push {settings.azure_acr_login_server}/fraud-detection-api:latest
    """
    
    return {
        'image_tag': image_tag,
        'registry': settings.azure_acr_login_server,
        'pushed_at': datetime.now().isoformat()
    }


def deploy_canary_5_percent(**context):
    """Deploy canary 5% using FraudPostgresHook"""
    ti = context['task_instance']
    push_info = ti.xcom_pull(task_ids='push_to_acr')
    
    image_tag = push_info['image_tag']
    
    # Use hook instead of direct SQLAlchemy
    hook = FraudPostgresHook()
    
    # Update deployment config
    query = """
        INSERT INTO deployment_config (model_version, image_tag, traffic_percentage, status)
        VALUES (%s, %s, %s, %s)
        ON CONFLICT (model_version) 
        DO UPDATE SET 
            traffic_percentage = EXCLUDED.traffic_percentage,
            status = EXCLUDED.status,
            updated_at = NOW()
    """
    
    model_version = push_info['image_tag'].split(':')[-1]
    hook.execute_query(query, (model_version, image_tag, 5, 'canary'))
    
    return {
        'deployment_stage': 'canary_5',
        'traffic_percentage': 5,
        'start_time': datetime.now().isoformat()
    }


def monitor_canary_5_percent(**context):
    """Monitor canary 5% using FraudPostgresHook with centralized thresholds"""
    import time
    
    ti = context['task_instance']
    deployment = ti.xcom_pull(task_ids='deploy_canary_5_percent')
    
    # Extract model version from deployment stage info
    deployment_stage = deployment.get('deployment_stage', 'canary_5')
    
    # Use hook instead of direct SQLAlchemy
    hook = FraudPostgresHook()
    
    # Monitor for monitoring window (from settings)
    monitoring_minutes = 30
    check_interval_seconds = 300  # 5 minutes
    num_checks = monitoring_minutes // (check_interval_seconds // 60)
    
    print(f"Monitoring canary deployment for {monitoring_minutes} minutes...")
    
    # Check metrics at regular intervals
    for i in range(num_checks):
        time.sleep(check_interval_seconds)
        
        # Check error rate
        query = """
            SELECT 
                COUNT(*) as total,
                COUNT(CASE WHEN error IS NOT NULL THEN 1 END) as errors
            FROM predictions
            WHERE created_at >= NOW() - INTERVAL '5 minutes'
        """
        
        result = hook.fetch_one(query)
        
        total = result[0]
        errors = result[1]
        
        error_rate = errors / total if total > 0 else 0
        
        print(f"Check {i+1}/{num_checks}: {total} predictions, error_rate={error_rate:.3f}")
        
        # Use centralized error rate threshold from settings
        max_error_rate = settings.max_error_rate if hasattr(settings, 'max_error_rate') else 0.05
        
        if error_rate > max_error_rate:
            raise ValueError(
                f"High error rate detected: {error_rate:.3f} > {max_error_rate}. "
                f"Rolling back deployment."
            )
    
    return {
        'monitoring_duration_minutes': monitoring_minutes,
        'status': 'healthy',
        'decision': 'promote_to_25_percent'
    }


def decide_promotion_25(**context):
    """Décider si promouvoir à 25%"""
    ti = context['task_instance']
    monitor_result = ti.xcom_pull(task_ids='monitor_canary_5_percent')
    
    if monitor_result['status'] == 'healthy':
        return 'deploy_canary_25_percent'
    else:
        return 'rollback_deployment'


def deploy_canary_25_percent(**context):
    """Déploie 25% du traffic"""
    ti = context['task_instance']
    deployment = ti.xcom_pull(task_ids='deploy_canary_5_percent')
    
    model_version = deployment['model_version']
    
    # Update load balancer pour 25%
    print(f"Promoting v{model_version} to 25% traffic...")
    
    return {
        'deployment': 'canary_25_percent',
        'model_version': model_version,
        'traffic_percentage': 25,
        'status': 'deployed'
    }


def monitor_canary_25_percent(**context):
    """Monitor canary 25% using FraudPostgresHook with centralized thresholds"""
    import time
    
    ti = context['task_instance']
    deployment = ti.xcom_pull(task_ids='deploy_canary_25_percent')
    
    print(f"Monitoring 25% canary for 1 hour...")
    
    # Use hook instead of direct SQLAlchemy
    hook = FraudPostgresHook()
    
    # Monitor for 1 hour
    monitoring_minutes = 60
    check_interval_seconds = 300  # 5 minutes
    num_checks = monitoring_minutes // (check_interval_seconds // 60)
    
    for i in range(num_checks):
        time.sleep(check_interval_seconds)
        
        query = """
            SELECT 
                COUNT(*) as total,
                AVG(probability) as avg_confidence,
                COUNT(CASE WHEN error IS NOT NULL THEN 1 END) as errors
            FROM predictions
            WHERE created_at >= NOW() - INTERVAL '5 minutes'
        """
        
        result = hook.fetch_one(query)
        
        total = result[0]
        errors = result[2] or 0
        error_rate = errors / total if total > 0 else 0
        
        print(f"Check {i+1}/{num_checks}: {total} predictions, error_rate={error_rate:.3f}")
        
        # Use centralized error rate threshold from settings
        max_error_rate = settings.max_error_rate if hasattr(settings, 'max_error_rate') else 0.05
        
        if error_rate > max_error_rate:
            raise ValueError(f"High error rate: {error_rate:.3f}")
    
    return {
        'monitoring_duration_minutes': monitoring_minutes,
        'status': 'healthy',
        'decision': 'promote_to_100_percent'
    }


def decide_promotion_100(**context):
    """Décider si promouvoir à 100%"""
    ti = context['task_instance']
    monitor_result = ti.xcom_pull(task_ids='monitor_canary_25_percent')
    
    if monitor_result['status'] == 'healthy':
        return 'deploy_full_production'
    else:
        return 'rollback_deployment'


def deploy_full_production(**context):
    """Déploie 100% du traffic (production) using hooks"""
    ti = context['task_instance']
    model_info = ti.xcom_pull(task_ids='validate_model')
    
    model_version = model_info['model_version']
    
    # Promouvoir dans MLflow using hook
    from plugins.hooks.mlflow_hook import MLflowHook
    mlflow_hook = MLflowHook()
    
    mlflow_hook.transition_model_stage(
        model_name=settings.mlflow_model_name,
        version=model_version,
        stage='production',
        archive_existing=True
    )
    
    # Update database using FraudPostgresHook
    db_hook = FraudPostgresHook()
    
    # Retire old production models
    query1 = """
        UPDATE model_versions 
        SET is_production = false, retired_at = NOW()
        WHERE is_production = true
    """
    
    # Mark new model as production
    query2 = """
        UPDATE model_versions
        SET is_production = true, deployed_at = NOW()
        WHERE version = %s
    """
    
    db_hook.execute_query(query1)
    db_hook.execute_query(query2, (f"v{model_version}",))
    
    print(f"✅ Model v{model_version} deployed to 100% production")
    
    return {
        'deployment': 'full_production',
        'model_version': model_version,
        'traffic_percentage': 100,
        'status': 'completed'
    }


def rollback_deployment(**context):
    """Rollback en cas de problème"""
    ti = context['task_instance']
    deployment = ti.xcom_pull(task_ids='deploy_canary_5_percent')
    
    model_version = deployment['model_version']
    
    print(f"🔴 Rolling back deployment v{model_version}")
    
    # Arrêter containers canary
    rollback_cmd = f"""
    az container delete \
        --resource-group fraud-detection-rg \
        --name fraud-api-canary-v{model_version} \
        --yes
    """
    
    # Alert équipe
    from plugins.operators.alert_operator import FraudDetectionAlertOperator
    
    alert_op = FraudDetectionAlertOperator(
        task_id='rollback_alert',
        alert_type='deployment_failed',
        severity='CRITICAL',
        message=f"Model deployment v{model_version} rolled back due to errors",
        details=deployment
    )
    
    alert_op.execute(context)
    
    return {
        'status': 'rolled_back',
        'model_version': model_version
    }


# Define DAG
with DAG(
    '05_model_deployment',
    default_args=default_args,
    description='Canary deployment progressif (5% → 25% → 100%)',
    schedule_interval=None,  # Manual trigger
    start_date=days_ago(1),
    catchup=False,
    tags=['deployment', 'production', 'canary']
) as dag:
    
    # Validation
    validate = PythonOperator(
        task_id='validate_model',
        python_callable=validate_model_for_deployment
    )
    
    # Build
    build = PythonOperator(
        task_id='build_docker_image',
        python_callable=build_docker_image
    )
    
    # Push ACR
    push = PythonOperator(
        task_id='push_to_acr',
        python_callable=push_to_acr
    )
    
    # Canary 5%
    deploy_5 = PythonOperator(
        task_id='deploy_canary_5_percent',
        python_callable=deploy_canary_5_percent
    )
    
    monitor_5 = PythonOperator(
        task_id='monitor_canary_5_percent',
        python_callable=monitor_canary_5_percent
    )
    
    decide_25 = BranchPythonOperator(
        task_id='decide_promotion_25',
        python_callable=decide_promotion_25
    )
    
    # Canary 25%
    deploy_25 = PythonOperator(
        task_id='deploy_canary_25_percent',
        python_callable=deploy_canary_25_percent
    )
    
    monitor_25 = PythonOperator(
        task_id='monitor_canary_25_percent',
        python_callable=monitor_canary_25_percent
    )
    
    decide_100 = BranchPythonOperator(
        task_id='decide_promotion_100',
        python_callable=decide_promotion_100
    )
    
    # Production 100%
    deploy_100 = PythonOperator(
        task_id='deploy_full_production',
        python_callable=deploy_full_production
    )
    
    # Rollback
    rollback = PythonOperator(
        task_id='rollback_deployment',
        python_callable=rollback_deployment,
        trigger_rule=TriggerRule.ONE_FAILED
    )
    
    # Dependencies
    validate >> build >> push >> deploy_5
    deploy_5 >> monitor_5 >> decide_25
    decide_25 >> [deploy_25, rollback]
    deploy_25 >> monitor_25 >> decide_100
    decide_100 >> [deploy_100, rollback]
