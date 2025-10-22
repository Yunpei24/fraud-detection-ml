"""
DAG 02: Drift Monitoring Pipeline
Runs hourly to detect data, target, and concept drift

Priority: #1 (MOST CRITICAL)
"""
from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator, BranchPythonOperator
from airflow.utils.dates import days_ago
import sys
from pathlib import Path

# Add config and plugins to path
AIRFLOW_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(AIRFLOW_ROOT / "config"))
sys.path.insert(0, str(AIRFLOW_ROOT / "plugins"))

from settings import settings
from module_loader import loader
from helpers import check_drift_severity, should_trigger_retraining

default_args = {
    'owner': 'ml-team',
    'depends_on_past': False,
    'email': settings.alert_email_recipients,
    'email_on_failure': True,
    'email_on_retry': False,
    'retries': 2,
    'retry_delay': timedelta(minutes=5)
}


def run_drift_monitoring(**context):
    """Execute drift monitoring from drift module"""
    from drift.src.pipelines.hourly_monitoring import run_hourly_monitoring
    from drift.src.config.settings import Settings
    
    drift_settings = Settings()
    result = run_hourly_monitoring(settings=drift_settings)
    
    # Store result in XCom for downstream tasks
    return result


def parse_drift_results(**context):
    """Analyze drift results and decide actions"""
    ti = context['task_instance']
    drift_result = ti.xcom_pull(task_ids='run_drift_monitoring')
    
    if not drift_result or drift_result.get('status') != 'success':
        return {'action': 'none', 'severity': 'info', 'reason': 'no_data_or_failed'}
    
    drift_data = drift_result.get('drift_results', {})
    
    # Check drift types
    data_drift = drift_data.get('data_drift', {}).get('drift_detected', False)
    target_drift = drift_data.get('target_drift', {}).get('drift_detected', False)
    concept_drift = drift_data.get('concept_drift', {}).get('drift_detected', False)
    
    # Determine action
    if concept_drift:
        # Concept drift = performance dégradée
        avg_psi = drift_data.get('data_drift', {}).get('avg_psi', 0)
        return {
            'action': 'trigger_retraining',
            'severity': 'critical',
            'drift_type': 'concept_drift',
            'avg_psi': avg_psi
        }
    elif data_drift:
        # Data drift = distribution changée
        avg_psi = drift_data.get('data_drift', {}).get('avg_psi', 0)
        severity = 'high' if avg_psi > 0.5 else 'medium'
        return {
            'action': 'trigger_retraining',
            'severity': severity,
            'drift_type': 'data_drift',
            'avg_psi': avg_psi
        }
    elif target_drift:
        # Target drift
        return {
            'action': 'alert_only',
            'severity': 'medium',
            'drift_type': 'target_drift'
        }
    else:
        return {'action': 'none', 'severity': 'info'}


def decide_next_step(**context):
    """Branch decision based on drift results"""
    ti = context['task_instance']
    decision = ti.xcom_pull(task_ids='parse_drift_results')
    
    action = decision.get('action', 'none')
    
    if action == 'trigger_retraining':
        return 'trigger_retraining_dag'
    elif action == 'alert_only':
        return 'send_drift_alert'
    else:
        return 'no_action'


def trigger_training_dag(**context):
    """Trigger training DAG if drift critical"""
    from airflow.api.common.trigger_dag import trigger_dag as airflow_trigger_dag
    
    ti = context['task_instance']
    decision = ti.xcom_pull(task_ids='parse_drift_results')
    
    dag_run = airflow_trigger_dag(
        dag_id='01_training_pipeline',
        run_id=f"drift_triggered_{context['ds']}",
        conf={
            'triggered_by': 'drift_detection',
            'severity': decision['severity'],
            'drift_type': decision.get('drift_type', 'unknown')
        }
    )
    
    # Save to retraining_triggers table
    import sqlalchemy as sa
    engine = sa.create_engine(settings.fraud_database_url)
    
    query = """
        INSERT INTO retraining_triggers 
        (trigger_reason, drift_type, drift_severity, airflow_dag_id, airflow_run_id, status)
        VALUES (%s, %s, %s, %s, %s, %s)
    """
    
    with engine.connect() as conn:
        conn.execute(sa.text(query), (
            f"Drift detected: {decision['severity']}",
            decision.get('drift_type', 'unknown'),
            decision['severity'],
            '01_training_pipeline',
            dag_run.run_id,
            'pending'
        ))
        conn.commit()
    
    return {'dag_run_id': dag_run.run_id, 'status': 'triggered'}


def send_drift_alert(**context):
    """Send alert for drift detection"""
    from plugins.operators.alert_operator import FraudDetectionAlertOperator
    
    ti = context['task_instance']
    decision = ti.xcom_pull(task_ids='parse_drift_results')
    
    alert_op = FraudDetectionAlertOperator(
        task_id='drift_alert_internal',
        alert_type='drift_detection',
        severity=decision['severity'].upper(),
        message=f"Drift detected: {decision.get('drift_type', 'unknown')}",
        details=decision
    )
    
    return alert_op.execute(context)


def no_action(**context):
    """No action needed - drift within acceptable thresholds"""
    return {'status': 'no_drift_detected', 'message': 'All metrics within acceptable range'}


def save_drift_metrics(**context):
    """Save drift metrics to database"""
    import sqlalchemy as sa
    
    ti = context['task_instance']
    drift_result = ti.xcom_pull(task_ids='run_drift_monitoring')
    
    if not drift_result:
        return {'status': 'no_data'}
    
    drift_data = drift_result.get('drift_results', {})
    engine = sa.create_engine(settings.fraud_database_url)
    
    # Save data drift metrics
    if 'data_drift' in drift_data:
        data_drift = drift_data['data_drift']
        query = """
            INSERT INTO drift_metrics 
            (metric_type, metric_name, metric_value, threshold, threshold_exceeded, severity)
            VALUES (%s, %s, %s, %s, %s, %s)
        """
        with engine.connect() as conn:
            conn.execute(sa.text(query), (
                'data_drift',
                'avg_psi',
                data_drift.get('avg_psi', 0),
                settings.data_drift_threshold,
                data_drift.get('drift_detected', False),
                'HIGH' if data_drift.get('avg_psi', 0) > 0.5 else 'MEDIUM'
            ))
            conn.commit()
    
    # Save concept drift metrics
    if 'concept_drift' in drift_data:
        concept_drift = drift_data['concept_drift']
        query = """
            INSERT INTO drift_metrics 
            (metric_type, metric_name, metric_value, threshold, threshold_exceeded, severity)
            VALUES (%s, %s, %s, %s, %s, %s)
        """
        with engine.connect() as conn:
            conn.execute(sa.text(query), (
                'concept_drift',
                'recall_drop',
                concept_drift.get('current_metrics', {}).get('recall', 0),
                settings.concept_drift_threshold,
                concept_drift.get('drift_detected', False),
                'CRITICAL' if concept_drift.get('drift_detected') else 'LOW'
            ))
            conn.commit()
    
    return {'status': 'metrics_saved', 'count': len(drift_data)}


# Define DAG
with DAG(
    '02_drift_monitoring',
    default_args=default_args,
    description='Hourly drift detection (Data, Target, Concept)',
    schedule_interval='0 * * * *',  # Every hour at minute 0
    start_date=days_ago(1),
    catchup=False,
    tags=['monitoring', 'drift', 'critical']
) as dag:
    
    # Task 1: Run drift monitoring
    run_drift = PythonOperator(
        task_id='run_drift_monitoring',
        python_callable=run_drift_monitoring
    )
    
    # Task 2: Parse results
    parse_results = PythonOperator(
        task_id='parse_drift_results',
        python_callable=parse_drift_results
    )
    
    # Task 3: Branch decision
    branch_decision = BranchPythonOperator(
        task_id='decide_next_step',
        python_callable=decide_next_step
    )
    
    # Task 4a: Trigger retraining
    trigger_retrain = PythonOperator(
        task_id='trigger_retraining_dag',
        python_callable=trigger_training_dag
    )
    
    # Task 4b: Send alert only
    alert_drift = PythonOperator(
        task_id='send_drift_alert',
        python_callable=send_drift_alert
    )
    
    # Task 4c: No action
    skip_action = PythonOperator(
        task_id='no_action',
        python_callable=no_action
    )
    
    # Task 5: Save metrics (runs regardless of branch)
    save_metrics = PythonOperator(
        task_id='save_drift_metrics',
        python_callable=save_drift_metrics,
        trigger_rule='all_done'  # Run even if upstream tasks fail
    )
    
    # Define task dependencies
    run_drift >> parse_results >> branch_decision
    branch_decision >> [trigger_retrain, alert_drift, skip_action]
    [trigger_retrain, alert_drift, skip_action] >> save_metrics
