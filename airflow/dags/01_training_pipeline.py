"""
DAG 01: Model Training Pipeline
Runs daily or triggered by drift detection

Priority: #2 (HIGH)
"""
from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator, BranchPythonOperator
from airflow.providers.databricks.operators.databricks import DatabricksSubmitRunOperator
from airflow.utils.dates import days_ago
import sys
from pathlib import Path

# Add config and plugins to path
AIRFLOW_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(AIRFLOW_ROOT / "config"))
sys.path.insert(0, str(AIRFLOW_ROOT / "plugins"))

from settings import settings
from module_loader import loader
from helpers import should_trigger_retraining, calculate_training_priority

default_args = {
    'owner': 'ml-team',
    'depends_on_past': False,
    'email': settings.alert_email_recipients,
    'email_on_failure': True,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=10)
}


def check_should_retrain(**context):
    """Intelligent decision: should we retrain?"""
    import sqlalchemy as sa
    
    # Check if triggered by drift (high priority)
    dag_run = context.get('dag_run')
    if dag_run and dag_run.conf.get('triggered_by') == 'drift_detection':
        return {'should_retrain': True, 'reason': 'drift_detected', 'priority': 'high'}
    
    # Check last training time
    engine = sa.create_engine(settings.fraud_database_url)
    
    query = """
        SELECT MAX(completed_at) as last_training
        FROM retraining_triggers
        WHERE status = 'completed'
    """
    
    with engine.connect() as conn:
        result = conn.execute(sa.text(query)).fetchone()
        last_training = result[0] if result else None
    
    # FIRST TIME TRAINING: Si jamais entrainé, toujours train
    if last_training is None:
        return {
            'should_retrain': True,
            'reason': 'initial_training (no previous training found)',
            'priority': 'high'
        }
    
    # If last training > cooldown period, allow retraining
    hours_since_training = (datetime.now() - last_training).total_seconds() / 3600
    
    if hours_since_training < settings.training_cooldown_hours:
        return {
            'should_retrain': False,
            'reason': f'cooldown_period ({hours_since_training:.1f}h < {settings.training_cooldown_hours}h)',
            'priority': 'low'
        }
    
    # Check training data volume
    query = """
        SELECT COUNT(*) as new_transactions
        FROM transactions
        WHERE created_at > (SELECT COALESCE(MAX(completed_at), NOW() - INTERVAL '7 days') FROM retraining_triggers WHERE status = 'completed')
    """
    
    with engine.connect() as conn:
        result = conn.execute(sa.text(query)).fetchone()
        new_transactions = result[0] if result else 0
    
    if new_transactions < settings.min_training_samples:
        return {
            'should_retrain': False,
            'reason': f'insufficient_data ({new_transactions} < {settings.min_training_samples})',
            'priority': 'low'
        }
    
    return {'should_retrain': True, 'reason': 'scheduled_daily', 'priority': 'normal'}


def decide_training_branch(**context):
    """Branch decision: train or skip"""
    ti = context['task_instance']
    decision = ti.xcom_pull(task_ids='check_should_retrain')
    
    if decision['should_retrain']:
        return 'load_training_data'
    else:
        return 'skip_training'


def load_training_data(**context):
    """Load and prepare training data from PostgreSQL"""
    import pandas as pd
    import sqlalchemy as sa
    
    engine = sa.create_engine(settings.fraud_database_url)
    
    # Load transactions with predictions (for validation)
    query = """
        SELECT 
            t.*,
            p.prediction,
            p.probability,
            cf.num_purchases_24h,
            cf.avg_transaction_amount,
            cf.days_since_last_transaction,
            cf.is_new_customer,
            mf.merchant_risk_score,
            mf.total_transactions_30d
        FROM transactions t
        LEFT JOIN predictions p ON t.transaction_id = p.transaction_id
        LEFT JOIN customer_features cf ON t.customer_id = cf.customer_id
        LEFT JOIN merchant_features mf ON t.merchant_id = mf.merchant_id
        WHERE t.created_at >= NOW() - INTERVAL '90 days'
        ORDER BY t.created_at DESC
    """
    
    df = pd.read_sql(query, engine)
    
    # Save to temporary location for Databricks
    output_path = f"/tmp/training_data_{context['ds_nodash']}.parquet"
    df.to_parquet(output_path, index=False)
    
    return {
        'data_path': output_path,
        'num_samples': len(df),
        'fraud_rate': df['is_fraud'].mean() if 'is_fraud' in df.columns else 0.0
    }


def skip_training(**context):
    """Skip training - conditions not met"""
    ti = context['task_instance']
    decision = ti.xcom_pull(task_ids='check_should_retrain')
    
    return {'status': 'skipped', 'reason': decision['reason']}


def validate_trained_model(**context):
    """Validate model performance before promotion"""
    ti = context['task_instance']
    
    # Get Databricks job result
    job_result = ti.xcom_pull(task_ids='train_models_databricks')
    
    if not job_result:
        raise ValueError("No training results from Databricks")
    
    # Extract metrics from job result
    metrics = job_result.get('metadata', {}).get('metrics', {})
    
    # Validation thresholds
    min_recall = settings.min_recall_threshold
    min_precision = settings.min_precision_threshold
    
    recall = metrics.get('recall', 0)
    precision = metrics.get('precision', 0)
    
    if recall < min_recall:
        raise ValueError(f"Model recall {recall:.3f} below threshold {min_recall}")
    
    if precision < min_precision:
        raise ValueError(f"Model precision {precision:.3f} below threshold {min_precision}")
    
    return {
        'status': 'validated',
        'metrics': metrics,
        'model_uri': job_result.get('model_uri')
    }


def register_model_to_mlflow(**context):
    """Register validated model to MLflow Model Registry"""
    import mlflow
    
    ti = context['task_instance']
    validation_result = ti.xcom_pull(task_ids='validate_models')
    
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    
    model_uri = validation_result['model_uri']
    model_name = settings.mlflow_model_name
    
    # Register model
    model_version = mlflow.register_model(
        model_uri=model_uri,
        name=model_name
    )
    
    # Save to model_versions table
    import sqlalchemy as sa
    engine = sa.create_engine(settings.fraud_database_url)
    
    query = """
        INSERT INTO model_versions 
        (version, model_name, model_uri, recall, precision, f1_score, registered_by)
        VALUES (%s, %s, %s, %s, %s, %s, %s)
    """
    
    metrics = validation_result['metrics']
    
    with engine.connect() as conn:
        conn.execute(sa.text(query), (
            model_version.version,
            model_name,
            model_uri,
            metrics.get('recall'),
            metrics.get('precision'),
            metrics.get('f1_score'),
            'airflow_training_pipeline'
        ))
        conn.commit()
    
    return {
        'model_name': model_name,
        'model_version': model_version.version,
        'model_uri': model_uri,
        'status': 'registered'
    }


# Databricks job configuration
databricks_job_config = {
    'new_cluster': {
        'spark_version': '13.3.x-scala2.12',
        'node_type_id': 'Standard_DS3_v2',
        'num_workers': 2,
        'spark_env_vars': {
            'PYSPARK_PYTHON': '/databricks/python3/bin/python3'
        }
    },
    'notebook_task': {
        'notebook_path': '/Workspace/fraud-detection/training',
        'base_parameters': {
            'data_path': '{{ ti.xcom_pull(task_ids="load_training_data")["data_path"] }}',
            'mlflow_tracking_uri': settings.mlflow_tracking_uri,
            'experiment_name': '/fraud-detection/training-runs'
        }
    },
    'libraries': [
        {'pypi': {'package': 'xgboost==1.7.6'}},
        {'pypi': {'package': 'scikit-learn==1.3.2'}},
        {'pypi': {'package': 'mlflow==2.10.2'}}
    ]
}


# Define DAG
with DAG(
    '01_training_pipeline',
    default_args=default_args,
    description='Daily model training with intelligent decision logic',
    schedule_interval='0 2 * * *',  # 2 AM daily
    start_date=days_ago(1),
    catchup=False,
    tags=['training', 'ml', 'high-priority']
) as dag:
    
    # Task 1: Check if training needed
    check_retrain = PythonOperator(
        task_id='check_should_retrain',
        python_callable=check_should_retrain
    )
    
    # Task 2: Branch decision
    branch_decision = BranchPythonOperator(
        task_id='decide_training_branch',
        python_callable=decide_training_branch
    )
    
    # Task 3a: Load training data
    load_data = PythonOperator(
        task_id='load_training_data',
        python_callable=load_training_data
    )
    
    # Task 3b: Skip training
    skip = PythonOperator(
        task_id='skip_training',
        python_callable=skip_training
    )
    
    # Task 4: Train models on Databricks
    train_databricks = DatabricksSubmitRunOperator(
        task_id='train_models_databricks',
        databricks_conn_id='databricks_default',
        json=databricks_job_config
    )
    
    # Task 5: Validate model
    validate = PythonOperator(
        task_id='validate_models',
        python_callable=validate_trained_model
    )
    
    # Task 6: Register to MLflow
    register = PythonOperator(
        task_id='register_to_mlflow',
        python_callable=register_model_to_mlflow
    )
    
    # Define dependencies
    check_retrain >> branch_decision
    branch_decision >> [load_data, skip]
    load_data >> train_databricks >> validate >> register
