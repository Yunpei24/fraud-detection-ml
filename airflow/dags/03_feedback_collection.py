"""
DAG 03: Feedback Collection Pipeline
Collecte les labels des analystes pour améliorer le modèle
Utilise data.src.storage.database pour queries

Schedule: Quotidien 1h AM
"""
from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.utils.dates import days_ago
import sys
from pathlib import Path

# Add config and plugins to path
AIRFLOW_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(AIRFLOW_ROOT / "config"))
sys.path.insert(0, str(AIRFLOW_ROOT / "plugins"))

from settings import settings
from module_loader import loader
from helpers import validate_training_metrics, should_trigger_retraining, format_metric_value

default_args = {
    'owner': 'ml-team',
    'depends_on_past': False,
    'email': settings.alert_email_recipients,
    'email_on_failure': True,
    'retries': 1,
    'retry_delay': timedelta(minutes=5)
}


def collect_analyst_labels(**context):
    """Collecte les labels confirmés par les analystes via DatabaseService"""
    import sqlalchemy as sa
    from plugins.hooks.postgres_hook import FraudPostgresHook
    
    hook = FraudPostgresHook()
    
    # Récupérer predictions avec labels analystes (dernières 24h)
    query = """
        INSERT INTO feedback_labels (transaction_id, predicted_label, analyst_label, confidence, feedback_quality)
        SELECT 
            p.transaction_id,
            p.prediction,
            t.is_fraud as analyst_label,
            p.probability,
            CASE 
                WHEN p.prediction = t.is_fraud THEN 'CORRECT'
                WHEN p.prediction = 1 AND t.is_fraud = 0 THEN 'FALSE_POSITIVE'
                WHEN p.prediction = 0 AND t.is_fraud = 1 THEN 'FALSE_NEGATIVE'
            END as feedback_quality
        FROM predictions p
        JOIN transactions t ON p.transaction_id = t.transaction_id
        WHERE p.created_at >= NOW() - INTERVAL '24 hours'
        AND t.analyst_reviewed = true
        AND NOT EXISTS (
            SELECT 1 FROM feedback_labels f 
            WHERE f.transaction_id = p.transaction_id
        )
    """
    
    rows_inserted = hook.execute_query(query)
    
    return {
        'status': 'success',
        'labels_collected': rows_inserted,
        'timestamp': datetime.now().isoformat()
    }


def analyze_feedback_quality(**context):
    """Analyse la qualité du feedback via PostgresHook"""
    from plugins.hooks.postgres_hook import FraudPostgresHook
    
    ti = context['task_instance']
    collection_result = ti.xcom_pull(task_ids='collect_analyst_labels')
    
    hook = FraudPostgresHook()
    
    # Calculer métriques feedback
    query = """
        SELECT 
            COUNT(*) as total_feedback,
            SUM(CASE WHEN feedback_quality = 'CORRECT' THEN 1 ELSE 0 END) as correct,
            SUM(CASE WHEN feedback_quality = 'FALSE_POSITIVE' THEN 1 ELSE 0 END) as false_positives,
            SUM(CASE WHEN feedback_quality = 'FALSE_NEGATIVE' THEN 1 ELSE 0 END) as false_negatives,
            AVG(confidence) as avg_confidence
        FROM feedback_labels
        WHERE created_at >= NOW() - INTERVAL '7 days'
    """
    
    result = hook.fetch_one(query)
    
    total = result[0] or 1
    metrics = {
        'total_feedback': result[0] or 0,
        'correct': result[1] or 0,
        'false_positives': result[2] or 0,
        'false_negatives': result[3] or 0,
        'avg_confidence': float(result[4] or 0),
        'accuracy': (result[1] or 0) / total,
        'fp_rate': (result[2] or 0) / total,
        'fn_rate': (result[3] or 0) / total
    }
    
    return metrics


def check_retraining_needed(**context):
    """Vérifie si retraining nécessaire basé sur feedback using helpers"""
    ti = context['task_instance']
    metrics = ti.xcom_pull(task_ids='analyze_feedback_quality')
    
    # Use centralized thresholds from settings
    min_accuracy = 0.90
    max_fn_rate = settings.max_false_negative_rate if hasattr(settings, 'max_false_negative_rate') else 0.05
    
    needs_retraining = (
        metrics['accuracy'] < min_accuracy or
        metrics['fn_rate'] > max_fn_rate
    )
    
    if needs_retraining:
        # Use helper to format values consistently
        reason = f"Accuracy {format_metric_value(metrics['accuracy'])} < {min_accuracy} or FN rate {format_metric_value(metrics['fn_rate'])} > {max_fn_rate}"
        
        return {
            'needs_retraining': True,
            'reason': reason,
            'metrics': metrics
        }
    
    return {
        'needs_retraining': False,
        'reason': 'Performance acceptable',
        'metrics': metrics
    }


def prepare_feedback_dataset(**context):
    """Prépare dataset avec feedback pour retraining using FraudPostgresHook"""
    import pandas as pd
    
    ti = context['task_instance']
    check_result = ti.xcom_pull(task_ids='check_retraining_needed')
    
    if not check_result['needs_retraining']:
        return {'status': 'skipped', 'reason': 'No retraining needed'}
    
    # Use hook instead of direct SQLAlchemy
    from plugins.hooks.postgres_hook import FraudPostgresHook
    hook = FraudPostgresHook()
    
    # Charger transactions avec feedback
    query = """
        SELECT 
            t.*,
            f.analyst_label,
            f.feedback_quality,
            cf.num_purchases_24h,
            cf.avg_transaction_amount,
            mf.merchant_risk_score
        FROM transactions t
        JOIN feedback_labels f ON t.transaction_id = f.transaction_id
        LEFT JOIN customer_features cf ON t.customer_id = cf.customer_id
        LEFT JOIN merchant_features mf ON t.merchant_id = mf.merchant_id
        WHERE f.created_at >= NOW() - INTERVAL '30 days'
    """
    
    results = hook.fetch_all(query)
    
    # Convert to DataFrame for parquet export
    columns = ['transaction_id', 'customer_id', 'merchant_id', 'amount', 'timestamp', 
               'is_fraud', 'analyst_label', 'feedback_quality', 'num_purchases_24h', 
               'avg_transaction_amount', 'merchant_risk_score']
    df = pd.DataFrame(results, columns=columns)
    
    # Sauvegarder pour training
    output_path = f"/tmp/feedback_dataset_{context['ds_nodash']}.parquet"
    df.to_parquet(output_path, index=False)
    
    return {
        'status': 'prepared',
        'dataset_path': output_path,
        'num_samples': len(df),
        'fraud_rate': float(df['analyst_label'].mean())
    }


def generate_feedback_report(**context):
    """Génère rapport de feedback"""
    ti = context['task_instance']
    
    collection_result = ti.xcom_pull(task_ids='collect_analyst_labels')
    quality_metrics = ti.xcom_pull(task_ids='analyze_feedback_quality')
    retraining_check = ti.xcom_pull(task_ids='check_retraining_needed')
    
    report = f"""
    === Feedback Collection Report ===
    Date: {context['ds']}
    
    COLLECTION:
    - Labels collected: {collection_result['labels_collected']}
    
    QUALITY (Last 7 days):
    - Total feedback: {quality_metrics['total_feedback']}
    - Accuracy: {quality_metrics['accuracy']:.2%}
    - False Positives: {quality_metrics['false_positives']} ({quality_metrics['fp_rate']:.2%})
    - False Negatives: {quality_metrics['false_negatives']} ({quality_metrics['fn_rate']:.2%})
    - Avg Confidence: {quality_metrics['avg_confidence']:.3f}
    
    RETRAINING:
    - Needed: {retraining_check['needs_retraining']}
    - Reason: {retraining_check['reason']}
    """
    
    print(report)
    
    # Sauvegarder dans database using FraudPostgresHook
    from plugins.hooks.postgres_hook import FraudPostgresHook
    hook = FraudPostgresHook()
    
    query = """
        INSERT INTO pipeline_execution_log 
        (pipeline_name, status, execution_time_seconds, records_processed, error_message)
        VALUES (%s, %s, %s, %s, %s)
    """
    
    hook.execute_query(query, (
        '03_feedback_collection',
        'SUCCESS',
        0,  # To be calculated
        collection_result['labels_collected'],
        None
    ))
    
    return {'report': report}


# Define DAG
with DAG(
    '03_feedback_collection',
    default_args=default_args,
    description='Collecte et analyse du feedback des analystes',
    schedule_interval='0 1 * * *',  # 1 AM daily
    start_date=days_ago(1),
    catchup=False,
    tags=['feedback', 'quality', 'monitoring']
) as dag:
    
    # Task 1: Collect labels
    collect = PythonOperator(
        task_id='collect_analyst_labels',
        python_callable=collect_analyst_labels
    )
    
    # Task 2: Analyze quality
    analyze = PythonOperator(
        task_id='analyze_feedback_quality',
        python_callable=analyze_feedback_quality
    )
    
    # Task 3: Check if retraining needed
    check_retrain = PythonOperator(
        task_id='check_retraining_needed',
        python_callable=check_retraining_needed
    )
    
    # Task 4: Prepare dataset
    prepare = PythonOperator(
        task_id='prepare_feedback_dataset',
        python_callable=prepare_feedback_dataset
    )
    
    # Task 5: Generate report
    report = PythonOperator(
        task_id='generate_feedback_report',
        python_callable=generate_feedback_report
    )
    
    # Dependencies
    collect >> analyze >> check_retrain >> prepare >> report
