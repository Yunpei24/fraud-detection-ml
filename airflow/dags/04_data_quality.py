"""
DAG 04: Data Quality Monitoring
Surveille la qualité des données en production
Utilise PostgresHook et helpers

Schedule: Quotidien 3h AM
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

default_args = {
    'owner': 'data-team',
    'depends_on_past': False,
    'email': settings.alert_email_recipients,
    'email_on_failure': True,
    'retries': 1,
    'retry_delay': timedelta(minutes=5)
}


def check_data_freshness(**context):
    """Vérifie fraîcheur des données via PostgresHook"""
    from plugins.hooks.postgres_hook import FraudPostgresHook
    
    hook = FraudPostgresHook()
    
    # Check dernière transaction
    query = """
        SELECT 
            MAX(created_at) as last_transaction,
            COUNT(*) as total_transactions_24h
        FROM transactions
        WHERE created_at >= NOW() - INTERVAL '24 hours'
    """
    
    result = hook.fetch_one(query)
    
    last_transaction = result[0]
    total_24h = result[1]
    
    # Alerte si pas de données depuis max_data_age_hours (from settings)
    if last_transaction:
        hours_ago = (datetime.now() - last_transaction).total_seconds() / 3600
        is_fresh = hours_ago < settings.max_data_age_hours
    else:
        hours_ago = None
        is_fresh = False
    
    return {
        'last_transaction': last_transaction.isoformat() if last_transaction else None,
        'hours_since_last': hours_ago,
        'total_24h': total_24h,
        'is_fresh': is_fresh,
        'status': 'OK' if is_fresh else 'STALE'
    }


def check_missing_values(**context):
    """Vérifie valeurs manquantes using FraudPostgresHook"""
    from plugins.hooks.postgres_hook import FraudPostgresHook
    hook = FraudPostgresHook()
    
    # Check NULL values dans colonnes critiques
    query = """
        SELECT 
            COUNT(*) as total,
            SUM(CASE WHEN amount IS NULL THEN 1 ELSE 0 END) as null_amount,
            SUM(CASE WHEN customer_id IS NULL THEN 1 ELSE 0 END) as null_customer,
            SUM(CASE WHEN merchant_id IS NULL THEN 1 ELSE 0 END) as null_merchant
        FROM transactions
        WHERE created_at >= NOW() - INTERVAL '24 hours'
    """
    
    result = hook.fetch_one(query)
    
    total = result[0] or 1
    
    return {
        'total_transactions': result[0],
        'null_amount': result[1],
        'null_customer': result[2],
        'null_merchant': result[3],
        'null_amount_pct': (result[1] or 0) / total * 100,
        'null_customer_pct': (result[2] or 0) / total * 100,
        'null_merchant_pct': (result[3] or 0) / total * 100,
        'status': 'OK' if (result[1] or 0) == 0 else 'WARNING'
    }


def check_data_ranges(**context):
    """Vérifie ranges des valeurs using FraudPostgresHook"""
    from plugins.hooks.postgres_hook import FraudPostgresHook
    hook = FraudPostgresHook()
    
    query = """
        SELECT 
            MIN(amount) as min_amount,
            MAX(amount) as max_amount,
            AVG(amount) as avg_amount,
            STDDEV(amount) as stddev_amount,
            COUNT(CASE WHEN amount < 0 THEN 1 END) as negative_amounts,
            COUNT(CASE WHEN amount > 10000 THEN 1 END) as very_high_amounts
        FROM transactions
        WHERE created_at >= NOW() - INTERVAL '24 hours'
    """
    
    result = hook.fetch_one(query)
    
    issues = []
    if result[4] > 0:  # negative amounts
        issues.append(f"Found {result[4]} negative amounts")
    if result[1] > 1000000:  # unrealistic max
        issues.append(f"Unrealistic max amount: {result[1]}")
    
    return {
        'min_amount': float(result[0] or 0),
        'max_amount': float(result[1] or 0),
        'avg_amount': float(result[2] or 0),
        'stddev_amount': float(result[3] or 0),
        'negative_amounts': result[4],
        'very_high_amounts': result[5],
        'issues': issues,
        'status': 'OK' if len(issues) == 0 else 'WARNING'
    }


def check_duplicates(**context):
    """Vérifie doublons using FraudPostgresHook"""
    from plugins.hooks.postgres_hook import FraudPostgresHook
    hook = FraudPostgresHook()
    
    query = """
        SELECT 
            COUNT(*) as total,
            COUNT(DISTINCT transaction_id) as unique_ids,
            COUNT(*) - COUNT(DISTINCT transaction_id) as duplicates
        FROM transactions
        WHERE created_at >= NOW() - INTERVAL '24 hours'
    """
    
    result = hook.fetch_one(query)
    
    return {
        'total_transactions': result[0],
        'unique_ids': result[1],
        'duplicates': result[2],
        'duplicate_rate': (result[2] / (result[0] or 1)) * 100,
        'status': 'OK' if result[2] == 0 else 'WARNING'
    }


def check_prediction_coverage(**context):
    """Vérifie que toutes transactions ont prédiction using FraudPostgresHook"""
    from plugins.hooks.postgres_hook import FraudPostgresHook
    hook = FraudPostgresHook()
    
    query = """
        SELECT 
            COUNT(DISTINCT t.transaction_id) as total_transactions,
            COUNT(DISTINCT p.transaction_id) as transactions_with_predictions,
            COUNT(DISTINCT t.transaction_id) - COUNT(DISTINCT p.transaction_id) as missing_predictions
        FROM transactions t
        LEFT JOIN predictions p ON t.transaction_id = p.transaction_id
        WHERE t.created_at >= NOW() - INTERVAL '24 hours'
    """
    
    result = hook.fetch_one(query)
    
    coverage = (result[1] / (result[0] or 1)) * 100
    
    # Use centralized coverage threshold
    min_coverage = settings.min_prediction_coverage if hasattr(settings, 'min_prediction_coverage') else 95
    
    return {
        'total_transactions': result[0],
        'with_predictions': result[1],
        'missing_predictions': result[2],
        'coverage_pct': coverage,
        'status': 'OK' if coverage >= min_coverage else 'CRITICAL'
    }


def save_quality_metrics(**context):
    """Sauvegarde métriques qualité using FraudPostgresHook"""
    from plugins.hooks.postgres_hook import FraudPostgresHook
    
    ti = context['task_instance']
    
    freshness = ti.xcom_pull(task_ids='check_data_freshness')
    missing = ti.xcom_pull(task_ids='check_missing_values')
    ranges = ti.xcom_pull(task_ids='check_data_ranges')
    duplicates = ti.xcom_pull(task_ids='check_duplicates')
    coverage = ti.xcom_pull(task_ids='check_prediction_coverage')
    
    # Use hook instead of direct SQLAlchemy
    hook = FraudPostgresHook()
    
    # Sauvegarder dans data_quality_log
    metrics = [
        ('data_freshness', 'hours_since_last', freshness['hours_since_last'] or 999, freshness['status']),
        ('missing_values', 'null_amount_pct', missing['null_amount_pct'], missing['status']),
        ('data_ranges', 'negative_amounts', ranges['negative_amounts'], ranges['status']),
        ('duplicates', 'duplicate_rate', duplicates['duplicate_rate'], duplicates['status']),
        ('prediction_coverage', 'coverage_pct', coverage['coverage_pct'], coverage['status'])
    ]
    
    query = """
        INSERT INTO data_quality_log 
        (check_type, metric_name, metric_value, status)
        VALUES (%s, %s, %s, %s)
    """
    
    for metric in metrics:
        hook.execute_query(query, metric)
    
    # Déterminer statut global
    statuses = [freshness['status'], missing['status'], ranges['status'], 
                duplicates['status'], coverage['status']]
    
    if 'CRITICAL' in statuses:
        global_status = 'CRITICAL'
    elif 'WARNING' in statuses:
        global_status = 'WARNING'
    else:
        global_status = 'OK'
    
    return {
        'global_status': global_status,
        'checks_passed': statuses.count('OK'),
        'checks_total': len(statuses),
        'timestamp': datetime.now().isoformat()
    }


def send_quality_alert(**context):
    """Envoie alerte si problèmes qualité"""
    ti = context['task_instance']
    summary = ti.xcom_pull(task_ids='save_quality_metrics')
    
    if summary['global_status'] == 'OK':
        return {'status': 'no_alert_needed'}
    
    freshness = ti.xcom_pull(task_ids='check_data_freshness')
    coverage = ti.xcom_pull(task_ids='check_prediction_coverage')
    
    from plugins.operators.alert_operator import FraudDetectionAlertOperator
    
    alert_op = FraudDetectionAlertOperator(
        task_id='quality_alert_internal',
        alert_type='data_quality',
        severity=summary['global_status'],
        message=f"Data quality issues detected: {summary['checks_passed']}/{summary['checks_total']} checks passed",
        details={
            'freshness': freshness,
            'coverage': coverage,
            'global_status': summary['global_status']
        }
    )
    
    return alert_op.execute(context)


# Define DAG
with DAG(
    '04_data_quality',
    default_args=default_args,
    description='Surveillance quotidienne de la qualité des données',
    schedule_interval='0 3 * * *',  # 3 AM daily
    start_date=days_ago(1),
    catchup=False,
    tags=['quality', 'monitoring', 'data']
) as dag:
    
    freshness = PythonOperator(
        task_id='check_data_freshness',
        python_callable=check_data_freshness
    )
    
    missing = PythonOperator(
        task_id='check_missing_values',
        python_callable=check_missing_values
    )
    
    ranges = PythonOperator(
        task_id='check_data_ranges',
        python_callable=check_data_ranges
    )
    
    dups = PythonOperator(
        task_id='check_duplicates',
        python_callable=check_duplicates
    )
    
    coverage = PythonOperator(
        task_id='check_prediction_coverage',
        python_callable=check_prediction_coverage
    )
    
    save = PythonOperator(
        task_id='save_quality_metrics',
        python_callable=save_quality_metrics
    )
    
    alert = PythonOperator(
        task_id='send_quality_alert',
        python_callable=send_quality_alert
    )
    
    # All checks in parallel, then save and alert
    [freshness, missing, ranges, dups, coverage] >> save >> alert
