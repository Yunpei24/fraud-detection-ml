"""
Sensors pour Airflow - Fraud Detection System
"""
from datetime import datetime, timedelta
from typing import Any, Optional
from airflow.sensors.base import BaseSensorOperator
from airflow.utils.decorators import apply_defaults


class MLflowModelSensor(BaseSensorOperator):
    """Sensor qui attend qu'un nouveau modèle soit disponible dans MLflow"""
    
    @apply_defaults
    def __init__(
        self,
        model_name: str,
        stage: str = 'staging',
        min_recall: float = 0.80,
        *args,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.model_name = model_name
        self.stage = stage
        self.min_recall = min_recall
    
    def poke(self, context: Any) -> bool:
        """Check if new model available"""
        from plugins.hooks.mlflow_hook import MLflowHook
        
        hook = MLflowHook()
        
        # Get latest model version in stage
        model_version = hook.get_latest_model_version(
            self.model_name,
            stage=self.stage
        )
        
        if not model_version:
            self.log.info(f"No model found in {self.stage} stage")
            return False
        
        # Check if model meets recall threshold
        run_id = model_version.run_id
        metrics = hook.get_model_metrics(run_id)
        
        recall = metrics.get('recall', 0)
        
        if recall < self.min_recall:
            self.log.warning(
                f"Model recall {recall:.3f} below threshold {self.min_recall}"
            )
            return False
        
        self.log.info(
            f"Model {model_version.version} ready with recall {recall:.3f}"
        )
        return True


class DriftDetectedSensor(BaseSensorOperator):
    """Sensor qui attend détection de drift critique"""
    
    @apply_defaults
    def __init__(
        self,
        check_interval_hours: int = 1,
        drift_threshold: float = 0.5,
        *args,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.check_interval_hours = check_interval_hours
        self.drift_threshold = drift_threshold
    
    def poke(self, context: Any) -> bool:
        """Check if critical drift detected"""
        from plugins.hooks.postgres_hook import FraudPostgresHook
        
        hook = FraudPostgresHook()
        
        # Get recent drift metrics
        df = hook.get_drift_metrics(hours=self.check_interval_hours)
        
        if df.empty:
            self.log.info("No drift metrics found")
            return False
        
        # Check for critical drift
        critical_drift = df[
            (df['threshold_exceeded'] == True) &
            (df['severity'].isin(['CRITICAL', 'HIGH']))
        ]
        
        if not critical_drift.empty:
            self.log.info(
                f"Critical drift detected: {len(critical_drift)} metrics"
            )
            return True
        
        return False


class DataFreshnessSensor(BaseSensorOperator):
    """Sensor qui attend des données fraîches"""
    
    @apply_defaults
    def __init__(
        self,
        max_age_hours: int = 2,
        min_records: int = 100,
        *args,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.max_age_hours = max_age_hours
        self.min_records = min_records
    
    def poke(self, context: Any) -> bool:
        """Check if fresh data available"""
        from plugins.hooks.postgres_hook import FraudPostgresHook
        import sqlalchemy as sa
        
        hook = FraudPostgresHook()
        
        query = f"""
            SELECT 
                MAX(created_at) as last_transaction,
                COUNT(*) as recent_count
            FROM transactions
            WHERE created_at >= NOW() - INTERVAL '{self.max_age_hours} hours'
        """
        
        result = hook.fetch_one(query)
        
        if not result or not result[0]:
            self.log.info("No recent transactions found")
            return False
        
        last_transaction = result[0]
        recent_count = result[1]
        
        hours_ago = (datetime.now() - last_transaction).total_seconds() / 3600
        
        if hours_ago > self.max_age_hours:
            self.log.warning(f"Data stale: {hours_ago:.1f} hours old")
            return False
        
        if recent_count < self.min_records:
            self.log.warning(
                f"Insufficient records: {recent_count} < {self.min_records}"
            )
            return False
        
        self.log.info(
            f"Fresh data available: {recent_count} records, "
            f"last updated {hours_ago:.1f}h ago"
        )
        return True
