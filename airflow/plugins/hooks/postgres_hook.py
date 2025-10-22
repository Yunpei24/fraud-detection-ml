"""
PostgreSQL Hook for Fraud Detection Database
Uses centralized configuration from airflow.src.config.settings
"""
from typing import Any, Dict, List, Optional
from airflow.hooks.base import BaseHook
import sqlalchemy as sa
import pandas as pd
import sys
from pathlib import Path

# Add airflow/src to path
AIRFLOW_SRC = Path(__file__).parent.parent.parent / "src"
sys.path.insert(0, str(AIRFLOW_SRC))

from config.settings import settings


class FraudPostgresHook(BaseHook):
    """Hook for interacting with fraud detection PostgreSQL database"""
    
    def __init__(self, database_url: Optional[str] = None):
        """
        Initialize PostgreSQL hook.
        
        Args:
            database_url: PostgreSQL connection URL (defaults to settings.fraud_database_url)
        """
        super().__init__()
        self.database_url = database_url or settings.fraud_database_url
        self.engine = None
    
    def _get_database_url(self) -> str:
        """Get database URL from settings"""
        from airflow.config.settings import settings
        return settings.fraud_database_url
    
    def get_engine(self) -> sa.Engine:
        """Get SQLAlchemy engine"""
        if self.engine is None:
            self.engine = sa.create_engine(self.database_url)
        return self.engine
    
    def execute_query(self, query: str, params: Optional[tuple] = None) -> Any:
        """Execute SQL query"""
        engine = self.get_engine()
        with engine.connect() as conn:
            result = conn.execute(sa.text(query), params or ())
            conn.commit()
            return result
    
    def fetch_one(self, query: str, params: Optional[tuple] = None) -> Optional[tuple]:
        """Fetch single row"""
        result = self.execute_query(query, params)
        return result.fetchone()
    
    def fetch_all(self, query: str, params: Optional[tuple] = None) -> List[tuple]:
        """Fetch all rows"""
        result = self.execute_query(query, params)
        return result.fetchall()
    
    def fetch_dataframe(self, query: str) -> pd.DataFrame:
        """Fetch results as pandas DataFrame"""
        engine = self.get_engine()
        return pd.read_sql(query, engine)
    
    def insert_many(self, table: str, records: List[Dict[str, Any]]) -> int:
        """Bulk insert records"""
        if not records:
            return 0
        
        df = pd.DataFrame(records)
        engine = self.get_engine()
        
        rows_inserted = df.to_sql(
            table, 
            engine, 
            if_exists='append', 
            index=False
        )
        
        return rows_inserted
    
    def get_drift_metrics(self, hours: int = 24) -> pd.DataFrame:
        """Get recent drift metrics"""
        query = f"""
            SELECT *
            FROM drift_metrics
            WHERE detected_at >= NOW() - INTERVAL '{hours} hours'
            ORDER BY detected_at DESC
        """
        return self.fetch_dataframe(query)
    
    def get_retraining_triggers(self, limit: int = 10) -> pd.DataFrame:
        """Get recent retraining triggers"""
        query = f"""
            SELECT *
            FROM retraining_triggers
            ORDER BY triggered_at DESC
            LIMIT {limit}
        """
        return self.fetch_dataframe(query)
    
    def get_model_versions(self, is_production: bool = None) -> pd.DataFrame:
        """Get model versions"""
        query = "SELECT * FROM model_versions"
        if is_production is not None:
            query += f" WHERE is_production = {is_production}"
        query += " ORDER BY registered_at DESC"
        return self.fetch_dataframe(query)
    
    def save_drift_metric(
        self, 
        metric_type: str,
        metric_name: str,
        metric_value: float,
        threshold: float,
        threshold_exceeded: bool,
        severity: str
    ) -> None:
        """Save drift metric"""
        query = """
            INSERT INTO drift_metrics 
            (metric_type, metric_name, metric_value, threshold, threshold_exceeded, severity)
            VALUES (%s, %s, %s, %s, %s, %s)
        """
        self.execute_query(query, (
            metric_type, metric_name, metric_value, 
            threshold, threshold_exceeded, severity
        ))
    
    def save_retraining_trigger(
        self,
        trigger_reason: str,
        drift_type: str,
        drift_severity: str,
        airflow_dag_id: str,
        airflow_run_id: str
    ) -> None:
        """Save retraining trigger"""
        query = """
            INSERT INTO retraining_triggers 
            (trigger_reason, drift_type, drift_severity, airflow_dag_id, airflow_run_id, status)
            VALUES (%s, %s, %s, %s, %s, 'pending')
        """
        self.execute_query(query, (
            trigger_reason, drift_type, drift_severity, 
            airflow_dag_id, airflow_run_id
        ))
    
    def update_retraining_status(
        self,
        airflow_run_id: str,
        status: str,
        model_version: Optional[str] = None
    ) -> None:
        """Update retraining trigger status"""
        if status == 'completed':
            query = """
                UPDATE retraining_triggers
                SET status = %s, completed_at = NOW(), model_version = %s
                WHERE airflow_run_id = %s
            """
            self.execute_query(query, (status, model_version, airflow_run_id))
        else:
            query = """
                UPDATE retraining_triggers
                SET status = %s
                WHERE airflow_run_id = %s
            """
            self.execute_query(query, (status, airflow_run_id))
    
    def close(self) -> None:
        """Close database connection"""
        if self.engine:
            self.engine.dispose()
