"""
Custom PostgreSQL Hook for Fraud Detection
===========================================
Extends Airflow PostgresHook with convenience methods for fraud detection queries.
"""
from typing import Any, List, Optional, Tuple
from airflow.providers.postgres.hooks.postgres import PostgresHook


class FraudPostgresHook(PostgresHook):
    """
    Custom PostgreSQL hook for fraud detection with convenience methods.
    
    This hook extends the standard Airflow PostgresHook with methods
    specifically designed for fraud detection queries.
    
    Usage:
        hook = FraudPostgresHook(postgres_conn_id='postgres_default')
        results = hook.fetch_all("SELECT * FROM transactions LIMIT 10")
    """
    
    def __init__(self, postgres_conn_id: str = 'postgres_default', **kwargs):
        """
        Initialize FraudPostgresHook.
        
        Args:
            postgres_conn_id: Airflow connection ID for PostgreSQL
            **kwargs: Additional arguments passed to PostgresHook
        """
        super().__init__(postgres_conn_id=postgres_conn_id, **kwargs)
    
    def fetch_all(
        self,
        query: str,
        parameters: Optional[Tuple[Any, ...]] = None
    ) -> List[Tuple[Any, ...]]:
        """
        Fetch all rows from a SQL query.
        
        Args:
            query: SQL query string
            parameters: Query parameters (tuple)
            
        Returns:
            List of tuples containing query results
            
        Example:
            results = hook.fetch_all(
                "SELECT * FROM transactions WHERE amount > %s",
                (100.0,)
            )
        """
        conn = self.get_conn()
        cursor = conn.cursor()
        
        try:
            if parameters:
                cursor.execute(query, parameters)
            else:
                cursor.execute(query)
            
            return cursor.fetchall()
        finally:
            cursor.close()
    
    def fetch_one(
        self,
        query: str,
        parameters: Optional[Tuple[Any, ...]] = None
    ) -> Optional[Tuple[Any, ...]]:
        """
        Fetch one row from a SQL query.
        
        Args:
            query: SQL query string
            parameters: Query parameters (tuple)
            
        Returns:
            Single tuple containing query result, or None if no results
            
        Example:
            result = hook.fetch_one(
                "SELECT COUNT(*) FROM transactions WHERE is_fraud = %s",
                (True,)
            )
        """
        conn = self.get_conn()
        cursor = conn.cursor()
        
        try:
            if parameters:
                cursor.execute(query, parameters)
            else:
                cursor.execute(query)
            
            return cursor.fetchone()
        finally:
            cursor.close()
    
    def execute_query(
        self,
        query: str,
        parameters: Optional[Tuple[Any, ...]] = None,
        autocommit: bool = True
    ) -> int:
        """
        Execute a SQL query and return number of affected rows.
        
        Useful for INSERT, UPDATE, DELETE operations.
        
        Args:
            query: SQL query string
            parameters: Query parameters (tuple)
            autocommit: Whether to auto-commit the transaction
            
        Returns:
            Number of affected rows
            
        Example:
            rows_affected = hook.execute_query(
                "UPDATE transactions SET processed = %s WHERE id = %s",
                (True, 12345)
            )
        """
        conn = self.get_conn()
        cursor = conn.cursor()
        
        try:
            if parameters:
                cursor.execute(query, parameters)
            else:
                cursor.execute(query)
            
            if autocommit:
                conn.commit()
            
            return cursor.rowcount
        except Exception as e:
            if autocommit:
                conn.rollback()
            raise e
        finally:
            cursor.close()
    
    def fetch_dataframe(
        self,
        query: str,
        parameters: Optional[Tuple[Any, ...]] = None
    ):
        """
        Fetch query results as a pandas DataFrame.
        
        Args:
            query: SQL query string
            parameters: Query parameters (tuple)
            
        Returns:
            pandas DataFrame with query results
            
        Example:
            df = hook.fetch_dataframe(
                "SELECT * FROM transactions WHERE created_at >= %s",
                ('2025-01-01',)
            )
        """
        import pandas as pd
        
        conn = self.get_conn()
        
        try:
            if parameters:
                df = pd.read_sql(query, conn, params=parameters)
            else:
                df = pd.read_sql(query, conn)
            
            return df
        finally:
            conn.close()
    
    def check_table_exists(self, table_name: str, schema: str = 'public') -> bool:
        """
        Check if a table exists in the database.
        
        Args:
            table_name: Name of the table
            schema: Database schema (default: 'public')
            
        Returns:
            True if table exists, False otherwise
        """
        query = """
            SELECT EXISTS (
                SELECT 1 
                FROM information_schema.tables 
                WHERE table_schema = %s 
                AND table_name = %s
            )
        """
        
        result = self.fetch_one(query, (schema, table_name))
        return result[0] if result else False
    
    def get_row_count(self, table_name: str, where_clause: str = "") -> int:
        """
        Get row count for a table.
        
        Args:
            table_name: Name of the table
            where_clause: Optional WHERE clause (without 'WHERE' keyword)
            
        Returns:
            Number of rows
            
        Example:
            count = hook.get_row_count(
                "transactions",
                "created_at >= NOW() - INTERVAL '24 hours'"
            )
        """
        if where_clause:
            query = f"SELECT COUNT(*) FROM {table_name} WHERE {where_clause}"
        else:
            query = f"SELECT COUNT(*) FROM {table_name}"
        
        result = self.fetch_one(query)
        return result[0] if result else 0
