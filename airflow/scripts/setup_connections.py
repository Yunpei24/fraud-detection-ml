#!/usr/bin/env python3
"""
Script pour configurer les connexions Airflow automatiquement
Usage: python setup_connections.py
"""

from airflow.models import Connection
from airflow import settings
import os


def setup_databricks_connection():
    """Configure Databricks connection"""
    conn_id = 'databricks_default'
    
    # Check if exists
    session = settings.Session()
    existing = session.query(Connection).filter(Connection.conn_id == conn_id).first()
    
    if existing:
        print(f"⚠️  Connection '{conn_id}' already exists")
        return
    
    # Create new connection
    conn = Connection(
        conn_id=conn_id,
        conn_type='databricks',
        host=os.getenv('DATABRICKS_HOST', 'https://your-workspace.cloud.databricks.com'),
        login='token',
        password=os.getenv('DATABRICKS_TOKEN', 'your-token-here')
    )
    
    session.add(conn)
    session.commit()
    session.close()
    
    print(f"✅ Databricks connection '{conn_id}' created")


def setup_postgres_connection():
    """Configure PostgreSQL fraud_db connection"""
    conn_id = 'fraud_postgres'
    
    session = settings.Session()
    existing = session.query(Connection).filter(Connection.conn_id == conn_id).first()
    
    if existing:
        print(f"⚠️  Connection '{conn_id}' already exists")
        return
    
    conn = Connection(
        conn_id=conn_id,
        conn_type='postgres',
        host='postgres-fraud',
        schema='fraud_db',
        login='postgres',
        password='postgres',
        port=5432
    )
    
    session.add(conn)
    session.commit()
    session.close()
    
    print(f"✅ PostgreSQL connection '{conn_id}' created")


def setup_http_connections():
    """Configure HTTP connections for API/Data/Drift modules"""
    
    connections = [
        {
            'conn_id': 'fraud_api',
            'host': os.getenv('API_BASE_URL', 'http://fraud-api:8000')
        },
        {
            'conn_id': 'fraud_data',
            'host': os.getenv('DATA_BASE_URL', 'http://fraud-data:8001')
        },
        {
            'conn_id': 'fraud_drift',
            'host': os.getenv('DRIFT_BASE_URL', 'http://fraud-drift:8002')
        }
    ]
    
    session = settings.Session()
    
    for conn_info in connections:
        conn_id = conn_info['conn_id']
        existing = session.query(Connection).filter(Connection.conn_id == conn_id).first()
        
        if existing:
            print(f"⚠️  Connection '{conn_id}' already exists")
            continue
        
        conn = Connection(
            conn_id=conn_id,
            conn_type='http',
            host=conn_info['host']
        )
        
        session.add(conn)
        print(f"✅ HTTP connection '{conn_id}' created")
    
    session.commit()
    session.close()


def main():
    """Setup all connections"""
    print("="*60)
    print("🔧 Configuration des connexions Airflow")
    print("="*60)
    
    try:
        setup_databricks_connection()
        setup_postgres_connection()
        setup_http_connections()
        
        print("\n✅ Toutes les connexions configurées!")
        print("\nVérifier dans l'UI: Admin > Connections")
        
    except Exception as e:
        print(f"\n❌ Erreur: {e}")
        print("Vous pouvez configurer manuellement dans l'UI Airflow")


if __name__ == '__main__':
    main()
