#!/usr/bin/env python3
"""
Script for configuring Airflow connections automatically
Usage: python setup_connections.py
"""

from airflow.models import Connection
from airflow import settings
import os


def setup_postgres_connection():
    """Configure PostgreSQL fraud_db connection"""
    conn_id = 'fraud_postgres'
    
    session = settings.Session()
    existing = session.query(Connection).filter(Connection.conn_id == conn_id).first()
    
    if existing:
        print(f" Connection '{conn_id}' already exists")
        return
    
    conn = Connection(
        conn_id=conn_id,
        conn_type='postgres',
        host='postgres',
        schema='fraud_detection',
        login='fraud_user',
        password='fraud_pass_dev_2024',
        port=5432
    )
    
    session.add(conn)
    session.commit()
    session.close()
    
    print(f" PostgreSQL connection '{conn_id}' created")


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
        print(f" HTTP connection '{conn_id}' created")
    
    session.commit()
    session.close()


def main():
    """Setup all connections"""
    print("="*60)
    print("🔧 Configuration of all Airflow connections")
    print("="*60)
    
    try:
        setup_postgres_connection()
        setup_http_connections()
        
        print("\n All the connections are configurated!")
        print("\nCheck in the UI: Admin > Connections")
        
    except Exception as e:
        print(f"\n Error: {e}")
        print("You can configure manually in the Airflow UI")


if __name__ == '__main__':
    main()
