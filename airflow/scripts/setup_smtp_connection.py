#!/usr/bin/env python3
"""
Set up SMTP connection for Airflow email alerts.
Run this script to create the SMTP connection in Airflow database.
"""

import os
from airflow import settings
from airflow.models import Connection

def setup_smtp_connection():
    """Set up Gmail SMTP connection in Airflow."""

    # Gmail SMTP settings
    conn_id = 'smtp_default'
    conn_type = 'smtp'
    host = 'smtp.gmail.com'
    login = 'emmanuelyunpei@gmail.com'  # Your Gmail address
    schema = ''  # Not used for SMTP
    port = 587

    # Get password from environment variable (set this securely)
    password = os.getenv('AIRFLOW_SMTP_PASSWORD')

    if not password:
        print("AIRFLOW_SMTP_PASSWORD environment variable not set")
        print("   SMTP connection will not be configured")
        print("   Set AIRFLOW_SMTP_PASSWORD to enable email alerts")
        return False

    # Check if connection already exists
    session = settings.Session()
    existing_conn = session.query(Connection).filter(Connection.conn_id == conn_id).first()

    if existing_conn:
        print(f"SMTP connection '{conn_id}' already exists. Updating...")
        existing_conn.conn_type = conn_type
        existing_conn.host = host
        existing_conn.login = login
        existing_conn.schema = schema
        existing_conn.port = port
        existing_conn.set_password(password)
    else:
        print(f"Creating new SMTP connection '{conn_id}'...")
        new_conn = Connection(
            conn_id=conn_id,
            conn_type=conn_type,
            host=host,
            login=login,
            schema=schema,
            port=port
        )
        new_conn.set_password(password)
        session.add(new_conn)

    session.commit()
    session.close()

    print("SMTP connection configured successfully!")
    print(f"   Connection ID: {conn_id}")
    print(f"   Host: {host}:{port}")
    print(f"   User: {login}")
    print("   Use STARTTLS: True")
    return True

if __name__ == "__main__":
    print("🔧 Setting up Airflow SMTP connection for Gmail...")
    success = setup_smtp_connection()
    if success:
        print("\nEmail alerts are now configured!")
        print("Test by running a DAG with email_on_failure=True")
    else:
        print("\nSetup failed. Please check your Gmail App Password.")