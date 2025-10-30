#!/bin/bash
# ============================================================================
# Database Creation Script
# This runs before schema.sql to create additional databases
# ============================================================================

set -e

# Wait for PostgreSQL to be ready
until pg_isready -U fraud_user -d postgres; do
  echo "Waiting for PostgreSQL to be ready..."
  sleep 2
done

echo "Creating additional databases..."

# Create separate database for MLflow to avoid table conflicts with Airflow
psql -U fraud_user -d postgres -c "CREATE DATABASE mlflow_db;" 2>/dev/null || echo "Database mlflow_db already exists"
psql -U fraud_user -d postgres -c "GRANT ALL PRIVILEGES ON DATABASE mlflow_db TO fraud_user;"

echo "Database creation completed."