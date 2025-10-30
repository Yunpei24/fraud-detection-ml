#!/bin/bash
# Script d'initialisation Airflow pour Docker
set -e

echo "Initializing Airflow database..."

# Wait for PostgreSQL
echo "Waiting for PostgreSQL to be ready..."
while ! nc -z postgres 5432; do
  sleep 1
done
echo "PostgreSQL is ready!"

# Initialize Airflow database
if airflow db check 2>/dev/null; then
  echo "Airflow DB already initialized"
else
  echo "Initializing Airflow DB..."
  airflow db init
  
  # Create admin user if not exists
  echo "Creating Airflow admin user..."
  airflow users create \
    --username admin \
    --firstname Admin \
    --lastname User \
    --role Admin \
    --email admin@localhost.com \
    --password admin || echo "Admin user already exists"
fi

echo "Airflow initialization complete!"
