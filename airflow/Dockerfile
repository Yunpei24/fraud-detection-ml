FROM apache/airflow:2.7.0-python3.10

USER root

# Install system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    libpq-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Switch to airflow user (Apache Airflow enforces this for pip)
USER airflow

# Install common package as airflow user
COPY --chown=airflow:root common/ /opt/airflow/common/
RUN pip install --user --no-cache-dir /opt/airflow/common

# Copy requirements
COPY airflow/requirements.txt /requirements.txt

# Install Python dependencies (sans -e ../common qui est déjà installé)
RUN pip install --no-cache-dir -r /requirements.txt

# Create necessary directories
RUN mkdir -p /opt/airflow/dags /opt/airflow/logs /opt/airflow/plugins /opt/airflow/config

# Copy Airflow configurations and DAGs
COPY airflow/config/ /opt/airflow/config/
COPY airflow/dags/ /opt/airflow/dags/

# Set working directory
WORKDIR /opt/airflow
