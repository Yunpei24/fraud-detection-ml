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

# Install Python dependencies
RUN pip install --no-cache-dir -r /requirements.txt

# Create necessary directories
RUN mkdir -p /opt/airflow/dags /opt/airflow/logs /opt/airflow/plugins /opt/airflow/config /opt/airflow/config_root

# IMPORTANT: Copy centralized config from project root
# This contains settings.py with Pydantic v1/v2 compatibility
COPY config/ /opt/airflow/config_root/

# Copy Airflow-specific configurations (constants.py, airflow.cfg)
COPY airflow/config/ /opt/airflow/config/

# Copy DAGs
COPY airflow/dags/ /opt/airflow/dags/

# Add config_root to PYTHONPATH so "from config import" finds it
ENV PYTHONPATH="/opt/airflow/config_root:${PYTHONPATH}"

# Set working directory
WORKDIR /opt/airflow
