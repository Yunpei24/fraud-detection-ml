# -----------------------------------------------------------
#  Fraud Detection ML - Training Dockerfile
#  Base: Python 3.10 Slim with optional CUDA support
# -----------------------------------------------------------

# 1. Base Image (lightweight, CPU by default)
FROM python:3.10-slim

# Uncomment the next line if you plan to use GPU with CUDA
# FROM nvidia/cuda:12.2.0-runtime-ubuntu22.04

# -----------------------------------------------------------
# 2. System dependencies
# -----------------------------------------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    curl \
    wget \
    && rm -rf /var/lib/apt/lists/*

# -----------------------------------------------------------
# 3. Set working directory
# -----------------------------------------------------------
WORKDIR /app

# -----------------------------------------------------------
# 4. Copy dependency file and install Python packages
# -----------------------------------------------------------

# Install common package first
COPY common/ /app/common/
RUN pip install --upgrade pip && \
    pip install --no-cache-dir \
    --default-timeout=100 \
    --retries 5 \
    /app/common

# Install training requirements
COPY training/requirements.txt ./requirements.txt

# Use pip cache optimizations and clean up after installation with timeout and retry settings
RUN pip install --no-cache-dir \
    --default-timeout=100 \
    --retries 5 \
    -r requirements.txt

# -----------------------------------------------------------
# 5. Copy centralized config and training codebase
# -----------------------------------------------------------

# IMPORTANT: Copy centralized config from project root
# This contains settings.py with Pydantic v2 that is used by training/src/config/settings.py
COPY config/ /app/config/

# Copy the entire training codebase into the container
COPY training/ .

# -----------------------------------------------------------
# 6. Environment variables
# -----------------------------------------------------------
ENV PYTHONUNBUFFERED=1 \
    MLFLOW_TRACKING_URI="file:mlruns" \
    MODEL_DIR="/app/models" \
    DATA_DIR="/app/data" \
    ARTIFACT_DIR="/app/training/artifacts" \
    PROMETHEUS_PORT=9095

# -----------------------------------------------------------
# 7. Create directories for data and models
# -----------------------------------------------------------
RUN mkdir -p $MODEL_DIR $DATA_DIR $ARTIFACT_DIR $DATA_DIR/raw

# -----------------------------------------------------------
# 7.5. Create non-root user for security + MLflow directory
# -----------------------------------------------------------
# Create MLflow directory for volume mounting (will be accessible by appuser)
RUN mkdir -p /mlflow/artifacts && \
    useradd -m -u 1000 appuser && \
    chown -R appuser:appuser /app /mlflow

# Switch to non-root user
USER appuser

# -----------------------------------------------------------
# 8. Expose Prometheus metrics port
# -----------------------------------------------------------
EXPOSE 9095

# -----------------------------------------------------------
# 9. Health check - verify metrics server is running
# -----------------------------------------------------------
HEALTHCHECK --interval=30s --timeout=10s --start-period=10s --retries=3 \
  CMD curl -f http://localhost:9095/metrics || exit 1

# -----------------------------------------------------------
# 10. Default command (run metrics server)
# Training jobs will be triggered by Airflow via DockerOperator
# -----------------------------------------------------------
CMD ["python", "metrics_server.py"]
