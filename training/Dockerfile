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
COPY requirements.txt ./requirements.txt

# Use pip cache optimizations and clean up after installation
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# -----------------------------------------------------------
# 5. Copy the entire training codebase into the container
# -----------------------------------------------------------
COPY . .

# -----------------------------------------------------------
# 6. Environment variables
# -----------------------------------------------------------
ENV PYTHONUNBUFFERED=1 \
    MLFLOW_TRACKING_URI="file:mlruns" \
    MODEL_DIR="/app/models" \
    DATA_DIR="/app/data" \
    ARTIFACT_DIR="/app/training/artifacts"

# -----------------------------------------------------------
# 7. Create directories for data and models
# -----------------------------------------------------------
RUN mkdir -p $MODEL_DIR $DATA_DIR $ARTIFACT_DIR

# -----------------------------------------------------------
# 8. Default command (train the model)
# -----------------------------------------------------------
CMD ["python", "-m", "training.src.pipelines.training_pipeline", "--config", "configs/model_xgb.yaml"]
