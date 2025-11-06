FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    postgresql-client \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install common package first
COPY common/ /app/common/
RUN pip install --no-cache-dir \
    --default-timeout=300 \
    --retries 5 \
    /app/common

# Copy requirements
COPY drift/requirements.txt .

# Install Python dependencies with timeout and retry settings
RUN pip install --no-cache-dir \
    --default-timeout=300 \
    --retries 5 \
    -r requirements.txt

# Copy application code
COPY drift/ .

# Create directories for reports and logs
RUN mkdir -p /app/reports /app/logs

# Create non-root user for security
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app

# Switch to non-root user
USER appuser

# Expose Prometheus metrics port
EXPOSE 9097

# Health check - verify metrics server is running
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
  CMD curl -f http://localhost:9097/metrics || exit 1

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app

# Run metrics server (exposes Prometheus metrics on port 9092)
CMD ["python", "metrics_server.py"]
