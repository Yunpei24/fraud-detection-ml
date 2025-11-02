# Multi-stage build for optimization
# Stage 1: Builder - compile dependencies
FROM python:3.10-slim as builder

WORKDIR /build

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Install common package first
COPY common/ /build/common/
RUN pip install --no-cache-dir --user /build/common

# Copy requirements
COPY data/requirements.txt .

# Install Python dependencies to /root/.local
RUN pip install --no-cache-dir --user -r requirements.txt


# Stage 2: Runtime - minimal image
FROM python:3.10-slim

WORKDIR /app

# Install only runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libpq5 \
    postgresql-client \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy Python packages from builder stage
COPY --from=builder /root/.local /root/.local

# IMPORTANT: Copy centralized config from project root
# This contains settings.py with Pydantic v2 that is used by data/src/config/settings.py
COPY config/ /app/config/

# Copy application code
COPY data/ .

# Create non-root user for security
RUN useradd -m -u 1000 appuser && \
    chown -R appuser:appuser /app && \
    cp -r /root/.local /home/appuser/.local && \
    chown -R appuser:appuser /home/appuser/.local

# Set environment variables
ENV PATH=/home/appuser/.local/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app

USER appuser

# Health check - verify metrics server is running
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:9091/metrics || exit 1

# Default command: run metrics server (exposes Prometheus metrics configured via PROMETHEUS_PORT env var)
CMD ["python", "metrics_server.py"]
