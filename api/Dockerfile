FROM python:3.10-slim as base

# Environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    libpq-dev \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# ============== BUILDER STAGE ==============
FROM base as builder

WORKDIR /build

# Install common package first
COPY common/ /build/common/
RUN pip install --user --no-warn-script-location --timeout=300 --retries=5 /build/common

# Copy and install API requirements
COPY api/requirements.txt .
RUN pip install --user --no-warn-script-location --timeout=300 --retries=5 -r requirements.txt

# ============== RUNTIME STAGE ==============
FROM base as runtime

# Create non-root user
RUN useradd -m -u 1000 appuser

WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /root/.local /home/appuser/.local

# Copy application code
COPY api/src/ ./src/
COPY api/scripts/ ./scripts/

# Create model storage directory for local deployment
RUN mkdir -p /mnt/fraud-models && chown appuser:appuser /mnt/fraud-models

# Set ownership
RUN chown -R appuser:appuser /app

# Switch to non-root user
USER appuser

# Update PATH
ENV PATH=/home/appuser/.local/bin:$PATH \
    PYTHONPATH=/app:/app/scripts

# Azure Web App uses dynamic PORT (default 8000 for local)
ENV PORT=8000
EXPOSE $PORT

# Health check (optional - Azure has its own)
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:${PORT}/health || exit 1

# Start with dynamic port from Azure
CMD uvicorn src.main:app --host 0.0.0.0 --port ${PORT} --log-level info