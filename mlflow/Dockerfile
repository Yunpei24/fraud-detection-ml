# Custom MLflow Dockerfile with PostgreSQL support
# Based on official MLflow image with psycopg2 installed

FROM ghcr.io/mlflow/mlflow:v2.10.2

# Switch to root to install packages
USER root

# Install psycopg2-binary for PostgreSQL backend support
# Note: psycopg2-binary is a standalone package that doesn't require compilation ==2.9.9
RUN pip install --no-cache-dir psycopg2-binary

# Install Python urllib for healthcheck (already in base image, but explicit)
# No additional packages needed

# Create directory for artifacts and backend storage
RUN mkdir -p /mlflow/artifacts /mlflow/backend && \
    chown -R 1000:1000 /mlflow

# Switch back to non-root user
USER 1000

# Expose MLflow port
EXPOSE 5000

# Default command (can be overridden in docker-compose)
CMD ["mlflow", "server", \
     "--host", "0.0.0.0", \
     "--port", "5000", \
     "--backend-store-uri", "file:///mlflow/backend", \
     "--default-artifact-root", "/mlflow/artifacts"]
