# Fraud Detection API

Production-ready FastAPI service for real-time fraud detection using ensemble machine learning models.

## Features

- **Ensemble ML Models**: XGBoost, Neural Network, Isolation Forest
- **Real-time Predictions**: Sub-second inference latency
- **Batch Processing**: Process multiple transactions at once
- **Model Versioning**: Manage multiple model versions with canary deployments
- **SMTP Alerts**: Email notifications for fraud detection
- **Prometheus Metrics**: Production monitoring and observability
- **Redis Caching**: Optional prediction caching
- **PostgreSQL Logging**: Transaction history and audit trail
- **Docker Ready**: Multi-stage containerized deployment
- **Comprehensive Testing**: 47+ unit tests with 80-97% coverage

## Prerequisites

- Python 3.10+
- PostgreSQL 12+ (optional, for logging)
- Redis 6+ (optional, for caching)
- Docker (for containerized deployment)

## Installation

### Local Development

```bash
cd fraud-detection-ml/api

# Create virtual environment
python -m venv venv_api
source venv_api/bin/activate  # On Windows: venv_api\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Configure environment
cp .env.example .env
# Edit .env with your configuration
```

### Docker Deployment

```bash
# Build image
docker build -t fraud-detection-api:latest .

# Run container
docker run -d \
  --name fraud-api \
  -p 8000:8000 \
  --env-file .env \
  fraud-detection-api:latest

# Check logs
docker logs -f fraud-api
```

## Running the API

### Development

```bash
uvicorn src.main:app --reload --host 0.0.0.0 --port 8000
```

### Production

```bash
uvicorn src.main:app --host 0.0.0.0 --port 8000 --workers 4
```

## API Endpoints

### Health Checks

- `GET /` - Root endpoint with API info
- `GET /health` - Basic health check
- `GET /metrics` - Prometheus metrics

### Predictions

- `POST /api/v1/predict` - Single transaction prediction (requires API key)
  ```json
  {
    "transaction_id": "TXN-001",
    "features": [0.5, 0.3, 0.8, ...]
  }
  ```

- `POST /api/v1/batch-predict` - Batch predictions (requires API key)
  ```json
  {
    "transactions": [
      {"transaction_id": "TXN-001", "features": [...]}
    ]
  }
  ```

### Admin (requires API key)

- `POST /admin/reload-model` - Reload ML models
- `GET /admin/model-version` - Get current model version

## Configuration

### Environment Variables

Key variables in `.env`:

```bash
# API Settings
API_HOST=0.0.0.0
API_PORT=8000
ENVIRONMENT=production

# Database (optional)
DATABASE_URL=postgresql://user:pass@localhost:5432/fraud_db

# Redis (optional)
REDIS_URL=redis://localhost:6379/0
ENABLE_CACHE=true

# Model Settings
MODEL_VERSION=1.0.0
FRAUD_THRESHOLD=0.5

# Email Alerts (SMTP)
SMTP_HOST=smtp.gmail.com
SMTP_PORT=587
SMTP_USER=your-email@gmail.com
SMTP_PASSWORD=your-app-password
ALERT_EMAIL_RECIPIENTS=alerts@company.com

# Security
REQUIRE_API_KEY=true
API_KEYS=your-api-key-here
ADMIN_TOKEN=your-admin-token-here
```

See `.env.example` for complete configuration options.

## Testing

### Run all tests

```bash
pytest tests/ -v --cov=src --cov-report=html
```

### Run specific test types

```bash
# Unit tests only
pytest tests/unit/ -v

# Test specific module
pytest tests/unit/test_alert_service.py -v

# Generate HTML coverage report
pytest tests/ --cov=src --cov-report=html
# Open htmlcov/index.html
```

## API Documentation

### Interactive Swagger UI

```
http://localhost:8000/docs
```

### ReDoc

```
http://localhost:8000/redoc
```

## Monitoring

### Health Check

```bash
curl http://localhost:8000/health
```

### Metrics (Prometheus)

```bash
curl http://localhost:8000/metrics
```

Available metrics:
- `fraud_detection_requests_total` - Total requests by endpoint
- `fraud_detection_errors_total` - Error counts by type
- `fraud_detection_predictions_total` - Predictions by label

## Architecture

### Components

- **FastAPI App**: Main application server
- **Prediction Service**: ML model inference
- **Alert Service**: SMTP-based fraud alerts
- **Model Registry**: Version management and canary deployments
- **Middleware**: Request logging, error handling, rate limiting

### Middleware Stack

1. **RequestLoggingMiddleware**: Logs all requests/responses
2. **ErrorHandlingMiddleware**: Centralized error handling
3. **RateLimitingMiddleware**: Request rate limiting (Redis-backed)
4. **CORSMiddleware**: Cross-origin request handling

### Database Schema

PostgreSQL tables (optional):
- `predictions` - Prediction history
- `alerts` - Alert logs
- `model_versions` - Model metadata

## Model Deployment

### Loading Models

Models are loaded from paths specified in `.env`:

```bash
MODEL_PATH_XGBOOST=models/xgboost_model.pkl
MODEL_PATH_NN=models/neural_network_model.pth
MODEL_PATH_ISOLATION_FOREST=models/isolation_forest_model.pkl
```

### Model Versioning

The API supports multiple model versions with canary deployments:

1. Register a new model version
2. Transition to Staging for testing
3. Use canary deployment to gradually increase traffic
4. Promote to Production when ready

### Reload Models

```bash
# Via API
curl -X POST http://localhost:8000/admin/reload-model \
  -H "Authorization: Bearer YOUR_ADMIN_TOKEN"

# Or restart the container
docker restart fraud-api
```

## Troubleshooting

### Port already in use

```bash
# Change port in .env or use -p flag
uvicorn src.main:app --port 8001
```

### Database connection error

Check `DATABASE_URL` in `.env` and ensure PostgreSQL is running:

```bash
psql $DATABASE_URL
```

### Redis connection error

Redis is optional. The API works without it but caching is disabled. Check `REDIS_URL` if you need caching.

### Email alerts not sending

Verify SMTP configuration in `.env`:
- Check `SMTP_HOST`, `SMTP_PORT`, `SMTP_USER`, `SMTP_PASSWORD`
- Verify `ALERT_EMAIL_RECIPIENTS` is set correctly
- For Gmail: Use app-specific password, not your account password

### Model not loading

Check that model files exist at the paths specified in `.env`. The API will log warnings on startup if files are missing.

## Performance

### Latency

- Single prediction: ~50-100ms
- Batch prediction (100 txns): ~500-800ms

### Throughput

- Theoretical: ~100-200 predictions/sec per worker
- With 4 workers: ~400-800 predictions/sec

### Resource Usage

- Memory: ~500MB base + models
- CPU: Varies by model complexity

## Security

- **API Key Authentication**: Required for prediction endpoints
- **JWT Authentication**: Required for sensitive operations (drift detection, explainability, audit)
- **Admin Token Authentication**: Required for administrative operations
- **Non-root Docker user**
- **CORS enabled**
- **Rate limiting** (Redis-backed)
- **Input validation** (Pydantic)

### API Key Usage

Include the API key in the `X-API-Key` header:

```bash
curl -X POST http://localhost:8000/api/v1/predict \
  -H "X-API-Key: your-api-key-here" \
  -H "Content-Type: application/json" \
  -d '{"transaction_id": "TXN-001", "features": [0.5, 0.3, 0.8]}'
```

Configure API keys via environment variables:
```bash
REQUIRE_API_KEY=true
API_KEYS=key1,key2,key3
```

## Deployment

### Azure Container Registry

```bash
az acr login --name <registry-name>
docker tag fraud-detection-api:latest <registry>.azurecr.io/fraud-detection-api:latest
docker push <registry>.azurecr.io/fraud-detection-api:latest
```

### Kubernetes

The API includes health checks for Kubernetes:

```yaml
livenessProbe:
  httpGet:
    path: /health
    port: 8000
  initialDelaySeconds: 30
  periodSeconds: 10

readinessProbe:
  httpGet:
    path: /health
    port: 8000
  initialDelaySeconds: 10
  periodSeconds: 5
```

## Development

### Code Quality

```bash
# Type checking
mypy src/

# Linting
pylint src/

# Format check
black --check src/
```

### Adding New Features

1. Create feature branch: `git checkout -b feature/my-feature`
2. Make changes and add tests
3. Run: `pytest tests/ -v --cov=src`
4. Submit pull request

## Project Structure

```
api/
├── src/
│   ├── main.py                 # FastAPI application
│   ├── config/                 # Configuration
│   ├── api/                    # Routes and middleware
│   ├── services/               # Business logic
│   │   ├── alert_service.py    # SMTP alerts
│   │   ├── model_versions.py   # Model registry
│   │   └── ...
│   └── models/                 # Data models
├── tests/                      # Test suite
│   ├── unit/                   # Unit tests
│   └── integration/            # Integration tests
├── requirements.txt            # Python dependencies
├── Dockerfile                  # Container configuration
└── README.md                   # This file
```

## Support

For issues or questions:
1. Check the troubleshooting section
2. Review logs: `docker logs fraud-api`
3. Open an issue in the repository

## License

See LICENSE file.

## Contributors

Fraud Detection Team

