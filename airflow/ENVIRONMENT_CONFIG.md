# Environment Configuration for Airflow DAGs

## Overview

The Airflow DAGs now use centralized configuration that automatically adapts to different environments (local development vs production). This eliminates hardcoded values and makes deployment seamless.

## Environment Detection

The system automatically detects the environment using the `ENVIRONMENT` environment variable:

- `ENVIRONMENT=local` (default): Uses local Docker images
- `ENVIRONMENT=production`: Uses Azure Container Registry images

## Docker Images by Environment

### Local Development (`ENVIRONMENT=local`)
```bash
fraud-detection/training:local
fraud-detection/api:local
fraud-detection/drift:local
fraud-detection/data-quality:local
```

### Production (`ENVIRONMENT=production`)
```bash
{your-registry}/training:latest
{your-registry}/api:latest
{your-registry}/drift:latest
{your-registry}/data-quality:latest
```

## Configuration

### For Local Development
```bash
export ENVIRONMENT=local
# No additional configuration needed
```

### For Production
```bash
export ENVIRONMENT=production
export DOCKERHUB_USERNAME=yoshua24
```

### In Docker Compose (Local)
```yaml
environment:
  - ENVIRONMENT=local
```

### In Azure/AWS (Production)
```yaml
environment:
  - ENVIRONMENT=production
  - DOCKERHUB_USERNAME=yoshua24
```

## Centralized Constants

All DAGs now import from `config.constants`:

```python
from config.constants import (
    ENV_VARS, DOCKER_NETWORK,
    DOCKER_IMAGE_TRAINING, DOCKER_IMAGE_API,
    TABLE_NAMES, SCHEDULES, THRESHOLDS
)
```

## Benefits

1. **No Hardcoded Values**: All configuration is centralized
2. **Environment Agnostic**: Same code works in local and production
3. **Easy Deployment**: Just change environment variables
4. **Maintainable**: Single source of truth for all constants
5. **Type Safe**: Constants are validated and documented

## Migration Complete

✅ All DAGs now use centralized constants
✅ Environment-aware Docker images
✅ No more hardcoded registry URLs
✅ Consistent configuration across all pipelines