# Airflow Configuration for Azure Shared Storage

## Airflow Variables to Set

In Airflow UI → Admin → Variables, add:

- **Key**: `azure_storage_account_key`
- **Value**: `<your-azure-storage-account-key>`

## Environment Variables in Docker Compose

Update your `docker-compose.yml` or environment files to include:

```yaml
environment:
  - AZURE_STORAGE_ACCOUNT=joshfraudstorageaccount
  - AZURE_STORAGE_ACCOUNT_KEY=${AZURE_STORAGE_ACCOUNT_KEY}
  - AZURE_STORAGE_MOUNT_PATH=/mnt/fraud-models
```

## Testing

After configuration, test the deployment scripts manually:

```bash
# Test canary deployment
docker run --rm \
  -e AZURE_STORAGE_ACCOUNT=joshfraudstorageaccount \
  -e AZURE_STORAGE_ACCOUNT_KEY=<key> \
  -e AZURE_STORAGE_MOUNT_PATH=/mnt/fraud-models \
  your-api-image \
  python scripts/deploy_canary.py --traffic 5 --model-uris "models:/fraud_detection_xgboost/Staging"

# Test promotion
docker run --rm \
  -e AZURE_STORAGE_ACCOUNT=joshfraudstorageaccount \
  -e AZURE_STORAGE_ACCOUNT_KEY=<key> \
  -e AZURE_STORAGE_MOUNT_PATH=/mnt/fraud-models \
  your-api-image \
  python scripts/promote_to_production.py --model-uris "models:/fraud_detection_xgboost/Staging"
```