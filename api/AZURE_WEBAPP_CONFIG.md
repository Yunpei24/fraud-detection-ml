# Azure Web App Configuration for Shared Storage (Option 1)

## Environment Variables to Set in Azure Web App

### Storage Configuration
AZURE_STORAGE_ACCOUNT=joshfraudstorageaccount
AZURE_STORAGE_ACCOUNT_KEY=<your-storage-account-key>
AZURE_STORAGE_MOUNT_PATH=/mnt/fraud-models

### Model Path Override
MODEL_PATH=/mnt/fraud-models

## Azure Web App Configuration Steps

### 1. Configure Path Mappings
In Azure Portal → Your Web App → Configuration → Path mappings:

- **Name**: fraud-models
- **Path**: /mnt/fraud-models
- **Account name**: joshfraudstorageaccount
- **Share name**: fraud-models
- **Access key**: <your-storage-account-key>
- **Type**: Azure Files

### 2. Environment Variables
In Azure Portal → Your Web App → Configuration → Application settings:

Add these environment variables:
- AZURE_STORAGE_ACCOUNT = joshfraudstorageaccount
- AZURE_STORAGE_ACCOUNT_KEY = <your-storage-account-key>
- AZURE_STORAGE_MOUNT_PATH = /mnt/fraud-models
- MODEL_PATH = /mnt/fraud-models

### 3. Verify Configuration
After deployment, check that:
- The mount path `/mnt/fraud-models` exists in the container
- Models can be written to and read from this path
- The traffic router uses the correct paths

## Testing the Configuration

Run the canary deployment test:
```bash
python test_canary_deployment.py
```

This will verify that:
1. Models can be saved to Azure File Share
2. Traffic routing works correctly
3. Model loading from shared storage works