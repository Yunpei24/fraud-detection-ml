# Azure Deployment Guide

This guide explains how to configure automatic API deployment to Azure Web App.

---

## Prerequisites

* Active Azure account and subscription
* Docker Hub account
* Azure CLI installed locally
* Permissions to create Azure resources

---

## Step 1: Create Azure Credentials

### Option A: Using Azure CLI (Recommended)

```bash
# 1. Log in to Azure
az login

# 2. Get your Subscription ID
SUBSCRIPTION_ID=$(az account show --query id -o tsv)
echo "Subscription ID: $SUBSCRIPTION_ID"

# 3. Create a Resource Group
az group create \
  --name fraud-detection-rg \
  --location westeurope

# 4. Create a Service Principal
az ad sp create-for-rbac \
  --name "fraud-detection-github-actions" \
  --role contributor \
  --scopes /subscriptions/$SUBSCRIPTION_ID/resourceGroups/fraud-detection-rg \
  --sdk-auth
```

**Output (SAVE IT!)**

```json
{
  "clientId": "abcd1234-...",
  "clientSecret": "your-secret-here",
  "subscriptionId": "12345678-...",
  "tenantId": "87654321-...",
  ...
}
```

---

## Step 2: Create the Azure Web App

```bash
# 1. Create an App Service Plan (Linux)
az appservice plan create \
  --name fraud-api-plan \
  --resource-group fraud-detection-rg \
  --is-linux \
  --sku B2

# 2. Create the Web App with Docker
az webapp create \
  --name fraud-detection-api \
  --resource-group fraud-detection-rg \
  --plan fraud-api-plan \
  --deployment-container-image-name jyen24/api:latest

# 3. Configure the port
az webapp config appsettings set \
  --name fraud-detection-api \
  --resource-group fraud-detection-rg \
  --settings WEBSITES_PORT=8000

# 4. Enable logs
az webapp log config \
  --name fraud-detection-api \
  --resource-group fraud-detection-rg \
  --docker-container-logging filesystem \
  --level information
```

---

## Step 3: Configure GitHub Secrets

### On GitHub:

1. Go to **Your repository → Settings → Secrets and variables → Actions**
2. Click **"New repository secret"**

### Create the following secrets:

| Secret Name            | Description                      | Example                                   |
| ---------------------- | -------------------------------- | ----------------------------------------- |
| `AZURE_CREDENTIALS`    | Full JSON from Service Principal | `{"clientId":"...","clientSecret":"..."}` |
| `AZURE_RESOURCE_GROUP` | Name of the resource group       | `fraud-detection-rg`                      |
| `DOCKERHUB_USERNAME`   | Your Docker Hub username         | `jyen24`                                  |
| `DOCKERHUB_TOKEN`      | Docker Hub access token          | `dckr_pat_...`                            |

### How to get the Docker Hub Token:

1. Go to [Docker Hub](https://hub.docker.com)
2. **Account Settings → Security → New Access Token**
3. Description: `GitHub Actions`
4. Permissions: `Read, Write, Delete`
5. Copy the token (it will not be shown again)

---

## Step 4: Configure Azure Environment Variables

```bash
# API environment variables
az webapp config appsettings set \
  --name fraud-detection-api \
  --resource-group fraud-detection-rg \
  --settings \
    ENVIRONMENT=production \
    POSTGRES_HOST=your-postgres-server.postgres.database.azure.com \
    POSTGRES_PORT=5432 \
    POSTGRES_DB=fraud_detection \
    POSTGRES_USER=fraud_user \
    POSTGRES_PASSWORD='your-secure-password' \
    REDIS_HOST=your-redis-cache.redis.cache.windows.net \
    REDIS_PORT=6380 \
    REDIS_PASSWORD='your-redis-password' \
    MLFLOW_TRACKING_URI=https://your-mlflow-server.azurewebsites.net \
    JWT_SECRET_KEY='your-jwt-secret-key' \
    JWT_ALGORITHM=HS256 \
    ACCESS_TOKEN_EXPIRE_MINUTES=1800 \
    LOG_LEVEL=INFO \
    CORS_ORIGINS='["https://your-frontend.com"]'
```

---

## Step 5: Test the Workflow

### Automatic Deployment

The workflow is triggered automatically when the **CI/CD Pipeline workflow completes successfully** on the `main` branch:

1. **Push to main**

```bash
git checkout main
git merge develop
git push origin main
```

This will:
- ✅ Trigger the `CI/CD Pipeline` workflow
- ⏳ Run code quality checks, unit tests, and Docker builds
- ✅ If successful, automatically trigger `Deploy API to Azure Web App`
- 🚀 Deploy the validated Docker image to Azure

2. **Pull Request merged into main**

* Create a PR from `develop` → `main`
* Merge the PR
* The CI/CD workflow will run first
* On success, the deployment workflow will run automatically

**Important:** The deployment will **only** run if the CI/CD Pipeline completes successfully. If tests fail or Docker builds fail, the deployment will be skipped to ensure only validated code reaches production.

### Manual Deployment

1. On GitHub, go to the **Actions** tab
2. Select **"Deploy API to Azure Web App"**
3. Click **Run workflow**, choose the `main` branch
4. Click **"Run workflow"**

---

## Step 6: Verify the Deployment

### 1. Check workflow status:

```
GitHub → Actions → Deploy API to Azure Web App → View the run
```

### 2. Test the deployed API:

```bash
# Health check
curl https://fraud-detection-api.azurewebsites.net/health

# API docs
open https://fraud-detection-api.azurewebsites.net/docs

# Metrics
curl https://fraud-detection-api.azurewebsites.net/metrics
```

### 3. View Azure logs:

```bash
# Real-time logs
az webapp log tail \
  --name fraud-detection-api \
  --resource-group fraud-detection-rg

# Download the last 100 lines
az webapp log download \
  --name fraud-detection-api \
  --resource-group fraud-detection-rg \
  --log-file logs.zip
```

---

## Workflow Jobs

The deployment workflow includes two jobs (the Docker image is already built by CI/CD):

### Job 1: Deploy

* Waits for CI/CD Pipeline to complete successfully
* Logs in to Azure
* Configures container with the latest validated image
* Updates app settings
* Restarts Web App
* Performs health check (10 attempts)

### Job 2: Test

* Tests endpoints (`/health`, `/docs`, `/metrics`)
* Sends success notification

**Note:** The Docker image build is handled by the `CI/CD Pipeline` workflow, ensuring that only tested and validated images are deployed.

---

## Workflow Triggers

```yaml
on:
  # Triggered after CI/CD Pipeline completes successfully
  workflow_run:
    workflows: ["CI/CD Pipeline"]
    types:
      - completed
    branches:
      - main
  
  # Manual deployment option
  workflow_dispatch:
```

**How it works:**

1. Any push or PR merge to `main` triggers the `CI/CD Pipeline` workflow
2. The CI/CD workflow runs tests, builds, and pushes Docker images
3. **Only if CI/CD succeeds**, the `Deploy API to Azure Web App` workflow is triggered
4. The deployment uses the Docker image that was just built and validated

The workflow does **not** trigger on:

* Push to `develop` or other branches
* Open (unmerged) pull requests
* Changes in unrelated folders (`airflow/`, `drift/`, etc.)

---

## Troubleshooting

### Error: "AZURE_CREDENTIALS not found"

Check that the secret is properly configured:

```
Settings → Secrets and variables → Actions → AZURE_CREDENTIALS
```

### Error: "Web App not found"

Make sure the Web App exists:

```bash
az webapp show \
  --name fraud-detection-api \
  --resource-group fraud-detection-rg
```

### Error: "Health check failed"

View logs:

```bash
az webapp log tail \
  --name fraud-detection-api \
  --resource-group fraud-detection-rg
```

### Docker image not updating

Force a pull of the latest image:

```bash
az webapp restart \
  --name fraud-detection-api \
  --resource-group fraud-detection-rg
```

---

## Estimated Costs

| Service     | SKU                 | Monthly Cost    |
| ----------- | ------------------- | --------------- |
| App Service | B2 (2 vCPU, 3.5 GB) | ~€55            |
| PostgreSQL  | Basic (1 vCore)     | ~€30            |
| Redis Cache | Basic C0 (250 MB)   | ~€15            |
| **Total**   |                     | **~€100/month** |

---

## Resources

* [Azure Web App Documentation](https://learn.microsoft.com/en-us/azure/app-service/)
* [GitHub Actions Azure Login](https://github.com/marketplace/actions/azure-login)
* [Docker Hub](https://hub.docker.com)

---

## Deployment Checklist

* [ ] Azure CLI installed and connected
* [ ] Service Principal created
* [ ] Resource Group created
* [ ] App Service Plan created
* [ ] Web App created
* [ ] GitHub Secrets configured

  * [ ] `AZURE_CREDENTIALS`
  * [ ] `AZURE_RESOURCE_GROUP`
  * [ ] `DOCKERHUB_USERNAME`
  * [ ] `DOCKERHUB_TOKEN`
* [ ] Azure environment variables configured
* [ ] PostgreSQL database created
* [ ] Redis Cache created
* [ ] GitHub Actions workflow tested
* [ ] Health check passed
* [ ] API publicly accessible

---

## Successful Deployment

Once all checks are passed, your API will be available at:

```
https://fraud-detection-api.azurewebsites.net
```

Interactive documentation:

```
https://fraud-detection-api.azurewebsites.net/docs
```
