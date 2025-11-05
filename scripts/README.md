# Scripts Directory

This directory contains utility scripts for building, deploying, and managing the Fraud Detection ML project across different environments.

## Available Scripts

### 1. `build-local.sh`
**Purpose**: Build and manage Docker services for local development

**Description**: Interactive script that provides a menu-driven interface for managing the local development environment using `docker-compose.local.yml`.

**Features**:
- Pull latest Docker images
- Build all services
- Start all services
- Stop all services
- Check service health
- View service logs
- View resource usage
- Clean up Docker resources
- Exit

**Usage**:
```bash
./scripts/build-local.sh
```

**Prerequisites**:
- Docker and Docker Compose installed
- `docker-compose.local.yml` configured
- Network connectivity for pulling images

---

### 2. `build-production.sh`
**Purpose**: Interactive management tool for VM1 and VM2 production environments

**Description**: Menu-driven script that provides comprehensive operations for both VM1 (application services) and VM2 (monitoring services). Offers a unified interface for deployment, health checks, log viewing, and cleanup operations.

**Features**:
- **Interactive Menu**: Select VM1, VM2, or exit
- **VM1 Operations**:
  - Pull latest Docker images
  - Start application services
  - Stop application services
  - Check service health
  - View service logs
  - Clean up Docker resources
- **VM2 Operations**:
  - Same operations as VM1
  - Additional Prometheus configuration validation
  - Automatic placeholder detection (`<VM1_PUBLIC_OR_PRIVATE_IP>`)
- **Color-coded Output**: Visual feedback for success, warnings, and errors
- **Automatic Validation**: Checks for required files before operations

**Usage**:
```bash
./scripts/build-production.sh
```

**Prerequisites**:
- Docker and Docker Compose installed
- `docker-compose.vm1.yml` configured (for VM1 operations)
- `docker-compose.vm2.yml` configured (for VM2 operations)
- `monitoring/prometheus.vm2.yml` configured with actual VM1 IP (for VM2)
- Docker Hub credentials configured

**Important for VM2**:
Before starting VM2 services, ensure `monitoring/prometheus.vm2.yml` has been updated with actual VM1 IP addresses. The script will warn you if placeholders are detected.

---

### 3. `deploy-production.sh`
**Purpose**: Automated production deployment with CLI flags

**Description**: Command-line deployment script that supports automated deployments for VM1, VM2, or both environments. Designed for CI/CD pipelines and scripted deployments.

**Usage**:
```bash
# Deploy VM1 only
./scripts/deploy-production.sh --vm1

# Deploy VM2 only
./scripts/deploy-production.sh --vm2

# Deploy both VMs
./scripts/deploy-production.sh --both

# Validate configuration without deploying
./scripts/deploy-production.sh --validate
```

**Features**:
- CLI flag-based operation (non-interactive)
- Validation mode for configuration testing
- Color-coded logging
- Automatic health checks after deployment
- Suitable for automation and CI/CD

**Prerequisites**:
- Same as `build-production.sh`
- Suitable for automated workflows

---

### 4. `deploy-vm1.sh`
**Purpose**: Deploy application services on VM1

**Description**: Production deployment script for VM1 that handles all application-layer services including databases, message queues, ML services, and orchestration.

**Services Deployed**:
- PostgreSQL (database)
- Redis (caching)
- Kafka & Zookeeper (event streaming)
- MLflow (model registry)
- Data pipeline service
- Drift detection service
- Training service
- Airflow (workflow orchestration)

**Usage**:
```bash
# On VM1
./scripts/deploy-vm1.sh
```

**Prerequisites**:
- Docker and Docker Compose installed on VM1
- `docker-compose.vm1.yml` configured
- Docker Hub credentials configured
- Environment variables set (see `.env.vm1`)

**Azure Requirements**:
- VM1 must be running in your Azure subscription
- Required ports open in Azure NSG
- VM1 must have internet access for Docker Hub

---

### 5. `deploy-vm2.sh`
**Purpose**: Deploy monitoring services on VM2

**Description**: Production deployment script for VM2 that handles all monitoring infrastructure including metrics collection, visualization, alerting, and system metrics.

**Services Deployed**:
- Prometheus (metrics collection)
- Grafana (visualization dashboards)
- Alertmanager (alert management)
- Node Exporter (system metrics)

**Usage**:
```bash
# On VM2
./scripts/deploy-vm2.sh
```

**Prerequisites**:
- Docker and Docker Compose installed on VM2
- `docker-compose.vm2.yml` configured
- `monitoring/prometheus.vm2.yml` configured with correct VM1 IP addresses
- Docker Hub credentials configured

**Azure Requirements**:
- VM2 must be running in colleague's Azure subscription
- Required ports open in Azure NSG (9090, 3001, 9093, 9100)
- VM2 must have network access to VM1 for metrics scraping
- VM2 must have internet access for Docker Hub

**Important Configuration**:
Before running, update `monitoring/prometheus.vm2.yml` with:
- Replace `<VM1_PUBLIC_OR_PRIVATE_IP>` with actual VM1 IP address
- Verify all target ports match VM1 services

---

## Script Comparison

| Feature | build-local.sh | build-production.sh | deploy-production.sh | deploy-vm1.sh | deploy-vm2.sh |
|---------|----------------|---------------------|----------------------|---------------|---------------|
| **Environment** | Local dev | VM1/VM2 prod | VM1/VM2 prod | VM1 only | VM2 only |
| **Interface** | Interactive menu | Interactive menu | CLI flags | Non-interactive | Non-interactive |
| **Use Case** | Development | Manual production ops | Automated deployment | VM1-specific deploy | VM2-specific deploy |
| **Automation-friendly** | No | No | Yes | Yes | Yes |
| **Multi-VM support** | N/A | Yes (menu selection) | Yes (CLI flags) | No | No |

---

## Cross-VM Communication Setup

### VM2 → VM1 Metrics Collection
For Prometheus on VM2 to scrape metrics from services on VM1:

1. **Network Configuration**:
   - Ensure VM2 can reach VM1 (private network or public IPs)
   - Open required ports on VM1's NSG:
     - 9095 (training-service)
     - 9096 (data-service)
     - 9097 (drift-service)
     - 9098 (mlflow)
     - 443 (api-service HTTPS)

2. **Prometheus Configuration**:
   - Update `monitoring/prometheus.vm2.yml`
   - Replace placeholder IPs with actual VM1 IP
   - Verify scrape intervals and timeouts

3. **Testing**:
   - Run deploy-vm2.sh or build-production.sh (VM2 option)
   - Access Prometheus UI: `http://<VM2_IP>:9090`
   - Check "Status → Targets" - all should show "UP"

---

## Deployment Workflows

### Local Development
```bash
# Start local environment
./scripts/build-local.sh
# Select option 3 (Start all services)
```

### Production Deployment - Interactive (Manual Operations)
```bash
# SSH into VM1 or VM2
ssh user@<VM_IP>

# Navigate to project
cd fraud-detection-ml

# Run interactive script
./scripts/build-production.sh

# Select VM1 or VM2 from menu
# Choose operations (start, stop, logs, etc.)
```

### Production Deployment - Automated (CI/CD)
```bash
# Deploy both VMs
./scripts/deploy-production.sh --both

# Or deploy individually
./scripts/deploy-production.sh --vm1
./scripts/deploy-production.sh --vm2

# Validate configuration first
./scripts/deploy-production.sh --validate
```

### Production Deployment - VM1 Specific
```bash
# SSH into VM1
ssh user@<VM1_IP>

# Navigate to project
cd fraud-detection-ml

# Pull latest code
git pull origin main

# Deploy services
./scripts/deploy-vm1.sh
```

### Production Deployment - VM2 Specific
```bash
# SSH into VM2
ssh user@<VM2_IP>

# Navigate to project
cd fraud-detection-ml

# Pull latest code
git pull origin main

# Update Prometheus configuration with VM1 IP
vim monitoring/prometheus.vm2.yml

# Deploy monitoring
./scripts/deploy-vm2.sh

# Access Grafana
# http://<VM2_IP>:3001
# Default credentials: admin/admin
```

---

## Troubleshooting

### Common Issues

1. **Docker Hub Rate Limits**:
   ```bash
   # Login to Docker Hub
   docker login
   # Enter credentials stored in GitHub Secrets
   ```

2. **Port Conflicts**:
   ```bash
   # Check if port is in use
   sudo lsof -i :<PORT>
   # Kill process if needed
   sudo kill -9 <PID>
   ```

3. **VM2 Cannot Reach VM1**:
   - Verify NSG rules on VM1
   - Test connectivity: `telnet <VM1_IP> 9095`
   - Check if services are bound to 0.0.0.0 (not 127.0.0.1)
   - Use `build-production.sh` VM2 menu to validate Prometheus config

4. **Prometheus Targets Down**:
   - Check `monitoring/prometheus.vm2.yml` IP addresses
   - Verify services are running on VM1
   - Check VM1 firewall rules
   - Review Prometheus logs: `docker logs prometheus`
   - Run `build-production.sh` → VM2 → View logs → prometheus

5. **Container Startup Failures**:
   ```bash
   # View logs
   docker-compose -f <compose-file> logs <service-name>
   
   # Restart specific service
   docker-compose -f <compose-file> restart <service-name>
   ```

6. **Placeholder IPs Not Replaced**:
   - `build-production.sh` will warn about `<VM1_PUBLIC_OR_PRIVATE_IP>` placeholders
   - Update `monitoring/prometheus.vm2.yml` before starting VM2 services
   - Use find/replace: `:<VM1_PUBLIC_OR_PRIVATE_IP>:` → `:<actual_IP>:`

---

## Maintenance

### Updating Services
```bash
# Interactive (build-production.sh)
./scripts/build-production.sh
# Select VM → Option 1 (Pull latest images)
# Then Option 2 (Start services)

# Automated
./scripts/deploy-production.sh --both

# Manual
docker-compose -f <compose-file> pull
docker-compose -f <compose-file> up -d
```

### Viewing Logs
```bash
# Using build-production.sh (interactive)
./scripts/build-production.sh
# Select VM → Option 5 (View logs)

# Manual
docker-compose -f <compose-file> logs -f
docker-compose -f <compose-file> logs -f <service-name>
docker-compose -f <compose-file> logs --tail=100 <service-name>
```

### Resource Cleanup
```bash
# Using build-production.sh (interactive)
./scripts/build-production.sh
# Select VM → Option 6 (Clean up)

# Manual
docker-compose -f <compose-file> down
docker-compose -f <compose-file> down -v  # ⚠️ deletes data
docker system prune -a
```

---

## Security Notes

- **Credentials**: Never commit credentials to Git
- **Environment Files**: Use `.env.vm1` and `.env.vm2` (gitignored)
- **Docker Hub**: Credentials stored in GitHub Secrets
- **VM Access**: Use SSH key authentication, disable password auth
- **Network**: Use Azure Private Network when possible
- **Ports**: Only expose necessary ports in NSG rules

---

## Additional Resources

- **Docker Compose Files**: Root directory (`docker-compose.*.yml`)
- **Prometheus Config**: `monitoring/prometheus.yml` (local), `monitoring/prometheus.vm2.yml` (production)
- **Environment Setup**: See `ENVIRONMENT_CONFIG.md` in respective service directories
- **Monitoring Guide**: `MONITORING_CORRECTIONS_REPORT.md`
- **VM Communication**: `Guide/COMM_VM1_VM2.md`