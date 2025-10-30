# Scripts Directory

This directory contains all shell scripts for building, deploying, and managing the Fraud Detection ML system.

## 📋 Available Scripts

### Local Development Scripts

#### `build-local.sh`
**Purpose**: Complete local development environment management
**Usage**: `./scripts/build-local.sh`
**Features**:
- Build all Docker images
- Start/stop all services
- Health checks for all services
- Full test suite (build + deploy + verify)
- Service logs and status monitoring

**Options**:
1. Build all images
2. Build specific image
3. Start all services
4. Start specific services
5. Stop all services
6. View logs
7. Clean up (remove volumes)
8. Full test (build + up + health checks)

### Production Deployment Scripts

#### `deploy-vm1.sh`
**Purpose**: Deploy application services on VM1 (your Azure subscription)
**Usage**: `bash scripts/deploy-vm1.sh` (run on VM1)
**Deploys**:
- PostgreSQL, Redis, Kafka, MLflow
- Data, Drift, Training services
- Airflow orchestration
- All application components

**Options**:
1. Full deployment (build + deploy)
2. Build images only
3. Deploy services only
4. Stop services
5. View status
6. View logs
7. Clean up
8. Health check

#### `deploy-vm2.sh`
**Purpose**: Deploy monitoring services on VM2 (colleague's Azure subscription)
**Usage**: `bash scripts/deploy-vm2.sh` (run on VM2)
**Deploys**:
- Prometheus, Grafana, Alertmanager
- Node Exporter for infrastructure metrics
- Complete monitoring stack

**Options**:
1. Full deployment
2. Deploy services only
3. Stop services
4. View status
5. View logs
6. Clean up
7. Health check
8. Configure VM1 IP

### Health Check Scripts

#### `vm1-health-check.sh`
**Purpose**: Comprehensive health checks for VM1 services
**Usage**: `bash scripts/vm1-health-check.sh`
**Checks**:
- Infrastructure: PostgreSQL, Redis, Kafka, Zookeeper
- Applications: MLflow, Data, Drift, Training services
- Orchestration: Airflow Webserver, Scheduler
- Metrics: External accessibility for VM2 scraping

#### `vm2-health-check.sh`
**Purpose**: Comprehensive health checks for VM2 monitoring services
**Usage**: `bash scripts/vm2-health-check.sh`
**Checks**:
- Monitoring: Prometheus, Grafana, Alertmanager, Node Exporter
- Connectivity: Cross-subscription access to VM1 services
- Targets: Prometheus target health
- Dashboards: Grafana dashboard provisioning
- Alerts: Alert rule loading

## 🚀 Quick Start

### Local Development
```bash
# From project root
./scripts/build-local.sh
# Choose option 8 for full test
```

### Production Deployment

**VM1 (Application Services)**:
```bash
# On VM1
git clone <repo>
cd fraud-detection-ml
bash scripts/deploy-vm1.sh
# Choose option 1 for full deployment
```

**VM2 (Monitoring Services)**:
```bash
# On VM2
git clone <repo>
cd fraud-detection-ml
bash scripts/deploy-vm2.sh
# Choose option 8 first to configure VM1 IP
# Then choose option 1 for full deployment
```

## 📊 Service URLs

### Local Development
- **API**: http://localhost:8000/docs
- **MLflow**: http://localhost:5000
- **Airflow**: http://localhost:8080 (admin/admin)
- **Prometheus**: http://localhost:9094
- **Grafana**: http://localhost:3000 (admin/admin_dev_2024)
- **Alertmanager**: http://localhost:9093

### Production
- **VM1 Services**: http://`<VM1_IP>`:8080 (Airflow)
- **VM2 Monitoring**: http://`<VM2_IP>`:3000 (Grafana)

## 🔧 Prerequisites

### Local Development
- Docker and Docker Compose
- Bash shell
- curl, jq (for health checks)

### Production VMs
- Ubuntu 20.04+ or similar Linux
- Docker and Docker Compose
- Git
- curl, jq, sed
- Network access between VM1 and VM2

## 🏗️ Architecture

### Local Development
All services run on a single machine using Docker Compose with local networking.

### Production
- **VM1**: Application services (databases, APIs, processing)
- **VM2**: Monitoring services (Prometheus, Grafana, alerts)
- Cross-subscription communication via public IPs

## 🔒 Security Notes

- Change all default passwords in production
- Use strong encryption keys
- Configure NSG rules properly for cross-subscription access
- Regular security updates and monitoring

## 🐛 Troubleshooting

### Common Issues
1. **Permission denied**: `chmod +x scripts/*.sh`
2. **Docker not running**: Start Docker Desktop/service
3. **Port conflicts**: Check `docker ps` for conflicts
4. **VM connectivity**: Verify NSG rules and public IPs

### Logs
```bash
# Local
docker-compose -f docker-compose.local.yml logs -f

# VM1
docker-compose -f docker-compose.vm1.yml logs -f

# VM2
docker-compose -f docker-compose.vm2.yml logs -f
```

### Health Checks
```bash
# Local/VM1
bash scripts/vm1-health-check.sh

# VM2
bash scripts/vm2-health-check.sh
```

## 📝 Maintenance

- Run health checks regularly
- Monitor disk space and resources
- Update Docker images periodically
- Backup volumes before major changes
- Test deployments in staging first