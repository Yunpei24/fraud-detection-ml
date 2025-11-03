# Fraud Detection ML System

A comprehensive machine learning system for fraud detection with MLOps capabilities, deployed across multiple Azure VMs with cross-subscription monitoring.

## 🏗️ Architecture

### Local Development
All services run on a single machine using Docker Compose with local networking.

### Production Deployment
- **VM1** (Your Azure subscription): Application services, databases, APIs
- **VM2** (Colleague's Azure subscription): Monitoring stack, dashboards, alerts
- Cross-subscription communication via public IPs with proper security

## 🚀 Quick Start

### ⚡ 5-Minute Setup (Local Development)

```bash
# 1. Clone repository
git clone https://github.com/Yunpei24/fraud-detection-ml.git
cd fraud-detection-ml

# 2. Download dataset (144 MB)
# Get from: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
# Place at: ./creditcard.csv

# 3. Create environment file
cp .env.production .env

# 4. Start services (builds images automatically)
./scripts/build-local.sh
# Choose option 8 (Full build + deployment + health checks)

# 5. Wait 5-10 minutes for first build, then access:
# - API:     http://localhost:8000/docs
# - MLflow:  http://localhost:5001
# - Airflow: http://localhost:8080 (admin/admin)
# - Grafana: http://localhost:3000 (admin/admin_dev_2024)
```

### 🎯 What Gets Deployed

**Local Development** (`docker-compose.local.yml`):
- ✅ All infrastructure (PostgreSQL, Redis, Kafka, Zookeeper)
- ✅ MLflow tracking server
- ✅ FastAPI application
- ✅ Data pipeline service
- ✅ Drift detection service
- ✅ Model training service
- ✅ Airflow orchestration (webserver, scheduler, worker)
- ✅ Complete monitoring stack (Prometheus, Grafana, Alertmanager)

**Total Services**: 16 containers

### 🐳 Docker Image Strategy

The project supports two approaches:

| Strategy | Use Case | Time | Network | Disk |
|----------|----------|------|---------|------|
| **Local Build** | Development, offline work, code changes | 15-30 min first time, then ~2 min | Base images only (~500 MB) | ~10 GB |
| **Registry Pull** | Production, quick setup, no code changes | 10-20 min | Full images (~12 GB total) | ~12 GB |

**Default**: Local builds are configured in `docker-compose.local.yml`

### 📋 Pre-flight Checklist

Before starting, verify:

```bash
# ✅ Docker is running
docker info

# ✅ Sufficient resources
docker system df
df -h  # Need 50+ GB free

# ✅ No port conflicts
lsof -i :8000  # API
lsof -i :5001  # MLflow
lsof -i :8080  # Airflow
lsof -i :3000  # Grafana

# ✅ Dataset exists
ls -lh creditcard.csv  # Should show ~144 MB file
```

### Local Development
```bash
# Clone repository
git clone <repository-url>
cd fraud-detection-ml

# Start full local environment
./scripts/build-local.sh
# Choose option 8 for complete build + deployment + health checks
```

### Production Deployment

**VM1 (Application Services)**:
```bash
# On VM1
git clone <repository-url>
cd fraud-detection-ml
bash scripts/deploy-vm1.sh
# Choose option 1 for full deployment
```

**VM2 (Monitoring Services)**:
```bash
# On VM2
git clone <repository-url>
cd fraud-detection-ml
bash scripts/deploy-vm2.sh
# Choose option 8 to configure VM1 IP first
# Then choose option 1 for full deployment
```

## 📊 Service URLs

### Local Development
- **API**: http://localhost:8000/docs (FastAPI with Swagger)
- **MLflow**: http://localhost:5000 (Model registry & experiments)
- **Airflow**: http://localhost:8080 (admin/admin) (Workflow orchestration)
- **Prometheus**: http://localhost:9094 (Metrics collection)
- **Grafana**: http://localhost:3000 (admin/admin_dev_2024) (Dashboards)
- **Alertmanager**: http://localhost:9093 (Alert management)

### Production
- **VM1 Services**: http://`<VM1_IP>`:8080 (Airflow)
- **VM2 Monitoring**: http://`<VM2_IP>`:3000 (Grafana)

## 📁 Project Structure

```
fraud-detection-ml/
├── api/                    # FastAPI application
├── data/                   # Data ingestion service
├── drift/                  # Model drift detection
├── training/               # Model training service
├── airflow/                # Workflow orchestration (DAGs, configs)
├── monitoring/             # Prometheus, Grafana, Alertmanager configs
├── scripts/                # Deployment and management scripts
├── docker-compose.local.yml # Local development stack
├── docker-compose.vm1.yml   # VM1 production services
├── docker-compose.vm2.yml   # VM2 monitoring services
├── .env.production         # Production environment variables
└── requirements.txt        # Python dependencies
```

## 🔧 Available Scripts

See [`scripts/README.md`](scripts/README.md) for detailed documentation.

- `scripts/build-local.sh` - Local development environment management
- `scripts/deploy-vm1.sh` - VM1 production deployment
- `scripts/deploy-vm2.sh` - VM2 monitoring deployment
- `scripts/vm1-health-check.sh` - VM1 service health checks
- `scripts/vm2-health-check.sh` - VM2 monitoring health checks

## 📈 Monitoring & Dashboards

The system includes comprehensive monitoring with 4 Grafana dashboards:

1. **System Overview** - Service health, throughput, alerts
2. **API Performance** - Request metrics, latency, errors, predictions
3. **Data Pipeline** - Processing performance, quality metrics, queues
4. **Drift Detection** - Model monitoring, performance, retraining triggers

### Alert Rules
- 20+ automated alerts for service health, performance, and business metrics
- Email, Slack, and webhook notifications
- Configurable thresholds and escalation policies

## 🛠️ Development

### Prerequisites

#### Required Software
- **Docker Desktop** (4.20+)
  - macOS: [Docker Desktop for Mac](https://docs.docker.com/desktop/install/mac-install/)
  - Windows: [Docker Desktop for Windows](https://docs.docker.com/desktop/install/windows-install/)
  - Linux: [Docker Engine](https://docs.docker.com/engine/install/)
- **Docker Compose** (2.0+) - Included with Docker Desktop
- **Git** (2.30+)
- **Bash shell** (zsh on macOS, bash on Linux)
- **Python 3.9+** (for local development without Docker)

#### System Requirements

**Minimum:**
- CPU: 4 cores
- RAM: 8 GB
- Disk: 50 GB free space
- Network: Stable broadband connection

**Recommended:**
- CPU: 8+ cores
- RAM: 16+ GB
- Disk: 100+ GB free space (SSD preferred)
- Network: High-speed connection for Docker image pulls

#### Docker Desktop Configuration

For optimal performance, configure Docker Desktop:

1. **macOS/Windows**: Docker Desktop → Preferences → Resources
   - **Memory**: 8 GB minimum, 12+ GB recommended
   - **CPUs**: 4 minimum, 6+ recommended
   - **Disk image size**: 60 GB minimum, 100+ GB recommended
   - **Swap**: 2 GB

2. **Enable BuildKit** (faster builds):
   ```bash
   export DOCKER_BUILDKIT=1
   export COMPOSE_DOCKER_CLI_BUILD=1
   ```

3. **Increase Docker Daemon Timeouts** (for large image pulls):
   
   Edit `~/.docker/daemon.json`:
   ```json
   {
     "max-concurrent-downloads": 3,
     "max-download-attempts": 5,
     "log-driver": "json-file",
     "log-opts": {
       "max-size": "10m",
       "max-file": "3"
     }
   }
   ```
   Then restart Docker Desktop.

#### Verify Installation
```bash
# Check Docker
docker --version
docker-compose --version
docker info

# Check available resources
docker system df
df -h

# Test Docker
docker run hello-world
```

### Environment Setup
```bash
# Copy environment file
cp .env.production .env

# Edit with your local settings
nano .env
```

### Building Services
```bash
# Build all services
./scripts/build-local.sh
# Choose option 1

# Build specific service
./scripts/build-local.sh
# Choose option 2, then select service
```

### Build Strategies

The project supports two build strategies:

#### 1. Local Build (Development)
Builds images locally from source code. Recommended for:
- Development and testing
- Making code changes
- Slow or unreliable internet connection
- First-time setup

```bash
# docker-compose.local.yml uses local builds:
# build:
#   context: .
#   dockerfile: ./api/Dockerfile
# image: fraud-detection/api:local
```

**Advantages:**
- No dependency on external registries
- Works offline (after initial base image pulls)
- Full control over build process
- Latest code changes

**Disadvantages:**
- Slower initial build (~10-20 minutes)
- Requires more disk space
- CPU-intensive

#### 2. Registry Pull (Production)
Pulls pre-built images from Docker Hub. Recommended for:
- Production deployment
- CI/CD pipelines
- Quick setup without source code changes
- Lower resource environments

```bash
# docker-compose.vm1.yml uses registry images:
# image: ${DOCKERHUB_USERNAME:-yoshua24}/api:latest
```

**Advantages:**
- Faster deployment
- Consistent builds across environments
- Less CPU/disk usage
- Pre-tested images

**Disadvantages:**
- Requires stable internet
- Dependent on registry availability
- Large image downloads (1-4 GB per service)

#### Switching Between Strategies

**To use local builds** (edit `docker-compose.local.yml`):
```yaml
api:
  build:
    context: .
    dockerfile: ./api/Dockerfile
  image: fraud-detection/api:local
  # Comment out or remove:
  # image: ${DOCKERHUB_USERNAME}/api:develop
```

**To use registry images**:
```yaml
api:
  image: ${DOCKERHUB_USERNAME:-yoshua24}/api:latest
  # Comment out build section
```

#### Hybrid Approach
```bash
# Pull most services, build only what changed
docker-compose -f docker-compose.local.yml pull postgres redis kafka
docker-compose -f docker-compose.local.yml build api data
docker-compose -f docker-compose.local.yml up -d
```

## 🚢 Production Deployment

### Pre-deployment Checklist
- [ ] Azure VMs provisioned (Ubuntu 20.04+)
- [ ] Docker installed on both VMs
- [ ] NSG rules configured for cross-subscription access
- [ ] Public IPs exchanged between team members
- [ ] Environment variables updated in `.env.production`

### Deployment Steps
1. **Coordinate with colleague** for IP exchange
2. **Deploy VM1** application services first
3. **Configure VM2** with VM1's public IP
4. **Deploy VM2** monitoring services
5. **Run health checks** on both VMs
6. **Configure alerts** and notifications

### Security Considerations
- Change all default passwords
- Use strong encryption keys
- Configure NSG rules for minimal required access
- Regular security updates
- Monitor access logs

## 🔍 Health Checks & Troubleshooting

### Automated Health Checks
```bash
# VM1 services
bash scripts/vm1-health-check.sh

# VM2 monitoring
bash scripts/vm2-health-check.sh
```

### Common Issues

#### 1. Docker Image Pull Errors

##### Error: `short read: expected X bytes but got Y: unexpected EOF`

**Description**: This error indicates that Docker image download was interrupted before completion.

**Root Causes**:
1. **Network Interruption**: Connection dropped during the ~3.5 GB download
2. **Timeout**: Download exceeded Docker daemon timeout (>1.5 hours)
3. **Disk Space**: Insufficient space to complete download
4. **Registry Issues**: Docker Hub/Registry server connection problems
5. **Memory Pressure**: System running out of RAM during extraction

**Immediate Solutions**:

```bash
# Solution 1: Clean and retry (Fastest)
docker system prune -a  # Remove partial downloads
./scripts/build-local.sh  # Retry

# Solution 2: Increase timeout and retry limits
# Edit ~/.docker/daemon.json (create if doesn't exist)
cat > ~/.docker/daemon.json << 'EOF'
{
  "max-concurrent-downloads": 3,
  "max-download-attempts": 5,
  "default-ulimits": {
    "nofile": {
      "Name": "nofile",
      "Hard": 64000,
      "Soft": 64000
    }
  }
}
EOF

# Restart Docker Desktop
# macOS: Restart from menu bar
# Linux: sudo systemctl restart docker

# Solution 3: Use local builds instead
# Edit docker-compose.local.yml - uncomment build sections
docker-compose -f docker-compose.local.yml build --no-cache
docker-compose -f docker-compose.local.yml up -d

# Solution 4: Pull images manually with retry
for i in {1..3}; do
  docker pull ${DOCKERHUB_USERNAME:-yoshua24}/api:latest && break
  echo "Retry $i failed, waiting 30s..."
  sleep 30
done

# Solution 5: Download during off-peak hours
# Large images (3-4 GB each) download faster with good connection
```

**Prevention**:
```bash
# Monitor download progress
docker pull yoshua24/api:latest &
watch -n 2 'docker images | head -5'

# Check network stability
ping -c 10 google.com
speedtest-cli  # Install: brew install speedtest-cli

# Ensure sufficient disk space (need 60+ GB free)
docker system df
df -h

# Optimize Docker resource allocation
# Docker Desktop → Settings → Resources:
# - Memory: 8+ GB
# - Disk: 100+ GB
# - CPU: 4+ cores
```

**Alternative: Use Local Builds**
If downloads consistently fail, switch to local builds:

```bash
# 1. Edit docker-compose.local.yml
# Change from:
#   image: ${DOCKERHUB_USERNAME}/api:latest
# To:
#   build:
#     context: .
#     dockerfile: ./api/Dockerfile
#   image: fraud-detection/api:local

# 2. Build locally
docker-compose -f docker-compose.local.yml build

# Pros: No network dependency, faster after first build
# Cons: Initial build takes 15-30 minutes, more CPU/disk intensive
```

**Debug Information**:
```bash
# Check Docker daemon logs
# macOS: ~/Library/Containers/com.docker.docker/Data/log/
# Linux: sudo journalctl -u docker.service

# Check available space
docker system df -v

# Check network
curl -I https://hub.docker.com

# Test single image pull
time docker pull yoshua24/api:latest
```

#### 2. Permission Issues
**Error**: `Permission denied` when running scripts

**Solution**:
```bash
chmod +x scripts/*.sh
# Or for all shell scripts
find . -name "*.sh" -exec chmod +x {} \;
```

#### 3. Port Conflicts
**Error**: `Port is already allocated`

**Solution**:
```bash
# Check what's using the port
lsof -i :8000  # Replace with your port
docker ps      # Check running containers

# Stop conflicting services
docker-compose -f docker-compose.local.yml down
# Or kill specific process
kill -9 <PID>
```

#### 4. VM Connectivity Issues
**Error**: Cannot connect to VM services

**Checklist**:
- [ ] Verify NSG rules allow traffic on required ports
- [ ] Check public IPs are correct in `.env.production`
- [ ] Test connectivity: `telnet <VM_IP> <PORT>`
- [ ] Verify firewall rules on both VMs
- [ ] Check if services are running: `docker ps`

#### 5. Metrics Not Appearing in Grafana
**Checklist**:
- [ ] Check Prometheus targets: http://localhost:9094/targets
- [ ] Verify service health: `docker-compose ps`
- [ ] Check Prometheus config: `monitoring/prometheus/prometheus.yml`
- [ ] Review service logs for metrics endpoint errors
- [ ] Ensure services expose metrics on correct ports

#### 6. Database Connection Errors
**Error**: `FATAL: password authentication failed`

**Solution**:
```bash
# 1. Verify environment variables
cat .env | grep POSTGRES

# 2. Check if PostgreSQL is running
docker exec fraud-postgres pg_isready -U fraud_user

# 3. Reset database (⚠️ Data loss)
docker-compose -f docker-compose.local.yml down -v
docker-compose -f docker-compose.local.yml up -d postgres

# 4. Test connection manually
docker exec -it fraud-postgres psql -U fraud_user -d fraud_detection
```

### Logs

#### View Service Logs
```bash
# Local development - all services
docker-compose -f docker-compose.local.yml logs -f

# Specific service
docker-compose -f docker-compose.local.yml logs -f api

# Last 100 lines
docker-compose -f docker-compose.local.yml logs -f --tail=100 api

# VM1 production
docker-compose -f docker-compose.vm1.yml logs -f

# VM2 monitoring
docker-compose -f docker-compose.vm2.yml logs -f
```

#### Debug Mode
```bash
# Enable debug logging
export LOG_LEVEL=DEBUG

# Run specific service with debug
docker-compose -f docker-compose.local.yml up api

# Check container health
docker inspect --format='{{json .State.Health}}' fraud-api | jq
```

### Performance Optimization

#### Reduce Docker Build Time
```bash
# Use build cache
docker-compose -f docker-compose.local.yml build

# Parallel builds (use multiple CPU cores)
docker-compose -f docker-compose.local.yml build --parallel

# Build specific service only
docker-compose -f docker-compose.local.yml build api
```

#### Reduce Image Size
- Use `.dockerignore` files
- Multi-stage builds (already implemented)
- Minimize layers in Dockerfiles
- Use Alpine-based images where possible

#### Monitor Resource Usage
```bash
# Container stats
docker stats

# System resource usage
docker system df

# Detailed container info
docker ps --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"
```

## 📚 Documentation

- [`Guide/PRODUCTION_DEPLOYMENT_GUIDE.md`](Guide/PRODUCTION_DEPLOYMENT_GUIDE.md) - Complete production setup guide
- [`monitoring/README.md`](monitoring/README.md) - Monitoring stack documentation
- [`scripts/README.md`](scripts/README.md) - Scripts documentation
- [`api/README.md`](api/README.md) - API service documentation

## ❓ FAQ (Frequently Asked Questions)

### General

**Q: How long does the initial setup take?**
A: 
- Local build: 15-30 minutes (building all images)
- Registry pull: 10-20 minutes (depending on network speed)
- After initial setup: 2-3 minutes to start all services

**Q: What is the minimum hardware required?**
A: 4 CPU cores, 8 GB RAM, 50 GB disk space. For better performance: 8+ cores, 16 GB RAM, 100 GB SSD.

**Q: Can I run this on Windows?**
A: Yes, with Docker Desktop for Windows. Use Git Bash or WSL2 for running shell scripts.

### Docker & Images

**Q: Why is Docker pulling images so slowly?**
A: 
- Images are large (1-4 GB each)
- Check your internet speed and stability
- Use `docker system prune` to free disk space
- Consider using local builds instead

**Q: Error: "no space left on device"**
A:
```bash
# Clean up Docker resources
docker system prune -a --volumes
# Increase Docker disk image size (Docker Desktop → Settings → Resources)
```

**Q: How do I switch from registry images to local builds?**
A: Edit `docker-compose.local.yml`, uncomment the `build:` sections and comment out `image:` lines that reference registries.

**Q: Can I use a different Docker registry?**
A: Yes, update `DOCKERHUB_USERNAME` in `.env` and push images to your registry.

### Services

**Q: Which services are essential for minimal setup?**
A: Minimum required: `postgres`, `redis`, `mlflow`, `api`. Optional: `kafka`, `airflow`, monitoring stack.

**Q: Can I run services individually?**
A:
```bash
docker-compose -f docker-compose.local.yml up -d postgres redis mlflow api
```

**Q: How do I reset everything and start fresh?**
A:
```bash
docker-compose -f docker-compose.local.yml down -v  # ⚠️ Deletes all data
docker system prune -a
./scripts/build-local.sh  # Choose option 1
```

**Q: Airflow UI shows "DAGs not loading"**
A: 
- Check `airflow/dags` folder is mounted correctly
- Verify DAG files have no syntax errors
- Check scheduler logs: `docker logs fraud-airflow-scheduler`

### Production

**Q: How do I deploy to production Azure VMs?**
A: Follow [`Guide/PRODUCTION_DEPLOYMENT_GUIDE.md`](Guide/PRODUCTION_DEPLOYMENT_GUIDE.md) for step-by-step instructions.

**Q: How do I secure the production deployment?**
A:
- Change all default passwords in `.env.production`
- Generate new keys for `AIRFLOW_FERNET_KEY`, `JWT_SECRET_KEY`
- Configure NSG rules to restrict access
- Enable SSL/TLS for all public endpoints
- Use Azure Key Vault for secrets

**Q: Can VM1 and VM2 be in different Azure regions?**
A: Yes, ensure cross-region connectivity and account for increased latency.

### Monitoring & Troubleshooting

**Q: Grafana shows no data**
A:
- Check Prometheus is scraping targets: http://localhost:9094/targets
- Verify services expose metrics on correct ports
- Check Prometheus configuration: `monitoring/prometheus/prometheus.yml`

**Q: How do I check if all services are healthy?**
A:
```bash
# Local
./scripts/build-local.sh  # Choose option 8 (full test)
# VM1
bash scripts/vm1-health-check.sh
# VM2
bash scripts/vm2-health-check.sh
```

**Q: Services keep restarting**
A:
- Check logs: `docker logs <container-name>`
- Verify environment variables are correct
- Check resource limits (memory, CPU)
- Verify database connectivity

### Data & Models

**Q: Where do I put the creditcard.csv dataset?**
A: Place it in the project root: `fraud-detection-ml/creditcard.csv`
Download from: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud

**Q: How do I train a new model?**
A:
- Trigger via Airflow DAG (`03_model_training`)
- Or manually: `docker exec fraud-training python -m src.training.train_model`

**Q: Where are trained models stored?**
A: Models are stored in MLflow artifacts volume and registered in MLflow model registry.

**Q: How do I update the model in production?**
A:
1. Train and register new model in MLflow
2. Promote to "Production" stage in MLflow UI
3. Restart API service or trigger hot reload

### Networking

**Q: How do services communicate?**
A: Via Docker network `fraud-detection-network`. All services can reach each other using container names (e.g., `http://api:8000`).

**Q: How do I expose additional ports?**
A: Edit `docker-compose.local.yml` and add port mappings in the `ports:` section.

**Q: Can I access services from outside the host?**
A: Yes, services bound to `0.0.0.0` are accessible. Configure firewall rules accordingly.

## 🔧 Advanced Configuration

### Using Custom Environment Files
```bash
# Use different env file
docker-compose -f docker-compose.local.yml --env-file .env.staging up -d
```

### Running Specific Service Combinations
```bash
# Infrastructure only
docker-compose -f docker-compose.local.yml up -d postgres redis kafka

# ML services only
docker-compose -f docker-compose.local.yml up -d mlflow training

# Monitoring stack only
docker-compose -f docker-compose.local.yml up -d prometheus grafana alertmanager
```

### Scaling Services
```bash
# Scale Airflow workers
docker-compose -f docker-compose.local.yml up -d --scale airflow-worker=3

# Note: Not all services support scaling (e.g., postgres, mlflow)
```

### Custom Network Configuration
```bash
# Create custom network
docker network create fraud-custom-network

# Update docker-compose.yml to use custom network
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make changes with proper testing
4. Update documentation
5. Submit a pull request

## 📄 License

See [LICENSE](LICENSE) file for details.

## 🆘 Support

For issues and questions:
1. Check the troubleshooting section
2. Review service logs
3. Run health checks
4. Check the documentation
5. Open an issue with detailed information

## 👨‍💻 Authors: Fraud Detection Team

1. Joshua Juste NIKIEMA
2. Olalekan Taofeek OLALUWOYE
3. Soulaimana Toihir DJALOUD