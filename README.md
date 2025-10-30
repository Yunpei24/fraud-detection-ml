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
- Docker & Docker Compose
- Git
- Bash shell
- Python 3.9+ (for local development)

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
- **Permission denied**: `chmod +x scripts/*.sh`
- **Port conflicts**: Check `docker ps` for conflicts
- **VM connectivity**: Verify NSG rules and public IPs
- **Metrics not appearing**: Check Prometheus targets and service logs

### Logs
```bash
# Local development
docker-compose -f docker-compose.local.yml logs -f

# VM1 production
docker-compose -f docker-compose.vm1.yml logs -f

# VM2 monitoring
docker-compose -f docker-compose.vm2.yml logs -f
```

## 📚 Documentation

- [`Guide/PRODUCTION_DEPLOYMENT_GUIDE.md`](Guide/PRODUCTION_DEPLOYMENT_GUIDE.md) - Complete production setup guide
- [`monitoring/README.md`](monitoring/README.md) - Monitoring stack documentation
- [`scripts/README.md`](scripts/README.md) - Scripts documentation
- [`api/README.md`](api/README.md) - API service documentation

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
