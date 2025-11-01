#!/bin/bash
# VM1 Production Deployment Script
# Deploys application services on VM1 (your Azure subscription)
# Run this on VM1 to deploy: postgres, redis, kafka, mlflow, data, drift, training, airflow

set -e  # Exit immediately if any command fails

echo "🚀 Fraud Detection ML - VM1 Production Deployment"
echo "=================================================="
echo ""

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging functions
log_info() {
    echo -e "${GREEN}✓${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}⚠${NC} $1"
}

log_error() {
    echo -e "${RED}✗${NC} $1"
}

log_step() {
    echo -e "${BLUE}➤${NC} $1"
}

# Check if running on Linux (Azure VM)
if [[ "$OSTYPE" != "linux-gnu"* ]]; then
    log_error "This script is designed for Linux (Azure VM). Current OS: $OSTYPE"
    exit 1
fi

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
    log_error "Docker is not running. Please start Docker service."
    exit 1
fi

log_info "Docker is running"

# Check if docker-compose is available
if ! command -v docker-compose &> /dev/null; then
    log_error "docker-compose is not installed. Please install it first."
    exit 1
fi

log_info "docker-compose is available"

# Check if .env.production exists
if [ ! -f .env.production ]; then
    log_error ".env.production file not found. Please ensure it exists in the project root."
    exit 1
fi

log_info ".env.production file found"

# Check if dataset exists
if [ ! -f creditcard.csv ]; then
    log_error "Dataset creditcard.csv not found. Please ensure it's in the project root."
    exit 1
fi

log_info "Dataset creditcard.csv found ($(du -h creditcard.csv | cut -f1))"

# Menu
echo ""
echo "VM1 Production Deployment Options:"
echo "  1) Full deployment (build + deploy)"
echo "  2) Build images only"
echo "  3) Deploy services only"
echo "  4) Stop all services"
echo "  5) View service status"
echo "  6) View logs"
echo "  7) Clean up (stop + remove volumes)"
echo "  8) Health check"
echo ""
read -p "Choice (1-8): " choice

case $choice in
    1)
        log_step "Starting full VM1 production deployment..."

        # Build all images
        log_info "Building all production images..."
        docker-compose -f docker-compose.vm1.yml build --no-cache
        log_info "Images built successfully ✅"

        # Deploy services
        log_info "Deploying VM1 services..."
        docker-compose -f docker-compose.vm1.yml --env-file .env.production up -d
        log_info "Services deployed successfully ✅"

        # Wait for services to start
        log_info "Waiting for services to initialize (60s)..."
        sleep 60

        # Run health checks
        log_step "Running health checks..."
        bash scripts/vm1-health-check.sh
        ;;

    2)
        log_step "Building production images for VM1..."

        # Build all images
        log_info "Building all production images..."
        docker-compose -f docker-compose.vm1.yml build --no-cache
        log_info "Images built successfully ✅"

        # List built images
        echo ""
        log_info "Built images:"
        docker images | grep fraud-detection
        ;;

    3)
        log_step "Deploying VM1 services..."

        # Check if images exist
        if ! docker images | grep -q fraud-detection; then
            log_warn "Production images not found. Building them first..."
            docker-compose -f docker-compose.vm1.yml build
        fi

        # Deploy services
        log_info "Starting VM1 services..."
        docker-compose -f docker-compose.vm1.yml --env-file .env.production up -d
        log_info "Services deployed successfully ✅"

        # Show service status
        echo ""
        log_info "Service status:"
        docker-compose -f docker-compose.vm1.yml ps
        ;;

    4)
        log_step "Stopping VM1 services..."
        docker-compose -f docker-compose.vm1.yml down
        log_info "All services stopped ✅"
        ;;

    5)
        log_step "VM1 service status:"
        docker-compose -f docker-compose.vm1.yml ps
        echo ""
        log_info "Container resource usage:"
        docker stats --no-stream $(docker-compose -f docker-compose.vm1.yml ps -q)
        ;;

    6)
        echo "Available services: postgres, redis, zookeeper, kafka, mlflow, data, drift, training, airflow-webserver, airflow-scheduler"
        read -p "Service name (leave empty for all): " service
        if [ -z "$service" ]; then
            docker-compose -f docker-compose.vm1.yml logs -f --tail=100
        else
            docker-compose -f docker-compose.vm1.yml logs -f --tail=100 $service
        fi
        ;;

    7)
        log_warn "⚠️  WARNING: This will remove all volumes (PostgreSQL, Redis, Kafka data, etc.)"
        read -p "Are you sure? (yes/no): " confirm
        if [ "$confirm" = "yes" ]; then
            docker-compose -f docker-compose.vm1.yml down -v
            log_info "Cleanup completed ✅"
        else
            log_info "Cancelled"
        fi
        ;;

    8)
        log_step "Running VM1 health checks..."
        bash scripts/vm1-health-check.sh
        ;;

    *)
        log_error "Invalid choice"
        exit 1
        ;;
esac

echo ""
if [ "$choice" = "1" ] || [ "$choice" = "3" ]; then
    echo "🎉 VM1 Deployment Summary:"
    echo "=========================="
    echo ""
    echo "✅ Services deployed on VM1:"
    echo "   • PostgreSQL: localhost:5432"
    echo "   • Redis: localhost:6379"
    echo "   • Kafka: localhost:9092 (internal), localhost:29092 (external)"
    echo "   • MLflow: localhost:5000"
    echo "   • Data Service: localhost:9091 (metrics)"
    echo "   • Drift Service: localhost:9092 (metrics)"
    echo "   • Training Service: localhost:9093 (metrics)"
    echo "   • Airflow: localhost:8080 (admin/admin)"
    echo ""
    echo "📊 Monitoring endpoints (accessible from VM2):"
    echo "   • Data metrics: http://<VM1_IP>:9091/metrics"
    echo "   • Drift metrics: http://<VM1_IP>:9092/metrics"
    echo "   • Training metrics: http://<VM1_IP>:9093/metrics"
    echo ""
    echo "🔗 Next steps:"
    echo "   1. Note your VM1 public IP: $(curl -s ifconfig.me)"
    echo "   2. Share this IP with your colleague for VM2 deployment"
    echo "   3. Configure NSG rules to allow VM2 access to ports 9091-9093"
    echo "   4. Run health checks: bash scripts/vm1-health-check.sh"
fi

echo ""
log_info "VM1 deployment script completed!"