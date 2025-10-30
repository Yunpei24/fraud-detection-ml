#!/bin/bash
# VM2 Production Deployment Script
# Deploys monitoring services on VM2 (colleague's Azure subscription)
# Run this on VM2 to deploy: prometheus, grafana, alertmanager, node-exporter

set -e  # Exit immediately if any command fails

echo "🚀 Fraud Detection ML - VM2 Monitoring Deployment"
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

# Check if VM1_IP is configured
if ! grep -q "<VM1_IP>" monitoring/prometheus.vm2.yml; then
    log_info "VM1 IP appears to be configured in prometheus config"
else
    log_warn "VM1 IP not configured in monitoring/prometheus.vm2.yml"
    log_warn "Please update <VM1_IP> with your colleague's VM1 public IP"
    read -p "VM1 Public IP: " vm1_ip
    if [ ! -z "$vm1_ip" ]; then
        sed -i "s/<VM1_IP>/$vm1_ip/g" monitoring/prometheus.vm2.yml
        log_info "VM1 IP configured: $vm1_ip"
    fi
fi

# Menu
echo ""
echo "VM2 Monitoring Deployment Options:"
echo "  1) Full deployment (deploy + configure)"
echo "  2) Deploy services only"
echo "  3) Stop all services"
echo "  4) View service status"
echo "  5) View logs"
echo "  6) Clean up (stop + remove volumes)"
echo "  7) Health check"
echo "  8) Configure VM1 IP"
echo ""
read -p "Choice (1-8): " choice

case $choice in
    1)
        log_step "Starting full VM2 monitoring deployment..."

        # Deploy services
        log_info "Deploying VM2 monitoring services..."
        docker-compose -f docker-compose.vm2.yml up -d
        log_info "Services deployed successfully ✅"

        # Wait for services to start
        log_info "Waiting for services to initialize (60s)..."
        sleep 60

        # Run health checks
        log_step "Running health checks..."
        bash scripts/vm2-health-check.sh
        ;;

    2)
        log_step "Deploying VM2 monitoring services..."

        # Deploy services
        log_info "Starting VM2 services..."
        docker-compose -f docker-compose.vm2.yml up -d
        log_info "Services deployed successfully ✅"

        # Show service status
        echo ""
        log_info "Service status:"
        docker-compose -f docker-compose.vm2.yml ps
        ;;

    3)
        log_step "Stopping VM2 services..."
        docker-compose -f docker-compose.vm2.yml down
        log_info "All services stopped ✅"
        ;;

    4)
        log_step "VM2 service status:"
        docker-compose -f docker-compose.vm2.yml ps
        echo ""
        log_info "Container resource usage:"
        docker stats --no-stream $(docker-compose -f docker-compose.vm2.yml ps -q)
        ;;

    5)
        echo "Available services: prometheus, grafana, alertmanager, node-exporter"
        read -p "Service name (leave empty for all): " service
        if [ -z "$service" ]; then
            docker-compose -f docker-compose.vm2.yml logs -f --tail=100
        else
            docker-compose -f docker-compose.vm2.yml logs -f --tail=100 $service
        fi
        ;;

    6)
        log_warn "⚠️  WARNING: This will remove all volumes (Grafana dashboards, Prometheus data, etc.)"
        read -p "Are you sure? (yes/no): " confirm
        if [ "$confirm" = "yes" ]; then
            docker-compose -f docker-compose.vm2.yml down -v
            log_info "Cleanup completed ✅"
        else
            log_info "Cancelled"
        fi
        ;;

    7)
        log_step "Running VM2 health checks..."
        bash scripts/vm2-health-check.sh
        ;;

    8)
        log_step "Configuring VM1 IP for Prometheus scraping..."
        read -p "Enter VM1 Public IP: " vm1_ip
        if [ ! -z "$vm1_ip" ]; then
            sed -i "s/<VM1_IP>/$vm1_ip/g" monitoring/prometheus.vm2.yml
            log_info "VM1 IP configured: $vm1_ip"
            log_info "Restarting Prometheus to apply changes..."
            docker-compose -f docker-compose.vm2.yml restart prometheus
            log_info "Prometheus restarted ✅"
        else
            log_error "No IP provided"
        fi
        ;;

    *)
        log_error "Invalid choice"
        exit 1
        ;;
esac

echo ""
if [ "$choice" = "1" ] || [ "$choice" = "2" ]; then
    echo "🎉 VM2 Deployment Summary:"
    echo "=========================="
    echo ""
    echo "✅ Monitoring services deployed on VM2:"
    echo "   • Prometheus: localhost:9090"
    echo "   • Grafana: localhost:3000 (admin/admin_dev_2024)"
    echo "   • Alertmanager: localhost:9093"
    echo "   • Node Exporter: localhost:9100"
    echo ""
    echo "📊 Available Dashboards:"
    echo "   • System Overview: Service health, throughput, alerts"
    echo "   • API Performance: Requests, latency, errors, predictions"
    echo "   • Data Pipeline: Processing, quality, queues"
    echo "   • Drift Detection: Model drift, performance, retraining"
    echo ""
    echo "🔗 Access URLs:"
    echo "   • Grafana: http://<VM2_IP>:3000"
    echo "   • Prometheus: http://<VM2_IP>:9090"
    echo "   • Alertmanager: http://<VM2_IP>:9093"
    echo ""
    echo "📋 Next steps:"
    echo "   1. Access Grafana and verify dashboards are populated"
    echo "   2. Check that VM1 services are being scraped by Prometheus"
    echo "   3. Configure alert notifications (email, Slack, etc.)"
    echo "   4. Run health checks: bash scripts/vm2-health-check.sh"
fi

echo ""
log_info "VM2 deployment script completed!"