#!/bin/bash

# Fraud Detection Monitoring Stack Deployment Script
# This script helps deploy the complete monitoring stack for the fraud detection system

set -e

echo "🚀 Starting Fraud Detection Monitoring Stack Deployment"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if Docker is running
check_docker() {
    if ! docker info > /dev/null 2>&1; then
        print_error "Docker is not running. Please start Docker first."
        exit 1
    fi
    print_success "Docker is running"
}

# Check if docker-compose is available
check_docker_compose() {
    if ! command -v docker-compose &> /dev/null; then
        print_error "docker-compose is not installed. Please install it first."
        exit 1
    fi
    print_success "docker-compose is available"
}

# Create necessary directories
create_directories() {
    print_status "Creating monitoring directories..."
    mkdir -p monitoring/grafana/provisioning/datasources
    mkdir -p monitoring/grafana/provisioning/dashboards
    mkdir -p monitoring/alertmanager/templates
    print_success "Directories created"
}

# Create Grafana datasource configuration
create_grafana_datasource() {
    print_status "Creating Grafana datasource configuration..."
    cat > monitoring/grafana/provisioning/datasources/prometheus.yml << EOF
apiVersion: 1

datasources:
  - name: Prometheus
    type: prometheus
    access: proxy
    url: http://prometheus:9090
    isDefault: true
    editable: true
EOF
    print_success "Grafana datasource configured"
}

# Create Grafana dashboard provisioning
create_grafana_dashboard_provisioning() {
    print_status "Creating Grafana dashboard provisioning..."
    cat > monitoring/grafana/provisioning/dashboards/dashboards.yml << EOF
apiVersion: 1

providers:
  - name: 'fraud-detection'
    type: file
    disableDeletion: false
    updateIntervalSeconds: 10
    allowUiUpdates: true
    options:
      path: /var/lib/grafana/dashboards
EOF
    print_success "Grafana dashboard provisioning configured"
}

# Set environment variables
set_environment() {
    print_status "Setting up environment variables..."
    if [ ! -f .env ]; then
        cat > .env << EOF
# Grafana Configuration
GRAFANA_ADMIN_PASSWORD=admin123

# Redis Configuration (if used)
REDIS_PASSWORD=

# Alertmanager SMTP (configure these)
SMTP_FROM=alerts@yourcompany.com
SMTP_USERNAME=alerts@yourcompany.com
SMTP_PASSWORD=your-app-password

# Slack Webhook (configure this)
SLACK_WEBHOOK_URL=https://hooks.slack.com/services/YOUR/SLACK/WEBHOOK
EOF
        print_warning "Created .env file with default values. Please update with your actual configuration!"
    else
        print_success "Environment file already exists"
    fi
}

# Start the monitoring stack
start_monitoring() {
    print_status "Starting monitoring stack..."
    docker-compose -f docker-compose.monitoring.yml up -d
    print_success "Monitoring stack started"
}

# Wait for services to be healthy
wait_for_services() {
    print_status "Waiting for services to be healthy..."
    sleep 30

    # Check Prometheus
    if curl -f http://localhost:9090/-/healthy > /dev/null 2>&1; then
        print_success "Prometheus is healthy"
    else
        print_warning "Prometheus health check failed"
    fi

    # Check Grafana
    if curl -f http://localhost:3000/api/health > /dev/null 2>&1; then
        print_success "Grafana is healthy"
    else
        print_warning "Grafana health check failed"
    fi
}

# Display access information
display_info() {
    echo ""
    echo "🎉 Monitoring stack deployment completed!"
    echo ""
    echo "Access your monitoring dashboards:"
    echo "  • Grafana:     http://localhost:3000 (admin/admin123)"
    echo "  • Prometheus:  http://localhost:9090"
    echo "  • Alertmanager: http://localhost:9093"
    echo ""
    echo "Available dashboards:"
    echo "  • System Overview"
    echo "  • API Performance"
    echo "  • Data Pipeline"
    echo "  • Drift Detection"
    echo ""
    echo "Next steps:"
    echo "1. Update .env file with your actual credentials"
    echo "2. Configure alert notifications in Grafana"
    echo "3. Verify that your services are exposing metrics"
    echo "4. Test alerts by triggering some failures"
}

# Main deployment function
main() {
    echo "Fraud Detection Monitoring Stack Deployment"
    echo "=========================================="

    check_docker
    check_docker_compose
    create_directories
    create_grafana_datasource
    create_grafana_dashboard_provisioning
    set_environment
    start_monitoring
    wait_for_services
    display_info
}

# Handle command line arguments
case "${1:-}" in
    "stop")
        print_status "Stopping monitoring stack..."
        docker-compose -f docker-compose.monitoring.yml down
        print_success "Monitoring stack stopped"
        ;;
    "restart")
        print_status "Restarting monitoring stack..."
        docker-compose -f docker-compose.monitoring.yml restart
        print_success "Monitoring stack restarted"
        ;;
    "logs")
        docker-compose -f docker-compose.monitoring.yml logs -f
        ;;
    "status")
        docker-compose -f docker-compose.monitoring.yml ps
        ;;
    *)
        main
        ;;
esac