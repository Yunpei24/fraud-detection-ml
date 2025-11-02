#!/bin/bash
# Build & Test Script for docker-compose.local.yml

set -e  # Exit immediately if any command fails

echo "🚀 Fraud Detection ML - Local Build & Test"
echo "==========================================="
echo ""

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Logging functions
log_info() {
    echo -e "${GREEN}✓${NC} $1"
}

# Prompted safe prune to free disk space before local builds
pre_build_prune() {
    echo ""
    read -p "Do you want to run a safe Docker prune to free space before building? (y/N): " prune_choice
    if [[ "$prune_choice" =~ ^[Yy]$ ]]; then
        log_warn "Running docker system prune and builder prune (non-interactive). This will remove unused images/containers/networks."
        docker system prune -af || true
        docker builder prune -af --filter until=24h || true
        log_info "Prune finished"
    else
        log_info "Skipping prune"
    fi
}

log_warn() {
    echo -e "${YELLOW}⚠${NC} $1"
}

log_error() {
    echo -e "${RED}✗${NC} $1"
}

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
    log_error "Docker is not running. Please start Docker Desktop."
    exit 1
fi

log_info "Docker is running"

# Check if .env file exists
if [ ! -f .env ]; then
    log_warn ".env file not found. Creating one with default values..."
    cat > .env << 'EOF'
POSTGRES_PASSWORD=fraud_pass_dev_2024
REDIS_PASSWORD=redis_pass_dev_2024
AIRFLOW_FERNET_KEY=ZmDfcTF7_60GrrY167zsiPd67pEvs0aGOv2oasOM1Pg=
GRAFANA_USER=admin
GRAFANA_PASSWORD=admin_dev_2024
ALERT_EMAIL=ml-alerts@localhost.com
EOF
    log_info ".env file created"
fi

# Check if the dataset exists
if [ ! -f creditcard.csv ]; then
    log_error "Dataset creditcard.csv not found at the project root!"
    echo "  Download it from: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud"
    echo "  Place it here: $(pwd)/creditcard.csv"
    exit 1
fi

log_info "Dataset creditcard.csv found ($(du -h creditcard.csv | cut -f1))"

# Menu
echo ""
echo "What would you like to do?"
echo "  1) Build all images (first time setup)"
echo "  2) Build a specific image"
echo "  3) Start all services"
echo "  4) Start specific services"
echo "  5) Stop all services"
echo "  6) View logs"
echo "  7) Clean up (down + remove volumes)"
echo "  8) Full test (build + up + health checks)"
echo ""
read -p "Choice (1-8): " choice

case $choice in
    1)
        log_info "Building all images..."
        pre_build_prune
        docker-compose -f docker-compose.local.yml build --no-cache
        log_info "Build completed ✅"
        ;;
    2)
        echo "Available services: api, data, drift, training, airflow-webserver, airflow-scheduler, prometheus, grafana, postgres, redis, mlflow"
        read -p "Service name: " service
    log_info "Building $service..."
    pre_build_prune
    docker-compose -f docker-compose.local.yml build --no-cache $service
        log_info "Build completed ✅"
        ;;
    3)
        log_info "Starting all services..."
        docker-compose -f docker-compose.local.yml up -d
        log_info "All services started ✅"
        echo ""
        echo "Available interfaces:"
        echo "  - API:        http://localhost:8000/docs"
        echo "  - MLflow:     http://localhost:5000"
        echo "  - Airflow:    http://localhost:8080 (admin/admin)"
        echo "  - Prometheus: http://localhost:9094"
        echo "  - Grafana:    http://localhost:3000 (admin/admin_dev_2024)"
        ;;
    4)
        echo "Services: postgres, redis, mlflow, api, data, drift, training, airflow-webserver, airflow-scheduler, prometheus, grafana"
        read -p "Services to start (separated by spaces): " services
        log_info "Starting $services..."
        docker-compose -f docker-compose.local.yml up -d $services
        log_info "Selected services started ✅"
        ;;
    5)
        log_info "Stopping all services..."
        docker-compose -f docker-compose.local.yml down
        log_info "All services stopped ✅"
        ;;
    6)
        read -p "Service name (leave empty for all): " service
        if [ -z "$service" ]; then
            docker-compose -f docker-compose.local.yml logs -f --tail=100
        else
            docker-compose -f docker-compose.local.yml logs -f --tail=100 $service
        fi
        ;;
    7)
        log_warn "⚠️  WARNING: This will remove all volumes (PostgreSQL, MLflow data, etc.)"
        read -p "Are you sure? (yes/no): " confirm
        if [ "$confirm" = "yes" ]; then
            docker-compose -f docker-compose.local.yml down -v
            log_info "Cleanup completed ✅"
        else
            log_info "Cancelled"
        fi
        ;;
    8)
        log_info "Full test - This may take 5-10 minutes..."
        echo ""
        
        # Build
        log_info "Step 1/5: Building images..."
        pre_build_prune
        docker-compose -f docker-compose.local.yml build
        
        # Start
        log_info "Step 2/5: Starting services..."
        docker-compose -f docker-compose.local.yml up -d
        
        # Wait
        log_info "Step 3/5: Waiting for startup (60s)..."
        sleep 60
        
        # Health checks
        log_info "Step 4/5: Checking service health..."
        
        echo ""
        echo "  PostgreSQL..."
        if docker exec fraud-postgres pg_isready -U fraud_user > /dev/null 2>&1; then
            log_info "  PostgreSQL: OK"
        else
            log_error "  PostgreSQL: FAIL"
        fi
        
        echo "  Redis..."
        if docker exec fraud-redis redis-cli ping > /dev/null 2>&1; then
            log_info "  Redis: OK"
        else
            log_error "  Redis: FAIL"
        fi
        
        echo "  MLflow..."
        if curl -s http://localhost:5000/health > /dev/null 2>&1; then
            log_info "  MLflow: OK"
        else
            log_error "  MLflow: FAIL"
        fi
        
        echo "  API..."
        if curl -s http://localhost:8000/health > /dev/null 2>&1; then
            log_info "  API: OK"
        else
            log_error "  API: FAIL"
        fi
        
        echo "  Data metrics..."
        if curl -s http://localhost:9091/metrics > /dev/null 2>&1; then
            log_info "  Data metrics: OK"
        else
            log_error "  Data metrics: FAIL"
        fi
        
        echo "  Drift metrics..."
        if curl -s http://localhost:9095/metrics > /dev/null 2>&1; then
            log_info "  Drift metrics: OK"
        else
            log_error "  Drift metrics: FAIL"
        fi
        
        echo "  Training metrics..."
        if curl -s http://localhost:9096/metrics > /dev/null 2>&1; then
            log_info "  Training metrics: OK"
        else
            log_error "  Training metrics: FAIL"
        fi
        
        echo "  Prometheus..."
        if curl -s http://localhost:9094 > /dev/null 2>&1; then
            log_info "  Prometheus: OK"
        else
            log_error "  Prometheus: FAIL"
        fi
        
        # Summary
        log_info "Step 5/5: Summary"
        echo ""
        docker-compose -f docker-compose.local.yml ps
        echo ""
        log_info "Full test completed ✅"
        echo ""
        echo "Available interfaces:"
        echo "  - API Docs:   http://localhost:8000/docs"
        echo "  - MLflow UI:  http://localhost:5000"
        echo "  - Airflow UI: http://localhost:8080 (admin/admin)"
        echo "  - Prometheus: http://localhost:9094"
        echo "  - Grafana:    http://localhost:3000 (admin/admin_dev_2024)"
        echo "  - Alertmanager: http://localhost:9093"
        ;;
    *)
        log_error "Invalid choice"
        exit 1
        ;;
esac

echo ""
log_info "Done!"
