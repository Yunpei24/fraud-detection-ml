#!/bin/bash
# Build & Deploy Script for VM1 and VM2 production environments

set -e  # Exit immediately if any command fails

echo "🚀 Fraud Detection ML - Production Build & Deploy"
echo "================================================="
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
    echo -e "${BLUE}➜${NC} $1"
}

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
    log_error "Docker is not running. Please start Docker."
    exit 1
fi

log_info "Docker is running"

# Select environment
echo ""
echo "Select target environment:"
echo "  1) VM1 (Application services: data, drift, training, airflow, mlflow, postgres, redis, kafka)"
echo "  2) VM2 (Monitoring services: prometheus, grafana, alertmanager, node-exporter)"
echo "  3) Both VM1 and VM2"
echo ""
read -p "Choice (1-3): " env_choice

case $env_choice in
    1)
        TARGET="VM1"
        COMPOSE_FILE="docker-compose.vm1.yml"
        ;;
    2)
        TARGET="VM2"
        COMPOSE_FILE="docker-compose.vm2.yml"
        ;;
    3)
        TARGET="BOTH"
        COMPOSE_FILE=""
        ;;
    *)
        log_error "Invalid choice"
        exit 1
        ;;
esac

log_info "Target: $TARGET"

# Function to handle VM1 operations
handle_vm1() {
    echo ""
    echo "VM1 Operations"
    echo "=============="
    echo ""
    echo "What would you like to do?"
    echo "  1) Pull images from Docker Hub"
    echo "  2) Start all services"
    echo "  3) Start specific services"
    echo "  4) Stop all services"
    echo "  5) View logs"
    echo "  6) Restart services"
    echo "  7) Check service health"
    echo "  8) Clean up (down + remove volumes)"
    echo ""
    read -p "Choice (1-8): " action

    case $action in
        1)
            log_info "Pulling images from Docker Hub..."
            
            # Check if DOCKERHUB_USERNAME is set
            if [ -z "$DOCKERHUB_USERNAME" ]; then
                log_warn "DOCKERHUB_USERNAME not set. Using default: yoshua24"
                export DOCKERHUB_USERNAME=yoshua24
            fi
            
            log_step "Pulling VM1 images..."
            docker-compose -f docker-compose.vm1.yml pull
            log_info "Images pulled successfully ✅"
            ;;
        2)
            log_info "Starting all VM1 services..."
            
            # Check for required files
            if [ ! -f creditcard.csv ]; then
                log_warn "creditcard.csv not found. Some services may fail."
            fi
            
            docker-compose -f docker-compose.vm1.yml up -d
            log_info "All services started ✅"
            echo ""
            echo "Available interfaces on VM1:"
            echo "  - MLflow:     http://localhost:5000"
            echo "  - Airflow:    http://localhost:8080 (admin/admin)"
            echo "  - PostgreSQL: localhost:5432"
            echo "  - Redis:      localhost:6379"
            echo "  - Kafka:      localhost:9092"
            echo ""
            echo "Metrics endpoints (accessible from VM2):"
            echo "  - Data:       http://localhost:9091/metrics"
            echo "  - Drift:      http://localhost:9095/metrics"
            echo "  - Training:   http://localhost:9096/metrics"
            ;;
        3)
            echo "Available services: postgres, redis, zookeeper, kafka, mlflow, data, drift, training,"
            echo "                    airflow-init, airflow-webserver, airflow-scheduler, airflow-worker"
            read -p "Services to start (separated by spaces): " services
            log_info "Starting $services..."
            docker-compose -f docker-compose.vm1.yml up -d $services
            log_info "Selected services started ✅"
            ;;
        4)
            log_info "Stopping all VM1 services..."
            docker-compose -f docker-compose.vm1.yml down
            log_info "All services stopped ✅"
            ;;
        5)
            read -p "Service name (leave empty for all): " service
            if [ -z "$service" ]; then
                docker-compose -f docker-compose.vm1.yml logs -f --tail=100
            else
                docker-compose -f docker-compose.vm1.yml logs -f --tail=100 $service
            fi
            ;;
        6)
            read -p "Service to restart (leave empty for all): " service
            if [ -z "$service" ]; then
                log_info "Restarting all services..."
                docker-compose -f docker-compose.vm1.yml restart
            else
                log_info "Restarting $service..."
                docker-compose -f docker-compose.vm1.yml restart $service
            fi
            log_info "Restart completed ✅"
            ;;
        7)
            log_info "Checking service health..."
            echo ""
            
            echo "PostgreSQL..."
            if docker exec fraud-postgres pg_isready -U fraud_user > /dev/null 2>&1; then
                log_info "PostgreSQL: UP"
            else
                log_error "PostgreSQL: DOWN"
            fi
            
            echo "Redis..."
            if docker exec fraud-redis redis-cli ping > /dev/null 2>&1; then
                log_info "Redis: UP"
            else
                log_error "Redis: DOWN"
            fi
            
            echo "Kafka..."
            if docker exec fraud-kafka kafka-broker-api-versions --bootstrap-server localhost:9092 > /dev/null 2>&1; then
                log_info "Kafka: UP"
            else
                log_error "Kafka: DOWN"
            fi
            
            echo "MLflow..."
            if curl -s http://localhost:5000/health > /dev/null 2>&1; then
                log_info "MLflow: UP"
            else
                log_error "MLflow: DOWN"
            fi
            
            echo "Data metrics..."
            if curl -s http://localhost:9091/metrics > /dev/null 2>&1; then
                log_info "Data metrics: UP"
            else
                log_error "Data metrics: DOWN"
            fi
            
            echo "Drift metrics..."
            if curl -s http://localhost:9095/metrics > /dev/null 2>&1; then
                log_info "Drift metrics: UP"
            else
                log_error "Drift metrics: DOWN"
            fi
            
            echo "Training metrics..."
            if curl -s http://localhost:9096/metrics > /dev/null 2>&1; then
                log_info "Training metrics: UP"
            else
                log_error "Training metrics: DOWN"
            fi
            
            echo "Airflow..."
            if curl -s http://localhost:8080/health > /dev/null 2>&1; then
                log_info "Airflow: UP"
            else
                log_error "Airflow: DOWN"
            fi
            
            echo ""
            log_step "Container status:"
            docker-compose -f docker-compose.vm1.yml ps
            ;;
        8)
            log_warn "⚠️  WARNING: This will remove all volumes (PostgreSQL, MLflow data, etc.)"
            read -p "Are you sure? Type 'yes' to confirm: " confirm
            if [ "$confirm" = "yes" ]; then
                docker-compose -f docker-compose.vm1.yml down -v
                log_info "Cleanup completed ✅"
            else
                log_info "Cancelled"
            fi
            ;;
        *)
            log_error "Invalid choice"
            exit 1
            ;;
    esac
}

# Function to handle VM2 operations
handle_vm2() {
    echo ""
    echo "VM2 Operations"
    echo "=============="
    echo ""
    
    # Check if placeholders are replaced
    if grep -q "<VM1_IP>\|<AZURE_WEBAPP_URL>" monitoring/prometheus/prometheus.vm2.yml 2>/dev/null; then
        log_error "Configuration error: Placeholders found in prometheus.vm2.yml"
        echo ""
        echo "Please replace the following before deployment:"
        echo "  - <VM1_IP> with actual VM1 IP address"
        echo "  - <AZURE_WEBAPP_URL> with actual Azure Web App URL"
        echo ""
        echo "Run: ./deploy-vm2-monitoring.sh for automated setup"
        exit 1
    fi
    
    echo "What would you like to do?"
    echo "  1) Pull images"
    echo "  2) Start all services"
    echo "  3) Start specific services"
    echo "  4) Stop all services"
    echo "  5) View logs"
    echo "  6) Restart services"
    echo "  7) Check service health"
    echo "  8) Validate Prometheus config"
    echo "  9) Clean up (down + remove volumes)"
    echo ""
    read -p "Choice (1-9): " action

    case $action in
        1)
            log_info "Pulling monitoring images..."
            docker-compose -f docker-compose.vm2.yml pull
            log_info "Images pulled successfully ✅"
            ;;
        2)
            log_info "Starting all VM2 monitoring services..."
            docker-compose -f docker-compose.vm2.yml up -d
            log_info "All services started ✅"
            echo ""
            echo "Available interfaces on VM2:"
            echo "  - Prometheus:    http://localhost:9090"
            echo "  - Grafana:       http://localhost:3000 (admin/admin)"
            echo "  - Alertmanager:  http://localhost:9093"
            echo "  - Node Exporter: http://localhost:9100"
            ;;
        3)
            echo "Available services: prometheus, grafana, alertmanager, node-exporter"
            read -p "Services to start (separated by spaces): " services
            log_info "Starting $services..."
            docker-compose -f docker-compose.vm2.yml up -d $services
            log_info "Selected services started ✅"
            ;;
        4)
            log_info "Stopping all VM2 services..."
            docker-compose -f docker-compose.vm2.yml down
            log_info "All services stopped ✅"
            ;;
        5)
            read -p "Service name (leave empty for all): " service
            if [ -z "$service" ]; then
                docker-compose -f docker-compose.vm2.yml logs -f --tail=100
            else
                docker-compose -f docker-compose.vm2.yml logs -f --tail=100 $service
            fi
            ;;
        6)
            read -p "Service to restart (leave empty for all): " service
            if [ -z "$service" ]; then
                log_info "Restarting all services..."
                docker-compose -f docker-compose.vm2.yml restart
            else
                log_info "Restarting $service..."
                docker-compose -f docker-compose.vm2.yml restart $service
            fi
            log_info "Restart completed ✅"
            ;;
        7)
            log_info "Checking monitoring service health..."
            echo ""
            
            echo "Prometheus..."
            if curl -s http://localhost:9090/-/healthy > /dev/null 2>&1; then
                log_info "Prometheus: UP"
            else
                log_error "Prometheus: DOWN"
            fi
            
            echo "Grafana..."
            if curl -s http://localhost:3000/api/health > /dev/null 2>&1; then
                log_info "Grafana: UP"
            else
                log_error "Grafana: DOWN"
            fi
            
            echo "Alertmanager..."
            if curl -s http://localhost:9093/-/healthy > /dev/null 2>&1; then
                log_info "Alertmanager: UP"
            else
                log_error "Alertmanager: DOWN"
            fi
            
            echo "Node Exporter..."
            if curl -s http://localhost:9100/metrics > /dev/null 2>&1; then
                log_info "Node Exporter: UP"
            else
                log_error "Node Exporter: DOWN"
            fi
            
            echo ""
            log_step "Prometheus targets status:"
            curl -s http://localhost:9090/api/v1/targets 2>/dev/null | \
                jq -r '.data.activeTargets[] | "\(.labels.job): \(.health)"' 2>/dev/null || \
                echo "  (Unable to fetch targets - Prometheus may be starting)"
            
            echo ""
            log_step "Container status:"
            docker-compose -f docker-compose.vm2.yml ps
            ;;
        8)
            log_info "Validating Prometheus configuration..."
            if docker run --rm -v $(pwd)/monitoring/prometheus:/etc/prometheus prom/prometheus:latest \
                promtool check config /etc/prometheus/prometheus.vm2.yml > /dev/null 2>&1; then
                log_info "Prometheus config is valid ✅"
            else
                log_error "Prometheus config has errors:"
                docker run --rm -v $(pwd)/monitoring/prometheus:/etc/prometheus prom/prometheus:latest \
                    promtool check config /etc/prometheus/prometheus.vm2.yml
                exit 1
            fi
            ;;
        9)
            log_warn "⚠️  WARNING: This will remove all volumes (Prometheus data, Grafana dashboards, etc.)"
            read -p "Are you sure? Type 'yes' to confirm: " confirm
            if [ "$confirm" = "yes" ]; then
                docker-compose -f docker-compose.vm2.yml down -v
                log_info "Cleanup completed ✅"
            else
                log_info "Cancelled"
            fi
            ;;
        *)
            log_error "Invalid choice"
            exit 1
            ;;
    esac
}

# Execute based on target
case $TARGET in
    VM1)
        handle_vm1
        ;;
    VM2)
        handle_vm2
        ;;
    BOTH)
        echo ""
        log_info "Processing VM1..."
        handle_vm1
        
        echo ""
        log_info "Processing VM2..."
        handle_vm2
        ;;
esac

echo ""
log_info "Done!"
