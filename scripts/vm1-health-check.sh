#!/bin/bash
# VM1 Health Check Script
# Comprehensive health checks for all VM1 services

set -e

echo "🔍 VM1 Services Health Check"
echo "============================"
echo ""

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m'

# Counter for issues
ISSUES=0

check_service() {
    local service_name=$1
    local check_command=$2
    local expected_output=$3

    echo -n "Checking $service_name... "

    if eval "$check_command" > /dev/null 2>&1; then
        if [ -z "$expected_output" ] || eval "$check_command" | grep -q "$expected_output"; then
            echo -e "${GREEN}✓ PASS${NC}"
            return 0
        else
            echo -e "${RED}✗ FAIL${NC} (unexpected output)"
            ((ISSUES++))
            return 1
        fi
    else
        echo -e "${RED}✗ FAIL${NC}"
        ((ISSUES++))
        return 1
    fi
}

echo "🗄️  Infrastructure Services:"
echo "-----------------------------"

# PostgreSQL
check_service "PostgreSQL" "docker exec fraud-postgres pg_isready -U fraud_user -d fraud_detection" "accepting connections"

# Redis
check_service "Redis" "docker exec fraud-redis redis-cli ping" "PONG"

# Zookeeper
check_service "Zookeeper" "docker exec fraud-zookeeper nc -z localhost 2181" ""

# Kafka
check_service "Kafka" "docker exec fraud-kafka kafka-broker-api-versions --bootstrap-server localhost:9092" "localhost:9092"

echo ""
echo "🔧 Application Services:"
echo "------------------------"

# MLflow
check_service "MLflow" "curl -s http://localhost:5000/health" "healthy"

# Data Service
check_service "Data Service" "curl -s http://localhost:9091/metrics" "fraud_data_"

# Drift Service
check_service "Drift Service" "curl -s http://localhost:9092/metrics" "fraud_drift_"

# Training Service
check_service "Training Service" "curl -s http://localhost:9093/metrics" "fraud_training_"

echo ""
echo "🎯 Orchestration Services:"
echo "--------------------------"

# Airflow Webserver
check_service "Airflow Webserver" "curl -s http://localhost:8080/health" '"status":"healthy"'

# Airflow Scheduler (check if process is running)
check_service "Airflow Scheduler" "docker exec fraud-airflow-scheduler ps aux | grep scheduler" "scheduler"

echo ""
echo "📊 Metrics Exposure Check:"
echo "--------------------------"

# Check if metrics are accessible from external (for VM2 scraping)
VM1_IP=$(curl -s ifconfig.me 2>/dev/null || echo "unknown")

if [ "$VM1_IP" != "unknown" ]; then
    echo "VM1 Public IP: $VM1_IP"
    echo ""

    echo "Testing external access to metrics endpoints:"
    check_service "Data Metrics (external)" "curl -s --max-time 5 http://$VM1_IP:9091/metrics" "fraud_data_"
    check_service "Drift Metrics (external)" "curl -s --max-time 5 http://$VM1_IP:9092/metrics" "fraud_drift_"
    check_service "Training Metrics (external)" "curl -s --max-time 5 http://$VM1_IP:9093/metrics" "fraud_training_"
else
    echo "Could not determine public IP. External access tests skipped."
fi

echo ""
echo "📋 Service Status Summary:"
echo "=========================="

# Show docker-compose status
docker-compose -f docker-compose.vm1.yml ps --format "table {{.Name}}\t{{.Status}}\t{{.Ports}}"

echo ""
echo "🎯 Health Check Results:"
echo "========================"

if [ $ISSUES -eq 0 ]; then
    echo -e "${GREEN}✅ All services are healthy! VM1 is ready for production.${NC}"
    echo ""
    echo "Next steps:"
    echo "1. Share your VM1 public IP ($VM1_IP) with your colleague"
    echo "2. Configure NSG rules to allow VM2 access to ports 9091-9093"
    echo "3. Proceed with VM2 deployment"
else
    echo -e "${RED}❌ Found $ISSUES issues that need attention.${NC}"
    echo ""
    echo "Troubleshooting:"
    echo "1. Check service logs: docker-compose -f docker-compose.vm1.yml logs <service_name>"
    echo "2. Restart failed services: docker-compose -f docker-compose.vm1.yml restart <service_name>"
    echo "3. Check resource usage: docker stats"
fi

echo ""
echo "🔍 Detailed logs available with: docker-compose -f docker-compose.vm1.yml logs -f"