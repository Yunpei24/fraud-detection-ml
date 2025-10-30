#!/bin/bash
# Production Deployment Verification Script
# Run this on each VM after deployment to verify everything is working

set -e

echo "🔍 Fraud Detection ML - Production Deployment Verification"
echo "=========================================================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to check service
check_service() {
    local name=$1
    local url=$2
    local expected_status=${3:-200}

    echo -n "Checking $name ($url)... "

    if curl -s --max-time 10 -o /dev/null -w "%{http_code}" "$url" | grep -q "$expected_status"; then
        echo -e "${GREEN}✅ OK${NC}"
        return 0
    else
        echo -e "${RED}❌ FAILED${NC}"
        return 1
    fi
}

# Function to check Docker service
check_docker_service() {
    local service_name=$1
    local container_name=$2

    echo -n "Checking Docker service $service_name... "

    if docker ps --filter "name=$container_name" --filter "status=running" | grep -q "$container_name"; then
        echo -e "${GREEN}✅ RUNNING${NC}"
        return 0
    else
        echo -e "${RED}❌ NOT RUNNING${NC}"
        return 1
    fi
}

# Detect which VM we're on based on running services
echo "Detecting VM type..."
if docker ps --filter "name=fraud-postgres" | grep -q fraud-postgres; then
    VM_TYPE="VM1"
    echo "📍 Detected VM1 (Application Services)"
elif docker ps --filter "name=fraud-prometheus" | grep -q fraud-prometheus; then
    VM_TYPE="VM2"
    echo "📍 Detected VM2 (Monitoring Services)"
else
    echo -e "${RED}❌ Cannot determine VM type. Are Docker services running?${NC}"
    exit 1
fi

echo ""
echo "🔧 Checking Docker Services..."
echo "------------------------------"

FAILED_SERVICES=()

if [ "$VM_TYPE" = "VM1" ]; then
    # VM1 Services
    services=(
        "PostgreSQL:fraud-postgres"
        "Redis:fraud-redis"
        "Kafka:fraud-kafka"
        "Zookeeper:fraud-zookeeper"
        "MLflow:fraud-mlflow"
        "Data Service:fraud-data"
        "Drift Service:fraud-drift"
        "Training Service:fraud-training"
        "Airflow Webserver:fraud-airflow-webserver"
        "Airflow Scheduler:fraud-airflow-scheduler"
    )

    for service in "${services[@]}"; do
        name=$(echo $service | cut -d: -f1)
        container=$(echo $service | cut -d: -f2)
        if ! check_docker_service "$name" "$container"; then
            FAILED_SERVICES+=("$name")
        fi
    done

elif [ "$VM_TYPE" = "VM2" ]; then
    # VM2 Services
    services=(
        "Prometheus:fraud-prometheus"
        "Grafana:fraud-grafana"
    )

    for service in "${services[@]}"; do
        name=$(echo $service | cut -d: -f1)
        container=$(echo $service | cut -d: -f2)
        if ! check_docker_service "$name" "$container"; then
            FAILED_SERVICES+=("$name")
        fi
    done
fi

echo ""
echo "🌐 Checking Network Endpoints..."
echo "---------------------------------"

if [ "$VM_TYPE" = "VM1" ]; then
    # VM1 Endpoints
    endpoints=(
        "Data Metrics:http://localhost:9091/metrics"
        "Drift Metrics:http://localhost:9092/metrics"
        "Training Metrics:http://localhost:9093/metrics"
        "Airflow:http://localhost:8080/health"
        "MLflow:http://localhost:5000/health"
    )

    for endpoint in "${endpoints[@]}"; do
        name=$(echo $endpoint | cut -d: -f1)
        url=$(echo $endpoint | cut -d: -f2)
        if ! check_service "$name" "$url"; then
            FAILED_SERVICES+=("$name endpoint")
        fi
    done

elif [ "$VM_TYPE" = "VM2" ]; then
    # VM2 Endpoints
    endpoints=(
        "Grafana:http://localhost:3000/api/health"
        "Prometheus:http://localhost:9090/-/healthy"
    )

    for endpoint in "${endpoints[@]}"; do
        name=$(echo $endpoint | cut -d: -f1)
        url=$(echo $endpoint | cut -d: -f2)
        if ! check_service "$name" "$url"; then
            FAILED_SERVICES+=("$name endpoint")
        fi
    done

    # Check if Prometheus can scrape VM1 (if VM1_IP is configured)
    echo ""
    echo "🔗 Checking Cross-Subscription Connectivity..."
    echo "---------------------------------------------"

    # Try to extract VM1_IP from prometheus config
    VM1_IP=$(grep -oP '(?<=targets: \[\')[^:]+' monitoring/prometheus.vm2.yml | grep -v '<VM1_IP>' | head -1)
    if [ -z "$VM1_IP" ] || [[ $VM1_IP == *"<VM1_IP>"* ]]; then
        echo -e "${YELLOW}⚠️  VM1_IP not configured in prometheus.vm2.yml${NC}"
        echo "   Please replace <VM1_IP> with your colleague's VM1 public IP address"
        echo "   This is required for cross-subscription monitoring to work"
    else
        echo "Testing connectivity to VM1 services at $VM1_IP..."
        echo "Note: These tests may fail if NSG rules aren't configured yet"

        # Test each metrics endpoint
        endpoints=("9091:data" "9092:drift" "9093:training")
        for endpoint in "${endpoints[@]}"; do
            port=$(echo $endpoint | cut -d: -f1)
            service=$(echo $endpoint | cut -d: -f2)
            echo -n "  $service metrics (port $port): "
            if curl -s --max-time 5 "http://$VM1_IP:$port/metrics" > /dev/null; then
                echo -e "${GREEN}✅ OK${NC}"
            else
                echo -e "${RED}❌ FAILED${NC}"
                echo "     Possible causes:"
                echo "     - NSG rules not configured on VM1"
                echo "     - VM1 IP address changed"
                echo "     - VM1 services not running"
                echo "     - Firewall blocking traffic"
            fi
        done

        # Check Prometheus targets status
        echo ""
        echo "Checking Prometheus targets status..."
        TARGETS_STATUS=$(curl -s "http://localhost:9090/api/v1/targets" | jq -r '.data.activeTargets[] | select(.labels.instance | contains("vm1")) | "\(.labels.job): \(.health)"')
        if [ -n "$TARGETS_STATUS" ]; then
            echo "$TARGETS_STATUS" | while read line; do
                if [[ $line == *"up"* ]]; then
                    echo -e "  $line ${GREEN}✅${NC}"
                else
                    echo -e "  $line ${RED}❌${NC}"
                fi
            done
        else
            echo -e "${YELLOW}⚠️  No VM1 targets found in Prometheus${NC}"
        fi
    fi
fi

echo ""
echo "📊 Deployment Summary"
echo "====================="

if [ ${#FAILED_SERVICES[@]} -eq 0 ]; then
    echo -e "${GREEN}✅ All checks passed! Deployment looks good.${NC}"
    if [ "$VM_TYPE" = "VM1" ]; then
        echo ""
        echo "Next steps:"
        echo "1. Note your VM1 IP address: $(curl -s ifconfig.me)"
        echo "2. Deploy VM2 with this IP address"
        echo "3. Configure Azure Web App for API service"
    elif [ "$VM_TYPE" = "VM2" ]; then
        echo ""
        echo "Next steps:"
        echo "1. Access Grafana: http://$(curl -s ifconfig.me):3000"
        echo "2. Verify Prometheus targets are UP"
        echo "3. Import or create dashboards"
    fi
else
    echo -e "${RED}❌ Some checks failed. Issues found:${NC}"
    for service in "${FAILED_SERVICES[@]}"; do
        echo -e "   - $service"
    done
    echo ""
    echo "Troubleshooting:"
    echo "1. Check Docker logs: docker logs <container_name>"
    echo "2. Verify environment variables in .env file"
    echo "3. Check network connectivity and firewall rules"
    echo "4. Review the PRODUCTION_DEPLOYMENT_GUIDE.md"
    exit 1
fi

echo ""
echo "📚 Useful Commands:"
echo "-------------------"
if [ "$VM_TYPE" = "VM1" ]; then
    echo "• View logs: docker-compose -f docker-compose.vm1.yml logs -f <service>"
    echo "• Restart service: docker-compose -f docker-compose.vm1.yml restart <service>"
    echo "• Check metrics: curl http://localhost:9091/metrics"
elif [ "$VM_TYPE" = "VM2" ]; then
    echo "• View logs: docker-compose -f docker-compose.vm2.yml logs -f <service>"
    echo "• Restart service: docker-compose -f docker-compose.vm2.yml restart <service>"
    echo "• Grafana: http://localhost:3000 (admin/admin_change_me)"
    echo "• Prometheus: http://localhost:9090"
fi