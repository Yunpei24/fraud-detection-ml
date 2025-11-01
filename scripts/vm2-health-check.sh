#!/bin/bash
# VM2 Health Check Script
# Comprehensive health checks for VM2 monitoring services

set -e

echo "🔍 VM2 Monitoring Services Health Check"
echo "======================================="
echo ""

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
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

echo "📊 Monitoring Services:"
echo "-----------------------"

# Prometheus
check_service "Prometheus" "curl -s http://localhost:9090/-/healthy" "Prometheus is Healthy"

# Grafana
check_service "Grafana" "curl -s http://localhost:3000/api/health" '"database":"ok"'

# Alertmanager
check_service "Alertmanager" "curl -s http://localhost:9093/-/healthy" "OK"

# Node Exporter
check_service "Node Exporter" "curl -s http://localhost:9100/metrics" "node_"

echo ""
echo "🔗 Cross-Subscription Connectivity:"
echo "-----------------------------------"

# Check if VM1 IP is configured
VM1_IP=$(grep -oP '(?<=targets: \[\")[^"]*' monitoring/prometheus.vm2.yml | head -1 | sed 's/<VM1_IP>//g')

if [ -z "$VM1_IP" ] || [[ $VM1_IP == *"<VM1_IP>"* ]]; then
    echo -e "${YELLOW}⚠️  VM1 IP not configured in Prometheus config${NC}"
    echo "   Please run: bash scripts/deploy-vm2.sh and choose option 8"
    ((ISSUES++))
else
    echo "VM1 IP configured: $VM1_IP"
    echo ""

    # Test connectivity to VM1 services
    echo "Testing connectivity to VM1 services:"
    check_service "VM1 Data Service" "curl -s --max-time 10 http://$VM1_IP:9091/metrics" "fraud_data_"
    check_service "VM1 Drift Service" "curl -s --max-time 10 http://$VM1_IP:9092/metrics" "fraud_drift_"
    check_service "VM1 Training Service" "curl -s --max-time 10 http://$VM1_IP:9093/metrics" "fraud_training_"
fi

echo ""
echo "📈 Prometheus Targets Status:"
echo "-----------------------------"

# Check Prometheus targets
if curl -s http://localhost:9090/api/v1/targets | jq -r '.data.activeTargets[] | "\(.labels.job): \(.health)"' > /tmp/targets 2>/dev/null; then
    echo "Active targets:"
    cat /tmp/targets
    echo ""

    # Check for unhealthy targets
    UNHEALTHY=$(grep -c "down\|unknown" /tmp/targets || echo "0")
    if [ "$UNHEALTHY" -gt 0 ]; then
        echo -e "${YELLOW}⚠️  Found $UNHEALTHY unhealthy targets${NC}"
        ((ISSUES++))
    else
        echo -e "${GREEN}✅ All targets are healthy${NC}"
    fi
else
    echo -e "${RED}❌ Could not query Prometheus targets${NC}"
    ((ISSUES++))
fi

echo ""
echo "📊 Grafana Dashboards:"
echo "----------------------"

# Check if dashboards are provisioned
DASHBOARDS=$(curl -s -u admin:admin_dev_2024 http://localhost:3000/api/search | jq -r '.[].title' 2>/dev/null || echo "")

if [ -n "$DASHBOARDS" ]; then
    echo "Provisioned dashboards:"
    echo "$DASHBOARDS" | sed 's/^/  • /'
    echo ""

    # Check if our specific dashboards exist
    EXPECTED_DASHBOARDS=("System Overview" "API Performance" "Data Pipeline" "Drift Detection")
    for dashboard in "${EXPECTED_DASHBOARDS[@]}"; do
        if echo "$DASHBOARDS" | grep -q "$dashboard"; then
            echo -e "${GREEN}✓ $dashboard dashboard found${NC}"
        else
            echo -e "${RED}✗ $dashboard dashboard missing${NC}"
            ((ISSUES++))
        fi
    done
else
    echo -e "${RED}❌ Could not retrieve dashboard list${NC}"
    ((ISSUES++))
fi

echo ""
echo "🚨 Alert Rules:"
echo "---------------"

# Check if alert rules are loaded
ALERTS=$(curl -s http://localhost:9090/api/v1/rules | jq -r '.data.groups[]?.rules[]?.name' 2>/dev/null || echo "")

if [ -n "$ALERTS" ]; then
    ALERT_COUNT=$(echo "$ALERTS" | wc -l)
    echo -e "${GREEN}✓ $ALERT_COUNT alert rules loaded${NC}"
    echo "Sample alerts:"
    echo "$ALERTS" | head -5 | sed 's/^/  • /'
    if [ "$ALERT_COUNT" -gt 5 ]; then
        echo "  ... and $(($ALERT_COUNT - 5)) more"
    fi
else
    echo -e "${RED}❌ No alert rules found${NC}"
    ((ISSUES++))
fi

echo ""
echo "📋 Service Status Summary:"
echo "=========================="

# Show docker-compose status
docker-compose -f docker-compose.vm2.yml ps --format "table {{.Name}}\t{{.Status}}\t{{.Ports}}"

echo ""
echo "🎯 Health Check Results:"
echo "========================"

VM2_IP=$(curl -s ifconfig.me 2>/dev/null || echo "unknown")

if [ $ISSUES -eq 0 ]; then
    echo -e "${GREEN}✅ All monitoring services are healthy! VM2 is fully operational.${NC}"
    echo ""
    echo "🎉 Production monitoring stack is ready!"
    echo ""
    echo "Access URLs:"
    echo "• Grafana: http://$VM2_IP:3000 (admin/admin_dev_2024)"
    echo "• Prometheus: http://$VM2_IP:9090"
    echo "• Alertmanager: http://$VM2_IP:9093"
    echo ""
    echo "Available dashboards:"
    echo "• System Overview - Service health and alerts"
    echo "• API Performance - Request metrics and latency"
    echo "• Data Pipeline - Processing and quality metrics"
    echo "• Drift Detection - Model monitoring and retraining"
else
    echo -e "${RED}❌ Found $ISSUES issues that need attention.${NC}"
    echo ""
    echo "Common issues:"
    echo "1. VM1 IP not configured: Run 'bash scripts/deploy-vm2.sh' option 8"
    echo "2. NSG rules blocking VM1->VM2 traffic"
    echo "3. VM1 services not running or metrics not exposed"
    echo "4. Check service logs: docker-compose -f docker-compose.vm2.yml logs <service>"
fi

echo ""
echo "🔍 Detailed logs available with: docker-compose -f docker-compose.vm2.yml logs -f"