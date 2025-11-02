#!/bin/bash

################################################################################
# Fraud Detection ML - Production Deployment Script
################################################################################
# Description: Automated deployment script for VM1 and VM2 in production
# Usage: bash scripts/deploy-production.sh [--vm1|--vm2|--both|--validate]
# Author: MLOps Team
# Date: 2025-11-01
################################################################################

set -e  # Exit on error
set -u  # Exit on undefined variable

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
MAGENTA='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Logging functions
log_info() {
    echo -e "${CYAN}ℹ️  [INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}✅ [SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}⚠️  [WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}❌ [ERROR]${NC} $1"
}

log_step() {
    echo ""
    echo -e "${MAGENTA}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${MAGENTA}  $1${NC}"
    echo -e "${MAGENTA}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo ""
}

# Banner
show_banner() {
    echo -e "${BLUE}"
    cat << "EOF"
╔═══════════════════════════════════════════════════════════════╗
║                                                               ║
║   🚀  FRAUD DETECTION ML - PRODUCTION DEPLOYMENT  🚀         ║
║                                                               ║
║   Automated deployment for VM1 (App) and VM2 (Monitoring)   ║
║                                                               ║
╚═══════════════════════════════════════════════════════════════╝
EOF
    echo -e "${NC}"
}

# Check if running as root
check_root() {
    if [ "$EUID" -eq 0 ]; then 
        log_error "Please do not run this script as root"
        exit 1
    fi
}

# Check prerequisites
check_prerequisites() {
    log_step "CHECKING PREREQUISITES"
    
    local missing_deps=0
    
    # Check Docker
    if ! command -v docker &> /dev/null; then
        log_error "Docker is not installed"
        missing_deps=1
    else
        log_success "Docker is installed ($(docker --version))"
    fi
    
    # Check Docker Compose
    if ! command -v docker-compose &> /dev/null && ! docker compose version &> /dev/null; then
        log_error "Docker Compose is not installed"
        missing_deps=1
    else
        log_success "Docker Compose is installed"
    fi
    
    # Check Git
    if ! command -v git &> /dev/null; then
        log_error "Git is not installed"
        missing_deps=1
    else
        log_success "Git is installed ($(git --version))"
    fi
    
    # Check curl
    if ! command -v curl &> /dev/null; then
        log_error "curl is not installed"
        missing_deps=1
    else
        log_success "curl is installed"
    fi
    
    # Check jq
    if ! command -v jq &> /dev/null; then
        log_warning "jq is not installed (optional, recommended for JSON parsing)"
    else
        log_success "jq is installed"
    fi
    
    if [ $missing_deps -eq 1 ]; then
        log_error "Missing required dependencies. Please install them first."
        exit 1
    fi
    
    log_success "All prerequisites met"
}

# Generate secure password
generate_password() {
    local length=${1:-32}
    python3 -c "import secrets; print(secrets.token_urlsafe($length))"
}

# Generate Fernet key
generate_fernet_key() {
    python3 -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())" 2>/dev/null || \
    openssl rand -base64 32
}

# Generate secret key
generate_secret_key() {
    python3 -c "import secrets; print(secrets.token_hex(32))"
}

# Interactive configuration
configure_deployment() {
    log_step "CONFIGURATION"
    
    echo -e "${CYAN}Please provide the following information:${NC}"
    echo ""
    
    # Deployment Type
    echo -e "${YELLOW}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${YELLOW}  Deployment Architecture${NC}"
    echo -e "${YELLOW}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo ""
    echo "Are VM1 and VM2 in the same Azure VNet (same subscription)?"
    echo "  - Same VNet: VMs can communicate via private IPs (more secure)"
    echo "  - Different Accounts: Must use public IPs for communication"
    echo ""
    read -p "Same VNet? (y/n): " SAME_VNET
    
    if [[ "$SAME_VNET" =~ ^[Yy]$ ]]; then
        DEPLOYMENT_TYPE="same-vnet"
        log_success "Deployment mode: SAME VNET (using private IPs)"
    else
        DEPLOYMENT_TYPE="cross-account"
        log_warning "Deployment mode: CROSS-ACCOUNT (using public IPs)"
        log_warning "⚠️  Metrics ports will be exposed on public internet"
        log_warning "⚠️  NSG rules will restrict access to VM2 IP only"
    fi
    
    echo ""
    echo -e "${YELLOW}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${YELLOW}  VM Configuration${NC}"
    echo -e "${YELLOW}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo ""
    
    # VM1 Configuration
    read -p "VM1 Public IP: " VM1_PUBLIC_IP
    
    if [ "$DEPLOYMENT_TYPE" = "same-vnet" ]; then
        read -p "VM1 Private IP: " VM1_PRIVATE_IP
        VM1_METRICS_IP="$VM1_PRIVATE_IP"
        log_info "Prometheus will scrape VM1 via private IP: $VM1_PRIVATE_IP"
    else
        VM1_PRIVATE_IP="$VM1_PUBLIC_IP"
        VM1_METRICS_IP="$VM1_PUBLIC_IP"
        log_warning "Prometheus will scrape VM1 via public IP: $VM1_PUBLIC_IP"
        log_warning "Make sure to configure NSG rules on VM1!"
    fi
    
    echo ""
    
    # VM2 Configuration
    read -p "VM2 Public IP: " VM2_PUBLIC_IP
    
    if [ "$DEPLOYMENT_TYPE" = "same-vnet" ]; then
        read -p "VM2 Private IP: " VM2_PRIVATE_IP
    else
        VM2_PRIVATE_IP="$VM2_PUBLIC_IP"
    fi
    
    echo ""
    echo -e "${YELLOW}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${YELLOW}  Azure Services${NC}"
    echo -e "${YELLOW}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo ""
    
    # Azure Web App
    read -p "Azure Web App URL (without https://): " AZURE_WEBAPP_URL
    
    # Email for alerts
    read -p "Alert Email Address: " ALERT_EMAIL
    
    # Generate passwords
    echo ""
    echo -e "${YELLOW}Generating secure credentials...${NC}"
    POSTGRES_PASSWORD=$(generate_password)
    REDIS_PASSWORD=$(generate_password)
    AIRFLOW_PASSWORD=$(generate_password)
    GRAFANA_PASSWORD=$(generate_password)
    AIRFLOW_FERNET_KEY=$(generate_fernet_key)
    AIRFLOW_SECRET_KEY=$(generate_secret_key)
    
    # DockerHub username
    read -p "DockerHub Username (default: yoshua24): " DOCKERHUB_USERNAME
    DOCKERHUB_USERNAME=${DOCKERHUB_USERNAME:-yoshua24}
    
    # Azure Storage (optional)
    read -p "Azure Storage Connection String (optional, press Enter to skip): " AZURE_STORAGE_CONNECTION_STRING
    
    echo ""
    log_success "Configuration complete"
    
    # Display deployment summary
    echo ""
    echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${CYAN}  Deployment Summary${NC}"
    echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo ""
    echo "Deployment Type: $DEPLOYMENT_TYPE"
    echo "VM1 Public IP:   $VM1_PUBLIC_IP"
    echo "VM1 Private IP:  $VM1_PRIVATE_IP"
    echo "VM1 Metrics IP:  $VM1_METRICS_IP"
    echo "VM2 Public IP:   $VM2_PUBLIC_IP"
    echo "VM2 Private IP:  $VM2_PRIVATE_IP"
    echo "Azure Web App:   https://$AZURE_WEBAPP_URL"
    echo "Alert Email:     $ALERT_EMAIL"
    
    if [ "$DEPLOYMENT_TYPE" = "cross-account" ]; then
        echo ""
        echo -e "${RED}⚠️  CROSS-ACCOUNT DEPLOYMENT WARNINGS:${NC}"
        echo "  • Metrics ports (9091, 9095, 9096) will be exposed publicly on VM1"
        echo "  • NSG rules will restrict access to VM2 IP: $VM2_PUBLIC_IP"
        echo "  • Data transfer between VMs will be charged as egress traffic"
        echo "  • Consider adding Basic Auth for additional security"
    fi
    echo ""
    
    # Save configuration to file
    cat > /tmp/deployment-config.env << EOF
DEPLOYMENT_TYPE=$DEPLOYMENT_TYPE
VM1_PUBLIC_IP=$VM1_PUBLIC_IP
VM1_PRIVATE_IP=$VM1_PRIVATE_IP
VM1_METRICS_IP=$VM1_METRICS_IP
VM2_PUBLIC_IP=$VM2_PUBLIC_IP
VM2_PRIVATE_IP=$VM2_PRIVATE_IP
AZURE_WEBAPP_URL=$AZURE_WEBAPP_URL
ALERT_EMAIL=$ALERT_EMAIL
POSTGRES_PASSWORD=$POSTGRES_PASSWORD
REDIS_PASSWORD=$REDIS_PASSWORD
AIRFLOW_PASSWORD=$AIRFLOW_PASSWORD
GRAFANA_PASSWORD=$GRAFANA_PASSWORD
AIRFLOW_FERNET_KEY=$AIRFLOW_FERNET_KEY
AIRFLOW_SECRET_KEY=$AIRFLOW_SECRET_KEY
DOCKERHUB_USERNAME=$DOCKERHUB_USERNAME
AZURE_STORAGE_CONNECTION_STRING=$AZURE_STORAGE_CONNECTION_STRING
EOF
    
    log_info "Configuration saved to /tmp/deployment-config.env"
}

# Load configuration
load_configuration() {
    if [ -f /tmp/deployment-config.env ]; then
        source /tmp/deployment-config.env
        log_success "Configuration loaded from /tmp/deployment-config.env"
    else
        log_error "Configuration file not found. Please run configuration first."
        exit 1
    fi
}

# Deploy VM1
deploy_vm1() {
    log_step "DEPLOYING VM1 (APPLICATION SERVER)"
    
    log_info "Connecting to VM1..."
    
    # Create .env file for VM1
    cat > /tmp/vm1.env << EOF
# PostgreSQL
POSTGRES_PASSWORD=$POSTGRES_PASSWORD

# Redis
REDIS_PASSWORD=$REDIS_PASSWORD

# Airflow
AIRFLOW_FERNET_KEY=$AIRFLOW_FERNET_KEY
AIRFLOW_SECRET_KEY=$AIRFLOW_SECRET_KEY
AIRFLOW_USERNAME=admin
AIRFLOW_PASSWORD=$AIRFLOW_PASSWORD
AIRFLOW_ADMIN_EMAIL=$ALERT_EMAIL
ALERT_EMAIL=$ALERT_EMAIL

# MLflow
MLFLOW_TRACKING_URI=http://mlflow:5000
MLFLOW_DEFAULT_ARTIFACT_ROOT=/mlflow/artifacts

# DockerHub
DOCKERHUB_USERNAME=$DOCKERHUB_USERNAME

# Azure (optional)
AZURE_STORAGE_CONNECTION_STRING=$AZURE_STORAGE_CONNECTION_STRING
AZURE_STORAGE_CONTAINER_NAME=fraud-models

# Environment
ENVIRONMENT=production
EOF
    
    log_info "Uploading .env file to VM1..."
    scp /tmp/vm1.env azureuser@$VM1_PUBLIC_IP:~/.env
    
    log_info "Executing deployment script on VM1..."
    
    ssh azureuser@$VM1_PUBLIC_IP << 'ENDSSH'
set -e

echo "🔹 Cloning repository..."
if [ ! -d "fraud-detection-ml" ]; then
    git clone https://github.com/Yunpei24/fraud-detection-ml.git
fi

cd fraud-detection-ml
git checkout develop
git pull origin develop

echo "🔹 Moving .env file..."
mv ~/.env .env

echo "🔹 Pulling Docker images..."
docker-compose -f docker-compose.vm1.yml pull

echo "🔹 Starting services..."
docker-compose -f docker-compose.vm1.yml up -d

echo "🔹 Waiting for services to be healthy (60s)..."
sleep 60

echo "🔹 Checking service status..."
docker-compose -f docker-compose.vm1.yml ps

echo "✅ VM1 deployment complete!"
ENDSSH
    
    if [ $? -eq 0 ]; then
        log_success "VM1 deployed successfully"
    else
        log_error "VM1 deployment failed"
        exit 1
    fi
}

# Deploy VM2
deploy_vm2() {
    log_step "DEPLOYING VM2 (MONITORING SERVER)"
    
    log_info "Connecting to VM2..."
    
    # Create .env file for VM2
    cat > /tmp/vm2.env << EOF
# Grafana
GRAFANA_USER=admin
GRAFANA_PASSWORD=$GRAFANA_PASSWORD
EOF
    
    log_info "Uploading .env file to VM2..."
    scp /tmp/vm2.env azureuser@$VM2_PUBLIC_IP:~/.env
    
    log_info "Configuring Prometheus targets..."
    
    ssh azureuser@$VM2_PUBLIC_IP << ENDSSH
set -e

echo "🔹 Cloning repository..."
if [ ! -d "fraud-detection-ml" ]; then
    git clone https://github.com/Yunpei24/fraud-detection-ml.git
fi

cd fraud-detection-ml
git checkout develop
git pull origin develop

echo "🔹 Moving .env file..."
mv ~/.env .env

echo "🔹 Configuring Prometheus targets..."
sed -i "s/<VM1_IP>/$VM1_METRICS_IP/g" monitoring/prometheus.vm2.yml
sed -i "s/<AZURE_WEBAPP_URL>/$AZURE_WEBAPP_URL/g" monitoring/prometheus.vm2.yml

echo "🔹 Verifying configuration..."
grep -E "targets.*:909" monitoring/prometheus.vm2.yml

echo "🔹 Pulling Docker images..."
docker-compose -f docker-compose.vm2.yml pull

echo "🔹 Starting monitoring services..."
docker-compose -f docker-compose.vm2.yml up -d

echo "🔹 Waiting for services to be healthy (30s)..."
sleep 30

echo "🔹 Checking service status..."
docker-compose -f docker-compose.vm2.yml ps

echo "✅ VM2 deployment complete!"
ENDSSH
    
    if [ $? -eq 0 ]; then
        log_success "VM2 deployed successfully"
    else
        log_error "VM2 deployment failed"
        exit 1
    fi
}

# Validate deployment
validate_deployment() {
    log_step "VALIDATING DEPLOYMENT"
    
    local validation_failed=0
    
    # Test VM1 metrics endpoints
    log_info "Testing VM1 metrics endpoints..."
    
    test_endpoint "Data Service (9091)" "http://$VM1_METRICS_IP:9091/metrics"
    [ $? -ne 0 ] && validation_failed=1
    
    test_endpoint "Drift Service (9095)" "http://$VM1_METRICS_IP:9095/metrics"
    [ $? -ne 0 ] && validation_failed=1
    
    test_endpoint "Training Service (9096)" "http://$VM1_METRICS_IP:9096/metrics"
    [ $? -ne 0 ] && validation_failed=1
    
    # Test VM2 monitoring stack
    log_info "Testing VM2 monitoring stack..."
    
    test_endpoint "Prometheus (9090)" "http://$VM2_PUBLIC_IP:9090/-/healthy"
    [ $? -ne 0 ] && validation_failed=1
    
    test_endpoint "Grafana (3000)" "http://$VM2_PUBLIC_IP:3000/api/health"
    [ $? -ne 0 ] && validation_failed=1
    
    test_endpoint "AlertManager (9093)" "http://$VM2_PUBLIC_IP:9093/-/healthy"
    [ $? -ne 0 ] && validation_failed=1
    
    # Test Prometheus targets
    log_info "Checking Prometheus targets..."
    
    if command -v jq &> /dev/null; then
        targets_response=$(curl -s "http://$VM2_PUBLIC_IP:9090/api/v1/targets")
        targets_up=$(echo "$targets_response" | jq -r '.data.activeTargets[] | select(.health=="up") | .labels.job' | wc -l)
        targets_total=$(echo "$targets_response" | jq -r '.data.activeTargets | length')
        
        log_info "Prometheus targets: $targets_up / $targets_total UP"
        
        if [ "$targets_up" -lt "$targets_total" ]; then
            log_warning "Some targets are DOWN:"
            echo "$targets_response" | jq -r '.data.activeTargets[] | select(.health!="up") | "\(.labels.job): \(.lastError)"'
            validation_failed=1
        else
            log_success "All Prometheus targets are UP"
        fi
    else
        log_warning "jq not installed, skipping detailed targets check"
    fi
    
    # Test Azure Web App
    log_info "Testing Azure Web App..."
    test_endpoint "API Health" "https://$AZURE_WEBAPP_URL/health"
    [ $? -ne 0 ] && validation_failed=1
    
    if [ $validation_failed -eq 1 ]; then
        log_error "Validation failed. Please check the errors above."
        return 1
    else
        log_success "All validation tests passed!"
        return 0
    fi
}

# Test endpoint helper
test_endpoint() {
    local name=$1
    local url=$2
    local expected_code=${3:-200}
    
    echo -n "  Testing $name... "
    http_code=$(curl -s -o /dev/null -w "%{http_code}" -m 10 "$url" 2>/dev/null || echo "000")
    
    if [ "$http_code" = "$expected_code" ]; then
        echo -e "${GREEN}✓ OK ($http_code)${NC}"
        return 0
    else
        echo -e "${RED}✗ FAILED ($http_code)${NC}"
        return 1
    fi
}

# Display credentials
display_credentials() {
    log_step "DEPLOYMENT CREDENTIALS"
    
    echo -e "${YELLOW}⚠️  IMPORTANT: Save these credentials securely!${NC}"
    echo ""
    
    cat << EOF
╔═══════════════════════════════════════════════════════════════╗
║                    DEPLOYMENT CREDENTIALS                     ║
╚═══════════════════════════════════════════════════════════════╝

VM1 (Application Server)
━━━━━━━━━━━━━━━━━━━━━━━━
  SSH:       ssh azureuser@$VM1_PUBLIC_IP
  Airflow:   http://$VM1_PUBLIC_IP:8080
    - Username: admin
    - Password: $AIRFLOW_PASSWORD
  MLflow:    http://$VM1_PUBLIC_IP:5000

VM2 (Monitoring Server)
━━━━━━━━━━━━━━━━━━━━━━━━
  SSH:       ssh azureuser@$VM2_PUBLIC_IP
  Grafana:   http://$VM2_PUBLIC_IP:3000
    - Username: admin
    - Password: $GRAFANA_PASSWORD
  Prometheus: http://$VM2_PUBLIC_IP:9090

Database & Cache
━━━━━━━━━━━━━━━━
  PostgreSQL Password: $POSTGRES_PASSWORD
  Redis Password:      $REDIS_PASSWORD

Azure Web App
━━━━━━━━━━━━━
  URL: https://$AZURE_WEBAPP_URL

Alert Email
━━━━━━━━━━━
  Email: $ALERT_EMAIL

EOF
    
    # Save to file
    cat > ~/fraud-detection-credentials.txt << EOF
FRAUD DETECTION ML - PRODUCTION CREDENTIALS
Generated: $(date)

VM1_PUBLIC_IP=$VM1_PUBLIC_IP
VM1_PRIVATE_IP=$VM1_PRIVATE_IP
VM2_PUBLIC_IP=$VM2_PUBLIC_IP
VM2_PRIVATE_IP=$VM2_PRIVATE_IP
AZURE_WEBAPP_URL=$AZURE_WEBAPP_URL

POSTGRES_PASSWORD=$POSTGRES_PASSWORD
REDIS_PASSWORD=$REDIS_PASSWORD
AIRFLOW_USERNAME=admin
AIRFLOW_PASSWORD=$AIRFLOW_PASSWORD
GRAFANA_USERNAME=admin
GRAFANA_PASSWORD=$GRAFANA_PASSWORD
AIRFLOW_FERNET_KEY=$AIRFLOW_FERNET_KEY
AIRFLOW_SECRET_KEY=$AIRFLOW_SECRET_KEY
ALERT_EMAIL=$ALERT_EMAIL
EOF
    
    log_info "Credentials saved to: ~/fraud-detection-credentials.txt"
    log_warning "Please move this file to a secure location!"
}

# Rollback function
rollback() {
    log_step "ROLLBACK"
    
    read -p "Are you sure you want to rollback? (yes/no): " confirm
    if [ "$confirm" != "yes" ]; then
        log_info "Rollback cancelled"
        return
    fi
    
    log_warning "Rolling back deployment..."
    
    if [ -n "${VM1_PUBLIC_IP:-}" ]; then
        log_info "Stopping VM1 services..."
        ssh azureuser@$VM1_PUBLIC_IP "cd fraud-detection-ml && docker-compose -f docker-compose.vm1.yml down" || true
    fi
    
    if [ -n "${VM2_PUBLIC_IP:-}" ]; then
        log_info "Stopping VM2 services..."
        ssh azureuser@$VM2_PUBLIC_IP "cd fraud-detection-ml && docker-compose -f docker-compose.vm2.yml down" || true
    fi
    
    log_success "Rollback complete"
}

# Show usage
show_usage() {
    cat << EOF
Usage: bash scripts/deploy-production.sh [OPTIONS]

OPTIONS:
    --configure     Run configuration wizard
    --vm1           Deploy VM1 (Application Server) only
    --vm2           Deploy VM2 (Monitoring Server) only
    --both          Deploy both VM1 and VM2 (default)
    --validate      Validate existing deployment
    --rollback      Rollback deployment
    --credentials   Display credentials
    --nsg-rules     Display NSG configuration commands (for cross-account)
    --help          Show this help message

EXAMPLES:
    # Full deployment with configuration
    bash scripts/deploy-production.sh --configure --both

    # Deploy VM1 only (after configuration)
    bash scripts/deploy-production.sh --vm1

    # Validate deployment
    bash scripts/deploy-production.sh --validate

    # Display NSG rules for cross-account setup
    bash scripts/deploy-production.sh --nsg-rules

    # Display credentials
    bash scripts/deploy-production.sh --credentials

EOF
}

# Display NSG configuration for cross-account
show_nsg_rules() {
    log_step "NSG CONFIGURATION FOR CROSS-ACCOUNT DEPLOYMENT"
    
    load_configuration
    
    if [ "$DEPLOYMENT_TYPE" != "cross-account" ]; then
        log_warning "This deployment is configured for same-vnet mode."
        log_info "NSG rules for inter-VM communication are handled by default VNet rules."
        return
    fi
    
    cat << EOF

${YELLOW}╔═══════════════════════════════════════════════════════════════╗
║         AZURE NSG RULES - CROSS-ACCOUNT DEPLOYMENT           ║
╚═══════════════════════════════════════════════════════════════╝${NC}

${RED}⚠️  IMPORTANT: VM1 and VM2 are in different Azure accounts/subscriptions.${NC}
${RED}⚠️  You MUST configure NSG rules to allow communication via public IPs.${NC}

${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}
${CYAN}  STEP 1: Configure NSG on VM1 (Allow Prometheus from VM2)${NC}
${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}

${GREEN}# Login to Azure Account #1 (where VM1 is located)${NC}
az login
az account set --subscription "<VM1_SUBSCRIPTION_ID>"

${GREEN}# Create rule to allow Prometheus scraping from VM2${NC}
az network nsg rule create \\
  --resource-group <VM1_RESOURCE_GROUP> \\
  --nsg-name <VM1_NSG_NAME> \\
  --name Allow-Prometheus-From-VM2 \\
  --priority 200 \\
  --source-address-prefixes "$VM2_PUBLIC_IP/32" \\
  --destination-port-ranges 9091 9095 9096 \\
  --protocol Tcp \\
  --access Allow \\
  --direction Inbound \\
  --description "Allow Prometheus scraping from VM2 (different account)"

${GREEN}# Verify the rule${NC}
az network nsg rule list \\
  --resource-group <VM1_RESOURCE_GROUP> \\
  --nsg-name <VM1_NSG_NAME> \\
  --query "[?name=='Allow-Prometheus-From-VM2']" \\
  --output table

${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}
${CYAN}  STEP 2: Configure NSG on VM2 (Allow Grafana Access)${NC}
${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}

${GREEN}# Login to Azure Account #2 (where VM2 is located)${NC}
az login
az account set --subscription "<VM2_SUBSCRIPTION_ID>"

${GREEN}# Allow Grafana access from Internet${NC}
az network nsg rule create \\
  --resource-group <VM2_RESOURCE_GROUP> \\
  --nsg-name <VM2_NSG_NAME> \\
  --name Allow-Grafana-Public \\
  --priority 300 \\
  --source-address-prefixes "Internet" \\
  --destination-port-ranges 3000 \\
  --protocol Tcp \\
  --access Allow \\
  --direction Inbound \\
  --description "Allow Grafana web access"

${GREEN}# (OPTIONAL) Restrict Prometheus to admin IP only${NC}
az network nsg rule create \\
  --resource-group <VM2_RESOURCE_GROUP> \\
  --nsg-name <VM2_NSG_NAME> \\
  --name Allow-Prometheus-Admin \\
  --priority 400 \\
  --source-address-prefixes "<YOUR_ADMIN_IP>/32" \\
  --destination-port-ranges 9090 \\
  --protocol Tcp \\
  --access Allow \\
  --direction Inbound \\
  --description "Allow Prometheus access for admin only"

${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}
${CYAN}  STEP 3: Verify Docker Services Listen on 0.0.0.0${NC}
${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}

${GREEN}# On VM1, check that services bind to 0.0.0.0 (not 127.0.0.1)${NC}
ssh azureuser@$VM1_PUBLIC_IP

# Inside VM1
docker exec fraud-data-service netstat -tlnp | grep 9091
docker exec fraud-drift-service netstat -tlnp | grep 9095
docker exec fraud-training-service netstat -tlnp | grep 9096

# Expected output: 0.0.0.0:9091, 0.0.0.0:9095, 0.0.0.0:9096

${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}
${CYAN}  STEP 4: Test Connectivity${NC}
${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}

${GREEN}# From VM2, test access to VM1 metrics${NC}
ssh azureuser@$VM2_PUBLIC_IP

# Inside VM2
curl -v http://$VM1_PUBLIC_IP:9091/metrics | head -20
curl -v http://$VM1_PUBLIC_IP:9095/metrics | head -20
curl -v http://$VM1_PUBLIC_IP:9096/metrics | head -20

# Expected: HTTP 200 OK with metrics

${GREEN}# Check Prometheus targets${NC}
curl http://localhost:9090/api/v1/targets | jq '.data.activeTargets[] | {job: .labels.job, health: .health}'

${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}
${CYAN}  Configuration Summary${NC}
${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}

VM1 Public IP:  $VM1_PUBLIC_IP
VM2 Public IP:  $VM2_PUBLIC_IP

NSG Rules Required:
  ✅ VM1: Allow $VM2_PUBLIC_IP → ports 9091, 9095, 9096
  ✅ VM2: Allow Internet → port 3000 (Grafana)
  ✅ (Optional) VM2: Allow <YOUR_IP> → port 9090 (Prometheus)

Prometheus Configuration:
  • Uses VM1 public IP: $VM1_PUBLIC_IP
  • File: monitoring/prometheus.vm2.yml

${YELLOW}⚠️  Security Recommendations:${NC}
  • Metrics ports are exposed publicly (restricted by NSG)
  • Consider adding Basic Auth for extra security
  • Monitor egress traffic costs (cross-account = charged)
  • Use Azure Monitor to track unauthorized access attempts

EOF

    log_info "For detailed security setup, see: Guide/CROSS_ACCOUNT_DEPLOYMENT.md"
}

# Main function
main() {
    show_banner
    check_root
    
    # Parse arguments
    local do_configure=0
    local do_vm1=0
    local do_vm2=0
    local do_validate=0
    local do_rollback=0
    local do_credentials=0
    local do_nsg_rules=0
    
    if [ $# -eq 0 ]; then
        show_usage
        exit 0
    fi
    
    while [ $# -gt 0 ]; do
        case "$1" in
            --configure)
                do_configure=1
                ;;
            --vm1)
                do_vm1=1
                ;;
            --vm2)
                do_vm2=1
                ;;
            --both)
                do_vm1=1
                do_vm2=1
                ;;
            --validate)
                do_validate=1
                ;;
            --rollback)
                do_rollback=1
                ;;
            --credentials)
                do_credentials=1
                ;;
            --nsg-rules)
                do_nsg_rules=1
                ;;
            --help)
                show_usage
                exit 0
                ;;
            *)
                log_error "Unknown option: $1"
                show_usage
                exit 1
                ;;
        esac
        shift
    done
    
    # Execute actions
    if [ $do_configure -eq 1 ]; then
        check_prerequisites
        configure_deployment
    fi
    
    if [ $do_rollback -eq 1 ]; then
        load_configuration
        rollback
        exit 0
    fi
    
    if [ $do_credentials -eq 1 ]; then
        load_configuration
        display_credentials
        exit 0
    fi
    
    if [ $do_nsg_rules -eq 1 ]; then
        show_nsg_rules
        exit 0
    fi
    
    if [ $do_vm1 -eq 1 ] || [ $do_vm2 -eq 1 ]; then
        check_prerequisites
        load_configuration
        
        # Show NSG warning for cross-account deployments
        if [ "$DEPLOYMENT_TYPE" = "cross-account" ]; then
            echo ""
            log_warning "═══════════════════════════════════════════════════════════════"
            log_warning "  CROSS-ACCOUNT DEPLOYMENT DETECTED"
            log_warning "═══════════════════════════════════════════════════════════════"
            log_warning "Before proceeding, ensure NSG rules are configured!"
            log_warning "Run: bash scripts/deploy-production.sh --nsg-rules"
            echo ""
            read -p "Have you configured the NSG rules? (yes/no): " nsg_confirmed
            if [ "$nsg_confirmed" != "yes" ]; then
                log_error "Please configure NSG rules first."
                log_info "Run: bash scripts/deploy-production.sh --nsg-rules"
                exit 1
            fi
        fi
        
        if [ $do_vm1 -eq 1 ]; then
            deploy_vm1
        fi
        
        if [ $do_vm2 -eq 1 ]; then
            deploy_vm2
        fi
        
        display_credentials
        
        # Show post-deployment NSG reminder for cross-account
        if [ "$DEPLOYMENT_TYPE" = "cross-account" ]; then
            echo ""
            log_warning "═══════════════════════════════════════════════════════════════"
            log_warning "  POST-DEPLOYMENT VERIFICATION"
            log_warning "═══════════════════════════════════════════════════════════════"
            log_info "Test connectivity from VM2 to VM1:"
            log_info "  ssh azureuser@$VM2_PUBLIC_IP"
            log_info "  curl http://$VM1_PUBLIC_IP:9091/metrics"
            log_info "  curl http://$VM1_PUBLIC_IP:9095/metrics"
            log_info "  curl http://$VM1_PUBLIC_IP:9096/metrics"
        fi
    fi
    
    if [ $do_validate -eq 1 ]; then
        load_configuration
        validate_deployment
    fi
    
    log_step "DEPLOYMENT COMPLETE"
    log_success "Fraud Detection ML is now running in production!"
    echo ""
    echo -e "${CYAN}Next steps:${NC}"
    echo "  1. Validate deployment: bash scripts/deploy-production.sh --validate"
    echo "  2. Access Grafana: http://$VM2_PUBLIC_IP:3000"
    echo "  3. Access Airflow: http://$VM1_PUBLIC_IP:8080"
    echo "  4. Check credentials: bash scripts/deploy-production.sh --credentials"
    
    if [ "$DEPLOYMENT_TYPE" = "cross-account" ]; then
        echo "  5. Verify NSG rules: bash scripts/deploy-production.sh --nsg-rules"
    fi
    echo ""
}

# Run main function
main "$@"
