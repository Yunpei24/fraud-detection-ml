# 🌐 Azure Cross-Subscription Deployment Guide

**Creation Date**: November 2, 2025  
**Version**: 1.0  
**Use Case**: VM1 and VM2 in different Azure accounts

---

## 📋 Table of Contents

1. [Overview](#overview)
2. [Cross-Subscription Architecture](#cross-subscription-architecture)
3. [Differences with Same-VNet Deployment](#differences-with-same-vnet-deployment)
4. [Mandatory NSG Configuration](#mandatory-nsg-configuration)
5. [Step-by-Step Deployment](#step-by-step-deployment)
6. [Advanced Security](#advanced-security)
7. [Troubleshooting](#troubleshooting)
8. [Costs and Optimization](#costs-and-optimization)

---

## 🎯 Overview

### **What is a Cross-Subscription Deployment?**

A cross-subscription deployment means that **VM1** (Application) and **VM2** (Monitoring) are hosted in **two different Azure accounts** (different subscriptions or different tenants).

**Consequences:**
- ❌ VMs **CANNOT** communicate via private IP (no shared VNet)
- ✅ VMs **MUST** communicate via **public IPs**
- ⚠️ Metrics ports **MUST be exposed** publicly (with NSG restrictions)
- 💰 Traffic between VMs is billed as **egress traffic** (Internet outbound)

---

## 🏗️ Cross-Subscription Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                    AZURE ACCOUNT #1 (Subscription A)                │
│                     Tenant: contoso.onmicrosoft.com                 │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  ┌────────────────────────────────────────────────────────────┐   │
│  │  VM1: Application Server                                   │   │
│  │  ══════════════════════════════════════════════════════════│   │
│  │  🌐 Public IP:  20.50.30.10                                │   │
│  │  🔒 Private IP: 10.0.1.4 (not accessible from VM2)         │   │
│  │                                                            │   │
│  │  Network Security Group (vm1-nsg):                        │   │
│  │  ┌──────────────────────────────────────────────────────┐ │   │
│  │  │ Rule: Allow-Prometheus-From-VM2                      │ │   │
│  │  │ Source: 52.170.10.20/32 (VM2 public IP)              │ │   │
│  │  │ Destination: Any                                     │ │   │
│  │  │ Ports: 9091, 9095, 9096                              │ │   │
│  │  │ Action: Allow                                        │ │   │
│  │  └──────────────────────────────────────────────────────┘ │   │
│  │                                                            │   │
│  │  Exposed Services:                                         │   │
│  │  • Data metrics:     0.0.0.0:9091  → Public              │   │
│  │  • Drift metrics:    0.0.0.0:9095  → Public              │   │
│  │  • Training metrics: 0.0.0.0:9096  → Public              │   │
│  └────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────┘
                                ↕
                      [PUBLIC INTERNET]
                   HTTP Requests via Public IP
              (Traffic billed as egress/ingress)
                                ↕
┌─────────────────────────────────────────────────────────────────────┐
│                    AZURE ACCOUNT #2 (Subscription B)                │
│                     Tenant: fabrikam.onmicrosoft.com                │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  ┌────────────────────────────────────────────────────────────┐   │
│  │  VM2: Monitoring Server                                    │   │
│  │  ══════════════════════════════════════════════════════════│   │
│  │  🌐 Public IP:  52.170.10.20                               │   │
│  │  🔒 Private IP: 10.1.2.5 (not accessible from VM1)         │   │
│  │                                                            │   │
│  │  Network Security Group (vm2-nsg):                        │   │
│  │  ┌──────────────────────────────────────────────────────┐ │   │
│  │  │ Rule: Allow-Grafana-Public                           │ │   │
│  │  │ Source: Internet                                     │ │   │
│  │  │ Destination: Any                                     │ │   │
│  │  │ Port: 3000                                           │ │   │
│  │  │ Action: Allow                                        │ │   │
│  │  └──────────────────────────────────────────────────────┘ │   │
│  │                                                            │   │
│  │  Prometheus Configuration:                                 │   │
│  │  • Scrape target: http://20.50.30.10:9091/metrics        │   │
│  │  • Scrape target: http://20.50.30.10:9095/metrics        │   │
│  │  • Scrape target: http://20.50.30.10:9096/metrics        │   │
│  └────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 🔄 Differences with Same-VNet Deployment

| Aspect | Same-VNet | Cross-Subscription |
|--------|-----------|-------------------|
| **Communication** | Private IP (10.x.x.x) | Public IP required |
| **Network Security** | VNet peering/integrated | Strict NSG required |
| **Public Exposure** | None (except web services) | Metrics ports public |
| **Prometheus Configuration** | `targets: ['10.0.1.4:9091']` | `targets: ['20.50.30.10:9091']` |
| **Network Costs** | Free (same region) | Billed ($0.05-0.10/GB) |
| **Latency** | <1ms | 10-50ms |
| **Setup Complexity** | Simple | Medium (NSG + validation) |
| **Security Risk** | Low | Medium (public exposure) |

---

## 🔐 Mandatory NSG Configuration

### **Prerequisites: Get Public IPs**

```bash
# On Azure Account #1 (VM1)
az login
az account set --subscription "<SUBSCRIPTION_A_ID>"

VM1_PUBLIC_IP=$(az vm show \
  --resource-group <VM1_RG> \
  --name vm1 \
  --show-details \
  --query publicIps \
  --output tsv)

echo "VM1 Public IP: $VM1_PUBLIC_IP"

# On Azure Account #2 (VM2)
az login
az account set --subscription "<SUBSCRIPTION_B_ID>"

VM2_PUBLIC_IP=$(az vm show \
  --resource-group <VM2_RG> \
  --name vm2 \
  --show-details \
  --query publicIps \
  --output tsv)

echo "VM2 Public IP: $VM2_PUBLIC_IP"
```

---

### **Step 1: Configure NSG on VM1**

**Objective:** Allow Prometheus (VM2) to scrape metrics from VM1.

```bash
# Connect to Azure account #1
az login
az account set --subscription "<SUBSCRIPTION_A_ID>"

# Create NSG rule
az network nsg rule create \
  --resource-group <VM1_RESOURCE_GROUP> \
  --nsg-name <VM1_NSG_NAME> \
  --name Allow-Prometheus-From-VM2 \
  --priority 200 \
  --source-address-prefixes "$VM2_PUBLIC_IP/32" \
  --destination-port-ranges 9091 9095 9096 \
  --protocol Tcp \
  --access Allow \
  --direction Inbound \
  --description "Allow Prometheus scraping from VM2 (cross-subscription)"

# Verify the rule
az network nsg rule show \
  --resource-group <VM1_RESOURCE_GROUP> \
  --nsg-name <VM1_NSG_NAME> \
  --name Allow-Prometheus-From-VM2 \
  --output table
```

**Expected result:**
```
Name                          Priority    SourceAddressPrefixes    DestinationPortRanges    Access
----------------------------  ----------  -----------------------  -----------------------  --------
Allow-Prometheus-From-VM2     200         52.170.10.20/32          9091,9095,9096          Allow
```

---

### **Step 2: Configure NSG on VM2**

**Objective:** Allow public access to Grafana.

```bash
# Connect to Azure account #2
az login
az account set --subscription "<SUBSCRIPTION_B_ID>"

# Rule 1: Grafana (public access)
az network nsg rule create \
  --resource-group <VM2_RESOURCE_GROUP> \
  --nsg-name <VM2_NSG_NAME> \
  --name Allow-Grafana-Public \
  --priority 300 \
  --source-address-prefixes "Internet" \
  --destination-port-ranges 3000 \
  --protocol Tcp \
  --access Allow \
  --direction Inbound \
  --description "Allow Grafana web interface"

# Rule 2 (OPTIONAL): Prometheus (restricted admin access)
YOUR_ADMIN_IP=$(curl -s ifconfig.me)

az network nsg rule create \
  --resource-group <VM2_RESOURCE_GROUP> \
  --nsg-name <VM2_NSG_NAME> \
  --name Allow-Prometheus-Admin \
  --priority 400 \
  --source-address-prefixes "$YOUR_ADMIN_IP/32" \
  --destination-port-ranges 9090 \
  --protocol Tcp \
  --access Allow \
  --direction Inbound \
  --description "Allow Prometheus access for admin only"
```

---

### **Step 3: Verify Docker Listens on 0.0.0.0**

**On VM1**, Docker services must listen on `0.0.0.0` (all interfaces) and not `127.0.0.1` (localhost).

```bash
# Connect to VM1
ssh azureuser@$VM1_PUBLIC_IP

# Check ports
sudo docker exec fraud-data-service netstat -tlnp | grep 9091
sudo docker exec fraud-drift-service netstat -tlnp | grep 9095
sudo docker exec fraud-training-service netstat -tlnp | grep 9096
```

**Expected result:**
```
tcp  0  0  0.0.0.0:9091  0.0.0.0:*  LISTEN  1/python
tcp  0  0  0.0.0.0:9095  0.0.0.0:*  LISTEN  1/python
tcp  0  0  0.0.0.0:9096  0.0.0.0:*  LISTEN  1/python
```

**If you see `127.0.0.1:9091`**, modify services to listen on `0.0.0.0`:

```python
# In data/metrics_server.py, drift/metrics_server.py, training/metrics_server.py
if __name__ == "__main__":
    start_http_server(9091, addr='0.0.0.0')  # ← Add addr='0.0.0.0'
```

---

## 🚀 Step-by-Step Deployment

### **Method 1: Automated Script (Recommended)**

```bash
# Clone the repo locally
git clone https://github.com/Yunpei24/fraud-detection-ml.git
cd fraud-detection-ml

# Launch configuration wizard
bash scripts/deploy-production.sh --configure

# Wizard responses:
# "Are VM1 and VM2 in the same Azure VNet? (y/n): n"  ← IMPORTANT
# "VM1 Public IP: 20.50.30.10"
# "VM2 Public IP: 52.170.10.20"
# ... (other configs)

# Display NSG commands to execute
bash scripts/deploy-production.sh --nsg-rules

# Copy-paste NSG commands into Azure CLI

# Deploy VM1 and VM2
bash scripts/deploy-production.sh --both

# Validate deployment
bash scripts/deploy-production.sh --validate
```

---

### **Method 2: Manual**

#### **On VM1 (Azure Account #1):**

```bash
# Connect to VM1
ssh azureuser@$VM1_PUBLIC_IP

# Clone the repo
git clone https://github.com/Yunpei24/fraud-detection-ml.git
cd fraud-detection-ml
git checkout develop

# Create .env file
cat > .env << 'EOF'
POSTGRES_PASSWORD=<SECURE_PASSWORD>
REDIS_PASSWORD=<SECURE_PASSWORD>
AIRFLOW_FERNET_KEY=<GENERATED_KEY>
AIRFLOW_SECRET_KEY=<GENERATED_KEY>
AIRFLOW_USERNAME=admin
AIRFLOW_PASSWORD=<SECURE_PASSWORD>
AIRFLOW_ADMIN_EMAIL=admin@example.com
ALERT_EMAIL=alerts@example.com
MLFLOW_TRACKING_URI=http://mlflow:5000
ENVIRONMENT=production
EOF

# Start services
docker-compose -f docker-compose.vm1.yml up -d

# Check services
docker-compose -f docker-compose.vm1.yml ps
```

#### **On VM2 (Azure Account #2):**

```bash
# Connect to VM2
ssh azureuser@$VM2_PUBLIC_IP

# Clone the repo
git clone https://github.com/Yunpei24/fraud-detection-ml.git
cd fraud-detection-ml
git checkout develop

# Create .env file
cat > .env << 'EOF'
GRAFANA_USER=admin
GRAFANA_PASSWORD=<SECURE_PASSWORD>
EOF

# Configure Prometheus with VM1 public IP
sed -i "s/<VM1_IP>/20.50.30.10/g" monitoring/prometheus.vm2.yml
sed -i "s/<AZURE_WEBAPP_URL>/your-app.azurewebsites.net/g" monitoring/prometheus.vm2.yml

# Verify configuration
grep -E "targets.*:909" monitoring/prometheus.vm2.yml
# Should display: targets: ['20.50.30.10:9091']

# Start monitoring
docker-compose -f docker-compose.vm2.yml up -d

# Check services
docker-compose -f docker-compose.vm2.yml ps
```

---

## 🛡️ Advanced Security

### **Option 1: Add Basic Auth with Nginx**

**On VM1**, install Nginx as a reverse proxy with authentication:

```bash
# Install Nginx
sudo apt update
sudo apt install nginx apache2-utils -y

# Create password
sudo htpasswd -c /etc/nginx/.htpasswd prometheus_user
# Enter a strong password: <PASSWORD>

# Configure Nginx
sudo nano /etc/nginx/sites-available/metrics-auth
```

**File content:**
```nginx
# ==============================================================================
# Nginx Reverse Proxy for Metrics with Basic Auth
# ==============================================================================
# This file protects metrics endpoints with HTTP Basic authentication
# Nginx listens on public ports (9091, 9095, 9097)
# and forwards to internal Docker ports

# Port 9091 - Data Service Metrics
server {
    listen 9091;
    server_name _;
    
    location /metrics {
        # Basic Auth protection
        auth_basic "Restricted - Prometheus Access Only";
        auth_basic_user_file /etc/nginx/.htpasswd;
        
        # Forward to data Docker container (internal port 9191)
        proxy_pass http://localhost:9191/metrics;
        
        # Required headers for proxying
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        
        # Timeouts
        proxy_connect_timeout 30s;
        proxy_send_timeout 30s;
        proxy_read_timeout 30s;
    }
    
    # Health check endpoint (without auth for monitoring)
    location /health {
        proxy_pass http://localhost:9191/health;
        proxy_set_header Host $host;
    }
}

# Port 9095 - Training Service Metrics
server {
    listen 9095;
    server_name _;
    
    location /metrics {
        auth_basic "Restricted - Prometheus Access Only";
        auth_basic_user_file /etc/nginx/.htpasswd;
        
        # Forward to training Docker container (internal port 9195)
        proxy_pass http://localhost:9195/metrics;
        
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        
        proxy_connect_timeout 30s;
        proxy_send_timeout 30s;
        proxy_read_timeout 30s;
    }
}

# Port 9097 - Drift Service Metrics
server {
    listen 9097;
    server_name _;
    
    location /metrics {
        auth_basic "Restricted - Prometheus Access Only";
        auth_basic_user_file /etc/nginx/.htpasswd;
        
        # Forward to drift Docker container (internal port 9197)
        proxy_pass http://localhost:9197/metrics;
        
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        
        proxy_connect_timeout 30s;
        proxy_send_timeout 30s;
        proxy_read_timeout 30s;
    }
}
```

**Enable and start:**
```bash
sudo ln -s /etc/nginx/sites-available/metrics-auth /etc/nginx/sites-enabled/
sudo nginx -t
sudo systemctl restart nginx
```

**Modify `docker-compose.vm1.yml` to use internal ports:**
```yaml
services:
  data:
    image: ${DOCKERHUB_USERNAME:-yoshua24}/data:latest
    container_name: fraud-data
    environment:
      - PROMETHEUS_PORT=9091  # Container port
    ports:
      # BEFORE (without Nginx): - "9091:9091"  # Publicly exposed ❌
      # AFTER (with Nginx):
      - "127.0.0.1:9191:9091"  # ✅ Accessible only from localhost
      #   ↑            ↑    ↑
      #   |            |    └─ Docker container port
      #   |            └────── VM port (localhost only)
      #   └─────────────────── Bind to localhost only
    networks:
      - fraud-network

  drift:
    image: ${DOCKERHUB_USERNAME:-yoshua24}/drift:latest
    container_name: fraud-drift
    environment:
      - PROMETHEUS_PORT=9097  # Container port
    ports:
      # BEFORE: - "9097:9097"  # Publicly exposed ❌
      # AFTER:
      - "127.0.0.1:9197:9097"  # ✅ Accessible only from localhost
    networks:
      - fraud-network

  training:
    image: ${DOCKERHUB_USERNAME:-yoshua24}/training:latest
    container_name: fraud-training
    environment:
      - PROMETHEUS_PORT=9095  # Container port
    ports:
      # BEFORE: - "9096:9095"  # Custom mapping publicly exposed ❌
      # AFTER:
      - "127.0.0.1:9195:9095"  # ✅ Accessible only from localhost
      #   ↑            ↑    ↑
      #   |            |    └─ Container port (9095)
      #   |            └────── VM port (9195, localhost only)
      #   └─────────────────── Bind to localhost only
    networks:
      - fraud-network
```

**Configure Prometheus on VM2 with Basic Auth:**
```yaml
# monitoring/prometheus.vm2.yml
scrape_configs:
  - job_name: 'fraud-data'
    static_configs:
      - targets: ['20.50.30.10:9091']
    basic_auth:
      username: 'prometheus_user'
      password: '<PASSWORD>'
```

---

### **Option 2: WireGuard VPN (Professional Solution)**

**Advantages:**
- ✅ Encrypted connection between VM1 and VM2
- ✅ No need to expose ports publicly
- ✅ Communication via "private IP tunnel"
- ✅ Free (open-source)

**Installation on VM1:**
```bash
sudo apt install wireguard -y
wg genkey | tee privatekey | wg pubkey > publickey

# Configure /etc/wireguard/wg0.conf
sudo nano /etc/wireguard/wg0.conf
```

**Content:**
```ini
[Interface]
PrivateKey = <VM1_PRIVATE_KEY>
Address = 10.99.0.1/24
ListenPort = 51820

[Peer]
PublicKey = <VM2_PUBLIC_KEY>
AllowedIPs = 10.99.0.2/32
```

**Installation on VM2:**
```bash
sudo apt install wireguard -y
wg genkey | tee privatekey | wg pubkey > publickey

sudo nano /etc/wireguard/wg0.conf
```

**Content:**
```ini
[Interface]
PrivateKey = <VM2_PRIVATE_KEY>
Address = 10.99.0.2/24

[Peer]
PublicKey = <VM1_PUBLIC_KEY>
Endpoint = 20.50.30.10:51820
AllowedIPs = 10.99.0.1/32
PersistentKeepalive = 25
```

**Start WireGuard:**
```bash
# On both VM1 and VM2
sudo wg-quick up wg0
sudo systemctl enable wg-quick@wg0
```

**Configure Prometheus to use tunnel IP:**
```yaml
# monitoring/prometheus.vm2.yml
scrape_configs:
  - job_name: 'fraud-data'
    static_configs:
      - targets: ['10.99.0.1:9091']  # ← Tunnel IP
```

**NSG: Open WireGuard port on VM1:**
```bash
az network nsg rule create \
  --resource-group <VM1_RG> \
  --nsg-name <VM1_NSG> \
  --name Allow-WireGuard \
  --priority 150 \
  --source-address-prefixes "$VM2_PUBLIC_IP/32" \
  --destination-port-ranges 51820 \
  --protocol Udp \
  --access Allow
```

---

## 🔍 Troubleshooting

### **Issue 1: Connection Timeout from VM2**

```bash
# On VM2
curl -v http://20.50.30.10:9091/metrics
# Error: curl: (7) Failed to connect to 20.50.30.10 port 9091: Connection timed out
```

**Possible causes:**
1. ❌ VM1 NSG doesn't contain rule allowing VM2
2. ❌ Docker service listens on `127.0.0.1` instead of `0.0.0.0`
3. ❌ VM2 public IP has changed (dynamic IP)
4. ❌ System firewall (ufw/iptables) blocks the port

**Solutions:**
```bash
# 1. Check NSG on VM1
az network nsg rule list \
  --resource-group <VM1_RG> \
  --nsg-name <VM1_NSG> \
  --query "[?name=='Allow-Prometheus-From-VM2']" \
  --output table

# 2. Check Docker binding on VM1
ssh azureuser@$VM1_PUBLIC_IP
sudo docker exec fraud-data-service netstat -tlnp | grep 9091
# Should display: 0.0.0.0:9091 (not 127.0.0.1:9091)

# 3. Check VM2 public IP
curl ifconfig.me
# Compare with IP in NSG rule

# 4. Check firewall on VM1
sudo ufw status
sudo iptables -L INPUT -n | grep 9091
```

---

### **Issue 2: Prometheus Targets "DOWN"**

```bash
# On VM2
curl http://localhost:9090/api/v1/targets | jq '.data.activeTargets[] | select(.health!="up")'
```

**Result:**
```json
{
  "job": "fraud-data",
  "health": "down",
  "lastError": "Get \"http://20.50.30.10:9091/metrics\": context deadline exceeded"
}
```

**Causes:**
1. ❌ IP in `prometheus.vm2.yml` is incorrect
2. ❌ NSG blocks the connection
3. ❌ Docker service is not started on VM1

**Solutions:**
```bash
# 1. Check Prometheus config
ssh azureuser@$VM2_PUBLIC_IP
grep -A5 "job_name.*fraud-data" ~/fraud-detection-ml/monitoring/prometheus.vm2.yml

# 2. Test manually from VM2
curl -v http://20.50.30.10:9091/metrics

# 3. Check services on VM1
ssh azureuser@$VM1_PUBLIC_IP
docker ps | grep data
docker logs fraud-data-service
```

---

### **Issue 3: High Traffic Costs**

**Symptom:** High Azure bill with "Data Transfer Out".

**Cause:** Prometheus scraping every 15s = ~5.5GB/month per service.

**Solutions:**

1. **Increase scraping interval:**
```yaml
# monitoring/prometheus.vm2.yml
scrape_configs:
  - job_name: 'fraud-data'
    scrape_interval: 60s  # Instead of 15s
```

2. **Move VM2 to same Azure account:**
```bash
# Create VM2 in Subscription A (same account as VM1)
# Use same VNet → Free communication
```

3. **Use Azure Monitor (PaaS):**
```bash
# Alternative: Send metrics to Azure Monitor
# Cost: $0.50/million metrics ingested
```

---

## 💰 Costs and Optimization

### **Traffic Cost Estimation**

| Component | Monthly Traffic | Cost/GB | Total/Month |
|-----------|----------------|---------|------------|
| Data metrics (15s) | 5.5 GB | $0.087 | $0.48 |
| Drift metrics (60s) | 1.4 GB | $0.087 | $0.12 |
| Training metrics (30s) | 2.8 GB | $0.087 | $0.24 |
| API metrics (10s) | Included (Azure Web App) | $0 | $0 |
| **TOTAL** | **9.7 GB** | - | **$0.84/month** |

**Note:** Azure Data Transfer Out rate (Europe West region, November 2025).

---

### **Optimization Strategies**

#### **1. Gzip Compression**

Enable compression in Prometheus:

```yaml
# monitoring/prometheus.vm2.yml
global:
  scrape_interval: 15s
  
scrape_configs:
  - job_name: 'fraud-data'
    static_configs:
      - targets: ['20.50.30.10:9091']
    metric_relabel_configs:
      - source_labels: [__name__]
        regex: '.*'
        action: keep
    # ~70% reduction with compression
```

**Savings:** ~$0.60/month

---

#### **2. Metrics Filtering**

Scrape only important metrics:

```yaml
# monitoring/prometheus.vm2.yml
scrape_configs:
  - job_name: 'fraud-data'
    static_configs:
      - targets: ['20.50.30.10:9091']
    metric_relabel_configs:
      - source_labels: [__name__]
        regex: 'fraud_(data|drift|training)_.*'  # Only fraud_* metrics
        action: keep
```

**Savings:** ~$0.30/month

---

#### **3. Local Aggregation**

Use Prometheus Pushgateway on VM1 for local aggregation:

```yaml
# VM1: Aggregate metrics locally every 5 minutes
# VM2: Scrape Pushgateway every 5 minutes (instead of 15s)
```

**Savings:** ~$0.70/month

---

## ✅ Deployment Checklist

### **Before Deployment**

- [ ] Get VM1 and VM2 public IPs
- [ ] Verify access to both Azure accounts
- [ ] Prepare credentials (passwords, keys)
- [ ] Read this entire guide

### **NSG Configuration**

- [ ] Create NSG rule on VM1: Allow VM2 → ports 9091, 9095, 9096
- [ ] Create NSG rule on VM2: Allow Internet → port 3000 (Grafana)
- [ ] (Optional) Create NSG rule on VM2: Allow Admin IP → port 9090
- [ ] Verify rules with `az network nsg rule list`

### **VM1 Deployment**

- [ ] SSH to VM1 functional
- [ ] Repo cloned on `develop` branch
- [ ] `.env` file created with credentials
- [ ] `docker-compose.vm1.yml` started
- [ ] All services `healthy`: `docker-compose ps`
- [ ] Ports listening on `0.0.0.0`: `netstat -tlnp`

### **VM2 Deployment**

- [ ] SSH to VM2 functional
- [ ] Repo cloned on `develop` branch
- [ ] `.env` file created for Grafana
- [ ] `prometheus.vm2.yml` configured with VM1 public IP
- [ ] `docker-compose.vm2.yml` started
- [ ] All services `healthy`: `docker-compose ps`

### **Validation**

- [ ] From VM2: `curl http://<VM1_IP>:9091/metrics` returns HTTP 200
- [ ] From VM2: `curl http://<VM1_IP>:9095/metrics` returns HTTP 200
- [ ] From VM2: `curl http://<VM1_IP>:9096/metrics` returns HTTP 200
- [ ] Prometheus targets: `curl localhost:9090/api/v1/targets` (all "up")
- [ ] Grafana accessible: `http://<VM2_IP>:3000`
- [ ] Dashboards display data

### **Security (Optional but Recommended)**

- [ ] Basic Auth configured with Nginx
- [ ] WireGuard VPN configured
- [ ] Alerts configured in AlertManager
- [ ] Logs monitored with Azure Monitor

---

## 📚 Additional Resources

- [Main Guide (Same-VNet)](./COMM_VM1_VM2.md)
- [Monitoring Corrections Report](../MONITORING_CORRECTIONS_REPORT.md)
- [Azure NSG Documentation](https://learn.microsoft.com/en-us/azure/virtual-network/network-security-groups-overview)
- [Prometheus Security Best Practices](https://prometheus.io/docs/operating/security/)
- [WireGuard Documentation](https://www.wireguard.com/quickstart/)

---

## 📝 Quick Reference Commands

### **Get VM Public IPs**
```bash
# VM1
az vm show --resource-group <RG> --name vm1 --show-details --query publicIps -o tsv

# VM2
az vm show --resource-group <RG> --name vm2 --show-details --query publicIps -o tsv
```

### **Test Connectivity from VM2 to VM1**
```bash
# Test metrics endpoints
curl -v http://<VM1_IP>:9091/metrics
curl -v http://<VM1_IP>:9095/metrics
curl -v http://<VM1_IP>:9096/metrics

# Test with basic auth (if configured)
curl -u prometheus_user:<PASSWORD> http://<VM1_IP>:9091/metrics
```

### **Check Prometheus Targets Status**
```bash
# On VM2
curl http://localhost:9090/api/v1/targets | jq '.data.activeTargets[] | {job, health, lastError}'
```

### **Monitor Docker Services**
```bash
# On VM1
docker-compose -f docker-compose.vm1.yml ps
docker-compose -f docker-compose.vm1.yml logs -f data drift training

# On VM2
docker-compose -f docker-compose.vm2.yml ps
docker-compose -f docker-compose.vm2.yml logs -f prometheus grafana
```

### **Check Network Listening Ports**
```bash
# On VM1 - verify services listen on 0.0.0.0
sudo netstat -tlnp | grep -E ':(9091|9095|9096)'

# Expected output:
# tcp 0 0 0.0.0.0:9091 0.0.0.0:* LISTEN <pid>/docker-proxy
# tcp 0 0 0.0.0.0:9095 0.0.0.0:* LISTEN <pid>/docker-proxy
# tcp 0 0 0.0.0.0:9096 0.0.0.0:* LISTEN <pid>/docker-proxy
```

### **View NSG Rules**
```bash
# List all rules
az network nsg rule list \
  --resource-group <RG> \
  --nsg-name <NSG_NAME> \
  --output table

# Show specific rule
az network nsg rule show \
  --resource-group <RG> \
  --nsg-name <NSG_NAME> \
  --name Allow-Prometheus-From-VM2 \
  --output json
```

### **Restart Services**
```bash
# On VM1
docker-compose -f docker-compose.vm1.yml restart data drift training

# On VM2
docker-compose -f docker-compose.vm2.yml restart prometheus grafana
```

---

## 🚨 Common Error Messages and Solutions

### **Error: "Connection refused"**
```
curl: (7) Failed to connect to 20.50.30.10 port 9091: Connection refused
```
**Solution:** Docker service is not running or crashed
```bash
ssh azureuser@$VM1_PUBLIC_IP
docker ps | grep data
docker logs fraud-data-service
docker-compose -f docker-compose.vm1.yml restart data
```

---

### **Error: "Connection timed out"**
```
curl: (7) Failed to connect to 20.50.30.10 port 9091: Connection timed out
```
**Solution:** NSG rule is missing or firewall blocks the connection
```bash
# Check NSG rules
az network nsg rule list --resource-group <RG> --nsg-name <NSG> --output table

# Check system firewall
ssh azureuser@$VM1_PUBLIC_IP
sudo ufw status
sudo iptables -L INPUT -n
```

---

### **Error: "context deadline exceeded"**
```
Get "http://20.50.30.10:9091/metrics": context deadline exceeded
```
**Solution:** Service is too slow or unresponsive
```bash
# Check service health
ssh azureuser@$VM1_PUBLIC_IP
docker stats fraud-data-service
docker logs fraud-data-service --tail 50

# Check resource usage
top
df -h
```

---

### **Error: "401 Unauthorized"** (with Basic Auth)
```
HTTP/1.1 401 Unauthorized
WWW-Authenticate: Basic realm="Restricted - Prometheus Access Only"
```
**Solution:** Wrong credentials in Prometheus config
```yaml
# Verify credentials in monitoring/prometheus.vm2.yml
scrape_configs:
  - job_name: 'fraud-data'
    basic_auth:
      username: 'prometheus_user'
      password: '<CORRECT_PASSWORD>'  # Check this
```

---

## 🎯 Architecture Decision Guide

### **When to Use Cross-Subscription Deployment**

✅ **Use cross-subscription when:**
- Different teams manage VM1 and VM2
- Compliance requires separate Azure accounts
- Budget separation is required
- Testing different Azure regions for disaster recovery
- You have existing infrastructure in different subscriptions

❌ **Avoid cross-subscription when:**
- You control both subscriptions
- Cost optimization is critical
- Low latency is essential
- Security concerns about public exposure
- You can use VNet peering

### **Alternative Architectures**

#### **Option 1: Same Subscription, Different VNets with Peering**
```
VM1 (VNet A) ←→ VNet Peering ←→ VM2 (VNet B)
Cost: Free (same region)
Latency: <1ms
Security: Private IPs only
```

#### **Option 2: Azure Private Link**
```
VM1 ←→ Private Endpoint ←→ Private Link Service ←→ VM2
Cost: $0.01/hour + $0.01/GB
Security: No public exposure
Complexity: High
```

#### **Option 3: Azure VPN Gateway**
```
VM1 (Subscription A) ←→ VPN Gateway ←→ VM2 (Subscription B)
Cost: $0.05/hour + $0.05/GB
Latency: 10-20ms
Security: Encrypted tunnel
```

---

## 🔒 Security Best Practices Summary

### **Network Layer**
- ✅ Use NSG rules with specific source IP (not 0.0.0.0/0)
- ✅ Regularly review and audit NSG rules
- ✅ Use Azure Network Watcher for traffic analysis
- ✅ Enable NSG flow logs for compliance

### **Application Layer**
- ✅ Implement Basic Auth or mutual TLS
- ✅ Use strong, unique passwords (minimum 16 characters)
- ✅ Rotate credentials every 90 days
- ✅ Store credentials in Azure Key Vault

### **Monitoring Layer**
- ✅ Enable Azure Monitor for VMs
- ✅ Configure alerts for unauthorized access attempts
- ✅ Monitor data transfer volumes for anomalies
- ✅ Set up Azure Security Center recommendations

### **Infrastructure Layer**
- ✅ Use static public IPs to avoid NSG rule updates
- ✅ Implement Azure Bastion for SSH access (no public SSH)
- ✅ Enable just-in-time (JIT) VM access
- ✅ Use Azure DDoS Protection Standard

---

## 📊 Monitoring Dashboard Examples

### **Grafana Dashboard for Cross-Subscription Metrics**

Create a dashboard that shows:

1. **Network Health Panel**
   - Prometheus scrape duration
   - Failed scrape attempts
   - Network latency between VMs

2. **Cost Tracking Panel**
   - Estimated data transfer costs
   - GB transferred per day
   - Monthly projection

3. **Security Panel**
   - Failed authentication attempts
   - Unusual traffic patterns
   - NSG rule changes

**Example Prometheus queries:**

```promql
# Scrape duration (should be <5s for cross-subscription)
prometheus_target_interval_length_seconds{job="fraud-data"}

# Failed scrapes
up{job=~"fraud.*"} == 0

# Data transfer estimate (bytes)
sum(increase(prometheus_target_scrapes_total[1d])) * avg(scrape_series_added)
```

---

## 🔄 Migration Path: Cross-Subscription to Same-VNet

If you later want to migrate to a same-VNet deployment:

### **Step 1: Create VM2 in Same Subscription**
```bash
# Create new VM2 in Subscription A
az vm create \
  --resource-group <VM1_RG> \
  --name vm2-new \
  --vnet-name <VM1_VNET> \
  --subnet monitoring-subnet \
  --image UbuntuLTS \
  --size Standard_B2s
```

### **Step 2: Update Prometheus Configuration**
```yaml
# Change from public IP to private IP
scrape_configs:
  - job_name: 'fraud-data'
    static_configs:
      - targets: ['10.0.1.4:9091']  # Private IP
    # Remove basic_auth (optional in private network)
```

### **Step 3: Remove NSG Public Rules**
```bash
# Delete public access rules
az network nsg rule delete \
  --resource-group <VM1_RG> \
  --nsg-name <VM1_NSG> \
  --name Allow-Prometheus-From-VM2
```

### **Step 4: Test and Decommission Old VM2**
```bash
# Test connectivity
curl http://10.0.1.4:9091/metrics

# Verify Prometheus targets
curl http://localhost:9090/api/v1/targets

# Delete old VM2 in Subscription B
az vm delete --resource-group <VM2_RG> --name vm2 --yes
```

---

## 💡 Tips and Tricks

### **Tip 1: Use Azure CLI Aliases**
```bash
# Add to ~/.bashrc
alias az-vm1='az account set --subscription "<SUB_A_ID>"'
alias az-vm2='az account set --subscription "<SUB_B_ID>"'

# Usage
az-vm1 && az vm list -o table
az-vm2 && az vm list -o table
```

### **Tip 2: Monitor NSG Changes with Azure Activity Log**
```bash
az monitor activity-log list \
  --resource-group <RG> \
  --namespace Microsoft.Network \
  --query "[?contains(operationName.value, 'securityRules')]" \
  --output table
```

### **Tip 3: Automate Public IP Checks**
```bash
# Create a script: check-ips.sh
#!/bin/bash
VM1_IP=$(az vm show -g <RG1> -n vm1 --show-details --query publicIps -o tsv)
VM2_IP=$(az vm show -g <RG2> -n vm2 --show-details --query publicIps -o tsv)

echo "VM1: $VM1_IP"
echo "VM2: $VM2_IP"

# Check if NSG rule matches current VM2 IP
RULE_IP=$(az network nsg rule show -g <RG1> --nsg-name <NSG> \
  -n Allow-Prometheus-From-VM2 --query sourceAddressPrefixes[0] -o tsv)

if [[ "$RULE_IP" != "$VM2_IP/32" ]]; then
  echo "⚠️  WARNING: NSG rule IP ($RULE_IP) does not match VM2 IP ($VM2_IP/32)"
  echo "Update with: az network nsg rule update -g <RG1> --nsg-name <NSG> -n Allow-Prometheus-From-VM2 --source-address-prefixes $VM2_IP/32"
fi
```

### **Tip 4: Setup Alert for High Data Transfer**
```bash
# Create Azure Monitor alert for >10GB/day
az monitor metrics alert create \
  --name high-egress-alert \
  --resource-group <RG> \
  --scopes /subscriptions/<SUB_ID>/resourceGroups/<RG>/providers/Microsoft.Network/networkInterfaces/<NIC> \
  --condition "total BytesSentRate > 10737418240" \
  --description "Alert when daily egress exceeds 10GB"
```

---

**Last Updated:** November 2, 2025  
**Author:** Fraud Detection Team  
**Status:** Production Ready ✅

---

## 📄 Document Version History

| Version | Date | Changes | Author |
|---------|------|---------|--------|
| 1.0 | Nov 2, 2025 | Initial release | MLOps Team |

---

## 🆘 Support and Contact

For issues or questions:
- **GitHub Issues:** [fraud-detection-ml/issues](https://github.com/Yunpei24/fraud-detection-ml)
- **Documentation:** [Project Wiki](https://github.com/Yunpei24/fraud-detection-ml/wiki)
- **Azure Support:** [Azure Portal Support](https://portal.azure.com/#blade/Microsoft_Azure_Support/HelpAndSupportBlade)