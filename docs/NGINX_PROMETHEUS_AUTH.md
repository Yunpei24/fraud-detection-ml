# 🔐 Nginx Reverse Proxy with Basic Authentication for Prometheus Metrics

**Date Created**: November 6, 2025  
**Version**: 1.0  
**Status**: Production Ready

---

## 📋 Table of Contents

1. [Overview](#overview)
2. [Architecture](#architecture)
3. [Security Benefits](#security-benefits)
4. [Prerequisites](#prerequisites)
5. [Configuration Analysis](#configuration-analysis)
6. [Step-by-Step Installation](#step-by-step-installation)
7. [Docker Compose Modifications](#docker-compose-modifications)
8. [Prometheus Configuration](#prometheus-configuration)
9. [Testing & Validation](#testing--validation)
10. [Security Best Practices](#security-best-practices)
11. [Troubleshooting](#troubleshooting)
12. [Rollback Procedure](#rollback-procedure)

---

## 🎯 Overview

This guide explains how to secure Prometheus metrics endpoints on VM1 using **Nginx reverse proxy with HTTP Basic Authentication**. Instead of exposing metrics ports directly to the internet, Nginx acts as a secure gateway that requires authentication before forwarding requests to Docker containers.

### **Problem Without Nginx** ❌

```
Internet → VM1:9091 → Docker Container (fraud-data)
         ↑
    No authentication!
    Anyone can scrape metrics
```

### **Solution With Nginx** ✅

```
Internet → VM1:9091 → Nginx (Basic Auth) → localhost:9191 → Docker Container
                      ↑
                Requires username/password
                Only localhost can access container ports
```

---

## 🏗️ Architecture

### **Before: Direct Exposure (Vulnerable)**

```
┌─────────────────────────────────────────────────────────────┐
│                        VM1 (Public)                         │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Internet → 0.0.0.0:9091 → Docker Container (fraud-data)   │
│             ↑                                               │
│             └─ No authentication barrier                    │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### **After: Nginx Proxy (Secured)**

```
┌───────────────────────────────────────────────────────────────────┐
│                        VM1 (Public)                               │
├───────────────────────────────────────────────────────────────────┤
│                                                                   │
│  Internet → 0.0.0.0:9091 → Nginx (Basic Auth)                    │
│                             ↓                                     │
│                       Request credentials                         │
│                             ↓                                     │
│                  ✅ If valid credentials                          │
│                             ↓                                     │
│              → 127.0.0.1:9191 → Docker Container (fraud-data)     │
│                 ↑                                                 │
│                 └─ Only accessible from localhost                 │
│                    (no direct internet access)                    │
│                                                                   │
└───────────────────────────────────────────────────────────────────┘
```

---

## 🛡️ Security Benefits

| Feature | Without Nginx | With Nginx |
|---------|---------------|------------|
| **Authentication** | ❌ None | ✅ HTTP Basic Auth |
| **Public Exposure** | ❌ Direct to containers | ✅ Only Nginx exposed |
| **Access Control** | ❌ Anyone | ✅ Authenticated users only |
| **Audit Logs** | ❌ None | ✅ Nginx access logs |
| **IP Filtering** | ⚠️ Requires NSG only | ✅ NSG + Nginx |
| **TLS/SSL** | ❌ Not possible | ✅ Can add SSL easily |

---

## ✅ Prerequisites

### **1. Services Running on VM1**

Ensure these services are deployed and running:
- **fraud-data** (metrics on port 9091)
- **fraud-training** (metrics on port 9095)
- **fraud-drift** (metrics on port 9097)

### **2. Software Requirements**

```bash
# Check if Docker is installed
docker --version

# Check if Docker Compose is installed
docker compose version
```

### **3. Network Access**

- SSH access to VM1
- Sudo/root privileges on VM1
- Prometheus on VM2 must be able to reach VM1's public IP

---

## 📊 Configuration Analysis

### **Current Docker Container Ports**

Based on analysis of `drift/Dockerfile`, `training/Dockerfile`, and `docker-compose.vm1.yml`:

| Service | Container Port | Current Host Mapping | Purpose |
|---------|---------------|---------------------|---------|
| **data** | 9091 | `9091:9091` | Data pipeline metrics |
| **training** | 9095 | `9096:9095` | Training metrics (custom mapping) |
| **drift** | 9097 | `9097:9097` | Drift detection metrics |

#### **Analysis Notes:**

1. **data**: Container exposes 9091 → Currently mapped to host 9091
2. **training**: Container exposes 9095 → Currently mapped to host 9096 (custom mapping)
3. **drift**: Container exposes 9097 → Currently mapped to host 9097

### **New Port Mappings (With Nginx)**

We'll bind container ports to **localhost only** with different host ports:

| Service | Container Port | New Host Mapping | Nginx Listens | Public Access |
|---------|---------------|------------------|---------------|---------------|
| **data** | 9091 | `127.0.0.1:9191:9091` | 0.0.0.0:9091 | ✅ Via Nginx |
| **training** | 9095 | `127.0.0.1:9195:9095` | 0.0.0.0:9095 | ✅ Via Nginx |
| **drift** | 9097 | `127.0.0.1:9197:9097` | 0.0.0.0:9097 | ✅ Via Nginx |

**Key Changes:**
- Container ports stay the same (no code changes needed)
- Host ports are bound to `127.0.0.1` (localhost only)
- Nginx listens on public interface and forwards to localhost ports
- Authentication required at Nginx layer

---

## 🔧 Step-by-Step Installation

### **Step 1: Connect to VM1**

```bash
ssh azureuser@<VM1_PUBLIC_IP>
```

---

### **Step 2: Install Required Software**

```bash
# Update package lists
sudo apt update

# Install Nginx and Apache utilities (for htpasswd)
sudo apt install -y nginx apache2-utils

# Verify installation
nginx -v
htpasswd -h
```

---

### **Step 3: Create Authentication File**

```bash
# Create password file for user "prometheus"
sudo htpasswd -c /etc/nginx/.htpasswd prometheus

# You'll be prompted to enter a password
# Example: prometheus_secure_2025
# Enter password: ************
# Re-type password: ************

# Verify the file was created
sudo cat /etc/nginx/.htpasswd
# Output: prometheus:$apr1$xyz123$...
```

**Security Note:** Use a strong password with at least 16 characters, including uppercase, lowercase, numbers, and symbols.

---

### **Step 4: Create Nginx Configuration File**

```bash
# Create the configuration file
sudo nano /etc/nginx/sites-available/metrics-auth
```

**Paste the following configuration:**

```nginx
# ==============================================================================
# Nginx Reverse Proxy for Prometheus Metrics with Basic Authentication
# ==============================================================================
# This configuration protects metrics endpoints with HTTP Basic Auth
# Nginx listens on public ports (9091, 9095, 9097) and forwards to internal Docker ports

# Port 9091 - Data Service Metrics
server {
    listen 9091;
    server_name _;
    
    # Metrics endpoint with authentication
    location /metrics {
        # Enable HTTP Basic Authentication
        auth_basic "Restricted - Prometheus Access Only";
        auth_basic_user_file /etc/nginx/.htpasswd;
        
        # Forward to Docker container (internal port 9191)
        proxy_pass http://localhost:9191/metrics;
        
        # Proxy headers
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        
        # Timeouts
        proxy_connect_timeout 30s;
        proxy_send_timeout 30s;
        proxy_read_timeout 30s;
    }
    
    # Health check endpoint (no authentication for monitoring)
    location /health {
        proxy_pass http://localhost:9191/health;
        proxy_set_header Host $host;
    }
    
    # Access logs
    access_log /var/log/nginx/metrics_data_access.log;
    error_log /var/log/nginx/metrics_data_error.log;
}

# Port 9095 - Training Service Metrics
server {
    listen 9095;
    server_name _;
    
    location /metrics {
        auth_basic "Restricted - Prometheus Access Only";
        auth_basic_user_file /etc/nginx/.htpasswd;
        
        # Forward to Docker container training (internal port 9195)
        # Note: Container listens on 9095, but we map it to 9195 on host
        proxy_pass http://localhost:9195/metrics;
        
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        
        proxy_connect_timeout 30s;
        proxy_send_timeout 30s;
        proxy_read_timeout 30s;
    }
    
    location /health {
        proxy_pass http://localhost:9195/health;
        proxy_set_header Host $host;
    }
    
    access_log /var/log/nginx/metrics_training_access.log;
    error_log /var/log/nginx/metrics_training_error.log;
}

# Port 9097 - Drift Service Metrics
server {
    listen 9097;
    server_name _;
    
    location /metrics {
        auth_basic "Restricted - Prometheus Access Only";
        auth_basic_user_file /etc/nginx/.htpasswd;
        
        # Forward to Docker container drift (internal port 9197)
        proxy_pass http://localhost:9197/metrics;
        
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        
        proxy_connect_timeout 30s;
        proxy_send_timeout 30s;
        proxy_read_timeout 30s;
    }
    
    location /health {
        proxy_pass http://localhost:9197/health;
        proxy_set_header Host $host;
    }
    
    access_log /var/log/nginx/metrics_drift_access.log;
    error_log /var/log/nginx/metrics_drift_error.log;
}
```

**Save and exit:** Press `Ctrl+X`, then `Y`, then `Enter`

---

### **Step 5: Enable Nginx Configuration**

```bash
# Create symbolic link to enable the site
sudo ln -s /etc/nginx/sites-available/metrics-auth /etc/nginx/sites-enabled/

# Test Nginx configuration (must show "syntax is ok")
sudo nginx -t

# Expected output:
# nginx: the configuration file /etc/nginx/nginx.conf syntax is ok
# nginx: configuration file /etc/nginx/nginx.conf test is successful
```

**If you see errors:**
- Check for typos in the configuration file
- Ensure `.htpasswd` file exists: `sudo ls -l /etc/nginx/.htpasswd`
- Check file permissions: `sudo chmod 644 /etc/nginx/.htpasswd`

---

### **Step 6: Restart Nginx**

```bash
# Restart Nginx to apply changes
sudo systemctl restart nginx

# Check Nginx status
sudo systemctl status nginx
# Should show: Active: active (running)

# Verify Nginx is listening on the correct ports
sudo netstat -tlnp | grep nginx

# Expected output:
# tcp  0.0.0.0:9091  0.0.0.0:*  LISTEN  1234/nginx
# tcp  0.0.0.0:9095  0.0.0.0:*  LISTEN  1234/nginx
# tcp  0.0.0.0:9097  0.0.0.0:*  LISTEN  1234/nginx
```

---

## 🐳 Docker Compose Modifications

### **Step 7: Backup Current Configuration**

```bash
cd ~/fraud-detection-ml

# Create a backup of the original file
cp docker-compose.vm1.yml docker-compose.vm1.yml.backup

# Verify backup
ls -l docker-compose.vm1.yml*
```

---

### **Step 8: Modify `docker-compose.vm1.yml`**

```bash
nano docker-compose.vm1.yml
```

**Find and modify the `ports` section for each service:**

#### **Before (Direct Exposure):**

```yaml
services:
  data:
    ports:
      - "9091:9091"  # ❌ Exposed to public internet

  drift:
    ports:
      - "9097:9097"  # ❌ Exposed to public internet

  training:
    ports:
      - "9096:9095"  # ❌ Exposed to public internet
```

#### **After (Localhost Only):**

```yaml
services:
  data:
    image: ${DOCKERHUB_USERNAME:-yoshua24}/data:latest
    container_name: fraud-data
    environment:
      - PROMETHEUS_PORT=9091  # Container port
    ports:
      - "127.0.0.1:9191:9091"  # ✅ Bind to localhost:9191 → container:9091
      #   ↑            ↑    ↑
      #   |            |    └─ Container port (unchanged)
      #   |            └────── Host port (internal, localhost only)
      #   └─────────────────── Bind address (localhost only)
    networks:
      - fraud-network
    # ... rest of configuration unchanged

  drift:
    image: ${DOCKERHUB_USERNAME:-yoshua24}/drift:latest
    container_name: fraud-drift
    environment:
      - PROMETHEUS_PORT=9097  # Container port
    ports:
      - "127.0.0.1:9197:9097"  # ✅ Bind to localhost:9197 → container:9097
    networks:
      - fraud-network
    # ... rest of configuration unchanged

  training:
    image: ${DOCKERHUB_USERNAME:-yoshua24}/training:latest
    container_name: fraud-training
    environment:
      - PROMETHEUS_PORT=9095  # Container port
    ports:
      - "127.0.0.1:9195:9095"  # ✅ Bind to localhost:9195 → container:9095
    networks:
      - fraud-network
    # ... rest of configuration unchanged
```

**Key Points:**
- `127.0.0.1:9191:9091` means:
  - Bind to `127.0.0.1` (localhost only, not accessible from internet)
  - Use host port `9191` (internal, Nginx will connect here)
  - Forward to container port `9091` (unchanged)
- No code changes needed in containers
- Only Docker port mappings are modified

**Save and exit:** `Ctrl+X`, `Y`, `Enter`

---

### **Step 9: Restart Docker Containers**

```bash
cd ~/fraud-detection-ml

# Stop all containers
docker-compose -f docker-compose.vm1.yml down

# Wait a few seconds
sleep 5

# Start containers with new configuration
docker-compose -f docker-compose.vm1.yml up -d

# Verify containers are running
docker ps

# Check that containers are listening on localhost
sudo netstat -tlnp | grep "9191\|9195\|9197"

# Expected output:
# tcp  127.0.0.1:9191  0.0.0.0:*  LISTEN  5678/docker-proxy
# tcp  127.0.0.1:9195  0.0.0.0:*  LISTEN  5679/docker-proxy
# tcp  127.0.0.1:9197  0.0.0.0:*  LISTEN  5680/docker-proxy
```

**Troubleshooting:**
- If containers fail to start, check logs: `docker-compose -f docker-compose.vm1.yml logs`
- If port conflicts, ensure no other services are using ports 9191, 9195, 9197
- Restore backup if needed: `mv docker-compose.vm1.yml.backup docker-compose.vm1.yml`

---

## 📊 Prometheus Configuration

### **Step 10: Configure Prometheus on VM2**

```bash
# Connect to VM2
ssh azureuser@<VM2_PUBLIC_IP>

cd ~/fraud-detection-ml/monitoring

# Backup original configuration
cp prometheus.vm2.yml prometheus.vm2.yml.backup

# Edit Prometheus configuration
nano prometheus.vm2.yml
```

**Add Basic Authentication to scrape configs:**

```yaml
# Global configuration
global:
  scrape_interval: 30s
  evaluation_interval: 30s

# Scrape configurations
scrape_configs:
  # Data Service Metrics
  - job_name: 'fraud-data'
    scrape_interval: 30s
    static_configs:
      - targets: ['<VM1_PUBLIC_IP>:9091']
    # Add Basic Authentication
    basic_auth:
      username: 'prometheus'
      password: 'prometheus_secure_2025'  # Use the password you created
    metrics_path: '/metrics'
    scheme: 'http'

  # Training Service Metrics
  - job_name: 'fraud-training'
    scrape_interval: 30s
    static_configs:
      - targets: ['<VM1_PUBLIC_IP>:9095']
    # Add Basic Authentication
    basic_auth:
      username: 'prometheus'
      password: 'prometheus_secure_2025'
    metrics_path: '/metrics'
    scheme: 'http'

  # Drift Service Metrics
  - job_name: 'fraud-drift'
    scrape_interval: 60s
    static_configs:
      - targets: ['<VM1_PUBLIC_IP>:9097']
    # Add Basic Authentication
    basic_auth:
      username: 'prometheus'
      password: 'prometheus_secure_2025'
    metrics_path: '/metrics'
    scheme: 'http'

  # API Service Metrics (if applicable)
  - job_name: 'fraud-api'
    scrape_interval: 30s
    static_configs:
      - targets: ['<AZURE_WEBAPP_URL>']
    metrics_path: '/metrics'
    scheme: 'https'
```

**Replace placeholders:**
- `<VM1_PUBLIC_IP>`: Your VM1's public IP address
- `<AZURE_WEBAPP_URL>`: Your Azure Web App URL (if deployed)
- `prometheus_secure_2025`: The actual password you created

**Save and exit:** `Ctrl+X`, `Y`, `Enter`

---

### **Step 11: Restart Prometheus**

```bash
cd ~/fraud-detection-ml

# Restart Prometheus container
docker-compose -f docker-compose.vm2.yml restart prometheus

# Check logs for any authentication errors
docker logs fraud-prometheus --tail 100

# Look for:
# - "Server is ready to receive web requests" ✅
# - No authentication errors ✅
```

---

## ✅ Testing & Validation

### **Test 1: Access WITHOUT Authentication (Should Fail)**

```bash
# From VM1 or your local machine
curl http://<VM1_PUBLIC_IP>:9091/metrics

# Expected output:
# <html>
# <head><title>401 Authorization Required</title></head>
# <body>
# <center><h1>401 Authorization Required</h1></center>
# <hr><center>nginx</center>
# </body>
# </html>
```

**✅ This is correct!** Nginx is blocking unauthenticated requests.

---

### **Test 2: Access WITH Authentication (Should Succeed)**

```bash
# From VM1
curl -u prometheus:prometheus_secure_2025 http://localhost:9091/metrics | head -20

# Expected output:
# # HELP data_pipeline_runs_total Total number of pipeline runs
# # TYPE data_pipeline_runs_total counter
# data_pipeline_runs_total 42.0
# ...

# From your local machine (or VM2)
curl -u prometheus:prometheus_secure_2025 http://<VM1_PUBLIC_IP>:9091/metrics | head -20

# Should also return metrics successfully
```

---

### **Test 3: Verify All Services**

```bash
# Test Data Service (port 9091)
curl -u prometheus:prometheus_secure_2025 http://<VM1_PUBLIC_IP>:9091/metrics | grep "data_pipeline"

# Test Training Service (port 9095)
curl -u prometheus:prometheus_secure_2025 http://<VM1_PUBLIC_IP>:9095/metrics | grep "fraud_training"

# Test Drift Service (port 9097)
curl -u prometheus:prometheus_secure_2025 http://<VM1_PUBLIC_IP>:9097/metrics | grep "drift_detection"
```

**All commands should return metrics lines with the specified prefixes.**

---

### **Test 4: Check Prometheus Targets**

```bash
# From your browser, access Prometheus UI on VM2
http://<VM2_PUBLIC_IP>:9090/targets

# Or via curl from VM2
curl -s http://localhost:9090/api/v1/targets | jq '.data.activeTargets[] | {job: .labels.job, health: .health, lastError: .lastError}'

# Expected output for each target:
# {
#   "job": "fraud-data",
#   "health": "up",
#   "lastError": ""
# }
```

**All targets should show `"health": "up"` with no errors.**

---

### **Test 5: Verify Nginx Access Logs**

```bash
# On VM1, check Nginx access logs
sudo tail -f /var/log/nginx/metrics_data_access.log

# Trigger a request
curl -u prometheus:prometheus_secure_2025 http://localhost:9091/metrics > /dev/null

# You should see a log entry like:
# 127.0.0.1 - prometheus [06/Nov/2025:10:30:00 +0000] "GET /metrics HTTP/1.1" 200 ...
```

---

## 🔒 Security Best Practices

### **1. Use Strong Passwords**

```bash
# Generate a strong random password
openssl rand -base64 32

# Example output: xK9mP2nQ4rT6vY8zA1bC3dE5fG7hJ9kL0mN==

# Update the password
sudo htpasswd /etc/nginx/.htpasswd prometheus
# Enter the new strong password
```

---

### **2. Store Credentials Securely (Prometheus)**

#### ⚠️ Important: Prometheus Limitation

**Prometheus does NOT support environment variable substitution in `prometheus.yml`**. Despite what you might expect, using `${PROMETHEUS_USER}` will NOT work - Prometheus will treat it as a literal string.

#### **Solution: Use Hardcoded Credentials in Protected File**

Since this is for an academic project, hardcoding credentials in a **protected file** is acceptable:

```bash
# On VM2, create prometheus configuration
cd ~/fraud-detection-ml/monitoring

# IMPORTANT: Replace <VM1_PUBLIC_IP> with actual IP
VM1_IP="20.123.45.100"  # Get with: ssh VM1 "curl ifconfig.me"

cat > prometheus.vm2.yml << EOF
global:
  scrape_interval: 15s
  evaluation_interval: 15s
  external_labels:
    cluster: 'fraud-detection'
    environment: 'production'

scrape_configs:
  # Data service metrics
  - job_name: 'fraud-data'
    static_configs:
      - targets: ['${VM1_IP}:9091']
    basic_auth:
      username: 'prometheus'
      password: 'xK9mP2nQ4rT6vY8zA1bC3dE5fG7hJ9kL0mN=='  # Use same password from Nginx htpasswd

  # Training service metrics
  - job_name: 'fraud-training'
    static_configs:
      - targets: ['${VM1_IP}:9095']
    basic_auth:
      username: 'prometheus'
      password: 'xK9mP2nQ4rT6vY8zA1bC3dE5fG7hJ9kL0mN=='

  # Drift detection metrics
  - job_name: 'fraud-drift'
    static_configs:
      - targets: ['${VM1_IP}:9097']
    basic_auth:
      username: 'prometheus'
      password: 'xK9mP2nQ4rT6vY8zA1bC3dE5fG7hJ9kL0mN=='
EOF

# Protect the file (only owner can read/write)
chmod 600 prometheus.vm2.yml

# Verify it's not tracked by git
echo "monitoring/prometheus.vm2.yml" >> ../.gitignore
```

#### **Why This Approach Works**

✅ **Security:**
- File is `chmod 600` (only owner can access)
- File is in `.gitignore` (not committed to repo)
- Credentials only exist on VM2 (monitoring server)
- Same password as Nginx (consistent authentication)

✅ **Simplicity:**
- No complex templating needed
- Easy to debug (exactly what Prometheus sees)
- Works immediately without additional setup

#### **Alternative: Dynamic Config Generation (Advanced)**

If you want to avoid hardcoding, use a script to generate the config:

```bash
# Create template
cat > prometheus.vm2.yml.template << 'TEMPLATE'
global:
  scrape_interval: 15s

scrape_configs:
  - job_name: 'fraud-data'
    static_configs:
      - targets: ['{{VM1_IP}}:9091']
    basic_auth:
      username: '{{PROM_USER}}'
      password: '{{PROM_PASS}}'
TEMPLATE

# Create credentials file
cat > .prometheus-secrets << 'EOF'
VM1_IP=20.123.45.100
PROM_USER=prometheus
PROM_PASS=xK9mP2nQ4rT6vY8zA1bC3dE5fG7hJ9kL0mN==
EOF

chmod 600 .prometheus-secrets

# Create generation script
cat > generate-config.sh << 'EOF'
#!/bin/bash
source .prometheus-secrets
sed -e "s|{{VM1_IP}}|$VM1_IP|g" \
    -e "s|{{PROM_USER}}|$PROM_USER|g" \
    -e "s|{{PROM_PASS}}|$PROM_PASS|g" \
    prometheus.vm2.yml.template > prometheus.vm2.yml
chmod 600 prometheus.vm2.yml
EOF

chmod +x generate-config.sh
./generate-config.sh
```

**For this project, use the first approach (hardcoded in protected file).** It's simpler and equally secure.

---

### **3. Restrict Access by IP (Optional)**

Add IP whitelisting in Nginx configuration:

```nginx
server {
    listen 9091;
    server_name _;
    
    # Allow only specific IPs
    allow <VM2_PUBLIC_IP>;  # Prometheus VM
    allow <YOUR_ADMIN_IP>;  # Your admin IP
    deny all;  # Deny everyone else
    
    location /metrics {
        auth_basic "Restricted - Prometheus Access Only";
        auth_basic_user_file /etc/nginx/.htpasswd;
        proxy_pass http://localhost:9191/metrics;
        # ...
    }
}
```

---

### **4. Enable HTTPS/TLS (Recommended for Production)**

```bash
# Install Certbot for Let's Encrypt SSL
sudo apt install certbot

# Generate certificate (requires a domain name)
sudo certbot certonly --standalone -d metrics.yourcompany.com
```

**Update Nginx configuration:**

```nginx
server {
    listen 9091 ssl http2;
    server_name metrics.yourcompany.com;
    
    ssl_certificate /etc/letsencrypt/live/metrics.yourcompany.com/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/metrics.yourcompany.com/privkey.pem;
    
    # Strong SSL configuration
    ssl_protocols TLSv1.2 TLSv1.3;
    ssl_ciphers HIGH:!aNULL:!MD5;
    
    location /metrics {
        auth_basic "Restricted - Prometheus Access Only";
        auth_basic_user_file /etc/nginx/.htpasswd;
        proxy_pass http://localhost:9191/metrics;
        # ...
    }
}
```

---

### **5. Monitor Nginx Logs**

```bash
# Set up log rotation
sudo nano /etc/logrotate.d/nginx-metrics

# Add:
/var/log/nginx/metrics_*_access.log {
    daily
    rotate 30
    compress
    delaycompress
    notifempty
    create 0640 www-data adm
    sharedscripts
    postrotate
        [ -f /var/run/nginx.pid ] && kill -USR1 `cat /var/run/nginx.pid`
    endscript
}
```

---

## 🚨 Troubleshooting

### **Problem 1: 401 Unauthorized Even With Correct Credentials**

**Symptoms:**
```bash
curl -u prometheus:password http://VM1:9091/metrics
# Returns: 401 Authorization Required
```

**Solutions:**

```bash
# 1. Verify password file exists and has correct permissions
sudo ls -l /etc/nginx/.htpasswd
# Should show: -rw-r--r-- 1 root root ... /etc/nginx/.htpasswd

sudo cat /etc/nginx/.htpasswd
# Should show: prometheus:$apr1$...

# 2. Test password manually
echo -n "password" | openssl passwd -apr1 -stdin

# 3. Recreate the password file
sudo rm /etc/nginx/.htpasswd
sudo htpasswd -c /etc/nginx/.htpasswd prometheus
# Enter password

# 4. Restart Nginx
sudo systemctl restart nginx
```

---

### **Problem 2: Prometheus Targets Show as "DOWN"**

**Symptoms:**
- Prometheus UI shows targets with red "DOWN" status
- Error: "context deadline exceeded" or "connection refused"

**Solutions:**

```bash
# 1. Test from VM2 manually
ssh azureuser@<VM2_PUBLIC_IP>
curl -v -u prometheus:password http://<VM1_PUBLIC_IP>:9091/metrics

# 2. Check Prometheus logs
docker logs fraud-prometheus --tail 100 | grep -i error

# 3. Verify VM2 can reach VM1 ports
telnet <VM1_PUBLIC_IP> 9091

# 4. Check Azure NSG rules
az network nsg rule list \
  --resource-group <RESOURCE_GROUP> \
  --nsg-name <VM1_NSG_NAME> \
  --query "[?destinationPortRange=='9091']" \
  --output table

# 5. Verify Nginx is listening
ssh azureuser@<VM1_PUBLIC_IP>
sudo netstat -tlnp | grep nginx
```

---

### **Problem 3: Docker Containers Can't Start**

**Symptoms:**
```
Error: bind: address already in use
```

**Solutions:**

```bash
# 1. Check what's using the port
sudo netstat -tlnp | grep 9191
sudo lsof -i :9191

# 2. Kill the process if necessary
sudo kill -9 <PID>

# 3. Or change the host port in docker-compose.vm1.yml
# From:
ports:
  - "127.0.0.1:9191:9091"
# To:
ports:
  - "127.0.0.1:9291:9091"

# Then update Nginx config to proxy_pass to 9291 instead
```

---

### **Problem 4: Metrics Endpoint Returns 502 Bad Gateway**

**Symptoms:**
```bash
curl -u prometheus:password http://VM1:9091/metrics
# Returns: 502 Bad Gateway
```

**Solutions:**

```bash
# 1. Check if Docker container is running
docker ps | grep fraud-data

# 2. Check if container is listening on the correct port
docker exec fraud-data netstat -tlnp | grep 9091

# 3. Test direct access to container (from VM1)
curl http://localhost:9191/metrics

# 4. Check Nginx error logs
sudo tail -f /var/log/nginx/metrics_data_error.log

# 5. Restart Docker containers
docker-compose -f docker-compose.vm1.yml restart data
```

---

### **Problem 5: Nginx Won't Start**

**Symptoms:**
```bash
sudo systemctl status nginx
# Shows: failed
```

**Solutions:**

```bash
# 1. Check detailed error logs
sudo journalctl -u nginx -n 50 --no-pager

# 2. Test configuration
sudo nginx -t
# Fix any syntax errors shown

# 3. Check if ports are already in use
sudo netstat -tlnp | grep ':9091\|:9095\|:9097'

# 4. Disable conflicting configurations
sudo rm /etc/nginx/sites-enabled/default

# 5. Restart Nginx
sudo systemctl restart nginx
```

---

## 🔄 Rollback Procedure

If you need to revert to the original configuration:

### **Step 1: Restore Docker Compose**

```bash
cd ~/fraud-detection-ml

# Stop containers
docker-compose -f docker-compose.vm1.yml down

# Restore backup
mv docker-compose.vm1.yml.backup docker-compose.vm1.yml

# Restart containers with original config
docker-compose -f docker-compose.vm1.yml up -d
```

---

### **Step 2: Disable Nginx Proxy**

```bash
# Remove symbolic link
sudo rm /etc/nginx/sites-enabled/metrics-auth

# Test configuration
sudo nginx -t

# Restart Nginx
sudo systemctl restart nginx
```

---

### **Step 3: Restore Prometheus Configuration**

```bash
# On VM2
cd ~/fraud-detection-ml/monitoring

# Restore backup
mv prometheus.vm2.yml.backup prometheus.vm2.yml

# Restart Prometheus
docker-compose -f docker-compose.vm2.yml restart prometheus
```

---

## 📊 Port Reference Table

| Service | Container Port | Container Listens | Host Port (Internal) | Host Bind | Nginx Listens (Public) | Auth Required |
|---------|---------------|-------------------|---------------------|-----------|----------------------|---------------|
| **data** | 9091 | 0.0.0.0:9091 | 9191 | 127.0.0.1 | 0.0.0.0:9091 | ✅ Yes |
| **training** | 9095 | 0.0.0.0:9095 | 9195 | 127.0.0.1 | 0.0.0.0:9095 | ✅ Yes |
| **drift** | 9097 | 0.0.0.0:9097 | 9197 | 127.0.0.1 | 0.0.0.0:9097 | ✅ Yes |

**Connection Flow:**
```
Prometheus (VM2) 
  → Internet 
  → VM1:9091 (Nginx)
  → Basic Auth Check
  → localhost:9191 (Docker Proxy)
  → Container:9091 (fraud-data)
```

---

## ✅ Validation Checklist

Before considering the setup complete, verify:

- [ ] **Nginx installed** and running: `sudo systemctl status nginx`
- [ ] **Password file created**: `sudo cat /etc/nginx/.htpasswd`
- [ ] **Nginx config valid**: `sudo nginx -t` shows "ok"
- [ ] **Nginx listening on ports**: `sudo netstat -tlnp | grep nginx` shows 9091, 9095, 9097
- [ ] **Docker containers restarted** with new port mappings
- [ ] **Containers listening on localhost**: `sudo netstat -tlnp | grep "9191\|9195\|9197"`
- [ ] **Auth test fails without credentials**: `curl http://VM1:9091/metrics` returns 401
- [ ] **Auth test succeeds with credentials**: `curl -u prometheus:pass http://VM1:9091/metrics` returns metrics
- [ ] **All 3 services tested** (data, training, drift)
- [ ] **Prometheus configured** with `basic_auth` in `prometheus.vm2.yml`
- [ ] **Prometheus restarted** on VM2
- [ ] **Prometheus targets UP**: Check `http://VM2:9090/targets` shows all green
- [ ] **Metrics visible in Prometheus**: Query `up{job="fraud-data"}` returns 1

---

## 📚 Additional Resources

### **Official Documentation**

- [Nginx HTTP Basic Authentication](http://nginx.org/en/docs/http/ngx_http_auth_basic_module.html)
- [Nginx Reverse Proxy](http://nginx.org/en/docs/http/ngx_http_proxy_module.html)
- [Prometheus Configuration](https://prometheus.io/docs/prometheus/latest/configuration/configuration/)
- [Docker Port Binding](https://docs.docker.com/config/containers/container-networking/)

### **Related Guides**

- `COMM_VM1_VM2.md` - Cross-VM Communication Setup
- `CROSS_SUBSCRIPTION_DEPLOYMENT.md` - Cross-Subscription Architecture
- `PRODUCTION_DEPLOYMENT_GUIDE.md` - Full Production Deployment

---

## 📝 Changelog

| Date | Version | Changes |
|------|---------|---------|
| 2025-11-06 | 1.0 | Initial documentation - Nginx Basic Auth setup |

---

**Author**: MLOps Team  
**Last Updated**: November 6, 2025  
**Status**: ✅ Production Ready
