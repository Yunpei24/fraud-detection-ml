# Nginx MLflow Tracking Server Proxy Guide

## 📋 Table of Contents

1. [Overview](#overview)
2. [Architecture](#architecture)
3. [Security Benefits](#security-benefits)
4. [Prerequisites](#prerequisites)
5. [Step-by-Step Configuration](#step-by-step-configuration)
6. [Testing & Validation](#testing--validation)
7. [Monitoring](#monitoring)
8. [Troubleshooting](#troubleshooting)
9. [Alternative Solutions](#alternative-solutions)

---

## 🎯 Overview

This guide explains how to securely expose your MLflow Tracking Server (running on VM1) to your Azure Web App API using **Nginx reverse proxy with IP whitelisting**.

### **The Problem**

Your FastAPI application on Azure Web App needs to access MLflow Tracking Server on VM1 to:
- Load ML models for predictions
- Log model metrics and parameters
- Track experiments
- Query model versions

However, MLflow is currently configured with `0.0.0.0:5001:5000`, which means it's **publicly accessible** to anyone on the internet without authentication.

### **The Solution**

Use **Nginx as a reverse proxy** to:
- ✅ Restrict access to Azure Web App IPs only (IP whitelisting)
- ✅ Add rate limiting to prevent abuse
- ✅ Provide centralized logging for all MLflow access
- ✅ Enable easy addition of SSL/TLS in the future
- ✅ **No code changes required in your API**

---

## 🏗️ Architecture

### **Before: Direct Exposure (Vulnerable)**

```
┌─────────────────────────────────────────────────────────┐
│                     Internet                            │
│                                                         │
│  ❌ Anyone can access MLflow                            │
│     http://<VM1_IP>:5001/api/2.0/mlflow/experiments    │
└───────────────────────┬─────────────────────────────────┘
                        │
                        │ No restrictions
                        │
┌───────────────────────┼─────────────────────────────────┐
│ VM1                   │                                 │
│                       ▼                                 │
│  ┌──────────────────────────────────────┐              │
│  │  MLflow Container                    │              │
│  │  0.0.0.0:5001:5000                   │              │
│  │  ❌ Publicly accessible               │              │
│  └──────────────────────────────────────┘              │
└─────────────────────────────────────────────────────────┘
```

### **After: Nginx Proxy (Secured)**

```
┌──────────────────────────────────────────────────────────┐
│              Azure Web App (FastAPI)                     │
│   fraud-detection-api-ammi-2025.azurewebsites.net        │
│                                                          │
│   MLFLOW_TRACKING_URI=http://<VM1_IP>:5000              │
└────────────────────┬─────────────────────────────────────┘
                     │
                     │ HTTP Request
                     │ Source IP: 20.19.1.44, etc.
                     │
                     ▼
┌──────────────────────────────────────────────────────────┐
│                    VM1 - Nginx Layer                     │
│                                                          │
│  ┌────────────────────────────────────────────────────┐ │
│  │  Nginx Reverse Proxy (Port 5000)                   │ │
│  │                                                    │ │
│  │  1. Check Source IP                               │ │
│  │     ├─ In whitelist? → Continue                   │ │
│  │     └─ Not in whitelist? → 403 Forbidden          │ │
│  │                                                    │ │
│  │  2. Rate Limiting (10 req/s per IP)               │ │
│  │                                                    │ │
│  │  3. Log Access                                    │ │
│  │                                                    │ │
│  │  4. Proxy to localhost:5001                       │ │
│  └────────────────────────────────────────────────────┘ │
│                           │                              │
│                           ▼                              │
│  ┌────────────────────────────────────────────────────┐ │
│  │  MLflow Container                                  │ │
│  │  127.0.0.1:5001:5000 (localhost only)             │ │
│  │  ✅ Not directly accessible from internet          │ │
│  └────────────────────────────────────────────────────┘ │
└──────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────┐
│              Internet (Other Users)                      │
│                                                          │
│  ❌ 403 Forbidden (IP not in whitelist)                  │
└──────────────────────────────────────────────────────────┘
```

### **Request Flow**

1. **Azure Web App** sends request to `http://<VM1_IP>:5000/api/2.0/mlflow/experiments/list`
2. **Nginx** receives request, checks source IP against whitelist
3. **If IP authorized**: Nginx forwards request to `localhost:5001` (MLflow container)
4. **MLflow** processes request, returns response
5. **Nginx** forwards response back to Azure Web App
6. **Nginx logs** record the entire transaction

---

## 🛡️ Security Benefits

| Feature | Without Nginx | With Nginx |
|---------|---------------|------------|
| **Public Access** | ❌ Anyone can access | ✅ Only Azure Web App |
| **IP Whitelisting** | ❌ Not possible | ✅ 13 Azure IPs only |
| **Rate Limiting** | ❌ No protection | ✅ 10 requests/second |
| **Access Logs** | ⚠️ Only MLflow logs | ✅ Nginx + MLflow logs |
| **DDoS Protection** | ❌ Vulnerable | ✅ Rate limiting + IP filter |
| **Future SSL/TLS** | ❌ Complex to add | ✅ Easy to enable |
| **API Code Changes** | - | ✅ **Zero changes needed** |

---

## 📋 Prerequisites

### 1. Azure Web App Outbound IPs

You need to know your Azure Web App's outbound IP addresses:

```bash
# Get Azure Web App outbound IPs
az webapp show \
  --name fraud-detection-api-ammi-2025 \
  --resource-group fraud-detection-rg \
  --query outboundIpAddresses \
  --output tsv

```

**Save these IPs** - you'll need them for configuration.

### 2. VM1 Access

Ensure you have:
- SSH access to VM1
- Sudo privileges
- Nginx installed (or ability to install it)

```bash
# SSH into VM1
ssh -i Fraud.VM1_key.pem frauduser@<VM1_IP>

# Check if Nginx is installed
nginx -v

# If not installed:
sudo apt-get update
sudo apt-get install -y nginx
```

### 3. Current MLflow Configuration

Check your current `docker-compose.vm1.yml`:

```yaml
services:
  mlflow:
    # ...
    ports:
      - "0.0.0.0:5001:5000"  # Currently exposed publicly
```

---

## 🔧 Step-by-Step Configuration

### **Step 1: Modify MLflow Docker Configuration**

**Edit `docker-compose.vm1.yml` on VM1:**

```bash
cd ~/fraud-detection-ml
nano docker-compose.vm1.yml
```

**Change MLflow port binding from `0.0.0.0` to `127.0.0.1`:**

```yaml
services:
  mlflow:
    image: ${DOCKERHUB_USERNAME:-yoshua24}/mlflow:latest
    container_name: fraud-mlflow
    command: >
      mlflow server
      --host 0.0.0.0
      --port 5000
      --backend-store-uri postgresql://mlflow_user:${MLFLOW_DB_PASSWORD}@postgres:5432/mlflow_db
      --default-artifact-root ${MLFLOW_ARTIFACT_ROOT:-/mlflow/artifacts}
      --serve-artifacts
    environment:
      - MLFLOW_TRACKING_URI=http://0.0.0.0:5000
    volumes:
      - mlflow_data:/mlflow/artifacts
    ports:
      - "127.0.0.1:5001:5000"  # ← CHANGE: Bind to localhost only
    networks:
      - fraud-network
    restart: unless-stopped
```

**Apply changes:**

```bash
docker-compose -f docker-compose.vm1.yml down
docker-compose -f docker-compose.vm1.yml up -d

# Verify MLflow is now on localhost only
docker ps | grep mlflow
# Should show: 127.0.0.1:5001->5000/tcp (not 0.0.0.0)
```

---

### **Step 2: Create Nginx Configuration**

**Create necessary directories and Nginx site configuration:**

```bash
# Create sites-available and sites-enabled directories if they don't exist
sudo mkdir -p /etc/nginx/sites-available
sudo mkdir -p /etc/nginx/sites-enabled

# Verify Nginx main config includes sites-enabled
# Check if this line exists in /etc/nginx/nginx.conf
sudo grep -q "include /etc/nginx/sites-enabled/\*;" /etc/nginx/nginx.conf || \
    echo "⚠️  Warning: You may need to add 'include /etc/nginx/sites-enabled/*;' to the http block in /etc/nginx/nginx.conf"

# Create Nginx configuration file for MLflow proxy
sudo nano /etc/nginx/sites-available/mlflow-proxy.conf
```

**Add this configuration (replace with your actual Azure IPs):**

```nginx
# ============================================================================
# MLflow Tracking Server Reverse Proxy
# ============================================================================
# File: /etc/nginx/sites-available/mlflow-proxy.conf
# Purpose: Secure MLflow access with IP whitelisting for Azure Web App
# ============================================================================

# Define allowed IPs (Azure Web App outbound IPs)
geo $allowed_mlflow_ip {
    default 0;  # Block by default
    
    # Azure Web App outbound IPs
    # Retrieved with: az webapp show --name api_name_deploy_azure_web --query "outboundIpAddresses"
    20.19.1.44 1;
    20.19.1.119 1;
    
    
    # Add more IPs if Azure Web App IPs change
}

# Rate limiting zone (10 requests per second per IP)
limit_req_zone $binary_remote_addr zone=mlflow_limit:10m rate=10r/s;

# MLflow upstream (localhost only)
upstream mlflow_backend {
    server 127.0.0.1:5001;
    
    # Connection pooling
    keepalive 32;
    keepalive_requests 100;
    keepalive_timeout 60s;
}

# Main MLflow proxy server
server {
    listen 5000;  # Public port
    server_name _;
    
    # Logging
    access_log /var/log/nginx/mlflow-access.log combined;
    error_log /var/log/nginx/mlflow-error.log warn;
    
    # Client settings
    client_max_body_size 100M;  # Allow large model uploads
    client_body_timeout 300s;
    
    # IP whitelisting check
    if ($allowed_mlflow_ip = 0) {
        return 403 "Access denied. IP not whitelisted.";
    }
    
    # Main location - proxy all requests to MLflow
    location / {
        # Rate limiting (10 req/s, burst up to 20)
        limit_req zone=mlflow_limit burst=20 nodelay;
        
        # Proxy to MLflow backend
        proxy_pass http://mlflow_backend;
        
        # Headers
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        
        # Timeouts (MLflow can be slow for large models)
        proxy_connect_timeout 60s;
        proxy_send_timeout 300s;
        proxy_read_timeout 300s;
        
        # Buffering (disable for streaming responses)
        proxy_buffering off;
        proxy_request_buffering off;
        
        # WebSocket support (if needed for future features)
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
    }
    
    # Health check endpoint (no rate limiting for monitoring)
    location /health {
        access_log off;
        limit_req off;
        proxy_pass http://mlflow_backend/health;
    }
    
    # Metrics endpoint (for Prometheus scraping)
    location /metrics {
        access_log off;
        limit_req off;
        proxy_pass http://mlflow_backend/metrics;
    }
}
```
**Or**
```nginx
# ==============================================================================
# NGINX MLFLOW TRACKING SERVER PROXY CONFIGURATION
# ==============================================================================
# File: /etc/nginx/sites-available/mlflow-proxy.conf
#
# Purpose: Secure MLflow Tracking Server access from Azure Web App
# - IP whitelisting (Azure Web App only)
# - Rate limiting (10 requests/second)
# - Health check endpoint (for monitoring)
# - Reverse proxy to localhost MLflow container
#
# Author: MLOps Team
# Last Updated: 2025-11-06
# ==============================================================================

# ==============================================================================
# IP WHITELIST - Azure Web App Outbound IPs
# ==============================================================================
# Define allowed IPs using geo module
# Only these IPs can access MLflow endpoints
geo $allowed_mlflow_ip {
    default 0;  # Block all by default
    
    # Azure Web App outbound IPs
    # Retrieved with: az webapp show --name fraud-detection-api-ammi-2025 \
    #                 --resource-group fraud-detection-rg \
    #                 --query outboundIpAddresses -o tsv
    20.19.1.44 1;
    20.19.1.119 1;
    20.19.1.218 1;
    20.19.2.151 1;
    20.19.2.185 1;
    20.19.2.239 1;
    51.138.218.150 1;
    51.138.223.105 1;
    20.74.97.20 1;
    51.138.216.249 1;
    20.74.98.69 1;
    20.74.99.131 1;
    20.111.1.5 1;
    
    # Note: To add more IPs in the future:
    # 1. Get new IPs: az webapp show --query outboundIpAddresses
    # 2. Add them here with value 1
    # 3. Reload Nginx: sudo systemctl reload nginx
}

# ==============================================================================
# RATE LIMITING ZONES
# ==============================================================================
# Limit to 10 requests per second per IP address
# Burst allows up to 20 requests temporarily
limit_req_zone $binary_remote_addr zone=mlflow_limit:10m rate=10r/s;

# Connection limit: max 20 concurrent connections per IP
limit_conn_zone $binary_remote_addr zone=mlflow_conn_limit:10m;

# ==============================================================================
# UPSTREAM DEFINITION - MLflow Backend
# ==============================================================================
# Define the backend MLflow server running in Docker
upstream mlflow_backend {
    # MLflow container on localhost (not exposed publicly)
    server 127.0.0.1:5001;
    
    # Keep 32 persistent connections for better performance
    keepalive 32;
    keepalive_timeout 60s;
    keepalive_requests 100;
}

# ==============================================================================
# MAIN SERVER BLOCK - MLflow Proxy
# ==============================================================================
server {
    # Listen on port 5000 (public port for MLflow)
    listen 5000;
    server_name _;
    
    # Logging
    access_log /var/log/nginx/mlflow-access.log;
    error_log /var/log/nginx/mlflow-error.log warn;
    
    # Security headers
    add_header X-Content-Type-Options "nosniff" always;
    add_header X-Frame-Options "DENY" always;
    add_header X-XSS-Protection "1; mode=block" always;
    
    # ==============================================================================
    # HEALTH CHECK ENDPOINT (No IP restriction for monitoring)
    # ==============================================================================
    # This endpoint can be used by monitoring tools without authentication
    # Uses MLflow's experiments/search API as health check (guaranteed to exist)
    location = /health {
        access_log off;  # Don't log health checks
        
        # Forward to MLflow experiments search (lightweight endpoint)
        proxy_pass http://mlflow_backend/api/2.0/mlflow/experiments/search;
        
        # Override method to POST (required by experiments/search)
        proxy_method POST;
        
        # Set required headers
        proxy_set_header Content-Type "application/json";
        proxy_set_header Host $host;
        
        # Send minimal request body (just get 1 experiment)
        proxy_set_body '{"max_results": 1}';
        
        # Timeouts
        proxy_connect_timeout 5s;
        proxy_send_timeout 5s;
        proxy_read_timeout 5s;
        
        # Return 200 if MLflow responds, 502 if down
    }
    
    # ==============================================================================
    # ALTERNATIVE HEALTH CHECK (If MLflow has native /health endpoint)
    # ==============================================================================
    # Uncomment if your MLflow version supports /api/2.0/mlflow/health
    # location /api/2.0/mlflow/health {
    #     access_log off;
    #     proxy_pass http://mlflow_backend/api/2.0/mlflow/health;
    #     proxy_set_header Host $host;
    # }
    
    # ==============================================================================
    # MAIN MLFLOW ENDPOINTS (With IP restriction and rate limiting)
    # ==============================================================================
    location / {
        # ----------------------------------------------------------------------
        # IP WHITELISTING CHECK
        # ----------------------------------------------------------------------
        # Block if IP is not in the allowed list
        if ($allowed_mlflow_ip = 0) {
            return 403 '{"error": "Access denied. Only Azure Web App can access this service."}';
        }
        
        # ----------------------------------------------------------------------
        # RATE LIMITING
        # ----------------------------------------------------------------------
        # Limit to 10 req/s with burst of 20
        # nodelay: don't delay requests within burst limit
        limit_req zone=mlflow_limit burst=20 nodelay;
        
        # Limit concurrent connections
        limit_conn mlflow_conn_limit 20;
        
        # ----------------------------------------------------------------------
        # PROXY CONFIGURATION
        # ----------------------------------------------------------------------
        # Forward all requests to MLflow backend
        proxy_pass http://mlflow_backend;
        
        # Preserve original request information
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_set_header X-Forwarded-Host $host;
        proxy_set_header X-Forwarded-Port $server_port;
        
        # ----------------------------------------------------------------------
        # TIMEOUTS (MLflow can be slow for large operations)
        # ----------------------------------------------------------------------
        proxy_connect_timeout 60s;  # Time to establish connection
        proxy_send_timeout 300s;     # Time to send request (large model uploads)
        proxy_read_timeout 300s;     # Time to read response (large model downloads)
        
        # ----------------------------------------------------------------------
        # BUFFERING SETTINGS
        # ----------------------------------------------------------------------
        # Disable buffering for streaming responses
        proxy_buffering off;
        proxy_request_buffering off;
        
        # Large file support (for model artifacts)
        client_max_body_size 500M;
        
        # Buffer sizes
        proxy_buffer_size 128k;
        proxy_buffers 4 256k;
        proxy_busy_buffers_size 256k;
        
        # ----------------------------------------------------------------------
        # HTTP VERSION
        # ----------------------------------------------------------------------
        proxy_http_version 1.1;
        
        # WebSocket support (if needed)
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
    }
    
    # ==============================================================================
    # SPECIFIC ENDPOINTS (Optional: Fine-grained control)
    # ==============================================================================
    
    # Experiments API
    location ~ ^/api/2\.0/mlflow/experiments {
        if ($allowed_mlflow_ip = 0) {
            return 403;
        }
        limit_req zone=mlflow_limit burst=20 nodelay;
        proxy_pass http://mlflow_backend;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
    
    # Runs API
    location ~ ^/api/2\.0/mlflow/runs {
        if ($allowed_mlflow_ip = 0) {
            return 403;
        }
        limit_req zone=mlflow_limit burst=20 nodelay;
        proxy_pass http://mlflow_backend;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
    
    # Models API (may have larger payloads)
    location ~ ^/api/2\.0/mlflow/model-versions {
        if ($allowed_mlflow_ip = 0) {
            return 403;
        }
        limit_req zone=mlflow_limit burst=30 nodelay;  # Higher burst for models
        client_max_body_size 1G;  # Allow larger uploads
        proxy_pass http://mlflow_backend;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_read_timeout 600s;  # Longer timeout for large models
    }
    
    # Artifacts (static files)
    location /get-artifact {
        if ($allowed_mlflow_ip = 0) {
            return 403;
        }
        limit_req zone=mlflow_limit burst=50 nodelay;  # Higher burst for downloads
        proxy_pass http://mlflow_backend;
        proxy_set_header Host $host;
        proxy_buffering on;  # Enable buffering for file downloads
    }
    
    # ==============================================================================
    # ERROR PAGES
    # ==============================================================================
    
    # 403 Forbidden (IP not allowed)
    error_page 403 = @forbidden;
    location @forbidden {
        default_type application/json;
        return 403 '{"error": "Access denied", "message": "Your IP is not authorized to access this service", "client_ip": "$remote_addr"}';
    }
    
    # 429 Too Many Requests (rate limit exceeded)
    error_page 429 = @ratelimit;
    location @ratelimit {
        default_type application/json;
        return 429 '{"error": "Rate limit exceeded", "message": "Too many requests. Please slow down.", "limit": "10 requests per second"}';
    }
    
    # 502 Bad Gateway (MLflow down)
    error_page 502 = @badgateway;
    location @badgateway {
        default_type application/json;
        return 502 '{"error": "Service unavailable", "message": "MLflow tracking server is not responding", "action": "Check MLflow container status: docker ps | grep mlflow"}';
    }
    
    # 504 Gateway Timeout
    error_page 504 = @timeout;
    location @timeout {
        default_type application/json;
        return 504 '{"error": "Gateway timeout", "message": "MLflow took too long to respond", "timeout": "300 seconds"}';
    }
}
```
```bash
# ==============================================================================
# MONITORING AND STATISTICS
# ==============================================================================
# Optional: Enable Nginx stub_status for monitoring
# Uncomment if you want to monitor Nginx performance
#
# server {
#     listen 8080;
#     server_name localhost;
#     
#     location /nginx_status {
#         stub_status on;
#         access_log off;
#         allow 127.0.0.1;
#         deny all;
#     }
# }

# ==============================================================================
# NOTES AND MAINTENANCE
# ==============================================================================
#
# Testing Configuration:
# ----------------------
# sudo nginx -t                    # Test syntax
# sudo systemctl reload nginx      # Apply changes
#
# View Logs:
# ----------
# sudo tail -f /var/log/nginx/mlflow-access.log
# sudo tail -f /var/log/nginx/mlflow-error.log
#
# Test Endpoints:
# ---------------
# # From VM2 (should work)
# curl http://localhost:5000/health
#
# # From allowed IP (should work)
# curl http://<VM2_PUBLIC_IP>:5000/health
#
# # From unauthorized IP (should return 403)
# curl http://<VM2_PUBLIC_IP>:5000/api/2.0/mlflow/experiments/list
#
# Update IP Whitelist:
# --------------------
# 1. Edit this file: sudo nano /etc/nginx/sites-available/mlflow-proxy.conf
# 2. Add new IPs in the geo $allowed_mlflow_ip block
# 3. Test: sudo nginx -t
# 4. Reload: sudo systemctl reload nginx
#
# Monitor Rate Limiting:
# ----------------------
# grep "limiting requests" /var/log/nginx/mlflow-error.log
# grep "403" /var/log/nginx/mlflow-access.log
#
# ==============================================================================
```

---

### **Step 3: Enable Nginx Configuration**

```bash
# Create symbolic link to enable the site
sudo ln -s /etc/nginx/sites-available/mlflow-proxy.conf /etc/nginx/sites-enabled/

# Verify the symlink was created
ls -la /etc/nginx/sites-enabled/mlflow-proxy.conf

# Test Nginx configuration
sudo nginx -t

# Expected output:
# nginx: the configuration file /etc/nginx/nginx.conf syntax is ok
# nginx: configuration file /etc/nginx/nginx.conf test is successful

# If you get an error about "include /etc/nginx/sites-enabled/*", add it to nginx.conf:
# sudo nano /etc/nginx/nginx.conf
# Inside the http {} block, add:
# include /etc/nginx/sites-enabled/*;

# Reload Nginx to apply changes
sudo systemctl reload nginx

# Verify Nginx is running
sudo systemctl status nginx

# Check that Nginx is listening on port 5000
sudo netstat -tlnp | grep :5000
# Expected: tcp 0 0.0.0.0:5000 0.0.0.0:* LISTEN <PID>/nginx
```

---

### **Step 4: Configure UFW Firewall**

**Allow MLflow access from Azure Web App IPs only:**

```bash
# Allow each Azure Web App outbound IP
sudo ufw allow from 20.19.1.44 to any port 5000 proto tcp comment "MLflow - Azure Web App"
sudo ufw allow from 20.19.1.119 to any port 5000 proto tcp comment "MLflow - Azure Web App"
sudo ufw allow from 20.19.1.218 to any port 5000 proto tcp comment "MLflow - Azure Web App"
sudo ufw allow from 20.19.2.151 to any port 5000 proto tcp comment "MLflow - Azure Web App"
sudo ufw allow from 20.19.2.185 to any port 5000 proto tcp comment "MLflow - Azure Web App"
sudo ufw allow from 20.19.2.239 to any port 5000 proto tcp comment "MLflow - Azure Web App"
sudo ufw allow from 51.138.218.150 to any port 5000 proto tcp comment "MLflow - Azure Web App"
sudo ufw allow from 51.138.223.105 to any port 5000 proto tcp comment "MLflow - Azure Web App"
sudo ufw allow from 20.74.97.20 to any port 5000 proto tcp comment "MLflow - Azure Web App"
sudo ufw allow from 51.138.216.249 to any port 5000 proto tcp comment "MLflow - Azure Web App"
sudo ufw allow from 20.74.98.69 to any port 5000 proto tcp comment "MLflow - Azure Web App"
sudo ufw allow from 20.74.99.131 to any port 5000 proto tcp comment "MLflow - Azure Web App"
sudo ufw allow from 20.111.1.5 to any port 5000 proto tcp comment "MLflow - Azure Web App"

# Reload firewall
sudo ufw reload

# Verify rules
sudo ufw status numbered
```

**Expected output:**

```
To                         Action      From
--                         ------      ----
[1] 22/tcp                 ALLOW       Anywhere
[2] 5000/tcp               ALLOW       20.19.1.44       # MLflow - Azure Web App
[3] 5000/tcp               ALLOW       20.19.1.119      # MLflow - Azure Web App
...
[14] 5000/tcp              ALLOW       20.111.1.5       # MLflow - Azure Web App
```

---

### **Step 5: Configure Azure Web App**

**Set MLflow tracking URI in Azure Web App settings:**

```bash
# Get VM1 public IP
VM1_PUBLIC_IP=$(az vm show \
  --resource-group fraud-detection-rg \
  --name VM1 \
  --show-details \
  --query publicIps \
  --output tsv)

echo "VM1 Public IP: $VM1_PUBLIC_IP"

# Update Azure Web App configuration
az webapp config appsettings set \
  --name fraud-detection-api-ammi-2025 \
  --resource-group fraud-detection-rg \
  --settings \
    MLFLOW_TRACKING_URI="http://${VM1_PUBLIC_IP}:5000"

# Restart the Web App to apply changes
az webapp restart \
  --name fraud-detection-api-ammi-2025 \
  --resource-group fraud-detection-rg

# Verify the setting
az webapp config appsettings list \
  --name fraud-detection-api-ammi-2025 \
  --resource-group fraud-detection-rg \
  --query "[?name=='MLFLOW_TRACKING_URI']"
```

---

### **Step 6: Restart Services**

```bash
# On VM1, restart Docker services
cd ~/fraud-detection-ml
docker-compose -f docker-compose.vm1.yml down
docker-compose -f docker-compose.vm1.yml up -d

# Check logs
docker logs fraud-mlflow --tail 50

# Verify MLflow is accessible through Nginx
curl -s http://localhost:5000/health
# Expected: {"status": "ok"} or similar
```

---

## ✅ Testing & Validation

### **Test 1: Local Access (Should Work)**

```bash
# On VM1, test direct access to MLflow container
curl http://localhost:5001/api/2.0/mlflow/experiments/list

# Test through Nginx proxy
curl http://localhost:5000/api/2.0/mlflow/experiments/list

# Both should return the same response
```

---

### **Test 2: External Access (Should Fail - IP Not Whitelisted)**

```bash
# From your local machine (NOT VM1)
curl http://<VM1_PUBLIC_IP>:5000/api/2.0/mlflow/experiments/list

# Expected: 403 Forbidden
# {"error": "Access denied. IP not whitelisted."}
```

---

### **Test 3: Azure Web App Access (Should Work)**

**Option A: Check API logs**

```bash
# Tail Azure Web App logs
az webapp log tail \
  --name fraud-detection-api-ammi-2025 \
  --resource-group fraud-detection-rg

# Look for MLflow connection messages
```

**Option B: Test API endpoints that use MLflow**

```bash
# Test model prediction endpoint
curl -X POST https://fraud-detection-api-ammi-2025.azurewebsites.net/predict \
  -H "Content-Type: application/json" \
  -d '{
    "Time": 0,
    "V1": -1.359807,
    "V2": -0.072781,
    "V3": 2.536347,
    "V4": 1.378155,
    "V5": -0.338321,
    "V6": 0.462388,
    "V7": 0.239599,
    "V8": 0.098698,
    "V9": 0.363787,
    "V10": 0.090794,
    "V11": -0.551600,
    "V12": -0.617801,
    "V13": -0.991390,
    "V14": -0.311169,
    "V15": 1.468177,
    "V16": -0.470401,
    "V17": 0.207971,
    "V18": 0.025791,
    "V19": 0.403993,
    "V20": 0.251412,
    "V21": -0.018307,
    "V22": 0.277838,
    "V23": -0.110474,
    "V24": 0.066928,
    "V25": 0.128539,
    "V26": -0.189115,
    "V27": 0.133558,
    "V28": -0.021053,
    "Amount": 149.62
  }'

# Expected: Prediction response with fraud probability
```

**Option C: Check available models endpoint**

```bash
# List available models (requires MLflow access)
curl https://fraud-detection-api-ammi-2025.azurewebsites.net/models/available

# Expected: List of models from MLflow
```

---

### **Test 4: Verify Nginx Logs**

```bash
# On VM1, check Nginx access logs
sudo tail -f /var/log/nginx/mlflow-access.log

# You should see requests from Azure Web App IPs:
# 20.19.1.44 - - [06/Nov/2025:12:34:56 +0000] "GET /api/2.0/mlflow/experiments/list HTTP/1.1" 200 ...

# Check error logs
sudo tail -f /var/log/nginx/mlflow-error.log

# Should be empty or contain only informational messages
```

---

### **Test 5: Rate Limiting**

```bash
# From a whitelisted IP (if you have access), send rapid requests
for i in {1..30}; do
  curl -s http://<VM1_IP>:5000/api/2.0/mlflow/experiments/list
done

# Expected: First 20 requests succeed (burst=20)
# Requests 21-30 may return 503 Service Temporarily Unavailable (rate limited)
```

---

## 📊 Monitoring

### **1. Nginx Access Logs**

Monitor all MLflow access in real-time:

```bash
# Watch access logs
sudo tail -f /var/log/nginx/mlflow-access.log

# Example log entry:
# 20.19.1.44 - - [06/Nov/2025:15:23:45 +0000] "GET /api/2.0/mlflow/experiments/list HTTP/1.1" 200 1234 "-" "python-requests/2.28.0"
```

### **2. Analyze Traffic Patterns**

```bash
# Count requests per IP
sudo awk '{print $1}' /var/log/nginx/mlflow-access.log | sort | uniq -c | sort -rn

# Count requests per endpoint
sudo awk '{print $7}' /var/log/nginx/mlflow-access.log | sort | uniq -c | sort -rn

# Check response codes
sudo awk '{print $9}' /var/log/nginx/mlflow-access.log | sort | uniq -c
```

### **3. Monitor MLflow Container**

```bash
# Check MLflow container stats
docker stats fraud-mlflow

# View MLflow logs
docker logs fraud-mlflow --tail 100 -f

# Check MLflow disk usage
docker exec fraud-mlflow du -sh /mlflow/artifacts
```

### **4. Create Monitoring Script**

```bash
# Create monitoring script
cat > ~/monitor-mlflow.sh << 'EOF'
#!/bin/bash
# MLflow Monitoring Script

echo "=== MLflow Status ==="
docker ps | grep fraud-mlflow

echo -e "\n=== Nginx MLflow Proxy Status ==="
sudo systemctl status nginx | grep -A 5 "Active:"

echo -e "\n=== Recent Access Logs (Last 10) ==="
sudo tail -10 /var/log/nginx/mlflow-access.log

echo -e "\n=== Request Count by IP (Last 1000 requests) ==="
sudo tail -1000 /var/log/nginx/mlflow-access.log | awk '{print $1}' | sort | uniq -c | sort -rn

echo -e "\n=== Error Count ==="
sudo grep -c "error" /var/log/nginx/mlflow-error.log

echo -e "\n=== Disk Usage ==="
docker exec fraud-mlflow du -sh /mlflow/artifacts 2>/dev/null || echo "Could not check artifacts"
EOF

chmod +x ~/monitor-mlflow.sh

# Run monitoring
~/monitor-mlflow.sh
```

---

## 🔧 Troubleshooting

### **Issue 1: 403 Forbidden from Azure Web App**

**Symptom:** Your API cannot access MLflow, returns 403 error.

**Solution:**

```bash
# 1. Verify Azure Web App IPs haven't changed
az webapp show \
  --name fraud-detection-api-ammi-2025 \
  --resource-group fraud-detection-rg \
  --query outboundIpAddresses -o tsv

# 2. Check Nginx geo configuration
sudo cat /etc/nginx/sites-available/mlflow-proxy | grep -A 15 "geo \$allowed_mlflow_ip"

# 3. Verify UFW rules
sudo ufw status | grep 5000

# 4. Test from VM1 (should work)
curl http://localhost:5000/health

# 5. Check Nginx error logs
sudo tail -50 /var/log/nginx/mlflow-error.log
```

---

### **Issue 2: Connection Timeout**

**Symptom:** Requests to MLflow timeout.

**Solution:**

```bash
# 1. Verify MLflow container is running
docker ps | grep fraud-mlflow

# 2. Check if MLflow responds locally
curl http://localhost:5001/health

# 3. Check if Nginx is listening on port 5000
sudo netstat -tlnp | grep :5000

# 4. Verify proxy_pass URL in Nginx config
sudo grep "proxy_pass" /etc/nginx/sites-available/mlflow-proxy

# 5. Increase timeouts in Nginx if needed
sudo nano /etc/nginx/sites-available/mlflow-proxy
# Increase proxy_read_timeout, proxy_send_timeout, proxy_connect_timeout

sudo systemctl reload nginx
```

---

### **Issue 3: 502 Bad Gateway**

**Symptom:** Nginx returns 502 error.

**Solution:**

```bash
# 1. Check if MLflow container is running
docker ps | grep mlflow

# 2. Check MLflow logs for errors
docker logs fraud-mlflow --tail 50

# 3. Verify MLflow is listening on correct port
docker exec fraud-mlflow netstat -tlnp | grep 5000

# 4. Test direct connection to container
docker exec fraud-mlflow curl http://localhost:5000/health

# 5. Restart MLflow if needed
docker-compose -f docker-compose.vm1.yml restart mlflow
```

---

### **Issue 4: Rate Limiting Too Strict**

**Symptom:** Legitimate requests are being rate limited.

**Solution:**

```bash
# Edit Nginx configuration
sudo nano /etc/nginx/sites-available/mlflow-proxy

# Increase rate limit:
# Change: rate=10r/s
# To:     rate=30r/s

# And increase burst:
# Change: burst=20
# To:     burst=50

# Test and reload
sudo nginx -t && sudo systemctl reload nginx
```

---

### **Issue 5: Large Model Upload Fails**

**Symptom:** Uploading large models to MLflow fails.

**Solution:**

```bash
# Edit Nginx configuration
sudo nano /etc/nginx/sites-available/mlflow-proxy

# Increase client_max_body_size:
# Change: client_max_body_size 100M;
# To:     client_max_body_size 500M;

# Also increase timeouts:
# proxy_send_timeout 600s;
# proxy_read_timeout 600s;

# Reload Nginx
sudo nginx -t && sudo systemctl reload nginx
```

---

## 🔄 Alternative Solutions

### **Option 1: Azure Container Instances (Managed MLflow)**

If you prefer a fully managed solution:

```bash
# Deploy MLflow as Azure Container Instance
az container create \
  --resource-group fraud-detection-rg \
  --name mlflow-tracking \
  --image ghcr.io/mlflow/mlflow:latest \
  --dns-name-label fraud-mlflow-ammi \
  --ports 5000 \
  --cpu 2 \
  --memory 4 \
  --environment-variables \
    BACKEND_STORE_URI="postgresql://user:pass@fraud-detection-db.postgres.database.azure.com/mlflow" \
    ARTIFACT_ROOT="wasbs://mlflow@fraudstorage.blob.core.windows.net/artifacts"

# Then use:
# MLFLOW_TRACKING_URI=http://fraud-mlflow-ammi.francecentral.azurecontainer.io:5000
```

**Pros:**
- ✅ Fully managed (no VM maintenance)
- ✅ Built-in networking features
- ✅ Easy scaling

**Cons:**
- ❌ Additional cost (~$30-50/month)
- ❌ Less control over configuration

---

### **Option 2: Azure App Service (MLflow as Web App)**

Deploy MLflow like your API:

```bash
# Create App Service for MLflow
az webapp create \
  --resource-group fraud-detection-rg \
  --plan fraud-detection-plan \
  --name fraud-mlflow-tracking \
  --runtime "PYTHON:3.10"

# Configure with startup command:
# mlflow server --host 0.0.0.0 --port 8000 --backend-store-uri postgresql://...
```

**Pros:**
- ✅ Same platform as API
- ✅ Built-in SSL/TLS
- ✅ Auto-scaling

**Cons:**
- ❌ Additional cost (~$10-20/month)
- ❌ Requires container registry setup

---

### **Option 3: VPN/Private Endpoint**

For maximum security (advanced):

```bash
# Create VPN between Azure and VM1
# Or use Azure Private Endpoint

# This allows private network communication
# MLflow not exposed to internet at all
```

**Pros:**
- ✅✅ Maximum security
- ✅ No public exposure

**Cons:**
- ❌ Complex setup
- ❌ Additional networking costs

---

## 📈 Performance Tuning

### **1. Enable Keepalive Connections**

Already configured in the provided Nginx config:

```nginx
upstream mlflow_backend {
    server 127.0.0.1:5001;
    keepalive 32;          # Keep 32 connections open
    keepalive_requests 100; # Reuse each connection 100 times
    keepalive_timeout 60s;  # Keep alive for 60 seconds
}
```

### **2. Enable Response Caching (Optional)**

For read-only endpoints (experiments list, etc.):

```nginx
# Add to server block
proxy_cache_path /var/cache/nginx/mlflow levels=1:2 keys_zone=mlflow_cache:10m max_size=1g inactive=60m;

location ~ ^/api/2.0/mlflow/(experiments|runs)/(list|search) {
    proxy_cache mlflow_cache;
    proxy_cache_valid 200 5m;
    proxy_cache_key "$scheme$request_method$host$request_uri";
    
    proxy_pass http://mlflow_backend;
    # ... other settings
}
```

### **3. Monitor Performance**

```bash
# Check Nginx worker connections
sudo nginx -V 2>&1 | grep worker_connections

# Monitor active connections
watch -n 1 "sudo netstat -an | grep :5000 | wc -l"

# Check connection states
sudo netstat -an | grep :5000 | awk '{print $6}' | sort | uniq -c
```

---

## 🔐 Security Checklist

Before going live, verify:

- [ ] MLflow container bound to localhost only (`127.0.0.1:5001`)
- [ ] Nginx configuration created and enabled
- [ ] All Azure Web App IPs in `geo $allowed_mlflow_ip` block
- [ ] UFW firewall rules configured (only Azure IPs on port 5000)
- [ ] Nginx listening on port 5000 (`netstat -tlnp | grep :5000`)
- [ ] Rate limiting configured (`limit_req_zone` in Nginx)
- [ ] Access logs enabled and monitored
- [ ] Azure Web App `MLFLOW_TRACKING_URI` updated
- [ ] Test: Unauthorized IPs blocked (403 response)
- [ ] Test: Azure Web App can access MLflow
- [ ] Test: API predictions work correctly
- [ ] Monitoring script created and scheduled

---

## 📊 Architecture Comparison

| Aspect | Without Nginx | With Nginx Proxy |
|--------|---------------|------------------|
| **URL** | `http://VM1_IP:5001` | `http://VM1_IP:5000` |
| **Container Binding** | `0.0.0.0:5001:5000` | `127.0.0.1:5001:5000` |
| **Public Access** | ❌ Anyone | ✅ Azure Web App only |
| **IP Whitelist** | ❌ Not possible | ✅ 13 Azure IPs |
| **Rate Limiting** | ❌ None | ✅ 10 req/s per IP |
| **Access Logs** | ⚠️ MLflow only | ✅ Nginx + MLflow |
| **Security Level** | ⚠️ Low | ✅ High |
| **API Code Changes** | - | ✅ **Zero** |
| **Setup Time** | 0 min | 15-20 min |
| **Cost** | $0 | $0 |

---

## 🎯 Summary

### **What You Achieved**

1. ✅ **Secured MLflow access** - Only Azure Web App can connect
2. ✅ **No API changes** - Same MLflow client code works
3. ✅ **Rate limiting** - Protection against abuse
4. ✅ **Centralized logging** - All access logged by Nginx
5. ✅ **Zero cost** - Uses existing VM1 resources
6. ✅ **Production-ready** - Suitable for academic and production use

### **Final Configuration**

```bash
# Azure Web App
MLFLOW_TRACKING_URI=http://<VM1_PUBLIC_IP>:5000

# VM1 - MLflow Container
127.0.0.1:5001:5000  # Localhost only

# VM1 - Nginx
Port 5000 → Reverse proxy → localhost:5001
IP whitelist: 13 Azure Web App IPs
Rate limit: 10 req/s per IP

# UFW Firewall
Allow port 5000 from Azure IPs only
```

### **Your API Code**

**No changes needed!** Your existing code continues to work:

```python
# api/src/services/model_versions.py
import mlflow
import os

# Automatically uses MLFLOW_TRACKING_URI from environment
mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))
client = mlflow.client.MlflowClient()

# All existing MLflow operations work without modification
experiments = client.search_experiments()
models = client.search_registered_models()
```

---

## 📚 Additional Resources

- **Nginx Reverse Proxy**: https://docs.nginx.com/nginx/admin-guide/web-server/reverse-proxy/
- **MLflow Tracking**: https://mlflow.org/docs/latest/tracking.html
- **Azure Web App Networking**: https://learn.microsoft.com/azure/app-service/networking-features
- **UFW Firewall**: https://help.ubuntu.com/community/UFW

---

**Document Version:** 1.0  
**Last Updated:** 2025-11-06  
**Maintained By:** Fraud Detection Team  
**Purpose:** Secure MLflow access for Azure Web App with zero API code changes
