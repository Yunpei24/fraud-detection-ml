# Security Architecture Documentation

**Fraud Detection ML Platform - Multi-Layer Security Design**

---

## Table of Contents

1. [Overview](#overview)
2. [Network Architecture](#network-architecture)
3. [Security Layers](#security-layers)
4. [Service-Specific Security](#service-specific-security)
5. [Authentication & Authorization](#authentication--authorization)
6. [IP Whitelisting](#ip-whitelisting)
7. [Encryption & Secure Communication](#encryption--secure-communication)
8. [Firewall Configuration](#firewall-configuration)
9. [Monitoring & Metrics Security](#monitoring--metrics-security)
10. [Best Practices](#best-practices)
11. [Security Checklist](#security-checklist)

---

## Overview

The Fraud Detection ML platform implements a **defense-in-depth security strategy** with multiple layers of protection:

- **Network Security Group (NSG)** - Azure cloud firewall
- **UFW (Uncomplicated Firewall)** - OS-level firewall on VM1
- **Nginx Reverse Proxy** - Application-level access control
- **SSL/TLS Encryption** - Secure data transmission
- **Authentication** - Password-based access control
- **Rate Limiting** - DDoS protection

### Deployment Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         INTERNET                                 │
└────────────┬────────────────────────────────────────────────────┘
             │
             ├─── Azure Web App (13 IPs)
             ├─── Render Backend (5 IPs/ranges)
             ├─── VM2 Prometheus (40.120.29.234)
             └─── Mac Local Dev (41.214.116.126)
             │
┌────────────▼─────────────────────────────────────────────────────┐
│                    AZURE NSG (Layer 1)                           │
│  - Rule 360: Azure Web App (Service Tag) → 5433                 │
│  - Rule 370: Azure Web App (13 IPs) → 5433                      │
│  - Rule 380: Render Backend (5 IPs) → 5433                      │
│  - Rule 390: VM2 Prometheus → 9091,9095,9097                    │
└────────────┬─────────────────────────────────────────────────────┘
             │
┌────────────▼─────────────────────────────────────────────────────┐
│                    VM1 (40.66.54.22)                             │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │           UFW FIREWALL (Layer 2)                          │  │
│  │  - 48 rules for specific IPs and ports                   │  │
│  │  - PostgreSQL: 5433 (14 IPs + 2 ranges)                  │  │
│  │  - MLflow: 5000 (14 IPs)                                 │  │
│  │  - Airflow: 8080 (1 IP)                                  │  │
│  │  - Metrics: 9091,9095,9097 (1 IP)                        │  │
│  └───────────────┬───────────────────────────────────────────┘  │
│                  │                                               │
│  ┌───────────────▼───────────────────────────────────────────┐  │
│  │           NGINX REVERSE PROXY (Layer 3)                   │  │
│  │                                                            │  │
│  │  ┌──────────────────────────────────────────────────┐    │  │
│  │  │ PostgreSQL Proxy (TCP Stream)                    │    │  │
│  │  │ Port: 5433 → 5432                                │    │  │
│  │  │ Security: UFW only (pass-through)                │    │  │
│  │  └──────────────────────────────────────────────────┘    │  │
│  │                                                            │  │
│  │  ┌──────────────────────────────────────────────────┐    │  │
│  │  │ MLflow Proxy (HTTP)                              │    │  │
│  │  │ Port: 5000 → 5001                                │    │  │
│  │  │ Security: Geo IP Whitelist + Rate Limiting       │    │  │
│  │  │ IPs: 14 authorized (Azure + Mac + VM2)           │    │  │
│  │  │ Rate: 10 req/s, burst 20                         │    │  │
│  │  └──────────────────────────────────────────────────┘    │  │
│  │                                                            │  │
│  │  ┌──────────────────────────────────────────────────┐    │  │
│  │  │ Metrics Proxy (HTTP with Basic Auth)            │    │  │
│  │  │ Ports: 9091,9095,9097 → 9191,9195,9197          │    │  │
│  │  │ Security: HTTP Basic Authentication              │    │  │
│  │  │ Credentials: prometheus / password               │    │  │
│  │  └──────────────────────────────────────────────────┘    │  │
│  └────────────────┬───────────────────────────────────────┘  │
│                   │                                            │
│  ┌────────────────▼───────────────────────────────────────┐  │
│  │         DOCKER CONTAINERS (Layer 4)                    │  │
│  │                                                         │  │
│  │  ┌────────────────────────────────────────────────┐   │  │
│  │  │ PostgreSQL:5432 (SSL + SCRAM-SHA-256)          │   │  │
│  │  │ - SSL: auto-generated certs                    │   │  │
│  │  │ - Auth: SCRAM-SHA-256 password hashing         │   │  │
│  │  │ - Users: fraud_user, admin                     │   │  │
│  │  └────────────────────────────────────────────────┘   │  │
│  │                                                         │  │
│  │  ┌────────────────────────────────────────────────┐   │  │
│  │  │ MLflow:5001 (localhost only)                   │   │  │
│  │  │ - Binding: 127.0.0.1:5001                      │   │  │
│  │  │ - Access: via Nginx proxy only                 │   │  │
│  │  └────────────────────────────────────────────────┘   │  │
│  │                                                         │  │
│  │  ┌────────────────────────────────────────────────┐   │  │
│  │  │ Metrics Services (localhost only)              │   │  │
│  │  │ - Data: 127.0.0.1:9191                         │   │  │
│  │  │ - Training: 127.0.0.1:9195                     │   │  │
│  │  │ - Drift: 127.0.0.1:9197                        │   │  │
│  │  │ - Access: via Nginx proxy only                 │   │  │
│  │  └────────────────────────────────────────────────┘   │  │
│  └─────────────────────────────────────────────────────┘  │
└────────────────────────────────────────────────────────────────┘
```

---

## Network Architecture

### VM1 - Production Services (40.66.54.22)

**Infrastructure:**
- OS: Ubuntu 20.04/22.04 LTS
- Cloud: Azure (West Europe or similar region)
- Network Security Group: Fraud.VM1-nsg
- Resource Group: fraud-detection-rg

**Services Deployed:**
- PostgreSQL 15 (port 5432 internal, 5433 external via Nginx)
- PgBouncer 1.24.1 (port 6432 internal)
- MLflow Tracking Server (port 5001 internal, 5000 external via Nginx)
- Apache Airflow (port 8080)
- Data Service (metrics on 9191 internal, 9091 external via Nginx)
- Training Service (metrics on 9195 internal, 9095 external via Nginx)
- Drift Service (metrics on 9197 internal, 9097 external via Nginx)
- Redis, Kafka (internal only)

### VM2 - Monitoring Stack (40.120.29.234)

**Infrastructure:**
- OS: Ubuntu 20.04/22.04 LTS
- Cloud: Azure (separate subscription)
- Purpose: Centralized monitoring and alerting

**Services Deployed:**
- Prometheus (port 9090)
- Grafana (port 3000)
- Alertmanager (port 9093)

### External Services

**Azure Web App (fraud-detection-api-ammi-2025):**
- Purpose: FastAPI backend
- Outbound IPs: 13 static IPs (20.19.x.x, 51.138.x.x, 20.74.x.x, 20.111.1.5)
- Access: PostgreSQL (5433), MLflow (5000)

**Render Backend (Node.js):**
- Purpose: Backend API
- Outbound IPs: 5 IPs/ranges (100.20.92.101, 44.225.181.72, 44.227.217.144, 74.220.48.0/24, 74.220.56.0/24)
- Access: PostgreSQL (5433)

**Mac Local Development:**
- IP: 41.214.116.126
- Access: MLflow (5000), Airflow (8080)

---

## Security Layers

### Layer 1: Azure Network Security Group (NSG)

**Purpose:** Cloud-level firewall managed by Azure

**Location:** Azure Portal → Fraud.VM1-nsg → Inbound security rules

**Configuration:**

| Priority | Name | Source | Ports | Protocol | Action | Description |
|----------|------|--------|-------|----------|--------|-------------|
| 360 | AllowAzureWebAppPostgreSQL | Service Tag: AppService | 5433 | TCP | Allow | Azure Web App via Service Tag |
| 370 | AllowAzureWebAppPostgreSQLIP | 13 Azure IPs | 5433 | TCP | Allow | Azure Web App via specific IPs |
| 380 | AllowRenderBackendPostgreSQL | 5 Render IPs/ranges | 5433 | TCP | Allow | Render backend Node.js |
| 390 | AllowPrometheusVM2 | 40.120.29.234 | 9091,9095,9097 | TCP | Allow | VM2 Prometheus metrics scraping |

**Management:**

```bash
# List all NSG rules
az network nsg rule list \
  --resource-group "fraud-detection-rg" \
  --nsg-name "Fraud.VM1-nsg" \
  --output table

# Create new rule
az network nsg rule create \
  --resource-group "fraud-detection-rg" \
  --nsg-name "Fraud.VM1-nsg" \
  --name "RuleName" \
  --priority 400 \
  --source-address-prefixes "1.2.3.4" \
  --destination-port-ranges 5433 \
  --protocol Tcp \
  --access Allow

# Delete rule
az network nsg rule delete \
  --resource-group "fraud-detection-rg" \
  --nsg-name "Fraud.VM1-nsg" \
  --name "RuleName"
```

**Key Features:**
- ✅ Blocks all traffic by default (deny-all rule)
- ✅ Only explicitly allowed IPs can reach VM1
- ✅ Service Tags automatically update when Azure service IPs change
- ✅ Centralized management via Azure Portal or CLI

---

### Layer 2: UFW (Uncomplicated Firewall)

**Purpose:** OS-level firewall on VM1 (Ubuntu)

**Status:** Active with 48+ rules

**Configuration File:** Managed via `ufw` command

**Key Rules:**

```bash
# PostgreSQL Access
ufw allow from 20.19.1.44 to any port 5433 comment "Azure Web App"
ufw allow from 20.19.1.119 to any port 5433 comment "Azure Web App"
ufw allow from 20.19.1.218 to any port 5433 comment "Azure Web App"
ufw allow from 20.19.2.151 to any port 5433 comment "Azure Web App"
ufw allow from 20.19.2.185 to any port 5433 comment "Azure Web App"
ufw allow from 20.19.2.239 to any port 5433 comment "Azure Web App"
ufw allow from 51.138.218.150 to any port 5433 comment "Azure Web App"
ufw allow from 51.138.223.105 to any port 5433 comment "Azure Web App"
ufw allow from 20.74.97.20 to any port 5433 comment "Azure Web App"
ufw allow from 51.138.216.249 to any port 5433 comment "Azure Web App"
ufw allow from 20.74.98.69 to any port 5433 comment "Azure Web App"
ufw allow from 20.74.99.131 to any port 5433 comment "Azure Web App"
ufw allow from 20.111.1.5 to any port 5433 comment "Azure Web App"
ufw allow from 40.120.29.234 to any port 5433 comment "PostgreSQL - Additional IP"

# Render Backend
ufw allow from 100.20.92.101 to any port 5433 comment "PostgreSQL - Render backend"
ufw allow from 44.225.181.72 to any port 5433 comment "PostgreSQL - Render backend"
ufw allow from 44.227.217.144 to any port 5433 comment "PostgreSQL - Render backend"
ufw allow from 74.220.48.0/24 to any port 5433 comment "PostgreSQL - Render backend range 1"
ufw allow from 74.220.56.0/24 to any port 5433 comment "PostgreSQL - Render backend range 2"

# MLflow Access
ufw allow from 20.19.1.44 to any port 5000 comment "MLflow - Azure Web App"
ufw allow from 20.19.1.119 to any port 5000 comment "MLflow - Azure Web App"
# ... (all 13 Azure IPs)
ufw allow from 41.214.116.126 to any port 5000 comment "MLflow - Mac Local Machine"
ufw allow from 40.120.29.234 to any port 5000 comment "MLflow - Additional IP"

# Airflow Access
ufw allow from 41.214.116.126 to any port 8080 comment "Airflow - Mac Local Machine"

# Prometheus Metrics (VM2)
ufw allow from 40.120.29.234 to any port 9091 comment "Prometheus - Data metrics"
ufw allow from 40.120.29.234 to any port 9095 comment "Prometheus - Training metrics"
ufw allow from 40.120.29.234 to any port 9097 comment "Prometheus - Drift metrics"

# SSH Access (always keep this!)
ufw allow 22/tcp comment "SSH access"
```

**Management Commands:**

```bash
# View all rules
sudo ufw status numbered

# Add new rule
sudo ufw allow from <IP> to any port <PORT> comment "Description"

# Delete rule by number
sudo ufw delete <number>

# Delete rule by specification
sudo ufw delete allow from <IP> to any port <PORT>

# Reload UFW
sudo ufw reload

# Enable/Disable
sudo ufw enable
sudo ufw disable
```

**Best Practices:**
- ✅ Always use comments for traceability
- ✅ Specific IPs only (no 0.0.0.0/0)
- ✅ Keep SSH port 22 open (avoid lockout!)
- ✅ Test before enabling in production
- ✅ Document all IP addresses and their purpose

---

### Layer 3: Nginx Reverse Proxy

**Purpose:** Application-level access control, SSL termination, rate limiting

**Configuration Files:**
- `/etc/nginx/streams-enabled/postgres-proxy.conf` (TCP stream)
- `/etc/nginx/sites-available/mlflow-proxy.conf` (HTTP)
- `/etc/nginx/sites-available/metrics-auth` (HTTP with Basic Auth)

#### 3.1 PostgreSQL Proxy (TCP Stream)

**File:** `/etc/nginx/streams-enabled/postgres-proxy.conf`

**Configuration:**

```nginx
# PostgreSQL TCP Proxy (Pass-through SSL)
# Port 5433 (external) -> 5432 (internal PostgreSQL with SSL)
# Security: IPs filtered by UFW

server {
    listen 5433;
    
    # Proxy TCP pass-through to internal PostgreSQL
    proxy_pass 127.0.0.1:5432;
    
    # Timeouts
    proxy_connect_timeout 10s;
    proxy_timeout 300s;
    
    # Buffer sizes
    proxy_buffer_size 16k;
}
```

**Security Model:**
- ✅ No IP whitelisting in Nginx (relies on UFW Layer 2)
- ✅ Pure TCP pass-through (SSL handled by PostgreSQL)
- ✅ Timeouts prevent connection hanging
- ✅ Port mapping: 5433 (public) → 5432 (private)

#### 3.2 MLflow Proxy (HTTP with Geo IP Whitelisting)

**File:** `/etc/nginx/sites-available/mlflow-proxy.conf`

**Key Features:**
- ✅ Geo module IP whitelisting
- ✅ Rate limiting (10 req/s, burst 20)
- ✅ Connection limiting (20 concurrent)
- ✅ Large file support (500MB+)
- ✅ Custom error pages (403, 429, 502, 504)
- ✅ Health check endpoint

**IP Whitelist Configuration:**

```nginx
geo $allowed_mlflow_ip {
    default 0;  # Block all by default
    
    # Azure Web App outbound IPs
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
    
    # VM2 Monitoring
    40.120.29.234 1;
    
    # Local development machine
    41.214.116.126 1;
}
```

**Rate Limiting:**

```nginx
# Limit to 10 requests per second per IP address
limit_req_zone $binary_remote_addr zone=mlflow_limit:10m rate=10r/s;

# Connection limit: max 20 concurrent connections per IP
limit_conn_zone $binary_remote_addr zone=mlflow_conn_limit:10m;
```

**Usage in Location Block:**

```nginx
location / {
    # IP whitelisting check
    if ($allowed_mlflow_ip = 0) {
        return 403 '{"error": "Access denied. Only authorized IPs can access."}';
    }
    
    # Rate limiting
    limit_req zone=mlflow_limit burst=20 nodelay;
    limit_conn mlflow_conn_limit 20;
    
    # Proxy to backend
    proxy_pass http://mlflow_backend;
    
    # Security headers
    proxy_set_header Host $host;
    proxy_set_header X-Real-IP $remote_addr;
    proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
}
```

**Management:**

```bash
# Test configuration
sudo nginx -t

# Reload without downtime
sudo systemctl reload nginx

# Restart (use if config changes are major)
sudo systemctl restart nginx

# View error logs
sudo tail -f /var/log/nginx/mlflow-error.log

# View access logs
sudo tail -f /var/log/nginx/mlflow-access.log
```

#### 3.3 Metrics Proxy (HTTP with Basic Authentication)

**File:** `/etc/nginx/sites-available/metrics-auth`

**Purpose:** Protect Prometheus metrics endpoints with username/password

**Configuration:**

```nginx
# Data Service Metrics (port 9091)
server {
    listen 9091;
    server_name _;
    
    location /metrics {
        auth_basic "Prometheus Metrics";
        auth_basic_user_file /etc/nginx/.htpasswd;
        
        proxy_pass http://localhost:9191/metrics;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}

# Training Service Metrics (port 9095)
server {
    listen 9095;
    server_name _;
    
    location /metrics {
        auth_basic "Prometheus Metrics";
        auth_basic_user_file /etc/nginx/.htpasswd;
        
        proxy_pass http://localhost:9195/metrics;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}

# Drift Service Metrics (port 9097)
server {
    listen 9097;
    server_name _;
    
    location /metrics {
        auth_basic "Prometheus Metrics";
        auth_basic_user_file /etc/nginx/.htpasswd;
        
        proxy_pass http://localhost:9197/metrics;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}
```

**Credentials Management:**

```bash
# Create password file
sudo htpasswd -c /etc/nginx/.htpasswd prometheus

# Add another user
sudo htpasswd /etc/nginx/.htpasswd another_user

# Set proper permissions
sudo chmod 644 /etc/nginx/.htpasswd

# Test authentication
curl -I http://localhost:9091/metrics
# Returns: HTTP/1.1 401 Unauthorized

curl -u prometheus:password http://localhost:9091/metrics
# Returns: Metrics data
```

**Security Features:**
- ✅ HTTP Basic Authentication (username/password required)
- ✅ Credentials stored as bcrypt hashes
- ✅ Port mapping to localhost-only Docker containers
- ✅ Prevents unauthorized metrics access

---

### Layer 4: Docker Container Security

**Purpose:** Isolate services, limit network exposure

#### 4.1 PostgreSQL Container

**Security Configuration:**

```yaml
# docker-compose.vm1.yml
postgres:
  image: postgres:15-alpine
  ports:
    - "127.0.0.1:5432:5432"  # Localhost only!
  environment:
    POSTGRES_PASSWORD_FILE: /run/secrets/postgres_password
    POSTGRES_INITDB_ARGS: "--auth-host=scram-sha-256 --auth-local=scram-sha-256"
  volumes:
    - ./ssl/server.crt:/var/lib/postgresql/server.crt:ro
    - ./ssl/server.key:/var/lib/postgresql/server.key:ro
  command: >
    postgres
    -c ssl=on
    -c ssl_cert_file=/var/lib/postgresql/server.crt
    -c ssl_key_file=/var/lib/postgresql/server.key
    -c password_encryption=scram-sha-256
```

**Security Features:**
- ✅ **Localhost binding:** `127.0.0.1:5432` (not exposed to internet)
- ✅ **SSL/TLS enabled:** Encrypted connections required
- ✅ **SCRAM-SHA-256:** Strong password hashing (better than MD5)
- ✅ **Docker secrets:** Passwords stored securely
- ✅ **Auto-generated SSL certificates** (can be replaced with Let's Encrypt)

**User Management:**

```sql
-- Create application user with limited privileges
CREATE USER fraud_user WITH PASSWORD 'secure_prod_password_2024_change_me' 
CONNECTION LIMIT 100;

-- Create admin user
CREATE USER admin WITH PASSWORD 'admin123' 
CREATEROLE CREATEDB;

-- Grant specific privileges only
GRANT CONNECT ON DATABASE fraud_detection TO fraud_user;
GRANT USAGE ON SCHEMA public TO fraud_user;
GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA public TO fraud_user;
```

#### 4.2 MLflow Container

**Security Configuration:**

```yaml
# docker-compose.vm1.yml
mlflow:
  image: your-registry/mlflow:latest
  ports:
    - "127.0.0.1:5001:5001"  # Localhost only!
  environment:
    MLFLOW_BACKEND_STORE_URI: postgresql://fraud_user:${POSTGRES_PASSWORD}@postgres:5432/mlflow
    MLFLOW_ARTIFACT_ROOT: /mlflow/artifacts
```

**Security Features:**
- ✅ **Localhost binding:** `127.0.0.1:5001` (access via Nginx only)
- ✅ **Database credentials:** Managed via environment variables
- ✅ **No direct internet exposure**

#### 4.3 Metrics Services (Data, Training, Drift)

**Security Configuration:**

```yaml
# docker-compose.vm1.yml
data:
  ports:
    - "127.0.0.1:9191:9091"  # Localhost only!

training:
  ports:
    - "127.0.0.1:9195:9095"  # Localhost only!

drift:
  ports:
    - "127.0.0.1:9197:9097"  # Localhost only!
```

**Security Features:**
- ✅ **Localhost binding:** All metrics on `127.0.0.1` only
- ✅ **Access via Nginx:** Requires Basic Auth
- ✅ **Port mapping:** Public ports (9091, 9095, 9097) → Private ports (9191, 9195, 9197)

---

## Service-Specific Security

### PostgreSQL Security

**Multi-Layer Protection:**

```
Client → NSG (5433) → UFW (5433) → Nginx (5433→5432) → PostgreSQL SSL (5432)
```

**Security Controls:**

1. **Network Level (NSG):**
   - Only Azure Web App and Render IPs allowed
   - Port 5433 only

2. **OS Level (UFW):**
   - 14 specific IPs + 2 IP ranges whitelisted
   - Port 5433 only

3. **Proxy Level (Nginx):**
   - TCP pass-through (no modification)
   - Timeout protection

4. **Application Level (PostgreSQL):**
   - SSL/TLS encryption mandatory
   - SCRAM-SHA-256 password hashing
   - User-level permissions
   - Connection pooling via PgBouncer

**Connection String Example:**

```bash
# Secure connection with SSL
postgresql://fraud_user:password@40.66.54.22:5433/fraud_detection?sslmode=require

# From Python
import psycopg2
conn = psycopg2.connect(
    host="40.66.54.22",
    port=5433,
    database="fraud_detection",
    user="fraud_user",
    password="secure_password",
    sslmode="require"
)
```

**Security Audit Checklist:**
- [ ] SSL certificates not expired
- [ ] Passwords use SCRAM-SHA-256 (not MD5)
- [ ] No superuser credentials in application code
- [ ] Connection pooling enabled (PgBouncer)
- [ ] Regular database backups
- [ ] Log file rotation configured

---

### MLflow Security

**Multi-Layer Protection:**

```
Client → NSG (5000) → UFW (5000) → Nginx Geo IP Filter (5000→5001) → MLflow (5001)
```

**Security Controls:**

1. **Network Level (NSG):**
   - No specific NSG rule (relies on default deny + UFW)

2. **OS Level (UFW):**
   - 14 specific IPs whitelisted
   - Port 5000 only

3. **Proxy Level (Nginx):**
   - **Geo IP whitelisting** (14 authorized IPs)
   - **Rate limiting:** 10 req/s, burst 20
   - **Connection limiting:** 20 concurrent connections
   - Custom error pages (403, 429, 502, 504)
   - Health check endpoint

4. **Application Level (MLflow):**
   - Database credentials managed via environment variables
   - Artifact storage with proper permissions

**Rate Limiting Behavior:**

```bash
# Normal request (within limit)
curl http://40.66.54.22:5000/api/2.0/mlflow/experiments/list
# Returns: 200 OK

# Rapid requests (exceeding 10 req/s)
for i in {1..30}; do curl http://40.66.54.22:5000/health; done
# First 10: 200 OK
# Next 10 (burst): 200 OK
# After burst: 429 Too Many Requests
```

**Access Control:**

```nginx
# Allowed IPs (in geo module)
40.120.29.234 → 1 (allowed)
192.168.1.1   → 0 (blocked, returns 403)
```

**Management:**

```bash
# Add new IP to whitelist
sudo nano /etc/nginx/sites-available/mlflow-proxy.conf

# Add under geo $allowed_mlflow_ip:
# 1.2.3.4 1;

# Test and reload
sudo nginx -t && sudo systemctl reload nginx

# Verify
curl -I http://40.66.54.22:5000/health
# Should return 200 OK from allowed IP
```

---

### Prometheus Metrics Security

**Multi-Layer Protection:**

```
Prometheus (VM2) → NSG (9091/9095/9097) → UFW (9091/9095/9097) 
→ Nginx Basic Auth (9091→9191) → Docker Container (9191)
```

**Security Controls:**

1. **Network Level (NSG):**
   - Rule 390: VM2 IP (40.120.29.234) → Ports 9091, 9095, 9097

2. **OS Level (UFW):**
   - VM2 IP whitelisted for ports 9091, 9095, 9097

3. **Proxy Level (Nginx):**
   - **HTTP Basic Authentication** required
   - Username: `prometheus`
   - Password: Stored in `/etc/nginx/.htpasswd` (bcrypt hash)

4. **Application Level (Docker):**
   - Containers bound to `127.0.0.1` only
   - No direct internet exposure

**Prometheus Configuration (VM2):**

```yaml
# monitoring/prometheus/prometheus.vm2.yml
scrape_configs:
  - job_name: 'fraud-data'
    scrape_interval: 30s
    basic_auth:
      username: 'prometheus'
      password: 'secure_password_here'
    static_configs:
      - targets: ['40.66.54.22:9091']

  - job_name: 'fraud-training'
    scrape_interval: 30s
    basic_auth:
      username: 'prometheus'
      password: 'secure_password_here'
    static_configs:
      - targets: ['40.66.54.22:9095']

  - job_name: 'fraud-drift'
    scrape_interval: 60s
    basic_auth:
      username: 'prometheus'
      password: 'secure_password_here'
    static_configs:
      - targets: ['40.66.54.22:9097']
```

**Testing Authentication:**

```bash
# Without credentials (should fail with 401)
curl -I http://40.66.54.22:9091/metrics
# HTTP/1.1 401 Unauthorized

# With credentials (should succeed)
curl -u prometheus:password http://40.66.54.22:9091/metrics
# Returns: Prometheus metrics data

# Check from VM2 Prometheus
# Targets page: http://40.66.54.22:9090/targets
# Should show all 3 targets as UP
```

---

## Authentication & Authorization

### PostgreSQL Authentication

**Method:** SCRAM-SHA-256 (SHA-256 Challenge-Response)

**Configuration:**

```bash
# postgresql.conf
password_encryption = scram-sha-256

# pg_hba.conf
host    all             all             0.0.0.0/0            scram-sha-256
```

**Password Hash Example:**

```sql
-- User creation
CREATE USER fraud_user WITH PASSWORD 'my_secure_password';

-- Password stored as:
-- SCRAM-SHA-256$4096:<salt>:<stored-key>:<server-key>
```

**Benefits:**
- ✅ Resistant to rainbow table attacks
- ✅ Salt per user prevents pre-computation
- ✅ Better than MD5 (legacy PostgreSQL default)
- ✅ Industry standard (SCRAM = Salted Challenge Response Authentication Mechanism)

**User Roles:**

```sql
-- Application user (limited privileges)
CREATE ROLE fraud_user WITH LOGIN PASSWORD 'secure_password' 
CONNECTION LIMIT 100;

GRANT CONNECT ON DATABASE fraud_detection TO fraud_user;
GRANT USAGE ON SCHEMA public TO fraud_user;
GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA public TO fraud_user;

-- Admin user (elevated privileges)
CREATE ROLE admin WITH LOGIN PASSWORD 'admin_password' 
CREATEROLE CREATEDB;

-- Read-only user (analytics)
CREATE ROLE analyst WITH LOGIN PASSWORD 'analyst_password';
GRANT CONNECT ON DATABASE fraud_detection TO analyst;
GRANT USAGE ON SCHEMA public TO analyst;
GRANT SELECT ON ALL TABLES IN SCHEMA public TO analyst;
```

### Nginx Basic Authentication

**Method:** HTTP Basic Authentication (RFC 7617)

**Password Storage:**

```bash
# Create password file
sudo htpasswd -c /etc/nginx/.htpasswd prometheus
# Prompts for password
# Password stored as bcrypt hash

# File content (/etc/nginx/.htpasswd):
prometheus:$2y$05$abc123...encrypted_password_hash
```

**Configuration:**

```nginx
server {
    listen 9091;
    
    location /metrics {
        auth_basic "Prometheus Metrics";
        auth_basic_user_file /etc/nginx/.htpasswd;
        
        proxy_pass http://localhost:9191/metrics;
    }
}
```

**Security Considerations:**
- ⚠️ **Not encrypted by itself** (credentials sent in Base64)
- ✅ **Solution:** Use with HTTPS/TLS (recommended for production)
- ✅ **Password hashed:** bcrypt with salt
- ✅ **Simple to implement:** No complex OAuth/JWT setup needed

**Usage:**

```bash
# From command line
curl -u prometheus:password http://40.66.54.22:9091/metrics

# From Prometheus
basic_auth:
  username: prometheus
  password: password

# From browser (prompts for login)
# http://40.66.54.22:9091/metrics
```

---

## IP Whitelisting

### Complete IP Registry

**Azure Web App (13 IPs):**
```
20.19.1.44
20.19.1.119
20.19.1.218
20.19.2.151
20.19.2.185
20.19.2.239
51.138.218.150
51.138.223.105
20.74.97.20
51.138.216.249
20.74.98.69
20.74.99.131
20.111.1.5
```

**Render Backend (5 IPs/Ranges):**
```
100.20.92.101
44.225.181.72
44.227.217.144
74.220.48.0/24  (256 IPs: 74.220.48.0 - 74.220.48.255)
74.220.56.0/24  (256 IPs: 74.220.56.0 - 74.220.56.255)
```

**VM2 Prometheus:**
```
40.120.29.234
```

**Mac Local Development:**
```
41.214.116.126
```

### IP Whitelist Management

**UFW (OS Firewall):**

```bash
# Add IP for PostgreSQL
sudo ufw allow from 1.2.3.4 to any port 5433 comment "Description"

# Add IP for MLflow
sudo ufw allow from 1.2.3.4 to any port 5000 comment "Description"

# Add IP for metrics
sudo ufw allow from 1.2.3.4 to any port 9091 comment "Description"

# View all rules
sudo ufw status numbered

# Delete rule
sudo ufw delete <number>
```

**Nginx (MLflow Geo Module):**

```bash
# Edit configuration
sudo nano /etc/nginx/sites-available/mlflow-proxy.conf

# Add IP under geo $allowed_mlflow_ip
1.2.3.4 1;

# Test and reload
sudo nginx -t
sudo systemctl reload nginx
```

**Azure NSG:**

```bash
# Add IP to existing rule
az network nsg rule update \
  --resource-group "fraud-detection-rg" \
  --nsg-name "Fraud.VM1-nsg" \
  --name "AllowAzureWebAppPostgreSQLIP" \
  --source-address-prefixes 20.19.1.44 20.19.1.119 1.2.3.4

# Or create new rule
az network nsg rule create \
  --resource-group "fraud-detection-rg" \
  --nsg-name "Fraud.VM1-nsg" \
  --name "AllowNewService" \
  --priority 400 \
  --source-address-prefixes 1.2.3.4 \
  --destination-port-ranges 5433 \
  --protocol Tcp \
  --access Allow
```

### IP Change Procedures

**When Azure Web App IPs change:**

1. Get new IPs:
```bash
az webapp show \
  --name fraud-detection-api-ammi-2025 \
  --resource-group fraud-detection-rg \
  --query outboundIpAddresses -o tsv
```

2. Update UFW:
```bash
# Add new IPs
sudo ufw allow from <NEW_IP> to any port 5433 comment "Azure Web App"
sudo ufw allow from <NEW_IP> to any port 5000 comment "MLflow - Azure Web App"
```

3. Update Nginx:
```bash
sudo nano /etc/nginx/sites-available/mlflow-proxy.conf
# Add: <NEW_IP> 1;
sudo nginx -t && sudo systemctl reload nginx
```

4. Update NSG:
```bash
az network nsg rule update \
  --resource-group "fraud-detection-rg" \
  --nsg-name "Fraud.VM1-nsg" \
  --name "AllowAzureWebAppPostgreSQLIP" \
  --source-address-prefixes <NEW_IP_LIST>
```

5. Test connection:
```bash
# From Azure Web App logs
# Should see successful PostgreSQL/MLflow connections
```

---

## Encryption & Secure Communication

### PostgreSQL SSL/TLS

**Configuration:**

```bash
# Generate self-signed certificate (development)
cd fraud-detection-ml/ssl
openssl req -new -x509 -days 365 -nodes -text \
  -out server.crt \
  -keyout server.key \
  -subj "/CN=fraud-ml-vm1"

chmod 600 server.key
chmod 644 server.crt
```

**Docker Compose SSL Mounting:**

```yaml
postgres:
  volumes:
    - ./ssl/server.crt:/var/lib/postgresql/server.crt:ro
    - ./ssl/server.key:/var/lib/postgresql/server.key:ro
  command: >
    postgres
    -c ssl=on
    -c ssl_cert_file=/var/lib/postgresql/server.crt
    -c ssl_key_file=/var/lib/postgresql/server.key
```

**Client Connection (SSL Required):**

```python
import psycopg2

# Require SSL (reject if server doesn't support SSL)
conn = psycopg2.connect(
    host="40.66.54.22",
    port=5433,
    database="fraud_detection",
    user="fraud_user",
    password="password",
    sslmode="require"  # Options: disable, allow, prefer, require, verify-ca, verify-full
)
```

**SSL Mode Comparison:**

| Mode | SSL Required? | Verifies Certificate? | Security Level |
|------|---------------|-----------------------|----------------|
| disable | No | No | ❌ Lowest |
| allow | Tries, fallback to plain | No | ⚠️ Low |
| prefer | Tries, fallback to plain | No | ⚠️ Medium |
| require | Yes | No | ✅ Good |
| verify-ca | Yes | Yes (CA only) | ✅ Better |
| verify-full | Yes | Yes (CA + hostname) | ✅ Best |

**Production Recommendation:**
- Use **Let's Encrypt** or corporate CA certificates
- Use **verify-full** mode for maximum security
- Rotate certificates annually

### Nginx SSL/TLS (Future Enhancement)

**Currently:** HTTP only (assumes Azure handles SSL termination)

**Recommended for Production:**

```nginx
server {
    listen 5000 ssl http2;
    
    ssl_certificate /etc/letsencrypt/live/fraud-ml.example.com/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/fraud-ml.example.com/privkey.pem;
    
    ssl_protocols TLSv1.2 TLSv1.3;
    ssl_ciphers HIGH:!aNULL:!MD5;
    ssl_prefer_server_ciphers on;
    
    # HSTS
    add_header Strict-Transport-Security "max-age=31536000; includeSubDomains" always;
    
    location / {
        # ... existing config
    }
}
```

**Benefits:**
- ✅ End-to-end encryption
- ✅ Prevents man-in-the-middle attacks
- ✅ Modern TLS protocols (1.2, 1.3)
- ✅ HSTS enforces HTTPS

---

## Firewall Configuration

### UFW Rules Summary

**Total Rules:** 48+

**Port Distribution:**
- **5433 (PostgreSQL):** 19 rules (13 Azure + 5 Render + 1 Additional)
- **5000 (MLflow):** 14 rules (13 Azure + 1 Mac + 1 Additional)
- **8080 (Airflow):** 1 rule (Mac)
- **9091, 9095, 9097 (Metrics):** 1 rule each (VM2)
- **22 (SSH):** 1 rule (Anywhere)

**Complete Rule Export:**

```bash
# Export all rules
sudo ufw status numbered > ufw-rules-backup.txt

# Or in detailed format
sudo ufw show added > ufw-rules-detailed.txt
```

**Restore Rules:**

```bash
# Disable UFW
sudo ufw disable

# Reset to defaults
sudo ufw reset

# Re-add rules from script
bash scripts/restore-ufw-rules.sh

# Enable
sudo ufw enable
```

### Azure NSG Rules Summary

**Total Rules:** 4 custom + default rules

**Custom Rules:**

| Priority | Name | Source | Destination | Ports | Protocol | Access |
|----------|------|--------|-------------|-------|----------|--------|
| 360 | AllowAzureWebAppPostgreSQL | Service Tag: AppService | Any | 5433 | TCP | Allow |
| 370 | AllowAzureWebAppPostgreSQLIP | 13 IPs | Any | 5433 | TCP | Allow |
| 380 | AllowRenderBackendPostgreSQL | 5 IPs/ranges | Any | 5433 | TCP | Allow |
| 390 | AllowPrometheusVM2 | 40.120.29.234 | Any | 9091,9095,9097 | TCP | Allow |

**Default Rules (lower priority, typically 65000+):**
- AllowVnetInBound
- AllowAzureLoadBalancerInBound
- DenyAllInBound

**Rule Priority Best Practices:**
- ✅ 100-199: Critical infrastructure
- ✅ 200-299: Application services
- ✅ 300-399: External access (PostgreSQL, MLflow)
- ✅ 400-499: Monitoring and metrics
- ✅ 500+: Testing and temporary rules

---

## Monitoring & Metrics Security

### Prometheus Security Model

**Architecture:**

```
Prometheus (VM2) 
  ↓ (Basic Auth)
Nginx (VM1 ports 9091, 9095, 9097)
  ↓ (localhost proxy)
Docker Containers (127.0.0.1:9191, 9195, 9197)
  ↓ (Prometheus Python client)
Application Metrics
```

**Security Layers:**

1. **Network Isolation:**
   - Docker containers bind to `127.0.0.1` only
   - Not accessible from internet directly

2. **Nginx Proxy:**
   - Maps public ports (9091, 9095, 9097) to private ports (9191, 9195, 9197)
   - Adds HTTP Basic Authentication

3. **Firewall:**
   - UFW allows only VM2 IP (40.120.29.234)
   - NSG rule 390 enforces same restriction

4. **Authentication:**
   - Username: `prometheus`
   - Password: Stored in `/etc/nginx/.htpasswd` (bcrypt)

### Metrics Endpoints

**Data Service (Port 9091):**
```bash
# From VM2 Prometheus
curl -u prometheus:password http://40.66.54.22:9091/metrics

# Metrics exposed:
# - kafka_consumer_lag
# - kafka_messages_processed_total
# - data_processing_duration_seconds
# - postgresql_connections
```

**Training Service (Port 9095):**
```bash
# From VM2 Prometheus
curl -u prometheus:password http://40.66.54.22:9095/metrics

# Metrics exposed:
# - model_training_duration_seconds
# - model_accuracy
# - model_precision
# - model_recall
# - training_samples_total
```

**Drift Service (Port 9097):**
```bash
# From VM2 Prometheus
curl -u prometheus:password http://40.66.54.22:9097/metrics

# Metrics exposed:
# - data_drift_detected
# - feature_drift_score
# - model_performance_degradation
# - drift_check_duration_seconds
```

### Grafana Dashboards (VM2)

**Access:** http://40.120.29.234:3000

**Security:**
- ⚠️ **Currently:** No authentication (default Grafana)
- ✅ **Recommended:** Enable authentication, HTTPS

**Enable Authentication:**

```yaml
# grafana/config/grafana.ini
[auth.anonymous]
enabled = false

[auth.basic]
enabled = true

[security]
admin_user = admin
admin_password = secure_password_change_me
```

**Add HTTPS:**

```yaml
# docker-compose.vm2.yml
grafana:
  environment:
    - GF_SERVER_PROTOCOL=https
    - GF_SERVER_CERT_FILE=/etc/grafana/ssl/cert.pem
    - GF_SERVER_CERT_KEY=/etc/grafana/ssl/key.pem
  volumes:
    - ./ssl:/etc/grafana/ssl:ro
```

---

## Best Practices

### General Security

- ✅ **Principle of Least Privilege:** Grant minimum necessary permissions
- ✅ **Defense in Depth:** Multiple security layers (NSG → UFW → Nginx → App)
- ✅ **Fail Secure:** Default deny, explicit allow
- ✅ **Audit Logging:** Enable logs for all services
- ✅ **Regular Updates:** Keep OS, Docker, services patched
- ✅ **Secrets Management:** Use environment variables, Docker secrets (never hardcode)

### Network Security

- ✅ **IP Whitelisting:** Specific IPs only, no 0.0.0.0/0
- ✅ **Port Minimization:** Expose only necessary ports
- ✅ **Localhost Binding:** Bind services to 127.0.0.1 when possible
- ✅ **Firewall Enabled:** UFW active on all VMs
- ✅ **NSG Rules:** Centralized control via Azure

### Application Security

- ✅ **Strong Passwords:** Minimum 16 characters, mixed case, numbers, symbols
- ✅ **Password Hashing:** SCRAM-SHA-256 (PostgreSQL), bcrypt (Nginx)
- ✅ **SSL/TLS:** Encrypt data in transit
- ✅ **Rate Limiting:** Prevent brute force and DDoS
- ✅ **Connection Limits:** Prevent resource exhaustion

### Monitoring & Compliance

- ✅ **Centralized Logging:** Ship logs to SIEM (future: ELK, Splunk)
- ✅ **Metrics Collection:** Prometheus + Grafana
- ✅ **Alerting:** Alertmanager for security events
- ✅ **Regular Audits:** Review firewall rules, user permissions
- ✅ **Incident Response:** Document procedures

### Maintenance

- ✅ **Certificate Rotation:** Renew SSL certificates before expiration
- ✅ **Password Rotation:** Change passwords quarterly
- ✅ **IP Whitelist Review:** Remove stale IPs monthly
- ✅ **Backup Testing:** Verify database backups work
- ✅ **Disaster Recovery:** Document recovery procedures

---

## Security Checklist

### Pre-Deployment

- [ ] All default passwords changed
- [ ] UFW firewall enabled and configured
- [ ] Azure NSG rules created
- [ ] SSL certificates generated (PostgreSQL)
- [ ] Nginx configurations tested (`nginx -t`)
- [ ] Docker containers bind to localhost only
- [ ] Environment variables set (no hardcoded secrets)
- [ ] SSH key-based authentication enabled
- [ ] Root login disabled

### Post-Deployment

- [ ] Verify PostgreSQL SSL connection works
- [ ] Test MLflow access from Azure Web App
- [ ] Confirm rate limiting working (429 errors)
- [ ] Check Prometheus scraping VM1 metrics
- [ ] Review Nginx access/error logs
- [ ] Verify UFW rules active (`ufw status`)
- [ ] Confirm NSG rules effective (test from unauthorized IP)
- [ ] Test Basic Auth on metrics endpoints (401 without creds)

### Monthly

- [ ] Review UFW firewall logs (`sudo journalctl -u ufw`)
- [ ] Check for failed authentication attempts
- [ ] Audit user list (PostgreSQL, Linux)
- [ ] Remove stale IPs from whitelist
- [ ] Update OS packages (`apt update && apt upgrade`)
- [ ] Review Nginx logs for anomalies
- [ ] Test disaster recovery procedures

### Quarterly

- [ ] Rotate database passwords
- [ ] Rotate Nginx Basic Auth passwords
- [ ] Review and update NSG rules
- [ ] Renew SSL certificates (if near expiration)
- [ ] Perform penetration testing
- [ ] Review access logs for suspicious activity
- [ ] Update documentation with any changes

### Annually

- [ ] Full security audit by external team
- [ ] Review and update security policies
- [ ] Disaster recovery drill
- [ ] Update incident response plan
- [ ] Compliance review (GDPR, HIPAA, etc.)

---

## Incident Response

### Security Incident Types

1. **Unauthorized Access Attempt**
   - Symptoms: Multiple 403/401 errors in logs, failed SSH attempts
   - Response: Block IP in UFW, review logs, notify team

2. **DDoS Attack**
   - Symptoms: 429 errors, high CPU/memory, slow response
   - Response: Enable aggressive rate limiting, block IP ranges, contact Azure support

3. **Data Breach**
   - Symptoms: Unexpected data access, suspicious queries
   - Response: Isolate system, preserve evidence, notify authorities, change all credentials

4. **Service Compromise**
   - Symptoms: Unexpected processes, modified files, privilege escalation
   - Response: Shutdown service, snapshot VM, forensic analysis, rebuild from clean state

### Response Procedures

**Step 1: Detect**
```bash
# Check recent access logs
sudo tail -1000 /var/log/nginx/mlflow-access.log | grep -v "200\|301"

# Check UFW blocks
sudo journalctl -u ufw | grep BLOCK | tail -100

# Check failed SSH attempts
sudo grep "Failed password" /var/log/auth.log | tail -50
```

**Step 2: Contain**
```bash
# Block suspicious IP immediately
sudo ufw insert 1 deny from <SUSPICIOUS_IP>

# Add to NSG deny rule
az network nsg rule create \
  --resource-group "fraud-detection-rg" \
  --nsg-name "Fraud.VM1-nsg" \
  --name "BlockMaliciousIP" \
  --priority 100 \
  --source-address-prefixes <SUSPICIOUS_IP> \
  --access Deny

# Shutdown affected service if necessary
docker compose -f docker-compose.vm1.yml stop <service>
```

**Step 3: Investigate**
```bash
# Export logs for analysis
sudo tar -czf incident-logs-$(date +%Y%m%d).tar.gz \
  /var/log/nginx/ \
  /var/log/auth.log \
  /var/log/syslog

# Database audit
docker exec postgres psql -U postgres -d fraud_detection \
  -c "SELECT * FROM pg_stat_activity WHERE state = 'active';"
```

**Step 4: Recover**
```bash
# Change compromised passwords
sudo htpasswd /etc/nginx/.htpasswd prometheus

# Rotate database credentials
# (Update .env, restart services)

# Rebuild from known-good state if necessary
```

**Step 5: Document**
- Incident timeline
- Root cause analysis
- Actions taken
- Lessons learned
- Preventive measures

---

## Contact & Support

**Security Team:**
- Primary Contact: [security@example.com](mailto:security@example.com)
- Emergency Hotline: +1-XXX-XXX-XXXX
- Incident Reports: [incidents@example.com](mailto:incidents@example.com)

**Documentation:**
- Architecture Diagrams: `/docs/architecture/`
- Runbooks: `/docs/runbooks/`
- Incident Response: `/docs/incident-response/`

**External Resources:**
- Azure Security Best Practices: https://learn.microsoft.com/azure/security
- OWASP Top 10: https://owasp.org/www-project-top-ten/
- CIS Benchmarks: https://www.cisecurity.org/cis-benchmarks/

---

## Appendix

### A. Complete UFW Rule List

```bash
# Export current rules
sudo ufw status numbered > docs/ufw-rules-$(date +%Y%m%d).txt
```

### B. Complete NSG Configuration

```bash
# Export NSG rules
az network nsg rule list \
  --resource-group "fraud-detection-rg" \
  --nsg-name "Fraud.VM1-nsg" \
  --output table > docs/nsg-rules-$(date +%Y%m%d).txt
```

### C. Nginx Configuration Files

Located in:
- `/etc/nginx/nginx.conf` (main config)
- `/etc/nginx/streams-enabled/postgres-proxy.conf`
- `/etc/nginx/sites-available/mlflow-proxy.conf`
- `/etc/nginx/sites-available/metrics-auth`

### D. Docker Security Scanning

```bash
# Scan images for vulnerabilities
docker scan fraudenzoubi/fraud-data:latest
docker scan fraudenzoubi/fraud-training:latest
docker scan fraudenzoubi/fraud-drift:latest

# Use Trivy for comprehensive scanning
trivy image fraudenzoubi/fraud-data:latest
```

### E. Compliance Mappings

**GDPR:**
- Data encryption at rest and in transit ✅
- Access logging and audit trails ✅
- Right to be forgotten (database deletion procedures) ✅

**HIPAA (if applicable):**
- Access controls (IP whitelisting, authentication) ✅
- Audit controls (logging) ✅
- Integrity controls (SSL/TLS) ✅
- Transmission security (encrypted connections) ✅

**SOC 2:**
- Security (multi-layer defense) ✅
- Availability (monitoring, alerting) ✅
- Processing Integrity (data validation) ✅
- Confidentiality (encryption, access control) ✅

---

## Revision History

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0 | 2025-11-07 | MLOps Team | Initial security architecture documentation |

---

**End of Document**
