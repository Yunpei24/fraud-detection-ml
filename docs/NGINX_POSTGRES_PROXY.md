# Nginx PostgreSQL Proxy Guide (Academic/Development Setup)

## ⚠️ Important Notice

**This guide is intended for academic/development environments ONLY.**

While Azure Database for PostgreSQL is the recommended production solution, this guide provides a cost-effective alternative for:
- 🎓 Academic projects and coursework
- 💰 Limited budget scenarios
- 🧪 Development and testing environments
- 📚 Learning database security concepts

**For production systems, always use Azure Database for PostgreSQL (see `AZURE_DATABASE_MIGRATION.md`).**

---

## Table of Contents
- [Overview](#overview)
- [Architecture](#architecture)
- [Security Considerations](#security-considerations)
- [Prerequisites](#prerequisites)
- [Step-by-Step Setup](#step-by-step-setup)
- [Testing & Validation](#testing--validation)
- [Monitoring](#monitoring)
- [Troubleshooting](#troubleshooting)
- [Cost Comparison](#cost-comparison)

---

## Overview

### What This Guide Does

This guide shows how to securely expose your PostgreSQL Docker container (running on VM1) to your Azure Web App API using **Nginx as a TCP/SSL reverse proxy**.

### Why Use Nginx for PostgreSQL?

**Advantages:**
- ✅ No additional Azure costs (uses existing VM1 resources)
- ✅ SSL/TLS encryption for database connections
- ✅ IP whitelisting and access control
- ✅ Connection rate limiting
- ✅ Centralized logging and monitoring
- ✅ Works with existing Docker Compose setup

**Disadvantages:**
- ❌ Manual management (no automatic backups like Azure DB)
- ❌ No built-in high availability
- ❌ You must manage security patches
- ❌ Performance depends on VM1 resources
- ❌ Single point of failure

---

## Architecture

### Current Setup (Insecure)
```
┌─────────────────────────────────────────────────────┐
│ VM1                                                 │
│                                                     │
│  ┌──────────────┐      ┌──────────────┐           │
│  │ PostgreSQL   │      │    Data      │           │
│  │ Container    │◀─────│   Service    │           │
│  │ Port: 5432   │      └──────────────┘           │
│  └──────────────┘                                  │
│         ▲                                          │
│         │ ❌ Would need public exposure            │
│         │    (DANGEROUS - no encryption)           │
└─────────┼──────────────────────────────────────────┘
          │
          │ Insecure connection
          │
┌─────────┼──────────────────────────────────────────┐
│ Azure   │                                          │
│         │                                          │
│  ┌──────┴───────┐                                 │
│  │  FastAPI     │                                 │
│  │  (Web App)   │                                 │
│  └──────────────┘                                 │
└─────────────────────────────────────────────────────┘
```

### Secure Setup with Nginx (This Guide)
```
┌──────────────────────────────────────────────────────────┐
│ VM1                                                      │
│                                                          │
│  ┌──────────────┐      ┌──────────────┐                │
│  │ PostgreSQL   │      │    Data      │                │
│  │ Container    │◀─────│   Service    │                │
│  │ 127.0.0.1    │      └──────────────┘                │
│  │ Port: 5432   │                                       │
│  └──────────────┘                                       │
│         ▲                                               │
│         │ Localhost only                                │
│         │ (not exposed)                                 │
│         │                                               │
│  ┌──────┴────────────────────────────────┐             │
│  │  Nginx TCP Stream Proxy               │             │
│  │  Port: 5433 (external)                │             │
│  │  ✓ SSL/TLS encryption                 │             │
│  │  ✓ IP whitelist (Azure Web App only)  │             │
│  │  ✓ Rate limiting                      │             │
│  │  ✓ Connection logging                 │             │
│  └───────────────────────────────────────┘             │
│         ▲                                               │
│         │ Firewall: Allow port 5433                     │
│         │ Only from Azure Web App IP                    │
└─────────┼───────────────────────────────────────────────┘
          │
          │ SSL/TLS encrypted connection
          │ (PostgreSQL sslmode=require)
          │
┌─────────┼───────────────────────────────────────────────┐
│ Azure   │                                               │
│         │                                               │
│  ┌──────┴────────────────────────┐                     │
│  │  FastAPI (Web App)            │                     │
│  │  Connection:                  │                     │
│  │  VM1_IP:5433                  │                     │
│  │  sslmode=require              │                     │
│  └───────────────────────────────┘                     │
└──────────────────────────────────────────────────────────┘
```

### Data Flow
1. **Azure Web App** connects to `VM1_IP:5433` with SSL
2. **Nginx** on VM1 validates:
   - Source IP (must be Azure Web App outbound IP)
   - Connection rate limits
   - Logs connection attempt
3. **Nginx** proxies connection to `localhost:5432` (PostgreSQL container)
4. **PostgreSQL** handles the query and returns response
5. **Nginx** encrypts response and sends back to Web App

---

## Security Considerations

### What This Setup Protects Against
- ✅ **Man-in-the-middle attacks** - SSL/TLS encryption
- ✅ **Unauthorized access** - IP whitelisting
- ✅ **DDoS attacks** - Rate limiting
- ✅ **Port scanning** - Non-standard port (5433)
- ✅ **Connection monitoring** - Centralized logging

### What This Setup Does NOT Protect Against
- ❌ **VM1 compromise** - If VM1 is hacked, database is exposed
- ❌ **Data loss** - No automatic backups (you must configure)
- ❌ **Hardware failure** - Single point of failure
- ❌ **Zero-day exploits** - You must manually patch PostgreSQL

### Required Security Measures
1. **Strong passwords** - Use 20+ character passwords
2. **Regular backups** - Daily automated backups (see backup section)
3. **Firewall rules** - UFW configured to allow only specific IPs
4. **SSL certificates** - Use Let's Encrypt or self-signed certs
5. **Log monitoring** - Monitor Nginx and PostgreSQL logs
6. **Regular updates** - Keep PostgreSQL and Nginx updated

---

## Prerequisites

### 1. VM1 Setup
- Ubuntu 20.04+ or Debian 11+
- Docker and Docker Compose installed
- UFW firewall enabled
- Nginx installed (we'll install if needed)
- OpenSSL installed

### 2. Azure Web App Information
```bash
# Get your Azure Web App outbound IPs
az webapp show \
  --name fraud-detection-api-ammi-2025 \
  --resource-group fraud-detection-rg \
  --query "outboundIpAddresses" \
  --output tsv

# Example output:
# 20.123.45.67,20.123.45.68,20.123.45.69,20.123.45.70
```

### 3. Required Tools Check
```bash
# On VM1, verify installations
docker --version
nginx -v
openssl version
ufw status

# Install missing tools
sudo apt-get update
sudo apt-get install -y nginx openssl ufw

# Be sure
sudo ufw allow 22/tcp  # SSH (sinon vous serez bloqué!)

# 1. Obtain the outgoing IP address of your Azure Web App
az webapp show \
  --name fraud-detection-api-ammi-2025 \
  --resource-group fraud-detection-rg \
  --query outboundIpAddresses -o tsv

# 2. Allow ONLY Azure Web App (replace with your actual IPs)
sudo ufw allow from 20.123.45.67 to any port 5433 proto tcp
sudo ufw allow from 20.123.45.68 to any port 5433 proto tcp

sudo ufw enable

# Then verify
sudo ufw status
```

---

## Step-by-Step Setup

### Step 1: Configure PostgreSQL Container for Localhost Only

**Modify `docker-compose.vm1.yml`:**

```yaml
services:
  postgres:
    image: postgres:15-alpine
    container_name: fraud-postgres
    environment:
      POSTGRES_DB: ${POSTGRES_DB:-fraud_detection}
      POSTGRES_USER: ${POSTGRES_USER:-fraud_user}
      POSTGRES_PASSWORD: ${POSTGRES_PASSWORD}
    volumes:
      - postgres_data:/var/lib/postgresql/data
      - ./data/src/schema.sql:/docker-entrypoint-initdb.d/schema.sql:ro
    ports:
      - "127.0.0.1:5432:5432"  # ← Bind to localhost ONLY
    networks:
      - fraud-network
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U ${POSTGRES_USER:-fraud_user}"]
      interval: 10s
      timeout: 5s
      retries: 5
    restart: unless-stopped
    
    # IMPORTANT: Enable SSL in PostgreSQL
    command: >
      -c ssl=on
      -c ssl_cert_file=/etc/ssl/certs/ssl-cert-snakeoil.pem
      -c ssl_key_file=/etc/ssl/private/ssl-cert-snakeoil.key
```

**Restart PostgreSQL container:**
```bash
cd /path/to/fraud-detection-ml
docker-compose -f docker-compose.vm1.yml up -d postgres
```

### Step 2: Generate SSL Certificates

#### Option A: Self-Signed Certificate (For Testing/Academic)
```bash
# Create directory for certificates
sudo mkdir -p /etc/nginx/ssl

# Generate self-signed certificate (valid for 1 year)
sudo openssl req -x509 -nodes -days 365 -newkey rsa:2048 \
  -keyout /etc/nginx/ssl/postgres-proxy.key \
  -out /etc/nginx/ssl/postgres-proxy.crt \
  -subj "/C=CA/ST=Quebec/L=Montreal/O=FraudDetection/OU=Academic/CN=$(curl -s ifconfig.me)"

# Set proper permissions
sudo chmod 600 /etc/nginx/ssl/postgres-proxy.key
sudo chmod 644 /etc/nginx/ssl/postgres-proxy.crt

# Verify certificate
openssl x509 -in /etc/nginx/ssl/postgres-proxy.crt -text -noout
```

#### Option B: Let's Encrypt Certificate (Production-like)
```bash
# Install certbot
sudo apt-get install -y certbot

# Get certificate (requires domain name)
sudo certbot certonly --standalone -d your-vm1-domain.com

# Certificates will be in:
# /etc/letsencrypt/live/your-vm1-domain.com/fullchain.pem
# /etc/letsencrypt/live/your-vm1-domain.com/privkey.pem
```

### Step 3: Configure Nginx TCP Stream Proxy

**Create Nginx stream configuration:**
**Create streams-enabled directory:**
```bash
sudo mkdir -p /etc/nginx/streams-enabled
```

```bash
# Edit stream configuration file
sudo nano /etc/nginx/streams-enabled/postgres-proxy.conf
```

**Add this configuration:**

```nginx
# PostgreSQL TCP Stream Proxy with SSL
# File: /etc/nginx/streams-enabled/postgres-proxy.conf

# Define allowed IPs (Azure Web App outbound IPs)
geo $allowed_ip {
    default 0;
    
    # Azure Web App outbound IPs (replace with your actual IPs)
    20.123.45.67 1;
    20.123.45.68 1;
    20.123.45.69 1;
    20.123.45.70 1;
    
    # Add more IPs as needed
    # Get IPs with: az webapp show --query "outboundIpAddresses"
}

# PostgreSQL upstream (localhost only)
upstream postgres_backend {
    server 127.0.0.1:5432;
    
    # Connection limits
    keepalive 32;
}

# Rate limiting zone (10 connections per IP per minute)
limit_conn_zone $binary_remote_addr zone=postgres_conn_limit:10m;

# PostgreSQL proxy server
server {
    listen 5433 ssl;  # External port with SSL
    
    # SSL certificates
    ssl_certificate /etc/nginx/ssl/postgres-proxy.crt;
    ssl_certificate_key /etc/nginx/ssl/postgres-proxy.key;
    
    # SSL security settings
    ssl_protocols TLSv1.2 TLSv1.3;
    ssl_ciphers HIGH:!aNULL:!MD5;
    ssl_prefer_server_ciphers on;
    ssl_session_cache shared:SSL:10m;
    ssl_session_timeout 10m;
    
    # Proxy settings
    proxy_pass postgres_backend;
    proxy_timeout 600s;
    proxy_connect_timeout 10s;
    
    # Connection limits
    limit_conn postgres_conn_limit 10;
    
    # Access control (only allowed IPs)
    # Note: geo module check happens before this
    proxy_protocol off;
}
```

### Step 4: Enable Nginx Stream Module

**Edit main Nginx configuration:**

```bash
# Edit nginx.conf
sudo nano /etc/nginx/nginx.conf
```

**Add stream block at the END of the file (outside http block):**

```nginx
# At the end of /etc/nginx/nginx.conf
# (AFTER the closing brace of the http block)

stream {
    # Logging
    log_format postgres_proxy '$remote_addr [$time_local] '
                              '$protocol $status $bytes_sent $bytes_received '
                              '$session_time "$upstream_addr" '
                              '"$upstream_bytes_sent" "$upstream_bytes_received" "$upstream_connect_time"';
    
    access_log /var/log/nginx/postgres-stream-access.log postgres_proxy;
    error_log /var/log/nginx/postgres-stream-error.log warn;
    
    # Include stream configurations
    include /etc/nginx/streams-enabled/*.conf;
}
```


### Step 5: Test and Reload Nginx

```bash
# Test configuration
sudo nginx -t

# Expected output:
# nginx: the configuration file /etc/nginx/nginx.conf syntax is ok
# nginx: configuration file /etc/nginx/nginx.conf test is successful

# Reload Nginx
sudo systemctl reload nginx

# Check Nginx status
sudo systemctl status nginx

# Verify Nginx is listening on port 5433
sudo netstat -tlnp | grep 5433
# Expected: tcp  0  0.0.0.0:5433  0.0.0.0:*  LISTEN  12345/nginx
```

### Step 6: Configure UFW Firewall

```bash
# Get Azure Web App outbound IPs
WEBAPP_IPS="20.123.45.67 20.123.45.68 20.123.45.69 20.123.45.70"

# Allow PostgreSQL proxy port from Azure Web App IPs only
for IP in $WEBAPP_IPS; do
    sudo ufw allow from $IP to any port 5433 proto tcp comment "Azure Web App to PostgreSQL"
done

# Verify firewall rules
sudo ufw status numbered

# Reload firewall
sudo ufw reload
```

### Step 7: Update Azure Web App Configuration

```bash
# Set environment variables for Azure Web App
WEBAPP_NAME="fraud-detection-api-ammi-2025"
RESOURCE_GROUP="fraud-detection-rg"
VM1_PUBLIC_IP="<YOUR_VM1_PUBLIC_IP>"  # Get with: curl ifconfig.me (on VM1)

# Update Web App settings
az webapp config appsettings set \
  --name $WEBAPP_NAME \
  --resource-group $RESOURCE_GROUP \
  --settings \
    POSTGRES_HOST="$VM1_PUBLIC_IP" \
    POSTGRES_PORT="5433" \
    POSTGRES_DB="fraud_detection" \
    POSTGRES_USER="fraud_user" \
    POSTGRES_PASSWORD="YourSecurePassword123!" \
    DATABASE_URL="postgresql://fraud_user:YourSecurePassword123!@${VM1_PUBLIC_IP}:5433/fraud_detection?sslmode=require"

# Restart Web App
az webapp restart --name $WEBAPP_NAME --resource-group $RESOURCE_GROUP

# Monitor logs
az webapp log tail --name $WEBAPP_NAME --resource-group $RESOURCE_GROUP
```

### Step 8: Configure Automated Backups

**Create backup script:**

```bash
# Create backup directory
sudo mkdir -p /var/backups/postgres
sudo chown $USER:$USER /var/backups/postgres

# Create backup script
nano ~/backup-postgres.sh
```

**Add this script:**

```bash
#!/bin/bash
# PostgreSQL Backup Script
# File: ~/backup-postgres.sh

BACKUP_DIR="/var/backups/postgres"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
POSTGRES_CONTAINER="fraud-postgres"
POSTGRES_DB="fraud_detection"
POSTGRES_USER="fraud_user"

# Retention: Keep backups for 7 days
RETENTION_DAYS=7

echo "[$(date)] Starting PostgreSQL backup..."

# Create backup
docker exec $POSTGRES_CONTAINER pg_dump -U $POSTGRES_USER $POSTGRES_DB | \
  gzip > "$BACKUP_DIR/postgres_${TIMESTAMP}.sql.gz"

if [ $? -eq 0 ]; then
    echo "[$(date)] Backup successful: postgres_${TIMESTAMP}.sql.gz"
    
    # Delete old backups
    find $BACKUP_DIR -name "postgres_*.sql.gz" -mtime +$RETENTION_DAYS -delete
    echo "[$(date)] Old backups cleaned (kept last $RETENTION_DAYS days)"
else
    echo "[$(date)] Backup FAILED!"
    exit 1
fi

# Show disk usage
echo "[$(date)] Backup directory size:"
du -sh $BACKUP_DIR
```

**Make script executable and schedule:**

```bash
# Make executable
chmod +x ~/backup-postgres.sh

# Test backup
~/backup-postgres.sh

# Schedule daily backups at 2 AM
crontab -e

# Add this line:
0 2 * * * /home/your-username/backup-postgres.sh >> /var/log/postgres-backup.log 2>&1
```

---

## Testing & Validation

### Step 1: Test Connection from VM1 (localhost)

```bash
# Test local connection (should work)
psql -h 127.0.0.1 -p 5432 -U fraud_user -d fraud_detection

# Expected: PostgreSQL prompt
# fraud_detection=> 
```

### Step 2: Test Nginx Proxy from VM1

```bash
# Install PostgreSQL client if needed
sudo apt-get install -y postgresql-client

# Test through Nginx proxy (with SSL)
PGSSLMODE=require psql -h localhost -p 5433 -U fraud_user -d fraud_detection

# Expected: PostgreSQL prompt
# fraud_detection=>

# If successful, exit
\q
```

### Step 3: Test from Another Machine (Simulate Azure Web App)

```bash
# From your local machine (NOT VM1)
# Replace VM1_IP with your actual VM1 public IP

psql "host=<VM1_IP> port=5433 dbname=fraud_detection user=fraud_user sslmode=require password=YourPassword"

# If connection is refused:
# - Check UFW: sudo ufw status
# - Check Nginx logs: sudo tail -f /var/log/nginx/postgres-stream-error.log
# - Verify your IP is whitelisted in /etc/nginx/streams-enabled/postgres-proxy.conf
```

### Step 4: Test from Azure Web App

```bash
# Use Azure Cloud Shell or SSH into Web App
az webapp ssh --name fraud-detection-api-ammi-2025 --resource-group fraud-detection-rg

# Inside Web App shell, test connection
python3 << EOF
import psycopg2
conn = psycopg2.connect(
    host="<VM1_IP>",
    port=5433,
    database="fraud_detection",
    user="fraud_user",
    password="YourPassword",
    sslmode="require"
)
print("✅ Connection successful!")
conn.close()
EOF
```

### Step 5: Validate API Health Endpoint

```bash
# Test API database health
curl https://fraud-detection-api-ammi-2025.azurewebsites.net/health/db

# Expected response:
# {
#   "status": "healthy",
#   "database": "connected",
#   "timestamp": "2025-11-06T12:34:56"
# }
```

---

## Monitoring

### 1. Monitor Nginx Logs

```bash
# Watch access logs in real-time
sudo tail -f /var/log/nginx/postgres-stream-access.log

# Example log entry:
# 20.123.45.67 [06/Nov/2025:12:34:56 +0000] TCP 200 1234 5678 45.678 "127.0.0.1:5432" "1234" "5678" "0.023"

# Watch error logs
sudo tail -f /var/log/nginx/postgres-stream-error.log
```

### 2. Monitor PostgreSQL Logs

```bash
# View PostgreSQL container logs
docker logs fraud-postgres --tail 50 --follow

# Look for connection attempts:
# LOG:  connection received: host=172.18.0.5 port=54321
# LOG:  connection authorized: user=fraud_user database=fraud_detection SSL enabled
```

### 3. Monitor Active Connections

```bash
# Check active PostgreSQL connections
docker exec fraud-postgres psql -U fraud_user -d fraud_detection -c "
SELECT 
    pid,
    usename,
    application_name,
    client_addr,
    state,
    query_start
FROM pg_stat_activity
WHERE datname = 'fraud_detection';
"
```

### 4. Set Up Connection Alerts

**Create monitoring script:**

```bash
nano ~/monitor-postgres-connections.sh
```

```bash
#!/bin/bash
# Monitor PostgreSQL connections
# File: ~/monitor-postgres-connections.sh

THRESHOLD=50  # Alert if connections exceed this
EMAIL="admin@example.com"

CONNECTIONS=$(docker exec fraud-postgres psql -U fraud_user -d fraud_detection -t -c "SELECT count(*) FROM pg_stat_activity WHERE datname = 'fraud_detection';")

if [ "$CONNECTIONS" -gt "$THRESHOLD" ]; then
    echo "⚠️ HIGH CONNECTION COUNT: $CONNECTIONS connections" | \
        mail -s "PostgreSQL Alert: High Connection Count" $EMAIL
fi
```

**Schedule check every 5 minutes:**

```bash
chmod +x ~/monitor-postgres-connections.sh

crontab -e
# Add:
*/5 * * * * /home/your-username/monitor-postgres-connections.sh
```

### 5. Monitor Disk Space

```bash
# Check PostgreSQL data volume size
docker exec fraud-postgres du -sh /var/lib/postgresql/data

# Check backup directory size
du -sh /var/backups/postgres

# Set up disk space alert
nano ~/check-disk-space.sh
```

```bash
#!/bin/bash
# Check disk space
THRESHOLD=80  # Alert at 80% full

USAGE=$(df -h / | awk 'NR==2 {print $5}' | sed 's/%//')

if [ "$USAGE" -gt "$THRESHOLD" ]; then
    echo "⚠️ DISK SPACE WARNING: ${USAGE}% used" | \
        mail -s "VM1 Disk Space Alert" admin@example.com
fi
```

---

## Troubleshooting

### Issue 1: Connection Timeout

**Symptom:** Azure Web App cannot connect to PostgreSQL

**Solutions:**

```bash
# A. Verify Nginx is listening on port 5433
sudo netstat -tlnp | grep 5433
# Should show: tcp  0.0.0.0:5433  LISTEN  nginx

# B. Check UFW firewall rules
sudo ufw status numbered
# Verify Azure Web App IPs are allowed on port 5433

# C. Test from VM1 localhost
psql "host=localhost port=5433 dbname=fraud_detection user=fraud_user sslmode=require"

# D. Check Nginx error logs
sudo tail -50 /var/log/nginx/postgres-stream-error.log

# E. Verify PostgreSQL container is running
docker ps | grep fraud-postgres

# F. Check if Azure Web App IP changed
az webapp show --name fraud-detection-api-ammi-2025 \
  --resource-group fraud-detection-rg \
  --query "outboundIpAddresses"
```

### Issue 2: SSL/TLS Errors

**Symptom:** `SSL error: certificate verify failed`

**Solutions:**

```bash
# A. Check SSL certificate validity
openssl x509 -in /etc/nginx/ssl/postgres-proxy.crt -text -noout | grep "Not After"

# B. Test SSL connection
openssl s_client -connect localhost:5433 -showcerts

# C. For self-signed certificates, use sslmode=require (not verify-full)
# In Azure Web App:
DATABASE_URL="postgresql://...?sslmode=require"  # Not verify-full

# D. Regenerate certificate if expired
sudo openssl req -x509 -nodes -days 365 -newkey rsa:2048 \
  -keyout /etc/nginx/ssl/postgres-proxy.key \
  -out /etc/nginx/ssl/postgres-proxy.crt \
  -subj "/C=CA/ST=Quebec/L=Montreal/O=FraudDetection/CN=$(curl -s ifconfig.me)"

sudo systemctl reload nginx
```

### Issue 3: Too Many Connections

**Symptom:** `FATAL: sorry, too many clients already`

**Solutions:**

```bash
# A. Check current connection count
docker exec fraud-postgres psql -U fraud_user -d fraud_detection -c "
SELECT count(*) FROM pg_stat_activity;
"

# B. Increase max_connections in PostgreSQL
# Edit docker-compose.vm1.yml:
services:
  postgres:
    command: >
      -c max_connections=100
      -c ssl=on
      ...

# Restart PostgreSQL
docker-compose -f docker-compose.vm1.yml up -d postgres

# C. Implement connection pooling (recommended)
# Use PgBouncer (see PgBouncer section below)
```

### Issue 4: Slow Performance

**Symptom:** Queries are slow through Nginx

**Solutions:**

```bash
# A. Check Nginx connection timeouts
# In /etc/nginx/streams-enabled/postgres-proxy.conf:
proxy_timeout 600s;  # Increase if needed

# B. Enable connection keepalive
upstream postgres_backend {
    server 127.0.0.1:5432;
    keepalive 32;  # Increase if needed
}

# C. Check PostgreSQL query performance
docker exec fraud-postgres psql -U fraud_user -d fraud_detection -c "
SELECT query, calls, total_exec_time, mean_exec_time 
FROM pg_stat_statements 
ORDER BY total_exec_time DESC 
LIMIT 10;
"

# D. Monitor VM1 resource usage
top
htop
docker stats
```

### Issue 5: IP Address Changed

**Symptom:** Azure Web App IP changed, connection refused

**Solutions:**

```bash
# A. Get new Azure Web App IPs
az webapp show --name fraud-detection-api-ammi-2025 \
  --resource-group fraud-detection-rg \
  --query "outboundIpAddresses" --output tsv

# B. Update Nginx geo block
sudo nano /etc/nginx/streams-enabled/postgres-proxy.conf
# Add new IPs to geo $allowed_ip block

# C. Update UFW firewall rules
NEW_IP="20.123.45.71"  # Replace with actual new IP
sudo ufw allow from $NEW_IP to any port 5433 proto tcp comment "Azure Web App"
sudo ufw reload

# D. Reload Nginx
sudo nginx -t && sudo systemctl reload nginx
```

---

## Advanced: Connection Pooling with PgBouncer

For better performance with multiple connections, add PgBouncer:

**Add to `docker-compose.vm1.yml`:**

```yaml
services:
  pgbouncer:
    image: edoburu/pgbouncer:latest
    container_name: fraud-pgbouncer
    environment:
      - DB_HOST=postgres
      - DB_PORT=5432
      - DB_USER=fraud_user
      - DB_PASSWORD=${POSTGRES_PASSWORD}
      - DB_NAME=fraud_detection
      - POOL_MODE=transaction
      - MAX_CLIENT_CONN=100
      - DEFAULT_POOL_SIZE=20
      - MIN_POOL_SIZE=5
      - RESERVE_POOL_SIZE=5
    ports:
      - "127.0.0.1:6432:6432"
    networks:
      - fraud-network
    depends_on:
      postgres:
        condition: service_healthy
    restart: unless-stopped
```

**Update Nginx to proxy to PgBouncer:**

```nginx
upstream postgres_backend {
    server 127.0.0.1:6432;  # PgBouncer instead of direct PostgreSQL
    keepalive 32;
}
```

**Restart services:**

```bash
docker-compose -f docker-compose.vm1.yml up -d pgbouncer
sudo systemctl reload nginx
```

---

## Cost Comparison

### Option 1: Nginx Proxy (This Guide)
**Monthly Costs:**
- VM1 resources: $0 (already running)
- Nginx: Free (open source)
- SSL certificates: $0 (self-signed or Let's Encrypt)
- **Total: $0/month additional cost**

**Time Investment:**
- Initial setup: 2-3 hours
- Maintenance: 1-2 hours/week (backups, monitoring, updates)

### Option 2: Azure Database for PostgreSQL
**Monthly Costs:**
- Burstable tier (B1ms): ~$20-30/month
- General Purpose (D2s_v3): ~$130-160/month
- **Total: $20-160/month**

**Time Investment:**
- Initial setup: 1-2 hours
- Maintenance: ~0 hours (fully managed)

### Recommendation by Use Case

| Use Case | Recommended Option | Reason |
|----------|-------------------|--------|
| Academic project | Nginx Proxy | Cost-effective, learning opportunity |
| Class assignment | Nginx Proxy | Sufficient for demonstrations |
| Startup MVP | Nginx Proxy | Save costs during validation phase |
| Production app | Azure Database | Reliability, backups, compliance |
| Enterprise | Azure Database | SLA, support, security features |

---

## Security Checklist

Before going live, verify:

- [ ] PostgreSQL container bound to localhost only (`127.0.0.1:5432`)
- [ ] Nginx SSL certificates configured and valid
- [ ] UFW firewall rules configured (only Azure Web App IPs)
- [ ] Strong PostgreSQL password (20+ characters)
- [ ] Automated daily backups configured and tested
- [ ] Backup restoration tested successfully
- [ ] Nginx access logs monitored
- [ ] PostgreSQL logs monitored
- [ ] Connection monitoring script running
- [ ] Disk space monitoring configured
- [ ] PostgreSQL and Nginx up to date
- [ ] Rate limiting configured in Nginx
- [ ] Connection pooling implemented (PgBouncer recommended)
- [ ] Emergency rollback plan documented

---

## Backup & Restore Procedures

### Manual Backup

```bash
# Create backup
docker exec fraud-postgres pg_dump -U fraud_user fraud_detection | \
  gzip > backup_$(date +%Y%m%d_%H%M%S).sql.gz

# Download backup to local machine
scp -i Fraud.VM1_key.pem user@VM1_IP:~/backup_*.sql.gz ./
```

### Restore from Backup

```bash
# Stop services using database
docker-compose -f docker-compose.vm1.yml stop data drift training mlflow airflow-webserver airflow-scheduler airflow-worker

# Restore database
gunzip -c backup_20251106_120000.sql.gz | \
  docker exec -i fraud-postgres psql -U fraud_user fraud_detection

# Restart services
docker-compose -f docker-compose.vm1.yml up -d
```

### Backup to Azure Blob Storage (Optional)

```bash
# Install Azure CLI
curl -sL https://aka.ms/InstallAzureCLIDeb | sudo bash

# Login
az login

# Upload backup
az storage blob upload \
  --account-name fraudbackups \
  --container-name postgres-backups \
  --name backup_$(date +%Y%m%d).sql.gz \
  --file backup_*.sql.gz
```

---

## Migration Path to Azure Database

When you're ready to migrate to Azure Database for PostgreSQL:

1. **Export data from current setup:**
   ```bash
   docker exec fraud-postgres pg_dump -U fraud_user fraud_detection > migration.sql
   ```

2. **Follow Azure Database Migration Guide:**
   - See `Guide/AZURE_DATABASE_MIGRATION.md`
   - Use exported data to initialize Azure Database

3. **Update services to use Azure Database:**
   - Change `docker-compose.vm1.yml` to `docker-compose.vm1.azure-db.yml`
   - Update Azure Web App connection strings

4. **Remove Nginx PostgreSQL proxy:**
   ```bash
   sudo rm /etc/nginx/streams-enabled/postgres-proxy.conf
   sudo systemctl reload nginx
   sudo ufw delete allow 5433
   ```

---

## Conclusion

This guide provides a **cost-effective solution for academic projects** while maintaining reasonable security. However, remember:

- ⚠️ **This is NOT a production-grade setup**
- ⏰ **Requires manual management and monitoring**
- 🔄 **Plan to migrate to Azure Database for production**

For production environments, always use **Azure Database for PostgreSQL** (see `AZURE_DATABASE_MIGRATION.md`).

---

## Additional Resources

- **PostgreSQL SSL Documentation**: https://www.postgresql.org/docs/15/ssl-tcp.html
- **Nginx Stream Module**: https://nginx.org/en/docs/stream/ngx_stream_core_module.html
- **UFW Firewall Guide**: https://help.ubuntu.com/community/UFW
- **PgBouncer Documentation**: https://www.pgbouncer.org/
- **PostgreSQL Performance Tuning**: https://wiki.postgresql.org/wiki/Performance_Optimization

---

**Document Version:** 1.0  
**Last Updated:** 2025-11-06  
**Maintained By:** Fraud Detection Team  
**Target Audience:** Academic/Development Environments
