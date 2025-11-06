# 🔄 Producer-Consumer Architecture: Kafka Streaming Setup

## 📋 Overview

This architecture implements a **producer-consumer** system with Kafka:

- **PRODUCER**: `00_transaction_producer.py` → Generates transactions and sends them to Kafka
- **CONSUMER**: `00_realtime_streaming.py` → Consumes from Kafka, predicts, and saves

---

## 🏗️ Complete Architecture
```
┌─────────────────────────────────────────────────────────────────────┐
│                    AIRFLOW DAG 1 (PRODUCER)                         │
│                  00_transaction_producer.py                         │
│                                                                     │
│  ┌──────────────────────────────────────────────────────────────┐ │
│  │ Schedule: Every 5 seconds                                     │ │
│  │ Generates: 50 transactions per run                           │ │
│  │ Fraud Rate: 5%                                               │ │
│  └──────────────────────────────────────────────────────────────┘ │
│                            ↓                                        │
│  ┌──────────────────────────────────────────────────────────────┐ │
│  │ DockerOperator:                                              │ │
│  │ python -m src.ingestion.transaction_simulator \              │ │
│  │   --mode batch \                                             │ │
│  │   --count 50 \                                               │ │
│  │   --fraud-rate 0.05                                          │ │
│  └──────────────────────────────────────────────────────────────┘ │
│                            ↓                                        │
└────────────────────────────┼────────────────────────────────────────┘
                             │
                             ↓
                    ┌────────────────────┐
                    │   KAFKA TOPIC      │
                    │  fraud-detection-  │
                    │   transactions     │
                    │                    │
                    │ 📊 Queue Size: 1K  │
                    │ ⏱️ Retention: 7d   │
                    └────────────────────┘
                             ↓
┌────────────────────────────┼────────────────────────────────────────┐
│                    AIRFLOW DAG 2 (CONSUMER)                         │
│                  00_realtime_streaming.py                           │
│                                                                     │
│  ┌──────────────────────────────────────────────────────────────┐ │
│  │ Schedule: Every 10 seconds                                    │ │
│  │ Consumes: Up to 100 transactions per run                     │ │
│  │ Processing: Clean → Predict → Save                           │ │
│  └──────────────────────────────────────────────────────────────┘ │
│                            ↓                                        │
│  ┌──────────────────────────────────────────────────────────────┐ │
│  │ DockerOperator:                                              │ │
│  │ python -m src.pipelines.realtime_pipeline \                  │ │
│  │   --mode batch \                                             │ │
│  │   --count 100                                                │ │
│  └──────────────────────────────────────────────────────────────┘ │
│                            ↓                                        │
│  ┌──────────────────────────────────────────────────────────────┐ │
│  │ 1. 📥 Consume from Kafka (KafkaConsumer)                     │ │
│  │ 2. 🧹 Clean & Preprocess (DataCleaner)                       │ │
│  │ 3. 🤖 Predict via API (POST /api/v1/batch-predict)           │ │
│  │ 4. 💾 Save to PostgreSQL (predictions table)                 │ │
│  │ 5. 📤 Send to Web App (fraud alerts)                         │ │
│  └──────────────────────────────────────────────────────────────┘ │
│                            ↓                                        │
└────────────────────────────┼────────────────────────────────────────┘
                             ↓
                    ┌────────────────────┐
                    │   PostgreSQL DB    │
                    │  predictions table │
                    │                    │
                    │ 📊 Predictions     │
                    │ ⚠️  Fraud Alerts    │
                    └────────────────────┘
```

---

## 🚀 Full Startup

### Step 1: Verify that Kafka is running
```bash
# Verify that Kafka is running
docker ps | grep kafka

# Should display:
# fraud-kafka
# fraud-zookeeper
```

### Step 2: Enable the 2 DAGs in Airflow UI
```bash
open http://localhost:8080

# Credentials
Username: admin
Password: admin
```

**In the Airflow interface:**

1. ✅ Enable `00_transaction_producer` (toggle ON)
2. ✅ Enable `00_realtime_streaming` (toggle ON)

### Step 3: Observe the DAGs in action
```bash
# Terminal 1: Producer logs
docker logs -f fraud-airflow-worker --tail=50 | grep "transaction_producer"

# Terminal 2: Consumer logs
docker logs -f fraud-airflow-worker --tail=50 | grep "realtime_streaming"

# Terminal 3: Kafka Consumer (view messages)

docker exec -it fraud-kafka kafka-console-consumer \
  --bootstrap-server localhost:9092 \
  --topic fraud-detection-transactions \
  --from-beginning
```

---

## ⏱️ Timing & Performance

| Component | Schedule | Transactions | Latency |
|-----------|----------|--------------|---------|
| **Producer** | Every 5s | 50/run | ~2s |
| **Consumer** | Every 10s | 100/run | ~5s |
| **Kafka Queue** | N/A | Buffer 1000+ | ~1ms |

**Total Throughput**:
- Producer: 50 txn × 12 runs/min = **600 txn/min** (10 txn/sec)
- Consumer: 100 txn × 6 runs/min = **600 txn/min** (10 txn/sec)
- **Perfect Balance** ✅

---

## 📊 Real-time Monitoring

### 1. Kafka Topic Lag (check if the consumer is keeping up)
```bash
docker exec -it fraud-kafka kafka-consumer-groups \
  --bootstrap-server localhost:9092 \
  --describe \
  --group fraud-detection-batch
```

**Interpretation**:
- `LAG = 0` → Consumer is keeping up perfectly ✅
- `LAG > 0` → Consumer is falling behind ⚠️
- `LAG > 500` → Consumer is overloaded 🚨

### 2. PostgreSQL Predictions Count
```bash
docker exec -it fraud-postgres psql -U postgres -d fraud_detection -c \
  "SELECT COUNT(*), 
          SUM(CASE WHEN predicted_fraud = 1 THEN 1 ELSE 0 END) as fraud_count,
          MAX(created_at) as last_prediction
   FROM predictions;"
```

### 3. Airflow DAG Metrics
```bash
# Airflow API
curl -s http://localhost:8080/api/v1/dags/00_transaction_producer/dagRuns \
  -u admin:admin123 | jq '.dag_runs[0] | {state, execution_date}'

curl -s http://localhost:8080/api/v1/dags/00_realtime_streaming/dagRuns \
  -u admin:admin123 | jq '.dag_runs[0] | {state, execution_date}'
```

---

## 🔧 Advanced Configuration

### Adjust Production Throughput

Edit `airflow/dags/00_transaction_producer.py`:
```python
# Lines 31-33
TRANSACTIONS_PER_RUN = 100  # 100 instead of 50 → 2x faster
FRAUD_RATE = 0.10  # 10% fraud instead of 5% → 2x more fraud

# Schedule (line 74)
schedule_interval="*/2 * * * * *"  # Every 2 seconds instead of 5
```

### Adjust Consumption

Edit `airflow/dags/00_realtime_streaming.py`:
```python
# Line 27
STREAMING_BATCH_SIZE = 200  # 200 instead of 100 → 2x faster

# Schedule (line 70)
schedule_interval="*/5 * * * * *"  # Every 5 seconds instead of 10
```

---

## 🐛 Troubleshooting

### Issue 1: "Kafka topic not found"
```bash
# Create the topic manually
docker exec -it fraud-kafka kafka-topics --create \
  --bootstrap-server localhost:9092 \
  --topic fraud-detection-transactions \
  --partitions 3 \
  --replication-factor 1
```

### Issue 2: "Consumer lag increasing"

**Cause**: Producer too fast, consumer too slow

**Solution**:
```python
# Option 1: Slow down the producer (schedule_interval="*/10 * * * * *")
# Option 2: Speed up the consumer (STREAMING_BATCH_SIZE = 200)
# Option 3: Parallelize the consumer (max_active_runs=3)
```

### Issue 3: "JWT authentication failed"
```bash
# Check API credentials
docker exec -it fraud-data env | grep API_

# Should display:
# API_URL=http://api:8000
# API_USERNAME=admin
# API_PASSWORD=admin123
```

### Issue 4: "Database connection timeout"
```bash
# Check PostgreSQL
docker exec -it fraud-postgres pg_isready

# Check tables
docker exec -it fraud-postgres psql -U postgres -d fraud_detection -c "\dt"
```

---

## 📈 Scaling Strategies

### Scenario 1: High Volume (1000 txn/sec)
```yaml
# docker-compose.local.yml
kafka:
  environment:
    KAFKA_NUM_PARTITIONS: 10  # 10 partitions instead of 3
```
```python
# 00_transaction_producer.py
max_active_runs=5  # 5 parallel producers

# 00_realtime_streaming.py
max_active_runs=10  # 10 parallel consumers
```

### Scenario 2: Low Latency (< 1 sec)
```python
# 00_transaction_producer.py
schedule_interval="* * * * * *"  # Every 1 second

# 00_realtime_streaming.py
schedule_interval="* * * * * *"  # Every 1 second
STREAMING_BATCH_SIZE = 20  # Small batches for low latency
```

### Scenario 3: Cost Optimization (reduce compute)
```python
# 00_transaction_producer.py
schedule_interval="0 * * * *"  # Every hour
TRANSACTIONS_PER_RUN = 10000  # Large batches

# 00_realtime_streaming.py
schedule_interval="*/5 * * * *"  # Every 5 minutes
STREAMING_BATCH_SIZE = 5000
```

---

## ✅ Validation Checklist

- [ ] Kafka & Zookeeper are running (`docker ps`)
- [ ] Topic `fraud-detection-transactions` exists
- [ ] DAG `00_transaction_producer` is enabled (Airflow UI)
- [ ] DAG `00_realtime_streaming` is enabled (Airflow UI)
- [ ] Both DAGs are running without errors (green logs)
- [ ] Kafka consumer lag = 0
- [ ] PostgreSQL `predictions` table is being populated
- [ ] API `/health` returns 200
- [ ] Grafana dashboard displays metrics

---

## 📚 References

- **TransactionSimulator Code**: `data/src/ingestion/transaction_simulator.py`
- **RealtimePipeline Code**: `data/src/pipelines/realtime_pipeline.py`
- **Kafka Configuration**: `docker-compose.local.yml` (lines 50-120)
- **Airflow Configuration**: `airflow/config/constants.py`

---

## 🎯 Next Steps

1. **Testing**: Manually trigger DAGs to verify functionality
2. **Monitoring**: Configure Grafana dashboards for visualization
3. **Alerting**: Set up AlertManager for detected fraud alerts
4. **Optimization**: Tune schedules according to actual volume

---

## 👨🏾‍💻 Contributors

Fraud Detection Team

1. Joshua Juste NIKIEMA
2. Olalekan Taofeek OLALUWOYE
3. Soulaimana Toihir DJALOUD