# Fraud Detection System - User Guide: Prediction & Investigation Flow

**Document Version:** 1.0  
**Last Updated:** November 7, 2025  
**Target Audience:** Fraud Analysts, System Users, Stakeholders

---

## Table of Contents

1. [System Overview](#system-overview)
2. [How Transactions Flow Through the System](#how-transactions-flow-through-the-system)
3. [Real-Time Prediction Process](#real-time-prediction-process)
4. [Using the Web Dashboard](#using-the-web-dashboard)
5. [Investigating Fraud Alerts](#investigating-fraud-alerts)
6. [Understanding SHAP Explanations](#understanding-shap-explanations)
7. [Providing Feedback](#providing-feedback)
8. [Monitoring System Performance](#monitoring-system-performance)
9. [FAQ](#faq)

---

## System Overview

### What Does This System Do?

The Fraud Detection System is an **automated machine learning platform** that:

1. **Monitors** credit card transactions in real-time
2. **Predicts** which transactions are likely fraudulent
3. **Alerts** analysts about high-risk transactions
4. **Explains** why a transaction was flagged using AI explainability
5. **Learns** from analyst feedback to improve over time

### Key Components (User Perspective)

```
┌──────────────────┐         ┌──────────────────┐         ┌──────────────────┐
│  Transaction     │────────▶│  ML Prediction   │────────▶│  Web Dashboard   │
│  Stream (Kafka)  │         │  Engine (API)    │         │  (Investigation) │
└──────────────────┘         └──────────────────┘         └──────────────────┘
        ↑                             ↑                            │
        │                             │                            │
        │                    ┌────────┴────────┐                  │
        │                    │  Ensemble Model │                  ↓
        │                    │  - XGBoost      │         ┌────────────────┐
        │                    │  - Neural Net   │         │  Analyst       │
        │                    │  - Iso Forest   │         │  Feedback      │
        │                    │  - Random Forest│         └────────────────┘
        │                    └─────────────────┘                  │
        │                                                          │
        └──────────────────────────────────────────────────────────┘
                          Model Retraining (Airflow)
```

---

## How Transactions Flow Through the System

### Step-by-Step Process

#### 1. **Transaction Generation** (Simulated)
   - **What happens:** A transaction simulator generates realistic credit card transactions
   - **Frequency:** ~50 transactions every 5 seconds
   - **Data includes:** Transaction amount, PCA-transformed features (V1-V28), timestamp
   - **User action:** None (automatic)

#### 2. **Transaction Ingestion** (Kafka)
   - **What happens:** Transactions are sent to a Kafka message queue
   - **Purpose:** Buffer transactions for reliable processing
   - **Topic:** `fraud-detection-transactions`
   - **User action:** None (automatic)

#### 3. **Real-Time Consumption** (Data Pipeline)
   - **What happens:** Airflow DAG consumes transactions from Kafka
   - **Frequency:** Every 10 seconds (processes ~100 transactions per batch)
   - **Processing:** Data cleaning, validation, feature extraction
   - **User action:** None (automatic)

#### 4. **Fraud Prediction** (ML API)
   - **What happens:** Cleaned transactions are sent to the prediction API
   - **Models used:** Ensemble of 4 ML models (weighted voting)
   - **Output:** 
     - `fraud_score`: Probability (0.0 to 1.0)
     - `is_fraud_predicted`: Boolean (True/False)
     - `risk_level`: "LOW", "MEDIUM", "HIGH", "CRITICAL"
   - **Performance:** ~50ms per prediction
   - **User action:** None (automatic)

#### 5. **Database Storage** (PostgreSQL)
   - **What happens:** Predictions are saved to the database
   - **Tables:**
     - `transactions`: Original transaction data
     - `predictions`: ML predictions with scores
   - **User action:** None (automatic)

#### 6. **Fraud Alerts** (WebSocket)
   - **What happens:** High-risk transactions are sent to the web dashboard
   - **Threshold:** `fraud_score >= 0.7` (configurable)
   - **Delivery:** Real-time via WebSocket connection
   - **User action:** **Analysts see alerts in dashboard** ✅

#### 7. **Investigation** (Web Dashboard)
   - **What happens:** Analysts review flagged transactions
   - **Tools available:**
     - Transaction details
     - SHAP explanations (feature importance)
     - Historical patterns
   - **User action:** **Analysts investigate and provide feedback** ✅

---

## Real-Time Prediction Process

### How the ML Model Makes Predictions

#### Input: Transaction Features

A transaction has **30 features**:
- `Time`: Seconds since first transaction (e.g., 12345.67)
- `V1` to `V28`: PCA-transformed features (anonymized for privacy)
  - These represent patterns like: transaction velocity, merchant type, location, etc.
- `Amount`: Transaction amount in dollars (e.g., $149.99)

**Example:**
```json
{
  "transaction_id": "TXN-789ABC",
  "Time": 12345.67,
  "V1": -0.234,
  "V2": 1.456,
  "V3": -0.789,
  ...
  "V28": 0.123,
  "amount": 149.99
}
```

#### Processing: Ensemble Model

The system uses **4 machine learning models** working together:

1. **XGBoost** (Gradient Boosting)
   - Weight: 40%
   - Best for: Detecting complex fraud patterns
   - Fast and accurate

2. **Neural Network** (Deep Learning)
   - Weight: 30%
   - Best for: Learning non-linear relationships
   - Handles complex interactions

3. **Random Forest** (Tree Ensemble)
   - Weight: 20%
   - Best for: Robust to outliers
   - Good generalization

4. **Isolation Forest** (Anomaly Detection)
   - Weight: 10%
   - Best for: Detecting unusual transactions
   - Identifies outliers

**Ensemble Voting:**
```
Final Score = (XGBoost × 0.4) + (Neural Net × 0.3) + (Random Forest × 0.2) + (Iso Forest × 0.1)
```

#### Output: Prediction Result

```json
{
  "transaction_id": "TXN-789ABC",
  "prediction": 1,               // 0=legitimate, 1=fraud
  "fraud_score": 0.87,           // Probability (87% fraud)
  "confidence": 0.92,            // Model confidence
  "risk_level": "HIGH",          // LOW/MEDIUM/HIGH/CRITICAL
  "model_version": "v2.3.1",
  "prediction_time": "2025-11-07T14:30:45Z",
  "processing_latency_ms": 52
}
```

### Risk Level Classification

| Fraud Score | Risk Level | Action |
|-------------|------------|--------|
| 0.00 - 0.30 | **LOW** | Auto-approve (no alert) |
| 0.30 - 0.50 | **MEDIUM** | Log for review |
| 0.50 - 0.70 | **HIGH** | Alert analyst |
| 0.70 - 1.00 | **CRITICAL** | **Immediate investigation** 🚨 |

---

## Using the Web Dashboard

### 1. Dashboard Overview

**URL:** `https://fraud-detection-web.vercel.app` (or your deployment URL)

**Login Credentials:**
- **Analyst:** `analyst` / `password`
- **Admin:** `admin` / `admin123`

### 2. Main Dashboard Page

**What you see:**

```
┌─────────────────────────────────────────────────────────────┐
│  FRAUD DETECTION DASHBOARD                      [Refresh ↻] │
├─────────────────────────────────────────────────────────────┤
│                                                               │
│  📊 Metrics Cards                                            │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐         │
│  │ Total       │  │ Fraud       │  │ Detection   │         │
│  │ Transactions│  │ Detected    │  │ Rate        │         │
│  │ 1,234,567   │  │ 4,521      │  │ 0.37%       │         │
│  └─────────────┘  └─────────────┘  └─────────────┘         │
│                                                               │
│  🚨 Real-Time Fraud Alerts                                   │
│  ┌─────────────────────────────────────────────────┐        │
│  │ TXN-789ABC | $149.99 | Score: 87% | 2 mins ago │        │
│  │ TXN-456DEF | $599.00 | Score: 92% | 5 mins ago │        │
│  │ TXN-123GHI | $299.50 | Score: 78% | 8 mins ago │        │
│  └─────────────────────────────────────────────────┘        │
│                                                               │
│  📈 Fraud Timeline (Last 24 Hours)                           │
│  [Line Chart showing fraud rate over time]                   │
│                                                               │
└─────────────────────────────────────────────────────────────┘
```

**Features:**
- **Auto-refresh:** Dashboard updates every 30 seconds
- **Real-time alerts:** New fraud alerts appear automatically via WebSocket
- **Metrics:** Live statistics on detection performance

### 3. Transactions Page

**Navigation:** Click "Transactions" in sidebar

**Features:**
- **Search:** Find transactions by ID, customer, amount
- **Filter:** Filter by fraud status, date range, score
- **Pagination:** View 100 transactions per page
- **View Details:** Click on any transaction to see full details

**Example:**
```
┌─────────────────────────────────────────────────────────────┐
│  TRANSACTIONS                             🔍 [Search...    ] │
├─────────────────────────────────────────────────────────────┤
│                                                               │
│  Filters: [All] [Fraud Only] [Legitimate] [Date Range]      │
│                                                               │
│  ID            Amount    Score   Status    Time             │
│  ───────────────────────────────────────────────────────── │
│  TXN-789ABC   $149.99    87%    🔴 Fraud   14:30:45        │
│  TXN-456DEF   $599.00    92%    🔴 Fraud   14:25:12        │
│  TXN-123GHI    $45.00     5%    🟢 OK      14:20:33        │
│  ...                                                         │
│                                                               │
│  [Previous]  Page 1 of 123  [Next]                          │
└─────────────────────────────────────────────────────────────┘
```

---

## Investigating Fraud Alerts

### Step 1: Click on a Transaction

When you click on a fraud alert or transaction, a **Transaction Modal** opens:

```
┌─────────────────────────────────────────────────────────────┐
│  TRANSACTION DETAILS                              [Close ✕] │
├─────────────────────────────────────────────────────────────┤
│                                                               │
│  Transaction ID: TXN-789ABC                                  │
│  Amount: $149.99                                             │
│  Time: 2025-11-07 14:30:45 UTC                              │
│                                                               │
│  🎯 PREDICTION                                               │
│  ┌────────────────────────────────────────────┐             │
│  │ Fraud Score:    87%  (HIGH RISK) 🔴       │             │
│  │ Confidence:     92%                        │             │
│  │ Model Version:  v2.3.1 (Ensemble)         │             │
│  │ Processing Time: 52ms                      │             │
│  └────────────────────────────────────────────┘             │
│                                                               │
│  📊 FEATURES (First 10)                                      │
│  V1:  -0.234  │  V2:   1.456  │  V3:  -0.789               │
│  V4:   0.567  │  V5:  -1.234  │  V6:   0.890               │
│  ...                                                         │
│  Amount: $149.99                                             │
│                                                               │
│  [View SHAP Explanation]  [Mark as Reviewed]                │
│                                                               │
└─────────────────────────────────────────────────────────────┘
```

### Step 2: View SHAP Explanation

Click **"View SHAP Explanation"** to understand **why** the model flagged this transaction.

---

## Understanding SHAP Explanations

### What is SHAP?

**SHAP** (SHapley Additive exPlanations) explains **which features** made the model predict fraud.

### SHAP Explanation Modal

```
┌─────────────────────────────────────────────────────────────┐
│  🧠 SHAP EXPLANATION - TXN-789ABC                 [Close ✕] │
├─────────────────────────────────────────────────────────────┤
│                                                               │
│  Model: Ensemble (XGBoost + Neural Network + RF + IF)       │
│                                                               │
│  📊 PREDICTION SUMMARY                                       │
│  ┌────────────────────────────────────────────┐             │
│  │ Fraud Score:    87%                        │             │
│  │ Base Value:     12% (population average)   │             │
│  │ Prediction:     +75% increase from base    │             │
│  └────────────────────────────────────────────┘             │
│                                                               │
│  🔍 TOP FEATURES CONTRIBUTING TO FRAUD PREDICTION            │
│                                                               │
│  Feature      Value     Impact    Contribution               │
│  ───────────────────────────────────────────────────────    │
│  V10         +2.45    ▶ FRAUD    +0.45  (Strongest)         │
│  V18         -1.89    ▶ FRAUD    +0.31                      │
│  amount      $149.99  ▶ FRAUD    +0.28                      │
│  V14         +0.67    ▶ FRAUD    +0.15                      │
│  V12         -0.34    ▶ FRAUD    +0.12                      │
│                                                               │
│  V5          -0.12    ◀ LEGIT    -0.23  (Against fraud)     │
│  V3          +0.45    ◀ LEGIT    -0.10                      │
│                                                               │
│  💡 INTERPRETATION                                           │
│  ┌────────────────────────────────────────────┐             │
│  │ This transaction is flagged because:       │             │
│  │                                             │             │
│  │ ✓ V10 value (+2.45) is unusually high     │             │
│  │   → Indicates abnormal transaction pattern │             │
│  │                                             │             │
│  │ ✓ Amount ($149.99) is in high-risk range  │             │
│  │   → Common fraud amount                    │             │
│  │                                             │             │
│  │ ✓ V18 value (-1.89) shows velocity risk   │             │
│  │   → Multiple transactions detected         │             │
│  │                                             │             │
│  │ ⚠ However, V5 suggests legitimacy         │             │
│  │   → Customer has good history              │             │
│  └────────────────────────────────────────────┘             │
│                                                               │
│  [Export Report]  [View Full Details]                       │
│                                                               │
└─────────────────────────────────────────────────────────────┘
```

### How to Read SHAP Values

#### 1. **Positive SHAP Values** (Red/Orange)
   - **Meaning:** Feature pushes prediction **toward FRAUD**
   - **Example:** `V10 = +2.45` has SHAP value `+0.45`
   - **Interpretation:** This unusual V10 value significantly increases fraud probability

#### 2. **Negative SHAP Values** (Blue/Green)
   - **Meaning:** Feature pushes prediction **toward LEGITIMATE**
   - **Example:** `V5 = -0.12` has SHAP value `-0.23`
   - **Interpretation:** This V5 value suggests the transaction is legitimate

#### 3. **Base Value**
   - **Meaning:** Average fraud rate in the population
   - **Example:** `12%` means 12% of all transactions are fraud
   - **Use:** Starting point before considering individual features

#### 4. **Prediction Value**
   - **Meaning:** Final fraud score after adding all SHAP contributions
   - **Calculation:** `Base Value + Sum(SHAP Values) = 87%`

### Visual Waterfall Example

```
Fraud Score Calculation:

Base Value (population avg)      12%  ┌─────┐
                                      │     │
+ V10 contribution              +45%  │     ├─────────────┐
+ V18 contribution              +31%  │     ├──────────┐  │
+ Amount contribution           +28%  │     ├────────┐ │  │
+ V14 contribution              +15%  │     ├───┐   │ │  │
+ V12 contribution              +12%  │     ├─┐ │   │ │  │
- V5 contribution (legitimate)  -23%  │   ┌─┘ │ │   │ │  │
- V3 contribution (legitimate)  -10%  │ ┌─┘   │ │   │ │  │
                                      │ │     │ │   │ │  │
Final Prediction                 87%  └─┴─────┴─┴───┴─┴──┘
                                        ▲
                                        │
                                   HIGH RISK
```

---

## Providing Feedback

### Why Feedback Matters

Your feedback helps the system:
1. **Improve accuracy** by learning from mistakes
2. **Reduce false positives** (legitimate transactions flagged as fraud)
3. **Catch new fraud patterns** that the model hasn't seen before

### How to Provide Feedback

#### Step 1: Review the Transaction & SHAP Explanation

#### Step 2: Make a Decision

In the Investigation page, you'll see:

```
┌─────────────────────────────────────────────────────────────┐
│  🔎 INVESTIGATION - PROVIDE FEEDBACK                         │
├─────────────────────────────────────────────────────────────┤
│                                                               │
│  Transaction TXN-789ABC reviewed by: analyst_john            │
│                                                               │
│  ❓ IS THIS TRANSACTION FRAUDULENT?                          │
│                                                               │
│  [✓ CONFIRM FRAUD]     [✗ MARK AS LEGITIMATE]               │
│                                                               │
│  📝 CONFIDENCE LEVEL                                         │
│  [★ ★ ★ ★ ☆] 4/5 stars                                     │
│                                                               │
│  💬 NOTES (Optional)                                         │
│  ┌────────────────────────────────────────────┐             │
│  │ Transaction looks suspicious due to:       │             │
│  │ - High amount for first purchase           │             │
│  │ - Unusual merchant category                │             │
│  │ - Verified with cardholder (confirmed)     │             │
│  └────────────────────────────────────────────┘             │
│                                                               │
│  [SUBMIT FEEDBACK]  [SKIP]                                   │
│                                                               │
└─────────────────────────────────────────────────────────────┘
```

#### Step 3: Submit Feedback

Your feedback is saved to the `feedback_labels` table:

```sql
INSERT INTO feedback_labels (
  transaction_id,
  analyst_id,
  confirmed_label,     -- 0=legitimate, 1=fraud
  confidence,          -- 1-5 stars
  feedback_notes,
  labeled_at
) VALUES (
  'TXN-789ABC',
  'analyst_john',
  1,                   -- Confirmed fraud
  4,                   -- 4/5 confidence
  'Verified with cardholder...',
  NOW()
);
```

#### Step 4: Model Retraining

- **Frequency:** Feedback is collected continuously
- **Retraining:** Models are retrained weekly (or when drift is detected)
- **Impact:** Your feedback is used in the next model version

---

## Monitoring System Performance

### 1. Model Performance Metrics

**Access:** Dashboard → "Metrics" card

**Key Metrics:**

| Metric | What It Means | Target |
|--------|---------------|--------|
| **Recall** | % of frauds detected | ≥ 95% |
| **Precision** | % of fraud alerts that are real | ≥ 85% |
| **False Positive Rate** | % of legitimate transactions flagged | ≤ 2% |
| **F1 Score** | Balance between recall & precision | ≥ 0.90 |

**Example:**
```
┌─────────────────────────────────────┐
│  MODEL PERFORMANCE (Last 24 Hours)  │
├─────────────────────────────────────┤
│  Recall:        96.2%  ✓ Above 95%  │
│  Precision:     87.4%  ✓ Above 85%  │
│  FPR:           1.8%   ✓ Below 2%   │
│  F1 Score:      0.916  ✓ Above 0.90 │
└─────────────────────────────────────┘
```

### 2. Drift Detection

**What is Drift?**
- **Data Drift:** Input data distribution changes (e.g., new fraud patterns)
- **Concept Drift:** Relationship between features and fraud changes
- **Target Drift:** Fraud rate changes significantly

**Monitoring:**
- **Automatic:** Drift is detected every hour by Airflow DAG
- **Alerts:** You'll be notified if drift exceeds thresholds
- **Action:** Model is automatically retrained when drift is detected

**Drift Dashboard:**
```
┌─────────────────────────────────────────────────┐
│  DRIFT MONITORING                               │
├─────────────────────────────────────────────────┤
│                                                  │
│  📊 Data Drift (PSI)                            │
│  Amount:    0.12  ✓ Normal (< 0.15)            │
│  V10:       0.08  ✓ Normal                     │
│  V18:       0.22  ⚠ Warning (> 0.15)           │
│                                                  │
│  🎯 Target Drift                                │
│  Fraud Rate: 0.42% → 0.38%  (-9.5%)           │
│  Status: ✓ Stable                              │
│                                                  │
│  🧠 Concept Drift                               │
│  Model Recall: 96% → 94%  (-2%)                │
│  Status: ⚠ Monitor closely                     │
│                                                  │
│  [View Full Report]  [Trigger Retraining]      │
│                                                  │
└─────────────────────────────────────────────────┘
```

### 3. System Health

**Access:** Dashboard → "System Status" indicator

**Health Checks:**
- ✅ **API:** Responding < 100ms
- ✅ **Database:** Connection healthy
- ✅ **Kafka:** Messages flowing
- ✅ **Models:** Loaded and responding

**If something is wrong:**
- 🔴 Red indicator appears
- Alert is sent to admin team
- Automatic recovery attempts
- Manual intervention if needed

---

## FAQ

### General Questions

**Q: How often are predictions made?**  
A: Predictions are made in **real-time** as transactions arrive. The system processes ~100 transactions every 10 seconds.

**Q: How accurate is the system?**  
A: The ensemble model achieves:
- **96% Recall** (detects 96% of frauds)
- **87% Precision** (87% of alerts are real fraud)
- **1.8% False Positive Rate** (only 1.8% of legitimate transactions are flagged)

**Q: What happens if the model makes a mistake?**  
A: Your feedback is crucial! When you mark a transaction as legitimate (false positive) or fraud (false negative), the system learns from it during the next retraining cycle.

### Technical Questions

**Q: Which features are most important for fraud detection?**  
A: The top features vary by model, but typically:
- `V10`, `V14`, `V17` (PCA features capturing fraud patterns)
- `amount` (transaction amount)
- `Time` (transaction timing)

Use SHAP explanations to see which features matter most for **each specific transaction**.

**Q: How does the ensemble model work?**  
A: It combines predictions from 4 models:
- **XGBoost (40%):** Gradient boosting trees
- **Neural Network (30%):** Deep learning
- **Random Forest (20%):** Decision tree ensemble
- **Isolation Forest (10%):** Anomaly detection

Final score = weighted average of individual model scores.

**Q: What is the "Base Value" in SHAP?**  
A: It's the **average fraud rate** in the training data (e.g., 12%). This is the starting point before considering individual transaction features.

**Q: Can I request explanations for old transactions?**  
A: Yes! Go to Transactions page → Search for transaction ID → Click "View Details" → "View SHAP Explanation"

### Workflow Questions

**Q: Do I need to review every transaction?**  
A: No. The system only sends **high-risk transactions** (fraud_score ≥ 0.7) for review. Most transactions are automatically approved.

**Q: How long do I have to review a transaction?**  
A: There's no strict time limit, but fraud alerts are sorted by **urgency**. Focus on:
1. **CRITICAL** (score ≥ 0.9): Review immediately
2. **HIGH** (score 0.7-0.9): Review within 1 hour
3. **MEDIUM** (score 0.5-0.7): Review when time permits

**Q: What if I'm unsure about a transaction?**  
A: Use SHAP explanations to understand the model's reasoning. If still unsure:
- Mark confidence as 2-3 stars (low confidence)
- Add notes explaining your uncertainty
- Skip the transaction for another analyst to review

**Q: How is feedback used for retraining?**  
A: Your feedback is saved in the `feedback_labels` table. During retraining:
- Confirmed frauds are added to the positive class
- False positives are added to the negative class
- Model learns to adjust weights based on your corrections

### Performance Questions

**Q: Why do I see legitimate transactions flagged as fraud?**  
A: This is called a "false positive." Possible reasons:
- Transaction pattern is similar to fraud (e.g., large amount, new merchant)
- Model is conservative to avoid missing real frauds
- Your feedback helps reduce false positives over time

**Q: Why did the model miss a fraud?**  
A: This is called a "false negative." Possible reasons:
- New fraud pattern not in training data
- Fraud score just below threshold (e.g., 0.68)
- Feature values similar to legitimate transactions
- Your feedback helps the model learn this pattern

**Q: How often are models updated?**  
A: Models are retrained:
- **Scheduled:** Weekly (every Sunday)
- **Drift-triggered:** When drift metrics exceed thresholds
- **Manual:** Admin can trigger retraining anytime

---

## Summary: Your Role as a Fraud Analyst

### Daily Workflow

1. **Monitor Dashboard**
   - Check real-time fraud alerts
   - Review system health indicators

2. **Investigate High-Priority Alerts**
   - Click on critical fraud alerts
   - Review transaction details
   - Analyze SHAP explanations

3. **Provide Feedback**
   - Confirm frauds or mark false positives
   - Add notes for context
   - Rate your confidence level

4. **Review Metrics**
   - Check model performance metrics
   - Monitor drift indicators
   - Report anomalies to admin team

### Impact of Your Work

✅ **Protects customers** from financial loss  
✅ **Improves model accuracy** through feedback  
✅ **Reduces false positives** over time  
✅ **Identifies new fraud patterns** early  
✅ **Ensures system reliability** through monitoring

---

## Additional Resources

- **Technical Documentation:** `/docs` folder
- **API Documentation:** `http://localhost:8000/docs` (Swagger)
- **Monitoring Dashboards:** `http://localhost:3000` (Grafana)
- **Support:** Contact ML team (Joshua, Olalekan, Soulaimana)

---

**Document End**

*This guide is maintained by the Fraud Detection ML Team. For updates or questions, please contact the team.*
