-- ============================================================================
-- PostgreSQL Database Schema for Fraud Detection
-- ============================================================================

-- Note: mlflow_db is created separately to avoid transaction issues in init scripts

-- ============================================================================
-- USERS TABLE - application users with role-based access
-- ============================================================================
CREATE TABLE IF NOT EXISTS users (
    id BIGSERIAL PRIMARY KEY,
    username VARCHAR(100) UNIQUE NOT NULL,
    email VARCHAR(255) UNIQUE NOT NULL,
    password_hash VARCHAR(255) NOT NULL,
    role VARCHAR(20) NOT NULL DEFAULT 'analyst',  -- 'admin', 'analyst', 'viewer'
    first_name VARCHAR(100),
    last_name VARCHAR(100),
    department VARCHAR(100),
    is_active BOOLEAN DEFAULT TRUE,
    is_verified BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    last_login TIMESTAMPTZ,
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_users_username ON users(username);
CREATE INDEX IF NOT EXISTS idx_users_email ON users(email);
CREATE INDEX IF NOT EXISTS idx_users_role ON users(role);
CREATE INDEX IF NOT EXISTS idx_users_is_active ON users(is_active);

-- ============================================================================
-- TRAINING_TRANSACTIONS TABLE - historical data used for model training
-- This contains the Kaggle creditcard.csv dataset with 284,807 transactions
-- Used ONLY for training, evaluation, and drift detection baseline
-- ============================================================================
CREATE TABLE IF NOT EXISTS training_transactions (
    id BIGSERIAL PRIMARY KEY,
    -- Time (seconds elapsed between this and first transaction)
    time INTEGER NOT NULL,
    -- PCA-transformed features V1-V28 (anonymized via PCA)
    v1 DECIMAL(10, 6),
    v2 DECIMAL(10, 6),
    v3 DECIMAL(10, 6),
    v4 DECIMAL(10, 6),
    v5 DECIMAL(10, 6),
    v6 DECIMAL(10, 6),
    v7 DECIMAL(10, 6),
    v8 DECIMAL(10, 6),
    v9 DECIMAL(10, 6),
    v10 DECIMAL(10, 6),
    v11 DECIMAL(10, 6),
    v12 DECIMAL(10, 6),
    v13 DECIMAL(10, 6),
    v14 DECIMAL(10, 6),
    v15 DECIMAL(10, 6),
    v16 DECIMAL(10, 6),
    v17 DECIMAL(10, 6),
    v18 DECIMAL(10, 6),
    v19 DECIMAL(10, 6),
    v20 DECIMAL(10, 6),
    v21 DECIMAL(10, 6),
    v22 DECIMAL(10, 6),
    v23 DECIMAL(10, 6),
    v24 DECIMAL(10, 6),
    v25 DECIMAL(10, 6),
    v26 DECIMAL(10, 6),
    v27 DECIMAL(10, 6),
    v28 DECIMAL(10, 6),
    -- amount and label
    amount DECIMAL(15, 2) NOT NULL,
    class INTEGER NOT NULL CHECK (class IN (0, 1)),  -- 0=legitimate, 1=fraud
    -- Metadata
    dataset_source VARCHAR(50) DEFAULT 'kaggle_creditcard',
    imported_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_training_time ON training_transactions(time);
CREATE INDEX IF NOT EXISTS idx_training_class ON training_transactions(class);
CREATE INDEX IF NOT EXISTS idx_training_amount ON training_transactions(amount);

-- ============================================================================
-- TRANSACTIONS TABLE - real-time production transactions
-- Stores all incoming transactions from Kafka for fraud detection
-- ============================================================================
CREATE TABLE IF NOT EXISTS transactions (
    id BIGSERIAL PRIMARY KEY,
    transaction_id VARCHAR(50) UNIQUE NOT NULL,
    -- Customer & Merchant
    customer_id VARCHAR(50) NOT NULL,
    merchant_id VARCHAR(50) NOT NULL,
    -- Transaction details
    amount DECIMAL(15, 2) NOT NULL,
    currency VARCHAR(3) NOT NULL DEFAULT 'USD',
    time TIMESTAMPTZ NOT NULL,
    -- Location
    customer_zip VARCHAR(20),
    merchant_zip VARCHAR(20),
    customer_country VARCHAR(2),
    merchant_country VARCHAR(2),
    -- Device & Session
    device_id VARCHAR(100),
    session_id VARCHAR(100),
    ip_address VARCHAR(45),
    -- Category
    mcc INTEGER,  -- Merchant Category Code
    transaction_type VARCHAR(50),
    -- Labels (ground truth, may be updated after investigation)
    is_fraud BOOLEAN DEFAULT FALSE,
    is_disputed BOOLEAN DEFAULT FALSE,
    -- Investigation tracking
    investigated_by BIGINT,  -- References users.id
    investigation_notes TEXT,
    investigated_at TIMESTAMPTZ,
    -- Metadata
    source_system VARCHAR(50) DEFAULT 'kafka',
    ingestion_timestamp TIMESTAMPTZ DEFAULT NOW(),
    created_at TIMESTAMPTZ DEFAULT NOW(),
    -- PCA Features (V1-V28) - populated by transaction_simulator for realistic testing
    v1 DECIMAL(10, 6),
    v2 DECIMAL(10, 6),
    v3 DECIMAL(10, 6),
    v4 DECIMAL(10, 6),
    v5 DECIMAL(10, 6),
    v6 DECIMAL(10, 6),
    v7 DECIMAL(10, 6),
    v8 DECIMAL(10, 6),
    v9 DECIMAL(10, 6),
    v10 DECIMAL(10, 6),
    v11 DECIMAL(10, 6),
    v12 DECIMAL(10, 6),
    v13 DECIMAL(10, 6),
    v14 DECIMAL(10, 6),
    v15 DECIMAL(10, 6),
    v16 DECIMAL(10, 6),
    v17 DECIMAL(10, 6),
    v18 DECIMAL(10, 6),
    v19 DECIMAL(10, 6),
    v20 DECIMAL(10, 6),
    v21 DECIMAL(10, 6),
    v22 DECIMAL(10, 6),
    v23 DECIMAL(10, 6),
    v24 DECIMAL(10, 6),
    v25 DECIMAL(10, 6),
    v26 DECIMAL(10, 6),
    v27 DECIMAL(10, 6),
    v28 DECIMAL(10, 6),
    FOREIGN KEY (investigated_by) REFERENCES users(id)
);

-- Indexes for performance
CREATE INDEX IF NOT EXISTS idx_transactions_customer_id ON transactions(customer_id);
CREATE INDEX IF NOT EXISTS idx_transactions_merchant_id ON transactions(merchant_id);
CREATE INDEX IF NOT EXISTS idx_transactions_time ON transactions(time);
CREATE INDEX IF NOT EXISTS idx_transactions_is_fraud ON transactions(is_fraud);
CREATE INDEX IF NOT EXISTS idx_transactions_customer_merchant ON transactions(customer_id, merchant_id);
CREATE INDEX IF NOT EXISTS idx_transactions_time_fraud ON transactions(time, is_fraud);
CREATE INDEX IF NOT EXISTS idx_transactions_ingestion ON transactions(ingestion_timestamp);

-- ============================================================================
-- PREDICTIONS TABLE - model predictions for transactions
-- Stores fraud detection predictions from the ML model
-- ============================================================================
CREATE TABLE IF NOT EXISTS predictions (
    id BIGSERIAL PRIMARY KEY,
    transaction_id VARCHAR(50) NOT NULL,
    -- Prediction results
    fraud_score DECIMAL(5, 4) NOT NULL CHECK (fraud_score BETWEEN 0 AND 1),
    is_fraud_predicted BOOLEAN NOT NULL,
    -- Model info
    model_version VARCHAR(20) NOT NULL,
    model_name VARCHAR(50) DEFAULT 'fraud_detection_model',
    -- Confidence & metadata
    confidence DECIMAL(5, 4),
    prediction_time TIMESTAMPTZ DEFAULT NOW(),
    prediction_latency_ms INTEGER,  -- Time taken for prediction
    -- Features used (for explainability)
    feature_importance JSONB,  -- Top features that influenced prediction
    -- Alert status
    alert_sent BOOLEAN DEFAULT FALSE,
    alert_sent_at TIMESTAMPTZ,
    FOREIGN KEY (transaction_id) REFERENCES transactions(transaction_id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_predictions_transaction_id ON predictions(transaction_id);
CREATE INDEX IF NOT EXISTS idx_predictions_time ON predictions(prediction_time);
CREATE INDEX IF NOT EXISTS idx_predictions_fraud_score ON predictions(fraud_score);
CREATE INDEX IF NOT EXISTS idx_predictions_is_fraud ON predictions(is_fraud_predicted);
CREATE INDEX IF NOT EXISTS idx_predictions_model_version ON predictions(model_version);
CREATE INDEX IF NOT EXISTS idx_predictions_time_fraud ON predictions(prediction_time, is_fraud_predicted);

-- ============================================================================
-- CUSTOMER_FEATURES TABLE - pre-computed customer features
-- ============================================================================
CREATE TABLE IF NOT EXISTS customer_features (
    id BIGSERIAL PRIMARY KEY,
    customer_id VARCHAR(50) UNIQUE NOT NULL,
    -- Transaction statistics
    total_transactions INT,
    avg_amount DECIMAL(15, 2),
    std_amount DECIMAL(15, 2),
    min_amount DECIMAL(15, 2),
    max_amount DECIMAL(15, 2),
    sum_amount DECIMAL(20, 2),
    -- Fraud statistics
    fraud_count INT DEFAULT 0,
    fraud_rate DECIMAL(5, 4) DEFAULT 0,
    last_update TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_customer_features_customer_id ON customer_features(customer_id);

-- ============================================================================
-- MERCHANT_FEATURES TABLE - pre-computed merchant features
-- ============================================================================
CREATE TABLE IF NOT EXISTS merchant_features (
    id BIGSERIAL PRIMARY KEY,
    merchant_id VARCHAR(50) UNIQUE NOT NULL,
    -- Transaction statistics
    total_transactions INT,
    avg_amount DECIMAL(15, 2),
    std_amount DECIMAL(15, 2),
    min_amount DECIMAL(15, 2),
    max_amount DECIMAL(15, 2),
    -- Fraud statistics
    fraud_count INT DEFAULT 0,
    fraud_rate DECIMAL(5, 4) DEFAULT 0,
    last_update TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_merchant_features_merchant_id ON merchant_features(merchant_id);

-- ============================================================================
-- DATA_QUALITY_LOG TABLE - tracks data quality issues
-- ============================================================================
CREATE TABLE IF NOT EXISTS data_quality_log (
    id BIGSERIAL PRIMARY KEY,
    check_type VARCHAR(50),
    severity VARCHAR(20),
    issue_description VARCHAR(500),
    affected_rows INT,
    check_timestamp TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_data_quality_check_type ON data_quality_log(check_type);
CREATE INDEX IF NOT EXISTS idx_data_quality_severity ON data_quality_log(severity);
CREATE INDEX IF NOT EXISTS idx_data_quality_timestamp ON data_quality_log(check_timestamp);

-- ============================================================================
-- PIPELINE_EXECUTION_LOG TABLE - tracks pipeline runs
-- ============================================================================
CREATE TABLE IF NOT EXISTS pipeline_execution_log (
    id BIGSERIAL PRIMARY KEY,
    pipeline_name VARCHAR(100),
    status VARCHAR(20),
    rows_processed INT,
    rows_stored INT,
    duration_seconds DECIMAL(10, 2),
    error_message TEXT,
    start_time TIMESTAMPTZ,
    end_time TIMESTAMPTZ,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_pipeline_name ON pipeline_execution_log(pipeline_name);
CREATE INDEX IF NOT EXISTS idx_pipeline_status ON pipeline_execution_log(status);
CREATE INDEX IF NOT EXISTS idx_pipeline_created_at ON pipeline_execution_log(created_at);

-- ============================================================================
-- DRIFT ANALYSIS RESULTS TABLE - stores comprehensive drift analysis results
-- ============================================================================
CREATE TABLE IF NOT EXISTS drift_analysis_results (
    id BIGSERIAL PRIMARY KEY,
    timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    analysis_window VARCHAR(20),
    reference_window VARCHAR(20),
    data_drift JSONB,
    target_drift JSONB,
    concept_drift JSONB,
    multivariate_drift JSONB,
    drift_summary JSONB
);

CREATE INDEX IF NOT EXISTS idx_drift_analysis_timestamp ON drift_analysis_results(timestamp);

-- ============================================================================
-- DRIFT REPORTS TABLE - stores automated drift reports
-- ============================================================================
CREATE TABLE IF NOT EXISTS drift_reports (
    id BIGSERIAL PRIMARY KEY,
    timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    summary JSONB,
    recommendations TEXT[],
    alerts JSONB,
    severity VARCHAR(20)
);

CREATE INDEX IF NOT EXISTS idx_drift_reports_timestamp ON drift_reports(timestamp);
CREATE INDEX IF NOT EXISTS idx_drift_reports_severity ON drift_reports(severity);

-- ============================================================================
-- DRIFT_METRICS TABLE - stores drift detection results
-- ============================================================================
CREATE TABLE IF NOT EXISTS drift_metrics (
    id BIGSERIAL PRIMARY KEY,
    timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    -- Type de drift
    metric_type VARCHAR(50) NOT NULL,  -- 'data_drift', 'target_drift', 'concept_drift'
    metric_name VARCHAR(100),          -- 'psi_amount', 'fraud_rate_change', 'recall_drop'
    -- Valeurs
    metric_value DECIMAL(10, 6) NOT NULL,
    threshold DECIMAL(5, 4) NOT NULL,
    threshold_exceeded BOOLEAN NOT NULL DEFAULT FALSE,
    -- Metadata
    severity VARCHAR(20),  -- 'LOW', 'MEDIUM', 'HIGH', 'CRITICAL'
    feature_name VARCHAR(100),  -- Pour data drift par feature
    details JSONB  -- Informations supplémentaires en JSON
);

CREATE INDEX IF NOT EXISTS idx_drift_timestamp ON drift_metrics(timestamp);
CREATE INDEX IF NOT EXISTS idx_drift_metric_type ON drift_metrics(metric_type);
CREATE INDEX IF NOT EXISTS idx_drift_threshold_exceeded ON drift_metrics(threshold_exceeded);
CREATE INDEX IF NOT EXISTS idx_drift_severity ON drift_metrics(severity);

-- ============================================================================
-- RETRAINING_TRIGGERS TABLE - tracks when model retraining is triggered
-- ============================================================================
CREATE TABLE IF NOT EXISTS retraining_triggers (
    id BIGSERIAL PRIMARY KEY,
    triggered_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    -- Raison du déclenchement
    trigger_reason TEXT NOT NULL,
    drift_type VARCHAR(50),  -- 'data_drift', 'target_drift', 'concept_drift'
    drift_severity VARCHAR(20),
    -- Airflow DAG info
    airflow_dag_id VARCHAR(100),
    airflow_run_id VARCHAR(100),
    airflow_execution_date TIMESTAMPTZ,
    -- Statut
    status VARCHAR(20) NOT NULL DEFAULT 'pending',  -- 'pending', 'running', 'completed', 'failed'
    completed_at TIMESTAMPTZ,
    error_message TEXT,
    -- Metadata
    drift_metrics_id BIGINT,
    new_model_version VARCHAR(50),
    FOREIGN KEY (drift_metrics_id) REFERENCES drift_metrics(id)
);

CREATE INDEX IF NOT EXISTS idx_retraining_triggered_at ON retraining_triggers(triggered_at);
CREATE INDEX IF NOT EXISTS idx_retraining_status ON retraining_triggers(status);
CREATE INDEX IF NOT EXISTS idx_retraining_dag_run ON retraining_triggers(airflow_dag_id, airflow_run_id);

-- ============================================================================
-- MODEL_VERSIONS TABLE - tracks all trained model versions
-- ============================================================================
CREATE TABLE IF NOT EXISTS model_versions (
    id BIGSERIAL PRIMARY KEY,
    version VARCHAR(50) UNIQUE NOT NULL,  -- 'v1.0.0', 'v1.1.0'
    -- Training info
    training_date TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    training_dataset_size INT,
    training_duration_seconds INT,
    -- Performance metrics
    recall DECIMAL(5, 4),
    precision DECIMAL(5, 4),
    f1_score DECIMAL(5, 4),
    auc_roc DECIMAL(5, 4),
    false_positive_rate DECIMAL(5, 4),
    -- Deployment status
    deployment_status VARCHAR(20) DEFAULT 'trained',  -- 'trained', 'staging', 'production', 'retired'
    deployed_at TIMESTAMPTZ,
    retired_at TIMESTAMPTZ,
    -- MLflow info
    mlflow_run_id VARCHAR(100),
    mlflow_model_uri TEXT,
    -- Metadata
    training_config JSONB,  -- Hyperparameters, etc.
    notes TEXT
);

CREATE INDEX IF NOT EXISTS idx_model_version ON model_versions(version);
CREATE INDEX IF NOT EXISTS idx_model_deployment_status ON model_versions(deployment_status);
CREATE INDEX IF NOT EXISTS idx_model_training_date ON model_versions(training_date);

-- ============================================================================
-- FEEDBACK_LABELS TABLE - analyst confirmations for predictions
-- ============================================================================
CREATE TABLE IF NOT EXISTS feedback_labels (
    id BIGSERIAL PRIMARY KEY,
    transaction_id VARCHAR(50) NOT NULL,
    -- Analyst info
    analyst_id VARCHAR(100) NOT NULL,
    analyst_email VARCHAR(255),
    -- Feedback
    confirmed_label INT NOT NULL CHECK (confirmed_label IN (0, 1)),  -- 0=legitimate, 1=fraud
    confidence INT CHECK (confidence BETWEEN 1 AND 5),  -- 1-5 stars
    feedback_notes TEXT,
    investigation_time_seconds INT,
    -- Timestamps
    labeled_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    -- Comparison avec prédiction originale
    original_prediction INT,
    original_confidence DECIMAL(5, 4),
    model_version VARCHAR(50),
    FOREIGN KEY (transaction_id) REFERENCES transactions(transaction_id)
);

CREATE INDEX IF NOT EXISTS idx_feedback_transaction ON feedback_labels(transaction_id);
CREATE INDEX IF NOT EXISTS idx_feedback_labeled_at ON feedback_labels(labeled_at);
CREATE INDEX IF NOT EXISTS idx_feedback_confirmed_label ON feedback_labels(confirmed_label);
CREATE INDEX IF NOT EXISTS idx_feedback_analyst ON feedback_labels(analyst_id);

-- ============================================================================
-- AIRFLOW_TASK_METRICS TABLE - custom metrics for Airflow tasks
-- ============================================================================
CREATE TABLE IF NOT EXISTS airflow_task_metrics (
    id BIGSERIAL PRIMARY KEY,
    -- Airflow task info
    dag_id VARCHAR(100) NOT NULL,
    task_id VARCHAR(100) NOT NULL,
    execution_date TIMESTAMPTZ NOT NULL,
    run_id VARCHAR(100),
    -- Performance metrics
    duration_seconds DECIMAL(10, 2),
    records_processed INT,
    records_failed INT,
    -- Resource usage
    cpu_percent DECIMAL(5, 2),
    memory_mb INT,
    -- Status
    status VARCHAR(20),  -- 'success', 'failed', 'running'
    error_message TEXT,
    -- Timestamps
    started_at TIMESTAMPTZ,
    completed_at TIMESTAMPTZ
);

CREATE INDEX IF NOT EXISTS idx_airflow_dag_task ON airflow_task_metrics(dag_id, task_id);
CREATE INDEX IF NOT EXISTS idx_airflow_execution_date ON airflow_task_metrics(execution_date);
CREATE INDEX IF NOT EXISTS idx_airflow_status ON airflow_task_metrics(status);

-- ============================================================================
-- VIEWS - convenient queries for common use cases
-- ============================================================================

-- View: Recent fraudulent transactions with predictions
CREATE OR REPLACE VIEW recent_frauds AS
SELECT
    t.transaction_id,
    t.customer_id,
    t.merchant_id,
    t.amount,
    t.time,
    p.fraud_score,
    p.model_version,
    p.prediction_time,
    t.is_fraud as actual_fraud,
    p.is_fraud_predicted as predicted_fraud
FROM transactions t
LEFT JOIN predictions p ON t.transaction_id = p.transaction_id
WHERE p.is_fraud_predicted = TRUE
ORDER BY t.time DESC;

-- View: Model performance metrics (confusion matrix)
CREATE OR REPLACE VIEW model_performance AS
SELECT
    COUNT(*) as total_predictions,
    SUM(CASE WHEN p.is_fraud_predicted = TRUE AND t.is_fraud = TRUE THEN 1 ELSE 0 END) as true_positives,
    SUM(CASE WHEN p.is_fraud_predicted = FALSE AND t.is_fraud = FALSE THEN 1 ELSE 0 END) as true_negatives,
    SUM(CASE WHEN p.is_fraud_predicted = TRUE AND t.is_fraud = FALSE THEN 1 ELSE 0 END) as false_positives,
    SUM(CASE WHEN p.is_fraud_predicted = FALSE AND t.is_fraud = TRUE THEN 1 ELSE 0 END) as false_negatives,
    p.model_version
FROM predictions p
JOIN transactions t ON p.transaction_id = t.transaction_id
WHERE t.is_fraud IS NOT NULL
GROUP BY p.model_version;

-- View: Training dataset statistics
CREATE OR REPLACE VIEW training_stats AS
SELECT
    COUNT(*) as total_transactions,
    SUM(CASE WHEN class = 1 THEN 1 ELSE 0 END) as fraud_count,
    SUM(CASE WHEN class = 0 THEN 1 ELSE 0 END) as legitimate_count,
    ROUND(AVG(CASE WHEN class = 1 THEN 1.0 ELSE 0.0 END), 4) as fraud_rate,
    ROUND(AVG(amount), 2) as avg_amount,
    ROUND(MIN(amount), 2) as min_amount,
    ROUND(MAX(amount), 2) as max_amount,
    ROUND(STDDEV(amount), 2) as stddev_amount
FROM training_transactions;

-- ============================================================================
-- TRIGGERS - automatic timestamp updates
-- ============================================================================

-- Function to update updated_at timestamp
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Trigger for users table
DROP TRIGGER IF EXISTS update_users_updated_at ON users;
CREATE TRIGGER update_users_updated_at
    BEFORE UPDATE ON users
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

-- ============================================================================
-- FUNCTIONS - useful database functions
-- ============================================================================

-- Function to calculate fraud rate for a time period
CREATE OR REPLACE FUNCTION calculate_fraud_rate(
    start_time TIMESTAMPTZ,
    end_time TIMESTAMPTZ
)
RETURNS DECIMAL(5, 4) AS $$
DECLARE
    total_count INTEGER;
    fraud_count INTEGER;
BEGIN
    SELECT COUNT(*), SUM(CASE WHEN is_fraud THEN 1 ELSE 0 END)
    INTO total_count, fraud_count
    FROM transactions
    WHERE time BETWEEN start_time AND end_time;

    IF total_count = 0 THEN
        RETURN 0;
    END IF;

    RETURN fraud_count::DECIMAL / total_count::DECIMAL;
END;
$$ LANGUAGE plpgsql;

-- ============================================================================
-- Additional INDEXES for performance
-- ============================================================================
CREATE INDEX IF NOT EXISTS idx_transactions_customer_merchant ON transactions(customer_id, merchant_id);
CREATE INDEX IF NOT EXISTS idx_transactions_datetime_range ON transactions(time, is_fraud);
CREATE INDEX IF NOT EXISTS idx_predictions_prediction_time_class ON predictions(prediction_time, is_fraud_predicted);
CREATE INDEX IF NOT EXISTS idx_transactions_time_fraud ON transactions(time, is_fraud);
CREATE INDEX IF NOT EXISTS idx_drift_metrics_timestamp_type ON drift_metrics(timestamp, metric_type);

-- ============================================================================
-- INITIAL DATA - create default admin user
-- Password: admin123 (hashed with bcrypt)
-- ============================================================================
INSERT INTO users (username, email, password_hash, role, first_name, last_name, is_active, is_verified)
VALUES (
    'admin',
    'admin@frauddetection.local',
    '$2b$12$LQv3c1yqBWVHxkd0LHAkCOYz6TtxMQJqhN8/LewY5GyYk3H.OwLKa',  -- admin123
    'admin',
    'System',
    'Administrator',
    TRUE,
    TRUE
)
ON CONFLICT (username) DO NOTHING;

-- ============================================================================
-- COMMENTS - documentation for tables
-- ============================================================================
COMMENT ON TABLE users IS 'Application users with role-based access control';
COMMENT ON TABLE training_transactions IS 'Historical transactions from Kaggle creditcard.csv used for model training';
COMMENT ON TABLE transactions IS 'Real-time production transactions from Kafka for fraud detection';
COMMENT ON TABLE predictions IS 'ML model predictions for transactions';

COMMENT ON COLUMN training_transactions.time IS 'Seconds elapsed since first transaction in dataset';
COMMENT ON COLUMN training_transactions.class IS '0=legitimate, 1=fraud';
COMMENT ON COLUMN transactions.v1 IS 'PCA feature V1 (anonymized)';
COMMENT ON COLUMN predictions.fraud_score IS 'Probability of fraud (0.0 to 1.0)';
