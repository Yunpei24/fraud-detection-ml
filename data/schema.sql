-- PostgreSQL Database Schema for Fraud Detection

-- ============================================================================
-- TRANSACTIONS TABLE - stores all incoming transactions
-- ============================================================================
CREATE TABLE IF NOT EXISTS transactions (
    id BIGSERIAL PRIMARY KEY,
    transaction_id VARCHAR(50) UNIQUE NOT NULL,
    customer_id VARCHAR(50) NOT NULL,
    merchant_id VARCHAR(50) NOT NULL,
    
    -- Transaction details
    amount DECIMAL(15, 2) NOT NULL,
    currency VARCHAR(3) NOT NULL,
    transaction_time TIMESTAMPTZ NOT NULL,
    
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
    mcc INT,
    transaction_type VARCHAR(50),
    
    -- Labels
    is_fraud BOOLEAN DEFAULT FALSE,
    is_disputed BOOLEAN DEFAULT FALSE,
    
    -- Metadata
    source_system VARCHAR(50) DEFAULT 'mobile',
    ingestion_timestamp TIMESTAMPTZ DEFAULT NOW(),
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_customer_id ON transactions(customer_id);
CREATE INDEX IF NOT EXISTS idx_merchant_id ON transactions(merchant_id);
CREATE INDEX IF NOT EXISTS idx_transaction_time ON transactions(transaction_time);
CREATE INDEX IF NOT EXISTS idx_is_fraud ON transactions(is_fraud);

-- ============================================================================
-- PREDICTIONS TABLE - stores model predictions
-- ============================================================================
CREATE TABLE IF NOT EXISTS predictions (
    id BIGSERIAL PRIMARY KEY,
    transaction_id VARCHAR(50) NOT NULL,
    fraud_score DECIMAL(5, 4) NOT NULL,
    is_fraud_predicted BOOLEAN NOT NULL,
    model_version VARCHAR(20) NOT NULL,
    confidence DECIMAL(5, 4),
    prediction_time TIMESTAMPTZ DEFAULT NOW(),
    
    FOREIGN KEY (transaction_id) REFERENCES transactions(transaction_id)
);

CREATE INDEX IF NOT EXISTS idx_predictions_transaction_id ON predictions(transaction_id);
CREATE INDEX IF NOT EXISTS idx_predictions_time ON predictions(prediction_time);
CREATE INDEX IF NOT EXISTS idx_predictions_fraud_score ON predictions(fraud_score);

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
-- DRIFT_METRICS TABLE - stores drift detection results (NEW)
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
-- RETRAINING_TRIGGERS TABLE - tracks when model retraining is triggered (NEW)
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
-- MODEL_VERSIONS TABLE - tracks all trained model versions (NEW)
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
-- FEEDBACK_LABELS TABLE - analyst confirmations for predictions (NEW)
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
-- AIRFLOW_TASK_METRICS TABLE - custom metrics for Airflow tasks (NEW)
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
-- Additional INDEXES for performance
-- ============================================================================
CREATE INDEX IF NOT EXISTS idx_transactions_customer_merchant ON transactions(customer_id, merchant_id);
CREATE INDEX IF NOT EXISTS idx_transactions_datetime_range ON transactions(transaction_time, is_fraud);
CREATE INDEX IF NOT EXISTS idx_predictions_prediction_time_class ON predictions(prediction_time, is_fraud_predicted);
CREATE INDEX IF NOT EXISTS idx_transactions_time_fraud ON transactions(transaction_time, is_fraud);
CREATE INDEX IF NOT EXISTS idx_drift_metrics_timestamp_type ON drift_metrics(timestamp, metric_type);
