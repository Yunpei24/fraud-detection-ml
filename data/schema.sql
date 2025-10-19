# SQL Server Database Schema for Fraud Detection

-- ============================================================================
-- TRANSACTIONS TABLE - stores all incoming transactions
-- ============================================================================
CREATE TABLE IF NOT EXISTS transactions (
    id BIGINT PRIMARY KEY IDENTITY(1,1),
    transaction_id NVARCHAR(50) UNIQUE NOT NULL,
    customer_id NVARCHAR(50) NOT NULL,
    merchant_id NVARCHAR(50) NOT NULL,
    
    -- Transaction details
    amount DECIMAL(15, 2) NOT NULL,
    currency NVARCHAR(3) NOT NULL,
    transaction_time DATETIME2 NOT NULL,
    
    -- Location
    customer_zip NVARCHAR(20),
    merchant_zip NVARCHAR(20),
    customer_country NVARCHAR(2),
    merchant_country NVARCHAR(2),
    
    -- Device & Session
    device_id NVARCHAR(100),
    session_id NVARCHAR(100),
    ip_address NVARCHAR(45),
    
    -- Category
    mcc INT,
    transaction_type NVARCHAR(50),
    
    -- Labels
    is_fraud BIT DEFAULT 0,
    is_disputed BIT DEFAULT 0,
    
    -- Metadata
    source_system NVARCHAR(50) DEFAULT 'mobile',
    ingestion_timestamp DATETIME2 DEFAULT GETUTCDATE(),
    created_at DATETIME2 DEFAULT GETUTCDATE(),
    
    INDEX idx_customer_id (customer_id),
    INDEX idx_merchant_id (merchant_id),
    INDEX idx_transaction_time (transaction_time),
    INDEX idx_is_fraud (is_fraud)
);

-- ============================================================================
-- PREDICTIONS TABLE - stores model predictions
-- ============================================================================
CREATE TABLE IF NOT EXISTS predictions (
    id BIGINT PRIMARY KEY IDENTITY(1,1),
    transaction_id NVARCHAR(50) NOT NULL,
    fraud_score DECIMAL(5, 4) NOT NULL,
    is_fraud_predicted BIT NOT NULL,
    model_version NVARCHAR(20) NOT NULL,
    confidence DECIMAL(5, 4),
    prediction_time DATETIME2 DEFAULT GETUTCDATE(),
    
    FOREIGN KEY (transaction_id) REFERENCES transactions(transaction_id),
    INDEX idx_transaction_id (transaction_id),
    INDEX idx_prediction_time (prediction_time)
);

-- ============================================================================
-- CUSTOMER_FEATURES TABLE - pre-computed customer features
-- ============================================================================
CREATE TABLE IF NOT EXISTS customer_features (
    id BIGINT PRIMARY KEY IDENTITY(1,1),
    customer_id NVARCHAR(50) UNIQUE NOT NULL,
    
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
    
    last_update DATETIME2 DEFAULT GETUTCDATE(),
    
    INDEX idx_customer_id (customer_id)
);

-- ============================================================================
-- MERCHANT_FEATURES TABLE - pre-computed merchant features
-- ============================================================================
CREATE TABLE IF NOT EXISTS merchant_features (
    id BIGINT PRIMARY KEY IDENTITY(1,1),
    merchant_id NVARCHAR(50) UNIQUE NOT NULL,
    
    -- Transaction statistics
    total_transactions INT,
    avg_amount DECIMAL(15, 2),
    std_amount DECIMAL(15, 2),
    min_amount DECIMAL(15, 2),
    max_amount DECIMAL(15, 2),
    
    -- Fraud statistics
    fraud_count INT DEFAULT 0,
    fraud_rate DECIMAL(5, 4) DEFAULT 0,
    
    last_update DATETIME2 DEFAULT GETUTCDATE(),
    
    INDEX idx_merchant_id (merchant_id)
);

-- ============================================================================
-- DATA_QUALITY_LOG TABLE - tracks data quality issues
-- ============================================================================
CREATE TABLE IF NOT EXISTS data_quality_log (
    id BIGINT PRIMARY KEY IDENTITY(1,1),
    check_type NVARCHAR(50),
    severity NVARCHAR(20),
    issue_description NVARCHAR(500),
    affected_rows INT,
    check_timestamp DATETIME2 DEFAULT GETUTCDATE(),
    
    INDEX idx_check_type (check_type),
    INDEX idx_severity (severity),
    INDEX idx_check_timestamp (check_timestamp)
);

-- ============================================================================
-- PIPELINE_EXECUTION_LOG TABLE - tracks pipeline runs
-- ============================================================================
CREATE TABLE IF NOT EXISTS pipeline_execution_log (
    id BIGINT PRIMARY KEY IDENTITY(1,1),
    pipeline_name NVARCHAR(100),
    status NVARCHAR(20),
    rows_processed INT,
    rows_stored INT,
    duration_seconds DECIMAL(10, 2),
    error_message NVARCHAR(1000),
    start_time DATETIME2,
    end_time DATETIME2,
    created_at DATETIME2 DEFAULT GETUTCDATE(),
    
    INDEX idx_pipeline_name (pipeline_name),
    INDEX idx_status (status),
    INDEX idx_created_at (created_at)
);

-- ============================================================================
-- CREATE INDEXES for performance
-- ============================================================================
CREATE INDEX idx_transactions_customer_merchant ON transactions(customer_id, merchant_id);
CREATE INDEX idx_transactions_datetime_range ON transactions(transaction_time, is_fraud);
CREATE INDEX idx_predictions_fraud_score ON predictions(fraud_score);
