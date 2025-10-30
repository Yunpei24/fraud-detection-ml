#!/usr/bin/env bash
#
# test_full_simulation.sh
#
# Script de test end-to-end pour le système de détection de fraude avec Kafka
# Teste le flux complet: Kafka → Simulateur → Pipeline → API → PostgreSQL
#
# Usage:
#   ./test_full_simulation.sh [--batch-size N] [--fraud-rate F] [--skip-setup]
#
# Options:
#   --batch-size N    Nombre de transactions à simuler (défaut: 1000)
#   --fraud-rate F    Taux de fraude (0.0-1.0, défaut: 0.05)
#   --skip-setup      Skip Docker setup (assume containers are running)
#   --help            Show this help message
#

set -e  # Exit on error
set -u  # Exit on undefined variable

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
BATCH_SIZE=${BATCH_SIZE:-1000}
FRAUD_RATE=${FRAUD_RATE:-0.05}
SKIP_SETUP=false
KAFKA_TOPIC="fraud-detection-transactions"
KAFKA_BOOTSTRAP_SERVERS="localhost:29092"
DB_CONTAINER="fraud_detection_db"
KAFKA_CONTAINER="fraud_detection_kafka"
DATA_CONTAINER="fraud_data_pipeline"

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --batch-size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --fraud-rate)
            FRAUD_RATE="$2"
            shift 2
            ;;
        --skip-setup)
            SKIP_SETUP=true
            shift
            ;;
        --help)
            grep '^#' "$0" | cut -c 3-
            exit 0
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Helper functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

check_docker() {
    if ! command -v docker &> /dev/null; then
        log_error "Docker is not installed or not in PATH"
        exit 1
    fi
    log_success "Docker is installed"
}

check_container() {
    local container=$1
    if ! docker ps --format '{{.Names}}' | grep -q "^${container}$"; then
        log_error "Container '${container}' is not running"
        return 1
    fi
    log_success "Container '${container}' is running"
    return 0
}

wait_for_container() {
    local container=$1
    local max_attempts=30
    local attempt=1
    
    log_info "Waiting for ${container} to be healthy..."
    
    while [ $attempt -le $max_attempts ]; do
        if docker inspect --format='{{.State.Health.Status}}' "$container" 2>/dev/null | grep -q "healthy"; then
            log_success "${container} is healthy"
            return 0
        fi
        
        echo -n "."
        sleep 2
        attempt=$((attempt + 1))
    done
    
    echo ""
    log_error "${container} did not become healthy in time"
    return 1
}

create_kafka_topic() {
    log_info "Creating Kafka topic: ${KAFKA_TOPIC}"
    
    # Check if topic already exists
    if docker exec "$KAFKA_CONTAINER" kafka-topics \
        --list \
        --bootstrap-server localhost:9092 \
        2>/dev/null | grep -q "^${KAFKA_TOPIC}$"; then
        log_warning "Topic '${KAFKA_TOPIC}' already exists"
        return 0
    fi
    
    # Create topic
    if docker exec "$KAFKA_CONTAINER" kafka-topics \
        --create \
        --topic "$KAFKA_TOPIC" \
        --bootstrap-server localhost:9092 \
        --partitions 3 \
        --replication-factor 1 \
        2>/dev/null; then
        log_success "Topic '${KAFKA_TOPIC}' created"
        return 0
    else
        log_error "Failed to create topic '${KAFKA_TOPIC}'"
        return 1
    fi
}

get_kafka_message_count() {
    local count=$(docker exec "$KAFKA_CONTAINER" kafka-run-class \
        kafka.tools.GetOffsetShell \
        --broker-list localhost:9092 \
        --topic "$KAFKA_TOPIC" \
        --time -1 \
        2>/dev/null | awk -F ':' '{sum += $3} END {print sum}')
    
    echo "${count:-0}"
}

run_simulator() {
    log_info "Running transaction simulator (batch size: ${BATCH_SIZE}, fraud rate: ${FRAUD_RATE})"
    
    if docker exec "$DATA_CONTAINER" python -m src.ingestion.transaction_simulator \
        --mode batch \
        --count "$BATCH_SIZE" \
        --fraud-rate "$FRAUD_RATE" \
        --kafka-servers kafka:9092 \
        --kafka-topic "$KAFKA_TOPIC" \
        2>&1 | tee /tmp/simulator_output.log; then
        log_success "Simulator completed successfully"
        return 0
    else
        log_error "Simulator failed"
        cat /tmp/simulator_output.log
        return 1
    fi
}

run_batch_pipeline() {
    log_info "Running batch prediction pipeline"
    
    if docker exec "$DATA_CONTAINER" python -m src.pipelines.realtime_pipeline \
        --mode batch \
        --count "$BATCH_SIZE" \
        --kafka-servers kafka:9092 \
        --kafka-topic "$KAFKA_TOPIC" \
        2>&1 | tee /tmp/pipeline_output.log; then
        log_success "Pipeline completed successfully"
        return 0
    else
        log_error "Pipeline failed"
        cat /tmp/pipeline_output.log
        return 1
    fi
}

verify_database() {
    log_info "Verifying PostgreSQL ingestion"
    
    # Count transactions in database
    local count=$(docker exec "$DB_CONTAINER" psql -U fraud_user -d fraud_detection \
        -t -c "SELECT COUNT(*) FROM transactions;" 2>/dev/null | xargs)
    
    if [ -z "$count" ] || [ "$count" -eq 0 ]; then
        log_error "No transactions found in database"
        return 1
    fi
    
    log_success "Found ${count} transactions in database"
    
    # Count predictions
    local pred_count=$(docker exec "$DB_CONTAINER" psql -U fraud_user -d fraud_detection \
        -t -c "SELECT COUNT(*) FROM transactions WHERE prediction IS NOT NULL;" 2>/dev/null | xargs)
    
    log_info "Transactions with predictions: ${pred_count}"
    
    # Check fraud rate
    local fraud_count=$(docker exec "$DB_CONTAINER" psql -U fraud_user -d fraud_detection \
        -t -c "SELECT COUNT(*) FROM transactions WHERE prediction = 1;" 2>/dev/null | xargs)
    
    local fraud_rate=$(echo "scale=2; $fraud_count * 100 / $count" | bc)
    log_info "Fraud detection rate: ${fraud_rate}%"
    
    return 0
}

print_summary() {
    echo ""
    echo "=========================================="
    echo "  FRAUD DETECTION SIMULATION SUMMARY"
    echo "=========================================="
    echo ""
    echo "Configuration:"
    echo "  - Batch Size: ${BATCH_SIZE}"
    echo "  - Fraud Rate: ${FRAUD_RATE}"
    echo "  - Kafka Topic: ${KAFKA_TOPIC}"
    echo ""
    
    # Kafka stats
    local kafka_messages=$(get_kafka_message_count)
    echo "Kafka Statistics:"
    echo "  - Messages in topic: ${kafka_messages}"
    echo ""
    
    # Database stats
    local db_count=$(docker exec "$DB_CONTAINER" psql -U fraud_user -d fraud_detection \
        -t -c "SELECT COUNT(*) FROM transactions;" 2>/dev/null | xargs || echo "0")
    local fraud_count=$(docker exec "$DB_CONTAINER" psql -U fraud_user -d fraud_detection \
        -t -c "SELECT COUNT(*) FROM transactions WHERE prediction = 1;" 2>/dev/null | xargs || echo "0")
    
    echo "Database Statistics:"
    echo "  - Total transactions: ${db_count}"
    echo "  - Frauds detected: ${fraud_count}"
    if [ "$db_count" -gt 0 ]; then
        local fraud_pct=$(echo "scale=2; $fraud_count * 100 / $db_count" | bc)
        echo "  - Fraud rate: ${fraud_pct}%"
    fi
    echo ""
    echo "=========================================="
}

cleanup() {
    log_info "Cleaning up temporary files"
    rm -f /tmp/simulator_output.log /tmp/pipeline_output.log
}

# Main execution
main() {
    echo ""
    echo "=========================================="
    echo "  FRAUD DETECTION - END TO END TEST"
    echo "=========================================="
    echo ""
    
    # Step 0: Check Docker
    log_info "Step 0: Checking Docker installation"
    check_docker
    echo ""
    
    # Step 1: Setup (if not skipped)
    if [ "$SKIP_SETUP" = false ]; then
        log_info "Step 1: Starting Docker containers"
        cd "$(dirname "$0")"
        docker compose up -d
        echo ""
        
        log_info "Step 2: Waiting for services to be healthy"
        wait_for_container "$DB_CONTAINER"
        wait_for_container "$KAFKA_CONTAINER"
        wait_for_container "$DATA_CONTAINER"
        echo ""
    else
        log_info "Step 1-2: Skipped (--skip-setup)"
        
        log_info "Checking required containers are running"
        check_container "$DB_CONTAINER" || exit 1
        check_container "$KAFKA_CONTAINER" || exit 1
        check_container "$DATA_CONTAINER" || exit 1
        echo ""
    fi
    
    # Step 3: Create Kafka topic
    log_info "Step 3: Creating Kafka topic"
    create_kafka_topic || exit 1
    echo ""
    
    # Step 4: Run simulator
    log_info "Step 4: Running transaction simulator"
    run_simulator || exit 1
    echo ""
    
    # Step 5: Verify Kafka ingestion
    log_info "Step 5: Verifying Kafka ingestion"
    local kafka_count=$(get_kafka_message_count)
    if [ "$kafka_count" -lt "$BATCH_SIZE" ]; then
        log_warning "Expected ${BATCH_SIZE} messages, found ${kafka_count}"
    else
        log_success "Kafka has ${kafka_count} messages"
    fi
    echo ""
    
    # Step 6: Run batch pipeline
    log_info "Step 6: Running batch prediction pipeline"
    run_batch_pipeline || exit 1
    echo ""
    
    # Step 7: Verify database
    log_info "Step 7: Verifying PostgreSQL ingestion"
    verify_database || exit 1
    echo ""
    
    # Step 8: Print summary
    print_summary
    
    # Cleanup
    cleanup
    
    log_success "All tests passed! ✅"
    exit 0
}

# Trap errors
trap 'log_error "Test failed at line $LINENO"; cleanup; exit 1' ERR

# Run main
main
