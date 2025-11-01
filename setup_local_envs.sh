#!/bin/bash
# ==============================================================================
# SETUP LOCAL DEVELOPMENT ENVIRONMENTS
# ==============================================================================
# Script pour initialiser tous les environnements virtuels locaux
# Crée les environnements virtuels pour chaque module et installe les dépendances
# ==============================================================================

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Module configurations
declare -A MODULES=(
    ["api"]="$PROJECT_ROOT/api:venv_api"
    ["data"]="$PROJECT_ROOT/data:venv"
    ["drift"]="$PROJECT_ROOT/drift:venv_drift"
    ["training"]="$PROJECT_ROOT/training:venv"
    ["airflow"]="$PROJECT_ROOT/airflow:venv_airflow"
)

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

# Setup virtual environment for a module
setup_module_venv() {
    local module_name=$1
    local module_config=$2

    IFS=':' read -r module_dir venv_name <<< "$module_config"

    log_info "Setting up $module_name module..."

    # Check if directory exists
    if [ ! -d "$module_dir" ]; then
        log_error "Module directory not found: $module_dir"
        return 1
    fi

    # Check if requirements.txt exists
    local requirements_file="$module_dir/requirements.txt"
    if [ ! -f "$requirements_file" ]; then
        log_warning "requirements.txt not found in $module_dir"
        requirements_file=""
    fi

    # Create virtual environment
    local venv_path="$module_dir/$venv_name"
    if [ -d "$venv_path" ]; then
        log_warning "Virtual environment already exists: $venv_path"
        read -p "Recreate it? (y/N): " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            log_info "Skipping $module_name"
            return 0
        fi
        rm -rf "$venv_path"
    fi

    log_info "Creating virtual environment: $venv_path"
    python3 -m venv "$venv_path"

    # Activate and install requirements
    log_info "Activating virtual environment and installing dependencies..."
    source "$venv_path/bin/activate"

    # Upgrade pip
    pip install --upgrade pip

    # Install requirements if file exists
    if [ -n "$requirements_file" ]; then
        pip install -r "$requirements_file"
        log_success "Installed requirements from $requirements_file"
    else
        log_warning "No requirements.txt found, installing basic packages..."
        pip install pytest pytest-cov
    fi

    # Deactivate
    deactivate

    log_success "$module_name setup completed ✅"
    return 0
}

# Show usage
usage() {
    echo "Usage: $0 [OPTIONS] [MODULES...]"
    echo ""
    echo "Options:"
    echo "  --all     Setup all modules (default)"
    echo "  --help    Show this help message"
    echo ""
    echo "Modules:"
    echo "  api      Setup API module"
    echo "  data     Setup Data module"
    echo "  drift    Setup Drift module"
    echo "  training Setup Training module"
    echo "  airflow  Setup Airflow module"
    echo ""
    echo "Examples:"
    echo "  $0              # Setup all modules"
    echo "  $0 api drift   # Setup only API and Drift modules"
}

# Parse arguments
SETUP_MODULES=()
ALL_MODULES=true

while [[ $# -gt 0 ]]; do
    case $1 in
        --all)
            ALL_MODULES=true
            shift
            ;;
        --help)
            usage
            exit 0
            ;;
        api|data|drift|training|airflow)
            SETUP_MODULES+=("$1")
            ALL_MODULES=false
            shift
            ;;
        *)
            log_error "Unknown option or module: $1"
            usage
            exit 1
            ;;
    esac
done

# Default to all modules if none specified
if [ "$ALL_MODULES" = true ] || [ ${#SETUP_MODULES[@]} -eq 0 ]; then
    SETUP_MODULES=("api" "data" "drift" "training" "airflow")
fi

# Main execution
main() {
    echo ""
    echo "=========================================="
    echo "  SETUP LOCAL DEVELOPMENT ENVIRONMENTS"
    echo "=========================================="
    echo ""
    log_info "Setting up modules: ${SETUP_MODULES[*]}"
    echo ""

    FAILED_MODULES=()
    SUCCESS_COUNT=0
    TOTAL_COUNT=${#SETUP_MODULES[@]}

    # Setup each module
    for module in "${SETUP_MODULES[@]}"; do
        if setup_module_venv "$module" "${MODULES[$module]}"; then
            ((SUCCESS_COUNT++))
        else
            FAILED_MODULES+=("$module")
        fi
        echo ""
    done

    # Summary
    echo "=========================================="
    echo "  SETUP SUMMARY"
    echo "=========================================="
    echo ""
    echo "Total modules: $TOTAL_COUNT"
    echo "Successful: $SUCCESS_COUNT"
    echo "Failed: $(($TOTAL_COUNT - $SUCCESS_COUNT))"

    if [ ${#FAILED_MODULES[@]} -eq 0 ]; then
        log_success "🎉 All environments setup successfully!"
        echo ""
        log_info "You can now run local tests with:"
        echo "  ./run_all_local_tests.sh"
        echo "  ./test_api_local.sh --all"
        echo "  ./test_data_local.sh --all"
        echo "  etc."
        exit 0
    else
        log_error "❌ Failed modules: ${FAILED_MODULES[*]}"
        exit 1
    fi
}

# Run main
main