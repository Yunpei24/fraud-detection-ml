.PHONY: help install install-dev test test-coverage test-unit test-integration test-performance lint format type-check security clean docker-build docker-run docker-push run-local docs

# Variables
PYTHON := python3
PIP := pip3
PYTEST := pytest
BLACK := black
FLAKE8 := flake8
MYPY := mypy
PYLINT := pylint
DOCKER_IMAGE := fraud-detection-drift:latest
DOCKER_REGISTRY := yoshua24
VENV := venv
SRC_DIR := src
TESTS_DIR := tests
DOCS_DIR := docs

# Colors for output
BLUE := \033[0;34m
GREEN := \033[0;32m
YELLOW := \033[0;33m
RED := \033[0;31m
NC := \033[0m # No Color

help: ## Show this help message
	@echo "$(BLUE)Drift Detection Module - Available Commands$(NC)"
	@echo ""
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "$(GREEN)%-20s$(NC) %s\n", $$1, $$2}'

# ============================================================================
# Installation & Setup
# ============================================================================

install: ## Install production dependencies
	@echo "$(BLUE)Installing production dependencies...$(NC)"
	$(PIP) install -r requirements.txt
	@echo "$(GREEN)✓ Installation complete$(NC)"

install-dev: ## Install all dependencies (prod + dev)
	@echo "$(BLUE)Installing all dependencies (prod + dev)...$(NC)"
	$(PIP) install -r requirements.txt
	@echo "$(GREEN)✓ All dependencies installed$(NC)"

venv: ## Create virtual environment
	@echo "$(BLUE)Creating virtual environment...$(NC)"
	$(PYTHON) -m venv $(VENV)
	@echo "$(GREEN)✓ Virtual environment created$(NC)"
	@echo "$(YELLOW)Activate with: source $(VENV)/bin/activate$(NC)"

# ============================================================================
# Testing
# ============================================================================

test: ## Run all tests (unit + integration)
	@echo "$(BLUE)Running all tests...$(NC)"
	$(PYTEST) $(TESTS_DIR) -v --tb=short
	@echo "$(GREEN)✓ All tests passed$(NC)"

test-unit: ## Run unit tests only
	@echo "$(BLUE)Running unit tests...$(NC)"
	$(PYTEST) $(TESTS_DIR)/unit -v --tb=short
	@echo "$(GREEN)✓ Unit tests passed$(NC)"

test-integration: ## Run integration tests only
	@echo "$(BLUE)Running integration tests...$(NC)"
	$(PYTEST) $(TESTS_DIR)/integration -v --tb=short
	@echo "$(GREEN)✓ Integration tests passed$(NC)"

test-coverage: ## Run tests with coverage report
	@echo "$(BLUE)Running tests with coverage...$(NC)"
	$(PYTEST) $(TESTS_DIR) --cov=$(SRC_DIR) --cov-report=html --cov-report=term-missing -v
	@echo "$(GREEN)✓ Coverage report generated: htmlcov/index.html$(NC)"

test-performance: ## Run performance benchmarks
	@echo "$(BLUE)Running performance benchmarks...$(NC)"
	$(PYTEST) $(TESTS_DIR) -v -m benchmark --benchmark-only
	@echo "$(GREEN)✓ Performance benchmarks complete$(NC)"

test-watch: ## Run tests in watch mode
	@echo "$(BLUE)Running tests in watch mode...$(NC)"
	$(PYTEST) $(TESTS_DIR) -v --looponfail

# ============================================================================
# Code Quality
# ============================================================================

lint: ## Run all linters (flake8, pylint)
	@echo "$(BLUE)Running linters...$(NC)"
	$(FLAKE8) $(SRC_DIR) $(TESTS_DIR) --max-line-length=120 --ignore=E203,W503
	$(PYLINT) $(SRC_DIR) --fail-under=8.0 --disable=C0111,C0103,R0913
	@echo "$(GREEN)✓ Linting passed$(NC)"

format: ## Format code with black
	@echo "$(BLUE)Formatting code with black...$(NC)"
	$(BLACK) $(SRC_DIR) $(TESTS_DIR) --line-length=120
	@echo "$(GREEN)✓ Code formatted$(NC)"

type-check: ## Run type checking with mypy
	@echo "$(BLUE)Running type checks...$(NC)"
	$(MYPY) $(SRC_DIR) --ignore-missing-imports --no-error-summary
	@echo "$(GREEN)✓ Type checks passed$(NC)"

security: ## Run security checks
	@echo "$(BLUE)Running security checks...$(NC)"
	@command -v bandit >/dev/null 2>&1 || (echo "$(YELLOW)Installing bandit...$(NC)" && $(PIP) install bandit)
	bandit -r $(SRC_DIR) -ll
	@echo "$(GREEN)✓ Security checks passed$(NC)"

quality: lint format type-check ## Run all code quality checks
	@echo "$(GREEN)✓ All quality checks passed$(NC)"

# ============================================================================
# Docker
# ============================================================================

docker-build: ## Build Docker image
	@echo "$(BLUE)Building Docker image: $(DOCKER_IMAGE)$(NC)"
	docker build -t $(DOCKER_IMAGE) -f Dockerfile .
	@echo "$(GREEN)✓ Docker image built$(NC)"

docker-run: docker-build ## Build and run Docker container locally
	@echo "$(BLUE)Running Docker container...$(NC)"
	docker run -it --rm \
		-v $(PWD):/app \
		-e ENV=development \
		$(DOCKER_IMAGE)
	@echo "$(GREEN)✓ Docker container stopped$(NC)"

docker-push: docker-build ## Build and push Docker image to registry
	@echo "$(BLUE)Pushing Docker image to registry...$(NC)"
	docker tag $(DOCKER_IMAGE) $(DOCKER_REGISTRY)/$(DOCKER_IMAGE)
	docker push $(DOCKER_REGISTRY)/$(DOCKER_IMAGE)
	@echo "$(GREEN)✓ Docker image pushed$(NC)"

# ============================================================================
# Documentation
# ============================================================================

docs: ## Generate documentation with Sphinx (if available)
	@echo "$(BLUE)Generating documentation...$(NC)"
	@if [ -d "$(DOCS_DIR)" ]; then \
		command -v sphinx-build >/dev/null 2>&1 && \
		sphinx-build -b html $(DOCS_DIR) $(DOCS_DIR)/_build || \
		echo "$(YELLOW)Sphinx not installed. Skipping documentation generation.$(NC)"; \
	else \
		echo "$(YELLOW)Documentation directory not found.$(NC)"; \
	fi
	@echo "$(GREEN)✓ Documentation ready$(NC)"

# ============================================================================
# Development
# ============================================================================

run-local: ## Run the drift detection service locally
	@echo "$(BLUE)Starting drift detection service locally...$(NC)"
	$(PYTHON) -m $(SRC_DIR).main
	@echo "$(GREEN)✓ Service stopped$(NC)"

debug: ## Run with debug logging
	@echo "$(BLUE)Running with debug logging...$(NC)"
	DEBUG=1 $(PYTHON) -m $(SRC_DIR).main
	@echo "$(GREEN)✓ Debug session ended$(NC)"

# ============================================================================
# Cleanup
# ============================================================================

clean: ## Remove build artifacts, cache, and temporary files
	@echo "$(BLUE)Cleaning up...$(NC)"
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type f -name "*.pyd" -delete
	find . -type f -name ".coverage" -delete
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".mypy_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "htmlcov" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "dist" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "build" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.tmp" -delete
	find . -type f -name "*.log" -delete
	@echo "$(GREEN)✓ Cleanup complete$(NC)"

clean-venv: ## Remove virtual environment
	@echo "$(BLUE)Removing virtual environment...$(NC)"
	rm -rf $(VENV)
	@echo "$(GREEN)✓ Virtual environment removed$(NC)"

clean-all: clean clean-venv ## Remove all generated files and virtual environment
	@echo "$(GREEN)✓ Complete cleanup done$(NC)"

# ============================================================================
# CI/CD Pipeline
# ============================================================================

ci: ## Run full CI pipeline (install, lint, test, coverage)
	@echo "$(BLUE)Running CI pipeline...$(NC)"
	@echo "$(YELLOW)Step 1: Installing dependencies...$(NC)"
	$(MAKE) install-dev
	@echo "$(YELLOW)Step 2: Running linters...$(NC)"
	$(MAKE) lint
	@echo "$(YELLOW)Step 3: Running type checks...$(NC)"
	$(MAKE) type-check
	@echo "$(YELLOW)Step 4: Running tests with coverage...$(NC)"
	$(MAKE) test-coverage
	@echo "$(GREEN)✓ CI pipeline passed$(NC)"

cd: ci docker-build ## Complete CI/CD pipeline (lint, test, build)
	@echo "$(GREEN)✓ CI/CD pipeline complete$(NC)"

# ============================================================================
# Default target
# ============================================================================

.DEFAULT_GOAL := help
