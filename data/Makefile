.PHONY: help build build-prod up down logs test clean shell verify push

# Colors
BLUE := \033[0;34m
GREEN := \033[0;32m
YELLOW := \033[1;33m
RED := \033[0;31m
NC := \033[0m # No Color

# Configuration
IMAGE_NAME := fraud-detection-data
REGISTRY := localhost:5000
TAG ?= latest

help:
	@echo "$(BLUE)╔═══════════════════════════════════════════════════════════╗$(NC)"
	@echo "$(BLUE)║   Fraud Detection Data Module - Docker Commands           ║$(NC)"
	@echo "$(BLUE)╚═══════════════════════════════════════════════════════════╝$(NC)"
	@echo ""
	@echo "$(YELLOW)Build Commands:$(NC)"
	@echo "  make build              Build Docker image (dev)"
	@echo "  make build-prod         Build Docker image (production)"
	@echo ""
	@echo "$(YELLOW)Docker Compose:$(NC)"
	@echo "  make up                 Start all services (docker-compose up -d)"
	@echo "  make down               Stop all services (docker-compose down)"
	@echo "  make logs               View pipeline logs (docker-compose logs -f)"
	@echo "  make shell              Open shell in data_pipeline container"
	@echo ""
	@echo "$(YELLOW)Testing:$(NC)"
	@echo "  make test               Run tests in container"
	@echo "  make verify             Run verify.py in container"
	@echo ""
	@echo "$(YELLOW)Registry:$(NC)"
	@echo "  make push               Push image to registry"
	@echo ""
	@echo "$(YELLOW)Maintenance:$(NC)"
	@echo "  make clean              Remove containers and volumes"
	@echo "  make help               Show this help message"
	@echo ""
	@echo "$(YELLOW)Examples:$(NC)"
	@echo "  make build TAG=v1.0.0"
	@echo "  make up"
	@echo "  make test"
	@echo "  make down"
	@echo ""

# Build commands
build:
	@echo "$(BLUE)🐳 Building Docker image (dev)...$(NC)"
	./build.sh $(TAG) $(REGISTRY)

build-prod:
	@echo "$(BLUE)🐳 Building Docker image (production)...$(NC)"
	@echo "$(YELLOW)⚠️  Building with optimizations for production$(NC)"
	./build.sh $(TAG) $(REGISTRY)

# Docker Compose commands
up:
	@echo "$(GREEN)🚀 Starting services with docker-compose...$(NC)"
	docker-compose up -d
	@echo ""
	@echo "$(GREEN)✓ Services started!$(NC)"
	@echo ""
	@echo "$(YELLOW)Services:$(NC)"
	@echo "  • PostgreSQL:    localhost:5432"
	@echo "  • Redis:         localhost:6379"
	@echo "  • Data Pipeline: localhost:8000"
	@echo ""
	@echo "$(YELLOW)View logs:$(NC)"
	@echo "  make logs"
	@echo ""

down:
	@echo "$(YELLOW)⏹️  Stopping services...$(NC)"
	docker-compose down
	@echo "$(GREEN)✓ Services stopped$(NC)"

logs:
	@echo "$(BLUE)📋 Showing logs (data_pipeline)...$(NC)"
	docker-compose logs -f data_pipeline

logs-all:
	@echo "$(BLUE)📋 Showing all logs...$(NC)"
	docker-compose logs -f

# Testing commands
test:
	@echo "$(BLUE)🧪 Running tests in container...$(NC)"
	docker-compose exec -T data_pipeline pytest tests/ -v --tb=short
	@echo ""
	@echo "$(GREEN)✓ Tests completed$(NC)"

test-coverage:
	@echo "$(BLUE)🧪 Running tests with coverage...$(NC)"
	docker-compose exec -T data_pipeline pytest tests/ -v --cov=src --cov-report=html
	@echo ""
	@echo "$(GREEN)✓ Coverage report: htmlcov/index.html$(NC)"

verify:
	@echo "$(BLUE)✓ Running verification script...$(NC)"
	docker-compose exec -T data_pipeline python verify.py
	@echo "$(GREEN)✓ Verification completed$(NC)"

# Shell access
shell:
	@echo "$(BLUE)🔓 Opening shell in data_pipeline container...$(NC)"
	@echo "$(YELLOW)Type 'exit' to quit$(NC)"
	docker-compose exec data_pipeline /bin/bash

shell-root:
	@echo "$(BLUE)🔓 Opening root shell in data_pipeline container...$(NC)"
	@echo "$(YELLOW)Type 'exit' to quit$(NC)"
	docker-compose exec -u root data_pipeline /bin/bash

# Registry commands
push:
	@echo "$(BLUE)📤 Pushing image to registry...$(NC)"
	docker push $(REGISTRY)/$(IMAGE_NAME):$(TAG)
	docker push $(REGISTRY)/$(IMAGE_NAME):latest
	@echo "$(GREEN)✓ Push completed$(NC)"

push-prod:
	@echo "$(BLUE)📤 Pushing production image to registry...$(NC)"
	@echo "$(YELLOW)Registry: $(REGISTRY)$(NC)"
	@echo "$(YELLOW)Image: $(IMAGE_NAME):$(TAG)$(NC)"
	docker push $(REGISTRY)/$(IMAGE_NAME):$(TAG)
	docker push $(REGISTRY)/$(IMAGE_NAME):latest
	@echo "$(GREEN)✓ Production image pushed$(NC)"

# Maintenance commands
clean:
	@echo "$(YELLOW)🧹 Cleaning up containers and volumes...$(NC)"
	docker-compose down -v
	@echo "$(GREEN)✓ Cleanup completed$(NC)"
	@echo ""
	@echo "$(YELLOW)Python cache cleanup:$(NC)"
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	find . -type f -name ".pytest_cache" -delete
	@echo "$(GREEN)✓ Python cache cleaned$(NC)"

prune:
	@echo "$(YELLOW)🧹 Docker system prune (remove unused images/volumes)...$(NC)"
	docker system prune -f
	@echo "$(GREEN)✓ Prune completed$(NC)"

# Status commands
status:
	@echo "$(BLUE)📊 Docker Compose Status:$(NC)"
	docker-compose ps
	@echo ""
	@echo "$(BLUE)🐳 Docker Images:$(NC)"
	docker images | grep $(IMAGE_NAME) || echo "No images built yet"
	@echo ""
	@echo "$(BLUE)💾 Docker Volumes:$(NC)"
	docker volume ls | grep fraud_detection || echo "No volumes created yet"

# Restart
restart: down up
	@echo "$(GREEN)✓ Services restarted$(NC)"

# Check dependencies
check:
	@echo "$(BLUE)🔍 Checking dependencies...$(NC)"
	@command -v docker >/dev/null 2>&1 || (echo "$(RED)✗ Docker not installed$(NC)" && exit 1)
	@command -v docker-compose >/dev/null 2>&1 || (echo "$(RED)✗ Docker Compose not installed$(NC)" && exit 1)
	@echo "$(GREEN)✓ Docker:$(NC) $$(docker --version)"
	@echo "$(GREEN)✓ Docker Compose:$(NC) $$(docker-compose --version)"
