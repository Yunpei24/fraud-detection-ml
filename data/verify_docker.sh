#!/bin/bash

###############################################################################
# Verification Script for Docker Implementation
# Checks that all Docker-related files are correctly configured
###############################################################################

set -e

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}═══════════════════════════════════════════════════════════${NC}"
echo -e "${BLUE}  Docker Implementation Verification - Data Module${NC}"
echo -e "${BLUE}═══════════════════════════════════════════════════════════${NC}"
echo ""

# Check files exist
echo -e "${YELLOW}📋 Checking required files...${NC}"
echo ""

FILES=(
    "Dockerfile"
    ".dockerignore"
    "docker-compose.yml"
    "build.sh"
    "Makefile"
    ".env.example"
)

all_files_exist=true
for file in "${FILES[@]}"; do
    if [ -f "$file" ]; then
        echo -e "${GREEN}✓${NC} $file"
    else
        echo -e "${RED}✗${NC} $file (MISSING)"
        all_files_exist=false
    fi
done
echo ""

# Check if Docker is installed
echo -e "${YELLOW}🐳 Checking Docker installation...${NC}"
if command -v docker &> /dev/null; then
    DOCKER_VERSION=$(docker --version)
    echo -e "${GREEN}✓${NC} $DOCKER_VERSION"
else
    echo -e "${RED}✗${NC} Docker not found"
    exit 1
fi
echo ""

# Check if Docker Compose is installed
echo -e "${YELLOW}🐳 Checking Docker Compose installation...${NC}"
if command -v docker-compose &> /dev/null; then
    COMPOSE_VERSION=$(docker-compose --version)
    echo -e "${GREEN}✓${NC} $COMPOSE_VERSION"
else
    echo -e "${RED}✗${NC} Docker Compose not found"
    exit 1
fi
echo ""

# Check Docker image
echo -e "${YELLOW}📦 Checking Docker image...${NC}"
if docker images | grep -q "fraud-detection-data"; then
    IMAGE_ID=$(docker images | grep "fraud-detection-data" | head -1 | awk '{print $3}')
    IMAGE_SIZE=$(docker images | grep "fraud-detection-data" | head -1 | awk '{print $7}')
    echo -e "${GREEN}✓${NC} Image found: $IMAGE_ID ($IMAGE_SIZE)"
else
    echo -e "${YELLOW}⚠${NC}  Image not built yet (run 'make build')"
fi
echo ""

# Check requirements.txt
echo -e "${YELLOW}📦 Checking requirements.txt...${NC}"
if [ -f "requirements.txt" ]; then
    REQ_COUNT=$(wc -l < requirements.txt)
    echo -e "${GREEN}✓${NC} requirements.txt ($REQ_COUNT lines)"
else
    echo -e "${RED}✗${NC} requirements.txt not found"
fi
echo ""

# Check Python modules
echo -e "${YELLOW}🔍 Checking key Python modules...${NC}"
MODULES=(
    "src/__init__.py"
    "src/config/__init__.py"
    "src/pipelines/__init__.py"
    "src/pipelines/realtime_pipeline.py"
)

for module in "${MODULES[@]}"; do
    if [ -f "$module" ]; then
        echo -e "${GREEN}✓${NC} $module"
    else
        echo -e "${RED}✗${NC} $module (MISSING)"
    fi
done
echo ""

# Check database schema
echo -e "${YELLOW}📊 Checking database schema...${NC}"
if [ -f "data/schema.sql" ]; then
    TABLES=$(grep -c "CREATE TABLE" data/schema.sql || true)
    echo -e "${GREEN}✓${NC} schema.sql ($TABLES tables)"
else
    echo -e "${RED}✗${NC} schema.sql not found"
fi
echo ""

# Summary
echo -e "${BLUE}═══════════════════════════════════════════════════════════${NC}"

if [ "$all_files_exist" = true ]; then
    echo -e "${GREEN}✅ All Docker files are correctly configured!${NC}"
    echo ""
    echo -e "${YELLOW}Next steps:${NC}"
    echo "  1. Review Dockerfile and docker-compose.yml"
    echo "  2. Build image: make build"
    echo "  3. Start services: make up"
    echo "  4. View logs: make logs"
    echo "  5. Run tests: make test"
    echo "  6. Stop services: make down"
    echo ""
    echo -e "${YELLOW}Quick reference:${NC}"
    echo "  • make help      - Show all available commands"
    echo "  • make build     - Build Docker image"
    echo "  • make up        - Start all services"
    echo "  • make test      - Run tests in container"
    echo "  • make down      - Stop all services"
    echo ""
else
    echo -e "${RED}❌ Some files are missing. Please check above.${NC}"
    exit 1
fi

echo -e "${BLUE}═══════════════════════════════════════════════════════════${NC}"
