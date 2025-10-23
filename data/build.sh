#!/bin/bash

###############################################################################
# Docker Build Script for Fraud Detection Data Module
# 
# Usage:
#   ./build.sh [tag] [registry]
#
# Examples:
#   ./build.sh latest              # Build as localhost:5000/fraud-detection-data:latest
#   ./build.sh v1.0.0 myregistry   # Build as myregistry/fraud-detection-data:v1.0.0
#
###############################################################################

set -e

# Configuration
IMAGE_NAME="fraud-detection-data"
IMAGE_TAG="${1:-latest}"
REGISTRY="${2:-localhost:5000}"
FULL_IMAGE="${REGISTRY}/${IMAGE_NAME}:${IMAGE_TAG}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}═══════════════════════════════════════════════════════════${NC}"
echo -e "${BLUE}  Docker Build: Fraud Detection Data Module${NC}"
echo -e "${BLUE}═══════════════════════════════════════════════════════════${NC}"
echo ""

# Check if Dockerfile exists
if [ ! -f "Dockerfile" ]; then
    echo -e "${RED}✗ Error: Dockerfile not found in current directory${NC}"
    exit 1
fi

echo -e "${YELLOW}📋 Configuration:${NC}"
echo "   Image Name:  ${IMAGE_NAME}"
echo "   Tag:         ${IMAGE_TAG}"
echo "   Registry:    ${REGISTRY}"
echo "   Full Image:  ${FULL_IMAGE}"
echo ""

# Get Docker version
DOCKER_VERSION=$(docker --version 2>/dev/null || echo "Docker not found")
echo -e "${YELLOW}🐳 ${DOCKER_VERSION}${NC}"
echo ""

# Start build
echo -e "${YELLOW}🔨 Building Docker image...${NC}"
echo ""

if docker build \
    --tag "${FULL_IMAGE}" \
    --tag "${REGISTRY}/${IMAGE_NAME}:latest" \
    --build-arg BUILD_DATE="$(date -u +'%Y-%m-%dT%H:%M:%SZ')" \
    --build-arg VCS_REF="$(git rev-parse --short HEAD 2>/dev/null || echo 'unknown')" \
    -f Dockerfile \
    .; then
    
    echo ""
    echo -e "${GREEN}✓ Build successful!${NC}"
    echo ""
    echo -e "${BLUE}═══════════════════════════════════════════════════════════${NC}"
    echo -e "${GREEN}✓ Image built: ${FULL_IMAGE}${NC}"
    echo -e "${BLUE}═══════════════════════════════════════════════════════════${NC}"
    echo ""
    
    # Show image info
    echo -e "${YELLOW}📊 Image Information:${NC}"
    docker images "${REGISTRY}/${IMAGE_NAME}" --format "table {{.Repository}}\t{{.Tag}}\t{{.Size}}\t{{.Created}}" | head -2
    echo ""
    
    # Provide next steps
    echo -e "${YELLOW}🚀 Next Steps:${NC}"
    echo ""
    echo "   Development (local docker-compose):"
    echo "   $ docker-compose up -d"
    echo ""
    echo "   Manual run:"
    echo "   $ docker run -it ${FULL_IMAGE}"
    echo ""
    echo "   Push to registry:"
    echo "   $ docker push ${FULL_IMAGE}"
    echo ""
    echo "   Kubernetes deployment:"
    echo "   $ kubectl set image deployment/data-pipeline data-pipeline=${FULL_IMAGE}"
    echo ""
    
else
    echo ""
    echo -e "${RED}✗ Build failed!${NC}"
    exit 1
fi
