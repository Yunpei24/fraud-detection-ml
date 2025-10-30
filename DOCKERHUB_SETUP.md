# Docker Hub Configuration for Local Development
# Set your Docker Hub username to use pre-built images from CI/CD pipeline

# Option 1: Set environment variable (recommended)
export DOCKERHUB_USERNAME=your_dockerhub_username

# Option 2: Or edit docker-compose.local.yml directly and replace:
# ${DOCKERHUB_USERNAME:-yoshua24} with your actual username

# Available image tags from CI/CD:
# - develop (for develop branch pushes)
# - latest (for main branch pushes)
# - main (for main branch)
# - develop-<sha> (commit-specific tags)
# - <pr-number> (for pull requests)

# Example usage:
# docker-compose -f docker-compose.local.yml up -d api
# This will pull yoshua24/api:develop from Docker Hub