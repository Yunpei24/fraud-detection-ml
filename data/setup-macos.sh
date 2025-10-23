#!/bin/bash
# macOS Setup Script for Data Module
# Handles cmake and OpenSSL issues with azure-eventhub installation

set -e

echo "🔧 Setting up Data Module for macOS..."
echo "========================================"

# Check Python version
PYTHON_VERSION=$(python --version 2>&1 | awk '{print $2}')
echo "📍 Python version: $PYTHON_VERSION"

# Check if cmake is installed
if ! command -v cmake &> /dev/null; then
    echo "❌ cmake not found. Installing via brew..."
    brew install cmake
else
    echo "✅ cmake already installed"
fi

# Check if OpenSSL is installed
if ! command -v openssl &> /dev/null; then
    echo "❌ OpenSSL not found. Installing via brew..."
    brew install openssl
else
    echo "✅ OpenSSL already installed"
fi

echo ""
echo "📦 Installing base requirements..."
pip install -r requirements.txt

echo ""
echo "✅ Base requirements installed successfully!"
echo ""
echo "Optional: Install Azure Event Hub support"
echo "Run: pip install -r requirements-azure.txt"
echo ""
echo "Or for macOS with cmake:"
echo "export LDFLAGS=\"-L/usr/local/opt/openssl/lib\""
echo "export CPPFLAGS=\"-I/usr/local/opt/openssl/include\""
echo "pip install -r requirements-azure.txt"
