#!/bin/bash
set -e

[ ! -d "venv_api" ] && python3 -m venv venv_api
source venv_api/bin/activate
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
[ ! -f ".env" ] && cp .env.example .env
[ ! -d "models" ] && mkdir -p models
echo "✅ Setup complete"
