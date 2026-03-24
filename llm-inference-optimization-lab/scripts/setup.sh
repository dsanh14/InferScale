#!/usr/bin/env bash
set -euo pipefail

echo "=== LLM Inference Optimization Lab — Setup ==="

# Create virtual environment if it doesn't exist
if [ ! -d ".venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv .venv
fi

echo "Activating virtual environment..."
source .venv/bin/activate

echo "Installing dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

echo "Creating required directories..."
mkdir -p logs results

echo "Copying .env.example to .env (if .env doesn't exist)..."
[ -f .env ] || cp .env.example .env

echo ""
echo "Setup complete. Activate with: source .venv/bin/activate"
