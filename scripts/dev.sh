#!/usr/bin/env bash
set -euo pipefail

# Development setup and validation script
echo "=== Setting up TR-RU Trade Forecasting Platform ==="

# 1. Check Python and Node versions
echo "Python version: $(python3 --version)"
echo "Node version: $(node --version)"
echo "NPM version: $(npm --version)"

# 2. Virtual environment setup
if [ ! -d ".venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv .venv
fi

echo "Activating virtual environment..."
source .venv/bin/activate

echo "Installing Python dependencies..."
pip install --upgrade pip
pip install -e ".[dev]"

# 3. Frontend setup
if [ -d "frontend" ]; then
    echo "Installing frontend dependencies..."
    npm --prefix frontend install
fi

# 4. Run tests and lint
echo "Running pytest..."
pytest tests/

echo "Running ruff check..."
ruff check .

echo "=== Setup and verification completed successfully! ==="
