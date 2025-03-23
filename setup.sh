#!/bin/bash

# Exit on error
set -e

echo "Setting up Trading System..."

# Create virtual environment if it doesn't exist
if [ ! -d ".venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv .venv
fi

# Activate virtual environment
source .venv/bin/activate

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Install dependencies
echo "Installing dependencies..."
pip install -r requirements.txt

# Create necessary directories
echo "Creating directory structure..."
mkdir -p logs
mkdir -p data
mkdir -p config

# Copy example config if it doesn't exist
if [ ! -f "config/config.json" ]; then
    echo "Creating default configuration..."
    cp config.example.json config/config.json
fi

# Make run script executable
chmod +x run_trading_system.py

echo "Setup complete! You can now run the trading system with:"
echo "./run_trading_system.py" 