#!/bin/bash
# Quick Start Script for EarthGPT

set -e

echo "=========================================="
echo "EarthGPT Quick Start Setup"
echo "=========================================="

# Check Python version
echo "Checking Python version..."
python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo "Python version: $python_version"

# Create virtual environment
echo "Creating virtual environment..."
python3 -m venv venv

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Install requirements
echo "Installing requirements..."
pip install -r requirements.txt

# Create data directories
echo "Creating data directories..."
mkdir -p data/images
mkdir -p outputs
mkdir -p logs
mkdir -p cache

echo ""
echo "=========================================="
echo "Setup Complete!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "1. Download datasets (DOTA, RSVQA, RSICD)"
echo "2. Run data preprocessing scripts"
echo "3. Start training with: python training/train.py"
echo ""
echo "For more information, see README.md"
echo ""
