#!/bin/bash

# Streak Analysis System Installation Script
# This script installs the streak analysis system and its dependencies

set -e

echo "🚀 Installing Streak Analysis System..."

# Check if Python 3.8+ is available
python_version=$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
required_version="3.8"

if [ "$(printf '%s\n' "$required_version" "$python_version" | sort -V | head -n1)" != "$required_version" ]; then
    echo "❌ Python 3.8+ is required. Found: $python_version"
    exit 1
fi

echo "✅ Python version: $python_version"

# Check if pip is available
if ! command -v pip3 &> /dev/null; then
    echo "❌ pip3 is not available. Please install pip first."
    exit 1
fi

echo "✅ pip3 is available"

# Create virtual environment (optional)
if [ "$1" = "--venv" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv venv
    source venv/bin/activate
    echo "✅ Virtual environment activated"
fi

# Upgrade pip
echo "⬆️  Upgrading pip..."
pip3 install --upgrade pip

# Install dependencies
echo "📦 Installing dependencies..."
pip3 install -r requirements.txt

# Install the package in development mode
echo "🔧 Installing streak analysis package..."
pip3 install -e .

echo "✅ Installation completed successfully!"

# Test the installation
echo "🧪 Testing installation..."
python3 test_basic.py

if [ $? -eq 0 ]; then
    echo "🎉 All tests passed! Installation is working correctly."
else
    echo "⚠️  Some tests failed. Please check the output above."
fi

echo ""
echo "📚 Next steps:"
echo "1. Ensure db_server.py is running on port 9002"
echo "2. Test connection: python -m streak_analysis.cli test-connection"
echo "3. Run analysis: python -m streak_analysis.cli analyze TQQQ --timeframe daily --lookback-days 365"
echo "4. Check examples/ directory for configuration files"
echo "5. Open examples/streak_analysis_demo.ipynb for Jupyter notebook demo"
echo ""
echo "📖 For more information, see README.md"
