#!/bin/bash
#
# Setup automation for prediction models
# This script helps you configure automated retraining
#

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "========================================================================"
echo "PREDICTION MODEL AUTOMATION SETUP"
echo "========================================================================"
echo ""

# Check if scripts are executable
if [ ! -x "scripts/retrain_models_auto.sh" ]; then
    echo "Making scripts executable..."
    chmod +x scripts/retrain_models_auto.sh
    chmod +x scripts/monitor_model_health.py
    echo "✅ Scripts made executable"
fi

# Create required directories
echo "Creating required directories..."
mkdir -p logs/retraining
mkdir -p logs/health_checks
mkdir -p logs/validation
mkdir -p models/production
mkdir -p results
echo "✅ Directories created"
echo ""

# Test retraining script (dry run)
echo "========================================================================"
echo "Testing retraining script..."
echo "========================================================================"
echo ""
echo "This will check if retraining is needed (won't actually retrain)..."
echo ""

./scripts/retrain_models_auto.sh || true

echo ""
echo "========================================================================"
echo "Testing health monitoring..."
echo "========================================================================"
echo ""

python scripts/monitor_model_health.py

echo ""
echo "========================================================================"
echo "SETUP COMPLETE"
echo "========================================================================"
echo ""
echo "Next steps:"
echo ""
echo "1. Review the cron entries in CRON_SETUP.txt"
echo ""
echo "2. Install cron jobs:"
echo "   crontab -e"
echo "   # Copy entries from CRON_SETUP.txt"
echo "   # Update paths and email address"
echo ""
echo "3. Test manual retraining:"
echo "   ./scripts/retrain_models_auto.sh --force"
echo ""
echo "4. Check model health anytime:"
echo "   python scripts/monitor_model_health.py"
echo ""
echo "5. View automation guide:"
echo "   cat MODEL_RETRAINING_GUIDE.md"
echo ""
echo "========================================================================"
echo ""
