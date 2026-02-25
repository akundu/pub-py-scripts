#!/bin/bash
#
# Quick Start Script for Continuous Trading Mode
#
# This script helps you launch the continuous mode system.
# It will open two terminal windows:
#   1. Dashboard (Flask web server)
#   2. Continuous mode (market monitor & alerts)
#

STOCKS_DIR="/Volumes/RAID1 NVME SSD 2TB/akundu_programs_dev/programs/python/pythonv3/pub-py-scripts/stocks"

echo "========================================="
echo "Continuous Trading Mode - Quick Start"
echo "========================================="
echo ""
echo "This will launch:"
echo "  1. Web Dashboard (http://localhost:5001)"
echo "  2. Continuous Mode (alerts & monitoring)"
echo ""
echo "Press Enter to continue or Ctrl+C to cancel..."
read

cd "$STOCKS_DIR" || exit 1

# Check if Flask is installed
if ! python -c "import flask" 2>/dev/null; then
    echo ""
    echo "⚠️  Flask not found. Dashboard will not work."
    echo "Install with: pip install flask"
    echo ""
    echo "Continue without dashboard? (y/n)"
    read -r response
    if [[ ! "$response" =~ ^[Yy]$ ]]; then
        echo "Cancelled."
        exit 0
    fi
    RUN_DASHBOARD=false
else
    RUN_DASHBOARD=true
fi

# Check if grid file exists
if [ ! -f "results/backtest_tight/grid_trading_ready.csv" ]; then
    echo ""
    echo "⚠️  Grid file not found!"
    echo "Run this first: python scripts/comprehensive_backtest.py --ticker NDX --backtest-days 90"
    echo ""
    exit 1
fi

echo ""
echo "Starting continuous mode..."
echo ""

# Launch dashboard in background (if Flask available)
if [ "$RUN_DASHBOARD" = true ]; then
    echo "Starting dashboard on http://localhost:5001 ..."
    python scripts/continuous/dashboard.py > logs/continuous/dashboard.log 2>&1 &
    DASHBOARD_PID=$!
    echo "Dashboard PID: $DASHBOARD_PID"
    sleep 2
    echo ""
fi

# Launch continuous mode
echo "Starting continuous mode..."
echo "Ticker: NDX"
echo "Trend: sideways"
echo ""
echo "Press Ctrl+C to stop"
echo ""

# Trap SIGINT to kill dashboard on exit
if [ "$RUN_DASHBOARD" = true ]; then
    trap "echo ''; echo 'Stopping...'; kill $DASHBOARD_PID 2>/dev/null; exit 0" SIGINT
fi

python scripts/continuous/continuous_mode.py --ticker NDX --trend sideways

# Cleanup
if [ "$RUN_DASHBOARD" = true ]; then
    kill $DASHBOARD_PID 2>/dev/null
fi

echo ""
echo "Continuous mode stopped."
