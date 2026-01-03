#!/bin/bash

# Polygon Real-time Streamer Demo Script
# This script demonstrates all the features of the professional streaming dashboard

echo "🚀 Starting Polygon Real-time Streamer Demo"
echo "=============================================="

# Check if rich library is available
if python -c "import rich" 2>/dev/null; then
    echo "✅ Rich library available - Full display mode enabled"
    DISPLAY_FLAG=""
else
    echo "⚠️  Rich library not available - Basic logging mode"
    echo "   Install with: pip install -r requirements-rich.txt"
    DISPLAY_FLAG="--no-display"
fi

echo ""
echo "📊 Available Demo Modes:"
echo "1. Full Dashboard (quotes + trades, 4 FPS refresh)"
echo "2. Quotes Only (5-second batching)"
echo "3. High-Frequency Mode (10 FPS refresh)"
echo "4. Basic Mode (no display, logging only)"
echo ""

# Default demo mode
DEMO_MODE=${1:-1}

case $DEMO_MODE in
    1)
        echo "🎯 Running Full Dashboard Mode..."
        python polygon_realtime_streamer.py \
            --symbols AAPL MSFT GOOGL NVDA TSLA AMZN META NFLX \
            --feed both \
            --batch-interval 5 \
            --display-refresh 4 \
            --db-server localhost:9100 \
            --test-mode 60
        ;;
    2)
        echo "📈 Running Quotes-Only Mode..."
        python polygon_realtime_streamer.py \
            --symbols AAPL MSFT GOOGL NVDA TSLA \
            --feed quotes \
            --batch-interval 5 \
            --display-refresh 2 \
            --db-server localhost:9100 \
            --test-mode 30
        ;;
    3)
        echo "⚡ Running High-Frequency Mode..."
        python polygon_realtime_streamer.py \
            --symbols AAPL MSFT GOOGL \
            --feed both \
            --batch-interval 1 \
            --display-refresh 10 \
            --db-server localhost:9100 \
            --test-mode 45
        ;;
    4)
        echo "📝 Running Basic Mode..."
        python polygon_realtime_streamer.py \
            --symbols AAPL MSFT GOOGL \
            --feed both \
            --no-display \
            --db-server localhost:9100 \
            --test-mode 20
        ;;
    *)
        echo "❌ Invalid demo mode. Use 1, 2, 3, or 4"
        exit 1
        ;;
esac

echo ""
echo "✅ Demo completed!"
echo "💡 Try different modes: ./run_streamer_demo.sh [1-4]"


