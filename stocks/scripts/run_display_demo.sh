#!/bin/bash

# Stock Display Dashboard Demo Script
# This script demonstrates the display-only functionality

echo "📊 Starting Stock Display Dashboard Demo"
echo "========================================"

# Check if rich library is available
if python -c "import rich" 2>/dev/null; then
    echo "✅ Rich library available - Display mode enabled"
else
    echo "❌ Rich library not available - Install with: pip install rich"
    exit 1
fi

echo ""
echo "📊 Available Display Modes:"
echo "1. Dedicated Display Dashboard (using scripts/stock_display_dashboard.py)"
echo "2. High-Frequency Display (1 FPS refresh)"
echo "3. Standard Display (2 FPS refresh)"
echo "4. Multi-Symbol Display (6+ symbols)"
echo ""

# Default demo mode
DEMO_MODE=${1:-1}
SYMBOLS="CART UBER TQQQ NFLX AMZN GOOG AAPL MSFT NVDA SPY QQQ META"

case $DEMO_MODE in
    1)
        echo "📈 Running Dedicated Display Dashboard..."
        python scripts/stock_display_dashboard.py \
            --symbols ${SYMBOLS} \
            --display-refresh 2 \
            --db-server ms1.kundu.dev:9100 \
            --test-mode 30
        ;;
    2)
        echo "⚡ Running High-Frequency Display..."
        python scripts/stock_display_dashboard.py \
            --symbols  ${SYMBOLS} \
            --display-refresh 1 \
            --db-server ms1.kundu.dev:9100 \
            --test-mode 20
        ;;
    3)
        echo "📊 Running Standard Display..."
        python scripts/stock_display_dashboard.py \
            --symbols  ${SYMBOLS} \
            --display-refresh 2 \
            --db-server ms1.kundu.dev:9100 \
            --test-mode 45
        ;;
    4)
        echo "🌐 Running Multi-Symbol Display..."
        python scripts/stock_display_dashboard.py \
            --symbols AAPL MSFT GOOGL NVDA TSLA AMZN META NFLX SPY QQQ \
            --display-refresh 1 \
            --db-server ms1.kundu.dev:9100 \
            --test-mode 60
        ;;
    *)
        echo "❌ Invalid demo mode. Use 1, 2, 3, or 4"
        exit 1
        ;;
esac

echo ""
echo "✅ Display demo completed!"
echo "💡 Try different modes: ./run_display_demo.sh [1-4]"
echo ""
echo "🔧 Usage Examples:"
echo "   # Dedicated display dashboard"
echo "   python scripts/stock_display_dashboard.py --symbols AAPL MSFT --display-refresh 1"
echo ""
echo "   # Stream data separately (in another terminal)"
echo "   python polygon_realtime_streamer.py --symbols AAPL MSFT --feed both --db-server ms1.kundu.dev:9100"
