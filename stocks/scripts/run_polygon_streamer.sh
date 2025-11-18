#!/bin/bash
# Quick start script for Polygon Real-time Streamer

set -e

echo "🚀 Polygon Real-time Streamer Quick Start"
echo "========================================"

# Check if Polygon API key is set
if [ -z "$POLYGON_API_KEY" ]; then
    echo "❌ POLYGON_API_KEY environment variable not set"
    echo "Please set it with: export POLYGON_API_KEY='your_api_key_here'"
    exit 1
fi

echo "✅ POLYGON_API_KEY is set"

# Check if database server is running
if curl -s "http://localhost:8080/health" > /dev/null 2>&1; then
    echo "✅ Database server is running"
else
    echo "⚠️  Database server is not running at localhost:8080"
    echo "Start it with: python db_server.py --db-file data/stocks.db --port 8080"
    echo ""
    read -p "Continue anyway? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

echo ""
echo "Starting Polygon Real-time Streamer..."
echo "Press Ctrl+C to stop"
echo ""

# Default: Stream major tech stocks
DEFAULT_SYMBOLS="AAPL MSFT GOOGL AMZN TSLA NVDA META NFLX"

# Allow override via command line arguments
if [ $# -gt 0 ]; then
    SYMBOLS="$@"
else
    SYMBOLS="$DEFAULT_SYMBOLS"
fi

echo "Streaming symbols: $SYMBOLS"
echo ""

# Run the streamer
python polygon_realtime_streamer.py \
    --symbols $SYMBOLS \
    --feed both \
    --symbols-per-connection 4 \
    --stats-interval 30 \
    --log-level INFO \
    --db-server localhost:8080


