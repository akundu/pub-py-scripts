#!/bin/bash
#
# Live Closing Price Prediction Setup Script
#
# This script helps you set up and run live predictions for NDX/SPX indices.
# It handles: loading historical data, starting continuous updates, and running predictions.
#
# Usage:
#   ./prediction_setup.sh --load-historical     # One-time: Load 365 days of historical data
#   ./prediction_setup.sh --start-continuous    # Start continuous data fetching (background)
#   ./prediction_setup.sh --predict NDX         # Get prediction for NDX
#   ./prediction_setup.sh --predict SPX         # Get prediction for SPX
#   ./prediction_setup.sh --check-data          # Verify database has required data
#   ./prediction_setup.sh --demo NDX            # Test prediction with synthetic data
#   ./prediction_setup.sh --stop-continuous     # Stop background continuous fetching

set -e

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
DB_PATH="${QUESTDB_URL:-questdb://admin:quest@localhost:8812/qdb}"
POLYGON_KEY="${POLYGON_API_KEY:-}"
INDEX_SYMBOLS="I:NDX I:SPX I:VIX1D"
CONTINUOUS_PID_FILE="/tmp/prediction_continuous_fetch.pid"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

check_polygon_key() {
    if [ -z "$POLYGON_KEY" ]; then
        log_error "POLYGON_API_KEY environment variable is not set"
        echo "Please set it with: export POLYGON_API_KEY=your_api_key_here"
        exit 1
    fi
}

load_historical() {
    check_polygon_key
    log_info "Loading 365 days of historical data for indices..."
    log_info "Symbols: $INDEX_SYMBOLS"
    log_info "Database: $DB_PATH"
    echo ""

    cd "$PROJECT_ROOT"
    python fetch_all_data.py \
        --symbols $INDEX_SYMBOLS \
        --fetch-market-data \
        --days-back 365 \
        --db-path "$DB_PATH" \
        --data-source polygon \
        --log-level INFO

    log_success "Historical data loaded successfully"
}

start_continuous() {
    check_polygon_key

    if [ -f "$CONTINUOUS_PID_FILE" ]; then
        OLD_PID=$(cat "$CONTINUOUS_PID_FILE")
        if ps -p "$OLD_PID" > /dev/null 2>&1; then
            log_warning "Continuous fetching already running (PID: $OLD_PID)"
            echo "Use --stop-continuous to stop it first"
            exit 1
        fi
    fi

    log_info "Starting continuous data fetching in background..."
    log_info "Symbols: $INDEX_SYMBOLS"
    log_info "Database: $DB_PATH"
    echo ""

    cd "$PROJECT_ROOT"
    nohup python fetch_all_data.py \
        --symbols $INDEX_SYMBOLS \
        --fetch-market-data \
        --continuous \
        --use-market-hours \
        --db-path "$DB_PATH" \
        --data-source polygon \
        --log-level WARNING \
        > /tmp/prediction_continuous_fetch.log 2>&1 &

    echo $! > "$CONTINUOUS_PID_FILE"
    log_success "Continuous fetching started (PID: $!)"
    echo "Log file: /tmp/prediction_continuous_fetch.log"
    echo "Use --stop-continuous to stop"
}

stop_continuous() {
    if [ ! -f "$CONTINUOUS_PID_FILE" ]; then
        log_warning "No continuous fetching process found"
        exit 0
    fi

    PID=$(cat "$CONTINUOUS_PID_FILE")
    if ps -p "$PID" > /dev/null 2>&1; then
        log_info "Stopping continuous fetching (PID: $PID)..."
        kill "$PID"
        rm -f "$CONTINUOUS_PID_FILE"
        log_success "Continuous fetching stopped"
    else
        log_warning "Process $PID is not running"
        rm -f "$CONTINUOUS_PID_FILE"
    fi
}

run_prediction() {
    local ticker="$1"
    local verbose="${2:-}"

    log_info "Running prediction for $ticker..."
    echo ""

    cd "$PROJECT_ROOT"

    if [ "$verbose" = "--verbose" ] || [ "$verbose" = "-v" ]; then
        python scripts/close_prediction_analysis.py --ticker "$ticker" --verbose --db-config "$DB_PATH"
    else
        python scripts/close_prediction_analysis.py --ticker "$ticker" --db-config "$DB_PATH"
    fi
}

run_demo() {
    local ticker="$1"

    log_info "Running demo prediction for $ticker (using synthetic data)..."
    echo ""

    cd "$PROJECT_ROOT"
    python scripts/close_prediction_analysis.py --ticker "$ticker" --demo --verbose
}

check_data() {
    log_info "Checking database for required data..."
    echo ""

    cd "$PROJECT_ROOT"
    python scripts/check_prediction_data.py --db-config "$DB_PATH"
}

show_help() {
    echo "Live Closing Price Prediction Setup Script"
    echo ""
    echo "Usage: $(basename "$0") [OPTION]"
    echo ""
    echo "Options:"
    echo "  --load-historical        Load 365 days of historical data (one-time setup)"
    echo "  --start-continuous       Start continuous data fetching in background"
    echo "  --stop-continuous        Stop background continuous fetching"
    echo "  --predict TICKER         Run prediction for TICKER (NDX or SPX)"
    echo "  --predict TICKER -v      Run prediction with verbose output"
    echo "  --demo TICKER            Run demo prediction with synthetic data"
    echo "  --check-data             Check if database has required data"
    echo "  --help                   Show this help message"
    echo ""
    echo "Environment Variables:"
    echo "  POLYGON_API_KEY          Your Polygon.io API key (required for data fetching)"
    echo "  QUESTDB_URL              QuestDB connection string"
    echo "                           (default: questdb://admin:quest@localhost:8812/qdb)"
    echo ""
    echo "Examples:"
    echo "  # First-time setup"
    echo "  export POLYGON_API_KEY=your_key_here"
    echo "  ./$(basename "$0") --load-historical"
    echo ""
    echo "  # Start continuous updates during market hours"
    echo "  ./$(basename "$0") --start-continuous"
    echo ""
    echo "  # Get current prediction"
    echo "  ./$(basename "$0") --predict NDX"
    echo ""
    echo "  # Test without database"
    echo "  ./$(basename "$0") --demo NDX"
}

# Main
case "${1:-}" in
    --load-historical)
        load_historical
        ;;
    --start-continuous)
        start_continuous
        ;;
    --stop-continuous)
        stop_continuous
        ;;
    --predict)
        if [ -z "${2:-}" ]; then
            log_error "Please specify a ticker (NDX or SPX)"
            exit 1
        fi
        run_prediction "$2" "${3:-}"
        ;;
    --demo)
        if [ -z "${2:-}" ]; then
            log_error "Please specify a ticker (NDX or SPX)"
            exit 1
        fi
        run_demo "$2"
        ;;
    --check-data)
        check_data
        ;;
    --help|-h|"")
        show_help
        ;;
    *)
        log_error "Unknown option: $1"
        echo "Use --help for usage information"
        exit 1
        ;;
esac
