#!/bin/bash
# NDX Optimal Credit Spread Strategy Runner
# Based on 6-month analysis with 100% win rate at optimal parameters

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

usage() {
    echo "NDX Optimal Credit Spread Strategy"
    echo ""
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  --mode [both|puts|calls]   Trading mode (default: both)"
    echo "  --trend [up|down|neutral]  Market trend for strategy selection"
    echo "  --start-date YYYY-MM-DD    Start date for backtest"
    echo "  --end-date YYYY-MM-DD      End date for backtest"
    echo "  --live                     Run in continuous mode for live trading"
    echo "  --dry-run                  Show parameters without executing"
    echo "  --help                     Show this help message"
    echo ""
    echo "Trend-Based Strategy Guide:"
    echo "  --trend up     -> Favors PUTS (bullish - you profit if market rises/stays flat)"
    echo "  --trend down   -> Favors CALLS (bearish - you profit if market falls/stays flat)"
    echo "  --trend neutral -> Uses BOTH with equal allocation"
    echo ""
    echo "Examples:"
    echo "  $0 --mode both --start-date 2026-01-27 --end-date 2026-01-31"
    echo "  $0 --trend up --live"
    echo "  $0 --mode puts --start-date 2026-02-03 --end-date 2026-02-03"
    exit 0
}

# Default values
MODE="both"
TREND=""
START_DATE=""
END_DATE=""
LIVE_MODE=false
DRY_RUN=false

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --mode)
            MODE="$2"
            shift 2
            ;;
        --trend)
            TREND="$2"
            shift 2
            ;;
        --start-date)
            START_DATE="$2"
            shift 2
            ;;
        --end-date)
            END_DATE="$2"
            shift 2
            ;;
        --live)
            LIVE_MODE=true
            shift
            ;;
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        --help)
            usage
            ;;
        *)
            echo "Unknown option: $1"
            usage
            ;;
    esac
done

# Apply trend-based mode selection
if [ -n "$TREND" ]; then
    case $TREND in
        up|bullish)
            echo -e "${GREEN}Trend: BULLISH - Favoring PUT credit spreads${NC}"
            echo "  -> Selling puts below market = profit if market rises or stays flat"
            MODE="puts"
            ;;
        down|bearish)
            echo -e "${RED}Trend: BEARISH - Favoring CALL credit spreads${NC}"
            echo "  -> Selling calls above market = profit if market falls or stays flat"
            MODE="calls"
            ;;
        neutral|sideways)
            echo -e "${YELLOW}Trend: NEUTRAL - Using BOTH put and call spreads${NC}"
            echo "  -> Iron condor style = profit if market stays in range"
            MODE="both"
            ;;
        *)
            echo "Unknown trend: $TREND (use up/down/neutral)"
            exit 1
            ;;
    esac
    echo ""
fi

# Set parameters based on mode
case $MODE in
    both)
        PERCENT_BEYOND="0.0330:0.0257"
        MAX_SPREAD_WIDTH="20:25"
        OPTION_TYPE="both"
        RISK_CAP=300000
        echo -e "${BLUE}Mode: BOTH (puts and calls)${NC}"
        echo "  PUT: 3.30% below close, width 15-20"
        echo "  CALL: 2.57% above close, width 20-25"
        ;;
    puts)
        PERCENT_BEYOND="0.0330"
        MAX_SPREAD_WIDTH="20"
        OPTION_TYPE="put"
        RISK_CAP=150000
        echo -e "${GREEN}Mode: PUTS ONLY${NC}"
        echo "  3.30% below close, width 15-20"
        ;;
    calls)
        PERCENT_BEYOND="0.0257"
        MAX_SPREAD_WIDTH="25"
        OPTION_TYPE="call"
        RISK_CAP=150000
        echo -e "${RED}Mode: CALLS ONLY${NC}"
        echo "  2.57% above close, width 20-25"
        ;;
    *)
        echo "Unknown mode: $MODE"
        exit 1
        ;;
esac

echo ""
echo "=== Strategy Parameters ==="
echo "Ticker: NDX"
echo "Percent Beyond: $PERCENT_BEYOND"
echo "Max Spread Width: $MAX_SPREAD_WIDTH"
echo "Risk Cap: \$$(printf "%'d" $RISK_CAP)"
echo "Profit Target: 80%"
echo "Trading Hours: 9:30 AM - 12:00 PM ET"
echo "Dynamic Width: Linear-500 (base=15, slope=500, min=10, max=50)"
echo ""

# Build command
CMD="python analyze_credit_spread_intervals.py"
CMD="$CMD --csv-dir ../options_csv_output"
CMD="$CMD --ticker NDX"
CMD="$CMD --option-type $OPTION_TYPE"
CMD="$CMD --percent-beyond $PERCENT_BEYOND"
CMD="$CMD --max-spread-width $MAX_SPREAD_WIDTH"
CMD="$CMD --min-spread-width 10"
CMD="$CMD --risk-cap $RISK_CAP"
CMD="$CMD --profit-target-pct 0.80"
CMD="$CMD --min-trading-hour 9"
CMD="$CMD --max-trading-hour 12"
CMD="$CMD --min-premium-diff 0.50"
CMD="$CMD --output-timezone America/New_York"
CMD="$CMD --dynamic-spread-width '{\"mode\": \"linear\", \"base_width\": 15, \"slope_factor\": 500, \"min_width\": 10, \"max_width\": 50}'"

# Add date range if provided
if [ -n "$START_DATE" ]; then
    CMD="$CMD --start-date $START_DATE"
fi
if [ -n "$END_DATE" ]; then
    CMD="$CMD --end-date $END_DATE"
fi

# Add live mode options
if [ "$LIVE_MODE" = true ]; then
    CMD="$CMD --continuous 30 --most-recent --best-only --curr-price --use-market-hours"
    echo "=== LIVE MODE ==="
    echo "Running continuously with 30-second intervals"
    echo "Using current price for calculations"
    echo ""
fi

# Show or execute
if [ "$DRY_RUN" = true ]; then
    echo "=== Command (dry run) ==="
    echo "$CMD"
else
    echo "=== Executing ==="
    eval "$CMD"
fi
