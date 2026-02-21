#!/bin/bash
#
# Full Backtesting & Optimization Pipeline
# Runs all analysis steps in sequence with parallelization
#
# Usage: ./run_full_analysis.sh [START_DATE] [END_DATE]
#
# Example: ./run_full_analysis.sh 2026-01-16 2026-02-15
#

set -e  # Exit on error

# Default dates (last month)
START_DATE=${1:-"2026-01-16"}
END_DATE=${2:-"2026-02-15"}
TICKER="NDX"
PROCESSES=$(sysctl -n hw.ncpu 2>/dev/null || nproc 2>/dev/null || echo "4")

echo "================================================================================"
echo "FULL BACKTESTING & OPTIMIZATION PIPELINE"
echo "================================================================================"
echo "Ticker:        $TICKER"
echo "Date Range:    $START_DATE to $END_DATE"
echo "Processes:     $PROCESSES"
echo "================================================================================"
echo ""

# Create results directory
mkdir -p results

# Step 1: Validation Tests
echo "================================================================================"
echo "STEP 1: VALIDATION TESTS"
echo "================================================================================"
echo "Running single-day validation to confirm fixes..."
echo ""

python scripts/daily_pnl_simulator.py \
  --ticker $TICKER \
  --start-date 2026-02-10 \
  --end-date 2026-02-10 \
  --dte 1 \
  --percentile 99 \
  --spread-width 20 \
  --output results/validation_test.csv

echo ""
echo "✓ Validation complete"
echo ""

# Step 2: Time-of-Day Analysis
echo "================================================================================"
echo "STEP 2: TIME-OF-DAY ANALYSIS"
echo "================================================================================"
echo "Analyzing optimal entry times for 0DTE strategies..."
echo ""

python scripts/time_of_day_analyzer.py \
  --ticker $TICKER \
  --start-date $START_DATE \
  --end-date $END_DATE \
  --processes $PROCESSES \
  --output results/time_of_day_analysis.csv

echo ""
echo "✓ Time-of-day analysis complete"
echo ""

# Step 3: Comprehensive Grid Search
echo "================================================================================"
echo "STEP 3: COMPREHENSIVE GRID SEARCH"
echo "================================================================================"
echo "Running full parameter sweep (756 configurations)..."
echo "This may take 4-8 hours depending on your hardware."
echo ""

python scripts/comprehensive_grid_search.py \
  --ticker $TICKER \
  --start-date $START_DATE \
  --end-date $END_DATE \
  --processes $PROCESSES \
  --min-roi 5.0 \
  --max-loss 30000 \
  --save-interval 100 \
  --output results/comprehensive_grid_search.csv

echo ""
echo "✓ Grid search complete"
echo ""

# Step 4: Position Sizing Optimization
echo "================================================================================"
echo "STEP 4: POSITION SIZING OPTIMIZATION"
echo "================================================================================"
echo "Optimizing position sizes for top strategies..."
echo ""

python scripts/position_sizing_optimizer.py \
  --results results/comprehensive_grid_search.csv \
  --capital-levels 25000 50000 100000 250000 500000 \
  --risk-tolerance moderate \
  --top-n 20 \
  --output results/position_sizing_recommendations.csv

echo ""
echo "✓ Position sizing complete"
echo ""

# Step 5: Portfolio Building
echo "================================================================================"
echo "STEP 5: PORTFOLIO BUILDING"
echo "================================================================================"
echo "Building diversified portfolios..."
echo ""

# Build optimal portfolios
python scripts/portfolio_builder.py \
  --results results/comprehensive_grid_search.csv \
  --sizing results/position_sizing_recommendations.csv \
  --capital 100000 \
  --max-correlation 0.7 \
  --min-strategies 3 \
  --max-strategies 5 \
  --top-n 20 \
  --num-portfolios 10 \
  --output results/optimal_portfolios.csv

# Build risk-tiered portfolios
python scripts/portfolio_builder.py \
  --results results/comprehensive_grid_search.csv \
  --capital 100000 \
  --build-tiered \
  --output results/risk_tiered_portfolios.csv

echo ""
echo "✓ Portfolio building complete"
echo ""

# Summary
echo "================================================================================"
echo "PIPELINE COMPLETE"
echo "================================================================================"
echo ""
echo "Results saved to:"
echo "  - results/validation_test.csv"
echo "  - results/time_of_day_analysis.csv"
echo "  - results/comprehensive_grid_search.csv"
echo "  - results/position_sizing_recommendations.csv"
echo "  - results/optimal_portfolios.csv"
echo "  - results/risk_tiered_portfolios.json"
echo ""
echo "Next steps:"
echo "  1. Review results in results/ directory"
echo "  2. Check DEPLOYMENT_GUIDE.md for trading instructions"
echo "  3. Start with paper trading to validate"
echo ""
echo "================================================================================"
