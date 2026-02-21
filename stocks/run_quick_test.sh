#!/bin/bash
#
# Quick Test - Runs limited configurations to verify setup
#
# Usage: ./run_quick_test.sh
#

set -e

TICKER="NDX"
START_DATE="2026-02-10"
END_DATE="2026-02-12"
PROCESSES=4

echo "================================================================================"
echo "QUICK TEST - Limited Configuration"
echo "================================================================================"
echo "This will run a small test to verify everything works."
echo ""

mkdir -p results

# Test 1: Single strategy validation
echo "Test 1: Single strategy validation..."
python scripts/daily_pnl_simulator.py \
  --ticker $TICKER \
  --start-date $START_DATE \
  --end-date $END_DATE \
  --dte 1 \
  --percentile 99 \
  --spread-width 20 \
  --max-positions 10 \
  --output results/quick_test_single.csv

echo "✓ Single strategy test passed"
echo ""

# Test 2: Grid search (limited)
echo "Test 2: Limited grid search (18 configs)..."
python scripts/comprehensive_grid_search.py \
  --ticker $TICKER \
  --start-date $START_DATE \
  --end-date $END_DATE \
  --processes $PROCESSES \
  --test-mode \
  --output results/quick_test_grid.csv

echo "✓ Grid search test passed"
echo ""

# Test 3: Position sizing
echo "Test 3: Position sizing optimization..."
python scripts/position_sizing_optimizer.py \
  --results results/quick_test_grid.csv \
  --capital-levels 100000 \
  --risk-tolerance moderate \
  --top-n 5 \
  --output results/quick_test_sizing.csv

echo "✓ Position sizing test passed"
echo ""

# Test 4: Portfolio building
echo "Test 4: Portfolio building..."
python scripts/portfolio_builder.py \
  --results results/quick_test_grid.csv \
  --capital 100000 \
  --max-correlation 0.8 \
  --top-n 5 \
  --num-portfolios 3 \
  --output results/quick_test_portfolios.csv

echo "✓ Portfolio building test passed"
echo ""

echo "================================================================================"
echo "ALL TESTS PASSED!"
echo "================================================================================"
echo ""
echo "Your system is configured correctly."
echo "Run ./run_full_analysis.sh to execute the complete pipeline."
echo ""
