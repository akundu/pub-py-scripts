#!/bin/bash

# Run time-allocated tiered strategy backtests with and without slope detection

mkdir -p comparison_results

echo "=============================================="
echo "Running NDX Time-Allocated Tiered Backtests"
echo "Comparing WITH vs WITHOUT slope detection"
echo "=============================================="

# Test period: Last week (2026-01-30 to 2026-02-06)
START_DATE="2026-01-30"
END_DATE="2026-02-06"

echo ""
echo "Period: $START_DATE to $END_DATE"
echo ""

# 1. Run WITH slope detection
echo "1. Running WITH slope detection..."
python scripts/analyze_credit_spread_intervals.py \
  --csv-dir csv_exports/options \
  --ticker NDX \
  --start-date $START_DATE \
  --end-date $END_DATE \
  --percent-beyond 0.025 \
  --strategy time_allocated_tiered \
  --strategy-config scripts/json/ta_tiered_strategy_ndx_permissive.json \
  --output-timezone US/Pacific \
  --log-level INFO \
  > comparison_results/with_slope.txt 2>&1

echo "✓ Completed (output saved to comparison_results/with_slope.txt)"

# 2. Run WITHOUT slope detection
echo ""
echo "2. Running WITHOUT slope detection (allocation only)..."
python scripts/analyze_credit_spread_intervals.py \
  --csv-dir csv_exports/options \
  --ticker NDX \
  --start-date $START_DATE \
  --end-date $END_DATE \
  --percent-beyond 0.025 \
  --strategy time_allocated_tiered \
  --strategy-config scripts/json/ta_tiered_strategy_ndx_permissive_no_slope.json \
  --output-timezone US/Pacific \
  --log-level INFO \
  > comparison_results/no_slope.txt 2>&1

echo "✓ Completed (output saved to comparison_results/no_slope.txt)"

echo ""
echo "=============================================="
echo "Extracting summary statistics..."
echo "=============================================="

# Extract key stats from both runs
echo ""
echo "WITH Slope Detection:"
echo "---------------------"
grep -E "Total (Days|P&L|Capital)|Win Rate|Average|ROI" comparison_results/with_slope.txt | head -20

echo ""
echo "WITHOUT Slope Detection (Allocation Only):"
echo "-------------------------------------------"
grep -E "Total (Days|P&L|Capital)|Win Rate|Average|ROI" comparison_results/no_slope.txt | head -20

echo ""
echo "=============================================="
echo "Detailed results saved to:"
echo "  - comparison_results/with_slope.txt"
echo "  - comparison_results/no_slope.txt"
echo "=============================================="
