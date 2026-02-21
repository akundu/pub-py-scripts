#!/bin/bash

# Comprehensive backtest comparison across multiple time periods

mkdir -p comparison_results/comprehensive

echo "=============================================="
echo "Comprehensive NDX Backtest Comparison"
echo "Testing WITH vs WITHOUT slope detection"
echo "=============================================="

# Test periods array: label, start_date, end_date
declare -a PERIODS=(
  "all_data:2026-01-22:2026-02-06"
  "last_week:2026-01-30:2026-02-06"
  "last_5_days:2026-02-02:2026-02-06"
  "single_day_feb04:2026-02-04:2026-02-04"
  "single_day_feb03:2026-02-03:2026-02-03"
)

# Function to run a single backtest
run_backtest() {
  local period_label=$1
  local start_date=$2
  local end_date=$3
  local slope_mode=$4  # "with" or "without"

  local config_file
  if [ "$slope_mode" = "with" ]; then
    config_file="scripts/json/ta_tiered_strategy_ndx_permissive.json"
  else
    config_file="scripts/json/ta_tiered_strategy_ndx_permissive_no_slope.json"
  fi

  local output_file="comparison_results/comprehensive/${period_label}_${slope_mode}_slope.txt"

  echo "  Running: $period_label ($start_date to $end_date) - $slope_mode slope..."

  python scripts/analyze_credit_spread_intervals.py \
    --csv-dir csv_exports/options \
    --ticker NDX \
    --start-date "$start_date" \
    --end-date "$end_date" \
    --percent-beyond 0.025 \
    --strategy time_allocated_tiered \
    --strategy-config "$config_file" \
    --output-timezone US/Pacific \
    > "$output_file" 2>&1

  # Extract key stats
  local win_rate=$(grep "Win Rate:" "$output_file" | tail -1 | awk '{print $3}')
  local roi=$(grep "ROI:" "$output_file" | tail -1 | awk '{print $2}')
  local total_pnl=$(grep "Net P&L:" "$output_file" | tail -1 | awk '{print $3}')

  echo "    ✓ Win Rate: $win_rate, ROI: $roi, P&L: $total_pnl"
}

# Run all tests
for period in "${PERIODS[@]}"; do
  IFS=':' read -r label start end <<< "$period"

  echo ""
  echo "Period: $label ($start to $end)"
  echo "----------------------------------------"

  run_backtest "$label" "$start" "$end" "with"
  run_backtest "$label" "$start" "$end" "without"
done

echo ""
echo "=============================================="
echo "Generating comparison table..."
echo "=============================================="

# Create summary table
{
  echo "# Comprehensive Slope Detection Comparison"
  echo ""
  echo "| Period | Dates | WITH Slope | WITHOUT Slope | Difference |"
  echo "|--------|-------|------------|---------------|------------|"

  for period in "${PERIODS[@]}"; do
    IFS=':' read -r label start end <<< "$period"

    # Extract ROI from both files
    with_file="comparison_results/comprehensive/${label}_with_slope.txt"
    without_file="comparison_results/comprehensive/${label}_without_slope.txt"

    if [ -f "$with_file" ] && [ -f "$without_file" ]; then
      with_roi=$(grep "ROI:" "$with_file" | tail -1 | awk '{print $2}')
      without_roi=$(grep "ROI:" "$without_file" | tail -1 | awk '{print $2}')

      echo "| $label | $start to $end | $with_roi | $without_roi | - |"
    fi
  done

  echo ""
  echo "Detailed results in: comparison_results/comprehensive/"

} > comparison_results/comprehensive/SUMMARY.md

cat comparison_results/comprehensive/SUMMARY.md

echo ""
echo "=============================================="
echo "Complete! Check comparison_results/comprehensive/"
echo "=============================================="
