#!/usr/bin/env python3
"""
Compare time-allocated tiered strategy performance with and without slope detection.

Runs backtests for different time periods to show the impact of slope-based entry timing.
"""

import subprocess
import sys
from datetime import datetime, timedelta
from pathlib import Path


def run_backtest(start_date, end_date, config_file, label):
    """Run a single backtest and return the command output."""
    cmd = [
        sys.executable,
        "scripts/analyze_credit_spread_intervals.py",
        "--csv-dir", "csv_exports/options",
        "--ticker", "NDX",
        "--start-date", start_date,
        "--end-date", end_date,
        "--percent-beyond", "0.025",  # Used for interval filtering, tiers use their own values
        "--strategy", "time_allocated_tiered",
        "--strategy-config", config_file,
        "--output-timezone", "US/Pacific"
    ]

    print(f"\n{'='*80}")
    print(f"Running: {label}")
    print(f"Period: {start_date} to {end_date}")
    print(f"Config: {config_file}")
    print(f"{'='*80}\n")

    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        print(f"ERROR running {label}:")
        print("STDERR:", result.stderr)
        print("STDOUT:", result.stdout)
        print("Return code:", result.returncode)
        return None

    return result.stdout


def extract_summary_stats(output):
    """Extract key statistics from backtest output."""
    if not output:
        return None

    stats = {}
    lines = output.split('\n')

    for i, line in enumerate(lines):
        # Look for summary statistics
        if 'Total Days Analyzed:' in line:
            stats['days'] = line.split(':')[1].strip()
        elif 'Total P&L:' in line:
            stats['total_pnl'] = line.split(':')[1].strip()
        elif 'Win Rate:' in line:
            stats['win_rate'] = line.split(':')[1].strip()
        elif 'Average Daily P&L:' in line:
            stats['avg_daily_pnl'] = line.split(':')[1].strip()
        elif 'Total Capital at Risk:' in line:
            stats['total_capital'] = line.split(':')[1].strip()
        elif 'ROI:' in line:
            stats['roi'] = line.split(':')[1].strip()

    return stats


def main():
    # Use available data range (based on actual CSV files)
    # We have data from 2026-01-22 to 2026-02-06

    # Full available period (all data)
    full_start = "2026-01-22"
    full_end = "2026-02-06"

    # Last week (Jan 30 - Feb 6)
    one_week_start = "2026-01-30"

    # Last 5 trading days (Feb 2 - Feb 6)
    five_days_start = "2026-02-02"

    # Config files
    with_slope_config = "scripts/json/ta_tiered_strategy_ndx.json"
    no_slope_config = "scripts/json/ta_tiered_strategy_ndx_no_slope.json"

    # Test periods
    periods = [
        ("All Available Data (2 weeks)", full_start, full_end),
        ("Last Week", one_week_start, full_end),
        ("Last 5 Days", five_days_start, full_end),
    ]

    results = {}

    for period_name, start, end in periods:
        print(f"\n\n{'#'*80}")
        print(f"# PERIOD: {period_name} ({start} to {end})")
        print(f"{'#'*80}")

        # Run with slope detection
        with_slope_output = run_backtest(
            start, end, with_slope_config,
            f"{period_name} - WITH Slope Detection"
        )

        # Run without slope detection
        no_slope_output = run_backtest(
            start, end, no_slope_config,
            f"{period_name} - WITHOUT Slope Detection (Allocation Only)"
        )

        # Store results
        results[period_name] = {
            'with_slope': extract_summary_stats(with_slope_output),
            'no_slope': extract_summary_stats(no_slope_output),
            'with_slope_output': with_slope_output,
            'no_slope_output': no_slope_output,
        }

    # Print comparison summary
    print(f"\n\n{'='*80}")
    print("COMPARISON SUMMARY")
    print(f"{'='*80}\n")

    for period_name in ["All Available Data (2 weeks)", "Last Week", "Last 5 Days"]:
        print(f"\n{period_name}:")
        print("-" * 40)

        with_slope = results[period_name]['with_slope']
        no_slope = results[period_name]['no_slope']

        if with_slope and no_slope:
            print(f"{'Metric':<25} {'With Slope':<20} {'Without Slope':<20}")
            print("-" * 65)

            for key in ['days', 'total_pnl', 'win_rate', 'avg_daily_pnl', 'roi']:
                if key in with_slope and key in no_slope:
                    print(f"{key:<25} {with_slope[key]:<20} {no_slope[key]:<20}")
        else:
            print("  [No data available]")

    # Save detailed output to files
    output_dir = Path("comparison_results")
    output_dir.mkdir(exist_ok=True)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    for period_name in ["All Available Data (2 weeks)", "Last Week", "Last 5 Days"]:
        period_slug = period_name.lower().replace(' ', '_')

        # With slope
        if results[period_name]['with_slope_output']:
            with_slope_file = output_dir / f"{period_slug}_with_slope_{timestamp}.txt"
            with open(with_slope_file, 'w') as f:
                f.write(results[period_name]['with_slope_output'])
            print(f"\nDetailed output saved to: {with_slope_file}")

        # Without slope
        if results[period_name]['no_slope_output']:
            no_slope_file = output_dir / f"{period_slug}_no_slope_{timestamp}.txt"
            with open(no_slope_file, 'w') as f:
                f.write(results[period_name]['no_slope_output'])
            print(f"Detailed output saved to: {no_slope_file}")


if __name__ == "__main__":
    main()
