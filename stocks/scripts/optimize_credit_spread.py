#!/usr/bin/env python3
"""
Credit Spread Parameter Optimizer - Thin wrapper.

Runs grid search mode of analyze_credit_spread_intervals.py.
Equivalent to:
    python analyze_credit_spread_intervals.py --grid-config <config.json> [options]

Usage:
    python optimize_credit_spread.py --config optimizer_config.json [--output results.csv] [--dry-run]
"""

import argparse
import asyncio
import sys
import os

# Add scripts dir to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from analyze_credit_spread_intervals import run_grid_search


def parse_optimizer_args():
    """Parse optimizer-specific args and map to grid search args."""
    parser = argparse.ArgumentParser(
        description="Credit Spread Parameter Optimizer - Grid search over backtest parameters"
    )
    parser.add_argument('--config', required=True, help='Path to JSON config file')
    parser.add_argument('--output', default='optimizer_results.csv', help='Output CSV path')
    parser.add_argument('--top-n', type=int, default=10, help='Number of top results to display')
    parser.add_argument('--sort-by', default='net_pnl',
                        choices=['net_pnl', 'profit_factor', 'win_rate', 'roi'],
                        help='Primary sort metric')
    parser.add_argument('--dry-run', action='store_true', help='Show combinations without executing')
    parser.add_argument('--resume', action='store_true', help='Resume from existing CSV')
    parser.add_argument('--cache-dir', default='.options_cache', help='Binary cache directory')
    parser.add_argument('--no-data-cache', action='store_true', help='Disable binary data cache')
    parser.add_argument('--log-level', default='WARNING', help='Log level (default: WARNING)')

    raw_args = parser.parse_args()

    # Map to the args format expected by run_grid_search
    class GridArgs:
        pass

    args = GridArgs()
    args.grid_config = raw_args.config
    args.grid_output = raw_args.output
    args.grid_top_n = raw_args.top_n
    args.grid_sort = raw_args.sort_by
    args.grid_dry_run = raw_args.dry_run
    args.grid_resume = raw_args.resume
    args.cache_dir = raw_args.cache_dir
    args.no_data_cache = raw_args.no_data_cache
    args.log_level = raw_args.log_level

    return args


if __name__ == '__main__':
    args = parse_optimizer_args()
    sys.exit(asyncio.run(run_grid_search(args)) or 0)
