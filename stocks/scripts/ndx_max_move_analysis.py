#!/usr/bin/env python3
"""
NDX/SPX Max Movement to Close Analysis — thin wrapper.

All logic has been moved to credit_spread_utils/max_move_utils.py.
You can also use: analyze_credit_spread_intervals.py --mode max-move

Usage:
    python scripts/ndx_max_move_analysis.py --ticker NDX
    python scripts/ndx_max_move_analysis.py --ticker SPX --days 60
"""

import argparse
import sys
from pathlib import Path

# Project path setup
CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

from credit_spread_utils.max_move_utils import run_max_move_analysis


def main():
    parser = argparse.ArgumentParser(
        description='NDX max movement to close analysis by 30-min time slots'
    )
    parser.add_argument(
        '--ticker', '-t',
        type=str,
        default='NDX',
        help='Ticker symbol (NDX, SPX, etc.)'
    )
    parser.add_argument(
        '--days', '-d',
        type=int,
        default=125,
        help='Number of trading days to analyze (default: 125, ~6 months)'
    )
    args = parser.parse_args()
    sys.exit(run_max_move_analysis(args) or 0)


if __name__ == '__main__':
    main()
