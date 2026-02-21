#!/usr/bin/env python3
"""
Equity Price Movement Analyzer — thin wrapper.

All logic has been moved to credit_spread_utils/price_movement_utils.py.
You can also use: analyze_credit_spread_intervals.py --mode price-movements

Usage:
    python scripts/analyze_price_movements.py --ticker I:NDX --no-plot
    python scripts/analyze_price_movements.py --ticker QQQ --from-time 10:00 --timezone PST
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

from credit_spread_utils.price_movement_utils import run_price_movement_analysis


def parse_args():
    """Parse command line arguments (standalone mode)."""
    parser = argparse.ArgumentParser(
        description='Analyze equity price movements from a given time to market close.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/analyze_price_movements.py --ticker SPY
  python scripts/analyze_price_movements.py --ticker QQQ --from-time 10:00 --timezone PST
  python scripts/analyze_price_movements.py --ticker I:NDX --from-time 11:30 --timezone EST --no-plot
        """
    )

    parser.add_argument('--ticker', required=True,
                        help='Ticker symbol (e.g., SPY, QQQ, I:NDX)')
    parser.add_argument('--data-dir', default='equities_output',
                        help='Directory containing equity data (default: equities_output)')
    parser.add_argument('--start-date', type=str, default=None,
                        help='Start date for analysis (YYYY-MM-DD)')
    parser.add_argument('--end-date', type=str, default=None,
                        help='End date for analysis (YYYY-MM-DD)')
    parser.add_argument('--from-time', type=str, default=None,
                        help='Time of day to measure FROM (HH:MM)')
    parser.add_argument('--timezone', type=str, default='PST',
                        choices=['PST', 'PDT', 'EST', 'EDT', 'UTC'],
                        help='Timezone for time inputs (default: PST)')
    parser.add_argument('--output', type=str, default='price_movements.png',
                        help='Output path for histogram image (default: price_movements.png)')
    parser.add_argument('--no-plot', action='store_true',
                        help='Skip histogram generation, print stats only')
    parser.add_argument('--day-direction', type=str, default=None,
                        choices=['up', 'down'],
                        help='Filter to only days that closed up or down vs prior day')

    return parser.parse_args()


def main():
    args = parse_args()
    # Map standalone arg names to what run_price_movement_analysis expects
    args.pm_timezone = args.timezone
    args.plot_output = args.output
    sys.exit(run_price_movement_analysis(args) or 0)


if __name__ == '__main__':
    main()
