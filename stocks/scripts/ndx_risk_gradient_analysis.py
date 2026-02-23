#!/usr/bin/env python3
"""
NDX Risk-Based Analysis Tool — thin wrapper.

All logic has been moved to credit_spread_utils/risk_gradient_utils.py.
You can also use: analyze_credit_spread_intervals.py --mode risk-gradient

Usage:
    python scripts/ndx_risk_gradient_analysis.py [--lookback-days 90] [--output-dir ./output]
    python scripts/ndx_risk_gradient_analysis.py --generate-config-only
    python scripts/ndx_risk_gradient_analysis.py --run-backtest --processes 8
"""

import argparse
import asyncio
import sys
from pathlib import Path

# Project path setup
CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

from credit_spread_utils.risk_gradient_utils import run_risk_gradient_analysis


def parse_args():
    """Parse command line arguments (standalone mode)."""
    parser = argparse.ArgumentParser(
        description='NDX Risk-Based Analysis Tool',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/ndx_risk_gradient_analysis.py
  python scripts/ndx_risk_gradient_analysis.py --lookback-days 60
  python scripts/ndx_risk_gradient_analysis.py --generate-config-only
        """
    )

    parser.add_argument('--ticker', default='NDX',
                        help='Ticker symbol (default: NDX)')
    parser.add_argument('--lookback-days', type=int, nargs='+', default=[90, 180],
                        help='Lookback periods in days (default: 90 180)')
    parser.add_argument('--output-dir', default=str(CURRENT_DIR),
                        help='Output directory for configs and results (default: scripts/)')
    parser.add_argument('--db-path', default=None,
                        help='Database connection string (default: $QUEST_DB_STRING)')
    parser.add_argument('--risk-cap', type=int, default=500000,
                        help='Risk cap in dollars (default: 500000)')
    parser.add_argument('--csv-dir', default='../options_csv_output',
                        help='Directory containing options CSV data')
    parser.add_argument('--min-trading-hour', type=int, default=9,
                        help='Minimum trading hour PST (default: 9)')
    parser.add_argument('--max-trading-hour', type=int, default=12,
                        help='Maximum trading hour PST (default: 12)')
    parser.add_argument('--gradient-steps', type=int, default=7,
                        help='Number of gradient steps from safe point (default: 7)')
    parser.add_argument('--step-size', type=float, default=0.0025,
                        help='Step size in decimal (default: 0.0025)')
    parser.add_argument('--generate-config-only', action='store_true',
                        help='Only generate config files, do not run backtest')
    parser.add_argument('--run-backtest', action='store_true',
                        help='Run backtest after generating configs')
    parser.add_argument('--processes', type=int, default=8,
                        help='Number of parallel processes for backtest (default: 8)')
    parser.add_argument('--log-level', default='INFO',
                        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                        help='Log level (default: INFO)')
    parser.add_argument('--time-analysis-config', type=str, default=None,
                        help='Path to a config file to run time-of-day analysis')
    parser.add_argument('--time-periods', type=str, nargs='+',
                        default=['3mo', '1mo', 'week1', 'week2', 'week3', 'week4'],
                        help='Time periods to analyze')
    parser.add_argument('--detailed-output', action='store_true',
                        help='Enable detailed output with hourly and 10-minute block breakdowns')

    return parser.parse_args()


async def main():
    args = parse_args()
    result = await run_risk_gradient_analysis(args)
    return result


if __name__ == '__main__':
    sys.exit(asyncio.run(main()) or 0)
