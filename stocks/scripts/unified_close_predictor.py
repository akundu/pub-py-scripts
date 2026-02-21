#!/usr/bin/env python3
"""
Unified Closing Price Predictor

Combines two prediction approaches into a single tool:
1. Percentile Range Model — time slot + above/below + 5-day realized vol scaling
2. Statistical Bucket Model — VIX1D regime, overnight gap, intraday move, momentum, etc.

For each band (P95, P98, P99, P100) the "best across best" combination takes
the wider (more conservative) range from both models.

Modes:
  backtest  — walk-forward accuracy comparison across models
  live      — real-time display with QuestDB (or --demo for CSV fallback)

Usage:
    python scripts/unified_close_predictor.py backtest --ticker NDX --test-days 10
    python scripts/unified_close_predictor.py backtest --ticker SPX --test-days 10
    python scripts/unified_close_predictor.py live --ticker NDX --interval 30
    python scripts/unified_close_predictor.py live --ticker NDX --demo
"""

import argparse
import asyncio
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.close_predictor.backtest import run_backtest
from scripts.close_predictor.live import run_demo_loop, run_live_loop


def main():
    parser = argparse.ArgumentParser(
        description="Unified Closing Price Predictor — combines percentile-range and statistical bucket models"
    )
    subparsers = parser.add_subparsers(dest="command", help="Mode")

    # Shared parent for common args
    shared = argparse.ArgumentParser(add_help=False)
    shared.add_argument('--ticker', '-t', type=str, default='NDX',
                        help='Ticker symbol (NDX, SPX)')
    shared.add_argument('--lookback', '-l', type=int, default=250,
                        help='Training lookback in trading days (default: 250)')

    # Backtest subcommand
    bt = subparsers.add_parser('backtest', parents=[shared],
                               help='Walk-forward backtest')
    bt.add_argument('--test-days', type=int, default=10,
                    help='Number of walk-forward test days (default: 10)')
    bt.add_argument('--no-vol-scale', action='store_true',
                    help='Disable vol-scaling for percentile model')
    bt.add_argument('--all-slots', action='store_true',
                    help='Show all 30-min slots instead of hourly summary')

    # Live subcommand
    lv = subparsers.add_parser('live', parents=[shared],
                               help='Live prediction display')
    lv.add_argument('--interval', '-i', type=int, default=30,
                    help='Refresh interval in seconds (default: 30)')
    lv.add_argument('--demo', action='store_true',
                    help='Demo mode using CSV data (no QuestDB required)')

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(1)

    if args.command == 'backtest':
        run_backtest(
            ticker=args.ticker,
            lookback=args.lookback,
            test_days=args.test_days,
            vol_scale=not args.no_vol_scale,
            all_slots=args.all_slots,
        )

    elif args.command == 'live':
        if args.demo:
            run_demo_loop(
                ticker=args.ticker,
                lookback=args.lookback,
                interval=args.interval,
                vol_scale=True,
            )
        else:
            asyncio.run(run_live_loop(
                ticker=args.ticker,
                lookback=args.lookback,
                interval=args.interval,
                vol_scale=True,
            ))


if __name__ == '__main__':
    main()
