#!/usr/bin/env python3
"""
Multiprocessing backtest runner for percentile-based spreads.

Runs comprehensive backtests across multiple configurations in parallel.
"""

import asyncio
import argparse
import logging
import sys
import json
from pathlib import Path
from datetime import datetime
from multiprocessing import Pool, cpu_count
from functools import partial
import pandas as pd
import os

# Add project root to path
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from credit_spread_utils.percentile_integration import PercentileSpreadIntegrator
from credit_spread_utils.data_loader import load_multi_dte_data
from credit_spread_utils.price_utils import get_previous_close_price
from common.questdb_db import StockQuestDB
from common.logging_utils import get_logger

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s'
)
logger = get_logger("backtest_runner", level="INFO")


async def run_single_config(
    ticker: str,
    start_date: str,
    end_date: str,
    dte: int,
    percentile: int,
    spread_width: float,
    flow_mode: str,
    csv_dir: str,
    db_config: str
) -> dict:
    """Run backtest for a single configuration."""

    config_name = f"DTE{dte}_p{percentile}_w{spread_width}_{flow_mode}"

    logger.info(f"[{config_name}] Loading data...")

    # Load data
    df = load_multi_dte_data(
        csv_dir=csv_dir,
        ticker=ticker,
        start_date=start_date,
        end_date=end_date,
        dte_buckets=(dte,),
        dte_tolerance=1,
        cache_dir=None,
        no_cache=True,
        logger=logger
    )

    if df.empty:
        logger.warning(f"[{config_name}] No data found")
        return None

    # Get trading dates
    df['trading_date'] = df['timestamp'].apply(
        lambda x: x.date() if hasattr(x, 'date') else pd.to_datetime(x).date()
    )
    trading_dates = sorted(df['trading_date'].unique())

    logger.info(f"[{config_name}] Processing {len(trading_dates)} days...")

    # Initialize
    db = StockQuestDB(db_config, logger=logger)
    integrator = PercentileSpreadIntegrator(db_config=db_config)

    all_trades = []
    days_processed = 0
    days_with_spreads = 0

    # Process each day
    for trading_date in trading_dates:
        day_df = df[df['trading_date'] == trading_date]
        first_ts = day_df['timestamp'].min()

        # Get previous close
        prev_close_result = await get_previous_close_price(db, ticker, first_ts, logger)
        if prev_close_result is None:
            continue

        prev_close, _ = prev_close_result
        days_processed += 1

        # Analyze
        trades = await integrator.analyze_single_day(
            options_df=day_df,
            ticker=ticker,
            trading_date=datetime.combine(trading_date, datetime.min.time()),
            prev_close=prev_close,
            dte=dte,
            percentile=percentile,
            spread_width=spread_width,
            profit_target_pct=0.5,
            flow_mode=flow_mode,
            use_mid=False
        )

        if trades:
            days_with_spreads += 1
        all_trades.extend(trades)

    # Aggregate results
    if not all_trades:
        logger.warning(f"[{config_name}] No spreads found")
        return {
            'config': config_name,
            'dte': dte,
            'percentile': percentile,
            'spread_width': spread_width,
            'flow_mode': flow_mode,
            'days_processed': days_processed,
            'days_with_spreads': 0,
            'total_spreads': 0,
            'avg_credit': 0,
            'avg_max_loss': 0,
            'avg_roi': 0
        }

    trades_df = pd.DataFrame(all_trades)

    result = {
        'config': config_name,
        'dte': dte,
        'percentile': percentile,
        'spread_width': spread_width,
        'flow_mode': flow_mode,
        'days_processed': days_processed,
        'days_with_spreads': days_with_spreads,
        'total_spreads': len(all_trades),
        'avg_credit': trades_df['entry_credit'].mean(),
        'avg_max_loss': trades_df['max_loss'].mean(),
        'avg_roi': (trades_df['entry_credit'] / trades_df['max_loss'] * 100).mean()
    }

    logger.info(
        f"[{config_name}] Complete: {len(all_trades)} spreads, "
        f"ROI={result['avg_roi']:.1f}%"
    )

    return result


def run_config_wrapper(args):
    """Wrapper for multiprocessing."""
    return asyncio.run(run_single_config(*args))


def main():
    parser = argparse.ArgumentParser(description='Run multiprocessing backtest')
    parser.add_argument('--ticker', default='NDX', help='Ticker symbol')
    parser.add_argument('--start-date', required=True, help='Start date')
    parser.add_argument('--end-date', required=True, help='End date')
    parser.add_argument('--processes', type=int, default=None, help='Number of processes')
    parser.add_argument('--csv-dir', default='options_csv_output_full', help='CSV directory')
    parser.add_argument('--output', default='backtest_results.csv', help='Output file')

    args = parser.parse_args()

    # Get DB config
    db_config = (
        os.getenv('QUEST_DB_STRING') or
        os.getenv('QUESTDB_CONNECTION_STRING') or
        os.getenv('QUESTDB_URL')
    )

    # Define configurations to test
    configs = []

    # Conservative configs (p95)
    for dte in [3, 5]:
        configs.append((args.ticker, args.start_date, args.end_date, dte, 95, 50, 'neutral', args.csv_dir, db_config))

    # Balanced configs (p97)
    for dte in [3, 5]:
        for width in [20, 30, 50]:
            configs.append((args.ticker, args.start_date, args.end_date, dte, 97, width, 'neutral', args.csv_dir, db_config))

    # Aggressive configs (p99)
    for dte in [0, 1, 3]:
        configs.append((args.ticker, args.start_date, args.end_date, dte, 99, 20, 'neutral', args.csv_dir, db_config))

    # With flow configs
    for dte in [3, 5]:
        configs.append((args.ticker, args.start_date, args.end_date, dte, 97, 30, 'with_flow', args.csv_dir, db_config))

    num_processes = args.processes or min(cpu_count(), len(configs))

    logger.info(f"\n{'='*80}")
    logger.info(f"Starting Multiprocessing Backtest")
    logger.info(f"{'='*80}")
    logger.info(f"Ticker: {args.ticker}")
    logger.info(f"Date Range: {args.start_date} to {args.end_date}")
    logger.info(f"Configurations: {len(configs)}")
    logger.info(f"Processes: {num_processes}")
    logger.info(f"{'='*80}\n")

    # Run in parallel
    with Pool(processes=num_processes) as pool:
        results = pool.map(run_config_wrapper, configs)

    # Filter out None results
    results = [r for r in results if r is not None]

    # Save results
    results_df = pd.DataFrame(results)
    results_df.to_csv(args.output, index=False)

    logger.info(f"\n{'='*80}")
    logger.info(f"BACKTEST COMPLETE")
    logger.info(f"{'='*80}")
    logger.info(f"Results saved to: {args.output}")
    logger.info(f"Total configs tested: {len(results)}")

    # Show top performers
    if not results_df.empty:
        logger.info(f"\nTop 5 by Average ROI:")
        top_roi = results_df.nlargest(5, 'avg_roi')[['config', 'total_spreads', 'avg_roi', 'avg_credit']]
        logger.info(f"\n{top_roi.to_string(index=False)}")

        logger.info(f"\nTop 5 by Total Spreads:")
        top_spreads = results_df.nlargest(5, 'total_spreads')[['config', 'total_spreads', 'avg_roi', 'avg_credit']]
        logger.info(f"\n{top_spreads.to_string(index=False)}")

    logger.info(f"\n{'='*80}\n")

    return 0


if __name__ == '__main__':
    sys.exit(main())
