#!/usr/bin/env python3
"""
Phase 1: Comprehensive Parameter Discovery Backtest

Tests expanded grid of configurations to identify optimal parameters:
- DTEs: 0, 1, 3, 5, 10
- Percentiles: 95, 96, 97, 98, 99, 100
- Spread Widths: 15, 20, 25, 30, 50
- Flow Modes: neutral, with_flow

Runs in parallel across multiple processes for performance.
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
logger = get_logger("phase1_backtest", level="INFO")


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

    config_name = f"DTE{dte}_p{percentile}_w{int(spread_width)}_{flow_mode}"

    logger.info(f"[{config_name}] Loading data...")

    # Determine CSV directory based on DTE
    if dte == 0:
        data_dir = 'options_csv_output'
    else:
        data_dir = csv_dir

    # Load data
    df = load_multi_dte_data(
        csv_dir=data_dir,
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
    parser = argparse.ArgumentParser(description='Phase 1: Comprehensive Parameter Discovery')
    parser.add_argument('--ticker', default='NDX', help='Ticker symbol')
    parser.add_argument('--start-date', required=True, help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end-date', required=True, help='End date (YYYY-MM-DD)')
    parser.add_argument('--processes', type=int, default=None, help='Number of processes')
    parser.add_argument('--csv-dir', default='options_csv_output_full', help='CSV directory for non-0DTE')
    parser.add_argument('--output', default='results/phase1_backtest.csv', help='Output file')
    parser.add_argument('--test-mode', action='store_true', help='Run limited configs for testing')

    args = parser.parse_args()

    # Get DB config
    db_config = (
        os.getenv('QUEST_DB_STRING') or
        os.getenv('QUESTDB_CONNECTION_STRING') or
        os.getenv('QUESTDB_URL')
    )

    if not db_config:
        print("ERROR: No database configuration found. Set QUEST_DB_STRING environment variable.")
        return 1

    # Define grid parameters
    if args.test_mode:
        # Limited grid for testing
        dtes = [0, 3, 5]
        percentiles = [95, 97, 99]
        spread_widths = [20, 50]
        flow_modes = ['neutral']
    else:
        # Full Phase 1 grid
        dtes = [0, 1, 3, 5, 10]
        percentiles = [95, 96, 97, 98, 99, 100]
        spread_widths = [15, 20, 25, 30, 50]
        flow_modes = ['neutral', 'with_flow']

    # Build configuration list
    configs = []
    for dte in dtes:
        for percentile in percentiles:
            for width in spread_widths:
                for flow_mode in flow_modes:
                    configs.append((
                        args.ticker,
                        args.start_date,
                        args.end_date,
                        dte,
                        percentile,
                        width,
                        flow_mode,
                        args.csv_dir,
                        db_config
                    ))

    num_processes = args.processes or min(cpu_count(), len(configs))

    logger.info(f"\n{'='*80}")
    logger.info(f"PHASE 1: COMPREHENSIVE PARAMETER DISCOVERY")
    logger.info(f"{'='*80}")
    logger.info(f"Ticker: {args.ticker}")
    logger.info(f"Date Range: {args.start_date} to {args.end_date}")
    logger.info(f"DTEs: {dtes}")
    logger.info(f"Percentiles: {percentiles}")
    logger.info(f"Spread Widths: {spread_widths}")
    logger.info(f"Flow Modes: {flow_modes}")
    logger.info(f"Total Configurations: {len(configs)}")
    logger.info(f"Processes: {num_processes}")
    logger.info(f"{'='*80}\n")

    # Run in parallel
    start_time = datetime.now()
    with Pool(processes=num_processes) as pool:
        results = pool.map(run_config_wrapper, configs)
    end_time = datetime.now()

    # Filter out None results
    results = [r for r in results if r is not None]

    # Save results
    results_df = pd.DataFrame(results)

    # Create output directory if needed
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    results_df.to_csv(args.output, index=False)

    runtime = (end_time - start_time).total_seconds()

    logger.info(f"\n{'='*80}")
    logger.info(f"PHASE 1 BACKTEST COMPLETE")
    logger.info(f"{'='*80}")
    logger.info(f"Results saved to: {args.output}")
    logger.info(f"Total configs tested: {len(results)}")
    logger.info(f"Runtime: {runtime:.1f} seconds ({runtime/60:.1f} minutes)")

    # Show top performers
    if not results_df.empty:
        logger.info(f"\nTop 10 by Average ROI:")
        top_roi = results_df.nlargest(10, 'avg_roi')[['config', 'total_spreads', 'avg_roi', 'avg_credit', 'days_with_spreads']]
        logger.info(f"\n{top_roi.to_string(index=False)}")

        logger.info(f"\nTop 10 by Total Spreads:")
        top_spreads = results_df.nlargest(10, 'total_spreads')[['config', 'total_spreads', 'avg_roi', 'avg_credit', 'days_with_spreads']]
        logger.info(f"\n{top_spreads.to_string(index=False)}")

        logger.info(f"\nTop 10 by Consistency (Days with Spreads):")
        top_consistency = results_df.nlargest(10, 'days_with_spreads')[['config', 'days_with_spreads', 'days_processed', 'total_spreads', 'avg_roi']]
        logger.info(f"\n{top_consistency.to_string(index=False)}")

        # Summary statistics
        logger.info(f"\n{'='*80}")
        logger.info(f"SUMMARY STATISTICS")
        logger.info(f"{'='*80}")
        logger.info(f"Configs with spreads: {len(results_df[results_df['total_spreads'] > 0])}/{len(results_df)}")
        logger.info(f"Average spreads per config: {results_df['total_spreads'].mean():.1f}")
        logger.info(f"Average ROI: {results_df[results_df['total_spreads'] > 0]['avg_roi'].mean():.1f}%")
        logger.info(f"Max ROI: {results_df['avg_roi'].max():.1f}%")

    logger.info(f"\n{'='*80}\n")

    return 0


if __name__ == '__main__':
    sys.exit(main())
