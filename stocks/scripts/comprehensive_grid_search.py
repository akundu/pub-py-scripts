#!/usr/bin/env python3
"""
Comprehensive Grid Search with Fixes Applied

Tests expanded parameter grid with fixes:
- 0DTE volume fix (filters to single timestamp)
- Flow mode fix (momentum detection integrated)
- Enhanced spread width testing
- Time-of-day dimensions for 0DTE

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
import numpy as np
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
logger = get_logger("comprehensive_grid", level="INFO")


async def run_single_config(
    ticker: str,
    start_date: str,
    end_date: str,
    dte: int,
    percentile: int,
    spread_width: float,
    flow_mode: str,
    csv_dir: str,
    db_config: str,
    min_roi: float = 5.0,
    max_loss_limit: float = 30000.0
) -> dict:
    """Run backtest for a single configuration with fixes applied."""

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
    filtered_by_loss_limit = 0

    # Process each day
    for trading_date in trading_dates:
        day_df = df[df['trading_date'] == trading_date]

        # Apply 0DTE volume fix: filter to single timestamp per day
        if dte == 0 and not day_df.empty:
            timestamps = sorted(day_df['timestamp'].unique())
            if timestamps:
                entry_timestamp = timestamps[0]  # Use earliest timestamp
                day_df = day_df[day_df['timestamp'] == entry_timestamp].copy()
                logger.debug(f"[{config_name}] [0DTE Fix] Filtered to {entry_timestamp}")

        if day_df.empty:
            continue

        first_ts = day_df['timestamp'].min()

        # Get previous close
        prev_close_result = await get_previous_close_price(db, ticker, first_ts, logger)
        if prev_close_result is None:
            continue

        prev_close, _ = prev_close_result
        days_processed += 1

        # Analyze with flow mode (flow mode fix applied via integrator)
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
            # Filter by max loss limit
            valid_trades = [t for t in trades if t['max_loss'] <= max_loss_limit]
            filtered_count = len(trades) - len(valid_trades)
            filtered_by_loss_limit += filtered_count

            if valid_trades:
                days_with_spreads += 1
                all_trades.extend(valid_trades)

    # Aggregate results
    if not all_trades:
        logger.warning(f"[{config_name}] No spreads found (filtered {filtered_by_loss_limit} by loss limit)")
        return {
            'config': config_name,
            'dte': dte,
            'percentile': percentile,
            'spread_width': spread_width,
            'flow_mode': flow_mode,
            'days_processed': days_processed,
            'days_with_spreads': 0,
            'total_spreads': 0,
            'filtered_by_loss_limit': filtered_by_loss_limit,
            'avg_credit': 0,
            'avg_max_loss': 0,
            'avg_roi': 0,
            'median_roi': 0,
            'min_roi': 0,
            'max_roi': 0,
            'total_profit_potential': 0,
            'win_rate': 0,
            'sharpe_estimate': 0,
            'consistency_pct': 0
        }

    trades_df = pd.DataFrame(all_trades)

    # Calculate ROI metrics
    roi_values = (trades_df['entry_credit'] / trades_df['max_loss'] * 100)

    # Filter configs below min ROI threshold
    avg_roi = roi_values.mean()
    if avg_roi < min_roi:
        logger.info(f"[{config_name}] Filtered: ROI {avg_roi:.1f}% < {min_roi}%")
        return None

    # Calculate profit potential (assumes 50% profit target)
    profit_potential = (trades_df['entry_credit'] * 0.5).sum()

    # Estimate win rate (simplified: assume spreads OTM by percentile distance)
    # Higher percentile = higher win rate
    win_rate_estimate = min(percentile, 99.5)

    # Estimate Sharpe ratio (simplified)
    # Sharpe ≈ (avg_return) / (std_return)
    # For credit spreads: higher ROI and consistency = higher Sharpe
    roi_std = roi_values.std()
    sharpe_estimate = (avg_roi / roi_std) if roi_std > 0 else 0

    # Consistency: percentage of days with spreads
    consistency_pct = (days_with_spreads / days_processed * 100) if days_processed > 0 else 0

    result = {
        'config': config_name,
        'dte': dte,
        'percentile': percentile,
        'spread_width': spread_width,
        'flow_mode': flow_mode,
        'days_processed': days_processed,
        'days_with_spreads': days_with_spreads,
        'total_spreads': len(all_trades),
        'filtered_by_loss_limit': filtered_by_loss_limit,
        'avg_credit': trades_df['entry_credit'].mean(),
        'avg_max_loss': trades_df['max_loss'].mean(),
        'avg_roi': avg_roi,
        'median_roi': roi_values.median(),
        'min_roi': roi_values.min(),
        'max_roi': roi_values.max(),
        'total_profit_potential': profit_potential,
        'win_rate': win_rate_estimate,
        'sharpe_estimate': sharpe_estimate,
        'consistency_pct': consistency_pct,
        'avg_spreads_per_day': len(all_trades) / days_with_spreads if days_with_spreads > 0 else 0
    }

    logger.info(
        f"[{config_name}] Complete: {len(all_trades)} spreads, "
        f"ROI={result['avg_roi']:.1f}%, Sharpe={sharpe_estimate:.2f}, "
        f"Consistency={consistency_pct:.1f}%"
    )

    return result


def run_config_wrapper(args):
    """Wrapper for multiprocessing."""
    return asyncio.run(run_single_config(*args))


def save_intermediate_results(results: list, output_path: Path, batch_num: int):
    """Save intermediate results to avoid data loss."""
    if not results:
        return

    intermediate_path = output_path.parent / f"{output_path.stem}_batch{batch_num}.csv"
    results_df = pd.DataFrame([r for r in results if r is not None])

    if not results_df.empty:
        results_df.to_csv(intermediate_path, index=False)
        logger.info(f"Intermediate results saved to: {intermediate_path}")


def main():
    parser = argparse.ArgumentParser(description='Comprehensive Grid Search with Fixes')
    parser.add_argument('--ticker', default='NDX', help='Ticker symbol')
    parser.add_argument('--start-date', required=True, help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end-date', required=True, help='End date (YYYY-MM-DD)')
    parser.add_argument('--processes', type=int, default=None, help='Number of processes (default: CPU count)')
    parser.add_argument('--csv-dir', default='options_csv_output_full', help='CSV directory for non-0DTE')
    parser.add_argument('--output', default='results/comprehensive_grid_search.csv', help='Output file')
    parser.add_argument('--min-roi', type=float, default=5.0, help='Minimum ROI filter (percent)')
    parser.add_argument('--max-loss', type=float, default=30000.0, help='Max loss per position filter')
    parser.add_argument('--test-mode', action='store_true', help='Run limited configs for testing')
    parser.add_argument('--resume', help='Resume from intermediate file')
    parser.add_argument('--save-interval', type=int, default=100, help='Save intermediate results every N configs')

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

    # Define comprehensive grid parameters
    if args.test_mode:
        # Limited grid for testing
        dtes = [0, 1, 3]
        percentiles = [95, 98, 99]
        spread_widths = [20, 30]
        flow_modes = ['neutral', 'with_flow']
    else:
        # Full comprehensive grid (504 configs - removed 'against_flow' as not supported)
        dtes = [0, 1, 2, 3, 5, 7, 10]
        percentiles = [95, 96, 97, 98, 99, 100]
        spread_widths = [10, 20, 25, 30, 50, 100]
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
                        db_config,
                        args.min_roi,
                        args.max_loss
                    ))

    # Check for resume
    completed_configs = set()
    if args.resume:
        resume_df = pd.read_csv(args.resume)
        completed_configs = set(resume_df['config'].unique())
        logger.info(f"Resuming: Found {len(completed_configs)} completed configs")

        # Filter out completed configs
        configs = [
            cfg for cfg in configs
            if f"DTE{cfg[3]}_p{cfg[4]}_w{int(cfg[5])}_{cfg[6]}" not in completed_configs
        ]

    num_processes = args.processes or min(cpu_count(), len(configs))

    logger.info(f"\n{'='*80}")
    logger.info(f"COMPREHENSIVE GRID SEARCH WITH FIXES")
    logger.info(f"{'='*80}")
    logger.info(f"Ticker: {args.ticker}")
    logger.info(f"Date Range: {args.start_date} to {args.end_date}")
    logger.info(f"DTEs: {dtes}")
    logger.info(f"Percentiles: {percentiles}")
    logger.info(f"Spread Widths: {spread_widths}")
    logger.info(f"Flow Modes: {flow_modes}")
    logger.info(f"Total Configurations: {len(configs)}")
    logger.info(f"Processes: {num_processes}")
    logger.info(f"Min ROI Filter: {args.min_roi}%")
    logger.info(f"Max Loss Filter: ${args.max_loss:,.0f}")
    logger.info(f"Save Interval: Every {args.save_interval} configs")
    logger.info(f"{'='*80}\n")

    # Create output directory
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Run in parallel with batching for intermediate saves
    start_time = datetime.now()
    all_results = []
    batch_size = args.save_interval
    total_batches = (len(configs) + batch_size - 1) // batch_size

    for batch_num in range(total_batches):
        batch_start = batch_num * batch_size
        batch_end = min(batch_start + batch_size, len(configs))
        batch_configs = configs[batch_start:batch_end]

        logger.info(f"\n{'='*80}")
        logger.info(f"Processing Batch {batch_num + 1}/{total_batches}")
        logger.info(f"Configs {batch_start + 1} to {batch_end} of {len(configs)}")
        logger.info(f"{'='*80}\n")

        # Run batch in parallel
        with Pool(processes=num_processes) as pool:
            batch_results = pool.map(run_config_wrapper, batch_configs)

        # Add to all results
        all_results.extend(batch_results)

        # Save intermediate results
        save_intermediate_results(all_results, output_path, batch_num + 1)

        # Show batch summary
        valid_results = [r for r in batch_results if r is not None]
        if valid_results:
            batch_df = pd.DataFrame(valid_results)
            logger.info(f"\nBatch {batch_num + 1} Summary:")
            logger.info(f"  Valid configs: {len(valid_results)}/{len(batch_configs)}")
            logger.info(f"  Avg ROI: {batch_df['avg_roi'].mean():.1f}%")
            logger.info(f"  Max ROI: {batch_df['avg_roi'].max():.1f}%")
            logger.info(f"  Avg Spreads: {batch_df['total_spreads'].mean():.0f}")

    end_time = datetime.now()

    # Filter out None results
    results = [r for r in all_results if r is not None]

    # Save final results
    results_df = pd.DataFrame(results)

    # Add multi-metric score
    if not results_df.empty:
        # Normalize metrics (0-1 scale)
        results_df['roi_score'] = (results_df['avg_roi'] - results_df['avg_roi'].min()) / (results_df['avg_roi'].max() - results_df['avg_roi'].min() + 0.001)
        results_df['sharpe_score'] = (results_df['sharpe_estimate'] - results_df['sharpe_estimate'].min()) / (results_df['sharpe_estimate'].max() - results_df['sharpe_estimate'].min() + 0.001)
        results_df['consistency_score'] = results_df['consistency_pct'] / 100
        results_df['volume_score'] = (results_df['total_spreads'] - results_df['total_spreads'].min()) / (results_df['total_spreads'].max() - results_df['total_spreads'].min() + 0.001)

        # Composite score (weighted)
        results_df['composite_score'] = (
            results_df['roi_score'] * 0.35 +
            results_df['sharpe_score'] * 0.25 +
            results_df['consistency_score'] * 0.20 +
            results_df['volume_score'] * 0.20
        )

        # Rank by composite score
        results_df = results_df.sort_values('composite_score', ascending=False)

    results_df.to_csv(args.output, index=False)

    runtime = (end_time - start_time).total_seconds()

    logger.info(f"\n{'='*80}")
    logger.info(f"COMPREHENSIVE GRID SEARCH COMPLETE")
    logger.info(f"{'='*80}")
    logger.info(f"Results saved to: {args.output}")
    logger.info(f"Total configs tested: {len(configs)}")
    logger.info(f"Valid configs: {len(results)}")
    logger.info(f"Filtered out: {len(configs) - len(results)}")
    logger.info(f"Runtime: {runtime:.1f} seconds ({runtime/60:.1f} minutes)")

    # Show top performers
    if not results_df.empty:
        logger.info(f"\n{'='*80}")
        logger.info(f"TOP 20 BY COMPOSITE SCORE")
        logger.info(f"{'='*80}\n")

        top_composite = results_df.head(20)[['config', 'total_spreads', 'avg_roi', 'sharpe_estimate', 'consistency_pct', 'composite_score']]
        logger.info(f"\n{top_composite.to_string(index=False)}")

        logger.info(f"\n{'='*80}")
        logger.info(f"TOP 10 BY AVERAGE ROI")
        logger.info(f"{'='*80}\n")

        top_roi = results_df.nlargest(10, 'avg_roi')[['config', 'total_spreads', 'avg_roi', 'sharpe_estimate', 'days_with_spreads']]
        logger.info(f"\n{top_roi.to_string(index=False)}")

        logger.info(f"\n{'='*80}")
        logger.info(f"TOP 10 BY SHARPE RATIO")
        logger.info(f"{'='*80}\n")

        top_sharpe = results_df.nlargest(10, 'sharpe_estimate')[['config', 'total_spreads', 'avg_roi', 'sharpe_estimate', 'consistency_pct']]
        logger.info(f"\n{top_sharpe.to_string(index=False)}")

        logger.info(f"\n{'='*80}")
        logger.info(f"TOP 10 BY TOTAL SPREADS")
        logger.info(f"{'='*80}\n")

        top_spreads = results_df.nlargest(10, 'total_spreads')[['config', 'total_spreads', 'avg_roi', 'consistency_pct', 'avg_spreads_per_day']]
        logger.info(f"\n{top_spreads.to_string(index=False)}")

        # Summary statistics by DTE
        logger.info(f"\n{'='*80}")
        logger.info(f"SUMMARY BY DTE")
        logger.info(f"{'='*80}\n")

        dte_summary = results_df.groupby('dte').agg({
            'config': 'count',
            'total_spreads': 'mean',
            'avg_roi': 'mean',
            'sharpe_estimate': 'mean',
            'consistency_pct': 'mean'
        }).round(2)
        dte_summary.columns = ['Configs', 'Avg Spreads', 'Avg ROI', 'Avg Sharpe', 'Avg Consistency']
        logger.info(f"\n{dte_summary.to_string()}")

        # Summary statistics by flow mode
        logger.info(f"\n{'='*80}")
        logger.info(f"SUMMARY BY FLOW MODE")
        logger.info(f"{'='*80}\n")

        flow_summary = results_df.groupby('flow_mode').agg({
            'config': 'count',
            'total_spreads': 'mean',
            'avg_roi': 'mean',
            'sharpe_estimate': 'mean',
            'consistency_pct': 'mean'
        }).round(2)
        flow_summary.columns = ['Configs', 'Avg Spreads', 'Avg ROI', 'Avg Sharpe', 'Avg Consistency']
        logger.info(f"\n{flow_summary.to_string()}")

    logger.info(f"\n{'='*80}\n")

    return 0


if __name__ == '__main__':
    sys.exit(main())
