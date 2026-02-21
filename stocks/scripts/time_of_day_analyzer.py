#!/usr/bin/env python3
"""
Time-of-Day Performance Analyzer

Analyzes credit spread performance across different trading windows to identify
optimal entry times throughout the day. Particularly useful for 0DTE strategies
where intraday timing matters.

For 0DTE: Analyzes multiple timestamps per day
For 1+ DTE: Analyzes single entry time per day
"""

import asyncio
import argparse
import sys
import json
from pathlib import Path
from datetime import datetime, time as dt_time, timedelta
from typing import List, Dict, Optional, Tuple
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

logger = get_logger("time_of_day_analyzer", level="INFO")


# Define trading windows
TRADING_WINDOWS = [
    ('09:30', '10:30', 'Market Open'),
    ('10:30', '11:30', 'Early Momentum'),
    ('11:30', '12:30', 'Mid Morning'),
    ('12:30', '13:30', 'Lunch Hour'),
    ('13:30', '14:30', 'Early Afternoon'),
    ('14:30', '15:30', 'Late Afternoon'),
    ('15:30', '16:00', 'Power Hour'),
]


def filter_to_time_window(
    df: pd.DataFrame,
    start_time_str: str,
    end_time_str: str
) -> pd.DataFrame:
    """
    Filter DataFrame to specific time window.

    Args:
        df: DataFrame with 'timestamp' column
        start_time_str: Start time as 'HH:MM'
        end_time_str: End time as 'HH:MM'

    Returns:
        Filtered DataFrame
    """
    if df.empty:
        return df

    # Parse times
    start_hour, start_min = map(int, start_time_str.split(':'))
    end_hour, end_min = map(int, end_time_str.split(':'))

    start_time = dt_time(start_hour, start_min)
    end_time = dt_time(end_hour, end_min)

    # Extract time from timestamp, converting UTC to Eastern time
    # 0DTE timestamps are stored as UTC ISO8601 (e.g., '2026-01-02T14:30:00+00:00')
    # 14:30 UTC = 09:30 ET (UTC-5 in winter, UTC-4 in summer)
    def _to_et_time(x):
        ts = x if hasattr(x, 'tzinfo') else pd.to_datetime(x)
        if ts.tzinfo is not None:
            # Timezone-aware: convert UTC -> ET
            import pytz
            et = pytz.timezone('America/New_York')
            ts = ts.astimezone(et)
        return ts.time()

    df['time_only'] = df['timestamp'].apply(_to_et_time)

    # Filter to window
    mask = (df['time_only'] >= start_time) & (df['time_only'] < end_time)
    filtered = df[mask].copy()

    # Clean up temporary column
    filtered = filtered.drop(columns=['time_only'])

    return filtered


async def analyze_time_window(
    ticker: str,
    start_date: str,
    end_date: str,
    dte: int,
    percentile: int,
    spread_width: float,
    flow_mode: str,
    time_window_start: str,
    time_window_end: str,
    window_label: str,
    csv_dir: str,
    db_config: str
) -> Optional[Dict]:
    """
    Analyze performance for a specific time window.

    Returns:
        Dict with results or None if no data
    """

    config_name = f"DTE{dte}_p{percentile}_w{int(spread_width)}_{flow_mode}"
    window_name = f"{time_window_start}-{time_window_end}"

    logger.info(f"[{config_name}] [{window_name}] Loading data...")

    # Determine CSV directory
    data_dir = 'options_csv_output' if dte == 0 else csv_dir

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
        logger.warning(f"[{config_name}] [{window_name}] No data")
        return None

    # For 0DTE, filter to time window
    if dte == 0:
        df = filter_to_time_window(df, time_window_start, time_window_end)

        if df.empty:
            logger.warning(f"[{config_name}] [{window_name}] No data in time window")
            return None

    # Get trading dates
    df['trading_date'] = df['timestamp'].apply(
        lambda x: x.date() if hasattr(x, 'date') else pd.to_datetime(x).date()
    )
    trading_dates = sorted(df['trading_date'].unique())

    # Initialize
    db = StockQuestDB(db_config, logger=logger)
    integrator = PercentileSpreadIntegrator(db_config=db_config)

    all_trades = []
    days_processed = 0
    days_with_spreads = 0

    # Process each day
    for trading_date in trading_dates:
        day_df = df[df['trading_date'] == trading_date]

        # For 0DTE with time window, filter to single timestamp per day
        if dte == 0 and not day_df.empty:
            timestamps = sorted(day_df['timestamp'].unique())
            if timestamps:
                entry_timestamp = timestamps[0]  # Use earliest in window
                day_df = day_df[day_df['timestamp'] == entry_timestamp].copy()

        if day_df.empty:
            continue

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

    # Calculate metrics
    if not all_trades:
        logger.warning(f"[{config_name}] [{window_name}] No spreads found")
        return None

    trades_df = pd.DataFrame(all_trades)

    # Calculate ROI
    roi_values = (trades_df['entry_credit'] / trades_df['max_loss'] * 100)

    # Calculate profit potential (assumes 50% profit target)
    profit_potential = (trades_df['entry_credit'] * 0.5).sum()

    # Consistency score: (days_with_spreads / days_processed) * avg_spreads_per_day
    avg_spreads_per_day = len(all_trades) / days_with_spreads if days_with_spreads > 0 else 0
    consistency_score = (days_with_spreads / days_processed * 100) * avg_spreads_per_day if days_processed > 0 else 0

    result = {
        'config': config_name,
        'dte': dte,
        'percentile': percentile,
        'spread_width': spread_width,
        'flow_mode': flow_mode,
        'time_window': window_name,
        'window_label': window_label,
        'days_processed': days_processed,
        'days_with_spreads': days_with_spreads,
        'total_opportunities': len(all_trades),
        'avg_opportunities_per_day': avg_spreads_per_day,
        'avg_roi': roi_values.mean(),
        'median_roi': roi_values.median(),
        'min_roi': roi_values.min(),
        'max_roi': roi_values.max(),
        'avg_credit': trades_df['entry_credit'].mean(),
        'avg_max_loss': trades_df['max_loss'].mean(),
        'total_profit_potential': profit_potential,
        'consistency_score': consistency_score
    }

    logger.info(
        f"[{config_name}] [{window_name}] Complete: {len(all_trades)} spreads, "
        f"ROI={result['avg_roi']:.1f}%, Profit=${profit_potential:,.0f}"
    )

    return result


def analyze_window_wrapper(args):
    """Wrapper for multiprocessing."""
    return asyncio.run(analyze_time_window(*args))


def build_analysis_tasks(
    ticker: str,
    start_date: str,
    end_date: str,
    configs: List[Tuple],
    csv_dir: str,
    db_config: str
) -> List[Tuple]:
    """
    Build list of analysis tasks for parallel execution.

    Returns:
        List of argument tuples for analyze_time_window
    """
    tasks = []

    for dte, percentile, spread_width, flow_mode in configs:
        # Only analyze time windows for 0DTE
        if dte == 0:
            windows_to_test = TRADING_WINDOWS
        else:
            # For 1+ DTE, just run once (no time window filtering)
            windows_to_test = [('09:30', '16:00', 'Full Day')]

        for start_time, end_time, label in windows_to_test:
            tasks.append((
                ticker,
                start_date,
                end_date,
                dte,
                percentile,
                spread_width,
                flow_mode,
                start_time,
                end_time,
                label,
                csv_dir,
                db_config
            ))

    return tasks


def main():
    parser = argparse.ArgumentParser(description='Time-of-Day Performance Analyzer')
    parser.add_argument('--ticker', default='NDX', help='Ticker symbol')
    parser.add_argument('--start-date', required=True, help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end-date', required=True, help='End date (YYYY-MM-DD)')
    parser.add_argument('--csv-dir', default='options_csv_output_full', help='CSV directory for 1+ DTE')
    parser.add_argument('--config-file', help='JSON file with configs to test')
    parser.add_argument('--dte', type=int, help='Single DTE to test')
    parser.add_argument('--percentile', type=int, help='Single percentile to test')
    parser.add_argument('--spread-width', type=float, help='Single spread width to test')
    parser.add_argument('--flow-mode', default='neutral', help='Flow mode')
    parser.add_argument('--output', default='results/time_of_day_analysis.csv', help='Output file')
    parser.add_argument('--processes', type=int, default=None, help='Number of processes (default: CPU count)')

    args = parser.parse_args()

    # Get DB config
    db_config = (
        os.getenv('QUEST_DB_STRING') or
        os.getenv('QUESTDB_CONNECTION_STRING') or
        os.getenv('QUESTDB_URL')
    )

    if not db_config:
        print("ERROR: No database configuration found")
        return 1

    # Build config list
    if args.config_file:
        # Load from JSON
        with open(args.config_file, 'r') as f:
            config_data = json.load(f)

        configs = []
        for cfg in config_data.get('configurations', []):
            configs.append((
                cfg['dte'],
                cfg['percentile'],
                cfg['spread_width'],
                cfg.get('flow_mode', 'neutral')
            ))

    elif args.dte is not None and args.percentile is not None and args.spread_width is not None:
        # Single config from args
        configs = [(args.dte, args.percentile, args.spread_width, args.flow_mode)]

    else:
        # Default: test top configs from Phase 1
        configs = [
            (0, 99, 20, 'neutral'),
            (0, 98, 20, 'with_flow'),
            (1, 99, 20, 'neutral'),
            (1, 99, 20, 'with_flow'),
            (3, 99, 30, 'neutral'),
        ]

    # Build analysis tasks
    tasks = build_analysis_tasks(
        ticker=args.ticker,
        start_date=args.start_date,
        end_date=args.end_date,
        configs=configs,
        csv_dir=args.csv_dir,
        db_config=db_config
    )

    num_processes = args.processes or min(cpu_count(), len(tasks))

    logger.info(f"\n{'='*80}")
    logger.info(f"TIME-OF-DAY PERFORMANCE ANALYZER")
    logger.info(f"{'='*80}")
    logger.info(f"Ticker: {args.ticker}")
    logger.info(f"Date Range: {args.start_date} to {args.end_date}")
    logger.info(f"Configurations: {len(configs)}")
    logger.info(f"Time Windows: {len(TRADING_WINDOWS)}")
    logger.info(f"Total Analysis Tasks: {len(tasks)}")
    logger.info(f"Processes: {num_processes}")
    logger.info(f"{'='*80}\n")

    # Run analysis in parallel
    start_time = datetime.now()
    with Pool(processes=num_processes) as pool:
        results = pool.map(analyze_window_wrapper, tasks)
    end_time = datetime.now()

    # Filter out None results
    results = [r for r in results if r is not None]
    results_df = pd.DataFrame(results)

    if results_df.empty:
        print("No results found")
        return 1

    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(output_path, index=False)

    runtime = (end_time - start_time).total_seconds()

    logger.info(f"\n{'='*80}")
    logger.info(f"ANALYSIS COMPLETE")
    logger.info(f"{'='*80}")
    logger.info(f"Results saved to: {output_path}")
    logger.info(f"Total results: {len(results_df)}")
    logger.info(f"Runtime: {runtime:.1f} seconds ({runtime/60:.1f} minutes)")

    # Show top performers
    if not results_df.empty:
        logger.info(f"\n{'='*80}")
        logger.info(f"TOP 10 BY TOTAL PROFIT POTENTIAL")
        logger.info(f"{'='*80}\n")

        top_profit = results_df.nlargest(10, 'total_profit_potential')[[
            'config', 'window_label', 'total_opportunities', 'avg_roi',
            'total_profit_potential', 'consistency_score'
        ]]
        logger.info(f"\n{top_profit.to_string(index=False)}")

        logger.info(f"\n{'='*80}")
        logger.info(f"TOP 10 BY AVERAGE ROI")
        logger.info(f"{'='*80}\n")

        top_roi = results_df.nlargest(10, 'avg_roi')[[
            'config', 'window_label', 'total_opportunities', 'avg_roi',
            'total_profit_potential', 'days_with_spreads'
        ]]
        logger.info(f"\n{top_roi.to_string(index=False)}")

        logger.info(f"\n{'='*80}")
        logger.info(f"TOP 10 BY OPPORTUNITY COUNT")
        logger.info(f"{'='*80}\n")

        top_opps = results_df.nlargest(10, 'total_opportunities')[[
            'config', 'window_label', 'total_opportunities', 'avg_roi',
            'avg_opportunities_per_day', 'consistency_score'
        ]]
        logger.info(f"\n{top_opps.to_string(index=False)}")

        # Best window analysis for 0DTE
        dte0_results = results_df[results_df['dte'] == 0]
        if not dte0_results.empty:
            logger.info(f"\n{'='*80}")
            logger.info(f"BEST TIME WINDOWS FOR 0DTE")
            logger.info(f"{'='*80}\n")

            window_summary = dte0_results.groupby('window_label').agg({
                'total_opportunities': 'sum',
                'avg_roi': 'mean',
                'total_profit_potential': 'sum',
                'consistency_score': 'mean'
            }).sort_values('total_profit_potential', ascending=False)

            logger.info(f"\n{window_summary.to_string()}")

    logger.info(f"\n{'='*80}\n")

    return 0


if __name__ == '__main__':
    sys.exit(main())
