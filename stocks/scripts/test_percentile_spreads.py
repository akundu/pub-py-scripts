#!/usr/bin/env python3
"""
Test script for percentile-based credit spread analysis.

This script demonstrates the percentile-based strike selection
system with a simple backtest on a few trading days.
"""

import asyncio
import argparse
import logging
import sys
from pathlib import Path
from datetime import datetime
import pandas as pd

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
logger = get_logger("test_percentile_spreads", level="INFO")


async def test_single_day(
    ticker: str,
    trading_date: str,
    dte: int,
    percentile: int,
    spread_width: float,
    flow_mode: str = 'neutral',
    csv_dir: str = 'options_csv_output_full',
    db_config: str = None
):
    """Test percentile spread analysis on a single day."""

    logger.info(f"\n{'='*80}")
    logger.info(f"Testing Percentile-Based Spreads")
    logger.info(f"{'='*80}")
    logger.info(f"Ticker: {ticker}")
    logger.info(f"Date: {trading_date}")
    logger.info(f"DTE: {dte}")
    logger.info(f"Percentile: p{percentile}")
    logger.info(f"Spread Width: {spread_width} points")
    logger.info(f"Flow Mode: {flow_mode}")
    logger.info(f"{'='*80}\n")

    # Load data
    logger.info(f"Loading options data from {csv_dir}/{ticker}...")
    try:
        df = load_multi_dte_data(
            csv_dir=csv_dir,
            ticker=ticker,
            start_date=trading_date,
            end_date=trading_date,
            dte_buckets=(dte,),
            dte_tolerance=1,
            cache_dir=None,
            no_cache=True,
            logger=logger
        )
    except Exception as e:
        logger.error(f"Failed to load data: {e}")
        return None

    if df.empty:
        logger.error(f"No data found for {ticker} on {trading_date}")
        return None

    logger.info(f"Loaded {len(df):,} option records")

    # Get previous close
    logger.info("Getting previous close price...")
    db = StockQuestDB(db_config, logger=logger)

    first_ts = df['timestamp'].min()
    prev_close_result = await get_previous_close_price(db, ticker, first_ts, logger)

    if prev_close_result is None:
        logger.error(f"Could not get previous close for {ticker} on {trading_date}")
        return None

    prev_close, prev_close_date = prev_close_result
    logger.info(f"Previous close: ${prev_close:.2f} (from {prev_close_date})")

    # Initialize integrator
    logger.info("\nInitializing percentile-based spread integrator...")
    integrator = PercentileSpreadIntegrator(db_config=db_config)

    # Analyze spreads
    logger.info(f"\nBuilding spreads with p{percentile} strikes...")
    trades = await integrator.analyze_single_day(
        options_df=df,
        ticker=ticker,
        trading_date=datetime.strptime(trading_date, '%Y-%m-%d'),
        prev_close=prev_close,
        dte=dte,
        percentile=percentile,
        spread_width=spread_width,
        profit_target_pct=0.5,
        flow_mode=flow_mode,
        use_mid=False
    )

    # Display results
    logger.info(f"\n{'='*80}")
    logger.info(f"RESULTS")
    logger.info(f"{'='*80}")
    logger.info(f"Total spreads found: {len(trades)}")

    if trades:
        logger.info(f"\nTop 5 Spreads by Entry Credit:")
        logger.info(f"{'-'*80}")

        # Sort by entry credit
        sorted_trades = sorted(trades, key=lambda x: x.get('entry_credit', 0), reverse=True)

        for i, trade in enumerate(sorted_trades[:5], 1):
            logger.info(f"\n#{i}: {trade['strategy_type'].upper()}")
            logger.info(f"  Entry Credit: ${trade['entry_credit']:.2f}")
            logger.info(f"  Max Loss: ${trade['max_loss']:.2f}")

            if 'rr_ratio' in trade:
                logger.info(f"  R/R Ratio: {trade['rr_ratio']:.3f}")

            if 'short_strike' in trade:
                logger.info(f"  Short Strike: {trade['short_strike']:.2f}")
                logger.info(f"  Long Strike: {trade['long_strike']:.2f}")

            if 'short_put_strike' in trade:
                logger.info(f"  Short Put Strike: {trade['short_put_strike']:.2f}")
                logger.info(f"  Short Call Strike: {trade['short_call_strike']:.2f}")

            # Calculate potential ROI
            if trade['max_loss'] > 0:
                entry_roi = (trade['entry_credit'] / trade['max_loss']) * 100
                logger.info(f"  Entry ROI: {entry_roi:.2f}%")

        # Summary statistics
        logger.info(f"\n{'-'*80}")
        logger.info(f"Summary Statistics:")
        logger.info(f"  Average Entry Credit: ${sum(t['entry_credit'] for t in trades) / len(trades):.2f}")
        logger.info(f"  Average Max Loss: ${sum(t['max_loss'] for t in trades) / len(trades):.2f}")

        avg_roi = sum((t['entry_credit'] / t['max_loss']) * 100 for t in trades if t['max_loss'] > 0) / len(trades)
        logger.info(f"  Average Entry ROI: {avg_roi:.2f}%")

    else:
        logger.warning("No spreads found with current parameters")

    logger.info(f"\n{'='*80}\n")

    return trades


async def test_multiple_days(
    ticker: str,
    start_date: str,
    end_date: str,
    dte: int,
    percentile: int,
    spread_width: float,
    flow_mode: str = 'neutral',
    csv_dir: str = 'options_csv_output_full',
    db_config: str = None
):
    """Test on multiple days and aggregate results."""

    logger.info(f"\n{'='*80}")
    logger.info(f"Multi-Day Percentile Spread Backtest")
    logger.info(f"{'='*80}")
    logger.info(f"Ticker: {ticker}")
    logger.info(f"Date Range: {start_date} to {end_date}")
    logger.info(f"DTE: {dte}, Percentile: p{percentile}, Width: {spread_width}")
    logger.info(f"{'='*80}\n")

    # Load data for date range
    logger.info(f"Loading multi-day options data...")
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
        logger.error("No data found")
        return

    # Get unique trading dates
    df['trading_date'] = df['timestamp'].apply(
        lambda x: x.date() if hasattr(x, 'date') else pd.to_datetime(x).date()
    )
    trading_dates = sorted(df['trading_date'].unique())
    logger.info(f"Found {len(trading_dates)} trading days")

    # Initialize
    db = StockQuestDB(db_config, logger=logger)
    integrator = PercentileSpreadIntegrator(db_config=db_config)

    all_trades = []

    # Process each day
    for i, trading_date in enumerate(trading_dates, 1):
        logger.info(f"\nProcessing day {i}/{len(trading_dates)}: {trading_date}")

        day_df = df[df['trading_date'] == trading_date]
        first_ts = day_df['timestamp'].min()

        # Get previous close
        prev_close_result = await get_previous_close_price(db, ticker, first_ts, logger)
        if prev_close_result is None:
            logger.warning(f"  No previous close, skipping")
            continue

        prev_close, _ = prev_close_result

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

        logger.info(f"  Found {len(trades)} spreads, prev_close=${prev_close:.2f}")
        all_trades.extend(trades)

    # Aggregate results
    logger.info(f"\n{'='*80}")
    logger.info(f"AGGREGATED RESULTS")
    logger.info(f"{'='*80}")
    logger.info(f"Total Trading Days: {len(trading_dates)}")
    logger.info(f"Total Spreads Found: {len(all_trades)}")

    if all_trades:
        trades_df = pd.DataFrame(all_trades)

        logger.info(f"\nBy Strategy Type:")
        for strategy, group in trades_df.groupby('strategy_type'):
            avg_credit = group['entry_credit'].mean()
            avg_loss = group['max_loss'].mean()
            avg_roi = (group['entry_credit'] / group['max_loss'] * 100).mean()

            logger.info(f"  {strategy}: {len(group)} spreads")
            logger.info(f"    Avg Credit: ${avg_credit:.2f}")
            logger.info(f"    Avg Max Loss: ${avg_loss:.2f}")
            logger.info(f"    Avg ROI: {avg_roi:.2f}%")

        logger.info(f"\nOverall:")
        logger.info(f"  Avg Credit: ${trades_df['entry_credit'].mean():.2f}")
        logger.info(f"  Avg Max Loss: ${trades_df['max_loss'].mean():.2f}")
        avg_overall_roi = (trades_df['entry_credit'] / trades_df['max_loss'] * 100).mean()
        logger.info(f"  Avg ROI: {avg_overall_roi:.2f}%")

    logger.info(f"{'='*80}\n")


async def main():
    parser = argparse.ArgumentParser(description='Test percentile-based credit spreads')
    parser.add_argument('--ticker', default='NDX', help='Ticker symbol')
    parser.add_argument('--date', help='Single trading date (YYYY-MM-DD)')
    parser.add_argument('--start-date', help='Start date for multi-day test')
    parser.add_argument('--end-date', help='End date for multi-day test')
    parser.add_argument('--dte', type=int, default=3, help='Days to expiration')
    parser.add_argument('--percentile', type=int, default=97, help='Percentile (95, 97, 99, etc.)')
    parser.add_argument('--spread-width', type=float, default=20.0, help='Spread width in points')
    parser.add_argument('--flow-mode', default='neutral', choices=['neutral', 'with_flow', 'against_flow'],
                        help='Flow mode strategy')
    parser.add_argument('--csv-dir', default='options_csv_output_full', help='CSV directory')
    parser.add_argument('--db-config', help='QuestDB connection string')

    args = parser.parse_args()

    # Use environment variable if db-config not provided
    if not args.db_config:
        import os
        args.db_config = (
            os.getenv('QUEST_DB_STRING') or
            os.getenv('QUESTDB_CONNECTION_STRING') or
            os.getenv('QUESTDB_URL')
        )

    if args.date:
        # Single day test
        await test_single_day(
            ticker=args.ticker,
            trading_date=args.date,
            dte=args.dte,
            percentile=args.percentile,
            spread_width=args.spread_width,
            flow_mode=args.flow_mode,
            csv_dir=args.csv_dir,
            db_config=args.db_config
        )
    elif args.start_date and args.end_date:
        # Multi-day test
        await test_multiple_days(
            ticker=args.ticker,
            start_date=args.start_date,
            end_date=args.end_date,
            dte=args.dte,
            percentile=args.percentile,
            spread_width=args.spread_width,
            flow_mode=args.flow_mode,
            csv_dir=args.csv_dir,
            db_config=args.db_config
        )
    else:
        logger.error("Must specify either --date or both --start-date and --end-date")
        return 1

    return 0


if __name__ == '__main__':
    sys.exit(asyncio.run(main()))
