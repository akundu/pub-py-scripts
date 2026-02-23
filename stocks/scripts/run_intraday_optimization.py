#!/usr/bin/env python3
"""
Intraday Optimization Runner

Comprehensive grid search to find optimal trading times throughout the day.
Tests all combinations of time windows, DTEs, percentiles, and spread widths.
"""

import asyncio
import argparse
import sys
from pathlib import Path
from datetime import datetime
import os

# Add project root to path
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from intraday_optimizer import (
    IntradayGridSearch,
    TrainingValidator,
    ScheduleGenerator
)
from common.logging_utils import get_logger

logger = get_logger("intraday_optimizer", level="INFO")


async def main():
    parser = argparse.ArgumentParser(
        description='Intraday Optimization System',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run full 3-month backtest
  python scripts/run_intraday_optimization.py \\
    --ticker NDX \\
    --training-months 3 \\
    --max-loss 30000 \\
    --min-roi 5.0

  # Quick test (1 week)
  python scripts/run_intraday_optimization.py \\
    --ticker NDX \\
    --training-months 0.25 \\
    --quick-test
        """
    )

    parser.add_argument('--ticker', default='NDX', help='Ticker symbol')
    parser.add_argument('--end-date', default=None,
                       help='End date (YYYY-MM-DD), defaults to today')
    parser.add_argument('--training-months', type=float, default=3.0,
                       help='Number of months for training')
    parser.add_argument('--validation-days', type=int, default=0,
                       help='Number of days for validation (0 = use all as training)')

    parser.add_argument('--max-loss', type=float, default=30000,
                       help='Max loss per trade ($)')
    parser.add_argument('--min-roi', type=float, default=5.0,
                       help='Minimum ROI percentage')

    parser.add_argument('--time-window-minutes', type=int, default=10,
                       help='Time window size (minutes)')
    parser.add_argument('--dte-min', type=int, default=0,
                       help='Minimum DTE')
    parser.add_argument('--dte-max', type=int, default=15,
                       help='Maximum DTE')
    parser.add_argument('--percentile-min', type=int, default=95,
                       help='Minimum percentile')
    parser.add_argument('--percentile-max', type=int, default=100,
                       help='Maximum percentile')
    parser.add_argument('--spread-widths', type=str, default='10,20,30,50,100',
                       help='Comma-separated spread widths')

    parser.add_argument('--strategy', default='maximize_roi',
                       choices=['maximize_roi', 'maximize_opportunities', 'balanced'],
                       help='Schedule generation strategy')
    parser.add_argument('--top-n', type=int, default=None,
                       help='Limit schedule to top N configs (None = unlimited)')

    parser.add_argument('--output-dir', default='results/intraday',
                       help='Output directory')
    parser.add_argument('--quick-test', action='store_true',
                       help='Quick test mode (1 week only)')

    args = parser.parse_args()

    # Quick test mode
    if args.quick_test:
        args.training_months = 0.25  # ~1 week
        args.validation_days = 0
        logger.info("Quick test mode: using 1 week of data")

    # Get database config
    db_config = (
        os.getenv('QUEST_DB_STRING') or
        os.getenv('QUESTDB_CONNECTION_STRING') or
        os.getenv('QUESTDB_URL')
    )

    if not db_config:
        print("ERROR: No database configuration found")
        return 1

    # Parse spread widths
    spread_widths = [float(w.strip()) for w in args.spread_widths.split(',')]

    # Determine date range
    end_date = args.end_date or datetime.now().strftime('%Y-%m-%d')

    validator = TrainingValidator(
        training_months=int(args.training_months),
        validation_days=args.validation_days
    )

    if args.validation_days > 0:
        (train_start, train_end), (val_start, val_end) = validator.get_periods(end_date)
        logger.info(f"Training: {train_start} to {train_end}")
        logger.info(f"Validation: {val_start} to {val_end}")
    else:
        # Use all data for training
        (train_start, train_end), _ = validator.get_periods(end_date)
        logger.info(f"Training (no validation): {train_start} to {train_end}")
        val_start, val_end = None, None

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Initialize grid search
    grid_search = IntradayGridSearch(
        time_window_minutes=args.time_window_minutes,
        dte_range=(args.dte_min, args.dte_max),
        percentile_range=(args.percentile_min, args.percentile_max),
        spread_widths=spread_widths,
        db_config=db_config
    )

    # Progress callback
    def progress(completed, total):
        pct = (completed / total) * 100
        logger.info(f"Progress: {completed:,}/{total:,} ({pct:.1f}%)")

    # Run grid search on training period
    logger.info("\n" + "="*80)
    logger.info("PHASE 1: TRAINING PERIOD ANALYSIS")
    logger.info("="*80)

    train_results = await grid_search.run_grid_search(
        ticker=args.ticker,
        start_date=train_start,
        end_date=train_end,
        max_loss_constraint=args.max_loss,
        min_roi_pct=args.min_roi,
        progress_callback=progress
    )

    # Save raw results
    train_raw_path = output_dir / 'intraday_training_raw.csv'
    train_results.to_csv(train_raw_path, index=False)
    logger.info(f"Raw training results saved: {train_raw_path}")

    # Aggregate results
    logger.info("\nAggregating training results...")
    train_agg = grid_search.aggregate_results(train_results)

    train_agg_path = output_dir / 'intraday_training_aggregated.csv'
    train_agg.to_csv(train_agg_path, index=False)
    logger.info(f"Aggregated training results saved: {train_agg_path}")

    # Generate schedule
    logger.info("\n" + "="*80)
    logger.info("PHASE 2: SCHEDULE GENERATION")
    logger.info("="*80)

    schedule_gen = ScheduleGenerator(block_size_minutes=15)
    schedule = schedule_gen.generate_schedule(
        agg_results=train_agg,
        strategy=args.strategy,
        top_n=args.top_n
    )

    schedule_path = output_dir / 'intraday_schedule.csv'
    schedule.to_csv(schedule_path, index=False)
    logger.info(f"Trading schedule saved: {schedule_path}")

    # Generate summary statistics
    summary = schedule_gen.generate_summary_statistics(
        schedule=schedule,
        training_period=(train_start, train_end),
        validation_period=(val_start, val_end) if val_start else None
    )

    # Print summary
    schedule_gen.print_summary(summary, schedule)

    # Save summary to file
    summary_path = output_dir / 'intraday_summary.txt'
    with open(summary_path, 'w') as f:
        f.write("="*80 + "\n")
        f.write("INTRADAY OPTIMIZATION SUMMARY\n")
        f.write("="*80 + "\n\n")
        f.write(f"TRAINING PERIOD: {summary['training_period']}\n")
        if summary['validation_period'] != "N/A":
            f.write(f"VALIDATION PERIOD: {summary['validation_period']}\n")
        f.write(f"\nCONFIGURATIONS:\n")
        f.write(f"  Total analyzed: {summary['total_configs_analyzed']:,}\n")
        f.write(f"  Tradeable: {summary['tradeable_configs']:,}\n")
        f.write(f"\nPERFORMANCE METRICS:\n")
        f.write(f"  Average ROI: {summary['avg_roi']:.2f}%\n")
        f.write(f"  Median ROI: {summary['median_roi']:.2f}%\n")
        f.write(f"  Max ROI: {summary['max_roi']:.2f}%\n")
        f.write(f"\nOPPORTUNITIES:\n")
        f.write(f"  Daily opportunities: {summary['total_daily_opportunities']:.1f}\n")
        f.write(f"  Avg entry credit: ${summary['avg_entry_credit']:.2f}\n")
        f.write(f"  Avg max loss: ${summary['avg_max_loss']:.2f}\n")
        f.write(f"\nCAPITAL REQUIREMENTS:\n")
        f.write(f"  Per day (unlimited): ${summary['capital_required_per_day']:,.0f}\n")
        f.write(f"\nEXPECTED PROFITS:\n")
        f.write(f"  Daily: ${summary['expected_daily_profit']:,.2f}\n")
        f.write(f"  Monthly: ${summary['expected_monthly_profit']:,.2f}\n")
        f.write(f"  Annual: ${summary['expected_annual_profit']:,.2f}\n")

    logger.info(f"Summary saved: {summary_path}")

    logger.info("\n" + "="*80)
    logger.info("OPTIMIZATION COMPLETE")
    logger.info("="*80)
    logger.info(f"Results saved to: {output_dir}")

    return 0


if __name__ == '__main__':
    sys.exit(asyncio.run(main()))
