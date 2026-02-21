#!/usr/bin/env python3
"""
Simplified Intraday Optimizer

Uses existing percentile infrastructure to test focused set of configurations.
Much simpler and faster than full grid search.
"""

import asyncio
import sys
from pathlib import Path
from datetime import datetime, timedelta
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

logger = get_logger("simplified_intraday", level="INFO")


async def test_single_config(
    ticker: str,
    date: datetime,
    dte: int,
    percentile: int,
    spread_width: float,
    db_config: str
) -> dict:
    """Test a single configuration on a single date."""

    date_str = date.strftime('%Y-%m-%d')

    # Load data
    csv_dir = 'options_csv_output' if dte == 0 else 'options_csv_output_full'

    try:
        df = load_multi_dte_data(
            csv_dir=csv_dir,
            ticker=ticker,
            start_date=date_str,
            end_date=date_str,
            dte_buckets=(dte,),
            dte_tolerance=1,
            cache_dir=None,
            no_cache=False,
            logger=logger
        )

        if df.empty:
            return None

        # Get previous close
        db = StockQuestDB(db_config, logger=logger)
        first_ts = df['timestamp'].min()
        prev_close_result = await get_previous_close_price(db, ticker, first_ts, logger)
        if prev_close_result is None:
            return None
        prev_close, _ = prev_close_result

        # Analyze
        integrator = PercentileSpreadIntegrator(db_config=db_config)
        trades = await integrator.analyze_single_day(
            options_df=df,
            ticker=ticker,
            trading_date=date,
            prev_close=prev_close,
            dte=dte,
            percentile=percentile,
            spread_width=spread_width,
            profit_target_pct=0.5,
            flow_mode='neutral',
            use_mid=False
        )

        if not trades:
            return None

        # Calculate metrics
        credits = [t.get('entry_credit', 0) for t in trades]
        max_losses = [t.get('max_loss', 0) for t in trades]

        avg_credit = sum(credits) / len(credits) if credits else 0
        avg_max_loss = sum(max_losses) / len(max_losses) if max_losses else 0

        # ROI = (50% of credit / max_loss) * 100
        rois = []
        for credit, max_loss in zip(credits, max_losses):
            if max_loss > 0:
                roi = ((credit * 0.5) / max_loss) * 100
                rois.append(roi)

        avg_roi = sum(rois) / len(rois) if rois else 0

        return {
            'date': date_str,
            'dte': dte,
            'percentile': percentile,
            'spread_width': spread_width,
            'num_spreads': len(trades),
            'avg_credit': avg_credit,
            'avg_max_loss': avg_max_loss,
            'avg_roi': avg_roi,
            'total_credit': sum(credits)
        }

    except Exception as e:
        logger.debug(f"Error testing config on {date_str}: {e}")
        return None


async def main():
    import argparse

    parser = argparse.ArgumentParser(description='Simplified Intraday Optimizer')
    parser.add_argument('--ticker', default='NDX')
    parser.add_argument('--start-date', required=True)
    parser.add_argument('--end-date', required=True)
    parser.add_argument('--output', default='results/simplified_intraday_results.csv')

    args = parser.parse_args()

    # Get DB config
    db_config = (
        os.getenv('QUEST_DB_STRING') or
        os.getenv('QUESTDB_CONNECTION_STRING') or
        os.getenv('QUESTDB_URL')
    )

    # Test focused set of configurations
    configs = [
        # DTE, percentile, spread_width
        (0, 95, 10), (0, 95, 20), (0, 95, 30), (0, 95, 50),
        (0, 98, 10), (0, 98, 20), (0, 98, 30), (0, 98, 50),
        (0, 99, 10), (0, 99, 20), (0, 99, 30), (0, 99, 50),
        (1, 95, 10), (1, 95, 20), (1, 95, 30), (1, 95, 50),
        (1, 98, 10), (1, 98, 20), (1, 98, 30), (1, 98, 50),
        (1, 99, 10), (1, 99, 20), (1, 99, 30), (1, 99, 50),
        (3, 95, 20), (3, 95, 30), (3, 95, 50), (3, 95, 100),
        (3, 98, 20), (3, 98, 30), (3, 98, 50), (3, 98, 100),
        (3, 99, 20), (3, 99, 30), (3, 99, 50), (3, 99, 100),
        (5, 95, 20), (5, 95, 50), (5, 95, 100),
        (5, 98, 20), (5, 98, 50), (5, 98, 100),
        (5, 99, 20), (5, 99, 50), (5, 99, 100),
        (7, 95, 50), (7, 95, 100),
        (7, 98, 50), (7, 98, 100),
        (7, 99, 50), (7, 99, 100),
        (10, 95, 50), (10, 95, 100),
        (10, 98, 50), (10, 98, 100),
        (10, 99, 50), (10, 99, 100),
    ]

    logger.info(f"Testing {len(configs)} configurations")

    # Generate date range
    start_dt = datetime.strptime(args.start_date, '%Y-%m-%d')
    end_dt = datetime.strptime(args.end_date, '%Y-%m-%d')

    trading_days = []
    current = start_dt
    while current <= end_dt:
        if current.weekday() < 5:  # Skip weekends
            trading_days.append(current)
        current += timedelta(days=1)

    logger.info(f"Testing across {len(trading_days)} trading days")
    logger.info(f"Total analyses: {len(configs) * len(trading_days)}")

    # Run tests
    results = []
    total = len(configs) * len(trading_days)
    completed = 0

    for date in trading_days:
        logger.info(f"Processing {date.strftime('%Y-%m-%d')}...")

        for dte, percentile, spread_width in configs:
            result = await test_single_config(
                ticker=args.ticker,
                date=date,
                dte=dte,
                percentile=percentile,
                spread_width=spread_width,
                db_config=db_config
            )

            if result:
                results.append(result)

            completed += 1
            if completed % 100 == 0:
                logger.info(f"Progress: {completed}/{total} ({100*completed/total:.1f}%)")

    # Convert to DataFrame
    df = pd.DataFrame(results)

    if df.empty:
        logger.error("No results generated!")
        return 1

    # Aggregate by configuration
    agg = df.groupby(['dte', 'percentile', 'spread_width']).agg({
        'num_spreads': ['sum', 'mean'],
        'avg_credit': 'mean',
        'avg_max_loss': 'mean',
        'avg_roi': 'mean',
        'total_credit': 'sum'
    }).reset_index()

    # Flatten columns
    agg.columns = [
        'dte', 'percentile', 'spread_width',
        'total_opportunities', 'avg_daily_opportunities',
        'avg_entry_credit', 'avg_max_loss', 'avg_roi',
        'total_credit_potential'
    ]

    # Filter by constraints
    agg = agg[
        (agg['avg_max_loss'] <= 30000) &
        (agg['avg_roi'] >= 5.0)
    ]

    # Sort by ROI and opportunities
    agg = agg.sort_values(['avg_roi', 'total_opportunities'], ascending=[False, False])

    # Add rank
    agg['rank'] = range(1, len(agg) + 1)

    # Save
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    agg.to_csv(output_path, index=False)

    # Print top 20
    print("\n" + "="*80)
    print("TOP 20 CONFIGURATIONS (Unlimited Capital)")
    print("="*80)
    print(f"\nPeriod: {args.start_date} to {args.end_date}")
    print(f"Configs tested: {len(configs)}")
    print(f"Configs meeting constraints: {len(agg)}")
    print(f"\nConstraints: Max Loss ≤ $30k, ROI ≥ 5%\n")

    top20 = agg.head(20)
    for _, row in top20.iterrows():
        print(f"{row['rank']:2d}. DTE={row['dte']:2d} p{row['percentile']:2d} w{row['spread_width']:3.0f} | "
              f"ROI={row['avg_roi']:5.2f}% | Opps={row['total_opportunities']:4.0f} | "
              f"Avg Loss=${row['avg_max_loss']:,.0f}")

    print("\n" + "="*80 + "\n")
    print(f"Full results saved to: {output_path}")

    return 0


if __name__ == '__main__':
    sys.exit(asyncio.run(main()))
