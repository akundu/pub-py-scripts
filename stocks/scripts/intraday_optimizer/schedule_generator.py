"""
Trading schedule generator.

Creates optimal trading schedules based on grid search results,
with support for various optimization strategies.
"""

from typing import Dict, List, Tuple
import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Add project root to path
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from common.logging_utils import get_logger

logger = get_logger("schedule_generator", level="INFO")


class ScheduleGenerator:
    """Generate optimal trading schedules."""

    def __init__(self, block_size_minutes: int = 15):
        """
        Initialize generator.

        Args:
            block_size_minutes: Size of schedule blocks (minutes)
        """
        self.block_size_minutes = block_size_minutes

    def consolidate_to_blocks(
        self,
        df: pd.DataFrame,
        window_minutes: int
    ) -> pd.DataFrame:
        """
        Consolidate 10-minute windows into larger blocks (e.g., 15-minute).

        Args:
            df: Aggregated results DataFrame
            window_minutes: Original window size

        Returns:
            DataFrame with block-level aggregation
        """
        # For now, if window_minutes equals block_size_minutes, no consolidation needed
        if window_minutes == self.block_size_minutes:
            df['time_block'] = df['start_time'] + '-' + df['end_time']
            return df

        # Otherwise, would need to map multiple windows to blocks
        # Simplified: just use the windows as-is
        df['time_block'] = df['start_time'] + '-' + df['end_time']
        return df

    def generate_schedule(
        self,
        agg_results: pd.DataFrame,
        strategy: str = 'maximize_roi',
        top_n: int = None
    ) -> pd.DataFrame:
        """
        Generate trading schedule.

        Args:
            agg_results: Aggregated grid search results
            strategy: Selection strategy:
                - 'maximize_roi': Select highest ROI configs
                - 'maximize_opportunities': Select most frequent configs
                - 'balanced': Balance ROI and opportunities
            top_n: Limit to top N configurations (None = unlimited)

        Returns:
            Schedule DataFrame
        """
        logger.info(f"Generating schedule with strategy: {strategy}")

        # Add time_block column
        agg_results = self.consolidate_to_blocks(
            agg_results.copy(),
            window_minutes=10  # Assuming 10-minute windows
        )

        # Calculate composite score based on strategy
        if strategy == 'maximize_roi':
            agg_results['score'] = agg_results['avg_roi']
        elif strategy == 'maximize_opportunities':
            agg_results['score'] = agg_results['total_opportunities']
        elif strategy == 'balanced':
            # Normalize both metrics and combine
            roi_norm = (agg_results['avg_roi'] - agg_results['avg_roi'].min()) / \
                       (agg_results['avg_roi'].max() - agg_results['avg_roi'].min() + 1e-6)
            opp_norm = (agg_results['total_opportunities'] - agg_results['total_opportunities'].min()) / \
                       (agg_results['total_opportunities'].max() - agg_results['total_opportunities'].min() + 1e-6)
            agg_results['score'] = (roi_norm + opp_norm) / 2
        else:
            raise ValueError(f"Unknown strategy: {strategy}")

        # Sort by score
        schedule = agg_results.sort_values('score', ascending=False).copy()

        # Limit to top N if specified
        if top_n:
            schedule = schedule.head(top_n)

        # Add rank
        schedule['rank'] = range(1, len(schedule) + 1)

        # Select columns for output
        output_cols = [
            'rank',
            'time_block',
            'start_time',
            'end_time',
            'dte',
            'percentile',
            'spread_width',
            'avg_roi',
            'std_roi',
            'avg_entry_credit',
            'avg_max_loss',
            'total_opportunities',
            'avg_daily_opportunities',
            'pct_days_meeting_constraints',
            'roi_sharpe',
            'score'
        ]

        schedule = schedule[output_cols].copy()

        # Add action column
        schedule['action'] = schedule.apply(
            lambda row: 'TRADE' if row['avg_roi'] >= 5.0 else 'SKIP',
            axis=1
        )

        logger.info(f"Schedule generated with {len(schedule)} configurations")

        return schedule

    def generate_summary_statistics(
        self,
        schedule: pd.DataFrame,
        training_period: Tuple[str, str],
        validation_period: Tuple[str, str] = None
    ) -> Dict[str, any]:
        """
        Generate summary statistics for the schedule.

        Args:
            schedule: Generated schedule
            training_period: (start_date, end_date) for training
            validation_period: Optional (start_date, end_date) for validation

        Returns:
            Dictionary with summary stats
        """
        tradeable = schedule[schedule['action'] == 'TRADE']

        summary = {
            'training_period': f"{training_period[0]} to {training_period[1]}",
            'validation_period': f"{validation_period[0]} to {validation_period[1]}" if validation_period else "N/A",
            'total_configs_analyzed': len(schedule),
            'tradeable_configs': len(tradeable),
            'avg_roi': tradeable['avg_roi'].mean() if len(tradeable) > 0 else 0,
            'median_roi': tradeable['avg_roi'].median() if len(tradeable) > 0 else 0,
            'max_roi': tradeable['avg_roi'].max() if len(tradeable) > 0 else 0,
            'total_daily_opportunities': tradeable['avg_daily_opportunities'].sum() if len(tradeable) > 0 else 0,
            'avg_entry_credit': tradeable['avg_entry_credit'].mean() if len(tradeable) > 0 else 0,
            'avg_max_loss': tradeable['avg_max_loss'].mean() if len(tradeable) > 0 else 0,
            'capital_required_per_day': tradeable['avg_max_loss'].sum() if len(tradeable) > 0 else 0,
        }

        # Calculate expected daily profit (sum of avg opportunities * avg ROI * avg max loss)
        if len(tradeable) > 0:
            daily_profits = []
            for _, row in tradeable.iterrows():
                # Expected profit per opportunity
                profit_per_opp = row['avg_max_loss'] * (row['avg_roi'] / 100)
                # Expected daily profit for this config
                daily_profit = profit_per_opp * row['avg_daily_opportunities']
                daily_profits.append(daily_profit)

            summary['expected_daily_profit'] = sum(daily_profits)
            summary['expected_monthly_profit'] = summary['expected_daily_profit'] * 21
            summary['expected_annual_profit'] = summary['expected_daily_profit'] * 252
        else:
            summary['expected_daily_profit'] = 0
            summary['expected_monthly_profit'] = 0
            summary['expected_annual_profit'] = 0

        return summary

    def print_summary(
        self,
        summary: Dict[str, any],
        schedule: pd.DataFrame
    ):
        """Print formatted summary."""
        print("\n" + "="*80)
        print("INTRADAY OPTIMIZATION SUMMARY")
        print("="*80)
        print(f"\nTRAINING PERIOD: {summary['training_period']}")
        if summary['validation_period'] != "N/A":
            print(f"VALIDATION PERIOD: {summary['validation_period']}")

        print(f"\nCONFIGURATIONS:")
        print(f"  Total analyzed: {summary['total_configs_analyzed']:,}")
        print(f"  Tradeable: {summary['tradeable_configs']:,}")

        print(f"\nPERFORMANCE METRICS:")
        print(f"  Average ROI: {summary['avg_roi']:.2f}%")
        print(f"  Median ROI: {summary['median_roi']:.2f}%")
        print(f"  Max ROI: {summary['max_roi']:.2f}%")

        print(f"\nOPPORTUNITIES:")
        print(f"  Daily opportunities: {summary['total_daily_opportunities']:.1f}")
        print(f"  Avg entry credit: ${summary['avg_entry_credit']:.2f}")
        print(f"  Avg max loss: ${summary['avg_max_loss']:.2f}")

        print(f"\nCAPITAL REQUIREMENTS:")
        print(f"  Per day (unlimited): ${summary['capital_required_per_day']:,.0f}")

        print(f"\nEXPECTED PROFITS (assuming all opportunities taken):")
        print(f"  Daily: ${summary['expected_daily_profit']:,.2f}")
        print(f"  Monthly: ${summary['expected_monthly_profit']:,.2f}")
        print(f"  Annual: ${summary['expected_annual_profit']:,.2f}")

        print(f"\nTOP 10 CONFIGURATIONS:")
        print("-"*80)
        top10 = schedule.head(10)
        for _, row in top10.iterrows():
            print(f"{row['rank']:2d}. {row['time_block']:11s} | "
                  f"DTE={row['dte']:2d} p{row['percentile']:2d} w{row['spread_width']:3.0f} | "
                  f"ROI={row['avg_roi']:5.2f}% | "
                  f"Opps={row['total_opportunities']:4.0f} | "
                  f"{row['action']}")

        print("="*80 + "\n")
