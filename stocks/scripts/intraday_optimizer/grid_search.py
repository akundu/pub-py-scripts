"""
Intraday grid search engine.

Runs comprehensive grid search across time windows, DTEs,
percentiles, and spread widths to find optimal trading configurations.
"""

import asyncio
import itertools
from datetime import datetime, timedelta, time
from typing import Dict, List, Tuple, Any
from pathlib import Path
import sys
import pandas as pd
import numpy as np
from multiprocessing import Pool
import os

# Add project root to path
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from .time_window_analyzer import TimeWindowAnalyzer, TimeWindowConfig
from common.logging_utils import get_logger

logger = get_logger("intraday_grid_search", level="INFO")


class IntradayGridSearch:
    """Grid search across time windows and parameters."""

    def __init__(
        self,
        time_window_minutes: int = 10,
        dte_range: Tuple[int, int] = (0, 15),
        percentile_range: Tuple[int, int] = (95, 100),
        spread_widths: List[float] = None,
        db_config: str = None
    ):
        """
        Initialize grid search.

        Args:
            time_window_minutes: Size of time windows (minutes)
            dte_range: (min_dte, max_dte)
            percentile_range: (min_percentile, max_percentile)
            spread_widths: List of spread widths to test
            db_config: QuestDB connection string
        """
        self.time_window_minutes = time_window_minutes
        self.dte_range = dte_range
        self.percentile_range = percentile_range
        self.spread_widths = spread_widths or [10, 20, 30, 50, 100]
        self.db_config = db_config or self._get_db_config()
        self.analyzer = TimeWindowAnalyzer(self.db_config)

    def _get_db_config(self) -> str:
        """Get database config from environment."""
        return (
            os.getenv('QUEST_DB_STRING') or
            os.getenv('QUESTDB_CONNECTION_STRING') or
            os.getenv('QUESTDB_URL')
        )

    def generate_time_windows(self) -> List[Tuple[str, str]]:
        """
        Generate time windows for the trading day.

        Returns:
            List of (start_time, end_time) tuples
        """
        windows = []
        market_open = time(9, 30)
        market_close = time(16, 0)

        current = datetime.combine(datetime.today(), market_open)
        end_dt = datetime.combine(datetime.today(), market_close)

        while current < end_dt:
            next_time = current + timedelta(minutes=self.time_window_minutes)
            if next_time <= end_dt:
                windows.append((
                    current.strftime('%H:%M'),
                    next_time.strftime('%H:%M')
                ))
            current = next_time

        return windows

    def generate_grid_configs(
        self,
        max_loss_constraint: float = 30000,
        min_roi_pct: float = 5.0
    ) -> List[Dict[str, Any]]:
        """
        Generate all grid configurations to test.

        Returns:
            List of config dictionaries
        """
        time_windows = self.generate_time_windows()
        dtes = range(self.dte_range[0], self.dte_range[1] + 1)
        percentiles = range(self.percentile_range[0], self.percentile_range[1] + 1)

        configs = []
        for (start_time, end_time), dte, percentile, width in itertools.product(
            time_windows, dtes, percentiles, self.spread_widths
        ):
            configs.append({
                'start_time': start_time,
                'end_time': end_time,
                'dte': dte,
                'percentile': percentile,
                'spread_width': width,
                'max_loss_constraint': max_loss_constraint,
                'min_roi_pct': min_roi_pct
            })

        logger.info(f"Generated {len(configs)} configurations to test")
        return configs

    async def analyze_single_day_config(
        self,
        ticker: str,
        date: datetime,
        config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Analyze a single configuration on a single day.

        Returns:
            Result dictionary with metrics
        """
        result = await self.analyzer.analyze_window(
            ticker=ticker,
            date=date,
            start_time=config['start_time'],
            end_time=config['end_time'],
            dte=config['dte'],
            percentile=config['percentile'],
            spread_width=config['spread_width'],
            max_loss_constraint=config['max_loss_constraint'],
            min_roi_pct=config['min_roi_pct']
        )

        # Add config info to result
        result.update({
            'date': date.strftime('%Y-%m-%d'),
            'start_time': config['start_time'],
            'end_time': config['end_time'],
            'dte': config['dte'],
            'percentile': config['percentile'],
            'spread_width': config['spread_width']
        })

        return result

    async def load_all_dte_data_for_date(
        self,
        ticker: str,
        date: datetime
    ) -> Dict[int, pd.DataFrame]:
        """
        Load all DTE data for a single date (optimized loading).

        Returns:
            Dict mapping DTE -> DataFrame
        """
        from credit_spread_utils.data_loader import load_multi_dte_data

        date_str = date.strftime('%Y-%m-%d')
        dte_data = {}

        # Load 0DTE (from options_csv_output)
        try:
            df0 = load_multi_dte_data(
                csv_dir='options_csv_output',
                ticker=ticker,
                start_date=date_str,
                end_date=date_str,
                dte_buckets=(0,),
                dte_tolerance=1,
                cache_dir=None,
                no_cache=False,  # Use cache
                logger=logger
            )
            if not df0.empty:
                dte_data[0] = df0
        except Exception as e:
            logger.debug(f"No 0DTE data for {date_str}: {e}")

        # Load 1-15 DTE (from options_csv_output_full)
        for dte in range(1, 16):
            try:
                df = load_multi_dte_data(
                    csv_dir='options_csv_output_full',
                    ticker=ticker,
                    start_date=date_str,
                    end_date=date_str,
                    dte_buckets=(dte,),
                    dte_tolerance=1,
                    cache_dir=None,
                    no_cache=False,  # Use cache
                    logger=logger
                )
                if not df.empty:
                    # Use only first timestamp for non-0DTE
                    timestamps = sorted(df['timestamp'].unique())
                    df = df[df['timestamp'] == timestamps[0]].copy()
                    dte_data[dte] = df
            except Exception as e:
                logger.debug(f"No DTE{dte} data for {date_str}: {e}")

        return dte_data

    async def run_grid_search(
        self,
        ticker: str,
        start_date: str,
        end_date: str,
        max_loss_constraint: float = 30000,
        min_roi_pct: float = 5.0,
        progress_callback: callable = None
    ) -> pd.DataFrame:
        """
        Run comprehensive grid search (OPTIMIZED version).

        Loads data once per day and tests all configs against cached data.

        Args:
            ticker: Ticker symbol
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            max_loss_constraint: Max loss per trade
            min_roi_pct: Minimum ROI percentage
            progress_callback: Optional callback for progress updates

        Returns:
            DataFrame with all results
        """
        logger.info(f"Starting OPTIMIZED grid search: {ticker} from {start_date} to {end_date}")

        # Generate all configurations
        configs = self.generate_grid_configs(max_loss_constraint, min_roi_pct)

        # Generate date range
        start_dt = datetime.strptime(start_date, '%Y-%m-%d')
        end_dt = datetime.strptime(end_date, '%Y-%m-%d')

        trading_days = []
        current = start_dt
        while current <= end_dt:
            # Skip weekends
            if current.weekday() < 5:
                trading_days.append(current)
            current += timedelta(days=1)

        logger.info(f"Testing {len(configs)} configs across {len(trading_days)} trading days")
        logger.info(f"Total analyses: {len(configs) * len(trading_days):,}")
        logger.info(f"Optimization: Loading data {len(trading_days)} times (once per day)")

        # Run analyses
        all_results = []
        total_days = len(trading_days)

        for day_idx, date in enumerate(trading_days):
            logger.info(f"Day {day_idx+1}/{total_days}: {date.strftime('%Y-%m-%d')} - Loading all DTE data...")

            # Load ALL DTE data for this date ONCE
            dte_data_cache = await self.load_all_dte_data_for_date(ticker, date)

            if not dte_data_cache:
                logger.warning(f"No data available for {date.strftime('%Y-%m-%d')}, skipping")
                continue

            logger.info(f"Loaded {len(dte_data_cache)} DTEs, testing {len(configs)} configs...")

            # Test all configs against this cached data
            for config_idx, config in enumerate(configs):
                dte = config['dte']

                # Get cached data for this DTE
                if dte not in dte_data_cache:
                    # No data for this DTE on this day
                    result = {
                        'date': date.strftime('%Y-%m-%d'),
                        'start_time': config['start_time'],
                        'end_time': config['end_time'],
                        'dte': dte,
                        'percentile': config['percentile'],
                        'spread_width': config['spread_width'],
                        'num_spreads': 0,
                        'avg_entry_credit': 0,
                        'avg_max_loss': 0,
                        'avg_roi': 0,
                        'total_credit_potential': 0,
                        'meets_constraints': False
                    }
                    all_results.append(result)
                    continue

                # Use cached data
                try:
                    result = await self.analyzer.analyze_window(
                        ticker=ticker,
                        date=date,
                        start_time=config['start_time'],
                        end_time=config['end_time'],
                        dte=dte,
                        percentile=config['percentile'],
                        spread_width=config['spread_width'],
                        max_loss_constraint=config['max_loss_constraint'],
                        min_roi_pct=config['min_roi_pct'],
                        cached_df=dte_data_cache[dte]
                    )

                    result.update({
                        'date': date.strftime('%Y-%m-%d'),
                        'start_time': config['start_time'],
                        'end_time': config['end_time'],
                        'dte': dte,
                        'percentile': config['percentile'],
                        'spread_width': config['spread_width']
                    })
                    all_results.append(result)

                except Exception as e:
                    logger.warning(f"Error testing config: {e}")
                    result = {
                        'date': date.strftime('%Y-%m-%d'),
                        'start_time': config['start_time'],
                        'end_time': config['end_time'],
                        'dte': dte,
                        'percentile': config['percentile'],
                        'spread_width': config['spread_width'],
                        'num_spreads': 0,
                        'avg_entry_credit': 0,
                        'avg_max_loss': 0,
                        'avg_roi': 0,
                        'total_credit_potential': 0,
                        'meets_constraints': False
                    }
                    all_results.append(result)

                # Progress update every 1000 configs
                if (config_idx + 1) % 1000 == 0:
                    logger.info(f"  Tested {config_idx+1}/{len(configs)} configs for this day")

        # Convert to DataFrame
        df = pd.DataFrame(all_results)

        logger.info(f"Grid search complete: {len(df)} results")
        return df

    def aggregate_results(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Aggregate results by configuration across all days.

        Args:
            df: Raw results DataFrame

        Returns:
            Aggregated DataFrame with summary statistics per config
        """
        # Group by configuration
        group_cols = ['start_time', 'end_time', 'dte', 'percentile', 'spread_width']

        agg_df = df.groupby(group_cols).agg({
            'num_spreads': ['sum', 'mean', 'std'],
            'avg_entry_credit': 'mean',
            'avg_max_loss': 'mean',
            'avg_roi': ['mean', 'std'],
            'total_credit_potential': 'sum',
            'meets_constraints': ['sum', 'mean']
        }).reset_index()

        # Flatten column names
        agg_df.columns = [
            'start_time', 'end_time', 'dte', 'percentile', 'spread_width',
            'total_opportunities', 'avg_daily_opportunities', 'std_daily_opportunities',
            'avg_entry_credit', 'avg_max_loss',
            'avg_roi', 'std_roi',
            'total_credit_potential',
            'days_meeting_constraints', 'pct_days_meeting_constraints'
        ]

        # Calculate Sharpe-like metric: avg_roi / std_roi
        agg_df['roi_sharpe'] = agg_df['avg_roi'] / (agg_df['std_roi'] + 1e-6)

        # Sort by total opportunities and ROI
        agg_df = agg_df.sort_values(
            ['total_opportunities', 'avg_roi'],
            ascending=[False, False]
        ).reset_index(drop=True)

        return agg_df
