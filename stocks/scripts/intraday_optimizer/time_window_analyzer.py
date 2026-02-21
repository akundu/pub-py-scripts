"""
Time window analyzer for intraday optimization.

Analyzes spreads available within specific time windows,
calculating metrics for strategy evaluation.
"""

import asyncio
from dataclasses import dataclass
from datetime import datetime, time, timedelta
from typing import Dict, List, Optional, Any
from pathlib import Path
import sys
import pandas as pd
import numpy as np

# Add project root to path
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from credit_spread_utils.percentile_integration import PercentileSpreadIntegrator
from credit_spread_utils.data_loader import load_multi_dte_data
from credit_spread_utils.price_utils import get_previous_close_price
from common.questdb_db import StockQuestDB
from common.logging_utils import get_logger

logger = get_logger("time_window_analyzer", level="INFO")


@dataclass
class TimeWindowConfig:
    """Configuration for a time window analysis."""
    start_time: str  # "09:30"
    end_time: str    # "09:40"
    dte: int
    percentile: int
    spread_width: float
    max_loss_constraint: float = 30000
    min_roi_pct: float = 5.0


class TimeWindowAnalyzer:
    """Analyze spreads for specific time windows."""

    def __init__(self, db_config: str):
        """
        Initialize analyzer.

        Args:
            db_config: QuestDB connection string
        """
        self.db_config = db_config
        self.integrator = PercentileSpreadIntegrator(db_config=db_config)
        self.db = StockQuestDB(db_config, logger=logger)

    async def analyze_window(
        self,
        ticker: str,
        date: datetime,
        start_time: str,
        end_time: str,
        dte: int,
        percentile: int,
        spread_width: float,
        max_loss_constraint: float = 30000,
        min_roi_pct: float = 5.0,
        cached_df: pd.DataFrame = None
    ) -> Dict[str, Any]:
        """
        Analyze spreads available in this time window.

        Args:
            ticker: Ticker symbol
            date: Trading date
            start_time: Window start (HH:MM)
            end_time: Window end (HH:MM)
            dte: Days to expiration
            percentile: Percentile for strikes
            spread_width: Spread width in points
            max_loss_constraint: Max loss per trade
            min_roi_pct: Minimum ROI percentage

        Returns:
            Dictionary with metrics:
            - num_spreads: Count of available spreads
            - avg_entry_credit: Average credit collected
            - avg_max_loss: Average max loss per spread
            - avg_roi: Average ROI %
            - meets_constraints: Boolean
            - trades: List of trade details
        """
        date_str = date.strftime('%Y-%m-%d')

        # Use cached data if provided, otherwise load
        if cached_df is not None:
            df = cached_df.copy()
        else:
            # Load data for this day
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
                    no_cache=True,
                    logger=logger
                )
            except Exception as e:
                logger.warning(f"Error loading data for {date_str}: {e}")
                return self._empty_result()

        if df.empty:
            return self._empty_result()

        try:
            # For 0DTE, filter to specific time window
            # For other DTEs, use first timestamp of day (already filtered in grid_search)
            if dte == 0:
                # Filter to time window
                df = self._filter_to_time_window(df, start_time, end_time)
                if df.empty:
                    return self._empty_result()

            # Get previous close
            first_ts = df['timestamp'].min()
            prev_close_result = await get_previous_close_price(self.db, ticker, first_ts, logger)
            if prev_close_result is None:
                return self._empty_result()

            prev_close, _ = prev_close_result

            # Find spreads
            trades = await self.integrator.analyze_single_day(
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
                return self._empty_result()

            # Calculate metrics
            entry_credits = [t.get('entry_credit', 0) for t in trades]
            max_losses = [t.get('max_loss', 0) for t in trades]

            avg_credit = np.mean(entry_credits) if entry_credits else 0
            avg_max_loss = np.mean(max_losses) if max_losses else 0

            # Calculate ROI: (profit target / max_loss) * 100
            # Profit target = 50% of credit
            rois = []
            for credit, max_loss in zip(entry_credits, max_losses):
                if max_loss > 0:
                    profit_target = credit * 0.5
                    roi = (profit_target / max_loss) * 100
                    rois.append(roi)

            avg_roi = np.mean(rois) if rois else 0

            # Check constraints
            meets_constraints = (
                avg_max_loss <= max_loss_constraint and
                avg_roi >= min_roi_pct
            )

            return {
                'num_spreads': len(trades),
                'avg_entry_credit': avg_credit,
                'avg_max_loss': avg_max_loss,
                'avg_roi': avg_roi,
                'total_credit_potential': sum(entry_credits),
                'meets_constraints': meets_constraints,
                'trades': trades
            }

        except Exception as e:
            logger.warning(f"Error analyzing window {start_time}-{end_time} on {date_str}: {e}")
            return self._empty_result()

    def _filter_to_time_window(
        self,
        df: pd.DataFrame,
        start_time: str,
        end_time: str
    ) -> pd.DataFrame:
        """Filter DataFrame to specific time window."""
        # Parse times
        start_hour, start_min = map(int, start_time.split(':'))
        end_hour, end_min = map(int, end_time.split(':'))

        start_t = time(start_hour, start_min)
        end_t = time(end_hour, end_min)

        # Filter timestamps
        df['time'] = df['timestamp'].dt.time
        mask = (df['time'] >= start_t) & (df['time'] < end_t)
        result = df[mask].copy()
        result = result.drop(columns=['time'])

        return result

    def _empty_result(self) -> Dict[str, Any]:
        """Return empty result structure."""
        return {
            'num_spreads': 0,
            'avg_entry_credit': 0,
            'avg_max_loss': 0,
            'avg_roi': 0,
            'total_credit_potential': 0,
            'meets_constraints': False,
            'trades': []
        }
