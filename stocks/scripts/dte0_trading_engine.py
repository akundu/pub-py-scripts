#!/usr/bin/env python3
"""
Professional 0DTE Trading Engine: Dynamic Probability Ladder & True Backtest Engine
"""

import sys
import os
import argparse
import asyncio
import logging
import warnings
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, time as dt_time
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum
import pytz

# Standard environment path injection for local modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from common.stock_db import get_stock_db
from common.logging_utils import get_logger
from common.display_utils import normalize_timezone_string

warnings.filterwarnings("ignore")

# ============================================================================
# Enums and Data Classes
# ============================================================================


class MarketRegime(Enum):
    LOW_VOL = "Low Volatility"
    NORMAL = "Normal"
    ELEVATED = "Elevated"
    HIGH_VOL = "High Volatility"


class TradeBias(Enum):
    TREND = "trend"
    REVERSION = "reversion"


class TradeDirection(Enum):
    PUT_CREDIT_SPREAD = "Put Credit Spread"
    CALL_CREDIT_SPREAD = "Call Credit Spread"


@dataclass
class ProbabilityLadder:
    anchor_price: float
    anchor_time: datetime
    p95_upper: float
    p99_upper: float
    mae_upper: float
    p95_lower: float
    p99_lower: float
    mae_lower: float
    range_decay_factor: float  # New: Percentage of daily range remaining
    # Empirical worst-case from open to close (fixed, independent of hour)
    worst_case_pct_up: float  # Worst ever % move up from open to close
    worst_case_pct_down: float  # Worst ever % move down from open to close
    empirical_p95_upper: float  # P95 of all historical open-to-close moves
    empirical_p99_upper: float  # P99 of all historical open-to-close moves
    empirical_p95_lower: float  # P5 of all historical open-to-close moves
    empirical_p99_lower: float  # P1 of all historical open-to-close moves
    # Hour-specific empirical ranges (from current hour to close)
    hour_empirical_p95_upper: float  # P95 from current hour to close (all data)
    hour_empirical_p99_upper: float  # P99 from current hour to close (all data)
    hour_empirical_p95_lower: float  # P5 from current hour to close (all data)
    hour_empirical_p99_lower: float  # P1 from current hour to close (all data)
    hour_twin_empirical_p95_upper: float  # P95 from current hour to close (twins only)
    hour_twin_empirical_p99_upper: float  # P99 from current hour to close (twins only)
    hour_twin_empirical_p95_lower: float  # P5 from current hour to close (twins only)
    hour_twin_empirical_p99_lower: float  # P1 from current hour to close (twins only)


# ============================================================================
# Technical Indicators
# ============================================================================


def calculate_vwap(df: pd.DataFrame) -> pd.Series:
    """Calculate Volume Weighted Average Price."""
    v = df["volume"]
    tp = (df["high"] + df["low"] + df["close"]) / 3
    return (tp * v).cumsum() / v.cumsum()


def get_success_confidence(
    num_twins: int, vix: float, price: float, vwap: float
) -> int:
    """Calculates a 1-10 confidence score based on sample size and VWAP alignment."""
    base_score = min(num_twins / 20, 7)
    vol_penalty = max(0, (vix - 15) / 10)
    vwap_penalty = 1.5 if abs(price - vwap) / price > 0.01 else 0
    return int(np.clip(base_score - vol_penalty - vwap_penalty + 3, 1, 10))


def classify_regime2(vix_value: float) -> MarketRegime:
    if vix_value < 15:
        return MarketRegime.LOW_VOL
    elif vix_value < 25:
        return MarketRegime.NORMAL
    elif vix_value < 35:
        return MarketRegime.ELEVATED
    else:
        return MarketRegime.HIGH_VOL


def classify_regime(vix_value: float) -> MarketRegime:
    """Recalibrated buckets for higher sensitivity to VIX1D spikes."""
    if vix_value < 12:
        return MarketRegime.LOW_VOL
    elif vix_value < 16:  # Lowered from 25
        return MarketRegime.NORMAL
    elif vix_value < 25:  # Lowered from 35
        return MarketRegime.ELEVATED
    else:
        return MarketRegime.HIGH_VOL


# ============================================================================
# Core Engine
# ============================================================================


def convert_hour_to_utc(
    date_str: str, hour_str: str, input_tz: str = "EST"
) -> datetime:
    """
    Convert a date and hour string from a timezone to UTC.

    Args:
        date_str: Date string in YYYY-MM-DD format
        hour_str: Time string in HH:MM format or just hour number (e.g., "9:30", "10:00", "9", "14")
        input_tz: Input timezone abbreviation (e.g., EST, EDT, PST, PDT) or full name (e.g., America/New_York)
                  Supports: EST/EDT, PST/PDT, CST/CDT, MST/MDT, UTC, GMT, and others
                  Default: EST (Eastern Standard Time)

    Returns:
        UTC datetime (timezone-naive, as expected by the database)
    """
    # Normalize timezone abbreviation to proper timezone name
    # This handles EST/EDT, PST/PDT, etc. and converts them to proper location-based timezones
    normalized_tz = normalize_timezone_string(input_tz)

    # Normalize hour string to HH:MM format
    # Handle cases like "9" -> "09:00", "9:30" -> "09:30", "10" -> "10:00"
    if ":" not in hour_str:
        # Just a number, assume it's the hour and add ":00" for minutes
        try:
            hour_num = int(hour_str)
            hour_str = f"{hour_num:02d}:00"
        except ValueError:
            raise ValueError(
                f"Invalid hour format: {hour_str}. Expected HH:MM or just hour number."
            )
    else:
        # Already has colon, ensure it's in HH:MM format
        parts = hour_str.split(":")
        if len(parts) != 2:
            raise ValueError(f"Invalid hour format: {hour_str}. Expected HH:MM format.")
        try:
            hour_num = int(parts[0])
            minute_num = int(parts[1])
            hour_str = f"{hour_num:02d}:{minute_num:02d}"
        except ValueError:
            raise ValueError(
                f"Invalid hour format: {hour_str}. Hours and minutes must be numbers."
            )

    # Parse the date and time
    dt_str = f"{date_str} {hour_str}"
    naive_dt = pd.to_datetime(dt_str)

    # Localize to input timezone (handles EST/EDT automatically via normalized timezone)
    tz = pytz.timezone(normalized_tz)
    localized_dt = tz.localize(naive_dt)

    # Convert to UTC and make timezone-naive (database expects naive UTC)
    utc_dt = localized_dt.astimezone(pytz.UTC)
    return utc_dt.replace(tzinfo=None)


class ZeroDTEEngine:
    def __init__(self, ticker: str, db_config: str, cache_dir: str = "./"):
        self.ticker = ticker
        self.db_config = db_config
        self.logger = get_logger("0dte_engine")
        self.cache_file = os.path.join(cache_dir, f"{ticker}_model_cache.parquet")
        self.db = None

    async def initialize(self):
        self.db = get_stock_db("questdb", db_config=self.db_config)
        await self.db._init_db()

    async def close(self):
        if self.db:
            await self.db.close()

    def _calculate_ladder(
        self,
        twins: pd.DataFrame,
        anchor_price: float,
        anchor_time: datetime,
        all_historical_data: Optional[pd.DataFrame] = None,
        day_open_price: Optional[float] = None,
    ) -> ProbabilityLadder:
        """
        Calculates ranges using Time-Anchored Relative Moves.
        Replaces get_intraday_move with get_relative_move logic.
        """

        def get_relative_move(group):
            # Find the twin's price at the exact same hour as today's anchor
            target_hour_data = group[group.index.time == anchor_time.time()]
            twin_anchor = (
                target_hour_data.iloc[0]["close"]
                if not target_hour_data.empty
                else group.iloc[0]["close"]
            )
            twin_close = group.iloc[-1]["close"]
            return (twin_close / twin_anchor - 1) * 100

        twin_moves = twins.groupby(twins.index.date).apply(get_relative_move)

        # Calculate Time-Decay (Range Contraction Factor)
        # Measures % of full-day twin volatility remaining from this anchor hour
        def get_decay(group):
            day_range = group["high"].max() - group["low"].min()
            if day_range <= 0 or pd.isna(day_range):
                return 1.0

            # Get the anchor time for this specific day (same date as group, same time as anchor_time)
            group_date = group.index[0].date()

            # Convert anchor_time to pd.Timestamp if it's not already
            if isinstance(anchor_time, datetime):
                anchor_ts = pd.Timestamp(anchor_time)
            else:
                anchor_ts = pd.Timestamp(anchor_time)

            # Create anchor timestamp for this specific day
            anchor_datetime_for_day = pd.Timestamp.combine(
                pd.Timestamp(group_date), anchor_ts.time()
            )

            # Ensure it's timezone-naive (matching group.index)
            if anchor_datetime_for_day.tz is not None:
                anchor_datetime_for_day = anchor_datetime_for_day.tz_localize(None)

            # Filter data from anchor time onwards
            remaining_data = group[group.index >= anchor_datetime_for_day]

            if remaining_data.empty:
                # If no data after anchor time, return 0 (all volatility has passed)
                return 0.0

            # Calculate remaining range: max high from anchor to EOD minus min low from anchor to EOD
            # This represents the volatility range that can still occur
            remaining_high = remaining_data["high"].max()
            remaining_low = remaining_data["low"].min()
            remaining_range = remaining_high - remaining_low

            # Ensure non-negative and valid
            if pd.isna(remaining_range) or remaining_range < 0:
                return 0.0

            return remaining_range / day_range

        decay_factors = twins.groupby(twins.index.date).apply(get_decay)
        # Filter out NaN values and calculate mean
        valid_decay_factors = decay_factors[~pd.isna(decay_factors)]
        decay_factor = (
            valid_decay_factors.mean() if len(valid_decay_factors) > 0 else 1.0
        )

        # Final safety check - ensure decay_factor is a valid number
        if pd.isna(decay_factor) or not np.isfinite(decay_factor):
            decay_factor = 1.0

        # Use day's open price for open->close calculations, fallback to anchor_price
        base_price_for_open_close = (
            day_open_price if day_open_price is not None else anchor_price
        )

        # Calculate empirical worst-case values from all historical data (independent of twins)
        # These are based on open->close and should be fixed regardless of hour
        worst_case_pct_up = 0.0
        worst_case_pct_down = 0.0
        empirical_p95_upper = base_price_for_open_close
        empirical_p99_upper = base_price_for_open_close
        empirical_p95_lower = base_price_for_open_close
        empirical_p99_lower = base_price_for_open_close

        # Hour-specific empirical ranges (from current hour to close)
        hour_empirical_p95_upper = anchor_price
        hour_empirical_p99_upper = anchor_price
        hour_empirical_p95_lower = anchor_price
        hour_empirical_p99_lower = anchor_price
        hour_twin_empirical_p95_upper = anchor_price
        hour_twin_empirical_p99_upper = anchor_price
        hour_twin_empirical_p95_lower = anchor_price
        hour_twin_empirical_p99_lower = anchor_price

        if all_historical_data is not None and not all_historical_data.empty:
            # Calculate open-to-close % moves for all historical days
            def get_open_to_close_move(group):
                if len(group) == 0:
                    return None
                day_open = group.iloc[0]["open"]
                day_close = group.iloc[-1]["close"]
                return (day_close / day_open - 1) * 100

            all_daily_moves = (
                all_historical_data.groupby(all_historical_data.index.date)
                .apply(get_open_to_close_move)
                .dropna()
            )

            if len(all_daily_moves) > 0:
                # Worst case moves (open->close)
                worst_case_pct_up = all_daily_moves.max()
                worst_case_pct_down = all_daily_moves.min()

                # Percentiles (open->close) - use base_price_for_open_close
                empirical_p95_upper = base_price_for_open_close * (
                    1 + np.percentile(all_daily_moves, 95) / 100
                )
                empirical_p99_upper = base_price_for_open_close * (
                    1 + np.percentile(all_daily_moves, 99) / 100
                )
                empirical_p95_lower = base_price_for_open_close * (
                    1 + np.percentile(all_daily_moves, 5) / 100
                )
                empirical_p99_lower = base_price_for_open_close * (
                    1 + np.percentile(all_daily_moves, 1) / 100
                )

            # Calculate hour-specific moves (from anchor hour to close) for all data
            def get_hour_to_close_move(group):
                if len(group) == 0:
                    return None
                # Find price at anchor hour for this day
                group_date = group.index[0].date()
                anchor_ts = (
                    pd.Timestamp(anchor_time)
                    if isinstance(anchor_time, datetime)
                    else pd.Timestamp(anchor_time)
                )
                anchor_datetime_for_day = pd.Timestamp.combine(
                    pd.Timestamp(group_date), anchor_ts.time()
                )
                if anchor_datetime_for_day.tz is not None:
                    anchor_datetime_for_day = anchor_datetime_for_day.tz_localize(None)

                # Find data at or after anchor time
                hour_data = group[group.index >= anchor_datetime_for_day]
                if hour_data.empty:
                    # Fallback to first bar if anchor hour not found
                    hour_price = group.iloc[0]["close"]
                else:
                    hour_price = hour_data.iloc[0]["close"]

                day_close = group.iloc[-1]["close"]
                return (day_close / hour_price - 1) * 100

            all_hour_to_close_moves = (
                all_historical_data.groupby(all_historical_data.index.date)
                .apply(get_hour_to_close_move)
                .dropna()
            )

            if len(all_hour_to_close_moves) > 0:
                hour_empirical_p95_upper = anchor_price * (
                    1 + np.percentile(all_hour_to_close_moves, 95) / 100
                )
                hour_empirical_p99_upper = anchor_price * (
                    1 + np.percentile(all_hour_to_close_moves, 99) / 100
                )
                hour_empirical_p95_lower = anchor_price * (
                    1 + np.percentile(all_hour_to_close_moves, 5) / 100
                )
                hour_empirical_p99_lower = anchor_price * (
                    1 + np.percentile(all_hour_to_close_moves, 1) / 100
                )

            # Calculate hour-specific moves for twins only
            if not twins.empty:
                twin_hour_to_close_moves = (
                    twins.groupby(twins.index.date)
                    .apply(get_hour_to_close_move)
                    .dropna()
                )
                if len(twin_hour_to_close_moves) > 0:
                    hour_twin_empirical_p95_upper = anchor_price * (
                        1 + np.percentile(twin_hour_to_close_moves, 95) / 100
                    )
                    hour_twin_empirical_p99_upper = anchor_price * (
                        1 + np.percentile(twin_hour_to_close_moves, 99) / 100
                    )
                    hour_twin_empirical_p95_lower = anchor_price * (
                        1 + np.percentile(twin_hour_to_close_moves, 5) / 100
                    )
                    hour_twin_empirical_p99_lower = anchor_price * (
                        1 + np.percentile(twin_hour_to_close_moves, 1) / 100
                    )

        return ProbabilityLadder(
            anchor_price=anchor_price,
            anchor_time=anchor_time,
            p95_upper=anchor_price * (1 + np.percentile(twin_moves, 95) / 100),
            p99_upper=anchor_price * (1 + np.percentile(twin_moves, 99) / 100),
            mae_upper=anchor_price * (1 + twin_moves.max() / 100),
            p95_lower=anchor_price * (1 + np.percentile(twin_moves, 5) / 100),
            p99_lower=anchor_price * (1 + np.percentile(twin_moves, 1) / 100),
            mae_lower=anchor_price * (1 + twin_moves.min() / 100),
            range_decay_factor=decay_factor,
            worst_case_pct_up=worst_case_pct_up,
            worst_case_pct_down=worst_case_pct_down,
            empirical_p95_upper=empirical_p95_upper,
            empirical_p99_upper=empirical_p99_upper,
            empirical_p95_lower=empirical_p95_lower,
            empirical_p99_lower=empirical_p99_lower,
            hour_empirical_p95_upper=hour_empirical_p95_upper,
            hour_empirical_p99_upper=hour_empirical_p99_upper,
            hour_empirical_p95_lower=hour_empirical_p95_lower,
            hour_empirical_p99_lower=hour_empirical_p99_lower,
            hour_twin_empirical_p95_upper=hour_twin_empirical_p95_upper,
            hour_twin_empirical_p99_upper=hour_twin_empirical_p99_upper,
            hour_twin_empirical_p95_lower=hour_twin_empirical_p95_lower,
            hour_twin_empirical_p99_lower=hour_twin_empirical_p99_lower,
        )

    # async def run_analysis2(self, mode, bias, date_str, anchor_hour, test_split=0.2):
    #     # Fetch data and twins (Logic from previous unified script)
    #     # ... [Data Fetching and Twin Matching Code] ...

    #     # PROBABILISTIC VS EMPIRICAL DASHBOARD
    #     print(
    #         f"\n{'='*80}\n ANALYSIS - {self.ticker} - {date_str} @ {anchor_hour}\n{'='*80}"
    #     )
    #     decay_display = (
    #         f"{ladder.range_decay_factor:.1%}"
    #         if not pd.isna(ladder.range_decay_factor)
    #         and np.isfinite(ladder.range_decay_factor)
    #         else "N/A"
    #     )
    #     print(f" TIME DECAY: {decay_display} of daily volatility typically remains.")

    #     print(f"\n--- PROBABILISTIC EXPECTATION (Based on {len(twins)} Twins) ---")
    #     print(f" P95 Confidence: [${ladder.p95_lower:.2f}, ${ladder.p95_upper:.2f}]")
    #     print(f" P99 Confidence: [${ladder.p99_lower:.2f}, ${ladder.p99_upper:.2f}]")

    #     print(f"\n--- EMPIRICAL DATA (Worst Case 'Hard Walls') ---")
    #     print(f" MAE Downside:  ${ladder.mae_lower:.2f}")
    #     print(f" MAE Upside:    ${ladder.mae_upper:.2f}")

    async def fetch_and_prepare(self, start_date, end_date):
        """Fetches data and enriches with VWAP, Gaps, and RSI-7."""
        # 1. Fetch Ticker and VIX1D data
        h_df = await self.db.get_stock_data(
            ticker=self.ticker,
            start_date=start_date,
            end_date=end_date,
            interval="hourly",
        )
        v_df = await self.db.get_stock_data(
            ticker="VIX1D", start_date=start_date, end_date=end_date, interval="hourly"
        )

        if h_df.empty:
            return pd.DataFrame()

        # 2. Add Time and VWAP Enrichment
        h_df["date"] = h_df.index.date
        h_df["vwap"] = h_df.groupby("date", group_keys=False).apply(calculate_vwap)

        # 3. Calculate Gap % from Previous Day Close
        daily_close = h_df.groupby("date")["close"].last().shift(1)
        h_df["prev_close"] = h_df["date"].map(daily_close)
        h_df["gap_pct"] = (h_df["open"] - h_df["prev_close"]) / h_df["prev_close"] * 100

        # 4. Calculate RSI-7 (Momentum)
        delta = h_df["close"].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=7).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=7).mean()
        rs = gain / loss
        h_df["rsi_7"] = 100 - (100 / (1 + rs))

        # 5. Join VIX1D and fill missing values
        h_df = h_df.join(
            v_df[["close"]].rename(columns={"close": "vix1d"}), how="left"
        ).ffill()

        return h_df

    async def load_model(
        self, use_cache: bool = True, required_date: Optional[str] = None
    ) -> pd.DataFrame:
        if use_cache and os.path.exists(self.cache_file):
            self.logger.info(
                f"Loading cached historical model from {self.cache_file}..."
            )
            df = pd.read_parquet(self.cache_file)

            # Ensure date column exists
            if "date" not in df.columns:
                df["date"] = df.index.date

            # Check if we need to fetch fresh data
            need_fresh_fetch = False
            target_date = None
            
            if required_date:
                # Specific date was requested
                target_date = required_date
            else:
                # For live mode, check if today's date is in cache
                target_date = datetime.now().strftime("%Y-%m-%d")
            
            if target_date:
                req_date_obj = pd.to_datetime(target_date).date()
                available_dates = set(df["date"].unique())

                if req_date_obj not in available_dates:
                    need_fresh_fetch = True
                    self.logger.info(
                        f"Date {target_date} not in cache, fetching fresh data..."
                    )

            if need_fresh_fetch:
                # Fetch fresh data that includes the required date
                end_date = max(target_date, datetime.now().strftime("%Y-%m-%d"))
                start_date = (datetime.now() - timedelta(days=730)).strftime(
                    "%Y-%m-%d"
                )
                df = await self.fetch_and_prepare(start_date, end_date)
                if not df.empty:
                    df.to_parquet(self.cache_file)
                    self.logger.info(
                        f"Model updated and saved to {self.cache_file}"
                    )

            return df

        self.logger.info("Rebuilding model from scratch (fetching 2 years)...")
        end_date = datetime.now().strftime("%Y-%m-%d")
        if required_date:
            end_date = max(required_date, end_date)
        start_date = (datetime.now() - timedelta(days=730)).strftime("%Y-%m-%d")
        df = await self.fetch_and_prepare(start_date, end_date)
        if not df.empty:
            df.to_parquet(self.cache_file)
        return df

    async def run_analysis(
        self,
        mode: str,
        bias: str,
        date_str: str,
        anchor_hour: str,
        test_split: float = 0.2,
        input_timezone: str = "EST",
    ):
        """
        Executes the 0DTE analysis/backtest using Time-Anchored Relative Moves,
        RSI-7 momentum matching, and recalibrated volatility regimes.
        """
        # 1. Load the model (use cache for live/historical, fresh for backtest)
        required_date = date_str if mode == "historical" else None
        df = await self.load_model(
            use_cache=(mode != "backtest"), required_date=required_date
        )
        if df.empty:
            self.logger.warning("No data loaded from model")
            print("No data available.")
            return

        # 1a. For live mode, fetch latest price from realtime system (prioritizes realtime > hourly > daily)
        if mode == "live":
            self.logger.info(f"Live mode: Attempting to fetch latest data for {self.ticker}")
            try:
                # First, explicitly fetch realtime data from the realtime_data table for today
                today_str = datetime.now().strftime("%Y-%m-%d")
                today_start = f"{today_str}T00:00:00"
                today_end = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
                
                realtime_data_today = None
                if hasattr(self.db, "get_realtime_data"):
                    try:
                        self.logger.info(
                            f"Fetching realtime data from realtime_data table for {self.ticker} "
                            f"from {today_start} to {today_end}"
                        )
                        realtime_data_today = await self.db.get_realtime_data(
                            self.ticker,
                            start_datetime=today_start,
                            end_datetime=today_end,
                            data_type="quote"
                        )
                        if not realtime_data_today.empty:
                            self.logger.info(
                                f"Found {len(realtime_data_today)} realtime data points for today"
                            )
                        else:
                            self.logger.info("No realtime data found in realtime_data table for today")
                    except Exception as e:
                        self.logger.warning(
                            f"Failed to fetch realtime data directly from table: {e}"
                        )
                
                # Check if database has price_service (QuestDB has it)
                if hasattr(self.db, "price_service"):
                    latest_price_data = (
                        await self.db.price_service.get_latest_price_with_data(
                            self.ticker, use_market_time=True
                        )
                    )
                    if latest_price_data:
                        source = latest_price_data.get("source", "unknown")
                        price = latest_price_data.get("price")
                        timestamp = latest_price_data.get("timestamp")
                        realtime_df = latest_price_data.get("realtime_df")
                        hourly_df = latest_price_data.get("hourly_df")
                        daily_df = latest_price_data.get("daily_df")

                        self.logger.info(
                            f"Live mode: Got latest price from {source}: ${price:.2f} at {timestamp}"
                        )
                        self.logger.debug(
                            f"Data availability - realtime_df: {realtime_df is not None and not realtime_df.empty if realtime_df is not None else False}, "
                            f"hourly_df: {hourly_df is not None and not hourly_df.empty if hourly_df is not None else False}, "
                            f"daily_df: {daily_df is not None and not daily_df.empty if daily_df is not None else False}"
                        )
                        if hourly_df is not None and not hourly_df.empty:
                            self.logger.debug(
                                f"hourly_df details - shape: {hourly_df.shape}, "
                                f"index type: {type(hourly_df.index)}, "
                                f"columns: {list(hourly_df.columns)}, "
                                f"index values (last 3): {list(hourly_df.index[-3:]) if len(hourly_df) >= 3 else list(hourly_df.index)}"
                            )

                        # If we have realtime data, add it to the dataframe
                        # Prefer explicitly fetched realtime_data_today if available, otherwise use price_service data
                        use_realtime_df = None
                        if realtime_data_today is not None and not realtime_data_today.empty:
                            # Use explicitly fetched realtime data from the table
                            use_realtime_df = realtime_data_today
                            self.logger.info(
                                f"Using realtime data directly from realtime_data table "
                                f"({len(use_realtime_df)} points)"
                            )
                        elif realtime_df is not None and not realtime_df.empty:
                            # Fallback to price_service realtime data
                            use_realtime_df = realtime_df
                            self.logger.info("Using realtime data from price_service")
                        
                        if use_realtime_df is not None and not use_realtime_df.empty:
                            # Convert realtime data to match hourly format
                            # Get the latest point (first row if sorted DESC, last row if sorted ASC)
                            if isinstance(use_realtime_df.index, pd.DatetimeIndex):
                                # Sort by index to get latest
                                use_realtime_df_sorted = use_realtime_df.sort_index(ascending=False)
                                latest_realtime = use_realtime_df_sorted.iloc[0]
                                rt_timestamp = use_realtime_df_sorted.index[0]
                            else:
                                latest_realtime = use_realtime_df.iloc[0]
                                rt_timestamp = latest_realtime.get("timestamp") or latest_realtime.name

                            # Check if timestamp is valid
                            if rt_timestamp is None:
                                self.logger.warning(
                                    "Realtime data timestamp is None, skipping"
                                )
                            else:
                                # Get price from realtime data
                                rt_price = float(latest_realtime.get("price", price))
                                
                                # Create a row compatible with hourly data format
                                new_row = pd.DataFrame(
                                    {
                                        "open": rt_price,
                                        "high": rt_price,
                                        "low": rt_price,
                                        "close": rt_price,
                                        "volume": latest_realtime.get("volume", 0),
                                        "date": pd.to_datetime(rt_timestamp).date(),
                                    },
                                    index=[pd.to_datetime(rt_timestamp)],
                                )

                                # Add VIX1D if available (use last known value)
                                if "vix1d" in df.columns and len(df) > 0:
                                    new_row["vix1d"] = df["vix1d"].iloc[-1]

                                # Append or update the dataframe
                                df = pd.concat([df, new_row]).sort_index()
                                self.logger.info(
                                    f"Added realtime data point at {rt_timestamp} with price ${rt_price:.2f}"
                                )

                        # If we have hourly data that's more recent than what's in df, update it
                        elif hourly_df is not None and not hourly_df.empty:
                            # Always try to use timestamp from price_service first if available
                            hr_timestamp = timestamp if timestamp is not None else None
                            
                            # If timestamp is None, try to extract from DataFrame
                            if hr_timestamp is None:
                                latest_hourly = hourly_df.iloc[-1]
                                
                                # First try: DatetimeIndex (most common case)
                                if isinstance(hourly_df.index, pd.DatetimeIndex) and len(hourly_df.index) > 0:
                                    hr_timestamp = hourly_df.index[-1]
                                # Second try: Try to convert index to datetime
                                elif len(hourly_df.index) > 0:
                                    try:
                                        hr_timestamp = pd.to_datetime(hourly_df.index[-1])
                                    except:
                                        pass
                                
                                # Third try: Check for datetime/timestamp columns
                                if hr_timestamp is None and hasattr(hourly_df, 'columns'):
                                    if 'datetime' in hourly_df.columns:
                                        try:
                                            hr_timestamp = pd.to_datetime(hourly_df['datetime'].iloc[-1])
                                        except:
                                            pass
                                    elif 'timestamp' in hourly_df.columns:
                                        try:
                                            hr_timestamp = pd.to_datetime(hourly_df['timestamp'].iloc[-1])
                                        except:
                                            pass
                                
                                # Fourth try: Check latest_hourly row attributes
                                if hr_timestamp is None:
                                    if hasattr(latest_hourly, 'name') and latest_hourly.name is not None:
                                        try:
                                            hr_timestamp = pd.to_datetime(latest_hourly.name)
                                        except:
                                            pass
                                    # Try accessing as Series
                                    if hr_timestamp is None and isinstance(latest_hourly, pd.Series):
                                        for col in ['datetime', 'timestamp', 'date']:
                                            if col in latest_hourly.index:
                                                try:
                                                    hr_timestamp = pd.to_datetime(latest_hourly[col])
                                                    break
                                                except:
                                                    pass

                            # Check if timestamp is valid before comparing
                            if hr_timestamp is None:
                                if price is not None:
                                    # Use current time as fallback - this handles cached data that lost its index
                                    hr_timestamp = datetime.now()
                                    print(f"[FALLBACK] Using current time as timestamp for hourly data: {hr_timestamp}")
                                    self.logger.info(
                                        f"Using current time as fallback timestamp for hourly data: {hr_timestamp}"
                                    )
                                else:
                                    print(f"[SKIP] Cannot process hourly data: hr_timestamp is None and price is None")
                                    self.logger.warning(
                                        f"Cannot process hourly data: no timestamp and no price available"
                                    )
                            
                            if hr_timestamp is not None:
                                # Check if this hourly bar is more recent than what we have
                                should_add = len(df) == 0
                                if not should_add and len(df) > 0:
                                    try:
                                        hr_ts = pd.to_datetime(hr_timestamp)
                                        should_add = hr_ts > df.index[-1]
                                    except (TypeError, ValueError) as e:
                                        self.logger.warning(
                                            f"Error comparing hourly timestamp: {e}, hr_timestamp={hr_timestamp}"
                                        )
                                        should_add = False

                                if should_add:
                                    latest_hourly = hourly_df.iloc[-1]
                                    # Safely extract values from Series/DataFrame row
                                    def safe_get(data, key, default):
                                        if hasattr(data, 'get'):
                                            return data.get(key, default)
                                        elif hasattr(data, '__getitem__'):
                                            try:
                                                return data[key]
                                            except (KeyError, IndexError):
                                                return default
                                        return default
                                    
                                    new_row = pd.DataFrame(
                                        {
                                            "open": safe_get(latest_hourly, "open", price),
                                            "high": safe_get(latest_hourly, "high", price),
                                            "low": safe_get(latest_hourly, "low", price),
                                            "close": price,
                                            "volume": safe_get(latest_hourly, "volume", 0),
                                            "date": pd.to_datetime(hr_timestamp).date(),
                                        },
                                        index=[pd.to_datetime(hr_timestamp)],
                                    )

                                    # Add VIX1D if available
                                    if "vix1d" in df.columns and len(df) > 0:
                                        new_row["vix1d"] = df["vix1d"].iloc[-1]

                                    df = pd.concat([df, new_row]).sort_index()
                                    self.logger.info(
                                        f"Added/updated hourly data at {hr_timestamp} with price ${price:.2f}"
                                    )

                        # If we have daily data that's more recent, update it
                        elif daily_df is not None and not daily_df.empty:
                            latest_daily = daily_df.iloc[-1]
                            dy_timestamp = (
                                daily_df.index[-1]
                                if isinstance(daily_df.index, pd.DatetimeIndex)
                                else latest_daily.get("date")
                            )

                            # Check if timestamp is valid before comparing
                            if dy_timestamp is None:
                                self.logger.warning(
                                    "Daily data timestamp is None, skipping"
                                )
                            else:
                                # Check if this daily bar is more recent than what we have
                                should_add = len(df) == 0
                                if not should_add and len(df) > 0:
                                    try:
                                        dy_ts = pd.to_datetime(dy_timestamp)
                                        should_add = dy_ts.date() > df["date"].iloc[-1]
                                    except (TypeError, ValueError) as e:
                                        self.logger.warning(
                                            f"Error comparing daily timestamp: {e}"
                                        )
                                        should_add = False

                                if should_add:
                                    # For daily data, we need to create hourly-like entries
                                    # Use the daily OHLC for the day
                                    day_start = pd.to_datetime(dy_timestamp).replace(
                                        hour=9, minute=30
                                    )
                                    new_row = pd.DataFrame(
                                        {
                                            "open": latest_daily.get("open", price),
                                            "high": latest_daily.get("high", price),
                                            "low": latest_daily.get("low", price),
                                            "close": price,
                                            "volume": latest_daily.get("volume", 0),
                                            "date": pd.to_datetime(dy_timestamp).date(),
                                        },
                                        index=[day_start],
                                    )

                                    # Add VIX1D if available
                                    if "vix1d" in df.columns and len(df) > 0:
                                        new_row["vix1d"] = df["vix1d"].iloc[-1]

                                    df = pd.concat([df, new_row]).sort_index()
                                    self.logger.info(
                                        f"Added/updated daily data at {dy_timestamp}"
                                    )

                        # Recalculate derived fields for the new data
                        if len(df) > 0:
                            # Recalculate date column
                            df["date"] = df.index.date
                            
                            # Log what dates we have after adding live data
                            unique_dates = sorted(df["date"].unique())
                            self.logger.info(
                                f"After adding live data: {len(unique_dates)} unique dates, "
                                f"latest: {unique_dates[-1] if unique_dates else 'N/A'}"
                            )
                        else:
                            self.logger.warning(
                                "No data in dataframe after live mode processing. "
                                "This means no realtime, hourly, or daily data was successfully added."
                            )

                            # Recalculate VWAP if we have volume data
                            if "volume" in df.columns:
                                df["vwap"] = df.groupby("date", group_keys=False).apply(
                                    calculate_vwap
                                )

                            # Recalculate gap_pct if we have prev_close
                            if "prev_close" in df.columns:
                                daily_close = (
                                    df.groupby("date")["close"].last().shift(1)
                                )
                                df["prev_close"] = df["date"].map(daily_close)
                                df["gap_pct"] = (
                                    (df["open"] - df["prev_close"])
                                    / df["prev_close"]
                                    * 100
                                )

                            # Recalculate RSI-7 if we have close prices
                            if "close" in df.columns:
                                delta = df["close"].diff()
                                gain = (
                                    (delta.where(delta > 0, 0)).rolling(window=7).mean()
                                )
                                loss = (
                                    (-delta.where(delta < 0, 0))
                                    .rolling(window=7)
                                    .mean()
                                )
                                rs = gain / loss
                                df["rsi_7"] = 100 - (100 / (1 + rs))
                    else:
                        self.logger.warning(
                            f"Live mode: No latest price data returned from price_service for {self.ticker}"
                        )
                        # If price_service didn't return data but we have realtime_data_today, use it
                        if realtime_data_today is not None and not realtime_data_today.empty:
                            self.logger.info(
                                "Using realtime data from realtime_data table as fallback"
                            )
                            # Get the latest point
                            if isinstance(realtime_data_today.index, pd.DatetimeIndex):
                                realtime_data_today_sorted = realtime_data_today.sort_index(ascending=False)
                                latest_realtime = realtime_data_today_sorted.iloc[0]
                                rt_timestamp = realtime_data_today_sorted.index[0]
                            else:
                                latest_realtime = realtime_data_today.iloc[0]
                                rt_timestamp = latest_realtime.get("timestamp") or latest_realtime.name
                            
                            if rt_timestamp is not None:
                                rt_price = float(latest_realtime.get("price", 0))
                                new_row = pd.DataFrame(
                                    {
                                        "open": rt_price,
                                        "high": rt_price,
                                        "low": rt_price,
                                        "close": rt_price,
                                        "volume": latest_realtime.get("volume", 0),
                                        "date": pd.to_datetime(rt_timestamp).date(),
                                    },
                                    index=[pd.to_datetime(rt_timestamp)],
                                )
                                
                                # Add VIX1D if available
                                if "vix1d" in df.columns and len(df) > 0:
                                    new_row["vix1d"] = df["vix1d"].iloc[-1]
                                
                                df = pd.concat([df, new_row]).sort_index()
                                self.logger.info(
                                    f"Added realtime data point (fallback) at {rt_timestamp} with price ${rt_price:.2f}"
                                )
                else:
                    self.logger.warning(
                        f"Live mode: Database does not have price_service attribute"
                    )
                    # If no price_service but we have realtime_data_today, use it
                    if realtime_data_today is not None and not realtime_data_today.empty:
                        self.logger.info(
                            "Using realtime data from realtime_data table (no price_service available)"
                        )
                        if isinstance(realtime_data_today.index, pd.DatetimeIndex):
                            realtime_data_today_sorted = realtime_data_today.sort_index(ascending=False)
                            latest_realtime = realtime_data_today_sorted.iloc[0]
                            rt_timestamp = realtime_data_today_sorted.index[0]
                        else:
                            latest_realtime = realtime_data_today.iloc[0]
                            rt_timestamp = latest_realtime.get("timestamp") or latest_realtime.name
                        
                        if rt_timestamp is not None:
                            rt_price = float(latest_realtime.get("price", 0))
                            new_row = pd.DataFrame(
                                {
                                    "open": rt_price,
                                    "high": rt_price,
                                    "low": rt_price,
                                    "close": rt_price,
                                    "volume": latest_realtime.get("volume", 0),
                                    "date": pd.to_datetime(rt_timestamp).date(),
                                },
                                index=[pd.to_datetime(rt_timestamp)],
                            )
                            
                            # Add VIX1D if available
                            if "vix1d" in df.columns and len(df) > 0:
                                new_row["vix1d"] = df["vix1d"].iloc[-1]
                            
                            df = pd.concat([df, new_row]).sort_index()
                            self.logger.info(
                                f"Added realtime data point at {rt_timestamp} with price ${rt_price:.2f}"
                            )

            except Exception as e:
                self.logger.warning(
                    f"Failed to fetch latest price data for live mode: {e}"
                )
                import traceback
                self.logger.debug(traceback.format_exc())
                # Continue with cached data if realtime fetch fails
            
            # Summary of live mode data addition
            if mode == "live":
                unique_dates_after = sorted(df["date"].unique()) if len(df) > 0 and "date" in df.columns else []
                target_date_obj = pd.to_datetime(date_str).date() if date_str else datetime.now().date()
                has_today_data = target_date_obj in unique_dates_after if unique_dates_after else False
                
                if has_today_data:
                    self.logger.info(
                        f"✓ Successfully added data for {target_date_obj}. "
                        f"Total dates in dataset: {len(unique_dates_after)}"
                    )
                else:
                    self.logger.warning(
                        f"✗ No data added for target date {target_date_obj}. "
                        f"Available dates: {unique_dates_after[-5:] if len(unique_dates_after) > 0 else 'None'}. "
                        f"This may cause the analysis to fail."
                    )

        # 2. Determine the dates to process
        all_dates = sorted(df["date"].unique())
        if len(all_dates) == 0:
            print("No dates found in dataset.")
            return

        if mode != "backtest":
            self.logger.info(
                f"Loaded data with {len(df)} rows covering {len(all_dates)} unique dates"
            )
            self.logger.info(
                f"Date range: {all_dates[0]} to {all_dates[-1]}"
            )

        split_idx = int(len(all_dates) * (1 - test_split))
        if mode == "backtest":
            test_dates = all_dates[split_idx:]
        else:
            # For live/historical mode, use provided date or today's date
            target_date_str = (
                date_str if date_str else datetime.now().strftime("%Y-%m-%d")
            )
            test_dates = [pd.to_datetime(target_date_str).date()]
            if mode != "backtest":
                self.logger.info(
                    f"Processing date: {test_dates[0]} (requested: {target_date_str})"
                )

        results = []
        skipped_no_anchor = 0
        skipped_no_twins = 0

        # 3. Process each date
        for target_date in test_dates:
            # Ensure date column exists and is the right type
            if "date" not in df.columns:
                df["date"] = df.index.date
            
            # Try filtering by date - handle type mismatches
            day_data = df[df["date"] == target_date]
            if day_data.empty:
                # Try alternative filtering methods
                day_data = df[df.index.date == target_date]
            
            if day_data.empty:
                skipped_no_anchor += 1
                if mode != "backtest":
                    print(
                        f"\n{'='*80}\n ANALYSIS - {self.ticker} - {target_date}\n{'='*80}"
                    )
                    print(f"ERROR: No data found for date {target_date} (type: {type(target_date)})")
                    print(f"Available dates in dataset: {len(all_dates)} unique dates")
                    if len(all_dates) > 0:
                        print(f"Date range: {all_dates[0]} to {all_dates[-1]}")
                        # Show closest dates
                        try:
                            target_dt = pd.to_datetime(target_date)
                            closest_dates = sorted(all_dates, key=lambda x: abs((pd.to_datetime(x) - target_dt).days))[:5]
                            print(f"Closest available dates: {closest_dates}")
                        except:
                            print(f"Sample available dates: {all_dates[:5]}")
                        # Debug: show date column types
                        if len(df) > 0:
                            print(f"Date column type: {type(df['date'].iloc[0])}")
                            print(f"Target date type: {type(target_date)}")
                            print(f"Sample dates in column: {df['date'].iloc[:5].tolist()}")
                continue

            # Convert anchor hour to UTC to match database timestamps
            anchor_time = convert_hour_to_utc(
                str(target_date), anchor_hour, input_tz=input_timezone
            )

            # Find the anchor point at or before the specified hour
            anchor_data_for_date = df.loc[
                (df.index <= anchor_time) & (df["date"] == target_date)
            ]

            if anchor_data_for_date.empty:
                # For historical predictions, use market open and calculate range for close
                if mode == "historical":
                    # Use the first bar (market open) of the day
                    anchor_data_for_date = day_data.iloc[:1]
                    actual_anchor_time = anchor_data_for_date.index[0]
                    # Use open price as anchor for close prediction
                    anchor_price_for_calc = day_data.iloc[0]["open"]
                    # For current state, use the first bar of the day (open)
                    current = day_data.iloc[0]
                else:
                    # For other modes, use first available bar
                    anchor_data_for_date = day_data.iloc[:1]
                    actual_anchor_time = anchor_data_for_date.index[0]
                    anchor_price_for_calc = None  # Will use current["close"] below
                    # Get the current market state at the anchor hour
                    historical_data = df.loc[df.index <= actual_anchor_time]
                    if historical_data.empty:
                        skipped_no_anchor += 1
                        if mode != "backtest":
                            print(
                                f"\n{'='*80}\n ANALYSIS - {self.ticker} - {target_date}\n{'='*80}"
                            )
                            print(f"ERROR: No historical data available at or before {actual_anchor_time}")
                            print(f"Available data range: {df.index.min()} to {df.index.max()}")
                        continue
                    current = historical_data.iloc[-1]
            else:
                actual_anchor_time = anchor_data_for_date.index[-1]
                anchor_price_for_calc = None  # Will use current["close"] below
                # Get the current market state at the anchor hour
                historical_data = df.loc[df.index <= actual_anchor_time]
                if historical_data.empty:
                    skipped_no_anchor += 1
                    if mode != "backtest":
                        print(
                            f"\n{'='*80}\n ANALYSIS - {self.ticker} - {target_date}\n{'='*80}"
                        )
                        print(f"ERROR: No historical data available at or before {actual_anchor_time}")
                        print(f"Available data range: {df.index.min()} to {df.index.max()}")
                    continue
                current = historical_data.iloc[-1]

            # Use open price if anchor hour wasn't present in historical mode, otherwise use current close
            if anchor_price_for_calc is not None:
                anchor_price = anchor_price_for_calc
            else:
                anchor_price = current["close"]

            # 4. TWIN MATCHING: Filter by Gap % AND RSI-7 momentum
            twins = df[
                (
                    df["gap_pct"].between(
                        current["gap_pct"] - 0.25, current["gap_pct"] + 0.25
                    )
                )
                & (df["rsi_7"].between(current["rsi_7"] - 5.0, current["rsi_7"] + 5.0))
                & (df.index.date < target_date)
            ]

            if twins.empty:
                skipped_no_twins += 1
                if mode != "backtest":
                    print(
                        f"ERROR: No matching twins found for Gap {current['gap_pct']:.2f}% and RSI {current['rsi_7']:.1f}"
                    )
                continue

            # 5. Calculate Ladder using Time-Anchored Relative Moves
            # Get day's open price for open->close empirical calculations
            day_open_price = day_data.iloc[0]["open"]
            # Pass full dataframe for empirical calculations
            ladder = self._calculate_ladder(
                twins,
                anchor_price,
                actual_anchor_time,
                all_historical_data=df,
                day_open_price=day_open_price,
            )

            # 6. Simulate Trade Outcome
            actual_close = day_data.iloc[-1]["close"]
            win = (actual_close <= ladder.p99_upper) and (
                actual_close >= ladder.p99_lower
            )

            # Recalibrated Regime Classification
            current_regime = classify_regime(current["vix1d"])

            results.append(
                {
                    "date": target_date,
                    "regime": current_regime,
                    "win": win,
                }
            )

            # 7. Print Dashboard for Single Analysis
            if mode != "backtest":
                print(
                    f"\n{'='*80}\n ANALYSIS - {self.ticker} - {target_date} @ {anchor_hour} {input_timezone}\n{'='*80}"
                )
                print(f" OPEN PRICE:     ${day_open_price:.2f}")
                print(f" CURRENT TIME:    {actual_anchor_time}")
                print(f" CURRENT PRICE:  ${anchor_price:.2f}")
                print(f" FINAL CLOSE:    ${actual_close:.2f}")
                print(f" REGIME: {current_regime.value}")

                decay_display = (
                    f"{ladder.range_decay_factor:.1%}"
                    if not pd.isna(ladder.range_decay_factor)
                    and np.isfinite(ladder.range_decay_factor)
                    else "N/A"
                )
                print(
                    f" TIME DECAY: {decay_display} of daily volatility typically remains."
                )

                print(
                    f"\n--- PROBABILISTIC EXPECTATION (Based on {len(twins)} Twins) ---"
                )
                print(
                    f" P95 Confidence: [${ladder.p95_lower:.2f}, ${ladder.p95_upper:.2f}]"
                )
                print(
                    f" P99 Confidence: [${ladder.p99_lower:.2f}, ${ladder.p99_upper:.2f}]"
                )

                print(f"\n--- EMPIRICAL DATA (Worst Case 'Hard Walls' from Twins) ---")
                print(f" MAE Downside:  ${ladder.mae_lower:.2f}")
                print(f" MAE Upside:    ${ladder.mae_upper:.2f}")

                print(f"\n--- EMPIRICAL DATA (Open->Close, All Historical Data) ---")
                print(
                    f" Worst Case Down: {ladder.worst_case_pct_down:.2f}% (${day_open_price * (1 + ladder.worst_case_pct_down / 100):.2f})"
                )
                print(
                    f" Worst Case Up:   {ladder.worst_case_pct_up:.2f}% (${day_open_price * (1 + ladder.worst_case_pct_up / 100):.2f})"
                )
                print(
                    f" P95 Range: [${ladder.empirical_p95_lower:.2f}, ${ladder.empirical_p95_upper:.2f}]"
                )
                print(
                    f" P99 Range: [${ladder.empirical_p99_lower:.2f}, ${ladder.empirical_p99_upper:.2f}]"
                )

                print(
                    f"\n--- EMPIRICAL DATA (From Current Hour->Close, All Historical Data) ---"
                )
                print(
                    f" P95 Range: [${ladder.hour_empirical_p95_lower:.2f}, ${ladder.hour_empirical_p95_upper:.2f}]"
                )
                print(
                    f" P99 Range: [${ladder.hour_empirical_p99_lower:.2f}, ${ladder.hour_empirical_p99_upper:.2f}]"
                )

                print(
                    f"\n--- EMPIRICAL DATA (From Current Hour->Close, Twins Only) ---"
                )
                print(
                    f" P95 Range: [${ladder.hour_twin_empirical_p95_lower:.2f}, ${ladder.hour_twin_empirical_p95_upper:.2f}]"
                )
                print(
                    f" P99 Range: [${ladder.hour_twin_empirical_p99_lower:.2f}, ${ladder.hour_twin_empirical_p99_upper:.2f}]"
                )

        # 8. Final Backtest Reporting
        if mode == "backtest":
            self.print_performance_report(results)

    # async def run_analysis2(
    #     self,
    #     mode,
    #     bias,
    #     date_str,
    #     anchor_hour,
    #     test_split=0.2,
    #     input_timezone: str = "EST",
    # ):
    #     # For historical mode, pass the required date to ensure it's in the cache
    #     required_date = date_str if mode == "historical" else None
    #     df = await self.load_model(
    #         use_cache=(mode != "backtest"), required_date=required_date
    #     )
    #     if df.empty:
    #         self.logger.warning("No data loaded from model")
    #         print("No data available.")
    #         return

    #     # Ensure date column exists and is the right type
    #     if "date" not in df.columns:
    #         df["date"] = df.index.date
    #     else:
    #         # Ensure date column contains date objects, not strings or other types
    #         if not isinstance(
    #             df["date"].iloc[0] if len(df) > 0 else None,
    #             type(pd.Timestamp.now().date()),
    #         ):
    #             # Try to convert if it's not already date objects
    #             try:
    #                 df["date"] = pd.to_datetime(df["date"]).dt.date
    #             except:
    #                 df["date"] = df.index.date

    #     all_dates = sorted(df["date"].unique())
    #     self.logger.info(
    #         f"Total unique dates in dataset: {len(all_dates)} "
    #         f"(from {all_dates[0] if all_dates else 'N/A'} to {all_dates[-1] if all_dates else 'N/A'})"
    #     )
    #     if len(all_dates) == 0:
    #         self.logger.error("No dates found in dataset")
    #         print("No dates found in dataset.")
    #         return

    #     split_idx = int(len(all_dates) * (1 - test_split))
    #     test_dates = (
    #         all_dates[split_idx:]
    #         if mode == "backtest"
    #         else [pd.to_datetime(date_str).date()]
    #     )
    #     self.logger.info(
    #         f"Test dates to process: {len(test_dates)} "
    #         f"(from {test_dates[0] if test_dates else 'N/A'} to {test_dates[-1] if test_dates else 'N/A'})"
    #     )

    #     results = []
    #     skipped_no_anchor = 0
    #     skipped_no_twins = 0
    #     for target_date in test_dates:
    #         # First check if we have any data for this date
    #         # Try multiple ways to filter by date in case of type mismatches
    #         day_data = df[df["date"] == target_date]
    #         if day_data.empty:
    #             # Try filtering by index date as fallback
    #             day_data = df[df.index.date == target_date]
    #         if day_data.empty:
    #             skipped_no_anchor += 1
    #             if mode != "backtest":
    #                 print(
    #                     f"\n{'='*80}\n ANALYSIS - {self.ticker} - {target_date}\n{'='*80}"
    #                 )
    #                 print(
    #                     f"ERROR: No data found for date {target_date} (type: {type(target_date)})"
    #                 )
    #                 print(f"Available dates in dataset: {len(all_dates)} unique dates")
    #                 # Show some sample dates
    #                 if len(all_dates) > 0:
    #                     print(
    #                         f"Sample dates: {all_dates[0]} (type: {type(all_dates[0])}) to {all_dates[-1]}"
    #                     )
    #                     # Check if target_date is in the list
    #                     if target_date in all_dates:
    #                         print(
    #                             f"WARNING: {target_date} is in all_dates but filtering failed!"
    #                         )
    #                     # Show a few dates around target_date
    #                     try:
    #                         idx = (
    #                             all_dates.index(target_date)
    #                             if target_date in all_dates
    #                             else -1
    #                         )
    #                         if idx >= 0:
    #                             start = max(0, idx - 2)
    #                             end = min(len(all_dates), idx + 3)
    #                             print(f"Dates around target: {all_dates[start:end]}")
    #                     except:
    #                         pass
    #             continue

    #         # Convert anchor hour from input timezone (default EST) to UTC (data is stored in UTC)
    #         # Supports timezone abbreviations (EST, EDT, PST, PDT, etc.) or full names
    #         anchor_time = convert_hour_to_utc(
    #             str(target_date), anchor_hour, input_tz=input_timezone
    #         )

    #         # Find the anchor point: data at or before anchor_time for this date
    #         # If anchor hour doesn't exist, use first available data point of the day
    #         anchor_data_for_date = df.loc[
    #             (df.index <= anchor_time) & (df["date"] == target_date)
    #         ]
    #         if anchor_data_for_date.empty:
    #             # No data at anchor hour, use first data point of the day
    #             anchor_data_for_date = day_data.iloc[:1]
    #             actual_anchor_time = anchor_data_for_date.index[0]
    #             self.logger.info(
    #                 f"No data at {anchor_hour} for {target_date}, using first available: {actual_anchor_time}"
    #             )
    #         else:
    #             actual_anchor_time = anchor_data_for_date.index[-1]

    #         # Get the current state at the anchor point (from full historical dataset)
    #         # This is what we use to find matching twins
    #         historical_data = df.loc[df.index <= actual_anchor_time]
    #         if historical_data.empty:
    #             skipped_no_anchor += 1
    #             if mode != "backtest":
    #                 print(
    #                     f"\n{'='*80}\n ANALYSIS - {self.ticker} - {target_date}\n{'='*80}"
    #                 )
    #                 print(
    #                     f"ERROR: No historical data available at or before {actual_anchor_time}"
    #                 )
    #                 print(f"Available data range: {df.index.min()} to {df.index.max()}")
    #             continue

    #         current = historical_data.iloc[-1]

    #         # Find twins: matching gap_pct and before target_date
    #         twins = df[
    #             (
    #                 df["gap_pct"].between(
    #                     current["gap_pct"] - 0.25, current["gap_pct"] + 0.25
    #                 )
    #             )
    #             & (df.index.date < target_date)
    #         ]
    #         if twins.empty:
    #             skipped_no_twins += 1
    #             if mode != "backtest":
    #                 print(
    #                     f"\n{'='*80}\n ANALYSIS - {self.ticker} - {target_date}\n{'='*80}"
    #                 )
    #                 print(f"ERROR: No matching historical 'twins' found")
    #                 print(
    #                     f"Current gap_pct: {current['gap_pct']:.2f}% "
    #                     f"(looking for matches in range: {current['gap_pct'] - 0.25:.2f}% to {current['gap_pct'] + 0.25:.2f}%)"
    #                 )
    #                 print(
    #                     f"Total historical records before {target_date}: {len(df[df.index.date < target_date])}"
    #                 )
    #             continue

    #         ladder = self._calculate_ladder(twins, current["close"], actual_anchor_time)

    #         # Get actual close for the target date (day_data already checked at start of loop)
    #         actual_close = day_data.iloc[-1]["close"]

    #         win = (actual_close <= ladder.p99_upper) and (
    #             actual_close >= ladder.p99_lower
    #         )
    #         results.append(
    #             {
    #                 "date": target_date,
    #                 "regime": classify_regime(current["vix1d"]),
    #                 "win": win,
    #             }
    #         )

    #         if mode != "backtest":
    #             print(
    #                 f"\n{'='*80}\n ANALYSIS - {self.ticker} - {target_date}\n{'='*80}"
    #             )
    #             print(f" REGIME: {classify_regime(current['vix1d']).value}")
    #             print(f" P95 RANGE: [${ladder.p95_lower:.2f}, ${ladder.p95_upper:.2f}]")
    #             print(f" P99 RANGE: [${ladder.p99_lower:.2f}, ${ladder.p99_upper:.2f}]")
    #             print(f" MAE WALLS: [${ladder.mae_lower:.2f}, ${ladder.mae_upper:.2f}]")

    #     if mode == "backtest":
    #         self.logger.info(
    #             f"Backtest complete: {len(results)} results, "
    #             f"{skipped_no_anchor} skipped (no anchor data), "
    #             f"{skipped_no_twins} skipped (no twins)"
    #         )
    #         self.print_performance_report(results)

    #     # PROBABILISTIC VS EMPIRICAL DASHBOARD
    #     print(
    #         f"\n{'='*80}\n ANALYSIS - {self.ticker} - {date_str} @ {anchor_hour}\n{'='*80}"
    #     )
    #     decay_display = (
    #         f"{ladder.range_decay_factor:.1%}"
    #         if not pd.isna(ladder.range_decay_factor)
    #         and np.isfinite(ladder.range_decay_factor)
    #         else "N/A"
    #     )
    #     print(f" TIME DECAY: {decay_display} of daily volatility typically remains.")

    #     print(f"\n--- PROBABILISTIC EXPECTATION (Based on {len(twins)} Twins) ---")
    #     print(f" P95 Confidence: [${ladder.p95_lower:.2f}, ${ladder.p95_upper:.2f}]")
    #     print(f" P99 Confidence: [${ladder.p99_lower:.2f}, ${ladder.p99_upper:.2f}]")

    #     print(f"\n--- EMPIRICAL DATA (Worst Case 'Hard Walls') ---")
    #     print(f" MAE Downside:  ${ladder.mae_lower:.2f}")
    #     print(f" MAE Upside:    ${ladder.mae_upper:.2f}")

    def print_performance_report(self, results):
        if not results:
            print(
                "\n"
                + "=" * 30
                + " PERFORMANCE REPORT "
                + "=" * 30
                + "\nNo results found. This could be due to:"
                + "\n  - No matching historical 'twins' found for test dates"
                + "\n  - Anchor hour not present in data for test dates"
                + "\n  - Insufficient historical data before test period"
                + "\nCheck logs for detailed information."
            )
            return
        res_df = pd.DataFrame(results)
        print(f"\n{'='*30} PERFORMANCE REPORT {'='*30}")
        print(f" {'Regime':<15} | {'Win Rate':<10} | {'Total Trades':<12}")
        print("-" * 65)
        for regime in res_df["regime"].unique():
            subset = res_df[res_df["regime"] == regime]
            win_rate = subset["win"].mean() * 100
            print(f" {regime.name:<15} | {win_rate:>8.1f}% | {len(subset):>12}")
        print("=" * 80)


async def main():
    parser = argparse.ArgumentParser(description="Professional 0DTE Trading Engine")
    parser.add_argument("--ticker", default="SPY")
    parser.add_argument(
        "--mode", choices=["live", "historical", "backtest"], default="live"
    )
    parser.add_argument("--bias", choices=["trend", "reversion"], default="trend")
    parser.add_argument("--date", help="YYYY-MM-DD")
    parser.add_argument(
        "--hour",
        default="10:30",
        help="Anchor hour in HH:MM format or just hour number (e.g., '10:30', '9', '14:00'). Default: 10:30. Interpreted in --timezone (default: EST)",
    )
    parser.add_argument(
        "--timezone",
        default="PST",
        help=(
            "Timezone for --hour parameter. Supports abbreviations (EST, EDT, PST, PDT, CST, CDT, MST, MDT, UTC, GMT, etc.) "
            "or full names (America/New_York, America/Los_Angeles, etc.). Default: PST. "
            "Data is stored in UTC, so the hour is converted from this timezone to UTC."
        ),
    )
    parser.add_argument("--test_split", type=float, default=0.2)
    parser.add_argument("--db_config", required=True)
    parser.add_argument("--cache_dir", default="./")

    args = parser.parse_args()
    engine = ZeroDTEEngine(args.ticker, args.db_config, cache_dir=args.cache_dir)
    try:
        await engine.initialize()
        target_date = args.date if args.date else datetime.now().strftime("%Y-%m-%d")
        await engine.run_analysis(
            args.mode,
            args.bias,
            target_date,
            args.hour,
            args.test_split,
            input_timezone=args.timezone,
        )
    except Exception as e:
        print(f"\nERROR: Exception occurred during analysis: {e}")
        import traceback
        traceback.print_exc()
        raise
    finally:
        await engine.close()


if __name__ == "__main__":
    asyncio.run(main())
