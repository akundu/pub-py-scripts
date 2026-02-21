#!/usr/bin/env python3
"""
CSV-based Closing Price Prediction Backtest

This script reads 5-minute interval data from equities_output CSV files
and runs backtests showing predictions at 15-minute intervals for the
last N trading days.

Usage:
    python scripts/csv_prediction_backtest.py --ticker NDX --days 5
    python scripts/csv_prediction_backtest.py --ticker SPX --days 5 --verbose
"""

import argparse
import os
import sys
from datetime import datetime, timedelta, time, date
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import pandas as pd
import numpy as np
from zoneinfo import ZoneInfo

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.strategy_utils.close_predictor import (
    PredictionContext,
    ClosePrediction,
    StatisticalClosePredictor,
    MLClosePredictor,
    EnsemblePredictor,
    BucketFeatures,
    VIXRegime,
    GapType,
    IntradayMove,
    DayOfWeek,
    PriorDayMove,
    IntradayRange,
    VIXChange,
    PriorClosePosition,
    Momentum5Day,
    FirstHourRange,
    MATrend,
    PriceVsMA50,
    OpeningDrive,
    GapFillStatus,
    TimeFromOpen,
    OpeningRangeBreakout,
    ConfidenceLevel,
    classify_intraday_move,
    classify_prior_day_move,
    classify_intraday_range,
    classify_vix_change,
    classify_prior_close_position,
    classify_momentum_5day,
    classify_first_hour_range,
    classify_ma_trend,
    classify_price_vs_ma50,
    classify_opening_drive,
    classify_gap_fill_status,
    classify_time_from_open,
    classify_opening_range_breakout,
    is_opex_week,
)

# Constants
ET_TZ = ZoneInfo("America/New_York")
UTC_TZ = ZoneInfo("UTC")
MARKET_OPEN_ET = time(9, 30)
MARKET_CLOSE_ET = time(16, 0)
EQUITIES_OUTPUT_DIR = Path(__file__).parent.parent / "equities_output"


@dataclass
class PredictionResult:
    """Single prediction result for backtesting."""
    date: str
    time_et: str
    current_price: float
    predicted_low: float
    predicted_mid: float
    predicted_high: float
    actual_close: float
    error_pct: float  # (predicted_mid - actual) / actual * 100
    in_range: bool  # actual close within predicted range
    confidence: str
    vix1d: Optional[float]
    samples: int


def load_csv_data(ticker: str, date_str: str) -> Optional[pd.DataFrame]:
    """Load CSV data for a specific ticker and date."""
    # Handle ticker format (I:NDX vs NDX)
    if not ticker.startswith("I:"):
        db_ticker = f"I:{ticker}"
    else:
        db_ticker = ticker

    csv_dir = EQUITIES_OUTPUT_DIR / db_ticker
    csv_file = csv_dir / f"{db_ticker}_equities_{date_str}.csv"

    if not csv_file.exists():
        return None

    df = pd.read_csv(csv_file, parse_dates=['timestamp'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
    df = df.sort_values('timestamp')
    return df


def get_available_dates(ticker: str, num_days: int = 5) -> List[str]:
    """Get list of available trading dates for a ticker."""
    if not ticker.startswith("I:"):
        db_ticker = f"I:{ticker}"
    else:
        db_ticker = ticker

    csv_dir = EQUITIES_OUTPUT_DIR / db_ticker
    if not csv_dir.exists():
        return []

    dates = []
    for f in sorted(csv_dir.glob(f"{db_ticker}_equities_*.csv")):
        date_str = f.stem.split("_")[-1]
        dates.append(date_str)

    return dates[-num_days:] if len(dates) > num_days else dates


def get_previous_close(ticker: str, current_date: str) -> Optional[float]:
    """Get closing price from the previous trading day."""
    if not ticker.startswith("I:"):
        db_ticker = f"I:{ticker}"
    else:
        db_ticker = ticker

    csv_dir = EQUITIES_OUTPUT_DIR / db_ticker

    # Get all available dates
    all_dates = []
    for f in sorted(csv_dir.glob(f"{db_ticker}_equities_*.csv")):
        date_str = f.stem.split("_")[-1]
        all_dates.append(date_str)

    # Find the date before current_date
    try:
        idx = all_dates.index(current_date)
        if idx == 0:
            return None
        prev_date = all_dates[idx - 1]
    except ValueError:
        return None

    # Load previous day's data and get closing price
    prev_df = load_csv_data(ticker, prev_date)
    if prev_df is None or prev_df.empty:
        return None

    # Get the last price of the day (close at 4 PM ET = 21:00 UTC)
    return prev_df.iloc[-1]['close']


def get_day_open(df: pd.DataFrame) -> float:
    """Get opening price for the day (9:30 AM ET = 14:30 UTC)."""
    # Filter to market hours (14:30 UTC onwards)
    market_df = df[df['timestamp'].dt.hour >= 14]
    if market_df.empty:
        return df.iloc[0]['open']
    return market_df.iloc[0]['open']


def get_day_close(df: pd.DataFrame) -> float:
    """Get closing price for the day (4:00 PM ET = 21:00 UTC)."""
    # The close is typically the last bar before/at 21:00 UTC
    close_time = df['timestamp'].dt.hour <= 21
    close_df = df[close_time]
    if close_df.empty:
        return df.iloc[-1]['close']
    return close_df.iloc[-1]['close']


def get_vix1d_at_time(date_str: str, target_time: datetime) -> Optional[float]:
    """Get VIX1D value at a specific time."""
    vix_df = load_csv_data("I:VIX1D", date_str)
    if vix_df is None or vix_df.empty:
        return None

    # Find the closest bar to target_time
    target_utc = target_time.astimezone(UTC_TZ)
    vix_df['time_diff'] = abs(vix_df['timestamp'] - target_utc)
    closest_idx = vix_df['time_diff'].idxmin()

    # Only use if within 15 minutes
    if vix_df.loc[closest_idx, 'time_diff'] > timedelta(minutes=15):
        # Fall back to most recent before target
        before = vix_df[vix_df['timestamp'] <= target_utc]
        if before.empty:
            return vix_df.iloc[0]['close']
        return before.iloc[-1]['close']

    return vix_df.loc[closest_idx, 'close']


def get_day_high_low(df: pd.DataFrame) -> Tuple[float, float]:
    """Get high and low for the day during market hours."""
    # Filter to market hours (14:30-21:00 UTC)
    market_df = df[(df['timestamp'].dt.hour >= 14) & (df['timestamp'].dt.hour <= 21)]
    if market_df.empty:
        return df['high'].max(), df['low'].min()
    return market_df['high'].max(), market_df['low'].min()


def get_first_hour_range(df: pd.DataFrame) -> Tuple[Optional[float], Optional[float]]:
    """Get high and low during first hour (9:30-10:30 AM ET = 14:30-15:30 UTC)."""
    # First hour: 14:30-15:30 UTC
    first_hour = df[(df['timestamp'].dt.hour == 14) |
                    ((df['timestamp'].dt.hour == 15) & (df['timestamp'].dt.minute <= 30))]
    if first_hour.empty:
        return None, None
    return first_hour['high'].max(), first_hour['low'].min()


def get_opening_range(df: pd.DataFrame) -> Tuple[Optional[float], Optional[float]]:
    """Get high and low during first 30 minutes (9:30-10:00 AM ET = 14:30-15:00 UTC)."""
    # First 30 min: 14:30-15:00 UTC
    first_30min = df[(df['timestamp'].dt.hour == 14) & (df['timestamp'].dt.minute >= 30)]
    if first_30min.empty:
        return None, None
    return first_30min['high'].max(), first_30min['low'].min()


def get_price_at_time(df: pd.DataFrame, hour_et: int, minute_et: int) -> Optional[float]:
    """Get price at a specific time in ET."""
    # Convert ET time to UTC (ET = UTC-5 during EST, UTC-4 during EDT)
    # For simplicity, use the closest bar
    target_hour_utc = hour_et + 5  # Approximate for EST

    # Find bar closest to target time
    target_bars = df[
        (df['timestamp'].dt.hour == target_hour_utc) &
        (df['timestamp'].dt.minute >= minute_et - 5) &
        (df['timestamp'].dt.minute <= minute_et + 5)
    ]

    if target_bars.empty:
        # Try EDT offset
        target_hour_utc = hour_et + 4
        target_bars = df[
            (df['timestamp'].dt.hour == target_hour_utc) &
            (df['timestamp'].dt.minute >= minute_et - 5) &
            (df['timestamp'].dt.minute <= minute_et + 5)
        ]

    if target_bars.empty:
        return None

    return target_bars.iloc[0]['close']


def compute_moving_averages(sorted_dates: List[str], date_to_close: Dict[str, float], current_idx: int) -> dict:
    """Compute 5/10/20/50-day SMAs from daily closes.

    Args:
        sorted_dates: List of dates in sorted order
        date_to_close: Mapping of date string to close price
        current_idx: Index of current date in sorted_dates

    Returns:
        Dict with ma5, ma10, ma20, ma50 (None if insufficient data)
    """
    result = {'ma5': None, 'ma10': None, 'ma20': None, 'ma50': None}

    for period, key in [(5, 'ma5'), (10, 'ma10'), (20, 'ma20'), (50, 'ma50')]:
        if current_idx >= period:
            closes = []
            for j in range(current_idx - period, current_idx):
                d = sorted_dates[j]
                if d in date_to_close:
                    closes.append(date_to_close[d])
            if len(closes) == period:
                result[key] = sum(closes) / period

    return result


def build_training_data(ticker: str, end_date: str, lookback_days: int = 90) -> pd.DataFrame:
    """Build training data from historical CSV files with all features."""
    if not ticker.startswith("I:"):
        db_ticker = f"I:{ticker}"
    else:
        db_ticker = ticker

    csv_dir = EQUITIES_OUTPUT_DIR / db_ticker

    # Get all available dates before end_date
    all_dates = []
    for f in sorted(csv_dir.glob(f"{db_ticker}_equities_*.csv")):
        date_str = f.stem.split("_")[-1]
        if date_str < end_date:
            all_dates.append(date_str)

    # Take last lookback_days
    train_dates = all_dates[-lookback_days:] if len(all_dates) > lookback_days else all_dates

    # Pre-load daily stats for all dates for efficient lookups
    daily_stats = {}
    vix_stats = {}

    for date_str in train_dates:
        df = load_csv_data(ticker, date_str)
        if df is None or df.empty:
            continue

        day_open = get_day_open(df)
        day_close = get_day_close(df)
        day_high, day_low = get_day_high_low(df)
        fh_high, fh_low = get_first_hour_range(df)
        or_high, or_low = get_opening_range(df)
        price_945 = get_price_at_time(df, 9, 45)

        daily_stats[date_str] = {
            'open': day_open,
            'close': day_close,
            'high': day_high,
            'low': day_low,
            'first_hour_high': fh_high,
            'first_hour_low': fh_low,
            'opening_range_high': or_high,
            'opening_range_low': or_low,
            'price_at_945': price_945,
        }

        # Get VIX1D for this day
        vix1d = get_vix1d_at_time(date_str, df.iloc[0]['timestamp'].to_pydatetime())
        vix_stats[date_str] = vix1d

    records = []

    # Build sliding window of historical closes for 5-day momentum
    date_to_close = {d: daily_stats[d]['close'] for d in daily_stats if d in daily_stats}
    sorted_dates = sorted(date_to_close.keys())

    # Pre-compute moving averages for each date
    date_to_mas = {}
    for idx, d in enumerate(sorted_dates):
        date_to_mas[d] = compute_moving_averages(sorted_dates, date_to_close, idx)

    for i, date_str in enumerate(train_dates):
        if date_str not in daily_stats:
            continue

        stats = daily_stats[date_str]
        df = load_csv_data(ticker, date_str)
        if df is None or df.empty:
            continue

        # Previous day data
        if i == 0:
            continue  # Skip first day - no previous data

        prev_date = train_dates[i - 1] if i > 0 else None
        prev_stats = daily_stats.get(prev_date) if prev_date else None

        if prev_stats is None:
            continue

        # Two days ago (for prior day move calculation)
        prev_prev_date = train_dates[i - 2] if i > 1 else None
        prev_prev_stats = daily_stats.get(prev_prev_date) if prev_prev_date else None

        # 5 days ago
        date_5days = train_dates[i - 5] if i >= 5 else None
        close_5days = daily_stats.get(date_5days, {}).get('close') if date_5days else None

        # Moving averages for this date
        mas = date_to_mas.get(date_str, {})

        # VIX data
        vix1d = vix_stats.get(date_str, 15.0)
        prev_vix1d = vix_stats.get(prev_date) if prev_date else None

        # Parse date for day of week
        dt = datetime.strptime(date_str, "%Y-%m-%d")
        day_of_week = dt.weekday()

        # Sample at each 15-minute interval during market hours
        for _, row in df.iterrows():
            ts = row['timestamp'].to_pydatetime()
            hour_utc = ts.hour
            minute = ts.minute

            # Only use data during market hours and at 15-min boundaries
            if 14 <= hour_utc <= 20 and minute in [0, 15, 30, 45]:
                ts_et = ts.astimezone(ET_TZ)
                hour_et = ts_et.hour

                if 9 <= hour_et <= 15:  # 9:30 AM to 3:45 PM
                    # Get intraday high/low up to this point
                    before = df[df['timestamp'] <= ts]
                    current_high = before['high'].max()
                    current_low = before['low'].min()

                    records.append({
                        'date': date_str,
                        'hour_et': hour_et + minute / 60,
                        'hour_price': row['close'],
                        'day_open': stats['open'],
                        'day_close': stats['close'],
                        'day_high': current_high,
                        'day_low': current_low,
                        'prev_close': prev_stats['close'],
                        'prev_day_close': prev_prev_stats['close'] if prev_prev_stats else None,
                        'prev_day_high': prev_stats['high'],
                        'prev_day_low': prev_stats['low'],
                        'vix1d': vix1d if vix1d else 15.0,
                        'prev_vix1d': prev_vix1d,
                        'day_of_week': day_of_week,
                        'close_5days_ago': close_5days,
                        'first_hour_high': stats['first_hour_high'],
                        'first_hour_low': stats['first_hour_low'],
                        # Open-time features
                        'opening_range_high': stats['opening_range_high'],
                        'opening_range_low': stats['opening_range_low'],
                        'price_at_945': stats['price_at_945'],
                        # Moving average features
                        'ma5': mas.get('ma5'),
                        'ma10': mas.get('ma10'),
                        'ma20': mas.get('ma20'),
                        'ma50': mas.get('ma50'),
                    })

    return pd.DataFrame(records)


async def append_today_from_questdb(
    db,
    ticker: str,
    today_date: str,
    hist_ctx: Dict,
    vix1d: Optional[float] = None
) -> pd.DataFrame:
    """
    Fetch today's intraday data from QuestDB realtime_data table and format as training rows.

    This appends ONLY today's data to supplement CSV historical training.
    Aggregates tick data into 5-minute OHLC bars.

    Args:
        db: QuestDB connection
        ticker: Ticker symbol (e.g., 'NDX' or 'I:NDX')
        today_date: Today's date string (YYYY-MM-DD)
        hist_ctx: Historical context dict from get_historical_context()
        vix1d: Current VIX1D value (optional)

    Returns:
        DataFrame with today's training rows in same format as build_training_data()
    """
    from datetime import datetime
    import asyncpg

    # Strip 'I:' prefix for realtime_data table (it uses 'NDX' not 'I:NDX')
    realtime_ticker = ticker.replace("I:", "")

    # Fetch today's tick data from realtime_data table
    try:
        # Get connection string from environment or db object
        import os
        db_config = os.getenv('QUEST_DB_STRING') or os.getenv('QUESTDB_CONNECTION_STRING')
        if not db_config:
            # Try to get from db object if it has the attribute
            db_config = getattr(db, 'db_config', None) or getattr(db, '_db_config', None)

        if not db_config:
            return pd.DataFrame()

        # Convert questdb:// to postgresql:// for asyncpg
        if db_config.startswith('questdb://'):
            db_config = db_config.replace('questdb://', 'postgresql://')

        # Query realtime_data table directly
        conn = await asyncpg.connect(db_config)

        tick_data = await conn.fetch(f"""
            SELECT timestamp, ticker, price
            FROM realtime_data
            WHERE ticker = '{realtime_ticker}'
            AND timestamp >= '{today_date} 00:00:00'::timestamp
            ORDER BY timestamp
        """)

        await conn.close()

        if not tick_data or len(tick_data) == 0:
            return pd.DataFrame()

        # Convert to DataFrame
        df = pd.DataFrame(tick_data, columns=['timestamp', 'ticker', 'price'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)

        # Aggregate into 5-minute OHLC bars
        df.set_index('timestamp', inplace=True)
        ohlc = df['price'].resample('5min').agg(['first', 'max', 'min', 'last', 'count'])
        ohlc.columns = ['open', 'high', 'low', 'close', 'volume']
        ohlc = ohlc[ohlc['volume'] > 0]  # Remove empty bars

        if ohlc.empty:
            return pd.DataFrame()

        # Reset index to get timestamp as column
        data = ohlc.reset_index()
        data.rename(columns={'index': 'timestamp'}, inplace=True)

    except Exception as e:
        print(f"Could not fetch today from QuestDB realtime_data: {e}")
        return pd.DataFrame()

    if data is None or data.empty:
        return pd.DataFrame()

    # Reset index and normalize columns
    df = data.reset_index()
    df.columns = [col.lower() for col in df.columns]

    if 'datetime' in df.columns:
        df.rename(columns={'datetime': 'timestamp'}, inplace=True)

    # Ensure timezone-aware timestamps
    if not isinstance(df['timestamp'].dtype, pd.DatetimeTZDtype):
        df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
    else:
        df['timestamp'] = df['timestamp'].dt.tz_convert('UTC')

    df = df.sort_values('timestamp')

    # Calculate daily stats for today
    day_open = df.iloc[0]['open']
    day_close = df.iloc[-1]['close']
    day_high = df['high'].max()
    day_low = df['low'].min()

    # Convert to ET for time-based features
    df['timestamp_et'] = df['timestamp'].dt.tz_convert(ET_TZ)

    # First hour range (9:30-10:30 ET)
    first_hour = df[
        (df['timestamp_et'].dt.hour == 9) |
        ((df['timestamp_et'].dt.hour == 10) & (df['timestamp_et'].dt.minute <= 30))
    ]
    fh_high = first_hour['high'].max() if not first_hour.empty else day_high
    fh_low = first_hour['low'].min() if not first_hour.empty else day_low

    # Opening range (9:30-10:00 ET)
    opening_range = df[
        (df['timestamp_et'].dt.hour == 9) & (df['timestamp_et'].dt.minute >= 30)
    ]
    or_high = opening_range['high'].max() if not opening_range.empty else day_high
    or_low = opening_range['low'].min() if not opening_range.empty else day_low

    # Price at 9:45 ET
    price_945_df = df[
        (df['timestamp_et'].dt.hour == 9) &
        (df['timestamp_et'].dt.minute >= 40) &
        (df['timestamp_et'].dt.minute <= 50)
    ]
    price_945 = price_945_df.iloc[0]['close'] if not price_945_df.empty else day_open

    # Get historical context
    prev_close = hist_ctx.get('day_1', {}).get('close', day_open)
    prev_day_close = hist_ctx.get('day_2', {}).get('close')
    prev_day_high = hist_ctx.get('day_1', {}).get('high', day_high)
    prev_day_low = hist_ctx.get('day_1', {}).get('low', day_low)
    close_5days_ago = hist_ctx.get('day_5', {}).get('close')

    # Moving averages from historical context
    ma5 = hist_ctx.get('ma5')
    ma10 = hist_ctx.get('ma10')
    ma20 = hist_ctx.get('ma20')
    ma50 = hist_ctx.get('ma50')

    # VIX
    if vix1d is None:
        vix1d = 15.0

    # Day of week
    dt = datetime.strptime(today_date, "%Y-%m-%d")
    day_of_week = dt.weekday()

    # Build training rows at 15-minute intervals
    records = []
    for _, row in df.iterrows():
        ts_et = row['timestamp_et'].to_pydatetime()
        hour_et = ts_et.hour
        minute = ts_et.minute

        # Only use market hours at 15-min boundaries
        if 9 <= hour_et <= 15 and minute in [0, 15, 30, 45]:
            # Get intraday high/low up to this point
            before = df[df['timestamp'] <= row['timestamp']]
            current_high = before['high'].max()
            current_low = before['low'].min()

            records.append({
                'date': today_date,
                'hour_et': hour_et + minute / 60,
                'hour_price': row['close'],
                'day_open': day_open,
                'day_close': day_close,
                'day_high': current_high,
                'day_low': current_low,
                'prev_close': prev_close,
                'prev_day_close': prev_day_close,
                'prev_day_high': prev_day_high,
                'prev_day_low': prev_day_low,
                'vix1d': vix1d,
                'prev_vix1d': None,  # Not critical for today's data
                'day_of_week': day_of_week,
                'close_5days_ago': close_5days_ago,
                'first_hour_high': fh_high,
                'first_hour_low': fh_low,
                'opening_range_high': or_high,
                'opening_range_low': or_low,
                'price_at_945': price_945,
                'ma5': ma5,
                'ma10': ma10,
                'ma20': ma20,
                'ma50': ma50,
            })

    return pd.DataFrame(records)


@dataclass
class DayContext:
    """Context data for a trading day used in predictions."""
    prev_close: float
    day_open: float
    vix1d: Optional[float]
    prev_day_close: Optional[float] = None
    prev_vix1d: Optional[float] = None
    prev_day_high: Optional[float] = None
    prev_day_low: Optional[float] = None
    close_5days_ago: Optional[float] = None
    first_hour_high: Optional[float] = None
    first_hour_low: Optional[float] = None
    # Open-time features
    opening_range_high: Optional[float] = None  # First 30 min high
    opening_range_low: Optional[float] = None   # First 30 min low
    price_at_945: Optional[float] = None        # Price at 9:45 AM ET
    # Moving average features
    ma5: Optional[float] = None    # 5-day simple moving average
    ma10: Optional[float] = None   # 10-day simple moving average
    ma20: Optional[float] = None   # 20-day simple moving average
    ma50: Optional[float] = None   # 50-day simple moving average


def make_prediction_at_time(
    predictor: StatisticalClosePredictor,
    df: pd.DataFrame,
    target_time: datetime,
    day_ctx: DayContext
) -> Optional[ClosePrediction]:
    """Make a prediction at a specific time using the trained predictor."""
    # Find the bar at or just before target_time
    target_utc = target_time.astimezone(UTC_TZ)
    before = df[df['timestamp'] <= target_utc]

    if before.empty:
        return None

    current_bar = before.iloc[-1]
    current_price = current_bar['close']

    # Get intraday high/low up to this point
    day_high = before['high'].max()
    day_low = before['low'].min()

    # Create prediction context with all features
    context = PredictionContext(
        ticker=df.iloc[0]['ticker'],
        current_price=current_price,
        prev_close=day_ctx.prev_close,
        day_open=day_ctx.day_open,
        current_time=target_time,
        vix1d=day_ctx.vix1d if day_ctx.vix1d else 15.0,
        day_high=day_high,
        day_low=day_low,
        # New features
        prev_day_close=day_ctx.prev_day_close,
        prev_vix1d=day_ctx.prev_vix1d,
        prev_day_high=day_ctx.prev_day_high,
        prev_day_low=day_ctx.prev_day_low,
        close_5days_ago=day_ctx.close_5days_ago,
        first_hour_high=day_ctx.first_hour_high,
        first_hour_low=day_ctx.first_hour_low,
        # Open-time features
        opening_range_high=day_ctx.opening_range_high,
        opening_range_low=day_ctx.opening_range_low,
        price_at_945=day_ctx.price_at_945,
        # Moving average features
        ma5=day_ctx.ma5,
        ma10=day_ctx.ma10,
        ma20=day_ctx.ma20,
        ma50=day_ctx.ma50,
    )

    # Make prediction
    try:
        prediction = predictor.predict(context)
        return prediction
    except Exception as e:
        print(f"  Warning: Prediction failed: {e}")
        return None


def get_historical_context(ticker: str, test_date: str, num_days_back: int = 55) -> Dict:
    """Get historical data needed for new features including moving averages."""
    if not ticker.startswith("I:"):
        db_ticker = f"I:{ticker}"
    else:
        db_ticker = ticker

    csv_dir = EQUITIES_OUTPUT_DIR / db_ticker

    # Get all dates
    all_dates = []
    for f in sorted(csv_dir.glob(f"{db_ticker}_equities_*.csv")):
        date_str = f.stem.split("_")[-1]
        all_dates.append(date_str)

    try:
        test_idx = all_dates.index(test_date)
    except ValueError:
        return {}

    result = {}

    # Collect closes for MA computation (need up to 50 prior days)
    historical_closes = []

    # Get previous days' data
    for i in range(1, num_days_back + 1):
        if test_idx - i >= 0:
            prev_date = all_dates[test_idx - i]
            prev_df = load_csv_data(ticker, prev_date)
            if prev_df is not None and not prev_df.empty:
                day_close = get_day_close(prev_df)
                historical_closes.append(day_close)

                # Only store detailed info for first 6 days
                if i <= 6:
                    day_high, day_low = get_day_high_low(prev_df)
                    result[f'day_{i}'] = {
                        'date': prev_date,
                        'close': day_close,
                        'high': day_high,
                        'low': day_low,
                        'vix1d': get_vix1d_at_time(prev_date, prev_df.iloc[0]['timestamp'].to_pydatetime())
                    }

    # Compute moving averages from historical closes (most recent first)
    if len(historical_closes) >= 5:
        result['ma5'] = sum(historical_closes[:5]) / 5
    if len(historical_closes) >= 10:
        result['ma10'] = sum(historical_closes[:10]) / 10
    if len(historical_closes) >= 20:
        result['ma20'] = sum(historical_closes[:20]) / 20
    if len(historical_closes) >= 50:
        result['ma50'] = sum(historical_closes[:50]) / 50

    return result


def compute_realized_vol(recent_closes: List[float], annualize: bool = True) -> float:
    """
    Compute realized volatility from list of closing prices.

    Args:
        recent_closes: List of closing prices (most recent first)
        annualize: If True, annualize the vol (multiply by sqrt(252))

    Returns:
        Realized volatility
    """
    if len(recent_closes) < 2:
        return 0.0

    # Reverse to chronological order
    closes = list(reversed(recent_closes))

    # Compute log returns
    log_returns = np.diff(np.log(closes))

    # Standard deviation
    vol = np.std(log_returns)

    if annualize:
        vol *= np.sqrt(252)

    return vol


def get_trailing_realized_vol(ticker: str, date_str: str, lookback_days: int = 5) -> Optional[float]:
    """
    Get trailing N-day realized vol for a specific date.

    Args:
        ticker: Ticker symbol
        date_str: Target date (YYYY-MM-DD)
        lookback_days: Number of days to look back

    Returns:
        Annualized realized volatility or None if insufficient data
    """
    if not ticker.startswith("I:"):
        db_ticker = f"I:{ticker}"
    else:
        db_ticker = ticker

    csv_dir = EQUITIES_OUTPUT_DIR / db_ticker

    # Get all dates
    all_dates = []
    for f in sorted(csv_dir.glob(f"{db_ticker}_equities_*.csv")):
        d_str = f.stem.split("_")[-1]
        all_dates.append(d_str)

    try:
        date_idx = all_dates.index(date_str)
    except ValueError:
        return None

    # Collect previous N days' closes
    closes = []
    for i in range(lookback_days):
        if date_idx - i >= 0:
            prev_date = all_dates[date_idx - i]
            prev_df = load_csv_data(ticker, prev_date)
            if prev_df is not None and not prev_df.empty:
                day_close = get_day_close(prev_df)
                closes.append(day_close)

    if len(closes) < 2:
        return None

    return compute_realized_vol(closes, annualize=True)


def get_historical_avg_vol(ticker: str, date_str: str, lookback_days: int = 90) -> Optional[float]:
    """
    Compute baseline average volatility over past N days.

    Uses rolling 5-day vols, returns median for robustness.
    Serves as denominator for vol scaling factor.

    Args:
        ticker: Ticker symbol
        date_str: Target date (YYYY-MM-DD)
        lookback_days: Number of days for baseline window

    Returns:
        Historical average volatility or None if insufficient data
    """
    if not ticker.startswith("I:"):
        db_ticker = f"I:{ticker}"
    else:
        db_ticker = ticker

    csv_dir = EQUITIES_OUTPUT_DIR / db_ticker

    # Get all dates
    all_dates = []
    for f in sorted(csv_dir.glob(f"{db_ticker}_equities_*.csv")):
        d_str = f.stem.split("_")[-1]
        all_dates.append(d_str)

    try:
        date_idx = all_dates.index(date_str)
    except ValueError:
        return None

    # Collect previous N days' closes
    closes = []
    for i in range(lookback_days + 5):  # Need extra for rolling window
        if date_idx - i >= 0:
            prev_date = all_dates[date_idx - i]
            prev_df = load_csv_data(ticker, prev_date)
            if prev_df is not None and not prev_df.empty:
                day_close = get_day_close(prev_df)
                closes.append(day_close)

    if len(closes) < 10:
        return None

    # Compute rolling 5-day vols
    rolling_vols = []
    for i in range(len(closes) - 5):
        window = closes[i:i+6]  # 6 points = 5 returns
        vol = compute_realized_vol(window, annualize=True)
        rolling_vols.append(vol)

    if len(rolling_vols) == 0:
        return None

    # Return median for robustness
    return np.median(rolling_vols)


def run_backtest(
    ticker: str,
    num_days: int = 5,
    lookback_days: int = 90,
    verbose: bool = False,
    feature_config: Optional[Dict[str, bool]] = None,
    use_ensemble: bool = False,
    ml_model_type: str = 'xgboost'
) -> List[PredictionResult]:
    """Run backtest for a ticker over the last N trading days.

    Args:
        ticker: Ticker symbol (NDX, SPX, etc.)
        num_days: Number of trading days to backtest
        lookback_days: Number of days for training data
        verbose: Show verbose output
        feature_config: Feature configuration dict
        use_ensemble: If True, use EnsemblePredictor combining statistical + ML
        ml_model_type: Type of ML model ('xgboost' or 'random_forest')

    Returns:
        List of PredictionResult objects
    """
    if not ticker.startswith("I:"):
        display_ticker = ticker
        db_ticker = f"I:{ticker}"
    else:
        display_ticker = ticker.replace("I:", "")
        db_ticker = ticker

    mode_str = "ENSEMBLE" if use_ensemble else "STATISTICAL"
    print(f"\n{'='*80}")
    print(f" CLOSING PRICE PREDICTION BACKTEST ({mode_str}) - {display_ticker}")
    print(f"{'='*80}")

    # Get available dates
    test_dates = get_available_dates(ticker, num_days)
    if len(test_dates) < 2:
        print(f"Error: Not enough data. Found {len(test_dates)} days, need at least 2.")
        return []

    print(f"\nTest period: {test_dates[0]} to {test_dates[-1]} ({len(test_dates)} days)")
    print(f"Training lookback: {lookback_days} days")

    # Default feature configuration
    if feature_config is None:
        feature_config = {
            'use_intraday_move': True,
            'use_day_of_week': False,
            'use_prior_day_move': True,
            'use_intraday_range': True,
            'use_vix_change': True,
            'use_prior_close_pos': True,
            'use_momentum_5day': True,
            'use_first_hour_range': True,
            'use_opex': True,
            # Open-time features
            'use_opening_drive': True,
            'use_gap_fill': True,
            'use_time_period': True,
            'use_orb': True,
            # Moving average features
            'use_ma_trend': True,
            'use_price_vs_ma50': True,
            'morning_mode': True,  # Disables time-dependent features in first hour
        }

    enabled_features = [k.replace('use_', '') for k, v in feature_config.items() if v]
    print(f"Enabled features: {enabled_features}")

    all_results = []

    for test_date in test_dates:
        if verbose:
            print(f"\n--- Testing {test_date} ---")

        # Build training data (excluding test date)
        train_df = build_training_data(ticker, test_date, lookback_days)
        if train_df.empty or len(train_df) < 50:
            print(f"  Warning: Insufficient training data for {test_date} ({len(train_df)} samples)")
            continue

        # Train predictor with configured features
        if use_ensemble:
            # Create ensemble predictor
            statistical = StatisticalClosePredictor(
                min_samples=5,
                **feature_config
            )
            predictor = EnsemblePredictor(
                statistical_predictor=statistical
            )
            # Configure ensemble weights
            predictor.ml_weight = 0.3  # 70% statistical, 30% ML
            predictor.ml_min_samples = 200
            predictor.ml_min_confidence = 0.5

            try:
                predictor.fit(train_df, fit_ml=True, ml_model_type=ml_model_type)
                if verbose and predictor.ml and predictor.ml.is_fitted:
                    print(f"  ML Model: {ml_model_type}, Val MAE: {predictor.ml.validation_mae:.4f}")
            except Exception as e:
                if verbose:
                    print(f"  Warning: ML fitting failed ({e}), using statistical only")
                predictor.fit(train_df, fit_ml=False)
        else:
            predictor = StatisticalClosePredictor(
                min_samples=5,
                **feature_config
            )
            predictor.fit(train_df)

        if verbose:
            if use_ensemble:
                print(f"  Training samples: {len(train_df)}, Valid buckets: {len(predictor.statistical.buckets)}")
            else:
                print(f"  Training samples: {len(train_df)}, Valid buckets: {len(predictor.buckets)}")

        # Load test day data
        test_df = load_csv_data(ticker, test_date)
        if test_df is None or test_df.empty:
            print(f"  Warning: No data for {test_date}")
            continue

        # Get historical context for new features
        hist_ctx = get_historical_context(ticker, test_date)

        # Get key prices for today
        day_1 = hist_ctx.get('day_1', {})
        day_2 = hist_ctx.get('day_2', {})
        day_5 = hist_ctx.get('day_5', {})

        prev_close = day_1.get('close')
        if prev_close is None:
            print(f"  Warning: No previous close for {test_date}")
            continue

        day_open = get_day_open(test_df)
        actual_close = get_day_close(test_df)
        fh_high, fh_low = get_first_hour_range(test_df)
        or_high, or_low = get_opening_range(test_df)
        price_945 = get_price_at_time(test_df, 9, 45)

        # Build day context with all features
        day_ctx = DayContext(
            prev_close=prev_close,
            day_open=day_open,
            vix1d=get_vix1d_at_time(test_date, test_df.iloc[0]['timestamp'].to_pydatetime()),
            prev_day_close=day_2.get('close'),
            prev_vix1d=day_1.get('vix1d'),
            prev_day_high=day_1.get('high'),
            prev_day_low=day_1.get('low'),
            close_5days_ago=day_5.get('close'),
            first_hour_high=fh_high,
            first_hour_low=fh_low,
            # Open-time features
            opening_range_high=or_high,
            opening_range_low=or_low,
            price_at_945=price_945,
            # Moving average features
            ma5=hist_ctx.get('ma5'),
            ma10=hist_ctx.get('ma10'),
            ma20=hist_ctx.get('ma20'),
            ma50=hist_ctx.get('ma50'),
        )

        if verbose:
            print(f"  Prev Close: ${prev_close:,.2f}, Open: ${day_open:,.2f}, Close: ${actual_close:,.2f}")

        # Generate 15-minute prediction times (9:30 AM to 3:45 PM ET)
        dt = datetime.strptime(test_date, "%Y-%m-%d")
        prediction_times = []

        for hour in range(9, 16):
            for minute in [0, 15, 30, 45]:
                if hour == 9 and minute < 30:
                    continue  # Market not open yet
                if hour == 15 and minute > 45:
                    continue  # After 3:45 PM

                pred_time = datetime(dt.year, dt.month, dt.day, hour, minute, tzinfo=ET_TZ)
                prediction_times.append(pred_time)

        # Make predictions at each time
        for pred_time in prediction_times:
            # Update VIX1D for this specific time
            day_ctx.vix1d = get_vix1d_at_time(test_date, pred_time)

            prediction = make_prediction_at_time(
                predictor, test_df, pred_time, day_ctx
            )

            if prediction is None:
                continue

            error_pct = (prediction.predicted_close_mid - actual_close) / actual_close * 100
            in_range = prediction.predicted_close_low <= actual_close <= prediction.predicted_close_high

            result = PredictionResult(
                date=test_date,
                time_et=pred_time.strftime("%H:%M"),
                current_price=prediction.predicted_close_mid,  # Will update with actual current
                predicted_low=prediction.predicted_close_low,
                predicted_mid=prediction.predicted_close_mid,
                predicted_high=prediction.predicted_close_high,
                actual_close=actual_close,
                error_pct=error_pct,
                in_range=in_range,
                confidence=prediction.confidence.value,
                vix1d=day_ctx.vix1d,
                samples=prediction.sample_size
            )

            # Get actual current price at prediction time
            target_utc = pred_time.astimezone(UTC_TZ)
            before = test_df[test_df['timestamp'] <= target_utc]
            if not before.empty:
                result.current_price = before.iloc[-1]['close']

            all_results.append(result)

    return all_results


def print_results(results: List[PredictionResult], ticker: str):
    """Print backtest results in a formatted table."""
    if not results:
        print("\nNo results to display.")
        return

    display_ticker = ticker.replace("I:", "") if ticker.startswith("I:") else ticker

    print(f"\n{'='*120}")
    print(f" 15-MINUTE PREDICTION RESULTS - {display_ticker}")
    print(f"{'='*120}")

    # Group by date
    current_date = None

    print(f"\n{'Date':<12} {'Time':<6} {'Current':>12} {'Pred Low':>12} {'Pred Mid':>12} {'Pred High':>12} {'Actual':>12} {'Error':>8} {'In Range':<8} {'Conf':<8}")
    print("-" * 120)

    for r in results:
        if r.date != current_date:
            if current_date is not None:
                print("-" * 120)
            current_date = r.date

        in_range_str = "YES" if r.in_range else "NO"
        in_range_color = "" if r.in_range else ""

        print(f"{r.date:<12} {r.time_et:<6} "
              f"${r.current_price:>10,.2f} "
              f"${r.predicted_low:>10,.2f} "
              f"${r.predicted_mid:>10,.2f} "
              f"${r.predicted_high:>10,.2f} "
              f"${r.actual_close:>10,.2f} "
              f"{r.error_pct:>+7.2f}% "
              f"{in_range_str:<8} "
              f"{r.confidence:<8}")

    # Summary statistics
    print(f"\n{'='*120}")
    print(" SUMMARY STATISTICS")
    print(f"{'='*120}")

    total = len(results)
    in_range_count = sum(1 for r in results if r.in_range)
    accuracy = in_range_count / total * 100 if total > 0 else 0

    errors = [r.error_pct for r in results]
    mae = np.mean(np.abs(errors))
    rmse = np.sqrt(np.mean(np.square(errors)))

    print(f"\nTotal Predictions: {total}")
    print(f"In Range Accuracy: {in_range_count}/{total} ({accuracy:.1f}%)")
    print(f"Mean Absolute Error: {mae:.3f}%")
    print(f"RMSE: {rmse:.3f}%")

    # Accuracy by time of day
    print(f"\n{'Time Block':<15} {'Count':>8} {'In Range':>10} {'Accuracy':>10} {'MAE':>10}")
    print("-" * 60)

    time_blocks = {}
    for r in results:
        hour = int(r.time_et.split(":")[0])
        block = f"{hour}:00-{hour}:59"
        if block not in time_blocks:
            time_blocks[block] = []
        time_blocks[block].append(r)

    for block in sorted(time_blocks.keys()):
        block_results = time_blocks[block]
        count = len(block_results)
        in_range = sum(1 for r in block_results if r.in_range)
        acc = in_range / count * 100 if count > 0 else 0
        block_mae = np.mean([abs(r.error_pct) for r in block_results])
        print(f"{block:<15} {count:>8} {in_range:>10} {acc:>9.1f}% {block_mae:>9.3f}%")

    # Accuracy by confidence level
    print(f"\n{'Confidence':<15} {'Count':>8} {'In Range':>10} {'Accuracy':>10}")
    print("-" * 50)

    conf_blocks = {}
    for r in results:
        if r.confidence not in conf_blocks:
            conf_blocks[r.confidence] = []
        conf_blocks[r.confidence].append(r)

    for conf in ['HIGH', 'MEDIUM', 'LOW', 'VERY_LOW']:
        if conf in conf_blocks:
            block_results = conf_blocks[conf]
            count = len(block_results)
            in_range = sum(1 for r in block_results if r.in_range)
            acc = in_range / count * 100 if count > 0 else 0
            print(f"{conf:<15} {count:>8} {in_range:>10} {acc:>9.1f}%")


def main():
    parser = argparse.ArgumentParser(
        description='Run closing price prediction backtest using CSV data'
    )
    parser.add_argument(
        '--ticker', '-t',
        type=str,
        default='NDX',
        help='Ticker symbol (NDX, SPX, etc.)'
    )
    parser.add_argument(
        '--days', '-d',
        type=int,
        default=5,
        help='Number of trading days to backtest (default: 5)'
    )
    parser.add_argument(
        '--lookback',
        type=int,
        default=90,
        help='Number of days for training data (default: 90)'
    )
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Show verbose output'
    )
    parser.add_argument(
        '--output', '-o',
        type=str,
        help='Save results to CSV file'
    )
    parser.add_argument(
        '--ensemble',
        action='store_true',
        help='Use ensemble predictor (statistical + ML)'
    )
    parser.add_argument(
        '--ml-model',
        type=str,
        default='xgboost',
        choices=['xgboost', 'random_forest'],
        help='ML model type for ensemble (default: xgboost)'
    )

    args = parser.parse_args()

    # Run backtest
    results = run_backtest(
        ticker=args.ticker,
        num_days=args.days,
        lookback_days=args.lookback,
        verbose=args.verbose,
        use_ensemble=args.ensemble,
        ml_model_type=args.ml_model
    )

    # Print results
    print_results(results, args.ticker)

    # Save to CSV if requested
    if args.output and results:
        df = pd.DataFrame([{
            'date': r.date,
            'time_et': r.time_et,
            'current_price': r.current_price,
            'predicted_low': r.predicted_low,
            'predicted_mid': r.predicted_mid,
            'predicted_high': r.predicted_high,
            'actual_close': r.actual_close,
            'error_pct': r.error_pct,
            'in_range': r.in_range,
            'confidence': r.confidence,
            'vix1d': r.vix1d,
            'samples': r.samples
        } for r in results])
        df.to_csv(args.output, index=False)
        print(f"\nResults saved to: {args.output}")


if __name__ == '__main__':
    main()
