#!/usr/bin/env python3
"""
Live Prediction Dashboard

Real-time closing price prediction dashboard for NDX/SPX indices.
Shows current predictions updated at configurable intervals during market hours.

Usage:
    python scripts/live_prediction_dashboard.py --ticker NDX --interval 60
    python scripts/live_prediction_dashboard.py --ticker SPX --interval 30 --verbose
"""

import argparse
import os
import sys
import time
from datetime import datetime, timedelta, time as dtime
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np
from zoneinfo import ZoneInfo

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.strategy_utils.close_predictor import (
    PredictionContext,
    ClosePrediction,
    StatisticalClosePredictor,
    ConfidenceLevel,
)
from scripts.csv_prediction_backtest import (
    load_csv_data,
    get_available_dates,
    get_day_open,
    get_day_close,
    get_first_hour_range,
    get_opening_range,
    get_price_at_time,
    get_vix1d_at_time,
    get_historical_context,
    build_training_data,
    DayContext,
)

# Constants
ET_TZ = ZoneInfo("America/New_York")
UTC_TZ = ZoneInfo("UTC")
MARKET_OPEN_ET = dtime(9, 30)
MARKET_CLOSE_ET = dtime(16, 0)
EQUITIES_OUTPUT_DIR = Path(__file__).parent.parent / "equities_output"


def clear_screen():
    """Clear the terminal screen."""
    os.system('cls' if os.name == 'nt' else 'clear')


def is_market_hours(dt: datetime) -> bool:
    """Check if datetime is during US market hours (9:30 AM - 4:00 PM ET)."""
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=ET_TZ)
    else:
        dt = dt.astimezone(ET_TZ)

    # Check if weekday (Mon=0, Sun=6)
    if dt.weekday() >= 5:
        return False

    market_open = dt.replace(hour=9, minute=30, second=0, microsecond=0)
    market_close = dt.replace(hour=16, minute=0, second=0, microsecond=0)

    return market_open <= dt <= market_close


def get_latest_price(ticker: str, date_str: str) -> Optional[Tuple[float, datetime]]:
    """Get the latest available price for a ticker."""
    df = load_csv_data(ticker, date_str)
    if df is None or df.empty:
        return None

    latest = df.iloc[-1]
    return latest['close'], latest['timestamp'].to_pydatetime()


def train_predictor(ticker: str, test_date: str, lookback_days: int = 480) -> Optional[StatisticalClosePredictor]:
    """Train the predictor on historical data."""
    train_df = build_training_data(ticker, test_date, lookback_days)
    if train_df.empty or len(train_df) < 50:
        return None

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
        'use_opening_drive': True,
        'use_gap_fill': True,
        'use_time_period': True,
        'use_orb': True,
        'morning_mode': True,
    }

    predictor = StatisticalClosePredictor(
        min_samples=5,
        **feature_config
    )
    predictor.fit(train_df)
    return predictor


def make_live_prediction(
    predictor: StatisticalClosePredictor,
    ticker: str,
    current_price: float,
    current_time: datetime,
    day_ctx: DayContext,
    day_high: float,
    day_low: float,
) -> Optional[ClosePrediction]:
    """Make a prediction using current market data."""
    if not ticker.startswith("I:"):
        db_ticker = f"I:{ticker}"
    else:
        db_ticker = ticker

    context = PredictionContext(
        ticker=db_ticker,
        current_price=current_price,
        prev_close=day_ctx.prev_close,
        day_open=day_ctx.day_open,
        current_time=current_time,
        vix1d=day_ctx.vix1d if day_ctx.vix1d else 15.0,
        day_high=day_high,
        day_low=day_low,
        prev_day_close=day_ctx.prev_day_close,
        prev_vix1d=day_ctx.prev_vix1d,
        prev_day_high=day_ctx.prev_day_high,
        prev_day_low=day_ctx.prev_day_low,
        close_5days_ago=day_ctx.close_5days_ago,
        first_hour_high=day_ctx.first_hour_high,
        first_hour_low=day_ctx.first_hour_low,
        opening_range_high=day_ctx.opening_range_high,
        opening_range_low=day_ctx.opening_range_low,
        price_at_945=day_ctx.price_at_945,
    )

    try:
        return predictor.predict(context)
    except Exception as e:
        print(f"Prediction error: {e}")
        return None


def format_price(price: float) -> str:
    """Format price with commas."""
    return f"${price:,.2f}"


def format_pct(pct: float) -> str:
    """Format percentage with sign."""
    return f"{pct:+.2f}%"


def print_dashboard(
    ticker: str,
    current_price: float,
    current_time: datetime,
    prediction: ClosePrediction,
    day_ctx: DayContext,
    day_high: float,
    day_low: float,
    training_samples: int
):
    """Print the prediction dashboard."""
    clear_screen()

    display_ticker = ticker.replace("I:", "") if ticker.startswith("I:") else ticker
    time_str = current_time.astimezone(ET_TZ).strftime("%I:%M:%S %p ET")
    date_str = current_time.astimezone(ET_TZ).strftime("%A, %B %d, %Y")

    # Calculate key metrics
    gap_pct = ((day_ctx.day_open - day_ctx.prev_close) / day_ctx.prev_close) * 100
    intraday_pct = ((current_price - day_ctx.day_open) / day_ctx.day_open) * 100
    from_prev_pct = ((current_price - day_ctx.prev_close) / day_ctx.prev_close) * 100
    range_pct = ((day_high - day_low) / day_ctx.day_open) * 100

    # Calculate buffer from predicted range
    low_buffer_pct = ((prediction.predicted_close_low - day_ctx.prev_close) / day_ctx.prev_close) * 100
    high_buffer_pct = ((prediction.predicted_close_high - day_ctx.prev_close) / day_ctx.prev_close) * 100

    print("=" * 72)
    print(f"  {display_ticker} LIVE CLOSING PRICE PREDICTION")
    print(f"  {date_str} - {time_str}")
    print("=" * 72)

    print(f"\n  {'CURRENT MARKET STATUS':^68}")
    print(f"  {'-' * 68}")
    print(f"  Current Price:    {format_price(current_price):>15}   From Open:  {format_pct(intraday_pct):>10}")
    print(f"  Previous Close:   {format_price(day_ctx.prev_close):>15}   From Prev:  {format_pct(from_prev_pct):>10}")
    print(f"  Today's Open:     {format_price(day_ctx.day_open):>15}   Gap:        {format_pct(gap_pct):>10}")
    print(f"  Day High:         {format_price(day_high):>15}   Day Range:  {format_pct(range_pct):>10}")
    print(f"  Day Low:          {format_price(day_low):>15}")
    if day_ctx.vix1d:
        print(f"  VIX1D:            {day_ctx.vix1d:>15.2f}")

    print(f"\n  {'PREDICTED CLOSING RANGE (80% Confidence)':^68}")
    print(f"  {'-' * 68}")

    # Confidence color
    conf_emoji = {
        ConfidenceLevel.HIGH: "[HIGH]",
        ConfidenceLevel.MEDIUM: "[MED]",
        ConfidenceLevel.LOW: "[LOW]",
        ConfidenceLevel.VERY_LOW: "[V.LOW]"
    }

    pred_low_from_current = ((prediction.predicted_close_low - current_price) / current_price) * 100
    pred_mid_from_current = ((prediction.predicted_close_mid - current_price) / current_price) * 100
    pred_high_from_current = ((prediction.predicted_close_high - current_price) / current_price) * 100

    print(f"  Low Estimate:     {format_price(prediction.predicted_close_low):>15}   {format_pct(pred_low_from_current):>10} from current")
    print(f"  Mid Estimate:     {format_price(prediction.predicted_close_mid):>15}   {format_pct(pred_mid_from_current):>10} from current")
    print(f"  High Estimate:    {format_price(prediction.predicted_close_high):>15}   {format_pct(pred_high_from_current):>10} from current")
    print(f"\n  Confidence:       {prediction.confidence.value:>15}   {conf_emoji.get(prediction.confidence, '')}")
    print(f"  Sample Size:      {prediction.sample_size:>15}")
    print(f"  Method:           {prediction.prediction_method:>15}")

    print(f"\n  {'CREDIT SPREAD IMPLICATIONS':^68}")
    print(f"  {'-' * 68}")
    print(f"  Put spreads safe BELOW:   {format_price(prediction.predicted_close_low):>12}  ({format_pct(low_buffer_pct)} from prev close)")
    print(f"  Call spreads safe ABOVE:  {format_price(prediction.predicted_close_high):>12}  ({format_pct(high_buffer_pct)} from prev close)")

    print(f"\n  {'RISK RECOMMENDATION':^68}")
    print(f"  {'-' * 68}")
    print(f"  Risk Level:       {prediction.recommended_risk_level:>15}/10")
    if prediction.risk_rationale:
        # Wrap long rationale
        rationale = prediction.risk_rationale[:60]
        print(f"  Rationale:        {rationale}")

    print(f"\n  {'TRAINING INFO':^68}")
    print(f"  {'-' * 68}")
    print(f"  Training samples: {training_samples:>15}")

    # Time until close
    now_et = current_time.astimezone(ET_TZ)
    close_time = now_et.replace(hour=16, minute=0, second=0, microsecond=0)
    time_to_close = close_time - now_et
    hours, remainder = divmod(int(time_to_close.total_seconds()), 3600)
    minutes, seconds = divmod(remainder, 60)
    print(f"  Time to close:    {hours}h {minutes}m {seconds}s")

    print("\n" + "=" * 72)
    print("  Press Ctrl+C to exit")
    print("=" * 72)


def run_dashboard(
    ticker: str,
    interval_seconds: int = 60,
    lookback_days: int = 480,
    verbose: bool = False
):
    """Run the live prediction dashboard."""
    if not ticker.startswith("I:"):
        db_ticker = f"I:{ticker}"
        display_ticker = ticker
    else:
        db_ticker = ticker
        display_ticker = ticker.replace("I:", "")

    print(f"\nInitializing {display_ticker} Live Prediction Dashboard...")
    print(f"Update interval: {interval_seconds} seconds")
    print(f"Training lookback: {lookback_days} days")

    # Get today's date
    now = datetime.now(ET_TZ)
    today_str = now.strftime("%Y-%m-%d")

    # Check if we have today's data
    available_dates = get_available_dates(ticker, num_days=5)
    if not available_dates:
        print(f"Error: No data available for {display_ticker}")
        return

    # Use most recent available date for training
    test_date = available_dates[-1]

    if verbose:
        print(f"Most recent data: {test_date}")

    # Check if market is open
    if not is_market_hours(now):
        print(f"\nMarket is currently closed.")
        print(f"Market hours: 9:30 AM - 4:00 PM ET, Monday-Friday")
        print(f"Current time: {now.strftime('%I:%M %p ET')}")

        # Ask if user wants to use historical data for demo
        print(f"\nUsing most recent available data ({test_date}) for demonstration.")

    # Train predictor
    print(f"\nTraining predictor on {lookback_days} days of historical data...")
    predictor = train_predictor(ticker, test_date, lookback_days)
    if predictor is None:
        print("Error: Could not train predictor - insufficient data")
        return

    training_samples = sum(len(moves) for moves in predictor.buckets.values())
    print(f"Training complete. {training_samples} samples, {len(predictor.buckets)} buckets")

    # Get historical context
    hist_ctx = get_historical_context(ticker, test_date)
    day_1 = hist_ctx.get('day_1', {})
    day_2 = hist_ctx.get('day_2', {})
    day_5 = hist_ctx.get('day_5', {})

    prev_close = day_1.get('close')
    if prev_close is None:
        print(f"Error: No previous close available")
        return

    # Load test day data
    test_df = load_csv_data(ticker, test_date)
    if test_df is None or test_df.empty:
        print(f"Error: No data for {test_date}")
        return

    day_open = get_day_open(test_df)
    fh_high, fh_low = get_first_hour_range(test_df)
    or_high, or_low = get_opening_range(test_df)
    price_945 = get_price_at_time(test_df, 9, 45)

    # Build day context
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
        opening_range_high=or_high,
        opening_range_low=or_low,
        price_at_945=price_945,
    )

    # For demo mode, iterate through the day's data
    print("\nStarting dashboard... (demo mode using historical data)")
    time.sleep(2)

    try:
        # Get all timestamps for the day at 15-min intervals
        timestamps = test_df[
            (test_df['timestamp'].dt.hour >= 14) &  # 9:30 AM ET onwards
            (test_df['timestamp'].dt.hour <= 20) &  # Until 4 PM ET
            (test_df['timestamp'].dt.minute % 15 == 0)  # 15-min intervals
        ]['timestamp'].tolist()

        if not timestamps:
            timestamps = test_df['timestamp'].tolist()

        for ts in timestamps:
            ts = ts.to_pydatetime()

            # Get data up to this time
            before = test_df[test_df['timestamp'] <= ts]
            if before.empty:
                continue

            current_price = before.iloc[-1]['close']
            day_high = before['high'].max()
            day_low = before['low'].min()

            # Update VIX1D
            day_ctx.vix1d = get_vix1d_at_time(test_date, ts)

            # Make prediction
            prediction = make_live_prediction(
                predictor, ticker, current_price, ts, day_ctx, day_high, day_low
            )

            if prediction:
                print_dashboard(
                    ticker, current_price, ts, prediction, day_ctx, day_high, day_low, training_samples
                )

            time.sleep(interval_seconds / 10)  # Speed up for demo

    except KeyboardInterrupt:
        print("\n\nDashboard stopped by user.")


def main():
    parser = argparse.ArgumentParser(
        description='Live closing price prediction dashboard'
    )
    parser.add_argument(
        '--ticker', '-t',
        type=str,
        default='NDX',
        help='Ticker symbol (NDX, SPX, etc.)'
    )
    parser.add_argument(
        '--interval', '-i',
        type=int,
        default=60,
        help='Update interval in seconds (default: 60)'
    )
    parser.add_argument(
        '--lookback',
        type=int,
        default=480,
        help='Number of days for training data (default: 480)'
    )
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Show verbose output'
    )

    args = parser.parse_args()

    run_dashboard(
        ticker=args.ticker,
        interval_seconds=args.interval,
        lookback_days=args.lookback,
        verbose=args.verbose
    )


if __name__ == '__main__':
    main()
