"""
Live and demo modes for the Unified Close Predictor.
"""

import asyncio
import os
import time as _time
from datetime import datetime
from typing import Optional

import pandas as pd

from .models import ET_TZ, _intraday_vol_cache
from .features import get_intraday_vol_factor
from .prediction import _train_statistical, make_unified_prediction
from .display import print_live_display

from scripts.percentile_range_backtest import (
    collect_all_data,
    TIME_SLOTS,
)
from scripts.csv_prediction_backtest import (
    load_csv_data,
    get_available_dates,
    get_day_open,
    get_previous_close,
    get_vix1d_at_time,
    get_first_hour_range,
    get_opening_range,
    get_price_at_time,
    get_historical_context,
    DayContext,
)


def _build_day_context(
    ticker: str,
    test_date: str,
    test_df: pd.DataFrame,
) -> Optional[DayContext]:
    """Build DayContext for a given date from CSV data."""
    day_open = get_day_open(test_df)
    fh_high, fh_low = get_first_hour_range(test_df)
    or_high, or_low = get_opening_range(test_df)
    price_945 = get_price_at_time(test_df, 9, 45)

    hist_ctx = get_historical_context(ticker, test_date)
    day_1 = hist_ctx.get('day_1', {})
    day_2 = hist_ctx.get('day_2', {})
    day_5 = hist_ctx.get('day_5', {})

    prev_close_val = day_1.get('close')
    if prev_close_val is None:
        prev_close_val = get_previous_close(ticker, test_date)
    if prev_close_val is None:
        return None

    return DayContext(
        prev_close=prev_close_val,
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
        ma5=hist_ctx.get('ma5'),
        ma10=hist_ctx.get('ma10'),
        ma20=hist_ctx.get('ma20'),
        ma50=hist_ctx.get('ma50'),
    )


def _find_nearest_time_label(hour_et: int, minute_et: int) -> str:
    """Find the nearest TIME_SLOT label for a given ET hour:minute."""
    best = None
    best_diff = 9999
    for h, m in TIME_SLOTS:
        diff = abs((h * 60 + m) - (hour_et * 60 + minute_et))
        if diff < best_diff:
            best_diff = diff
            best = f"{h}:{m:02d}"
    return best or "10:00"


def run_demo_loop(
    ticker: str,
    lookback: int = 250,
    interval: int = 30,
    vol_scale: bool = True,
):
    """Demo mode -- iterate through the most recent day's CSV data."""
    display_ticker = ticker.replace("I:", "") if ticker.startswith("I:") else ticker

    print(f"\nInitializing {display_ticker} Unified Predictor (demo mode)...")

    all_dates = get_available_dates(ticker, lookback + 20)
    if len(all_dates) < lookback + 5:
        print(f"Not enough data. Have {len(all_dates)} dates.")
        return

    test_date = all_dates[-1]
    print(f"Using data from {test_date}")

    # Collect percentile data
    print("Collecting percentile data...")
    pct_df = collect_all_data(ticker, all_dates)
    if pct_df is None or pct_df.empty:
        print("No percentile data.")
        return

    unique_dates = sorted(pct_df['date'].unique())
    test_idx = unique_dates.index(test_date) if test_date in unique_dates else len(unique_dates) - 1
    pct_train_dates = set(unique_dates[max(0, test_idx - lookback):test_idx])
    train_dates_sorted = unique_dates[max(0, test_idx - lookback):test_idx]

    # Clear intraday vol cache
    _intraday_vol_cache.clear()

    # Train statistical model
    print("Training statistical model...")
    stat_predictor = _train_statistical(ticker, test_date, lookback)
    if stat_predictor:
        # Check if it's LightGBM or Statistical predictor
        if hasattr(stat_predictor, 'buckets'):
            total_samples = sum(len(m) for m in stat_predictor.buckets.values())
            print(f"Statistical model: {total_samples} samples, {len(stat_predictor.buckets)} buckets")
        else:
            # LightGBM predictor
            predictor_type = type(stat_predictor).__name__
            print(f"{predictor_type}: Trained successfully")

    # Load test day
    test_df = load_csv_data(ticker, test_date)
    if test_df is None or test_df.empty:
        print(f"No data for {test_date}")
        return

    day_ctx = _build_day_context(ticker, test_date, test_df)
    if day_ctx is None:
        print("Could not build day context.")
        return

    prev_close_val = day_ctx.prev_close

    # Get realized vol for test date from pct_df
    test_pct_rows = pct_df[pct_df['date'] == test_date]
    current_vol = None
    if not test_pct_rows.empty:
        vol_vals = test_pct_rows['realized_vol'].dropna()
        if not vol_vals.empty:
            current_vol = vol_vals.iloc[0]

    # Iterate through 15-min timestamps
    timestamps = test_df[
        (test_df['timestamp'].dt.hour >= 14) &
        (test_df['timestamp'].dt.hour <= 20) &
        (test_df['timestamp'].dt.minute % 15 == 0)
    ]['timestamp'].tolist()

    if not timestamps:
        timestamps = test_df['timestamp'].tolist()

    print(f"\nStarting demo... ({len(timestamps)} time points)")
    _time.sleep(1)

    try:
        for ts in timestamps:
            ts = ts.to_pydatetime()
            ts_et = ts.astimezone(ET_TZ)
            hour_et = ts_et.hour
            minute_et = ts_et.minute

            if hour_et < 9 or (hour_et == 9 and minute_et < 30) or hour_et >= 16:
                continue

            time_label = _find_nearest_time_label(hour_et, minute_et)

            # Data up to this time
            before = test_df[test_df['timestamp'] <= ts]
            if before.empty:
                continue

            current_price = before.iloc[-1]['close']
            day_high_now = before['high'].max()
            day_low_now = before['low'].min()

            day_ctx.vix1d = get_vix1d_at_time(test_date, ts)

            # Compute intraday vol factor
            ivol_factor = get_intraday_vol_factor(
                ticker, test_date, time_label, test_df, train_dates_sorted,
            )

            pred = make_unified_prediction(
                pct_df=pct_df,
                predictor=stat_predictor,
                ticker=ticker,
                current_price=current_price,
                prev_close=prev_close_val,
                current_time=ts_et,
                time_label=time_label,
                day_ctx=day_ctx,
                day_high=day_high_now,
                day_low=day_low_now,
                train_dates=pct_train_dates,
                current_vol=current_vol,
                vol_scale=vol_scale,
                data_source="CSV demo",
                intraday_vol_factor=ivol_factor,
            )

            if pred:
                print_live_display(pred, interval)

            _time.sleep(interval / 10)  # speed up for demo

    except KeyboardInterrupt:
        print("\n\nDemo stopped.")


async def run_live_loop(
    ticker: str,
    lookback: int = 250,
    interval: int = 30,
    vol_scale: bool = True,
):
    """Live mode -- fetch prices from QuestDB and display predictions."""
    display_ticker = ticker.replace("I:", "") if ticker.startswith("I:") else ticker
    db_ticker = f"I:{ticker}" if not ticker.startswith("I:") else ticker

    print(f"\nInitializing {display_ticker} Unified Predictor (live mode)...")

    # Import and connect to QuestDB
    from common.stock_db import get_stock_db

    db = get_stock_db(
        'questdb',
        db_config=os.getenv('QUESTDB_CONNECTION_STRING'),
        enable_cache=True,
        redis_url=os.getenv('REDIS_URL'),
    )

    # Prepare percentile data from CSV
    all_dates = get_available_dates(ticker, lookback + 20)
    if len(all_dates) < lookback + 5:
        print(f"Not enough CSV data for percentile model. Have {len(all_dates)} dates.")
        return

    print("Collecting percentile data...")
    pct_df = collect_all_data(ticker, all_dates)
    if pct_df is None or pct_df.empty:
        print("No percentile data.")
        return

    unique_dates = sorted(pct_df['date'].unique())
    pct_train_dates = set(unique_dates[:-1])  # all except last
    train_dates_sorted = unique_dates[:-1]

    # Clear intraday vol cache
    _intraday_vol_cache.clear()

    # Get realized vol from most recent date
    last_date = unique_dates[-1]
    last_rows = pct_df[pct_df['date'] == last_date]
    current_vol = None
    if not last_rows.empty:
        vol_vals = last_rows['realized_vol'].dropna()
        if not vol_vals.empty:
            current_vol = vol_vals.iloc[0]

    # Train statistical model on latest available
    test_date = all_dates[-1]
    print("Training statistical model...")
    stat_predictor = _train_statistical(ticker, test_date, lookback)
    if stat_predictor:
        # Check if it's LightGBM or Statistical predictor
        if hasattr(stat_predictor, 'buckets'):
            total_samples = sum(len(m) for m in stat_predictor.buckets.values())
            print(f"Statistical model: {total_samples} samples, {len(stat_predictor.buckets)} buckets")
        else:
            # LightGBM predictor
            predictor_type = type(stat_predictor).__name__
            print(f"{predictor_type}: Trained successfully")

    # Build day context from CSV for features that don't change intraday
    test_df = load_csv_data(ticker, test_date)
    day_ctx = None
    prev_close_val = None

    if test_df is not None and not test_df.empty:
        day_ctx = _build_day_context(ticker, test_date, test_df)
        if day_ctx:
            prev_close_val = day_ctx.prev_close

    print(f"\nStarting live loop (interval: {interval}s)...")

    try:
        while True:
            now_et = datetime.now(ET_TZ)

            # Check market hours
            if now_et.weekday() >= 5:
                print(f"\rMarket closed (weekend). Waiting...", end="", flush=True)
                await asyncio.sleep(60)
                continue

            mkt_open = now_et.replace(hour=9, minute=30, second=0, microsecond=0)
            mkt_close = now_et.replace(hour=16, minute=0, second=0, microsecond=0)
            if now_et < mkt_open or now_et > mkt_close:
                print(f"\rMarket closed ({now_et.strftime('%H:%M ET')}). Waiting...", end="", flush=True)
                await asyncio.sleep(60)
                continue

            # Fetch current price from QuestDB
            try:
                data = await db.get_latest_price_with_data(db_ticker)
                if data is None or data.get('price') is None:
                    print(f"\rNo price data from QuestDB. Retrying...", end="", flush=True)
                    await asyncio.sleep(10)
                    continue

                current_price = data['price']
                source = data.get('source', 'unknown')
            except Exception as e:
                print(f"\rQuestDB error: {e}. Retrying...", end="", flush=True)
                await asyncio.sleep(10)
                continue

            # Fetch previous close from QuestDB if not already set
            if prev_close_val is None:
                try:
                    prev_data = await db.get_previous_close_prices([db_ticker])
                    prev_close_val = prev_data.get(db_ticker)
                except Exception:
                    pass

            if prev_close_val is None:
                print(f"\rNo previous close available. Retrying...", end="", flush=True)
                await asyncio.sleep(10)
                continue

            # Fetch VIX1D
            try:
                vix_data = await db.get_latest_price_with_data('I:VIX1D')
                if vix_data and vix_data.get('price'):
                    if day_ctx:
                        day_ctx.vix1d = vix_data['price']
            except Exception:
                pass

            # Build a minimal DayContext if we don't have one
            if day_ctx is None:
                day_ctx = DayContext(
                    prev_close=prev_close_val,
                    day_open=current_price,
                    vix1d=15.0,
                )

            time_label = _find_nearest_time_label(now_et.hour, now_et.minute)

            # Compute intraday vol factor using latest CSV data
            ivol_factor = 1.0
            if test_df is not None and not test_df.empty:
                ivol_factor = get_intraday_vol_factor(
                    ticker, test_date, time_label, test_df, train_dates_sorted,
                )

            pred = make_unified_prediction(
                pct_df=pct_df,
                predictor=stat_predictor,
                ticker=ticker,
                current_price=current_price,
                prev_close=prev_close_val,
                current_time=now_et,
                time_label=time_label,
                day_ctx=day_ctx,
                day_high=current_price,
                day_low=current_price,
                train_dates=pct_train_dates,
                current_vol=current_vol,
                vol_scale=vol_scale,
                data_source=f"QuestDB {source}",
                intraday_vol_factor=ivol_factor,
            )

            if pred:
                print_live_display(pred, interval)

            await asyncio.sleep(interval)

    except KeyboardInterrupt:
        print("\n\nLive mode stopped.")
