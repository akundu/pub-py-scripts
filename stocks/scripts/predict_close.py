#!/usr/bin/env python3
"""
Predict or train close-price models for NDX / SPX.

Usage:
    # Predict (default mode)
    python scripts/predict_close.py NDX                         # 0DTE prediction
    python scripts/predict_close.py NDX --days-ahead 5          # Multi-day prediction
    python scripts/predict_close.py predict NDX --days-ahead 5  # Explicit subcommand

    # Train
    python scripts/predict_close.py train NDX                              # 0DTE only
    python scripts/predict_close.py train NDX --max-dte 20                 # 0DTE + multi-day
    python scripts/predict_close.py train NDX --clear-cache --clear-models # Full clean retrain
"""

import sys
import os
from pathlib import Path

# Add parent directory to path FIRST, before any local imports
# This allows both scripts.xxx imports and common.xxx imports to work
sys.path.insert(0, str(Path(__file__).parent.parent))

import asyncio
import warnings
import pickle
import json
import numpy as np
warnings.filterwarnings('ignore')

from datetime import datetime, timedelta, date, timezone
from scripts.close_predictor.models import ET_TZ, LGBM_BAND_WIDTH_SCALE
from scripts.close_predictor.live import _build_day_context, _find_nearest_time_label
from scripts.close_predictor.prediction import _train_statistical, make_unified_prediction
from scripts.close_predictor.features import get_intraday_vol_factor
from scripts.percentile_range_backtest import collect_all_data
from scripts.csv_prediction_backtest import get_available_dates, load_csv_data, get_vix1d_at_time, get_historical_context, DayContext, append_today_from_questdb, build_training_data
from common.stock_db import get_stock_db
from common.market_hours import is_market_hours

# Model version - increment when model code changes to invalidate cache
MODEL_VERSION = "1.0.0"
CACHE_DIR = Path(".cache")


def _us_market_holidays(year: int) -> set:
    """Return a set of US market holiday dates for the given year.

    Covers NYSE holidays: New Year's, MLK Day, Presidents Day, Good Friday,
    Memorial Day, Juneteenth, Independence Day, Labor Day, Thanksgiving, Christmas.
    """
    from datetime import date as _date
    holidays = set()

    def nth_weekday(year, month, weekday, n):
        """nth occurrence of weekday (0=Mon) in given month."""
        d = _date(year, month, 1)
        # Move to first occurrence of weekday
        diff = (weekday - d.weekday()) % 7
        d = d + timedelta(days=diff)
        return d + timedelta(weeks=n - 1)

    def last_weekday(year, month, weekday):
        """Last occurrence of weekday in given month."""
        import calendar
        last_day = _date(year, month, calendar.monthrange(year, month)[1])
        diff = (last_day.weekday() - weekday) % 7
        return last_day - timedelta(days=diff)

    def observed(d):
        """If holiday falls on weekend, observe on adjacent weekday."""
        if d.weekday() == 5:  # Saturday -> Friday
            return d - timedelta(days=1)
        if d.weekday() == 6:  # Sunday -> Monday
            return d + timedelta(days=1)
        return d

    # New Year's Day (Jan 1)
    holidays.add(observed(_date(year, 1, 1)))
    # MLK Day (3rd Monday in January)
    holidays.add(nth_weekday(year, 1, 0, 3))
    # Presidents Day (3rd Monday in February)
    holidays.add(nth_weekday(year, 2, 0, 3))
    # Good Friday (2 days before Easter - approximated)
    # Easter algorithm (Anonymous Gregorian)
    a = year % 19; b = year // 100; c = year % 100
    d2 = (b - b // 4 - (b - (b + 8) // 25 + 1) // 3 + 19 * a + 15) % 30
    e = (32 + 2 * (b % 4) + 2 * (c // 4) - d2 - (c % 4)) % 7
    f = d2 + e - 7 * ((a + 11 * d2 + 22 * e) // 451) + 114
    easter = _date(year, f // 31, f % 31 + 1)
    holidays.add(easter - timedelta(days=2))  # Good Friday
    # Memorial Day (last Monday in May)
    holidays.add(last_weekday(year, 5, 0))
    # Juneteenth (June 19, from 2022)
    if year >= 2022:
        holidays.add(observed(_date(year, 6, 19)))
    # Independence Day (July 4)
    holidays.add(observed(_date(year, 7, 4)))
    # Labor Day (1st Monday in September)
    holidays.add(nth_weekday(year, 9, 0, 1))
    # Thanksgiving (4th Thursday in November)
    holidays.add(nth_weekday(year, 11, 3, 4))
    # Christmas (Dec 25)
    holidays.add(observed(_date(year, 12, 25)))

    return holidays


def get_last_trading_day(current_date: date) -> date:
    """Get the last trading day (Mon-Fri, excluding US market holidays) before current_date.

    Args:
        current_date: The reference date

    Returns:
        The last trading day (excludes weekends and major US market holidays)
    """
    # Build holiday set for current and previous year (to handle Jan 1 edge case)
    holidays = _us_market_holidays(current_date.year) | _us_market_holidays(current_date.year - 1)

    prev_date = current_date - timedelta(days=1)

    # Skip weekends and holidays
    while prev_date.weekday() >= 5 or prev_date in holidays:
        prev_date = prev_date - timedelta(days=1)

    return prev_date


def get_nth_trading_day(start_date: date, n: int) -> date:
    """Get the date that is N trading days after start_date.

    Args:
        start_date: The starting date (typically today)
        n: Number of trading days to advance

    Returns:
        The date N trading days after start_date (excluding weekends and US market holidays)
    """
    holidays = _us_market_holidays(start_date.year) | _us_market_holidays(start_date.year + 1)

    current = start_date
    count = 0
    while count < n:
        current = current + timedelta(days=1)
        if current.weekday() < 5 and current not in holidays:
            count += 1

    return current


def get_cache_path(ticker: str, training_date: str, lookback: int) -> tuple[Path, Path]:
    """Get paths for model cache and metadata."""
    CACHE_DIR.mkdir(exist_ok=True)
    base_name = f"lgbm_predictor_{ticker}_{training_date}_{lookback}"
    model_path = CACHE_DIR / f"{base_name}.pkl"
    meta_path = CACHE_DIR / f"{base_name}.json"
    return model_path, meta_path


def save_model_cache(predictor, ticker: str, training_date: str, lookback: int, pct_df, training_approach: str = "UNKNOWN"):
    """Save trained model and metadata to cache."""
    model_path, meta_path = get_cache_path(ticker, training_date, lookback)

    try:
        # Save model
        with open(model_path, 'wb') as f:
            pickle.dump(predictor, f)

        # Save metadata
        metadata = {
            'ticker': ticker,
            'training_date': training_date,
            'lookback': lookback,
            'model_version': MODEL_VERSION,
            'model_type': type(predictor).__name__,
            'band_width_scale': LGBM_BAND_WIDTH_SCALE,
            'cached_at': datetime.now(ET_TZ).isoformat(),
            'pct_df_shape': pct_df.shape if pct_df is not None else None,
            'training_approach': training_approach,
        }
        with open(meta_path, 'w') as f:
            json.dump(metadata, f, indent=2)

        return True
    except Exception as e:
        print(f"⚠️  Failed to cache model: {e}")
        return False


def load_model_cache(ticker: str, training_date: str, lookback: int, force_retrain: bool = False):
    """Load cached model if valid, otherwise return None."""
    if force_retrain:
        return None, None

    model_path, meta_path = get_cache_path(ticker, training_date, lookback)

    if not model_path.exists() or not meta_path.exists():
        return None, None

    try:
        # Load metadata
        with open(meta_path, 'r') as f:
            metadata = json.load(f)

        # Validate metadata
        if metadata.get('model_version') != MODEL_VERSION:
            print(f"⚠️  Cache invalid: model version mismatch (cached: {metadata.get('model_version')}, current: {MODEL_VERSION})")
            return None, None

        if metadata.get('band_width_scale') != LGBM_BAND_WIDTH_SCALE:
            print(f"⚠️  Cache invalid: band width changed (cached: {metadata.get('band_width_scale')}, current: {LGBM_BAND_WIDTH_SCALE})")
            return None, None

        # Load model
        with open(model_path, 'rb') as f:
            predictor = pickle.load(f)

        cached_at = metadata.get('cached_at', 'unknown')
        return predictor, metadata

    except Exception as e:
        print(f"⚠️  Failed to load cached model: {e}")
        return None, None


def clear_old_caches(ticker: str, current_training_date: str, lookback: int):
    """Remove old cached models for this ticker."""
    try:
        pattern = f"lgbm_predictor_{ticker}_*_{lookback}.*"
        for cache_file in CACHE_DIR.glob(pattern):
            # Keep current cache, remove others
            if current_training_date not in cache_file.name:
                cache_file.unlink()
                print(f"Removed old cache: {cache_file.name}")
    except Exception as e:
        print(f"⚠️  Failed to clear old caches: {e}")


def clear_cache_files(ticker):
    """Clear 0DTE .cache/lgbm_predictor_{ticker}_* files."""
    cleared = 0
    if CACHE_DIR.exists():
        for f in CACHE_DIR.glob(f"lgbm_predictor_{ticker}_*"):
            f.unlink()
            cleared += 1
    if cleared:
        print(f"✓ Cleared {cleared} cache files for {ticker}")
    else:
        print(f"ℹ️  No cache files found for {ticker}")
    return cleared


def clear_model_files(ticker, output_dir=Path('models/production')):
    """Clear models/production/{ticker}_latest/ directory."""
    latest_dir = output_dir / f"{ticker}_latest"
    cleared = 0
    if latest_dir.exists():
        for f in latest_dir.iterdir():
            if f.is_file():
                f.unlink()
                cleared += 1
    if cleared:
        print(f"✓ Cleared {cleared} model files from {latest_dir}")
    else:
        print(f"ℹ️  No model files found in {latest_dir}")
    return cleared


async def train_0dte_model(ticker, lookback=120, db_config=None):
    """Train 0DTE LightGBM model and save to cache. Returns the predictor."""
    print(f"\n{'='*80}")
    print(f"TRAINING 0DTE MODEL - {ticker}")
    print(f"{'='*80}\n")

    # Load CSV data
    all_dates = get_available_dates(ticker, lookback + 20)
    if len(all_dates) < lookback:
        print(f"❌ Not enough training data. Need {lookback} days, have {len(all_dates)}")
        return None

    training_date = all_dates[-1]
    print(f"Training data through: {training_date}")
    print(f"Building training data from {lookback} days of CSV files...")

    train_df = build_training_data(ticker, training_date, lookback)
    if train_df.empty or len(train_df) < 50:
        print("❌ Insufficient training data from CSV")
        return None

    print(f"✓ Loaded {len(train_df)} training samples from CSV")

    from scripts.strategy_utils.close_predictor import LGBMClosePredictor
    from scripts.close_predictor.models import (
        LGBM_N_ESTIMATORS, LGBM_LEARNING_RATE,
        LGBM_MAX_DEPTH, LGBM_MIN_CHILD_SAMPLES,
    )

    stat_predictor = LGBMClosePredictor(
        n_estimators=LGBM_N_ESTIMATORS,
        learning_rate=LGBM_LEARNING_RATE,
        max_depth=LGBM_MAX_DEPTH,
        min_child_samples=LGBM_MIN_CHILD_SAMPLES,
        band_width_scale=LGBM_BAND_WIDTH_SCALE,
        use_fallback=True,
    )

    # Clean training data
    initial_rows = len(train_df)
    train_df_clean = train_df.dropna()
    rows_dropped = initial_rows - len(train_df_clean)
    if rows_dropped > 0:
        print(f"⚠️  Dropped {rows_dropped} rows with NaN values ({rows_dropped/initial_rows*100:.1f}%)")

    if len(train_df_clean) < 100:
        print(f"❌ Insufficient training data after cleaning: {len(train_df_clean)} rows")
        return None

    stat_predictor.fit(train_df_clean)
    print(f"✓ Model trained: {type(stat_predictor).__name__}")

    # Save to cache
    pct_df = collect_all_data(ticker, all_dates)
    if save_model_cache(stat_predictor, ticker, training_date, lookback, pct_df, "STATIC"):
        print(f"✓ Model cached")
        clear_old_caches(ticker, training_date, lookback)

    return stat_predictor


def _load_vix_data(vix_ticker, all_dates):
    """Load VIX/VIX1D historical data for multi-day training."""
    vix_data = {}
    for date_str in all_dates:
        df = load_csv_data(vix_ticker, date_str)
        if df is not None and not df.empty:
            vix_data[date_str] = df
    return vix_data


def train_multi_day_models(ticker, train_days=250, validate_days=30,
                           max_dte=20, output_dir=Path('models/production')):
    """Train multi-day LightGBM ensemble (DTE 1-N). Returns validation stats."""
    import pandas as pd
    from scripts.close_predictor.multi_day_features import compute_market_context, MarketContext
    from scripts.close_predictor.multi_day_lgbm import MultiDayEnsemble

    print(f"\n{'='*80}")
    print(f"MULTI-DAY MODEL RETRAINING")
    print(f"{'='*80}\n")
    print(f"Ticker:           {ticker}")
    print(f"Training window:  {train_days} days")
    print(f"Validation:       {validate_days} days")
    print(f"Max DTE:          {max_dte}")
    print(f"Timestamp:        {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    # Load historical data
    print("Loading historical data...")
    total_lookback = train_days + validate_days + max_dte + 20
    all_dates = get_available_dates(ticker, total_lookback)

    data_by_date = {}
    for date_str in all_dates:
        df = load_csv_data(ticker, date_str)
        if df is not None and not df.empty:
            data_by_date[date_str] = df

    print(f"✓ Loaded {len(all_dates)} trading days")

    # Load VIX and VIX1D data
    print("Loading VIX and VIX1D data...")
    vix_data_by_date = _load_vix_data('VIX', all_dates)
    vix1d_data_by_date = _load_vix_data('VIX1D', all_dates)
    print(f"✓ Loaded VIX data for {len(vix_data_by_date)} days, VIX1D for {len(vix1d_data_by_date)} days")

    # Determine train/validate split
    validate_start_idx = len(all_dates) - validate_days - max_dte
    train_end_idx = validate_start_idx
    train_start_idx = max(0, train_end_idx - train_days)

    train_dates = all_dates[train_start_idx:train_end_idx]
    validate_dates = all_dates[validate_start_idx:validate_start_idx + validate_days]

    print(f"\nTrain period:     {train_dates[0]} to {train_dates[-1]} ({len(train_dates)} days)")
    print(f"Validate period:  {validate_dates[0]} to {validate_dates[-1]} ({len(validate_dates)} days)")

    # Compute market contexts
    print("\nComputing market context features...")
    all_contexts = {}

    for i, date_str in enumerate(all_dates):
        if date_str not in data_by_date:
            continue

        lookback_idx = max(0, i - 60)
        history_dates = all_dates[lookback_idx:i+1]

        history_rows = []
        for d in history_dates:
            if d in data_by_date:
                df = data_by_date[d]
                history_rows.append({
                    'date': d,
                    'open': df.iloc[0]['open'] if 'open' in df.columns else df.iloc[0]['close'],
                    'close': df.iloc[-1]['close'],
                    'high': df['high'].max() if 'high' in df.columns else df.iloc[-1]['close'],
                    'low': df['low'].min() if 'low' in df.columns else df.iloc[-1]['close'],
                    'volume': df['volume'].sum() if 'volume' in df.columns else 0,
                })

        if not history_rows:
            continue

        price_history = pd.DataFrame(history_rows)
        current_price = price_history.iloc[-1]['close']
        current_date = datetime.strptime(date_str, '%Y-%m-%d').date()

        # Build VIX history
        vix_history = None
        if vix_data_by_date:
            vix_rows = []
            for d in history_dates:
                if d in vix_data_by_date:
                    vdf = vix_data_by_date[d]
                    if not vdf.empty:
                        vix_rows.append({'date': d, 'close': vdf.iloc[-1]['close']})
            if vix_rows:
                vix_history = pd.DataFrame(vix_rows)

        # Build VIX1D history
        vix1d_history = None
        if vix1d_data_by_date:
            vix1d_rows = []
            for d in history_dates:
                if d in vix1d_data_by_date:
                    v1df = vix1d_data_by_date[d]
                    if not v1df.empty:
                        vix1d_rows.append({'date': d, 'close': v1df.iloc[-1]['close']})
            if vix1d_rows:
                vix1d_history = pd.DataFrame(vix1d_rows)

        ctx = compute_market_context(
            ticker=ticker,
            current_price=current_price,
            current_date=current_date,
            price_history=price_history,
            vix_history=vix_history,
            vix1d_history=vix1d_history,
            iv_data=None,
        )
        all_contexts[date_str] = ctx

    print(f"✓ Computed contexts for {len(all_contexts)} dates")

    # Compute forward returns
    print(f"\nComputing {max_dte}-day forward returns...")
    returns_by_dte = {}

    for dte in range(1, max_dte + 1):
        returns_by_dte[dte] = {}
        for i, start_date in enumerate(all_dates):
            if i + dte >= len(all_dates):
                continue
            end_date = all_dates[i + dte]
            if start_date not in data_by_date or end_date not in data_by_date:
                continue

            start_close = data_by_date[start_date].iloc[-1]['close']
            end_close = data_by_date[end_date].iloc[-1]['close']
            forward_return = (end_close - start_close) / start_close * 100
            returns_by_dte[dte][start_date] = forward_return

    print(f"✓ Computed returns for DTE 1-{max_dte}")

    # Train ensemble
    print("\nTraining LightGBM ensemble models...")
    ensemble = MultiDayEnsemble()
    train_contexts_by_date = {d: all_contexts[d] for d in train_dates if d in all_contexts}

    ensemble_stats = ensemble.train_all(
        all_dates=train_dates,
        contexts_by_date=train_contexts_by_date,
        returns_by_dte=returns_by_dte,
        max_dte=max_dte,
    )

    # Save models
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    model_dir = output_dir / f"{ticker}_{timestamp}"
    ensemble.save_all(model_dir)

    # Also save to 'latest' directory (overwrite previous)
    latest_dir = output_dir / f"{ticker}_latest"
    ensemble.save_all(latest_dir)

    print(f"\n✓ Saved models to:")
    print(f"  - {model_dir} (timestamped)")
    print(f"  - {latest_dir} (latest)")

    # Validation summary
    print(f"\n{'='*80}")
    print("VALIDATION SUMMARY")
    print(f"{'='*80}\n")
    print(f"{'DTE':<6} {'Train RMSE':<12} {'Val RMSE':<12} {'Status':<20}")
    print(f"{'─'*60}")

    for dte in range(1, min(max_dte + 1, 21)):
        if dte in ensemble_stats:
            stats = ensemble_stats[dte]
            train_rmse = stats['train_rmse']
            val_rmse = stats['val_rmse']

            if val_rmse < 2.0:
                status = "✓ Excellent"
            elif val_rmse < 3.0:
                status = "✓ Good"
            elif val_rmse < 4.0:
                status = "⚠ Acceptable"
            else:
                status = "⚠ High Error"

            print(f"{dte:<6} {train_rmse:>9.2f}%   {val_rmse:>9.2f}%   {status}")

    # Overall assessment
    print(f"\n{'─'*60}")
    avg_val_rmse = np.mean([ensemble_stats[dte]['val_rmse'] for dte in ensemble_stats])
    max_val_rmse = max([ensemble_stats[dte]['val_rmse'] for dte in ensemble_stats])

    print(f"\nAverage Validation RMSE: {avg_val_rmse:.2f}%")
    print(f"Maximum Validation RMSE: {max_val_rmse:.2f}%")

    if max_val_rmse < 3.0:
        print("\n✓ Models are PRODUCTION READY")
    elif max_val_rmse < 4.0:
        print("\n⚠ Models are acceptable but should be monitored")
    else:
        print("\n⚠ Models show high error - consider retraining with different parameters")

    print(f"\n{'='*80}\n")

    # Save training metadata
    metadata = {
        'ticker': ticker,
        'train_period': f"{train_dates[0]} to {train_dates[-1]}",
        'train_days': len(train_dates),
        'validate_period': f"{validate_dates[0]} to {validate_dates[-1]}",
        'timestamp': timestamp,
        'avg_val_rmse': float(avg_val_rmse),
        'max_val_rmse': float(max_val_rmse),
        'validation_stats': {dte: {
            'train_rmse': float(stats['train_rmse']),
            'val_rmse': float(stats['val_rmse']),
        } for dte, stats in ensemble_stats.items()}
    }

    # Save to timestamped directory
    metadata_file = model_dir / "training_metadata.json"
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)

    # Also save to latest directory
    latest_metadata_file = latest_dir / "training_metadata.json"
    with open(latest_metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"✓ Saved training metadata to:")
    print(f"  - {metadata_file}")
    print(f"  - {latest_metadata_file}")

    # Clear prediction cache to force refresh
    prediction_cache_dir = Path(__file__).parent.parent / ".prediction_cache"
    if prediction_cache_dir.exists():
        import glob as glob_mod
        cache_pattern = str(prediction_cache_dir / f"future_{ticker}_*.json")
        cleared_count = 0
        for cache_file in glob_mod.glob(cache_pattern):
            try:
                Path(cache_file).unlink()
                cleared_count += 1
            except Exception:
                pass

        if cleared_count > 0:
            print(f"\n✓ Cleared {cleared_count} cached prediction(s) for {ticker}")

    return ensemble_stats


async def predict_future_close(ticker: str, days_ahead: int, current_price: float, lookback: int = 120):
    """Predict close price N trading days in the future using historical patterns.

    Args:
        ticker: Ticker symbol (NDX or SPX)
        days_ahead: Number of trading days ahead to predict
        current_price: Current/starting price
        lookback: Number of historical days to analyze

    Returns:
        Dictionary with prediction results
    """
    print(f"\n{'='*80}")
    print(f"MULTI-DAY FORECAST: {days_ahead} TRADING DAYS AHEAD")
    print(f"{'='*80}\n")

    # Load historical data
    all_dates = get_available_dates(ticker, lookback + days_ahead + 20)
    if len(all_dates) < lookback:
        print(f"❌ Not enough data. Need {lookback} days, have {len(all_dates)}")
        return None

    print(f"Analyzing {len(all_dates)} days of historical data...")

    # Calculate N-day returns from historical data
    n_day_returns = []
    for i in range(len(all_dates) - days_ahead):
        start_date = all_dates[i]
        end_date = all_dates[i + days_ahead]

        # Get closing prices
        start_df = load_csv_data(ticker, start_date)
        end_df = load_csv_data(ticker, end_date)

        if start_df is not None and end_df is not None and not start_df.empty and not end_df.empty:
            start_close = start_df.iloc[-1]['close']
            end_close = end_df.iloc[-1]['close']
            pct_return = (end_close - start_close) / start_close * 100
            n_day_returns.append(pct_return)

    if len(n_day_returns) < 50:
        print(f"❌ Insufficient data for {days_ahead}-day prediction (need 50 samples, have {len(n_day_returns)})")
        return None

    n_day_returns = np.array(n_day_returns)
    print(f"✓ Found {len(n_day_returns)} historical {days_ahead}-day periods\n")

    # Calculate percentile bands
    percentiles = {
        'P75': (np.percentile(n_day_returns, 12.5), np.percentile(n_day_returns, 87.5)),
        'P80': (np.percentile(n_day_returns, 10.0), np.percentile(n_day_returns, 90.0)),
        'P85': (np.percentile(n_day_returns, 7.5), np.percentile(n_day_returns, 92.5)),
        'P90': (np.percentile(n_day_returns, 5.0), np.percentile(n_day_returns, 95.0)),
        'P95': (np.percentile(n_day_returns, 2.5), np.percentile(n_day_returns, 97.5)),
        'P97': (np.percentile(n_day_returns, 1.5), np.percentile(n_day_returns, 98.5)),
        'P98': (np.percentile(n_day_returns, 1.0), np.percentile(n_day_returns, 99.0)),
        'P99': (np.percentile(n_day_returns, 0.5), np.percentile(n_day_returns, 99.5)),
    }

    # Calculate statistics
    mean_return = np.mean(n_day_returns)
    median_return = np.median(n_day_returns)
    std_return = np.std(n_day_returns)

    # Convert to price predictions
    expected_price = current_price * (1 + mean_return / 100)
    median_price = current_price * (1 + median_return / 100)

    # Compute target date
    target_date = get_nth_trading_day(date.today(), days_ahead)
    target_date_str = target_date.strftime("%A, %B %d, %Y")

    # Display results
    print(f"Starting Price:     ${current_price:,.2f}")
    print(f"Target:             {days_ahead} trading days ahead")
    print(f"Target Date:        {target_date_str}\n")

    print(f"{'='*80}")
    print(f"FORECAST STATISTICS ({days_ahead}-day returns)")
    print(f"{'='*80}")
    print(f"Mean Return:        {mean_return:+.2f}%")
    print(f"Median Return:      {median_return:+.2f}%")
    print(f"Std Deviation:      {std_return:.2f}%")
    print(f"Best {days_ahead}-day:        {np.max(n_day_returns):+.2f}%")
    print(f"Worst {days_ahead}-day:       {np.min(n_day_returns):+.2f}%\n")

    print(f"{'='*80}")
    print(f"FORECAST PRICES")
    print(f"{'='*80}")
    print(f"Expected Close (mean):   ${expected_price:,.2f} ({mean_return:+.2f}%)")
    print(f"Median Close:            ${median_price:,.2f} ({median_return:+.2f}%)\n")

    print(f"{'='*80}")
    print(f"CONFIDENCE BANDS")
    print(f"{'='*80}\n")

    for band_name in ['P75', 'P80', 'P85', 'P90', 'P95', 'P97', 'P98', 'P99']:
        lo_pct, hi_pct = percentiles[band_name]
        lo_price = current_price * (1 + lo_pct / 100)
        hi_price = current_price * (1 + hi_pct / 100)
        width_pct = hi_pct - lo_pct

        print(f"{band_name} Band ({band_name[1:]}% confidence):")
        print(f"  Lower:  ${lo_price:,.2f} ({lo_pct:+.2f}%)")
        print(f"  Upper:  ${hi_price:,.2f} ({hi_pct:+.2f}%)")
        print(f"  Width:  {width_pct:.2f}%\n")

    # Recommendation
    _, hi_p95 = percentiles['P95']
    lo_p95, _ = percentiles['P95']

    print(f"{'='*80}")
    print(f"FORECAST SUMMARY")
    print(f"{'='*80}")
    print(f"Expected {days_ahead}-day close: ${expected_price:,.2f}")
    print(f"P95 Range: ${current_price * (1 + lo_p95/100):,.2f} - ${current_price * (1 + hi_p95/100):,.2f}")
    print(f"{'='*80}\n")

    return {
        'days_ahead': days_ahead,
        'target_date': target_date.isoformat(),
        'target_date_str': target_date_str,
        'current_price': current_price,
        'expected_price': expected_price,
        'median_price': median_price,
        'mean_return': mean_return,
        'median_return': median_return,
        'std_return': std_return,
        'percentiles': percentiles,
        'n_samples': len(n_day_returns),
    }


async def _predict_future_close_unified(ticker: str, days_ahead: int, lookback: int, db_config: Optional[str], use_time_decay: bool = True, use_intraday_vol: bool = True) -> Optional['UnifiedPrediction']:
    """Multi-day ahead prediction using Ensemble Combined method (RECOMMENDED).

    Computes predictions using all 4 methods and displays comparison:
    1. Baseline: Simple percentile distribution
    2. Conditional: Feature-weighted distribution
    3. Ensemble: LightGBM ML model
    4. Ensemble Combined: Conservative blend (RECOMMENDED)

    Args:
        ticker: Ticker symbol (NDX or SPX)
        days_ahead: Number of trading days ahead to predict
        lookback: Number of historical days to analyze
        db_config: QuestDB connection string
        use_time_decay: Enable time decay factor (default True)
        use_intraday_vol: Enable intraday volatility scaling (default True)

    Returns:
        UnifiedPrediction with ensemble_combined bands as primary prediction
    """
    from scripts.close_predictor.models import UnifiedPrediction, UnifiedBand
    from scripts.close_predictor.bands import map_percentile_to_bands
    from scripts.close_predictor.multi_day_features import compute_market_context
    from scripts.close_predictor.multi_day_predictor import predict_with_conditional_distribution
    from scripts.close_predictor.multi_day_lgbm import MultiDayLGBMPredictor
    from pathlib import Path

    print(f"\n{'='*80}")
    print(f"MULTI-DAY PREDICTION: {days_ahead} trading days ahead")
    print(f"{'='*80}\n")

    # Get current price from QuestDB, fall back to CSV if needed
    current_price = None
    current_date_str = None
    try:
        from common.questdb_db import StockQuestDB
        db = StockQuestDB(db_config, logger=None)
        current_price = await db.get_latest_price(f"I:{ticker}")
        if current_price:
            print(f"✓ Current price from QuestDB: ${current_price:,.2f}")
    except Exception as e:
        print(f"⚠️  QuestDB failed: {e}")

    # Fall back to CSV if QuestDB didn't work
    if current_price is None:
        print(f"⚠️  No QuestDB price available, using latest CSV close...")
        all_dates_temp = get_available_dates(ticker, 5)
        if not all_dates_temp:
            print(f"❌ No historical data available for {ticker}")
            return None

        latest_date = all_dates_temp[-1]
        latest_df = load_csv_data(ticker, latest_date)

        if latest_df is None or latest_df.empty:
            print(f"❌ Could not load data for {latest_date}")
            return None

        current_price = latest_df.iloc[-1]['close']
        current_date_str = latest_date
        print(f"✓ Using latest close from CSV: ${current_price:,.2f} (date: {latest_date})")

    if current_price is None or current_price <= 0:
        print(f"❌ Could not get valid price for {ticker}")
        return None

    # Load historical data
    all_dates = get_available_dates(ticker, lookback + days_ahead + 60)
    if len(all_dates) < lookback:
        print(f"❌ Not enough data. Need {lookback} days, have {len(all_dates)}")
        return None

    # Get the previous close (last CSV close) as static reference for baseline bands
    prev_close_for_baseline = None
    if all_dates:
        last_csv_df = load_csv_data(ticker, all_dates[-1])
        if last_csv_df is not None and not last_csv_df.empty:
            prev_close_for_baseline = last_csv_df.iloc[-1]['close']
            print(f"✓ Previous close (last CSV): ${prev_close_for_baseline:,.2f} (date: {all_dates[-1]})")
    if prev_close_for_baseline is None:
        prev_close_for_baseline = current_price  # Fallback

    # Calculate N-day returns from historical data
    n_day_returns = []
    data_by_date = {}

    for i in range(len(all_dates) - days_ahead):
        start_date = all_dates[i]
        end_date = all_dates[i + days_ahead]

        start_df = load_csv_data(ticker, start_date)
        end_df = load_csv_data(ticker, end_date)

        if start_df is not None and not start_df.empty:
            data_by_date[start_date] = start_df

        if start_df is not None and end_df is not None:
            start_close = start_df.iloc[-1]['close']
            end_close = end_df.iloc[-1]['close']
            pct_return = (end_close - start_close) / start_close * 100
            n_day_returns.append(pct_return)

    # TIER 1 FEATURES: Load VIX and VIX1D data for improved predictions
    print("Loading VIX and VIX1D data for TIER 1 features...")
    vix_data_by_date = {}
    vix1d_data_by_date = {}
    for date_str in all_dates:
        vix_df = load_csv_data('VIX', date_str)
        if vix_df is not None and not vix_df.empty:
            vix_data_by_date[date_str] = vix_df

        vix1d_df = load_csv_data('VIX1D', date_str)
        if vix1d_df is not None and not vix1d_df.empty:
            vix1d_data_by_date[date_str] = vix1d_df
    print(f"✓ Loaded VIX data for {len(vix_data_by_date)} days, VIX1D for {len(vix1d_data_by_date)} days")

    if len(n_day_returns) < 50:
        print(f"❌ Insufficient data for {days_ahead}-day prediction (need 50 samples, have {len(n_day_returns)})")
        return None

    n_day_returns_array = np.array(n_day_returns)
    print(f"✓ Analyzed {len(n_day_returns)} historical {days_ahead}-day periods\n")

    # Compute target date
    target_date = get_nth_trading_day(date.today(), days_ahead)

    # --- METHOD 1: Baseline (simple percentile) --- uses prev_close as static reference
    baseline_bands = map_percentile_to_bands(n_day_returns_array, prev_close_for_baseline)

    # --- METHOD 2 & 4: Conditional + Ensemble Combined (requires market context) ---
    conditional_bands = {}
    ensemble_bands = {}
    ensemble_combined_bands = {}

    # Compute current market context
    current_date = datetime.strptime(current_date_str, '%Y-%m-%d').date() if current_date_str else date.today()

    # Build price history for context calculation
    lookback_idx = max(0, len(all_dates) - 60)
    history_dates = all_dates[lookback_idx:]

    history_rows = []
    for d in history_dates:
        if d in data_by_date:
            df = data_by_date[d]
            history_rows.append({
                'date': d,
                'close': df.iloc[-1]['close'],
                'high': df['high'].max() if 'high' in df.columns else df.iloc[-1]['close'],
                'low': df['low'].min() if 'low' in df.columns else df.iloc[-1]['close'],
                'volume': df['volume'].sum() if 'volume' in df.columns else 0,
            })

    if history_rows:
        import pandas as pd
        price_history = pd.DataFrame(history_rows)

        # Build VIX history for TIER 1 features
        vix_history = None
        if vix_data_by_date:
            vix_rows = []
            for d in history_dates:
                if d in vix_data_by_date:
                    vdf = vix_data_by_date[d]
                    if not vdf.empty:
                        vix_rows.append({'date': d, 'close': vdf.iloc[-1]['close']})
            if vix_rows:
                vix_history = pd.DataFrame(vix_rows)

        # Build VIX1D history for TIER 1 features
        vix1d_history = None
        if vix1d_data_by_date:
            vix1d_rows = []
            for d in history_dates:
                if d in vix1d_data_by_date:
                    v1df = vix1d_data_by_date[d]
                    if not v1df.empty:
                        vix1d_rows.append({'date': d, 'close': v1df.iloc[-1]['close']})
            if vix1d_rows:
                vix1d_history = pd.DataFrame(vix1d_rows)

        # Get IV data for TIER 1 features
        iv_data = None
        try:
            from common.questdb_db import StockQuestDB
            from common.financial_data import get_financial_info
            db = StockQuestDB(db_config, logger=None)
            fin_info = await get_financial_info(ticker, db, include_iv_analysis=True)
            if fin_info:
                iv_data = {
                    'iv_rank': fin_info.get('iv_rank'),
                    'iv_90d_rank': fin_info.get('iv_90d_rank'),
                    'iv_30d': fin_info.get('iv_30d'),
                    'iv_90d': fin_info.get('iv_90d'),
                }
                print(f"✓ Loaded IV data: rank={iv_data.get('iv_rank')}, 30d={iv_data.get('iv_30d')}%, 90d={iv_data.get('iv_90d')}%")
        except Exception as e:
            print(f"⚠️  Could not load IV data: {e}")

        try:
            current_context = compute_market_context(
                ticker=ticker,
                current_price=current_price,
                current_date=current_date,
                price_history=price_history,
                vix_history=vix_history,
                vix1d_history=vix1d_history,
                iv_data=iv_data,
            )

            # Build historical contexts (use recent subset for performance)
            from scripts.close_predictor.multi_day_features import compute_historical_contexts
            train_dates = all_dates[max(0, len(all_dates) - lookback):len(all_dates) - days_ahead]
            historical_contexts = compute_historical_contexts(
                ticker=ticker,
                all_dates=train_dates,
                price_data_by_date=data_by_date,
                vix_data_by_date=vix_data_by_date,
                vix1d_data_by_date=vix1d_data_by_date,
                lookback_days=60,
            )

            # Get historical vols and returns for those dates
            train_returns = []
            train_vols = []
            for td in train_dates:
                if td in data_by_date:
                    # Find corresponding return
                    idx = all_dates.index(td) if td in all_dates else -1
                    if idx >= 0 and idx < len(all_dates) - days_ahead:
                        ret_idx = idx - (len(all_dates) - lookback - days_ahead)
                        if 0 <= ret_idx < len(n_day_returns):
                            train_returns.append(n_day_returns[ret_idx])
                        else:
                            train_returns.append(n_day_returns[min(ret_idx, len(n_day_returns)-1)])
                    # Get vol from context
                    ctx_idx = train_dates.index(td)
                    if ctx_idx < len(historical_contexts):
                        train_vols.append(historical_contexts[ctx_idx].realized_vol_5d)

            # Compute time decay and intraday volatility factors
            effective_days_ahead = float(days_ahead)
            intraday_vol_factor = 1.0

            if use_time_decay or use_intraday_vol:
                from scripts.close_predictor.models import ET_TZ
                current_time = datetime.now(ET_TZ)
                market_open_hour = 9
                market_open_minute = 30
                market_close_hour = 16
                market_close_minute = 0

                # Compute hours remaining in trading day
                hours_to_close = 0.0
                if current_time.hour < market_open_hour or (current_time.hour == market_open_hour and current_time.minute < market_open_minute):
                    # Before market open - full day ahead
                    hours_to_close = 6.5
                elif current_time.hour < market_close_hour or (current_time.hour == market_close_hour and current_time.minute == 0):
                    # During market hours
                    hours_to_close = (market_close_hour - current_time.hour) + (market_close_minute - current_time.minute) / 60.0
                else:
                    # After market close
                    hours_to_close = 0.0

                # Apply time decay if enabled
                if use_time_decay:
                    # As trading day progresses, effective DTE decreases
                    # Example: 5-day prediction at 3:50 PM → ~4.03 effective days
                    fraction_of_day_remaining = hours_to_close / 6.5 if hours_to_close > 0 else 0.0
                    effective_days_ahead = days_ahead - (1.0 - fraction_of_day_remaining)
                    # Don't go below 0.5 days (minimum uncertainty)
                    effective_days_ahead = max(0.5, effective_days_ahead)
                    print(f"Time decay: {hours_to_close:.2f} hours to close → effective DTE = {effective_days_ahead:.2f} (vs {days_ahead} nominal)")

                # Apply intraday volatility scaling if enabled
                if use_intraday_vol:
                    # Get today's high/low from latest data
                    today_high = current_price
                    today_low = current_price
                    prev_close_price = None

                    # Try to get from history_rows if available
                    if history_rows:
                        latest_row = history_rows[-1]
                        today_high = max(latest_row.get('high', current_price), current_price)
                        today_low = min(latest_row.get('low', current_price), current_price)

                        # Get previous close for intraday range calculation
                        if len(history_rows) >= 2:
                            prev_close_price = history_rows[-2]['close']

                    if prev_close_price and prev_close_price > 0:
                        intraday_range_pct = (today_high - today_low) / prev_close_price * 100

                        # Scale volatility factor based on intraday range
                        # Normal day: 0.5-1.5% range → factor = 1.0
                        # Volatile day: 3%+ range → factor = 1.3-1.5
                        if intraday_range_pct > 2.0:
                            intraday_vol_factor = 1.0 + (intraday_range_pct - 1.5) / 10.0
                            intraday_vol_factor = min(1.5, intraday_vol_factor)  # Cap at 1.5x
                            print(f"Intraday vol: range={intraday_range_pct:.2f}% → vol factor = {intraday_vol_factor:.2f}x")
                        else:
                            print(f"Intraday vol: range={intraday_range_pct:.2f}% (normal) → vol factor = 1.00x")

            # Conditional prediction
            if len(train_returns) >= 50 and len(train_returns) == len(historical_contexts[:len(train_returns)]):
                conditional_bands = predict_with_conditional_distribution(
                    ticker=ticker,
                    days_ahead=days_ahead,
                    current_price=current_price,
                    current_context=current_context,
                    n_day_returns=train_returns,
                    historical_contexts=historical_contexts[:len(train_returns)],
                    historical_realized_vols=train_vols if len(train_vols) == len(train_returns) else None,
                    use_weighting=True,
                    use_regime_filter=True,
                    use_vol_scaling=True,
                    effective_days_ahead=effective_days_ahead,
                    intraday_vol_factor=intraday_vol_factor,
                )

        except Exception as e:
            print(f"⚠️  Conditional prediction failed: {e}")
            current_context = None

    # --- METHOD 3: Ensemble (LightGBM) ---
    # Use production models (retrained monthly by automated system)
    # Priority order:
    # 1. models/production/{ticker}/lgbm_Xdte.pkl (automated retraining - ticker-specific)
    # 2. models/production/lgbm_Xdte.pkl (legacy single-ticker deployment)
    # 3. models/production/{ticker}_latest/lgbm_Xdte.pkl (old manual deployment)
    # 4. results/multi_day_backtest/models/lgbm_Xdte.pkl (backtest results)

    base_dir = Path(__file__).parent.parent

    # Try new ticker-specific automated retraining location first
    model_path = base_dir / "models" / "production" / ticker / f"lgbm_{days_ahead}dte.pkl"

    if model_path.exists():
        model_dir = base_dir / "models" / "production" / ticker
    else:
        # Fallback to legacy single-ticker location (backward compatibility)
        model_path = base_dir / "models" / "production" / f"lgbm_{days_ahead}dte.pkl"

        if model_path.exists():
            model_dir = base_dir / "models" / "production"
        else:
            # Fallback to old ticker-specific location
            model_dir = base_dir / "models" / "production" / f"{ticker}_latest"
            model_path = model_dir / f"lgbm_{days_ahead}dte.pkl"

            if not model_path.exists():
                # Final fallback to backtest models
                model_dir = base_dir / "results" / "multi_day_backtest" / "models"
                model_path = model_dir / f"lgbm_{days_ahead}dte.pkl"

    if model_path.exists() and current_context:
        try:
            predictor = MultiDayLGBMPredictor.load(model_path)
            ensemble_bands = predictor.predict_distribution(
                current_context,
                current_price,
                effective_days_ahead=effective_days_ahead if use_time_decay else None,
                intraday_vol_factor=intraday_vol_factor if use_intraday_vol else 1.0,
            )
            print(f"✓ Loaded ensemble model from: {model_dir}")
        except Exception as e:
            print(f"⚠️  Ensemble model loading failed: {e}")

    # --- METHOD 4: Ensemble Combined (conservative blend) ---
    if conditional_bands and ensemble_bands:
        for band_name in ['P95', 'P97', 'P98', 'P99', 'P100']:
            if band_name in conditional_bands and band_name in ensemble_bands:
                cb = conditional_bands[band_name]
                eb = ensemble_bands[band_name]

                # Take wider of the two (more conservative)
                lo_price = min(cb.lo_price, eb.lo_price)
                hi_price = max(cb.hi_price, eb.hi_price)
                width_pts = hi_price - lo_price
                width_pct = (hi_price - lo_price) / current_price * 100.0 if current_price else 0.0

                ensemble_combined_bands[band_name] = UnifiedBand(
                    name=band_name,
                    lo_price=lo_price,
                    hi_price=hi_price,
                    lo_pct=(lo_price - current_price) / current_price * 100.0 if current_price else 0.0,
                    hi_pct=(hi_price - current_price) / current_price * 100.0 if current_price else 0.0,
                    width_pts=width_pts,
                    width_pct=width_pct,
                    source="ensemble_combined",
                )

    # Display results
    print(f"Current Price:      ${current_price:,.2f}")
    print(f"Target Date:        {target_date.strftime('%A, %B %d, %Y')} ({days_ahead} trading days)\n")

    # SMART FALLBACK: Determine recommended method using regime detection & confidence scoring
    from scripts.close_predictor.regime_detector import RegimeDetector
    from pathlib import Path

    # Try to load regime state to check for regime changes
    regime_status = None
    regime_recommendation = None
    fallback_reason = None

    try:
        model_dir = Path(__file__).parent.parent / "models" / "production"
        cache_dir = Path(__file__).parent.parent / "models" / "regime_cache"

        # Load metadata to get training RMSE
        metadata_file = model_dir / ticker / "metadata.json"
        if metadata_file.exists():
            import json
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)

            # Create regime detector
            detector = RegimeDetector.create_for_model(
                ticker=ticker,
                days_ahead=days_ahead,
                model_metadata=metadata,
                cache_dir=cache_dir,
            )
            regime_status = detector.get_status()
            regime_recommendation = regime_status['recommended_method']

            if regime_status.get('is_regime_changed'):
                fallback_reason = regime_status.get('fallback_reason', 'Regime change detected')
    except Exception as e:
        print(f"⚠️  Could not load regime detector: {e}")
        regime_recommendation = None

    # Method selection with fallback hierarchy
    if regime_recommendation == "baseline":
        # Severe regime change - use baseline
        recommended_method = "Baseline (Simple Percentile)"
        recommended_bands = baseline_bands
        print(f"⚠️  REGIME CHANGE DETECTED: Using baseline method")
        print(f"   Reason: {fallback_reason}")

    elif regime_recommendation == "conditional" or not ensemble_bands:
        # Moderate regime change or no ensemble available - use conditional
        recommended_method = "Conditional (Feature-Weighted)"
        recommended_bands = conditional_bands if conditional_bands else baseline_bands
        if fallback_reason:
            print(f"ℹ️  Using conditional method (regime confidence)")
            print(f"   Reason: {fallback_reason}")

    elif ensemble_bands:
        # Check model confidence if ensemble is available
        try:
            # Get confidence score
            confidence_score = 0.8  # Default high confidence
            if hasattr(loaded_predictor, 'get_prediction_confidence'):
                confidence_score = loaded_predictor.get_prediction_confidence(
                    context=current_context,
                    recent_errors=None,  # Could track this in regime detector
                )

            # Use ensemble if confidence is high
            if confidence_score >= 0.7:
                recommended_method = "Ensemble Combined"
                recommended_bands = ensemble_combined_bands if ensemble_combined_bands else conditional_bands
                print(f"✓ Using ensemble method (confidence: {confidence_score:.1%})")
            else:
                # Low confidence - fallback to conditional
                recommended_method = "Conditional (Feature-Weighted)"
                recommended_bands = conditional_bands if conditional_bands else baseline_bands
                print(f"ℹ️  Low model confidence ({confidence_score:.1%}) - using conditional method")
        except Exception as e:
            # Error checking confidence - use conditional
            recommended_method = "Conditional (Feature-Weighted)"
            recommended_bands = conditional_bands if conditional_bands else baseline_bands
    else:
        # Fallback to conditional or baseline
        recommended_method = "Conditional (Feature-Weighted)"
        if conditional_bands:
            recommended_bands = conditional_bands
        else:
            recommended_method = "Baseline (Simple Percentile)"
            recommended_bands = baseline_bands

    # Show all methods for comparison
    methods_to_show = [
        ("Baseline (Simple Percentile)", baseline_bands, "Reference", False),
        ("🏆 Conditional (Feature-Weighted)", conditional_bands, "⭐ RECOMMENDED - 37% tighter bands, 97-99% hit rate", True),
        ("Ensemble (LightGBM)", ensemble_bands, "Alternative (wider bands, 100% hit rate)", False),
        ("Ensemble Combined", ensemble_combined_bands, "Conservative blend", False),
    ]

    print(f"{'='*80}")
    print(f"PREDICTION METHODS COMPARISON")
    print(f"{'='*80}")
    print(f"Based on 180-day backtest validation:")
    print(f"  • Conditional: 37-39% TIGHTER bands than baseline (97-99% hit rate)")
    print(f"  • Ensemble: 24-58% WIDER bands than baseline (100% hit rate, too conservative)")
    print(f"  • Recommendation: Use Conditional for best capital efficiency")
    print(f"{'='*80}\n")

    for method_name, bands, description, is_recommended in methods_to_show:
        if not bands:
            continue

        print(f"{'='*80}")
        print(f"{method_name}")
        print(f"{description}")
        print(f"{'='*80}")

        for band_name in ['P95', 'P97', 'P98', 'P99']:
            if band_name in bands:
                b = bands[band_name]
                print(f"{band_name:6}  ${b.lo_price:>10,.2f} - ${b.hi_price:>10,.2f}   "
                      f"(±{b.width_pts/2:>6,.0f} pts, ±{b.width_pct/2:>5.2f}%)")
        print()

    # Use last CSV close as prev_close (static reference)
    prev_close = prev_close_for_baseline

    # Return UnifiedPrediction with Ensemble Combined as primary
    # Fall back to conditional, then baseline if ensemble_combined not available
    primary_bands = ensemble_combined_bands if ensemble_combined_bands else (
        conditional_bands if conditional_bands else baseline_bands
    )

    # Convert bands to dict format for all 4 methods
    def bands_to_dict(bands):
        if not bands:
            return {}
        return {
            name: {
                'lo_price': float(band.lo_price),
                'hi_price': float(band.hi_price),
                'lo_pct': float(band.lo_pct),
                'hi_pct': float(band.hi_pct),
                'width_pct': float(band.width_pct),
                'width_pts': float(band.width_pts),
            }
            for name, band in bands.items()
        }

    pred = UnifiedPrediction(
        ticker=ticker,
        current_price=current_price,
        prev_close=prev_close,
        hours_to_close=0.0,  # N/A for multi-day
        time_label=f"{days_ahead}DTE",
        above_prev=current_price >= prev_close,
        percentile_bands=primary_bands,  # Use ensemble_combined as primary
        statistical_bands=ensemble_bands if ensemble_bands else {},
        combined_bands=primary_bands,
        confidence="HIGH" if ensemble_combined_bands else "MEDIUM",
        risk_level=None,
    )

    # Add all 4 ensemble methods with dynamic recommendation based on regime/confidence
    # Mark the actually selected method as recommended
    pred.ensemble_methods = [
        {
            'method': 'Baseline (Percentile)',
            'description': 'Reference method - simple percentile distribution',
            'bands': bands_to_dict(baseline_bands),
            'recommended': recommended_method == "Baseline (Simple Percentile)",
            'backtest_performance': 'Reference (100% hit rate)',
            'fallback_reason': fallback_reason if recommended_method == "Baseline (Simple Percentile)" else None,
        },
        {
            'method': 'Conditional (Feature-Weighted)',
            'description': '⭐ Best balance of tight bands and reliability',
            'bands': bands_to_dict(conditional_bands),
            'recommended': recommended_method == "Conditional (Feature-Weighted)",
            'backtest_performance': '37-39% tighter bands, 97-99% hit rate',
            'fallback_reason': fallback_reason if recommended_method == "Conditional (Feature-Weighted)" else None,
        },
        {
            'method': 'Ensemble (LightGBM)',
            'description': 'Machine learning - too conservative for trading',
            'bands': bands_to_dict(ensemble_bands),
            'recommended': False,  # Never directly recommended (use Combined instead)
            'backtest_performance': '24-58% wider bands, 100% hit rate',
        },
        {
            'method': 'Ensemble Combined',
            'description': 'Conservative blend - use when models are confident',
            'bands': bands_to_dict(ensemble_combined_bands),
            'recommended': recommended_method == "Ensemble Combined",
            'backtest_performance': '24-58% wider bands, 100% hit rate',
            'fallback_reason': None,
        },
    ]

    # Add regime status to prediction for monitoring
    if regime_status:
        pred.regime_status = regime_status

    # Add additional fields expected by web UI (for compatibility with old format)
    mean_return = float(np.mean(n_day_returns_array))
    median_return = float(np.median(n_day_returns_array))
    std_return = float(np.std(n_day_returns_array))

    pred.target_date = target_date.isoformat() if hasattr(target_date, 'isoformat') else str(target_date)
    pred.target_date_str = target_date.strftime("%A, %B %d, %Y") if hasattr(target_date, 'strftime') else str(target_date)
    pred.expected_price = current_price * (1 + mean_return / 100)
    pred.mean_return = mean_return
    pred.median_return = median_return
    pred.std_return = std_return
    pred.n_samples = len(n_day_returns_array)

    print(f"{'='*80}")
    print(f"PRIMARY PREDICTION: {'Ensemble Combined' if ensemble_combined_bands else ('Conditional' if conditional_bands else 'Baseline')}")
    print(f"{'='*80}")

    print(f"{'='*80}\n")

    return pred


async def predict_close(ticker='NDX', lookback=120, force_retrain=False, similar_days_count=10, db_config=None, days_ahead=0, target_date=None, use_time_decay=True, use_intraday_vol=True):
    """Make a prediction for today's close (or future date) using LIVE QuestDB data.

    Args:
        ticker: Ticker symbol (NDX or SPX)
        lookback: Number of historical days to analyze
        force_retrain: Force model retraining
        similar_days_count: Number of similar days to show
        db_config: QuestDB connection string
        days_ahead: Number of trading days ahead to predict (0 = today)
        target_date: Specific future date to predict (date object or YYYY-MM-DD string)
        use_time_decay: Enable time decay factor for multi-day predictions (default True)
        use_intraday_vol: Enable intraday volatility scaling for multi-day predictions (default True)

    Returns:
        UnifiedPrediction object with percentile_bands, statistical_bands, combined_bands
    """

    # Convert target_date to days_ahead if provided
    if target_date is not None:
        if isinstance(target_date, str):
            target_date = datetime.strptime(target_date, '%Y-%m-%d').date()
        today = datetime.now(ET_TZ).date()
        # Count trading days between now and target
        days_ahead = 0
        check_date = today
        while check_date < target_date:
            check_date = check_date + timedelta(days=1)
            if check_date.weekday() < 5:  # Mon-Fri
                days_ahead += 1

    # For multi-day ahead predictions, use historical N-day return distribution
    if days_ahead > 0:
        return await _predict_future_close_unified(ticker, days_ahead, lookback, db_config, use_time_decay, use_intraday_vol)

    print(f"\n{'='*80}")
    print(f"PREDICT TODAY'S CLOSE - {ticker} (LIVE DATA)")
    print(f"{'='*80}\n")

    # Connect to QuestDB
    print("Connecting to QuestDB...")
    if db_config is None:
        db_config = (
            os.getenv('QUEST_DB_STRING', '')
            or os.getenv('QUESTDB_CONNECTION_STRING', '')
            or os.getenv('QUESTDB_URL', '')
            or 'postgresql://admin:quest@localhost:8812/qdb'
        )
    redis_url = os.getenv('REDIS_URL', 'redis://localhost:6379/0')

    db = get_stock_db(
        'questdb',
        db_config=db_config,
        enable_cache=True,
        redis_url=redis_url,
    )

    # QuestDB stores tickers without the I: prefix
    db_ticker = ticker.replace("I:", "") if ticker.startswith("I:") else ticker

    # Get current price from QuestDB (latest close or realtime price)
    print(f"Fetching current price from QuestDB...")
    current_price = None
    current_time = None
    current_date = None
    data_source = None

    try:
        current_data = await db.get_latest_price_with_data(db_ticker)
        if current_data and current_data.get('price') is not None:
            current_price = current_data['price']
            current_time_utc = current_data.get('timestamp')
            data_source = current_data.get('source', 'QuestDB')

            # Convert to ET
            if current_time_utc:
                current_time = current_time_utc.astimezone(ET_TZ)
                current_date = current_time.date()
            else:
                current_time = datetime.now(ET_TZ)
                current_date = current_time.date()

            print(f"✓ Current price: ${current_price:,.2f} (source: {data_source}, date: {current_date})")
        else:
            print(f"❌ No price data available in QuestDB")
            return

    except Exception as e:
        print(f"❌ QuestDB error: {e}")
        return

    # Get previous close - try QuestDB first, then fall back to CSV
    print(f"Fetching previous close...")
    prev_close = None
    prev_close_date = None
    prev_close_source = None  # Track which source was used

    # Determine expected previous trading day
    expected_prev_day = get_last_trading_day(current_date)
    print(f"Expected previous trading day: {expected_prev_day}")

    # Try QuestDB first (faster and includes today's data)
    try:
        import asyncpg

        async with db.connection.get_connection() as conn:
            # Get the most recent daily close before current_date
            row = await conn.fetchrow(
                """
                SELECT date, close
                FROM daily_prices
                WHERE ticker = $1 AND date < $2
                ORDER BY date DESC
                LIMIT 1
                """,
                db_ticker,
                current_date
            )

            if row:
                prev_close_candidate = row['close']
                prev_close_date_candidate = row['date']

                # Convert datetime to date if needed
                if hasattr(prev_close_date_candidate, 'date'):
                    prev_close_date_candidate = prev_close_date_candidate.date()

                # Check if QuestDB returned the expected trading day
                # Must match exactly or be within 1 day (for after-hours edge cases)
                days_diff = (expected_prev_day - prev_close_date_candidate).days
                if days_diff == 0:
                    # Perfect match
                    prev_close = prev_close_candidate
                    prev_close_date = prev_close_date_candidate
                    prev_close_source = "QuestDB"
                    print(f"✓ Previous close (from QuestDB): ${prev_close:,.2f} (date: {prev_close_date})")
                elif days_diff == 1:
                    # QuestDB daily_prices is 1 day behind - try intraday prices table
                    print(f"⚠️  QuestDB daily_prices is 1 day behind (got {prev_close_date_candidate}, expected {expected_prev_day})")
                    print(f"    Trying to aggregate from intraday prices table...")

                    try:
                        # Get the last price from the expected previous day
                        # Convert date to datetime for the query
                        from datetime import datetime as dt
                        start_of_day = dt.combine(expected_prev_day, dt.min.time())
                        end_of_day = dt.combine(expected_prev_day, dt.max.time())

                        intraday_row = await conn.fetchrow(
                            """
                            SELECT price, timestamp
                            FROM realtime_data
                            WHERE ticker = $1
                                AND timestamp >= $2
                                AND timestamp <= $3
                            ORDER BY timestamp DESC
                            LIMIT 1
                            """,
                            db_ticker,
                            start_of_day,
                            end_of_day
                        )

                        if intraday_row and intraday_row['price']:
                            prev_close = intraday_row['price']
                            prev_close_date = expected_prev_day
                            prev_close_source = "QuestDB intraday (aggregated)"
                            print(f"✓ Previous close (from QuestDB intraday): ${prev_close:,.2f} (date: {prev_close_date})")
                        else:
                            print(f"    No intraday data found, will check CSV...")
                    except Exception as e:
                        print(f"    Intraday lookup failed: {e}, will check CSV...")
                else:
                    print(f"⚠️  QuestDB data is stale (got {prev_close_date_candidate}, expected {expected_prev_day}), trying CSV...")
            else:
                print(f"⚠️  No previous close in QuestDB before {current_date}, trying CSV...")
    except Exception as e:
        print(f"⚠️  QuestDB lookup failed: {e}, trying CSV...")

    # If QuestDB didn't work, fall back to CSV
    if prev_close is None:
        try:
            all_dates_temp = get_available_dates(ticker, 20)
            if all_dates_temp:
                # Get the most recent CSV date - this is the previous trading day
                latest_csv_date = all_dates_temp[-1]
                latest_csv_date_obj = datetime.strptime(latest_csv_date, '%Y-%m-%d').date()

                # Only use CSV if it's recent (within last 5 calendar days)
                days_since_csv = (current_date - latest_csv_date_obj).days
                if days_since_csv <= 5:
                    # Load the CSV data for that date and get the closing price
                    csv_df = load_csv_data(ticker, latest_csv_date)
                    if csv_df is not None and not csv_df.empty:
                        # Get the last price (EOD close) from that day
                        prev_close = csv_df.iloc[-1]['close']
                        prev_close_date = latest_csv_date_obj
                        prev_close_source = "CSV"
                        print(f"✓ Previous close (from CSV): ${prev_close:,.2f} (date: {prev_close_date})")
                else:
                    print(f"⚠️  CSV data is stale ({days_since_csv} days old)")
        except Exception as e:
            print(f"⚠️  CSV lookup also failed: {e}")

    # Final fallback
    if prev_close is None:
        print(f"⚠️  Using current price as previous close (no historical data available)")
        prev_close = current_price
        prev_close_date = current_date
        prev_close_source = "Fallback (current price)"

    # Get VIX from QuestDB
    print(f"Fetching VIX1D from QuestDB...")
    vix1d = None

    try:
        vix_data = await db.get_latest_price_with_data('VIX1D')
        if vix_data and vix_data.get('price'):
            vix1d = vix_data['price']
            print(f"✓ VIX1D: {vix1d:.2f}")
        else:
            print(f"⚠️  VIX1D not available in QuestDB, will use default")
    except Exception as e:
        print(f"⚠️  Failed to fetch VIX: {e}")

    # Get available CSV data for TRAINING
    print(f"\nLoading historical CSV data for training...")
    all_dates = get_available_dates(ticker, lookback + 20)
    if len(all_dates) < lookback:
        print(f"❌ Not enough training data. Need {lookback} days, have {len(all_dates)}")
        return

    training_date = all_dates[-1]
    print(f"Training data through: {training_date}")

    # Use default VIX if not available from QuestDB
    if vix1d is None:
        vix1d = 15.0
        print(f"ℹ️  Using default VIX1D: {vix1d:.2f}")

    # Determine which training approach to use based on current time
    # STATIC: Train on historical data only (9:30 AM, 3:00 PM - higher failure risk)
    # DYNAMIC: Include today's intraday data (10:00 AM - 2:30 PM, 3:30 PM - better capital efficiency)
    current_hour = current_time.hour + current_time.minute / 60.0
    use_dynamic = True  # Default to dynamic
    training_approach = "DYNAMIC"

    # Use STATIC at specific hours where dynamic showed failures in backtesting
    if current_hour <= 9.75:  # 9:30 AM (9.5) to 9:45 AM
        use_dynamic = False
        training_approach = "STATIC"
        print(f"\n⚠️  Using STATIC approach (9:30 AM - risk hour based on backtest)")
    elif 14.75 <= current_hour <= 15.25:  # 2:45 PM to 3:15 PM (around 3:00 PM)
        use_dynamic = False
        training_approach = "STATIC"
        print(f"\n⚠️  Using STATIC approach (3:00 PM - risk hour based on backtest)")
    else:
        print(f"\n✓ Using DYNAMIC approach (includes today's data for better capital efficiency)")

    # Try to load cached model
    print(f"\nChecking for cached model...")
    stat_predictor, cache_meta = load_model_cache(ticker, training_date, lookback, force_retrain)

    if stat_predictor is not None:
        cached_at = cache_meta.get('cached_at', 'unknown time')
        cached_approach = cache_meta.get('training_approach', 'UNKNOWN')

        # If cached model doesn't match desired approach, retrain
        if cached_approach != training_approach:
            print(f"⚠️  Cached model uses {cached_approach} approach, but need {training_approach}")
            print(f"   Retraining with {training_approach} approach...")
            stat_predictor = None
            cache_meta = None
        else:
            print(f"✓ Loaded cached model from {cached_at}")
            print(f"  Model: {cache_meta.get('model_type')}")
            print(f"  Approach: {cached_approach}")
            print(f"  Version: {cache_meta.get('model_version')}")
            print(f"  Band width: {cache_meta.get('band_width_scale')}x")

    if stat_predictor is None:
        # Train new model
        if force_retrain:
            print(f"🔄 Force retraining model with {training_approach} approach...")
        else:
            print(f"No valid cache found, training new model with {training_approach} approach...")

        # Build training data from CSV
        print(f"Building training data from {lookback} days of CSV files...")
        train_df = build_training_data(ticker, training_date, lookback)

        if train_df.empty or len(train_df) < 50:
            print("❌ Insufficient training data from CSV")
            return

        print(f"✓ Loaded {len(train_df)} training samples from CSV (historical)")

        # DYNAMIC approach: Try to append today's data from QuestDB
        if use_dynamic:
            today_str = datetime.now(ET_TZ).strftime('%Y-%m-%d')
            if today_str != training_date:
                print(f"[DYNAMIC] Appending today's data ({today_str}) from QuestDB...")
                try:
                    # Get historical context for today's features
                    hist_ctx = get_historical_context(ticker, training_date, num_days_back=55)

                    # Fetch today's intraday data from QuestDB up to current time
                    today_df = await append_today_from_questdb(db, ticker, today_str, hist_ctx, vix1d)

                    if today_df is not None and not today_df.empty:
                        # Filter to only include data up to current hour
                        # (append_today_from_questdb samples at 15-min intervals)
                        today_df_filtered = today_df[today_df['hour_et'] <= current_hour]

                        if not today_df_filtered.empty:
                            print(f"✓ Appended {len(today_df_filtered)} samples from today up to {current_hour:.1f}")
                            import pandas as pd
                            train_df = pd.concat([train_df, today_df_filtered], ignore_index=True)
                            print(f"✓ Total training samples: {len(train_df)} (historical + today)")
                        else:
                            print(f"ℹ️  No samples from today up to current hour")
                    else:
                        print(f"ℹ️  No data available for today from QuestDB")
                except Exception as e:
                    print(f"⚠️  Could not append today's data: {e}")
                    print(f"   Continuing with CSV data only (falling back to STATIC)")
        else:
            print(f"[STATIC] Training on historical data only (not including today)")

        # Train model on combined data
        print(f"Training LightGBM predictor on {len(train_df)} samples...")

        from scripts.strategy_utils.close_predictor import LGBMClosePredictor
        from scripts.close_predictor.models import (
            LGBM_N_ESTIMATORS,
            LGBM_LEARNING_RATE,
            LGBM_MAX_DEPTH,
            LGBM_MIN_CHILD_SAMPLES,
            LGBM_BAND_WIDTH_SCALE,
        )

        stat_predictor = LGBMClosePredictor(
            n_estimators=LGBM_N_ESTIMATORS,
            learning_rate=LGBM_LEARNING_RATE,
            max_depth=LGBM_MAX_DEPTH,
            min_child_samples=LGBM_MIN_CHILD_SAMPLES,
            band_width_scale=LGBM_BAND_WIDTH_SCALE,
            use_fallback=True,
        )

        # Validate training data - remove rows with NaN in target or features
        initial_rows = len(train_df)
        train_df_clean = train_df.dropna()
        rows_dropped = initial_rows - len(train_df_clean)

        if rows_dropped > 0:
            print(f"⚠️  Dropped {rows_dropped} rows with NaN values ({rows_dropped/initial_rows*100:.1f}%)")

        if len(train_df_clean) < 100:
            print(f"❌ Insufficient training data after cleaning: {len(train_df_clean)} rows (need at least 100)")
            return

        stat_predictor.fit(train_df_clean)
        print(f"✓ Model trained: {type(stat_predictor).__name__}")

    # Collect percentile data for training
    print("Collecting percentile data...")
    pct_df = collect_all_data(ticker, all_dates)
    if pct_df is None or pct_df.empty:
        print("❌ No percentile data")
        return

    unique_dates = sorted(pct_df['date'].unique())
    # Constrain percentile training to match lookback window
    if len(unique_dates) > lookback:
        unique_dates = unique_dates[-lookback:]
    pct_train_dates = set(unique_dates)
    train_dates_sorted = unique_dates

    # Save model to cache if it was just trained
    if cache_meta is None and stat_predictor is not None:
        print("Saving model to cache...")
        if save_model_cache(stat_predictor, ticker, training_date, lookback, pct_df, training_approach):
            print(f"✓ Model cached for future use ({training_approach} approach)")
            # Clean up old caches
            clear_old_caches(ticker, training_date, lookback)

    # Get historical context for features (MA, previous days, etc.)
    print("Getting historical context...")
    hist_ctx = get_historical_context(ticker, training_date, num_days_back=55)

    # Build day context from historical data + live VIX
    day_ctx = DayContext(
        prev_close=prev_close,
        day_open=prev_close,  # We'll use prev_close as approximation for day open
        vix1d=vix1d,
        prev_day_close=hist_ctx.get('day_2', {}).get('close'),
        prev_vix1d=hist_ctx.get('day_1', {}).get('vix1d'),
        prev_day_high=hist_ctx.get('day_1', {}).get('high'),
        prev_day_low=hist_ctx.get('day_1', {}).get('low'),
        close_5days_ago=hist_ctx.get('day_5', {}).get('close'),
        ma5=hist_ctx.get('ma5'),
        ma10=hist_ctx.get('ma10'),
        ma20=hist_ctx.get('ma20'),
        ma50=hist_ctx.get('ma50'),
    )

    # Use current live time
    time_label = _find_nearest_time_label(current_time.hour, current_time.minute)

    hours_to_close = (16 - current_time.hour) + (0 - current_time.minute) / 60.0
    if hours_to_close < 0:
        hours_to_close = 0

    # Estimate day high/low (use current price as proxy if not available)
    day_high = current_price
    day_low = current_price

    print(f"\n{'='*80}")
    print(f"CURRENT MARKET STATE (QUESTDB DATA)")
    print(f"{'='*80}")
    print(f"Time:           {current_time.strftime('%B %d, %Y at %I:%M %p ET')}")
    print(f"Time to Close:  {hours_to_close:.1f} hours")
    print(f"\nCurrent Price:  ${current_price:,.2f}")
    print(f"Previous Close: ${prev_close:,.2f} (source: {prev_close_source}, date: {prev_close_date})")
    print(f"Move from Prev: {((current_price - prev_close) / prev_close * 100):+.2f}%")
    print(f"VIX1D:          {vix1d:.2f}")
    print(f"Data Source:    {data_source}")
    print(f"\nTraining Approach: {training_approach}")
    if training_approach == "STATIC":
        print(f"  (Using historical data only - safer at this hour based on backtest)")
    else:
        print(f"  (Including today's data - better capital efficiency)")

    # Load test data for day high/low (but don't use for intraday vol scaling)
    # Disable intraday vol factor to match validated backtest approach
    # Backtest achieved 98.9% accuracy WITHOUT intraday vol scaling
    test_df = None
    try:
        test_df = load_csv_data(ticker, training_date)
    except:
        pass

    ivol_factor = 1.0

    # Make prediction using QuestDB price data
    print(f"\n{'='*80}")
    print("GENERATING PREDICTION WITH QUESTDB DATA...")
    print(f"{'='*80}")

    pred = make_unified_prediction(
        pct_df=pct_df,
        predictor=stat_predictor,
        ticker=ticker,
        current_price=current_price,
        prev_close=prev_close,
        current_time=current_time,
        time_label=time_label,
        day_ctx=day_ctx,
        day_high=day_high,
        day_low=day_low,
        train_dates=pct_train_dates,
        current_vol=None,
        vol_scale=True,
        data_source=f'QuestDB ({data_source})',
        intraday_vol_factor=ivol_factor,
    )

    if not pred:
        print("❌ Prediction failed")
        return

    # Add all prediction methods as a dynamic attribute (for 0DTE comparison display)
    def bands_to_dict_0dte(bands):
        if not bands:
            return {}
        return {
            name: {
                'lo_price': float(band.lo_price),
                'hi_price': float(band.hi_price),
                'lo_pct': float(band.lo_pct),
                'hi_pct': float(band.hi_pct),
                'width_pct': float(band.width_pct),
                'width_pts': float(band.width_pts),
            }
            for name, band in bands.items()
        }

    pred.ensemble_methods = [
        {
            'method': 'Percentile (Historical)',
            'description': 'Historical percentile distribution',
            'bands': bands_to_dict_0dte(pred.percentile_bands),
            'recommended': False,
            'backtest_performance': 'Baseline reference',
        },
        {
            'method': 'LightGBM (Statistical)',
            'description': 'Machine learning statistical model',
            'bands': bands_to_dict_0dte(pred.statistical_bands),
            'recommended': False,
            'backtest_performance': 'ML-based prediction',
        },
        {
            'method': 'Combined (Blended)',
            'description': '⭐ RECOMMENDED - Blend of percentile and statistical',
            'bands': bands_to_dict_0dte(pred.combined_bands),
            'recommended': True,
            'backtest_performance': 'Best balance for 0DTE',
        },
    ]

    # Display predictions
    print(f"\n{'='*80}")
    print(f"PREDICTED CLOSE RANGES (LightGBM)")
    print(f"{'='*80}")
    print(f"Model: {type(stat_predictor).__name__}")
    print(f"Training: {lookback} days (~{len(train_dates_sorted)*19:,} samples)")
    print(f"Band Width: 3.0x (78.3% historical hit rate)")

    # Show predictions from each model + combined
    print(f"\n{'='*80}")
    print("PREDICTION BANDS BY MODEL")
    print(f"{'='*80}")

    # 1. LightGBM Model (Statistical)
    if pred.statistical_bands:
        print(f"\n{'-'*80}")
        print("1. LightGBM MODEL (ML-based quantile regression)")
        print(f"{'-'*80}")

        for band_name in ['P95', 'P97', 'P98', 'P99', 'P100']:
            if band_name in pred.statistical_bands:
                band = pred.statistical_bands[band_name]
                print(f"  {band_name}: ${band.lo_price:,.2f} - ${band.hi_price:,.2f}  ({band.width_pct:.2f}% width)")

    # 2. Percentile Model (Historical)
    if pred.percentile_bands:
        print(f"\n{'-'*80}")
        print("2. PERCENTILE MODEL (Historical distribution)")
        print(f"{'-'*80}")

        for band_name in ['P95', 'P97', 'P98', 'P99', 'P100']:
            if band_name in pred.percentile_bands:
                band = pred.percentile_bands[band_name]
                print(f"  {band_name}: ${band.lo_price:,.2f} - ${band.hi_price:,.2f}  ({band.width_pct:.2f}% width)")

    # 3. Combined (Wider of both)
    if pred.combined_bands:
        print(f"\n{'-'*80}")
        print("3. COMBINED PREDICTION (Wider range from both models)")
        print(f"{'-'*80}")
        print("   *** USE THIS FOR TRADING - MOST CONSERVATIVE ***")
        print(f"{'-'*80}")

        for band_name in ['P95', 'P97', 'P98', 'P99', 'P100']:
            if band_name in pred.combined_bands:
                band = pred.combined_bands[band_name]
                midpoint = (band.lo_price + band.hi_price) / 2

                print(f"\n{band_name} Band:")
                print(f"  Lower:    ${band.lo_price:,.2f}  ({band.lo_pct:+.2f}% from current)")
                print(f"  Mid:      ${midpoint:,.2f}  ({((midpoint-current_price)/current_price*100):+.2f}%)")
                print(f"  Upper:    ${band.hi_price:,.2f}  ({band.hi_pct:+.2f}%)")
                print(f"  Width:    {band.width_pct:.2f}% (${band.width_pts:,.2f})")

    # Metadata
    print(f"\n{'='*80}")
    print(f"CONFIDENCE METRICS")
    print(f"{'='*80}")
    print(f"Confidence:  {pred.confidence or 'MEDIUM'}")
    if pred.risk_level:
        print(f"Risk Level:  {pred.risk_level}/10")

    # Intelligent band recommendation
    from scripts.close_predictor.band_selector import recommend_band, format_recommendation, RiskProfile

    # Get day's high/low for trend analysis
    if test_df is not None and not test_df.empty:
        day_high = test_df['high'].max()
        day_low = test_df['low'].min()
    else:
        day_high = current_price
        day_low = current_price

    # Get recommendation for different risk profiles
    print(f"\n{'='*80}")
    print("INTELLIGENT BAND SELECTION")
    print(f"{'='*80}")
    print(f"\nCurrent Price: ${current_price:,.2f}")
    print(f"{'='*80}\n")

    for risk_profile in [RiskProfile.AGGRESSIVE, RiskProfile.MODERATE, RiskProfile.CONSERVATIVE]:
        rec = recommend_band(
            vix=vix1d if vix1d else 15.0,
            hours_to_close=hours_to_close,
            current_price=current_price,
            prev_close=prev_close,
            day_high=day_high,
            day_low=day_low,
            risk_profile=risk_profile,
            realized_vol=getattr(pred, 'realized_vol', None),
            historical_avg_vol=None,
        )

        # Get the recommended band details from combined bands
        if rec.recommended_band in pred.combined_bands:
            band = pred.combined_bands[rec.recommended_band]
            lo_price = band.lo_price
            hi_price = band.hi_price
            lo_pct = (lo_price - current_price) / current_price * 100
            hi_pct = (hi_price - current_price) / current_price * 100
            lo_amt = lo_price - current_price
            hi_amt = hi_price - current_price
            width_pct = band.width_pct

            print(f"{risk_profile.value.upper()} Profile:")
            print(f"  → Use {rec.recommended_band} Band")
            print(f"     Range: ${lo_price:,.2f} to ${hi_price:,.2f}")
            print(f"     Move from Current: {lo_pct:+.2f}% to {hi_pct:+.2f}% (${lo_amt:+,.2f} to ${hi_amt:+,.2f})")
            print(f"     Band Width: {width_pct:.2f}%")
            print(f"     Hit Rate: {rec.expected_hit_rate*100:.0f}% | Opportunity Score: {rec.opportunity_score*100:.0f}/100")
            print(f"     {rec.rationale}")
        else:
            # Fallback if band not available
            print(f"{risk_profile.value.upper()} Profile:")
            print(f"  → Use {rec.recommended_band} Band")
            print(f"     Hit Rate: {rec.expected_hit_rate*100:.0f}% | Opportunity: {rec.opportunity_score*100:.0f}/100")
            print(f"     {rec.rationale}")
        print()

    # Similar days analysis
    if pct_df is not None and not pct_df.empty:
        try:
            from scripts.close_predictor.similar_days import find_similar_days, format_similar_days

            # Calculate today's characteristics
            gap_pct = (current_price - prev_close) / prev_close * 100
            day_open = test_df.iloc[0]['open'] if test_df is not None and not test_df.empty else current_price
            intraday_move = (current_price - day_open) / day_open * 100 if day_open > 0 else 0

            # Find similar days (90%+ similarity)
            similar_days = find_similar_days(
                pct_df=pct_df,
                current_vix=vix1d if vix1d else 15.0,
                current_gap_pct=gap_pct,
                current_intraday_move=intraday_move,
                current_price=current_price,
                prev_close=prev_close,
                time_label=pred.time_label,
                top_n=50,  # Capture up to 50 matches
                min_similarity=90.0,  # Only 90%+ similar days
            )

            if similar_days:
                print(format_similar_days(similar_days, current_price, show_top_n=similar_days_count))
        except Exception as e:
            print(f"\n⚠️  Similar days analysis unavailable: {e}")

    # Time-based expectations
    if hours_to_close <= 1:
        print(f"\n✓ High accuracy period (near close)")
        print(f"  Expected hit rate: 80-90%")
        print(f"  Expected midpoint error: <0.5%")
    elif hours_to_close <= 2:
        print(f"\n✓ Good accuracy period (afternoon)")
        print(f"  Expected hit rate: 75-85%")
        print(f"  Expected midpoint error: ~0.64%")
    else:
        print(f"\n⚠️  Lower accuracy period (early day)")
        print(f"  Expected hit rate: 65-75%")
        print(f"  Expected midpoint error: ~1.0%")

    print(f"\n{'='*80}")
    print(f"PREDICTED CLOSE: ${midpoint:,.2f}")
    print(f"RANGE: ${band.lo_price:,.2f} - ${band.hi_price:,.2f}")
    print(f"{'='*80}\n")

    # Add similar days and training approach to prediction object for web interface
    pred.training_approach = training_approach

    # Compute similar days and add to prediction object
    similar_days_list = None
    if pct_df is not None and not pct_df.empty:
        try:
            from scripts.close_predictor.similar_days import find_similar_days

            # Calculate today's characteristics
            gap_pct = (current_price - prev_close) / prev_close * 100
            day_open = test_df.iloc[0]['open'] if test_df is not None and not test_df.empty else current_price
            intraday_move = (current_price - day_open) / day_open * 100 if day_open > 0 else 0

            # Find similar days (90%+ similarity)
            similar_days_objs = find_similar_days(
                pct_df=pct_df,
                current_vix=vix1d if vix1d else 15.0,
                current_gap_pct=gap_pct,
                current_intraday_move=intraday_move,
                current_price=current_price,
                prev_close=prev_close,
                time_label=pred.time_label,
                top_n=50,  # Capture up to 50 matches
                min_similarity=90.0,  # Only 90%+ similar days
            )

            # Convert SimilarDay objects to dicts for JSON serialization
            if similar_days_objs:
                similar_days_list = [
                    {
                        'date': sd.date,
                        'similarity_score': sd.similarity_score,
                        'vix': sd.vix,
                        'gap_pct': sd.gap_pct,
                        'intraday_move_pct': sd.intraday_move_pct,
                        'actual_close_move': sd.actual_close_move,
                        'time_label': sd.time_label,
                        'outcome': sd.outcome,
                    }
                    for sd in similar_days_objs
                ]
        except Exception as e:
            # If similar days computation fails, just skip it
            pass

    pred.similar_days = similar_days_list

    # Return the prediction object for programmatic use
    return pred


def _build_predict_parser(parser):
    """Add predict-mode arguments to a parser."""
    parser.add_argument('ticker', nargs='?', default='NDX', choices=['NDX', 'SPX'],
                        help='Ticker symbol to predict (default: NDX)')
    parser.add_argument('--retrain', '--force-retrain',
                        action='store_true', dest='force_retrain',
                        help='Force retrain model, ignore cached version')
    parser.add_argument('--similar-days', type=int, default=10, metavar='N',
                        help='Number of similar historical days to display (default: 10, max: 50)')
    parser.add_argument('--days-ahead', type=int, metavar='N',
                        help='Predict close N trading days ahead (e.g., 5 for next Friday)')
    parser.add_argument('--target-date', type=str, metavar='YYYY-MM-DD',
                        help='Predict close for specific future date (e.g., 2026-02-20)')
    parser.add_argument('--lookback', type=int, default=120, metavar='N',
                        help='Number of historical trading days for training (default: 120)')
    parser.add_argument('--db', type=str, default=None, metavar='CONNECTION_STRING',
                        help='QuestDB connection string (default: QUEST_DB_STRING env)')
    parser.add_argument('--no-time-decay', action='store_true', dest='no_time_decay',
                        help='Disable time decay factor for multi-day predictions')
    parser.add_argument('--no-intraday-vol', action='store_true', dest='no_intraday_vol',
                        help='Disable intraday volatility scaling for multi-day predictions')


def _build_train_parser(parser):
    """Add train-mode arguments to a parser."""
    parser.add_argument('ticker', nargs='?', default='NDX', choices=['NDX', 'SPX'],
                        help='Ticker symbol to train (default: NDX)')
    parser.add_argument('--lookback', type=int, default=120, metavar='N',
                        help='0DTE training lookback in trading days (default: 120)')
    parser.add_argument('--db', type=str, default=None, metavar='CONNECTION_STRING',
                        help='QuestDB connection string (default: QUEST_DB_STRING env)')
    parser.add_argument('--max-dte', type=int, default=0, metavar='N',
                        help='Also train multi-day models for DTE 1..N (e.g., 20). 0 = 0DTE only.')
    parser.add_argument('--train-days', type=int, default=250, metavar='N',
                        help='Multi-day training window in days (default: 250)')
    parser.add_argument('--validate-days', type=int, default=30, metavar='N',
                        help='Multi-day validation window in days (default: 30)')
    parser.add_argument('--output-dir', type=Path, default=Path('models/production'),
                        help='Output directory for multi-day models (default: models/production)')
    parser.add_argument('--clear-cache', action='store_true',
                        help='Clear .cache/ files for this ticker before training')
    parser.add_argument('--clear-models', action='store_true',
                        help='Clear models/production/{ticker}_latest/ before training')


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        description='''
Predict or train close-price models for NDX / SPX.

If no subcommand is given, defaults to "predict".
        ''',
        epilog='''
Examples:
  # Predict (default)
  %(prog)s NDX                                 0DTE prediction
  %(prog)s NDX --days-ahead 5                  Multi-day prediction
  %(prog)s predict NDX --days-ahead 5          Explicit predict subcommand

  # Train
  %(prog)s train NDX                           Train 0DTE model only
  %(prog)s train NDX --max-dte 20              Train 0DTE + multi-day (1-20)
  %(prog)s train NDX --clear-cache --clear-models --max-dte 20
                                               Full clean retrain
        ''',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest='mode')

    # predict subcommand
    predict_parser = subparsers.add_parser(
        'predict',
        help='Predict close price (default mode)',
        epilog='''
Examples:
  %(prog)s NDX                    0DTE prediction
  %(prog)s SPX --days-ahead 3     3-day ahead prediction
  %(prog)s NDX --retrain          Force model retrain then predict
  %(prog)s NDX --target-date 2026-03-05
        ''',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    _build_predict_parser(predict_parser)

    # train subcommand
    train_parser = subparsers.add_parser(
        'train',
        help='Train prediction models',
        epilog='''
Examples:
  %(prog)s NDX                              Train 0DTE model
  %(prog)s NDX --max-dte 20                 Train 0DTE + multi-day (1-20)
  %(prog)s NDX --max-dte 20 --train-days 250
  %(prog)s NDX --clear-cache                Clear cache then retrain
  %(prog)s NDX --clear-models               Clear latest models then retrain
  %(prog)s NDX --clear-cache --clear-models --max-dte 20
        ''',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    _build_train_parser(train_parser)

    # --- Default mode handling ---
    # If first positional arg is not a subcommand, assume "predict"
    args = parser.parse_args()
    if args.mode is None:
        # Re-parse with 'predict' prepended
        args = parser.parse_args(['predict'] + sys.argv[1:])

    if args.mode == 'predict':
        # Validate predict-specific args
        if hasattr(args, 'similar_days') and (args.similar_days < 1 or args.similar_days > 50):
            parser.error("--similar-days must be between 1 and 50")

        if hasattr(args, 'lookback') and (args.lookback < 30 or args.lookback > 1260):
            parser.error("--lookback must be between 30 and 1260 (5 years)")

        if getattr(args, 'days_ahead', None) and getattr(args, 'target_date', None):
            parser.error("Cannot specify both --days-ahead and --target-date")

        if getattr(args, 'days_ahead', None) and args.days_ahead < 1:
            parser.error("--days-ahead must be at least 1")

        if getattr(args, 'target_date', None):
            try:
                target_date = datetime.strptime(args.target_date, '%Y-%m-%d').date()
                if target_date <= datetime.now(ET_TZ).date():
                    parser.error("--target-date must be a future date")
            except ValueError:
                parser.error("--target-date must be in YYYY-MM-DD format")

        async def run_predict():
            db_config = (
                args.db
                or os.getenv('QUEST_DB_STRING', '')
                or os.getenv('QUESTDB_CONNECTION_STRING', '')
                or os.getenv('QUESTDB_URL', '')
                or 'postgresql://admin:quest@localhost:8812/qdb'
            )

            days_ahead = 0
            if args.days_ahead or args.target_date:
                if args.target_date:
                    target_date = datetime.strptime(args.target_date, '%Y-%m-%d').date()
                    today = datetime.now(ET_TZ).date()
                    days_ahead = 0
                    check_date = today
                    while check_date < target_date:
                        check_date = check_date + timedelta(days=1)
                        if check_date.weekday() < 5:
                            days_ahead += 1
                    print(f"Target Date: {target_date} ({days_ahead} trading days from now)\n")
                else:
                    days_ahead = args.days_ahead

            await predict_close(
                ticker=args.ticker,
                lookback=args.lookback,
                force_retrain=args.force_retrain,
                similar_days_count=args.similar_days,
                db_config=db_config,
                days_ahead=days_ahead,
                use_time_decay=not args.no_time_decay,
                use_intraday_vol=not args.no_intraday_vol,
            )

        asyncio.run(run_predict())

    elif args.mode == 'train':
        async def run_train():
            # Clear cache/models if requested
            if args.clear_cache:
                clear_cache_files(args.ticker)
            if args.clear_models:
                clear_model_files(args.ticker, args.output_dir)

            # Train 0DTE model
            db_config = (
                args.db
                or os.getenv('QUEST_DB_STRING', '')
                or os.getenv('QUESTDB_CONNECTION_STRING', '')
                or os.getenv('QUESTDB_URL', '')
                or 'postgresql://admin:quest@localhost:8812/qdb'
            )
            predictor = await train_0dte_model(args.ticker, args.lookback, db_config)
            if predictor:
                print(f"\n✓ 0DTE model training complete")
            else:
                print(f"\n⚠️  0DTE model training failed")

            # Train multi-day models if requested
            if args.max_dte > 0:
                train_multi_day_models(
                    ticker=args.ticker,
                    train_days=args.train_days,
                    validate_days=args.validate_days,
                    max_dte=args.max_dte,
                    output_dir=args.output_dir,
                )

        asyncio.run(run_train())
