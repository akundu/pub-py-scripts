#!/usr/bin/env python3
"""
Comprehensive Prediction Backtest

Tests the full prediction system across:
- 0DTE (same-day close): 13 time slots × 3 models (LightGBM, Percentile, Combined)
- Multi-day (1, 2, 5, 10 DTE): 4 models (Baseline, Conditional, Ensemble, Ensemble Combined)

Usage:
    python scripts/backtest_comprehensive.py --ticker NDX
    python scripts/backtest_comprehensive.py --ticker NDX --skip-0dte
    python scripts/backtest_comprehensive.py --ticker NDX --skip-multiday
    python scripts/backtest_comprehensive.py --ticker NDX --test-days 22
"""

import sys
import os
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import time as time_mod
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
from zoneinfo import ZoneInfo

from scripts.csv_prediction_backtest import (
    get_available_dates,
    load_csv_data,
    get_day_close,
    get_day_open,
    get_day_high_low,
    get_first_hour_range,
    get_opening_range,
    get_price_at_time,
    get_vix1d_at_time,
    get_historical_context,
    DayContext,
)
from scripts.percentile_range_backtest import (
    collect_all_data,
    HOURS_TO_CLOSE,
    get_price_at_slot,
)
from scripts.close_predictor.prediction import (
    _train_statistical,
    make_unified_prediction,
)
from scripts.close_predictor.bands import map_percentile_to_bands, combine_bands
from scripts.close_predictor.models import UnifiedBand, UNIFIED_BAND_NAMES

ET_TZ = ZoneInfo("America/New_York")
UTC_TZ = ZoneInfo("UTC")

# 13 half-hour time slots for 0DTE
HALF_HOUR_SLOTS = [
    (9, 30), (10, 0), (10, 30), (11, 0), (11, 30),
    (12, 0), (12, 30), (13, 0), (13, 30), (14, 0),
    (14, 30), (15, 0), (15, 30),
]
HALF_HOUR_LABELS = [f"{h}:{m:02d}" for h, m in HALF_HOUR_SLOTS]

# Band levels to evaluate
EVAL_BANDS = ["P95", "P97", "P98", "P99"]


# ============================================================================
# 0DTE Backtest
# ============================================================================

@dataclass
class ZeroDTEResult:
    """Single 0DTE prediction result."""
    date: str
    hour: str
    model: str  # percentile, statistical, combined

    # Band prices
    p95_lo: float
    p95_hi: float
    p97_lo: float
    p97_hi: float
    p98_lo: float
    p98_hi: float
    p99_lo: float
    p99_hi: float

    actual_close: float
    current_price: float

    # Hit flags
    p95_hit: bool
    p97_hit: bool
    p98_hit: bool
    p99_hit: bool

    # Widths (% of current price)
    p95_width: float
    p97_width: float
    p98_width: float
    p99_width: float

    # Midpoint error (% of current price)
    midpoint_error: float

    vix: Optional[float] = None


def extract_band_info(
    bands: Dict[str, UnifiedBand],
    actual_close: float,
    current_price: float,
) -> dict:
    """Extract hit/miss, width, midpoint info from band dict."""
    info = {}
    for band_name in EVAL_BANDS:
        if band_name in bands:
            b = bands[band_name]
            info[f"{band_name.lower()}_lo"] = b.lo_price
            info[f"{band_name.lower()}_hi"] = b.hi_price
            info[f"{band_name.lower()}_hit"] = b.lo_price <= actual_close <= b.hi_price
            info[f"{band_name.lower()}_width"] = b.width_pct
        else:
            info[f"{band_name.lower()}_lo"] = 0.0
            info[f"{band_name.lower()}_hi"] = 0.0
            info[f"{band_name.lower()}_hit"] = False
            info[f"{band_name.lower()}_width"] = 0.0

    # Midpoint error from P97 band
    if "P97" in bands:
        mid = (bands["P97"].lo_price + bands["P97"].hi_price) / 2
    else:
        mid = current_price
    info["midpoint_error"] = abs(mid - actual_close) / current_price * 100
    return info


def make_0dte_result(
    date: str,
    hour: str,
    model: str,
    bands: Dict[str, UnifiedBand],
    actual_close: float,
    current_price: float,
    vix: Optional[float] = None,
) -> ZeroDTEResult:
    """Build a ZeroDTEResult from band dict."""
    info = extract_band_info(bands, actual_close, current_price)
    return ZeroDTEResult(
        date=date,
        hour=hour,
        model=model,
        p95_lo=info["p95_lo"], p95_hi=info["p95_hi"],
        p97_lo=info["p97_lo"], p97_hi=info["p97_hi"],
        p98_lo=info["p98_lo"], p98_hi=info["p98_hi"],
        p99_lo=info["p99_lo"], p99_hi=info["p99_hi"],
        actual_close=actual_close,
        current_price=current_price,
        p95_hit=info["p95_hit"], p97_hit=info["p97_hit"],
        p98_hit=info["p98_hit"], p99_hit=info["p99_hit"],
        p95_width=info["p95_width"], p97_width=info["p97_width"],
        p98_width=info["p98_width"], p99_width=info["p99_width"],
        midpoint_error=info["midpoint_error"],
        vix=vix,
    )


def run_0dte_backtest(
    ticker: str,
    test_days: int,
    train_days: int,
) -> List[ZeroDTEResult]:
    """Run 0DTE backtest across all test dates and time slots."""
    print(f"\n{'='*80}")
    print("PART 1: 0DTE BACKTEST (same-day close)")
    print(f"{'='*80}\n")

    # Get all available dates
    needed = train_days + test_days + 20
    all_dates = get_available_dates(ticker, needed)
    if len(all_dates) < train_days + 5:
        print(f"Not enough data: have {len(all_dates)} dates, need at least {train_days + 5}")
        return []

    # Split: last test_days for testing, rest for training
    test_dates = all_dates[-test_days:]
    print(f"Test period:  {test_dates[0]} to {test_dates[-1]} ({len(test_dates)} days)")
    print(f"Training:     {train_days} days lookback per test date\n")

    # Collect percentile data from ALL dates (training uses subset via train_dates set)
    print("Collecting percentile model data...")
    pct_df = collect_all_data(ticker, all_dates)
    print(f"  Collected {len(pct_df)} slot records across {pct_df['date'].nunique()} days\n")

    results = []
    t0 = time_mod.time()

    for test_idx, test_date in enumerate(test_dates):
        # Train LightGBM once per test date
        predictor = _train_statistical(ticker, test_date, train_days)

        # Build training date set for percentile model (dates before test_date)
        date_idx = all_dates.index(test_date)
        train_start = max(0, date_idx - train_days)
        train_date_set = set(all_dates[train_start:date_idx])

        # Load day's CSV data
        day_df = load_csv_data(ticker, test_date)
        if day_df is None or day_df.empty:
            continue

        actual_close = get_day_close(day_df)
        day_open_price = get_day_open(day_df)

        # Get historical context for DayContext
        hist_ctx = get_historical_context(ticker, test_date)
        if not hist_ctx:
            continue

        # Build base DayContext
        day_1 = hist_ctx.get('day_1', {})
        day_2 = hist_ctx.get('day_2', {})
        day_5 = hist_ctx.get('day_5', {})

        # Get VIX1D for this date
        vix1d = get_vix1d_at_time(test_date, day_df.iloc[0]['timestamp'].to_pydatetime())

        # Get first-hour and opening-range data
        fh_high, fh_low = get_first_hour_range(day_df)
        or_high, or_low = get_opening_range(day_df)
        price_945 = get_price_at_time(day_df, 9, 45)

        base_day_ctx = DayContext(
            prev_close=day_1.get('close', day_open_price),
            day_open=day_open_price,
            vix1d=vix1d,
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

        prev_close = base_day_ctx.prev_close

        # Get realized vol for this date (from pct_df)
        date_vols = pct_df[pct_df['date'] == test_date]['realized_vol']
        current_vol = date_vols.iloc[0] if not date_vols.empty and not pd.isna(date_vols.iloc[0]) else None

        # For each time slot
        for hour_et, minute_et in HALF_HOUR_SLOTS:
            time_label = f"{hour_et}:{minute_et:02d}"

            # Get current price at this slot
            current_price = get_price_at_slot(day_df, hour_et, minute_et)
            if current_price is None:
                continue

            # Get day high/low up to this time
            # Convert ET to approximate UTC for filtering
            for utc_offset in (5, 4):
                target_hour_utc = hour_et + utc_offset
                target_ts = day_df['timestamp'].iloc[0].replace(
                    hour=target_hour_utc, minute=minute_et, second=0
                )
                before = day_df[day_df['timestamp'] <= target_ts]
                if not before.empty:
                    break
            else:
                before = day_df

            if before.empty:
                continue

            day_high = before['high'].max()
            day_low = before['low'].min()

            # Build current_time as timezone-aware datetime
            dt = datetime.strptime(test_date, "%Y-%m-%d")
            current_time = dt.replace(
                hour=hour_et, minute=minute_et, tzinfo=ET_TZ
            )

            # Call make_unified_prediction to get all 3 band sets
            unified = make_unified_prediction(
                pct_df=pct_df,
                predictor=predictor,
                ticker=ticker,
                current_price=current_price,
                prev_close=prev_close,
                current_time=current_time,
                time_label=time_label,
                day_ctx=base_day_ctx,
                day_high=day_high,
                day_low=day_low,
                train_dates=train_date_set,
                current_vol=current_vol,
                vol_scale=True,
            )

            if unified is None:
                continue

            # Record results for each model type
            for model_name, bands in [
                ("percentile", unified.percentile_bands),
                ("statistical", unified.statistical_bands),
                ("combined", unified.combined_bands),
            ]:
                if not bands:
                    continue
                results.append(make_0dte_result(
                    date=test_date,
                    hour=time_label,
                    model=model_name,
                    bands=bands,
                    actual_close=actual_close,
                    current_price=current_price,
                    vix=vix1d,
                ))

        elapsed = time_mod.time() - t0
        if (test_idx + 1) % 5 == 0 or test_idx == len(test_dates) - 1:
            rate = (test_idx + 1) / elapsed if elapsed > 0 else 0
            eta = (len(test_dates) - test_idx - 1) / rate if rate > 0 else 0
            print(f"  [{test_idx+1}/{len(test_dates)}] {test_date} "
                  f"({elapsed:.0f}s elapsed, ~{eta:.0f}s remaining)")

    print(f"\n  Total 0DTE results: {len(results)}")
    return results


# ============================================================================
# Multi-day Backtest
# ============================================================================

@dataclass
class MultiDayResult:
    """Single multi-day prediction result."""
    date: str
    dte: int
    model: str  # baseline, conditional, ensemble, ensemble_combined

    # Band prices
    p95_lo: float
    p95_hi: float
    p97_lo: float
    p97_hi: float
    p98_lo: float
    p98_hi: float
    p99_lo: float
    p99_hi: float

    actual_close: float
    current_price: float
    actual_return_pct: float

    # Hit flags
    p95_hit: bool
    p97_hit: bool
    p98_hit: bool
    p99_hit: bool

    # Widths (% of current price)
    p95_width: float
    p97_width: float
    p98_width: float
    p99_width: float

    # Midpoint error (% of current price)
    midpoint_error: float

    vix: Optional[float] = None
    vol_regime: str = "medium"


def make_multiday_result(
    date: str,
    dte: int,
    model: str,
    bands: Dict[str, UnifiedBand],
    actual_close: float,
    current_price: float,
    context=None,
) -> MultiDayResult:
    """Build a MultiDayResult from band dict."""
    info = extract_band_info(bands, actual_close, current_price)
    actual_return = (actual_close - current_price) / current_price * 100
    return MultiDayResult(
        date=date,
        dte=dte,
        model=model,
        p95_lo=info["p95_lo"], p95_hi=info["p95_hi"],
        p97_lo=info["p97_lo"], p97_hi=info["p97_hi"],
        p98_lo=info["p98_lo"], p98_hi=info["p98_hi"],
        p99_lo=info["p99_lo"], p99_hi=info["p99_hi"],
        actual_close=actual_close,
        current_price=current_price,
        actual_return_pct=actual_return,
        p95_hit=info["p95_hit"], p97_hit=info["p97_hit"],
        p98_hit=info["p98_hit"], p99_hit=info["p99_hit"],
        p95_width=info["p95_width"], p97_width=info["p97_width"],
        p98_width=info["p98_width"], p99_width=info["p99_width"],
        midpoint_error=info["midpoint_error"],
        vix=context.vix if context else None,
        vol_regime=context.vol_regime if context else "medium",
    )


def run_multiday_backtest(
    ticker: str,
    test_days: int,
    train_days: int,
    dtes: List[int] = None,
) -> List[MultiDayResult]:
    """Run multi-day backtest across test dates and DTEs."""
    if dtes is None:
        dtes = [1, 2, 5, 10]
    max_dte = max(dtes)

    print(f"\n{'='*80}")
    print("PART 2: MULTI-DAY BACKTEST")
    print(f"{'='*80}\n")

    from scripts.close_predictor.multi_day_features import (
        compute_market_context,
        MarketContext,
    )
    from scripts.close_predictor.multi_day_predictor import (
        predict_with_conditional_distribution,
    )
    from scripts.close_predictor.multi_day_lgbm import MultiDayEnsemble

    # Load historical data
    total_lookback = test_days + train_days + max_dte + 30
    all_dates = get_available_dates(ticker, total_lookback)
    print(f"Loaded {len(all_dates)} trading days")

    data_by_date = {}
    for date_str in all_dates:
        df = load_csv_data(ticker, date_str)
        if df is not None and not df.empty:
            data_by_date[date_str] = df

    # Load VIX and VIX1D data
    print("Loading VIX and VIX1D data...")
    vix_data_by_date = {}
    vix1d_data_by_date = {}
    for date_str in all_dates:
        vdf = load_csv_data("VIX", date_str)
        if vdf is not None and not vdf.empty:
            vix_data_by_date[date_str] = vdf
        v1df = load_csv_data("VIX1D", date_str)
        if v1df is not None and not v1df.empty:
            vix1d_data_by_date[date_str] = v1df
    print(f"  VIX: {len(vix_data_by_date)} days, VIX1D: {len(vix1d_data_by_date)} days")

    # Compute forward returns
    print(f"Computing forward returns for DTEs {dtes}...")
    returns_by_dte: Dict[int, Dict[str, float]] = {}
    for dte in dtes:
        returns_by_dte[dte] = {}
        for i, start_date in enumerate(all_dates):
            if i + dte >= len(all_dates):
                continue
            end_date = all_dates[i + dte]
            if start_date not in data_by_date or end_date not in data_by_date:
                continue
            start_close = data_by_date[start_date].iloc[-1]['close']
            end_close = data_by_date[end_date].iloc[-1]['close']
            returns_by_dte[dte][start_date] = (end_close - start_close) / start_close * 100

    # Split train/test
    test_start_idx = len(all_dates) - test_days - max_dte
    test_dates = all_dates[test_start_idx:len(all_dates) - max_dte]
    train_end_idx = test_start_idx
    train_start_idx = max(0, train_end_idx - train_days)
    train_dates = all_dates[train_start_idx:train_end_idx]

    print(f"Train period: {train_dates[0]} to {train_dates[-1]} ({len(train_dates)} days)")
    print(f"Test period:  {test_dates[0]} to {test_dates[-1]} ({len(test_dates)} days)")

    # Compute market contexts
    print("Computing market contexts...")
    all_contexts: Dict[str, MarketContext] = {}
    for i, date_str in enumerate(all_dates):
        if date_str not in data_by_date:
            continue

        lookback_idx = max(0, i - 60)
        history_dates = all_dates[lookback_idx:i + 1]

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

        # VIX history
        vix_history = None
        vix_rows = []
        for d in history_dates:
            if d in vix_data_by_date:
                vdf = vix_data_by_date[d]
                if not vdf.empty:
                    vix_rows.append({'date': d, 'close': vdf.iloc[-1]['close']})
        if vix_rows:
            vix_history = pd.DataFrame(vix_rows)

        # VIX1D history
        vix1d_history = None
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

    print(f"  Computed contexts for {len(all_contexts)} dates")

    # Train LGBM ensemble
    print("\nTraining LGBM ensemble models...")
    ensemble = MultiDayEnsemble()
    train_contexts_by_date = {d: all_contexts[d] for d in train_dates if d in all_contexts}
    ensemble.train_all(
        all_dates=train_dates,
        contexts_by_date=train_contexts_by_date,
        returns_by_dte=returns_by_dte,
        max_dte=max_dte,
    )

    # Run backtest
    print(f"\nRunning backtest on {len(test_dates)} test days...")
    results = []
    t0 = time_mod.time()

    for test_idx, test_date in enumerate(test_dates):
        if test_date not in data_by_date or test_date not in all_contexts:
            continue

        current_price = data_by_date[test_date].iloc[-1]['close']
        current_context = all_contexts[test_date]

        for dte in dtes:
            if test_date not in returns_by_dte.get(dte, {}):
                continue

            forward_return = returns_by_dte[dte][test_date]
            actual_close = current_price * (1 + forward_return / 100)

            # Build historical N-day returns from training period
            train_returns = []
            train_contexts_list = []
            train_vols = []

            for td in train_dates:
                if td in returns_by_dte.get(dte, {}) and td in all_contexts:
                    train_returns.append(returns_by_dte[dte][td])
                    train_contexts_list.append(all_contexts[td])
                    train_vols.append(all_contexts[td].realized_vol_5d)

            if len(train_returns) < 50:
                continue

            # 1. Baseline prediction
            baseline_bands = map_percentile_to_bands(np.array(train_returns), current_price)
            results.append(make_multiday_result(
                test_date, dte, 'baseline', baseline_bands,
                actual_close, current_price, current_context,
            ))

            # 2. Conditional prediction
            conditional_bands = predict_with_conditional_distribution(
                ticker=ticker,
                days_ahead=dte,
                current_price=current_price,
                current_context=current_context,
                n_day_returns=train_returns,
                historical_contexts=train_contexts_list,
                historical_realized_vols=train_vols,
                use_weighting=True,
                use_regime_filter=True,
                use_vol_scaling=True,
            )
            results.append(make_multiday_result(
                test_date, dte, 'conditional', conditional_bands,
                actual_close, current_price, current_context,
            ))

            # 3. Ensemble LGBM prediction
            ensemble_bands = ensemble.predict(dte, current_context, current_price)
            if ensemble_bands:
                results.append(make_multiday_result(
                    test_date, dte, 'ensemble', ensemble_bands,
                    actual_close, current_price, current_context,
                ))

                # 4. Ensemble Combined: wider of conditional + ensemble
                combined_bands = {}
                for band_name in EVAL_BANDS:
                    if band_name in conditional_bands and band_name in ensemble_bands:
                        cb = conditional_bands[band_name]
                        eb = ensemble_bands[band_name]
                        combined_bands[band_name] = UnifiedBand(
                            name=band_name,
                            lo_price=min(cb.lo_price, eb.lo_price),
                            hi_price=max(cb.hi_price, eb.hi_price),
                            lo_pct=min(cb.lo_pct, eb.lo_pct),
                            hi_pct=max(cb.hi_pct, eb.hi_pct),
                            width_pts=max(cb.hi_price, eb.hi_price) - min(cb.lo_price, eb.lo_price),
                            width_pct=(max(cb.hi_price, eb.hi_price) - min(cb.lo_price, eb.lo_price)) / current_price * 100,
                            source="ensemble_combined",
                        )

                if combined_bands:
                    results.append(make_multiday_result(
                        test_date, dte, 'ensemble_combined', combined_bands,
                        actual_close, current_price, current_context,
                    ))

        if (test_idx + 1) % 10 == 0 or test_idx == len(test_dates) - 1:
            elapsed = time_mod.time() - t0
            print(f"  [{test_idx+1}/{len(test_dates)}] {test_date} ({elapsed:.0f}s)")

    print(f"\n  Total multi-day results: {len(results)}")
    return results


# ============================================================================
# Summarization and output
# ============================================================================

def summarize_0dte(results: List[ZeroDTEResult], test_days: int) -> pd.DataFrame:
    """Summarize 0DTE results by model, hour, and period."""
    if not results:
        return pd.DataFrame()

    df = pd.DataFrame([asdict(r) for r in results])
    all_dates_sorted = sorted(df['date'].unique())

    # Define periods: 1-month (last 22 days) and 3-month (all)
    periods = [("3mo", all_dates_sorted)]
    if len(all_dates_sorted) > 22:
        periods.append(("1mo", all_dates_sorted[-22:]))

    summary_rows = []
    for period_name, period_dates in periods:
        pdf = df[df['date'].isin(period_dates)]

        for model in sorted(pdf['model'].unique()):
            for hour in HALF_HOUR_LABELS:
                subset = pdf[(pdf['model'] == model) & (pdf['hour'] == hour)]
                if subset.empty:
                    continue

                summary_rows.append({
                    'model': model,
                    'hour': hour,
                    'period': period_name,
                    'n_samples': len(subset),
                    'p95_hit_rate': subset['p95_hit'].mean() * 100,
                    'p97_hit_rate': subset['p97_hit'].mean() * 100,
                    'p98_hit_rate': subset['p98_hit'].mean() * 100,
                    'p99_hit_rate': subset['p99_hit'].mean() * 100,
                    'p95_avg_width': subset['p95_width'].mean(),
                    'p97_avg_width': subset['p97_width'].mean(),
                    'p98_avg_width': subset['p98_width'].mean(),
                    'p99_avg_width': subset['p99_width'].mean(),
                    'avg_midpoint_error': subset['midpoint_error'].mean(),
                    'median_midpoint_error': subset['midpoint_error'].median(),
                })

    return pd.DataFrame(summary_rows)


def summarize_multiday(results: List[MultiDayResult], test_days: int) -> pd.DataFrame:
    """Summarize multi-day results by model, DTE, and period."""
    if not results:
        return pd.DataFrame()

    df = pd.DataFrame([asdict(r) for r in results])
    all_dates_sorted = sorted(df['date'].unique())

    periods = [("3mo", all_dates_sorted)]
    if len(all_dates_sorted) > 22:
        periods.append(("1mo", all_dates_sorted[-22:]))

    summary_rows = []
    for period_name, period_dates in periods:
        pdf = df[df['date'].isin(period_dates)]

        for model in sorted(pdf['model'].unique()):
            for dte in sorted(pdf['dte'].unique()):
                subset = pdf[(pdf['model'] == model) & (pdf['dte'] == dte)]
                if subset.empty:
                    continue

                summary_rows.append({
                    'model': model,
                    'dte': int(dte),
                    'period': period_name,
                    'n_samples': len(subset),
                    'p95_hit_rate': subset['p95_hit'].mean() * 100,
                    'p97_hit_rate': subset['p97_hit'].mean() * 100,
                    'p98_hit_rate': subset['p98_hit'].mean() * 100,
                    'p99_hit_rate': subset['p99_hit'].mean() * 100,
                    'p95_avg_width': subset['p95_width'].mean(),
                    'p97_avg_width': subset['p97_width'].mean(),
                    'p98_avg_width': subset['p98_width'].mean(),
                    'p99_avg_width': subset['p99_width'].mean(),
                    'avg_midpoint_error': subset['midpoint_error'].mean(),
                    'median_midpoint_error': subset['midpoint_error'].median(),
                })

    return pd.DataFrame(summary_rows)


def print_0dte_tables(summary_df: pd.DataFrame):
    """Print formatted 0DTE summary tables."""
    if summary_df.empty:
        return

    print(f"\n{'='*100}")
    print("0DTE SUMMARY TABLES")
    print(f"{'='*100}")

    for period in sorted(summary_df['period'].unique(), reverse=True):
        pdf = summary_df[summary_df['period'] == period]
        print(f"\n--- Period: {period} ---\n")

        # Table by model (averaged across hours)
        print(f"{'Model':<15} {'N':<6} {'P95 Hit%':<10} {'P97 Hit%':<10} {'P98 Hit%':<10} "
              f"{'P99 Hit%':<10} {'P97 Width':<10} {'Midpt Err':<10}")
        print(f"{'─'*81}")

        for model in sorted(pdf['model'].unique()):
            mdf = pdf[pdf['model'] == model]
            n = mdf['n_samples'].sum()
            # Weighted average by sample count
            w = mdf['n_samples']
            print(f"{model:<15} {n:<6} "
                  f"{np.average(mdf['p95_hit_rate'], weights=w):>7.1f}%  "
                  f"{np.average(mdf['p97_hit_rate'], weights=w):>7.1f}%  "
                  f"{np.average(mdf['p98_hit_rate'], weights=w):>7.1f}%  "
                  f"{np.average(mdf['p99_hit_rate'], weights=w):>7.1f}%  "
                  f"{np.average(mdf['p97_avg_width'], weights=w):>7.2f}%  "
                  f"{np.average(mdf['avg_midpoint_error'], weights=w):>7.2f}%")

        # Table by hour for combined model
        combined = pdf[pdf['model'] == 'combined']
        if not combined.empty:
            print(f"\n  Combined model by time slot:")
            print(f"  {'Hour':<8} {'N':<6} {'P95 Hit%':<10} {'P97 Hit%':<10} {'P98 Hit%':<10} "
                  f"{'P97 Width':<10} {'Midpt Err':<10}")
            print(f"  {'─'*70}")
            for _, row in combined.iterrows():
                print(f"  {row['hour']:<8} {row['n_samples']:<6.0f} "
                      f"{row['p95_hit_rate']:>7.1f}%  {row['p97_hit_rate']:>7.1f}%  "
                      f"{row['p98_hit_rate']:>7.1f}%  {row['p97_avg_width']:>7.2f}%  "
                      f"{row['avg_midpoint_error']:>7.2f}%")


def print_multiday_tables(summary_df: pd.DataFrame):
    """Print formatted multi-day summary tables."""
    if summary_df.empty:
        return

    print(f"\n{'='*100}")
    print("MULTI-DAY SUMMARY TABLES")
    print(f"{'='*100}")

    for period in sorted(summary_df['period'].unique(), reverse=True):
        pdf = summary_df[summary_df['period'] == period]
        print(f"\n--- Period: {period} ---\n")

        for dte in sorted(pdf['dte'].unique()):
            dte_df = pdf[pdf['dte'] == dte]
            print(f"\n  DTE {dte}:")
            print(f"  {'Model':<22} {'N':<6} {'P95 Hit%':<10} {'P97 Hit%':<10} {'P98 Hit%':<10} "
                  f"{'P99 Hit%':<10} {'P97 Width':<10} {'Midpt Err':<10}")
            print(f"  {'─'*88}")

            for _, row in dte_df.iterrows():
                print(f"  {row['model']:<22} {row['n_samples']:<6.0f} "
                      f"{row['p95_hit_rate']:>7.1f}%  {row['p97_hit_rate']:>7.1f}%  "
                      f"{row['p98_hit_rate']:>7.1f}%  {row['p99_hit_rate']:>7.1f}%  "
                      f"{row['p97_avg_width']:>7.2f}%  {row['avg_midpoint_error']:>7.2f}%")


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Comprehensive Prediction Backtest')
    parser.add_argument('--ticker', type=str, default='NDX', help='Ticker symbol')
    parser.add_argument('--test-days', type=int, default=66, help='Number of test days (default: 66 = ~3 months)')
    parser.add_argument('--train-days', type=int, default=250, help='Training lookback days')
    parser.add_argument('--skip-0dte', action='store_true', help='Skip 0DTE backtest')
    parser.add_argument('--skip-multiday', action='store_true', help='Skip multi-day backtest')
    parser.add_argument('--output-dir', type=Path, default=Path('results/comprehensive_backtest'),
                        help='Output directory')
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*80}")
    print(f"COMPREHENSIVE PREDICTION BACKTEST")
    print(f"{'='*80}\n")
    print(f"Ticker:       {args.ticker}")
    print(f"Test period:  Last {args.test_days} trading days")
    print(f"Training:     {args.train_days} days lookback")
    print(f"0DTE:         {'SKIP' if args.skip_0dte else '13 time slots x 3 models'}")
    print(f"Multi-day:    {'SKIP' if args.skip_multiday else 'DTEs 1,2,5,10 x 4 models'}")
    print(f"Output:       {args.output_dir}")

    total_start = time_mod.time()

    # ---- Part 1: 0DTE ----
    dte0_results = []
    if not args.skip_0dte:
        dte0_results = run_0dte_backtest(args.ticker, args.test_days, args.train_days)

        if dte0_results:
            # Save detailed CSV
            detail_df = pd.DataFrame([asdict(r) for r in dte0_results])
            detail_file = args.output_dir / f"0dte_detailed_{args.ticker}.csv"
            detail_df.to_csv(detail_file, index=False)
            print(f"\n  Saved 0DTE detailed results -> {detail_file}")

            # Save summary CSV
            summary_0dte = summarize_0dte(dte0_results, args.test_days)
            summary_file = args.output_dir / f"0dte_summary_{args.ticker}.csv"
            summary_0dte.to_csv(summary_file, index=False)
            print(f"  Saved 0DTE summary -> {summary_file}")

    # ---- Part 2: Multi-day ----
    md_results = []
    if not args.skip_multiday:
        md_results = run_multiday_backtest(args.ticker, args.test_days, args.train_days)

        if md_results:
            # Save detailed CSV
            detail_df = pd.DataFrame([asdict(r) for r in md_results])
            detail_file = args.output_dir / f"multiday_detailed_{args.ticker}.csv"
            detail_df.to_csv(detail_file, index=False)
            print(f"\n  Saved multi-day detailed results -> {detail_file}")

            # Save summary CSV
            summary_md = summarize_multiday(md_results, args.test_days)
            summary_file = args.output_dir / f"multiday_summary_{args.ticker}.csv"
            summary_md.to_csv(summary_file, index=False)
            print(f"  Saved multi-day summary -> {summary_file}")

    # ---- Print summary tables ----
    if dte0_results:
        print_0dte_tables(summarize_0dte(dte0_results, args.test_days))

    if md_results:
        print_multiday_tables(summarize_multiday(md_results, args.test_days))

    total_elapsed = time_mod.time() - total_start
    print(f"\n{'='*80}")
    print(f"COMPLETED in {total_elapsed:.0f}s ({total_elapsed/60:.1f} min)")
    print(f"{'='*80}\n")


if __name__ == '__main__':
    main()
