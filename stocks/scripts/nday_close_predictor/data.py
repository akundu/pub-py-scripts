"""
Data loading and feature engineering for the N-day forward close predictor.

Builds a daily feature matrix from:
  - Historical close prices (from existing CSVs)
  - VIX1D values (from existing VIX CSV data)
  - Derived: rolling returns, realized vol, MA deviations, regime labels
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from scripts.csv_prediction_backtest import (
    load_csv_data,
    get_available_dates,
    get_day_close,
    get_vix1d_at_time,
)


# ---------------------------------------------------------------------------
# VIX regime thresholds (consistent with band_selector.py)
# ---------------------------------------------------------------------------
VIX_REGIME_LOW      = 12.0
VIX_REGIME_NORMAL   = 20.0
VIX_REGIME_ELEVATED = 30.0

VIX_REGIME_LABELS = {0: "low_vol", 1: "normal", 2: "elevated", 3: "high_vol"}


def vix_to_regime(vix: float) -> int:
    if vix < VIX_REGIME_LOW:
        return 0
    elif vix < VIX_REGIME_NORMAL:
        return 1
    elif vix < VIX_REGIME_ELEVATED:
        return 2
    else:
        return 3


# ---------------------------------------------------------------------------
# Step 1: Load raw daily series (close + VIX) for all dates
# ---------------------------------------------------------------------------

def load_daily_series(ticker: str, all_dates: List[str]) -> pd.DataFrame:
    """
    Load daily close and opening VIX1D for every date in all_dates.

    Returns a DataFrame sorted by date with columns:
        date, close, vix
    Missing VIX is forward/backward filled.
    """
    records = []
    for date_str in all_dates:
        df = load_csv_data(ticker, date_str)
        if df is None or df.empty:
            continue

        close = get_day_close(df)

        # VIX at market open (first bar)
        try:
            first_ts = df.iloc[0]['timestamp'].to_pydatetime()
        except Exception:
            first_ts = None

        vix = get_vix1d_at_time(date_str, first_ts) if first_ts else None

        records.append({"date": date_str, "close": close, "vix": vix})

    result = pd.DataFrame(records)
    if result.empty:
        return result

    result = result.sort_values("date").reset_index(drop=True)

    # Forward-fill missing VIX (use previous day's value)
    result["vix"] = result["vix"].ffill().bfill().fillna(15.0)
    return result


# ---------------------------------------------------------------------------
# Step 2: Build full feature matrix with forward returns
# ---------------------------------------------------------------------------

def build_feature_matrix(
    daily: pd.DataFrame,
    horizons: List[int] = (1, 3, 7, 14),
) -> pd.DataFrame:
    """
    Compute features and N-day forward return targets for every date.

    Input: daily DataFrame from load_daily_series() (date, close, vix)
    Output: DataFrame with one row per date, columns:
        date, close, vix, vix_regime,
        return_1d, return_5d, return_21d,
        realized_vol_5d, realized_vol_21d, vol_ratio,
        dist_from_ma20, dist_from_ma50,
        above_ma200 (bool),
        day_of_week (0=Mon … 4=Fri),
        vix_5d_change,
        forward_return_{N}  for each N in horizons   (% return, NaN if not available)
    """
    if daily.empty or len(daily) < 25:
        return pd.DataFrame()

    closes = daily["close"].values
    vixes  = daily["vix"].values
    dates  = daily["date"].values
    n      = len(closes)

    rows = []

    for i in range(n):
        # Need at least 21 prior bars for all features
        if i < 21:
            continue

        c0 = closes[i]
        if c0 <= 0:
            continue

        # ---- Momentum (log returns for accuracy) ----
        ret_1d  = (closes[i] - closes[i-1])  / closes[i-1] * 100 if i >= 1  else 0.0
        ret_5d  = (closes[i] - closes[i-5])  / closes[i-5]  * 100 if i >= 5  else 0.0
        ret_21d = (closes[i] - closes[i-21]) / closes[i-21] * 100 if i >= 21 else 0.0

        # ---- Realized volatility (stdev of daily returns) ----
        rets_5  = [(closes[j] - closes[j-1]) / closes[j-1] * 100 for j in range(i-4, i+1)]
        rets_21 = [(closes[j] - closes[j-1]) / closes[j-1] * 100 for j in range(i-20, i+1)]
        rv5  = float(np.std(rets_5))
        rv21 = float(np.std(rets_21))
        vol_ratio = rv5 / rv21 if rv21 > 0 else 1.0

        # ---- Moving averages ----
        ma20  = float(np.mean(closes[i-19:i+1]))
        ma50  = float(np.mean(closes[i-49:i+1])) if i >= 49 else float(np.mean(closes[:i+1]))
        ma200 = float(np.mean(closes[i-199:i+1])) if i >= 199 else None

        dist_ma20 = (c0 - ma20)  / ma20  * 100
        dist_ma50 = (c0 - ma50)  / ma50  * 100
        above_200 = (c0 > ma200) if ma200 is not None else None

        # ---- VIX features ----
        vix_now = vixes[i]
        vix_5d_ago = vixes[i-5] if i >= 5 else vix_now
        vix_5d_change = vix_now - vix_5d_ago
        regime = vix_to_regime(vix_now)

        # ---- Calendar ----
        try:
            dow = datetime.strptime(dates[i], "%Y-%m-%d").weekday()  # 0=Mon, 4=Fri
        except Exception:
            dow = 2

        row = {
            "date":             dates[i],
            "close":            c0,
            "vix":              vix_now,
            "vix_regime":       regime,
            "vix_5d_change":    vix_5d_change,
            "return_1d":        ret_1d,
            "return_5d":        ret_5d,
            "return_21d":       ret_21d,
            "realized_vol_5d":  rv5,
            "realized_vol_21d": rv21,
            "vol_ratio":        vol_ratio,
            "dist_from_ma20":   dist_ma20,
            "dist_from_ma50":   dist_ma50,
            "above_ma200":      1 if above_200 else (0 if above_200 is not None else 0),
            "day_of_week":      dow,
        }

        # ---- Forward returns (targets) ----
        # horizons are in calendar days; find actual trading dates N+ cal days ahead
        for h in horizons:
            # Walk forward until we find a date >= date[i] + h cal days
            target_dt_str = _add_calendar_days(dates[i], h)
            fwd_close = _find_forward_close(closes, dates, i, target_dt_str)
            row[f"forward_return_{h}d"] = (
                (fwd_close - c0) / c0 * 100 if fwd_close is not None else np.nan
            )

        rows.append(row)

    return pd.DataFrame(rows).reset_index(drop=True)


def _add_calendar_days(date_str: str, days: int) -> str:
    """Return date_str + days calendar days as 'YYYY-MM-DD'."""
    from datetime import timedelta
    d = datetime.strptime(date_str, "%Y-%m-%d") + timedelta(days=days)
    return d.strftime("%Y-%m-%d")


def _find_forward_close(
    closes: np.ndarray,
    dates: np.ndarray,
    from_idx: int,
    target_date_str: str,
) -> Optional[float]:
    """
    Find the close price on the first trading date >= target_date_str.
    Returns None if beyond available data.
    """
    for j in range(from_idx + 1, len(dates)):
        if dates[j] >= target_date_str:
            return float(closes[j])
    return None


# ---------------------------------------------------------------------------
# Feature column list (used by model training + prediction)
# ---------------------------------------------------------------------------

FEATURE_COLS = [
    "vix",
    "vix_regime",
    "vix_5d_change",
    "return_1d",
    "return_5d",
    "return_21d",
    "realized_vol_5d",
    "realized_vol_21d",
    "vol_ratio",
    "dist_from_ma20",
    "dist_from_ma50",
    "above_ma200",
    "day_of_week",
]
