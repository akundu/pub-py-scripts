"""VMaxMin.1 — Dynamic Mean-Reversion Credit Spread Tracker (Core Engine).

Sells a 0DTE OTM or ITM credit spread at market open, optionally rolls
throughout the day, and protects at EOD via DTE+1 roll or stop loss.

Configurable dimensions:
  - leg_placement: "otm" or "itm"
  - depth_pct: how far OTM/ITM to place legs (None = closest)
  - min_spread_width: force minimum width ($)
  - stop_loss_mode: "credit_multiple" or "width_pct" (None = disabled)
  - sizing_mode: "budget" or "credit_multiple"
  - roll_mode: "none", "eod_itm", "midday_dte1", "conditional_dte1"
"""

import glob
import math
import os
from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

try:
    import pytz
    US_PACIFIC = pytz.timezone("US/Pacific")
except ImportError:
    US_PACIFIC = None


# ── Time Helpers ─────────────────────────────────────────────────────────────

def _utc_to_pacific(ts_utc) -> str:
    """Convert UTC timestamp to Pacific HH:MM string (DST-aware)."""
    if US_PACIFIC is not None:
        local = ts_utc.astimezone(US_PACIFIC)
        return f"{local.hour:02d}:{local.minute:02d}"
    h = (ts_utc.hour - 8) % 24
    return f"{h:02d}:{ts_utc.minute:02d}"


def _time_to_mins(t: str) -> int:
    h, m = t.split(":")
    return int(h) * 60 + int(m)


# ── Default Config ───────────────────────────────────────────────────────────

DEFAULT_CONFIG = {
    "entry_time_pacific": "06:35",
    "max_per_transaction": 50000,
    "commission_per_transaction": 10,
    "num_contracts": None,

    # Leg placement & depth
    "leg_placement": "otm",       # "otm" or "itm"
    "depth_pct": None,            # None = closest strike, float = % distance from price

    # Width
    "min_spread_width": None,     # None = use min_width_steps, float = force min width in $
    "min_width_steps": {"SPX": 5, "NDX": 10, "RUT": 5},
    "max_width_multiplier": 10,

    # Stop loss
    "stop_loss_mode": None,       # None, "credit_multiple", "width_pct"
    "stop_loss_value": None,      # e.g. 2 for 2x credit, or 0.50 for 50% of width

    # Position sizing
    "sizing_mode": "budget",      # "budget" or "credit_multiple"
    "sizing_credit_multiple": None,  # e.g. 10 = max_loss = 10 x credit
    "max_contracts": 50,             # hard cap on contract count

    # Roll design
    "roll_mode": "eod_itm",       # "none", "eod_itm", "midday_dte1", "conditional_dte1"
    "roll_check_times_pacific": ["07:35", "08:35", "09:35", "10:35", "11:35"],
    "midday_roll_breach_pct": 0.50,        # for midday: roll if breached > X% of width
    "conditional_roll_min_recovery": 0.50,  # for conditional: new credit >= X% of close debit

    # Legacy (mapped to roll_mode internally)
    "intraday_rolls": False,
    "eod_roll": True,

    # Proximity window (for eod_itm mode)
    "proximity_check_start_pacific": "12:30",
    "proximity_check_end_pacific": "12:50",
    "proximity_check_interval_mins": 5,
    "proximity_threshold_pct": 0.005,

    # Call-track mode: always sell call spread, roll up when price > short strike
    "strategy_mode": "directional",            # "directional" (existing) or "call_track"
    "call_track_check_times_pacific": [                     # Intraday checks (12:45 EOD is separate)
        "07:05", "07:35", "08:05", "08:35",
        "09:05", "09:35", "10:05", "10:35",
        "11:05", "11:35", "12:05", "12:35",
    ],
    "call_track_roll_interval_hours": 2,       # Fallback if check_times not set
    "call_track_roll_budget_pct": 0.25,        # Max 25% of original credit spent on rolls
    "call_track_eod_time_pacific": "12:45",    # EOD proximity check time
    "call_track_eod_proximity_pct": 0.003,     # 0.3% — roll to DTE+1 if within this
    "call_track_unlimited_budget": False,      # True = no cap on roll costs
    "call_track_leg_placement": "best_roi",      # "best_roi" = highest ROI strike, "nearest", "itm", "otm"
    "call_track_depth_pct": 0.003,             # min distance from price for strike placement (0.3%)

    # Layer mode enhancements
    "layer_dual_entry": True,                  # Open both call + put at entry (default True)
    "layer_entry_directions": "both",          # "both", "call", "put" — which spreads to open at entry
    "layer_eod_exit_pct": 0.0,                 # % of width breached to trigger EOD roll (0 = any ITM)
    "max_roll_count": 99,                      # Stop rolling after N (99 = effectively infinite)
    "roll_recovery_threshold": 99.0,           # Stop if cumulative_cost >= original_credit * this
    "roll_match_contracts": False,             # False = size roll to cover close debit (credit-neutral)
    "roll_max_width_mult": 5,                  # Max width on roll = original_width × this
    "roll_max_contract_mult": 2,               # Max contracts on roll = original_count × this
    "roll_max_chain_contracts": None,           # None = no cap (best_roi entries limit snowball naturally)

    # Layer breach threshold: new HOD/LOD must exceed previous by this much
    "layer_breach_min_points": None,           # None = use min_width_step (e.g. 5 for RUT)

    # Entry window: scan this range for best spread (not just one snapshot)
    "layer_entry_window_start": "06:30",       # Start of entry scan window
    "layer_entry_window_end": "06:45",         # End of entry scan window

    # Adaptive ROI entry: aspiration starts high and relaxes to floor at window end
    "layer_entry_min_roi": 0.50,               # Starting aspiration (50% ROI)
    "layer_entry_min_roi_floor": 0.0,          # Floor at window end (0 = accept anything)

    # Percentile-triggered entry (strategy_mode = "percentile_layer")
    "percentile_entry_pN": 75,                 # Which percentile band triggers entry
    "percentile_lookback": 120,                # Trading days of history for percentile calc
    "percentile_spread_width": 5,              # Forced spread width ($)
    "percentile_layering": True,               # Enable/disable HOD/LOD layering after trigger
    "percentile_layer_cutoff": "11:45",        # No layering after this time
    "percentile_roll_time": "12:30",           # Roll check time
    "percentile_roll_proximity": 0.003,        # 0.3% proximity to trigger roll
    "percentile_roll_dte": 2,                  # Roll to DTE+N

    # VIX filtering for percentile_layer mode
    "percentile_vix_enabled": False,           # Enable VIX-based entry filtering
    "percentile_vix_min": None,                # Skip entry if VIX open < this (e.g., 18)
    "percentile_vix_max": None,                # Skip entry if VIX open > this (e.g., 35)
    "percentile_vix_regime_skip": [],          # Skip entry on these regimes (e.g., ["extreme"])
    "percentile_vix_scale_contracts": False,   # Scale contracts by VIX regime multiplier

    # EOD scan: check every minute from scan_start to scan_end
    "layer_eod_scan_start": "12:50",           # Start scanning
    "layer_eod_scan_end": "13:00",             # Stop scanning (positions not rolled settle here)
    "layer_eod_proximity": 0.002,              # Roll if ITM or within 0.2% of short strike

    # Data dirs
    "equity_dir": "equities_output",
    "options_0dte_dir": "options_csv_output_full",
    "options_dte1_dir": "csv_exports/options",
}

TICKER_START_DATES = {"RUT": "2025-01-02", "SPX": "2026-02-15", "NDX": "2026-02-15"}


# ── Data Structures ──────────────────────────────────────────────────────────

@dataclass
class SpreadPosition:
    """Represents an open credit spread position."""
    direction: str
    short_strike: float
    long_strike: float
    width: float
    credit_per_share: float
    num_contracts: int
    entry_time: str
    entry_price: float
    dte: int = 0

    @property
    def total_credit(self) -> float:
        return self.credit_per_share * 100 * self.num_contracts

    @property
    def max_loss(self) -> float:
        return (self.width - self.credit_per_share) * 100 * self.num_contracts


@dataclass
class TradeRecord:
    event: str
    time_pacific: str
    direction: str
    short_strike: float
    long_strike: float
    width: float
    credit_or_debit: float
    num_contracts: int
    commission: float
    underlying_price: float
    dte: int = 0
    notes: str = ""


@dataclass
class RolledPosition:
    """Tracks a position that was rolled to DTE+N for multi-day evaluation."""
    direction: str              # "call" or "put"
    short_strike: float
    long_strike: float
    width: float
    credit_per_share: float     # credit from this DTE+N leg
    num_contracts: int
    expiration_date: str        # YYYY-MM-DD when this leg expires

    # Lifecycle
    original_entry_date: str    # the day the very first 0DTE was opened
    original_credit: float      # total $ credit from original 0DTE entry
    cumulative_roll_cost: float # sum of (close_debit - new_credit) across all rolls
    roll_count: int             # 1 = first roll, 2 = second, etc.


@dataclass
class DayResult:
    ticker: str
    date: str
    direction: str = ""
    trades: List[TradeRecord] = field(default_factory=list)
    total_credits: float = 0.0
    total_debits: float = 0.0
    total_commissions: float = 0.0
    num_rolls: int = 0
    eod_rolled_to_dte1: bool = False
    stopped_out: bool = False
    final_pnl: float = 0.0
    close_price: float = 0.0
    open_price: float = 0.0
    hod: float = 0.0
    lod: float = 0.0
    failure_reason: str = ""

    @property
    def net_pnl(self) -> float:
        return self.total_credits - self.total_debits - self.total_commissions


# ── Data Loading ─────────────────────────────────────────────────────────────

def load_equity_bars_df(ticker: str, trade_date: str, equity_dir: str) -> pd.DataFrame:
    path = os.path.join(equity_dir, f"I:{ticker}",
                        f"I:{ticker}_equities_{trade_date}.csv")
    if not os.path.exists(path):
        return pd.DataFrame()
    try:
        df = pd.read_csv(path)
    except Exception:
        return pd.DataFrame()
    if df.empty:
        return df
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df["time_pacific"] = df["timestamp"].apply(_utc_to_pacific)
    df["time_mins"] = df["time_pacific"].apply(_time_to_mins)
    return df


def load_equity_prices(ticker: str, trade_date: str, equity_dir: str) -> dict:
    df = load_equity_bars_df(ticker, trade_date, equity_dir)
    if df.empty:
        return {}
    return {row["time_pacific"]: float(row["close"]) for _, row in df.iterrows()}


def get_price_at_time(prices: dict, target: str, tolerance_mins: int = 10) -> Optional[float]:
    if not prices:
        return None
    target_mins = _time_to_mins(target)
    best_key, best_diff = None, 999999
    for k in prices:
        diff = target_mins - _time_to_mins(k)
        if 0 <= diff < best_diff:
            best_diff = diff
            best_key = k
    if best_key is None or best_diff > tolerance_mins:
        return None
    return prices[best_key]


def get_hod_lod_in_range(equity_df: pd.DataFrame,
                         start_mins: int, end_mins: int) -> Tuple[float, float]:
    mask = (equity_df["time_mins"] >= start_mins) & (equity_df["time_mins"] <= end_mins)
    subset = equity_df[mask]
    if subset.empty:
        return 0.0, float("inf")
    return float(subset["high"].max()), float(subset["low"].min())


def load_0dte_options(ticker: str, trade_date: str,
                      options_dir: str) -> Optional[pd.DataFrame]:
    path = os.path.join(options_dir, ticker, f"{ticker}_options_{trade_date}.csv")
    if not os.path.exists(path):
        return None
    try:
        df = pd.read_csv(path)
    except Exception:
        return None
    if df.empty:
        return None
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df["time_pacific"] = df["timestamp"].apply(_utc_to_pacific)
    for col in ["bid", "ask", "strike", "volume"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
    return df


def load_dte1_options(ticker: str, expiration_date: str, snapshot_date: str,
                      csv_exports_dir: str,
                      fallback_options_dir: str = None) -> Optional[pd.DataFrame]:
    # Primary: csv_exports format ({dir}/{ticker}/{expiration_date}.csv)
    path = os.path.join(csv_exports_dir, ticker, f"{expiration_date}.csv")
    if not os.path.exists(path) and fallback_options_dir:
        # Fallback: load next-day 0DTE file and filter for the expiration we need
        # The snapshot_date file may contain options expiring on expiration_date
        fb_path = os.path.join(fallback_options_dir, ticker,
                               f"{ticker}_options_{snapshot_date}.csv")
        if os.path.exists(fb_path):
            path = fb_path
        else:
            # Or load the expiration_date file (next day's data at open)
            fb_path2 = os.path.join(fallback_options_dir, ticker,
                                    f"{ticker}_options_{expiration_date}.csv")
            if os.path.exists(fb_path2):
                path = fb_path2
            else:
                return None
    if not os.path.exists(path):
        return None
    try:
        df = pd.read_csv(path)
    except Exception:
        return None
    if df.empty:
        return None
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    # Filter to the target expiration if 'expiration' column exists
    if "expiration" in df.columns:
        df_exp = df[df["expiration"] == expiration_date].copy()
        if not df_exp.empty:
            df = df_exp
    df["time_pacific"] = df["timestamp"].apply(_utc_to_pacific)
    for col in ["bid", "ask", "strike", "volume"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
    return df


def snap_options_to_time(options_df: pd.DataFrame, target_time: str,
                         tolerance_mins: int = 10) -> pd.DataFrame:
    target_mins = _time_to_mins(target_time)
    times = options_df["time_pacific"].unique()
    best_time, best_diff = None, 999999
    for t in times:
        diff = abs(_time_to_mins(t[:5]) - target_mins)
        if diff < best_diff:
            best_diff = diff
            best_time = t
    if best_time is None or best_diff > tolerance_mins:
        return pd.DataFrame()
    return options_df[options_df["time_pacific"] == best_time].copy()


def get_trading_dates(ticker: str, equity_dir: str, start_date: str,
                      end_date: str) -> List[str]:
    d = os.path.join(equity_dir, f"I:{ticker}")
    files = sorted(glob.glob(os.path.join(d, "*.csv")))
    dates = []
    for f in files:
        try:
            dt = os.path.basename(f).split("_equities_")[1].replace(".csv", "")
            if start_date <= dt <= end_date:
                dates.append(dt)
        except Exception:
            continue
    return sorted(dates)


def get_prev_close(ticker: str, trade_date: str, all_dates: List[str],
                   equity_dir: str) -> Optional[float]:
    try:
        idx = all_dates.index(trade_date)
    except ValueError:
        return None
    if idx == 0:
        return None
    prev_date = all_dates[idx - 1]
    prices = load_equity_prices(ticker, prev_date, equity_dir)
    if not prices:
        return None
    sorted_times = sorted(prices.keys(), key=_time_to_mins)
    return prices[sorted_times[-1]] if sorted_times else None


def get_next_trading_date(trade_date: str, all_dates: List[str]) -> Optional[str]:
    try:
        idx = all_dates.index(trade_date)
    except ValueError:
        return None
    if idx + 1 >= len(all_dates):
        return None
    return all_dates[idx + 1]


# ── Spread Construction ──────────────────────────────────────────────────────

def filter_valid_quotes(options_snap: pd.DataFrame, direction: str) -> pd.DataFrame:
    if options_snap.empty or "type" not in options_snap.columns:
        return pd.DataFrame()
    side = options_snap[options_snap["type"] == direction].copy()
    if side.empty:
        return side
    side = side[(side["bid"] > 0) & (side["ask"] > 0) & (side["bid"] < side["ask"])]
    if side.empty:
        return side
    side["_spread"] = side["ask"] - side["bid"]
    side = (side.sort_values(["volume", "_spread"], ascending=[False, True])
            .drop_duplicates("strike", keep="first")
            .drop(columns=["_spread"]))
    return side


def find_credit_spread(options_snap: pd.DataFrame, current_price: float,
                       direction: str, min_step: float, max_width: float,
                       otm_target_price: Optional[float] = None,
                       leg_placement: str = "otm",
                       depth_pct: Optional[float] = None) -> Optional[Dict]:
    """Find narrowest credit spread with positive net credit.

    Args:
        depth_pct: Minimum % distance from price for strike placement.
                   For OTM calls: short >= price * (1 + depth_pct).
                   For ITM calls: long <= price * (1 - depth_pct).
    """
    target = otm_target_price if otm_target_price is not None else current_price
    side = filter_valid_quotes(options_snap, direction)
    if len(side) < 2:
        return None

    if leg_placement == "itm":
        return _find_itm_spread(side, target, direction, min_step, max_width, depth_pct)
    elif leg_placement == "nearest":
        return _find_nearest_spread(side, target, direction, min_step, max_width)
    elif leg_placement == "best_roi":
        proximity = depth_pct if depth_pct else 0.003
        return _find_best_roi_spread(options_snap, target, direction, min_step,
                                     max_width, proximity_pct=proximity)
    else:
        return _find_otm_spread(side, target, direction, min_step, max_width, depth_pct)


def _find_otm_spread(side: pd.DataFrame, target: float, direction: str,
                     min_step: float, max_width: float,
                     depth_pct: Optional[float] = None,
                     max_credit_pct: float = 0.30) -> Optional[Dict]:
    """Both legs OTM: calls above price, puts below price."""
    if direction == "call":
        min_strike = target * (1 + depth_pct) if depth_pct else target
        short_candidates = side[side["strike"] >= min_strike].sort_values("strike")
    else:
        max_strike = target * (1 - depth_pct) if depth_pct else target
        short_candidates = side[side["strike"] <= max_strike].sort_values(
            "strike", ascending=False)

    for _, short_row in short_candidates.head(10).iterrows():
        short_strike = float(short_row["strike"])
        short_bid = float(short_row["bid"])

        for width_mult in range(1, int(max_width / min_step) + 1):
            width = min_step * width_mult
            if direction == "call":
                long_strike = short_strike + width
            else:
                long_strike = short_strike - width

            long_row = side[side["strike"] == long_strike]
            if long_row.empty:
                continue
            long_ask = float(long_row.iloc[0]["ask"])

            credit = short_bid - long_ask
            if 0 < credit < width * max_credit_pct:
                return {
                    "short_strike": short_strike,
                    "long_strike": long_strike,
                    "short_bid": short_bid,
                    "long_ask": long_ask,
                    "credit": credit,
                    "width": width,
                }

    return None


def _find_nearest_spread(side: pd.DataFrame, target: float, direction: str,
                         min_step: float, max_width: float,
                         max_credit_pct: float = 0.70) -> Optional[Dict]:
    """Find best credit spread: highest credit, narrowest width, nearest to target.

    Evaluates ALL valid strike pairs within range and picks the best one.
    For calls: short at lower strike, long at higher strike.
    For puts: short at higher strike, long at lower strike.
    Credit capped at max_credit_pct of width to reject bad/stale quotes.
    """
    strikes = sorted(side["strike"].unique())
    if len(strikes) < 2:
        return None

    # Build lookup for bid/ask by strike
    bids = {}
    asks = {}
    for _, row in side.iterrows():
        s = float(row["strike"])
        bids[s] = float(row["bid"])
        asks[s] = float(row["ask"])

    # Evaluate all valid pairs near the target
    candidates = []
    near_strikes = [s for s in strikes if abs(s - target) <= max_width * 2]

    for short_strike in near_strikes:
        short_bid = bids[short_strike]
        for width_mult in range(1, int(max_width / min_step) + 1):
            width = min_step * width_mult
            if direction == "call":
                long_strike = short_strike + width
            else:
                long_strike = short_strike - width

            if long_strike not in asks:
                continue
            long_ask = asks[long_strike]

            credit = short_bid - long_ask
            if 0 < credit < width * max_credit_pct:
                dist = abs(short_strike - target)
                candidates.append({
                    "short_strike": short_strike,
                    "long_strike": long_strike,
                    "short_bid": short_bid,
                    "long_ask": long_ask,
                    "credit": credit,
                    "width": width,
                    "_dist": dist,
                })

    if not candidates:
        return None

    # Sort: narrowest width, then closest to target, then highest credit
    candidates.sort(key=lambda c: (c["width"], c["_dist"], -c["credit"]))
    best = candidates[0]
    del best["_dist"]
    return best


def _find_best_roi_spread(options_snap: pd.DataFrame, target: float,
                          direction: str, min_step: float, max_width: float,
                          proximity_pct: float = 0.003,
                          max_credit_pct: float = 0.70) -> Optional[Dict]:
    """Find best ROI spread within proximity of target price.

    Evaluates ALL valid strike pairs where the short strike is within
    proximity_pct of target. Legs can be ITM, OTM, or mixed — doesn't matter.
    Picks: narrowest width first, then highest ROI.

    ROI = credit / (width - credit).
    """
    side = filter_valid_quotes(options_snap, direction)
    if len(side) < 2:
        return None

    strikes = sorted(side["strike"].unique())
    bids = {}
    asks = {}
    for _, row in side.iterrows():
        s = float(row["strike"])
        bids[s] = float(row["bid"])
        asks[s] = float(row["ask"])

    # Short strike must be within proximity_pct of target
    proximity_pts = target * proximity_pct
    near_strikes = [s for s in strikes if abs(s - target) <= proximity_pts]

    candidates = []
    for short_strike in near_strikes:
        short_bid = bids[short_strike]
        for width_mult in range(1, int(max_width / min_step) + 1):
            width = min_step * width_mult
            if direction == "call":
                long_strike = short_strike + width
            else:
                long_strike = short_strike - width

            if long_strike not in asks:
                continue
            long_ask = asks[long_strike]

            credit = short_bid - long_ask
            if credit <= 0 or credit >= width * max_credit_pct:
                continue

            roi = credit / (width - credit)
            candidates.append({
                "short_strike": short_strike,
                "long_strike": long_strike,
                "short_bid": short_bid,
                "long_ask": long_ask,
                "credit": credit,
                "width": width,
                "_roi": roi,
            })

    if not candidates:
        return None

    # Sort: narrowest width first, then highest ROI
    candidates.sort(key=lambda c: (c["width"], -c["_roi"]))
    best = candidates[0]
    del best["_roi"]
    return best


def _find_itm_spread(side: pd.DataFrame, target: float, direction: str,
                     min_step: float, max_width: float,
                     depth_pct: Optional[float] = None) -> Optional[Dict]:
    """Both legs ITM: calls below price, puts above price."""
    if direction == "call":
        max_long = target * (1 - depth_pct) if depth_pct else target
        long_candidates = side[side["strike"] <= max_long].sort_values(
            "strike", ascending=False)

        for _, long_row in long_candidates.head(10).iterrows():
            long_strike = float(long_row["strike"])
            long_ask = float(long_row["ask"])

            for width_mult in range(1, int(max_width / min_step) + 1):
                width = min_step * width_mult
                short_strike = long_strike - width
                if short_strike <= 0:
                    continue
                short_row = side[side["strike"] == short_strike]
                if short_row.empty:
                    continue
                short_bid = float(short_row.iloc[0]["bid"])

                credit = short_bid - long_ask
                if 0 < credit < width:  # credit >= width means bad quotes
                    return {
                        "short_strike": short_strike,
                        "long_strike": long_strike,
                        "short_bid": short_bid,
                        "long_ask": long_ask,
                        "credit": credit,
                        "width": width,
                    }

    else:  # put
        min_long = target * (1 + depth_pct) if depth_pct else target
        long_candidates = side[side["strike"] >= min_long].sort_values("strike")

        for _, long_row in long_candidates.head(10).iterrows():
            long_strike = float(long_row["strike"])
            long_ask = float(long_row["ask"])

            for width_mult in range(1, int(max_width / min_step) + 1):
                width = min_step * width_mult
                short_strike = long_strike + width
                short_row = side[side["strike"] == short_strike]
                if short_row.empty:
                    continue
                short_bid = float(short_row.iloc[0]["bid"])

                credit = short_bid - long_ask
                if 0 < credit < width:  # credit >= width means bad quotes
                    return {
                        "short_strike": short_strike,
                        "long_strike": long_strike,
                        "short_bid": short_bid,
                        "long_ask": long_ask,
                        "credit": credit,
                        "width": width,
                    }

    return None


def find_debit_spread(options_snap: pd.DataFrame, current_price: float,
                      direction: str, min_step: float, max_width: float,
                      depth_pct: Optional[float] = None) -> Optional[Dict]:
    """Find a debit spread (long spread) near ATM.

    A debit spread profits when the underlying moves in the spread direction.
    - Call debit spread (bull): buy lower call (near ATM), sell higher call.
      Debit = long_ask - short_bid. Max profit = width - debit.
    - Put debit spread (bear): buy higher put (near ATM), sell lower put.
      Debit = long_ask - short_bid. Max profit = width - debit.

    Returns dict with: long_strike, short_strike, debit, width, max_profit.
    """
    side = filter_valid_quotes(options_snap, direction)
    if len(side) < 2:
        return None

    strikes = sorted(side["strike"].unique())
    bids = {}
    asks = {}
    for _, row in side.iterrows():
        s = float(row["strike"])
        bids[s] = float(row["bid"])
        asks[s] = float(row["ask"])

    # Long strike near ATM (within depth_pct or closest)
    if depth_pct:
        proximity = current_price * depth_pct
    else:
        proximity = current_price * 0.005  # default 0.5%

    candidates = []
    for long_strike in strikes:
        if abs(long_strike - current_price) > proximity * 3:
            continue
        long_ask = asks.get(long_strike)
        if long_ask is None or long_ask <= 0:
            continue

        for width_mult in range(1, int(max_width / min_step) + 1):
            width = min_step * width_mult
            if direction == "call":
                short_strike = long_strike + width  # sell higher call
            else:
                short_strike = long_strike - width  # sell lower put

            short_bid = bids.get(short_strike)
            if short_bid is None or short_bid <= 0:
                continue

            debit = long_ask - short_bid
            if debit <= 0 or debit >= width:
                continue  # no valid debit or overpaying

            max_profit = width - debit
            roi = max_profit / debit if debit > 0 else 0
            dist = abs(long_strike - current_price)

            candidates.append({
                "long_strike": long_strike,
                "short_strike": short_strike,
                "long_ask": long_ask,
                "short_bid": short_bid,
                "debit": debit,
                "width": width,
                "max_profit": max_profit,
                "credit": -debit,  # negative for compatibility
                "_roi": roi,
                "_dist": dist,
            })

    if not candidates:
        return None

    # Prefer: narrowest width, then closest to ATM, then best ROI
    candidates.sort(key=lambda c: (c["width"], c["_dist"], -c["_roi"]))
    best = candidates[0]
    del best["_roi"]
    del best["_dist"]
    return best


def close_spread_cost(options_snap: pd.DataFrame, position: SpreadPosition) -> Optional[float]:
    """Debit per share to close (positive number), or None."""
    side = filter_valid_quotes(options_snap, position.direction)
    if side.empty:
        return None
    short_row = side[side["strike"] == position.short_strike]
    long_row = side[side["strike"] == position.long_strike]
    if short_row.empty or long_row.empty:
        return None
    debit = float(short_row.iloc[0]["ask"]) - float(long_row.iloc[0]["bid"])
    return max(debit, 0)


# ── Core Engine ──────────────────────────────────────────────────────────────

class VMaxMinEngine:
    """Runs the vMaxMin.1 strategy for a single trading day."""

    def __init__(self, config: Optional[Dict] = None):
        self.config = {**DEFAULT_CONFIG, **(config or {})}

    def _get_min_step(self, ticker: str) -> float:
        steps = self.config["min_width_steps"]
        return steps.get(ticker, 5)

    def _get_effective_min_width(self, ticker: str) -> float:
        forced = self.config.get("min_spread_width")
        if forced:
            return float(forced)
        return self._get_min_step(ticker)

    def _get_max_width(self, ticker: str) -> float:
        return self._get_min_step(ticker) * self.config["max_width_multiplier"]

    def _calc_contracts(self, credit_per_share: float, width: float) -> int:
        explicit = self.config.get("num_contracts")
        if explicit:
            return explicit

        cap = self.config.get("max_contracts", 50)

        sizing = self.config.get("sizing_mode", "budget")
        if sizing == "credit_multiple":
            mult = self.config.get("sizing_credit_multiple", 10)
            if mult and credit_per_share > 0:
                target_max_loss = credit_per_share * 100 * mult
                max_loss_per_contract = (width - credit_per_share) * 100
                if max_loss_per_contract <= 0:
                    return 1
                return min(cap, max(1, int(target_max_loss / max_loss_per_contract)))

        # Default: budget / credit (invest total budget, recoup via credit)
        budget = self.config["max_per_transaction"]
        credit_per_contract = credit_per_share * 100
        if credit_per_contract <= 0:
            return 1
        return min(cap, max(1, int(budget / credit_per_contract)))

    def _calc_roll_contracts(self, close_debit_total: float,
                             new_credit_per_share: float,
                             new_width: float) -> Optional[int]:
        new_credit_per_contract = new_credit_per_share * 100
        if new_credit_per_contract <= 0:
            return None
        needed = math.ceil(close_debit_total / new_credit_per_contract)
        if needed < 1:
            needed = 1
        max_loss_per_contract = (new_width - new_credit_per_share) * 100
        if max_loss_per_contract > 0:
            budget_cap = int(self.config["max_per_transaction"] / max_loss_per_contract)
            needed = min(needed, max(1, budget_cap))
        return needed

    def _check_stop_loss(self, position: SpreadPosition,
                         options_snap: pd.DataFrame) -> Optional[float]:
        """Check if stop loss triggered. Returns close debit/share if yes, else None."""
        mode = self.config.get("stop_loss_mode")
        value = self.config.get("stop_loss_value")
        if not mode or value is None:
            return None

        debit = close_spread_cost(options_snap, position)
        if debit is None:
            return None

        unrealized_loss_per_share = debit - position.credit_per_share

        if mode == "credit_multiple":
            threshold = position.credit_per_share * value
            if unrealized_loss_per_share > threshold:
                return debit
        elif mode == "width_pct":
            threshold = position.width * value
            if unrealized_loss_per_share > threshold:
                return debit

        return None

    def _is_position_itm(self, position: SpreadPosition, price: float) -> bool:
        if position.direction == "call":
            return price > position.short_strike
        else:
            return price < position.short_strike

    def _breach_pct(self, position: SpreadPosition, price: float) -> float:
        """How far price has breached past short strike as fraction of width."""
        if position.direction == "call":
            breach = price - position.short_strike
        else:
            breach = position.short_strike - price
        return breach / position.width if position.width > 0 else 0

    def _resolve_roll_mode(self) -> str:
        """Map legacy flags to roll_mode."""
        rm = self.config.get("roll_mode")
        if rm and rm != "eod_itm":
            return rm
        # Legacy compat
        if not self.config.get("eod_roll", True):
            return "none"
        return self.config.get("roll_mode", "eod_itm")

    def run_single_day(self, ticker: str, trade_date: str,
                       equity_df: pd.DataFrame, equity_prices: dict,
                       options_0dte: Optional[pd.DataFrame],
                       all_dates: List[str],
                       prev_close: Optional[float],
                       carried_positions: Optional[List["RolledPosition"]] = None,
                       ) -> Tuple[DayResult, List["RolledPosition"]]:
        """Run strategy for one day. Returns (DayResult, new_carries).

        For non-layer modes, new_carries is always [].
        """
        mode = self.config.get("strategy_mode", "directional")
        if mode == "call_track":
            return (self._run_call_track(ticker, trade_date, equity_df,
                                         equity_prices, options_0dte,
                                         all_dates, prev_close), [])
        if mode == "layer":
            return self._run_layer(ticker, trade_date, equity_df,
                                   equity_prices, options_0dte,
                                   all_dates, prev_close,
                                   carried_positions=carried_positions or [])
        if mode == "percentile_layer":
            return self._run_percentile_layer(
                ticker, trade_date, equity_df, equity_prices,
                options_0dte, all_dates, prev_close,
                carried_positions=carried_positions or [])
        result = DayResult(ticker=ticker, date=trade_date)
        commission = self.config["commission_per_transaction"]
        entry_time = self.config["entry_time_pacific"]
        min_step = self._get_min_step(ticker)
        eff_min_width = self._get_effective_min_width(ticker)
        max_width = self._get_max_width(ticker)
        roll_mode = self._resolve_roll_mode()

        # --- Get entry price ---
        entry_price = get_price_at_time(equity_prices, entry_time)
        if entry_price is None:
            result.failure_reason = f"No price at {entry_time}"
            return (result, [])

        sorted_times = sorted(equity_prices.keys(), key=_time_to_mins)
        result.close_price = equity_prices[sorted_times[-1]] if sorted_times else 0
        result.open_price = entry_price

        # --- Direction ---
        if prev_close is None:
            result.failure_reason = "No previous close"
            return (result, [])

        direction = "call" if entry_price >= prev_close else "put"
        result.direction = direction

        if options_0dte is None or options_0dte.empty:
            result.failure_reason = "No 0DTE options data"
            return (result, [])

        # --- Entry ---
        entry_snap = snap_options_to_time(options_0dte, entry_time)
        if entry_snap.empty:
            result.failure_reason = f"No options snapshot at {entry_time}"
            return (result, [])

        leg_placement = self.config.get("leg_placement", "otm")
        depth_pct = self.config.get("depth_pct")
        spread = find_credit_spread(entry_snap, entry_price, direction,
                                    eff_min_width, max_width,
                                    leg_placement=leg_placement,
                                    depth_pct=depth_pct)
        if spread is None:
            result.failure_reason = f"No valid spread ({leg_placement}, depth={depth_pct})"
            return (result, [])

        num_contracts = self._calc_contracts(spread["credit"], spread["width"])
        position = SpreadPosition(
            direction=direction,
            short_strike=spread["short_strike"],
            long_strike=spread["long_strike"],
            width=spread["width"],
            credit_per_share=spread["credit"],
            num_contracts=num_contracts,
            entry_time=entry_time,
            entry_price=entry_price,
            dte=0,
        )
        result.total_credits += position.total_credit
        result.total_commissions += commission
        result.trades.append(TradeRecord(
            event="entry", time_pacific=entry_time, direction=direction,
            short_strike=spread["short_strike"], long_strike=spread["long_strike"],
            width=spread["width"], credit_or_debit=position.total_credit,
            num_contracts=num_contracts, commission=commission,
            underlying_price=entry_price, dte=0,
        ))

        # --- Track HOD/LOD ---
        entry_mins = _time_to_mins(entry_time)
        hod, lod = entry_price, entry_price
        if not equity_df.empty:
            h, l = get_hod_lod_in_range(equity_df, 0, entry_mins)
            if h > 0:
                hod = max(hod, h)
            if l < float("inf"):
                lod = min(lod, l)

        # --- Intraday checks (stop loss + midday rolls) ---
        roll_times = self.config["roll_check_times_pacific"]
        for roll_time in roll_times:
            if position is None:
                break
            roll_mins = _time_to_mins(roll_time)
            if roll_mins <= entry_mins:
                continue

            # Update HOD/LOD
            if not equity_df.empty:
                h, l = get_hod_lod_in_range(equity_df, entry_mins, roll_mins)
                if h > 0:
                    hod = max(hod, h)
                if l < float("inf"):
                    lod = min(lod, l)

            roll_snap = snap_options_to_time(options_0dte, roll_time)
            if roll_snap.empty:
                continue

            current_price_at_check = get_price_at_time(equity_prices, roll_time)
            if current_price_at_check is None:
                continue

            # --- Stop loss check ---
            sl_debit = self._check_stop_loss(position, roll_snap)
            if sl_debit is not None:
                close_total = sl_debit * 100 * position.num_contracts
                result.total_debits += close_total
                result.total_commissions += commission
                result.stopped_out = True
                result.trades.append(TradeRecord(
                    event="stop_loss_close", time_pacific=roll_time,
                    direction=direction,
                    short_strike=position.short_strike, long_strike=position.long_strike,
                    width=position.width, credit_or_debit=-close_total,
                    num_contracts=position.num_contracts, commission=commission,
                    underlying_price=current_price_at_check, dte=0,
                    notes=f"SL: debit=${sl_debit:.2f}/shr",
                ))
                position = None
                break

            # --- Midday DTE+1 roll check ---
            if roll_mode in ("midday_dte1", "conditional_dte1"):
                breach = self._breach_pct(position, current_price_at_check)
                min_breach = self.config.get("midday_roll_breach_pct", 0.50)
                if breach > min_breach:
                    rolled = self._attempt_dte1_roll(
                        result, position, ticker, trade_date, all_dates,
                        options_0dte, roll_time, current_price_at_check,
                        eff_min_width, max_width, commission, roll_mode)
                    if rolled:
                        position = None
                        break

        result.hod = hod
        result.lod = lod

        # --- EOD roll (eod_itm mode only) ---
        if position is not None and position.dte == 0 and roll_mode == "eod_itm":
            position = self._handle_eod_roll(
                result, position, direction, ticker, trade_date, all_dates,
                equity_prices, options_0dte, eff_min_width, max_width, commission)

        # --- End of Day Settlement ---
        if position is not None and position.dte == 0:
            close = result.close_price
            is_itm = self._is_position_itm(position, close)

            if is_itm:
                if direction == "call":
                    intrinsic = min(close - position.short_strike, position.width)
                else:
                    intrinsic = min(position.short_strike - close, position.width)
                loss = intrinsic * 100 * position.num_contracts
                result.total_debits += loss
                result.trades.append(TradeRecord(
                    event="expiration_itm", time_pacific="13:00", direction=direction,
                    short_strike=position.short_strike, long_strike=position.long_strike,
                    width=position.width, credit_or_debit=-loss,
                    num_contracts=position.num_contracts, commission=0,
                    underlying_price=close, dte=0,
                    notes=f"ITM by {intrinsic:.2f}",
                ))
            else:
                result.trades.append(TradeRecord(
                    event="expiration_otm", time_pacific="13:00", direction=direction,
                    short_strike=position.short_strike, long_strike=position.long_strike,
                    width=position.width, credit_or_debit=0,
                    num_contracts=position.num_contracts, commission=0,
                    underlying_price=close, dte=0,
                    notes="Expired OTM — full profit",
                ))

        result.final_pnl = result.net_pnl
        return (result, [])

    # ── Call-Track Mode ────────────────────────────────────────────────────

    def _run_call_track(self, ticker: str, trade_date: str,
                        equity_df: pd.DataFrame, equity_prices: dict,
                        options_0dte: Optional[pd.DataFrame],
                        all_dates: List[str],
                        prev_close: Optional[float]) -> DayResult:
        """Call-track mode: always sell call spread, roll up when price > short strike.

        - Entry at market open: sell ATM/barely-OTM call credit spread
        - Every N hours: if price > short_strike, close + reopen at new price
        - Roll budget: cumulative roll costs ≤ X% of original credit
        - EOD: if price within Y% of short strike, roll to DTE+1
        """
        result = DayResult(ticker=ticker, date=trade_date)
        commission = self.config["commission_per_transaction"]
        entry_time = self.config["entry_time_pacific"]
        min_step = self._get_min_step(ticker)
        eff_min_width = self._get_effective_min_width(ticker)
        max_width = self._get_max_width(ticker)

        # --- Entry price ---
        entry_price = get_price_at_time(equity_prices, entry_time)
        if entry_price is None:
            result.failure_reason = f"No price at {entry_time}"
            return result

        sorted_times = sorted(equity_prices.keys(), key=_time_to_mins)
        result.close_price = equity_prices[sorted_times[-1]] if sorted_times else 0
        result.open_price = entry_price
        result.direction = "call"

        if options_0dte is None or options_0dte.empty:
            result.failure_reason = "No 0DTE options data"
            return result

        # --- Entry: always call credit spread at current price ---
        # Use tight 5-min tolerance so we only use fresh 5-min snapshots
        entry_snap = snap_options_to_time(options_0dte, entry_time, tolerance_mins=5)
        if entry_snap.empty:
            result.failure_reason = f"No options snapshot at {entry_time}"
            return result

        ct_leg = self.config.get("call_track_leg_placement", "itm")
        ct_depth = self.config.get("call_track_depth_pct", None)
        spread = find_credit_spread(entry_snap, entry_price, "call",
                                    eff_min_width, max_width,
                                    leg_placement=ct_leg, depth_pct=ct_depth)
        if spread is None:
            result.failure_reason = "No valid call spread at entry"
            return result

        num_contracts = self._calc_contracts(spread["credit"], spread["width"])
        position = SpreadPosition(
            direction="call",
            short_strike=spread["short_strike"],
            long_strike=spread["long_strike"],
            width=spread["width"],
            credit_per_share=spread["credit"],
            num_contracts=num_contracts,
            entry_time=entry_time,
            entry_price=entry_price,
            dte=0,
        )

        original_credit_total = position.total_credit
        result.total_credits += original_credit_total
        result.total_commissions += commission
        result.trades.append(TradeRecord(
            event="entry", time_pacific=entry_time, direction="call",
            short_strike=spread["short_strike"], long_strike=spread["long_strike"],
            width=spread["width"], credit_or_debit=original_credit_total,
            num_contracts=num_contracts, commission=commission,
            underlying_price=entry_price, dte=0,
        ))

        # --- Roll budget tracking ---
        cumulative_roll_cost = 0.0
        unlimited = self.config.get("call_track_unlimited_budget", False)
        roll_budget = (float("inf") if unlimited else
                       original_credit_total * self.config.get(
                           "call_track_roll_budget_pct", 0.25))

        # --- Roll check times ---
        entry_mins = _time_to_mins(entry_time)
        eod_time = self.config.get("call_track_eod_time_pacific", "12:45")
        eod_mins = _time_to_mins(eod_time)

        explicit_times = self.config.get("call_track_check_times_pacific")
        if explicit_times:
            check_times = [t for t in explicit_times if _time_to_mins(t) > entry_mins]
        else:
            interval_hours = self.config.get("call_track_roll_interval_hours", 2)
            check_times = []
            t = entry_mins + interval_hours * 60
            while t < eod_mins:
                check_times.append(f"{t // 60:02d}:{t % 60:02d}")
                t += interval_hours * 60

        # --- HOD/LOD tracking ---
        hod, lod = entry_price, entry_price

        # --- Intraday roll checks ---
        for check_time in check_times:
            if position is None:
                break

            check_mins = _time_to_mins(check_time)

            # Update HOD/LOD
            if not equity_df.empty:
                h, l = get_hod_lod_in_range(equity_df, entry_mins, check_mins)
                if h > 0:
                    hod = max(hod, h)
                if l < float("inf"):
                    lod = min(lod, l)

            current_price = get_price_at_time(equity_prices, check_time)
            if current_price is None:
                continue

            # Roll trigger: price > short_strike (spread going ITM)
            if current_price <= position.short_strike:
                continue

            roll_snap = snap_options_to_time(options_0dte, check_time, tolerance_mins=5)

            # Try 0DTE roll first: close existing + open new 0DTE
            rolled_0dte = False
            if not roll_snap.empty:
                close_debit = close_spread_cost(roll_snap, position)
                if close_debit is not None:
                    close_total = close_debit * 100 * position.num_contracts
                    new_spread = find_credit_spread(
                        roll_snap, current_price, "call",
                        eff_min_width, max_width,
                        leg_placement=ct_leg, depth_pct=ct_depth)
                    if new_spread is not None:
                        new_credit_total = new_spread["credit"] * 100 * position.num_contracts
                        roll_cost = close_total - new_credit_total

                        if roll_cost > 0 and cumulative_roll_cost + roll_cost > roll_budget:
                            result.trades.append(TradeRecord(
                                event="roll_skipped", time_pacific=check_time,
                                direction="call",
                                short_strike=position.short_strike,
                                long_strike=position.long_strike,
                                width=position.width, credit_or_debit=0,
                                num_contracts=position.num_contracts, commission=0,
                                underlying_price=current_price, dte=0,
                                notes=f"Budget: cost=${roll_cost:.0f} > remaining=${roll_budget - cumulative_roll_cost:.0f}",
                            ))
                            continue

                        # Execute 0DTE roll
                        result.total_debits += close_total
                        result.total_credits += new_credit_total
                        result.total_commissions += commission
                        if roll_cost > 0:
                            cumulative_roll_cost += roll_cost
                        result.num_rolls += 1

                        result.trades.append(TradeRecord(
                            event="roll_close", time_pacific=check_time,
                            direction="call",
                            short_strike=position.short_strike,
                            long_strike=position.long_strike,
                            width=position.width, credit_or_debit=-close_total,
                            num_contracts=position.num_contracts, commission=0,
                            underlying_price=current_price, dte=0,
                            notes=f"Price {current_price:.0f} > short {position.short_strike:.0f}",
                        ))
                        result.trades.append(TradeRecord(
                            event="roll_open", time_pacific=check_time,
                            direction="call",
                            short_strike=new_spread["short_strike"],
                            long_strike=new_spread["long_strike"],
                            width=new_spread["width"],
                            credit_or_debit=new_credit_total,
                            num_contracts=position.num_contracts,
                            commission=commission,
                            underlying_price=current_price, dte=0,
                            notes=f"Roll cost: ${roll_cost:.0f} (used ${cumulative_roll_cost:.0f}/{roll_budget:.0f})",
                        ))

                        position = SpreadPosition(
                            direction="call",
                            short_strike=new_spread["short_strike"],
                            long_strike=new_spread["long_strike"],
                            width=new_spread["width"],
                            credit_per_share=new_spread["credit"],
                            num_contracts=position.num_contracts,
                            entry_time=check_time,
                            entry_price=current_price,
                            dte=0,
                        )
                        rolled_0dte = True

            # Fallback: if 0DTE roll failed, try DTE+1 roll
            if not rolled_0dte:
                rolled = self._attempt_dte1_roll(
                    result, position, ticker, trade_date, all_dates,
                    options_0dte, check_time, current_price,
                    eff_min_width, max_width, commission, "intraday_dte1")
                if rolled:
                    position = None
                    break
                else:
                    result.trades.append(TradeRecord(
                        event="roll_skipped", time_pacific=check_time,
                        direction="call",
                        short_strike=position.short_strike,
                        long_strike=position.long_strike,
                        width=position.width, credit_or_debit=0,
                        num_contracts=position.num_contracts, commission=0,
                        underlying_price=current_price, dte=0,
                        notes="No 0DTE or DTE+1 roll possible",
                    ))

        result.hod = hod
        result.lod = lod

        # --- EOD proximity check → roll to DTE+1 ---
        if position is not None and position.dte == 0:
            eod_proximity = self.config.get("call_track_eod_proximity_pct", 0.003)
            eod_price = get_price_at_time(equity_prices, eod_time)

            if eod_price is not None:
                distance_pct = abs(eod_price - position.short_strike) / eod_price
                if distance_pct <= eod_proximity:
                    rolled = self._attempt_dte1_roll(
                        result, position, ticker, trade_date, all_dates,
                        options_0dte, eod_time, eod_price,
                        eff_min_width, max_width, commission, "eod_itm")
                    if rolled:
                        position = None

        # --- End of day settlement ---
        if position is not None and position.dte == 0:
            close = result.close_price
            is_itm = close > position.short_strike  # always call

            if is_itm:
                intrinsic = min(close - position.short_strike, position.width)
                loss = intrinsic * 100 * position.num_contracts
                result.total_debits += loss
                result.trades.append(TradeRecord(
                    event="expiration_itm", time_pacific="13:00", direction="call",
                    short_strike=position.short_strike,
                    long_strike=position.long_strike,
                    width=position.width, credit_or_debit=-loss,
                    num_contracts=position.num_contracts, commission=0,
                    underlying_price=close, dte=0,
                    notes=f"ITM by {intrinsic:.2f}",
                ))
            else:
                result.trades.append(TradeRecord(
                    event="expiration_otm", time_pacific="13:00", direction="call",
                    short_strike=position.short_strike,
                    long_strike=position.long_strike,
                    width=position.width, credit_or_debit=0,
                    num_contracts=position.num_contracts, commission=0,
                    underlying_price=close, dte=0,
                    notes="Expired OTM — full profit",
                ))

        result.final_pnl = result.net_pnl
        return result

    # ── Layer Mode ───────────────────────────────────────────────────────

    def _run_layer(self, ticker: str, trade_date: str,
                   equity_df: pd.DataFrame, equity_prices: dict,
                   options_0dte: Optional[pd.DataFrame],
                   all_dates: List[str],
                   prev_close: Optional[float],
                   carried_positions: Optional[List["RolledPosition"]] = None,
                   ) -> Tuple[DayResult, List["RolledPosition"]]:
        """Layer mode: accumulate spreads on new HOD/LOD, roll ITM at EOD.

        - START: load carried positions expiring today into positions list
        - 06:35: dual entry — open call spread + put spread (or single if configured)
        - 08:35, 10:35: if new HOD → add call spread; if new LOD → add put spread
        - 12:45: roll all ITM spreads to DTE+1 (with roll limits for carried positions)
        - 13:00: remaining positions expire

        Returns (DayResult, new_carries) where new_carries are RolledPositions
        for future days.
        """
        carried_positions = carried_positions or []
        new_carries: List[RolledPosition] = []
        result = DayResult(ticker=ticker, date=trade_date)
        commission = self.config["commission_per_transaction"]
        entry_time = self.config["entry_time_pacific"]
        min_step = self._get_min_step(ticker)
        eff_min_width = self._get_effective_min_width(ticker)
        max_width = self._get_max_width(ticker)
        max_roll_count = self.config.get("max_roll_count", 3)
        roll_recovery_threshold = self.config.get("roll_recovery_threshold", 1.0)

        # --- Entry price ---
        entry_price = get_price_at_time(equity_prices, entry_time)
        if entry_price is None:
            result.failure_reason = f"No price at {entry_time}"
            return (result, [])

        sorted_times = sorted(equity_prices.keys(), key=_time_to_mins)
        result.close_price = equity_prices[sorted_times[-1]] if sorted_times else 0
        result.open_price = entry_price

        if prev_close is None:
            result.failure_reason = "No previous close"
            return (result, [])

        if options_0dte is None or options_0dte.empty:
            result.failure_reason = "No 0DTE options data"
            return (result, [])

        # --- Initial direction ---
        direction = "call" if entry_price >= prev_close else "put"

        ct_leg = self.config.get("call_track_leg_placement", "nearest")
        ct_depth = self.config.get("call_track_depth_pct")

        # --- Auto-size contracts from budget if not explicitly set ---
        explicit_contracts = self.config.get("num_contracts")
        if explicit_contracts:
            num_contracts = explicit_contracts
        else:
            daily_budget = self.config.get("daily_budget", 100000)
            check_times_cfg = self.config.get("call_track_check_times_pacific", [])
            # Max positions: 2 entries + up to 2 per check time
            max_positions = 2 + len(check_times_cfg) * 2
            num_contracts = max(1, int(daily_budget / (max_positions * eff_min_width * 100)))

        # --- Load carried positions expiring today ---
        positions: List[SpreadPosition] = []
        # Map position index → RolledPosition for roll limit checks
        carried_map: Dict[int, RolledPosition] = {}

        for rp in carried_positions:
            if rp.expiration_date != trade_date:
                continue
            pos = SpreadPosition(
                direction=rp.direction,
                short_strike=rp.short_strike,
                long_strike=rp.long_strike,
                width=rp.width,
                credit_per_share=rp.credit_per_share,
                num_contracts=rp.num_contracts,
                entry_time="carried",
                entry_price=entry_price,
                dte=1,
            )
            idx = len(positions)
            positions.append(pos)
            carried_map[idx] = rp
            result.trades.append(TradeRecord(
                event="carried_position", time_pacific=entry_time,
                direction=rp.direction,
                short_strike=rp.short_strike, long_strike=rp.long_strike,
                width=rp.width, credit_or_debit=0,
                num_contracts=rp.num_contracts, commission=0,
                underlying_price=entry_price, dte=1,
                notes=f"Carry from {rp.original_entry_date}, roll#{rp.roll_count}",
            ))

        # --- Entry spread(s): scan window for best ROI ---
        entry_dir_cfg = self.config.get("layer_entry_directions", "both")
        dual_entry = self.config.get("layer_dual_entry", True)
        if entry_dir_cfg == "call":
            entry_dirs = ["call"]
        elif entry_dir_cfg == "put":
            entry_dirs = ["put"]
        elif dual_entry:
            entry_dirs = ["call", "put"]
        else:
            entry_dirs = [direction]

        window_start = _time_to_mins(self.config.get("layer_entry_window_start", "06:30"))
        window_end = _time_to_mins(self.config.get("layer_entry_window_end", "06:45"))

        # Adaptive ROI: starts at aspiration, relaxes linearly to floor at window end
        min_roi_start = self.config.get("layer_entry_min_roi", 0.50)
        min_roi_floor = self.config.get("layer_entry_min_roi_floor", 0.0)
        window_range = max(window_end - window_start, 1)

        # Scan each minute in the window with adaptive ROI threshold
        accepted_spreads: Dict[str, Tuple[Dict, str, float, float]] = {}  # dir → (spread, time, price, threshold)
        best_fallback: Dict[str, Tuple[Dict, str, float, float]] = {}     # dir → (spread, time, price, roi)

        for scan_mins in range(window_start, window_end + 1):
            scan_time = f"{scan_mins // 60:02d}:{scan_mins % 60:02d}"
            snap = snap_options_to_time(options_0dte, scan_time, tolerance_mins=3)
            if snap.empty:
                continue
            scan_price = get_price_at_time(equity_prices, scan_time)
            if scan_price is None:
                continue

            progress = (scan_mins - window_start) / window_range  # 0.0→1.0
            dynamic_min_roi = min_roi_start - (min_roi_start - min_roi_floor) * progress

            for entry_dir in entry_dirs:
                if entry_dir in accepted_spreads:
                    continue  # already accepted for this direction

                # Try best_roi first (0.3% proximity), fall back to nearest
                spread = find_credit_spread(snap, scan_price, entry_dir,
                                            eff_min_width, max_width,
                                            leg_placement="best_roi",
                                            depth_pct=0.003)
                if spread is None:
                    spread = find_credit_spread(snap, scan_price, entry_dir,
                                                eff_min_width, max_width,
                                                leg_placement=ct_leg, depth_pct=ct_depth)
                if spread is None:
                    continue

                roi = spread["credit"] / (spread["width"] - spread["credit"]) if spread["width"] > spread["credit"] else 0

                # Track best fallback regardless
                prev_fb = best_fallback.get(entry_dir)
                if prev_fb is None or roi > prev_fb[3]:
                    best_fallback[entry_dir] = (spread, scan_time, scan_price, roi)

                # Accept immediately if meets dynamic threshold
                if roi >= dynamic_min_roi:
                    accepted_spreads[entry_dir] = (spread, scan_time, scan_price, dynamic_min_roi)

        # Use accepted spreads, falling back to best-seen for directions not accepted
        final_spreads: Dict[str, Tuple[Dict, str, float, str]] = {}  # dir → (spread, time, price, notes_extra)
        for entry_dir in entry_dirs:
            if entry_dir in accepted_spreads:
                sp, tm, pr, thresh = accepted_spreads[entry_dir]
                roi_val = sp["credit"] / (sp["width"] - sp["credit"]) * 100 if sp["width"] > sp["credit"] else 0
                final_spreads[entry_dir] = (sp, tm, pr, f"ROI={roi_val:.0f}%, threshold={thresh*100:.0f}%")
            elif entry_dir in best_fallback:
                sp, tm, pr, roi_val_raw = best_fallback[entry_dir]
                final_spreads[entry_dir] = (sp, tm, pr, f"ROI={roi_val_raw*100:.0f}%, fallback best-seen")

        any_entry = False
        for entry_dir in entry_dirs:
            if entry_dir not in final_spreads:
                continue
            spread, used_time, used_price, roi_notes = final_spreads[entry_dir]
            pos = SpreadPosition(
                direction=entry_dir,
                short_strike=spread["short_strike"],
                long_strike=spread["long_strike"],
                width=spread["width"],
                credit_per_share=spread["credit"],
                num_contracts=num_contracts,
                entry_time=used_time,
                entry_price=used_price,
                dte=0,
            )
            positions.append(pos)
            result.total_credits += pos.total_credit
            result.total_commissions += commission
            result.trades.append(TradeRecord(
                event="entry", time_pacific=used_time, direction=entry_dir,
                short_strike=spread["short_strike"], long_strike=spread["long_strike"],
                width=spread["width"], credit_or_debit=pos.total_credit,
                num_contracts=num_contracts, commission=commission,
                underlying_price=used_price, dte=0,
                notes=f"Initial {entry_dir} spread ({roi_notes})",
            ))
            any_entry = True

        if not any_entry and not carried_positions:
            result.failure_reason = f"No valid spread at entry"
            return (result, [])

        if len(entry_dirs) > 1 and any_entry:
            result.direction = "both"
        elif entry_dirs:
            result.direction = entry_dirs[0]
        else:
            result.direction = direction

        # --- HOD/LOD tracking ---
        entry_mins = _time_to_mins(entry_time)
        hod, lod = entry_price, entry_price
        prev_hod, prev_lod = hod, lod

        # --- Intraday checks: layer new spreads on new extremes ---
        check_times = self.config.get("call_track_check_times_pacific",
                                      ["08:35", "10:35"])
        check_times = [t for t in check_times if _time_to_mins(t) > entry_mins]

        for check_time in check_times:
            check_mins = _time_to_mins(check_time)

            # Update HOD/LOD from bars
            if not equity_df.empty:
                h, l = get_hod_lod_in_range(equity_df, entry_mins, check_mins)
                if h > 0:
                    hod = max(hod, h)
                if l < float("inf"):
                    lod = min(lod, l)

            current_price = get_price_at_time(equity_prices, check_time)
            if current_price is None:
                continue

            snap = snap_options_to_time(options_0dte, check_time, tolerance_mins=5)
            if snap.empty:
                continue

            breach_min = self.config.get("layer_breach_min_points") or eff_min_width
            new_hod = hod >= prev_hod + breach_min
            new_lod = lod <= prev_lod - breach_min

            # New HOD → add call spread at HOD level (sell at the high)
            if new_hod:
                call_spread = find_credit_spread(
                    snap, hod, "call",
                    eff_min_width, max_width,
                    leg_placement="best_roi", depth_pct=0.003)
                if call_spread is None:
                    call_spread = find_credit_spread(
                        snap, hod, "call",
                        eff_min_width, max_width,
                        leg_placement=ct_leg, depth_pct=ct_depth)
                if call_spread is not None:
                    pos = SpreadPosition(
                        direction="call",
                        short_strike=call_spread["short_strike"],
                        long_strike=call_spread["long_strike"],
                        width=call_spread["width"],
                        credit_per_share=call_spread["credit"],
                        num_contracts=num_contracts,
                        entry_time=check_time,
                        entry_price=current_price,
                        dte=0,
                    )
                    positions.append(pos)
                    result.total_credits += pos.total_credit
                    result.total_commissions += commission
                    result.trades.append(TradeRecord(
                        event="layer_add", time_pacific=check_time,
                        direction="call",
                        short_strike=call_spread["short_strike"],
                        long_strike=call_spread["long_strike"],
                        width=call_spread["width"],
                        credit_or_debit=pos.total_credit,
                        num_contracts=num_contracts, commission=commission,
                        underlying_price=current_price, dte=0,
                        notes=f"New HOD {hod:.0f} (prev {prev_hod:.0f}, breach>={breach_min:.0f})",
                    ))
                    prev_hod = hod  # ratchet up only when layer fires

            # New LOD → add put spread at LOD level (sell at the low)
            if new_lod:
                put_spread = find_credit_spread(
                    snap, lod, "put",
                    eff_min_width, max_width,
                    leg_placement="best_roi", depth_pct=0.003)
                if put_spread is None:
                    put_spread = find_credit_spread(
                        snap, lod, "put",
                        eff_min_width, max_width,
                        leg_placement=ct_leg, depth_pct=ct_depth)
                if put_spread is not None:
                    pos = SpreadPosition(
                        direction="put",
                        short_strike=put_spread["short_strike"],
                        long_strike=put_spread["long_strike"],
                        width=put_spread["width"],
                        credit_per_share=put_spread["credit"],
                        num_contracts=num_contracts,
                        entry_time=check_time,
                        entry_price=current_price,
                        dte=0,
                    )
                    positions.append(pos)
                    result.total_credits += pos.total_credit
                    result.total_commissions += commission
                    result.trades.append(TradeRecord(
                        event="layer_add", time_pacific=check_time,
                        direction="put",
                        short_strike=put_spread["short_strike"],
                        long_strike=put_spread["long_strike"],
                        width=put_spread["width"],
                        credit_or_debit=pos.total_credit,
                        num_contracts=num_contracts, commission=commission,
                        underlying_price=current_price, dte=0,
                        notes=f"New LOD {lod:.0f} (prev {prev_lod:.0f}, breach>={breach_min:.0f})",
                    ))
                    prev_lod = lod  # ratchet down only when layer fires

        result.hod = hod
        result.lod = lod

        # --- EOD: minute-by-minute scan from eod_scan_start to eod_scan_end ---
        # At each minute: if position is ITM or within proximity of short strike → roll
        # Once rolled, position is removed from scan. Anything surviving settles at 13:00.
        eod_scan_start = _time_to_mins(self.config.get("layer_eod_scan_start", "12:50"))
        eod_scan_end = _time_to_mins(self.config.get("layer_eod_scan_end", "13:00"))
        eod_proximity = self.config.get("layer_eod_proximity", 0.002)  # 0.2%

        remaining = list(positions)
        checked_set = set()  # positions already decided (rolled or blocked)

        for scan_mins in range(eod_scan_start, eod_scan_end + 1):
            scan_time = f"{scan_mins // 60:02d}:{scan_mins % 60:02d}"
            scan_price = get_price_at_time(equity_prices, scan_time)
            if scan_price is None:
                continue

            still_remaining = []
            for pos in remaining:
                pos_id = id(pos)
                if pos_id in checked_set:
                    still_remaining.append(pos)  # already decided, keep for settlement
                    continue

                idx = positions.index(pos) if pos in positions else -1

                # Check: ITM or within proximity of short strike
                is_itm = self._is_position_itm(pos, scan_price)
                if not is_itm and eod_proximity > 0:
                    distance_pct = abs(scan_price - pos.short_strike) / scan_price if scan_price > 0 else 1
                    is_threatened = distance_pct <= eod_proximity
                else:
                    is_threatened = is_itm

                if not is_threatened:
                    still_remaining.append(pos)
                    continue

                # Check roll limits for carried positions
                rp = carried_map.get(idx) if idx >= 0 else None
                if rp is not None:
                    if rp.roll_count >= max_roll_count:
                        still_remaining.append(pos)  # stays for 13:00 settlement
                        result.trades.append(TradeRecord(
                            event="roll_limit_reached", time_pacific=scan_time,
                            direction=pos.direction,
                            short_strike=pos.short_strike, long_strike=pos.long_strike,
                            width=pos.width, credit_or_debit=0,
                            num_contracts=pos.num_contracts, commission=0,
                            underlying_price=scan_price, dte=pos.dte,
                            notes=f"Max rolls ({max_roll_count}) reached, settling at 13:00",
                        ))
                        checked_set.add(pos_id)
                        continue
                    if rp.cumulative_roll_cost >= rp.original_credit * roll_recovery_threshold:
                        still_remaining.append(pos)
                        result.trades.append(TradeRecord(
                            event="roll_cost_exceeded", time_pacific=scan_time,
                            direction=pos.direction,
                            short_strike=pos.short_strike, long_strike=pos.long_strike,
                            width=pos.width, credit_or_debit=0,
                            num_contracts=pos.num_contracts, commission=0,
                            underlying_price=scan_price, dte=pos.dte,
                            notes=f"Roll cost ${rp.cumulative_roll_cost:.0f} >= threshold ${rp.original_credit * roll_recovery_threshold:.0f}",
                        ))
                        checked_set.add(pos_id)
                        continue

                # Attempt DTE+1 roll — position leaves remaining if successful
                new_rp = self._attempt_dte1_roll_layer(
                    result, pos, ticker, trade_date, all_dates,
                    options_0dte, scan_time, scan_price,
                    eff_min_width, max_width, commission, rp)
                if new_rp is not None:
                    new_carries.append(new_rp)
                    result.num_rolls += 1
                # Don't add to still_remaining — position is handled
                checked_set.add(pos_id)

            remaining = still_remaining

        # --- 13:00: settle remaining positions ---
        close = result.close_price
        for pos in remaining:
            is_itm = self._is_position_itm(pos, close)
            if is_itm:
                if pos.direction == "call":
                    intrinsic = min(close - pos.short_strike, pos.width)
                else:
                    intrinsic = min(pos.short_strike - close, pos.width)
                loss = intrinsic * 100 * pos.num_contracts
                result.total_debits += loss
                result.trades.append(TradeRecord(
                    event="expiration_itm", time_pacific="13:00",
                    direction=pos.direction,
                    short_strike=pos.short_strike, long_strike=pos.long_strike,
                    width=pos.width, credit_or_debit=-loss,
                    num_contracts=pos.num_contracts, commission=0,
                    underlying_price=close, dte=pos.dte,
                    notes=f"ITM by {intrinsic:.2f}",
                ))
            else:
                result.trades.append(TradeRecord(
                    event="expiration_otm", time_pacific="13:00",
                    direction=pos.direction,
                    short_strike=pos.short_strike, long_strike=pos.long_strike,
                    width=pos.width, credit_or_debit=0,
                    num_contracts=pos.num_contracts, commission=0,
                    underlying_price=close, dte=pos.dte,
                    notes="Expired OTM — full profit",
                ))

        result.final_pnl = result.net_pnl
        return (result, new_carries)

    # ── Percentile Layer Mode ─────────────────────────────────────────────

    def _precompute_daily_closes(self, ticker: str, all_dates: List[str],
                                  equity_dir: str) -> Dict[str, float]:
        """Load close prices for all dates (cached after first call)."""
        if not hasattr(self, '_daily_closes_cache') or self._daily_closes_cache is None:
            closes: Dict[str, float] = {}
            for d in all_dates:
                df = load_equity_bars_df(ticker, d, equity_dir)
                if not df.empty:
                    closes[d] = float(df['close'].iloc[-1])
            self._daily_closes_cache = closes
        return self._daily_closes_cache

    def _compute_percentile_bands(self, trade_date: str, all_dates: List[str],
                                   daily_closes: Dict[str, float],
                                   prev_close: float) -> Optional[Dict[str, float]]:
        """Compute P(N) call and put price levels from historical close-to-close returns."""
        pN = self.config.get("percentile_entry_pN", 75)
        lookback = self.config.get("percentile_lookback", 120)

        if trade_date not in all_dates:
            return None
        idx = all_dates.index(trade_date)
        start = max(0, idx - lookback)
        window_dates = all_dates[start:idx]
        closes_arr = np.array([daily_closes[d] for d in window_dates if d in daily_closes])
        if len(closes_arr) < 10:
            return None

        returns = (closes_arr[1:] - closes_arr[:-1]) / closes_arr[:-1]
        up_returns = returns[returns > 0]
        down_returns = returns[returns < 0]

        call_level = None
        put_level = None
        if len(up_returns) > 0:
            call_level = prev_close * (1 + np.percentile(up_returns, pN))
        if len(down_returns) > 0:
            put_level = prev_close * (1 - np.percentile(np.abs(down_returns), pN))

        return {"call": call_level, "put": put_level}

    def _run_percentile_layer(self, ticker: str, trade_date: str,
                               equity_df: pd.DataFrame, equity_prices: dict,
                               options_0dte: Optional[pd.DataFrame],
                               all_dates: List[str],
                               prev_close: Optional[float],
                               carried_positions: Optional[List["RolledPosition"]] = None,
                               ) -> Tuple[DayResult, List["RolledPosition"]]:
        """Percentile-triggered layer mode.

        Entry: only when price breaches P(N) band from close-to-close returns.
        Direction: follows the move (up -> call spread, down -> put spread).
        Both legs ITM at entry. Optional layering on HOD/LOD (same dir only).
        Roll at configurable time if within proximity. Settle at 13:00.
        """
        carried_positions = carried_positions or []
        new_carries: List[RolledPosition] = []
        result = DayResult(ticker=ticker, date=trade_date)
        commission = self.config["commission_per_transaction"]
        equity_dir = self.config.get("equity_dir", "equities_output")
        spread_width = self.config.get("percentile_spread_width", 5)
        layering = self.config.get("percentile_layering", True)
        layer_cutoff = self.config.get("percentile_layer_cutoff", "11:45")
        roll_time = self.config.get("percentile_roll_time", "12:30")
        roll_proximity = self.config.get("percentile_roll_proximity", 0.003)
        roll_dte = self.config.get("percentile_roll_dte", 2)
        num_contracts = self.config.get("num_contracts") or 1
        leg_placement = self.config.get("percentile_leg_placement", "itm")  # itm, otm, debit
        otm_depth = self.config.get("percentile_otm_depth", 0.001)  # 0.1% for near-ATM OTM
        is_debit = leg_placement == "debit"

        # Validate data
        if equity_df.empty or prev_close is None:
            result.failure_reason = "No equity data or prev_close"
            return (result, [])
        if options_0dte is None or options_0dte.empty:
            result.failure_reason = "No 0DTE options data"
            return (result, [])

        sorted_times = sorted(equity_prices.keys(), key=_time_to_mins)
        if not sorted_times:
            result.failure_reason = "No price data"
            return (result, [])
        result.open_price = equity_prices[sorted_times[0]]
        result.close_price = equity_prices[sorted_times[-1]]

        # --- Compute percentile bands ---
        daily_closes = self._precompute_daily_closes(ticker, all_dates, equity_dir)
        bands = self._compute_percentile_bands(trade_date, all_dates, daily_closes, prev_close)
        if bands is None:
            result.failure_reason = "Insufficient history for percentile bands"
            return (result, [])

        call_level = bands["call"]
        put_level = bands["put"]
        pN = self.config.get("percentile_entry_pN", 75)

        # --- VIX filtering ---
        vix_open = None
        vix_regime = None
        if self.config.get("percentile_vix_enabled", False):
            vix_df = load_equity_bars_df("VIX", trade_date, "equities_output")
            if not vix_df.empty:
                vix_open = float(vix_df["close"].iloc[0])

            vix_min = self.config.get("percentile_vix_min")
            vix_max = self.config.get("percentile_vix_max")
            regime_skip = self.config.get("percentile_vix_regime_skip", [])

            if vix_min and vix_open is not None and vix_open < vix_min:
                result.failure_reason = f"VIX {vix_open:.1f} < min {vix_min}"
                # Still handle carries below
                if not carried_positions:
                    return (result, [])
            if vix_max and vix_open is not None and vix_open > vix_max:
                result.failure_reason = f"VIX {vix_open:.1f} > max {vix_max}"
                if not carried_positions:
                    return (result, [])

            # VIX regime check (inline — avoids dependency on signal generator in subprocess)
            if regime_skip and vix_open is not None:
                vix_daily_cache = getattr(self, '_vix_daily_cache', None)
                if vix_daily_cache is None:
                    vix_daily_cache = {}
                    vix_dir = os.path.join("equities_output", "I:VIX")
                    if os.path.isdir(vix_dir):
                        for fname in sorted(os.listdir(vix_dir)):
                            if not fname.endswith(".csv"):
                                continue
                            ds = fname.split("_")[-1].replace(".csv", "")
                            try:
                                vdf = pd.read_csv(os.path.join(vix_dir, fname), usecols=["close"])
                                if not vdf.empty:
                                    vix_daily_cache[ds] = float(vdf["close"].iloc[-1])
                            except Exception:
                                pass
                    self._vix_daily_cache = vix_daily_cache

                lookback_vix = [v for d, v in sorted(vix_daily_cache.items()) if d < trade_date][-60:]
                if len(lookback_vix) >= 10:
                    pct_rank = float(np.sum(np.array(lookback_vix) < vix_open) / len(lookback_vix) * 100)
                    if pct_rank < 30:
                        vix_regime = "low"
                    elif pct_rank < 70:
                        vix_regime = "normal"
                    elif pct_rank < 90:
                        vix_regime = "high"
                    else:
                        vix_regime = "extreme"

                    if vix_regime in regime_skip:
                        result.failure_reason = f"VIX regime '{vix_regime}' skipped (VIX={vix_open:.1f})"
                        if not carried_positions:
                            return (result, [])

            # Scale contracts by VIX regime
            if self.config.get("percentile_vix_scale_contracts") and vix_regime:
                vix_multipliers = {"low": 1.2, "normal": 1.0, "high": 0.6, "extreme": 0.25}
                mult = vix_multipliers.get(vix_regime, 1.0)
                num_contracts = max(1, int(num_contracts * mult))

        # --- Load carried positions expiring today ---
        positions: List[SpreadPosition] = []
        carried_map: Dict[int, RolledPosition] = {}
        for rp in carried_positions:
            if rp.expiration_date != trade_date:
                continue
            pos = SpreadPosition(
                direction=rp.direction, short_strike=rp.short_strike,
                long_strike=rp.long_strike, width=rp.width,
                credit_per_share=rp.credit_per_share,
                num_contracts=rp.num_contracts,
                entry_time="carried", entry_price=result.open_price, dte=1,
            )
            idx = len(positions)
            positions.append(pos)
            carried_map[idx] = rp
            result.trades.append(TradeRecord(
                event="carried_position", time_pacific="06:30",
                direction=rp.direction,
                short_strike=rp.short_strike, long_strike=rp.long_strike,
                width=rp.width, credit_or_debit=0,
                num_contracts=rp.num_contracts, commission=0,
                underlying_price=result.open_price, dte=1,
                notes=f"Carry from {rp.original_entry_date}, roll#{rp.roll_count}",
            ))

        # --- Scan 5-min bars for percentile band breach ---
        trigger_dir = None
        trigger_time = None
        trigger_price = None

        for _, bar in equity_df.iterrows():
            bar_time = bar["time_pacific"]
            bar_mins = _time_to_mins(bar_time)
            if bar_mins < _time_to_mins("06:30"):
                continue

            bar_high = float(bar["high"])
            bar_low = float(bar["low"])
            bar_close = float(bar["close"])

            if call_level is not None and bar_high >= call_level:
                trigger_dir = "call"
                trigger_time = bar_time
                trigger_price = bar_close
                break
            if put_level is not None and bar_low <= put_level:
                trigger_dir = "put"
                trigger_time = bar_time
                trigger_price = bar_close
                break

        if trigger_dir is None:
            # No trigger today — but still handle carried positions
            if not carried_positions:
                result.failure_reason = f"Price stayed within P{pN} bands (call={call_level:.1f}, put={put_level:.1f})"
                return (result, [])
            # Fall through to EOD handling for carries
            result.trades.append(TradeRecord(
                event="no_trigger", time_pacific="13:00",
                direction="none", short_strike=0, long_strike=0,
                width=0, credit_or_debit=0, num_contracts=0, commission=0,
                underlying_price=result.close_price, dte=0,
                notes=f"P{pN} not breached (call={call_level:.1f}, put={put_level:.1f})",
            ))
        else:
            # --- Entry: ITM credit spread at trigger price ---
            snap = snap_options_to_time(options_0dte, trigger_time, tolerance_mins=5)
            if snap.empty:
                snap = snap_options_to_time(options_0dte, trigger_time, tolerance_mins=10)

            entry_spread = None
            if not snap.empty:
                if is_debit:
                    entry_spread = find_debit_spread(
                        snap, trigger_price, trigger_dir,
                        spread_width, spread_width,
                        depth_pct=otm_depth)
                else:
                    entry_spread = find_credit_spread(
                        snap, trigger_price, trigger_dir,
                        spread_width, spread_width,
                        leg_placement=leg_placement,
                        depth_pct=otm_depth if leg_placement == "otm" else None)

            if entry_spread is not None:
                if is_debit:
                    # Debit spread: we pay the debit upfront
                    debit_per_share = entry_spread["debit"]
                    debit_total = debit_per_share * 100 * num_contracts
                    max_profit_total = entry_spread["max_profit"] * 100 * num_contracts
                    pos = SpreadPosition(
                        direction=trigger_dir,
                        short_strike=entry_spread["short_strike"],
                        long_strike=entry_spread["long_strike"],
                        width=entry_spread["width"],
                        credit_per_share=-debit_per_share,  # negative = debit
                        num_contracts=num_contracts,
                        entry_time=trigger_time,
                        entry_price=trigger_price,
                        dte=0,
                    )
                    positions.append(pos)
                    result.total_debits += debit_total
                    result.total_commissions += commission
                    roi_pct = entry_spread["max_profit"] / debit_per_share * 100 if debit_per_share > 0 else 0
                    result.trades.append(TradeRecord(
                        event="entry", time_pacific=trigger_time,
                        direction=trigger_dir,
                        short_strike=entry_spread["short_strike"],
                        long_strike=entry_spread["long_strike"],
                        width=entry_spread["width"],
                        credit_or_debit=-debit_total,
                        num_contracts=num_contracts, commission=commission,
                        underlying_price=trigger_price, dte=0,
                        notes=f"P{pN} {trigger_dir} breach @{trigger_price:.1f} "
                              f"DEBIT={debit_per_share:.2f} maxProfit={entry_spread['max_profit']:.2f} "
                              f"ROI={roi_pct:.0f}% w={entry_spread['width']}",
                    ))
                else:
                    # Credit spread (OTM or ITM)
                    pos = SpreadPosition(
                        direction=trigger_dir,
                        short_strike=entry_spread["short_strike"],
                        long_strike=entry_spread["long_strike"],
                        width=entry_spread["width"],
                        credit_per_share=entry_spread["credit"],
                        num_contracts=num_contracts,
                        entry_time=trigger_time,
                        entry_price=trigger_price,
                        dte=0,
                    )
                    positions.append(pos)
                    result.total_credits += pos.total_credit
                    result.total_commissions += commission
                    roi_pct = entry_spread["credit"] / (entry_spread["width"] - entry_spread["credit"]) * 100 \
                        if entry_spread["width"] > entry_spread["credit"] else 0
                    result.trades.append(TradeRecord(
                        event="entry", time_pacific=trigger_time,
                        direction=trigger_dir,
                        short_strike=entry_spread["short_strike"],
                        long_strike=entry_spread["long_strike"],
                        width=entry_spread["width"],
                        credit_or_debit=pos.total_credit,
                        num_contracts=num_contracts, commission=commission,
                        underlying_price=trigger_price, dte=0,
                        notes=f"P{pN} {trigger_dir} breach @{trigger_price:.1f} "
                              f"(level={call_level if trigger_dir == 'call' else put_level:.1f}, "
                              f"ROI={roi_pct:.0f}%, {leg_placement.upper()}, w={entry_spread['width']}"
                              f"{f', VIX={vix_open:.1f} {vix_regime}' if vix_open else ''})",
                    ))
                result.direction = trigger_dir
            else:
                result.trades.append(TradeRecord(
                    event="entry_failed", time_pacific=trigger_time,
                    direction=trigger_dir, short_strike=0, long_strike=0,
                    width=0, credit_or_debit=0, num_contracts=0, commission=0,
                    underlying_price=trigger_price, dte=0,
                    notes=f"P{pN} {trigger_dir} triggered but no {leg_placement} spread at w={spread_width}",
                ))

        # --- HOD/LOD layering (same direction only, before cutoff) ---
        if layering and trigger_dir is not None:
            trigger_mins = _time_to_mins(trigger_time) if trigger_time else 0
            cutoff_mins = _time_to_mins(layer_cutoff)
            breach_min = self.config.get("layer_breach_min_points") or spread_width

            hod = trigger_price or result.open_price
            lod = trigger_price or result.open_price
            prev_extreme = hod if trigger_dir == "call" else lod

            # Check at regular intervals
            check_interval = 30  # every 30 mins
            for check_mins in range(trigger_mins + check_interval, cutoff_mins + 1, check_interval):
                check_time = f"{check_mins // 60:02d}:{check_mins % 60:02d}"
                if not equity_df.empty:
                    h, l = get_hod_lod_in_range(equity_df, trigger_mins, check_mins)
                    if h > 0:
                        hod = max(hod, h)
                    if l < float("inf"):
                        lod = min(lod, l)

                check_price = get_price_at_time(equity_prices, check_time)
                if check_price is None:
                    continue
                snap = snap_options_to_time(options_0dte, check_time, tolerance_mins=5)
                if snap.empty:
                    continue

                # Only layer in the trigger direction
                should_layer = False
                layer_ref_price = check_price
                if trigger_dir == "call" and hod >= prev_extreme + breach_min:
                    should_layer = True
                    layer_ref_price = hod
                elif trigger_dir == "put" and lod <= prev_extreme - breach_min:
                    should_layer = True
                    layer_ref_price = lod

                if should_layer:
                    if is_debit:
                        layer_spread = find_debit_spread(
                            snap, layer_ref_price, trigger_dir,
                            spread_width, spread_width,
                            depth_pct=otm_depth)
                    else:
                        layer_spread = find_credit_spread(
                            snap, layer_ref_price, trigger_dir,
                            spread_width, spread_width,
                            leg_placement=leg_placement,
                            depth_pct=otm_depth if leg_placement == "otm" else None)
                    if layer_spread is not None:
                        if is_debit:
                            layer_debit = layer_spread["debit"]
                            layer_total = layer_debit * 100 * num_contracts
                            pos = SpreadPosition(
                                direction=trigger_dir,
                                short_strike=layer_spread["short_strike"],
                                long_strike=layer_spread["long_strike"],
                                width=layer_spread["width"],
                                credit_per_share=-layer_debit,
                                num_contracts=num_contracts,
                                entry_time=check_time,
                                entry_price=check_price,
                                dte=0,
                            )
                            positions.append(pos)
                            result.total_debits += layer_total
                            result.total_commissions += commission
                            result.trades.append(TradeRecord(
                                event="layer_add", time_pacific=check_time,
                                direction=trigger_dir,
                                short_strike=layer_spread["short_strike"],
                                long_strike=layer_spread["long_strike"],
                                width=layer_spread["width"],
                                credit_or_debit=-layer_total,
                                num_contracts=num_contracts, commission=commission,
                                underlying_price=check_price, dte=0,
                                notes=f"DEBIT layer {'HOD' if trigger_dir == 'call' else 'LOD'} "
                                      f"breach to {hod if trigger_dir == 'call' else lod:.1f}",
                            ))
                        else:
                            pos = SpreadPosition(
                                direction=trigger_dir,
                                short_strike=layer_spread["short_strike"],
                                long_strike=layer_spread["long_strike"],
                                width=layer_spread["width"],
                                credit_per_share=layer_spread["credit"],
                                num_contracts=num_contracts,
                                entry_time=check_time,
                                entry_price=check_price,
                                dte=0,
                            )
                            positions.append(pos)
                            result.total_credits += pos.total_credit
                            result.total_commissions += commission
                            result.trades.append(TradeRecord(
                                event="layer_add", time_pacific=check_time,
                                direction=trigger_dir,
                                short_strike=layer_spread["short_strike"],
                                long_strike=layer_spread["long_strike"],
                                width=layer_spread["width"],
                                credit_or_debit=pos.total_credit,
                                num_contracts=num_contracts, commission=commission,
                                underlying_price=check_price, dte=0,
                                notes=f"{'HOD' if trigger_dir == 'call' else 'LOD'} "
                                      f"breach to {hod if trigger_dir == 'call' else lod:.1f} "
                                      f"(prev {prev_extreme:.1f}, min={breach_min})",
                            ))
                        prev_extreme = hod if trigger_dir == "call" else lod

        # Update HOD/LOD for result
        if not equity_df.empty:
            h, l = get_hod_lod_in_range(equity_df, _time_to_mins("06:30"), _time_to_mins("13:00"))
            result.hod = h if h > 0 else result.open_price
            result.lod = l if l < float("inf") else result.open_price

        # --- Roll check at configured time ---
        roll_mins = _time_to_mins(roll_time)
        roll_price = get_price_at_time(equity_prices, roll_time)
        rolled_ids: set = set()

        if roll_price is not None and positions:
            roll_snap = snap_options_to_time(options_0dte, roll_time, tolerance_mins=5)
            for i, pos in enumerate(positions):
                dist_pct = abs(roll_price - pos.short_strike) / roll_price if roll_price > 0 else 1
                is_itm = self._is_position_itm(pos, roll_price)
                if not (is_itm or dist_pct <= roll_proximity):
                    continue

                # Close position
                close_debit = None
                if not roll_snap.empty:
                    close_debit = close_spread_cost(roll_snap, pos)
                if close_debit is None:
                    if pos.direction == "call":
                        intr = max(0, roll_price - pos.short_strike)
                    else:
                        intr = max(0, pos.short_strike - roll_price)
                    close_debit = min(intr, pos.width)

                close_total = close_debit * 100 * pos.num_contracts
                result.total_debits += close_total
                result.total_commissions += commission

                # Find DTE+N target
                target_date = trade_date
                for _ in range(roll_dte):
                    nd = get_next_trading_date(target_date, all_dates)
                    if nd:
                        target_date = nd
                if target_date == trade_date:
                    result.trades.append(TradeRecord(
                        event="roll_close", time_pacific=roll_time,
                        direction=pos.direction,
                        short_strike=pos.short_strike, long_strike=pos.long_strike,
                        width=pos.width, credit_or_debit=-close_total,
                        num_contracts=pos.num_contracts, commission=commission,
                        underlying_price=roll_price, dte=pos.dte,
                        notes=f"No DTE+{roll_dte} date",
                    ))
                    rolled_ids.add(i)
                    continue

                # Try to find DTE+N options (check expiration column or load separate file)
                dte_opts = None
                if options_0dte is not None and 'expiration' in options_0dte.columns:
                    dte_filtered = options_0dte[options_0dte['expiration'] == target_date]
                    if not dte_filtered.empty:
                        dte_opts = dte_filtered
                if dte_opts is None:
                    dte_opts = load_dte1_options(ticker, target_date, trade_date,
                                                 self.config.get("options_dte1_dir", "csv_exports/options"),
                                                 fallback_options_dir=self.config.get("options_0dte_dir", "options_csv_output_full_5"))

                new_spread = None
                if dte_opts is not None and not dte_opts.empty:
                    dte_snap = snap_options_to_time(dte_opts, roll_time, tolerance_mins=30)
                    if not dte_snap.empty:
                        new_spread = find_credit_spread(
                            dte_snap, roll_price, pos.direction,
                            spread_width, spread_width,
                            leg_placement="itm")

                if new_spread is not None:
                    new_cr = new_spread["credit"] * 100 * pos.num_contracts
                    result.total_credits += new_cr
                    result.total_commissions += commission
                    result.num_rolls += 1

                    rp_existing = carried_map.get(i)
                    if rp_existing:
                        rc = rp_existing.roll_count + 1
                        orig_date = rp_existing.original_entry_date
                        orig_cr = rp_existing.original_credit
                        cum_cost = rp_existing.cumulative_roll_cost + max(0, close_total - new_cr)
                    else:
                        rc = 1
                        orig_date = trade_date
                        orig_cr = pos.total_credit
                        cum_cost = max(0, close_total - new_cr)

                    result.trades.append(TradeRecord(
                        event="roll_close", time_pacific=roll_time,
                        direction=pos.direction,
                        short_strike=pos.short_strike, long_strike=pos.long_strike,
                        width=pos.width, credit_or_debit=-close_total,
                        num_contracts=pos.num_contracts, commission=commission,
                        underlying_price=roll_price, dte=pos.dte,
                        notes=f"Close for roll to DTE+{roll_dte}",
                    ))
                    result.trades.append(TradeRecord(
                        event="roll_open", time_pacific=roll_time,
                        direction=pos.direction,
                        short_strike=new_spread["short_strike"],
                        long_strike=new_spread["long_strike"],
                        width=new_spread["width"], credit_or_debit=new_cr,
                        num_contracts=pos.num_contracts, commission=commission,
                        underlying_price=roll_price, dte=roll_dte,
                        notes=f"Rolled to {target_date} (DTE+{roll_dte}, roll#{rc})",
                    ))
                    new_carries.append(RolledPosition(
                        direction=pos.direction,
                        short_strike=new_spread["short_strike"],
                        long_strike=new_spread["long_strike"],
                        width=new_spread["width"],
                        credit_per_share=new_spread["credit"],
                        num_contracts=pos.num_contracts,
                        expiration_date=target_date,
                        original_entry_date=orig_date,
                        original_credit=orig_cr,
                        cumulative_roll_cost=cum_cost,
                        roll_count=rc,
                    ))
                else:
                    result.trades.append(TradeRecord(
                        event="roll_close", time_pacific=roll_time,
                        direction=pos.direction,
                        short_strike=pos.short_strike, long_strike=pos.long_strike,
                        width=pos.width, credit_or_debit=-close_total,
                        num_contracts=pos.num_contracts, commission=commission,
                        underlying_price=roll_price, dte=pos.dte,
                        notes=f"Roll failed: no DTE+{roll_dte} spread",
                    ))
                rolled_ids.add(i)

        # --- 13:00: settle remaining positions ---
        close = result.close_price
        for i, pos in enumerate(positions):
            if i in rolled_ids:
                continue

            if is_debit and pos.credit_per_share < 0:
                # Debit spread: profit when spread finishes ITM
                # For call debit: profit = max(0, close - long_strike) - max(0, close - short_strike)
                # Simplified: if close > short_strike → max profit = width
                #             if close < long_strike → max loss = debit paid (0 payout)
                #             in between → partial profit
                if pos.direction == "call":
                    long_val = max(0, close - pos.long_strike)
                    short_val = max(0, close - pos.short_strike)
                else:
                    long_val = max(0, pos.long_strike - close)
                    short_val = max(0, pos.short_strike - close)
                payout = long_val - short_val
                payout = max(0, min(payout, pos.width))
                payout_total = payout * 100 * pos.num_contracts
                if payout_total > 0:
                    result.total_credits += payout_total
                    result.trades.append(TradeRecord(
                        event="expiration_itm", time_pacific="13:00",
                        direction=pos.direction,
                        short_strike=pos.short_strike, long_strike=pos.long_strike,
                        width=pos.width, credit_or_debit=payout_total,
                        num_contracts=pos.num_contracts, commission=0,
                        underlying_price=close, dte=pos.dte,
                        notes=f"DEBIT spread payout={payout:.2f}/share",
                    ))
                else:
                    result.trades.append(TradeRecord(
                        event="expiration_otm", time_pacific="13:00",
                        direction=pos.direction,
                        short_strike=pos.short_strike, long_strike=pos.long_strike,
                        width=pos.width, credit_or_debit=0,
                        num_contracts=pos.num_contracts, commission=0,
                        underlying_price=close, dte=pos.dte,
                        notes="DEBIT spread expired worthless",
                    ))
            else:
                # Credit spread: standard settlement
                is_itm = self._is_position_itm(pos, close)
                if is_itm:
                    if pos.direction == "call":
                        intrinsic = min(close - pos.short_strike, pos.width)
                    else:
                        intrinsic = min(pos.short_strike - close, pos.width)
                    loss = intrinsic * 100 * pos.num_contracts
                    result.total_debits += loss
                    result.trades.append(TradeRecord(
                        event="expiration_itm", time_pacific="13:00",
                        direction=pos.direction,
                        short_strike=pos.short_strike, long_strike=pos.long_strike,
                        width=pos.width, credit_or_debit=-loss,
                        num_contracts=pos.num_contracts, commission=0,
                        underlying_price=close, dte=pos.dte,
                        notes=f"ITM by {intrinsic:.2f}",
                    ))
                else:
                    result.trades.append(TradeRecord(
                        event="expiration_otm", time_pacific="13:00",
                        direction=pos.direction,
                        short_strike=pos.short_strike, long_strike=pos.long_strike,
                        width=pos.width, credit_or_debit=0,
                        num_contracts=pos.num_contracts, commission=0,
                        underlying_price=close, dte=pos.dte,
                        notes="Expired OTM — full profit",
                    ))

        result.final_pnl = result.net_pnl
        return (result, new_carries)

    def _attempt_dte1_roll_layer(self, result: DayResult, position: SpreadPosition,
                                  ticker: str, trade_date: str, all_dates: List[str],
                                  options_0dte: Optional[pd.DataFrame],
                                  time_str: str, price: float,
                                  min_step: float, max_width: float,
                                  commission: float,
                                  existing_rp: Optional[RolledPosition] = None,
                                  ) -> Optional[RolledPosition]:
        """Roll a position to DTE+1. Returns RolledPosition if successful, else None.

        If existing_rp is provided, this is a re-roll of a carried position.
        """
        # Close current position — try market quotes, fall back to intrinsic
        close_debit = None
        if options_0dte is not None and not options_0dte.empty:
            close_snap = snap_options_to_time(options_0dte, time_str)
            if not close_snap.empty:
                close_debit = close_spread_cost(close_snap, position)

        if close_debit is None:
            if position.direction == "call":
                intrinsic = max(0, price - position.short_strike)
            else:
                intrinsic = max(0, position.short_strike - price)
            close_debit = min(intrinsic, position.width)

        close_total = close_debit * 100 * position.num_contracts

        # Load DTE+1 options
        next_date = get_next_trading_date(trade_date, all_dates)
        if next_date is None:
            result.total_debits += close_total
            result.total_commissions += commission
            result.trades.append(TradeRecord(
                event="eod_roll_close", time_pacific=time_str,
                direction=position.direction,
                short_strike=position.short_strike, long_strike=position.long_strike,
                width=position.width, credit_or_debit=-close_total,
                num_contracts=position.num_contracts, commission=commission,
                underlying_price=price, dte=position.dte,
                notes="No next trading date",
            ))
            return None

        dte1_opts = load_dte1_options(ticker, next_date, trade_date,
                                      self.config["options_dte1_dir"],
                                      fallback_options_dir=self.config.get("options_0dte_dir", "options_csv_output_full_5"))
        if dte1_opts is None or dte1_opts.empty:
            result.total_debits += close_total
            result.total_commissions += commission
            result.trades.append(TradeRecord(
                event="eod_roll_close", time_pacific=time_str,
                direction=position.direction,
                short_strike=position.short_strike, long_strike=position.long_strike,
                width=position.width, credit_or_debit=-close_total,
                num_contracts=position.num_contracts, commission=commission,
                underlying_price=price, dte=position.dte,
                notes="No DTE+1 options",
            ))
            return None

        dte1_snap = snap_options_to_time(dte1_opts, time_str, tolerance_mins=30)
        if dte1_snap.empty:
            result.total_debits += close_total
            result.total_commissions += commission
            result.trades.append(TradeRecord(
                event="eod_roll_close", time_pacific=time_str,
                direction=position.direction,
                short_strike=position.short_strike, long_strike=position.long_strike,
                width=position.width, credit_or_debit=-close_total,
                num_contracts=position.num_contracts, commission=commission,
                underlying_price=price, dte=position.dte,
                notes="No DTE+1 snapshot",
            ))
            return None

        # Find new spread — try progressively wider until credit covers close debit
        leg_placement = self.config.get("call_track_leg_placement", "nearest")
        depth_pct = self.config.get("call_track_depth_pct")
        roll_max_width_mult = self.config.get("roll_max_width_mult", 5)
        roll_max_contract_mult = self.config.get("roll_max_contract_mult", 2)

        new_spread = None
        new_contracts = position.num_contracts

        if self.config.get("roll_match_contracts", True):
            # Try widths from min up to roll_max_width_mult × original width
            best_spread = None
            orig_width = position.width
            cap_width = orig_width * roll_max_width_mult
            max_contracts = int(position.num_contracts * roll_max_contract_mult)

            for try_max_w in range(int(min_step), int(cap_width) + 1, int(min_step)):
                sp = find_credit_spread(dte1_snap, price, position.direction,
                                        min_step, try_max_w,
                                        leg_placement=leg_placement,
                                        depth_pct=depth_pct)
                if sp is not None:
                    best_spread = sp
                    # Check if 1:1 contracts covers the close debit
                    if sp["credit"] * 100 * position.num_contracts >= close_total:
                        break  # found a width that covers at 1:1

            if best_spread is not None:
                new_spread = best_spread
                credit_at_match = new_spread["credit"] * 100 * position.num_contracts
                if credit_at_match >= close_total:
                    new_contracts = position.num_contracts
                else:
                    # Widen didn't fully cover — allow up to max_contract_mult
                    needed = math.ceil(close_total / (new_spread["credit"] * 100))
                    new_contracts = min(needed, max_contracts)
        else:
            new_spread = find_credit_spread(dte1_snap, price, position.direction,
                                            min_step, max_width,
                                            leg_placement=leg_placement,
                                            depth_pct=depth_pct)
            if new_spread is not None:
                calc = self._calc_roll_contracts(
                    close_total, new_spread["credit"], new_spread["width"])
                new_contracts = calc if calc is not None else position.num_contracts

        # Apply chain contract cap
        chain_cap = self.config.get("roll_max_chain_contracts")
        if chain_cap and new_spread is not None and new_contracts > chain_cap:
            new_contracts = chain_cap

        if new_spread is None:
            result.total_debits += close_total
            result.total_commissions += commission
            result.trades.append(TradeRecord(
                event="eod_roll_close", time_pacific=time_str,
                direction=position.direction,
                short_strike=position.short_strike, long_strike=position.long_strike,
                width=position.width, credit_or_debit=-close_total,
                num_contracts=position.num_contracts, commission=commission,
                underlying_price=price, dte=position.dte,
                notes="No DTE+1 spread found",
            ))
            return None

        new_credit_total = new_spread["credit"] * 100 * new_contracts
        roll_cost = close_total - new_credit_total

        result.total_debits += close_total
        result.total_commissions += commission
        event_prefix = "eod_roll"
        result.trades.append(TradeRecord(
            event=f"{event_prefix}_close", time_pacific=time_str,
            direction=position.direction,
            short_strike=position.short_strike, long_strike=position.long_strike,
            width=position.width, credit_or_debit=-close_total,
            num_contracts=position.num_contracts, commission=commission,
            underlying_price=price, dte=position.dte,
            notes="eod_itm_layer trigger",
        ))

        result.total_credits += new_credit_total
        result.total_commissions += commission
        result.eod_rolled_to_dte1 = True

        # Determine roll lineage
        if existing_rp is not None:
            new_roll_count = existing_rp.roll_count + 1
            original_entry_date = existing_rp.original_entry_date
            original_credit = existing_rp.original_credit
            cumulative_cost = existing_rp.cumulative_roll_cost + max(0, roll_cost)
        else:
            new_roll_count = 1
            original_entry_date = trade_date
            original_credit = position.total_credit
            cumulative_cost = max(0, roll_cost)

        result.trades.append(TradeRecord(
            event=f"{event_prefix}_open", time_pacific=time_str,
            direction=position.direction,
            short_strike=new_spread["short_strike"], long_strike=new_spread["long_strike"],
            width=new_spread["width"], credit_or_debit=new_credit_total,
            num_contracts=new_contracts, commission=commission,
            underlying_price=price, dte=1,
            notes=f"Rolled to {next_date} (roll#{new_roll_count}, cum_cost=${cumulative_cost:.0f})",
        ))

        return RolledPosition(
            direction=position.direction,
            short_strike=new_spread["short_strike"],
            long_strike=new_spread["long_strike"],
            width=new_spread["width"],
            credit_per_share=new_spread["credit"],
            num_contracts=new_contracts,
            expiration_date=next_date,
            original_entry_date=original_entry_date,
            original_credit=original_credit,
            cumulative_roll_cost=cumulative_cost,
            roll_count=new_roll_count,
        )

    def _handle_eod_roll(self, result, position, direction, ticker, trade_date,
                         all_dates, equity_prices, options_0dte,
                         min_step, max_width, commission):
        """EOD proximity check — roll to DTE+1 if ITM or threatened."""
        prox_start = self.config["proximity_check_start_pacific"]
        prox_end = self.config["proximity_check_end_pacific"]
        prox_interval = self.config["proximity_check_interval_mins"]
        prox_threshold = self.config["proximity_threshold_pct"]

        t = _time_to_mins(prox_start)
        end_mins = _time_to_mins(prox_end)

        while t <= end_mins:
            time_str = f"{t // 60:02d}:{t % 60:02d}"
            price = get_price_at_time(equity_prices, time_str)
            if price is not None:
                is_itm = self._is_position_itm(position, price)
                if direction == "call":
                    is_threatened = (position.short_strike - price) / price < prox_threshold
                else:
                    is_threatened = (price - position.short_strike) / price < prox_threshold

                if is_itm or is_threatened:
                    rolled = self._attempt_dte1_roll(
                        result, position, ticker, trade_date, all_dates,
                        options_0dte, time_str, price,
                        min_step, max_width, commission, "eod_itm")
                    if rolled:
                        return None  # position rolled away
                break
            t += prox_interval

        return position

    def _attempt_dte1_roll(self, result, position, ticker, trade_date,
                           all_dates, options_0dte, time_str, price,
                           min_step, max_width, commission, mode) -> bool:
        """Roll to DTE+1. Returns True if successful."""
        # Close 0DTE — try market quotes first, fall back to intrinsic value
        close_debit = None
        if options_0dte is not None and not options_0dte.empty:
            close_snap = snap_options_to_time(options_0dte, time_str)
            if not close_snap.empty:
                close_debit = close_spread_cost(close_snap, position)

        if close_debit is None:
            # Estimate close cost from intrinsic value
            if position.direction == "call":
                intrinsic = max(0, price - position.short_strike)
            else:
                intrinsic = max(0, position.short_strike - price)
            close_debit = min(intrinsic, position.width)

        close_total = close_debit * 100 * position.num_contracts

        # Load DTE+1 options
        next_date = get_next_trading_date(trade_date, all_dates)
        if next_date is None:
            # Can't open DTE+1, just close
            result.total_debits += close_total
            result.total_commissions += commission
            result.trades.append(TradeRecord(
                event="eod_roll_close", time_pacific=time_str,
                direction=position.direction,
                short_strike=position.short_strike, long_strike=position.long_strike,
                width=position.width, credit_or_debit=-close_total,
                num_contracts=position.num_contracts, commission=commission,
                underlying_price=price, dte=0, notes="No next trading date",
            ))
            return True

        dte1_opts = load_dte1_options(ticker, next_date, trade_date,
                                      self.config["options_dte1_dir"],
                                      fallback_options_dir=self.config.get("options_0dte_dir", "options_csv_output_full_5"))
        if dte1_opts is None or dte1_opts.empty:
            result.total_debits += close_total
            result.total_commissions += commission
            result.trades.append(TradeRecord(
                event="eod_roll_close", time_pacific=time_str,
                direction=position.direction,
                short_strike=position.short_strike, long_strike=position.long_strike,
                width=position.width, credit_or_debit=-close_total,
                num_contracts=position.num_contracts, commission=commission,
                underlying_price=price, dte=0, notes="No DTE+1 options",
            ))
            return True

        dte1_snap = snap_options_to_time(dte1_opts, time_str, tolerance_mins=30)
        if dte1_snap.empty:
            result.total_debits += close_total
            result.total_commissions += commission
            result.trades.append(TradeRecord(
                event="eod_roll_close", time_pacific=time_str,
                direction=position.direction,
                short_strike=position.short_strike, long_strike=position.long_strike,
                width=position.width, credit_or_debit=-close_total,
                num_contracts=position.num_contracts, commission=commission,
                underlying_price=price, dte=0, notes="No DTE+1 snapshot",
            ))
            return True

        # Use call_track placement if in that mode, else directional settings
        if self.config.get("strategy_mode") == "call_track":
            leg_placement = self.config.get("call_track_leg_placement", "nearest")
            depth_pct = self.config.get("call_track_depth_pct")
        else:
            leg_placement = self.config.get("leg_placement", "otm")
            depth_pct = self.config.get("depth_pct")
        new_spread = find_credit_spread(dte1_snap, price, position.direction,
                                        min_step, max_width,
                                        leg_placement=leg_placement,
                                        depth_pct=depth_pct)
        if new_spread is None:
            result.total_debits += close_total
            result.total_commissions += commission
            result.trades.append(TradeRecord(
                event="eod_roll_close", time_pacific=time_str,
                direction=position.direction,
                short_strike=position.short_strike, long_strike=position.long_strike,
                width=position.width, credit_or_debit=-close_total,
                num_contracts=position.num_contracts, commission=commission,
                underlying_price=price, dte=0, notes="No DTE+1 spread found",
            ))
            return True

        # Conditional mode: check if new credit covers enough of close debit
        if mode == "conditional_dte1":
            min_recovery = self.config.get("conditional_roll_min_recovery", 0.50)
            new_contracts = self._calc_roll_contracts(
                close_total, new_spread["credit"], new_spread["width"])
            if new_contracts is not None:
                projected_credit = new_spread["credit"] * 100 * new_contracts
                if close_total > 0 and projected_credit / close_total < min_recovery:
                    return False  # Don't roll — recovery too low

        # Commit the close
        result.total_debits += close_total
        result.total_commissions += commission
        event_prefix = "midday_dte1" if mode in ("midday_dte1", "conditional_dte1") else "eod_roll"
        result.trades.append(TradeRecord(
            event=f"{event_prefix}_close", time_pacific=time_str,
            direction=position.direction,
            short_strike=position.short_strike, long_strike=position.long_strike,
            width=position.width, credit_or_debit=-close_total,
            num_contracts=position.num_contracts, commission=commission,
            underlying_price=price, dte=0,
            notes=f"{mode} trigger",
        ))

        # Open DTE+1
        new_contracts = self._calc_roll_contracts(
            close_total, new_spread["credit"], new_spread["width"])
        if new_contracts is None:
            return True

        new_credit = new_spread["credit"] * 100 * new_contracts
        result.total_credits += new_credit
        result.total_commissions += commission
        result.eod_rolled_to_dte1 = True
        result.num_rolls += 1
        result.trades.append(TradeRecord(
            event=f"{event_prefix}_open", time_pacific=time_str,
            direction=position.direction,
            short_strike=new_spread["short_strike"], long_strike=new_spread["long_strike"],
            width=new_spread["width"], credit_or_debit=new_credit,
            num_contracts=new_contracts, commission=commission,
            underlying_price=price, dte=1,
            notes=f"Rolled to {next_date}",
        ))
        return True
