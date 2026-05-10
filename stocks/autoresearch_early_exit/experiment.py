"""
experiment.py — Early-exit threshold sweep engine.

Given a config dict describing an early-exit rule, simulates the rule against
the pre-computed per-trade intraday observations and returns portfolio metrics
compared to a hold-to-expiry baseline.

Usage (standalone):
    python3 experiment.py

Data dependency:
    Run first (once):
        python3 run_early_exit_analysis.py --save-raw
    Produces: results/dte_comparison/early_exit_raw_obs.parquet
"""
from __future__ import annotations

import math
from collections import defaultdict
from pathlib import Path
from statistics import mean, pstdev

import pandas as pd

ROOT = Path(__file__).parent.parent
RAW_OBS_PATH = ROOT / "results" / "dte_comparison" / "early_exit_raw_obs.parquet"

CAPITAL = 1_000_000.0
TRADING_DAYS_PER_YEAR = 252

# ── Config space ──────────────────────────────────────────────────────────────
# profit_target_pct  : close when pct_captured >= X  (integer, 30–95)
# stop_loss_pct      : close when pct_captured <= -Y  (integer, 50–300; 9999=disabled)
# apply_dtes         : list of DTE ints to apply the rule to
# apply_tiers        : list of tier strings ("aggressive","moderate","conservative")
# apply_sides        : list of sides ("put","call")

PROFIT_TARGETS = list(range(30, 96, 5))          # 30, 35, … 95
STOP_LOSSES    = [9999, 50, 100, 150, 200, 300]  # 9999 = disabled
APPLY_DTES     = [[0], [1, 2], [2, 3], [1, 2, 3], [0, 1, 2, 3]]
APPLY_TIERS    = [
    ["aggressive"],
    ["moderate"],
    ["conservative"],
    ["aggressive", "moderate"],
    ["moderate", "conservative"],
    ["aggressive", "moderate", "conservative"],
]
APPLY_SIDES    = [["put"], ["call"], ["put", "call"]]

ALL_DTES  = [0, 1, 2, 3]
ALL_TIERS = ["aggressive", "moderate", "conservative"]
ALL_SIDES = ["put", "call"]


def default_config() -> dict:
    return {
        "profit_target_pct": 70,
        "stop_loss_pct": 9999,
        "apply_dtes": [1, 2, 3],
        "apply_tiers": ["aggressive", "moderate", "conservative"],
        "apply_sides": ["put"],
    }


# ── Data loading (module-level cache) ────────────────────────────────────────

_RAW: pd.DataFrame | None = None


def _load() -> pd.DataFrame:
    global _RAW
    if _RAW is None:
        if not RAW_OBS_PATH.exists():
            raise FileNotFoundError(
                f"Raw observations not found at {RAW_OBS_PATH}.\n"
                "Run:  python3 run_early_exit_analysis.py --save-raw"
            )
        df = pd.read_parquet(RAW_OBS_PATH)
        # Ensure types
        df["date"] = df["date"].astype(str)
        df["dte"]  = df["dte"].astype(int)
        df["pct_captured"]  = df["pct_captured"].astype(float)
        df["entry_credit"]  = df["entry_credit"].astype(float)
        df["max_loss"]      = df["max_loss"].astype(float)
        df["roi_pct"]       = df["roi_pct"].astype(float)
        df["nroi"]          = df["nroi"].astype(float)
        df["hours_elapsed"] = df["hours_elapsed"].astype(float)
        _RAW = df
    return _RAW


# ── Trade reconstruction ──────────────────────────────────────────────────────

def _build_trade_trajectories(df: pd.DataFrame) -> dict[tuple, list[dict]]:
    """Group raw observations by unique trade key, returning snaps in time order."""
    trades: dict[tuple, list[dict]] = defaultdict(list)
    for _, row in df.iterrows():
        key = (row["date"], row["ticker"], int(row["dte"]),
               row["tier"], row["side"])
        trades[key].append({
            "snap_label":    row["snap_label"],
            "hours_elapsed": float(row["hours_elapsed"]),
            "pct_captured":  float(row["pct_captured"]),
            "entry_credit":  float(row["entry_credit"]),
            "max_loss":      float(row["max_loss"]),
            "roi_pct":       float(row["roi_pct"]),
            "nroi":          float(row["nroi"]),
        })
    # Sort each trade's snaps by hours_elapsed
    for key in trades:
        trades[key].sort(key=lambda r: r["hours_elapsed"])
    return trades


# ── Simulation ────────────────────────────────────────────────────────────────

def simulate_rule(config: dict) -> dict:
    """
    Simulate an early-exit rule against all trades in the raw obs.

    For each trade:
    - If the rule applies (DTE/tier/side match), scan snaps in order.
    - Exit at the first snap where pct_captured >= profit_target_pct
      OR pct_captured <= -stop_loss_pct.
    - Realized P&L (per unit) = pct_captured/100 * entry_credit (early)
                               = roi_pct/100 * max_loss          (hold)
    - Both are normalised to $ per $1 of max_loss (ROI units):
        hold_roi   = roi_pct / 100
        early_roi  = (pct_captured / 100 * entry_credit) / max_loss

    Returns a dict with per-date portfolio P&L comparing early-exit vs hold
    (normalised so $1M of max_loss is deployed across all trades each day).
    """
    df = _load()

    profit_target  = config["profit_target_pct"]
    stop_loss      = config["stop_loss_pct"]    # disabled if 9999
    apply_dtes     = set(config["apply_dtes"])
    apply_tiers    = set(config["apply_tiers"])
    apply_sides    = set(config["apply_sides"])

    trades = _build_trade_trajectories(df)

    # Per-date P&L (dict: date_str → {hold: float, early: float})
    daily_hold:  dict[str, float] = defaultdict(float)
    daily_early: dict[str, float] = defaultdict(float)

    for (date_str, ticker, dte, tier, side), snaps in trades.items():
        if not snaps:
            continue
        # Use last snap's metadata (same for all snaps of a trade)
        s0 = snaps[0]
        entry_credit = s0["entry_credit"]
        max_loss_val = s0["max_loss"]
        roi_pct_hold = s0["roi_pct"]

        if max_loss_val <= 0:
            continue

        # Normalise: roi per $1 of max_loss
        hold_roi = roi_pct_hold / 100.0

        rule_applies = (
            dte  in apply_dtes  and
            tier in apply_tiers and
            side in apply_sides
        )

        if not rule_applies:
            # No rule → hold to expiry
            daily_hold[date_str]  += hold_roi
            daily_early[date_str] += hold_roi
            continue

        # Scan snaps for early exit trigger
        exit_roi: float | None = None
        for snap in snaps:
            cap = snap["pct_captured"]
            if cap >= profit_target:
                exit_roi = (cap / 100.0 * entry_credit) / max_loss_val
                break
            if stop_loss < 9999 and cap <= -stop_loss:
                exit_roi = (cap / 100.0 * entry_credit) / max_loss_val
                break

        daily_hold[date_str]  += hold_roi
        daily_early[date_str] += exit_roi if exit_roi is not None else hold_roi

    return daily_hold, daily_early


def _portfolio_metrics(daily_pnl: dict[str, float], n_trades: int) -> dict:
    if not daily_pnl:
        return {"annualized_sharpe": -9.0, "annualized_roi_pct": 0.0,
                "max_drawdown_pct": 100.0, "win_day_pct": 0.0, "n_days": 0}
    pnls = [v for _, v in sorted(daily_pnl.items())]
    n = len(pnls)
    avg = mean(pnls)
    std = pstdev(pnls) if n > 1 else 1e-9
    daily_sharpe = avg / std if std > 1e-12 else 0.0
    ann_sharpe   = daily_sharpe * math.sqrt(TRADING_DAYS_PER_YEAR)
    ann_roi      = avg * TRADING_DAYS_PER_YEAR * 100.0
    # Drawdown on cumulative ROI curve
    eq = [0.0]
    for p in pnls:
        eq.append(eq[-1] + p)
    peak = 0.0; max_dd = 0.0
    for e in eq:
        if e > peak:
            peak = e
        dd = peak - e
        if dd > max_dd:
            max_dd = dd
    # Express dd as % of peak+1 (avoid div/0 on small portfolios)
    max_dd_pct = max_dd / (abs(peak) + 1.0) * 100.0
    win_days = sum(1 for p in pnls if p > 0)
    return {
        "annualized_sharpe":  ann_sharpe,
        "annualized_roi_pct": ann_roi,
        "max_drawdown_pct":   max_dd_pct,
        "win_day_pct":        100.0 * win_days / n,
        "n_days":             n,
        "n_trades":           n_trades,
    }


def score_metrics(m: dict) -> float:
    if m["annualized_roi_pct"] <= 0:
        return m["annualized_sharpe"] - 1.0
    dd_pen = min(1.0, m["max_drawdown_pct"] / 30.0)
    return m["annualized_sharpe"] * (1.0 - dd_pen)


def run_experiment(config: dict) -> dict:
    """
    Run one early-exit config and return metrics dict.

    Returns keys: score, annualized_sharpe, annualized_roi_pct, max_drawdown_pct,
                  win_day_pct, n_days, n_trades, improvement_pct
    (improvement_pct = early-exit sharpe vs hold-to-expiry sharpe, in %)
    """
    try:
        daily_hold, daily_early = simulate_rule(config)
    except Exception as e:
        return {"score": -999.0, "reason": str(e)}

    n_trades = len(_build_trade_trajectories(_load()))

    m_hold  = _portfolio_metrics(daily_hold, n_trades)
    m_early = _portfolio_metrics(daily_early, n_trades)

    m_early["score"] = score_metrics(m_early)
    sh_hold = m_hold["annualized_sharpe"]
    sh_early = m_early["annualized_sharpe"]
    m_early["improvement_pct"] = (
        (sh_early - sh_hold) / abs(sh_hold) * 100.0
        if abs(sh_hold) > 0.01 else 0.0
    )
    m_early["hold_sharpe"] = sh_hold
    return m_early


# ── Standalone test ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    cfg = default_config()
    print(f"Default config: {cfg}")
    r = run_experiment(cfg)
    print(f"  score          = {r.get('score', 'NA'):.4f}")
    print(f"  ann_sharpe     = {r.get('annualized_sharpe', 0):.3f}  (hold={r.get('hold_sharpe',0):.3f})")
    print(f"  ann_roi        = {r.get('annualized_roi_pct', 0):.2f}%")
    print(f"  max_dd         = {r.get('max_drawdown_pct', 0):.2f}%")
    print(f"  win_days       = {r.get('win_day_pct', 0):.1f}%")
    print(f"  improvement    = {r.get('improvement_pct', 0):+.1f}%  vs hold")
    print(f"  n_trades       = {r.get('n_trades', 0)}")
