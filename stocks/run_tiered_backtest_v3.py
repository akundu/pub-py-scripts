#!/usr/bin/env python3
"""Tiered Portfolio Backtest v3 — Cross-Ticker Selection with Volume Awareness.

Reuses v2's per-ticker×tier backtests, then applies a cross-ticker selection
layer: for each tier at each 10-minute interval, compares all available tickers
and picks the best opportunity based on credit/risk ratio, volume adequacy, and
bid-ask tightness.

Key differences from v2:
  - Single unified $500K/day budget shared across all tickers
  - Volume-adjusted contract sizing (can't trade 132 contracts if volume is 8)
  - Cross-ticker scoring selects the best ticker per tier per interval
  - Shows which ticker won and why, plus volume impact analysis

Usage:
  python run_tiered_backtest_v3.py                     # Full run (backtests + analysis)
  python run_tiered_backtest_v3.py --analyze           # Skip backtests, re-analyze
  python run_tiered_backtest_v3.py --no-volume-cap     # Disable volume caps (v2-style)
  python run_tiered_backtest_v3.py --volume-fill-pct 0.50  # Allow 50% of volume
  python run_tiered_backtest_v3.py --weights 0.5,0.25,0.25  # Custom scoring weights
"""

import argparse
import math
import os
import sys
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore", category=FutureWarning)

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "results" / "tiered_portfolio_v3"
OPTIONS_DIR = BASE_DIR / "options_csv_output_full"

sys.path.insert(0, str(BASE_DIR))
from scripts.live_trading.advisor.tier_config import (
    TIERS,
    TICKERS,
    TICKER_PARAMS,
    MAX_RISK_PER_TRADE,
    DAILY_BUDGET,
    MAX_TRADES_PER_WINDOW,
    TRADE_WINDOW_MINUTES,
    STRATEGY_DEFAULTS,
    get_spread_width,
    get_ticker_param,
)

# Import v2 functions for backtest execution and trade loading
from run_tiered_backtest_v2 import (
    run_all_backtests,
    load_all_trades,
    compute_metrics,
)

# v2 results directory (backtests write here)
V2_OUTPUT_DIR = BASE_DIR / "results" / "tiered_portfolio_v2"


# ---------------------------------------------------------------------------
# Phase 1: Options CSV cache for volume/bid-ask lookups
# ---------------------------------------------------------------------------

_options_cache: dict = {}


def _load_options_csv(ticker: str, trading_date: str) -> pd.DataFrame:
    """Load options CSV for a ticker and date, cached per (ticker, date)."""
    key = (ticker, trading_date)
    if key in _options_cache:
        return _options_cache[key]

    csv_dir = OPTIONS_DIR / ticker
    # Try common naming patterns
    for pattern in [
        f"{ticker}_options_{trading_date}.csv",
        f"*_{trading_date}.csv",
        f"*{trading_date}*.csv",
    ]:
        matches = list(csv_dir.glob(pattern))
        if matches:
            df = pd.read_csv(matches[0])
            df["volume"] = pd.to_numeric(df.get("volume", 0), errors="coerce").fillna(0).astype(int)
            if "bid" in df.columns:
                df["bid"] = pd.to_numeric(df["bid"], errors="coerce").fillna(0)
            if "ask" in df.columns:
                df["ask"] = pd.to_numeric(df["ask"], errors="coerce").fillna(0)
            _options_cache[key] = df
            return df

    _options_cache[key] = pd.DataFrame()
    return pd.DataFrame()


def _get_strike_liquidity(options_df: pd.DataFrame, strike: float,
                          option_type: str) -> dict:
    """Get volume and bid-ask for a specific strike from the options chain.

    Aggregates across timestamps: uses max volume and the latest bid/ask.
    """
    if options_df.empty:
        return {"volume": 0, "bid": 0, "ask": 0, "bid_ask_spread_pct": 1.0}

    mask = (options_df["strike"] == strike) & (options_df["type"] == option_type)
    rows = options_df[mask]
    if rows.empty:
        # Try nearest strike within 1 point
        mask_near = (abs(options_df["strike"] - strike) <= 1) & (options_df["type"] == option_type)
        rows = options_df[mask_near]
    if rows.empty:
        return {"volume": 0, "bid": 0, "ask": 0, "bid_ask_spread_pct": 1.0}

    vol = int(rows["volume"].max())
    # Use the row with highest volume for bid/ask (most representative)
    best = rows.loc[rows["volume"].idxmax()]
    bid = float(best.get("bid", 0))
    ask = float(best.get("ask", 0))
    mid = (bid + ask) / 2 if (bid + ask) > 0 else 1
    ba_spread = (ask - bid) / mid if mid > 0 else 1.0

    return {"volume": vol, "bid": bid, "ask": ask, "bid_ask_spread_pct": ba_spread}


# ---------------------------------------------------------------------------
# Phase 2: Enrich trades with liquidity data
# ---------------------------------------------------------------------------

def enrich_trades_with_liquidity(trades: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """Add volume and bid-ask data to each trade by looking up options CSVs.

    Adds columns: short_volume, long_volume, min_leg_volume,
    short_bid_ask_pct, long_bid_ask_pct, avg_bid_ask_pct.
    """
    if trades.empty:
        return trades

    trades = trades.copy()

    # Pre-compute trading_date strings
    if "trading_date" in trades.columns:
        date_col = trades["trading_date"]
    elif "entry_dt" in trades.columns:
        date_col = trades["entry_dt"].dt.date.astype(str)
    else:
        date_col = trades["entry_date"].astype(str)

    short_vols, long_vols, min_vols = [], [], []
    short_bas, long_bas, avg_bas = [], [], []

    for idx, row in trades.iterrows():
        td = str(date_col.loc[idx])
        options_df = _load_options_csv(ticker, td)

        otype = row.get("option_type", "put")
        short_strike = row.get("short_strike", 0)
        long_strike = row.get("long_strike", 0)

        short_liq = _get_strike_liquidity(options_df, short_strike, otype)
        long_liq = _get_strike_liquidity(options_df, long_strike, otype)

        short_vols.append(short_liq["volume"])
        long_vols.append(long_liq["volume"])
        min_vols.append(min(short_liq["volume"], long_liq["volume"]))
        short_bas.append(short_liq["bid_ask_spread_pct"])
        long_bas.append(long_liq["bid_ask_spread_pct"])
        avg_bas.append((short_liq["bid_ask_spread_pct"] + long_liq["bid_ask_spread_pct"]) / 2)

    trades["short_volume"] = short_vols
    trades["long_volume"] = long_vols
    trades["min_leg_volume"] = min_vols
    trades["short_bid_ask_pct"] = short_bas
    trades["long_bid_ask_pct"] = long_bas
    trades["avg_bid_ask_pct"] = avg_bas

    return trades


# ---------------------------------------------------------------------------
# Phase 3: Scoring and selection
# ---------------------------------------------------------------------------

def score_trade(row: pd.Series, weights: tuple = (0.40, 0.30, 0.30)) -> float:
    """Score a trade for cross-ticker comparison.  Higher = better.

    Components:
      1. Credit/Risk ratio (w=0.40): credit_per_contract / max_risk_per_contract
      2. Volume adequacy (w=0.30): min_leg_volume relative to requested contracts
      3. Bid-ask tightness (w=0.30): tighter spread = higher score

    Returns float in [0, 1].
    """
    w_credit, w_volume, w_bidask = weights

    # Credit/Risk component
    num_contracts = max(row.get("num_contracts", 1), 1)
    credit = abs(row.get("credit", 0))
    max_loss = abs(row.get("max_loss", credit))
    credit_risk = (credit / max_loss) if max_loss > 0 else 0
    # Normalize: typical range 0.05 to 0.50
    cr_score = min(1.0, max(0.0, (credit_risk - 0.05) / 0.45))

    # Volume component
    min_vol = row.get("min_leg_volume", 0)
    vol_ratio = min_vol / num_contracts if num_contracts > 0 else 0
    # Logarithmic scale: 1.0 if ratio >= 5x, 0.0 if <= 0
    if vol_ratio <= 0:
        vol_score = 0.0
    else:
        vol_score = min(1.0, math.log(1 + vol_ratio) / math.log(6))

    # Bid-ask component
    ba_pct = row.get("avg_bid_ask_pct", 0.5)
    # Score: 1.0 if spread < 1%, 0.0 if spread > 25%
    ba_score = max(0.0, min(1.0, (0.25 - ba_pct) / 0.24))

    return w_credit * cr_score + w_volume * vol_score + w_bidask * ba_score


def volume_adjusted_contracts(requested: int, min_leg_volume: int,
                              volume_fill_pct: float = 0.25) -> int:
    """Cap contracts based on available volume.

    Never trade more than volume_fill_pct of the minimum leg's volume.
    This prevents market impact and ensures realistic fills.
    """
    if min_leg_volume <= 0:
        return 0
    max_from_volume = max(1, int(min_leg_volume * volume_fill_pct))
    return min(requested, max_from_volume)


# ---------------------------------------------------------------------------
# Phase 4: Cross-ticker portfolio simulation
# ---------------------------------------------------------------------------

def simulate_cross_ticker_portfolio(
    all_trades: pd.DataFrame,
    weights: tuple = (0.40, 0.30, 0.30),
    volume_fill_pct: float = 0.25,
    apply_volume_cap: bool = True,
    fallback_enabled: bool = True,
) -> tuple:
    """Cross-ticker portfolio simulation with unified budget.

    For each (date, interval, tier), scores all tickers and picks the best.
    Applies volume-adjusted contract sizing and unified budget constraint.

    Returns: (accepted_trades, rejected_trades, selection_log)
    """
    if all_trades.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    trades = all_trades.copy()

    # Create interval key: round entry_time to 10-minute boundary
    trades["interval_key"] = trades["entry_dt"].dt.floor("10min")
    trades["slot_key"] = (
        trades["entry_date"].astype(str) + "_" +
        trades["interval_key"].dt.strftime("%H%M") + "_" +
        trades["dte_tier"]
    )

    # Score all trades
    trades["score"] = trades.apply(lambda r: score_trade(r, weights), axis=1)

    # Sort by time, then priority within each slot
    trades = trades.sort_values(["entry_dt", "priority", "score"],
                                ascending=[True, True, False]).copy()

    # Track results
    trades["portfolio_accepted"] = False
    trades["reject_reason"] = ""
    trades["original_contracts"] = trades["num_contracts"].copy()
    trades["adjusted_contracts"] = trades["num_contracts"].copy()
    trades["volume_cap_applied"] = False
    trades["selection_reason"] = ""

    daily_used = {}
    recent_entries = []
    window_td = pd.Timedelta(minutes=TRADE_WINDOW_MINUTES)
    processed_slots = set()
    selection_log = []

    # Process slot by slot
    for slot_key, slot_group in trades.groupby("slot_key", sort=False):
        # Within each slot, pick the best ticker
        candidates = slot_group.sort_values("score", ascending=False)

        for _, candidate in candidates.iterrows():
            idx = candidate.name
            day = candidate["entry_date"]
            entry_ts = candidate["entry_dt"]
            ticker = candidate["ticker"]
            orig_contracts = int(candidate["num_contracts"])
            min_vol = int(candidate["min_leg_volume"])

            # Volume-adjusted sizing
            if apply_volume_cap:
                adj_contracts = volume_adjusted_contracts(
                    orig_contracts, min_vol, volume_fill_pct)
            else:
                adj_contracts = orig_contracts

            if adj_contracts == 0:
                # No volume — try fallback
                trades.at[idx, "reject_reason"] = "no_volume"
                if fallback_enabled:
                    continue  # Try next ticker in this slot
                else:
                    break

            # Scale risk proportionally
            scale = adj_contracts / orig_contracts if orig_contracts > 0 else 1
            adjusted_risk = abs(candidate.get("max_loss", MAX_RISK_PER_TRADE)) * scale

            if day not in daily_used:
                daily_used[day] = 0.0

            # Budget check
            if daily_used[day] + adjusted_risk > DAILY_BUDGET:
                trades.at[idx, "reject_reason"] = "budget"
                if fallback_enabled:
                    continue
                else:
                    break

            # Rate limit check
            cutoff = entry_ts - window_td
            recent_entries = [t for t in recent_entries if t > cutoff]
            if len(recent_entries) >= MAX_TRADES_PER_WINDOW:
                trades.at[idx, "reject_reason"] = "rate_limit"
                if fallback_enabled:
                    continue
                else:
                    break

            # Accept this trade
            trades.at[idx, "portfolio_accepted"] = True
            trades.at[idx, "adjusted_contracts"] = adj_contracts
            trades.at[idx, "volume_cap_applied"] = (adj_contracts < orig_contracts)
            daily_used[day] += adjusted_risk
            recent_entries.append(entry_ts)

            # Build selection reason
            competing = candidates["ticker"].unique().tolist()
            reason = f"best_score({candidate['score']:.3f})"
            if len(competing) > 1:
                reason += f" vs {','.join(c for c in competing if c != ticker)}"
            if adj_contracts < orig_contracts:
                reason += f" vol_cap:{orig_contracts}->{adj_contracts}"
            trades.at[idx, "selection_reason"] = reason

            # Log selection
            selection_log.append({
                "slot_key": slot_key,
                "date": day,
                "time": str(entry_ts),
                "tier": candidate["dte_tier"],
                "winner": ticker,
                "score": candidate["score"],
                "competing_tickers": ",".join(competing),
                "num_candidates": len(candidates),
                "orig_contracts": orig_contracts,
                "adj_contracts": adj_contracts,
                "min_leg_volume": min_vol,
                "credit_risk_ratio": float(candidate["credit"]) / max(abs(float(candidate.get("max_loss", 1))), 1),
            })

            break  # Only accept one trade per slot

    # Compute adjusted PnL
    accepted = trades[trades["portfolio_accepted"]].copy()
    rejected = trades[~trades["portfolio_accepted"]].copy()

    if len(accepted) > 0:
        scale_factor = accepted["adjusted_contracts"] / accepted["original_contracts"].clip(lower=1)
        accepted["adjusted_pnl"] = accepted["pnl"] * scale_factor
        accepted["adjusted_credit"] = accepted["credit"] * scale_factor
        accepted["adjusted_max_loss"] = accepted["max_loss"].abs() * scale_factor
    else:
        accepted["adjusted_pnl"] = []
        accepted["adjusted_credit"] = []
        accepted["adjusted_max_loss"] = []

    sel_df = pd.DataFrame(selection_log) if selection_log else pd.DataFrame()

    n_budget = int((rejected["reject_reason"] == "budget").sum())
    n_vol = int((rejected["reject_reason"] == "no_volume").sum())
    n_rate = int((rejected["reject_reason"] == "rate_limit").sum())
    n_vcap = int(accepted["volume_cap_applied"].sum()) if len(accepted) > 0 else 0

    print(f"  Cross-ticker simulation: {len(accepted)} accepted, {len(rejected)} rejected")
    print(f"    Rejections: budget={n_budget}, no_volume={n_vol}, rate_limit={n_rate}")
    print(f"    Volume-capped: {n_vcap} trades had contracts reduced")

    return accepted, rejected, sel_df


# ---------------------------------------------------------------------------
# Phase 5: Analysis and metrics
# ---------------------------------------------------------------------------

def print_selection_summary(sel_df: pd.DataFrame, accepted: pd.DataFrame):
    """Print cross-ticker selection analysis."""
    print()
    print("=" * 100)
    print("  CROSS-TICKER SELECTION SUMMARY")
    print("=" * 100)

    if sel_df.empty:
        print("  No selection data.")
        return

    # Overall ticker wins
    wins = sel_df["winner"].value_counts()
    total = len(sel_df)
    print(f"\n  Total slots: {total}")
    print(f"  {'Ticker':<8} {'Selected':>10} {'Pct':>8}")
    print(f"  {'─'*8} {'─'*10} {'─'*8}")
    for ticker, count in wins.items():
        print(f"  {ticker:<8} {count:>10} {count/total*100:>7.1f}%")

    # Per-tier breakdown
    print(f"\n  Selection by tier:")
    print(f"  {'Tier':<20}", end="")
    for t in wins.index:
        print(f" {t:>8}", end="")
    print(f" {'Total':>8}")
    print(f"  {'─'*20}", end="")
    for _ in wins.index:
        print(f" {'─'*8}", end="")
    print(f" {'─'*8}")

    for tier in sorted(sel_df["tier"].unique()):
        tier_data = sel_df[sel_df["tier"] == tier]
        tier_wins = tier_data["winner"].value_counts()
        print(f"  {tier:<20}", end="")
        for t in wins.index:
            count = tier_wins.get(t, 0)
            print(f" {count:>8}", end="")
        print(f" {len(tier_data):>8}")

    # Volume impact
    if len(accepted) > 0:
        print(f"\n  Volume Impact:")
        for ticker in sorted(accepted["ticker"].unique()):
            t_df = accepted[accepted["ticker"] == ticker]
            if len(t_df) == 0:
                continue
            orig_avg = t_df["original_contracts"].mean()
            adj_avg = t_df["adjusted_contracts"].mean()
            pct_reduce = (1 - adj_avg / orig_avg) * 100 if orig_avg > 0 else 0
            print(f"    {ticker}: avg contracts {orig_avg:.0f} -> {adj_avg:.0f} "
                  f"({pct_reduce:+.0f}% reduction), "
                  f"avg min_leg_volume: {t_df['min_leg_volume'].mean():.0f}")

    print("=" * 100)


def print_v3_metrics(accepted: pd.DataFrame):
    """Print portfolio-level metrics using adjusted PnL."""
    print()
    print("=" * 100)
    print("  V3 PORTFOLIO METRICS (Volume-Adjusted)")
    print("=" * 100)

    if len(accepted) == 0:
        print("  No accepted trades.")
        return

    pnl = accepted["adjusted_pnl"].values
    total_trades = len(pnl)
    wins = int((pnl > 0).sum())
    losses = int((pnl <= 0).sum())
    net_pnl = float(pnl.sum())
    total_credit = float(accepted["adjusted_credit"].sum())
    total_risk = float(accepted["adjusted_max_loss"].sum())
    roi = (total_credit / total_risk * 100) if total_risk > 0 else 0

    if total_trades > 1 and pnl.std() > 0:
        sharpe = (pnl.mean() / pnl.std(ddof=1)) * np.sqrt(252)
    else:
        sharpe = 0

    cum_pnl = np.cumsum(pnl)
    peak = np.maximum.accumulate(cum_pnl)
    max_dd = float((peak - cum_pnl).max())

    total_gains = float(pnl[pnl > 0].sum())
    total_losses_val = float(np.abs(pnl[pnl <= 0]).sum())
    pf = total_gains / total_losses_val if total_losses_val > 0 else float("inf")

    print(f"  Trades:          {total_trades:,}")
    print(f"  Win rate:        {wins/total_trades*100:.1f}%")
    print(f"  Net P&L:         ${net_pnl:,.0f}")
    print(f"  Avg P&L/trade:   ${net_pnl/total_trades:,.0f}")
    print(f"  Total credit:    ${total_credit:,.0f}")
    print(f"  Total risk:      ${total_risk:,.0f}")
    print(f"  ROI:             {roi:.1f}%")
    print(f"  Sharpe:          {sharpe:.2f}")
    print(f"  Max drawdown:    ${max_dd:,.0f}")
    print(f"  Profit factor:   {pf:.2f}" if pf < 1000 else f"  Profit factor:   inf")

    # Per-ticker breakdown
    print(f"\n  Per-ticker:")
    print(f"  {'Ticker':<8} {'Trades':>8} {'WR%':>7} {'Net P&L':>14} {'Avg Contracts':>14} {'ROI%':>8}")
    print(f"  {'─'*8} {'─'*8} {'─'*7} {'─'*14} {'─'*14} {'─'*8}")
    for ticker in sorted(accepted["ticker"].unique()):
        t_df = accepted[accepted["ticker"] == ticker]
        t_pnl = t_df["adjusted_pnl"]
        t_wins = (t_pnl > 0).sum()
        t_credit = t_df["adjusted_credit"].sum()
        t_risk = t_df["adjusted_max_loss"].sum()
        t_roi = (t_credit / t_risk * 100) if t_risk > 0 else 0
        print(f"  {ticker:<8} {len(t_df):>8} {t_wins/len(t_df)*100:>6.1f}% "
              f"${t_pnl.sum():>12,.0f} {t_df['adjusted_contracts'].mean():>13.1f} "
              f"{t_roi:>7.1f}%")

    print("=" * 100)


# ---------------------------------------------------------------------------
# Phase 6: Charts
# ---------------------------------------------------------------------------

TICKER_COLORS = {"NDX": "#e74c3c", "SPX": "#3498db", "RUT": "#2ecc71"}


def chart_ticker_selection(sel_df: pd.DataFrame, chart_dir: Path):
    """Bar chart: how often each ticker was selected, by tier."""
    if sel_df.empty:
        return
    chart_dir.mkdir(parents=True, exist_ok=True)

    tiers = sorted(sel_df["tier"].unique())
    tickers_found = sorted(sel_df["winner"].unique())
    x = np.arange(len(tiers))
    width = 0.25

    fig, ax = plt.subplots(figsize=(16, 7))
    for i, ticker in enumerate(tickers_found):
        counts = []
        for tier in tiers:
            c = len(sel_df[(sel_df["tier"] == tier) & (sel_df["winner"] == ticker)])
            counts.append(c)
        ax.bar(x + i * width, counts, width, label=ticker,
               color=TICKER_COLORS.get(ticker, "gray"), alpha=0.85)

    ax.set_xticks(x + width * (len(tickers_found) - 1) / 2)
    ax.set_xticklabels(tiers, rotation=45, ha="right", fontsize=8)
    ax.set_title("Cross-Ticker Selection Frequency by Tier", fontsize=14, fontweight="bold")
    ax.set_ylabel("Times Selected")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    fig.savefig(chart_dir / "ticker_selection_by_tier.png", dpi=150)
    plt.close(fig)


def chart_selection_over_time(sel_df: pd.DataFrame, chart_dir: Path):
    """Monthly stacked bar: ticker selection over time."""
    if sel_df.empty:
        return
    chart_dir.mkdir(parents=True, exist_ok=True)

    sel_df = sel_df.copy()
    sel_df["month"] = pd.to_datetime(sel_df["date"]).dt.to_period("M")
    months = sorted(sel_df["month"].unique())
    tickers_found = sorted(sel_df["winner"].unique())

    fig, ax = plt.subplots(figsize=(16, 6))
    x = np.arange(len(months))
    bottom = np.zeros(len(months))

    for ticker in tickers_found:
        counts = []
        for m in months:
            c = len(sel_df[(sel_df["month"] == m) & (sel_df["winner"] == ticker)])
            counts.append(c)
        counts = np.array(counts)
        ax.bar(x, counts, bottom=bottom, label=ticker,
               color=TICKER_COLORS.get(ticker, "gray"), alpha=0.85)
        bottom += counts

    ax.set_xticks(x)
    ax.set_xticklabels([str(m) for m in months], rotation=45, ha="right", fontsize=8)
    ax.set_title("Ticker Selection Over Time", fontsize=14, fontweight="bold")
    ax.set_ylabel("Trades Selected")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    fig.savefig(chart_dir / "ticker_selection_over_time.png", dpi=150)
    plt.close(fig)


def chart_volume_impact(accepted: pd.DataFrame, chart_dir: Path):
    """Grouped bar: original vs adjusted contracts per ticker."""
    if accepted.empty:
        return
    chart_dir.mkdir(parents=True, exist_ok=True)

    tickers_found = sorted(accepted["ticker"].unique())
    orig = [accepted[accepted["ticker"] == t]["original_contracts"].mean() for t in tickers_found]
    adj = [accepted[accepted["ticker"] == t]["adjusted_contracts"].mean() for t in tickers_found]

    x = np.arange(len(tickers_found))
    width = 0.35
    fig, ax = plt.subplots(figsize=(10, 6))
    bars1 = ax.bar(x - width / 2, orig, width, label="Original (v2)", alpha=0.7, color="#888")
    bars2 = ax.bar(x + width / 2, adj, width, label="Volume-Adjusted (v3)",
                   color=[TICKER_COLORS.get(t, "gray") for t in tickers_found], alpha=0.85)

    for bar, val in zip(bars1, orig):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                f"{val:.0f}", ha="center", va="bottom", fontsize=10)
    for bar, val in zip(bars2, adj):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                f"{val:.0f}", ha="center", va="bottom", fontsize=10)

    ax.set_xticks(x)
    ax.set_xticklabels(tickers_found, fontsize=12)
    ax.set_title("Avg Contracts per Trade: Original vs Volume-Adjusted", fontsize=14, fontweight="bold")
    ax.set_ylabel("Contracts")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    fig.savefig(chart_dir / "volume_impact.png", dpi=150)
    plt.close(fig)


def chart_cumulative_pnl_v3(accepted: pd.DataFrame, chart_dir: Path):
    """Cumulative P&L: per-ticker and combined, using adjusted PnL."""
    if accepted.empty:
        return
    chart_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(16, 8))
    sorted_all = accepted.sort_values("entry_dt")

    for ticker in sorted(accepted["ticker"].unique()):
        t_df = sorted_all[sorted_all["ticker"] == ticker]
        if len(t_df) == 0:
            continue
        cum = t_df["adjusted_pnl"].cumsum()
        ax.plot(t_df["entry_dt"].values, cum.values, label=ticker,
                color=TICKER_COLORS.get(ticker, "gray"), linewidth=1.5, alpha=0.8)

    # Combined
    combined_cum = sorted_all["adjusted_pnl"].cumsum()
    ax.plot(sorted_all["entry_dt"].values, combined_cum.values,
            label="COMBINED", color="black", linewidth=3, alpha=0.9)

    ax.set_title("V3 Cumulative P&L (Volume-Adjusted)", fontsize=14, fontweight="bold")
    ax.set_xlabel("Date")
    ax.set_ylabel("Cumulative P&L ($)")
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color="gray", linewidth=0.5)
    fig.autofmt_xdate()
    plt.tight_layout()
    fig.savefig(chart_dir / "cumulative_pnl_v3.png", dpi=150)
    plt.close(fig)


def chart_score_distribution(all_trades: pd.DataFrame, accepted: pd.DataFrame,
                             chart_dir: Path):
    """Histogram: winning trade scores vs all trade scores."""
    if all_trades.empty:
        return
    chart_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(12, 6))
    bins = np.linspace(0, 1, 40)

    if "score" in all_trades.columns:
        ax.hist(all_trades["score"].values, bins=bins, alpha=0.4, color="#888",
                label=f"All candidates ({len(all_trades)})", edgecolor="white")
    if "score" in accepted.columns and len(accepted) > 0:
        ax.hist(accepted["score"].values, bins=bins, alpha=0.7, color="#3498db",
                label=f"Selected ({len(accepted)})", edgecolor="white")

    ax.set_title("Trade Score Distribution", fontsize=14, fontweight="bold")
    ax.set_xlabel("Composite Score")
    ax.set_ylabel("Frequency")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(chart_dir / "score_distribution.png", dpi=150)
    plt.close(fig)


def chart_drawdown_v3(accepted: pd.DataFrame, chart_dir: Path):
    """Drawdown chart using adjusted PnL."""
    if accepted.empty:
        return
    chart_dir.mkdir(parents=True, exist_ok=True)

    sorted_df = accepted.sort_values("entry_dt")
    cum_pnl = sorted_df["adjusted_pnl"].cumsum().values
    peak = np.maximum.accumulate(cum_pnl)
    dd = peak - cum_pnl

    fig, ax = plt.subplots(figsize=(16, 5))
    ax.fill_between(sorted_df["entry_dt"].values, 0, -dd, color="#e74c3c", alpha=0.5)
    ax.plot(sorted_df["entry_dt"].values, -dd, color="#c0392b", linewidth=1)
    ax.set_title("V3 Portfolio Drawdown (Volume-Adjusted)", fontsize=14, fontweight="bold")
    ax.set_xlabel("Date")
    ax.set_ylabel("Drawdown ($)")
    ax.grid(True, alpha=0.3)
    fig.autofmt_xdate()
    plt.tight_layout()
    fig.savefig(chart_dir / "drawdown_v3.png", dpi=150)
    plt.close(fig)


def chart_monthly_pnl_v3(accepted: pd.DataFrame, chart_dir: Path):
    """Monthly P&L stacked by ticker."""
    if accepted.empty:
        return
    chart_dir.mkdir(parents=True, exist_ok=True)

    accepted = accepted.copy()
    accepted["exit_month"] = pd.to_datetime(accepted["exit_time"], errors="coerce").dt.to_period("M")
    months = sorted(accepted["exit_month"].dropna().unique())
    tickers_found = sorted(accepted["ticker"].unique())

    fig, ax = plt.subplots(figsize=(16, 7))
    x = np.arange(len(months))
    width = 0.8 / max(len(tickers_found), 1)

    for i, ticker in enumerate(tickers_found):
        pnls = []
        for m in months:
            p = accepted[(accepted["exit_month"] == m) & (accepted["ticker"] == ticker)]["adjusted_pnl"].sum()
            pnls.append(float(p))
        ax.bar(x + i * width, pnls, width, label=ticker,
               color=TICKER_COLORS.get(ticker, "gray"), alpha=0.85)

    ax.set_xticks(x + width * (len(tickers_found) - 1) / 2)
    ax.set_xticklabels([str(m) for m in months], rotation=45, ha="right")
    ax.set_title("V3 Monthly P&L by Ticker (Volume-Adjusted)", fontsize=14, fontweight="bold")
    ax.set_ylabel("P&L ($)")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    ax.axhline(y=0, color="black", linewidth=0.5)
    plt.tight_layout()
    fig.savefig(chart_dir / "monthly_pnl_v3.png", dpi=150)
    plt.close(fig)


def generate_all_v3_charts(all_trades: pd.DataFrame, accepted: pd.DataFrame,
                           sel_df: pd.DataFrame, chart_dir: Path):
    """Generate all v3 charts."""
    chart_dir.mkdir(parents=True, exist_ok=True)
    charts = [
        ("ticker_selection_by_tier", lambda: chart_ticker_selection(sel_df, chart_dir)),
        ("ticker_selection_over_time", lambda: chart_selection_over_time(sel_df, chart_dir)),
        ("volume_impact", lambda: chart_volume_impact(accepted, chart_dir)),
        ("cumulative_pnl_v3", lambda: chart_cumulative_pnl_v3(accepted, chart_dir)),
        ("score_distribution", lambda: chart_score_distribution(all_trades, accepted, chart_dir)),
        ("drawdown_v3", lambda: chart_drawdown_v3(accepted, chart_dir)),
        ("monthly_pnl_v3", lambda: chart_monthly_pnl_v3(accepted, chart_dir)),
    ]

    print(f"\nGenerating {len(charts)} charts...")
    for i, (name, func) in enumerate(charts, 1):
        func()
        print(f"  [{i}/{len(charts)}] {name}.png")

    print(f"Charts saved to {chart_dir}/")


# ---------------------------------------------------------------------------
# Phase 7: HTML Report
# ---------------------------------------------------------------------------

def generate_v3_html_report(accepted: pd.DataFrame, rejected: pd.DataFrame,
                            sel_df: pd.DataFrame, all_trades: pd.DataFrame):
    """Generate comprehensive v3 HTML report."""
    now = datetime.now()
    today_str = now.astimezone().strftime("%Y-%m-%d %H:%M:%S %Z")
    today_date = now.strftime("%Y-%m-%d")
    report_name = f"report_tiered_portfolio_v3_{today_date}.html"

    # Metrics using adjusted PnL
    pnl = accepted["adjusted_pnl"].values if len(accepted) > 0 else np.array([])
    total_trades = len(pnl)
    wins = int((pnl > 0).sum()) if total_trades > 0 else 0
    net_pnl = float(pnl.sum()) if total_trades > 0 else 0
    wr = wins / total_trades * 100 if total_trades > 0 else 0
    total_credit = float(accepted["adjusted_credit"].sum()) if total_trades > 0 else 0
    total_risk = float(accepted["adjusted_max_loss"].sum()) if total_trades > 0 else 0
    roi = (total_credit / total_risk * 100) if total_risk > 0 else 0

    sharpe = 0
    if total_trades > 1 and pnl.std() > 0:
        sharpe = (pnl.mean() / pnl.std(ddof=1)) * np.sqrt(252)

    cum_pnl = np.cumsum(pnl) if total_trades > 0 else np.array([0])
    peak = np.maximum.accumulate(cum_pnl)
    max_dd = float((peak - cum_pnl).max()) if len(cum_pnl) > 0 else 0

    total_gains = float(pnl[pnl > 0].sum()) if total_trades > 0 else 0
    total_losses_val = float(np.abs(pnl[pnl <= 0]).sum()) if total_trades > 0 else 0
    pf = total_gains / total_losses_val if total_losses_val > 0 else float("inf")
    pf_str = f"{pf:.2f}" if pf < 1000 else "&infin;"

    # Selection stats
    total_slots = len(sel_df) if not sel_df.empty else 0
    ticker_wins = sel_df["winner"].value_counts().to_dict() if not sel_df.empty else {}

    # Volume cap stats
    n_vcap = int(accepted["volume_cap_applied"].sum()) if total_trades > 0 else 0
    avg_orig = accepted["original_contracts"].mean() if total_trades > 0 else 0
    avg_adj = accepted["adjusted_contracts"].mean() if total_trades > 0 else 0

    # Per-ticker metrics
    ticker_rows = ""
    for ticker in sorted(accepted["ticker"].unique()) if total_trades > 0 else []:
        t_df = accepted[accepted["ticker"] == ticker]
        t_pnl = t_df["adjusted_pnl"]
        t_wins = (t_pnl > 0).sum()
        t_credit = t_df["adjusted_credit"].sum()
        t_risk = t_df["adjusted_max_loss"].sum()
        t_roi = (t_credit / t_risk * 100) if t_risk > 0 else 0
        t_selected = ticker_wins.get(ticker, 0)
        t_pct = (t_selected / total_slots * 100) if total_slots > 0 else 0
        t_avg_vol = t_df["min_leg_volume"].mean()
        t_avg_orig = t_df["original_contracts"].mean()
        t_avg_adj = t_df["adjusted_contracts"].mean()
        t_vol_reduce = ((1 - t_avg_adj / t_avg_orig) * 100) if t_avg_orig > 0 else 0
        color = "positive" if t_pnl.sum() >= 0 else "negative"
        ticker_rows += f"""      <tr>
        <td style="color:{TICKER_COLORS.get(ticker, '#888')}; font-weight:700;">{ticker}</td>
        <td>{len(t_df)}</td><td>{t_wins/len(t_df)*100:.1f}%</td>
        <td class="{color}">${t_pnl.sum():,.0f}</td>
        <td>{t_roi:.1f}%</td>
        <td>{t_selected} ({t_pct:.0f}%)</td>
        <td>{t_avg_vol:.0f}</td>
        <td>{t_avg_orig:.0f} &rarr; {t_avg_adj:.0f}</td>
        <td>{t_vol_reduce:.0f}%</td>
      </tr>\n"""

    # Per-tier table
    tier_rows = ""
    for t in TIERS:
        t_df = accepted[accepted["dte_tier"] == t["label"]] if total_trades > 0 else pd.DataFrame()
        if len(t_df) == 0:
            tier_rows += f"""      <tr>
        <td>{t['label']}</td><td>{t['priority']}</td><td>{t['dte']}</td><td>P{t['percentile']}</td>
        <td>0</td><td>-</td><td>$0</td><td>-</td><td>-</td>
      </tr>\n"""
            continue
        t_pnl = t_df["adjusted_pnl"]
        t_wins = (t_pnl > 0).sum()
        t_dominant = t_df["ticker"].value_counts().index[0]
        t_dominant_pct = t_df["ticker"].value_counts().iloc[0] / len(t_df) * 100
        color = "positive" if t_pnl.sum() >= 0 else "negative"
        tier_rows += f"""      <tr>
        <td>{t['label']}</td><td>{t['priority']}</td><td>{t['dte']}</td><td>P{t['percentile']}</td>
        <td>{len(t_df)}</td><td>{t_wins/len(t_df)*100:.1f}%</td>
        <td class="{color}">${t_pnl.sum():,.0f}</td>
        <td>${t_pnl.mean():,.0f}</td>
        <td style="color:{TICKER_COLORS.get(t_dominant, '#888')}">{t_dominant} ({t_dominant_pct:.0f}%)</td>
      </tr>\n"""

    # Volume liquidity table
    vol_comparison = ""
    for ticker in ["NDX", "SPX", "RUT"]:
        t_df = accepted[accepted["ticker"] == ticker] if total_trades > 0 else pd.DataFrame()
        if len(t_df) == 0:
            vol_comparison += f"""      <tr>
        <td>{ticker}</td><td>0</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td>
      </tr>\n"""
            continue
        vol_comparison += f"""      <tr>
        <td style="color:{TICKER_COLORS.get(ticker, '#888')}; font-weight:700;">{ticker}</td>
        <td>{len(t_df)}</td>
        <td>{t_df['min_leg_volume'].mean():.0f}</td>
        <td>{t_df['min_leg_volume'].median():.0f}</td>
        <td>{t_df['original_contracts'].mean():.0f}</td>
        <td>{t_df['adjusted_contracts'].mean():.0f}</td>
        <td>{t_df['avg_bid_ask_pct'].mean()*100:.1f}%</td>
      </tr>\n"""

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Tiered Portfolio v3 — Cross-Ticker Selection — {today_str}</title>
<style>
  :root {{
    --bg: #0d1117; --card: #161b22; --border: #30363d; --text: #e6edf3;
    --muted: #8b949e; --accent: #58a6ff; --green: #3fb950; --red: #f85149; --orange: #d29922;
  }}
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Helvetica, Arial, sans-serif;
    background: var(--bg); color: var(--text); line-height: 1.6; }}
  .hero {{ background: linear-gradient(135deg, #0d1117 0%, #1a2332 50%, #0d1117 100%);
    border-bottom: 1px solid var(--border); padding: 60px 40px; text-align: center; }}
  .hero h1 {{ font-size: 2.4em; font-weight: 700; background: linear-gradient(90deg, #f39c12, var(--green));
    -webkit-background-clip: text; -webkit-text-fill-color: transparent; }}
  .hero .subtitle {{ font-size: 1.15em; color: var(--muted); max-width: 900px; margin: 10px auto 0; }}
  .hero .date {{ margin-top: 12px; font-size: 0.9em; color: var(--muted); }}
  .container {{ max-width: 1300px; margin: 0 auto; padding: 40px 24px; }}
  .kpi-strip {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(160px, 1fr)); gap: 14px; margin-bottom: 48px; }}
  .kpi {{ background: var(--card); border: 1px solid var(--border); border-radius: 12px; padding: 18px; text-align: center; }}
  .kpi .value {{ font-size: 1.7em; font-weight: 700; color: var(--green); }}
  .kpi .value.warn {{ color: var(--orange); }}
  .kpi .label {{ font-size: 0.82em; color: var(--muted); margin-top: 4px; text-transform: uppercase; letter-spacing: 0.05em; }}
  .section {{ margin-bottom: 56px; }}
  .section h2 {{ font-size: 1.5em; font-weight: 600; border-bottom: 2px solid var(--accent); padding-bottom: 8px; display: inline-block; margin-bottom: 8px; }}
  .section .narrative {{ color: var(--muted); font-size: 1.0em; margin: 12px 0 24px 0; max-width: 950px; line-height: 1.7; }}
  .section .narrative strong {{ color: var(--text); }}
  .chart-container {{ background: var(--card); border: 1px solid var(--border); border-radius: 12px; padding: 16px; margin-bottom: 16px; overflow: hidden; }}
  .chart-container img {{ width: 100%; height: auto; border-radius: 8px; display: block; }}
  table {{ width: 100%; border-collapse: collapse; font-size: 0.88em; background: var(--card); border-radius: 12px; overflow: hidden; border: 1px solid var(--border); }}
  th {{ background: #1c2333; color: var(--accent); padding: 10px 12px; text-align: right; font-weight: 600; text-transform: uppercase; font-size: 0.78em; letter-spacing: 0.04em; border-bottom: 2px solid var(--border); }}
  th:first-child {{ text-align: left; }}
  td {{ padding: 8px 12px; text-align: right; border-bottom: 1px solid var(--border); font-variant-numeric: tabular-nums; }}
  td:first-child {{ text-align: left; font-weight: 600; }}
  tr:last-child td {{ border-bottom: none; }}
  tr:hover td {{ background: rgba(88, 166, 255, 0.04); }}
  .positive {{ color: var(--green); }}
  .negative {{ color: var(--red); }}
  .callout {{ background: rgba(88, 166, 255, 0.06); border: 1px solid rgba(88, 166, 255, 0.2); border-left: 4px solid var(--accent); border-radius: 8px; padding: 16px 20px; margin: 16px 0; font-size: 0.95em; color: var(--muted); }}
  .callout strong {{ color: var(--text); }}
  .footer {{ text-align: center; padding: 32px; color: var(--muted); font-size: 0.85em; border-top: 1px solid var(--border); margin-top: 48px; }}
  @media (max-width: 800px) {{ .kpi-strip {{ grid-template-columns: repeat(2, 1fr); }} .hero {{ padding: 40px 20px; }} .hero h1 {{ font-size: 1.6em; }} }}
</style>
</head>
<body>

<div class="hero">
  <h1>Tiered Portfolio v3 — Cross-Ticker Selection</h1>
  <div class="subtitle">
    Compares NDX, SPX, RUT at each 10-minute interval and picks the best opportunity
    based on credit/risk, volume adequacy, and bid-ask tightness. Unified $500K/day budget.
    Volume-adjusted contract sizing ensures realistic fills.
  </div>
  <div class="date">Generated: {today_str}</div>
</div>

<div class="container">

<!-- Portfolio KPIs -->
<div class="kpi-strip">
  <div class="kpi"><div class="value">{total_trades:,}</div><div class="label">Trades</div></div>
  <div class="kpi"><div class="value">{wr:.1f}%</div><div class="label">Win Rate</div></div>
  <div class="kpi"><div class="value">${net_pnl/1e6:.2f}M</div><div class="label">Net P&amp;L</div></div>
  <div class="kpi"><div class="value">{roi:.1f}%</div><div class="label">ROI</div></div>
  <div class="kpi"><div class="value">{sharpe:.2f}</div><div class="label">Sharpe</div></div>
  <div class="kpi"><div class="value">{pf_str}</div><div class="label">Profit Factor</div></div>
  <div class="kpi"><div class="value warn">${max_dd:,.0f}</div><div class="label">Max Drawdown</div></div>
  <div class="kpi"><div class="value">{total_slots}</div><div class="label">Decision Slots</div></div>
  <div class="kpi"><div class="value">{n_vcap}</div><div class="label">Volume-Capped</div></div>
  <div class="kpi"><div class="value">{avg_orig:.0f}&rarr;{avg_adj:.0f}</div><div class="label">Avg Contracts</div></div>
</div>

<div class="callout">
  <strong>v3 vs v2:</strong> v2 runs each ticker independently with its own $500K/day budget.
  v3 uses a <strong>single unified budget</strong> and picks the best ticker per tier per interval,
  with contract sizes capped at 25% of available volume for realistic fills.
  This dramatically impacts tickers like RUT (median volume: 8) where v2 assumed 132 contracts per trade.
</div>

<!-- Cross-Ticker Selection -->
<div class="section">
  <h2>Cross-Ticker Selection</h2>
  <div class="narrative">
    At each 10-minute interval for each tier, the system scores all available ticker candidates
    and selects the best one. The scoring formula weights credit/risk ratio (40%), volume adequacy (30%),
    and bid-ask tightness (30%).
  </div>
  <div class="chart-container"><img src="charts/ticker_selection_by_tier.png" alt="Ticker Selection by Tier"></div>
  <div class="chart-container"><img src="charts/ticker_selection_over_time.png" alt="Ticker Selection Over Time"></div>
</div>

<!-- Per-Ticker Performance -->
<div class="section">
  <h2>Per-Ticker Performance</h2>
  <table>
    <thead><tr>
      <th>Ticker</th><th>Trades</th><th>WR%</th><th>Net P&amp;L</th><th>ROI%</th>
      <th>Selected</th><th>Avg Vol</th><th>Contracts</th><th>Vol Reduction</th>
    </tr></thead>
    <tbody>
{ticker_rows}
    </tbody>
  </table>
</div>

<!-- Per-Tier Performance -->
<div class="section">
  <h2>Per-Tier Performance</h2>
  <table>
    <thead><tr>
      <th>Tier</th><th>Pri</th><th>DTE</th><th>Pctl</th>
      <th>Trades</th><th>WR%</th><th>Net P&amp;L</th><th>Avg P&amp;L</th><th>Dominant Ticker</th>
    </tr></thead>
    <tbody>
{tier_rows}
    </tbody>
  </table>
</div>

<!-- Volume & Liquidity Analysis -->
<div class="section">
  <h2>Volume &amp; Liquidity Analysis</h2>
  <div class="narrative">
    Volume is the key differentiator between v2 and v3. SPX has the deepest options liquidity
    (median volume ~95 per strike), while RUT has the thinnest (median ~8). Contract sizes
    are capped at 25% of the minimum leg's volume to ensure realistic fills.
  </div>
  <table>
    <thead><tr>
      <th>Ticker</th><th>Trades</th><th>Avg Min Vol</th><th>Median Min Vol</th>
      <th>Orig Contracts</th><th>Adj Contracts</th><th>Avg Bid-Ask</th>
    </tr></thead>
    <tbody>
{vol_comparison}
    </tbody>
  </table>
  <div class="chart-container"><img src="charts/volume_impact.png" alt="Volume Impact"></div>
</div>

<!-- Cumulative P&L -->
<div class="section">
  <h2>Cumulative P&amp;L</h2>
  <div class="chart-container"><img src="charts/cumulative_pnl_v3.png" alt="V3 Cumulative P&L"></div>
</div>

<!-- Monthly P&L -->
<div class="section">
  <h2>Monthly P&amp;L by Ticker</h2>
  <div class="chart-container"><img src="charts/monthly_pnl_v3.png" alt="V3 Monthly P&L"></div>
</div>

<!-- Drawdown -->
<div class="section">
  <h2>Portfolio Drawdown</h2>
  <div class="chart-container"><img src="charts/drawdown_v3.png" alt="V3 Drawdown"></div>
</div>

<!-- Score Distribution -->
<div class="section">
  <h2>Trade Score Distribution</h2>
  <div class="narrative">
    Each candidate trade is scored on a 0&ndash;1 scale. Higher scores indicate
    better credit/risk, higher volume, and tighter bid-ask spreads. Selected trades
    are the winners at each interval.
  </div>
  <div class="chart-container"><img src="charts/score_distribution.png" alt="Score Distribution"></div>
</div>

<!-- Configuration -->
<div class="section">
  <h2>Configuration</h2>
  <table style="max-width: 650px;">
    <thead><tr><th>Parameter</th><th>Value</th></tr></thead>
    <tbody>
      <tr><td>Tickers</td><td>NDX, SPX, RUT</td></tr>
      <tr><td>Daily Budget</td><td>${DAILY_BUDGET:,} (unified across tickers)</td></tr>
      <tr><td>Max Risk/Trade</td><td>${MAX_RISK_PER_TRADE:,}</td></tr>
      <tr><td>Volume Fill %</td><td>25% (max fraction of available volume)</td></tr>
      <tr><td>Scoring Weights</td><td>Credit/Risk: 40%, Volume: 30%, Bid-Ask: 30%</td></tr>
      <tr><td>Fallback</td><td>Enabled (if top ticker has no volume, try next)</td></tr>
      <tr><td>Tiers</td><td>{len(TIERS)}</td></tr>
      <tr><td>Source</td><td>profiles/tiered_v2.yaml (shared with v2)</td></tr>
    </tbody>
  </table>
</div>

</div>
<div class="footer">Tiered Portfolio v3 &mdash; Cross-Ticker Selection &mdash; Generated {today_str}</div>
</body>
</html>"""

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    report_path = OUTPUT_DIR / report_name
    with open(report_path, "w") as f:
        f.write(html)

    index_path = OUTPUT_DIR / "index.html"
    if index_path.exists() or index_path.is_symlink():
        index_path.unlink()
    index_path.symlink_to(report_name)

    print(f"\nHTML report saved to {report_path}")
    print(f"index.html -> {report_name}")
    return report_path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='''
Tiered Portfolio Backtest v3 — Cross-Ticker Selection.

Runs the same 9 DTE tiers as v2 for each ticker, then applies a cross-ticker
selection layer: at each 10-minute interval, compares all tickers and picks the
best opportunity based on credit/risk ratio, volume adequacy, and bid-ask tightness.

Uses a single unified $500K/day budget shared across all tickers.
Volume-adjusted contract sizing ensures realistic fills.
        ''',
        epilog='''
Examples:
  %(prog)s
      Full run: backtests + cross-ticker analysis + report

  %(prog)s --analyze
      Skip backtests, re-analyze existing v2 results

  %(prog)s --no-volume-cap
      Disable volume-based contract capping (v2-style sizing)

  %(prog)s --volume-fill-pct 0.50
      Allow filling up to 50%% of available volume (default: 25%%)

  %(prog)s --weights 0.5,0.25,0.25
      Custom scoring weights: credit_risk, volume, bidask

  %(prog)s --help
      Show this help message
        ''',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--analyze", action="store_true",
                        help="Skip backtests, only run cross-ticker analysis on existing v2 results")
    parser.add_argument("--volume-fill-pct", type=float, default=0.25,
                        help="Max fraction of available volume to consume per trade (default: 0.25)")
    parser.add_argument("--weights", type=str, default="0.40,0.30,0.30",
                        help="Scoring weights: credit_risk,volume,bidask (default: 0.40,0.30,0.30)")
    parser.add_argument("--no-volume-cap", action="store_true",
                        help="Disable volume-based contract capping")
    parser.add_argument("--no-fallback", action="store_true",
                        help="Disable fallback to 2nd-best ticker if top has no volume")

    args = parser.parse_args()

    weights = tuple(float(w) for w in args.weights.split(","))
    if len(weights) != 3 or abs(sum(weights) - 1.0) > 0.01:
        print(f"ERROR: weights must sum to 1.0, got {weights} = {sum(weights):.3f}")
        sys.exit(1)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Phase 1: Run backtests (reuse v2)
    if not args.analyze:
        print("Phase 1: Running backtests (reusing v2 engine)...")
        run_all_backtests(TICKERS)

    # Phase 2: Load and enrich trades from all tickers
    print("\nPhase 2: Loading and enriching trades with volume data...")
    all_enriched = []
    for ticker in TICKERS:
        print(f"\n  Loading {ticker} trades...")
        trades = load_all_trades(ticker)
        if len(trades) == 0:
            print(f"    No trades for {ticker}, skipping.")
            continue
        print(f"    {len(trades)} raw trades loaded")

        print(f"    Enriching with volume/bid-ask data...")
        enriched = enrich_trades_with_liquidity(trades, ticker)
        all_enriched.append(enriched)
        print(f"    Done. Avg min_leg_volume: {enriched['min_leg_volume'].mean():.1f}, "
              f"Avg bid-ask: {enriched['avg_bid_ask_pct'].mean()*100:.1f}%")

    if not all_enriched:
        print("ERROR: No trade data found for any ticker.")
        sys.exit(1)

    all_trades = pd.concat(all_enriched, ignore_index=True)
    print(f"\nTotal enriched trades across all tickers: {len(all_trades):,}")

    # Phase 3: Cross-ticker portfolio simulation
    print("\nPhase 3: Cross-ticker portfolio simulation...")
    accepted, rejected, sel_df = simulate_cross_ticker_portfolio(
        all_trades,
        weights=weights,
        volume_fill_pct=args.volume_fill_pct,
        apply_volume_cap=not args.no_volume_cap,
        fallback_enabled=not args.no_fallback,
    )

    # Phase 4: Analysis
    print_selection_summary(sel_df, accepted)
    print_v3_metrics(accepted)

    # Save CSVs
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    if len(accepted) > 0:
        # Convert period columns to strings for CSV
        save_cols = [c for c in accepted.columns if c not in ("exit_month",)]
        accepted[save_cols].to_csv(OUTPUT_DIR / "portfolio_trades.csv", index=False)
    if not sel_df.empty:
        sel_df.to_csv(OUTPUT_DIR / "cross_ticker_selections.csv", index=False)

    # Phase 5: Charts
    chart_dir = OUTPUT_DIR / "charts"
    generate_all_v3_charts(all_trades, accepted, sel_df, chart_dir)

    # Phase 6: HTML Report
    generate_v3_html_report(accepted, rejected, sel_df, all_trades)

    # Open report
    report_path = list(OUTPUT_DIR.glob("report_*.html"))
    if report_path:
        os.system(f'open "{report_path[-1]}"')

    print("\nDone!")


if __name__ == "__main__":
    main()
