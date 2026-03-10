#!/usr/bin/env python3
"""Tiered Multi-DTE Portfolio Backtest v2 — Multi-Ticker, Priority-Ordered.

Runs 9 DTE tiers for each configured ticker (NDX, SPX, RUT) with per-ticker
spread widths and parameters from profiles/tiered_v2.yaml (single source of
truth).  Results are combined into a single tabbed HTML report.

Usage:
  python run_tiered_backtest_v2.py                   # All tickers
  python run_tiered_backtest_v2.py --ticker NDX      # Single ticker
  python run_tiered_backtest_v2.py --analyze         # Skip backtests
  python run_tiered_backtest_v2.py --ticker SPX --analyze
"""

import argparse
import sys
import os
import yaml
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from multiprocessing import Pool

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "results" / "tiered_portfolio_v2"
CHART_DIR = OUTPUT_DIR / "charts"

# Import shared tier definitions — all derived from profiles/tiered_v2.yaml
sys.path.insert(0, str(BASE_DIR))
from scripts.live_trading.advisor.tier_config import (
    TIERS,
    TICKERS,
    DEFAULT_TICKER,
    TICKER_PARAMS,
    MAX_RISK_PER_TRADE,
    DAILY_BUDGET,
    MAX_TRADES_PER_WINDOW,
    TRADE_WINDOW_MINUTES,
    STRATEGY_DEFAULTS,
    THETA_PARAMS_0DTE,
    THETA_PARAMS_MULTI_DAY,
    EXIT_RULES,
    get_spread_width,
    get_ticker_param,
)


# ---------------------------------------------------------------------------
# Phase 1: Config + Execution
# ---------------------------------------------------------------------------

def _tier_config(tier: dict, ticker: str = "NDX") -> dict:
    """Build config dict for a single tier + ticker.

    Strategy params are read from STRATEGY_DEFAULTS (which comes from
    profiles/tiered_v2.yaml) so that backtest, live advisor, and report
    all use the same source.  Spread widths and other numeric params are
    resolved per-ticker via TICKER_PARAMS.
    """
    dte = tier["dte"]
    sd = STRATEGY_DEFAULTS  # shorthand
    tp = TICKER_PARAMS.get(ticker, {})

    theta = THETA_PARAMS_0DTE if dte == 0 else THETA_PARAMS_MULTI_DAY

    # Resolve per-ticker overrides
    actual_spread_width = get_spread_width(tier, ticker)
    actual_min_credit = tp.get("min_credit", sd.get("min_credit", 0.75))
    actual_max_move_cap = tp.get("max_move_cap", 150)
    start_date = tp.get("backtest_start", "2024-01-02")
    end_date = tp.get("backtest_end", "2026-03-06")

    strategy_params = {
        "dte": dte,
        "percentile": tier["percentile"],
        "lookback": sd.get("lookback", 120),
        "option_types": sd.get("option_types", ["put", "call"]),
        "spread_width": actual_spread_width,
        "interval_minutes": sd.get("interval_minutes", 10),
        "entry_start_utc": tier["entry_start"],
        "entry_end_utc": tier["entry_end"],
        "num_contracts": sd.get("num_contracts", 1),
        "max_loss_estimate": MAX_RISK_PER_TRADE,
        "profit_target_0dte": sd.get("profit_target_0dte", 0.75),
        "profit_target_multiday": sd.get("profit_target_multiday", 0.50),
        "min_roi_per_day": sd.get("min_roi_per_day", 0.025),
        "min_credit": actual_min_credit,
        "min_total_credit": sd.get("min_total_credit", 0),
        "min_credit_per_point": sd.get("min_credit_per_point", 0),
        "max_contracts": sd.get("max_contracts", 0),
        "use_mid": sd.get("use_mid", False),
        "min_volume": sd.get("min_volume", 2),
        "stop_loss_multiplier": sd.get("stop_loss_multiplier", 0),
        "roll_enabled": sd.get("roll_enabled", True),
        "max_rolls": sd.get("max_rolls", 2),
        "roll_check_start_utc": sd.get("roll_check_start_utc", "18:00"),
        "roll_proximity_pct": sd.get("roll_proximity_pct", 0.005),
        "directional_entry": tier["directional"],
        "contract_sizing": sd.get("contract_sizing", "max_budget"),
        "_exit_mode": "theta",
        "_theta_ahead": theta["ahead"],
        "_theta_min_decay": theta["min_decay"],
        "_theta_cut_behind": theta["cut_behind"],
        "_theta_cut_min_time": theta["cut_min_time"],
    }

    if tier["eod_threshold"] is not None:
        strategy_params["pursuit_eod_threshold"] = tier["eod_threshold"]

    return {
        "infra": {
            "ticker": ticker,
            "start_date": start_date,
            "end_date": end_date,
            "lookback_days": 180,
            "num_processes": 1,
            "output_dir": f"results/tiered_portfolio_v2/{ticker}/{tier['label']}",
        },
        "providers": [
            {"name": "csv_equity", "role": "equity", "params": {"csv_dir": "equities_output"}},
            {"name": "csv_options", "role": "options", "params": {
                "csv_dir": "options_csv_output_full",
                "dte_buckets": list(range(0, max(dte + 2, 12))),
            }},
        ],
        "strategy": {
            "name": "percentile_entry_credit_spread",
            "params": strategy_params,
        },
        "constraints": {
            "budget": {
                "max_spend_per_transaction": MAX_RISK_PER_TRADE,
                "daily_budget": DAILY_BUDGET,
            },
            "trading_hours": {
                "entry_start": tier["entry_start"],
                "entry_end": tier["entry_end"],
            },
            "exit_rules": {
                "profit_target_pct": 0.75,
                "mode": "first_triggered",
            },
        },
        "report": {
            "formats": ["csv"],
            "metrics": ["win_rate", "roi", "sharpe", "max_drawdown", "profit_factor"],
        },
    }


def run_single(args):
    """Run a single backtest config in a subprocess."""
    label, config_dict = args

    sys.path.insert(0, str(BASE_DIR))
    os.chdir(str(BASE_DIR))

    import scripts.backtesting.providers.csv_equity_provider   # noqa
    import scripts.backtesting.providers.csv_options_provider   # noqa
    import scripts.backtesting.instruments.credit_spread        # noqa
    import scripts.backtesting.strategies.credit_spread.percentile_entry  # noqa

    from scripts.backtesting.config import BacktestConfig
    from scripts.backtesting.engine import BacktestEngine

    params = config_dict["strategy"]["params"]
    exit_mode = params.pop("_exit_mode", "flat")
    theta_ahead = params.pop("_theta_ahead", 0.25)
    theta_min_decay = params.pop("_theta_min_decay", 0.50)
    theta_cut_behind = params.pop("_theta_cut_behind", 0.40)
    theta_cut_min_time = params.pop("_theta_cut_min_time", 0.60)

    config_path = BASE_DIR / f"_tmp_config_{label}.yaml"
    with open(config_path, "w") as f:
        yaml.dump(config_dict, f)

    try:
        config = BacktestConfig.from_yaml(str(config_path))
        engine = BacktestEngine(config)

        if exit_mode == "theta":
            from scripts.backtesting.constraints.exit_rules.theta_decay_exit import ThetaDecayExit

            theta_exit = ThetaDecayExit(
                take_profit_ahead_pct=theta_ahead,
                min_decay_pct=theta_min_decay,
                cut_behind_pct=theta_cut_behind,
                cut_min_time_pct=theta_cut_min_time,
            )

            def patched_run(dry_run=False):
                engine.provider = engine._build_providers()
                engine.constraints = engine._build_constraints()
                engine.exit_manager = engine._build_exit_manager()
                engine.collector = engine._build_collector()
                engine.strategy = engine._build_strategy()

                if engine.strategy.exit_manager is not None:
                    new_rules = []
                    for rule in engine.strategy.exit_manager._rules:
                        if rule.name == "profit_target":
                            new_rules.append(theta_exit)
                        else:
                            new_rules.append(rule)
                    engine.strategy.exit_manager._rules = new_rules

                trading_dates = engine._resolve_dates(engine.config.infra.ticker)
                engine.strategy.setup()

                all_results = []
                for trading_date in trading_dates:
                    try:
                        day_results = engine._process_day(engine.config.infra.ticker, trading_date)
                        all_results.extend(day_results)
                    except Exception:
                        pass

                engine.strategy.teardown()
                engine.provider.close()
                engine.collector.add_batch(all_results)
                summary = engine.collector.summarize()
                engine._generate_reports(summary)
                return summary

            engine.run = patched_run

        results = engine.run()

        if results and "metrics" in results:
            m = results["metrics"]
            return (label, {
                "total_trades": m.get("total_trades", 0),
                "win_rate": m.get("win_rate", 0),
                "wins": m.get("wins", 0),
                "losses": m.get("losses", 0),
                "net_pnl": m.get("net_pnl", 0),
                "total_credits": m.get("total_credits", 0),
                "total_gains": m.get("total_gains", 0),
                "total_losses": m.get("total_losses", 0),
                "avg_pnl": m.get("avg_pnl", 0),
                "roi": m.get("roi", 0),
                "sharpe": m.get("sharpe", 0),
                "max_drawdown": m.get("max_drawdown", 0),
                "profit_factor": m.get("profit_factor", 0),
            })
        return (label, {"error": "no results"})
    finally:
        if config_path.exists():
            config_path.unlink()


def run_all_backtests(ticker: str = "NDX"):
    """Execute all tier configs in parallel for a single ticker."""
    configs = [(f"{ticker}_{t['label']}", _tier_config(t, ticker)) for t in TIERS]

    tp = TICKER_PARAMS.get(ticker, {})
    start = tp.get("backtest_start", "2024-01-02")
    end = tp.get("backtest_end", "2026-03-06")

    print("=" * 110)
    print(f"Tiered Multi-DTE Portfolio Backtest v2 — {ticker}")
    print("=" * 110)
    print()
    for t in TIERS:
        mode = t["directional"]
        if t["eod_threshold"]:
            mode += f" ({t['eod_threshold']*100:.1f}% threshold)"
        sw = get_spread_width(t, ticker)
        print(f"  P{t['priority']} {t['label']:>16}  DTE={t['dte']:<3} P{t['percentile']:<3} {sw}pt  {mode}")
    print()
    print(f"Risk: ${MAX_RISK_PER_TRADE:,}/trade, ${DAILY_BUDGET:,}/day unified budget")
    print(f"Period:   {start} to {end}")
    print(f"Configs:  {len(configs)}")
    print("=" * 110)
    print()
    print(f"Running {len(configs)} configs for {ticker} with Pool(8)...")
    print()

    with Pool(processes=min(8, len(configs))) as pool:
        results = pool.map(run_single, configs)

    results_dict = dict(results)

    print()
    print(f"{'Pri':>3} {'Tier':>18} {'Trades':>8} {'Wins':>6} {'WR%':>7} {'Net P&L':>12} {'Sharpe':>8} {'ROI%':>8}")
    print("-" * 75)
    for t in TIERS:
        key = f"{ticker}_{t['label']}"
        r = results_dict.get(key, {})
        if "error" in r:
            print(f"  {t['priority']:>1} {t['label']:>16}  ERROR")
        else:
            print(f"  {t['priority']:>1} {t['label']:>16} {r['total_trades']:>8} {r['wins']:>6} {r['win_rate']:>6.1f}% "
                  f"${r['net_pnl']:>10,.0f} {r['sharpe']:>7.2f} {r['roi']:>6.1f}%")
    print()
    return results_dict


# ---------------------------------------------------------------------------
# Phase 2: Post-processing
# ---------------------------------------------------------------------------

def load_all_trades(ticker: str = "NDX") -> pd.DataFrame:
    """Load trades.csv from each tier for a given ticker, add metadata columns."""
    frames = []
    ticker_dir = OUTPUT_DIR / ticker
    for t in TIERS:
        trades_path = ticker_dir / t["label"] / "trades.csv"
        if not trades_path.exists():
            print(f"  WARNING: {trades_path} not found, skipping {t['label']}")
            continue
        df = pd.read_csv(trades_path)
        df["ticker"] = ticker
        df["dte_tier"] = t["label"]
        df["dte_config"] = t["dte"]
        df["percentile"] = t["percentile"]
        df["spread_width"] = get_spread_width(t, ticker)
        df["priority"] = t["priority"]
        df["tier_type"] = "eod" if "eod" in t["label"] else "intraday"
        frames.append(df)

    if not frames:
        print(f"WARNING: No trades.csv files found for {ticker}.")
        return pd.DataFrame()

    trades = pd.concat(frames, ignore_index=True)

    trades["entry_dt"] = pd.to_datetime(trades["entry_time"], errors="coerce")
    trades["exit_dt"] = pd.to_datetime(trades["exit_time"], errors="coerce")
    trades["entry_hour_utc"] = trades["entry_dt"].dt.hour
    trades["entry_date"] = trades["entry_dt"].dt.date
    trades["exit_date"] = trades["exit_dt"].dt.date
    trades["exit_month"] = trades["exit_dt"].dt.to_period("M")
    trades["is_win"] = trades["pnl"] > 0
    trades["days_held"] = (trades["exit_dt"] - trades["entry_dt"]).dt.total_seconds() / 86400

    return trades




def simulate_portfolio(trades: pd.DataFrame) -> pd.DataFrame:
    """Replay trades chronologically with priority ordering and unified daily budget.

    For each day, sort trades by priority (lower = higher priority), then by entry time.
    Accept trades until the daily budget is exhausted. Trades that exceed the budget are
    marked as rejected.  Additionally enforces a rate limit of MAX_TRADES_PER_WINDOW
    trades per TRADE_WINDOW_MINUTES-minute rolling window.
    """
    trades = trades.sort_values(["entry_dt", "priority"]).copy()
    trades["portfolio_accepted"] = False
    trades["reject_reason"] = ""

    daily_used = {}
    recent_entries: list = []   # timestamps of recently accepted trades
    window_td = pd.Timedelta(minutes=TRADE_WINDOW_MINUTES)
    rate_rejected = 0

    for idx, row in trades.iterrows():
        day = row["entry_date"]
        if day not in daily_used:
            daily_used[day] = 0.0

        trade_risk = abs(row.get("max_loss", MAX_RISK_PER_TRADE))
        if trade_risk <= 0:
            trade_risk = MAX_RISK_PER_TRADE

        entry_ts = row["entry_dt"]

        # Budget check
        if daily_used[day] + trade_risk > DAILY_BUDGET:
            trades.at[idx, "reject_reason"] = "budget"
            continue

        # Rate-limit check: count accepted trades in the trailing window
        cutoff = entry_ts - window_td
        recent_entries = [t for t in recent_entries if t > cutoff]
        if len(recent_entries) >= MAX_TRADES_PER_WINDOW:
            trades.at[idx, "reject_reason"] = "rate_limit"
            rate_rejected += 1
            continue

        trades.at[idx, "portfolio_accepted"] = True
        daily_used[day] += trade_risk
        recent_entries.append(entry_ts)

    accepted = trades[trades["portfolio_accepted"]].copy()
    rejected = trades[~trades["portfolio_accepted"]].copy()
    budget_rejected = int((trades["reject_reason"] == "budget").sum())

    print(f"  Portfolio simulation: {len(accepted)} accepted, {len(rejected)} rejected "
          f"(budget: {budget_rejected}, rate-limit: {rate_rejected})")

    return accepted, rejected


def compute_metrics(df: pd.DataFrame) -> dict:
    """Compute summary metrics for a group of trades.

    ROI = total_credits / total_max_risk (credit collected vs risk taken).
    ROI/Day = ROI / (DTE + 1), normalized for the capital allocation window.
    """
    if len(df) == 0:
        return {
            "trades": 0, "wins": 0, "losses": 0, "win_rate": 0,
            "net_pnl": 0, "avg_pnl": 0, "total_credits": 0,
            "total_gains": 0, "total_losses": 0,
            "roi": 0, "roi_per_day": 0, "avg_dte_window": 0,
            "sharpe": 0, "max_drawdown": 0, "profit_factor": 0,
        }

    pnl = df["pnl"].values
    wins = int((pnl > 0).sum())
    losses = int((pnl <= 0).sum())
    total_trades = len(pnl)
    total_gains = float(pnl[pnl > 0].sum())
    total_losses = float(np.abs(pnl[pnl <= 0]).sum())
    net_pnl = total_gains - total_losses

    if "credit" in df.columns:
        total_credits = float(df["credit"].sum())
    else:
        total_credits = 0

    if "max_loss" in df.columns:
        total_risk = float(df["max_loss"].abs().sum())
    else:
        total_risk = total_credits if total_credits > 0 else 1

    # ROI = credit / max_risk (premium yield as % of risk)
    roi = (total_credits / total_risk * 100) if total_risk > 0 else 0

    # Average DTE window = dte_config + 1 (entry day counts)
    # e.g. DTE 0 → 1 day, DTE 2 → 3 days, DTE 5 → 6 days
    if "dte_config" in df.columns:
        avg_dte_window = float(df["dte_config"].mean()) + 1
    else:
        avg_dte_window = 1

    # ROI/Day = ROI normalized by the DTE window (not actual hold time)
    roi_per_day = roi / avg_dte_window

    if total_trades > 1 and pnl.std() > 0:
        sharpe = (pnl.mean() / pnl.std(ddof=1)) * np.sqrt(252)
    else:
        sharpe = 0

    cum_pnl = np.cumsum(pnl)
    peak = np.maximum.accumulate(cum_pnl)
    drawdown = peak - cum_pnl
    max_dd = float(drawdown.max()) if len(drawdown) > 0 else 0

    pf = (total_gains / total_losses) if total_losses > 0 else float("inf")

    return {
        "trades": total_trades,
        "wins": wins,
        "losses": losses,
        "win_rate": (wins / total_trades * 100) if total_trades > 0 else 0,
        "net_pnl": net_pnl,
        "avg_pnl": net_pnl / total_trades if total_trades > 0 else 0,
        "total_credits": total_credits,
        "total_gains": total_gains,
        "total_losses": total_losses,
        "roi": roi,
        "roi_per_day": roi_per_day,
        "avg_dte_window": avg_dte_window,
        "sharpe": sharpe,
        "max_drawdown": max_dd,
        "profit_factor": pf,
    }


# ---------------------------------------------------------------------------
# Phase 2: Tables
# ---------------------------------------------------------------------------

def print_tier_summary(trades: pd.DataFrame, label: str = "ALL TRADES"):
    """Print per-tier metrics table with ROI and ROI/Day."""
    print()
    print("=" * 170)
    print(f"  PER-TIER METRICS ({label})")
    print("=" * 170)
    header = (f"  {'Pri':>3} {'Tier':>18} {'DTE':>4} {'Pctl':>5} {'Width':>5} {'Trades':>7} {'Wins':>5} {'Losses':>6} {'WR%':>7} "
              f"{'Net P&L':>12} {'Avg P&L':>10} {'ROI%':>7} {'ROI/D%':>7} {'Window':>7} {'Sharpe':>7} {'MaxDD':>10} {'PF':>8}")
    print(header)
    print("  " + "-" * 166)

    rows = []
    for t in TIERS:
        df_tier = trades[trades["dte_tier"] == t["label"]]
        m = compute_metrics(df_tier)
        pf_str = f"{m['profit_factor']:.2f}" if m['profit_factor'] < 1000 else "inf"
        print(f"  {t['priority']:>3} {t['label']:>18} {t['dte']:>4} P{t['percentile']:>3} {t['spread_width']:>3}pt {m['trades']:>7} {m['wins']:>5} {m['losses']:>6} {m['win_rate']:>6.1f}% "
              f"${m['net_pnl']:>10,.0f} ${m['avg_pnl']:>8,.0f} {m['roi']:>6.1f}% {m['roi_per_day']:>6.1f}% {m['avg_dte_window']:>6.2f}d {m['sharpe']:>6.2f} ${m['max_drawdown']:>8,.0f} {pf_str:>8}")
        m["tier"] = t["label"]
        m["dte"] = t["dte"]
        m["percentile"] = t["percentile"]
        m["priority"] = t["priority"]
        rows.append(m)

    print("  " + "-" * 166)
    m = compute_metrics(trades)
    pf_str = f"{m['profit_factor']:.2f}" if m['profit_factor'] < 1000 else "inf"
    print(f"  {'':>3} {'COMBINED':>18} {'all':>4} {'all':>5} {'mix':>5} {m['trades']:>7} {m['wins']:>5} {m['losses']:>6} {m['win_rate']:>6.1f}% "
          f"${m['net_pnl']:>10,.0f} ${m['avg_pnl']:>8,.0f} {m['roi']:>6.1f}% {m['roi_per_day']:>6.1f}% {m['avg_dte_window']:>6.2f}d {m['sharpe']:>6.2f} ${m['max_drawdown']:>8,.0f} {pf_str:>8}")
    print("=" * 170)
    return pd.DataFrame(rows)


def print_monthly_breakdown(trades: pd.DataFrame):
    """Print monthly P&L breakdown by tier type."""
    print()
    print("=" * 100)
    print("  MONTHLY P&L BREAKDOWN")
    print("=" * 100)

    months = sorted(trades["exit_month"].dropna().unique())
    intraday_tiers = [t for t in TIERS if "eod" not in t["label"]]
    eod_tiers = [t for t in TIERS if "eod" in t["label"]]

    hdr = f"  {'Month':>10} {'Intraday':>14} {'EOD':>14} {'Combined':>14} {'Trades':>8} {'WR%':>7}"
    print(hdr)
    print("  " + "-" * 72)

    rows = []
    for month in months:
        df_month = trades[trades["exit_month"] == month]
        intra_pnl = float(df_month[df_month["tier_type"] == "intraday"]["pnl"].sum())
        eod_pnl = float(df_month[df_month["tier_type"] == "eod"]["pnl"].sum())
        combined = intra_pnl + eod_pnl
        n_trades = len(df_month)
        wins = (df_month["pnl"] > 0).sum()
        wr = (wins / n_trades * 100) if n_trades > 0 else 0

        print(f"  {str(month):>10} ${intra_pnl:>12,.0f} ${eod_pnl:>12,.0f} ${combined:>12,.0f} {n_trades:>8} {wr:>6.1f}%")
        rows.append({"month": str(month), "intraday": intra_pnl, "eod": eod_pnl, "combined": combined, "trades": n_trades, "win_rate": wr})

    print("=" * 100)
    return pd.DataFrame(rows)


def print_hour_summary(trades: pd.DataFrame):
    """Print per-entry-hour metrics."""
    print()
    print("=" * 100)
    print("  SUMMARY BY ENTRY HOUR (UTC / ET)")
    print("=" * 100)
    hours = sorted(trades["entry_hour_utc"].dropna().unique().astype(int))
    header = f"  {'UTC':>6} {'ET':>6} {'Trades':>7} {'Wins':>5} {'WR%':>7} {'Net P&L':>12} {'Avg P&L':>10} {'Sharpe':>7}"
    print(header)
    print("  " + "-" * 65)

    rows = []
    for h in hours:
        df_h = trades[trades["entry_hour_utc"] == h]
        m = compute_metrics(df_h)
        et = h - 5 if h >= 5 else h + 19  # rough UTC->ET
        print(f"  {h:>4}:xx {et:>4}:xx {m['trades']:>7} {m['wins']:>5} {m['win_rate']:>6.1f}% "
              f"${m['net_pnl']:>10,.0f} ${m['avg_pnl']:>8,.0f} {m['sharpe']:>6.2f}")
        m["hour_utc"] = h
        rows.append(m)

    print("=" * 100)
    return pd.DataFrame(rows)


def print_loss_analysis(trades: pd.DataFrame):
    """Analyze losing trades: dates, causes, patterns."""
    losers = trades[trades["pnl"] < 0].copy()

    print()
    print("=" * 120)
    print(f"  LOSS ANALYSIS ({len(losers)} losing trades)")
    print("=" * 120)

    if len(losers) == 0:
        print("  No losing trades!")
        print("=" * 120)
        return pd.DataFrame()

    # Group by date
    losers["loss_date"] = losers["entry_dt"].dt.date
    daily_losses = losers.groupby("loss_date").agg(
        count=("pnl", "size"),
        total_loss=("pnl", "sum"),
        tiers=("dte_tier", lambda x: ", ".join(sorted(x.unique()))),
        option_types=("option_type", lambda x: ", ".join(sorted(x.unique()))),
    ).sort_values("total_loss")

    print()
    print("  LOSS DAYS (sorted by severity):")
    print(f"  {'Date':>12} {'Trades':>7} {'Total Loss':>12} {'Tiers':>40} {'Direction':>12}")
    print("  " + "-" * 90)

    for dt, row in daily_losses.iterrows():
        print(f"  {str(dt):>12} {row['count']:>7} ${row['total_loss']:>10,.0f} {row['tiers']:>40} {row['option_types']:>12}")

    # By tier
    print()
    print("  LOSSES BY TIER:")
    print(f"  {'Tier':>18} {'Losses':>7} {'Total Loss':>12} {'Avg Loss':>10}")
    print("  " + "-" * 50)
    tier_losses = losers.groupby("dte_tier")["pnl"].agg(["count", "sum", "mean"])
    for tier, row in tier_losses.iterrows():
        print(f"  {tier:>18} {int(row['count']):>7} ${row['sum']:>10,.0f} ${row['mean']:>8,.0f}")

    # By exit reason
    if "exit_reason" in losers.columns:
        print()
        print("  LOSSES BY EXIT REASON:")
        for reason, group in losers.groupby("exit_reason"):
            short_reason = str(reason)[:60]
            print(f"  {short_reason:>60}: {len(group)} trades, ${group['pnl'].sum():>10,.0f}")

    print("=" * 120)
    return daily_losses


def print_daily_stats(trades: pd.DataFrame):
    """Print daily P&L statistics."""
    print()
    print("=" * 80)
    print("  DAILY P&L STATISTICS (portfolio)")
    print("=" * 80)

    daily = trades.groupby("exit_date")["pnl"].sum()
    print(f"  Trading days:       {len(daily)}")
    print(f"  Mean daily P&L:     ${daily.mean():>10,.0f}")
    print(f"  Median daily P&L:   ${daily.median():>10,.0f}")
    print(f"  Std daily P&L:      ${daily.std():>10,.0f}")
    print(f"  Best day:           ${daily.max():>10,.0f}")
    print(f"  Worst day:          ${daily.min():>10,.0f}")
    pos = (daily > 0).sum()
    neg = (daily < 0).sum()
    print(f"  Positive days:      {pos} ({pos/len(daily)*100:.1f}%)")
    print(f"  Negative days:      {neg} ({neg/len(daily)*100:.1f}%)")
    print(f"  Zero days:          {(daily == 0).sum()}")

    # Weekly P&L
    daily.index = pd.to_datetime(daily.index)
    weekly = daily.resample("W").sum()
    weeks_above_50k = (weekly >= 50000).sum()
    print(f"\n  Weeks >= $50K P&L:  {weeks_above_50k} of {len(weekly)} ({weeks_above_50k/len(weekly)*100:.1f}%)")
    print(f"  Avg weekly P&L:     ${weekly.mean():>10,.0f}")
    print(f"  Min weekly P&L:     ${weekly.min():>10,.0f}")
    print("=" * 80)

    return daily


def save_csvs(all_trades: pd.DataFrame, portfolio_trades: pd.DataFrame,
              tier_summary_all: pd.DataFrame, tier_summary_port: pd.DataFrame,
              monthly_df: pd.DataFrame, hour_summary: pd.DataFrame):
    """Save CSV outputs."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    def safe_save(df, path):
        out = df.copy()
        for col in out.columns:
            if hasattr(out[col], 'dt') and hasattr(out[col].dt, 'to_period'):
                pass
            if out[col].dtype == 'period[M]':
                out[col] = out[col].astype(str)
        out.to_csv(path, index=False)

    safe_save(all_trades, OUTPUT_DIR / "all_trades_raw.csv")
    safe_save(portfolio_trades, OUTPUT_DIR / "portfolio_trades.csv")
    tier_summary_all.to_csv(OUTPUT_DIR / "summary_by_tier_raw.csv", index=False)
    tier_summary_port.to_csv(OUTPUT_DIR / "summary_by_tier_portfolio.csv", index=False)
    monthly_df.to_csv(OUTPUT_DIR / "monthly_breakdown.csv", index=False)
    hour_summary.to_csv(OUTPUT_DIR / "summary_by_hour.csv", index=False)

    print(f"\nCSV files saved to {OUTPUT_DIR}/")


# ---------------------------------------------------------------------------
# Phase 3: Charts
# ---------------------------------------------------------------------------

TIER_COLORS = {
    "dte0_p95":     "#e74c3c",
    "dte1_p90":     "#3498db",
    "dte2_p90":     "#2ecc71",
    "dte3_p80":     "#9b59b6",
    "dte5_p75":     "#f39c12",
    "dte10_p90":    "#1abc9c",
    "dte1_p90_eod": "#e67e22",
    "dte2_p90_eod": "#16a085",
    "dte3_p90_eod": "#8e44ad",
}


def chart_cumulative_pnl(trades: pd.DataFrame, suffix="", ticker="NDX", chart_dir=None):
    """Cumulative P&L per tier + combined."""
    chart_dir = chart_dir or CHART_DIR
    fig, ax = plt.subplots(figsize=(16, 8))

    for t in TIERS:
        df_tier = trades[trades["dte_tier"] == t["label"]].sort_values("exit_dt")
        if len(df_tier) == 0:
            continue
        cum_pnl = df_tier["pnl"].cumsum()
        lw = 1.0 if "eod" in t["label"] else 1.5
        ls = "--" if "eod" in t["label"] else "-"
        ax.plot(df_tier["exit_dt"].values, cum_pnl.values,
                label=t["label"], color=TIER_COLORS.get(t["label"], "gray"),
                linewidth=lw, linestyle=ls, alpha=0.8)

    combined = trades.sort_values("exit_dt")
    ax.plot(combined["exit_dt"].values, combined["pnl"].cumsum().values,
            label="COMBINED", color="black", linewidth=3, alpha=0.9)

    ax.set_title(f"Cumulative P&L: {ticker} Tiered Portfolio{suffix}", fontsize=14, fontweight="bold")
    ax.set_xlabel("Date")
    ax.set_ylabel("Cumulative P&L ($)")
    ax.legend(loc="upper left", fontsize=8, ncol=2)
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color="gray", linewidth=0.5)
    fig.autofmt_xdate()
    plt.tight_layout()
    fname = f"cumulative_pnl{'_portfolio' if suffix else ''}.png"
    fig.savefig(chart_dir / fname, dpi=150)
    plt.close(fig)
    return fname


def chart_monthly_pnl(trades: pd.DataFrame, ticker="NDX", chart_dir=None):
    """Stacked monthly bars: intraday vs EOD."""
    chart_dir = chart_dir or CHART_DIR
    months = sorted(trades["exit_month"].dropna().unique())
    month_strs = [str(m) for m in months]

    fig, ax = plt.subplots(figsize=(16, 7))
    x = np.arange(len(months))
    width = 0.35

    intra_pnls = []
    eod_pnls = []
    for month in months:
        df_m = trades[trades["exit_month"] == month]
        intra_pnls.append(float(df_m[df_m["tier_type"] == "intraday"]["pnl"].sum()))
        eod_pnls.append(float(df_m[df_m["tier_type"] == "eod"]["pnl"].sum()))

    ax.bar(x - width/2, intra_pnls, width, label="Intraday", color="#3498db", alpha=0.85)
    ax.bar(x + width/2, eod_pnls, width, label="EOD", color="#e67e22", alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels(month_strs, rotation=45, ha="right")
    ax.set_title(f"{ticker} Monthly P&L: Intraday vs EOD", fontsize=14, fontweight="bold")
    ax.set_ylabel("P&L ($)")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    ax.axhline(y=0, color="black", linewidth=0.5)
    plt.tight_layout()
    fig.savefig(chart_dir / "monthly_pnl_intraday_vs_eod.png", dpi=150)
    plt.close(fig)


def chart_drawdown(trades: pd.DataFrame, ticker="NDX", chart_dir=None):
    """Portfolio drawdown over time."""
    chart_dir = chart_dir or CHART_DIR
    combined = trades.sort_values("exit_dt")
    cum_pnl = combined["pnl"].cumsum().values
    peak = np.maximum.accumulate(cum_pnl)
    dd = peak - cum_pnl

    fig, ax = plt.subplots(figsize=(16, 5))
    ax.fill_between(combined["exit_dt"].values, 0, -dd, color="#e74c3c", alpha=0.5)
    ax.plot(combined["exit_dt"].values, -dd, color="#c0392b", linewidth=1)
    ax.set_title(f"{ticker} Portfolio Drawdown", fontsize=14, fontweight="bold")
    ax.set_xlabel("Date")
    ax.set_ylabel("Drawdown ($)")
    ax.grid(True, alpha=0.3)
    fig.autofmt_xdate()
    plt.tight_layout()
    fig.savefig(chart_dir / "portfolio_drawdown.png", dpi=150)
    plt.close(fig)


def chart_tier_summary(trades: pd.DataFrame, ticker="NDX", chart_dir=None):
    """Per-tier bar charts: P&L, trade count, win rate."""
    chart_dir = chart_dir or CHART_DIR
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    labels = [t["label"] for t in TIERS]
    colors = [TIER_COLORS.get(l, "gray") for l in labels]

    metrics_list = []
    for t in TIERS:
        df_tier = trades[trades["dte_tier"] == t["label"]]
        metrics_list.append(compute_metrics(df_tier))

    for ax_idx, (title, key, fmt) in enumerate([
        ("Net P&L", "net_pnl", "${:,.0f}"),
        ("Trade Count", "trades", "{}"),
        ("Win Rate", "win_rate", "{:.1f}%"),
    ]):
        ax = axes[ax_idx]
        vals = [m[key] for m in metrics_list]
        bars = ax.bar(range(len(labels)), vals, color=colors, alpha=0.85)
        ax.set_title(title, fontweight="bold")
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=7)
        ax.grid(True, axis="y", alpha=0.3)
        if key == "net_pnl":
            ax.axhline(y=0, color="black", linewidth=0.5)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                    fmt.format(val), ha="center", va="bottom", fontsize=7)

    plt.suptitle(f"{ticker} Tier Summary", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    fig.savefig(chart_dir / "tier_summary.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def chart_daily_histogram(trades: pd.DataFrame, ticker="NDX", chart_dir=None):
    """Daily P&L distribution histogram."""
    chart_dir = chart_dir or CHART_DIR
    daily = trades.groupby("exit_date")["pnl"].sum()

    fig, ax = plt.subplots(figsize=(14, 6))
    ax.hist(daily.values, bins=50, color="#3498db", alpha=0.7, edgecolor="white")
    ax.axvline(x=0, color="black", linewidth=1)
    ax.axvline(x=daily.mean(), color="red", linewidth=1.5, linestyle="--",
               label=f"Mean: ${daily.mean():,.0f}")
    ax.axvline(x=daily.median(), color="green", linewidth=1.5, linestyle="--",
               label=f"Median: ${daily.median():,.0f}")
    ax.set_title(f"{ticker} Daily P&L Distribution (Portfolio)", fontsize=14, fontweight="bold")
    ax.set_xlabel("Daily P&L ($)")
    ax.set_ylabel("Frequency")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(chart_dir / "daily_pnl_histogram.png", dpi=150)
    plt.close(fig)


def chart_weekly_pnl(trades: pd.DataFrame, ticker="NDX", chart_dir=None):
    """Weekly P&L bar chart with $50K target line."""
    chart_dir = chart_dir or CHART_DIR
    daily = trades.groupby("exit_date")["pnl"].sum()
    daily.index = pd.to_datetime(daily.index)
    weekly = daily.resample("W").sum()

    fig, ax = plt.subplots(figsize=(16, 6))
    colors = ["#2ecc71" if v >= 50000 else "#e74c3c" if v < 0 else "#f39c12" for v in weekly.values]
    ax.bar(range(len(weekly)), weekly.values, color=colors, alpha=0.8)
    ax.axhline(y=50000, color="blue", linewidth=1.5, linestyle="--", label="$50K target")
    ax.axhline(y=0, color="black", linewidth=0.5)

    ax.set_title(f"{ticker} Weekly P&L (green = >= $50K, orange = $0-$50K, red = negative)", fontsize=13, fontweight="bold")
    ax.set_xlabel("Week")
    ax.set_ylabel("Weekly P&L ($)")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)

    tick_positions = list(range(0, len(weekly), 4))
    tick_labels = [str(weekly.index[i].date()) for i in tick_positions if i < len(weekly)]
    ax.set_xticks(tick_positions[:len(tick_labels)])
    ax.set_xticklabels(tick_labels, rotation=45, ha="right", fontsize=8)

    plt.tight_layout()
    fig.savefig(chart_dir / "weekly_pnl.png", dpi=150)
    plt.close(fig)


def generate_all_charts(all_trades: pd.DataFrame, portfolio_trades: pd.DataFrame,
                        ticker: str = "NDX", chart_dir: Path = None):
    """Generate all charts for a single ticker."""
    chart_dir = chart_dir or CHART_DIR
    chart_dir.mkdir(parents=True, exist_ok=True)

    charts = [
        ("cumulative_pnl (raw)", lambda: chart_cumulative_pnl(all_trades, ticker=ticker, chart_dir=chart_dir)),
        ("cumulative_pnl_portfolio", lambda: chart_cumulative_pnl(portfolio_trades, " (Budget-Constrained)", ticker=ticker, chart_dir=chart_dir)),
        ("monthly_pnl", lambda: chart_monthly_pnl(portfolio_trades, ticker=ticker, chart_dir=chart_dir)),
        ("portfolio_drawdown", lambda: chart_drawdown(portfolio_trades, ticker=ticker, chart_dir=chart_dir)),
        ("tier_summary", lambda: chart_tier_summary(portfolio_trades, ticker=ticker, chart_dir=chart_dir)),
        ("daily_pnl_histogram", lambda: chart_daily_histogram(portfolio_trades, ticker=ticker, chart_dir=chart_dir)),
        ("weekly_pnl", lambda: chart_weekly_pnl(portfolio_trades, ticker=ticker, chart_dir=chart_dir)),
    ]

    print(f"\nGenerating {len(charts)} charts for {ticker}...")
    for i, (name, func) in enumerate(charts, 1):
        func()
        print(f"  [{i}/{len(charts)}] {name}.png")

    print(f"\nAll charts saved to {chart_dir}/")


# ---------------------------------------------------------------------------
# Phase 4: HTML Report
# ---------------------------------------------------------------------------

def _build_roll_analysis_html(portfolio_trades: pd.DataFrame) -> str:
    """Build HTML for the roll analysis section."""
    rolled = portfolio_trades[portfolio_trades["roll_count"] > 0]
    original_legs = portfolio_trades[portfolio_trades["exit_reason"] == "roll_trigger_itm"]

    n_rolled = len(rolled)
    n_original = len(original_legs)
    rolled_pnl = rolled["pnl"].sum() if n_rolled else 0
    original_pnl = original_legs["pnl"].sum() if n_original else 0
    net = rolled_pnl + original_pnl
    chains_profitable = int((rolled["total_chain_pnl"] > 0).sum()) if n_rolled else 0

    # Per-date breakdown
    loss_dates = set(original_legs["trading_date"].unique())
    roll_dates = set(rolled["trading_date"].unique())
    both_dates = sorted(loss_dates & roll_dates)
    only_loss_dates = sorted(loss_dates - roll_dates)

    date_rows = ""
    for d in both_dates:
        d_losses = original_legs[original_legs["trading_date"] == d]
        d_wins = rolled[rolled["trading_date"] == d]
        d_net = d_losses["pnl"].sum() + d_wins["pnl"].sum()
        color = "var(--green)" if d_net >= 0 else "var(--red)"
        date_rows += f"""<tr>
          <td>{d}</td><td>{len(d_losses)}</td><td style="color:var(--red)">${d_losses['pnl'].sum():,.0f}</td>
          <td>{len(d_wins)}</td><td style="color:var(--green)">${d_wins['pnl'].sum():,.0f}</td>
          <td style="color:{color}">${d_net:,.0f}</td></tr>\n"""
    for d in only_loss_dates:
        d_losses = original_legs[original_legs["trading_date"] == d]
        date_rows += f"""<tr>
          <td>{d}</td><td>{len(d_losses)}</td><td style="color:var(--red)">${d_losses['pnl'].sum():,.0f}</td>
          <td>0</td><td>-</td>
          <td style="color:var(--red)">${d_losses['pnl'].sum():,.0f}</td></tr>\n"""

    return f"""
  <div style="display:grid;grid-template-columns:repeat(3,1fr);gap:20px;margin:20px 0;">
    <div class="kpi"><div class="kpi-value" style="color:var(--green)">${rolled_pnl:,.0f}</div><div class="kpi-label">Roll Wins ({n_rolled} trades)</div></div>
    <div class="kpi"><div class="kpi-value" style="color:var(--red)">${original_pnl:,.0f}</div><div class="kpi-label">Original Losses ({n_original} trades)</div></div>
    <div class="kpi"><div class="kpi-value" style="color:{'var(--green)' if net >= 0 else 'var(--red)'}">${net:,.0f}</div><div class="kpi-label">Net Rolling Impact</div></div>
  </div>
  <p>{chains_profitable} of {n_rolled} roll chains were net profitable. Rolling contributed <strong>${net:,.0f}</strong> to the portfolio.</p>
  <table>
    <thead><tr><th>Date</th><th>Losses</th><th>Loss P&amp;L</th><th>Rolls</th><th>Roll P&amp;L</th><th>Net</th></tr></thead>
    <tbody>{date_rows}</tbody>
  </table>"""


def _utc_to_display(utc_str: str) -> str:
    """Convert a UTC time string like '18:00' to 'HH:MM AM/PM ET / HH:MM AM/PM PST'."""
    parts = utc_str.split(":")
    utc_h, utc_m = int(parts[0]), int(parts[1])
    # ET = UTC - 4 (EDT), PST = UTC - 7 (PDT)
    et_h = (utc_h - 4) % 24
    pst_h = (utc_h - 7) % 24
    def _fmt(h, m):
        suffix = "AM" if h < 12 else "PM"
        display_h = h if 1 <= h <= 12 else (h - 12 if h > 12 else 12)
        return f"{display_h}:{m:02d} {suffix}"
    return f"{_fmt(et_h, utc_m)} ET / {_fmt(pst_h, utc_m)} PST"


def _build_ticker_tab_html(ticker: str, all_trades: pd.DataFrame,
                           portfolio_trades: pd.DataFrame, monthly_df: pd.DataFrame,
                           loss_df: pd.DataFrame, daily_pnl: pd.Series) -> str:
    """Build the HTML content for a single ticker tab."""
    m = compute_metrics(portfolio_trades)
    m_raw = compute_metrics(all_trades)

    # Weekly stats
    weekly = daily_pnl.groupby(pd.Grouper(freq="W")).sum()
    weeks_above_50k = int((weekly >= 50000).sum())
    total_weeks = max(len(weekly), 1)

    tp = TICKER_PARAMS.get(ticker, {})
    actual_min_credit = tp.get("min_credit", STRATEGY_DEFAULTS.get("min_credit", 0.75))
    actual_max_move_cap = tp.get("max_move_cap", 150)
    start_date = tp.get("backtest_start", "2024-01-02")
    end_date = tp.get("backtest_end", "2026-03-06")

    roll_check_utc = STRATEGY_DEFAULTS["roll_check_start_utc"]
    roll_time_display = _utc_to_display(roll_check_utc)
    roll_proximity = STRATEGY_DEFAULTS["roll_proximity_pct"]
    roll_proximity_pct_display = f"{roll_proximity * 100:.1f}%"
    max_rolls = STRATEGY_DEFAULTS["max_rolls"]

    # Per-tier table rows
    tier_rows_html = ""
    for t in TIERS:
        df_t = portfolio_trades[portfolio_trades["dte_tier"] == t["label"]]
        tm = compute_metrics(df_t)
        color = TIER_COLORS.get(t["label"], "#888")
        sw = get_spread_width(t, ticker)
        pf_str = f"{tm['profit_factor']:.2f}" if tm['profit_factor'] < 1000 else "&infin;"
        tier_rows_html += f"""      <tr>
        <td><span class="tier-dot" style="background:{color};"></span>{t['label']}</td>
        <td>{t['priority']}</td><td>{t['dte']}</td><td>P{t['percentile']}</td><td>{sw}pt</td>
        <td>{tm['trades']}</td><td>{tm['wins']}</td><td>{tm['losses']}</td>
        <td>{tm['win_rate']:.1f}%</td>
        <td class="{'positive' if tm['net_pnl']>=0 else 'negative'}">${tm['net_pnl']:,.0f}</td>
        <td>${tm['avg_pnl']:,.0f}</td>
        <td>{tm['roi']:.1f}%</td><td>{tm['roi_per_day']:.1f}%</td><td>{tm['avg_dte_window']:.2f}d</td>
        <td>{tm['sharpe']:.2f}</td>
        <td class="{'negative' if tm['max_drawdown']>0 else ''}">${tm['max_drawdown']:,.0f}</td>
        <td>{pf_str}</td>
      </tr>\n"""

    pf_str_c = f"{m['profit_factor']:.2f}" if m['profit_factor'] < 1000 else "&infin;"
    tier_rows_html += f"""      <tr class="combined-row">
        <td>COMBINED</td><td></td><td>all</td><td>all</td><td>mix</td>
        <td>{m['trades']}</td><td>{m['wins']}</td><td>{m['losses']}</td>
        <td>{m['win_rate']:.1f}%</td>
        <td class="positive">${m['net_pnl']:,.0f}</td>
        <td>${m['avg_pnl']:,.0f}</td>
        <td>{m['roi']:.1f}%</td><td>{m['roi_per_day']:.1f}%</td><td>{m['avg_dte_window']:.1f}d</td>
        <td>{m['sharpe']:.2f}</td>
        <td class="negative">${m['max_drawdown']:,.0f}</td>
        <td>{pf_str_c}</td>
      </tr>"""

    # Monthly rows
    monthly_rows_html = ""
    for _, row in monthly_df.iterrows():
        monthly_rows_html += f"""      <tr>
        <td>{row['month']}</td>
        <td class="{'positive' if row['intraday']>=0 else 'negative'}">${row['intraday']:,.0f}</td>
        <td class="{'positive' if row['eod']>=0 else 'negative'}">${row['eod']:,.0f}</td>
        <td class="{'positive' if row['combined']>=0 else 'negative'}"><strong>${row['combined']:,.0f}</strong></td>
        <td>{int(row['trades'])}</td><td>{row['win_rate']:.1f}%</td>
      </tr>\n"""

    # Loss rows
    loss_rows_html = ""
    if len(loss_df) > 0:
        for dt, row in loss_df.iterrows():
            loss_rows_html += f"""      <tr>
        <td>{dt}</td><td>{int(row['count'])}</td>
        <td class="negative">${row['total_loss']:,.0f}</td>
        <td>{row['tiers']}</td><td>{row['option_types']}</td>
      </tr>\n"""

    # Roll analysis
    roll_html = ""
    if "roll_count" in portfolio_trades.columns:
        roll_html = _build_roll_analysis_html(portfolio_trades)
    else:
        roll_html = "<p style='color: var(--muted);'>No roll data available.</p>"

    chart_prefix = f"{ticker}/charts"

    return f"""
<div class="kpi-strip">
  <div class="kpi"><div class="value">{m['trades']}</div><div class="label">Trades (Portfolio)</div></div>
  <div class="kpi"><div class="value">{m['win_rate']:.1f}%</div><div class="label">Win Rate</div></div>
  <div class="kpi"><div class="value">${m['net_pnl']/1e6:.2f}M</div><div class="label">Net P&amp;L</div></div>
  <div class="kpi"><div class="value">{m['roi']:.1f}%</div><div class="label">ROI (Credit/Risk)</div></div>
  <div class="kpi"><div class="value">{m['roi_per_day']:.1f}%</div><div class="label">ROI / Day</div></div>
  <div class="kpi"><div class="value">{m['sharpe']:.2f}</div><div class="label">Sharpe</div></div>
  <div class="kpi"><div class="value">{pf_str_c}</div><div class="label">Profit Factor</div></div>
  <div class="kpi"><div class="value warn">${m['max_drawdown']:,.0f}</div><div class="label">Max Drawdown</div></div>
  <div class="kpi"><div class="value">{weeks_above_50k}/{total_weeks}</div><div class="label">Weeks &ge; $50K</div></div>
  <div class="kpi"><div class="value">{m_raw['trades']}</div><div class="label">Raw Trades (Pre-Budget)</div></div>
</div>

<div class="section">
  <h2>{ticker} Per-Tier Performance</h2>
  <div class="narrative">
    Period: <strong>{start_date}</strong> to <strong>{end_date}</strong>.
    Spread widths scaled for {ticker} (max_move_cap: {actual_max_move_cap}pt, min_credit: ${actual_min_credit}).
  </div>
  <table>
    <thead><tr>
      <th>Tier</th><th>Pri</th><th>DTE</th><th>Pctl</th><th>Width</th>
      <th>Trades</th><th>Wins</th><th>Losses</th><th>WR%</th>
      <th>Net P&amp;L</th><th>Avg P&amp;L</th><th>ROI%</th><th>ROI/Day</th><th>Window</th>
      <th>Sharpe</th><th>Max DD</th><th>PF</th>
    </tr></thead>
    <tbody>
{tier_rows_html}
    </tbody>
  </table>
  <div class="chart-container" style="margin-top: 24px;"><img src="{chart_prefix}/tier_summary.png" alt="{ticker} Tier Summary"></div>
</div>

<div class="section">
  <h2>{ticker} Cumulative P&amp;L</h2>
  <div class="chart-container"><img src="{chart_prefix}/cumulative_pnl_portfolio.png" alt="{ticker} Cumulative P&L"></div>
</div>

<div class="section">
  <h2>{ticker} Monthly P&amp;L Breakdown</h2>
  <div class="chart-container"><img src="{chart_prefix}/monthly_pnl_intraday_vs_eod.png" alt="{ticker} Monthly P&L"></div>
  <table>
    <thead><tr><th>Month</th><th>Intraday</th><th>EOD</th><th>Combined</th><th>Trades</th><th>WR%</th></tr></thead>
    <tbody>
{monthly_rows_html}
    </tbody>
  </table>
</div>

<div class="section">
  <h2>{ticker} Portfolio Drawdown</h2>
  <div class="chart-container"><img src="{chart_prefix}/portfolio_drawdown.png" alt="{ticker} Drawdown"></div>
</div>

<div class="section">
  <h2>{ticker} Weekly P&amp;L</h2>
  <div class="narrative">
    <strong>{weeks_above_50k} of {total_weeks} weeks</strong> ({weeks_above_50k/total_weeks*100:.1f}%) met the $50K target.
    Average weekly P&amp;L: <strong>${weekly.mean():,.0f}</strong>.
  </div>
  <div class="chart-container"><img src="{chart_prefix}/weekly_pnl.png" alt="{ticker} Weekly P&L"></div>
</div>

<div class="section">
  <h2>{ticker} Daily P&amp;L Distribution</h2>
  <div class="chart-container"><img src="{chart_prefix}/daily_pnl_histogram.png" alt="{ticker} Daily P&L Distribution"></div>
</div>

<div class="section">
  <h2>{ticker} Loss Analysis</h2>
  {"<p style='color: var(--green); font-size: 1.2em; text-align: center; padding: 40px;'>Zero losing trades!</p>" if len(loss_df) == 0 else f'''
  <table>
    <thead><tr><th>Date</th><th>Trades</th><th>Total Loss</th><th>Tiers</th><th>Direction</th></tr></thead>
    <tbody>
{loss_rows_html}
    </tbody>
  </table>'''}
</div>

<div class="section">
  <h2>{ticker} Roll Analysis</h2>
  {roll_html}
</div>

<div class="section">
  <h2>{ticker} Configuration</h2>
  <table style="max-width: 650px;">
    <thead><tr><th>Parameter</th><th>Value</th></tr></thead>
    <tbody>
      <tr><td>Ticker</td><td>{ticker}</td></tr>
      <tr><td>Period</td><td>{start_date} to {end_date}</td></tr>
      <tr><td>Spread Widths</td><td>{', '.join(f"{get_spread_width(t, ticker)}pt" for t in TIERS[:3])} (P90+), {', '.join(f"{get_spread_width(t, ticker)}pt" for t in TIERS[3:5])} (&lt;P90)</td></tr>
      <tr><td>Max Move Cap</td><td>{actual_max_move_cap} pts</td></tr>
      <tr><td>Min Credit</td><td>${actual_min_credit}</td></tr>
      <tr><td>Risk Per Trade</td><td>${MAX_RISK_PER_TRADE:,}</td></tr>
      <tr><td>Daily Budget</td><td>${DAILY_BUDGET:,}</td></tr>
      <tr><td>Rolling</td><td>Enabled ({roll_time_display}, {roll_proximity_pct_display} proximity, max {max_rolls})</td></tr>
      <tr><td>Exit Mode</td><td>Theta Decay</td></tr>
    </tbody>
  </table>
</div>"""


def generate_html_report(ticker_data: dict):
    """Generate comprehensive tabbed HTML report for all tickers.

    ticker_data: dict of ticker -> {
        'all_trades': DataFrame, 'portfolio_trades': DataFrame,
        'monthly_df': DataFrame, 'loss_df': DataFrame, 'daily_pnl': Series,
    }
    """
    now = datetime.now()
    today_str = now.astimezone().strftime("%Y-%m-%d %H:%M:%S %Z")
    today_date = now.strftime("%Y-%m-%d")
    report_name = f"report_tiered_portfolio_v2_{today_date}.html"

    active_tickers = [t for t in TICKERS if t in ticker_data]

    # Build tab buttons and tab content
    tab_buttons = ""
    tab_contents = ""
    for i, ticker in enumerate(active_tickers):
        active_class = " active" if i == 0 else ""
        d = ticker_data[ticker]
        m = compute_metrics(d["portfolio_trades"])

        # Tab button with key metric
        pnl_str = f"${m['net_pnl']/1e6:.1f}M" if abs(m['net_pnl']) >= 1e6 else f"${m['net_pnl']/1e3:.0f}K"
        tab_buttons += f'    <button class="tab-btn{active_class}" onclick="switchTab(\'{ticker}\')" id="tab-btn-{ticker}">{ticker} <span class="tab-metric">{pnl_str} | {m["win_rate"]:.0f}%</span></button>\n'

        # Tab content
        display = "block" if i == 0 else "none"
        tab_content = _build_ticker_tab_html(
            ticker, d["all_trades"], d["portfolio_trades"],
            d["monthly_df"], d["loss_df"], d["daily_pnl"],
        )
        tab_contents += f'  <div class="tab-content" id="tab-{ticker}" style="display:{display};">\n{tab_content}\n  </div>\n'

    # Summary KPI across all tickers
    all_portfolio = pd.concat([d["portfolio_trades"] for d in ticker_data.values()], ignore_index=True)
    m_all = compute_metrics(all_portfolio)
    pf_all = f"{m_all['profit_factor']:.2f}" if m_all['profit_factor'] < 1000 else "&infin;"

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Tiered Portfolio v2 — Multi-Ticker — {today_str}</title>
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
  .hero h1 {{ font-size: 2.4em; font-weight: 700; background: linear-gradient(90deg, var(--accent), var(--green));
    -webkit-background-clip: text; -webkit-text-fill-color: transparent; }}
  .hero .subtitle {{ font-size: 1.15em; color: var(--muted); max-width: 800px; margin: 10px auto 0; }}
  .hero .date {{ margin-top: 12px; font-size: 0.9em; color: var(--muted); }}
  .container {{ max-width: 1300px; margin: 0 auto; padding: 40px 24px; }}
  .kpi-strip {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(160px, 1fr)); gap: 14px; margin-bottom: 48px; }}
  .kpi {{ background: var(--card); border: 1px solid var(--border); border-radius: 12px; padding: 18px; text-align: center; }}
  .kpi .value {{ font-size: 1.7em; font-weight: 700; color: var(--green); }}
  .kpi .value.warn {{ color: var(--orange); }}
  .kpi .value.danger {{ color: var(--red); }}
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
  tr.combined-row td {{ background: rgba(88, 166, 255, 0.08); font-weight: 700; border-top: 2px solid var(--accent); }}
  .positive {{ color: var(--green); }}
  .negative {{ color: var(--red); }}
  .tier-dot {{ display: inline-block; width: 10px; height: 10px; border-radius: 50%; margin-right: 6px; vertical-align: middle; }}
  .callout {{ background: rgba(88, 166, 255, 0.06); border: 1px solid rgba(88, 166, 255, 0.2); border-left: 4px solid var(--accent); border-radius: 8px; padding: 16px 20px; margin: 16px 0; font-size: 0.95em; color: var(--muted); }}
  .callout strong {{ color: var(--text); }}
  .footer {{ text-align: center; padding: 32px; color: var(--muted); font-size: 0.85em; border-top: 1px solid var(--border); margin-top: 48px; }}
  /* Tab styles */
  .tab-bar {{ display: flex; gap: 4px; margin-bottom: 32px; border-bottom: 2px solid var(--border); padding-bottom: 0; }}
  .tab-btn {{ background: var(--card); border: 1px solid var(--border); border-bottom: none; border-radius: 8px 8px 0 0;
    padding: 12px 24px; color: var(--muted); font-size: 1.1em; font-weight: 600; cursor: pointer;
    transition: all 0.2s; }}
  .tab-btn:hover {{ background: #1c2333; color: var(--text); }}
  .tab-btn.active {{ background: var(--accent); color: #fff; border-color: var(--accent); }}
  .tab-metric {{ font-size: 0.75em; font-weight: 400; opacity: 0.8; margin-left: 6px; }}
  .tab-content {{ display: none; }}
  @media (max-width: 800px) {{ .kpi-strip {{ grid-template-columns: repeat(2, 1fr); }} .hero {{ padding: 40px 20px; }} .hero h1 {{ font-size: 1.6em; }} .tab-btn {{ padding: 8px 12px; font-size: 0.9em; }} }}
</style>
<script>
function switchTab(ticker) {{
  document.querySelectorAll('.tab-content').forEach(el => el.style.display = 'none');
  document.querySelectorAll('.tab-btn').forEach(el => el.classList.remove('active'));
  document.getElementById('tab-' + ticker).style.display = 'block';
  document.getElementById('tab-btn-' + ticker).classList.add('active');
}}
</script>
</head>
<body>

<div class="hero">
  <h1>Tiered Multi-DTE Portfolio Backtest v2</h1>
  <div class="subtitle">
    9 DTE tiers &times; {len(active_tickers)} tickers ({', '.join(active_tickers)}) with priority-ordered capital allocation,
    unified $500K/day budget, tiered spread widths, rolling enabled.
  </div>
  <div class="date">Generated: {today_str}</div>
</div>

<div class="container">

<!-- Cross-ticker summary KPIs -->
<div class="section">
  <h2>All Tickers Combined</h2>
  <div class="kpi-strip">
    <div class="kpi"><div class="value">{m_all['trades']}</div><div class="label">Total Trades</div></div>
    <div class="kpi"><div class="value">{m_all['win_rate']:.1f}%</div><div class="label">Win Rate</div></div>
    <div class="kpi"><div class="value">${m_all['net_pnl']/1e6:.2f}M</div><div class="label">Net P&amp;L</div></div>
    <div class="kpi"><div class="value">{m_all['roi']:.1f}%</div><div class="label">ROI</div></div>
    <div class="kpi"><div class="value">{m_all['sharpe']:.2f}</div><div class="label">Sharpe</div></div>
    <div class="kpi"><div class="value">{pf_all}</div><div class="label">Profit Factor</div></div>
    <div class="kpi"><div class="value warn">${m_all['max_drawdown']:,.0f}</div><div class="label">Max Drawdown</div></div>
    <div class="kpi"><div class="value">{len(active_tickers)}</div><div class="label">Tickers</div></div>
  </div>

  <!-- Per-ticker summary table -->
  <table style="max-width: 900px;">
    <thead><tr><th>Ticker</th><th>Period</th><th>Trades</th><th>WR%</th><th>Net P&amp;L</th><th>ROI%</th><th>ROI/Day</th><th>Sharpe</th><th>Max DD</th></tr></thead>
    <tbody>"""

    for ticker in active_tickers:
        d = ticker_data[ticker]
        tm = compute_metrics(d["portfolio_trades"])
        tp = TICKER_PARAMS.get(ticker, {})
        html += f"""
      <tr>
        <td>{ticker}</td>
        <td>{tp.get('backtest_start', '?')} &ndash; {tp.get('backtest_end', '?')}</td>
        <td>{tm['trades']}</td><td>{tm['win_rate']:.1f}%</td>
        <td class="{'positive' if tm['net_pnl']>=0 else 'negative'}">${tm['net_pnl']:,.0f}</td>
        <td>{tm['roi']:.1f}%</td><td>{tm['roi_per_day']:.1f}%</td>
        <td>{tm['sharpe']:.2f}</td>
        <td class="{'negative' if tm['max_drawdown']>0 else ''}">${tm['max_drawdown']:,.0f}</td>
      </tr>"""

    html += f"""
    </tbody>
  </table>
</div>

<!-- Ticker tabs -->
<div class="tab-bar">
{tab_buttons}
</div>

{tab_contents}

</div>
<div class="footer">Tiered Portfolio v2 &mdash; {', '.join(active_tickers)} Credit Spreads &mdash; Generated {today_str}</div>
</body>
</html>"""

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

def _process_ticker(ticker: str, run_backtests: bool = True) -> dict:
    """Run full pipeline for a single ticker: backtest, load, simulate, charts."""
    print(f"\n{'='*80}")
    print(f"  Processing {ticker}")
    print(f"{'='*80}")

    ticker_output = OUTPUT_DIR / ticker
    ticker_chart_dir = ticker_output / "charts"
    ticker_output.mkdir(parents=True, exist_ok=True)

    # Phase 1: Run backtests
    if run_backtests:
        run_all_backtests(ticker)

    # Phase 2: Load trades
    print(f"\nLoading {ticker} trades data...")
    all_trades = load_all_trades(ticker)
    if len(all_trades) == 0:
        print(f"  No trades found for {ticker}, skipping.")
        return None
    print(f"  Loaded {len(all_trades)} raw trades across {all_trades['dte_tier'].nunique()} tiers")

    # Portfolio simulation
    print(f"\nRunning {ticker} portfolio simulation...")
    portfolio_trades, rejected_trades = simulate_portfolio(all_trades)

    # Tables
    print(f"\n--- {ticker} RAW RESULTS ---")
    tier_summary_raw = print_tier_summary(all_trades, f"{ticker} RAW")
    print(f"\n--- {ticker} PORTFOLIO RESULTS ---")
    tier_summary_port = print_tier_summary(portfolio_trades, f"{ticker} PORTFOLIO")
    monthly_df = print_monthly_breakdown(portfolio_trades)
    hour_summary = print_hour_summary(portfolio_trades)
    loss_df = print_loss_analysis(portfolio_trades)
    daily_pnl = print_daily_stats(portfolio_trades)

    # Save CSVs to per-ticker dir
    save_csvs(all_trades, portfolio_trades, tier_summary_raw, tier_summary_port, monthly_df, hour_summary)

    # Charts
    generate_all_charts(all_trades, portfolio_trades, ticker=ticker, chart_dir=ticker_chart_dir)

    return {
        "all_trades": all_trades,
        "portfolio_trades": portfolio_trades,
        "monthly_df": monthly_df,
        "loss_df": loss_df,
        "daily_pnl": daily_pnl,
    }


def main():
    parser = argparse.ArgumentParser(
        description='''
Tiered Multi-DTE Portfolio Backtest v2 — Multi-Ticker.

Runs 9 DTE tiers (6 intraday + 3 EOD) for each configured ticker (NDX, SPX,
RUT) with per-ticker spread widths and parameters.  Results are combined into
a single tabbed HTML report.

All configuration comes from profiles/tiered_v2.yaml (single source of truth).
        ''',
        epilog='''
Examples:
  %(prog)s
      Full run: all tickers, backtests + analysis + charts + HTML report

  %(prog)s --analyze
      Skip backtests, re-analyze existing trades.csv files

  %(prog)s --ticker NDX
      Run only NDX (skip SPX, RUT)

  %(prog)s --ticker NDX --ticker SPX
      Run NDX and SPX only

  %(prog)s --help
      Show this help message
        ''',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--analyze", action="store_true",
                        help="Skip backtests, only run analysis on existing results")
    parser.add_argument("--ticker", action="append", dest="tickers",
                        help="Run specific ticker(s) only (can be repeated). Default: all from config.")
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    tickers_to_run = args.tickers if args.tickers else TICKERS
    print(f"Tickers: {', '.join(tickers_to_run)}")

    # Process each ticker
    ticker_data = {}
    for ticker in tickers_to_run:
        result = _process_ticker(ticker, run_backtests=not args.analyze)
        if result is not None:
            ticker_data[ticker] = result

    if not ticker_data:
        print("ERROR: No trade data found for any ticker.")
        sys.exit(1)

    # Phase 4: Combined HTML report
    generate_html_report(ticker_data)

    # Open report
    report_path = list(OUTPUT_DIR.glob("report_*.html"))
    if report_path:
        os.system(f'open "{report_path[-1]}"')

    print("\nDone!")


if __name__ == "__main__":
    main()
