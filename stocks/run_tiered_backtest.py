#!/usr/bin/env python3
"""Tiered Multi-DTE Portfolio Backtest for NDX Credit Spreads.

Combines all learnings from prior sweeps (comprehensive, pursuit_eod, directional,
theta_decay, stoploss) into a single production-ready backtest. Runs a tiered
portfolio strategy across 4 DTE buckets simultaneously, each with optimized
percentile and directional rules, constrained by realistic budgets.

Tiers:
  DTE 0  | P99  | pursuit (intraday)               | Tightest strikes, highest risk
  DTE 1  | P95  | pursuit_eod (1.0% threshold)      | EOD momentum filter
  DTE 2  | P90  | pursuit (intraday)                 | Best risk-adjusted returns
  DTE 5  | P80  | pursuit (intraday)                 | Widest strikes, 100% WR

Shared rules: no stop loss, theta decay exit, 14:00-15:00 UTC entry window,
50pt spread width, $0.75 min credit, max_budget sizing, $75K/trade, $100K/tier/day.

Phases:
  1. Execute: 4 backtest configs in parallel via Pool(4)
  2. Analyze: Per-tier metrics, combined portfolio, monthly/hourly breakdowns
  3. Chart: Cumulative P&L, monthly bars, drawdown, summary bars, daily histogram

Usage:
  python run_tiered_backtest.py            # Full run (backtests + analysis)
  python run_tiered_backtest.py --analyze  # Skip backtests, re-analyze only
"""

import argparse
import sys
import os
import yaml
import numpy as np
import pandas as pd
from pathlib import Path
from multiprocessing import Pool

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "results" / "tiered_portfolio"
CHART_DIR = OUTPUT_DIR / "charts"

# Tier definitions: (dte, percentile, directional_mode, eod_threshold_or_None)
TIERS = [
    {"dte": 0, "percentile": 99, "directional": "pursuit",     "eod_threshold": None,  "label": "dte0_p99"},
    {"dte": 1, "percentile": 95, "directional": "pursuit_eod", "eod_threshold": 0.01,  "label": "dte1_p95_eod"},
    {"dte": 2, "percentile": 90, "directional": "pursuit",     "eod_threshold": None,  "label": "dte2_p90"},
    {"dte": 5, "percentile": 80, "directional": "pursuit",     "eod_threshold": None,  "label": "dte5_p80"},
]

TIER_LABELS = [t["label"] for t in TIERS]


# ---------------------------------------------------------------------------
# Phase 1: Config + Execution
# ---------------------------------------------------------------------------

def _tier_config(tier: dict) -> dict:
    """Build config dict for a single tier."""
    dte = tier["dte"]

    # Theta params: conservative for 0DTE, relaxed for multi-day
    if dte == 0:
        theta_ahead = 0.35
        theta_min_decay = 0.60
        theta_cut_behind = 0.50
        theta_cut_min_time = 0.70
    else:
        theta_ahead = 0.25
        theta_min_decay = 0.50
        theta_cut_behind = 0.40
        theta_cut_min_time = 0.60

    strategy_params = {
        "dte": dte,
        "percentile": tier["percentile"],
        "lookback": 120,
        "option_types": ["put", "call"],
        "spread_width": 50,
        "interval_minutes": 10,
        "entry_start_utc": "14:00",
        "entry_end_utc": "15:00",
        "num_contracts": 1,
        "max_loss_estimate": 75000,
        "profit_target_0dte": 0.75,
        "profit_target_multiday": 0.50,
        "min_roi_per_day": 0.025,
        "min_credit": 0.75,
        "min_total_credit": 0,
        "min_credit_per_point": 0,
        "max_contracts": 0,
        "stop_loss_multiplier": 0,
        "roll_enabled": False,
        "max_rolls": 0,
        "directional_entry": tier["directional"],
        "contract_sizing": "max_budget",
        # Custom keys popped before YAML dump
        "_exit_mode": "theta",
        "_theta_ahead": theta_ahead,
        "_theta_min_decay": theta_min_decay,
        "_theta_cut_behind": theta_cut_behind,
        "_theta_cut_min_time": theta_cut_min_time,
    }

    if tier["eod_threshold"] is not None:
        strategy_params["pursuit_eod_threshold"] = tier["eod_threshold"]

    return {
        "infra": {
            "ticker": "NDX",
            "start_date": "2025-03-01",
            "end_date": "2026-03-05",
            "lookback_days": 180,
            "num_processes": 1,
            "output_dir": f"results/tiered_portfolio/{tier['label']}",
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
                "max_spend_per_transaction": 75000,
                "daily_budget": 100000,
            },
            "trading_hours": {
                "entry_start": "14:00",
                "entry_end": "15:00",
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


def _make_configs():
    """Generate 4 tier configs."""
    configs = []
    for tier in TIERS:
        cfg = _tier_config(tier)
        configs.append((tier["label"], cfg))
    return configs


def run_single(args):
    """Run a single backtest config in a subprocess."""
    label, config_dict = args

    sys.path.insert(0, str(BASE_DIR))
    os.chdir(str(BASE_DIR))

    # Registry imports must happen in each subprocess
    import scripts.backtesting.providers.csv_equity_provider   # noqa
    import scripts.backtesting.providers.csv_options_provider   # noqa
    import scripts.backtesting.instruments.credit_spread        # noqa
    import scripts.backtesting.strategies.credit_spread.percentile_entry  # noqa

    from scripts.backtesting.config import BacktestConfig
    from scripts.backtesting.engine import BacktestEngine

    # Pop custom exit params before writing YAML
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

                # Replace profit_target with theta_decay in exit manager
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


def run_all_backtests():
    """Execute all 4 tier configs in parallel."""
    configs = _make_configs()

    print("=" * 100)
    print("Tiered Multi-DTE Portfolio Backtest")
    print("=" * 100)
    print()
    for t in TIERS:
        mode = t["directional"]
        if t["eod_threshold"]:
            mode += f" ({t['eod_threshold']*100:.1f}% threshold)"
        print(f"  {t['label']:>16}  DTE={t['dte']}  P{t['percentile']}  {mode}")
    print()
    print("Shared: no stop loss, theta decay exit, 14:00-15:00 UTC entry,")
    print("        50pt spreads, $0.75 min credit, max_budget sizing")
    print(f"Budget: $75K/trade, $100K/tier/day, $400K combined/day")
    print(f"Period: 2025-03-01 to 2026-03-05")
    print(f"Total configs: {len(configs)}")
    print("=" * 100)
    print()
    print(f"Running {len(configs)} configs with Pool(4)...")
    print()

    with Pool(processes=min(4, len(configs))) as pool:
        results = pool.map(run_single, configs)

    results_dict = dict(results)

    # Print quick summary
    print()
    print(f"{'Tier':>18} {'Trades':>8} {'Wins':>6} {'WR%':>7} {'Net P&L':>12} {'Sharpe':>8} {'ROI%':>8}")
    print("-" * 70)
    for t in TIERS:
        label = t["label"]
        r = results_dict.get(label, {})
        if "error" in r:
            print(f"  {label:>16}  ERROR")
        else:
            print(f"  {label:>16} {r['total_trades']:>8} {r['wins']:>6} {r['win_rate']:>6.1f}% "
                  f"${r['net_pnl']:>10,.0f} {r['sharpe']:>7.2f} {r['roi']:>6.1f}%")
    print()

    return results_dict


# ---------------------------------------------------------------------------
# Phase 2: Post-processing
# ---------------------------------------------------------------------------

def load_all_trades() -> pd.DataFrame:
    """Load trades.csv from each tier subdir, add dte_tier column."""
    frames = []
    for t in TIERS:
        trades_path = OUTPUT_DIR / t["label"] / "trades.csv"
        if not trades_path.exists():
            print(f"  WARNING: {trades_path} not found, skipping {t['label']}")
            continue
        df = pd.read_csv(trades_path)
        df["dte_tier"] = t["label"]
        df["dte_config"] = t["dte"]
        df["percentile"] = t["percentile"]
        frames.append(df)

    if not frames:
        print("ERROR: No trades.csv files found. Run backtests first.")
        sys.exit(1)

    trades = pd.concat(frames, ignore_index=True)

    # Parse timestamps
    trades["entry_dt"] = pd.to_datetime(trades["entry_time"], errors="coerce")
    trades["exit_dt"] = pd.to_datetime(trades["exit_time"], errors="coerce")

    # Entry hour (UTC) and PST
    trades["entry_hour_utc"] = trades["entry_dt"].dt.hour
    trades["entry_hour_pst"] = trades["entry_hour_utc"] - 8

    # Date columns
    trades["entry_date"] = trades["entry_dt"].dt.date
    trades["exit_date"] = trades["exit_dt"].dt.date
    trades["exit_month"] = trades["exit_dt"].dt.to_period("M")

    # Win/loss flag
    trades["is_win"] = trades["pnl"] > 0

    # Days held
    trades["days_held"] = (trades["exit_dt"] - trades["entry_dt"]).dt.total_seconds() / 86400

    return trades


def compute_metrics(df: pd.DataFrame) -> dict:
    """Compute summary metrics for a group of trades."""
    if len(df) == 0:
        return {
            "trades": 0, "wins": 0, "losses": 0, "win_rate": 0,
            "net_pnl": 0, "avg_pnl": 0, "total_credits": 0,
            "total_gains": 0, "total_losses": 0,
            "roi": 0, "sharpe": 0, "max_drawdown": 0, "profit_factor": 0,
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
        total_credits = float(df["initial_credit"].sum() * df["num_contracts"].sum() * 100) if "initial_credit" in df.columns else 0

    if "max_loss" in df.columns:
        total_risk = float(df["max_loss"].abs().sum())
    else:
        total_risk = total_credits

    roi = (total_credits / total_risk * 100) if total_risk > 0 else 0

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
        "sharpe": sharpe,
        "max_drawdown": max_dd,
        "profit_factor": pf,
    }


# ---------------------------------------------------------------------------
# Phase 2: Output -- Tables
# ---------------------------------------------------------------------------

def print_tier_summary(trades: pd.DataFrame):
    """Print per-tier metrics table."""
    print()
    print("=" * 130)
    print("  PER-TIER METRICS")
    print("=" * 130)
    header = (f"  {'Tier':>18} {'DTE':>4} {'Pctl':>5} {'Trades':>7} {'Wins':>5} {'Losses':>6} {'WR%':>7} "
              f"{'Net P&L':>12} {'Avg P&L':>10} {'ROI%':>7} {'Sharpe':>7} {'MaxDD':>10} {'PF':>6}")
    print(header)
    print("  " + "-" * 126)

    rows = []
    for t in TIERS:
        df_tier = trades[trades["dte_tier"] == t["label"]]
        m = compute_metrics(df_tier)
        print(f"  {t['label']:>18} {t['dte']:>4} P{t['percentile']:>3} {m['trades']:>7} {m['wins']:>5} {m['losses']:>6} {m['win_rate']:>6.1f}% "
              f"${m['net_pnl']:>10,.0f} ${m['avg_pnl']:>8,.0f} {m['roi']:>6.1f}% "
              f"{m['sharpe']:>6.2f} ${m['max_drawdown']:>8,.0f} {m['profit_factor']:>5.2f}")
        m["tier"] = t["label"]
        m["dte"] = t["dte"]
        m["percentile"] = t["percentile"]
        rows.append(m)

    # Combined portfolio
    print("  " + "-" * 126)
    m = compute_metrics(trades)
    print(f"  {'COMBINED':>18} {'all':>4} {'all':>5} {m['trades']:>7} {m['wins']:>5} {m['losses']:>6} {m['win_rate']:>6.1f}% "
          f"${m['net_pnl']:>10,.0f} ${m['avg_pnl']:>8,.0f} {m['roi']:>6.1f}% "
          f"{m['sharpe']:>6.2f} ${m['max_drawdown']:>8,.0f} {m['profit_factor']:>5.2f}")

    print("=" * 130)
    return pd.DataFrame(rows)


def print_monthly_breakdown(trades: pd.DataFrame):
    """Print per-month breakdown with per-tier and combined P&L."""
    print()
    print("=" * 110)
    print("  MONTHLY BREAKDOWN BY TIER")
    print("=" * 110)

    months = sorted(trades["exit_month"].dropna().unique())

    hdr = f"  {'Month':>10}"
    for t in TIERS:
        hdr += f"{t['label']:>16}"
    hdr += f"{'COMBINED':>16}"
    print(hdr)
    print("  " + "-" * (10 + 16 * (len(TIERS) + 1)))

    rows = []
    for month in months:
        df_month = trades[trades["exit_month"] == month]
        line = f"  {str(month):>10}"
        row = {"month": str(month)}
        total_pnl = 0
        for t in TIERS:
            df_cell = df_month[df_month["dte_tier"] == t["label"]]
            pnl = float(df_cell["pnl"].sum()) if len(df_cell) > 0 else 0
            line += f" ${pnl:>12,.0f}".rjust(16)
            row[t["label"]] = pnl
            total_pnl += pnl
        line += f" ${total_pnl:>12,.0f}".rjust(16)
        row["combined"] = total_pnl
        print(line)
        rows.append(row)

    print("=" * 110)
    return pd.DataFrame(rows)


def print_hour_summary(trades: pd.DataFrame):
    """Print per-entry-hour metrics table."""
    print()
    print("=" * 110)
    print("  SUMMARY BY ENTRY HOUR (UTC)")
    print("=" * 110)
    hours = sorted(trades["entry_hour_utc"].dropna().unique().astype(int))
    header = (f"  {'Hour':>6} {'PST':>5} {'Trades':>7} {'Wins':>5} {'WR%':>7} "
              f"{'Net P&L':>12} {'Avg P&L':>10} {'Sharpe':>7}")
    print(header)
    print("  " + "-" * 65)

    rows = []
    for h in hours:
        df_h = trades[trades["entry_hour_utc"] == h]
        m = compute_metrics(df_h)
        pst = h - 8
        print(f"  {h:>4}:xx {pst:>3}:xx {m['trades']:>7} {m['wins']:>5} {m['win_rate']:>6.1f}% "
              f"${m['net_pnl']:>10,.0f} ${m['avg_pnl']:>8,.0f} {m['sharpe']:>6.2f}")
        m["hour_utc"] = h
        m["hour_pst"] = pst
        rows.append(m)

    print("=" * 110)
    return pd.DataFrame(rows)


def print_daily_pnl_stats(trades: pd.DataFrame):
    """Print daily P&L statistics for the combined portfolio."""
    print()
    print("=" * 80)
    print("  DAILY P&L STATISTICS (combined portfolio)")
    print("=" * 80)

    # Aggregate by exit date
    daily = trades.groupby("exit_date")["pnl"].sum()
    print(f"  Trading days:       {len(daily)}")
    print(f"  Mean daily P&L:     ${daily.mean():>10,.0f}")
    print(f"  Median daily P&L:   ${daily.median():>10,.0f}")
    print(f"  Std daily P&L:      ${daily.std():>10,.0f}")
    print(f"  Best day:           ${daily.max():>10,.0f}")
    print(f"  Worst day:          ${daily.min():>10,.0f}")
    print(f"  Positive days:      {(daily > 0).sum()} ({(daily > 0).mean()*100:.1f}%)")
    print(f"  Negative days:      {(daily < 0).sum()} ({(daily < 0).mean()*100:.1f}%)")
    print(f"  Zero days:          {(daily == 0).sum()}")
    print("=" * 80)

    return daily


def save_csvs(trades: pd.DataFrame, tier_summary: pd.DataFrame,
              monthly_df: pd.DataFrame, hour_summary: pd.DataFrame):
    """Save all CSV outputs."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Convert Period columns to strings for CSV
    trades_out = trades.copy()
    if "exit_month" in trades_out.columns:
        trades_out["exit_month"] = trades_out["exit_month"].astype(str)

    trades_out.to_csv(OUTPUT_DIR / "all_trades.csv", index=False)
    tier_summary.to_csv(OUTPUT_DIR / "summary_by_tier.csv", index=False)
    monthly_df.to_csv(OUTPUT_DIR / "monthly_breakdown.csv", index=False)
    hour_summary.to_csv(OUTPUT_DIR / "summary_by_hour.csv", index=False)

    print(f"\nCSV files saved to {OUTPUT_DIR}/")


# ---------------------------------------------------------------------------
# Phase 3: Charts
# ---------------------------------------------------------------------------

TIER_COLORS = {
    "dte0_p99": "#e74c3c",       # red
    "dte1_p95_eod": "#3498db",   # blue
    "dte2_p90": "#2ecc71",       # green
    "dte5_p80": "#f39c12",       # orange
}


def chart_cumulative_pnl(trades: pd.DataFrame):
    """Cumulative P&L: one line per tier + bold combined portfolio line."""
    fig, ax = plt.subplots(figsize=(14, 7))

    # Per-tier lines
    for t in TIERS:
        df_tier = trades[trades["dte_tier"] == t["label"]].sort_values("exit_dt")
        if len(df_tier) == 0:
            continue
        cum_pnl = df_tier["pnl"].cumsum()
        ax.plot(df_tier["exit_dt"].values, cum_pnl.values,
                label=t["label"], color=TIER_COLORS[t["label"]], linewidth=1.5, alpha=0.8)

    # Combined portfolio line
    combined = trades.sort_values("exit_dt")
    cum_combined = combined["pnl"].cumsum()
    ax.plot(combined["exit_dt"].values, cum_combined.values,
            label="COMBINED", color="black", linewidth=3, alpha=0.9)

    ax.set_title("Cumulative P&L: Tiered Portfolio", fontsize=14, fontweight="bold")
    ax.set_xlabel("Date")
    ax.set_ylabel("Cumulative P&L ($)")
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color="gray", linewidth=0.5)

    fig.autofmt_xdate()
    plt.tight_layout()
    fig.savefig(CHART_DIR / "cumulative_pnl.png", dpi=150)
    plt.close(fig)


def chart_monthly_pnl(trades: pd.DataFrame):
    """Stacked bar chart: monthly P&L by tier."""
    months = sorted(trades["exit_month"].dropna().unique())
    month_strs = [str(m) for m in months]

    fig, ax = plt.subplots(figsize=(14, 7))
    x = np.arange(len(months))
    width = 0.6

    bottom_pos = np.zeros(len(months))
    bottom_neg = np.zeros(len(months))

    for t in TIERS:
        pnls = []
        for month in months:
            df_cell = trades[(trades["dte_tier"] == t["label"]) & (trades["exit_month"] == month)]
            pnls.append(float(df_cell["pnl"].sum()) if len(df_cell) > 0 else 0)
        pnls = np.array(pnls)

        pos = np.where(pnls >= 0, pnls, 0)
        neg = np.where(pnls < 0, pnls, 0)

        ax.bar(x, pos, width, bottom=bottom_pos, label=t["label"],
               color=TIER_COLORS[t["label"]], alpha=0.85)
        ax.bar(x, neg, width, bottom=bottom_neg,
               color=TIER_COLORS[t["label"]], alpha=0.85)

        bottom_pos += pos
        bottom_neg += neg

    ax.set_xticks(x)
    ax.set_xticklabels(month_strs, rotation=45, ha="right")
    ax.set_title("Monthly P&L by Tier (stacked)", fontsize=14, fontweight="bold")
    ax.set_ylabel("P&L ($)")
    ax.legend(loc="best")
    ax.grid(True, axis="y", alpha=0.3)
    ax.axhline(y=0, color="black", linewidth=0.5)

    plt.tight_layout()
    fig.savefig(CHART_DIR / "monthly_pnl_stacked.png", dpi=150)
    plt.close(fig)


def chart_portfolio_drawdown(trades: pd.DataFrame):
    """Portfolio drawdown over time."""
    combined = trades.sort_values("exit_dt")
    cum_pnl = combined["pnl"].cumsum().values
    peak = np.maximum.accumulate(cum_pnl)
    drawdown = peak - cum_pnl

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.fill_between(combined["exit_dt"].values, 0, -drawdown,
                    color="#e74c3c", alpha=0.5)
    ax.plot(combined["exit_dt"].values, -drawdown, color="#c0392b", linewidth=1)

    ax.set_title("Portfolio Drawdown", fontsize=14, fontweight="bold")
    ax.set_xlabel("Date")
    ax.set_ylabel("Drawdown ($)")
    ax.grid(True, alpha=0.3)

    fig.autofmt_xdate()
    plt.tight_layout()
    fig.savefig(CHART_DIR / "portfolio_drawdown.png", dpi=150)
    plt.close(fig)


def chart_tier_summary_bars(trades: pd.DataFrame):
    """Per-tier summary bar chart: net P&L, trade count, win rate."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    labels = [t["label"] for t in TIERS]
    colors = [TIER_COLORS[l] for l in labels]
    metrics_list = []
    for t in TIERS:
        df_tier = trades[trades["dte_tier"] == t["label"]]
        metrics_list.append(compute_metrics(df_tier))

    # Net P&L
    ax = axes[0]
    vals = [m["net_pnl"] for m in metrics_list]
    bars = ax.bar(labels, vals, color=colors, alpha=0.85)
    ax.set_title("Net P&L by Tier", fontweight="bold")
    ax.set_ylabel("Net P&L ($)")
    ax.grid(True, axis="y", alpha=0.3)
    ax.axhline(y=0, color="black", linewidth=0.5)
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                f"${val:,.0f}", ha="center", va="bottom", fontsize=8)

    # Trade count
    ax = axes[1]
    vals = [m["trades"] for m in metrics_list]
    bars = ax.bar(labels, vals, color=colors, alpha=0.85)
    ax.set_title("Trade Count by Tier", fontweight="bold")
    ax.set_ylabel("Trades")
    ax.grid(True, axis="y", alpha=0.3)
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                str(val), ha="center", va="bottom", fontsize=8)

    # Win rate
    ax = axes[2]
    vals = [m["win_rate"] for m in metrics_list]
    bars = ax.bar(labels, vals, color=colors, alpha=0.85)
    ax.set_title("Win Rate by Tier", fontweight="bold")
    ax.set_ylabel("Win Rate (%)")
    ax.set_ylim(0, 105)
    ax.grid(True, axis="y", alpha=0.3)
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                f"{val:.1f}%", ha="center", va="bottom", fontsize=8)

    for ax in axes:
        ax.tick_params(axis="x", rotation=30)

    plt.suptitle("Tier Summary", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    fig.savefig(CHART_DIR / "tier_summary.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def chart_daily_pnl_histogram(trades: pd.DataFrame):
    """Daily P&L distribution histogram."""
    daily = trades.groupby("exit_date")["pnl"].sum()

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.hist(daily.values, bins=40, color="#3498db", alpha=0.7, edgecolor="white")
    ax.axvline(x=0, color="black", linewidth=1)
    ax.axvline(x=daily.mean(), color="red", linewidth=1.5, linestyle="--",
               label=f"Mean: ${daily.mean():,.0f}")
    ax.axvline(x=daily.median(), color="green", linewidth=1.5, linestyle="--",
               label=f"Median: ${daily.median():,.0f}")

    ax.set_title("Daily P&L Distribution (Combined Portfolio)", fontsize=14, fontweight="bold")
    ax.set_xlabel("Daily P&L ($)")
    ax.set_ylabel("Frequency")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(CHART_DIR / "daily_pnl_histogram.png", dpi=150)
    plt.close(fig)


def generate_all_charts(trades: pd.DataFrame):
    """Generate all charts."""
    CHART_DIR.mkdir(parents=True, exist_ok=True)

    charts = [
        ("cumulative_pnl", chart_cumulative_pnl),
        ("monthly_pnl_stacked", chart_monthly_pnl),
        ("portfolio_drawdown", chart_portfolio_drawdown),
        ("tier_summary", chart_tier_summary_bars),
        ("daily_pnl_histogram", chart_daily_pnl_histogram),
    ]

    print(f"\nGenerating {len(charts)} charts...")
    for i, (name, func) in enumerate(charts, 1):
        func(trades)
        print(f"  [{i}/{len(charts)}] {name}.png")

    print(f"\nAll charts saved to {CHART_DIR}/")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='''
Tiered Multi-DTE Portfolio Backtest for NDX Credit Spreads.

Runs 4 DTE tiers simultaneously with optimized settings from prior sweeps:

  DTE 0 | P99  | pursuit (intraday)          | Tightest strikes
  DTE 1 | P95  | pursuit_eod (1.0% thresh)   | EOD momentum filter
  DTE 2 | P90  | pursuit (intraday)          | Best risk-adjusted
  DTE 5 | P80  | pursuit (intraday)          | Widest, 100% WR

All tiers: no stop loss, theta decay exit, 14:00-15:00 UTC entry,
50pt spreads, $0.75 min credit, max_budget sizing.

Budget: $75K/trade, $100K/tier/day, $400K combined/day.
Period: 2025-03-01 to 2026-03-05.
        ''',
        epilog='''
Examples:
  %(prog)s
      Full run: backtests + analysis + charts

  %(prog)s --analyze
      Skip backtests, re-analyze existing trades.csv files

  %(prog)s --help
      Show this help message
        ''',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--analyze", action="store_true",
                        help="Skip backtests, only run analysis on existing results")
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Phase 1: Run backtests (unless --analyze)
    if not args.analyze:
        run_all_backtests()
    else:
        print("Skipping backtests (--analyze mode)")

    # Phase 2: Load and enrich trades
    print("\nLoading trades data...")
    trades = load_all_trades()
    print(f"  Loaded {len(trades)} trades across {trades['dte_tier'].nunique()} tiers")

    # Phase 2: Tables
    tier_summary = print_tier_summary(trades)
    monthly_df = print_monthly_breakdown(trades)
    hour_summary = print_hour_summary(trades)
    daily_pnl = print_daily_pnl_stats(trades)

    # Save CSVs
    save_csvs(trades, tier_summary, monthly_df, hour_summary)

    # Phase 3: Charts
    generate_all_charts(trades)

    print("\nDone!")


if __name__ == "__main__":
    main()
