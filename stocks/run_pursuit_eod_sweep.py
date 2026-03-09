#!/usr/bin/env python3
"""Pursuit-EOD NDX Credit Spread Sweep across percentiles and EOD thresholds.

Evaluates the pursuit_eod directional mode: at each day's close, compare the last
5-min bar to the previous close. If the absolute move exceeds a threshold, lock
the pursuit direction for the entire next trading day (up -> sell calls, down ->
sell puts). If below threshold, skip trading the next day entirely. All configs
use 1DTE options and max_budget contract sizing.

Axes:
  Percentiles: P75, P80, P90, P95, P98, P100
  Thresholds:  0.5%, 0.75%, 1.0%, 1.25%, 1.5%, 2.0%
  Total: 36 configs

Phases:
  1. Execute: 36 backtest configs in parallel via Pool(8)
  2. Analyze: Load trades.csv, add derived columns, compute grouped metrics
  3. Chart: Generate matplotlib PNG charts + console/CSV summary tables

Usage:
  python run_pursuit_eod_sweep.py            # Full run (backtests + analysis)
  python run_pursuit_eod_sweep.py --analyze  # Skip backtests, re-analyze only
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
OUTPUT_DIR = BASE_DIR / "results" / "pursuit_eod_sweep"
CHART_DIR = OUTPUT_DIR / "charts"

PERCENTILES = [75, 80, 90, 95, 98, 100]
THRESHOLDS = [0.005, 0.0075, 0.01, 0.0125, 0.015, 0.02]
THRESHOLD_LABELS = ["0.50%", "0.75%", "1.00%", "1.25%", "1.50%", "2.00%"]

DTE = 1  # All configs use 1DTE


# ---------------------------------------------------------------------------
# Phase 1: Config + Execution
# ---------------------------------------------------------------------------

def _base_config(percentile: int, threshold: float) -> dict:
    """Build config dict for a given percentile and EOD threshold."""
    label = f"p{percentile}_t{threshold:.4f}"

    # Theta params for DTE 1
    theta_ahead = 0.25
    theta_min_decay = 0.50
    theta_cut_behind = 0.40
    theta_cut_min_time = 0.60

    return {
        "infra": {
            "ticker": "NDX",
            "start_date": "2025-09-09",
            "end_date": "2026-03-07",
            "lookback_days": 180,
            "num_processes": 1,
            "output_dir": f"results/pursuit_eod_sweep/{label}",
        },
        "providers": [
            {"name": "csv_equity", "role": "equity", "params": {"csv_dir": "equities_output"}},
            {"name": "csv_options", "role": "options", "params": {
                "csv_dir": "options_csv_output_full",
                "dte_buckets": list(range(0, 12)),
            }},
        ],
        "strategy": {
            "name": "percentile_entry_credit_spread",
            "params": {
                "dte": DTE,
                "percentile": percentile,
                "lookback": 120,
                "option_types": ["put", "call"],
                "spread_width": 50,
                "interval_minutes": 10,
                "entry_start_utc": "13:00",
                "entry_end_utc": "17:00",
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
                "directional_entry": "pursuit_eod",
                "pursuit_eod_threshold": threshold,
                "contract_sizing": "max_budget",
                # Custom keys popped before YAML dump
                "_exit_mode": "theta",
                "_theta_ahead": theta_ahead,
                "_theta_min_decay": theta_min_decay,
                "_theta_cut_behind": theta_cut_behind,
                "_theta_cut_min_time": theta_cut_min_time,
            },
        },
        "constraints": {
            "budget": {
                "max_spend_per_transaction": 75000,
                "daily_budget": 400000,
            },
            "trading_hours": {
                "entry_start": "13:00",
                "entry_end": "17:00",
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
    """Generate 36 configs (6 percentiles x 6 thresholds)."""
    configs = []
    for pct in PERCENTILES:
        for threshold in THRESHOLDS:
            label = f"p{pct}_t{threshold:.4f}"
            cfg = _base_config(pct, threshold)
            configs.append((label, cfg))
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
    """Execute all 36 configs in parallel."""
    configs = _make_configs()

    print("=" * 100)
    print("Pursuit-EOD NDX Credit Spread Sweep")
    print("pursuit_eod + max_budget sizing, 50pt spreads, budget=$75K/trade, DTE=1")
    print("Theta decay exit, no stop loss, no rolling, entry 13:00-17:00 UTC")
    print(f"Percentiles: {PERCENTILES}")
    print(f"Thresholds:  {THRESHOLD_LABELS}")
    print(f"Total configs: {len(configs)}")
    print(f"Period: 2025-03-01 to 2026-02-28 (1 year)")
    print("=" * 100)
    print()
    print(f"Running {len(configs)} configs with Pool(8)...")
    print()

    with Pool(processes=min(8, len(configs))) as pool:
        results = pool.map(run_single, configs)

    results_dict = dict(results)

    # Print quick summary
    print()
    print(f"{'Label':>20} {'Trades':>8} {'Wins':>6} {'WR%':>7} {'Net P&L':>12} {'Sharpe':>8} {'ROI%':>8}")
    print("-" * 75)
    for pct in PERCENTILES:
        for threshold in THRESHOLDS:
            label = f"p{pct}_t{threshold:.4f}"
            r = results_dict.get(label, {})
            if "error" in r:
                print(f"  {label:>18}  ERROR")
            else:
                print(f"  {label:>18} {r['total_trades']:>8} {r['wins']:>6} {r['win_rate']:>6.1f}% "
                      f"${r['net_pnl']:>10,.0f} {r['sharpe']:>7.2f} {r['roi']:>6.1f}%")
        print()

    return results_dict


# ---------------------------------------------------------------------------
# Phase 2: Post-processing
# ---------------------------------------------------------------------------

def _threshold_from_label(label: str) -> float:
    """Extract threshold from label like 'p95_t0.0125'."""
    parts = label.split("_t")
    return float(parts[1]) if len(parts) == 2 else 0.0


def load_all_trades() -> pd.DataFrame:
    """Load trades.csv from each subdir, add derived columns."""
    frames = []
    for pct in PERCENTILES:
        for threshold in THRESHOLDS:
            label = f"p{pct}_t{threshold:.4f}"
            trades_path = OUTPUT_DIR / label / "trades.csv"
            if not trades_path.exists():
                print(f"  WARNING: {trades_path} not found, skipping {label}")
                continue
            df = pd.read_csv(trades_path)
            df["threshold"] = threshold
            df["percentile"] = pct
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
# Phase 3: Output -- Tables
# ---------------------------------------------------------------------------

def print_pct_threshold_summary(trades: pd.DataFrame):
    """Print per-percentile x threshold metrics table."""
    print()
    print("=" * 130)
    print("  SUMMARY BY PERCENTILE x THRESHOLD")
    print("=" * 130)
    header = (f"  {'Pctile':>6} {'Thresh':>7} {'Trades':>7} {'Wins':>5} {'Losses':>6} {'WR%':>7} "
              f"{'Net P&L':>12} {'Avg P&L':>10} {'ROI%':>7} {'Sharpe':>7} {'MaxDD':>10} {'PF':>6}")
    print(header)
    print("  " + "-" * 126)

    rows = []
    for pct in PERCENTILES:
        for i, threshold in enumerate(THRESHOLDS):
            df_cell = trades[(trades["percentile"] == pct) & (np.isclose(trades["threshold"], threshold))]
            m = compute_metrics(df_cell)
            print(f"  P{pct:>4} {THRESHOLD_LABELS[i]:>7} {m['trades']:>7} {m['wins']:>5} {m['losses']:>6} {m['win_rate']:>6.1f}% "
                  f"${m['net_pnl']:>10,.0f} ${m['avg_pnl']:>8,.0f} {m['roi']:>6.1f}% "
                  f"{m['sharpe']:>6.2f} ${m['max_drawdown']:>8,.0f} {m['profit_factor']:>5.2f}")
            m["percentile"] = pct
            m["threshold"] = threshold
            rows.append(m)
        print()

    print("=" * 130)
    return pd.DataFrame(rows)


def print_pct_summary(trades: pd.DataFrame):
    """Print per-percentile aggregated metrics."""
    print()
    print("=" * 120)
    print("  SUMMARY BY PERCENTILE (all thresholds combined)")
    print("=" * 120)
    header = (f"  {'Pctile':>6} {'Trades':>7} {'Wins':>5} {'Losses':>6} {'WR%':>7} "
              f"{'Net P&L':>12} {'Avg P&L':>10} {'ROI%':>7} {'Sharpe':>7} {'MaxDD':>10} {'PF':>6}")
    print(header)
    print("  " + "-" * 116)

    rows = []
    for pct in PERCENTILES:
        df_pct = trades[trades["percentile"] == pct]
        m = compute_metrics(df_pct)
        print(f"  P{pct:>4} {m['trades']:>7} {m['wins']:>5} {m['losses']:>6} {m['win_rate']:>6.1f}% "
              f"${m['net_pnl']:>10,.0f} ${m['avg_pnl']:>8,.0f} {m['roi']:>6.1f}% "
              f"{m['sharpe']:>6.2f} ${m['max_drawdown']:>8,.0f} {m['profit_factor']:>5.2f}")
        m["percentile"] = pct
        rows.append(m)

    print("=" * 120)
    return pd.DataFrame(rows)


def print_threshold_summary(trades: pd.DataFrame):
    """Print per-threshold aggregated metrics."""
    print()
    print("=" * 120)
    print("  SUMMARY BY THRESHOLD (all percentiles combined)")
    print("=" * 120)
    header = (f"  {'Thresh':>7} {'Trades':>7} {'Wins':>5} {'Losses':>6} {'WR%':>7} "
              f"{'Net P&L':>12} {'Avg P&L':>10} {'ROI%':>7} {'Sharpe':>7} {'MaxDD':>10} {'PF':>6}")
    print(header)
    print("  " + "-" * 116)

    rows = []
    for i, threshold in enumerate(THRESHOLDS):
        df_t = trades[np.isclose(trades["threshold"], threshold)]
        m = compute_metrics(df_t)
        print(f"  {THRESHOLD_LABELS[i]:>7} {m['trades']:>7} {m['wins']:>5} {m['losses']:>6} {m['win_rate']:>6.1f}% "
              f"${m['net_pnl']:>10,.0f} ${m['avg_pnl']:>8,.0f} {m['roi']:>6.1f}% "
              f"{m['sharpe']:>6.2f} ${m['max_drawdown']:>8,.0f} {m['profit_factor']:>5.2f}")
        m["threshold"] = threshold
        rows.append(m)

    print("=" * 120)
    return pd.DataFrame(rows)


def print_hour_summary(trades: pd.DataFrame):
    """Print per-entry-hour metrics table."""
    print()
    print("=" * 110)
    print("  SUMMARY BY ENTRY HOUR (UTC, all percentiles/thresholds)")
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


def print_cross_tab(trades: pd.DataFrame):
    """Print Percentile x Threshold pivot tables for net P&L and win rate."""
    print()
    print("=" * 110)
    print("  CROSS TAB: PERCENTILE x THRESHOLD -- Net P&L ($)")
    print("=" * 110)

    hdr = f"  {'Pctile':>6}"
    for tl in THRESHOLD_LABELS:
        hdr += f"{'T ' + tl:>14}"
    print(hdr)
    print("  " + "-" * (6 + 14 * len(THRESHOLDS)))

    pnl_rows = []
    wr_rows = []
    for pct in PERCENTILES:
        row_pnl = {"percentile": pct}
        row_wr = {"percentile": pct}
        line = f"  P{pct:>4}"
        for i, threshold in enumerate(THRESHOLDS):
            df_cell = trades[(trades["percentile"] == pct) & (np.isclose(trades["threshold"], threshold))]
            m = compute_metrics(df_cell)
            line += f" ${m['net_pnl']:>10,.0f}".rjust(14)
            row_pnl[f"t{threshold:.4f}"] = m["net_pnl"]
            row_wr[f"t{threshold:.4f}"] = m["win_rate"]
        print(line)
        pnl_rows.append(row_pnl)
        wr_rows.append(row_wr)

    print()
    print("  CROSS TAB: PERCENTILE x THRESHOLD -- Win Rate (%)")
    print("  " + "-" * (6 + 14 * len(THRESHOLDS)))
    hdr = f"  {'Pctile':>6}"
    for tl in THRESHOLD_LABELS:
        hdr += f"{'T ' + tl:>14}"
    print(hdr)
    print("  " + "-" * (6 + 14 * len(THRESHOLDS)))

    for pct in PERCENTILES:
        line = f"  P{pct:>4}"
        for threshold in THRESHOLDS:
            df_cell = trades[(trades["percentile"] == pct) & (np.isclose(trades["threshold"], threshold))]
            m = compute_metrics(df_cell)
            line += f" {m['win_rate']:>10.1f}%".rjust(14)
        print(line)

    print()
    print("  CROSS TAB: PERCENTILE x THRESHOLD -- Sharpe Ratio")
    print("  " + "-" * (6 + 14 * len(THRESHOLDS)))
    hdr = f"  {'Pctile':>6}"
    for tl in THRESHOLD_LABELS:
        hdr += f"{'T ' + tl:>14}"
    print(hdr)
    print("  " + "-" * (6 + 14 * len(THRESHOLDS)))

    for pct in PERCENTILES:
        line = f"  P{pct:>4}"
        for threshold in THRESHOLDS:
            df_cell = trades[(trades["percentile"] == pct) & (np.isclose(trades["threshold"], threshold))]
            m = compute_metrics(df_cell)
            line += f" {m['sharpe']:>10.2f}".rjust(14)
        print(line)

    print("=" * 110)
    return pd.DataFrame(pnl_rows), pd.DataFrame(wr_rows)


def save_csvs(trades: pd.DataFrame, pct_thresh_summary: pd.DataFrame,
              pct_summary: pd.DataFrame, thresh_summary: pd.DataFrame,
              hour_summary: pd.DataFrame, pnl_tab: pd.DataFrame, wr_tab: pd.DataFrame):
    """Save all CSV outputs."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    trades.to_csv(OUTPUT_DIR / "all_trades.csv", index=False)
    pct_thresh_summary.to_csv(OUTPUT_DIR / "summary_by_pct_threshold.csv", index=False)
    pct_summary.to_csv(OUTPUT_DIR / "summary_by_percentile.csv", index=False)
    thresh_summary.to_csv(OUTPUT_DIR / "summary_by_threshold.csv", index=False)
    hour_summary.to_csv(OUTPUT_DIR / "summary_by_hour.csv", index=False)
    pnl_tab.to_csv(OUTPUT_DIR / "cross_tab_pnl.csv", index=False)
    wr_tab.to_csv(OUTPUT_DIR / "cross_tab_winrate.csv", index=False)

    print(f"\nCSV files saved to {OUTPUT_DIR}/")


# ---------------------------------------------------------------------------
# Phase 3: Output -- Charts
# ---------------------------------------------------------------------------

def chart_heatmap_pnl(trades: pd.DataFrame):
    """Heatmap: Percentile x Threshold -> net P&L."""
    fig, ax = plt.subplots(figsize=(12, 6))

    data = np.zeros((len(PERCENTILES), len(THRESHOLDS)))
    for i, pct in enumerate(PERCENTILES):
        for j, threshold in enumerate(THRESHOLDS):
            df_cell = trades[(trades["percentile"] == pct) & (np.isclose(trades["threshold"], threshold))]
            m = compute_metrics(df_cell)
            data[i, j] = m["net_pnl"]

    im = ax.imshow(data, cmap="RdYlGn", aspect="auto")
    ax.set_xticks(range(len(THRESHOLDS)))
    ax.set_xticklabels(THRESHOLD_LABELS)
    ax.set_yticks(range(len(PERCENTILES)))
    ax.set_yticklabels([f"P{p}" for p in PERCENTILES])

    for i in range(len(PERCENTILES)):
        for j in range(len(THRESHOLDS)):
            val = data[i, j]
            color = "black" if abs(val) < (np.abs(data).max() * 0.6) else "white"
            ax.text(j, i, f"${val:,.0f}", ha="center", va="center", fontsize=8, color=color)

    plt.colorbar(im, ax=ax, label="Net P&L ($)")
    ax.set_title("Net P&L: Percentile x EOD Threshold (DTE=1)", fontsize=14, fontweight="bold")
    ax.set_xlabel("EOD Threshold")
    ax.set_ylabel("Entry Percentile")

    plt.tight_layout()
    fig.savefig(CHART_DIR / "heatmap_net_pnl.png", dpi=150)
    plt.close(fig)


def chart_heatmap_winrate(trades: pd.DataFrame):
    """Heatmap: Percentile x Threshold -> win rate."""
    fig, ax = plt.subplots(figsize=(12, 6))

    data = np.zeros((len(PERCENTILES), len(THRESHOLDS)))
    for i, pct in enumerate(PERCENTILES):
        for j, threshold in enumerate(THRESHOLDS):
            df_cell = trades[(trades["percentile"] == pct) & (np.isclose(trades["threshold"], threshold))]
            m = compute_metrics(df_cell)
            data[i, j] = m["win_rate"]

    im = ax.imshow(data, cmap="YlGn", aspect="auto", vmin=50, vmax=100)
    ax.set_xticks(range(len(THRESHOLDS)))
    ax.set_xticklabels(THRESHOLD_LABELS)
    ax.set_yticks(range(len(PERCENTILES)))
    ax.set_yticklabels([f"P{p}" for p in PERCENTILES])

    for i in range(len(PERCENTILES)):
        for j in range(len(THRESHOLDS)):
            val = data[i, j]
            ax.text(j, i, f"{val:.1f}%", ha="center", va="center", fontsize=8)

    plt.colorbar(im, ax=ax, label="Win Rate (%)")
    ax.set_title("Win Rate: Percentile x EOD Threshold (DTE=1)", fontsize=14, fontweight="bold")
    ax.set_xlabel("EOD Threshold")
    ax.set_ylabel("Entry Percentile")

    plt.tight_layout()
    fig.savefig(CHART_DIR / "heatmap_winrate.png", dpi=150)
    plt.close(fig)


def chart_heatmap_sharpe(trades: pd.DataFrame):
    """Heatmap: Percentile x Threshold -> Sharpe ratio."""
    fig, ax = plt.subplots(figsize=(12, 6))

    data = np.zeros((len(PERCENTILES), len(THRESHOLDS)))
    for i, pct in enumerate(PERCENTILES):
        for j, threshold in enumerate(THRESHOLDS):
            df_cell = trades[(trades["percentile"] == pct) & (np.isclose(trades["threshold"], threshold))]
            m = compute_metrics(df_cell)
            data[i, j] = m["sharpe"]

    vmax = max(abs(np.nanmin(data)), abs(np.nanmax(data)), 1)
    im = ax.imshow(data, cmap="RdYlGn", aspect="auto", vmin=-vmax, vmax=vmax)
    ax.set_xticks(range(len(THRESHOLDS)))
    ax.set_xticklabels(THRESHOLD_LABELS)
    ax.set_yticks(range(len(PERCENTILES)))
    ax.set_yticklabels([f"P{p}" for p in PERCENTILES])

    for i in range(len(PERCENTILES)):
        for j in range(len(THRESHOLDS)):
            val = data[i, j]
            color = "black" if abs(val) < (vmax * 0.6) else "white"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=8, color=color)

    plt.colorbar(im, ax=ax, label="Sharpe Ratio")
    ax.set_title("Sharpe Ratio: Percentile x EOD Threshold (DTE=1)", fontsize=14, fontweight="bold")
    ax.set_xlabel("EOD Threshold")
    ax.set_ylabel("Entry Percentile")

    plt.tight_layout()
    fig.savefig(CHART_DIR / "heatmap_sharpe.png", dpi=150)
    plt.close(fig)


def chart_cumulative_pnl_by_pct(trades: pd.DataFrame):
    """Line chart: cumulative P&L over time, one line per percentile."""
    fig, ax = plt.subplots(figsize=(14, 7))

    colors = plt.cm.viridis(np.linspace(0.15, 0.95, len(PERCENTILES)))
    for idx, pct in enumerate(PERCENTILES):
        df_pct = trades[trades["percentile"] == pct].sort_values("exit_dt")
        if len(df_pct) == 0:
            continue
        cum_pnl = df_pct["pnl"].cumsum()
        ax.plot(df_pct["exit_dt"].values, cum_pnl.values,
                label=f"P{pct}", color=colors[idx], linewidth=1.5)

    ax.set_title("Cumulative P&L by Entry Percentile (all thresholds, DTE=1)", fontsize=14, fontweight="bold")
    ax.set_xlabel("Date")
    ax.set_ylabel("Cumulative P&L ($)")
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color="black", linewidth=0.5)

    fig.autofmt_xdate()
    plt.tight_layout()
    fig.savefig(CHART_DIR / "cumulative_pnl_by_percentile.png", dpi=150)
    plt.close(fig)


def chart_cumulative_pnl_by_threshold(trades: pd.DataFrame):
    """Line chart: cumulative P&L over time, one line per threshold."""
    fig, ax = plt.subplots(figsize=(14, 7))

    colors = plt.cm.tab10(np.linspace(0, 0.6, len(THRESHOLDS)))
    for idx, threshold in enumerate(THRESHOLDS):
        df_t = trades[np.isclose(trades["threshold"], threshold)].sort_values("exit_dt")
        if len(df_t) == 0:
            continue
        cum_pnl = df_t["pnl"].cumsum()
        ax.plot(df_t["exit_dt"].values, cum_pnl.values,
                label=THRESHOLD_LABELS[idx], color=colors[idx], linewidth=1.5)

    ax.set_title("Cumulative P&L by EOD Threshold (all percentiles, DTE=1)", fontsize=14, fontweight="bold")
    ax.set_xlabel("Date")
    ax.set_ylabel("Cumulative P&L ($)")
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color="black", linewidth=0.5)

    fig.autofmt_xdate()
    plt.tight_layout()
    fig.savefig(CHART_DIR / "cumulative_pnl_by_threshold.png", dpi=150)
    plt.close(fig)


def chart_net_pnl_bars(trades: pd.DataFrame):
    """Grouped bar chart: net P&L per threshold, grouped by percentile."""
    fig, ax = plt.subplots(figsize=(14, 7))

    x = np.arange(len(THRESHOLDS))
    width = 0.13
    colors = plt.cm.viridis(np.linspace(0.15, 0.95, len(PERCENTILES)))

    for idx, pct in enumerate(PERCENTILES):
        pnls = []
        for threshold in THRESHOLDS:
            df_cell = trades[(trades["percentile"] == pct) & (np.isclose(trades["threshold"], threshold))]
            m = compute_metrics(df_cell)
            pnls.append(m["net_pnl"])
        offset = (idx - len(PERCENTILES) / 2 + 0.5) * width
        ax.bar(x + offset, pnls, width, label=f"P{pct}", color=colors[idx])

    ax.set_xticks(x)
    ax.set_xticklabels(THRESHOLD_LABELS)
    ax.set_title("Net P&L by EOD Threshold and Percentile (DTE=1)", fontsize=14, fontweight="bold")
    ax.set_xlabel("EOD Threshold")
    ax.set_ylabel("Net P&L ($)")
    ax.legend(loc="best")
    ax.grid(True, axis="y", alpha=0.3)
    ax.axhline(y=0, color="black", linewidth=0.5)

    plt.tight_layout()
    fig.savefig(CHART_DIR / "net_pnl_by_threshold_and_pct.png", dpi=150)
    plt.close(fig)


def chart_sharpe_bars(trades: pd.DataFrame):
    """Grouped bar chart: Sharpe per threshold, grouped by percentile."""
    fig, ax = plt.subplots(figsize=(14, 7))

    x = np.arange(len(THRESHOLDS))
    width = 0.13
    colors = plt.cm.viridis(np.linspace(0.15, 0.95, len(PERCENTILES)))

    for idx, pct in enumerate(PERCENTILES):
        sharpes = []
        for threshold in THRESHOLDS:
            df_cell = trades[(trades["percentile"] == pct) & (np.isclose(trades["threshold"], threshold))]
            m = compute_metrics(df_cell)
            sharpes.append(m["sharpe"])
        offset = (idx - len(PERCENTILES) / 2 + 0.5) * width
        ax.bar(x + offset, sharpes, width, label=f"P{pct}", color=colors[idx])

    ax.set_xticks(x)
    ax.set_xticklabels(THRESHOLD_LABELS)
    ax.set_title("Sharpe Ratio by EOD Threshold and Percentile (DTE=1)", fontsize=14, fontweight="bold")
    ax.set_xlabel("EOD Threshold")
    ax.set_ylabel("Sharpe Ratio (annualized)")
    ax.legend(loc="best")
    ax.grid(True, axis="y", alpha=0.3)
    ax.axhline(y=0, color="black", linewidth=0.5)

    plt.tight_layout()
    fig.savefig(CHART_DIR / "sharpe_by_threshold_and_pct.png", dpi=150)
    plt.close(fig)


def chart_avg_pnl_by_hour(trades: pd.DataFrame):
    """Line chart: avg P&L by entry hour, one line per percentile."""
    hours = sorted(trades["entry_hour_utc"].dropna().unique().astype(int))

    fig, ax = plt.subplots(figsize=(12, 7))
    colors = plt.cm.viridis(np.linspace(0.15, 0.95, len(PERCENTILES)))

    for idx, pct in enumerate(PERCENTILES):
        avg_pnls = []
        valid_hours = []
        for h in hours:
            df_cell = trades[(trades["percentile"] == pct) & (trades["entry_hour_utc"] == h)]
            if len(df_cell) > 0:
                avg_pnls.append(df_cell["pnl"].mean())
                valid_hours.append(h)
        if valid_hours:
            ax.plot(valid_hours, avg_pnls, marker="o", label=f"P{pct}",
                    color=colors[idx], linewidth=1.5, markersize=5)

    ax.set_xticks(hours)
    ax.set_xticklabels([f"{h}:00 UTC\n({h-8}:00 PST)" for h in hours], fontsize=8)

    ax.set_title("Average P&L by Entry Hour and Percentile (DTE=1)", fontsize=14, fontweight="bold")
    ax.set_xlabel("Entry Hour")
    ax.set_ylabel("Avg P&L per Trade ($)")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color="black", linewidth=0.5)

    plt.tight_layout()
    fig.savefig(CHART_DIR / "avg_pnl_by_hour_and_pct.png", dpi=150)
    plt.close(fig)


def generate_all_charts(trades: pd.DataFrame):
    """Generate all charts."""
    CHART_DIR.mkdir(parents=True, exist_ok=True)

    charts = [
        ("heatmap_net_pnl", chart_heatmap_pnl),
        ("heatmap_winrate", chart_heatmap_winrate),
        ("heatmap_sharpe", chart_heatmap_sharpe),
        ("cumulative_pnl_by_percentile", chart_cumulative_pnl_by_pct),
        ("cumulative_pnl_by_threshold", chart_cumulative_pnl_by_threshold),
        ("net_pnl_by_threshold_and_pct", chart_net_pnl_bars),
        ("sharpe_by_threshold_and_pct", chart_sharpe_bars),
        ("avg_pnl_by_hour_and_pct", chart_avg_pnl_by_hour),
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
Pursuit-EOD NDX Credit Spread Sweep across percentiles and EOD thresholds.

Uses end-of-day momentum to lock the next day's direction. At each day's close,
compares the last 5-min bar's close to the previous day's close. If the absolute
move exceeds a threshold, locks pursuit direction for the entire next trading day
(price went up -> sell calls, price went down -> sell puts). If below threshold,
skips trading the next day entirely. All configs use 1DTE options.

Percentiles: P75, P80, P90, P95, P98, P100
Thresholds:  0.5%, 0.75%, 1.0%, 1.25%, 1.5%, 2.0%
Total: 36 configs
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
    print(f"  Loaded {len(trades)} trades across {trades['percentile'].nunique()} percentiles "
          f"and {trades['threshold'].nunique()} thresholds")

    # Phase 3: Tables
    pct_thresh_summary = print_pct_threshold_summary(trades)
    pct_summary = print_pct_summary(trades)
    thresh_summary = print_threshold_summary(trades)
    hour_summary = print_hour_summary(trades)
    pnl_tab, wr_tab = print_cross_tab(trades)

    # Save CSVs
    save_csvs(trades, pct_thresh_summary, pct_summary, thresh_summary, hour_summary, pnl_tab, wr_tab)

    # Phase 3: Charts
    generate_all_charts(trades)

    print("\nDone!")


if __name__ == "__main__":
    main()
