#!/usr/bin/env python3
"""Optimal $500K/Day Deployment — Strategy/Ticker/Timing Sweep.

Three-phase sweep to find the best way to deploy $500K/day across available
strategies, tickers, DTEs, and time windows. Optimizes for credit/max_risk (ROI)
while fully utilizing the budget.

Phases:
  A. Individual Strategy Profiling — run every viable strategy × ticker × DTE
     combo as a standalone backtest (embarrassingly parallel via Pool(8)).
  B. Rank and Select — score by normalized ROI, filter by win rate and trade
     count, produce ranked table.
  C. Orchestrated Portfolio Optimization — compose top configs into portfolio
     variants and run through the orchestrator with $500K/day budget.

Usage:
  python run_strategy_sweep.py --start-date 2025-06-13 --end-date 2026-03-13

  python run_strategy_sweep.py --start-date 2025-06-13 --end-date 2026-03-13 \\
      --daily-budget 500000 --workers 8

  python run_strategy_sweep.py --dry-run
      Preview all configs without running backtests

  python run_strategy_sweep.py --analyze
      Skip Phase A backtests, re-analyze existing results

  python run_strategy_sweep.py --phase-a-only
      Run Phase A only (individual profiling), skip orchestration

  python run_strategy_sweep.py --help
      Show this help message
"""

import argparse
import json
import logging
import os
import sys
import tempfile
import traceback
from datetime import date
from multiprocessing import Pool, cpu_count
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE_DIR = Path(__file__).resolve().parent

# ---------------------------------------------------------------------------
# Strategy × Ticker × DTE config definitions
# ---------------------------------------------------------------------------

def _build_sweep_configs(start_date: str, end_date: str) -> list:
    """Build all (label, config_dict) tuples for Phase A.

    Returns list of (label, config_dict) where label encodes
    strategy_ticker_dteN[_window].
    """
    configs = []

    # --- percentile_entry: NDX, SPX, RUT × DTE 0,1,2 × morning/full ---
    for ticker in ["NDX", "SPX", "RUT"]:
        for dte in [0, 1, 2]:
            for window_name, entry_start, entry_end in [
                ("morning", "14:30", "16:30"),
                ("full", "13:00", "17:00"),
            ]:
                label = f"pctile_{ticker}_dte{dte}_{window_name}"
                # 0DTE uses options_csv_output, multi-day uses options_csv_output_full
                csv_dir = "options_csv_output" if dte == 0 else "options_csv_output_full"
                dte_buckets = [0] if dte == 0 else list(range(0, max(dte + 2, 5)))
                configs.append((label, {
                    "infra": {
                        "ticker": ticker,
                        "start_date": start_date,
                        "end_date": end_date,
                        "lookback_days": 180,
                        "num_processes": 1,
                        "output_dir": f"results/strategy_sweep/{label}",
                    },
                    "providers": [
                        {"name": "csv_equity", "role": "equity",
                         "params": {"csv_dir": "equities_output"}},
                        {"name": "csv_options", "role": "options",
                         "params": {"csv_dir": csv_dir, "dte_buckets": dte_buckets}},
                    ],
                    "strategy": {
                        "name": "percentile_entry_credit_spread",
                        "params": {
                            "dte": dte,
                            "percentile": 95,
                            "lookback": 120,
                            "option_types": ["put", "call"],
                            "spread_width": 50,
                            "interval_minutes": 10,
                            "entry_start_utc": entry_start,
                            "entry_end_utc": entry_end,
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
                            "directional_entry": "pursuit",
                            "contract_sizing": "max_budget",
                        },
                    },
                    "constraints": {
                        "budget": {
                            "max_spend_per_transaction": 75000,
                            "daily_budget": 400000,
                        },
                        "trading_hours": {
                            "entry_start": entry_start,
                            "entry_end": entry_end,
                        },
                        "exit_rules": {
                            "profit_target_pct": 0.75 if dte == 0 else 0.50,
                            "mode": "first_triggered",
                        },
                    },
                    "report": {
                        "formats": ["csv"],
                        "metrics": ["win_rate", "roi", "sharpe",
                                    "max_drawdown", "profit_factor"],
                    },
                }))

    # --- iv_regime_condor: NDX, SPX, RUT × DTE 0 ---
    for ticker in ["NDX", "SPX", "RUT"]:
        label = f"iv_regime_{ticker}"
        configs.append((label, {
            "infra": {
                "ticker": ticker,
                "start_date": start_date,
                "end_date": end_date,
                "lookback_days": 250,
                "num_processes": 1,
                "output_dir": f"results/strategy_sweep/{label}",
            },
            "providers": [
                {"name": "csv_equity", "role": "equity",
                 "params": {"csv_dir": "equities_output"}},
                {"name": "csv_options", "role": "options",
                 "params": {"csv_dir": "options_csv_output",
                            "dte_buckets": [0, 1]}},
            ],
            "strategy": {
                "name": "iv_regime_condor",
                "params": {
                    "percentile": 95,
                    "iron_condor_percentile": 85,
                    "lookback": 120,
                    "dte": 0,
                    "spread_width": 50,
                    "vix_csv_dir": "equities_output/I:VIX",
                    "vix_lookback": 60,
                    "low_vol_threshold": 30,
                    "entry_start_utc": "14:00",
                    "entry_end_utc": "17:00",
                    "interval_minutes": 30,
                    "num_contracts": 1,
                    "max_loss_estimate": 10000,
                    "min_credit": 0.30,
                    "use_mid": True,
                },
            },
            "constraints": {
                "budget": {
                    "max_spend_per_transaction": 20000,
                    "daily_budget": 100000,
                },
                "trading_hours": {
                    "entry_start": "14:00",
                    "entry_end": "17:00",
                    "forced_exit_time": "20:45",
                },
                "exit_rules": {
                    "profit_target_pct": 0.50,
                    "stop_loss_pct": 2.0,
                    "time_exit": "20:30",
                    "mode": "first_triggered",
                },
            },
            "report": {
                "formats": ["csv"],
                "metrics": ["win_rate", "roi", "sharpe",
                            "max_drawdown", "profit_factor"],
            },
        }))

    # --- weekly_iron_condor: NDX, SPX, RUT ---
    for ticker in ["NDX", "SPX", "RUT"]:
        label = f"weekly_ic_{ticker}"
        configs.append((label, {
            "infra": {
                "ticker": ticker,
                "start_date": start_date,
                "end_date": end_date,
                "lookback_days": 250,
                "num_processes": 1,
                "output_dir": f"results/strategy_sweep/{label}",
            },
            "providers": [
                {"name": "csv_equity", "role": "equity",
                 "params": {"csv_dir": "equities_output"}},
                {"name": "csv_options", "role": "options",
                 "params": {"csv_dir": "options_csv_output_full",
                            "dte_buckets": list(range(0, 11))}},
            ],
            "strategy": {
                "name": "weekly_iron_condor",
                "params": {
                    "percentile": 80,
                    "lookback": 120,
                    "dte_windows": [5, 7, 10],
                    "spread_width": 50,
                    "entry_days": [0, 1],
                    "entry_start_utc": "14:00",
                    "entry_end_utc": "17:00",
                    "vix_csv_dir": "equities_output/I:VIX",
                    "vix_lookback": 60,
                    "skip_vix_regimes": ["extreme"],
                    "num_contracts": 1,
                    "max_loss_estimate": 10000,
                    "min_credit": 0.50,
                    "use_mid": True,
                },
            },
            "constraints": {
                "budget": {
                    "max_spend_per_transaction": 20000,
                    "daily_budget": 100000,
                },
                "trading_hours": {
                    "entry_start": "14:00",
                    "entry_end": "17:00",
                },
                "exit_rules": {
                    "profit_target_pct": 0.50,
                    "stop_loss_pct": 2.0,
                    "mode": "first_triggered",
                },
            },
            "report": {
                "formats": ["csv"],
                "metrics": ["win_rate", "roi", "sharpe",
                            "max_drawdown", "profit_factor"],
            },
        }))

    # --- tail_hedged: NDX, SPX, RUT × DTE 0 ---
    for ticker in ["NDX", "SPX", "RUT"]:
        label = f"tail_hedge_{ticker}"
        configs.append((label, {
            "infra": {
                "ticker": ticker,
                "start_date": start_date,
                "end_date": end_date,
                "lookback_days": 250,
                "num_processes": 1,
                "output_dir": f"results/strategy_sweep/{label}",
            },
            "providers": [
                {"name": "csv_equity", "role": "equity",
                 "params": {"csv_dir": "equities_output"}},
                {"name": "csv_options", "role": "options",
                 "params": {"csv_dir": "options_csv_output",
                            "dte_buckets": [0]}},
            ],
            "strategy": {
                "name": "tail_hedged_credit_spread",
                "params": {
                    "percentile": 95,
                    "hedge_percentile": 99,
                    "lookback": 120,
                    "dte": 0,
                    "spread_width": 50,
                    "hedge_spread_width": 50,
                    "base_hedge_pct": 0.05,
                    "vix_csv_dir": "equities_output/I:VIX",
                    "vix_lookback": 60,
                    "option_types": ["put", "call"],
                    "entry_start_utc": "13:00",
                    "entry_end_utc": "17:00",
                    "interval_minutes": 10,
                    "num_contracts": 1,
                    "max_loss_estimate": 10000,
                    "min_credit": 0.30,
                    "use_mid": True,
                },
            },
            "constraints": {
                "budget": {
                    "max_spend_per_transaction": 20000,
                    "daily_budget": 100000,
                },
                "trading_hours": {
                    "entry_start": "13:00",
                    "entry_end": "17:00",
                    "forced_exit_time": "20:45",
                },
                "exit_rules": {
                    "profit_target_pct": 0.50,
                    "stop_loss_pct": 2.0,
                    "time_exit": "20:30",
                    "mode": "first_triggered",
                },
            },
            "report": {
                "formats": ["csv"],
                "metrics": ["win_rate", "roi", "sharpe",
                            "max_drawdown", "profit_factor"],
            },
        }))

    # --- zero_dte: NDX, SPX, RUT ---
    for ticker in ["NDX", "SPX", "RUT"]:
        label = f"zero_dte_{ticker}"
        configs.append((label, {
            "infra": {
                "ticker": ticker,
                "start_date": start_date,
                "end_date": end_date,
                "lookback_days": 100,
                "num_processes": 1,
                "output_dir": f"results/strategy_sweep/{label}",
            },
            "providers": [
                {"name": "csv_equity", "role": "equity",
                 "params": {"csv_dir": "equities_output"}},
                {"name": "csv_options", "role": "options",
                 "params": {"csv_dir": "options_csv_output",
                            "dte_buckets": [0]}},
            ],
            "strategy": {
                "name": "zero_dte_credit_spread",
                "params": {
                    "option_types": ["put", "call"],
                    "percent_beyond": "0.03:0.05",
                    "instruments": ["credit_spread"],
                    "entry_strategy": "single_entry",
                    "num_contracts": 1,
                    "max_loss_estimate": 10000,
                },
            },
            "constraints": {
                "budget": {
                    "max_spend_per_transaction": 20000,
                    "daily_budget": 100000,
                },
                "trading_hours": {
                    "entry_start": "14:45",
                    "entry_end": "20:00",
                    "forced_exit_time": "20:45",
                },
                "exit_rules": {
                    "profit_target_pct": 0.50,
                    "stop_loss_pct": 2.0,
                    "time_exit": "20:30",
                    "mode": "first_triggered",
                },
            },
            "report": {
                "formats": ["csv"],
                "metrics": ["win_rate", "roi", "sharpe",
                            "max_drawdown", "profit_factor"],
            },
        }))

    # --- tqqq_momentum_scalper: TQQQ ---
    label = "tqqq_scalper"
    configs.append((label, {
        "infra": {
            "ticker": "TQQQ",
            "start_date": start_date,
            "end_date": end_date,
            "lookback_days": 30,
            "num_processes": 1,
            "output_dir": f"results/strategy_sweep/{label}",
        },
        "providers": [
            {"name": "csv_equity", "role": "equity",
             "params": {"csv_dir": "equities_output"}},
            {"name": "csv_options", "role": "options",
             "params": {"csv_dir": "options_csv_output_full",
                        "dte_buckets": [0, 1]}},
        ],
        "strategy": {
            "name": "tqqq_momentum_scalper",
            "params": {
                "signal_mode": "combined",
                "option_types": ["put", "call"],
                "percent_beyond": "0.02:0.02",
                "min_width": 1,
                "max_width": 2,
                "num_contracts": 10,
                "max_loss_estimate": 10000,
                "min_consecutive_down": 3,
                "min_consecutive_up": 4,
                "max_gap_pct": 0.005,
                "instruments": ["credit_spread"],
            },
        },
        "constraints": {
            "budget": {
                "max_spend_per_transaction": 10000,
                "daily_budget": 50000,
            },
            "trading_hours": {
                "entry_start": "09:30",
                "entry_end": "15:00",
                "forced_exit_time": "15:45",
            },
            "exit_rules": {
                "profit_target_pct": 0.50,
                "stop_loss_pct": 2.0,
                "time_exit": "15:30",
                "mode": "first_triggered",
            },
        },
        "report": {
            "formats": ["csv"],
            "metrics": ["win_rate", "roi", "sharpe",
                        "max_drawdown", "profit_factor"],
        },
    }))

    return configs


# ---------------------------------------------------------------------------
# Phase A: Run individual backtests
# ---------------------------------------------------------------------------

def _run_single_config(args):
    """Run a single backtest config in a subprocess."""
    label, config_dict = args

    sys.path.insert(0, str(BASE_DIR))
    os.chdir(str(BASE_DIR))

    # Registry imports — must happen in each subprocess
    import scripts.backtesting.providers.csv_equity_provider         # noqa: F401
    import scripts.backtesting.providers.csv_options_provider         # noqa: F401
    import scripts.backtesting.instruments.credit_spread              # noqa: F401
    import scripts.backtesting.instruments.iron_condor                # noqa: F401
    import scripts.backtesting.strategies.credit_spread.zero_dte      # noqa: F401
    import scripts.backtesting.strategies.credit_spread.multi_day     # noqa: F401
    import scripts.backtesting.strategies.credit_spread.scale_in      # noqa: F401
    import scripts.backtesting.strategies.credit_spread.tiered        # noqa: F401
    import scripts.backtesting.strategies.credit_spread.time_allocated      # noqa: F401
    import scripts.backtesting.strategies.credit_spread.gate_filtered       # noqa: F401
    import scripts.backtesting.strategies.credit_spread.percentile_entry    # noqa: F401
    import scripts.backtesting.strategies.credit_spread.tqqq_momentum_scalper  # noqa: F401
    import scripts.backtesting.strategies.credit_spread.iv_regime_condor       # noqa: F401
    import scripts.backtesting.strategies.credit_spread.weekly_iron_condor     # noqa: F401
    import scripts.backtesting.strategies.credit_spread.tail_hedged            # noqa: F401

    from scripts.backtesting.config import BacktestConfig
    from scripts.backtesting.engine import BacktestEngine

    # Write temp YAML for this config
    config_path = BASE_DIR / f"_tmp_sweep_{label}.yaml"
    with open(config_path, "w") as f:
        yaml.dump(config_dict, f)

    try:
        config = BacktestConfig.from_yaml(str(config_path))
        engine = BacktestEngine(config)
        results = engine.run()

        if results and "metrics" in results:
            m = results["metrics"]
            # Extract avg credit and avg max_loss from individual trade results
            trade_results = results.get("results", [])
            avg_credit = 0.0
            avg_max_loss = 0.0
            if trade_results:
                credits = [t.get("credit", t.get("initial_credit", 0))
                           for t in trade_results if isinstance(t, dict)]
                max_losses = [abs(t.get("max_loss", 0))
                              for t in trade_results if isinstance(t, dict)]
                if credits:
                    avg_credit = sum(credits) / len(credits)
                if max_losses:
                    avg_max_loss = sum(max_losses) / len(max_losses)

            return (label, {
                "strategy": config_dict["strategy"]["name"],
                "ticker": config_dict["infra"]["ticker"],
                "dte": config_dict["strategy"]["params"].get("dte", 0),
                "total_trades": m.get("total_trades", 0),
                "win_rate": m.get("win_rate", 0),
                "wins": m.get("wins", 0),
                "losses": m.get("losses", 0),
                "net_pnl": m.get("net_pnl", 0),
                "total_credits": m.get("total_credits", 0),
                "roi": m.get("roi", 0),
                "sharpe": m.get("sharpe", 0),
                "max_drawdown": m.get("max_drawdown", 0),
                "profit_factor": m.get("profit_factor", 0),
                "avg_credit": avg_credit,
                "avg_max_loss": avg_max_loss,
                "avg_credit_risk_ratio": (avg_credit / avg_max_loss
                                          if avg_max_loss > 0 else 0),
            })
        return (label, {"error": "no results", "strategy": config_dict["strategy"]["name"],
                        "ticker": config_dict["infra"]["ticker"]})

    except Exception as e:
        return (label, {
            "error": str(e),
            "traceback": traceback.format_exc(),
            "strategy": config_dict["strategy"]["name"],
            "ticker": config_dict["infra"]["ticker"],
        })
    finally:
        if config_path.exists():
            config_path.unlink()


def run_phase_a(configs: list, workers: int) -> pd.DataFrame:
    """Phase A: run all individual backtests in parallel."""
    print("=" * 100)
    print("PHASE A: Individual Strategy Profiling")
    print(f"  Configs: {len(configs)}")
    print(f"  Workers: {workers}")
    print("=" * 100)
    print()

    # Show what will run
    for label, cfg in configs:
        strategy = cfg["strategy"]["name"]
        ticker = cfg["infra"]["ticker"]
        dte = cfg["strategy"]["params"].get("dte", "?")
        print(f"  {label:<40} {strategy:<35} {ticker:<6} DTE={dte}")
    print()

    print(f"Running {len(configs)} configs with Pool({workers})...")
    with Pool(processes=workers) as pool:
        results = pool.map(_run_single_config, configs)

    results_dict = dict(results)

    # Print quick summary
    print()
    print(f"{'Label':<40} {'Strategy':<25} {'Tkr':<5} {'Trades':>7} "
          f"{'WR%':>7} {'Net P&L':>12} {'Sharpe':>8} {'ROI%':>8}")
    print("-" * 120)
    for label, r in sorted(results_dict.items()):
        if "error" in r:
            print(f"  {label:<38} {r.get('strategy','?'):<25} "
                  f"{r.get('ticker','?'):<5} ERROR: {r['error'][:40]}")
        else:
            print(f"  {label:<38} {r['strategy']:<25} {r['ticker']:<5} "
                  f"{r['total_trades']:>7} {r['win_rate']:>6.1f}% "
                  f"${r['net_pnl']:>10,.0f} {r['sharpe']:>7.2f} "
                  f"{r['roi']:>6.1f}%")
    print()

    # Build DataFrame
    rows = []
    for label, r in results_dict.items():
        if "error" not in r:
            r["label"] = label
            rows.append(r)
    df = pd.DataFrame(rows)

    return df


# ---------------------------------------------------------------------------
# Phase B: Rank and Select
# ---------------------------------------------------------------------------

def run_phase_b(df: pd.DataFrame,
                min_win_rate: float = 80.0,
                min_trades: int = 20,
                min_profit_factor: float = 2.0,
                top_n: int = 15) -> pd.DataFrame:
    """Phase B: score, filter, and rank configs."""
    print("=" * 100)
    print("PHASE B: Rank and Select Top Combinations")
    print(f"  Filters: win_rate >= {min_win_rate}%, trades >= {min_trades}, "
          f"profit_factor >= {min_profit_factor}")
    print(f"  Top N: {top_n}")
    print("=" * 100)
    print()

    if len(df) == 0:
        print("  No results to rank.")
        return df

    # Compute normalized ROI: (avg_credit / avg_max_loss) / max(1, dte)
    df = df.copy()
    df["normalized_roi"] = df.apply(
        lambda r: (r["avg_credit_risk_ratio"] / max(1, r["dte"]))
        if r["avg_credit_risk_ratio"] > 0 else 0,
        axis=1,
    )

    # Composite score: weighted blend of normalized ROI, Sharpe, win rate
    roi_norm = df["normalized_roi"] / df["normalized_roi"].max() if df["normalized_roi"].max() > 0 else 0
    sharpe_norm = df["sharpe"].clip(lower=0) / max(df["sharpe"].max(), 1)
    wr_norm = df["win_rate"] / 100.0
    df["composite_score"] = 0.50 * roi_norm + 0.30 * sharpe_norm + 0.20 * wr_norm

    # Apply filters
    qualified = df[
        (df["win_rate"] >= min_win_rate) &
        (df["total_trades"] >= min_trades) &
        (df["profit_factor"] >= min_profit_factor)
    ].copy()

    print(f"  {len(df)} total configs → {len(qualified)} pass filters")

    # Rank by composite score
    qualified = qualified.sort_values("composite_score", ascending=False)
    qualified = qualified.head(top_n).reset_index(drop=True)
    qualified["rank"] = range(1, len(qualified) + 1)

    # Print ranked table
    print()
    print(f"{'Rank':>4} {'Label':<40} {'Strategy':<25} {'Tkr':<5} "
          f"{'DTE':>3} {'Trades':>6} {'WR%':>6} {'ROI%':>7} "
          f"{'Sharpe':>7} {'NormROI':>8} {'Score':>6}")
    print("-" * 130)
    for _, r in qualified.iterrows():
        print(f"  {r['rank']:>2} {r['label']:<40} "
              f"{r['strategy']:<25} {r['ticker']:<5} {r['dte']:>3} "
              f"{r['total_trades']:>6} {r['win_rate']:>5.1f}% "
              f"{r['roi']:>6.1f}% {r['sharpe']:>6.2f} "
              f"{r['normalized_roi']:>7.4f} {r['composite_score']:>5.3f}")
    print()

    return qualified


# ---------------------------------------------------------------------------
# Phase C: Orchestrated Portfolio Optimization
# ---------------------------------------------------------------------------

def _build_portfolio_configs(ranked_df: pd.DataFrame, all_df: pd.DataFrame,
                             start_date: str, end_date: str,
                             daily_budget: int,
                             output_dir: str) -> list:
    """Build orchestration YAML configs for portfolio variants.

    Returns list of (portfolio_name, yaml_path) tuples.
    """
    portfolios = []
    configs_dir = Path(output_dir) / "phase_c_configs"
    configs_dir.mkdir(parents=True, exist_ok=True)

    if len(ranked_df) == 0:
        return portfolios

    def _make_instance(row, priority=5):
        """Create an algo instance dict from a ranked row."""
        label = row["label"]
        strategy = row["strategy"]
        ticker = row["ticker"]
        dte = int(row["dte"])

        # Point to the base YAML config for this strategy (absolute path)
        configs_base = str(BASE_DIR / "scripts" / "backtesting" / "configs")
        config_map = {
            "percentile_entry_credit_spread": "percentile_entry_ndx.yaml",
            "iv_regime_condor": "iv_regime_condor_ndx.yaml",
            "weekly_iron_condor": "weekly_iron_condor_ndx.yaml",
            "tail_hedged_credit_spread": "tail_hedged_ndx.yaml",
            "zero_dte_credit_spread": "credit_spread_0dte_ndx.yaml",
            "tqqq_momentum_scalper": "tqqq_momentum_scalper.yaml",
        }
        base_config = os.path.join(
            configs_base,
            config_map.get(strategy, "credit_spread_0dte_ndx.yaml"),
        )

        overrides = {"ticker": ticker}
        if dte > 0:
            overrides["dte"] = dte

        return {
            "algo": strategy,
            "id": f"sweep:{label}",
            "config": base_config,
            "overrides": overrides,
            "triggers": ["always"],
            "priority": priority,
        }

    def _write_orchestration(name, instances, budget=None):
        """Write an orchestration YAML and return its path."""
        budget = budget or daily_budget
        manifest = {
            "orchestration": {
                "name": name,
                "lookback_days": 250,
                "selection_mode": "best_score",
                "daily_budget": budget,
                "output_dir": f"{output_dir}/phase_c_portfolios/{name}",
                "start_date": start_date,
                "end_date": end_date,
                "phase2_mode": "interval",
                "interval_minutes": 15,
                "top_n": 3,
                "scoring_weights": [0.80, 0.10, 0.10],
                "triggers": {
                    "always": {"type": "always"},
                },
                "instances": instances,
            }
        }
        path = configs_dir / f"{name}.yaml"
        with open(path, "w") as f:
            yaml.dump(manifest, f, default_flow_style=False)
        return str(path)

    # 1. Best single config (all-in)
    if len(ranked_df) >= 1:
        best = ranked_df.iloc[0]
        inst = _make_instance(best, priority=1)
        path = _write_orchestration(
            "best_single", [inst], daily_budget)
        portfolios.append(("best_single", path))

    # 2. Top 3 per ticker — diversified across tickers
    top_per_ticker = []
    for ticker in ranked_df["ticker"].unique():
        ticker_rows = ranked_df[ranked_df["ticker"] == ticker].head(3)
        for _, row in ticker_rows.iterrows():
            top_per_ticker.append(_make_instance(row, priority=3))
    if top_per_ticker:
        path = _write_orchestration(
            "top3_per_ticker", top_per_ticker, daily_budget)
        portfolios.append(("top3_per_ticker", path))

    # 3. Top 1 per ticker + TQQQ — concentrated picks
    concentrated = []
    for ticker in ["NDX", "SPX", "RUT"]:
        ticker_rows = ranked_df[ranked_df["ticker"] == ticker].head(1)
        for _, row in ticker_rows.iterrows():
            concentrated.append(_make_instance(row, priority=2))
    # Add TQQQ if present
    tqqq_rows = ranked_df[ranked_df["ticker"] == "TQQQ"].head(1)
    for _, row in tqqq_rows.iterrows():
        concentrated.append(_make_instance(row, priority=4))
    if concentrated:
        path = _write_orchestration(
            "concentrated", concentrated, daily_budget)
        portfolios.append(("concentrated", path))

    # 4. IC-heavy portfolio — iron condor strategies prioritized
    ic_strategies = ["iv_regime_condor", "weekly_iron_condor"]
    ic_rows = ranked_df[ranked_df["strategy"].isin(ic_strategies)]
    ic_instances = []
    for _, row in ic_rows.iterrows():
        ic_instances.append(_make_instance(row, priority=2))
    # Add best non-IC as complement
    non_ic = ranked_df[~ranked_df["strategy"].isin(ic_strategies)].head(2)
    for _, row in non_ic.iterrows():
        ic_instances.append(_make_instance(row, priority=5))
    if ic_instances:
        path = _write_orchestration(
            "ic_heavy", ic_instances, daily_budget)
        portfolios.append(("ic_heavy", path))

    # 5. Time-diversified — morning + full-day configs
    morning = ranked_df[ranked_df["label"].str.contains("morning", na=False)].head(3)
    full_day = ranked_df[ranked_df["label"].str.contains("full", na=False)].head(3)
    time_instances = []
    for _, row in morning.iterrows():
        time_instances.append(_make_instance(row, priority=3))
    for _, row in full_day.iterrows():
        time_instances.append(_make_instance(row, priority=3))
    # Fill with any remaining top configs
    if len(time_instances) < 3:
        for _, row in ranked_df.head(3).iterrows():
            inst = _make_instance(row, priority=5)
            if inst["id"] not in [i["id"] for i in time_instances]:
                time_instances.append(inst)
    if time_instances:
        path = _write_orchestration(
            "time_diversified", time_instances, daily_budget)
        portfolios.append(("time_diversified", path))

    return portfolios


def _aggregate_per_instance_metrics(output_dir: str, name: str) -> dict:
    """Aggregate metrics from per-instance CSV files in a portfolio's output dir."""
    per_inst_dir = os.path.join(output_dir, "phase_c_portfolios", name, "per_instance")
    if not os.path.isdir(per_inst_dir):
        return {}

    total_trades = total_wins = total_losses = 0
    total_pnl = total_credits = total_gains = total_loss_amt = 0.0
    max_dd = 0.0
    attribution = {}

    for inst_name in sorted(os.listdir(per_inst_dir)):
        mf = os.path.join(per_inst_dir, inst_name, "metrics.csv")
        if not os.path.exists(mf):
            continue
        try:
            m = pd.read_csv(mf)
            if len(m) == 0:
                continue
            r = m.iloc[0]
            trades = int(r["total_trades"])
            wins = int(r["wins"])
            losses = int(r["losses"])
            pnl = float(r["net_pnl"])
            credits = float(r["total_credits"])
            gains = float(r["total_gains"])
            loss_amt = float(r["total_losses"])
            dd = float(r["max_drawdown"])

            total_trades += trades
            total_wins += wins
            total_losses += losses
            total_pnl += pnl
            total_credits += credits
            total_gains += gains
            total_loss_amt += loss_amt
            max_dd = max(max_dd, dd)

            attribution[inst_name] = {
                "trades": trades,
                "metrics": {
                    "win_rate": float(r["win_rate"]),
                    "net_pnl": pnl,
                    "sharpe": float(r["sharpe"]),
                    "roi": float(r["roi"]),
                },
            }
        except Exception:
            continue

    wr = (total_wins / total_trades * 100) if total_trades > 0 else 0
    pf = (total_gains / total_loss_amt) if total_loss_amt > 0 else float("inf")
    roi = (total_pnl / total_credits * 100) if total_credits > 0 else 0
    sharpe_vals = [a["metrics"]["sharpe"] for a in attribution.values()
                   if a["metrics"]["sharpe"] > 0]
    avg_sharpe = sum(sharpe_vals) / len(sharpe_vals) if sharpe_vals else 0

    return {
        "combined_metrics": {
            "total_trades": total_trades,
            "win_rate": wr,
            "roi": roi,
            "sharpe": avg_sharpe,
            "net_pnl": total_pnl,
            "max_drawdown": max_dd,
            "profit_factor": pf,
        },
        "per_algo_attribution": attribution,
    }


def _run_single_orchestration(name: str, config_path: str,
                              output_dir: str) -> tuple:
    """Run a single orchestration (in-process, since orchestrator uses its own Pool)."""
    from scripts.backtesting.orchestration.manifest import OrchestrationManifest
    from scripts.backtesting.orchestration.engine import OrchestratorEngine

    try:
        manifest = OrchestrationManifest.load(config_path)
        engine = OrchestratorEngine(manifest)
        summary = engine.run()
        engine.save_results(summary)

        # Aggregate from per-instance CSVs (more reliable than combined_metrics)
        agg = _aggregate_per_instance_metrics(output_dir, name)
        combined = agg.get("combined_metrics", summary.get("combined_metrics", {}))
        attribution = agg.get("per_algo_attribution",
                              summary.get("per_algo_attribution", {}))

        return (name, {
            "success": True,
            "total_accepted": combined.get("total_trades",
                                           summary.get("total_accepted", 0)),
            "total_rejected": summary.get("total_rejected", 0),
            "combined_metrics": combined,
            "per_algo_attribution": attribution,
        })
    except Exception as e:
        return (name, {
            "success": False,
            "error": str(e),
            "traceback": traceback.format_exc(),
        })


def run_phase_c(portfolios: list, workers: int,
                output_dir: str = "results/strategy_sweep") -> dict:
    """Phase C: run orchestrated portfolio variants sequentially.

    Each orchestration uses its own internal multiprocessing Pool for Phase 1,
    so we run them sequentially to avoid nested daemonic process errors.
    """
    print("=" * 100)
    print("PHASE C: Orchestrated Portfolio Optimization")
    print(f"  Portfolio variants: {len(portfolios)}")
    print(f"  (sequential — each orchestration uses internal Pool({workers}))")
    print("=" * 100)
    print()

    for name, path in portfolios:
        print(f"  {name:<30} {path}")
    print()

    if not portfolios:
        print("  No portfolios to run (Phase B yielded no qualified configs).")
        return {}

    results = []
    for i, (name, path) in enumerate(portfolios, 1):
        print(f"[{i}/{len(portfolios)}] Running portfolio: {name}...")
        result = _run_single_orchestration(name, path, output_dir)
        results.append(result)

    results_dict = dict(results)

    # Print summary
    print()
    print(f"{'Portfolio':<30} {'Trades':>7} {'WR%':>7} {'ROI%':>8} "
          f"{'Sharpe':>8} {'Net P&L':>12} {'MaxDD':>10} {'PF':>7}")
    print("-" * 100)
    for name, r in sorted(results_dict.items()):
        if not r.get("success"):
            print(f"  {name:<28} ERROR: {r.get('error', 'unknown')[:50]}")
        else:
            m = r.get("combined_metrics", {})
            print(f"  {name:<28} {r['total_accepted']:>7} "
                  f"{m.get('win_rate', 0):>6.1f}% "
                  f"{m.get('roi', 0):>7.1f}% "
                  f"{m.get('sharpe', 0):>7.2f} "
                  f"${m.get('net_pnl', 0):>10,.0f} "
                  f"${m.get('max_drawdown', 0):>8,.0f} "
                  f"{m.get('profit_factor', 0):>6.2f}")
    print()

    return results_dict


# ---------------------------------------------------------------------------
# Charts
# ---------------------------------------------------------------------------

def generate_charts(phase_a_df: pd.DataFrame, phase_b_df: pd.DataFrame,
                    phase_c_results: dict, chart_dir: Path):
    """Generate all charts for the sweep report."""
    chart_dir.mkdir(parents=True, exist_ok=True)

    if len(phase_a_df) == 0:
        return

    # 1. Net P&L by Strategy (sorted by P&L descending)
    fig, ax = plt.subplots(figsize=(12, 6))
    strat_pnl = phase_a_df.groupby("strategy")["net_pnl"].sum().sort_values(ascending=True)
    colors = ["#3fb950" if v >= 0 else "#f85149" for v in strat_pnl.values]
    strat_pnl.plot(kind="barh", ax=ax, color=colors)
    ax.set_title("Total Net P&L by Strategy", fontsize=14, fontweight="bold")
    ax.set_xlabel("Net P&L ($)")
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f"${x:,.0f}"))
    ax.grid(True, axis="x", alpha=0.3)
    plt.tight_layout()
    fig.savefig(chart_dir / "pnl_by_strategy.png", dpi=150)
    plt.close(fig)

    # 2. Net P&L by Ticker (sorted by P&L descending)
    fig, ax = plt.subplots(figsize=(10, 6))
    ticker_pnl = phase_a_df.groupby("ticker")["net_pnl"].sum().sort_values(ascending=True)
    colors = ["#3fb950" if v >= 0 else "#f85149" for v in ticker_pnl.values]
    ticker_pnl.plot(kind="barh", ax=ax, color=colors)
    ax.set_title("Total Net P&L by Ticker", fontsize=14, fontweight="bold")
    ax.set_xlabel("Net P&L ($)")
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f"${x:,.0f}"))
    ax.grid(True, axis="x", alpha=0.3)
    plt.tight_layout()
    fig.savefig(chart_dir / "pnl_by_ticker.png", dpi=150)
    plt.close(fig)

    # 3. Sharpe by Strategy × Ticker heatmap
    if len(phase_a_df) > 1:
        fig, ax = plt.subplots(figsize=(12, 8))
        pivot = phase_a_df.pivot_table(
            values="sharpe", index="strategy", columns="ticker",
            aggfunc="mean"
        )
        if pivot.shape[0] > 0 and pivot.shape[1] > 0:
            im = ax.imshow(pivot.values, cmap="RdYlGn", aspect="auto")
            ax.set_xticks(range(len(pivot.columns)))
            ax.set_xticklabels(pivot.columns)
            ax.set_yticks(range(len(pivot.index)))
            ax.set_yticklabels(pivot.index, fontsize=8)
            for i in range(len(pivot.index)):
                for j in range(len(pivot.columns)):
                    val = pivot.values[i, j]
                    if not np.isnan(val):
                        ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                                fontsize=8)
            plt.colorbar(im, ax=ax, label="Sharpe Ratio")
            ax.set_title("Sharpe Ratio: Strategy x Ticker", fontsize=14,
                          fontweight="bold")
        plt.tight_layout()
        fig.savefig(chart_dir / "sharpe_strategy_ticker.png", dpi=150)
        plt.close(fig)

    # 4. Portfolio comparison (Phase C) — sorted by Net P&L descending
    if phase_c_results:
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        # Collect and sort by net P&L descending
        portfolio_data = []
        for name, r in phase_c_results.items():
            if r.get("success"):
                m = r.get("combined_metrics", {})
                portfolio_data.append((
                    name, m.get("roi", 0), m.get("sharpe", 0),
                    m.get("net_pnl", 0),
                ))
        portfolio_data.sort(key=lambda x: x[3], reverse=True)
        names = [d[0].replace("_", "\n") for d in portfolio_data]
        roi_vals = [d[1] for d in portfolio_data]
        sharpe_vals = [d[2] for d in portfolio_data]
        pnl_vals = [d[3] for d in portfolio_data]

        if names:
            x = range(len(names))
            colors_roi = ["#58a6ff" if v >= 0 else "#f85149" for v in roi_vals]
            colors_sharpe = ["#58a6ff" if v >= 0 else "#f85149" for v in sharpe_vals]
            colors_pnl = ["#3fb950" if v >= 0 else "#f85149" for v in pnl_vals]

            axes[0].bar(x, roi_vals, color=colors_roi)
            axes[0].set_xticks(x)
            axes[0].set_xticklabels(names, fontsize=8)
            axes[0].set_title("ROI (%)", fontweight="bold")
            axes[0].grid(True, axis="y", alpha=0.3)

            axes[1].bar(x, sharpe_vals, color=colors_sharpe)
            axes[1].set_xticks(x)
            axes[1].set_xticklabels(names, fontsize=8)
            axes[1].set_title("Sharpe Ratio", fontweight="bold")
            axes[1].grid(True, axis="y", alpha=0.3)

            axes[2].bar(x, pnl_vals, color=colors_pnl)
            axes[2].set_xticks(x)
            axes[2].set_xticklabels(names, fontsize=8)
            axes[2].set_title("Net P&L ($)", fontweight="bold")
            axes[2].grid(True, axis="y", alpha=0.3)
            axes[2].yaxis.set_major_formatter(
                plt.FuncFormatter(lambda x, p: f"${x:,.0f}"))

        plt.suptitle("Portfolio Comparison", fontsize=16, fontweight="bold")
        plt.tight_layout()
        fig.savefig(chart_dir / "portfolio_comparison.png", dpi=150)
        plt.close(fig)

    # 5. Phase B ranked configs — sorted by Net P&L
    if len(phase_b_df) > 0:
        sorted_b = phase_b_df.sort_values("net_pnl", ascending=True)
        fig, ax = plt.subplots(figsize=(14, 7))
        labels = [f"{r['label']}" for _, r in sorted_b.iterrows()]
        pnl_values = sorted_b["net_pnl"].values
        colors = ["#3fb950" if v >= 0 else "#f85149" for v in pnl_values]
        ax.barh(range(len(labels)), pnl_values, color=colors)
        ax.set_yticks(range(len(labels)))
        ax.set_yticklabels(labels, fontsize=8)
        ax.set_xlabel("Net P&L ($)")
        ax.xaxis.set_major_formatter(
            plt.FuncFormatter(lambda x, p: f"${x:,.0f}"))
        ax.set_title("Top Configs by Net P&L (Phase B)", fontsize=14,
                      fontweight="bold")
        ax.grid(True, axis="x", alpha=0.3)
        plt.tight_layout()
        fig.savefig(chart_dir / "top_configs_ranked.png", dpi=150)
        plt.close(fig)

    print(f"Charts saved to {chart_dir}/")


# ---------------------------------------------------------------------------
# HTML Report
# ---------------------------------------------------------------------------

def generate_html_report(phase_a_df: pd.DataFrame, phase_b_df: pd.DataFrame,
                         phase_c_results: dict, output_dir: str,
                         start_date: str, end_date: str, daily_budget: int):
    """Generate comprehensive HTML report."""
    today = date.today().isoformat()
    report_path = os.path.join(output_dir, f"report_sweep_{today}.html")

    # --- Phase A table rows --- sorted by net P&L descending
    phase_a_rows = ""
    for _, r in phase_a_df.sort_values("net_pnl", ascending=False).iterrows():
        phase_a_rows += f"""<tr>
            <td>{r['label']}</td>
            <td>{r['strategy']}</td>
            <td>{r['ticker']}</td>
            <td>{int(r['dte'])}</td>
            <td>{int(r['total_trades'])}</td>
            <td>{r['win_rate']:.1f}%</td>
            <td>{r['roi']:.1f}%</td>
            <td>{r['sharpe']:.2f}</td>
            <td>${r['net_pnl']:,.0f}</td>
            <td>${r['max_drawdown']:,.0f}</td>
            <td>{r['profit_factor']:.2f}</td>
            <td>{r['avg_credit_risk_ratio']:.4f}</td>
        </tr>"""

    # --- Phase B table rows ---
    phase_b_rows = ""
    for _, r in phase_b_df.iterrows():
        phase_b_rows += f"""<tr>
            <td style="font-weight:bold">{int(r['rank'])}</td>
            <td>{r['label']}</td>
            <td>{r['strategy']}</td>
            <td>{r['ticker']}</td>
            <td>{int(r['dte'])}</td>
            <td>{int(r['total_trades'])}</td>
            <td>{r['win_rate']:.1f}%</td>
            <td>{r['roi']:.1f}%</td>
            <td>{r['sharpe']:.2f}</td>
            <td>{r['normalized_roi']:.4f}</td>
            <td>{r['composite_score']:.3f}</td>
        </tr>"""

    # --- Phase C table rows ---
    # Best portfolio = highest net P&L (budget utilization matters more than ROI
    # for a $500K/day deployment target)
    phase_c_rows = ""
    best_portfolio = None
    best_pnl = -999
    for name, r in phase_c_results.items():
        if not r.get("success"):
            continue
        m = r.get("combined_metrics", {})
        pnl = m.get("net_pnl", 0)
        if pnl > best_pnl:
            best_pnl = pnl
            best_portfolio = name

    # Sort portfolios by net P&L descending
    sorted_portfolios = sorted(
        phase_c_results.items(),
        key=lambda x: x[1].get("combined_metrics", {}).get("net_pnl", 0)
        if x[1].get("success") else -1,
        reverse=True,
    )
    for name, r in sorted_portfolios:
        if not r.get("success"):
            phase_c_rows += f"""<tr>
                <td>{name}</td>
                <td colspan="7" style="color:#f85149">ERROR: {r.get('error', 'unknown')[:80]}</td>
            </tr>"""
            continue
        m = r.get("combined_metrics", {})
        style = (' style="background:#1a2332;font-weight:bold"'
                 if name == best_portfolio else '')
        phase_c_rows += f"""<tr{style}>
            <td>{name}</td>
            <td>{m.get('total_trades', r.get('total_accepted', 0))}</td>
            <td>{m.get('win_rate', 0):.1f}%</td>
            <td>{m.get('roi', 0):.1f}%</td>
            <td>{m.get('sharpe', 0):.2f}</td>
            <td>${m.get('net_pnl', 0):,.0f}</td>
            <td>${m.get('max_drawdown', 0):,.0f}</td>
            <td>{m.get('profit_factor', 0):.2f}</td>
        </tr>"""

    # --- Attribution rows (from best portfolio) ---
    attribution_rows = ""
    if best_portfolio and phase_c_results.get(best_portfolio, {}).get("success"):
        attr = phase_c_results[best_portfolio].get("per_algo_attribution", {})
        for iid, data in attr.items():
            m = data.get("metrics", {})
            attribution_rows += f"""<tr>
                <td>{iid}</td>
                <td>{data.get('algo_name', '')}</td>
                <td>{data.get('ticker', '')}</td>
                <td>{data.get('trades', 0)}</td>
                <td>{m.get('win_rate', 0):.1f}%</td>
                <td>${m.get('net_pnl', 0):,.0f}</td>
            </tr>"""

    # --- Recommendations ---
    recommendations = ""
    if best_portfolio and phase_c_results.get(best_portfolio, {}).get("success"):
        bp = phase_c_results[best_portfolio]
        bm = bp.get("combined_metrics", {})
        # Find highest ROI portfolio for comparison
        roi_best = max(
            ((n, r) for n, r in phase_c_results.items() if r.get("success")),
            key=lambda x: x[1].get("combined_metrics", {}).get("roi", 0),
        )
        roi_name, roi_data = roi_best
        roi_m = roi_data.get("combined_metrics", {})
        recommendations = f"""
        <div class="section">
        <h2>Recommendations</h2>
        <ol>
            <li><strong>Best portfolio for ${daily_budget:,}/day deployment:
                <code>{best_portfolio}</code></strong> &mdash;
                {bm.get('total_trades', 0):,} trades,
                {bm.get('win_rate', 0):.1f}% win rate,
                {bm.get('roi', 0):.1f}% ROI,
                ${bm.get('net_pnl', 0):,.0f} net P&L,
                Sharpe {bm.get('sharpe', 0):.2f},
                max drawdown ${bm.get('max_drawdown', 0):,.0f}.</li>
            <li><strong>Highest ROI (capital efficiency):
                <code>{roi_name}</code></strong> &mdash;
                {roi_m.get('roi', 0):.1f}% ROI,
                ${roi_m.get('net_pnl', 0):,.0f} net P&L
                ({roi_m.get('total_trades', 0)} trades). Best for
                smaller capital allocation.</li>
            <li>Deploy ${daily_budget:,}/day using the
                <code>{best_portfolio}</code> orchestration config at
                <code>results/strategy_sweep/phase_c_configs/{best_portfolio}.yaml</code>.</li>
            <li>The orchestration YAML is ready for
                <code>run_orchestrated_backtest.py</code> or
                <code>run_live_advisor.py</code>.</li>
        </ol>
        </div>"""

    # --- KPI strip values ---
    total_configs = len(phase_a_df)
    qualified = len(phase_b_df)
    best_wr = phase_b_df["win_rate"].max() if len(phase_b_df) > 0 else 0
    best_sharpe = phase_b_df["sharpe"].max() if len(phase_b_df) > 0 else 0

    html = f"""<!DOCTYPE html>
<html><head>
<title>Strategy Sweep Report — {today}</title>
<style>
body {{ background: #0d1117; color: #c9d1d9; font-family: -apple-system, BlinkMacSystemFont, sans-serif; margin: 0; padding: 20px; }}
.hero {{ background: linear-gradient(135deg, #161b22, #0d1117); border: 1px solid #30363d; border-radius: 12px; padding: 30px; margin-bottom: 24px; text-align: center; }}
.hero h1 {{ color: #58a6ff; margin: 0; font-size: 28px; }}
.hero .subtitle {{ color: #8b949e; margin-top: 8px; }}
.kpi-strip {{ display: flex; gap: 16px; margin-bottom: 24px; flex-wrap: wrap; }}
.kpi {{ background: #161b22; border: 1px solid #30363d; border-radius: 8px; padding: 16px; flex: 1; min-width: 120px; text-align: center; }}
.kpi .value {{ font-size: 24px; font-weight: bold; color: #58a6ff; }}
.kpi .label {{ color: #8b949e; font-size: 12px; margin-top: 4px; }}
table {{ width: 100%; border-collapse: collapse; margin-bottom: 24px; background: #161b22; border-radius: 8px; overflow: hidden; }}
th {{ background: #21262d; color: #8b949e; padding: 12px; text-align: left; font-size: 12px; text-transform: uppercase; }}
td {{ padding: 10px 12px; border-bottom: 1px solid #21262d; font-size: 13px; }}
h2 {{ color: #58a6ff; border-bottom: 1px solid #30363d; padding-bottom: 8px; }}
h3 {{ color: #c9d1d9; }}
.section {{ background: #161b22; border: 1px solid #30363d; border-radius: 8px; padding: 20px; margin-bottom: 24px; }}
.chart-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(500px, 1fr)); gap: 16px; }}
.chart-card {{ background: #161b22; border: 1px solid #30363d; border-radius: 8px; padding: 16px; }}
.chart-card img {{ width: 100%; border-radius: 4px; }}
.chart-card p {{ color: #8b949e; font-size: 13px; margin-top: 8px; }}
ol li {{ margin-bottom: 8px; }}
</style></head><body>

<div class="hero">
    <h1>$500K/Day Optimal Deployment Sweep</h1>
    <div class="subtitle">Strategy × Ticker × DTE Profiling &mdash; {start_date} to {end_date} &mdash; {today}</div>
</div>

<div class="kpi-strip">
    <div class="kpi"><div class="value">{total_configs}</div><div class="label">Configs Tested</div></div>
    <div class="kpi"><div class="value">{qualified}</div><div class="label">Qualified</div></div>
    <div class="kpi"><div class="value">{len(phase_c_results)}</div><div class="label">Portfolios</div></div>
    <div class="kpi"><div class="value">{best_wr:.1f}%</div><div class="label">Best Win Rate</div></div>
    <div class="kpi"><div class="value">{best_sharpe:.2f}</div><div class="label">Best Sharpe</div></div>
    <div class="kpi"><div class="value">${daily_budget:,}</div><div class="label">Daily Budget</div></div>
</div>

<div class="section">
<h2>Phase A: Individual Strategy Profiling ({total_configs} configs)</h2>
<p>Every viable strategy × ticker × DTE combination run as a standalone backtest.
   Sorted by ROI descending.</p>
<div style="overflow-x:auto">
<table>
<tr><th>Config</th><th>Strategy</th><th>Ticker</th><th>DTE</th><th>Trades</th>
    <th>Win Rate</th><th>ROI</th><th>Sharpe</th><th>Net P&L</th>
    <th>Max DD</th><th>PF</th><th>Credit/Risk</th></tr>
{phase_a_rows}
</table>
</div>
</div>

<div class="section">
<h2>Phase B: Ranked Top Configs ({qualified} qualified)</h2>
<p>Scored by normalized ROI = (credit/max_loss) / max(1, DTE). Filtered by
   win rate &ge; 80%, trades &ge; 20, profit factor &ge; 2.0. Composite score blends
   normalized ROI (50%), Sharpe (30%), win rate (20%).</p>
<table>
<tr><th>Rank</th><th>Config</th><th>Strategy</th><th>Ticker</th><th>DTE</th>
    <th>Trades</th><th>Win Rate</th><th>ROI</th><th>Sharpe</th>
    <th>Norm ROI</th><th>Score</th></tr>
{phase_b_rows}
</table>
</div>

<div class="section">
<h2>Phase C: Portfolio Comparison</h2>
<p>Top configs composed into portfolio variants and run through the orchestrator
   with ${daily_budget:,}/day budget, 15-min intervals, best_score selection.</p>
<table>
<tr><th>Portfolio</th><th>Trades</th><th>Win Rate</th><th>ROI</th><th>Sharpe</th>
    <th>Net P&L</th><th>Max DD</th><th>PF</th></tr>
{phase_c_rows}
</table>
</div>

{"" if not attribution_rows else f'''
<div class="section">
<h2>Best Portfolio Attribution: {best_portfolio}</h2>
<p>Which algo contributed which trades to the winning portfolio.</p>
<table>
<tr><th>Instance</th><th>Algo</th><th>Ticker</th><th>Trades</th><th>Win Rate</th><th>Net P&L</th></tr>
{attribution_rows}
</table>
</div>
'''}

{recommendations}

<div class="section">
<h2>Charts</h2>
<div class="chart-grid">
    <div class="chart-card">
        <h3>Net P&amp;L by Strategy</h3>
        <img src="charts/pnl_by_strategy.png" alt="P&amp;L by Strategy">
        <p>Total net P&amp;L across all ticker/DTE combinations for each strategy.
           Shows which strategies generated the most absolute profit over the backtest period.</p>
    </div>
    <div class="chart-card">
        <h3>Net P&amp;L by Ticker</h3>
        <img src="charts/pnl_by_ticker.png" alt="P&amp;L by Ticker">
        <p>Total net P&amp;L by underlying ticker. Shows which tickers generated
           the most profit and where capital should be concentrated.</p>
    </div>
    <div class="chart-card">
        <h3>Sharpe: Strategy × Ticker</h3>
        <img src="charts/sharpe_strategy_ticker.png" alt="Sharpe Heatmap">
        <p>Sharpe ratio heatmap showing which strategy/ticker combinations
           deliver the most consistent risk-adjusted returns.</p>
    </div>
    <div class="chart-card">
        <h3>Top Configs Ranked</h3>
        <img src="charts/top_configs_ranked.png" alt="Top Configs">
        <p>Phase B composite scores for the top qualified configs. Blends
           normalized ROI (50%), Sharpe (30%), and win rate (20%).</p>
    </div>
    <div class="chart-card">
        <h3>Portfolio Comparison</h3>
        <img src="charts/portfolio_comparison.png" alt="Portfolio Comparison">
        <p>Side-by-side comparison of all portfolio variants across ROI,
           Sharpe ratio, and absolute net P&L.</p>
    </div>
</div>
</div>

<div class="section">
<h2>Methodology</h2>
<table>
<tr><th>Parameter</th><th>Value</th></tr>
<tr><td>Backtest Period</td><td>{start_date} to {end_date}</td></tr>
<tr><td>Daily Budget</td><td>${daily_budget:,}</td></tr>
<tr><td>Phase A Workers</td><td>8 (capped)</td></tr>
<tr><td>Phase B Filters</td><td>WR &ge; 80%, trades &ge; 20, PF &ge; 2.0</td></tr>
<tr><td>Phase B Scoring</td><td>50% norm ROI + 30% Sharpe + 20% WR</td></tr>
<tr><td>Phase C Mode</td><td>Interval (15-min), best_score selection</td></tr>
<tr><td>Phase C Scoring Weights</td><td>[0.80, 0.10, 0.10] (ROI-dominant)</td></tr>
<tr><td>Strategies Tested</td><td>percentile_entry, iv_regime_condor, weekly_iron_condor, tail_hedged, zero_dte, tqqq_momentum_scalper</td></tr>
<tr><td>Tickers</td><td>NDX, SPX, RUT, TQQQ</td></tr>
</table>
</div>

<div class="section" style="text-align:center;color:#8b949e">
Generated with <a href="https://claude.com/claude-code" style="color:#58a6ff">Claude Code</a>
</div>

</body></html>"""

    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    with open(report_path, "w") as f:
        f.write(html)

    # Create/update index.html symlink pointing to the latest report
    report_name = os.path.basename(report_path)
    index_path = os.path.join(output_dir, "index.html")
    if os.path.exists(index_path) or os.path.islink(index_path):
        os.unlink(index_path)
    os.symlink(report_name, index_path)
    print(f"index.html -> {report_name}")

    return report_path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="""
Optimal $500K/Day Deployment — Strategy/Ticker/Timing Sweep.

Three-phase sweep to find the best way to deploy $500K/day across strategies,
tickers, DTEs, and time windows:

  Phase A: Profile every viable strategy × ticker × DTE combo independently
           (~31 configs, parallel via multiprocessing).
  Phase B: Score by normalized ROI, filter by quality, rank top N.
  Phase C: Compose top configs into portfolio variants and run through the
           orchestrator to find the optimal mix.

Output: ranked comparison report with charts and deployment recommendations.
        """,
        epilog="""
Examples:
  %(prog)s --start-date 2025-06-13 --end-date 2026-03-13
      Full sweep with default $500K budget

  %(prog)s --start-date 2025-06-13 --end-date 2026-03-13 --daily-budget 300000
      Full sweep with $300K/day budget

  %(prog)s --dry-run
      Preview all configs without running any backtests

  %(prog)s --analyze
      Skip Phase A backtests, re-analyze existing phase_a_results.csv

  %(prog)s --phase-a-only
      Run Phase A only, skip orchestrated portfolio optimization

  %(prog)s --workers 4
      Limit parallelism to 4 workers
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument("--start-date", default="2025-06-13",
                        help="Backtest start date (default: 2025-06-13)")
    parser.add_argument("--end-date", default="2026-03-13",
                        help="Backtest end date (default: 2026-03-13)")
    parser.add_argument("--daily-budget", type=int, default=500000,
                        help="Daily deployment budget in dollars (default: 500000)")
    parser.add_argument("--output-dir", default="results/strategy_sweep",
                        help="Output directory (default: results/strategy_sweep)")
    parser.add_argument("--workers", type=int, default=min(8, cpu_count()),
                        help=f"Number of parallel workers (default: {min(8, cpu_count())})")
    parser.add_argument("--top-n", type=int, default=15,
                        help="Number of top configs to select in Phase B (default: 15)")
    parser.add_argument("--min-win-rate", type=float, default=80.0,
                        help="Minimum win rate filter for Phase B (default: 80.0)")
    parser.add_argument("--min-trades", type=int, default=20,
                        help="Minimum trade count filter for Phase B (default: 20)")
    parser.add_argument("--min-profit-factor", type=float, default=2.0,
                        help="Minimum profit factor filter for Phase B (default: 2.0)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Preview configs without running backtests")
    parser.add_argument("--analyze", action="store_true",
                        help="Skip Phase A, re-analyze existing phase_a_results.csv")
    parser.add_argument("--phase-a-only", action="store_true",
                        help="Run Phase A only, skip Phase C orchestration")
    parser.add_argument("--report-only", action="store_true",
                        help="Regenerate report and charts from existing results "
                             "(no backtests, no orchestration)")

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    chart_dir = output_dir / "charts"

    # Build configs
    configs = _build_sweep_configs(args.start_date, args.end_date)

    print()
    print("=" * 100)
    print(f"  STRATEGY SWEEP: Optimal ${args.daily_budget:,}/Day Deployment")
    print(f"  Period: {args.start_date} to {args.end_date}")
    print(f"  Configs: {len(configs)} | Workers: {args.workers}")
    print(f"  Output: {output_dir}")
    print("=" * 100)
    print()

    # --- Dry run ---
    if args.dry_run:
        print("DRY RUN — showing all configs that would execute:\n")
        for i, (label, cfg) in enumerate(configs, 1):
            strategy = cfg["strategy"]["name"]
            ticker = cfg["infra"]["ticker"]
            dte = cfg["strategy"]["params"].get("dte", "?")
            window = f"{cfg['constraints']['trading_hours']['entry_start']}-{cfg['constraints']['trading_hours']['entry_end']}"
            print(f"  [{i:>2}] {label:<40} {strategy:<35} {ticker:<6} "
                  f"DTE={dte} {window}")
        print(f"\nTotal: {len(configs)} configs")
        print(f"Estimated runtime: ~{len(configs) // args.workers * 30} min "
              f"({len(configs)} configs / {args.workers} workers)")
        return

    # --- Phase A ---
    if args.analyze:
        csv_path = output_dir / "phase_a_results.csv"
        if csv_path.exists():
            print(f"Loading existing Phase A results from {csv_path}")
            phase_a_df = pd.read_csv(csv_path)
        else:
            print(f"ERROR: {csv_path} not found. Run without --analyze first.")
            sys.exit(1)
    else:
        phase_a_df = run_phase_a(configs, args.workers)
        if len(phase_a_df) > 0:
            csv_path = output_dir / "phase_a_results.csv"
            phase_a_df.to_csv(csv_path, index=False)
            print(f"Phase A results saved to {csv_path}")

    # --- Phase B ---
    phase_b_df = run_phase_b(
        phase_a_df,
        min_win_rate=args.min_win_rate,
        min_trades=args.min_trades,
        min_profit_factor=args.min_profit_factor,
        top_n=args.top_n,
    )
    if len(phase_b_df) > 0:
        csv_path = output_dir / "phase_b_ranked.csv"
        phase_b_df.to_csv(csv_path, index=False)
        print(f"Phase B results saved to {csv_path}")

    # --- Phase C ---
    phase_c_results = {}
    if args.report_only:
        # Load existing Phase C results from per-instance CSVs
        pc_dir = output_dir / "phase_c_portfolios"
        if pc_dir.exists():
            for pname in sorted(os.listdir(pc_dir)):
                ppath = pc_dir / pname
                if ppath.is_dir() and (ppath / "per_instance").exists():
                    agg = _aggregate_per_instance_metrics(str(output_dir), pname)
                    if agg:
                        phase_c_results[pname] = {
                            "success": True,
                            "total_accepted": agg["combined_metrics"].get(
                                "total_trades", 0),
                            "combined_metrics": agg["combined_metrics"],
                            "per_algo_attribution": agg["per_algo_attribution"],
                        }
            print(f"Loaded {len(phase_c_results)} portfolio results from "
                  f"existing per-instance data")
    elif not args.phase_a_only:
        portfolios = _build_portfolio_configs(
            phase_b_df, phase_a_df,
            args.start_date, args.end_date,
            args.daily_budget,
            str(output_dir),
        )
        if portfolios:
            phase_c_results = run_phase_c(portfolios, args.workers,
                                              str(output_dir))
    else:
        print("Skipping Phase C (--phase-a-only)")

    # --- Charts ---
    generate_charts(phase_a_df, phase_b_df, phase_c_results, chart_dir)

    # --- HTML Report ---
    report_path = generate_html_report(
        phase_a_df, phase_b_df, phase_c_results,
        str(output_dir), args.start_date, args.end_date, args.daily_budget,
    )
    print(f"\nHTML report: {report_path}")
    print("\nDone!")


if __name__ == "__main__":
    main()
