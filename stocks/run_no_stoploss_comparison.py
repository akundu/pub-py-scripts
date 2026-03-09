#!/usr/bin/env python3
"""Compare with stop loss vs no stop loss for NDX credit spreads.

Budget: $300K daily, $30K per transaction, both put+call.
"""

import sys
import os
import yaml
from pathlib import Path
from multiprocessing import Pool

BASE_DIR = Path(__file__).resolve().parent


def make_config(stop_loss_mult: float, label: str) -> dict:
    """Create config dict. stop_loss_mult=0 means disabled."""
    cfg = {
        "infra": {
            "ticker": "NDX",
            "start_date": "2025-03-01",
            "end_date": "2026-02-28",
            "lookback_days": 180,
            "num_processes": 1,
            "output_dir": f"results/stoploss_comparison/{label}",
        },
        "providers": [
            {"name": "csv_equity", "role": "equity", "params": {"csv_dir": "equities_output"}},
            {"name": "csv_options", "role": "options", "params": {
                "csv_dir": "options_csv_output_full",
                "dte_buckets": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            }},
        ],
        "strategy": {
            "name": "percentile_entry_credit_spread",
            "params": {
                "dte": 0,
                "percentile": 95,
                "lookback": 120,
                "option_types": ["put", "call"],
                "spread_width": 50,
                "interval_minutes": 10,
                "entry_start_utc": "13:00",
                "entry_end_utc": "17:00",
                "num_contracts": 1,
                "max_loss_estimate": 10000,
                "profit_target_0dte": 0.75,
                "profit_target_multiday": 0.50,
                "min_roi_per_day": 0.025,
                "min_credit": 0.75,
                "min_total_credit": 0,
                "min_credit_per_point": 0,
                "max_contracts": 0,
                "stop_loss_multiplier": stop_loss_mult,
                "roll_enabled": True,
                "roll_check_start_utc": "18:00",
                "early_itm_check_utc": "14:00",
                "max_move_cap": 150,
                "roll_percentile": 90,
                "roll_min_dte": 3,
                "roll_max_dte": 10,
                "max_roll_width": 50,
                "max_rolls": 2,
            },
        },
        "constraints": {
            "budget": {
                "max_spend_per_transaction": 30000,
                "daily_budget": 300000,
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
            "formats": ["console", "csv"],
            "metrics": ["win_rate", "roi", "sharpe", "max_drawdown", "profit_factor"],
        },
    }
    # Only add stop_loss_pct to exit_rules if we want it
    if stop_loss_mult > 0:
        cfg["constraints"]["exit_rules"]["stop_loss_pct"] = stop_loss_mult
        cfg["strategy"]["params"]["stop_loss_multiplier"] = stop_loss_mult
    else:
        # Explicitly remove stop loss
        cfg["strategy"]["params"]["stop_loss_multiplier"] = 0
    return cfg


def run_single(args):
    """Run a single backtest config."""
    label, config_dict = args

    sys.path.insert(0, str(BASE_DIR))
    os.chdir(str(BASE_DIR))

    import scripts.backtesting.providers.csv_equity_provider   # noqa
    import scripts.backtesting.providers.csv_options_provider   # noqa
    import scripts.backtesting.instruments.credit_spread        # noqa
    import scripts.backtesting.strategies.credit_spread.percentile_entry  # noqa

    from scripts.backtesting.config import BacktestConfig
    from scripts.backtesting.engine import BacktestEngine

    config_path = BASE_DIR / f"_tmp_config_{label}.yaml"
    with open(config_path, "w") as f:
        yaml.dump(config_dict, f)

    try:
        config = BacktestConfig.from_yaml(str(config_path))
        engine = BacktestEngine(config)
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


def main():
    configs = [
        ("with_stoploss_3x", make_config(stop_loss_mult=3.0, label="with_stoploss_3x")),
        ("no_stoploss", make_config(stop_loss_mult=0, label="no_stoploss")),
    ]

    print("=" * 80)
    print("NDX Credit Spread: Stop Loss 3x vs No Stop Loss")
    print(f"Budget: $300K daily, $30K per transaction, both put+call")
    print(f"Period: 2025-03-01 to 2026-02-28 (1 year)")
    print(f"P95 entry, 0DTE, 50pt spreads, rolling enabled (max 2)")
    print("=" * 80)
    print()

    with Pool(processes=2) as pool:
        results = pool.map(run_single, configs)

    results_dict = dict(results)

    print()
    print("=" * 80)
    print(f"{'Metric':<25} {'3x Stop Loss':>18} {'No Stop Loss':>18} {'Delta':>15}")
    print("-" * 80)

    r_sl = results_dict.get("with_stoploss_3x", {})
    r_no = results_dict.get("no_stoploss", {})

    if "error" in r_sl or "error" in r_no:
        print(f"ERROR: with_sl={r_sl}, no_sl={r_no}")
        return

    metrics = [
        ("Total trades", "total_trades", "d"),
        ("Wins", "wins", "d"),
        ("Losses", "losses", "d"),
        ("Win rate (%)", "win_rate", ".1f"),
        ("Net P&L ($)", "net_pnl", ",.2f"),
        ("Total credits ($)", "total_credits", ",.2f"),
        ("Total gains ($)", "total_gains", ",.2f"),
        ("Total losses ($)", "total_losses", ",.2f"),
        ("Avg P&L/trade ($)", "avg_pnl", ",.2f"),
        ("ROI (%)", "roi", ".1f"),
        ("Sharpe ratio", "sharpe", ".2f"),
        ("Max drawdown ($)", "max_drawdown", ",.2f"),
        ("Profit factor", "profit_factor", ".2f"),
    ]

    for name, key, fmt in metrics:
        v_sl = r_sl.get(key, 0)
        v_no = r_no.get(key, 0)
        delta = v_no - v_sl

        v_sl_str = f"{v_sl:{fmt}}"
        v_no_str = f"{v_no:{fmt}}"

        if isinstance(delta, float):
            if abs(delta) < 0.01:
                delta_str = "—"
            else:
                sign = "+" if delta > 0 else ""
                delta_str = f"{sign}{delta:{fmt}}"
        else:
            if delta == 0:
                delta_str = "—"
            else:
                sign = "+" if delta > 0 else ""
                delta_str = f"{sign}{delta:{fmt}}"

        print(f"  {name:<23} {v_sl_str:>18} {v_no_str:>18} {delta_str:>15}")

    print("=" * 80)


if __name__ == "__main__":
    main()
