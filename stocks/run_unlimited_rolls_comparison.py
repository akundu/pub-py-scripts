#!/usr/bin/env python3
"""Compare max_rolls=2 vs unlimited rolls for NDX credit spreads.

Budget: $300K daily, $30K per transaction, both put+call sides.
"""

import sys
import os
import copy
import yaml
from pathlib import Path
from multiprocessing import Pool

BASE_DIR = Path(__file__).resolve().parent


def make_config(max_rolls: int, label: str) -> dict:
    """Create config dict with specified max_rolls."""
    return {
        "infra": {
            "ticker": "NDX",
            "start_date": "2025-03-01",
            "end_date": "2026-02-28",
            "lookback_days": 180,
            "num_processes": 1,
            "output_dir": f"results/rolls_comparison/{label}",
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
                "stop_loss_multiplier": 3.0,
                "roll_enabled": True,
                "roll_check_start_utc": "18:00",
                "early_itm_check_utc": "14:00",
                "max_move_cap": 150,
                "roll_percentile": 90,
                "roll_min_dte": 3,
                "roll_max_dte": 10,
                "max_roll_width": 50,
                "max_rolls": max_rolls,
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
                "stop_loss_pct": 3.0,
                "mode": "first_triggered",
            },
        },
        "report": {
            "formats": ["console", "csv"],
            "metrics": ["win_rate", "roi", "sharpe", "max_drawdown", "profit_factor"],
        },
    }


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

    # Write temp config
    config_path = BASE_DIR / f"_tmp_config_{label}.yaml"
    with open(config_path, "w") as f:
        yaml.dump(config_dict, f)

    try:
        config = BacktestConfig.from_yaml(str(config_path))
        engine = BacktestEngine(config)
        results = engine.run()

        # Extract key metrics
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
        ("max_rolls_2", make_config(max_rolls=2, label="max_rolls_2")),
        ("unlimited_rolls", make_config(max_rolls=999, label="unlimited_rolls")),
    ]

    print("=" * 80)
    print("NDX Credit Spread: max_rolls=2 vs Unlimited Rolls Comparison")
    print(f"Budget: $300K daily, $30K per transaction, both put+call")
    print(f"Period: 2025-03-01 to 2026-02-28 (1 year)")
    print("=" * 80)
    print()

    with Pool(processes=2) as pool:
        results = pool.map(run_single, configs)

    results_dict = dict(results)

    # Display comparison
    print()
    print("=" * 80)
    print(f"{'Metric':<25} {'max_rolls=2':>18} {'unlimited_rolls':>18} {'Delta':>15}")
    print("-" * 80)

    r2 = results_dict.get("max_rolls_2", {})
    ru = results_dict.get("unlimited_rolls", {})

    if "error" in r2 or "error" in ru:
        print(f"ERROR: max_rolls_2={r2}, unlimited={ru}")
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
        v2 = r2.get(key, 0)
        vu = ru.get(key, 0)
        delta = vu - v2

        v2_str = f"{v2:{fmt}}"
        vu_str = f"{vu:{fmt}}"

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

        print(f"  {name:<23} {v2_str:>18} {vu_str:>18} {delta_str:>15}")

    print("=" * 80)


if __name__ == "__main__":
    main()
