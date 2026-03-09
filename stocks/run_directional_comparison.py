#!/usr/bin/env python3
"""Compare directional entry modes: both sides vs momentum vs contrarian.

Momentum: price above prev_close → sell puts (bullish), below → sell calls (bearish)
Contrarian: opposite of momentum
Both: always enter put + call (current default)

Tests across DTE 0, 1, 2, 5, 10 at P80.
"""

import sys
import os
import yaml
from pathlib import Path
from multiprocessing import Pool

BASE_DIR = Path(__file__).resolve().parent


def _base_config(label: str, dte: int, directional: str) -> dict:
    return {
        "infra": {
            "ticker": "NDX",
            "start_date": "2025-03-01",
            "end_date": "2026-02-28",
            "lookback_days": 180,
            "num_processes": 1,
            "output_dir": f"results/directional/{label}",
        },
        "providers": [
            {"name": "csv_equity", "role": "equity", "params": {"csv_dir": "equities_output"}},
            {"name": "csv_options", "role": "options", "params": {
                "csv_dir": "options_csv_output_full",
                "dte_buckets": list(range(0, max(dte + 2, 5))),
            }},
        ],
        "strategy": {
            "name": "percentile_entry_credit_spread",
            "params": {
                "dte": dte,
                "percentile": 80,
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
                "directional_entry": directional,
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


def run_single(args):
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
    dtes = [0, 2, 5, 10]
    modes = ["both", "momentum", "contrarian", "contrarian_double"]

    configs = []
    for dte in dtes:
        for mode in modes:
            label = f"dte{dte}_{mode}"
            configs.append((label, _base_config(label, dte, mode)))

    print("=" * 120)
    print("NDX Credit Spread: Directional Entry Comparison")
    print(f"P80, 1 contract, 50pt spreads, no stop loss, no rolling")
    print(f"Period: 2025-03-01 to 2026-02-28 (1 year)")
    print()
    print("Modes:")
    print("  both:              Enter 1 put + 1 call at every interval (current default)")
    print("  momentum:          Price > prev close → sell puts only; price < prev close → sell calls only")
    print("  contrarian:        Price > prev close → sell calls only; price < prev close → sell puts only")
    print("  contrarian_double: Enter both sides, but 2 contracts on contrarian side (1 momentum + 2 contrarian)")
    print("=" * 120)
    print()

    with Pool(processes=min(8, len(configs))) as pool:
        results = pool.map(run_single, configs)

    results_dict = dict(results)

    metrics = [
        ("Total trades", "total_trades", "d"),
        ("Wins", "wins", "d"),
        ("Losses", "losses", "d"),
        ("Win rate (%)", "win_rate", ".1f"),
        ("Net P&L ($)", "net_pnl", ",.0f"),
        ("Total credits ($)", "total_credits", ",.0f"),
        ("Total gains ($)", "total_gains", ",.0f"),
        ("Total losses ($)", "total_losses", ",.0f"),
        ("Avg P&L/trade ($)", "avg_pnl", ",.0f"),
        ("ROI (%)", "roi", ".1f"),
        ("Sharpe ratio", "sharpe", ".2f"),
        ("Max drawdown ($)", "max_drawdown", ",.0f"),
        ("Profit factor", "profit_factor", ".2f"),
    ]

    for dte in dtes:
        print()
        print(f"{'='*100}")
        print(f"  DTE = {dte}")
        print(f"{'='*100}")
        header = f"  {'Metric':<22}"
        for mode in modes:
            header += f" {mode:>18}"
        print(header)
        print(f"  {'-'*96}")

        labels = [f"dte{dte}_{m}" for m in modes]
        for name, key, fmt in metrics:
            row = f"  {name:<22}"
            for lbl in labels:
                r = results_dict.get(lbl, {})
                if "error" in r:
                    row += f" {'ERROR':>18}"
                else:
                    v = r.get(key, 0)
                    row += f" {format(v, fmt):>18}"
            print(row)

    # Summary table across DTEs
    print()
    print("=" * 120)
    print("SUMMARY: Net P&L by DTE and Mode")
    print("-" * 120)
    header = f"  {'DTE':<6}"
    for mode in modes:
        header += f" {mode + ' P&L':>18} {mode + ' WR%':>10} {mode + ' Sharpe':>12}"
    print(header)
    print(f"  {'-'*114}")

    for dte in dtes:
        row = f"  {dte:<6}"
        for mode in modes:
            lbl = f"dte{dte}_{mode}"
            r = results_dict.get(lbl, {})
            if "error" in r:
                row += f" {'ERROR':>18} {'':>10} {'':>12}"
            else:
                row += f" ${r.get('net_pnl', 0):>16,.0f} {r.get('win_rate', 0):>9.1f}% {r.get('sharpe', 0):>11.2f}"
        print(row)

    print("=" * 120)


if __name__ == "__main__":
    main()
