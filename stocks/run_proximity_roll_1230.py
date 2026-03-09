#!/usr/bin/env python3
"""Dual-trigger proximity roll: proximity 0.5% after 12:30pm OR $20K loss threshold.

Budget: $400K daily, $75K max loss per transaction, $2.5K min credit per trade.
No stop loss, unlimited rolls.
"""

import sys
import os
import yaml
from pathlib import Path
from multiprocessing import Pool

BASE_DIR = Path(__file__).resolve().parent


def _base_config(label: str) -> dict:
    return {
        "infra": {
            "ticker": "NDX",
            "start_date": "2025-03-01",
            "end_date": "2026-02-28",
            "lookback_days": 180,
            "num_processes": 1,
            "output_dir": f"results/proximity_roll_1230/{label}",
        },
        "providers": [
            {"name": "csv_equity", "role": "equity", "params": {"csv_dir": "equities_output"}},
            {"name": "csv_options", "role": "options", "params": {
                "csv_dir": "options_csv_output_full",
                "dte_buckets": list(range(0, 31)),
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
                "max_loss_estimate": 75000,
                "profit_target_0dte": 0.75,
                "profit_target_multiday": 0.50,
                "min_roi_per_day": 0.025,
                "min_credit": 0.75,
                "min_total_credit": 2500,
                "min_credit_per_point": 0,
                "max_contracts": 0,
                "stop_loss_multiplier": 0,
                "roll_enabled": True,
                "roll_percentile": 90,
                "roll_min_dte": 3,
                "roll_max_dte": 30,
                "max_roll_width": 50,
                "max_rolls": 999,
                "use_proximity_roll": True,
                "proximity_pct": 0.01,
                "early_itm_check_utc": "23:59",
                "max_move_cap": 999999,
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
            "formats": ["console", "csv"],
            "metrics": ["win_rate", "roi", "sharpe", "max_drawdown", "profit_factor"],
        },
    }


def config_no_roll():
    """No stop loss, no rolling (hold to expiration)."""
    cfg = _base_config("no_roll")
    cfg["strategy"]["params"]["roll_enabled"] = False
    cfg["strategy"]["params"]["max_rolls"] = 0
    cfg["strategy"]["params"]["use_proximity_roll"] = False
    return cfg


def config_proximity_only():
    """0.5% proximity roll at 12:30pm PST, NO loss threshold."""
    cfg = _base_config("prox_only_05pct_1230pm")
    cfg["strategy"]["params"]["roll_check_start_utc"] = "19:30"
    cfg["strategy"]["params"]["proximity_pct"] = 0.005
    cfg["strategy"]["params"]["max_loss_trigger"] = 0  # disabled
    return cfg


def config_dual_20k():
    """0.5% proximity at 12:30pm + $20K loss threshold anytime."""
    cfg = _base_config("dual_05pct_20k")
    cfg["strategy"]["params"]["roll_check_start_utc"] = "19:30"
    cfg["strategy"]["params"]["proximity_pct"] = 0.005
    cfg["strategy"]["params"]["max_loss_trigger"] = 20000
    return cfg


def config_dual_10k():
    """0.5% proximity at 12:30pm + $10K loss threshold anytime."""
    cfg = _base_config("dual_05pct_10k")
    cfg["strategy"]["params"]["roll_check_start_utc"] = "19:30"
    cfg["strategy"]["params"]["proximity_pct"] = 0.005
    cfg["strategy"]["params"]["max_loss_trigger"] = 10000
    return cfg


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

    use_proximity = config_dict["strategy"]["params"].pop("use_proximity_roll", False)
    proximity_pct = config_dict["strategy"]["params"].pop("proximity_pct", 0.005)
    max_loss_trigger = config_dict["strategy"]["params"].pop("max_loss_trigger", 0)

    config_path = BASE_DIR / f"_tmp_config_{label}.yaml"
    with open(config_path, "w") as f:
        yaml.dump(config_dict, f)

    try:
        config = BacktestConfig.from_yaml(str(config_path))
        engine = BacktestEngine(config)

        if use_proximity:
            from scripts.backtesting.constraints.exit_rules.proximity_roll import ProximityRollExit
            max_rolls_val = config_dict["strategy"]["params"].get("max_rolls", 999)
            roll_check_start = config_dict["strategy"]["params"].get("roll_check_start_utc", "19:00")

            original_run = engine.run

            def patched_run(dry_run=False, _prox_pct=proximity_pct, _max_r=max_rolls_val, _rcs=roll_check_start, _mlt=max_loss_trigger):
                engine.provider = engine._build_providers()
                engine.constraints = engine._build_constraints()
                engine.exit_manager = engine._build_exit_manager()
                engine.collector = engine._build_collector()
                engine.strategy = engine._build_strategy()

                trading_dates = engine._resolve_dates(engine.config.infra.ticker)
                engine.strategy.setup()

                prox_exit = ProximityRollExit(
                    proximity_pct=_prox_pct,
                    roll_check_start_utc=_rcs,
                    max_rolls=_max_r,
                    max_loss_trigger=_mlt,
                )
                engine.strategy._roll_trigger = prox_exit
                if engine.strategy.exit_manager is not None:
                    new_rules = []
                    for rule in engine.strategy.exit_manager._rules:
                        if rule.name == "roll_trigger":
                            new_rules.append(prox_exit)
                        else:
                            new_rules.append(rule)
                    engine.strategy.exit_manager._rules = new_rules

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


def main():
    configs = [
        ("1_no_roll", config_no_roll()),
        ("2_prox_only", config_proximity_only()),
        ("3_dual_20k", config_dual_20k()),
        ("4_dual_10k", config_dual_10k()),
    ]

    print("=" * 105)
    print("NDX Credit Spread: Dual-Trigger Roll Comparison")
    print(f"Budget: $400K daily, $75K max loss/txn, $2.5K min credit/trade, no stop loss")
    print(f"Period: 2025-03-01 to 2026-02-28 (1 year), P95 entry, 0DTE, 50pt spreads")
    print()
    print("Configs:")
    print("  1) No roll:           Hold to expiration, no rolling")
    print("  2) Prox only:         0.5% proximity after 12:30pm PST, no loss threshold")
    print("  3) Dual 0.5%+$20K:    0.5% proximity after 12:30pm OR $20K loss anytime")
    print("  4) Dual 0.5%+$10K:    0.5% proximity after 12:30pm OR $10K loss anytime")
    print("=" * 105)
    print()

    with Pool(processes=4) as pool:
        results = pool.map(run_single, configs)

    results_dict = dict(results)

    labels = ["1_no_roll", "2_prox_only", "3_dual_20k", "4_dual_10k"]
    short_labels = ["No Roll", "Prox Only", "Dual $20K", "Dual $10K"]

    print()
    print("=" * 105)
    header = f"{'Metric':<25}"
    for sl in short_labels:
        header += f" {sl:>18}"
    print(header)
    print("-" * 105)

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

    for name, key, fmt in metrics:
        row = f"  {name:<23}"
        for lbl in labels:
            r = results_dict.get(lbl, {})
            if "error" in r:
                row += f" {'ERROR':>18}"
            else:
                v = r.get(key, 0)
                row += f" {format(v, fmt):>18}"
        print(row)

    # Delta vs no-roll baseline
    print("-" * 105)
    print("  Delta vs No Roll:")
    base = results_dict.get("1_no_roll", {})
    if "error" not in base:
        for name, key, fmt in metrics:
            row = f"    {name:<21}"
            row += f" {'—':>18}"
            for lbl in labels[1:]:
                r = results_dict.get(lbl, {})
                if "error" in r:
                    row += f" {'—':>18}"
                else:
                    delta = r.get(key, 0) - base.get(key, 0)
                    if isinstance(delta, float) and abs(delta) < 0.01:
                        row += f" {'—':>18}"
                    elif isinstance(delta, int) and delta == 0:
                        row += f" {'—':>18}"
                    else:
                        sign = "+" if delta > 0 else ""
                        row += f" {sign + format(delta, fmt):>18}"

            print(row)

    print("=" * 105)


if __name__ == "__main__":
    main()
