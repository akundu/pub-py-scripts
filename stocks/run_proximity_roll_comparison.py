#!/usr/bin/env python3
"""Compare: no stop loss + proximity roll (0.5% of strike, after 12pm PST, unlimited rolls)
vs current baseline (3x stop loss + P95 roll trigger, max 2 rolls).

Budget: $300K daily, $30K per transaction, both put+call.
"""

import sys
import os
import yaml
from pathlib import Path
from multiprocessing import Pool

BASE_DIR = Path(__file__).resolve().parent

# ── Configs ──────────────────────────────────────────────────────────────────

def _base_config(label: str) -> dict:
    return {
        "infra": {
            "ticker": "NDX",
            "start_date": "2025-03-01",
            "end_date": "2026-02-28",
            "lookback_days": 180,
            "num_processes": 1,
            "output_dir": f"results/proximity_roll/{label}",
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
                "roll_enabled": True,
                "roll_percentile": 90,
                "roll_min_dte": 3,
                "roll_max_dte": 10,
                "max_roll_width": 50,
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


def config_baseline():
    """Current: 3x stop loss + P95 roll trigger, max 2 rolls."""
    cfg = _base_config("baseline_stoploss_p95roll")
    cfg["strategy"]["params"].update({
        "stop_loss_multiplier": 3.0,
        "max_rolls": 2,
        "roll_check_start_utc": "18:00",
        "early_itm_check_utc": "14:00",
        "max_move_cap": 150,
    })
    cfg["constraints"]["exit_rules"]["stop_loss_pct"] = 3.0
    return cfg


def config_no_sl_no_roll():
    """No stop loss, no rolling."""
    cfg = _base_config("no_stoploss_no_roll")
    cfg["strategy"]["params"].update({
        "stop_loss_multiplier": 0,
        "roll_enabled": False,
        "max_rolls": 0,
    })
    return cfg


def config_no_sl_proximity_roll():
    """No stop loss + proximity roll (0.5% of strike, 12pm PST, unlimited)."""
    cfg = _base_config("no_stoploss_proximity_roll")
    cfg["strategy"]["params"].update({
        "stop_loss_multiplier": 0,
        "max_rolls": 999,
        # Custom proximity roll params (handled by monkey-patch)
        "use_proximity_roll": True,
        "proximity_pct": 0.005,        # 0.5%
        "roll_check_start_utc": "19:00",  # 12pm PST = 19:00 UTC
        "early_itm_check_utc": "23:59",   # Effectively disabled
        "max_move_cap": 999999,            # Effectively disabled
    })
    return cfg


def config_no_sl_proximity_roll_1pct():
    """No stop loss + proximity roll (1.0% of strike, 12pm PST, unlimited)."""
    cfg = _base_config("no_stoploss_proximity_roll_1pct")
    cfg["strategy"]["params"].update({
        "stop_loss_multiplier": 0,
        "max_rolls": 999,
        "use_proximity_roll": True,
        "proximity_pct": 0.01,            # 1.0%
        "roll_check_start_utc": "19:00",
        "early_itm_check_utc": "23:59",
        "max_move_cap": 999999,
    })
    return cfg


# ── Runner ───────────────────────────────────────────────────────────────────

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

    config_path = BASE_DIR / f"_tmp_config_{label}.yaml"
    with open(config_path, "w") as f:
        yaml.dump(config_dict, f)

    try:
        config = BacktestConfig.from_yaml(str(config_path))
        engine = BacktestEngine(config)

        # Monkey-patch: after strategy.setup(), replace RollTriggerExit with ProximityRollExit
        if use_proximity:
            from scripts.backtesting.constraints.exit_rules.proximity_roll import ProximityRollExit
            max_rolls_val = config_dict["strategy"]["params"].get("max_rolls", 999)
            roll_check_start = config_dict["strategy"]["params"].get("roll_check_start_utc", "19:00")

            original_run = engine.run

            def patched_run(dry_run=False, _prox_pct=proximity_pct, _max_r=max_rolls_val, _rcs=roll_check_start):
                # Build components normally
                engine.provider = engine._build_providers()
                engine.constraints = engine._build_constraints()
                engine.exit_manager = engine._build_exit_manager()
                engine.collector = engine._build_collector()
                engine.strategy = engine._build_strategy()

                trading_dates = engine._resolve_dates(engine.config.infra.ticker)

                # Setup strategy (this injects the original RollTriggerExit)
                engine.strategy.setup()

                # NOW replace with ProximityRollExit
                prox_exit = ProximityRollExit(
                    proximity_pct=_prox_pct,
                    roll_check_start_utc=_rcs,
                    max_rolls=_max_r,
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

                # Run the trading loop
                all_results = []
                for trading_date in trading_dates:
                    try:
                        day_results = engine._process_day(engine.config.infra.ticker, trading_date)
                        all_results.extend(day_results)
                    except Exception as e:
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
        ("1_baseline", config_baseline()),
        ("2_no_sl_no_roll", config_no_sl_no_roll()),
        ("3_no_sl_prox_0.5pct", config_no_sl_proximity_roll()),
        ("4_no_sl_prox_1.0pct", config_no_sl_proximity_roll_1pct()),
    ]

    print("=" * 100)
    print("NDX Credit Spread: Stop Loss / Roll Strategy Comparison")
    print(f"Budget: $300K daily, $30K per transaction, both put+call")
    print(f"Period: 2025-03-01 to 2026-02-28 (1 year), P95 entry, 0DTE, 50pt spreads")
    print()
    print("Configs:")
    print("  1) Baseline:        3x stop loss + P95 dynamic roll (max 2 rolls)")
    print("  2) No SL, no roll:  No stop loss, no rolling (hold to expiration)")
    print("  3) No SL + 0.5% roll: No stop loss, roll if within 0.5% of strike after 12pm PST, unlimited")
    print("  4) No SL + 1.0% roll: No stop loss, roll if within 1.0% of strike after 12pm PST, unlimited")
    print("=" * 100)
    print()

    with Pool(processes=4) as pool:
        results = pool.map(run_single, configs)

    results_dict = dict(results)

    labels = ["1_baseline", "2_no_sl_no_roll", "3_no_sl_prox_0.5pct", "4_no_sl_prox_1.0pct"]
    short_labels = ["Baseline", "No SL/Roll", "0.5% Roll", "1.0% Roll"]

    print()
    print("=" * 100)
    header = f"{'Metric':<25}"
    for sl in short_labels:
        header += f" {sl:>16}"
    print(header)
    print("-" * 100)

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
                row += f" {'ERROR':>16}"
            else:
                v = r.get(key, 0)
                row += f" {format(v, fmt):>16}"
        print(row)

    # Delta row vs baseline
    print("-" * 100)
    print("  Delta vs Baseline:")
    base = results_dict.get("1_baseline", {})
    if "error" not in base:
        for name, key, fmt in metrics:
            row = f"    {name:<21}"
            row += f" {'—':>16}"  # baseline column
            for lbl in labels[1:]:
                r = results_dict.get(lbl, {})
                if "error" in r:
                    row += f" {'—':>16}"
                else:
                    delta = r.get(key, 0) - base.get(key, 0)
                    if isinstance(delta, float) and abs(delta) < 0.01:
                        row += f" {'—':>16}"
                    elif isinstance(delta, int) and delta == 0:
                        row += f" {'—':>16}"
                    else:
                        sign = "+" if delta > 0 else ""
                        row += f" {sign + format(delta, fmt):>16}"
                print_row = True
            print(row)

    print("=" * 100)


if __name__ == "__main__":
    main()
