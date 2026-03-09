#!/usr/bin/env python3
"""Compare exit strategies: flat 75% profit target vs theta decay curve.

Tests:
  1. flat_75: Current default — exit at 75% of credit captured
  2. theta_default: Theta decay curve with default params (ahead=0.25, min=0.50)
  3. theta_aggressive: Tighter ahead threshold (0.15) and lower min (0.40)
  4. theta_conservative: Wider ahead threshold (0.35) and higher min (0.60)

All configs use contrarian_double mode (best P&L from directional comparison).
"""

import sys
import os
import yaml
from pathlib import Path
from multiprocessing import Pool

BASE_DIR = Path(__file__).resolve().parent


def _base_config(label: str, dte: int) -> dict:
    return {
        "infra": {
            "ticker": "NDX",
            "start_date": "2025-03-01",
            "end_date": "2026-02-28",
            "lookback_days": 180,
            "num_processes": 1,
            "output_dir": f"results/theta_decay/{label}",
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
                "directional_entry": "contrarian_double",
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
    """Generate all test configurations."""
    configs = []
    dtes = [0, 2, 5]

    for dte in dtes:
        # 1. Flat 75% profit target (baseline)
        label = f"dte{dte}_flat75"
        cfg = _base_config(label, dte)
        cfg["strategy"]["params"]["_exit_mode"] = "flat"
        configs.append((label, cfg))

        # 2. Theta decay — default
        label = f"dte{dte}_theta_default"
        cfg = _base_config(label, dte)
        cfg["strategy"]["params"]["_exit_mode"] = "theta"
        cfg["strategy"]["params"]["_theta_ahead"] = 0.25
        cfg["strategy"]["params"]["_theta_min_decay"] = 0.50
        cfg["strategy"]["params"]["_theta_cut_behind"] = 0.40
        cfg["strategy"]["params"]["_theta_cut_min_time"] = 0.60
        configs.append((label, cfg))

        # 3. Theta decay — aggressive (exit sooner)
        label = f"dte{dte}_theta_aggr"
        cfg = _base_config(label, dte)
        cfg["strategy"]["params"]["_exit_mode"] = "theta"
        cfg["strategy"]["params"]["_theta_ahead"] = 0.15
        cfg["strategy"]["params"]["_theta_min_decay"] = 0.40
        cfg["strategy"]["params"]["_theta_cut_behind"] = 0.30
        cfg["strategy"]["params"]["_theta_cut_min_time"] = 0.50
        configs.append((label, cfg))

        # 4. Theta decay — conservative (hold longer)
        label = f"dte{dte}_theta_cons"
        cfg = _base_config(label, dte)
        cfg["strategy"]["params"]["_exit_mode"] = "theta"
        cfg["strategy"]["params"]["_theta_ahead"] = 0.35
        cfg["strategy"]["params"]["_theta_min_decay"] = 0.60
        cfg["strategy"]["params"]["_theta_cut_behind"] = 0.50
        cfg["strategy"]["params"]["_theta_cut_min_time"] = 0.70
        configs.append((label, cfg))

    return configs


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

    # Extract custom exit params before writing YAML
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
            # Replace profit_target exit rule with theta decay
            from scripts.backtesting.constraints.exit_rules.theta_decay_exit import ThetaDecayExit

            theta_exit = ThetaDecayExit(
                take_profit_ahead_pct=theta_ahead,
                min_decay_pct=theta_min_decay,
                cut_behind_pct=theta_cut_behind,
                cut_min_time_pct=theta_cut_min_time,
            )

            # Patch: after engine builds strategy, replace exit rules
            original_run = engine.run

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


def main():
    configs = _make_configs()

    print("=" * 120)
    print("NDX Credit Spread: Theta Decay Exit vs Flat Profit Target")
    print(f"P80, contrarian_double mode, 1 contract base, 50pt spreads, no stop loss")
    print(f"Period: 2025-03-01 to 2026-02-28 (1 year)")
    print()
    print("Exit modes:")
    print("  flat_75:       Close at 75% of credit captured (current default)")
    print("  theta_default: Theta curve — ahead 25ppt+50% min decay, cut behind 40ppt after 60% time")
    print("  theta_aggr:    Theta curve — ahead 15ppt+40% min decay, cut behind 30ppt after 50% time")
    print("  theta_cons:    Theta curve — ahead 35ppt+60% min decay, cut behind 50ppt after 70% time")
    print("=" * 120)
    print()

    with Pool(processes=min(8, len(configs))) as pool:
        results = pool.map(run_single, configs)

    results_dict = dict(results)

    metrics_list = [
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

    dtes = [0, 2, 5]
    modes = ["flat75", "theta_default", "theta_aggr", "theta_cons"]
    short_labels = ["Flat 75%", "Theta Def", "Theta Aggr", "Theta Cons"]

    for dte in dtes:
        print()
        print(f"{'='*100}")
        print(f"  DTE = {dte}")
        print(f"{'='*100}")
        header = f"  {'Metric':<22}"
        for sl in short_labels:
            header += f" {sl:>18}"
        print(header)
        print(f"  {'-'*96}")

        labels = [f"dte{dte}_{m}" for m in modes]
        for name, key, fmt in metrics_list:
            row = f"  {name:<22}"
            for lbl in labels:
                r = results_dict.get(lbl, {})
                if "error" in r:
                    row += f" {'ERROR':>18}"
                else:
                    v = r.get(key, 0)
                    row += f" {format(v, fmt):>18}"
            print(row)

    # Summary
    print()
    print("=" * 120)
    print("SUMMARY")
    print("-" * 120)
    header = f"  {'DTE':<6}"
    for sl in short_labels:
        header += f" {sl + ' P&L':>15} {sl + ' WR':>8} {sl + ' Sh':>8}"
    print(header)
    print(f"  {'-'*114}")

    for dte in dtes:
        row = f"  {dte:<6}"
        for mode in modes:
            lbl = f"dte{dte}_{mode}"
            r = results_dict.get(lbl, {})
            if "error" in r:
                row += f" {'ERR':>15} {'':>8} {'':>8}"
            else:
                row += f" ${r.get('net_pnl', 0):>13,.0f} {r.get('win_rate', 0):>6.1f}% {r.get('sharpe', 0):>7.2f}"
        print(row)
    print("=" * 120)


if __name__ == "__main__":
    main()
