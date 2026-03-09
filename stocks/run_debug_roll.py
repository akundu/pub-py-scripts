#!/usr/bin/env python3
"""Debug: trace exactly why rolls fail on specific dates."""

import sys
import os
from pathlib import Path
from datetime import date

BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))
os.chdir(str(BASE_DIR))

import scripts.backtesting.providers.csv_equity_provider   # noqa
import scripts.backtesting.providers.csv_options_provider   # noqa
import scripts.backtesting.instruments.credit_spread        # noqa
import scripts.backtesting.strategies.credit_spread.percentile_entry  # noqa

from scripts.backtesting.strategies.credit_spread.percentile_entry import PercentileEntryCreditSpreadStrategy

# Monkey-patch _execute_roll to add debug output
original_execute_roll = PercentileEntryCreditSpreadStrategy._execute_roll

def debug_execute_roll(self, pos_dict, exit_signal, day_context):
    params = self.config.params
    position = pos_dict["position"]
    roll_count = pos_dict.get("roll_count", 0)

    roll_min_dte = params.get("roll_min_dte", 3)
    roll_max_dte = params.get("roll_max_dte", 10)
    dte_progression = [roll_min_dte, min(5, roll_max_dte), roll_max_dte]
    new_dte = dte_progression[min(roll_count, len(dte_progression) - 1)]

    roll_percentile = params.get("roll_percentile", min(params.get("percentile", 95), 90))
    pct_data = day_context.signals.get("percentile_range", {})
    strikes_by_dte = pct_data.get("strikes", {})
    dte_strikes = strikes_by_dte.get(new_dte, strikes_by_dte.get(0, {}))
    pct_strikes = dte_strikes.get(roll_percentile, dte_strikes.get(params.get("percentile", 95), {}))
    target_strike = pct_strikes.get(position.option_type)

    options = day_context.options_data
    opts_count = len(options) if options is not None else 0
    dte_filtered_count = 0
    if options is not None and "dte" in options.columns:
        filtered = options[(options["dte"] >= new_dte - 1) & (options["dte"] <= new_dte + 1)]
        dte_filtered_count = len(filtered)
        dtes_avail = sorted(options["dte"].unique().tolist())
    else:
        dtes_avail = []

    print(f"\n  ROLL DEBUG [{day_context.trading_date}]:")
    print(f"    Position: {position.option_type} {position.short_strike}/{position.long_strike}")
    print(f"    Exit: {exit_signal.reason} at price {exit_signal.exit_price:.2f}")
    print(f"    Roll #{roll_count} → target DTE {new_dte}")
    print(f"    Target strike (P{roll_percentile} DTE{new_dte}): {target_strike}")
    print(f"    Signals strikes_by_dte keys: {list(strikes_by_dte.keys())}")
    print(f"    Options rows: {opts_count}, DTEs available: {dtes_avail}")
    print(f"    After DTE filter ({new_dte-1}-{new_dte+1}): {dte_filtered_count} rows")

    result = original_execute_roll(self, pos_dict, exit_signal, day_context)

    if result is None:
        print(f"    RESULT: _execute_roll returned None (complete failure)")
    else:
        closed, new_pos = result
        if new_pos is None:
            print(f"    RESULT: Roll FAILED — closed position but no replacement built")
        else:
            print(f"    RESULT: Roll SUCCESS → {new_pos['position'].option_type} "
                  f"{new_pos['position'].short_strike}/{new_pos['position'].long_strike} DTE={new_pos['dte']}")

    return result

PercentileEntryCreditSpreadStrategy._execute_roll = debug_execute_roll

# Now run the backtest with proximity roll
import yaml
from scripts.backtesting.config import BacktestConfig
from scripts.backtesting.engine import BacktestEngine
from scripts.backtesting.constraints.exit_rules.proximity_roll import ProximityRollExit

config_dict = {
    "infra": {
        "ticker": "NDX",
        "start_date": "2025-03-01",
        "end_date": "2026-02-28",
        "lookback_days": 180,
        "num_processes": 1,
        "output_dir": "results/debug_roll",
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
            "dte": 0, "percentile": 95, "lookback": 120,
            "option_types": ["put", "call"], "spread_width": 50,
            "interval_minutes": 10, "entry_start_utc": "13:00", "entry_end_utc": "17:00",
            "num_contracts": 1, "max_loss_estimate": 75000,
            "profit_target_0dte": 0.75, "profit_target_multiday": 0.50,
            "min_roi_per_day": 0.025, "min_credit": 0.75,
            "min_total_credit": 2500, "min_credit_per_point": 0, "max_contracts": 0,
            "stop_loss_multiplier": 0,
            "roll_enabled": True, "max_rolls": 999,
            "roll_check_start_utc": "19:30", "early_itm_check_utc": "23:59",
            "max_move_cap": 999999, "roll_percentile": 90,
            "roll_min_dte": 3, "roll_max_dte": 10, "max_roll_width": 50,
        },
    },
    "constraints": {
        "budget": {"max_spend_per_transaction": 75000, "daily_budget": 400000},
        "trading_hours": {"entry_start": "13:00", "entry_end": "17:00"},
        "exit_rules": {"profit_target_pct": 0.75, "mode": "first_triggered"},
    },
    "report": {"formats": ["csv"], "metrics": ["win_rate", "roi"]},
}

config_path = BASE_DIR / "_tmp_debug.yaml"
with open(config_path, "w") as f:
    yaml.dump(config_dict, f)

config = BacktestConfig.from_yaml(str(config_path))
engine = BacktestEngine(config)

# Build components and setup
engine.provider = engine._build_providers()
engine.constraints = engine._build_constraints()
engine.exit_manager = engine._build_exit_manager()
engine.collector = engine._build_collector()
engine.strategy = engine._build_strategy()
trading_dates = engine._resolve_dates("NDX")
engine.strategy.setup()

# Inject proximity roll
prox_exit = ProximityRollExit(proximity_pct=0.01, roll_check_start_utc="19:30", max_rolls=999)
engine.strategy._roll_trigger = prox_exit
if engine.strategy.exit_manager is not None:
    new_rules = []
    for rule in engine.strategy.exit_manager._rules:
        if rule.name == "roll_trigger":
            new_rules.append(prox_exit)
        else:
            new_rules.append(rule)
    engine.strategy.exit_manager._rules = new_rules

# Run
all_results = []
for td in trading_dates:
    try:
        day_results = engine._process_day("NDX", td)
        all_results.extend(day_results)
    except Exception as e:
        print(f"Error on {td}: {e}")

engine.strategy.teardown()
engine.provider.close()

config_path.unlink()

# Summary
losses = [r for r in all_results if r.get("pnl", 0) < 0]
rolls = [r for r in all_results if r.get("metadata", {}).get("roll_count", 0) > 0]
print(f"\n\nSummary: {len(all_results)} trades, {len(losses)} losses, {len(rolls)} rolled replacements")
