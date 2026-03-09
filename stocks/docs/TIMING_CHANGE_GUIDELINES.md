# Timing Change Guidelines for Roll/Exit Rules

This document describes how to change timing parameters (roll check times, exit
windows, 0DTE warning times) across the entire codebase so that backtests and
live advisors stay in sync.

## Canonical Time: Roll Check Start

**Current setting: 18:00 UTC = 11:00 AM PST = 2:00 PM ET**

This is the time after which the system starts checking whether open positions
need to be rolled (i.e., the short strike is threatened by remaining price
movement before close).

## All Locations That Must Be Updated

When changing the roll check start time, you MUST update **all** of these
locations. Missing one creates a divergence between backtest and live results.

### 1. Backtesting Exit Rules (defaults in Python code)

| File | Parameter | Current |
|------|-----------|---------|
| `scripts/backtesting/constraints/exit_rules/roll_trigger.py` | `roll_check_start_utc` default | `"18:00"` |
| `scripts/backtesting/constraints/exit_rules/proximity_roll.py` | `roll_check_start_utc` default | `"18:00"` |

These are the **source of truth** for backtesting. Both `RollTriggerExit` and
`ProximityRollExit` use this parameter to gate when proximity/P95 roll checks
activate.

### 2. Backtesting Configs (YAML files)

| File | Line | Current |
|------|------|---------|
| `scripts/backtesting/configs/percentile_entry_ndx.yaml` | `roll_check_start_utc` | `"18:00"` |
| `scripts/backtesting/configs/percentile_entry_grid_sweep.yaml` | `roll_check_start_utc` | `"18:00"` |

### 3. Tiered Backtest Runner

| File | Location | Current |
|------|----------|---------|
| `run_tiered_backtest_v2.py` | `_tier_config()` → `roll_check_start_utc` | `"18:00"` |
| `run_tiered_backtest_v2.py` | Print statement (line ~275) | descriptive text |
| `run_tiered_backtest_v2.py` | Module docstring | descriptive text |

### 4. Live Advisor — Profile System

| File | Location | Current |
|------|----------|---------|
| `scripts/live_trading/advisor/tier_config.py` | `STRATEGY_DEFAULTS["roll_check_start_utc"]` | `"18:00"` |
| `scripts/live_trading/advisor/profile_loader.py` | `ExitRuleConfig.roll_check_start_utc` default | `"18:00"` |
| `scripts/live_trading/advisor/profile_loader.py` | `_parse_exit_rules()` fallback | `"18:00"` |
| `scripts/live_trading/advisor/profile_loader.py` | `from_tier_config()` fallback | `"18:00"` |
| `scripts/live_trading/advisor/profiles/tiered_v2.yaml` | `exit_rules.roll_check_start_utc` | `"18:00"` |
| `scripts/live_trading/advisor/profiles/tiered_v2.yaml` | `strategy_defaults.roll_check_start_utc` | `"18:00"` |

### 5. Live Advisor — Evaluator

| File | Location | What It Does |
|------|----------|-------------|
| `scripts/live_trading/advisor/tier_evaluator.py` | `evaluate_exits()` | Reads `exit_rules.roll_check_start_utc` from profile (no hardcoded time) |

The 0DTE proximity warning now also uses `exit_rules.roll_check_start_utc`
from the profile rather than a hardcoded `time(19, 30)`.

### 6. Live Trading Paper Config

| File | Line | Current |
|------|------|---------|
| `scripts/live_trading/configs/ndx_credit_spread_paper.yaml` | `roll_check_start_utc` | `"18:00"` |

### 7. Strategy Implementations

| File | Parameter | Current |
|------|-----------|---------|
| `scripts/backtesting/strategies/credit_spread/percentile_entry.py` | `roll_check_start_utc` fallback | `"18:00"` |
| `scripts/live_trading/strategies/ndx_credit_spread.py` | `roll_check_start_utc` fallback | `"18:00"` |

### 8. Tests

| File | What to Update |
|------|----------------|
| `tests/test_live_advisor.py` | `_make_test_profile()` strategy_defaults, and any `moves_to_close` time slots / datetime test fixtures |
| `tests/test_backtesting_percentile_entry.py` | Explicit `roll_check_start_utc="18:00"` in `RollTriggerExit()` calls |

### 9. Documentation

| File | Sections |
|------|----------|
| `docs/NDX_DAILY_TRADING_ALGORITHM.md` | Roll decision point time references |
| `docs/NDX_CREDIT_SPREAD_PLAYBOOK.md` | `roll_check_start_utc` parameter table |

## Step-by-Step Procedure for Changing the Roll Check Time

1. **Pick the new UTC time.** Convert from your desired PST/ET:
   - PST + 8 = UTC (standard time)
   - PDT + 7 = UTC (daylight saving)
   - ET + 5 = UTC (standard) / ET + 4 = UTC (daylight saving)

2. **Update the two exit rule defaults** (roll_trigger.py, proximity_roll.py).

3. **Update all YAML configs** — grep for `roll_check_start_utc` across the repo.

4. **Update `run_tiered_backtest_v2.py`** — the `_tier_config()` function and
   print/docstring comments.

5. **Update `tier_config.py`** — `STRATEGY_DEFAULTS["roll_check_start_utc"]`.

6. **Update `profile_loader.py`** — the `ExitRuleConfig` dataclass default, and
   both fallback strings in `_parse_exit_rules()` and `from_tier_config()`.

7. **Update advisor profile YAMLs** in `profiles/`.

8. **Update tests** — search for the old UTC time string in test files and update
   both the config dicts and any datetime fixtures that simulate "after roll check".

9. **Update docs** — NDX_DAILY_TRADING_ALGORITHM.md, NDX_CREDIT_SPREAD_PLAYBOOK.md.

10. **Run tests:** `python -m pytest tests/test_live_advisor.py tests/test_backtesting_exit_rules.py tests/test_backtesting_percentile_entry.py -v`

11. **Re-run the backtest:** `python run_tiered_backtest_v2.py`

12. **Compare results** — the backtest output dir is `results/tiered_portfolio_v2/`.
    Each tier has its own `trades.csv`. Compare win rate, net P&L, and roll counts
    before/after the timing change.

## Related Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `roll_check_start_utc` | `"18:00"` | When P95 roll / proximity roll checks begin |
| `early_itm_check_utc` | `"14:00"` | Early morning ITM check (rolls immediately if breached) |
| `max_move_cap` | `150` | Cap on P95 remaining-move in points |
| `roll_proximity_pct` | `0.005` | Proximity threshold (0.5% of strike) |
| `max_rolls` | `2` | Max rolls per position chain |
| `zero_dte_proximity_warn` | `0.005` | 0DTE OTM warning threshold |

## Timezone Reference

| UTC | PST (Nov-Mar) | PDT (Mar-Nov) | ET (Nov-Mar) | ET (Mar-Nov) |
|-----|---------------|---------------|--------------|---------------|
| 14:00 | 6:00 AM | 7:00 AM | 9:00 AM | 10:00 AM |
| 18:00 | 10:00 AM | 11:00 AM | 1:00 PM | 2:00 PM |
| 19:00 | 11:00 AM | 12:00 PM | 2:00 PM | 3:00 PM |
| 19:30 | 11:30 AM | 12:30 PM | 2:30 PM | 3:30 PM |
| 20:00 | 12:00 PM | 1:00 PM | 3:00 PM | 4:00 PM |

**Note:** Market close is 20:00 UTC (4:00 PM ET / 1:00 PM PDT).
