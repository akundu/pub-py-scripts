# Backtesting Framework Guide

This is the authoritative reference for the modular backtesting framework.
Any Claude Code session working on backtesting should read this file first.

## Architecture Overview

The framework separates concerns into composable layers:

```
YAML Config -> Engine -> { Providers, Strategy, Constraints, ExitRules } -> Results -> Reporters
```

- **Providers**: Data sources (CSV files, QuestDB). Multiple providers per test via CompositeProvider.
- **Signals**: Prediction/analysis generators (close predictor, percentile model, etc.). NOT strategies -- they feed INTO strategies.
- **Instruments**: Tradeable structures with P&L math (credit spread, iron condor, strangle, straddle).
- **Strategies**: Trading logic. Credit spread variants are the primary strategies. Only strategies decide when to enter/exit.
- **Constraints**: Entry gates (budget limits, trading hours, rate limiting). Composable via ConstraintChain.
- **Exit Rules**: Position management (profit target, stop loss, time-based). Composable via CompositeExit (first-triggered wins).
- **Results**: Collection, metrics (Sharpe, drawdown, profit factor), and reporting (console, CSV, JSON).
- **Grid Sweep**: Presentation layer that runs any strategy across parameter combinations. Not a strategy.

## Quick Start

```bash
# Run a backtest from project root (stocks/)
python -m scripts.backtesting.runner --config scripts/backtesting/configs/credit_spread_0dte_ndx.yaml

# Dry run (preview without executing)
python -m scripts.backtesting.runner --config scripts/backtesting/configs/credit_spread_0dte_ndx.yaml --dry-run

# Grid sweep over parameter combinations
python -m scripts.backtesting.runner --config scripts/backtesting/configs/grid_sweep_comprehensive.yaml

# Override config parameters via CLI
python -m scripts.backtesting.runner --config scripts/backtesting/configs/credit_spread_0dte_ndx.yaml \
    --ticker SPX --start-date 2026-01-01 --end-date 2026-02-28

# Run framework tests
python -m pytest tests/test_backtesting_config.py tests/test_backtesting_constraints.py \
    tests/test_backtesting_exit_rules.py tests/test_backtesting_instruments.py \
    tests/test_backtesting_collector.py -v
```

## Directory Structure

```
scripts/backtesting/
    BACKTESTING.md                     # THIS FILE -- framework guide
    engine.py                          # BacktestEngine orchestrator
    config.py                          # Config dataclasses + YAML/JSON loader
    runner.py                          # CLI entry point (argparse, --help for usage)

    providers/
        base.py                        # DataProvider ABC
        registry.py                    # DataProviderRegistry
        composite_provider.py          # CompositeProvider (multiple providers per test)
        csv_equity_provider.py         # Reads OHLCV from equities_output/ CSVs
        csv_options_provider.py        # Reads options chains from options_csv_output/ CSVs
        questdb_provider.py            # Reads from QuestDB (realtime + daily)

    signals/
        base.py                        # SignalGenerator ABC
        registry.py                    # SignalGeneratorRegistry
        close_predictor.py             # Wraps close_predictor/prediction.py
        percentile_model.py            # Wraps common/range_percentiles.py
        conditional_model.py           # Wraps conditional similarity weighting
        band_selector.py              # Wraps close_predictor/band_selector.py

    instruments/
        base.py                        # Instrument ABC + InstrumentPosition + PositionResult
        factory.py                     # InstrumentFactory registry
        pnl.py                         # Shared P&L math (spread, iron condor, strangle, straddle)
        credit_spread.py               # Wraps spread_builder.build_credit_spreads()
        iron_condor.py                 # Wraps iron_condor_builder.IronCondorBuilder
        strangle.py                    # Short strangle
        straddle.py                    # Short straddle

    strategies/
        base.py                        # BacktestStrategy ABC + DayContext dataclass
        registry.py                    # BacktestStrategyRegistry
        credit_spread/
            base_credit_spread.py      # BaseCreditSpreadStrategy (common CS logic)
            zero_dte.py                # ZeroDTEStrategy -- same-day expiration
            multi_day.py               # MultiDayDTEStrategy -- 1-20 day DTE
            scale_in.py                # ScaleInStrategy -- layered entries on breach
            tiered.py                  # TieredStrategy -- simultaneous multi-percentile
            time_allocated.py          # TimeAllocatedStrategy -- hourly window budgets
            gate_filtered.py           # GateFilteredStrategy -- prediction-gated entries

    constraints/
        base.py                        # Constraint ABC + ConstraintChain + ConstraintContext
        budget/
            max_spend.py               # MaxSpendPerTransaction
            daily_budget.py            # DailyBudget (tracks capital lifecycle)
            gradual_distribution.py    # GradualDistribution (sliding window rate limit)
        trading_hours/
            entry_window.py            # EntryWindow (min/max entry times)
            forced_exit.py             # ForcedExit (no entries past this time)
        exit_rules/
            base_exit.py               # ExitRule ABC + ExitSignal dataclass
            time_exit.py               # TimeBasedExit
            profit_target.py           # ProfitTargetExit (% of max credit)
            stop_loss.py               # StopLossExit (% of initial credit)
            composite_exit.py          # CompositeExit (first-triggered wins)

    results/
        collector.py                   # ResultCollector (aggregates trade results)
        metrics.py                     # StandardMetrics (win_rate, roi, sharpe, drawdown, etc.)
        grid_sweep.py                  # GridSweep (param sweep presentation layer)
        reporters/
            base.py                    # ReportGenerator ABC
            console.py                 # ConsoleReporter (formatted stdout)
            csv_reporter.py            # CSVReporter (trades.csv + metrics.csv)
            json_reporter.py           # JSONReporter (full results JSON)
            comparison.py              # ComparisonReporter (cross-run comparison)
            grid_reporter.py           # GridReporter (grid sweep ranking)

    parallel/
        executor.py                    # BacktestExecutor (multiprocessing)

    configs/                           # YAML config files (one per strategy variant)
        credit_spread_0dte_ndx.yaml
        credit_spread_multiday_ndx.yaml
        credit_spread_scale_in.yaml
        credit_spread_tiered.yaml
        credit_spread_gate.yaml
        grid_sweep_comprehensive.yaml
        grid_sweep_intraday.yaml

    utils/
        compare_results.py             # Cross-run comparison CLI
        analyze_grid.py                # Grid result analysis CLI
        generate_report.py             # Report generation CLI
```

## Inheritance Hierarchy

```
DataProvider (ABC)
├── CSVEquityProvider          # CSV OHLCV bars
├── CSVOptionsProvider         # CSV options chains
└── QuestDBProvider            # QuestDB realtime/daily

SignalGenerator (ABC)
├── ClosePredictorSignal       # LightGBM + statistical predictions
├── PercentileModelSignal      # Historical distribution bands
├── ConditionalModelSignal     # Similarity-weighted predictions
└── BandSelectorSignal         # Intelligent band selection

Instrument (ABC)
├── CreditSpreadInstrument     # Vertical credit spreads
├── IronCondorInstrument       # Put spread + call spread
├── StrangleInstrument         # Short put + short call (different strikes)
└── StraddleInstrument         # Short put + short call (same strike)

BacktestStrategy (ABC)
└── BaseCreditSpreadStrategy   # Common CS logic (constraint checking, P&L eval)
    ├── ZeroDTEStrategy        # 0DTE same-day expiration
    ├── MultiDayDTEStrategy    # 1-20 day DTE, multi-day tracking
    ├── ScaleInStrategy        # Layered entries on price breach
    ├── TieredStrategy         # Simultaneous multi-percentile entries
    ├── TimeAllocatedStrategy  # Hourly window-based entries
    └── GateFilteredStrategy   # Prediction-gated entries

Constraint (ABC)
├── MaxSpendPerTransaction     # Per-position capital cap
├── DailyBudget                # Daily capital with lifecycle (freed on close)
├── GradualDistribution        # Sliding window rate limit
├── EntryWindow                # Time-of-day entry restriction
└── ForcedExit                 # No entries past deadline

ExitRule (ABC)
├── TimeBasedExit              # Exit at specific time
├── ProfitTargetExit           # Exit at % of max profit
├── StopLossExit               # Exit at % loss threshold
└── CompositeExit              # First-triggered wins
```

## How to Add a New Strategy

This is the most common extension point. Follow this pattern:

### Step 1: Create the strategy file

Create `scripts/backtesting/strategies/credit_spread/my_strategy.py`:

```python
"""MyStrategy -- description of what makes this strategy unique."""

from datetime import datetime
from typing import Dict, List

from .base_credit_spread import BaseCreditSpreadStrategy
from ..base import DayContext
from ..registry import BacktestStrategyRegistry


class MyStrategy(BaseCreditSpreadStrategy):
    """One-line description."""

    @property
    def name(self) -> str:
        return "my_strategy"

    def generate_signals(self, day_context: DayContext) -> List[Dict]:
        """Generate entry signals. This is where your strategy logic lives."""
        params = self.config.params
        option_types = params.get("option_types", ["put", "call"])
        percent_beyond = params.get("percent_beyond", "0.03:0.05")
        num_contracts = params.get("num_contracts", 1)

        signals = []
        if day_context.options_data is None or day_context.options_data.empty:
            return signals

        for opt_type in option_types:
            signals.append({
                "option_type": opt_type,
                "percent_beyond": percent_beyond,
                "instrument": "credit_spread",  # or "iron_condor"
                "num_contracts": num_contracts,
                "timestamp": datetime.combine(
                    day_context.trading_date, datetime.min.time()
                ),
                "max_loss": params.get("max_loss_estimate", 10000),
            })

        return signals

    # execute_signals() and evaluate() are inherited from BaseCreditSpreadStrategy.
    # Override them only if your strategy needs custom position building or exit logic.


# REQUIRED: Register so the engine can find it by name
BacktestStrategyRegistry.register("my_strategy", MyStrategy)
```

### Step 2: Create a YAML config

Create `scripts/backtesting/configs/my_strategy.yaml`:

```yaml
infra:
  ticker: NDX
  start_date: "2025-12-01"
  end_date: "2026-02-28"
  output_dir: results/my_strategy

providers:
  - name: csv_equity
    role: equity
    params:
      csv_dir: equities_output
  - name: csv_options
    role: options
    params:
      csv_dir: options_csv_output
      dte_buckets: [0]

strategy:
  name: my_strategy      # Must match the registered name
  params:
    option_types: [put, call]
    percent_beyond: "0.03:0.05"
    num_contracts: 1
    max_loss_estimate: 10000
    # Add any custom params your strategy needs here

constraints:
  budget:
    daily_budget: 100000
  trading_hours:
    entry_start: "09:45"
    entry_end: "15:00"
  exit_rules:
    profit_target_pct: 0.50
    stop_loss_pct: 2.0
    time_exit: "15:30"
    mode: first_triggered

report:
  formats: [console, csv]
  metrics: [win_rate, roi, sharpe, profit_factor]
```

### Step 3: Import it in runner.py (for auto-registration)

Add to `scripts/backtesting/runner.py` in the `main()` function:
```python
import scripts.backtesting.strategies.credit_spread.my_strategy  # noqa: F401
```

### Step 4: Write tests

Create `tests/test_backtesting_my_strategy.py` testing signal generation logic.

### Step 5: Run

```bash
python -m scripts.backtesting.runner --config scripts/backtesting/configs/my_strategy.yaml
```

## How to Add a New Constraint

```python
# scripts/backtesting/constraints/budget/my_constraint.py
from ..base import Constraint, ConstraintContext, ConstraintResult

class MyConstraint(Constraint):
    def __init__(self, threshold: float):
        self._threshold = threshold

    @property
    def name(self) -> str:
        return "my_constraint"

    def check(self, context: ConstraintContext) -> ConstraintResult:
        if context.position_capital > self._threshold:
            return ConstraintResult.reject(self.name, f"Exceeds {self._threshold}")
        return ConstraintResult.allow()

    # Optional: override reset_day(), on_position_opened(), on_position_closed()
```

Then add it in `engine.py`'s `_build_constraints()` method, or manually to the ConstraintChain.

## How to Add a New Exit Rule

```python
# scripts/backtesting/constraints/exit_rules/my_exit.py
from .base_exit import ExitRule, ExitSignal

class MyExit(ExitRule):
    @property
    def name(self) -> str:
        return "my_exit"

    def should_exit(self, position, current_price, current_time, day_context=None):
        if some_condition:
            return ExitSignal(
                triggered=True, rule_name=self.name,
                exit_price=current_price, exit_time=current_time,
                reason="my_reason"
            )
        return None
```

## How to Add a New Instrument

```python
# scripts/backtesting/instruments/my_instrument.py
from .base import Instrument, InstrumentPosition, PositionResult
from .factory import InstrumentFactory

class MyInstrument(Instrument):
    @property
    def name(self) -> str:
        return "my_instrument"

    def build_position(self, options_data, signal, prev_close):
        # Build and return InstrumentPosition or None
        ...

    def calculate_pnl(self, position, exit_price):
        # Calculate and return PositionResult
        ...

InstrumentFactory.register("my_instrument", MyInstrument)
```

## Config Schema Reference

All config dataclasses are in `scripts/backtesting/config.py`:

| Dataclass | Key Fields |
|-----------|------------|
| `InfraConfig` | ticker, start_date, end_date, lookback_days, num_processes, output_dir |
| `ProviderEntry` | name (registry key), role (equity/options/realtime), params (dict) |
| `StrategyConfig` | name (registry key), params (strategy-specific dict) |
| `BudgetConfig` | max_spend_per_transaction, daily_budget, gradual_distribution |
| `TradingHoursConfig` | entry_start, entry_end, forced_exit_time |
| `ExitRulesConfig` | profit_target_pct, stop_loss_pct, time_exit, mode |
| `GridSweepConfig` | param_grid (dotted keys -> value lists), batch_size |

### Signal dict format (returned by generate_signals)

Strategies return a list of signal dicts. Required keys for credit spread:
```python
{
    "option_type": "put" or "call",
    "percent_beyond": "0.03:0.05",     # put_pct:call_pct
    "instrument": "credit_spread",      # registry key
    "num_contracts": 1,
    "timestamp": datetime,
    "max_loss": 10000,                  # estimated, for constraint checking
}
```

### Position dict format (passed to evaluate)

execute_signals returns a list of position dicts:
```python
{
    "position": InstrumentPosition,     # from instrument.build_position()
    "signal": dict,                     # the original signal
}
```

### Result dict format (returned by evaluate, stored in collector)

evaluate returns dicts from PositionResult.to_dict():
```python
{
    "instrument_type", "option_type", "entry_time", "exit_time",
    "short_strike", "long_strike", "initial_credit", "max_loss",
    "num_contracts", "exit_price", "pnl", "pnl_per_contract",
    "exit_reason", "credit", "trading_date",
}
```

## Data Sources

### Options CSV Directories

There are **two** options data directories. Use the correct one based on DTE:

| Directory | Contents | Use For |
|-----------|----------|---------|
| `options_csv_output/` | 0DTE options only (same-day expiration) | 0DTE strategies (`dte_buckets: [0]`) |
| `options_csv_output_full/` | Full options chain (0-29+ DTE, multiple expirations) | Multi-day strategies (`dte_buckets: [1,2,3,...]`) |

Both directories have the same CSV format: `{TICKER}_options_{YYYY-MM-DD}.csv` with columns:
`timestamp, ticker, type, strike, expiration, bid, ask, day_close, vwap, fmv, delta, gamma, theta, vega, implied_volatility, volume`

The provider calculates DTE automatically from the `expiration` column vs the trading date.

### Equity CSV Directory

| Directory | Contents |
|-----------|----------|
| `equities_output/` | 5-minute OHLCV bars. Subdirs use `I:` prefix for indices (e.g., `I:NDX/`, `I:SPX/`). The provider handles this automatically. |

### Timestamps

All CSV timestamps are in **UTC** (e.g., `2026-02-27 14:30:00+00:00` = 9:30 AM ET). Configure constraint times in UTC accordingly:

| ET Time | UTC Time |
|---------|----------|
| 09:30 | 14:30 |
| 09:45 | 14:45 |
| 15:00 | 20:00 |
| 15:30 | 20:30 |
| 15:45 | 20:45 |
| 16:00 | 21:00 |

## Existing Code Wrapped (NOT modified, NOT deleted)

The framework wraps these existing modules via providers/signals/instruments:

| Existing Module | Wrapped By |
|---|---|
| `scripts/csv_prediction_backtest.py` | `providers/csv_equity_provider.py` |
| `scripts/credit_spread_utils/data_loader.py` | `providers/csv_options_provider.py` |
| `common/questdb_db.py` | `providers/questdb_provider.py` |
| `scripts/credit_spread_utils/spread_builder.py` | `instruments/credit_spread.py` |
| `scripts/credit_spread_utils/iron_condor_builder.py` | `instruments/iron_condor.py` |
| `scripts/credit_spread_utils/backtest_engine.py` | `instruments/pnl.py` |
| `scripts/credit_spread_utils/capital_utils.py` | `constraints/budget/daily_budget.py` |
| `scripts/credit_spread_utils/rate_limiter.py` | `constraints/budget/gradual_distribution.py` |
| `scripts/credit_spread_utils/metrics.py` | `results/metrics.py` |
| `scripts/close_predictor/prediction.py` | `signals/close_predictor.py` |

## Test Files

| Test File | Covers |
|-----------|--------|
| `tests/test_backtesting_config.py` | Config loading, parsing, YAML, deep_set |
| `tests/test_backtesting_constraints.py` | All constraints + ConstraintChain |
| `tests/test_backtesting_exit_rules.py` | All exit rules + CompositeExit |
| `tests/test_backtesting_instruments.py` | P&L calculations + InstrumentFactory |
| `tests/test_backtesting_collector.py` | ResultCollector + StandardMetrics |
