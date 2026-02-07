# Credit Spread Analysis - Architecture & Strategy Framework

> **MANDATE: This document MUST be updated with every code change to the credit
> spread analysis system. When adding, modifying, or removing modules, commands,
> strategies, or features, update the relevant sections below. This is the
> single source of truth for the system architecture.**

---

## Table of Contents

1. [System Overview](#system-overview)
2. [Architecture](#architecture)
3. [Module Map](#module-map)
4. [Daily Workflow Commands](#daily-workflow-commands)
5. [Strategy Framework](#strategy-framework)
6. [Grid Search](#grid-search)
7. [Feature Flags Reference](#feature-flags-reference)
8. [Configuration](#configuration)
9. [Adding a New Strategy](#adding-a-new-strategy)
10. [Supporting Scripts](#supporting-scripts)
11. [Change Log](#change-log)

---

## System Overview

The credit spread analysis system backtests 0DTE (zero days to expiration)
credit spread strategies on NDX and SPX options data. It reads CSV files
containing options chain snapshots at 15-minute intervals and evaluates
spread opportunities against historical price data stored in QuestDB.

**Key capabilities:**

- Single-run backtesting with configurable parameters
- Grid search optimization across parameter combinations
- Continuous live monitoring during market hours
- Dynamic spread width calculation based on strike distance
- Scale-in on breach (layered entry) strategy
- Tiered investment (multi-tier concurrent positions) strategy
- Delta-based filtering using Black-Scholes with VIX1D implied volatility
- Rate limiting (sliding window and time-block based)
- Binary caching for fast subsequent loads
- Multiprocessing for parallel CSV processing

**Main entry point:** `scripts/analyze_credit_spread_intervals.py` (~1,505 lines)

---

## Architecture

### High-Level Data Flow

```
CSV Files (options_csv_output/TICKER/)
    │
    ▼
data_loader.py ─── load_data_cached() ─── binary cache (.options_cache/)
    │
    ▼
interval_analyzer.py ─── analyze_interval() ─── QuestDB (price data)
    │                         │
    │                    spread_builder.py ─── build_credit_spreads()
    │                         │
    │                    backtest_engine.py ─── calculate_spread_pnl()
    │
    ▼
main() dispatcher
    ├── Standard results ─── metrics.py / output_formatter.py
    ├── Grid search     ─── grid_search.py ─── parallel workers
    ├── Continuous mode  ─── continuous_runner.py
    ├── Scale-in        ─── scale_in_utils.py
    ├── Tiered          ─── tiered_investment_utils.py
    └── Strategy framework ─── strategies/ package
```

### Directory Structure

```
stocks/
├── scripts/
│   ├── analyze_credit_spread_intervals.py    # Main entry point (1,505 lines)
│   ├── fetch_index_prices.py                 # Price movement analysis (separate)
│   ├── ndx_risk_gradient_analysis.py         # Risk gradient analysis (separate)
│   ├── ndx_daily_calculator.py               # Daily strike calculator (separate)
│   ├── grid_config_*.json                    # Grid search configurations
│   ├── scale_in_config_*.json                # Scale-in strategy configs
│   ├── tiered_config_*.json                  # Tiered strategy configs
│   └── credit_spread_utils/                  # Utility modules package
│       ├── __init__.py                       # Package init, exports all modules
│       ├── arg_parser.py                     # CLI argument parsing (~610 lines)
│       ├── backtest_engine.py                # P&L calculation, profit targets
│       ├── capital_utils.py                  # Capital lifecycle management
│       ├── continuous_runner.py              # Continuous analysis mode
│       ├── data_loader.py                    # CSV loading, binary caching
│       ├── delta_utils.py                    # Delta calculation, BS model
│       ├── dynamic_width_utils.py            # Dynamic spread width calculation
│       ├── grid_search.py                    # Grid search optimization engine
│       ├── interval_analyzer.py              # Core interval analysis
│       ├── metrics.py                        # Trading statistics, reporting
│       ├── output_formatter.py               # Display and printing utilities
│       ├── price_utils.py                    # Price fetching from QuestDB
│       ├── rate_limiter.py                   # Sliding window rate limiting
│       ├── scale_in_utils.py                 # Scale-in on breach logic
│       ├── spread_builder.py                 # Spread building, option pricing
│       ├── tiered_investment_utils.py        # Tiered investment logic
│       ├── time_block_rate_limiter.py        # Time-block rate limiting
│       ├── timezone_utils.py                 # Timezone handling
│       └── strategies/                       # Strategy framework package
│           ├── __init__.py                   # Imports concrete strategies
│           ├── base.py                       # BaseStrategy ABC, StrategyConfig
│           ├── registry.py                   # StrategyRegistry
│           ├── single_entry.py               # SingleEntryStrategy (default)
│           ├── scale_in_strategy.py          # ScaleInStrategy
│           └── tiered_strategy.py            # TieredStrategy
├── docs/
│   └── CREDIT_SPREAD_STRATEGIES.md           # This file
├── common/
│   ├── questdb_db.py                         # QuestDB database interface
│   ├── logging_utils.py                      # Logging configuration
│   └── market_hours.py                       # Market hours awareness
└── options_csv_output/                       # CSV data directory
    ├── NDX/                                  # NDX_options_YYYY-MM-DD.csv
    └── SPX/                                  # SPX_options_YYYY-MM-DD.csv
```

---

## Module Map

| Module | Key Functions / Classes | Purpose |
|--------|------------------------|---------|
| `analyze_credit_spread_intervals.py` | `main()`, `analyze_scale_in_trade()` | Entry point, CLI dispatch, scale-in analysis |
| `arg_parser.py` | `parse_args()` | CLI argument definitions with help text |
| `spread_builder.py` | `parse_percent_beyond()`, `parse_max_spread_width()`, `parse_min_premium_diff()`, `calculate_option_price()`, `build_credit_spreads()` | Spread construction and option pricing |
| `backtest_engine.py` | `calculate_spread_pnl()`, `check_profit_target_hit()`, `find_option_at_timestamp()` | P&L calculation, profit target checking |
| `interval_analyzer.py` | `analyze_interval()`, `round_to_15_minutes()`, `parse_pst_timestamp()` | Core 15-minute interval analysis |
| `metrics.py` | `compute_metrics()`, `filter_top_n_per_day()`, `print_trading_statistics()`, `print_hourly_summary()`, `print_10min_block_breakdown()`, `generate_hourly_histogram()` | Statistics computation and reporting |
| `data_loader.py` | `find_csv_files_in_dir()`, `load_data_cached()`, `compute_cache_key()`, `clear_cache()`, `process_single_csv()`, `process_single_csv_sync()` | CSV loading, binary caching, parallel processing |
| `grid_search.py` | `run_grid_search()`, `run_backtest_with_params()`, `_build_strategy_for_combo()` | Grid search optimization engine |
| `output_formatter.py` | `format_best_current_option()`, `format_summary_line()`, `build_summary_parts()` | Display and printing utilities |
| `continuous_runner.py` | `run_continuous_analysis()`, `_run_single_analysis_iteration()` | Continuous live monitoring mode |
| `capital_utils.py` | `calculate_position_capital()`, `filter_results_by_capital_limit()` | Capital lifecycle management |
| `price_utils.py` | `get_current_day_close_price()`, `get_previous_close_price()`, `get_previous_open_price()`, `get_current_day_open_price()`, `get_price_at_time()` | Price fetching from QuestDB |
| `timezone_utils.py` | `resolve_timezone()`, `format_timestamp()`, `normalize_timestamp()` | Timezone handling and conversion |
| `rate_limiter.py` | `SlidingWindowRateLimiter` | Sliding window rate limiting |
| `time_block_rate_limiter.py` | `TimeBlockRateLimiter` | Time-block based rate limiting |
| `dynamic_width_utils.py` | Dynamic spread width calculation | Width based on strike distance |
| `scale_in_utils.py` | `ScaleInConfig`, `initialize_scale_in_trade()`, `calculate_layered_pnl()`, `process_price_update()`, `generate_scale_in_summary()`, `load_scale_in_config()` | Scale-in on breach logic |
| `delta_utils.py` | `DeltaFilterConfig`, `parse_delta_range()`, `format_delta_filter_info()` | Delta calculation, Black-Scholes |
| `tiered_investment_utils.py` | `TieredInvestmentConfig`, `initialize_tiered_trade()`, `calculate_all_tiers_pnl()`, `generate_tiered_summary()`, `load_tiered_config()` | Tiered investment logic |
| `strategies/__init__.py` | Package imports | Triggers auto-registration of strategies |
| `strategies/base.py` | `BaseStrategy`, `StrategyConfig`, `StrategyResult` | Abstract base classes |
| `strategies/registry.py` | `StrategyRegistry` | Strategy lookup by name |
| `strategies/single_entry.py` | `SingleEntryStrategy` | Default best-spread-per-interval |
| `strategies/scale_in_strategy.py` | `ScaleInStrategy` | Layered entry on breach |
| `strategies/tiered_strategy.py` | `TieredStrategy` | Multi-tier position management |

---

## Daily Workflow Commands

All commands run from the project root (`stocks/` directory).

### Backtesting (Historical Analysis)

**Standard backtest - NDX, configurable date range:**

```bash
python scripts/analyze_credit_spread_intervals.py \
    --csv-dir options_csv_output --ticker NDX \
    --start-date 2025-11-01 --end-date 2026-02-07 \
    --percent-beyond 0.005:0.015 --max-spread-width 20:30 \
    --risk-cap 500000 --profit-target-pct 0.80 \
    --min-trading-hour 6 --max-trading-hour 12 \
    --output-timezone America/Los_Angeles --summary
```

**Backtest with dynamic spread widths (Linear-500):**

```bash
python scripts/analyze_credit_spread_intervals.py \
    --csv-dir options_csv_output --ticker NDX \
    --start-date 2025-11-01 \
    --percent-beyond 0.005:0.015 --max-spread-width 50 \
    --dynamic-spread-width '{"mode":"linear","base_width":15,"slope_factor":500,"min_width":10,"max_width":50}' \
    --risk-cap 500000 --profit-target-pct 0.80 --summary
```

**Backtest with scale-in strategy:**

```bash
python scripts/analyze_credit_spread_intervals.py \
    --csv-dir options_csv_output --ticker NDX \
    --start-date 2025-11-01 \
    --percent-beyond 0.025:0.026 --max-spread-width 25 \
    --scale-in-enabled --scale-in-config scripts/scale_in_config_ndx.json \
    --risk-cap 500000 --summary
```

**Backtest with tiered investment strategy:**

```bash
python scripts/analyze_credit_spread_intervals.py \
    --csv-dir options_csv_output --ticker NDX \
    --start-date 2025-11-01 \
    --percent-beyond 0.02 --max-spread-width 50 \
    --tiered-enabled --tiered-config scripts/tiered_config_ndx.json \
    --summary
```

**Backtest with delta filtering (VIX1D-based IV):**

```bash
python scripts/analyze_credit_spread_intervals.py \
    --csv-dir options_csv_output --ticker NDX \
    --start-date 2025-11-01 \
    --percent-beyond 0.005 --max-spread-width 50 \
    --max-short-delta 0.15 --use-vix1d \
    --option-type put --summary
```

### Grid Search (Parameter Optimization)

**Multi-timeframe grid search (run each timeframe separately):**

```bash
# 1-week window
python scripts/analyze_credit_spread_intervals.py \
    --grid-config scripts/grid_config_ndx_1wk_100pct.json \
    --grid-output scripts/ndx_1wk_100pct_results.csv \
    --grid-sort net_pnl --grid-top-n 20 --log-level WARNING

# 2-week window
python scripts/analyze_credit_spread_intervals.py \
    --grid-config scripts/grid_config_ndx_2wk_100pct.json \
    --grid-output scripts/ndx_2wk_100pct_results.csv \
    --grid-sort net_pnl --grid-top-n 20 --log-level WARNING

# 1-month, 3-month, 6-month: same pattern with corresponding grid configs
```

**Delta grid search (NDX puts, delta 0.01-0.20):**

```bash
python scripts/analyze_credit_spread_intervals.py \
    --grid-config scripts/ndx_optimal_puts_config.json \
    --grid-output scripts/ndx_delta_puts_results.csv \
    --grid-sort net_pnl --grid-top-n 20 --log-level WARNING --no-data-cache
```

**Resume an interrupted grid search:**

```bash
python scripts/analyze_credit_spread_intervals.py \
    --grid-config scripts/grid_config_ndx_1mo_100pct.json \
    --grid-output scripts/ndx_1mo_100pct_results.csv \
    --grid-resume --grid-sort net_pnl --grid-top-n 20
```

**Grid search with strategy framework:**

```bash
python scripts/analyze_credit_spread_intervals.py \
    --grid-config scripts/grid_config_with_strategy.json \
    --grid-output scripts/strategy_grid_results.csv \
    --grid-sort net_pnl --grid-top-n 20
```

Grid config with strategy section:

```json
{
  "strategy": {"name": "tiered", "config_file": "tiered_config_ndx.json"},
  "fixed_params": {
    "csv_dir": "options_csv_output",
    "underlying_ticker": "NDX",
    "max_spread_width": "50",
    "start_date": "2025-11-05",
    "end_date": "2026-02-02"
  },
  "grid_params": {
    "strategy.feature_flags.greedy_t3_first": [true, false],
    "percent_beyond": [0.015, 0.020, 0.025]
  }
}
```

### Continuous Mode (Live Monitoring)

**Live monitoring during market hours:**

```bash
python scripts/analyze_credit_spread_intervals.py \
    --csv-dir options_csv_output --ticker NDX \
    --percent-beyond 0.005:0.015 --max-spread-width 20:30 \
    --risk-cap 500000 --profit-target-pct 0.80 \
    --continuous 10 --most-recent --best-only \
    --curr-price --use-market-hours \
    --output-timezone America/Los_Angeles
```

**Continuous with auto-stop after N runs:**

```bash
python scripts/analyze_credit_spread_intervals.py \
    --csv-dir options_csv_output --ticker NDX \
    --percent-beyond 0.005:0.015 --max-spread-width 20:30 \
    --continuous 15 --most-recent --best-only \
    --curr-price --use-market-hours \
    --run-once-before-wait --continuous-max-runs 100
```

---

## Strategy Framework

### Overview

The strategy framework provides a pluggable architecture for credit spread entry
and position management. It replaces hard-coded strategy logic with an abstract
interface that allows multiple trading strategies to coexist under a single
backtest and grid search engine.

### Strategy Lifecycle

Every strategy follows the same lifecycle:

```
select_entries(day_results, prev_close, option_type, **kwargs)
    -> List[Dict]  (positions)

calculate_pnl(positions, close_price, **kwargs)
    -> StrategyResult  (aggregated P&L with per-position detail)
```

1. **`select_entries()`** receives the interval analysis results for one trading
   day and returns a list of position dictionaries.

2. **`calculate_pnl()`** receives those positions along with the closing price
   and returns a `StrategyResult` dataclass.

### Auto-Registration

Concrete strategies register themselves at import time using the
`@StrategyRegistry.register` class decorator:

```python
from .registry import StrategyRegistry

@StrategyRegistry.register
class MyStrategy(BaseStrategy):
    @property
    def name(self) -> str:
        return "my_strategy"
    ...
```

### Core Data Classes

**`StrategyConfig`** -- holds `enabled` (bool) and `feature_flags` (dict).
Provides `get_flag(name, default)`, `has_flag(name)`, `to_dict()`, `from_dict()`.

**`StrategyResult`** -- returned by `calculate_pnl()`:

| Field | Type | Description |
|-------|------|-------------|
| `strategy_name` | `str` | Name of the strategy |
| `trading_date` | `Optional[datetime]` | Date of the trading day |
| `option_type` | `str` | `"call"` or `"put"` |
| `total_credit` | `float` | Sum of all position credits |
| `total_max_loss` | `float` | Sum of all position max losses |
| `total_pnl` | `Optional[float]` | Net P&L across all positions |
| `positions` | `List[Dict]` | Per-position detail |
| `metadata` | `Dict[str, Any]` | Optional additional data |

### Available Strategies

#### single_entry

Default strategy. Selects the single best spread per interval based on highest
net credit. No external config required.

```bash
python scripts/analyze_credit_spread_intervals.py \
    --csv-dir options_csv_output --ticker NDX \
    --percent-beyond 0.02 --max-spread-width 50 \
    --strategy single_entry --option-type put --summary
```

#### scale_in

Layered entry on breach. Positions enter in layers (L1/L2/L3) as the underlying
price breaches previous layer strikes. Requires `config_file` pointing to a
scale-in JSON configuration.

Feature flags: `aggressive_l1` (placeholder for tighter L1 percent_beyond).

```bash
python scripts/analyze_credit_spread_intervals.py \
    --csv-dir options_csv_output --ticker NDX \
    --percent-beyond 0.02 --max-spread-width 50 \
    --strategy scale_in \
    --strategy-config '{"config_file": "scripts/scale_in_config_ndx.json"}' \
    --option-type put --summary
```

#### tiered

Multi-tier position management. Multiple tiers activate simultaneously, each
with its own percent_beyond, spread width, and contract count. Requires
`config_file` pointing to a tiered investment JSON configuration.

Feature flags: `greedy_t3_first` (reverse tier order), `wait_for_slope`
(placeholder for slope-aware entry).

```bash
python scripts/analyze_credit_spread_intervals.py \
    --csv-dir options_csv_output --ticker NDX \
    --percent-beyond 0.02 --max-spread-width 50 \
    --strategy tiered \
    --strategy-config '{"config_file": "scripts/tiered_config_ndx.json"}' \
    --option-type put --summary
```

### Backward Compatibility

Legacy CLI flags continue to work:

| Legacy Flag | Equivalent Strategy Framework Usage |
|-------------|-------------------------------------|
| `--scale-in-enabled --scale-in-config file.json` | `--strategy scale_in --strategy-config '{"config_file": "file.json"}'` |
| `--tiered-enabled --tiered-config file.json` | `--strategy tiered --strategy-config '{"config_file": "file.json"}'` |
| `--scale-in-summary-only` | Still works alongside `--strategy scale_in` |
| `--tiered-summary-only` | Still works alongside `--strategy tiered` |

---

## Grid Search

The grid search engine (`grid_search.py`) runs parameter sweeps across any
combination of backtest parameters. It supports:

- Parallel execution via multiprocessing
- Resume from existing results (skip computed combos)
- Strategy parameterization including feature flags
- CSV output for analysis

### Grid Config Format

```json
{
  "fixed_params": {
    "csv_dir": "options_csv_output",
    "underlying_ticker": "NDX",
    "percent_beyond": "0.005",
    "max_spread_width": "50",
    "risk_cap": 5000,
    "max_credit_width_ratio": 0.60,
    "max_trading_hour": 15,
    "min_trading_hour": 7,
    "option_type": "put",
    "use_vix1d": true,
    "start_date": "2025-11-05",
    "end_date": "2026-02-02"
  },
  "grid_params": {
    "max_short_delta": [0.01, 0.02, 0.05, 0.10, 0.15, 0.20]
  }
}
```

### Strategy-Aware Grid Search

Add a `"strategy"` section to enable strategy parameterization:

```json
{
  "strategy": {
    "name": "tiered",
    "config_file": "tiered_config_ndx.json"
  },
  "grid_params": {
    "strategy.feature_flags.greedy_t3_first": [true, false],
    "percent_beyond": [0.015, 0.020, 0.025]
  }
}
```

Keys prefixed with `strategy.feature_flags.` are extracted per combo and merged
into the strategy config's feature flags before creating the strategy instance.

---

## Feature Flags Reference

| Strategy | Flag | Type | Default | Description |
|----------|------|------|---------|-------------|
| `scale_in` | `aggressive_l1` | `bool` | `false` | Placeholder for tighter L1 percent_beyond |
| `tiered` | `greedy_t3_first` | `bool` | `false` | Reverse tier evaluation order (T3 first) |
| `tiered` | `wait_for_slope` | `bool` | `false` | Placeholder for slope-aware entry logic |

Feature flags are read via `StrategyConfig.get_flag()` and applied in each
strategy's `apply_feature_flags()` method.

---

## Configuration

### CLI Arguments

The strategy framework adds two arguments (in `arg_parser.py`):

```
--strategy {single_entry,scale_in,tiered}
    Strategy to use. Default: None (legacy code paths).

--strategy-config path/to/config.json
    Path to JSON strategy configuration with feature flags.
```

### JSON Strategy Config Schema

```json
{
  "strategy": "tiered",
  "enabled": true,
  "feature_flags": {"greedy_t3_first": true},
  "config_file": "tiered_config_ndx.json"
}
```

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `strategy` | `str` | No | Strategy name (informational) |
| `enabled` | `bool` | No | Enable/disable. Default: `true` |
| `feature_flags` | `dict` | No | Key-value pairs. Default: `{}` |
| `config_file` | `str` | Conditional | Path to strategy-specific config. Required for `scale_in` and `tiered` |

### Scale-In Config Fields

| Field | Type | Description |
|-------|------|-------------|
| `enabled` | `bool` | Master enable switch |
| `total_capital` | `float` | Total capital across all layers |
| `spread_width` | `float` | Fixed spread width for all layers |
| `min_time_between_layers_minutes` | `int` | Min gap between layer activations |
| `layers.{put\|call}[].level` | `int` | Layer number (1, 2, 3) |
| `layers.{put\|call}[].percent_beyond` | `float` | Percent beyond close |
| `layers.{put\|call}[].capital_pct` | `float` | Fraction of total_capital |
| `layers.{put\|call}[].trigger` | `str` | `"entry"`, `"L1_breach"`, `"L2_breach"` |

### Tiered Config Fields

| Field | Type | Description |
|-------|------|-------------|
| `enabled` | `bool` | Master enable switch |
| `tiers.{put\|call}[].level` | `int` | Tier number (1, 2, 3, ...) |
| `tiers.{put\|call}[].percent_beyond` | `float` | Percent beyond close |
| `tiers.{put\|call}[].num_contracts` | `int` | Contract count |
| `tiers.{put\|call}[].spread_width` | `float` | Spread width (strike distance) |

---

## Adding a New Strategy

### Step 1: Create the Strategy File

Create `scripts/credit_spread_utils/strategies/my_strategy.py`:

```python
from typing import Any, Dict, List, Optional
from .base import BaseStrategy, StrategyConfig, StrategyResult
from .registry import StrategyRegistry

@StrategyRegistry.register
class MyStrategy(BaseStrategy):
    @property
    def name(self) -> str:
        return "my_strategy"

    def validate_config(self) -> bool:
        return True

    def select_entries(self, day_results, prev_close, option_type, **kw):
        return []

    def calculate_pnl(self, positions, close_price, **kw):
        return StrategyResult(
            strategy_name=self.name, option_type=kw.get('option_type', 'put'),
            total_credit=0, total_max_loss=0, total_pnl=0, positions=[])

    @classmethod
    def from_json(cls, config_dict, logger=None):
        config = StrategyConfig.from_dict(config_dict) if config_dict else StrategyConfig()
        return cls(config=config, logger=logger)
```

### Step 2: Import in `strategies/__init__.py`

```python
from . import my_strategy
```

### Step 3: Add to `arg_parser.py` choices

Update `--strategy` choices in `_add_strategy_args()`:

```python
choices=['single_entry', 'scale_in', 'tiered', 'my_strategy'],
```

### Step 4: Update this document

Add the strategy to the Available Strategies section and Feature Flags table.

---

## Supporting Scripts

These scripts are **separate** from the main analyzer and were not affected by
the modularization. They produce complementary analysis results.

| Script | Purpose | Example |
|--------|---------|---------|
| `scripts/fetch_index_prices.py` | Close-to-close price movements, intraday extremes, 52-week range tables | `python scripts/fetch_index_prices.py --ticker NDX --period 6mo` |
| `scripts/ndx_risk_gradient_analysis.py` | Risk gradient analysis (safe points, offset levels) | `python scripts/ndx_risk_gradient_analysis.py` |
| `scripts/ndx_daily_calculator.py` | Daily strike level calculator for live trading | `python scripts/ndx_daily_calculator.py` |

---

## Change Log

Track all significant changes here. Format: `YYYY-MM-DD: Description`.

- **2026-02-07**: Modularization complete. Extracted 8 modules from monolithic
  5,307-line file into `credit_spread_utils/` package. Created strategy
  framework with `strategies/` subpackage (BaseStrategy ABC, StrategyRegistry,
  SingleEntryStrategy, ScaleInStrategy, TieredStrategy). Wired strategy
  framework into main pipeline and grid search. Main file reduced to 1,505
  lines. Added daily workflow commands to arg_parser.py epilog. Created this
  architecture document.
