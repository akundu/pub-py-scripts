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
10. [Weekly Data Generation Commands](#weekly-data-generation-commands)
11. [Supporting Scripts](#supporting-scripts)
12. [Change Log](#change-log)

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
│   ├── analyze_credit_spread_intervals.py    # Main entry point (~1,520 lines)
│   ├── analyze_price_movements.py            # Thin wrapper → --mode price-movements
│   ├── ndx_max_move_analysis.py              # Thin wrapper → --mode max-move
│   ├── ndx_risk_gradient_analysis.py         # Thin wrapper → --mode risk-gradient
│   ├── fetch_index_prices.py                 # Current index prices (separate)
│   ├── ndx_daily_calculator.py               # Daily strike calculator (separate)
│   ├── grid_config_*.json                    # Grid search configurations
│   ├── scale_in_config_*.json                # Scale-in strategy configs
│   ├── tiered_config_*.json                  # Tiered strategy configs
│   └── credit_spread_utils/                  # Utility modules package
│       ├── __init__.py                       # Package init, exports all modules
│       ├── arg_parser.py                     # CLI argument parsing (~800 lines)
│       ├── backtest_engine.py                # P&L calculation, profit targets
│       ├── capital_utils.py                  # Capital lifecycle management
│       ├── continuous_runner.py              # Continuous analysis mode
│       ├── data_loader.py                    # CSV loading, binary caching
│       ├── delta_utils.py                    # Delta calculation, BS model
│       ├── dynamic_width_utils.py            # Dynamic spread width calculation
│       ├── grid_search.py                    # Grid search optimization engine
│       ├── interval_analyzer.py              # Core interval analysis
│       ├── max_move_utils.py                 # Intraday extreme movement tables
│       ├── metrics.py                        # Trading statistics, reporting
│       ├── output_formatter.py               # Display and printing utilities
│       ├── price_movement_utils.py           # Close-to-close / time-to-close movements
│       ├── price_utils.py                    # Price fetching from QuestDB
│       ├── rate_limiter.py                   # Sliding window rate limiting
│       ├── risk_gradient_utils.py            # Risk gradient analysis + config gen
│       ├── scale_in_utils.py                 # Scale-in on breach logic
│       ├── spread_builder.py                 # Spread building, option pricing
│       ├── tiered_investment_utils.py        # Tiered investment logic
│       ├── time_allocated_tiered_utils.py    # Time-allocated tiered logic
│       ├── predictor_tier_adapter.py         # Predictor confidence → tier adjustment
│       ├── close_predictor_gate.py           # Close predictor risk gate
│       ├── time_block_rate_limiter.py        # Time-block rate limiting
│       ├── timezone_utils.py                 # Timezone handling
│       └── strategies/                       # Strategy framework package
│           ├── __init__.py                   # Imports concrete strategies
│           ├── base.py                       # BaseStrategy ABC, StrategyConfig
│           ├── registry.py                   # StrategyRegistry
│           ├── single_entry.py               # SingleEntryStrategy (default)
│           ├── scale_in_strategy.py          # ScaleInStrategy
│           ├── tiered_strategy.py            # TieredStrategy
│           └── time_allocated_tiered_strategy.py  # TimeAllocatedTieredStrategy
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
| `price_movement_utils.py` | `load_ticker_data()`, `calculate_movements()`, `calculate_statistics()`, `print_statistics()`, `generate_histogram()`, `run_price_movement_analysis()` | Close-to-close and time-to-close price movement analysis |
| `max_move_utils.py` | `load_csv_data()`, `get_available_dates()`, `get_day_close()`, `get_previous_close()`, `analyze_max_moves()`, `print_tables()`, `run_max_move_analysis()` | Intraday extreme movement tables by 30-min time slots |
| `risk_gradient_utils.py` | `query_historical_max_movements()`, `generate_risk_gradient_values()`, `create_grid_config()`, `run_risk_gradient_analysis()` | Risk gradient analysis with QuestDB + grid config generation |
| `time_allocated_tiered_utils.py` | `TimeAllocatedTieredConfig`, `allocate_across_windows()`, `calculate_time_allocated_pnl()`, `generate_time_allocated_summary()`, `check_slope_flattened()` | Time-allocated tiered investment with hourly windows, slope detection |
| `strategies/time_allocated_tiered_strategy.py` | `TimeAllocatedTieredStrategy` | Time-allocated tiered strategy wrapper |
| `close_predictor_gate.py` | `ClosePredictorGate`, `ClosePredictorGateConfig` | Close predictor risk gate for filtering/annotating spreads |
| `predictor_tier_adapter.py` | `compute_predictor_adjustment()`, `apply_adjustment_to_tiers()`, `validate_strikes_against_bands()` | Adapter: prediction confidence → tier ROI/budget adjustments |

---

## Analysis Modes (`--mode`)

The main entry point supports multiple analysis modes via the `--mode` argument.
When `--mode` is not `credit-spread`, the existing validation (requiring
`--percent-beyond`, etc.) is bypassed and mode-specific arguments are used.

| Mode | Description | Replaces |
|------|-------------|----------|
| `credit-spread` | Default. Existing credit spread analysis. | (current default) |
| `price-movements` | Close-to-close or time-to-close price movement statistics | `analyze_price_movements.py` |
| `max-move` | Intraday extreme movement tables (30-min slots) | `ndx_max_move_analysis.py` |
| `risk-gradient` | Risk gradient analysis from historical safe points | `ndx_risk_gradient_analysis.py` |

### Price Movements Mode

```bash
# Close-to-close analysis
python scripts/analyze_credit_spread_intervals.py \
    --mode price-movements --ticker I:NDX --no-plot

# Time-to-close analysis (from 11:30 AM EST)
python scripts/analyze_credit_spread_intervals.py \
    --mode price-movements --ticker I:NDX --from-time 11:30 --pm-timezone EST --no-plot

# Down days only
python scripts/analyze_credit_spread_intervals.py \
    --mode price-movements --ticker I:NDX --no-plot --day-direction down
```

### Max Move Mode

```bash
# NDX 125-day analysis
python scripts/analyze_credit_spread_intervals.py \
    --mode max-move --ticker NDX --days 125

# SPX 60-day analysis
python scripts/analyze_credit_spread_intervals.py \
    --mode max-move --ticker SPX --days 60
```

### Risk Gradient Mode

```bash
# Generate configs only
python scripts/analyze_credit_spread_intervals.py \
    --mode risk-gradient --ticker NDX --lookback-days 90 180 --generate-config-only

# Full analysis with backtest
python scripts/analyze_credit_spread_intervals.py \
    --mode risk-gradient --ticker NDX --lookback-days 90 180 --run-backtest --processes 8
```

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

**Continuous with time-allocated tiered strategy and close predictor integration:**

```bash
python scripts/analyze_credit_spread_intervals.py \
    --csv-path ./csv_exports/options/NDX/$(date +%Y-%m-%d).csv \
    --option-type both --percent-beyond 0.02 --max-spread-width 55:55 \
    --underlying-ticker NDX --db-path $QUEST_DB_STRING \
    --risk-cap 35000 --min-contract-price 1 --max-credit-width-ratio 0.6 \
    --max-strike-distance-pct 0.034 --min-trading-hour 6 --max-trading-hour 10 \
    --profit-target-pct 0.90 --no-data-cache --log-level WARNING --summary \
    --strategy time_allocated_tiered \
    --strategy-config scripts/json/ta_tiered_strategy_ndx_realtime.json \
    --continuous 60 --most-recent
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

#### time_allocated_tiered

Time-based capital allocation across hourly windows (6:30am-9:30am PST) with
tier priority deployment (T3->T2->T1), directional bias, slope-based entry
timing, and carry-forward of unused budget. Requires `config_file` pointing
to a time-allocated tiered JSON configuration.

**Close predictor integration:** When `close_predictor_integration.enabled` is
`true` in the config and a `ClosePredictorGate` is available (via continuous
mode's `--close-predictor-gate` or strategy kwargs), the strategy dynamically
adjusts ROI thresholds and budget allocation per window based on the
prediction's confidence level and band width. HIGH confidence (narrow bands,
later in day) lowers ROI thresholds for more aggressive deployment; LOW
confidence (wide bands, early morning) raises them for conservative deployment.
Additionally, `band_strike_validation` can reject positions whose short strikes
fall inside the predicted close band.

**Continuous mode support:** The `--strategy time_allocated_tiered` flag now
works with `--continuous`, executing the strategy on each iteration's results.

Feature flags: `carry_forward_unused` (enable/disable carry-forward),
`require_double_flatten` (require 2 consecutive flat bars),
`vix_dynamic_thresholds` (placeholder for VIX-based ROI scaling).

**Slope detection behavior:** When slope detection is enabled, the strategy
checks each interval's 5-minute equity bars for momentum flattening. Only
intervals from the flattened point onward are used for spread selection
(post-flattened intervals). If there are insufficient lookback bars (common
at the start of a window), the interval is treated as "not flattened" — the
strategy waits for more data rather than deploying immediately. When the
intraday DataFrame is completely absent (no equities data at all), the
interval is treated as flattened to avoid blocking.

**Window start offset:** Each hourly window supports a `start_minute_pst`
field (default 0) to offset the start within the hour. The default NDX
configs use `start_minute_pst: 30` for the 6am window, meaning deployment
begins at 6:30am PST (when equities data first becomes available).

```bash
python scripts/analyze_credit_spread_intervals.py \
    --csv-dir options_csv_output --ticker NDX \
    --start-date 2025-12-01 --end-date 2026-02-09 \
    --percent-beyond 0.02 --max-spread-width 60 \
    --strategy time_allocated_tiered \
    --strategy-config '{"config_file": "scripts/json/time_allocated_tiered_config_ndx.json"}' \
    --min-trading-hour 6 --max-trading-hour 10 \
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

**Config override aliasing:** Grid search JSON uses `slope_detection` as the
key name (matching the JSON config structure), but the Python dataclass field
is `slope_config`. The grid search engine maps `slope_detection` ->
`slope_config` automatically via `attr_aliases` in `_apply_config_overrides()`.

### Time-Allocated Tiered Grid Search (Slope Comparison)

Sweep tier spread widths while comparing slope detection ON vs OFF. Uses
`linked_params` to keep put/call widths in sync:

```json
{
  "description": "NDX time-allocated tiered - width sweep with slope comparison",
  "fixed_params": {
    "csv_dir": "options_csv_output",
    "underlying_ticker": "NDX",
    "start_date": "2025-07-01",
    "end_date": "2026-01-31",
    "option_type": "both",
    "min_spread_width": 5,
    "max_spread_width": "200:200"
  },
  "strategy": {
    "strategy": "time_allocated_tiered",
    "enabled": true,
    "config_file": "scripts/json/time_allocated_tiered_config_ndx_permissive.json"
  },
  "grid_params": {
    "strategy.config.slope_detection.skip_slope": [false, true],
    "strategy.config.tiers.put.0.spread_width": {"min": 30, "max": 60, "step": 5},
    "strategy.config.tiers.put.1.spread_width": {"min": 20, "max": 50, "step": 5},
    "strategy.config.tiers.put.2.spread_width": {"min": 10, "max": 30, "step": 5}
  },
  "linked_params": {
    "strategy.config.tiers.call.0.spread_width": "strategy.config.tiers.put.0.spread_width",
    "strategy.config.tiers.call.1.spread_width": "strategy.config.tiers.put.1.spread_width",
    "strategy.config.tiers.call.2.spread_width": "strategy.config.tiers.put.2.spread_width"
  }
}
```

Run example:

```bash
python scripts/analyze_credit_spread_intervals.py \
    --grid-config scripts/json/grid_config_ndx_ta_tiered_width_sweep.json \
    --grid-output scripts/csv/ndx_ta_width_sweep_results.csv \
    --grid-sort net_pnl --grid-top-n 20 --log-level WARNING --processes 14
```

---

## Feature Flags Reference

| Strategy | Flag | Type | Default | Description |
|----------|------|------|---------|-------------|
| `scale_in` | `aggressive_l1` | `bool` | `false` | Placeholder for tighter L1 percent_beyond |
| `tiered` | `greedy_t3_first` | `bool` | `false` | Reverse tier evaluation order (T3 first) |
| `tiered` | `wait_for_slope` | `bool` | `false` | Placeholder for slope-aware entry logic |
| `time_allocated_tiered` | `carry_forward_unused` | `bool` | `true` | Enable/disable carry-forward of unused budget between windows |
| `time_allocated_tiered` | `require_double_flatten` | `bool` | `false` | Require 2 consecutive bars of slope flattening before entry |
| `time_allocated_tiered` | `vix_dynamic_thresholds` | `bool` | `false` | Placeholder: scale ROI thresholds by VIX level |

Feature flags are read via `StrategyConfig.get_flag()` and applied in each
strategy's `apply_feature_flags()` method.

---

## Configuration

### CLI Arguments

The strategy framework adds two arguments (in `arg_parser.py`):

```
--strategy {single_entry,scale_in,tiered,time_allocated_tiered}
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

### Time-Allocated Tiered Config Fields

| Field | Type | Description |
|-------|------|-------------|
| `enabled` | `bool` | Master enable switch |
| `total_capital` | `float` | Total capital across all windows |
| `ticker` | `str` | Underlying ticker for equities data (e.g., `"I:NDX"`) |
| `equities_dir` | `str` | Directory for 5-min equity bar CSVs |
| `carry_forward_decay` | `float` | Decay factor for unused budget carry-forward (0.0-1.0) |
| `direction_priority_split` | `float` | Capital split favoring the directional side (e.g., 0.70) |
| `max_concurrent_exposure` | `float\|null` | Max total exposure cap (null = unlimited) |
| `hourly_windows[].label` | `str` | Window label (e.g., `"6am"`, `"7am"`) |
| `hourly_windows[].start_hour_pst` | `int` | Window start hour in PST |
| `hourly_windows[].start_minute_pst` | `int` | Minute offset within start hour (default 0, use 30 for 6:30am) |
| `hourly_windows[].end_hour_pst` | `int` | Window end hour in PST |
| `hourly_windows[].end_minute_pst` | `int` | Minute offset within end hour (0 for full hours, 30 for 9:30) |
| `hourly_windows[].budget_pct` | `float\|"remainder"` | Fraction of total_capital (last window uses `"remainder"`) |
| `tiers.{put\|call}[].level` | `int` | Tier number (1, 2, 3) |
| `tiers.{put\|call}[].percent_beyond` | `float` | Percent beyond close |
| `tiers.{put\|call}[].spread_width` | `float` | Spread width (strike distance) |
| `tiers.{put\|call}[].roi_threshold` | `float` | Minimum ROI to deploy this tier |
| `tiers.{put\|call}[].max_cumulative_budget_pct` | `float` | Max cumulative budget fraction (T3=0.65, T2=0.95, T1=1.0) |
| `slope_detection.lookback_bars` | `int` | Number of 5-min bars for slope averaging (default 5) |
| `slope_detection.flatten_ratio_threshold` | `float` | Max \|instant/avg\| ratio to consider flattened (default 0.4) |
| `slope_detection.min_directional_move_pct` | `float` | Below this, market is "flat" (default 0.0005) |
| `slope_detection.require_double_flatten` | `bool` | Require 2 consecutive flat bars (default false) |
| `slope_detection.skip_slope` | `bool` | Skip slope detection entirely (default false) |
| `profit_targets.{label}` | `float` | Profit target percentage per window (e.g., `"6am": 0.60`) |
| `close_predictor_integration.enabled` | `bool` | Enable predictor-based tier adjustment (default false) |
| `close_predictor_integration.band_level` | `str` | Which band to check: P95/P97/P98/P99 (default P95) |
| `close_predictor_integration.confidence_roi_multipliers` | `dict` | Confidence -> ROI multiplier map (HIGH=0.70, MEDIUM=0.90, LOW=1.10, VERY_LOW=1.40) |
| `close_predictor_integration.budget_scale_clamp` | `[float, float]` | Min/max clamp for budget scaling (default [0.5, 1.5]) |
| `close_predictor_integration.time_of_day_penalties` | `dict` | Window label -> confidence penalty (6am=0.15, 7am=0.10, 8am=0.05, 9am=0.0) |
| `close_predictor_integration.band_strike_validation` | `bool` | Reject positions with short strike inside predicted band (default true) |

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
choices=['single_entry', 'scale_in', 'tiered', 'time_allocated_tiered', 'my_strategy'],
```

### Step 4: Update this document

Add the strategy to the Available Strategies section and Feature Flags table.

---

## Weekly Data Generation Commands

These commands regenerate the analyses captured in `scripts/summarized_view.txt`.
Run weekly (e.g., every Friday after close) from the project root (`stocks/`).

### Date Setup

```bash
# macOS date arithmetic for rolling windows
END_DATE=$(date +%Y-%m-%d)
START_1WK=$(date -v-7d +%Y-%m-%d)
START_2WK=$(date -v-14d +%Y-%m-%d)
START_1MO=$(date -v-1m +%Y-%m-%d)
START_3MO=$(date -v-3m +%Y-%m-%d)
START_6MO=$(date -v-6m +%Y-%m-%d)
```

### 1. Close-to-Close Price Movement Statistics

Produces percentile tables (P95/P97/P99/P100) for daily price movements.

```bash
# NDX - all days, down days only, up days only (using --mode)
python scripts/analyze_credit_spread_intervals.py --mode price-movements --ticker I:NDX --no-plot
python scripts/analyze_credit_spread_intervals.py --mode price-movements --ticker I:NDX --no-plot --day-direction down
python scripts/analyze_credit_spread_intervals.py --mode price-movements --ticker I:NDX --no-plot --day-direction up

# SPX - same set
python scripts/analyze_credit_spread_intervals.py --mode price-movements --ticker I:SPX --no-plot
python scripts/analyze_credit_spread_intervals.py --mode price-movements --ticker I:SPX --no-plot --day-direction down
python scripts/analyze_credit_spread_intervals.py --mode price-movements --ticker I:SPX --no-plot --day-direction up

# Legacy thin wrapper still works:
# python scripts/analyze_price_movements.py --ticker I:NDX --no-plot
```

### 2. Last N Minutes to Close Analysis

Measures how much the index moves in the final 5-10 minutes before close.

```bash
# NDX - last 10 min and last 5 min
python scripts/analyze_credit_spread_intervals.py --mode price-movements --ticker I:NDX --from-time 15:50 --pm-timezone EST --no-plot
python scripts/analyze_credit_spread_intervals.py --mode price-movements --ticker I:NDX --from-time 15:55 --pm-timezone EST --no-plot

# SPX - same
python scripts/analyze_credit_spread_intervals.py --mode price-movements --ticker I:SPX --from-time 15:50 --pm-timezone EST --no-plot
python scripts/analyze_credit_spread_intervals.py --mode price-movements --ticker I:SPX --from-time 15:55 --pm-timezone EST --no-plot

# Legacy thin wrapper still works:
# python scripts/analyze_price_movements.py --ticker I:NDX --from-time 15:50 --timezone EST --no-plot
```

### 3. Intraday Extreme Movement Tables (30-min slots)

Shows max upside/downside movement from each half-hour slot to close, split by
days above vs below previous close. Produces P95-P100 tables.

```bash
python scripts/analyze_credit_spread_intervals.py --mode max-move --ticker NDX --days 125
python scripts/analyze_credit_spread_intervals.py --mode max-move --ticker SPX --days 125

# Legacy thin wrapper still works:
# python scripts/ndx_max_move_analysis.py --ticker NDX --days 125
```

### 4. Risk Gradient Analysis

Starts from the "zero risk" point (100th percentile price movement) and tests
shifting toward the money in steps to map the risk/reward gradient.

```bash
python scripts/analyze_credit_spread_intervals.py \
    --mode risk-gradient --ticker NDX --lookback-days 90 180 --run-backtest --processes 8

python scripts/analyze_credit_spread_intervals.py \
    --mode risk-gradient --ticker SPX --lookback-days 90 180 --run-backtest --processes 8

# Legacy thin wrapper still works:
# python scripts/ndx_risk_gradient_analysis.py --ticker NDX --lookback-days 90 180 --run-backtest
```

### 5. Multi-Timeframe Grid Search (Parameter Convergence)

Sweeps percent_beyond, max_spread_width, and trading hours across 5 rolling
windows per ticker (1wk, 2wk, 1mo, 3mo, 6mo). Requires JSON config files with
rolling dates. Example for NDX 1-month:

```bash
cat > /tmp/grid_ndx_1mo.json << EOF
{
  "fixed_params": {
    "csv_dir": "options_csv_output", "underlying_ticker": "NDX",
    "risk_cap": 5000, "max_credit_width_ratio": 0.60,
    "start_date": "$START_1MO", "end_date": "$END_DATE"
  },
  "grid_params": {
    "percent_beyond": ["0.005","0.010","0.015","0.020","0.025","0.030","0.035"],
    "max_spread_width": ["10","15","20","25","30"],
    "min_trading_hour": [6,7,8,9],
    "max_trading_hour": [11,12,13,14,15]
  }
}
EOF
python scripts/analyze_credit_spread_intervals.py \
    --grid-config /tmp/grid_ndx_1mo.json \
    --grid-output scripts/ndx_1mo_100pct_results.csv \
    --grid-sort net_pnl --grid-top-n 20 --log-level WARNING
```

Repeat with `START_1WK`/`START_2WK`/`START_3MO`/`START_6MO` and matching output
filenames. Run same set for SPX (change `underlying_ticker` and output names).

**Full timeframe list per ticker (10 grid searches total):**

| Ticker | Window | Start Date Var | Output File |
|--------|--------|---------------|-------------|
| NDX | 1 week | `$START_1WK` | `ndx_1wk_100pct_results.csv` |
| NDX | 2 weeks | `$START_2WK` | `ndx_2wk_100pct_results.csv` |
| NDX | 1 month | `$START_1MO` | `ndx_1mo_100pct_results.csv` |
| NDX | 3 months | `$START_3MO` | `ndx_3mo_100pct_results.csv` |
| NDX | 6 months | `$START_6MO` | `ndx_6mo_100pct_results.csv` |
| SPX | 1 week | `$START_1WK` | `spx_1wk_100pct_results.csv` |
| SPX | 2 weeks | `$START_2WK` | `spx_2wk_100pct_results.csv` |
| SPX | 1 month | `$START_1MO` | `spx_1mo_100pct_results.csv` |
| SPX | 3 months | `$START_3MO` | `spx_3mo_100pct_results.csv` |
| SPX | 6 months | `$START_6MO` | `spx_6mo_100pct_results.csv` |

### 6. Delta Grid Search

Sweeps delta 0.01-0.20 to find optimal delta filter per instrument and side.
Four runs total (NDX calls, NDX puts, SPX calls, SPX puts).

```bash
# NDX Calls
cat > /tmp/grid_ndx_delta_calls.json << EOF
{
  "fixed_params": {
    "csv_dir": "options_csv_output", "underlying_ticker": "NDX",
    "percent_beyond": "0.005", "max_spread_width": "50",
    "risk_cap": 5000, "max_credit_width_ratio": 0.60,
    "max_trading_hour": 15, "min_trading_hour": 7,
    "option_type": "call", "use_vix1d": true,
    "start_date": "$START_3MO", "end_date": "$END_DATE"
  },
  "grid_params": {
    "max_short_delta": [0.01,0.02,0.03,0.04,0.05,0.06,0.07,0.08,0.09,0.10,
                        0.11,0.12,0.13,0.14,0.15,0.16,0.17,0.18,0.19,0.20]
  }
}
EOF
python scripts/analyze_credit_spread_intervals.py \
    --grid-config /tmp/grid_ndx_delta_calls.json \
    --grid-output scripts/ndx_delta_calls_results.csv \
    --grid-sort net_pnl --grid-top-n 20 --log-level WARNING --no-data-cache

# NDX Puts - same config but change "option_type": "put"
# SPX Calls - change "underlying_ticker": "SPX", "option_type": "call"
# SPX Puts  - change "underlying_ticker": "SPX", "option_type": "put"
```

**Full delta grid list (4 runs):**

| Config | Output File |
|--------|-------------|
| NDX + call | `ndx_delta_calls_results.csv` |
| NDX + put | `ndx_delta_puts_results.csv` |
| SPX + call | `spx_delta_calls_results.csv` |
| SPX + put | `spx_delta_puts_results.csv` |

### 7. Dynamic Spread Width Comparison

Compares fixed-width baseline vs Linear-500, Linear-1000, and Stepped configs.

```bash
# Baseline (fixed width 20:30)
python scripts/analyze_credit_spread_intervals.py \
    --csv-dir options_csv_output --ticker NDX \
    --start-date "$START_3MO" --end-date "$END_DATE" \
    --percent-beyond 0.005:0.015 --max-spread-width 20:30 \
    --risk-cap 300000 --profit-target-pct 0.80 \
    --min-trading-hour 9 --max-trading-hour 12 --summary-only

# Linear-500 (recommended)
python scripts/analyze_credit_spread_intervals.py \
    --csv-dir options_csv_output --ticker NDX \
    --start-date "$START_3MO" --end-date "$END_DATE" \
    --percent-beyond 0.005:0.015 --max-spread-width 50 \
    --dynamic-spread-width '{"mode":"linear","base_width":15,"slope_factor":500,"min_width":10,"max_width":50}' \
    --risk-cap 300000 --profit-target-pct 0.80 \
    --min-trading-hour 9 --max-trading-hour 12 --summary-only

# Linear-1000
python scripts/analyze_credit_spread_intervals.py \
    --csv-dir options_csv_output --ticker NDX \
    --start-date "$START_3MO" --end-date "$END_DATE" \
    --percent-beyond 0.005:0.015 --max-spread-width 50 \
    --dynamic-spread-width '{"mode":"linear","base_width":15,"slope_factor":1000,"min_width":10,"max_width":50}' \
    --risk-cap 300000 --profit-target-pct 0.80 \
    --min-trading-hour 9 --max-trading-hour 12 --summary-only

# Stepped
python scripts/analyze_credit_spread_intervals.py \
    --csv-dir options_csv_output --ticker NDX \
    --start-date "$START_3MO" --end-date "$END_DATE" \
    --percent-beyond 0.005:0.015 --max-spread-width 50 \
    --dynamic-spread-width '{"mode":"stepped","steps":[{"threshold":0.01,"width":15},{"threshold":0.02,"width":25},{"threshold":0.03,"width":35}]}' \
    --risk-cap 300000 --profit-target-pct 0.80 \
    --min-trading-hour 9 --max-trading-hour 12 --summary-only
```

Repeat all 4 for SPX (change `--ticker SPX`).

### Summary of Weekly Runs

| Category | NDX Runs | SPX Runs | Total |
|----------|----------|----------|-------|
| Price movement stats | 3 | 3 | 6 |
| Last N min to close | 2 | 2 | 4 |
| Intraday extremes | 1 | 1 | 2 |
| Risk gradient | 1 | 1 | 2 |
| Multi-timeframe grid | 5 | 5 | 10 |
| Delta grid | 2 | 2 | 4 |
| Dynamic width compare | 4 | 4 | 8 |
| **Total** | **18** | **18** | **36** |

---

## Supporting Scripts

These scripts complement the main analyzer. Scripts marked as **thin wrappers**
delegate to `credit_spread_utils/` and can also be invoked via `--mode`.

| Script | Purpose | Status | Example |
|--------|---------|--------|---------|
| `scripts/fetch_index_prices.py` | Current index prices (Yahoo Finance) | Standalone | `python scripts/fetch_index_prices.py --all` |
| `scripts/analyze_price_movements.py` | Close-to-close / time-to-close movements | Thin wrapper → `--mode price-movements` | `python scripts/analyze_price_movements.py --ticker I:NDX --no-plot` |
| `scripts/ndx_max_move_analysis.py` | Intraday extreme movement tables | Thin wrapper → `--mode max-move` | `python scripts/ndx_max_move_analysis.py --ticker NDX --days 125` |
| `scripts/ndx_risk_gradient_analysis.py` | Risk gradient analysis | Thin wrapper → `--mode risk-gradient` | `python scripts/ndx_risk_gradient_analysis.py --run-backtest` |
| `scripts/ndx_daily_calculator.py` | Daily strike level calculator for live trading | Standalone | `python scripts/ndx_daily_calculator.py --fetch` |

---

## Change Log

Track all significant changes here. Format: `YYYY-MM-DD: Description`.

- **2026-02-10**: Integrated close predictor confidence into time-allocated
  tiered strategy. Created `predictor_tier_adapter.py` with
  `compute_predictor_adjustment()`, `apply_adjustment_to_tiers()`, and
  `validate_strikes_against_bands()`. Added `ClosePredictorIntegrationConfig`
  dataclass to `time_allocated_tiered_utils.py` with per-window ROI multipliers,
  budget scaling, time-of-day penalties, and band strike validation. Updated
  `TimeAllocatedTieredStrategy.select_entries()` to dynamically adjust tier
  thresholds based on prediction confidence. Enabled `--strategy` flag in
  `--continuous` mode by extracting `_execute_strategy_on_results()` helper
  and passing strategy through `continuous_runner.py`. Added
  `close_predictor_integration` section to NDX realtime JSON config. Created
  comprehensive tests (`test_predictor_tier_adapter.py`, 30+ tests).
- **2026-02-09**: Fixed slope detection having zero effect in grid search.
  Three root causes fixed: (1) `check_slope_flattened()` returned
  `flattened=True` on insufficient lookback bars — now returns `False` so the
  strategy waits for enough data. (2) After slope flattened, all window
  intervals were used for spread selection — now only post-flattened intervals
  are used (`window_intervals[idx:]`). (3) Grid search `_apply_config_overrides()`
  couldn't map JSON key `slope_detection` to Python field `slope_config` —
  added `attr_aliases` mapping in `grid_search.py`. Also added
  `start_minute_pst` field to `HourlyWindowConfig` so 6am window can start at
  6:30am PST (when equities data becomes available). Updated all 5 JSON configs.
  Added time-allocated tiered config fields documentation.
- **2026-02-08**: Added `time_allocated_tiered` strategy with time-based
  capital allocation across hourly windows (6am-9:30am PST), tier priority
  deployment (T3->T2->T1), slope-based entry timing, directional bias, and
  carry-forward. Created `time_allocated_tiered_utils.py`,
  `strategies/time_allocated_tiered_strategy.py`,
  `json/time_allocated_tiered_config_ndx.json`. Extended `grid_search.py`
  with `strategy.config.*` key support for grid parameter overrides. Added
  comprehensive tests (`test_time_allocated_tiered.py`).
- **2026-02-08**: Consolidated 3 standalone analysis scripts into the main
  analyzer via `--mode` argument. Created `price_movement_utils.py`,
  `max_move_utils.py`, and `risk_gradient_utils.py` in `credit_spread_utils/`.
  Original scripts (`analyze_price_movements.py`, `ndx_max_move_analysis.py`,
  `ndx_risk_gradient_analysis.py`) converted to thin wrappers. Added
  `--mode price-movements|max-move|risk-gradient` to `arg_parser.py`. Added
  mode dispatch to `analyze_credit_spread_intervals.py`. Created tests for all
  3 new utility modules (65 tests). Updated Weekly Data Generation Commands to
  use `--mode` syntax. Updated Supporting Scripts table.
- **2026-02-07**: Modularization complete. Extracted 8 modules from monolithic
  5,307-line file into `credit_spread_utils/` package. Created strategy
  framework with `strategies/` subpackage (BaseStrategy ABC, StrategyRegistry,
  SingleEntryStrategy, ScaleInStrategy, TieredStrategy). Wired strategy
  framework into main pipeline and grid search. Main file reduced to 1,505
  lines. Added daily workflow commands to arg_parser.py epilog. Created this
  architecture document.
