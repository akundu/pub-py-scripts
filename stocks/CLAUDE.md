# Project: Stocks Analysis System

## Environment Configuration

### QuestDB Connection

**REQUIRED**: Set the QuestDB connection string for accessing realtime and historical data:

```bash
export QUEST_DB_STRING="questdb://stock_user:stock_password@lin1.kundu.dev:8812/stock_data"
```

**Usage Notes**:
- The `realtime_data` table contains tick-level price updates (updated every ~5 seconds)
- The daily tables are only populated after market close
- **Always use `realtime_data` for today's intraday data**, not daily tables
- Ticker format in `realtime_data`: Use `'NDX'`, `'SPX'` (no `'I:'` prefix)
- The prediction code automatically aggregates tick data into 5-minute OHLC bars

**Alternative Environment Variables** (fallback order):
1. `QUEST_DB_STRING` (preferred)
2. `QUESTDB_CONNECTION_STRING`
3. `QUESTDB_URL`

**Connection String Format**:
```
questdb://user:password@host:port/database
```

The code automatically converts `questdb://` to `postgresql://` for asyncpg compatibility.

## Command Line Interface Requirements

**All command-line programs MUST include comprehensive help screens.** Every script that accepts command line arguments must use `argparse` with:

1. **Description**: Clear explanation of what the program does
2. **Argument help**: Detailed help text for each parameter
3. **Examples section**: Show common usage patterns with `epilog`
4. **--help support**: Standard `-h`/`--help` flag

### Help Screen Template

```python
import argparse

parser = argparse.ArgumentParser(
    description='''
Clear description of what this program does.
Can span multiple lines.
    ''',
    epilog='''
Examples:
  %(prog)s --option1 value1
      Description of what this does

  %(prog)s --option2 value2 --option3
      Another example

  %(prog)s --help
      Show this help message
    ''',
    formatter_class=argparse.RawDescriptionHelpFormatter
)
parser.add_argument('--option', help='Clear description of this option')
```

### Good Examples in Codebase

- `scripts/backtest_close_predictor.py`: Hour-by-hour backtest with examples
- `scripts/compare_static_vs_dynamic_backtest.py`: Training approach comparison
- `predict_close.py`: Live prediction with hybrid approach

## Close Price Prediction Models

### Model Architecture

The system uses a **dual-model approach** combining:

1. **LightGBM Model** (ML-based quantile regression)
   - Predicts P10/P50/P90 percentiles using 21 features
   - Extended to P95/P97/P98/P99/P100 via tail extrapolation
   - Fast, feature-rich, adaptive to market conditions

2. **Percentile Model** (Historical distribution)
   - Pure historical distribution of price movements
   - Time-of-day and market condition filters
   - Robust to model overfitting

3. **Combined Prediction** (Recommended for trading)
   - Takes the **wider range** from both models
   - Most conservative approach
   - Best risk management

### Configuration Parameters

**File**: `scripts/close_predictor/models.py`

```python
LGBM_BAND_WIDTH_SCALE = 3.0  # Scale factor for base P10-P90 bands
                              # Higher = wider bands, more conservative
                              # Recommended range: 3.0-7.0
```

**File**: `scripts/close_predictor/bands.py`

```python
extension_factor = 0.5  # Multiplier for P95+ tail extensions
                        # Higher = wider bands beyond P90
                        # Recommended range: 0.5-3.0
```

### Calibrating Band Accuracy

To find optimal multipliers that achieve target hit rates (P95=95%, P98=98%, P99=99%):

```bash
# Run calibration backtest
python scripts/calibrate_band_multipliers.py NDX --days 60

# Then run accuracy backtest with new settings
python scripts/backtest_band_accuracy.py NDX --days 90
```

**Expected Results**:
- P95 bands: 80-95% hit rate, 1.5-3.0% width
- P98 bands: 95-98% hit rate, 2.0-4.0% width
- P99 bands: 98-100% hit rate, 3.0-7.0% width

**If LightGBM bands are too narrow**:
1. Increase `LGBM_BAND_WIDTH_SCALE` from 3.0 → 5.0 or 7.0
2. Increase `extension_factor` from 0.5 → 1.0 or 2.0
3. Use Combined bands (recommended) which include Percentile model

## Testing Requirements

**Tests MUST be run after every code change.** Before committing, always verify that all tests pass.

### Running Tests

```bash
# Run all tests (from project root: stocks/)
python -m pytest tests/ -v

# Run credit spread utility tests only
python -m pytest tests/test_analyze_credit_spread_intervals.py tests/test_price_movement_utils.py tests/test_max_move_utils.py tests/test_risk_gradient_utils.py tests/test_delta_utils.py -v

# Run a single test file
python -m pytest tests/test_price_movement_utils.py -v
```

### Test Configuration

- `pytest.ini` at project root configures test discovery
- Test files live in `tests/` and follow the `test_*.py` naming convention
- Async tests use `asyncio_mode = auto`

### Key Test Files

| Test File | Covers |
|-----------|--------|
| `test_analyze_credit_spread_intervals.py` | Main analyzer, arg parser, grid search |
| `test_price_movement_utils.py` | `credit_spread_utils/price_movement_utils.py` |
| `test_max_move_utils.py` | `credit_spread_utils/max_move_utils.py` |
| `test_risk_gradient_utils.py` | `credit_spread_utils/risk_gradient_utils.py` |
| `test_delta_utils.py` | `credit_spread_utils/delta_utils.py` |

## Architecture Reference

See `docs/CREDIT_SPREAD_STRATEGIES.md` for the full architecture document including module map, analysis modes, and CLI usage.

### Main Entry Point

`scripts/analyze_credit_spread_intervals.py` supports multiple analysis modes via `--mode`:

- `credit-spread` (default) -- credit spread backtesting
- `price-movements` -- close-to-close / time-to-close price movement stats
- `max-move` -- intraday extreme movement tables
- `risk-gradient` -- risk gradient analysis from historical safe points

### Project Layout

```
stocks/
  scripts/                          # Main scripts
    analyze_credit_spread_intervals.py  # Main entry point
    credit_spread_utils/            # Utility modules package
      strategies/                   # Strategy framework (pluggable)
  tests/                            # All test files
  docs/                             # Architecture docs
  common/                           # Shared DB/logging utilities
```

## Modular Backtesting Framework

The `scripts/backtesting/` directory contains a composable backtesting framework for options strategies. See `scripts/backtesting/BACKTESTING.md` for the full guide.

### Quick Start

```bash
# Run a backtest
python -m scripts.backtesting.runner --config scripts/backtesting/configs/credit_spread_0dte_ndx.yaml

# Dry run
python -m scripts.backtesting.runner --config scripts/backtesting/configs/credit_spread_0dte_ndx.yaml --dry-run

# Grid sweep
python -m scripts.backtesting.runner --config scripts/backtesting/configs/grid_sweep_comprehensive.yaml
```

### Key Components

| Component | Location | Purpose |
|-----------|----------|---------|
| Engine | `scripts/backtesting/engine.py` | Central orchestrator |
| Config | `scripts/backtesting/config.py` | YAML/JSON config loading |
| Providers | `scripts/backtesting/providers/` | Data sources (CSV, QuestDB) |
| Signals | `scripts/backtesting/signals/` | Prediction models |
| Instruments | `scripts/backtesting/instruments/` | Credit spread, iron condor, etc. |
| Strategies | `scripts/backtesting/strategies/` | Trading logic |
| Constraints | `scripts/backtesting/constraints/` | Budget, timing, exit rules |
| Results | `scripts/backtesting/results/` | Metrics, reporting |

### Running Backtesting Tests

```bash
python -m pytest tests/test_backtesting_config.py tests/test_backtesting_constraints.py tests/test_backtesting_exit_rules.py tests/test_backtesting_instruments.py tests/test_backtesting_collector.py -v
```

### TQQQ Momentum Scalper Strategy

A short-term 0DTE credit spread strategy for TQQQ using three signals: Opening Range Breakout (ORB), consecutive-day mean reversion, and small gap fade. See `results/TQQQ_MOMENTUM_SCALPER_ANALYSIS.md` for the full analysis.

```bash
# Single run
python -m scripts.backtesting.runner --config scripts/backtesting/configs/tqqq_momentum_scalper.yaml

# Parameter sweep (54 configs, multiprocessed)
python run_tqqq_momentum_sweep.py
```

**Key files:**
- Strategy: `scripts/backtesting/strategies/credit_spread/tqqq_momentum_scalper.py`
- Config: `scripts/backtesting/configs/tqqq_momentum_scalper.yaml`
- Sweep runner: `run_tqqq_momentum_sweep.py`
- Full analysis: `results/TQQQ_MOMENTUM_SCALPER_ANALYSIS.md`

**Best config (1-year backtest):** combined mode, 2% OTM, 10 contracts = 87 trades, 94% win rate, $92K net P&L, 188% ROI.

**Data:** Uses `options_csv_output_full/TQQQ/` (full chain with 0-1 DTE) and `equities_output/TQQQ/` (5-min bars).

## Live Paper Trading Platform

The `scripts/live_trading/` directory contains a live paper trading platform that runs strategies during market hours. It reuses the backtesting framework's abstractions (`DataProvider`, `SignalGenerator`, `Constraint`, `ExitRule`, `Instrument`) with live data from QuestDB and `csv_exports/options/`.

### Quick Start

```bash
# Run live paper trading
python -m scripts.live_trading.runner --config scripts/live_trading/configs/ndx_credit_spread_paper.yaml

# Dry run (no QuestDB needed)
python -m scripts.live_trading.runner --config scripts/live_trading/configs/ndx_credit_spread_paper.yaml --dry-run

# Performance report
python -m scripts.live_trading.runner --performance --days 30

# Show open positions
python -m scripts.live_trading.runner --positions

# Show recent journal entries
python -m scripts.live_trading.runner --journal --days 7
```

### Key Components

| Component | Location | Purpose |
|-----------|----------|---------|
| Engine | `scripts/live_trading/engine.py` | Main orchestrator + tick loop |
| Config | `scripts/live_trading/config.py` | YAML config with `live:` section |
| Executor | `scripts/live_trading/executor.py` | `PaperExecutor` (instant fill), `LiveExecutor` (stub) |
| Position Store | `scripts/live_trading/position_store.py` | JSON-backed persistent positions |
| Trade Journal | `scripts/live_trading/trade_journal.py` | JSONL decision log |
| Providers | `scripts/live_trading/providers/` | `realtime_equity` (QuestDB), `realtime_options` (CSV snapshots) |
| Strategies | `scripts/live_trading/strategies/` | `NDXCreditSpreadLiveStrategy` (playbook) |
| Runner | `scripts/live_trading/runner.py` | CLI entry point |
| Config YAML | `scripts/live_trading/configs/` | Trading configurations |

### Running Live Trading Tests

```bash
python -m pytest tests/test_live_trading.py -v
```

### Architecture

The `LiveEngine` runs a tick loop during market hours:
1. Fetch current price from QuestDB `realtime_data`
2. Check exit rules on all open positions (every tick)
3. Generate new entry signals at configured intervals
4. Route orders through `OrderExecutor` (paper fills instantly)
5. Persist positions to JSON, decisions to JSONL journal

Positions survive restarts via JSON persistence. The `OrderExecutor` ABC is designed for future exchange connectivity (IBKR/TDA) but only `PaperExecutor` is implemented now.

## Live Advisor — Profile-Based System

The `scripts/live_trading/advisor/` package provides a **generic, profile-based** live trading advisor. Any backtest configuration can be run as a live advisor by defining a YAML profile.

### Quick Start

```bash
# List available profiles
python run_live_advisor.py --list-profiles

# Run the 9-tier NDX advisor (current production config)
python run_live_advisor.py --profile tiered_v2

# Dry run (no QuestDB needed)
python run_live_advisor.py --profile tiered_v2 --dry-run

# Single-tier profile
python run_live_advisor.py --profile single_p90_dte2 --dry-run

# Override ticker
python run_live_advisor.py --profile tiered_v2 --ticker SPX

# Load from arbitrary YAML path
python run_live_advisor.py --profile ./my_custom_profile.yaml

# Legacy entry point (backwards compat, loads tiered_v2)
python run_live_advisor_v2.py --dry-run
```

### Key Components

| Component | Location | Purpose |
|-----------|----------|---------|
| Profile Loader | `scripts/live_trading/advisor/profile_loader.py` | `AdvisorProfile` dataclass + YAML loader + validator |
| Direction Modes | `scripts/live_trading/advisor/direction_modes.py` | Pluggable directional logic (pursuit, pursuit_eod, etc.) |
| Tier Evaluator | `scripts/live_trading/advisor/tier_evaluator.py` | Evaluates all tiers, accepts `AdvisorProfile` |
| Display | `scripts/live_trading/advisor/advisor_display.py` | Terminal UI, reads all config from profile |
| Position Tracker | `scripts/live_trading/advisor/position_tracker.py` | JSON-backed positions, profile-keyed dirs |
| Tier Config | `scripts/live_trading/advisor/tier_config.py` | Legacy constants (still used by `run_tiered_backtest_v2.py`) |
| Profiles | `scripts/live_trading/advisor/profiles/*.yaml` | Profile definitions |
| Runner | `run_live_advisor.py` | Generic CLI entry point |

### Profile YAML Format

Profiles live in `scripts/live_trading/advisor/profiles/<name>.yaml`. Each profile defines:
- `name`, `ticker` — identity
- `risk` — max_risk_per_trade, daily_budget, rate limits
- `providers` — equity/options data source dirs
- `signal` — signal generator name + params
- `tiers` — ordered list with label, priority, directional mode, DTE, percentile, etc.
- `exit_rules` — roll timing, proximity thresholds, 0DTE warnings
- `strategy_defaults` — min_credit, use_mid, contract_sizing, etc.

### Adding a New Profile

1. Create `scripts/live_trading/advisor/profiles/<name>.yaml` (use `tiered_v2.yaml` as template)
2. Define tiers with directional modes (`pursuit`, `pursuit_eod`, or register new ones in `direction_modes.py`)
3. Test: `python run_live_advisor.py --profile <name> --dry-run`

### Tiered Portfolio v2 Backtest

The canonical backtest runner is `run_tiered_backtest_v2.py`. It uses `tier_config.py` for tier definitions and runs all 9 tiers in parallel.

```bash
python run_tiered_backtest_v2.py            # Full run (9 tiers, ~10 min)
python run_tiered_backtest_v2.py --analyze  # Re-analyze only (skip backtests)
```

**Results**: `results/tiered_portfolio_v2/` — per-tier CSVs, portfolio simulation, charts, and HTML report.

### Timing Parameters — CRITICAL

**All roll/exit timing is unified at 18:00 UTC (11:00 AM PST / 2:00 PM ET).**

When changing timing parameters, you MUST update all locations listed in `docs/TIMING_CHANGE_GUIDELINES.md`. This includes exit rule defaults, YAML configs, profile loader defaults, tier_config.py, backtest runner, and tests.

### Running Advisor Tests

```bash
python -m pytest tests/test_live_advisor.py -v
```

## Backtest HTML Report Requirements

**Every backtest run (sweep, tiered portfolio, single config, etc.) MUST produce an HTML report** saved alongside the results. This applies to all `run_*.py` sweep scripts and any new backtest runners.

### Report Standards

1. **Filename**: Use a unique, descriptive name with the date: `report_{description}_{YYYY-MM-DD}.html`
   - Examples: `report_tiered_portfolio_2026-03-08.html`, `report_comprehensive_sweep_2026-03-10.html`
   - Saved in the same output directory as the backtest results (e.g., `results/tiered_portfolio/`)

2. **Structure** (follow the template in `results/tiered_portfolio/report.html`):
   - **Hero banner**: Strategy name, subtitle, date range
   - **KPI strip**: Key metrics at a glance (trades, win rate, net P&L, Sharpe, max drawdown, profit factor)
   - **Strategy overview**: Tier/config cards explaining the setup and rationale
   - **Per-config/tier performance table**: Full metrics with combined row
   - **Charts with narrative**: Each chart accompanied by plain-English explanation of what it shows and why it matters
   - **Monthly breakdown**: Table + chart showing regime sensitivity
   - **Drawdown analysis**: Chart + narrative, include caveats about backtest limitations
   - **Daily P&L distribution**: Histogram with mean/median statistics
   - **Key takeaways**: Numbered insights synthesized from the results
   - **Methodology table**: All config parameters used

3. **Styling**: Dark theme (background `#0d1117`), GitHub-inspired, responsive. Use the CSS from the template report.

4. **Charts**: Reference as relative paths (`charts/*.png`) so the report is self-contained within its output directory.

5. **Narrative quality**: Each chart section must include 2-4 sentences explaining:
   - What the chart shows
   - What the key patterns/outliers are
   - Why it matters for the strategy

### Reference Implementation

See `results/tiered_portfolio/report.html` for the canonical example of a well-structured backtest report.

## Strategy Documentation

**Every backtest strategy MUST have a documentation file in `docs/strategies/`.** When creating or documenting a new strategy, create `docs/strategies/<strategy_name>.md` with:

1. **Overview**: What the strategy does in 2-3 sentences
2. **Strategy Logic**: How signals are generated, tier/config breakdown tables
3. **Exit Rules**: Roll timing, proximity thresholds, profit targets, stop losses
4. **Risk Management**: Budget, position sizing, rate limits
5. **Commands**: Exact CLI commands for live advisor and backtesting
6. **Results**: Output directory, report location, latest backtest metrics
7. **Key Files**: Table mapping files to their purpose

### Existing Strategy Docs

| Strategy | Doc |
|----------|-----|
| Tiered Portfolio v2 (NDX) | `docs/strategies/tiered_portfolio_v2.md` |
| TQQQ Momentum Scalper | `results/TQQQ_MOMENTUM_SCALPER_ANALYSIS.md` |
