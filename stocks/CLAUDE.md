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

### Polygon.io API

**REQUIRED** for downloading historical equities and options data:

```bash
export POLYGON_API_KEY="your_polygon_api_key"
```

## Data Download Pipeline

### Automated (Crontab)

Daily at 3:10 AM local time, crontab triggers:
```bash
curl 'http://localhost:9102/run_script?script=ms1_cron.sh'
```

This runs `run_scripts/ms1_cron.sh` which downloads equities + options and retrains models. Default: last 2 days. Override with query params: `?start_date=YYYY-MM-DD&end_date=YYYY-MM-DD` or `?days_back=N`.

### Manual Download Commands

**Equities** (5-min OHLCV bars):
```bash
python3 scripts/equities_download.py I:VIX1D I:VIX SPY DJX I:DJX TQQQ QQQ I:NDX I:SPX I:RUT \
  --start 2025-01-01 --end 2025-03-13 --output-dir ./equities_output
```
Output: `equities_output/{TICKER}/{TICKER}_equities_YYYY-MM-DD.csv`

**0DTE Options** (5-min bars, same-day expiration):
```bash
python3 scripts/options_chain_download.py SPX NDX RUT \
  --zero-dte-date-start 2025-01-01 --zero-dte-date-end 2025-03-13 \
  --max-connections 30 --num-processes 2 --interval 5min \
  --format-chain-csv --output-dir options_csv_output/
```
Output: `options_csv_output/{TICKER}/{TICKER}_options_YYYY-MM-DD.csv`

**Multi-Day Options** (15-min bars, 30-day rolling window):
```bash
python3 scripts/options_chain_download.py SPX NDX DJX TQQQ RUT \
  --track-from 2025-01-01 --track-end 2025-03-13 --track-days 30 \
  --interval-minutes 15 --chunk-days 7 --max-connections 20 \
  --num-processes 12 --window-workers 5 --skip-existing \
  --format-chain-csv --output-dir ./options_csv_output_full/
```
Output: `options_csv_output_full/{TICKER}/{TICKER}_options_YYYY-MM-DD.csv`

### Data Directories

| Directory | Content | Interval | Key Tickers |
|-----------|---------|----------|-------------|
| `equities_output/` | Equity 5-min bars | 5min | I:NDX, I:SPX, TQQQ, SPY, QQQ, I:VIX, I:VIX1D, I:RUT, I:DJX, DJX |
| `options_csv_output/` | 0DTE options chains | 5min | SPX, NDX, RUT |
| `options_csv_output_full/` | Multi-day options (30d window) | 15min | SPX, NDX, DJX, TQQQ, RUT |
| `csv_exports/options/` | Live options snapshots | realtime | NDX, SPX, RUT, DJX, TQQQ, AVGO, etc. |

**Note**: `equities_output/` has symlinks: `NDX`→`I:NDX`, `SPX`→`I:SPX`, `RUT`→`I:RUT`, `DJX`→`I:DJX`

### Key Scripts

| Script | Purpose |
|--------|---------|
| `scripts/equities_download.py` | Download equity bars from Polygon.io |
| `scripts/options_chain_download.py` | Download options chains from Polygon.io |
| `run_scripts/ms1_cron.sh` | Daily orchestrator (download + retrain) |
| `run_scripts/build_close_models.sh` | Model retraining pipeline |
| `scripts/retrain_models_auto.sh` | LightGBM retraining (1-20 DTE) |

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

**Tests MUST be run after every code change.** Before committing, always verify that all tests pass. **Every code change must include corresponding test additions or updates.** Do not add features or fix bugs without updating tests. All tests must pass before the change is considered complete.

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

### Strategy Development Guide

**If you are building a new strategy, READ `docs/STRATEGY_DEVELOPMENT_GUIDE.md` FIRST.** It covers:
- How to create a strategy (file, config, registration, tests)
- How to inherit from or compose existing strategies
- How to use the orchestrator (REQUIRED for all comparisons and production work)
- Performance considerations (multiprocessing, data optimization)
- Documentation and reporting requirements

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

### vMaxMin Layer Strategy

A 0DTE credit spread strategy that opens both call and put spreads at market open, layers additional spreads at HOD/LOD throughout the day, and rolls ITM positions to DTE+1 using credit-neutral sizing with infinite rolls.

```bash
# Simulate a date
python run_live_advisor.py --profile vmaxmin_layer --simulate 2026-03-20 --sim-speed 0

# Or directly
python run_vmaxmin_live.py --simulate 2026-03-20 --num-contracts 40

# Backtest
python scripts/backtesting/scripts/run_vmaxmin_backtest.py \
    --tickers RUT --layer --lookback-days 120 --num-contracts 40 \
    --daily-budget 100000 --risk-cap 500000 -v

# Check carry positions
python run_vmaxmin_live.py --positions
```

**Key Parameters:**
- Entry: `best_roi` spread finder, scan 06:30-06:45 window, 0.3% proximity
- Layers: check 08:35, 10:35 for new HOD/LOD
- EOD: scan 12:50-13:00 every minute, roll if ITM or within 0.2%
- Rolls: credit-neutral sizing, no chain cap, infinite rolls
- Risk: $100K/day budget, $500K max exposure, 3-day skip limit

**6-Month Backtest Results (40 contracts):**

| Ticker | True P&L | Annualized | Win Rate | Avg/Day |
|--------|----------|------------|----------|---------|
| RUT | +$5.18M | +$10.5M | 100% | +$41,763 |
| SPX | +$2.22M | +$4.5M | 100% | +$17,921 |
| NDX | +$10.6M | +$21.5M | 100% | +$85,509 |

**Key files:**
- Core engine: `scripts/backtesting/scripts/vmaxmin_engine.py`
- Backtest CLI: `scripts/backtesting/scripts/run_vmaxmin_backtest.py`
- Single-day simulator: `scripts/backtesting/scripts/run_vmaxmin_simulate.py`
- Live/simulate runner: `run_vmaxmin_live.py`
- Profile config: `scripts/live_trading/advisor/profiles/vmaxmin_layer.yaml`
- Backtest YAML config: `scripts/backtesting/configs/vmaxmin_layer_rut.yaml`
- Full analysis: `docs/strategies/vmaxmin_layer_analysis.md`
- Tests: `tests/test_vmaxmin.py` (88 tests)

### Roll Cost Analysis Tool

Standalone analysis tool that calculates the net cost of rolling breached 0DTE credit spreads to future expirations. Answers: "At what time of day, to which DTE, and at what strike placement should I roll?"

**Two scripts:**
- `scripts/roll_cost_table.py` — Core engine: generates roll cost data and ASCII tables
- `scripts/generate_roll_cost_report.py` — Full pipeline: runs data for multiple tickers, generates charts, builds a tabbed HTML report with playbooks

```bash
# Single ticker analysis
python scripts/roll_cost_table.py --ticker RUT --start 2026-01-01 --end 2026-03-29 \
    --spread-width 20 --options-dir ./options_csv_output_full

# Multi-ticker report with charts (one command does everything)
python scripts/generate_roll_cost_report.py --start 2026-01-01 --end 2026-03-29 \
    --tickers RUT:20 SPX:10 NDX:50 --options-dir ./options_csv_output_full \
    --output-dir ./results/roll_cost_q1_2026
```

**Key parameters:**
- `--entry-breach-pcts` (default: 100 75 50 25) — how deep ITM the 0DTE spread is
- `--target-breach-pcts` (default: 100 50 25 0) — where the new short strike lands (100=same strikes, 0=ATM)
- `--check-times` — times in PST (default: 08:30 to 12:55, 30-min intervals)
- `--roll-dtes` (default: 1 2 3 5) — DTE targets for the roll
- `--options-dir` / `--equities-dir` — configurable data sources

**How it works:**
1. For each day/time, constructs a hypothetical 0DTE spread at the specified breach depth
2. Calculates close cost (buy back 0DTE) and open credit (sell DTE+N spread) using mid pricing
3. Reports net = close_debit - open_credit (negative = credit received, positive = cost)
4. If options dir has multi-exp data (e.g. `options_csv_output_full_15`), uses same-moment pricing; otherwise uses cross-day opening snapshots for DTE+N

**Report output includes:**
- Roll playbook with best time/DTE/strike recommendations per option type
- Decision tree for puts vs calls
- Summary tables, heatmaps (entry x target x time x DTE), line charts, distributions
- Collapsible detailed tables for every parameter combination
- Tabbed interface when multiple tickers are analyzed

**Key files:**
- Core engine: `scripts/roll_cost_table.py`
- Report generator: `scripts/generate_roll_cost_report.py`
- Strategy doc: `docs/strategies/roll_cost_analysis.md`
- Tests: `tests/test_roll_cost_table.py` (34 tests)

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

## Live Advisor — Unified Config System

The live advisor, simulator, and orchestrated backtest all share a **single config file** that controls all trading parameters. Mode-specific differences (e.g., mid pricing for backtests) are handled via `mode_overrides`.

### Unified Config: The Single Source of Truth

**All trading parameters live in one file:**

```
scripts/live_trading/advisor/profiles/adaptive_v5.yaml
```

This file controls: tickers, percentiles, spread widths, tiers, ROI thresholds, risk limits, exit rules, adaptive budget, and everything else. All three modes read from it:

- **Live**: `python run_live_advisor.py --profile adaptive_v5 --live`
- **Simulate**: `python run_live_advisor.py --profile adaptive_v5 --simulate 2026-03-20`
- **Backtest**: `scripts/backtesting/configs/orchestration_adaptive_budget.yaml` references it via `profile: adaptive_v5`

Mode-specific overrides are in the `mode_overrides` section at the bottom of the YAML:
```yaml
mode_overrides:
  backtest:
    strategy_defaults:
      use_mid: true              # backtest uses mid prices
    exit_rules:
      profit_target_pct: 0.50    # backtest manages exits
      stop_loss_pct: 2.0
  live: {}                        # no overrides
  simulate: {}                    # no overrides
```

**To change trading behavior**: Edit `adaptive_v5.yaml`. All three modes pick it up.

**To change backtest-only behavior**: Edit `mode_overrides.backtest` in the same file.

**To change backtest run parameters** (date range, output dir, instances): Edit `orchestration_adaptive_budget.yaml`.

### Quick Start

```bash
# Live advisor with UTP/IBKR (all tickers)
python run_live_advisor.py --profile adaptive_v5 --live

# Live with specific ticker and faster refresh
python run_live_advisor.py --profile adaptive_v5 --live --ticker RUT --interval 30

# Simulate a historical day from CSV
python run_live_advisor.py --profile adaptive_v5 --simulate 2026-03-20 --sim-speed 5

# Fast simulation scan
python run_live_advisor.py --profile adaptive_v5 --simulate 2026-03-20 --sim-speed 0.1 --no-interactive

# Dry run (show config and exit)
python run_live_advisor.py --profile adaptive_v5 --dry-run

# Orchestrated backtest
python -m scripts.backtesting.scripts.run_orchestrated_backtest \
    --config scripts/backtesting/configs/orchestration_adaptive_budget.yaml
```

### Interactive Commands (Live/Simulate)

| Command | Action |
|---------|--------|
| `buy <order_id>` | Confirm a trade (e.g., `buy RUT_P2420_D0`) |
| `close <pos_id>` | Close a position |
| `close <pos_id> <price>` | Close with exit price for P&L |
| `flush` | Close all open positions |
| `p` | Show positions |
| `s` | Summary |
| `q` | Quit (Ctrl+C also works, twice to force) |

### Key Components

| Component | Location | Purpose |
|-----------|----------|---------|
| **Unified Config** | `scripts/live_trading/advisor/profiles/adaptive_v5.yaml` | Single source of truth for all trading params |
| Profile Loader | `scripts/live_trading/advisor/profile_loader.py` | Loads YAML with mode-specific overrides |
| Tier Evaluator | `scripts/live_trading/advisor/tier_evaluator.py` | Evaluates tiers, builds spreads, tracks rejections |
| Display | `scripts/live_trading/advisor/advisor_display.py` | Terminal UI with multi-ticker prices |
| Position Tracker | `scripts/live_trading/advisor/position_tracker.py` | JSON-backed positions, rate limiting |
| Spread Builder | `scripts/credit_spread_utils/spread_builder.py` | Core spread construction (shared across all modes) |
| Roll Builder | `scripts/credit_spread_utils/roll_builder.py` | Roll spread with width expansion (shared) |
| UTP Provider | `scripts/live_trading/providers/utp_provider.py` | IBKR quotes/options via UTP daemon |
| CSV Exports Provider | `scripts/backtesting/providers/csv_exports_options_provider.py` | Historical options for simulation |
| Runner | `run_live_advisor.py` | CLI entry point for live/simulate |
| Backtest Runner | `scripts/backtesting/scripts/run_orchestrated_backtest.py` | CLI for orchestrated backtest |
| Backtest Config | `scripts/backtesting/configs/orchestration_adaptive_budget.yaml` | Backtest-only: dates, instances, scoring |

### Shared Code Across All Modes

The following code is shared — backtest, live, and simulate all call the same functions:

- `spread_builder.build_credit_spreads()` — spread construction, min_otm_pct, bid/ask pricing
- `CreditSpreadInstrument.build_position()` — instrument wrapper
- `roll_builder.build_roll_spread()` — roll width expansion, BTC cost comparison
- `AdaptiveBudgetConfig` — ROI tier thresholds and multipliers

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

## Universal Trade Platform (UTP)

The `live_trading/universal-trade-platform/` directory contains a unified multi-broker trading API (FastAPI) supporting Robinhood, E\*TRADE, and IBKR. **See `live_trading/universal-trade-platform/CLAUDE.md` for full UTP-specific context.**

### Quick Start — Unified CLI (`utp.py`)

Everything is in two files: `utp.py` (CLI + server) and `tests/test_utp.py` (all 532 tests).

```bash
cd live_trading/universal-trade-platform
python utp.py portfolio                          # View positions, P&L
python utp.py quote SPY AAPL                     # Real-time quotes
python utp.py options RUT --type CALL --live     # Option chain quotes
python utp.py options SPX --list-expirations     # List available expirations
python utp.py margin credit-spread --symbol SPX --short-strike 5500 \
  --long-strike 5475 --option-type PUT --expiration 2026-03-20
python utp.py trade equity --symbol SPY --side BUY --quantity 1
python utp.py trade --validate-all               # Test all 5 trade types
python utp.py trade replay <pos-id> --live       # Replay a trade (local or portfolio spread ID)
python utp.py trades --live                      # Today's trades (IBKR real-time P&L)
python utp.py trades --all --live                # All trades (open + closed)
python utp.py close <pos-id> --live              # Close position by ID (MARKET order default)
python utp.py close <pos-id> --net-price 0.10 --live  # Close at LIMIT $0.10 debit
python utp.py playbook execute playbooks/example_mixed.yaml
python utp.py status                             # System dashboard
python utp.py journal --days 7                   # Trade history
python utp.py performance --days 30              # Performance metrics
python utp.py reconcile --flush --show --live    # Flush + reconcile + show positions
python utp.py readiness --symbol SPX --paper     # IBKR connectivity test
python utp.py daemon --paper                             # Start always-on daemon
python utp.py daemon --live --advisor-profile tiered_v2  # With advisor signals
python utp.py repl                                       # Interactive REPL
python utp.py server                             # Start API server
python -m pytest tests/ -v                       # Run all 532 tests
```

### Core Capabilities

- **Multi-broker trading**: Unified REST API across Robinhood, E\*TRADE, IBKR
- **Transaction ledger**: Append-only JSONL log of every trade event with snapshots
- **Paper trading**: Dry-run trades tracked with real P&L computation
- **Dashboard**: REST + terminal display for positions, cash, performance metrics
- **Auto-expiration**: Background loop closes expired options at EOD
- **Position sync**: Polls all brokers every 5 min for out-of-band positions
- **CSV import**: Ingest Robinhood/E\*TRADE CSV exports for unified history
- **Real IBKR**: Full `ib_insync` integration (set `IBKR_ACCOUNT_ID` to activate)
- **Source attribution**: Every position/transaction tagged: `live_api`, `paper`, `csv_import`, `external_sync`
- **Trade playbooks**: YAML instruction files for batch trade execution (equity, options, spreads, iron condors)
- **Reconciliation**: Compare system positions against broker-reported positions
- **Option chains**: View available expirations, strikes, and live bid/ask/volume for any underlying
- **Status dashboard**: Unified view of active positions, pending orders, recent trades, cache stats
- **Always-on daemon**: Server-first architecture with persistent IBKR connection, LAN trust, REPL, Python client library

### Key Endpoints

| Endpoint | Purpose |
|----------|---------|
| `POST /trade/execute` | Execute trade (add `X-Dry-Run: true` for paper) |
| `GET /dashboard/summary` | Active positions, P&L, source breakdown |
| `GET /dashboard/performance` | Win rate, Sharpe, drawdown, profit factor |
| `GET /ledger/entries` | Query transaction log (filter by broker, type, source) |
| `POST /account/sync` | Manual position sync across brokers |
| `POST /import/csv?broker=robinhood` | Import CSV transaction history |
| `POST /playbook/execute` | Execute YAML trade playbook (add `X-Dry-Run: true` for paper) |
| `POST /playbook/validate` | Validate playbook structure without executing |
| `GET /account/reconciliation` | Compare system vs broker positions |
| `GET /dashboard/status` | Full status: positions, orders, trades, cache |
| `POST /trade/close` | Close position by ID |
| `GET /account/trades` | Trade history |
| `GET /account/orders` | Open orders |
| `GET /market/options/{symbol}` | Option chain data |
| `GET /dashboard/advisor/recommendations` | Current advisor signals |

### Architecture

```
app/
├── services/           # ledger, position_store, dashboard, expiration, sync, csv_importer, metrics, playbook
├── routes/             # trade, market, account, ledger, dashboard, import, auth, ws, playbook
├── core/providers/     # robinhood (stub), etrade (stub), ibkr (stub + live via ib_insync)
└── main.py             # Lifespan: init services, register providers, start background tasks
```

### Testing

532 tests in `tests/test_utp.py`. Tests use `tmp_path` for isolated persistence. Run from `live_trading/universal-trade-platform/`:

```bash
python -m pytest tests/ -v
python -m pytest tests/test_utp.py -k "TestLedger" -v   # Filter by class
```

### UTP Documentation

Full docs in `live_trading/universal-trade-platform/docs/`:
- `architecture.md` — system design, persistence, background tasks, data flows
- `api_reference.md` — all 33 endpoints with request/response schemas
- `configuration.md` — all environment variables
- `providers.md` — broker provider interface, IBKR live, adding new brokers
- `ibkr_setup_guide.md` — TWS/IB Gateway connection walkthrough
- `testing.md` — 202 tests, fixtures, test class descriptions
- `playbook.md` — trade playbook system, reconciliation, status dashboard
- `authentication.md` — API key + OAuth2/JWT flows
- `symbology.md` — symbol mapping across brokers
- `websockets.md` — real-time streaming

**HTML docs**: `docs/html/` (dark-themed, 10 pages + index). Rebuild after any `.md` changes: `python3 docs/build_html.py`
