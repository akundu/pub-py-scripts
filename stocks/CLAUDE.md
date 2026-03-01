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
