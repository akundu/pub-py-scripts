# Unified Close Predictor

End-of-day closing price range predictor combining two models into a single tool for NDX and SPX.

## System Overview

The Unified Close Predictor produces confidence bands (P95, P98, P99, P100) that estimate where the closing price will land, given the current price and time of day. It combines:

1. **Percentile Range Model** — historical move distributions filtered by time slot, above/below prev close, and vol-scaled
2. **Statistical Bucket Model** — VIX1D regime, overnight gap, intraday move, momentum, and other features

The combined output takes the **wider (more conservative)** range from both models at each band level.

## Architecture

```
┌─────────────────────────────────────────────────────┐
│                   CLI (argparse)                    │
│           unified_close_predictor.py                │
└────────┬───────────────────────────────┬────────────┘
         │                               │
    ┌────▼────┐                    ┌─────▼────┐
    │backtest │                    │   live   │
    │         │                    │  / demo  │
    └────┬────┘                    └────┬─────┘
         │                              │
    ┌────▼──────────────────────────────▼─────┐
    │            prediction.py                │
    │  train_both_models()                    │
    │  compute_percentile_prediction()        │
    │  compute_statistical_prediction()       │
    │  make_unified_prediction()              │
    └────┬──────────────┬────────────────┬────┘
         │              │                │
    ┌────▼────┐   ┌─────▼─────┐   ┌─────▼─────┐
    │bands.py │   │features.py│   │display.py  │
    └────┬────┘   └─────┬─────┘   └────────────┘
         │              │
    ┌────▼──────────────▼─────┐
    │       models.py         │
    │  UnifiedBand             │
    │  UnifiedPrediction       │
    │  Constants               │
    └─────────────────────────┘
```

## Module Map

| Module | Lines | Purpose |
|--------|-------|---------|
| `models.py` | ~75 | Constants (`ET_TZ`, `FULL_DAY_BARS`, `UNIFIED_BAND_NAMES`), `UnifiedBand` and `UnifiedPrediction` dataclasses, `STAT_FEATURE_CONFIG`, shared `_intraday_vol_cache` |
| `bands.py` | ~130 | `map_statistical_to_bands()`, `map_percentile_to_bands()`, `combine_bands()` |
| `features.py` | ~170 | `detect_reversal_strength()`, `compute_intraday_vol_from_bars()`, `compute_historical_avg_intraday_vol()`, `compute_intraday_vol_factor()`, `get_intraday_vol_factor()` |
| `prediction.py` | ~230 | `train_both_models()`, `_train_statistical()`, `compute_percentile_prediction()`, `compute_statistical_prediction()`, `make_unified_prediction()` |
| `display.py` | ~130 | `print_live_display()`, `print_backtest_results()`, formatting helpers |
| `backtest.py` | ~200 | `run_backtest()` — walk-forward backtest engine |
| `live.py` | ~260 | `run_demo_loop()`, `run_live_loop()`, `_build_day_context()`, `_find_nearest_time_label()` |
| `__init__.py` | ~40 | Re-exports key symbols for convenient imports |

## CLI Commands

### Backtest

Walk-forward accuracy comparison across models:

```bash
# Default: 10 test days, NDX, vol-scaled
python scripts/unified_close_predictor.py backtest --ticker NDX --test-days 10

# SPX with more test days
python scripts/unified_close_predictor.py backtest --ticker SPX --test-days 20

# Raw mode (no vol-scaling), all 30-min slots
python scripts/unified_close_predictor.py backtest --ticker NDX --test-days 10 --no-vol-scale --all-slots

# Custom lookback
python scripts/unified_close_predictor.py backtest --ticker NDX --lookback 500 --test-days 15
```

### Live Mode

Real-time predictions using QuestDB:

```bash
# Default: NDX, 30s refresh
python scripts/unified_close_predictor.py live --ticker NDX --interval 30

# SPX with faster refresh
python scripts/unified_close_predictor.py live --ticker SPX --interval 10
```

### Demo Mode

CSV-based simulation of the most recent trading day:

```bash
python scripts/unified_close_predictor.py live --ticker NDX --demo
python scripts/unified_close_predictor.py live --ticker SPX --demo
```

## Prediction Models

### Percentile Range Model

Collects historical closing-price moves relative to the current price at each 30-minute time slot. Filters by above/below previous close and applies 5-day realized volatility scaling.

Output: empirical percentile bands from the move distribution.

### Statistical Bucket Model

Uses `StatisticalClosePredictor` trained on features:
- VIX1D regime
- Overnight gap size
- Intraday move from open
- Intraday range (high-low)
- 5-day momentum
- First hour range
- Opening range breakout
- Day-of-week, OPEX proximity
- Prior-day signals

Output: P10/P90 prediction interval, extrapolated to P95-P100.

### Combined Output

For each band level (P95, P98, P99, P100):
- `lo_price = min(percentile_lo, statistical_lo)`
- `hi_price = max(percentile_hi, statistical_hi)`

This "best across best" approach ensures the combined range is always at least as wide as either model alone.

## Band System

| Band | Coverage | Percentile Range |
|------|----------|-----------------|
| P95 | 95% | 2.5th - 97.5th |
| P98 | 98% | 1.0th - 99.0th |
| P99 | 99% | 0.5th - 99.5th |
| P100 | 100% | 0.0th - 100.0th |

## Feature Engineering

### Reversal Detection

`detect_reversal_strength()` returns a blend weight in [0.0, 0.5] based on three signals:

1. **Overshoot/Undershoot** — has the day already traded on the opposite side of prev_close?
2. **Directional signal** — is price moving toward prev_close?
3. **Proximity** — how close is current price to prev_close?

When the blend weight > 0, opposite-condition training data is mixed into the percentile model to widen the relevant tail.

### Intraday Volatility Adaptation

`get_intraday_vol_factor()` computes how today's realized intraday vol compares to the historical average at the same time of day:

- `factor = current_vol / historical_avg`, clipped to [0.5, 2.0]
- Factor > 1.0 widens bands (volatile day)
- Factor < 1.0 tightens bands (quiet day)

## Backtest Results Example

```
=======================================================================
 NDX ABOVE Prev Close — Unified Backtest Accuracy
 (does actual close fall within predicted range?)
=======================================================================
                    P95                P98                P99               P100
HrsLeft  Time    N  Pctl    Stat    Comb   Pctl    Stat    Comb ...
6.0h     10:00   8   88%     75%    100%    88%     88%    100% ...
5.0h     11:00   8  100%     88%    100%   100%     88%    100% ...
4.0h     12:00   8  100%     88%    100%   100%    100%    100% ...
...
```

## Adding New Features

To add a new feature to the prediction pipeline:

1. **Add constants/types** to `models.py` if needed
2. **Implement feature logic** in `features.py`
3. **Integrate into prediction** in `prediction.py`:
   - Add parameters to `make_unified_prediction()`
   - Pass the feature to `compute_percentile_prediction()` or `compute_statistical_prediction()`
4. **Update display** in `display.py` to show the new feature
5. **Thread through backtest** in `backtest.py`
6. **Thread through live/demo** in `live.py`
7. **Add tests** in `tests/test_unified_close_predictor.py`

## Dependencies

- `numpy`, `pandas` — numerical computation
- `scripts.percentile_range_backtest` — percentile data collection + vol scaling
- `scripts.strategy_utils.close_predictor` — statistical bucket model
- `scripts.csv_prediction_backtest` — CSV data loading + training data construction
- `common.stock_db` — QuestDB access (live mode only, lazy import)

## Testing

```bash
# Run all unified close predictor tests
pytest tests/test_unified_close_predictor.py -v

# Run a specific test class
pytest tests/test_unified_close_predictor.py::TestDetectReversalStrength -v

# Quick import check
python -c "from scripts.close_predictor import run_backtest, UnifiedBand, detect_reversal_strength; print('OK')"
```

## Change Log

- **2025-02-08**: Modularized into `scripts/close_predictor/` package (models, bands, features, prediction, display, backtest, live). Added unit tests. Reversal detection + intraday vol features preserved from monolith.
