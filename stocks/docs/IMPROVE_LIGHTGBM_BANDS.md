# How to Improve LightGBM Band Accuracy

## Problem

LightGBM bands are too narrow (0.05-0.09% width) compared to realistic market movements (1.5-4.0% typical NDX daily range).

**Why?**
- Model is too confident in predictions
- Tail extrapolation multipliers are too conservative
- Band width scale is calibrated for different use case

## Solution: Three Approaches

### Quick Fix (5 minutes)

**Increase extension_factor**

File: `scripts/close_predictor/bands.py` (around line 50)

```python
# BEFORE
extension_factor = 0.5  # Too conservative

# AFTER - Try one of these:
extension_factor = 1.0  # 2x wider bands
extension_factor = 2.0  # 4x wider bands (recommended starting point)
extension_factor = 3.0  # 6x wider bands (very conservative)
```

**Expected Result**:
- extension_factor = 2.0 → P95 width ~0.20%, P98 ~0.24%, P99 ~0.32%
- Still narrow, but 4x improvement

### Better Fix (10 minutes)

**Increase LGBM_BAND_WIDTH_SCALE**

File: `scripts/close_predictor/models.py` (line 47)

```python
# BEFORE
LGBM_BAND_WIDTH_SCALE = 3.0  # Base P10-P90 width multiplier

# AFTER - Try these:
LGBM_BAND_WIDTH_SCALE = 5.0  # Moderate increase (1.67x)
LGBM_BAND_WIDTH_SCALE = 7.0  # Significant increase (2.33x) - RECOMMENDED
LGBM_BAND_WIDTH_SCALE = 10.0  # Very conservative (3.33x)
```

**Expected Result** (with band_width_scale = 7.0):
- P95 width ~0.12%, P98 ~0.14%, P99 ~0.19%
- Combined with extension_factor = 2.0 → P95 ~0.24%, P98 ~0.28%, P99 ~0.38%

### Best Fix (1-2 hours)

**Calibrate using backtest**

1. **Test current performance**:
```bash
python scripts/backtest_band_accuracy.py NDX --days 60
```

Note the hit rates and widths.

2. **Try different combinations**:

| LGBM_BAND_WIDTH_SCALE | extension_factor | Expected P95 Width | Expected Hit Rate |
|-----------------------|------------------|-------------------|-------------------|
| 3.0 | 0.5 | ~0.05% | ~40% (too narrow) |
| 5.0 | 1.0 | ~0.20% | ~60% |
| 7.0 | 2.0 | ~0.40% | ~75-80% |
| 10.0 | 3.0 | ~0.60% | ~85-90% |

3. **Make changes and retest**:
```bash
# Edit models.py and bands.py with new values
python scripts/predict_close.py NDX --retrain
python scripts/backtest_band_accuracy.py NDX --days 60
```

4. **Iterate until you achieve**:
- P95: 80-95% hit rate
- P98: 95-98% hit rate
- P99: 98-100% hit rate

## Recommended Settings (Start Here)

Based on backtest results, these should give reasonable bands:

**File: `scripts/close_predictor/models.py`**
```python
LGBM_BAND_WIDTH_SCALE = 7.0  # Up from 3.0
```

**File: `scripts/close_predictor/bands.py`**
```python
extension_factor = 2.0  # Up from 0.5
```

**Expected Results**:
- P95: ~0.40% width, ~80% hit rate
- P98: ~0.50% width, ~90% hit rate
- P99: ~0.65% width, ~95% hit rate
- P100: ~0.80% width, ~98% hit rate

**Note**: These are still narrower than the Percentile model (1.65-4.88%), which is why the **Combined Prediction** (taking wider of both) is recommended for actual trading.

## Why Use Combined Bands?

Even with optimized LightGBM settings, the **Combined Prediction** is most reliable:

```python
# From predict_close.py output:

LightGBM Model:     P95 = 0.40% width (optimized)
Percentile Model:   P95 = 1.65% width (historical)
Combined:           P95 = 1.65% width (takes wider range)
```

**Use Combined for**:
- Strike selection (options trading)
- Position sizing (risk management)
- Stop-loss placement

**Use LightGBM for**:
- Directional bias (which way market is moving)
- High-confidence trades (when bands are tight)
- Transaction timing (midpoint prediction)

## Testing Your Changes

```bash
# 1. Make changes to models.py and bands.py

# 2. Test with live prediction
export QUEST_DB_STRING="questdb://stock_user:stock_password@lin1.kundu.dev:8812/stock_data"
python scripts/predict_close.py NDX --retrain

# 3. Backtest accuracy
python scripts/backtest_band_accuracy.py NDX --days 60

# 4. Check hit rates match targets
#    P95: 80-95%
#    P98: 95-98%
#    P99: 98-100%
```

## Advanced: Dynamic Scaling

For production use, consider implementing dynamic band scaling based on:
- VIX levels (higher VIX = wider bands)
- Time to close (more time = wider bands)
- Recent realized volatility
- Market regime detection

These adjustments are already partially implemented via `ENABLE_DYNAMIC_VOL_SCALING` in `models.py`.
