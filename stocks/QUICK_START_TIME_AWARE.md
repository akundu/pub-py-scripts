# Quick Start: Time-Aware Multi-Day Predictions

## TL;DR

Multi-day predictions now automatically adjust for:
1. **Time of day** - Bands tighten as market close approaches
2. **Current volatility** - Bands widen on volatile days

**Default:** Both features are **ENABLED**. Use flags to disable if needed.

---

## Quick Commands

### 1. Normal Usage (Time-Aware Enabled)

```bash
# 5-day prediction with time-aware features
python scripts/predict_close_now.py NDX --days-ahead 5

# Output will show:
# Time decay: 4.25 hours to close → effective DTE = 4.35 (vs 5 nominal)
# Intraday vol: range=1.85% (normal) → vol factor = 1.00x
```

### 2. Disable Time-Aware Features (Old Behavior)

```bash
# Revert to original behavior
python scripts/predict_close_now.py NDX --days-ahead 5 --no-time-decay --no-intraday-vol
```

### 3. Quick Validation Test

```bash
# Run both versions and compare
python scripts/predict_close_now.py NDX --days-ahead 5 > new_behavior.txt
python scripts/predict_close_now.py NDX --days-ahead 5 --no-time-decay --no-intraday-vol > old_behavior.txt
diff new_behavior.txt old_behavior.txt
```

### 4. Backtest Comparison (30-Day)

```bash
# Create results directories
mkdir -p results/baseline_30d results/improved_30d

# Run baseline (old behavior) - IF backtest script supports the flags
python scripts/backtest_multi_day.py \\
  --ticker NDX --test-days 30 --train-days 250 --max-dte 20 \\
  --no-time-decay --no-intraday-vol \\
  --output-dir results/baseline_30d/

# Run improved (new behavior) - IF backtest script supports the flags
python scripts/backtest_multi_day.py \\
  --ticker NDX --test-days 30 --train-days 250 --max-dte 20 \\
  --output-dir results/improved_30d/

# Compare results
python scripts/compare_backtest_results.py \\
  --baseline results/baseline_30d/ \\
  --improved results/improved_30d/ \\
  --output comparison_30d.md

# View report
cat comparison_30d.md
```

**Note:** The backtest script may need updating to support `--no-time-decay` and `--no-intraday-vol` flags. If not available, the script will use the default behavior (features enabled).

---

## What Changed?

### Before (Old Behavior)
- 10 AM prediction: P99 bands = ±6.0%
- 3:50 PM prediction: P99 bands = ±6.0% (same!)
- Calm day vs volatile day: No difference

### After (New Behavior)
- 10 AM prediction: P99 bands = ±6.0% (baseline)
- 3:50 PM prediction: P99 bands = ±5.8% (2-4% tighter)
- Calm day: ±6.0%, Volatile day: ±6.9% (15% wider)

---

## Expected Impact

| Scenario | Band Width Change | Accuracy Improvement |
|----------|------------------|---------------------|
| End of trading day | -2% to -4% | Better calibration |
| Volatile day (3%+ intraday) | +15% to +25% | Better coverage |
| Overall RMSE | -5% to -10% | Better predictions |
| Hit rate | ≥99% (maintain) | No degradation |

---

## Rollback

If you need to revert:

```bash
# Option 1: Use flags (easiest)
python scripts/predict_close_now.py NDX --days-ahead 5 --no-time-decay --no-intraday-vol

# Option 2: Git revert (permanent)
git log --oneline | grep -i "time-aware"
git revert <commit-hash>
```

---

## Files Modified

1. `scripts/predict_close_now.py` - Main prediction script
2. `scripts/close_predictor/multi_day_predictor.py` - Conditional predictor
3. `scripts/close_predictor/multi_day_lgbm.py` - LGBM predictor
4. `scripts/compare_backtest_results.py` - New comparison tool

---

## Next Steps

1. ✅ **Test basic functionality** (5 min)
   ```bash
   python scripts/predict_close_now.py NDX --days-ahead 5
   # Verify you see "Time decay" and "Intraday vol" in output
   ```

2. ⚠️ **Quick backtest** (30 min) - If backtest script supports flags
   ```bash
   # Run 30-day baseline vs improved comparison
   # See command #4 above
   ```

3. 📊 **Review results** (10 min)
   ```bash
   cat comparison_30d.md
   # Check if hit rate ≥99% and RMSE improved
   ```

4. ✅ **Decision**
   - If results good → keep time-aware features enabled (default)
   - If results bad → disable via flags or revert

---

## Support

- Full documentation: `TIME_AWARE_MULTIDAY_IMPLEMENTATION.md`
- Help: `python scripts/predict_close_now.py --help`
- Issues: Check conversation transcript or implementation file

---

**Quick Tip:** Run predictions at different times of day (morning, midday, near close) and compare the band widths. You should see them narrow as close approaches!
