# Model Retraining Guide

**Last Updated:** February 22, 2026

---

## 🎯 Overview

Prediction models need periodic retraining to:
1. **Adapt to regime changes** (new VIX levels, volatility patterns)
2. **Incorporate recent data** (market evolves over time)
3. **Maintain accuracy** (prevent staleness and drift)
4. **Stay profitable** (band widths must stay tight)

This guide covers when, how, and how often to retrain each model type.

---

## 📅 When to Retrain

### 1. **Scheduled Retraining** (Recommended)

| Model Type | Frequency | Why |
|------------|-----------|-----|
| **Multi-Day Ensemble (LightGBM)** | **Monthly** | Captures recent patterns, prevents staleness |
| **Multi-Day Conditional** | **Quarterly** | Uses raw data (less sensitive to staleness) |
| **0DTE Statistical** | **Monthly** | Intraday patterns change frequently |
| **Baseline (Percentile)** | **Never** | Uses raw historical data (no training) |

**Recommendation:** Retrain all models on the **first Saturday of each month** to avoid market hours.

---

### 2. **Trigger-Based Retraining** (Advanced)

Retrain immediately if any of these conditions are met:

#### Condition 1: Accuracy Degradation
```
Recent RMSE > Validation RMSE × 1.5

Example:
- Validation RMSE: 2.0%
- Recent 30-day RMSE: 3.5%
- Trigger: 3.5 > (2.0 × 1.5) = 3.0 ✓ RETRAIN NOW
```

#### Condition 2: Hit Rate Drop
```
Recent P99 Hit Rate < 95%

Example:
- Expected P99 hit rate: 99%
- Last 30 predictions: 92% hit rate
- Trigger: 92% < 95% ✓ RETRAIN NOW
```

#### Condition 3: Regime Change
```
VIX regime changed and stayed in new regime for 10+ days

Example:
- Previous regime: VIX 12-18 (low/medium)
- New regime: VIX 22-28 (high)
- Days in new regime: 12
- Trigger: Regime changed + 12 > 10 ✓ RETRAIN NOW
```

#### Condition 4: Major Market Event
```
Manually triggered after:
- Market crashes (>5% single-day move)
- Fed policy changes
- Black swan events
- Extended market closures
```

---

## 🔧 How to Retrain Each Model

### 1. Multi-Day Ensemble (LightGBM)

**Models:** 1DTE, 2DTE, 3DTE, 5DTE, 7DTE, 10DTE, 15DTE, 20DTE

**Command:**
```bash
# Retrain on most recent 250 days (1 year)
python scripts/backtest_multi_day.py \
  --ticker NDX \
  --train-days 250 \
  --test-days 30 \
  --max-dte 20 \
  --train-lgbm \
  --output-dir results/retraining_$(date +%Y%m%d)
```

**What This Does:**
1. Loads last 250 days of market data
2. Computes market features (VIX, volume, momentum)
3. Trains 8 separate LightGBM quantile models (one per DTE)
4. Validates on last 30 days
5. Saves models to `results/retraining_YYYYMMDD/models/`
6. Generates performance report

**Time Required:** ~15-30 minutes (depends on data size)

**Output Files:**
```
results/retraining_YYYYMMDD/
├── models/
│   ├── ensemble_1dte.pkl
│   ├── ensemble_2dte.pkl
│   ├── ensemble_5dte.pkl
│   ├── ensemble_10dte.pkl
│   └── ...
├── detailed_predictions.csv
├── summary.csv
└── stats.json
```

---

### 2. Multi-Day Conditional (Feature-Weighted)

**Models:** Same DTEs as Ensemble

**Command:**
```bash
# Conditional doesn't need explicit training (uses raw data)
# But you should validate performance periodically

python scripts/backtest_multi_day.py \
  --ticker NDX \
  --train-days 250 \
  --test-days 30 \
  --max-dte 20 \
  --output-dir results/conditional_validation_$(date +%Y%m%d)
```

**What This Does:**
1. Validates Conditional performance on last 30 days
2. NO model files saved (Conditional uses raw historical data)
3. Generates performance report
4. Use this to verify Conditional is still outperforming

**Time Required:** ~5-10 minutes

**Note:** Conditional doesn't need retraining because it uses raw historical distributions. However, you should validate that it's still performing well.

---

### 3. 0DTE Statistical (LightGBM)

**Models:** Intraday prediction at multiple time slots

**Command:**
```bash
# 0DTE models require different backtest script
python scripts/backtest_comprehensive.py \
  --ticker NDX \
  --lookback 250 \
  --test-days 30 \
  --train-statistical \
  --output-dir results/0dte_retraining_$(date +%Y%m%d)
```

**What This Does:**
1. Trains Statistical LightGBM for intraday predictions
2. Validates on last 30 trading days
3. Saves models for each time slot (9:30, 10:00, ..., 3:30)

**Time Required:** ~20-40 minutes

**Note:** 0DTE Statistical has low hit rates (32-55%). Consider using Combined/Percentile instead, which don't require training.

---

### 4. Baseline (Simple Percentile)

**No retraining needed** - uses raw historical percentiles.

Just ensure the data cache is up to date (happens automatically).

---

## 🤖 Automated Retraining

### Option 1: Monthly Cron Job (Recommended)

Create a cron job that runs retraining automatically.

**Setup:**

1. Create retraining script:

```bash
cat > ~/stocks/retrain_models.sh << 'EOF'
#!/bin/bash
# Automated model retraining script
# Runs on first Saturday of each month

set -e

# Change to project directory
cd /Volumes/RAID1\ NVME\ SSD\ 2TB/akundu_programs_dev/programs/python/pythonv3/pub-py-scripts/stocks

# Activate virtual environment if needed
# source .venv/bin/activate

# Generate timestamp
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_DIR="logs/retraining"
mkdir -p "$LOG_DIR"

echo "================================================================"
echo "AUTOMATED MODEL RETRAINING - $TIMESTAMP"
echo "================================================================"

# 1. Retrain Multi-Day Ensemble (LightGBM)
echo ""
echo "1. Retraining Multi-Day Ensemble (1-20 DTE)..."
python scripts/backtest_multi_day.py \
  --ticker NDX \
  --train-days 250 \
  --test-days 30 \
  --max-dte 20 \
  --train-lgbm \
  --output-dir "results/auto_retrain_$TIMESTAMP" \
  2>&1 | tee "$LOG_DIR/ensemble_$TIMESTAMP.log"

# Check if training succeeded
if [ $? -eq 0 ]; then
    echo "✅ Ensemble retraining complete"

    # Copy models to production directory
    PROD_DIR="models/production"
    mkdir -p "$PROD_DIR"
    cp -r "results/auto_retrain_$TIMESTAMP/models/"* "$PROD_DIR/"

    # Save metadata
    echo "{\"retrained_at\": \"$TIMESTAMP\", \"train_days\": 250, \"test_days\": 30}" > "$PROD_DIR/metadata.json"

    echo "✅ Models deployed to $PROD_DIR"
else
    echo "❌ Ensemble retraining failed"
    exit 1
fi

# 2. Validate Conditional performance
echo ""
echo "2. Validating Multi-Day Conditional..."
python scripts/backtest_multi_day.py \
  --ticker NDX \
  --train-days 250 \
  --test-days 30 \
  --max-dte 20 \
  --output-dir "results/conditional_validation_$TIMESTAMP" \
  2>&1 | tee "$LOG_DIR/conditional_$TIMESTAMP.log"

# 3. Analyze performance
echo ""
echo "3. Analyzing retrained model performance..."
python scripts/analyze_performance_close_prices.py \
  --train-days 250 \
  2>&1 | tee "$LOG_DIR/analysis_$TIMESTAMP.log"

# 4. Send notification (optional)
echo ""
echo "================================================================"
echo "RETRAINING COMPLETE - $TIMESTAMP"
echo "================================================================"
echo "Logs saved to: $LOG_DIR/"
echo "Models saved to: models/production/"

# Optional: Send email notification
# echo "Model retraining completed at $TIMESTAMP" | mail -s "Trading Models Retrained" your@email.com

EOF

chmod +x ~/stocks/retrain_models.sh
```

2. Add to crontab (runs first Saturday of month at 2 AM):

```bash
# Edit crontab
crontab -e

# Add this line (runs at 2 AM on first Saturday of each month)
0 2 * * 6 [ $(date +\%d) -le 7 ] && ~/stocks/retrain_models.sh
```

**Alternative schedule options:**
```bash
# Monthly on 1st at 2 AM
0 2 1 * * ~/stocks/retrain_models.sh

# Every Sunday at 2 AM
0 2 * * 0 ~/stocks/retrain_models.sh

# Last day of month at 2 AM
0 2 28-31 * * [ $(date -d tomorrow +\%d) -eq 1 ] && ~/stocks/retrain_models.sh
```

---

### Option 2: Manual Monthly Retraining

Create a simple wrapper script:

```bash
cat > retrain_now.sh << 'EOF'
#!/bin/bash
# Manual retraining script

cd /Volumes/RAID1\ NVME\ SSD\ 2TB/akundu_programs_dev/programs/python/pythonv3/pub-py-scripts/stocks

TIMESTAMP=$(date +%Y%m%d)

echo "Starting model retraining..."

# Retrain ensemble
python scripts/backtest_multi_day.py \
  --ticker NDX \
  --train-days 250 \
  --test-days 30 \
  --max-dte 20 \
  --train-lgbm \
  --output-dir "results/retrain_$TIMESTAMP"

echo "Retraining complete! Results in: results/retrain_$TIMESTAMP"
EOF

chmod +x retrain_now.sh
```

**Usage:**
```bash
# Run whenever you want to retrain
./retrain_now.sh
```

---

## 📊 Monitoring Retraining Performance

### 1. Compare Before/After Performance

After retraining, compare new models to old ones:

```bash
# Analyze new models
python scripts/analyze_performance_close_prices.py \
  --train-days 250

# Compare to baseline
python scripts/backtest_multi_day.py \
  --ticker NDX \
  --train-days 250 \
  --test-days 30 \
  --max-dte 20 \
  --train-lgbm \
  --output-dir results/comparison_$(date +%Y%m%d)
```

**Look for:**
- ✅ RMSE improvement (lower is better)
- ✅ Hit rate maintained (≥98%)
- ✅ Band widths similar or tighter
- ❌ Red flag: RMSE increased >20%
- ❌ Red flag: Hit rate dropped <95%

---

### 2. Track Performance Over Time

Create a monitoring dashboard:

```python
# scripts/monitor_model_performance.py
import pandas as pd
from pathlib import Path
import json

def track_retraining_history():
    """Track model performance across retraining cycles."""

    history = []
    results_dir = Path('results')

    # Find all retraining runs
    for retrain_dir in sorted(results_dir.glob('retrain_*')):
        summary_file = retrain_dir / 'summary.csv'
        if not summary_file.exists():
            continue

        # Load results
        df = pd.read_csv(summary_file)

        # Extract timestamp from directory name
        timestamp = retrain_dir.name.replace('retrain_', '')

        # Aggregate metrics
        for method in ['baseline', 'conditional', 'ensemble', 'ensemble_combined']:
            method_data = df[df['method'] == method]

            avg_rmse = method_data['avg_midpoint_error'].mean()
            avg_hit_rate = method_data['p99_hit_rate'].mean()
            avg_width = method_data['p99_avg_width'].mean()

            history.append({
                'timestamp': timestamp,
                'method': method,
                'avg_rmse': avg_rmse,
                'avg_hit_rate': avg_hit_rate,
                'avg_width': avg_width,
            })

    history_df = pd.DataFrame(history)
    history_df.to_csv('results/retraining_history.csv', index=False)

    print("\n📊 Model Performance History:")
    print(history_df.to_string())

    # Alert on degradation
    for method in history_df['method'].unique():
        method_history = history_df[history_df['method'] == method].sort_values('timestamp')

        if len(method_history) >= 2:
            latest = method_history.iloc[-1]
            previous = method_history.iloc[-2]

            rmse_change = ((latest['avg_rmse'] / previous['avg_rmse']) - 1) * 100

            if rmse_change > 20:
                print(f"\n⚠️  WARNING: {method} RMSE increased {rmse_change:.1f}% since last retraining!")
            elif rmse_change < -10:
                print(f"\n✅ IMPROVEMENT: {method} RMSE decreased {abs(rmse_change):.1f}% since last retraining!")

if __name__ == '__main__':
    track_retraining_history()
```

**Usage:**
```bash
python scripts/monitor_model_performance.py
```

---

## 🎯 Best Practices

### 1. **Training Window Size**

| Prediction Horizon | Recommended Window | Rationale |
|-------------------|--------------------|-----------|
| 0DTE | 90-180 days (3-6 months) | Recent data most relevant for intraday |
| 1-5 DTE | 180-250 days (6-12 months) | Balance recency and sample size |
| 10-20 DTE | 250-365 days (1-1.5 years) | Need full market cycle |

**Current backtest uses 250 days (1 year) - optimal for most cases.**

---

### 2. **Validation Period**

Always validate on **30-60 days** of out-of-sample data:
- Too small (<20 days): High variance, unreliable
- Too large (>90 days): Overlaps with training, biased

---

### 3. **Model Versioning**

Keep track of model versions:

```bash
# Save models with version
results/
├── models_v1_20260101/
├── models_v2_20260201/
├── models_v3_20260301/
└── production/  # Symlink to latest

# Create symlink to latest
ln -sf models_v3_20260301 production
```

---

### 4. **Rollback Strategy**

If new models perform worse:

```bash
# Revert to previous models
cd models/
rm -rf production
ln -sf models_v2_20260201 production

echo "Rolled back to v2"
```

---

### 5. **Testing Before Deployment**

```bash
# 1. Retrain in test directory
python scripts/backtest_multi_day.py \
  --train-lgbm \
  --output-dir results/test_retrain_$(date +%Y%m%d)

# 2. Validate performance
python scripts/analyze_performance_close_prices.py

# 3. Compare to production
# If RMSE < production RMSE + 0.5%, deploy
# Otherwise, investigate or rollback

# 4. Deploy to production
cp -r results/test_retrain_*/models/* models/production/
```

---

## 🚨 Red Flags (Don't Deploy)

**DO NOT deploy retrained models if:**

1. ❌ **RMSE increased >20%** vs previous version
2. ❌ **P99 hit rate dropped <95%** on validation set
3. ❌ **Band widths increased >30%** (less profitable)
4. ❌ **Training data includes known anomalies** (flash crashes, halts)
5. ❌ **Validation set too small** (<20 days)

**Instead:**
- Investigate root cause
- Adjust training window
- Check data quality
- Keep using previous models

---

## 📋 Retraining Checklist

```
[ ] 1. Verify data quality (no gaps, correct prices)
[ ] 2. Check market regime (VIX, volatility)
[ ] 3. Run retraining script
[ ] 4. Validate on out-of-sample data (30+ days)
[ ] 5. Compare to previous model performance
[ ] 6. Check for red flags (RMSE, hit rate, band width)
[ ] 7. If all good: Deploy to production
[ ] 8. If red flags: Investigate and/or rollback
[ ] 9. Update retraining log
[ ] 10. Monitor first week of production use
```

---

## 📊 Sample Retraining Schedule

### Monthly (1st Saturday)
```
2:00 AM - Automated retraining starts
2:15 AM - Ensemble models retrained
2:30 AM - Conditional validation complete
2:45 AM - Performance analysis generated
3:00 AM - Models deployed to production (if passed checks)
3:15 AM - Email notification sent
```

### Trigger-Based (As Needed)
```
Condition: VIX spiked from 15 to 28, stayed high for 12 days
Action: Manual retrain with last 120 days (capture new regime)
Validation: Test on last 20 days in new regime
Deploy: If RMSE < 3% and hit rate > 95%
```

---

## 🎓 FAQ

### Q: How often should I retrain?

**A:** **Monthly for Ensemble, Quarterly for Conditional**
- Ensemble (LightGBM): Monthly (learns patterns, needs fresh data)
- Conditional: Quarterly (uses raw data, less sensitive)
- 0DTE Statistical: Monthly (intraday patterns change)
- Baseline: Never (raw percentiles)

---

### Q: Can I retrain during market hours?

**A:** **Technically yes, but NOT recommended**
- Training is CPU-intensive (may slow live predictions)
- Better to retrain on weekends or after-hours
- Deploy new models before market open Monday

---

### Q: What if RMSE gets worse after retraining?

**A:** **Don't deploy - rollback or investigate**
1. Check if validation set is representative
2. Try different training window (180-365 days)
3. Verify data quality
4. Check for regime change
5. If still worse: keep old models

---

### Q: Do I need to retrain all DTEs or just some?

**A:** **Retrain all at once (recommended)**
- Ensures consistency across time horizons
- Training 8 DTEs takes ~30 mins total
- Partial retraining can cause discontinuities

---

### Q: How much historical data do I need?

**A:** **Minimum 250 days (1 year), optimal 250-500 days**
- Less than 180 days: High variance, unreliable
- 250 days: Optimal for most predictions
- 500+ days: Good for Conditional, risky for Ensemble

---

## 🔧 Quick Commands Reference

```bash
# Monthly retrain (Ensemble + validation)
python scripts/backtest_multi_day.py --ticker NDX --train-days 250 --test-days 30 --max-dte 20 --train-lgbm --output-dir results/retrain_$(date +%Y%m%d)

# Validate Conditional performance
python scripts/backtest_multi_day.py --ticker NDX --train-days 250 --test-days 30 --max-dte 20 --output-dir results/conditional_val_$(date +%Y%m%d)

# Analyze performance
python scripts/analyze_performance_close_prices.py --train-days 250

# Monitor performance history
python scripts/monitor_model_performance.py

# Walk-forward validation (tests retraining every 30 days)
python scripts/backtest_multi_day.py --ticker NDX --train-days 250 --test-days 180 --max-dte 20 --train-lgbm --walk-forward --step-size 30 --output-dir results/walk_forward
```

---

**Last Updated:** February 22, 2026
**Recommended Schedule:** Monthly (1st Saturday at 2 AM)
**Default Training Window:** 250 days (1 year)
