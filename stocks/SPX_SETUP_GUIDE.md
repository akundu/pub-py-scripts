# SPX Automated Retraining Setup Guide

## Quick Start (5 Minutes)

Your NDX models are already set up and running. Here's how to add SPX:

### Step 1: Run Initial SPX Retraining

```bash
# Train SPX models for the first time
./scripts/retrain_models_auto.sh --ticker SPX

# This will:
# ✅ Train 20 models (1-20 DTE) on SPX data
# ✅ Validate performance (RMSE < 4%, hit rate ≥ 95%)
# ✅ Deploy to models/production/SPX/
# ✅ Create backup in models/backup_SPX_{date}/
# ⏱️  Takes ~2 minutes
```

### Step 2: Verify SPX Models

```bash
# Check model health
python scripts/monitor_model_health.py --ticker SPX

# Expected output:
# ✅ OK: Models healthy
# ✅ All required model files present
# ✅ Validation RMSE: ~1-2%
# ✅ Hit rate: 99-100%
```

### Step 3: Test SPX Predictions

```bash
# Test 0DTE prediction
python scripts/predict_close_now.py SPX

# Test 5-day ahead prediction
python scripts/predict_close_now.py SPX --days-ahead 5

# Should load models from: models/production/SPX/
```

### Step 4: Add Cron Jobs

```bash
# Edit crontab
crontab -e

# Add these two lines (update PROJECT_DIR and email):
0 3 * * 6 [ $(date +\%d) -le 7 ] && cd "$PROJECT_DIR" && ./scripts/retrain_models_auto.sh --ticker SPX >> logs/retraining/cron_SPX.log 2>&1
5 6 * * 1-5 cd "$PROJECT_DIR" && python scripts/monitor_model_health.py --ticker SPX --alert-email your@email.com >> logs/health_checks/health_SPX.log 2>&1
```

**Done!** SPX models will now retrain monthly and health checks run daily.

---

## What You Now Have

### Model Structure

```
models/production/
├── NDX/
│   ├── lgbm_1dte.pkl through lgbm_20dte.pkl  (20 models)
│   └── metadata.json
└── SPX/
    ├── lgbm_1dte.pkl through lgbm_20dte.pkl  (20 models)
    └── metadata.json
```

### Automated Schedule

| Time | Task | Ticker |
|------|------|--------|
| **1st Sat 2AM** | Model Retraining | NDX |
| **1st Sat 3AM** | Model Retraining | SPX |
| **Daily 6AM** | Health Check + Alert | NDX |
| **Daily 6:05AM** | Health Check + Alert | SPX |

### How Predictions Work

1. **User requests prediction:**
   - Web: `/predictions/SPX`
   - API: `/predictions/api/lazy/future/SPX/5`
   - CLI: `python scripts/predict_close_now.py SPX --days-ahead 5`

2. **System loads correct models:**
   - NDX request → loads `models/production/NDX/lgbm_Xdte.pkl`
   - SPX request → loads `models/production/SPX/lgbm_Xdte.pkl`

3. **Fallback order (if ticker-specific models not found):**
   - Priority 1: `models/production/{TICKER}/lgbm_Xdte.pkl` ← **NEW**
   - Priority 2: `models/production/lgbm_Xdte.pkl` ← Legacy single-ticker
   - Priority 3: `models/production/{TICKER}_latest/lgbm_Xdte.pkl` ← Old manual
   - Priority 4: `results/multi_day_backtest/models/lgbm_Xdte.pkl` ← Backtest

---

## Comparison: NDX vs SPX

### NDX Status
- ✅ Models deployed: `models/production/NDX/`
- ✅ Retrained: 2026-02-22 (today)
- ✅ RMSE: 1.55%
- ✅ Hit rate: 100%
- ✅ Automation: Configured (if cron installed)

### SPX Status (After Running Step 1)
- ✅ Models deployed: `models/production/SPX/`
- ✅ Retrained: (today's date)
- ✅ RMSE: ~1-2% (expected)
- ✅ Hit rate: 99-100% (expected)
- ⏸️  Automation: Needs cron installation (Step 4)

---

## Manual Operations

### Check Health
```bash
# Check both tickers
python scripts/monitor_model_health.py --ticker NDX
python scripts/monitor_model_health.py --ticker SPX

# Or use a loop
for ticker in NDX SPX; do
  echo "=== $ticker ==="
  python scripts/monitor_model_health.py --ticker $ticker
  echo ""
done
```

### Force Retrain Both
```bash
# Retrain both NDX and SPX (ignores 25-day threshold)
./scripts/retrain_models_auto.sh --ticker NDX --force
./scripts/retrain_models_auto.sh --ticker SPX --force
```

### Rollback to Backup
```bash
# List available backups
ls -lt models/backup_SPX_*/

# Restore from specific backup
cp -r models/backup_SPX_20260222/* models/production/SPX/

# Verify restoration
python scripts/monitor_model_health.py --ticker SPX
```

---

## Expected Performance

Based on 180-day backtest validation:

### Multi-Day Predictions (1-20 DTE)

| Method | Band Width vs Baseline | Hit Rate | Recommended |
|--------|------------------------|----------|-------------|
| **Conditional** | **37-39% TIGHTER** | 97-99% | ⭐ YES |
| Baseline | Reference (100%) | 100% | Fallback |
| Ensemble | 24-58% WIDER | 100% | Too conservative |

### 0DTE Predictions (Same Day)

| Method | Performance | Recommended |
|--------|-------------|-------------|
| Combined (Blended) | Best balance | ⭐ YES |
| Percentile | Good baseline | Fallback |
| LightGBM | ML-based | Alternative |

**Both NDX and SPX use the same methodology, so expect similar performance.**

---

## Troubleshooting

### SPX retraining fails with "Not enough data"
```bash
# Check if SPX CSV data exists
ls -lh csv_data/historical/SPX/ | head -20

# Verify recent data
python -c "from scripts.csv_prediction_backtest import get_available_dates; \
           dates = get_available_dates('SPX', 260); \
           print(f'SPX has {len(dates)} days of data')"

# Need at least 250 days for training
```

### SPX predictions fail with "Model not found"
```bash
# Check if models were deployed
ls -lh models/production/SPX/

# If empty, run retraining:
./scripts/retrain_models_auto.sh --ticker SPX --force

# Check logs for errors:
tail -100 logs/retraining/retrain_SPX_*.log
```

### Validation fails (RMSE > 4%)
This means SPX market conditions changed significantly. Options:
1. **Reduce training window** (try 125 days instead of 250)
2. **Check data quality** (ensure CSV files are complete)
3. **Wait and retry** (temporary market anomaly)

```bash
# Try with shorter training window (edit script temporarily)
# Or skip validation for manual review:
./scripts/retrain_models_auto.sh --ticker SPX --skip-deploy

# Then review results manually:
cat results/auto_retrain_SPX_{date}/summary.csv
```

---

## File Locations

### Models
- **NDX:** `models/production/NDX/lgbm_{1-20}dte.pkl`
- **SPX:** `models/production/SPX/lgbm_{1-20}dte.pkl`

### Logs
- **NDX Retraining:** `logs/retraining/retrain_NDX_*.log`
- **SPX Retraining:** `logs/retraining/retrain_SPX_*.log`
- **NDX Health:** `logs/health_checks/health_NDX.log`
- **SPX Health:** `logs/health_checks/health_SPX.log`

### Results
- **NDX:** `results/auto_retrain_NDX_{date}/`
- **SPX:** `results/auto_retrain_SPX_{date}/`

### Backups
- **NDX:** `models/backup_NDX_{date}/`
- **SPX:** `models/backup_SPX_{date}/`

---

## Next Steps

1. ✅ **Run Step 1** - Initial SPX retraining (~2 min)
2. ✅ **Verify Step 2** - Check model health
3. ✅ **Test Step 3** - Make test predictions
4. ⏸️  **Add Step 4** - Install cron jobs for automation

**After Step 4, both NDX and SPX will maintain themselves automatically!**

---

## Summary

**What Changed:**
- ✅ Retraining script now accepts `--ticker` parameter
- ✅ Models stored in ticker-specific subdirectories
- ✅ Prediction script auto-detects ticker and loads correct models
- ✅ Health monitor supports both NDX and SPX
- ✅ Backward compatible with existing NDX setup

**What's the Same:**
- Same training methodology (250 days, LightGBM)
- Same validation criteria (RMSE < 4%, hit rate ≥ 95%)
- Same monthly retraining schedule
- Same daily health checks

**Why This Matters:**
- Can trade both NDX and SPX with optimized models
- Each ticker gets market-specific training
- Independent retraining schedules (SPX at 3AM, NDX at 2AM)
- Isolated failures (SPX issues don't affect NDX)

🚀 **You're ready to add SPX automated retraining!**
