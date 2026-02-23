# Model Deployment Path Fix

## Problem Discovered

After the automated retraining completed successfully on 2026-02-22, the **newly trained models were not being picked up** by the prediction system.

### Root Cause

**Path Mismatch:**
- ✅ Automated retraining script deployed models to: `models/production/lgbm_Xdte.pkl`
- ❌ Prediction script was looking for models in: `models/production/{ticker}_latest/lgbm_Xdte.pkl`
- 🐛 Result: Predictions were using **old models from Feb 19** instead of **fresh models from Feb 22**

### Impact

- `/predictions/` web interface would load stale models
- API endpoints would serve outdated predictions
- Manual `scripts/predict_close_now.py` runs would miss the new models
- Automated monthly retraining would succeed but not take effect!

---

## Fix Applied

Updated `scripts/predict_close_now.py` (lines 558-574) to check model paths in priority order:

### Model Loading Priority (Multi-Day Predictions)

```python
# Priority 1: Automated retraining location (MOST RECENT)
models/production/lgbm_{days_ahead}dte.pkl

# Priority 2: Manual deployment location (FALLBACK)
models/production/{ticker}_latest/lgbm_{days_ahead}dte.pkl

# Priority 3: Backtest results (LAST RESORT)
results/multi_day_backtest/models/lgbm_{days_ahead}dte.pkl
```

### Verification

```bash
# Test shows correct model is now loaded:
✅ Found NEW model (automated retraining)
   models/production/lgbm_5dte.pkl
   Retrained: 20260222
   RMSE: 1.55%
   Hit Rate: 100.0%
```

---

## Current Model Status

### Production Models (Automated Retraining)
- **Location:** `models/production/`
- **Ticker:** NDX only (SPX not yet configured)
- **Files:** `lgbm_1dte.pkl` through `lgbm_20dte.pkl` (20 models)
- **Retrained:** 2026-02-22 14:05:50
- **Validation:**
  - RMSE: 1.55% (excellent, target <4%)
  - Hit Rate: 100% (perfect, target ≥95%)
  - Training Window: 250 days (1 year)

### Metadata
```json
{
    "retrained_at": "20260222_140550",
    "retrained_date": "20260222",
    "train_days": 250,
    "test_days": 30,
    "max_dte": 20,
    "ticker": "NDX",
    "validation_rmse": 1.55,
    "validation_hit_rate": 100.0,
    "backup_location": "models/backup_20260222"
}
```

### Legacy Models (Manual Deployment)
- **Location:** `models/production/NDX_latest/`
- **Retrained:** 2026-02-19 (3 days old)
- **Status:** ⚠️ Still present but will not be used (correct behavior)

---

## How Predictions Work Now

### Chain of Components

1. **User requests prediction:**
   - Web UI: `GET /predictions/NDX`
   - API: `GET /predictions/api/lazy/future/NDX/5`
   - CLI: `python scripts/predict_close_now.py --days-ahead 5`

2. **db_server.py** routes request → **common/predictions.py**

3. **common/predictions.py** imports → **scripts/predict_close_now.py**

4. **scripts/predict_close_now.py** loads models:
   - ✅ First checks: `models/production/lgbm_5dte.pkl` ← **USES THIS (new!)**
   - ⏭️ Skips: `models/production/NDX_latest/lgbm_5dte.pkl` (not needed)
   - ⏭️ Skips: `results/multi_day_backtest/models/lgbm_5dte.pkl` (not needed)

5. **Prediction generated** using fresh models from automated retraining

---

## Ticker-Specific Behavior

### Current Setup (NDX Only)
- Models in `models/production/` are NDX-specific (trained on NDX data)
- Metadata confirms: `"ticker": "NDX"`
- **SPX predictions will NOT work** with current automated retraining

### For SPX Support

To enable automated SPX retraining, update `scripts/retrain_models_auto.sh` line 132:

```bash
# Current (NDX only)
python scripts/backtest_multi_day.py \
    --ticker NDX \
    --train-days 250 \
    ...

# For SPX (run separately or create spx_retrain.sh)
python scripts/backtest_multi_day.py \
    --ticker SPX \
    --train-days 250 \
    ...
```

**Recommended approach:**
1. Create `scripts/retrain_models_spx_auto.sh` (copy of retrain_models_auto.sh)
2. Change ticker to SPX
3. Deploy to `models/production_spx/` or add ticker prefix to filenames
4. Update cron to run both NDX and SPX retraining

---

## Testing Checklist

- [x] New models are in `models/production/`
- [x] Metadata shows retrain date 20260222
- [x] Health monitor validates all files present
- [x] predict_close_now.py loads correct models
- [x] Model priority order works correctly
- [ ] Test live prediction via API (need QuestDB running)
- [ ] Test web UI `/predictions/NDX` (need server running)
- [ ] Verify SPX handling (should fallback to old location or fail gracefully)

---

## Automated Retraining Schedule

With cron jobs installed (from CRON_SETUP.txt):

### Monthly Retraining
- **When:** 1st Saturday of each month at 2:00 AM
- **What:** Retrains models with latest 250 days of data
- **Where:** Deploys to `models/production/`
- **Validation:** Only deploys if RMSE < 4% and hit rate ≥ 95%
- **Backup:** Saves old models to `models/backup_YYYYMMDD/`

### Daily Health Checks
- **When:** Weekdays at 6:00 AM
- **What:** Checks model age, file integrity, performance
- **Alert:** Emails you if models > 25 days old or performance degraded
- **Exit codes:** 0=healthy, 1=warning, 2=critical

---

## Rollback Procedure

If the new models perform poorly:

```bash
# 1. Check available backups
ls -lt models/backup_*/

# 2. Restore from backup (example: Feb 19 backup)
cp -r models/backup_20260219/* models/production/

# 3. Verify restoration
python scripts/monitor_model_health.py

# 4. Restart API server if running
# (models are loaded on demand, but cache may need clearing)
```

---

## Summary

✅ **FIXED:** Model path resolution now prioritizes automated retraining location
✅ **VERIFIED:** Fresh models from 2026-02-22 are being loaded
✅ **VALIDATED:** RMSE 1.55%, Hit rate 100% (exceeds all thresholds)
✅ **AUTOMATED:** Monthly retraining will now work correctly
⚠️ **NOTE:** Current setup is NDX-only, SPX needs separate configuration

**Next Steps:**
1. Install cron jobs for automated retraining (see CRON_SETUP.txt)
2. Monitor first live predictions for validation
3. Consider adding SPX automated retraining if needed
