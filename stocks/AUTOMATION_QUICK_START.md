# Automated Model Retraining - Quick Start Guide

**Last Updated:** February 22, 2026

---

## 🚀 Quick Setup (5 Minutes)

### Step 1: Make Scripts Executable

```bash
cd /Volumes/RAID1\ NVME\ SSD\ 2TB/akundu_programs_dev/programs/python/pythonv3/pub-py-scripts/stocks

chmod +x scripts/retrain_models_auto.sh
chmod +x scripts/monitor_model_health.py
chmod +x setup_automation.sh
```

---

### Step 2: Run Setup Script

```bash
./setup_automation.sh
```

This creates directories and tests the scripts.

---

### Step 3: Install Cron Jobs

```bash
# Open crontab editor
crontab -e

# Copy these lines (update paths and email!):
```

**Paste these cron entries:**

```cron
# ============================================================================
# Trading Model Automation
# ============================================================================

SHELL=/bin/bash
PATH=/usr/local/bin:/usr/bin:/bin
MAILTO=your@email.com  # UPDATE THIS

# Project directory - UPDATE THIS!
PROJECT_DIR=/Volumes/RAID1\ NVME\ SSD\ 2TB/akundu_programs_dev/programs/python/pythonv3/pub-py-scripts/stocks

# Monthly retraining: 1st Saturday at 2:00 AM
0 2 * * 6 [ $(date +\%d) -le 7 ] && cd "$PROJECT_DIR" && ./scripts/retrain_models_auto.sh >> logs/retraining/cron_$(date +\%Y\%m).log 2>&1

# Daily health check: Weekdays at 6:00 AM
0 6 * * 1-5 cd "$PROJECT_DIR" && python scripts/monitor_model_health.py --alert-email your@email.com >> logs/health_checks/health_$(date +\%Y\%m).log 2>&1

# Weekly validation: Sundays at 3:00 AM
0 3 * * 0 cd "$PROJECT_DIR" && python scripts/analyze_performance_close_prices.py --train-days 250 >> logs/validation/weekly_$(date +\%Y\%m\%d).log 2>&1

# Monthly cleanup: 1st of month at 4:00 AM
0 4 1 * * find "$PROJECT_DIR/logs" -name "*.log" -mtime +90 -delete
0 4 1 * * cd "$PROJECT_DIR/results" && ls -dt auto_retrain_* 2>/dev/null | tail -n +7 | xargs rm -rf 2>/dev/null || true
```

**Save and exit:** ESC → `:wq` (vim) or CTRL+X (nano)

**Verify:**
```bash
crontab -l
```

---

## ✅ What Just Happened?

You now have **fully automated model management**:

### 📅 **Monthly Retraining** (1st Saturday @ 2 AM)
- Automatically retrains all models (1-20 DTE)
- Uses last 250 days (1 year) of data by default (configurable via `--train-days`)
- Validates on last 30 days
- Deploys to production if quality checks pass
- Backs up old models
- Sends email notification

### 🩺 **Daily Health Checks** (Weekdays @ 6 AM)
- Checks model age
- Verifies all files present
- Monitors recent performance
- **Sends email alert** if retraining needed

### 📊 **Weekly Validation** (Sundays @ 3 AM)
- Analyzes model performance
- Tracks accuracy trends
- Logs results for review

### 🧹 **Monthly Cleanup** (1st @ 4 AM)
- Removes logs older than 90 days
- Keeps last 6 retraining results
- Frees up disk space

---

## 🎮 Manual Commands

### Check Model Health
```bash
python scripts/monitor_model_health.py
```

**Output:**
```
================================================================================
MODEL HEALTH CHECK
================================================================================
Status: ✅ OK: Models healthy

1. MODEL AGE
  Days old: 12
  Last retrained: 20260210

2. MODEL FILES
  All required model files present

3. RECENT PERFORMANCE
  RMSE: 1.85% (threshold: 4.0%)
  Hit Rate: 98.3% (threshold: 95%)

RECOMMENDATION: No action needed - models healthy
================================================================================
```

---

### Manual Retraining
```bash
# Show full help screen with all options
./scripts/retrain_models_auto.sh --help

# Force retrain (skip age check)
./scripts/retrain_models_auto.sh --force

# Retrain but don't deploy (test first)
./scripts/retrain_models_auto.sh --force --skip-deploy

# Use a custom training window (default: 250 days / ~1 year)
./scripts/retrain_models_auto.sh --force --train-days 120   # 6 months
./scripts/retrain_models_auto.sh --force --train-days 500   # 2 years
```

---

### Analyze Performance
```bash
# Quick analysis
python scripts/analyze_performance_close_prices.py

# Full day intraday analysis
python scripts/analyze_performance_close_prices.py --show-intraday

# With training window comparison
python scripts/analyze_performance_close_prices.py --compare-windows
```

---

## 📊 Monitoring Dashboards

### View Recent Logs

```bash
# Latest retraining log
tail -100 logs/retraining/cron_$(date +%Y%m).log

# Latest health check
tail -50 logs/health_checks/health_$(date +%Y%m).log

# Latest validation
cat logs/validation/weekly_$(date +%Y%m%d).log
```

---

### Check Last Retraining

```bash
cat models/production/metadata.json
```

**Output:**
```json
{
  "retrained_at": "20260201_020015",
  "retrained_date": "20260201",
  "train_days": 250,
  "test_days": 30,
  "ticker": "NDX",
  "validation_rmse": 1.85,
  "validation_hit_rate": 98.3
}
```

---

## 🔔 Email Notifications

You'll receive emails for:

✅ **Successful retraining** (if enabled in script)
⚠️ **Health check warnings** (models >25 days old)
🚨 **Critical alerts** (models >40 days old or missing files)

**To enable email notifications:**

1. Edit `scripts/retrain_models_auto.sh`
2. Uncomment the last line:
```bash
# Line 298: Uncomment this
echo "Model retraining completed successfully" | mail -s "✅ Models Retrained" your@email.com
```

---

## 🚨 Troubleshooting

### Cron Job Not Running?

**Check cron logs:**
```bash
# macOS
log show --predicate 'process == "cron"' --last 1h

# Linux
grep CRON /var/log/syslog
```

**Verify paths:**
```bash
# Make sure PROJECT_DIR is correct in crontab
crontab -l | grep PROJECT_DIR
```

---

### Retraining Failed?

**Check logs:**
```bash
tail -200 logs/retraining/cron_$(date +%Y%m).log
```

**Common issues:**
- Python virtual environment not activated
- Insufficient disk space
- Data not available
- Network issues (if fetching remote data)

---

### Health Check Shows "Critical"?

**Run manual health check:**
```bash
python scripts/monitor_model_health.py
```

**If models missing:**
```bash
# Initial training (first time)
./scripts/retrain_models_auto.sh --force
```

**If models too old:**
```bash
# Retrain now
./scripts/retrain_models_auto.sh --force
```

---

## 📋 Validation Checklist

After setting up automation, verify:

```bash
# 1. Check cron is installed and running
which cron || echo "Cron not found"

# 2. Verify crontab entries
crontab -l

# 3. Check scripts are executable
ls -la scripts/retrain_models_auto.sh
ls -la scripts/monitor_model_health.py

# 4. Test health monitoring
python scripts/monitor_model_health.py

# 5. Test retraining (dry run - won't actually retrain if models <25 days old)
./scripts/retrain_models_auto.sh

# 6. Check directory structure
ls -la logs/
ls -la models/
ls -la results/
```

---

## 🎯 Next Steps

### First Time Setup

1. **Initial Training** (run once):
```bash
./scripts/retrain_models_auto.sh --force
```

2. **Verify Deployment**:
```bash
ls -la models/production/
cat models/production/metadata.json
```

3. **Test Health Check**:
```bash
python scripts/monitor_model_health.py
```

4. **Wait for Cron** (or test manually):
```bash
# Test daily health check
cd "$PROJECT_DIR" && python scripts/monitor_model_health.py --alert-email your@email.com

# Test weekly validation
cd "$PROJECT_DIR" && python scripts/analyze_performance_close_prices.py --train-days 250
```

---

### Ongoing Maintenance

**Monthly (after automated retraining):**
1. Check email for retraining notification
2. Review performance report in logs
3. Verify new models deployed to production

**Weekly:**
1. Review validation results
2. Check health status

**On Alert:**
1. Check health status: `python scripts/monitor_model_health.py`
2. Review logs: `tail -100 logs/retraining/cron_*.log`
3. Manually retrain if needed: `./scripts/retrain_models_auto.sh --force`

---

## 📚 Documentation

- **`MODEL_RETRAINING_GUIDE.md`** - Comprehensive retraining guide
- **`CRON_SETUP.txt`** - Detailed cron configuration
- **`ENSEMBLE_VS_CONDITIONAL_EXPLAINED.md`** - Model comparison
- **`MODEL_PERFORMANCE_EVIDENCE.md`** - Performance proof
- **`AUTOMATION_QUICK_START.md`** - This file

---

## 🎓 Summary

**You now have:**
✅ Automated monthly retraining
✅ Daily health monitoring with email alerts
✅ Weekly performance validation
✅ Automatic cleanup of old files
✅ Manual control when needed
✅ Rollback capability if issues arise

**Hands-off operation:**
- Models retrain themselves monthly
- You get email alerts if issues
- Logs track everything
- Backups prevent data loss

**Manual override available anytime:**
- Show help: `./scripts/retrain_models_auto.sh --help`
- Force retrain: `./scripts/retrain_models_auto.sh --force`
- Custom window: `./scripts/retrain_models_auto.sh --force --train-days 120`
- Check health: `python scripts/monitor_model_health.py`
- Analyze: `python scripts/analyze_performance_close_prices.py`

---

**Last Updated:** February 22, 2026
**Status:** Production Ready
