# Breaking Changes: Training Dates Now Required

## ⚠️ IMPORTANT: Breaking Change

**Effective immediately**, the backtesting framework **requires** separate training and prediction date ranges. This is a breaking change that affects all existing scripts and workflows.

## What Changed

### Training Dates Are Now Mandatory

Previously, training dates were optional. Now they are **required** and must be explicitly specified:

- `--training-start`: Training start date (REQUIRED)
- `--training-end`: Training end date (REQUIRED)

### Training Dates Must Differ from Prediction Dates

To ensure proper train/test split and prevent data leakage, the system now enforces:

- ✅ Training dates **MUST** be different from prediction dates
- ✅ Both training dates must be specified
- ✅ Training period cannot be the same as prediction period
- ⚠️ System warns if training period overlaps with prediction period

## Why This Change?

This change enforces machine learning best practices:

1. **Prevents Data Leakage**: Models cannot be trained on the same data they're tested on
2. **Realistic Evaluation**: Performance metrics reflect true out-of-sample results
3. **Production Readiness**: Ensures models are validated properly before deployment
4. **Scientific Rigor**: Follows standard ML/AI validation methodology

## How to Migrate

### Step 1: Identify Affected Scripts

Any script using the backtesting CLI needs to be updated. Look for:

```bash
# Old format (will no longer work)
python -m stock_backtest.cli.main --symbols AAPL --strategy markov_int --start 2023-01-01 --end 2024-10-31
```

### Step 2: Add Training Dates

Update all scripts to include training dates:

```bash
# New format (required)
python -m stock_backtest.cli.main \
  --symbols AAPL \
  --strategy markov_int \
  --training-start 2020-01-01 \
  --training-end 2023-12-31 \
  --start 2024-01-01 \
  --end 2024-10-31
```

### Step 3: Ensure Separate Periods

Make sure training and prediction dates are different:

- ❌ **Invalid**: `--training-start 2023-01-01 --training-end 2024-10-31 --start 2023-01-01 --end 2024-10-31`
- ✅ **Valid**: `--training-start 2020-01-01 --training-end 2022-12-31 --start 2023-01-01 --end 2024-10-31`

### Step 4: Update Configuration Files

If using YAML/JSON configuration files, add training dates:

```yaml
# config.yaml
training_start: "2020-01-01"
training_end: "2023-12-31"
start: "2024-01-01"
end: "2024-10-31"
```

## Common Migration Patterns

### Pattern 1: Recent Data Analysis

**Before:**
```bash
--start 2024-01-01 --end 2024-10-31
```

**After:**
```bash
--training-start 2020-01-01 --training-end 2023-12-31 \
--start 2024-01-01 --end 2024-10-31
```

### Pattern 2: Multi-Year Backtest

**Before:**
```bash
--start 2020-01-01 --end 2024-10-31
```

**After:**
```bash
--training-start 2015-01-01 --training-end 2019-12-31 \
--start 2020-01-01 --end 2024-10-31
```

### Pattern 3: Short-Term Trading

**Before:**
```bash
--start 2024-06-01 --end 2024-10-31
```

**After:**
```bash
--training-start 2023-01-01 --training-end 2024-05-31 \
--start 2024-06-01 --end 2024-10-31
```

## Error Messages

### Missing Training Dates

```
ValueError: Training dates are required. Please specify both --training-start 
and --training-end. Training dates must be different from prediction dates to 
ensure proper train/test split.
```

**Solution**: Add both `--training-start` and `--training-end` arguments.

### Same Training and Prediction Dates

```
ValueError: Training dates cannot be the same as prediction dates. Use separate 
date ranges for training and prediction to avoid data leakage.
```

**Solution**: Use different date ranges for training and prediction.

### Overlapping Periods (Warning)

```
UserWarning: Training period (2023-12-31) overlaps with prediction period 
(2023-06-01). This may cause data leakage. Consider using non-overlapping periods.
```

**Solution**: Adjust dates so training ends before prediction begins.

## Validation Rules

The system now enforces these rules:

1. ✅ `training_start_date < training_end_date`
2. ✅ `start_date < end_date`
3. ✅ `training_start_date != start_date OR training_end_date != end_date`
4. ⚠️ Warning if `training_end_date > start_date` (overlapping periods)

## Recommended Practices

### Best Practice: Non-Overlapping Periods

```bash
# RECOMMENDED: Training ends before prediction begins
--training-start 2020-01-01 --training-end 2022-12-31 \
--start 2023-01-01 --end 2024-10-31
```

### Acceptable: Small Gap

```bash
# ACCEPTABLE: Small gap between training and prediction
--training-start 2020-01-01 --training-end 2023-11-30 \
--start 2023-12-01 --end 2024-10-31
```

### Avoid: Overlapping Periods

```bash
# NOT RECOMMENDED: Training overlaps with prediction
--training-start 2020-01-01 --training-end 2024-06-30 \
--start 2024-01-01 --end 2024-10-31
# This triggers a warning about potential data leakage
```

## Timeline

- **Before**: Training dates optional, defaulted to prediction dates
- **Now**: Training dates required, must differ from prediction dates
- **Impact**: All existing scripts must be updated

## Need Help?

1. See `TRAIN_TEST_SPLIT_CHANGES.md` for detailed technical documentation
2. Check `examples/train_test_split_example.sh` for working examples
3. Review `examples/train_test_split_config.yaml` for configuration examples

## Quick Reference

### Required Arguments
- `--training-start YYYY-MM-DD` [REQUIRED]
- `--training-end YYYY-MM-DD` [REQUIRED]
- `--start YYYY-MM-DD` [optional, default: 90 days ago]
- `--end YYYY-MM-DD` [optional, default: today]

### Validation
- Training dates cannot equal prediction dates
- Both training dates must be provided
- Warning if periods overlap

### Example Command
```bash
python -m stock_backtest.cli.main \
  --symbols AAPL \
  --strategy markov_int \
  --training-start 2020-01-01 \
  --training-end 2023-12-31 \
  --start 2024-01-01 \
  --end 2024-10-31
```

---

**Questions?** Run `python -m stock_backtest.cli.main --help` for full documentation.

