# Percentile Backtest - Commands Reference

This document lists all commands used for running percentile-based backtests and analyzing results.

---

## Initial Backtest (13 Configs)

**Command:**
```bash
python scripts/run_percentile_backtest.py \
  --ticker NDX \
  --start-date 2026-02-03 \
  --end-date 2026-02-13 \
  --processes 4 \
  --output results/backtest_comprehensive.csv
```

**Runtime**: 52 seconds
**Configs**: 13
**Results**: `results/backtest_comprehensive.csv`, `BACKTEST_RESULTS_SUMMARY.md`

---

## Phase 1 Test Mode (18 Configs)

**Command:**
```bash
python scripts/run_percentile_backtest_phase1.py \
  --start-date 2026-02-03 \
  --end-date 2026-02-13 \
  --processes 4 \
  --test-mode \
  --output results/phase1_test.csv
```

**Runtime**: 71 seconds
**Configs**: 18 (3 DTEs × 3 percentiles × 2 widths)
**Purpose**: Validate Phase 1 script before full run

---

## Phase 1 Comprehensive (300 Configs) ✅

**Command:**
```bash
python scripts/run_percentile_backtest_phase1.py \
  --start-date 2026-01-05 \
  --end-date 2026-02-13 \
  --processes 6 \
  --output results/phase1_comprehensive.csv
```

**Runtime**: 150 minutes (2.5 hours)
**Configs**: 300 (5 DTEs × 6 percentiles × 5 widths × 2 flow modes)
**Trading Days**: 29
**Success Rate**: 252/300 configs (84%)
**Total Spreads**: 24,525,672
**Results**: `results/phase1_comprehensive.csv`

---

## Analyze Phase 1 Results

**Command:**
```bash
python scripts/analyze_phase1_results.py \
  --input results/phase1_comprehensive.csv \
  --top-n 20
```

**Output**:
- Console: Detailed analysis by ROI, volume, consistency, risk-adjusted score
- File: `results/phase1_top_configs.csv` (top 20 for Phase 2)
- Summary: `PHASE1_RESULTS_SUMMARY.md`

---

## Run Single-Day Test

**Command (1 DTE, p99, 20pt spread):**
```bash
python scripts/test_percentile_spreads.py \
  --ticker NDX \
  --date 2026-02-10 \
  --dte 1 \
  --percentile 99 \
  --spread-width 20 \
  --flow-mode neutral \
  --profit-target-pct 0.5
```

**Use Cases**:
- Test new configurations quickly
- Validate strategy on specific trading days
- Debug spread construction logic

---

## Run Multi-Day Test

**Command (3-7 DTE range):**
```bash
python scripts/test_percentile_spreads.py \
  --ticker NDX \
  --start-date 2026-02-03 \
  --end-date 2026-02-13 \
  --dte 3 \
  --percentile 95 \
  --spread-width 50 \
  --flow-mode neutral \
  --profit-target-pct 0.5
```

---

## Production Strategy Commands

### Strategy A: Best Overall (DTE 1, p99, 20pt)
```bash
python scripts/test_percentile_spreads.py \
  --ticker NDX \
  --date 2026-02-XX \
  --dte 1 \
  --percentile 99 \
  --spread-width 20 \
  --flow-mode neutral \
  --profit-target-pct 0.5
```
**Expected**: ~2,930 spreads/day, 527.5% ROI, 72.4% consistency

---

### Strategy B: Highest Volume (DTE 1, p99, 30pt)
```bash
python scripts/test_percentile_spreads.py \
  --ticker NDX \
  --date 2026-02-XX \
  --dte 1 \
  --percentile 99 \
  --spread-width 30 \
  --flow-mode neutral \
  --profit-target-pct 0.5
```
**Expected**: ~3,440 spreads/day, 435.6% ROI, 75.9% consistency

---

### Strategy C: Most Consistent (0 DTE, p98, 20pt)
```bash
python scripts/test_percentile_spreads.py \
  --ticker NDX \
  --date 2026-02-XX \
  --dte 0 \
  --percentile 98 \
  --spread-width 20 \
  --flow-mode neutral \
  --profit-target-pct 0.5
```
**Expected**: ~1,931 spreads/day, 355.2% ROI, 86.2% consistency

---

### Strategy D: Conservative (DTE 3, p95, 50pt)
```bash
python scripts/test_percentile_spreads.py \
  --ticker NDX \
  --date 2026-02-XX \
  --dte 3 \
  --percentile 95 \
  --spread-width 50 \
  --flow-mode neutral \
  --profit-target-pct 0.5
```
**Expected**: ~130 spreads/day, 274.2% ROI, 76.0% consistency

---

## Running Tests

### Unit Tests
```bash
# Test percentile strike selector
pytest tests/test_percentile_strike_selector.py -v

# Test exit strategy manager
pytest tests/test_exit_strategy_manager.py -v

# Test momentum detector
pytest tests/test_momentum_detector.py -v

# Run all tests
pytest tests/test_percentile*.py tests/test_exit*.py tests/test_momentum*.py -v
```

**Status**: 24/24 tests passing ✅

---

## Parameter Reference

### DTEs Available
- `--dte 0` - Same day expiration (0DTE)
- `--dte 1` - Next day expiration (best overall performance)
- `--dte 3` - 3 days out
- `--dte 5` - 5 days out
- `--dte 10` - 10 days out

### Percentiles Available
- `--percentile 95` - 5% breach risk (conservative)
- `--percentile 96` - 4% breach risk
- `--percentile 97` - 3% breach risk (balanced)
- `--percentile 98` - 2% breach risk
- `--percentile 99` - 1% breach risk (aggressive, best consistency)
- `--percentile 100` - 0% historical breach (max move)

### Spread Widths
- `--spread-width 15` - Tight spreads, highest ROI, lower consistency
- `--spread-width 20` - **Optimal for most cases**
- `--spread-width 25` - Good balance
- `--spread-width 30` - Higher premium
- `--spread-width 50` - Wide spreads, best consistency (82.4%)

### Flow Modes
- `--flow-mode neutral` - **Recommended** - Iron condors, no directional bias
- `--flow-mode with_flow` - ⚠️ Data quality issues, needs momentum integration fix
- `--flow-mode against_flow` - Not yet tested

### Profit Targets
- `--profit-target-pct 0.3` - 30% of max profit
- `--profit-target-pct 0.5` - **Recommended** - 50% of max profit
- `--profit-target-pct 0.7` - 70% of max profit

---

## Troubleshooting

### Database Connection Error
```bash
# Set environment variable first
export QUEST_DB_STRING="postgresql://user:pass@host:port/db"

# Then run command
python scripts/test_percentile_spreads.py ...
```

### No Spreads Found
- Try wider spread width (50pt instead of 20pt)
- Try lower percentile (p95 instead of p99)
- Check if options data exists for that date
- Verify CSV files in `options_csv_output/` (0DTE) or `options_csv_output_full/` (1+ DTE)

### Unrealistic ROI Values
- Filter results with analysis script (removes ROI > 10,000%)
- Issue mainly affects with_flow mode
- Use neutral mode for clean results

---

## File Locations

### Input Data
- **0DTE options**: `options_csv_output/NDX/`
- **Non-0DTE options**: `options_csv_output_full/NDX/`

### Output Files
- **Initial backtest**: `results/backtest_comprehensive.csv`
- **Phase 1 test**: `results/phase1_test.csv`
- **Phase 1 full**: `results/phase1_comprehensive.csv`
- **Top configs**: `results/phase1_top_configs.csv`

### Documentation
- **Initial results**: `BACKTEST_RESULTS_SUMMARY.md`
- **Phase 1 results**: `PHASE1_RESULTS_SUMMARY.md`
- **Implementation status**: `PERCENTILE_IMPLEMENTATION_STATUS.md`
- **Commands reference**: `BACKTEST_COMMANDS_REFERENCE.md` (this file)

### Scripts
- **Single/multi-day test**: `scripts/test_percentile_spreads.py`
- **Multiprocessing (13 configs)**: `scripts/run_percentile_backtest.py`
- **Phase 1 grid (300 configs)**: `scripts/run_percentile_backtest_phase1.py`
- **Results analyzer**: `scripts/analyze_phase1_results.py`

---

## Quick Start

**To test the #1 recommended strategy today:**
```bash
python scripts/test_percentile_spreads.py \
  --ticker NDX \
  --date $(date +%Y-%m-%d) \
  --dte 1 \
  --percentile 99 \
  --spread-width 20 \
  --flow-mode neutral \
  --profit-target-pct 0.5
```

**To run a custom backtest:**
```bash
python scripts/run_percentile_backtest_phase1.py \
  --start-date 2026-01-15 \
  --end-date 2026-02-15 \
  --processes 6 \
  --output results/my_backtest.csv

# Then analyze
python scripts/analyze_phase1_results.py \
  --input results/my_backtest.csv \
  --top-n 20
```

---

**Last Updated**: February 15, 2026
**Status**: ✅ All systems operational
**Next**: Phase 2 grid search or production deployment
