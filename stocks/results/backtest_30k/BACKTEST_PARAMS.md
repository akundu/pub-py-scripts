# NDX Credit Spread Backtest Parameters — Session Reference
_Created: 2026-02-17 | Reload for daily reassessment_

## Backtest Parameters
```
Script:          scripts/comprehensive_backtest.py
Capital/trade:   $30,000 max risk
Profit target:   50% of credit received
Backtest window: 90 trading days
Band window:     100 rolling calendar days of price history
DTEs tested:     0, 1, 3, 5, 10
Spread types:    put_spread, call_spread, iron_condor
Flow modes:      with_flow, against_flow, neutral
Spread widths:   50, 100, 150, 200, 250, 300 NDX points
Processes:       18
```

## Run Commands
```bash
# Backtest (90 days, $30k risk):
cd /Volumes/RAID1\ NVME\ SSD\ 2TB/akundu_programs_dev/programs/python/pythonv3/pub-py-scripts/stocks
python scripts/comprehensive_backtest.py \
  --processes 18 --backtest-days 90 --band-days 100 \
  --dte 0 1 3 5 10 --cache --out-dir results/backtest_30k

# Re-run WITHOUT cache (fresh data):
python scripts/comprehensive_backtest.py \
  --processes 18 --backtest-days 90 --band-days 100 \
  --dte 0 1 3 5 10 --out-dir results/backtest_30k
```

## Constants (scripts/comprehensive_backtest.py lines 35-60)
```python
MAX_RISK      = 30_000    # $ max risk per trade
PROFIT_TGT    = 0.50      # exit at 50% of credit
BAND_DAYS     = 100       # rolling window for percentile bands
BACKTEST_DAYS = 90        # days to backtest
DTE_TARGETS   = [0, 1, 3, 5, 10]
SPREAD_WIDTHS = [50, 100, 150, 200, 250, 300]
BAND_CONFIGS  = {
    'P95':  (2.5, 97.5),  # 95th pct — widest strikes tested
    'P97':  (1.5, 98.5),
    'P98':  (1.0, 99.0),
    'P99':  (0.5, 99.5),
    'P100': (0.0, 100.0), # 100th pct — tightest strikes (min/max of 100d moves)
}
# Time buckets (ET times → displayed in PST = ET-3h):
# Bucket A: 09:30-09:50 ET = 06:30-06:50 PST (10-min intervals)
# Bucket B: 10:00-11:45 ET = 07:00-08:45 PST (15-min intervals)
# Bucket C: 12:00-15:30 ET = 09:00-12:30 PST (30-min intervals)
```

## Success Criteria (grid_analysis_30k.csv column: `successful`)
A config is marked `successful=True` when ALL of:
- `win_rate_pct >= 85%`
- `roi_pct >= 5%`  (roi = avg_credit / avg_max_risk × 100)
- `n_trades >= 10`
- `sharpe > 0`
- `avg_pnl > 0`

## Key Results (90-day backtest, $30k/trade)
| Metric | Value |
|--------|-------|
| Total configs tested | 25,650 |
| Non-empty configs | 20,610 |
| Total trades | 712,695 |
| Directional configs | 12,780 |
| Successful (all criteria) | 5,287 (41.4%) |

## Flow Mode Results
| Flow | Avg Win% | Avg P&L | Avg Sharpe |
|------|---------|---------|------------|
| with_flow    | 92.5% | $713 | 0.35 |
| against_flow | 93.1% | $681 | 0.31 |
| neutral      | 93.0% | $511 | 0.23 |

## Best Entry Times (PST)
| Rank | Time PST | ET | Avg P&L | Avg Sharpe |
|------|----------|----|---------|------------|
| 1 | 07:15 | 10:15 | $877 | 0.272 |
| 2 | 08:45 | 11:45 | $783 | 0.285 |
| 3 | 09:00 | 12:00 | $736 | 0.241 |
| 4 | 11:30 | 14:30 | $700 | 0.334 |
| 5 | 12:30 | 15:30 | $619 | 0.357 |

## Avoid
- DTE=10 (negative aggregate Sharpe: -0.229)
- Call spread standalone (negative avg P&L)
- Entry times: 08:00, 11:00 PST (negative avg P&L)
- P95 band alone (weakest ROI and Sharpe)

## Output Files
| File | Description |
|------|-------------|
| `results/backtest_30k/grid_summary.csv` | Raw grid (20,610 configs) |
| `results/backtest_30k/all_trades.csv` | All 712,695 trades |
| `results/backtest_30k/grid_analysis_30k.csv` | Annotated with ROI, buckets, success flag |
| `results/backtest_30k/grid_dte{N}.csv` | Per-DTE breakdown |
| `results/backtest_30k/INVESTMENT_PLAN_feb18.md` | Today's trading plan |
| `results/backtest_30k/BACKTEST_PARAMS.md` | This file — reload tomorrow |

## Tomorrow's Reassessment Checklist
1. Re-run backtest without --cache to include today's data
2. Compare new grid vs this one — did best configs hold?
3. Track actual trades vs predicted win rates
4. Adjust position sizing if win rate diverges > 10% from backtest
5. Note any new configs entering top 10 by sharpe