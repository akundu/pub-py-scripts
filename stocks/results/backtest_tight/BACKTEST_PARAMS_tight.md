# NDX Backtest Parameters — Tight Spreads ($30k/trade)
_Reload tomorrow to reassess_

## Run Command
```bash
cd '/Volumes/RAID1 NVME SSD 2TB/akundu_programs_dev/programs/python/pythonv3/pub-py-scripts/stocks'
python scripts/comprehensive_backtest.py \
  --processes 18 --backtest-days 90 --band-days 100 \
  --dte 0 1 3 5 10 --cache --out-dir results/backtest_tight
# Drop --cache to load fresh data including today
```

## Spread Width Caps
| Band | 0DTE | 1DTE (+20%) | 3DTE (+72.8%) |
|------|------|------------|--------------|
| P95  | ≤30  | ≤36        | ≤52          |
| P97  | ≤30  | ≤36        | ≤52          |
| P98  | ≤40  | ≤48        | ≤69          |
| P99  | ≤50  | ≤60        | ≤86          |
| P100 | ≤50  | ≤60        | ≤86          |

## Key Results
| Metric | Value |
|--------|-------|
| Configs tested | 49,248 |
| Non-empty | 40,944 |
| Total trades | 1,480,236 |
| Successful | 19,633 (48.0%) |

## Best Entry Time (PST): 07:15 (10:15 ET) — avg P&L $752/contract
## Best Flow: with_flow (avg P&L $441) slightly beats against_flow ($461 Sharpe)
## Avoid DTE=10 (negative aggregate Sharpe)