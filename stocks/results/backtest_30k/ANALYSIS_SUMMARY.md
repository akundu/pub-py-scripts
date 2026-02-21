# $30k Backtest Grid Analysis — Summary
**Generated:** 2026-02-17  
**Source:** `grid_summary.csv` → `grid_analysis_30k.csv`

---

## Dataset Overview
- **Total configs loaded:** 20,610
- **Total trades:** 712,695
- **Directional configs (with_flow + against_flow):** 12,780
- **Successful configs (all 5 criteria met):** 5,287 (41.4%)
- **Failed configs:** 7,493 (58.6%)

### Success Criteria
All five must pass:
1. `win_rate_pct >= 85%`
2. `roi_pct >= 5%` (avg_credit / avg_max_risk × 100)
3. `n_trades >= 10`
4. `sharpe > 0`
5. `avg_pnl > 0`

---

## Time Buckets
| ID | PST Window | ET Window | Interval |
|----|-----------|-----------|----------|
| A  | 06:30–06:50 | 09:30–09:50 | 10-min |
| B  | 07:00–08:45 | 10:00–11:45 | 15-min |
| C  | 09:00–12:30 | 12:00–15:30 | 30-min |

---

## Best Time Bucket Per DTE (by avg ROI, successful configs)

| DTE | Best Bucket | Avg ROI | Avg Win Rate | Avg P&L | Avg Sharpe |
|-----|-------------|---------|-------------|---------|------------|
| 0   | B (07:00–08:45 PST) | 7.9% | 93.8% | $850 | 0.456 |
| 1   | B (07:00–08:45 PST) | 18.3% | 95.8% | $822 | 0.355 |
| 3   | C (09:00–12:30 PST) | 26.6% | 92.3% | $699 | 0.299 |
| 5   | C (09:00–12:30 PST) | 27.6% | 92.6% | $693 | 0.313 |
| 10  | A (06:30–06:50 PST) | 35.6% | 93.4% | $264 | 0.271 |

---

## Aggregate Stats — Successful Configs by DTE × Flow

| DTE | Flow | # Configs | Avg ROI | Max ROI | Avg Win% | Avg P&L | Avg Sharpe |
|-----|------|-----------|---------|---------|---------|---------|------------|
| 0   | against_flow | 23  | 7.9%  | 18.7%  | 92.4% | $889  | 0.388 |
| 0   | with_flow    | 26  | 7.2%  | 10.5%  | 93.0% | $825  | 0.405 |
| 1   | against_flow | 1032 | 16.2% | 125.3% | 96.3% | $746  | 0.359 |
| 1   | with_flow    | 1203 | 18.6% | 125.3% | 95.8% | $878  | 0.387 |
| 3   | against_flow | 609  | 21.4% | 248.5% | 93.1% | $715  | 0.266 |
| 3   | with_flow    | 839  | 23.2% | 256.8% | 93.3% | $751  | 0.324 |
| 5   | against_flow | 644  | 23.0% | 250.5% | 93.2% | $740  | 0.282 |
| 5   | with_flow    | 723  | 24.8% | 259.1% | 93.7% | $676  | 0.343 |
| 10  | against_flow | 69   | 30.6% | 155.2% | 91.6% | $1,000 | 0.264 |
| 10  | with_flow    | 119  | 24.4% | 119.9% | 92.4% | $695  | 0.264 |

---

## Investment Plan — Feb 18, 2026 (Tuesday, first day after Presidents' Day)
**Capital:** $500k+ | **Max risk per trade:** $30k | **Profit target:** 50%

### Tier 1: Highest Conviction (Sharpe > 1.0, 100% WR, ≥10 trades)

| DTE | Band | Entry (PST/ET) | Bucket | Type | Flow | Width | ROI | AvgCredit | AvgP&L | Sharpe |
|-----|------|---------------|--------|------|------|-------|-----|-----------|--------|--------|
| 5   | P100 | 06:30 / 09:30 | A | put_spread    | against_flow | 300 | 5.2%  | $1,482 | $741   | 1.546 |
| 0   | P97  | 07:00 / 10:00 | B | iron_condor   | with_flow    | 300 | 7.0%  | $2,076 | $2,076 | 1.283 |
| 0   | P97  | 07:00 / 10:00 | B | iron_condor   | against_flow | 300 | 7.0%  | $2,076 | $2,076 | 1.283 |
| 3   | P100 | 07:00 / 10:00 | B | put_spread    | with_flow    | 50  | 28.3% | $1,928 | $964   | 1.269 |
| 1   | P98  | 07:00 / 10:00 | B | put_spread    | with_flow    | 300 | 29.0% | $6,819 | $3,409 | 1.257 |
| 5   | P95  | 12:30 / 15:30 | C | call_spread   | against_flow | 300 | 7.0%  | $1,738 | $869   | 1.142 |
| 3   | P99  | 07:00 / 10:00 | B | call_spread   | with_flow    | 300 | 7.2%  | $2,017 | $1,008 | 1.123 |
| 1   | P98  | 11:30 / 14:30 | C | call_spread   | with_flow    | 250 | 7.3%  | $1,834 | $917   | 1.112 |
| 5   | P99  | 07:00 / 10:00 | B | call_spread   | with_flow    | 300 | 11.8% | $2,848 | $1,424 | 1.089 |
| 10  | P99  | 09:30 / 12:30 | C | put_spread    | with_flow    | 300 | 9.8%  | $2,687 | $1,522 | 1.088 |

### Tier 2: Strong (Sharpe 0.5–1.0, WR ≥ 90%, ≥15 trades, sorted by TotalP&L)

| DTE | Band | Entry (PST/ET) | Bucket | Type | Flow | Width | ROI | AvgCredit | TotalP&L | Sharpe |
|-----|------|---------------|--------|------|------|-------|-----|-----------|----------|--------|
| 1   | P100 | 08:15 / 11:15 | B | iron_condor | with_flow    | 150 | 28.9% | $4,926 | $98,734 | 0.761 |
| 1   | P100 | 08:15 / 11:15 | B | iron_condor | against_flow | 150 | 28.9% | $4,926 | $98,734 | 0.761 |
| 1   | P99  | 08:15 / 11:15 | B | iron_condor | with_flow    | 150 | 21.4% | $3,769 | $97,520 | 0.610 |
| 1   | P99  | 08:15 / 11:15 | B | iron_condor | against_flow | 150 | 21.4% | $3,769 | $97,520 | 0.610 |
| 1   | P97  | 06:30 / 09:30 | A | iron_condor | against_flow | 300 | 19.8% | $5,282 | $92,888 | 0.645 |
| 1   | P97  | 06:30 / 09:30 | A | iron_condor | with_flow    | 300 | 19.8% | $5,282 | $92,888 | 0.645 |
| 1   | P97  | 11:30 / 14:30 | C | iron_condor | against_flow | 150 | 20.6% | $3,288 | $90,404 | 0.647 |
| 1   | P97  | 11:30 / 14:30 | C | iron_condor | with_flow    | 150 | 20.6% | $3,288 | $90,404 | 0.647 |

### Position Sizing ($30k max risk per trade, $500k capital)

| Risk Level | Daily Capital % | Max Positions |
|------------|----------------|---------------|
| Conservative | 10% | 1 |
| Moderate | 25% | 4 |
| Aggressive | 50% | 8 |
| Hard max (capital) | 100% | 16 |

---

## Output Files
- `grid_analysis_30k.csv` — Full annotated grid (12,780 directional rows, 30 columns)
- `ANALYSIS_SUMMARY.md` — This file

