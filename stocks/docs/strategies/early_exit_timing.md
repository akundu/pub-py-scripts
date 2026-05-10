# Early Exit Timing — Credit Spreads

Empirical analysis of when to close a credit spread before expiration to
maximize capital efficiency. Derived from **8,501 actual entries** repriced
at every 30-min intraday snapshot using live bid/ask quotes from
`options_csv_output_full/`.

## Capital Model

- **$600K total · $50K per trade · 12 concurrent slots**
- Freeing a slot early has value = `days_freed × DTE0_nROI × $50K`
- DTE0 moderate nROI (per day, per $50K slot): SPX $2,805 · NDX $2,815 · RUT $2,860

---

## Key Empirical Findings

### 1. Theta decay is back-loaded

For OTM credit spreads (p90–p99 percentile strikes), most credit decays in
the **final hours** of the position's life, not in the morning. Early exit
rarely improves daily ROI unless the freed slot is immediately redeployed.

### 2. Spread-value stop-losses don't work

Of all trades that hit any intraday drawdown threshold (−50% to −200%), **100%
recovered to a profitable expiry**. A spread-value stop fires on temporary wide
bid-ask quotes, not genuine breaches. Use underlying-price proximity instead.

### 3. For same check-time, reinvestment cancels out in the PT comparison

When comparing 70% vs 75% vs 80% profit target **at the same check time**,
both freed slots earn the same DTE0 reinvestment. The comparison reduces to
simply: how much of the original credit did you capture? This matters because
**80% dominates 70% for DTE1 and DTE3** even after accounting for higher hit
rates at lower thresholds.

---

## Profit Target — Check Times

Close when `pct_captured = (entry_credit − current_spread_value) / entry_credit ≥ PT%`
at the following check times:

| DTE | Check time | Notes |
|-----|-----------|-------|
| DTE0 | SPX @ 12:30 ET · NDX @ 12:00 ET · RUT @ 12:00 ET | p75 of all trades crosses 80% here |
| DTE1 | D0 EOD 15:30 ET (same day as entry) | |
| DTE2 | D1 EOD 15:30 ET (day after entry) | |
| DTE3 | D1 EOD 15:30 ET (day after entry) — frees **2 days** | Most compelling case |

---

## Stop Loss — Underlying Distance

Do **not** use spread value as a stop trigger. Use underlying price proximity
to short strike:

| DTE | Threshold | Check after | Action |
|-----|-----------|-------------|--------|
| DTE0 | within **0.5%** of short strike | 11:00 ET | Close immediately |
| DTE1 | within **1.0%** of short strike | 15:00 ET (EOD) | Close or roll |
| DTE2+ | within **1.5%** of short strike | 15:00 ET (EOD) | Roll (don't close) |

```python
PROXIMITY_THRESHOLD = {0: 0.005, 1: 0.010, 2: 0.015, 3: 0.015}
STOP_AFTER_ET       = {0: "11:00", 1: "15:00", 2: "15:00", 3: "15:00"}

for position in open_positions:
    if now_et < STOP_AFTER_ET[position.dte]:
        continue  # skip open-hour noise
    threshold = PROXIMITY_THRESHOLD[position.dte]
    if is_put  and underlying < short_strike * (1 - threshold):
        action = "close" if dte == 0 else ("close_or_roll" if dte == 1 else "roll")
    if is_call and underlying > short_strike * (1 + threshold):
        action = "close" if dte == 0 else ("close_or_roll" if dte == 1 else "roll")
```

---

## Per-Slot Net Benefit at 80% PT ($50K slot, moderate tier)

`net_benefit = 80% × hold_$ + days_freed × reinvest_$/day − hold_$`

| Ticker | DTE | Check time | Days freed | Early exit $ | Hold $ | Net benefit |
|--------|-----|-----------|-----------|-------------|--------|-------------|
| SPX | 1 | D0 EOD | 1 | $4,360 + $2,805 = $7,165 | $5,450 | **+$1,715** |
| SPX | 2 | D1 EOD | 1 | $4,420 + $2,805 = $7,225 | $5,525 | **+$1,700** |
| SPX | 3 | D1 EOD | 2 | $5,400 + $5,610 = $11,010 | $6,750 | **+$4,260** |
| NDX | 1 | D0 EOD | 1 | $5,760 + $2,815 = $8,575 | $7,200 | **+$1,375** |
| NDX | 2 | D1 EOD | 1 | $4,144 + $2,815 = $6,959 | $5,180 | **+$1,779** |
| NDX | 3 | D1 EOD | 2 | $7,584 + $5,630 = $13,214 | $9,480 | **+$3,734** |
| RUT | 1 | D0 EOD | 1 | $4,520 + $2,860 = $7,380 | $5,650 | **+$1,730** |
| RUT | 2 | D1 EOD | 1 | $4,100 + $2,860 = $6,960 | $5,125 | **+$1,835** |
| RUT | 3 | D1 EOD | 2 | $5,896 + $5,720 = $11,616 | $7,370 | **+$4,246** |

---

## 70% vs 75% vs 80% — Full-Portfolio EV Analysis

The per-trigger net benefit favors 80%. But the correct comparison weights by
hit rate — most trades don't hit the PT threshold at the check time.

### Correct comparison: 100 DTE1 SPX trades at $50K each

Baseline (hold all): 100 × $5,450 = **$545,000**

| PT level | Hit rate | Per-trigger $ | Non-triggered $ | Portfolio total | Lift vs hold | Lift/trade |
|----------|---------|--------------|----------------|----------------|-------------|-----------|
| 70% | ~32% | $3,815 + $2,805 = $6,620 | $5,450 | $582,440 | +$37,440 | **+$374** |
| 75% | ~28% | $4,088 + $2,805 = $6,893 | $5,450 | $585,404 | +$40,404 | **+$404** |
| 80% | ~25% | $4,360 + $2,805 = $7,165 | $5,450 | $587,875 | +$42,875 | **+$429** |

**80% wins for DTE1.** The higher hit rate at lower PTs does not compensate
for the forfeited credit. The lift formula is:

```
lift/trade = hit_rate × (PT_pct × hold_$ + reinvest_$ − hold_$)
           = hit_rate × (reinvest_$ − (1 − PT_pct) × hold_$)
```

### The reinvestment term cancels in the PT comparison

Because 70%, 75%, and 80% are all checked at the **same time** (D0 EOD for
DTE1), and all free the **same number of days** (1), the reinvestment income
is identical across all three. The comparison reduces to:

```
Δ(70 vs 80) = hit_rate_70 × 0.70 × hold_$ − hit_rate_80 × 0.80 × hold_$
            ≈ 0.32 × 0.70 × $5,450 − 0.25 × 0.80 × $5,450
            = $1,221 − $1,090 = +$131 in favor of 80%
```

### DTE2: exception — 70-75% wins on full-portfolio EV

At D1 EOD for DTE2, the capture distribution is wider — more trades sit in
the 40–80% band. Lowering PT to 70% picks up many more triggers:

| PT | Hit rate | Lift/trade (NDX DTE2) |
|----|---------|----------------------|
| 70% | ~42% | **+$530** |
| 75% | ~35% | **+$532** |
| 80% | ~27% | **+$480** |

70–75% beats 80% for DTE2. Use **70%** as the DTE2 check.

---

## DTE0: Where Lower PT Earns More Reinvestment Time

For DTE0, check time shifts earlier when PT is lower, enabling **same-day
redeployment** into a fresh DTE0 spread with more theta-decay time remaining:

| PT | Check time (NDX/RUT) | Remaining day | Extra hour vs 80% | Approx extra income |
|----|---------------------|--------------|-------------------|---------------------|
| 70% | ~11:00–11:30 ET | 3.5–4 hrs | +60 min | ~+$430 on new trade |
| 75% | ~11:30–12:00 ET | 3–3.5 hrs | +30 min | ~+$215 on new trade |
| 80% | ~12:00–12:30 ET | 2.5–3 hrs | baseline | — |

At 70% you give up 10% of original credit (≈$286 for SPX DTE0), but gain
~$430 in extra reinvest time. **Net: ~+$144 per triggered DTE0** in favor of
70% PT — but **only if you have a replacement trade ready immediately**.

Without a queued replacement trade, 80% is strictly better for DTE0.

---

## PT Level Summary by DTE

| DTE | Best PT | Reason |
|-----|---------|--------|
| DTE0 (same-day redeploy available) | **70–75%** | Earlier close = more intraday reinvest time |
| DTE0 (no redeploy queued) | **80%** | Maximize credit captured |
| DTE1 | **80%** | Bimodal D0 EOD distribution; lower PT loses credit without gaining hit rate |
| DTE2 | **70%** | More trades in the 70–80% band; portfolio EV favors lower threshold |
| DTE3 | **75–80%** | Large absolute credit at stake; don't give it up cheaply |
| Any, risk event imminent | **70%** | Hard floor — don't hold through Fed/CPI/earnings |

---

## Pseudocode — Full Exit Logic

```python
PT_CHECK_TIME = {
    "SPX": {0: "12:30", 1: "D0_EOD", 2: "D1_EOD", 3: "D1_EOD"},
    "NDX": {0: "12:00", 1: "D0_EOD", 2: "D1_EOD", 3: "D1_EOD"},
    "RUT": {0: "12:00", 1: "D0_EOD", 2: "D1_EOD", 3: "D1_EOD"},
}

PT_THRESHOLD = {
    0: 0.80,   # DTE0: 80% (or 70-75% if same-day redeploy is queued)
    1: 0.80,   # DTE1: 80%
    2: 0.70,   # DTE2: 70% (portfolio EV favors lower threshold)
    3: 0.80,   # DTE3: 80% (large credit; don't discount cheaply)
}

PROXIMITY_THRESHOLD = {0: 0.005, 1: 0.010, 2: 0.015, 3: 0.015}
STOP_AFTER_ET       = {0: "11:00", 1: "15:00", 2: "15:00", 3: "15:00"}

for position in open_positions:
    spread_value  = short_ask - long_bid
    pct_captured  = (entry_credit - spread_value) / entry_credit * 100
    check_reached = now >= PT_CHECK_TIME[ticker][dte]
    pt            = PT_THRESHOLD[dte] * 100

    # Profit target
    if pct_captured >= pt and check_reached:
        close_and_redeploy()

    # Stop loss (underlying distance)
    threshold = PROXIMITY_THRESHOLD[dte]
    if now_et >= STOP_AFTER_ET[dte]:
        if is_put  and underlying < short_strike * (1 - threshold):
            close_or_roll(dte)
        if is_call and underlying > short_strike * (1 + threshold):
            close_or_roll(dte)

    # Risk-event floor
    if risk_event_imminent and pct_captured >= 70:
        close_and_redeploy()
```

---

## nROI Reference (moderate tier, per day per $50K slot)

| Ticker | DTE0 | DTE1 | DTE2 | DTE3 |
|--------|------|------|------|------|
| SPX | $2,805 (5.61%/day) | $2,865 (5.73%/day) | $1,975 (3.95%/day) | $1,685 (3.37%/day) |
| NDX | $2,815 (5.63%/day) | $3,625 (7.25%/day) | $1,850 (3.70%/day) | $2,370 (4.74%/day) |
| RUT | $2,860 (5.72%/day) | $2,750 (5.50%/day) | $1,835 (3.67%/day) | $1,845 (3.69%/day) |

---

## Files

| File | Purpose |
|------|---------|
| `run_early_exit_analysis.py` | Generates `results/dte_comparison/report_early_exit.html` |
| `run_comprehensive_report.py` | Tab 11 — Early Exit Timing |
| `autoresearch_early_exit/experiment.py` | Simulation engine (profit target + stop loss sweep) |
| `autoresearch_early_exit/loop.py` | Hill-climbing sweep (Phase 1 seeds + Phase 2 hill-climb) |
| `results/dte_comparison/early_exit_raw_obs.parquet` | 29,202 raw observations (one per trade × snapshot) |

## Reproduce

```bash
# Run full early-exit analysis (generates report_early_exit.html)
python3 run_early_exit_analysis.py

# Run comprehensive report (Tab 11 = Early Exit Timing)
python3 run_comprehensive_report.py

# Run autoresearch sweep (hill-climbing, ~400 trials)
python3 autoresearch_early_exit/loop.py --trials 400

# Isolated sweeps
python3 autoresearch_early_exit/loop.py --fix-stop-loss 9999   # PT only
python3 autoresearch_early_exit/loop.py --fix-profit-target 9999  # SL only
```
