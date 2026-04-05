# autoresearch — Roll Cost Optimization

Autonomous research to find the cheapest way to roll breached 0DTE credit spreads
on **RUT** and **NDX**, minimizing:
1. Net roll cost (debit paid or credit received)
2. Worst-case liability (95th percentile debit)
3. Number of contracts needed (smaller position = less risk)
4. Spread width (narrower = less capital tied up)

## Data Sources

- **Equities**: `equities_output/I:RUT/`, `equities_output/I:NDX/` (5-min OHLCV)
- **Options**: `options_csv_output_full_5/RUT/`, `options_csv_output_full_5/NDX/` (5-min 0DTE chains)

## Tool

```bash
python scripts/roll_cost_table.py --ticker RUT --start 2025-10-01 --end 2026-03-27 \
    --spread-width 20 \
    --entry-breach-pcts 100 75 50 25 0 -5 \
    --target-breach-pcts 100 75 50 25 0 \
    --check-times 09:00 09:30 10:00 10:30 11:00 11:30 12:00 12:15 12:30 12:45 12:55 \
    --roll-dtes 1 2 3 4 5 \
    --options-dir ./options_csv_output_full_5 \
    --output-dir results_auto/roll_cost/rut_full
```

## Optimization Metric

For each (breach_level, option_type) combination, find the (DTE, target_breach, time) that:
- **Minimizes avg net cost** (most negative = most credit)
- **Maximizes credit probability** (% of days you receive credit, not pay debit)
- **Minimizes worst-case** (95th percentile — the worst 1-in-20 day)

Composite score: `avg_net * (1 + credit_pct/100)` — lower is better.

## Key Findings (617K observations, 122-123 days each)

### The Decision Matrix

| Breach | RUT Put | RUT Call | NDX Put | NDX Call | Action |
|--------|---------|---------|---------|---------|--------|
| **-5% (OTM)** | -$8.31 (92%) | -$7.94 (97%) | -$16.55 (83%) | -$20.54 (87%) | **ROLL NOW** |
| **0% (ATM)** | -$7.92 (92%) | -$7.64 (95%) | -$15.76 (81%) | -$19.60 (85%) | **ROLL** |
| **25% ITM** | -$4.85 (79%) | -$4.95 (83%) | -$9.06 (68%) | -$12.36 (74%) | **ROLL** |
| **50% ITM** | -$2.55 (57%) | -$2.35 (65%) | -$4.43 (35%) | -$3.92 (55%) | **MARGINAL** |
| **75% ITM** | -$2.65 (54%) | -$1.73 (45%) | -$3.04 (34%) | +$0.03 (38%) | **AVOID for NDX** |
| **100% ITM** | -$2.84 (53%) | -$2.06 (49%) | -$3.55 (37%) | +$1.23 (35%) | **ACCEPT LOSS** |

### Seven Rules for Minimum-Cost Rolls

1. **Roll at 12:55 PM PT** for puts (all breaches) and calls (0-50% breach). Credit prob goes from ~50% morning to 79-97% at close.

2. **Roll calls 75-100% breach at 11:00 AM** — deeply breached calls get WORSE at close (33% credit at 12:55 vs 44% at 11:00).

3. **DTE+1 is almost always optimal** — more time value in DTE+2-5 doesn't compensate for the wider bid/ask and lower fill rate.

4. **Puts: same strikes (target=100%)** — keeping the position in place costs least because you avoid strike-moving slippage.

5. **Calls: target 25% (move 75% toward ATM)** on RUT. **Calls: same strikes** on NDX. Different tickers have different call-side dynamics.

6. **RUT rolls are 2-3x cheaper than NDX** in absolute dollars, and have 10-20% higher credit probability at every breach level. RUT is the safer ticker to roll.

7. **Width 20-25pt is most capital-efficient** for RUT (24-25% of width as credit). For minimum absolute cost, use 10pt spreads. For minimum worst-case, use 10-15pt.

### Minimum Contracts Needed

The cheapest roll requires the fewest contracts. For a given total position:
- **Wider spread + fewer contracts** = fewer rolls needed but higher cost per roll
- **Narrower spread + more contracts** = more rolls needed but lower cost each

At 25% breach, DTE+1, 12:55 PM:
| Width | Credit/Spread | Worst/Spread | For $100K position: Contracts | Rolls to Cover |
|-------|--------------|-------------|------------------------------|----------------|
| 10pt | -$2.20 | +$3.71 | 100 contracts | 100 rolls max |
| 20pt | -$4.85 | +$3.55 | 50 contracts | 50 rolls max |
| 25pt | -$6.14 | +$4.09 | 40 contracts | 40 rolls max |

**20pt width is the sweet spot**: fewest rolls needed per dollar of exposure, best credit probability (79-83%), and moderate worst case.

### Bid/Ask Slippage

Real-world slippage costs ~$0.60/spread ($1,200 per 20-lot). Even with slippage:
- 0% breach at 12:55: -$7.29 avg credit (88% probability)
- 25% breach at 12:55: -$4.34 avg credit (74% probability)

## Experiment Loop Parameters

When optimizing, test these dimensions:

1. **Time**: every 5 minutes from 12:00-12:55 (granular sweet spot search)
2. **Width**: 5, 10, 15, 20, 25, 30 points
3. **DTE**: 1, 2, 3, 4, 5
4. **Target breach**: 0%, 25%, 50%, 75%, 100%
5. **Bid/ask vs mid**: real-world slippage impact
6. **Ticker**: RUT vs NDX cross-comparison

Record results in `results_auto/roll_cost/roll_cost.csv`.
