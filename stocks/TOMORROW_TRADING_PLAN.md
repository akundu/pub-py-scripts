# Trading Plan for Monday, February 17, 2026

## Backtest Status

**Comprehensive Grid Search:** COMPLETE
- Tested: 504 configurations (7 DTEs x 6 percentiles x 6 widths x 2 flow modes)
- Period: January 16 - February 15, 2026 (20 trading days)
- Processes: 16 cores
- Results: `results/comprehensive_grid_search_full.csv`

**Key Backtest Findings:**
- 'with_flow' mode: ZERO opportunities found across all 20 days (only neutral/iron condor works)
- DTE1 neutral mode dominates all other DTEs
- 0DTE: Very low volume (4-6 spreads/day after fix) - not practical for daily trading
- Best consistency: 90% of trading days have opportunities for top strategies

---

## CONFIRMED OPTIMAL STRATEGIES (From Actual Backtests)

### Strategy #1: DTE1_p95_w50_neutral (PRIMARY RECOMMENDATION)

**Configuration:**
```
Ticker:        NDX
DTE:           1 day (expires Tuesday Feb 18)
Percentile:    95th
Spread Width:  50 points
Flow Mode:     Neutral (Iron Condors)
```

**Backtested Performance (20 days):**
- Practical Score:    0.689 (ranked #3 overall, best balance)
- Consistency:        90% (18 of 20 trading days had opportunities)
- Win Rate:           95.0%
- Daily Opportunities: ~12,956 spread opportunities/day
- Avg Credit:         $23.13/spread
- Avg Max Loss:       $26.87/spread
- Median Daily ROI:   71.1%
- Total 20-Day Vol:   259,115 spreads

**Why This Strategy:**
- Highest consistency (90%) - you can trade this nearly every day
- Large credit collected ($23.13 vs $15.26 for w30)
- 95% win rate over 20 trading days
- Massive volume (~13k opportunities/day) - never runs out of trades
- Balanced risk/reward with 50-point spread width

**Entry Details:**
- Time: 10:00-11:00 AM (after opening volatility settles)
- Strikes: System selects based on 95th percentile
- Position Size: $2,690 max loss per position (1 contract)
- Max Positions: 20-37 (based on capital - see sizing table below)

---

### Strategy #2: DTE1_p95_w30_neutral (CONSERVATIVE ALTERNATIVE)

**Configuration:**
```
Ticker:        NDX
DTE:           1 day (expires Tuesday Feb 18)
Percentile:    95th
Spread Width:  30 points
Flow Mode:     Neutral (Iron Condors)
```

**Backtested Performance (20 days):**
- Practical Score:    0.648 (ranked #10 overall)
- Consistency:        85% (17 of 20 trading days)
- Win Rate:           95.0%
- Daily Opportunities: ~8,950 spread opportunities/day
- Avg Credit:         $15.26/spread
- Avg Max Loss:       $14.74/spread
- Median Daily ROI:   92.8%
- Total 20-Day Vol:   179,010 spreads

**Why This Strategy:**
- Lower max loss ($14.74 vs $26.87 for w50) - better for smaller accounts
- 95% win rate
- Higher median ROI percentage (92.8% vs 71.1%)
- Still very high volume (~9k opportunities/day)
- Good for capital-constrained traders

---

### Strategy #3: DTE1_p98_w30_neutral (HIGH WIN RATE)

**Configuration:**
```
Ticker:        NDX
DTE:           1 day (expires Tuesday Feb 18)
Percentile:    98th
Spread Width:  30 points
Flow Mode:     Neutral (Iron Condors)
```

**Backtested Performance (20 days):**
- Practical Score:    0.620 (ranked #20 overall)
- Consistency:        80% (16 of 20 trading days)
- Win Rate:           98.0% (HIGHEST AMONG PRACTICAL STRATEGIES)
- Daily Opportunities: ~6,943 spread opportunities/day
- Avg Credit:         $14.75/spread
- Avg Max Loss:       $15.25/spread
- Median Daily ROI:   83.5%
- Total 20-Day Vol:   138,868 spreads

**Why This Strategy:**
- Highest win rate (98%) among consistent strategies
- Good volume (~7k opportunities/day)
- Conservative 30-point width with manageable max loss
- Good for risk-averse traders who prioritize win rate

---

## KEY DISCOVERIES FROM BACKTEST

### 1. 'with_flow' Mode Does Not Work
- **ALL 252 with_flow configs had ZERO opportunities**
- Only neutral (iron condor) mode produces viable trades
- Do not attempt directional flow strategies until further investigation

### 2. 0DTE Is Not Practical
- After volume fix (single timestamp filter), 0DTE only has 4-6 spreads/day
- Too few opportunities for meaningful position sizing
- Focus exclusively on DTE1+ strategies

### 3. DTE1 Dominates All Other Expirations
- DTE0: avg ~9 spreads/day (too low)
- DTE1: avg ~89,000 spreads/day (massively better)
- DTE2: avg ~3,296 spreads/day (decent but lower than DTE1)
- DTE3+: drops off significantly

### 4. Wide Spreads (w50, w100) Have Better Consistency
- w50, w100: 90% consistency (18/20 days)
- w30: 80-85% consistency
- w20: 85% consistency
- w10: 85% consistency

### 5. Percentile 95-97 Sweet Spot
- P95-97: Best balance of volume and win rate
- P98-99: Higher win rate but lower consistency
- P100: Good win rate (99.5%) but lower volume

---

## POSITION SIZING BY CAPITAL

### If You Have $50,000:

**Strategy: DTE1_p95_w30_neutral**
```
Max loss per iron condor: $1,474 (1 contract, $14.74 x 100 multiplier)
Max positions: 17 (based on 50% capital risk = $25,000 / $1,474)
Total risk: $25,058 (50% of capital)
Expected credit collected (per day): 17 x $1,526 = $25,942
Expected profit (50% target, 95% win rate):
  Winners (95%): 16 x $763 = $12,208
  Losers (5%): 1 x (-$1,474) = -$1,474
  Net: ~$10,734 (21% return on capital at risk)
```

**Strategy: DTE1_p95_w50_neutral**
```
Max loss per iron condor: $2,687 (1 contract, $26.87 x 100 multiplier)
Max positions: 9 (based on 50% capital risk = $25,000 / $2,687)
Total risk: $24,183 (48% of capital)
Expected credit collected (per day): 9 x $2,313 = $20,817
Expected profit (50% target, 95% win rate):
  Winners (95%): 9 x $1,156 = $10,408
  Losers (5%): 0-1 x (-$2,687) = -$134
  Net: ~$10,274 (20% return on capital at risk)
```

### If You Have $100,000:

**Strategy: DTE1_p95_w30_neutral (RECOMMENDED)**
```
Max loss per iron condor: $1,474 (1 contract)
Max positions: 34 (based on 50% capital risk = $50,000 / $1,474)
Total risk: $50,116 (50% of capital)
Expected credit collected (per day): 34 x $1,526 = $51,884
Expected profit (50% target, 95% win rate):
  Winners (95%): 32 x $763 = $24,416
  Losers (5%): 2 x (-$1,474) = -$2,948
  Net: ~$21,468 (21% return on capital at risk)
```

**Strategy: DTE1_p95_w50_neutral**
```
Max loss per iron condor: $2,687 (1 contract)
Max positions: 18 (based on 50% capital risk = $50,000 / $2,687)
Total risk: $48,366 (48% of capital)
Expected credit collected (per day): 18 x $2,313 = $41,634
Expected profit (50% target, 95% win rate):
  Winners (95%): 17 x $1,156 = $19,660
  Losers (5%): 1 x (-$2,687) = -$2,687
  Net: ~$16,973 (17% return on capital at risk)
```

### If You Have $250,000:

**Strategy: DTE1_p95_w30_neutral**
```
Max loss per iron condor: $1,474 (1 contract)
Max positions: 85 (based on 50% capital = $125,000 / $1,474)
Total risk: $125,290 (50% of capital)
Expected profit: ~$53,670 (21% return on capital at risk)
```

**Strategy: DTE1_p95_w50_neutral**
```
Max loss per iron condor: $2,687 (1 contract)
Max positions: 46 (based on 50% capital = $125,000 / $2,687)
Total risk: $123,602 (49% of capital)
Expected profit: ~$43,422 (17% return on capital at risk)
```

---

## TOMORROW'S SCHEDULE (Feb 17, 2026)

### 9:00-9:25 AM: Pre-Market Prep
- [ ] Check NDX futures movement
- [ ] Check VIX level (if >30, reduce positions by 50%)
- [ ] Review any major news/events
- [ ] Note: Feb 17 is Presidents Day - MARKET IS CLOSED. Plan for TUESDAY Feb 18.

**IMPORTANT: February 17, 2026 is Presidents Day - US markets are CLOSED.**
**This plan applies to Tuesday, February 18, 2026.**

### 10:00-11:00 AM: Primary Entry Window (Tuesday, Feb 18)
- [ ] 10:00: Scan for DTE1 opportunities (p95, w30 or w50, neutral)
- [ ] 10:05-10:30: Enter positions (use limit orders at midpoint)
- [ ] Verify both put and call sides of iron condor
- [ ] Confirm credits received match backtest expectations

### 11:00 AM - 3:30 PM: Monitoring
- [ ] Check positions every 2 hours
- [ ] Close any positions hitting 50% profit target
- [ ] No new entries after 11:00 AM for DTE1

### 4:00-4:30 PM: After Hours
- [ ] Review day's performance
- [ ] Calculate total P&L (positions expire Tuesday = Wednesday morning P&L)
- [ ] Note: DTE1 positions entered Tuesday expire Wednesday Feb 19

---

## RISK MANAGEMENT RULES

### Portfolio-Level Stops
```
STOP if daily unrealized loss reaches 5% of capital
REDUCE sizes if VIX > 30 (cut positions by 50%)
NO NEW ENTRIES if 3 consecutive losing days
CASH RESERVE: Always keep 25% of capital in cash
```

### Position-Level Rules
```
Max loss per iron condor: As specified per strategy
Profit target: 50% of credit received
Stop loss: 150% of credit received (roll or close)
Time stop: Close all DTE1 positions by 3:30 PM on expiration day
```

### VIX-Based Sizing
```
VIX < 20:   Full position size
VIX 20-25:  Full position size
VIX 25-30:  Reduce by 25%
VIX > 30:   Reduce by 50% or skip
VIX > 40:   Skip all new entries
```

---

## ENTRY CHECKLIST

Before entering ANY position:

**Configuration:**
- [ ] DTE = 1 (expires next day)
- [ ] Percentile = 95 (or 98 for Strategy #3)
- [ ] Spread Width = 30 or 50 points
- [ ] Flow Mode = Neutral (Iron Condor, not directional)

**Strike Verification:**
- [ ] Both put and call sides present
- [ ] Strikes are OTM (outside current price)
- [ ] Iron condor structure confirmed

**Credit Verification:**
- [ ] Credit received matches strategy expectation
- [ ] Min credit: >$10/spread (for w30) or >$20/spread (for w50)
- [ ] ROI >50% (credit/max loss ratio)

**Risk Check:**
- [ ] Position size within limits
- [ ] Total portfolio risk <50% of capital
- [ ] VIX level acceptable

---

## MONITORING COMMANDS

To check your strategies using the backtest tools:

```bash
# View top strategies from comprehensive grid search
python3 -c "
import pandas as pd
df = pd.read_csv('results/comprehensive_grid_search_full.csv')
df_active = df[df['total_spreads'] > 0]
df_top = df_active[df_active['flow_mode']=='neutral'].sort_values('composite_score', ascending=False)
print(df_top[['config','consistency_pct','win_rate','avg_credit','avg_max_loss','median_roi']].head(10).to_string())
"

# Run position sizing optimizer
python3 scripts/position_sizing_optimizer.py --results-file results/comprehensive_grid_search_full.csv

# Build optimal portfolio
python3 scripts/portfolio_builder.py --results-file results/comprehensive_grid_search_full.csv
```

---

## IMPORTANT NOTES

1. **MARKET CLOSED FEB 17**: Presidents Day - plan applies to Feb 18 (Tuesday)

2. **No Directional Strategies**: 'with_flow' mode failed to find any opportunities. Use neutral/iron condor only.

3. **No 0DTE Today**: After backtest fix, 0DTE only has 4-6 opportunities/day - not worth it.

4. **DTE1 Is King**: All top strategies are 1-day-to-expiration iron condors. Positions entered Tuesday expire Wednesday.

5. **Check Comprehensive Results**: Full analysis in `results/comprehensive_grid_search_full.csv` (264 configs, 504 tested).

---

## FINAL RECOMMENDATIONS SUMMARY

| Priority | Strategy | Consistency | WinRate | MaxLoss/Pos | Credit/Pos | Daily Opps |
|----------|----------|-------------|---------|-------------|------------|------------|
| PRIMARY  | DTE1_p95_w50_neutral | 90% | 95% | $2,687 | $2,313 | ~12,956 |
| ALT #1   | DTE1_p95_w30_neutral | 85% | 95% | $1,474 | $1,526 | ~8,950 |
| ALT #2   | DTE1_p98_w30_neutral | 80% | 98% | $1,525 | $1,475 | ~6,943 |

**Recommended for most traders: DTE1_p95_w30_neutral**
- Lower max loss ($1,474) = better for risk management
- Higher median ROI% (92.8% vs 71.1%)
- Still 85% consistency (17 of 20 days)
- More capital efficient

**Recommended for larger accounts ($250k+): DTE1_p95_w50_neutral**
- Highest consistency (90%)
- Larger absolute credits ($2,313/contract)
- More opportunities per day

---

**Last Updated:** February 16, 2026, 5:00 PM
**Backtest Period:** January 16 - February 15, 2026 (20 trading days)
**Configurations Tested:** 504 (7 DTEs x 6 percentiles x 6 widths x 2 flow modes)
**Grid Search:** COMPLETE - `results/comprehensive_grid_search_full.csv`
**Next Step:** Run position sizing optimizer and portfolio builder for detailed allocation

**REMEMBER: Markets are CLOSED February 17 (Presidents Day). Next trading day: February 18.**
