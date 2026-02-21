
## Feb 18 Investment Plan
> Market opens after Presidents' Day holiday. Check flow at 06:30 PST.

### Pre-Market Checklist
- [ ] Note NDX futures direction vs Friday close
- [ ] Determine flow direction: NDX > Friday close → with_flow | NDX < Friday close → against_flow
- [ ] Check VIX: if >25 reduce positions by 50%; if >35 skip 0DTE entirely

### Entry Schedule
#### Bucket A — 06:30–06:50 PST (09:30–09:50 ET)
_First 20 minutes. Widest win rates. Use as direction confirmation._

| DTE | Band | Type | Flow | Width | n_contr | WR% | ROI% | Credit/30k | PnL/30k | Sharpe |
|-----|------|------|------|-------|---------|-----|------|-----------|---------|--------|
| 1DTE | P100 | put_spread | with_flow | w=5 | ×42 | 100% | 135.9% | $40,778 | $20,389 | 0.697 |
| 1DTE | P99 | put_spread | with_flow | w=10 | ×30 | 100% | 105.3% | $31,590 | $15,795 | 0.808 |
| 1DTE | P99 | put_spread | with_flow | w=5 | ×30 | 100% | 105.3% | $31,590 | $15,795 | 0.808 |
| 1DTE | P98 | put_spread | with_flow | w=10 | ×35 | 100% | 102.8% | $30,848 | $15,424 | 0.677 |
| 1DTE | P98 | put_spread | with_flow | w=5 | ×35 | 100% | 102.8% | $30,848 | $15,424 | 0.677 |

#### Bucket B — 07:00–08:45 PST (10:00–11:45 ET) ← PRIMARY WINDOW
_Highest avg P&L: $752 at 07:15 PST. Best overall entry zone._

| DTE | Band | Time PST | Type | Flow | Width | n_contr | WR% | ROI% | Credit/30k | PnL/30k | Sharpe |
|-----|------|----------|------|------|-------|---------|-----|------|-----------|---------|--------|
| 1DTE | P95 | 07:15 | iron_condor | against_flow | w=10 | ×12 | 97% | 90.4% | $27,125 | $13,788 | 0.526 |
| 1DTE | P95 | 07:15 | iron_condor | with_flow | w=10 | ×12 | 97% | 90.4% | $27,125 | $13,788 | 0.526 |
| 1DTE | P95 | 07:15 | iron_condor | against_flow | w=5 | ×12 | 97% | 86.6% | $25,973 | $13,224 | 0.502 |
| 1DTE | P95 | 07:15 | iron_condor | with_flow | w=5 | ×12 | 97% | 86.6% | $25,973 | $13,224 | 0.502 |
| 1DTE | P95 | 07:15 | iron_condor | neutral | w=10 | ×11 | 98% | 75.7% | $22,703 | $11,535 | 0.505 |
| 1DTE | P95 | 07:00 | call_spread | against_flow | w=10 | ×27 | 91% | 72.8% | $21,841 | $8,612 | 0.544 |
| 1DTE | P100 | 07:00 | iron_condor | against_flow | w=10 | ×6 | 98% | 64.0% | $19,216 | $9,472 | 0.569 |
| 1DTE | P100 | 07:00 | iron_condor | with_flow | w=10 | ×6 | 98% | 64.0% | $19,216 | $9,472 | 0.569 |

#### Bucket C — 09:00–12:30 PST (12:00–15:30 ET)
_Best Sharpe (0.327 at 12:30 PST). Good for DTE=3/5._

| DTE | Band | Time PST | Type | Flow | Width | n_contr | WR% | ROI% | Credit/30k | PnL/30k | Sharpe |
|-----|------|----------|------|------|-------|---------|-----|------|-----------|---------|--------|
| 1DTE | P95 | 12:00 | iron_condor | with_flow | w=5 | ×12 | 96% | 89.2% | $26,747 | $13,603 | 0.508 |
| 1DTE | P95 | 12:00 | iron_condor | against_flow | w=5 | ×12 | 96% | 89.2% | $26,747 | $13,603 | 0.508 |
| 1DTE | P95 | 12:00 | iron_condor | against_flow | w=10 | ×11 | 96% | 82.8% | $24,847 | $12,429 | 0.518 |
| 1DTE | P95 | 12:00 | iron_condor | with_flow | w=10 | ×11 | 96% | 82.8% | $24,847 | $12,429 | 0.518 |
| 5DTE | P99 | 11:30 | put_spread | with_flow | w=15 | ×4 | 94% | 68.7% | $20,607 | $2,565 | 0.636 |
| 5DTE | P99 | 11:30 | put_spread | with_flow | w=20 | ×4 | 94% | 68.7% | $20,607 | $2,565 | 0.636 |

### Position Sizing ($30k max risk, $500k capital)
| Risk Level | Positions | Max Daily Exposure | Notes |
|------------|-----------|-------------------|-------|
| Conservative (10%) | 1–2 | $60k | ≤2 simultaneous spreads |
| Moderate (25%)     | 4   | $120k | One per major time bucket |
| Aggressive (50%)   | 8   | $240k | Full bucket coverage |

### Rules
1. Determine **with_flow vs against_flow** before ANY entry — check NDX vs Friday close
2. Use **iron_condor** when direction unclear (highest avg dollar P&L: $575/contract)
3. Use **put_spread** on with_flow (bullish) days — 95.1% win rate
4. **n_contracts** in table = how many 1-lot contracts to deploy $30k at avg risk
5. Close at **50% profit target** — don't hold to expiry unless 0DTE
6. DTE=0: close by 15:50 ET if not hit target
7. DTE≥1: close next morning at 09:35 ET if profit target not hit