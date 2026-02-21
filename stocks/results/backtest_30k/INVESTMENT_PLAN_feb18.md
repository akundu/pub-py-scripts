# NDX Credit Spread Investment Plan — Feb 18, 2026
> Next trading day after Presidents' Day holiday (market closed Feb 17)
> Generated: 2026-02-17

## Setup
- **Ticker:** NDX (Nasdaq-100 Index, $100 multiplier)
- **Capital per trade:** $30,000 max risk
- **Profit target:** Close at 50% of credit received
- **Exit:** Market order when credit decays to 50%
- **Positions today:** 1-4 (conservative), 5-8 (moderate)

## FLOW CHECK (do this first at market open)
Compare NDX price now vs:
- **0DTE:** vs 10 minutes ago → if higher = with_flow (bullish), lower = against_flow (bearish)
- **>0DTE:** vs yesterday's EOD close → if higher = with_flow, lower = against_flow
- Note: BOTH with_flow and against_flow are valid — pick matching spread type

## ENTRY SCHEDULE

### 06:30–06:50 PST (09:30–09:50 ET) — Bucket A
_First 20 minutes. High win rate, lower ROI. Good for confirming direction._

| Priority | DTE | Band | Type | Flow | Width | ROI% | AvgCredit | AvgP&L | Sharpe |
|----------|-----|------|------|------|-------|------|-----------|--------|--------|
| #1 | 5DTE | P100 | put_spread | against_flow | w=300 | 5.2% | $1,482 | $741 | 1.546 |
| #2 | 5DTE | P99 | call_spread | with_flow | w=300 | 5.6% | $1,582 | $791 | 1.070 |
| #3 | 0DTE | P98 | iron_condor | against_flow | w=300 | 8.1% | $2,376 | $1,892 | 1.024 |

### 07:00–08:45 PST (10:00–11:45 ET) — Bucket B
_PRIMARY window. Best avg P&L ($877 at 07:15 PST). Most opportunities._

| Priority | DTE | Band | Type | Flow | Width | ROI% | AvgCredit | AvgP&L | Sharpe |
|----------|-----|------|------|------|-------|------|-----------|--------|--------|
| #1 | 1DTE | P99 | call_spread | with_flow | w=100 | 72.0% | $6,786 | $705 | 0.609 |
| #2 | 5DTE | P95 | call_spread | with_flow | w=100 | 60.7% | $6,400 | $836 | 0.625 |
| #3 | 3DTE | P100 | iron_condor | with_flow | w=300 | 59.0% | $13,759 | $4,322 | 0.617 |
| #4 | 3DTE | P100 | iron_condor | against_flow | w=300 | 59.0% | $13,759 | $4,322 | 0.617 |
| #5 | 5DTE | P100 | iron_condor | with_flow | w=300 | 57.1% | $14,254 | $1,027 | 0.557 |

### 09:00–12:30 PST (12:00–15:30 ET) — Bucket C
_SECONDARY window. Best Sharpe (0.357 at 12:30 PST). Best for multi-DTE._

| Priority | DTE | Band | Type | Flow | Width | ROI% | AvgCredit | AvgP&L | Sharpe |
|----------|-----|------|------|------|-------|------|-----------|--------|--------|
| #1 | 10DTE | P97 | call_spread | against_flow | w=300 | 104.8% | $12,050 | $2,220 | 0.559 |
| #2 | 1DTE | P97 | call_spread | against_flow | w=50 | 63.7% | $2,170 | $882 | 0.897 |
| #3 | 1DTE | P100 | call_spread | with_flow | w=300 | 52.5% | $8,999 | $3,727 | 0.732 |
| #4 | 3DTE | P99 | put_spread | against_flow | w=50 | 49.2% | $2,900 | $1,025 | 0.635 |

## POSITION SIZING
| Risk Level | Max Positions | Max Daily Loss | Target Daily Gain |
|------------|--------------|----------------|-------------------|
| Conservative (10%) | 1–2 | $60k | $10–30k |
| Moderate (25%)     | 4   | $120k | $30–60k |
| Aggressive (50%)   | 8   | $240k | $60–120k|

## RULES
1. **Check flow direction** before entering any trade
2. **Iron condor** = best for neutral/unclear days (highest avg dollar P&L)
3. **Put spread** = best for with_flow (bullish) days (highest win rate 95.9%)
4. **Avoid call spread standalone** — use only as part of iron condor
5. **Skip DTE=10** entirely — negative aggregate Sharpe
6. **Preferred DTE:** 1 (most opportunities, robust) or 0 (same-day, fast)
7. **Profit target:** Always close at 50% credit captured
8. **Hard stop:** Close any position at 2× credit debit (150% of max expected loss)

## RISK MANAGEMENT
- Max risk per trade: $30,000 (verified by: spread_width × 100 − credit_received ≤ $30,000)
- NDX multiplier: ×100 (each point = $100)
- For iron condor: 4 legs total (short put + long put + short call + long call)
- Hold overnight only for DTE>0 — exit 0DTE before 15:55 ET
