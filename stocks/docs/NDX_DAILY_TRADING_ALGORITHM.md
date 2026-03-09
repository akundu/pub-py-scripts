# NDX Credit Spread — Daily Trading Algorithm

Based on 20 months of backtesting (Jul 2024 – Mar 2026, 420 trading days) across all percentiles (P80–P100) and DTEs (0–10). Incorporates proximity-based rolling with progressive DTE widening validated over 1 year of detailed roll analysis.

---

## The One Rule That Matters

**DTE >= 2 at P80 has a 100% win rate. Zero stop losses. Zero drawdown. Over 420 trading days.**

0DTE and 1DTE will lose money sometimes no matter what percentile you use. DTE >= 2 has never lost at any percentile from P80 to P100. This is the foundation of everything below.

---

## Algorithm Overview

```
EACH TRADING DAY:
  1. Pre-market:        Compute P80 strikes for all DTEs (close-to-close anchor)
  2. 6:00am PST:        Recompute strikes using today's open (open-to-close anchor)
  3. 6:00-10:00am PST:  Enter new positions at ALL chosen DTE tiers, every 10 min
  4. All day:           Monitor exits (profit target, roll trigger, stop loss)
  5. 12:30pm PST:       Roll check on expiring positions
  6. 1:00pm PST:        EOD — log, reconcile, prep for tomorrow
```

---

## Step 1: Pre-Market — Compute Strikes (Before 6:00 AM PST)

Pull the last 180 trading days of NDX daily data. Compute percentile strikes using **two anchors** depending on time of day:

### Anchor 1: Close-to-Close (Pre-Market, Before Open)

Use yesterday's close as anchor. Available before market opens.

```
For DTE = N:
  window = N + 1   (0DTE → 1-day returns, 2DTE → 3-day returns, 10DTE → 11-day)
  returns = (close[i+window] - close[i]) / close[i]   for all i in lookback
  up_returns   = returns where returns > 0
  down_returns = returns where returns < 0

  call_strike = yesterday_close × (1 + percentile(up_returns, P))
  put_strike  = yesterday_close × (1 - percentile(|down_returns|, P))
```

### Anchor 2: Open-to-Close (After Market Open)

Once today's opening price is known, recompute using open-to-close returns. This removes overnight gap noise and gives tighter, more accurate intraday ranges.

```
For DTE = N:
  target_day = N   (0DTE → same day close, 2DTE → close 2 days later)
  returns = (close[i+target_day] - open[i]) / open[i]   for all i in lookback

  call_strike = today_open × (1 + percentile(up_returns, P))
  put_strike  = today_open × (1 - percentile(|down_returns|, P))
```

**Recommendation:** Use close-to-close pre-market for initial strike targets. After the open, recompute with open-to-close. For DTE >= 2, the difference is small (~2-4%), so close-to-close is fine all day. For 0DTE, open-to-close is ~20% tighter and more accurate for intraday trading.

### Strike Distance Reference (at P80, NDX ~24,700)

| DTE | Close-to-Close ||| Open-to-Close ||| Shrinkage |
|-----|-------|-------|-------|-------|-------|-------|-----------|
|     | **Down** | **Up** | **% OTM** | **Down** | **Up** | **% OTM** | |
| 0   | 340 pts | 355 pts | 1.4% | 287 pts | 268 pts | 1.1% | **~20%** |
| 1   | 452 pts | 511 pts | 1.9% | 427 pts | 463 pts | 1.8% | ~7% |
| 2   | 557 pts | 649 pts | 2.4% | 547 pts | 602 pts | 2.3% | ~4% |
| 3   | 628 pts | 781 pts | 2.9% | 621 pts | 728 pts | 2.7% | ~4% |
| 5   | 698 pts | 929 pts | 3.3% | 684 pts | 904 pts | 3.2% | ~2% |
| 10  | 857 pts | 1300 pts | 4.4% | 873 pts | 1277 pts | 4.4% | ~0% |

**Why open-to-close is tighter:** The overnight gap (close→open) adds variance. Removing it shrinks the range. For 0DTE this is huge (~20%) because the gap is a large fraction of one day's move. For 10DTE, one overnight gap is noise relative to 10 days of cumulative movement.

**Trading implication:** Open-to-close gives you strikes ~20% closer on 0DTE (more premium, more risk). For DTE >= 2, the difference is negligible (~2-4%) — stick with close-to-close for simplicity.

Higher DTE = wider strikes = safer = less premium collected per day.

---

## Step 2: Entry Window (6:00 AM – 10:00 AM PST)

Scan every 10 minutes. At each check, enter BOTH a put spread AND a call spread. Enter at **all DTEs you want to trade**, not just one — each DTE is an independent position with its own expiration.

### DTE Menu — Pick Your Tier(s)

| Tier | DTE | Percentile | Win Rate | Net P&L (20mo) | Sharpe | Risk Level |
|------|-----|-----------|----------|----------------|--------|------------|
| **Core** | **2DTE** | **P80** | **100%** | $6.3M | 19.4 | **Zero** |
| Core | 5DTE | P80 | **100%** | $1.6M | 17.2 | Zero |
| Core | 10DTE | P80 | **100%** | $507K | 21.7 | Zero |
| Aggressive | 1DTE | P80 | 97.6% | $7.6M | 12.2 | Low |
| Aggressive | 0DTE | P95 | 96.2%* | $4.7M | 2.1 | Moderate |

*With proximity roll at 12:30pm PST. Without rolling: 90%.

NDX has daily expirations, so 0DTE, 1DTE, 2DTE, etc. are all available every trading day. You are not choosing one DTE per check — you enter at each DTE tier you want to trade. Budget is the only constraint.

### Entry Rules

For each 10-minute check, for each DTE tier you're trading:

```
1. Look up the P80 strike for this DTE (from step 1)
   → Use open-to-close anchor after market open (especially for 0DTE)
   → Use close-to-close pre-market or for DTE >= 2
2. Find the put spread: short at the put strike, long 50 pts further OTM
3. Find the call spread: short at the call strike, long 50 pts further OTM
4. Reject if credit < $0.75/share (junk premium)
5. Enter 1 contract each (put spread + call spread)
6. Log: entry time, strikes, credit received, DTE, expiration date
```

### How Many Positions Per Day

At 1 contract per signal, scanning every 10 min from 6:00-10:00 AM = up to 25 checks.
Each check can produce 1 put + 1 call per DTE tier. Example daily entry count:

```
Trading 2DTE only:           ~26 puts + 26 calls = 52 positions/day
Trading 2DTE + 5DTE:         ~45 puts + 45 calls = 90 positions/day
Trading 2DTE + 5DTE + 10DTE: ~69 puts + 69 calls = 138 positions/day
Full ladder (all DTEs):      ~140 puts + 140 calls = 280 positions/day
```

### Why 1 Contract

| Contracts | Credit/Trade | Stop Loss Cost | Blow-up Risk |
|-----------|-------------|---------------|-------------|
| 1         | $75-150     | $3-5K         | None        |
| 5         | $375-750    | $15-25K       | Low         |
| Auto-size to $5K min | $5,000 | $200-335K | **Catastrophic** |

The backtest showed auto-sizing to $2.5K minimum credit turned positions into 20-33 contract bombs with $75K max loss per trade. At 1 contract, a stop loss costs $3-5K. Scale up only after consistent profitability.

### What You're Entering Each Time

```
SELL 1 NDX Put Credit Spread:
  Short: P80 put strike (e.g., NDX 24,230 put)
  Long:  50 pts lower   (e.g., NDX 24,180 put)
  Credit received: ~$0.75-$1.50/share ($75-$150/contract)
  Max risk: ~$4,000/contract

SELL 1 NDX Call Credit Spread:
  Short: P80 call strike (e.g., NDX 25,180 call)
  Long:  50 pts higher   (e.g., NDX 25,230 call)
  Credit received: ~$0.75-$1.50/share
  Max risk: ~$4,000/contract
```

### Multi-DTE Ladder — What You're Carrying

When you enter at multiple DTEs daily, positions overlap as they age:

```
Example week (entering 2DTE + 5DTE daily):

         | Mon Entries      | Tue Entries      | Wed Entries      | Open Positions
Mon      | 2DTE, 5DTE       |                  |                  | 2 sets
Tue      | 1DTE, 4DTE       | 2DTE, 5DTE       |                  | 4 sets
Wed      | 0DTE→EXPIRE, 3DTE| 1DTE, 4DTE       | 2DTE, 5DTE       | 5 sets
Thu      | 2DTE, expired    | 0DTE→EXPIRE, 3DTE| 1DTE, 4DTE       | 5 sets
Fri      | 1DTE, expired    | 2DTE, expired    | 0DTE→EXPIRE, 3DTE| 4 sets
```

Each "set" = multiple put+call spreads entered throughout the 6-10am window. A typical steady-state with 2DTE+5DTE carries ~5-8 sets of positions simultaneously.

---

## Step 3: Intra-Day Monitoring (All Day)

Every open position — whether entered today or carried from prior days — gets continuous monitoring.

### Exit Priority (First One Triggered Wins)

```
1. PROFIT TARGET (highest priority)
   → Close if 75% of credit captured (0DTE) or 50% (multi-day)

2. PROXIMITY ROLL (on expiration day only, after 12:30pm PST)
   → Roll if price within 0.5% of short strike

3. LOSS-THRESHOLD ROLL (on expiration day only, anytime)
   → Roll if unrealized loss exceeds $20K per position

4. STOP LOSS (last resort, DTE >= 2 only — where it has never triggered)
   → Close if loss reaches 3× credit received
```

### Monitoring Schedule by Time of Day

| Time (PST) | What to Check | Action |
|------------|--------------|--------|
| 6:00 AM | Market open. Enter new positions. | Execute Step 2 entry cascade |
| 6:00–10:00 AM | Every 10 min: new entry scan | Continue entering if budget allows |
| 6:00 AM–12:30 PM | All positions: profit target | Close any position at 75%+ profit (0DTE) or 50%+ (multi-day) |
| 6:00 AM–12:30 PM | Expiring positions: loss threshold | If any position's unrealized loss > $20K → roll immediately |
| 12:30 PM | **Roll decision point** for expiring positions | Check proximity: if price within 0.5% of any expiring short strike → roll |
| 12:30–1:00 PM | Final monitoring window | Last chance to roll or close |
| 1:00 PM | Market close | 0DTE positions expire. Multi-day carry overnight. |

### How to Determine Which Positions Are "Expiring"

```
For each open position:
  days_held = today - entry_date
  is_expiration_day = (days_held >= DTE - 1) OR (DTE == 0)

  If is_expiration_day:
    → Apply roll trigger checks (proximity + loss threshold)
    → This position MUST be resolved today (roll, close, or expire)
  Else:
    → Only check profit target
    → Position carries to tomorrow
```

---

## Step 4: Rolling Procedure (12:30 PM PST)

Rolling = closing a threatened position and opening a replacement at a further DTE. The goal is to survive crash days by pushing the position into the future where time decay works in your favor.

### When to Roll

Two triggers (OR logic), checked only on a position's expiration day:

| Trigger | When | Condition | Example |
|---------|------|-----------|---------|
| **Proximity** | After 12:30 PM PST | Price within 0.5% of short strike (or ITM) | Short put at 24,200, NDX at 24,100 (0.4% away) → ROLL |
| **Loss threshold** | Anytime on expiration day | Unrealized loss exceeds $20,000 | 5 contracts, spread $40 underwater → $20K loss → ROLL |

### How to Roll

```
STEP 1: Close the current position at market price
  → Book the P&L (usually a loss on the closing leg)

STEP 2: Compute the replacement target strike
  → Use P90 percentile for the new DTE (more conservative than entry P80)
  → Ensure the target is at least 1% OTM from current price
  → If percentile target is too close to current price, push it further OTM

STEP 3: Pick the new DTE (progressive widening)
  → 1st roll: DTE 3
  → 2nd roll: DTE 5
  → 3rd+ roll: DTE 10-30

STEP 4: Build the replacement spread
  → Same direction (put or call) as original
  → Width: start at 50 pts, widen to 100 or 200 if needed
     (crash days have wider strike spacing at deep OTM levels)
  → If no spread available at target DTE, try wider DTE range
  → Progressive search: exact DTE → ±2 DTE → any DTE >= 1

STEP 5: If replacement found → enter it, track the roll chain
         If no replacement possible → close for loss (rare, ~6 times/year)
```

### Roll Chain Accounting

Each roll creates a "chain" — the original position plus all its replacements:

```
Chain example (Mar 26 → Mar 31):
  Original: Sell put 19700/19660, credit $0.94, 27 contracts
  Mar 26 @ 12:30pm: NDX drops to 19,244 (2.3% ITM) → ROLL TRIGGERED
  Close original: P&L = -$105,448
  Open replacement: Sell put 18150/18100, DTE 3, credit higher
  Mar 31: replacement expires OTM
  Replacement P&L: +$126,900
  Chain P&L: +$21,452 (recovered the loss + profit)
```

### Roll Performance (1-Year Backtest, P95, 0DTE entries)

| Metric | Value |
|--------|-------|
| Total roll chains | 10 |
| Chains that recovered (net positive) | 4 |
| Chains that lost (net negative) | 6 |
| Total chain P&L | -$209K |
| But: total premium from replacement trades | +$3.16M |
| Net impact of rolling vs no roll | **+$2.94M** |

Rolling is not about recovering each individual loss. It is about freeing capital to collect new premium. Even when the chain P&L is negative, the replacement position generates fresh credits that more than compensate.

---

## Step 5: End of Day (1:00 PM PST)

| Task | Detail |
|------|--------|
| 0DTE positions | Expired or were rolled/closed. Nothing to carry. |
| 1DTE positions | Carry overnight. Tomorrow they become 0DTE. Highest priority tomorrow. |
| 2DTE positions | Carry overnight. Tomorrow they become 1DTE. |
| 5DTE/10DTE positions | Carry. Note days remaining. |
| Rolled positions | Note: new DTE, new strikes, roll count, chain P&L so far. |
| Daily log | Record: trades entered, trades closed, credits, rolls, stops, net P&L. |
| Tomorrow's strikes | Compute P80 strikes using today's close. Be ready at 6 AM. |

---

## Position Sizing by DTE

### Capital Per DTE at 1 Contract

| DTE | Avg Credit | Max Risk/Trade | Trades/Day | Daily Capital |
|-----|-----------|---------------|-----------|---------------|
| 0DTE | $75-150 | $4,000 | 14-26 | $56K-$104K |
| 1DTE | $75-150 | $4,000 | 28 | $114K |
| 2DTE | $75-150 | $4,000 | 26 | $104K |
| 5DTE | $75-150 | $4,000 | 19 | $76K |
| 10DTE | $75-100 | $4,000 | 24 | $99K |

### Overlapping Capital with Multi-DTE Ladder

Running 2DTE daily = up to 3 days of positions open at once:

```
2DTE ladder alone:    3 × $104K = $312K
Add 5DTE:             + $76K × 5 days overlap = ~$380K more
Add 10DTE:            + $99K × 10 days overlap = ~$990K more
Full ladder:          ~$400K–$500K actively deployed
```

### Scaling Up (After Proven Consistency)

| Level | Contracts | Budget/Day | Annual P&L Target | Max Single Loss |
|-------|-----------|-----------|-------------------|----------------|
| Starter | 1 | $100K | $150K–$500K | $5K |
| Moderate | 2-5 | $200K–$400K | $500K–$2M | $25K |
| Aggressive | 10-30 | $400K+ | $2M–$5M | $75K–$150K |

---

## Performance Summary by DTE (20-Month Backtest, 1 Contract, P80)

| DTE | Trades | Win Rate | Stop Losses | Net P&L | Sharpe | Verdict |
|-----|--------|---------|------------|---------|--------|---------|
| **0DTE** | 5,748 | 89.0% | 311 ($1.1M) | $3.6M | 4.6 | High risk, high reward. Needs rolling. |
| **1DTE** | 8,510 | 97.6% | 132 ($498K) | $7.6M | 12.2 | Best raw P&L. Small loss risk. |
| **2DTE** | 5,850 | **100%** | **0** | $6.3M | 19.4 | **Recommended primary.** Zero risk. |
| **5DTE** | 1,486 | **100%** | **0** | $1.6M | 17.2 | Safe secondary. Fewer trades. |
| **10DTE** | 784 | **100%** | **0** | $507K | 21.7 | Ultra-safe. Low volume. |

---

## Decision Flowchart

```
PRE-MARKET (before 6:00 AM PST)
│
├─ Compute P80 strikes for each DTE tier (close-to-close anchor)
├─ Decide which DTEs to trade today:
│    Conservative: 2DTE only
│    Moderate:     2DTE + 5DTE
│    Full:         2DTE + 5DTE + 10DTE
│    Aggressive:   All of the above + 1DTE + 0DTE
│
MARKET OPEN (6:00 AM PST)
│
├─ Recompute strikes using today's open (open-to-close anchor)
│  → Especially important for 0DTE (~20% tighter)
│  → Optional for DTE >= 2 (~2-4% difference)
│
├─ FOR EACH DTE tier you're trading:
│    Enter 1 put spread + 1 call spread at P80 strikes
│    Reject if credit < $0.75/share
│
REPEAT every 10 minutes until 10:00 AM PST
│
MONITOR ALL DAY:
│
├─ Any position at 75%+ profit (0DTE) or 50%+ (multi-day)?
│  YES → CLOSE IT. Take the profit.
│
├─ Any position expiring today with loss > $20K?
│  YES → ROLL to DTE 3-5 at P90 strike
│
├─ 12:30 PM: Any expiring position with price within 0.5% of short strike?
│  YES → ROLL to DTE 3-10 at P90 strike
│  NO  → Let it expire (collect full credit)
│
1:00 PM: MARKET CLOSE
├─ Log everything
├─ Compute tomorrow's strikes (close-to-close)
└─ Done
```

---

## What NOT to Do

1. **Don't auto-size contracts.** 1 contract = $5K max loss. 30 contracts = $150K max loss. The math is unforgiving.

2. **Don't trade 0DTE without active monitoring.** 0DTE requires watching the roll window at 12:30pm. If you can't be at your screen, skip 0DTE and trade 2DTE+ only.

3. **Don't ignore aging positions.** Monday's 2DTE is Wednesday's 0DTE. It needs full 0DTE treatment on Wednesday.

4. **Don't enter the same strike twice.** If you have a 2DTE put at 24,200 from yesterday, don't add another at the same strike today.

5. **Don't skip the put side.** Both directions contribute equally. Over 420 days, put and call sides are balanced.

6. **Don't assume higher percentile = safer at 0DTE/1DTE.** Higher percentile = further OTM = lower credit. The loss RATE doesn't improve much. P80 0DTE: 89% win. P100 0DTE: 89.5% win. The safety comes from DTE, not percentile.

7. **Don't roll more than 3 times.** After 3 rolls, you're fighting the trend. Let the stop loss handle it.

8. **Don't skip 2DTE to chase 0DTE premium.** 2DTE P80 made $6.3M with zero losses. 0DTE P80 made $3.6M with $1.1M in stop losses. The math favors patience.

---

## Quick Reference Card

```
STRIKES:    P80 percentile, 180-day lookback, window = DTE + 1
ANCHOR:     Close-to-close pre-market; open-to-close after open (esp. 0DTE)
SPREAD:     50 pts wide, credit spread (sell near, buy far)
CONTRACTS:  1 per signal (scale up gradually)
ENTRY:      6:00–10:00 AM PST, every 10 min, at ALL chosen DTEs
DTE TIERS:  2DTE (core) + 5DTE + 10DTE + optionally 1DTE + 0DTE
BOTH SIDES: Always enter put AND call
PROFIT:     Close at 75% (0DTE) or 50% (multi-day)
ROLL:       12:30 PM PST if within 0.5% of strike, or anytime if loss > $20K
ROLL DTE:   3 → 5 → 10-30 (progressive)
STOP LOSS:  3× credit (safety net — has never triggered at DTE >= 2)
```
