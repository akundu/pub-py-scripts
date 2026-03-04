# NDX Credit Spread Daily Playbook

Generated from backtesting analysis across 420 trading days (Jul 2024 – Mar 2026) and validated on the most recent 120 trading days (Sep 2025 – Mar 2026). All results use bid/ask pricing (realistic), 50-point spread width, 1 contract per signal, P80 close-to-close percentile strikes.

---

## Table of Contents

1. [Core Principle](#core-principle)
2. [Position Sizing](#position-sizing)
3. [Backtest Results Summary](#backtest-results-summary)
4. [Daily Schedule](#daily-schedule)
5. [Multi-Day Position Lifecycle](#multi-day-position-lifecycle)
6. [Rolling Procedure](#rolling-procedure)
7. [Capital Requirements](#capital-requirements)
8. [What NOT To Do](#what-not-to-do)
9. [Full Data Tables](#full-data-tables)

---

## Core Principle

**DTE >= 2 is the hard safety boundary.** Over 420 trading days, every percentile from P80 to P100 has a 100% win rate at DTE=2+. Zero stop losses, zero drawdown. 0DTE and 1DTE have stop losses at every percentile — no amount of conservatism makes them fully safe over long periods.

**Use 1 contract per signal.** Do NOT auto-size to a minimum credit target. The $5K-minimum auto-sizing turned $75 credits into 50-67 contract positions with $200-300K stop losses. At 1 contract, a stop loss costs $3-5K (survivable).

---

## Position Sizing

| Approach | Contracts | Stop Loss Size | Trades/Day (2DTE) | Notes |
|----------|-----------|---------------|-------------------|-------|
| **1 contract (recommended)** | 1 | $3-5K | ~26 | Safe, many trades, manageable losses |
| 2-5 contracts | 2-5 | $8-25K | ~26 | Linear scaling, moderate risk |
| $5K min credit (dangerous) | 50-67 | $200-300K | ~10 | Auto-sizes contracts, catastrophic on stop loss |

**Why auto-sizing is dangerous**: A P95 0DTE strike far OTM yields ~$0.75/share credit. To meet $5K minimum: `ceil($5000 / $75) = 67 contracts`. Stop loss at 3x on a 50pt spread: `$50 × 67 × 100 = $335K` max loss.

---

## Backtest Results Summary

### 20-Month Results (Jul 2024 – Mar 2026, 1 contract)

#### Stop Losses Matrix
| | DTE=0 | DTE=1 | DTE=2 | DTE=5 | DTE=10 |
|---|---|---|---|---|---|
| P80 | 311 (-$1.1M) | 132 (-$498K) | **0** | **0** | **0** |
| P85 | 313 (-$1.2M) | 244 (-$962K) | **0** | **0** | **0** |
| P90 | 224 (-$787K) | 139 (-$449K) | **0** | **0** | **0** |
| P95 | 125 (-$509K) | 82 (-$325K) | **0** | **0** | **0** |
| P98 | 152 (-$550K) | 16 (-$28K) | **0** | **0** | **0** |
| P99 | 148 (-$501K) | 22 (-$87K) | **0** | **0** | **0** |
| P100 | 66 (-$228K) | 16 (-$78K) | **0** | **0** | **0** |

#### Net P&L Matrix
| | DTE=0 | DTE=1 | DTE=2 | DTE=5 | DTE=10 |
|---|---|---|---|---|---|
| P80 | $3.6M | $7.6M | $6.3M | $1.6M | $507K |
| P85 | $2.3M | $5.8M | $5.0M | $1.5M | $454K |
| P90 | $2.0M | $5.0M | $3.6M | $865K | $399K |
| P95 | $1.1M | $3.4M | $1.7M | $500K | $150K |
| P98 | $482K | $1.7M | $1.2M | $165K | $43K |
| P99 | $526K | $1.5M | $703K | $128K | $12K |
| P100 | $375K | $869K | $451K | $40K | $14K |

#### Win Rate Matrix
| | DTE=0 | DTE=1 | DTE=2 | DTE=5 | DTE=10 |
|---|---|---|---|---|---|
| P80 | 89.0% | 97.6% | **100%** | **100%** | **100%** |
| P85 | 88.5% | 96.2% | **100%** | **100%** | **100%** |
| P90 | 90.7% | 97.4% | **100%** | **100%** | **100%** |
| P95 | 90.1% | 97.6% | **100%** | **100%** | **100%** |
| P98 | 91.2% | 97.8% | **100%** | **100%** | **100%** |
| P99 | 88.0% | 97.2% | **100%** | **100%** | **100%** |
| P100 | 89.5% | 97.4% | **100%** | **100%** | **100%** |

#### Sharpe Ratio Matrix
| | DTE=0 | DTE=1 | DTE=2 | DTE=5 | DTE=10 |
|---|---|---|---|---|---|
| P80 | 4.6 | **12.2** | 19.4 | 17.2 | 21.7 |
| P85 | 3.3 | 8.7 | 17.0 | 16.1 | 17.1 |
| P90 | 3.9 | 10.9 | 16.8 | 16.0 | 21.3 |
| P95 | 2.9 | 9.2 | 17.4 | 16.9 | 32.2 |
| P98 | 2.1 | 9.0 | 14.2 | 19.7 | 11.9 |
| P99 | 2.9 | 7.9 | 12.8 | 22.5 | 20.9 |
| P100 | 2.9 | 7.1 | 16.0 | 21.4 | **40.3** |

### Last 120 Trading Days (Sep 2025 – Mar 2026, 1 contract)

#### Stop Losses Matrix (Recent)
| | DTE=0 | DTE=1 | DTE=2 | DTE=5 | DTE=10 |
|---|---|---|---|---|---|
| **P80** | **0** | **0** | **0** | **0** | **0** |
| **P85** | **0** | 19 (-$53K) | **0** | **0** | **0** |
| P90 | 44 (-$194K) | 22 (-$86K) | **0** | **0** | **0** |
| P95 | 22 (-$108K) | **0** | **0** | **0** | **0** |
| P98 | 22 (-$36K) | **0** | **0** | **0** | **0** |
| **P99** | **0** | **0** | **0** | **0** | **0** |
| **P100** | **0** | **0** | **0** | **0** | **0** |

#### Best Configs (Last 120 Days)
| DTE | Best | Trades | Win% | Net P&L | Sharpe | Stops |
|-----|------|--------|------|---------|--------|-------|
| 0DTE | P80 | 1,390 | 89.2% | $960K | 6.3 | 0 (but 102 rolls) |
| 1DTE | P80 | 2,182 | 98.5% | $2.0M | 13.7 | 0 |
| 1DTE | P95 | 1,177 | 100% | $798K | 15.7 | 0 |
| 2DTE | P80 | 1,466 | 100% | $1.5M | 18.2 | 0 |
| 5DTE | P80 | 340 | 100% | $333K | 16.6 | 0 |
| 10DTE | P80 | 168 | 100% | $148K | 23.0 | 0 |

#### Monthly Breakdown — 2DTE P80 (Last 120 Days)
| Month | Trades | Win% | Net P&L |
|-------|--------|------|---------|
| 2025-09 | 199 | 100% | $111,067 |
| 2025-10 | 308 | 100% | $392,524 |
| 2025-11 | 272 | 100% | $309,840 |
| 2025-12 | 288 | 100% | $208,064 |
| 2026-01 | 256 | 100% | $280,064 |
| 2026-02 | 143 | 100% | $196,783 |

---

## Daily Schedule

### 1. Night Before / Pre-Market (Before 5:55 AM PST)

**Review open positions from previous days:**

| Check | Action |
|-------|--------|
| 2DTE positions opened yesterday | Now 1DTE. These become your highest-priority monitor items today. They expire tomorrow. |
| 2DTE positions opened 2 days ago | Now 0DTE. Highest urgency. Apply 0DTE exit rules — tighter monitoring, roll trigger active. |
| 5DTE positions | Count days held. If this is the last day before expiry, treat like 0DTE for exit monitoring. |
| 10DTE positions | Count days held. If nearing expiry, increase monitoring frequency. |
| Rolled positions | Check roll count (max 2). Note the new DTE and strikes. |
| Any position at 90%+ profit | Pre-set to close at market open if still near target. |

**Key rule**: Every multi-day position ages by 1 DTE each night. A 2DTE opened Monday becomes 1DTE Tuesday and 0DTE Wednesday. On its final day, apply 0DTE exit rules.

### 2. Compute Strikes (5:55 AM PST / 12:55 UTC)

```
1. Pull last 180 trading days of NDX daily closes
2. For each DTE you plan to trade (2, 5, 10, optionally 1):
   - window = DTE + 1  (2DTE -> 3-day returns, 1DTE -> 2-day returns)
   - Compute close-to-close returns over that window
   - Split into up_returns (positive) and down_returns (negative)
   - call_strike = prev_close * (1 + percentile(up_returns, 80))
   - put_strike  = prev_close * (1 - percentile(abs(down_returns), 80))
3. Record today's target strikes for each DTE
```

### 3. Entry Window (6:00 AM - 10:00 AM PST / 13:00 - 17:00 UTC)

**Every 10 minutes, scan for new entries in this priority:**

| Priority | What to Enter | Details |
|----------|--------------|---------|
| **1st** | 2DTE put + call spread at P80 strikes | Sell 1 contract each, 50pt wide. Always enter BOTH put and call. |
| **2nd** | 5DTE put + call spreads at P80 | If 2DTE options don't exist for today. |
| **3rd** | 10DTE put + call spreads at P80 | If neither 2DTE nor 5DTE available. |
| **4th (optional)** | 1DTE put + call at P80 or P95 | Only if you accept ~1-2% loss rate. P95 = 100% win rate in last 120 days but fewer trades. |

**What you're selling each time:**
- Credit received: ~$0.75-$1.50 per share ($75-$150 per contract)
- Max risk per contract: ~$4,000 (spread width $50 x 100 shares - credit)
- Stop loss trigger: 3x the credit received

### 4. Intra-Day Monitoring (All Day)

**For every open position (including carries from prior days):**

| Time (PST) | Check | Action |
|------------|-------|--------|
| Continuously | **Profit target** | If position value <= 5% of original credit (95% profit captured) -> **close it** |
| 7:00 AM | **Early ITM check** on 0DTE/1DTE positions | If NDX price has already breached your short strike -> **roll immediately** to DTE 3-5 at P90 strike |
| 11:00 AM+ | **Roll trigger** on positions expiring today | Compute P95 remaining-move-to-close from historical data. If `distance_to_short_strike <= P95_remaining_move` -> **roll** |
| All day | **Stop loss** | If spread value hits 3x the credit you received -> **close for loss** ($3-5K per contract) |

**Exit priority**: Profit target -> Roll trigger -> Stop loss (first one triggered wins)

### 5. End of Day (1:00 PM PST / Market Close)

| Task | Detail |
|------|--------|
| **0DTE positions** | Should have expired or been closed. If still open, they settle at close. |
| **1DTE positions** | Carry overnight. Tomorrow they become 0DTE — monitor closely. |
| **2DTE positions** | Carry overnight. Tomorrow they become 1DTE. |
| **5/10DTE positions** | Carry. Count days to expiry. |
| **Log the day** | Record: trades entered, trades closed, credits collected, any stops/rolls, net P&L. |
| **Compute tomorrow's strikes** | If you want to be ready at 6 AM, compute P80 strikes now using today's close. |

---

## Multi-Day Position Lifecycle

### Example: A Single 2DTE Position

```
Day 1 (Entry Day):
  6:00 AM  - Enter 2DTE put + call spread at P80 strikes
  6:00-1:00 - Monitor for 95% profit target
  1:00 PM  - Position carries overnight as 1DTE

Day 2 (Now 1DTE):
  6:00 AM  - DO NOT re-enter same strikes (position already open)
  7:00 AM  - Early ITM check: if breached, roll immediately
  11:00 AM - Roll trigger check: P95 remaining move vs distance to strike
  All day  - Monitor profit target (95%) and stop loss (3x)
  1:00 PM  - Position carries overnight as 0DTE

Day 3 (Now 0DTE - Expiration):
  6:00 AM  - Highest priority monitoring
  7:00 AM  - Early ITM check -> roll if breached
  11:00 AM - Roll trigger active, check every bar
  1:00 PM  - Expires. If still OTM, collect full credit. If ITM, settle at loss.
```

### Overlapping Positions on Any Given Day

On a typical Wednesday, you might have:
- **New 2DTE entries** from today (expire Friday)
- **Yesterday's 2DTE** now at 1DTE (expire Thursday)
- **Monday's 2DTE** now at 0DTE (expire today) — highest priority
- **Last week's 5DTE** still running — check days remaining
- **Any rolled positions** from prior days — track roll count

### Position Aging Table

| Opened | Mon Status | Tue Status | Wed Status | Thu Status | Fri Status |
|--------|-----------|-----------|-----------|-----------|-----------|
| Mon 2DTE | **ENTRY** | 1DTE (monitor) | 0DTE (expires) | - | - |
| Tue 2DTE | - | **ENTRY** | 1DTE (monitor) | 0DTE (expires) | - |
| Wed 2DTE | - | - | **ENTRY** | 1DTE (monitor) | 0DTE (expires) |
| Mon 5DTE | **ENTRY** | 4DTE | 3DTE | 2DTE | 1DTE |
| Mon 10DTE | **ENTRY** | 9DTE | 8DTE | 7DTE | 6DTE |

---

## Rolling Procedure

When a roll trigger fires (position threatened on its expiration day):

```
1. Close the current position (realize the P&L)
2. Compute new P90 strikes at the roll DTE:
   - 1st roll: DTE 3
   - 2nd roll: DTE 5
   - 3rd roll (max): DTE 10
3. Open new spread at the new strike, same direction (put/call), 50pt wide
4. Track: original entry credit, cumulative P&L, roll count
5. If roll count = 2 and still threatened -> let stop loss handle it
```

**Roll trigger conditions:**
- After 11:00 AM PST: P95 remaining-move-to-close >= distance to short strike
- At 7:00 AM PST: Short strike already breached (ITM) -> roll immediately
- Only on the position's expiration day (last day of DTE)
- Max P95 move capped at 150 points (configurable)

---

## Capital Requirements

### At 1 Contract Per Trade

| Tier | Trades/Day | Credit/Trade | Risk/Trade | Daily Capital |
|------|-----------|-------------|-----------|---------------|
| 2DTE P80 | 26 | $1,022 | $3,978 | $104K |
| 1DTE P80 | 28 | $990 | $4,010 | $114K |
| 5DTE P80 | 19 | $979 | $4,021 | $76K |
| 10DTE P80 | 24 | $882 | $4,118 | $99K |

### With Overlapping Multi-Day Positions

If running 2DTE every day, you'll have up to 3 days of positions open simultaneously:
- Today's new entries (2DTE): ~$104K
- Yesterday's entries (now 1DTE): ~$104K
- Day before's entries (now 0DTE): ~$104K
- **Total for 2DTE ladder: ~$312K**

Add 5DTE and 10DTE overlaps: **~$400-500K fully deployed**

---

## What NOT To Do

1. **Don't size up contracts to chase bigger credits.** The $5K-minimum auto-sizing turned 1-contract positions into 50-67 contract bombs with $200-300K stop losses.

2. **Don't trade 0DTE below P80.** Even in the last 120 days, P90 0DTE had 44 stop losses. Over 20 months, 0DTE P80 had 311 stop losses.

3. **Don't ignore aging positions.** A 2DTE from Monday is a 0DTE on Wednesday — it needs full 0DTE attention (ITM checks, roll trigger, tighter monitoring).

4. **Don't enter the same strike twice.** If you already have a 2DTE put at 21,200 from yesterday (now 1DTE), don't add another today at the same strike.

5. **Don't skip the roll check after 11am.** The dynamic roll saved positions that would have been stop losses — 0DTE P80 had 102 rolls and 0 stop losses in the last 120 days.

6. **Don't assume higher percentile = safer at 0DTE/1DTE.** P98/P99/P100 have fewer trades but the same stop loss rate. Higher percentile = further OTM = lower credit = more contracts needed if auto-sizing = worse blowups.

7. **Don't skip the put side.** Both directions contribute equally to returns.

---

## Full Data Tables

### Cascade Recommendation (All Data Combined)

| Priority | DTE | Percentile | 20-Month Net | 120-Day Net | 20-Month Stops | 120-Day Stops |
|----------|-----|-----------|-------------|------------|---------------|--------------|
| **1st** | **2DTE** | **P80** | **$6.3M** | **$1.5M** | **0** | **0** |
| 2nd | 5DTE | P80 | $1.6M | $333K | 0 | 0 |
| 3rd | 10DTE | P80 | $507K | $148K | 0 | 0 |
| Optional | 1DTE | P80 | $7.6M | $2.0M | 132 | 0 |
| Optional | 1DTE | P95 | $3.4M | $798K | 82 | 0 |
| **Avoid** | **0DTE** | **any** | varies | varies | 66-313 | 0-44 |

### Percentile Explanation

- **P80**: Strike placed at the 80th percentile of historical close-to-close returns. 80% of historical moves stayed within this range.
- **P95**: 95th percentile. More conservative (further OTM), less credit, fewer trades.
- **P100**: Maximum historical move. Most conservative possible.
- **Window**: For DTE=N, uses (N+1)-day returns. DTE=2 uses 3-day close-to-close returns.

### Key Definitions

| Term | Meaning |
|------|---------|
| DTE | Days To Expiration of the option contract |
| 0DTE | Expires today |
| 1DTE | Expires tomorrow |
| 2DTE | Expires day after tomorrow |
| Credit spread | Sell near strike, buy far strike. Collect premium. Max loss = width - credit. |
| P80 strike | Strike price at the 80th percentile of historical moves |
| Roll | Close threatened position, reopen at further DTE with new strikes |
| Stop loss 3x | Close if spread value reaches 3x the credit received |
| Profit target 95% | Close when 95% of credit has been captured through theta decay |

### Configurable Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `percentile` | 80 | Which percentile to use for strike placement |
| `spread_width` | 50 | Width of credit spread in points |
| `profit_target` | 0.95 | Close at 95% of premium collected |
| `stop_loss_multiplier` | 3.0 | Close if loss reaches 3x premium |
| `interval_minutes` | 10 | Check for new entries every 10 min |
| `entry_start_utc` | 13:00 | 6:00 AM PST |
| `entry_end_utc` | 17:00 | 10:00 AM PST |
| `roll_check_start_utc` | 18:00 | 11:00 AM PST |
| `early_itm_check_utc` | 14:00 | 7:00 AM PST |
| `max_move_cap` | 150 | Cap P95 remaining move at 150 points |
| `roll_min_dte` | 3 | First roll targets DTE 3 |
| `roll_max_dte` | 10 | Max roll DTE |
| `max_rolls` | 2 | Maximum rolls per position chain |
| `min_credit` | 0.75 | Reject options with credit < $0.75/share |
| `min_credit_per_point` | 0 | Min credit/width ratio (0.04 = $2/50pt). 0=disabled |
| `max_contracts` | 0 | Cap on contracts per trade. 0=unlimited |
| `min_total_credit` | 0 | Min total credit per trade ($). 0=disabled |
| `use_mid` | false | false=bid/ask (realistic), true=mid-price (optimistic) |
| `lookback` | 180 | Trading days of history for percentile computation |

### Running Backtests

```bash
# Single config
python -m scripts.backtesting.runner --config scripts/backtesting/configs/percentile_entry_ndx.yaml

# Full sweep (35 configs: 7 percentiles x 5 DTEs)
python run_cascade_sweep.py

# Configurable in run_cascade_sweep.py header:
#   DTE_VALUES, PERCENTILES, PROFIT_TARGET, MIN_TOTAL_CREDIT,
#   MIN_CREDIT_PER_POINT, MAX_CONTRACTS, USE_MID, START_DATE, END_DATE
```
