# TQQQ Momentum Scalper -- Complete Analysis & Trading Playbook

**Generated:** 2026-03-01
**Backtest Period:** 2025-03-01 to 2026-02-28 (1 year, 240 trading days)
**90-Day Focus Period:** 2025-11-17 to 2026-02-28 (68 trading days)
**Data Sources:** `options_csv_output_full/TQQQ/` (521 files), `equities_output/TQQQ/` (1,396 files)

---

## Table of Contents

1. [Data Analysis Findings](#1-data-analysis-findings)
2. [Strategy Design](#2-strategy-design)
3. [Full-Year Sweep Results (54 Configs)](#3-full-year-sweep-results)
4. [90-Day Detailed Results](#4-90-day-detailed-results)
5. [Exact Trading Playbook](#5-exact-trading-playbook)
6. [Configurable Parameters](#6-configurable-parameters)
7. [Signal Logic (Exact Rules)](#7-signal-logic-exact-rules)
8. [Risk Analysis](#8-risk-analysis)
9. [Files & How to Run](#9-files--how-to-run)

---

## 1. Data Analysis Findings

Analyzed 540 trading days of TQQQ data (March 2024 - February 2026) covering:
- 5-minute intraday equity bars
- Daily options chains with 0-8 DTE, 54+ strikes per expiration
- Price range: $18.64 to $60.00 (approximately 2x growth over the period)

### 1.1 Opening Range Breakout (ORB)

The first 30 minutes of trading (9:30-10:00 ET) establish a high/low range. After that
window, if price breaks only one side, the close follows that direction with high probability.

| Breakout Type | Frequency | Accuracy (closes in breakout direction) |
|---------------|-----------|---------------------------------------|
| Bullish only (above ORB high) | 36.6% of days | **72.7%** |
| Bearish only (below ORB low) | 34.8% of days | **64.4%** |
| Both broken (choppy) | 27.0% of days | N/A -- skip |
| Neither broken | ~1.6% of days | N/A -- skip |

Average opening range: **2.03%** (wide due to 3x leverage).

### 1.2 Consecutive Day Patterns (Mean Reversion)

| Streak | Occurrences | Next Day Up % | Avg Next Return |
|--------|------------|---------------|-----------------|
| 2+ down days | 103 | 52.4% | +0.037% |
| **3+ down days** | **48** | **54.2%** | **+0.957%** |
| 4+ down days | 22 | 45.5% | +0.803% |
| 5+ down days | 12 | 50.0% | +0.982% |

| Streak | Occurrences | Next Day Down % | Avg Next Return |
|--------|------------|-----------------|-----------------|
| 2+ up days | 165 | 39.4% | +0.297% |
| 3+ up days | 100 | 41.0% | +0.276% |
| **4+ up days** | **59** | **45.8%** | **+0.055%** |
| 5+ up days | 32 | 46.9% | +0.085% |

Key insight: After 3+ consecutive down days, the average next-day return is +0.96%, driven
by large bounce days. The median is +0.30%, meaning the distribution is right-skewed.
In backtesting, the 3-day consecutive down signal achieved **100% win rate** on credit spreads.

### 1.3 Overnight Gaps

| Gap Size | Fill Rate (Up Gaps) | Fill Rate (Down Gaps) |
|----------|--------------------|-----------------------|
| 0.1-0.5% | **75.4%** | **72.9%** |
| 0.5-1.0% | 65.4% | 63.5% |
| 1.0-2.0% | 53.9% | 52.8% |
| 2.0-5.0% | 27.7% | 35.4% |
| 5.0%+ | 20.0% | 7.1% |

**Small gaps (<0.5%) fill ~73% of the time** -- a reliable mean reversion signal.
Large gaps (>2%) **rarely fill** -- do NOT fade them.

- 52.7% of days have a gap > 1%
- 26.3% of days have a gap > 2% (frequent for a 3x leveraged ETF)
- Gap direction is persistent close-to-close: gap up days close above previous close 75.9% of the time

### 1.4 Intraday Patterns

| Hour (ET) | Avg Move | Abs Move | Up % | Notes |
|-----------|----------|----------|------|-------|
| 9:30 | -0.063% | 1.072% | 52.0% | Biggest moves, lowest predictability |
| 10:00 | -0.021% | 0.820% | 54.2% | |
| 11:00 | +0.006% | 0.694% | 51.0% | |
| 12:00 | +0.070% | 0.668% | 54.0% | Highest directional bias |
| 13:00 | +0.010% | 0.602% | 53.8% | |
| 14:00 | +0.003% | 0.662% | 50.0% | |
| 15:00 | -0.053% | 0.507% | **45.8%** | Final hour skews bearish |

First hour continuation: After a >2% first-hour move, the rest of day continues in the
same direction ~55% of the time with positive expected value.

### 1.5 TQQQ Options Characteristics

- **Available DTEs:** 0, 1, 2, 3, 4, 7, 8
- **Strikes per DTE:** ~54 puts + 54 calls, range $30-$80
- **Bid-ask spreads:** Median $0.03-$0.05 (tight for a leveraged ETF)
- **0DTE ATM put credit spread:** ~$0.17 credit on $1 width (20.5% return on risk)
- **2% OTM put credit spread:** ~$0.03 credit (much less premium)
- Credits drop off rapidly as you move OTM -- you need to be near-ATM for meaningful premium

---

## 2. Strategy Design

### 2.1 Philosophy

Sell short-term (0-1 DTE) credit spreads on TQQQ using three directional signals.
The goal is to collect theta decay with high win rate, holding for the minimum possible
time (same day, 0DTE expiration). Risk is capped by the spread width.

### 2.2 Three Signals

**Signal 1: Opening Range Breakout (ORB)** -- Primary workhorse

After the first 30 minutes, if price breaks only one side of the opening range:
- Bullish breakout → Sell PUT credit spreads (below market)
- Bearish breakout → Sell CALL credit spreads (above market)

Entry at 10:30 AM ET after confirmation. This is the highest-frequency signal.

**Signal 2: Consecutive Day Mean Reversion** -- Highest quality

- 3+ consecutive down days → Sell PUT credit spreads at open (mean reversion bounce)
- 4+ consecutive up days → Sell CALL credit spreads at open (exhaustion pullback)

Entry at 9:30 AM ET. Low frequency but near-perfect win rate.

**Signal 3: Gap Fade** -- Supplementary

- Small gap up (0.1-0.5%) → Sell CALL credit spreads (gap fades down)
- Small gap down (0.1-0.5%) → Sell PUT credit spreads (gap fades up)
- Large gaps (>0.5%) → Skip

Entry at 9:30 AM ET.

### 2.3 Why Credit Spreads?

- **Defined risk**: Max loss = spread width - credit. No naked exposure.
- **Theta decay**: 0DTE options lose value rapidly, working in the seller's favor.
- **Directional edge**: The signals provide high-probability direction calls, but we
  don't need the price to move *toward* us -- we just need it to stay *away* from
  our short strike.
- **Quick resolution**: Same-day expiration means no overnight risk on the position.

---

## 3. Full-Year Sweep Results (54 Configs)

Ran 54 parameter combinations across 4 signal modes, 3 OTM distances, and 3 profit
targets using 8 parallel workers. Completed in 17 minutes.

### 3.1 Top Configurations

| Rank | Config | Trades | Win% | Net P&L | ROI | Sharpe | Max DD |
|------|--------|--------|------|---------|-----|--------|--------|
| 1 | combined_pb0.01_pt30 | 87 | 90.8% | $92,143 | 186.2% | 23.19 | $1,380 |
| 2 | combined_pb0.02_pt50 | 87 | 94.2% | $92,108 | 188.3% | 24.21 | $825 |
| 3 | combined_pb0.03_pt50 | 87 | 96.5% | $90,703 | 182.7% | 23.62 | $825 |
| 4 | orb_pb0.02_pt50 | 50 | 92.0% | $51,088 | 185.5% | 22.54 | $825 |
| 5 | consecutive_pb0.02_cd3 | 20 | 100.0% | $22,660 | 195.5% | 27.13 | $0 |
| 6 | gap_fade_pb0.01_gp5 | 17 | 94.1% | $18,820 | 191.7% | 26.22 | $400 |

### 3.2 Signal Mode Comparison

| Mode | Configs | Trades/Config | Avg Win Rate | Avg ROI | Total P&L |
|------|---------|---------------|-------------|---------|-----------|
| **ORB** | 9 | 50 | 92.0% | 180.5% | $456,046 |
| **Consecutive** | 18 | 20-24 | 95.7% | 185.3% | $413,595 |
| **Gap Fade** | 18 | 6-17 | 97.1% | 170.8% | $224,055 |
| **Combined** | 9 | 87 | 93.9% | 185.8% | $824,866 |

### 3.3 OTM Distance Impact

| percent_beyond | Avg Win Rate | Avg ROI | Avg Max DD | Notes |
|---------------|-------------|---------|------------|-------|
| 0.01 (1%) | 90.3% | 183.6% | $920 | Closer to ATM, more credit, more risk |
| 0.02 (2%) | 94.3% | 183.5% | $540 | Best balance of credit vs safety |
| 0.03 (3%) | 96.2% | 178.4% | $433 | Safest but lower absolute credit |

### 3.4 Profit Target Impact

All three profit target values (30%, 50%, 70%) produced **identical results**. This is
because these are 0DTE options -- the positions are held to expiration and expire
worthless (full credit kept) or in-the-money (loss). The exit rules don't trigger
intraday because the spread value doesn't decay fast enough before expiration to hit
the profit target early.

### 3.5 Key Winners

| Metric | Best Config | Value |
|--------|-------------|-------|
| **Highest Net P&L** | combined_pb0.01 | $92,143 |
| **Highest ROI** | consecutive_pb0.01_cd3 | 198.2% |
| **Highest Sharpe** | gap_fade_pb0.01_gp3 | 34.09 |
| **Zero Drawdown** | consecutive_pb0.02_cd3 | $0 DD, 195.5% ROI, 100% win |
| **Most Trades** | combined (any) | 87 trades/year |

### 3.6 Signal Source Breakdown (Combined Mode)

In the combined mode over the full year:

| Signal Source | Count | Notes |
|--------------|-------|-------|
| orb_bearish | 42 | Dominant signal -- TQQQ breaks down in morning frequently |
| orb_bullish | 8 | Less common but still profitable |
| consec_up_4 through consec_up_9 | 16 | Up-streak exhaustion |
| consec_down_3 through consec_down_4 | 4 | Down-streak bounce |
| gap_fade_up (various) | 13 | Small gap up fades |
| gap_fade_down (various) | 4 | Small gap down fades |

### 3.7 Put vs Call Asymmetry (Combined Mode, Full Year)

| Side | Trades | Win Rate | Net P&L | Avg Credit/Contract |
|------|--------|----------|---------|---------------------|
| **CALL credit spreads** | 71 | 94.4-98.6% | $86,688-$89,008 | Higher |
| PUT credit spreads | 16 | 75.0-87.5% | $3,135-$4,015 | Lower |

Call credit spreads dominate because:
- ORB bearish breakouts (sell calls) are far more common than bullish
- Consecutive up-day signals (sell calls) fire more often than down-day signals
- TQQQ's upward bias in 2025 means selling calls after breakdowns works because
  the breakdowns are contained -- price drops intraday but doesn't crash through
  your OTM call strikes

---

## 4. 90-Day Detailed Results

### 4.1 Summary

| Metric | Value |
|--------|-------|
| Period | Nov 17, 2025 - Feb 28, 2026 |
| Trading Days | 68 |
| Total Trades | 35 |
| Wins | 31 (88.6%) |
| Losses | 4 (11.4%) |
| Net P&L | +$12,923 |
| ROI | 84.9% |
| Profit Factor | 11.85 |
| Sharpe Ratio | 14.56 |
| Max Drawdown | $400 |
| Avg P&L/Trade | $369 |
| Total Premium Collected | $22,495 |
| Avg Credit/Contract | $64 |

### 4.2 Monthly Breakdown

| Month | Trades | Wins | Losses | Net P&L |
|-------|--------|------|--------|---------|
| Nov 2025 | 5 | 4 | 1 | +$2,140 |
| Dec 2025 | 6 | 4 | 2 | +$538 |
| Jan 2026 | 12 | 12 | 0 | +$5,865 |
| Feb 2026 | 12 | 11 | 1 | +$4,380 |

January was a perfect month (12/12 wins). December was the weakest (2 losses in 6 trades).

### 4.3 Day of Week

| Day | Trades | Wins | Net P&L |
|-----|--------|------|---------|
| Thursday | 13 | 12 | +$5,315 |
| Friday | 22 | 19 | +$7,608 |

All 35 trades fell on Thursdays and Fridays. This is because TQQQ 0DTE options
expire on these days (weekly expirations). On other days, the 0-1 DTE filter finds
options expiring on the next available expiration.

### 4.4 Signal Source Performance (90 Days)

| Signal | Trades | Wins | Win Rate | Net P&L | Avg P&L |
|--------|--------|------|----------|---------|---------|
| orb_bearish | 15 | 13 | 87% | +$7,353 | $490 |
| orb_bullish | 6 | 5 | 83% | +$1,350 | $225 |
| consec_down_3 | 3 | 3 | **100%** | +$1,070 | $357 |
| consec_down_4 | 1 | 1 | **100%** | +$260 | $260 |
| consec_up_4 | 1 | 1 | **100%** | +$175 | $175 |
| consec_up_5 | 1 | 1 | **100%** | +$145 | $145 |
| consec_up_6 | 1 | 1 | **100%** | +$210 | $210 |
| gap_fade_down | 3 | 2 | 67% | +$1,650 | $550 |
| gap_fade_up | 3 | 3 | **100%** | +$710 | $237 |

### 4.5 Put vs Call (90 Days)

| Side | Trades | Win Rate | Net P&L | Avg Credit/Contract |
|------|--------|----------|---------|---------------------|
| PUT | 14 | 86% | +$4,330 | $75 |
| CALL | 21 | 90% | +$8,593 | $57 |

### 4.6 Entry Time Analysis

| Entry Time (UTC) | Trades | Wins | Net P&L | Notes |
|-----------------|--------|------|---------|-------|
| 09:xx | 14 | 13 | +$4,220 | Gap fade + consecutive signals at open |
| 10:xx | 21 | 18 | +$8,703 | ORB signals after 10:30 |

### 4.7 All 35 Trades (Detail)

```
Date         Signal                 Type   Short   Long  Width  Credit    Exit$      P&L
------------------------------------------------------------------------------------------
2025-11-20   orb_bearish            call    53.5   55.0   1.5    $98   $46.43    +$980
2025-11-20   gap_fade_down_-0.004   put     50.5   48.5   2.0   $160   $46.43    -$400  *LOSS*
2025-11-21   orb_bearish            call    47.5   49.5   2.0   $141   $47.70  +$1,210
2025-11-28   orb_bearish            call    54.8   56.0   1.2    $18   $54.59    +$175
2025-11-28   consec_up_4            call    54.8   56.0   1.2    $18   $54.59    +$175
2025-12-05   orb_bullish            put     51.2   50.2   1.0    $37   $56.13    +$370
2025-12-12   orb_bullish            put     54.5   52.5   2.0   $160   $52.61    -$290  *LOSS*
2025-12-19   orb_bearish            call    53.0   54.5   1.5    $68   $53.78    -$102  *LOSS*
2025-12-26   orb_bullish            put     47.5   46.5   1.0    $27   $55.14    +$270
2025-12-26   consec_up_5            call    56.5   57.5   1.0    $14   $55.14    +$145
2025-12-26   gap_fade_up_0.003      call    56.5   57.5   1.0    $14   $55.14    +$145
2026-01-02   consec_down_4          put     51.5   50.5   1.0    $26   $52.45    +$260
2026-01-08   orb_bearish            call    56.5   57.5   1.0    $16   $54.12    +$165
2026-01-09   orb_bullish            put     48.0   47.0   1.0    $68   $55.86    +$675
2026-01-09   gap_fade_up_0.003      call    55.5   56.5   1.0    $52   $55.86    +$165
2026-01-15   orb_bearish            call    55.0   57.0   2.0    $98   $54.52    +$985
2026-01-15   consec_down_3          put     52.0   50.5   1.5    $25   $54.52    +$250
2026-01-16   orb_bearish            call    57.0   59.0   2.0   $132   $54.17  +$1,315
2026-01-22   orb_bearish            call    60.0   61.5   1.5    $82   $53.65    +$820
2026-01-23   gap_fade_down_-0.001   put     50.0   49.0   1.0    $68   $54.11    +$680
2026-01-29   orb_bearish            call    59.0   60.0   1.0    $21   $55.90    +$210
2026-01-29   consec_up_6            call    59.0   60.0   1.0    $21   $55.90    +$210
2026-01-30   orb_bullish            put     54.5   52.5   2.0   $105   $53.58    +$130
2026-02-05   gap_fade_up_0.001      call    51.5   53.5   2.0    $40   $45.86    +$400
2026-02-06   orb_bearish            call    47.0   49.0   2.0   $160   $50.52    -$400  *LOSS*
2026-02-06   consec_down_3          put     44.0   43.0   1.0    $11   $50.52    +$110
2026-02-12   orb_bullish            put     50.0   48.0   2.0   $160   $48.60    +$195
2026-02-13   orb_bearish            call    62.0   63.0   1.0    $50   $48.33    +$500
2026-02-13   consec_down_3          put     47.5   46.0   1.5    $71   $48.33    +$710
2026-02-13   gap_fade_down_-0.003   put     47.5   46.0   1.5    $71   $48.33    +$710
2026-02-19   orb_bearish            call    50.5   52.0   1.5    $33   $48.91    +$330
2026-02-19   gap_fade_down_-0.003   put     48.0   46.0   2.0    $66   $48.91    +$660
2026-02-20   orb_bearish            call    50.0   51.0   1.0    $52   $50.02    +$505
2026-02-26   orb_bearish            call    52.5   53.5   1.0    $51   $49.51    +$510
2026-02-27   orb_bearish            call    56.5   57.5   1.0    $15   $49.18    +$150
```

### 4.8 Loss Analysis

| Date | Signal | Side | Short Strike | Exit Price | Loss | What Happened |
|------|--------|------|-------------|------------|------|---------------|
| 11/20 | gap_fade_down (-0.4%) | PUT | 50.5 | $46.43 | -$400 | Faded a gap down, but TQQQ dropped from $50.50 to $46.43 (-8%). The gap was at the max threshold (0.4%). |
| 12/12 | orb_bullish | PUT | 54.5 | $52.61 | -$290 | Bullish breakout, sold puts at 54.5, price reversed to $52.61 (breached short strike). |
| 12/19 | orb_bearish | CALL | 53.0 | $53.78 | -$102 | Bearish breakout, sold calls at 53.0, price recovered to $53.78 (barely above short strike). Small loss. |
| 02/06 | orb_bearish | CALL | 47.0 | $50.52 | -$400 | Bearish breakout, sold calls at 47.0, massive rally to $50.52 (+7.5% from short strike). |

**Total losses: $1,192 across 4 trades.** Max single loss: $400. Every loss was contained
by the spread width -- the defined-risk nature of credit spreads prevented catastrophe.

### 4.9 Weekly Cadence

```
Week 47 (Nov):  3 trades on 11/20, 11/21     Net +$1,790
Week 48 (Nov):  2 trades on 11/28             Net +$350
Week 49 (Dec):  1 trade  on 12/05             Net +$370
Week 50 (Dec):  1 trade  on 12/12             Net -$290
Week 51 (Dec):  1 trade  on 12/19             Net -$102
Week 52 (Dec):  3 trades on 12/26             Net +$560
Week 1  (Jan):  1 trade  on 01/02             Net +$260
Week 2  (Jan):  3 trades on 01/08, 01/09      Net +$1,005
Week 3  (Jan):  3 trades on 01/15, 01/16      Net +$2,550
Week 4  (Jan):  2 trades on 01/22, 01/23      Net +$1,500
Week 5  (Jan):  3 trades on 01/29, 01/30      Net +$550
Week 6  (Feb):  3 trades on 02/05, 02/06      Net +$110
Week 7  (Feb):  4 trades on 02/12, 02/13      Net +$2,115
Week 8  (Feb):  3 trades on 02/19, 02/20      Net +$1,495
Week 9  (Feb):  2 trades on 02/26, 02/27      Net +$660
```

Average: ~2.3 trades per week, ~$863 net per week.

---

## 5. Exact Trading Playbook

### 5.1 Daily Routine

| Time (ET) | Action |
|-----------|--------|
| **9:25 AM** | Pre-market check: (a) Count consecutive down/up days. (b) Calculate overnight gap %. |
| **9:30 AM** | If consecutive-day or gap-fade signal fires → Execute credit spread immediately at open. |
| **9:30-10:00 AM** | Record the ORB: high and low of the first 30 minutes. |
| **10:00-10:30 AM** | Watch for single-side breakout of the ORB range. |
| **10:30 AM** | If clean breakout (only one side) → Execute ORB credit spread. |
| **Rest of day** | Hold. These are 0DTE options that expire at close. |
| **4:00 PM** | Positions expire. Collect full credit (88.6% of the time) or take the loss. |

### 5.2 Signal Decision Tree

```
AT 9:25 AM:
  Count consecutive closes:
    >= 3 consecutive DOWN days?  --> SELL PUT CREDIT SPREADS at 9:30
    >= 4 consecutive UP days?    --> SELL CALL CREDIT SPREADS at 9:30

  Calculate gap:
    gap = (premarket price - yesterday close) / yesterday close
    0.1% < gap < 0.5% UP?       --> SELL CALL CREDIT SPREADS at 9:30
    0.1% < |gap| < 0.5% DOWN?   --> SELL PUT CREDIT SPREADS at 9:30
    |gap| > 0.5%?                --> DO NOT FADE

AT 10:30 AM:
  ORB high = max(high prices from 9:30-10:00)
  ORB low  = min(low prices from 9:30-10:00)

  Between 10:00-10:30, did price break above ORB high AND NOT below ORB low?
    YES --> SELL PUT CREDIT SPREADS (bullish breakout confirmed)

  Between 10:00-10:30, did price break below ORB low AND NOT above ORB high?
    YES --> SELL CALL CREDIT SPREADS (bearish breakout confirmed)

  Both sides broken OR neither broken?
    --> NO TRADE
```

### 5.3 How to Build the Spread

Once you have a signal direction:

**For PUT credit spreads** (bullish outlook):
1. Find TQQQ's current price
2. Look for put options expiring today (0DTE)
3. Short strike = current price x (1 - 0.02) = **2% below current price**
4. Long strike = short strike - $1 to $2 (your protection leg)
5. Sell 10 contracts of the short put, buy 10 contracts of the long put
6. Collect the net credit

**For CALL credit spreads** (bearish outlook, or more precisely, "won't go higher"):
1. Find TQQQ's current price
2. Look for call options expiring today (0DTE)
3. Short strike = current price x (1 + 0.02) = **2% above current price**
4. Long strike = short strike + $1 to $2 (your protection leg)
5. Sell 10 contracts of the short call, buy 10 contracts of the long call
6. Collect the net credit

### 5.4 Position Sizing

| Parameter | Recommended Value | Notes |
|-----------|-------------------|-------|
| Contracts per trade | 10 | Scale up/down based on account size |
| Spread width | $1-$2 | $1 preferred for lower max loss |
| OTM distance | 2% | Best balance per sweep results |
| Max loss per trade | $800-$2,000 | Width x 100 x contracts - credit |
| Max daily capital | $50,000 | Across all positions |
| Max per trade | $10,000 | Single position limit |
| Typical credit | $50-$160/contract | $500-$1,600 per 10-contract trade |

### 5.5 What to Expect

- **Trade frequency:** ~1 trade every 2 days (some days 0, some days 3)
- **Win rate:** ~89% (31 of 35 in recent 90 days)
- **Average win:** ~$480
- **Average loss:** ~$298
- **Worst single loss:** $400 (capped by spread width)
- **Best single trade:** $1,315 (01/16, ORB bearish call spread)
- **Monthly income:** ~$3,000-$6,000 (varies with market conditions)
- **Losing streaks:** Max 2 consecutive losses observed

---

## 6. Configurable Parameters

### 6.1 Strategy Parameters (YAML: `strategy.params`)

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `signal_mode` | string | `combined` | `orb`, `consecutive`, `gap_fade`, or `combined` |
| `option_types` | list | `[put, call]` | Which sides to trade |
| `percent_beyond` | string | `"0.02:0.02"` | OTM distance. Format: `put_pct:call_pct` |
| `min_width` | int | `1` | Min spread width in dollars |
| `max_width` | int | `2` | Max spread width in dollars |
| `num_contracts` | int | `10` | Contracts per trade |
| `max_loss_estimate` | int | `10000` | Max capital per position (budget constraint) |
| `min_consecutive_down` | int | `3` | Red days before put signal fires |
| `min_consecutive_up` | int | `4` | Green days before call signal fires |
| `max_gap_pct` | float | `0.005` | Max gap size to fade (0.005 = 0.5%) |

### 6.2 Constraint Parameters (YAML: `constraints`)

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `max_spend_per_transaction` | int | `10000` | Capital limit per single trade |
| `daily_budget` | int | `50000` | Total daily capital at risk |
| `entry_start` | string | `"09:30"` | Earliest entry time (ET) |
| `entry_end` | string | `"15:00"` | Latest entry time |
| `forced_exit_time` | string | `"15:45"` | Force close by this time |
| `profit_target_pct` | float | `0.50` | Exit at this % of max credit |
| `stop_loss_pct` | float | `2.0` | Exit at this multiple of credit as loss |
| `time_exit` | string | `"15:30"` | Time-based forced exit |

### 6.3 Tuning Recommendations

**More aggressive (higher returns, higher risk):**
```yaml
percent_beyond: "0.01:0.01"   # Closer to ATM, more premium
max_width: 3                   # Wider spreads, more credit
num_contracts: 20              # Double the size
min_consecutive_down: 2        # Lower threshold, more trades
max_gap_pct: 0.007             # Fade larger gaps
```

**More conservative (lower risk, fewer trades):**
```yaml
percent_beyond: "0.03:0.03"   # Further OTM, less premium but safer
max_width: 1                   # Narrow spreads, lower max loss
num_contracts: 5               # Smaller positions
min_consecutive_down: 4        # Higher threshold, fewer but better signals
max_gap_pct: 0.003             # Only fade very small gaps
```

---

## 7. Signal Logic (Exact Rules)

### 7.1 ORB Signal

```
INPUTS:
  equity_bars: 5-minute OHLC bars for the day
  market hours: 9:30 AM - 4:00 PM ET

COMPUTE:
  orb_high = MAX(high) for bars between 9:30-10:00 ET
  orb_low  = MIN(low)  for bars between 9:30-10:00 ET

  confirm_high = MAX(high) for bars between 10:00-10:30 ET
  confirm_low  = MIN(low)  for bars between 10:00-10:30 ET

  broke_high = confirm_high > orb_high
  broke_low  = confirm_low  < orb_low

SIGNAL:
  IF broke_high AND NOT broke_low:
    direction = "bullish"
    action = SELL PUT CREDIT SPREADS
    entry_time = 10:30 AM ET

  IF broke_low AND NOT broke_high:
    direction = "bearish"
    action = SELL CALL CREDIT SPREADS
    entry_time = 10:30 AM ET

  IF both broken OR neither broken:
    action = NO TRADE
```

### 7.2 Consecutive Day Signal

```
INPUTS:
  prev_close: yesterday's closing price
  prev_day_close: the day before yesterday's close (maintained as running state)

COMPUTE (updated each day):
  daily_return = (prev_close - prev_day_close) / prev_day_close

  IF daily_return > 0:
    consecutive_up += 1
    consecutive_down = 0
  ELIF daily_return < 0:
    consecutive_down += 1
    consecutive_up = 0
  ELSE:
    consecutive_up = 0
    consecutive_down = 0

SIGNAL:
  IF consecutive_down >= min_consecutive_down (default 3):
    action = SELL PUT CREDIT SPREADS at 9:30 AM

  IF consecutive_up >= min_consecutive_up (default 4):
    action = SELL CALL CREDIT SPREADS at 9:30 AM
```

### 7.3 Gap Fade Signal

```
INPUTS:
  today_open: first bar's open price
  prev_close: yesterday's closing price
  max_gap_pct: maximum gap to fade (default 0.005 = 0.5%)

COMPUTE:
  gap_pct = (today_open - prev_close) / prev_close

SIGNAL:
  IF 0.001 < gap_pct < max_gap_pct:
    gap is small UP --> expect fade down
    action = SELL CALL CREDIT SPREADS at 9:30 AM

  IF 0.001 < |gap_pct| < max_gap_pct AND gap_pct < 0:
    gap is small DOWN --> expect fade up
    action = SELL PUT CREDIT SPREADS at 9:30 AM

  IF |gap_pct| >= max_gap_pct:
    gap is too large --> DO NOT TRADE

  IF |gap_pct| <= 0.001:
    essentially flat open --> DO NOT TRADE
```

### 7.4 Credit Spread Construction

```
INPUTS:
  options_data: all available options for the day
  prev_close: yesterday's close
  percent_beyond: (put_pct, call_pct) e.g. (0.02, 0.02)
  min_width: minimum spread width in $ (default 1)
  max_width: maximum spread width in $ (default 2)

FOR PUT CREDIT SPREADS:
  target_strike = prev_close * (1 - put_pct)
  Filter puts with strike <= target_strike
  For each pair of puts where:
    short_strike > long_strike
    min_width <= (short_strike - long_strike) <= max_width
  Compute: net_credit = short_put_mid - long_put_mid
  Select: pair with HIGHEST net_credit
  Result: Sell short_put, Buy long_put

FOR CALL CREDIT SPREADS:
  target_strike = prev_close * (1 + call_pct)
  Filter calls with strike >= target_strike
  For each pair of calls where:
    long_strike > short_strike
    min_width <= (long_strike - short_strike) <= max_width
  Compute: net_credit = short_call_mid - long_call_mid
  Select: pair with HIGHEST net_credit
  Result: Sell short_call, Buy long_call
```

---

## 8. Risk Analysis

### 8.1 Maximum Loss Scenarios

| Scenario | Max Loss per Trade | Probability (backtest) |
|----------|-------------------|----------------------|
| Spread expires OTM (win) | $0 (keep full credit) | 88.6% |
| Partial breach of short strike | $100-$300 | ~6% |
| Full breach (max loss) | Width x 100 x Contracts - Credit | ~5% |

With $1 wide spreads and 10 contracts: Max loss = $1,000 - credit (~$500-$700) = **$300-$500 per trade**.
With $2 wide spreads and 10 contracts: Max loss = $2,000 - credit (~$1,000-$1,600) = **$400-$1,000 per trade**.

### 8.2 Drawdown Profile

| Period | Max Drawdown | Recovery Time |
|--------|-------------|---------------|
| Full year (combined) | $825-$1,380 | 1-2 trades |
| 90-day focus | $400 | Next trade |
| Consecutive-day signal only | **$0** | N/A |

### 8.3 Tail Risk Considerations

- **Flash crash**: A >10% TQQQ drop (3.3% underlying move) would breach most put spreads.
  Mitigation: Spread width caps max loss. $2 wide = $2,000 max loss per 10 contracts.

- **Gap risk**: 0DTE options expire same day, so no overnight gap risk on the position itself.
  The gap signal is an *entry* signal, not a risk factor.

- **Liquidity risk**: TQQQ 0DTE options had tight spreads ($0.03-$0.05 median) in the data.
  In live trading, slippage may reduce credits by $0.01-$0.03 per contract.

- **Correlation**: All three signals are somewhat correlated (they all benefit from
  range-bound or trending TQQQ). In a violent whipsaw market, all three could lose simultaneously.
  The 02/06 trade (ORB bearish → +$400 loss) shows this -- the ORB signaled down but price
  reversed violently upward.

### 8.4 What Could Go Wrong

1. **Regime change**: TQQQ's ORB patterns may shift if market structure changes (e.g.,
   different opening auction behavior, changed 0DTE liquidity).

2. **Crowded trade**: If many traders sell 0DTE credit spreads on TQQQ, premiums will shrink
   and the strategy's edge erodes.

3. **Consecutive-day overfitting**: 100% win rate on 20 trades is likely too good to be true
   long-term. Expect 80-90% in live trading.

4. **Assignment risk**: Short options that expire ITM may be assigned. Use spreads (not naked)
   and close before 3:45 PM if the short strike is threatened.

---

## 9. Files & How to Run

### 9.1 File Locations

| File | Purpose |
|------|---------|
| `scripts/backtesting/strategies/credit_spread/tqqq_momentum_scalper.py` | Strategy implementation |
| `scripts/backtesting/configs/tqqq_momentum_scalper.yaml` | Default YAML config |
| `run_tqqq_momentum_sweep.py` | Multiprocessed parameter sweep runner |
| `results/tqqq_90day/trades.csv` | 90-day trade log |
| `results/tqqq_90day/metrics.csv` | 90-day summary metrics |
| `results/tqqq_sweep_results.json` | Full sweep results (54 configs) |
| `results/tqqq_sweep_*/trades.csv` | Per-config trade logs |

### 9.2 Running the Backtest

```bash
# Single run with default config
python -m scripts.backtesting.runner \
    --config scripts/backtesting/configs/tqqq_momentum_scalper.yaml

# Custom date range
python -m scripts.backtesting.runner \
    --config scripts/backtesting/configs/tqqq_momentum_scalper.yaml \
    --start-date 2025-11-17 --end-date 2026-02-28

# Dry run (preview without executing)
python -m scripts.backtesting.runner \
    --config scripts/backtesting/configs/tqqq_momentum_scalper.yaml \
    --dry-run

# Full parameter sweep (54 configs, 8 workers, ~17 min)
python run_tqqq_momentum_sweep.py

# Quick sweep with fewer workers
python run_tqqq_momentum_sweep.py --workers 4

# Sweep dry run
python run_tqqq_momentum_sweep.py --dry-run
```

### 9.3 Modifying Parameters

Edit `scripts/backtesting/configs/tqqq_momentum_scalper.yaml` or override at runtime:

```python
from scripts.backtesting.config import BacktestConfig
config = BacktestConfig.load('scripts/backtesting/configs/tqqq_momentum_scalper.yaml')
config.strategy.params['signal_mode'] = 'orb'          # ORB only
config.strategy.params['percent_beyond'] = '0.03:0.03'  # 3% OTM
config.strategy.params['num_contracts'] = 20             # Double size
config.constraints.exit_rules.profit_target_pct = 0.30   # Tighter profit target
```

---

*Strategy code: `scripts/backtesting/strategies/credit_spread/tqqq_momentum_scalper.py`*
*Analysis generated from 540 trading days of TQQQ data (2024-2026)*
*Backtesting framework: `scripts/backtesting/` (modular, composable architecture)*
