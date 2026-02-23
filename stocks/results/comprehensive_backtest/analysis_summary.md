# Comprehensive Backtest Analysis: NDX Prediction System

**Test period:** 66 trading days (2025-11-14 to 2026-02-20)
**Ticker:** NDX | **Training:** 250-day lookback

---

## Key Insight: Width Is Everything

For selling option spreads, a high hit rate with a wide band is worthless — you need **tight bands that still hit**. The right metric is **efficiency = hit rate / width**. This completely reshuffles the rankings.

---

## PART 1: 0DTE — Where and When to Trade

### Model Selection: Percentile Wins, Statistical Is Dead Weight

| Model | P98 Hit | P97 Width | Efficiency |
|-------|---------|-----------|------------|
| **Percentile** | 91.7% | 1.81% | 50.7 |
| **Combined** | 91.8% | 1.81% | 50.7 |
| Statistical | 27.6% | 0.24% | 115.0 (but unusable — too tight, too many misses) |

The combined model differs from percentile in only **5 out of 852 slots** (0.59%). The statistical model is effectively a non-contributor. **Use percentile alone** — it's simpler and equivalent.

### The Money Table: Time Slot Ranking by Width-Adjusted Efficiency

Ranked by what matters — **accuracy per unit of width consumed** (P98 hit rate / P98 avg width):

| Rank | Slot | P98 Hit% | P98 Width | Efficiency | P97 Hit% | P97 Width | Verdict |
|------|------|----------|-----------|------------|----------|-----------|---------|
| **1** | **15:30** | **98.4%** | **1.05%** | **93.5** | 96.9% | 0.97% | **BEST SLOT** |
| **2** | **14:30** | **95.3%** | **1.27%** | **74.9** | 95.3% | 1.18% | **Excellent** |
| **3** | **15:00** | **81.2%** | **0.78%** | **104.3** | 79.7% | 0.73% | Tightest band; see caveat |
| 4 | 14:00 | 90.9% | 1.41% | 64.5 | 90.9% | 1.30% | Good |
| 5 | 13:30 | 93.9% | 1.68% | 55.9 | 90.9% | 1.60% | Good |
| 6 | 13:00 | 93.9% | 1.76% | 53.3 | 92.4% | 1.65% | Good |
| 7 | 12:30 | 90.9% | 1.76% | 51.5 | 90.9% | 1.66% | Decent |
| 8 | 11:30 | 92.4% | 2.25% | 41.1 | 92.4% | 2.06% | Wide |
| 9 | 12:00 | 90.9% | 2.25% | 40.4 | 89.4% | 2.10% | Wide |
| 10 | 11:00 | 90.9% | 2.28% | 39.9 | 90.9% | 2.10% | Wide |
| 11 | 10:30 | 92.4% | 2.51% | 36.8 | 92.4% | 2.33% | Too wide |
| 12 | 10:00 | 92.4% | 3.00% | 30.8 | 90.9% | 2.80% | Too wide |
| **13** | **9:30** | **89.4%** | **3.13%** | **28.6** | 87.9% | 2.97% | **WORST SLOT** |

### Actionable 0DTE Strategy

**Tier 1 — High conviction (enter trades here):**

- **15:30 (3:30 PM)** — The clear winner. 98.4% P98 hit rate at only 1.05% width. You're selling ~210 points of NDX spread with near-certainty. Only 1 miss in 64 observations.
- **14:30 (2:30 PM)** — 95.3% P98 hit rate at 1.27% width (~254 pts). Only 3 misses in 64 observations. Excellent risk/reward.

**Tier 2 — Good edge:**

- **13:00-14:00** — ~91-94% P98 hit rate at 1.4-1.8% width. Solid but wider. Good if you need more time for the trade to develop.

**Tier 3 — Avoid for width-sensitive strategies:**

- **9:30-10:30** — Bands are 2.3-3.1% wide. You're giving up 460-620 NDX points to get ~90% accuracy. Poor risk/reward for credit spreads.

**The 15:00 Paradox:**

The 15:00 slot has the **tightest band** (0.78%) but only 81% hit rate. Analysis of misses shows they're **balanced directionally** (5 upper, 7 lower) and mostly marginal (<0.15% miss distance). This is a band that's optimally tight — it's not broken, it's just at the edge. Usable for aggressive traders who accept the 19% miss rate in exchange for extremely tight positioning.

### Miss Clustering Warning

Misses are **not random** — they cluster on specific days:

| Date | VIX | Misses | Pattern |
|------|-----|--------|---------|
| 2025-12-10 | 13.78 | 12 of 13 slots | Strong uptrend, all upper-band violations |
| 2026-01-29 | 10.35 | 11 of 13 slots | Strong uptrend on low-VIX day |
| 2025-11-20 | 13.75 | 9 of 13 slots | Strong downtrend, all lower-band violations |

**Pattern: Low VIX + strong directional move = miss day.** When VIX is suppressed but the market makes a large directional move, the model underestimates the range. Consider widening bands or reducing position size on days where VIX < 12 and the first-hour move exceeds 0.5%.

### 1-Month vs 3-Month Stability

Results are consistent. 1-month P97 hit rate (combined): **92.7%** vs 3-month: **90.8%**. The recent month is slightly better, suggesting no model degradation.

---

## PART 2: Multi-Day — DTE Selection

### The Width Problem Scales With DTE

| DTE | Best Model | P97 Hit% | P97 Width | Efficiency |
|-----|-----------|----------|-----------|------------|
| **1** | **Conditional** | **95.5%** | **4.32%** | **22.1** |
| 2 | Conditional | 93.9% | 5.62% | 16.7 |
| 5 | Conditional | 93.9% | 8.97% | 10.5 |
| 10 | Conditional | 100% | 12.59% | 7.9 |

For comparison, the "safe" models (baseline, ensemble, ensemble_combined) achieve 100% P97 hit rates across all DTEs — but at enormous width (6-18%).

### Model Comparison (3-month, P97 level)

| Model | 1DTE Hit | 1DTE Width | 5DTE Hit | 5DTE Width | 10DTE Hit | 10DTE Width |
|-------|----------|------------|----------|------------|-----------|-------------|
| **Conditional** | **95.5%** | **4.32%** | **93.9%** | **8.97%** | **100%** | **12.59%** |
| Baseline | 100% | 6.31% | 100% | 13.90% | 100% | 17.62% |
| Ensemble | 100% | 9.56% | 100% | 11.47% | 95.5% | 13.25% |
| Ens. Combined | 100% | 9.61% | 100% | 12.12% | 100% | 15.10% |

**Conditional is the clear width winner** — 30-40% narrower than baseline at every DTE, with only 4-6% fewer hits. For selling spreads, this is the optimal model.

### Actionable Multi-Day Strategy

**1 DTE — Best multi-day opportunity:**
- Conditional model: 95.5% hit rate at 4.32% width
- That's ~865 NDX points of range. Tight enough for wide iron condors
- Only 3 misses in 66 days

**2 DTE — Still viable:**
- 93.9% hit at 5.62% width (~1,125 pts)
- 4 misses in 66 days; 2 of the misses were marginal

**5+ DTE — Width becomes prohibitive:**
- 5 DTE: ~9% width (~1,800 pts). Hard to structure profitable spreads
- 10 DTE: ~12.6% width (~2,520 pts). Not practical for directional credit spreads

**If you need 100% hit rate** (zero tolerance for misses), use ensemble_combined — but accept 2-2.5x wider bands.

---

## SYNTHESIS: Optimal Investment Playbook

### Best Opportunities (ranked by efficiency)

| Priority | Strategy | When | Model | Hit% | Width | Notes |
|----------|----------|------|-------|------|-------|-------|
| **#1** | **0DTE at 15:30** | Daily, 3:30 PM | Percentile P98 | 98.4% | 1.05% | ~210 pts NDX. Best edge in system |
| **#2** | **0DTE at 14:30** | Daily, 2:30 PM | Percentile P98 | 95.3% | 1.27% | ~254 pts. Enter here if missed 15:30 |
| **#3** | **1DTE** | EOD | Conditional P97 | 95.5% | 4.32% | ~865 pts. Next-day expiry spreads |
| **#4** | **0DTE at 13:00-14:00** | Midday | Percentile P98 | 91-94% | 1.4-1.8% | ~280-360 pts. More time to manage |
| **#5** | **2DTE** | EOD | Conditional P97 | 93.9% | 5.62% | ~1,125 pts. Wider but 2 days to manage |

### What to Avoid

- **0DTE morning entries (9:30-10:30)** — 3% width for 90% accuracy is terrible efficiency
- **0DTE at 15:00** — Unless you accept 19% miss rate for the tightest possible band
- **5+ DTE with conditional model** — Width > 9% destroys spread economics
- **Ensemble/baseline models** — 100% hit rate sounds great but the width makes it unprofitable
- **Statistical model** — Remove it entirely; it adds complexity with zero benefit

### Risk Management Rules

1. **Low VIX caution (VIX < 12):** Widen bands by 20% or reduce size. Three of the worst miss-cluster days had VIX 7-14 with strong directional moves.
2. **First-hour directional filter:** If price moves > 0.5% in first hour on a low-VIX day, consider skipping 0DTE or using P99 instead of P98.
3. **No 0DTE morning entries for tight spreads.** The band narrows naturally as the day progresses — let time work for you by entering at 14:00+.
4. **For multi-day, conditional model only.** It's the only model with acceptable width. Accept the ~5% miss rate and size positions accordingly.
