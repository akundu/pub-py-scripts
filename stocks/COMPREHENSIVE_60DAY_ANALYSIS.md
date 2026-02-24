# Comprehensive 60-Day Backtest Analysis

**Generated:** 2026-02-23
**Test Period:** 2025-11-24 to 2026-02-20 (60 trading days)
**Training:** 250 days lookback
**Ticker:** NDX

---

## 🚨 **CRITICAL FINDINGS**

### ❌ **0DTE PREDICTIONS BELOW TARGET**

| Model | P95 Hit% | P97 Hit% | P98 Hit% | P99 Hit% | Target | Status |
|-------|----------|----------|----------|----------|--------|--------|
| **Combined** | 89.8% | **92.4%** | 93.0% | 95.2% | ≥99% | ❌ **FAIL** |
| **Percentile** | 89.8% | **92.4%** | 92.9% | 95.1% | ≥99% | ❌ **FAIL** |
| **Statistical** | 19.1% | **24.4%** | 28.9% | 38.9% | ≥99% | ❌ **TERRIBLE** |

**Verdict:** 0DTE predictions are **significantly below the 99% hit rate target**. Only achieving 92.4% at P97 and 95.2% at P99.

---

## 📊 Part 1: 0DTE Analysis (Same-Day Close)

### Overall Performance (3-Month Period)

**P97 Hit Rate by Time of Day:**

| Time | Hit Rate | Band Width | Midpt Error | Status |
|------|----------|------------|-------------|--------|
| **9:30 AM** | 88.3% | 2.94% | 0.68% | ❌ **Worst** |
| **10:00 AM** | 91.7% | 2.75% | 0.61% | ⚠️ Poor |
| 10:30 AM | 93.3% | 2.29% | 0.49% | ⚠️ Below target |
| 11:00 AM | 93.3% | 2.09% | 0.43% | ⚠️ Below target |
| 11:30 AM | 93.3% | 2.04% | 0.37% | ⚠️ Below target |
| 12:00 PM | 93.3% | 2.06% | 0.35% | ⚠️ Below target |
| 12:30 PM | 91.7% | 1.63% | 0.29% | ⚠️ Poor |
| 1:00 PM | 93.3% | 1.60% | 0.29% | ⚠️ Below target |
| 1:30 PM | 93.3% | 1.56% | 0.27% | ⚠️ Below target |
| 2:00 PM | 91.7% | 1.29% | 0.24% | ⚠️ Poor |
| **2:30 PM** | 96.6% | 1.15% | 0.18% | ✅ **Best** |
| **3:00 PM** | 84.5% | 0.72% | 0.17% | ❌ **Very Poor** |
| 3:30 PM | 96.6% | 0.94% | 0.13% | ⚠️ Just below |

### Key Observations

1. **Early Morning Problem (9:30-10:00 AM)**
   - Hit rate: 88-92% (way below target)
   - Band width: 2.75-2.94% (wide)
   - Issue: **High uncertainty early in day**
   - Missing factor: **Opening volatility, gap dynamics**

2. **Mid-Afternoon Drop (3:00 PM)**
   - Hit rate: **84.5%** (WORST time)
   - Band width: 0.72% (too narrow!)
   - Issue: **Bands are TOO TIGHT at 3 PM**
   - Missing factor: **Late-day volatility spikes**

3. **Best Performance (2:30 PM & 3:30 PM)**
   - Hit rate: 96.6% (close to target)
   - Why? Market has settled, patterns clearer
   - Still not at 99% target

### Patterns by Time of Day

**Band Width vs Hit Rate:**
- Wide bands (2.5-3%) → Low hit rates (88-93%)
- Narrow bands (0.7-1.5%) → Variable hit rates (85-97%)
- **Issue:** Band width NOT directly correlated with hit rate
- **Conclusion:** Current band selection logic is suboptimal

---

## 📊 Part 2: Multi-Day Analysis

### Performance by DTE (P97 Bands, 3-Month)

| DTE | Conditional Hit% | Baseline Hit% | Ensemble Hit% | Width (Cond) | Status |
|-----|-----------------|---------------|---------------|--------------|--------|
| **1** | **100.0%** | 98.3% | 100.0% | 4.32% | ✅ **Perfect** |
| **2** | **93.3%** | 100.0% | 100.0% | 5.58% | ⚠️ Below target |
| **5** | **93.3%** | 100.0% | 100.0% | 8.83% | ⚠️ Below target |
| **10** | **100.0%** | 100.0% | 100.0% | 12.49% | ✅ **Perfect** |

### Key Observations

1. **1DTE & 10DTE: Excellent**
   - 100% hit rate (perfect!)
   - Time-aware features working well

2. **2DTE & 5DTE: Below Target**
   - Only 93.3% hit rate (need 99%)
   - Issue: These DTEs are in "transition zone"
   - Too far for 0DTE logic, too close for long-term logic

3. **Conditional vs Baseline/Ensemble**
   - Conditional has **tighter bands** (good for capital efficiency)
   - But **lower hit rates** on 2DTE & 5DTE (bad for coverage)
   - Baseline/Ensemble have 100% hit but **wider bands** (conservative)

---

## 🔍 Root Cause Analysis

### Why Is 0DTE Performance Poor?

#### 1. **Statistical Model is Broken** (24.4% hit rate)
- Uses only historical percentiles without context
- Ignores current market conditions
- **Needs complete overhaul**

#### 2. **Band Selection is Static**
- Current system: Select band based on time + VIX
- Missing: Real-time volatility, market regime, recent moves
- **Needs dynamic adjustment**

#### 3. **Early Morning Uncertainty**
- Opening gap not properly modeled
- First hour volatility not captured
- Pre-market signals ignored

#### 4. **Late Afternoon Volatility**
- 3:00 PM has lowest hit rate (84.5%)
- Bands too narrow at this critical time
- Late-day moves (closing auctions) not modeled

#### 5. **Intraday Regime Changes**
- Market can shift regimes during day (calm → volatile)
- Current predictions don't adapt mid-day
- Need real-time regime detection

---

## 💡 Improvement Suggestions

### **For 0DTE (Priority: HIGH)**

#### 1. **Replace Statistical Model Entirely**

**Current:** Simple historical percentiles (24.4% hit rate)

**Proposed:** Context-aware ensemble model
```python
# Use similar approach to multi-day conditional predictor
0dte_prediction = combine([
    percentile_baseline,  # Historical distribution
    lgbm_context_model,   # ML model with current context
    similar_days_model,   # Matching historical patterns
])

# Weight based on time of day
if hours_to_close > 5:
    # Early: rely more on similar days
    weights = [0.2, 0.3, 0.5]
elif hours_to_close < 1:
    # Late: rely more on percentile (clearer picture)
    weights = [0.5, 0.3, 0.2]
else:
    # Mid-day: balanced
    weights = [0.33, 0.33, 0.34]
```

**Expected improvement:** 24.4% → 95%+ hit rate

---

#### 2. **Dynamic Band Width Adjustment**

**Current:** Static band width based on time slot

**Proposed:** Real-time adjustment based on:
```python
# Factors to consider:
1. Current intraday volatility (high - low) / open
2. Recent 5-minute price swings
3. Volume profile (high volume = more conviction)
4. VIX/VIX1D ratio (stress indicator)
5. Time decay factor (less time = less uncertainty)

# Adjustment formula:
base_width = percentile_bands[time_slot]
intraday_vol_factor = compute_intraday_vol()
regime_factor = detect_regime_change()
volume_factor = analyze_volume_profile()

adjusted_width = base_width * (
    1.0 +
    0.3 * (intraday_vol_factor - 1.0) +  # ±30% for volatility
    0.2 * (regime_factor - 1.0) +        # ±20% for regime shift
    -0.1 * volume_factor                  # -10% if high conviction
)
```

**Expected improvement:** 92.4% → 97%+ hit rate

---

#### 3. **Opening Gap Model (9:30-10:30 AM)**

**Problem:** 88-93% hit rate in first hour (worst performance)

**Root cause:** Opening gaps create uncertainty
- Pre-market moves not modeled
- Gap fill probability not estimated
- Opening volatility spike not captured

**Proposed solution:**
```python
# Add opening-specific features:
1. Gap size (current - prev_close) / prev_close
2. Gap direction (up or down)
3. Pre-market volume (if available)
4. Overnight futures moves
5. Historical gap fill probability

# Opening gap adjustment:
if hour == 9.5:  # 9:30 AM
    gap_pct = abs(current_price - prev_close) / prev_close
    if gap_pct > 0.01:  # >1% gap
        # Widen bands by gap magnitude
        band_width *= (1 + gap_pct * 2)  # 2% gap → +4% width
```

**Expected improvement:** 88.3% → 95%+ hit rate at 9:30 AM

---

#### 4. **Late-Day Volatility Model (2:30-4:00 PM)**

**Problem:** 3:00 PM has 84.5% hit rate (worst time of day)

**Root cause:** Bands too narrow (0.72%), missing late-day moves
- Closing auction volatility
- End-of-day portfolio rebalancing
- News-driven late moves

**Proposed solution:**
```python
# Late-day specific logic:
if hours_to_close < 1.5:  # After 2:30 PM
    # Check for unusual late-day activity:
    recent_5min_range = (high_5min - low_5min) / current_price

    if recent_5min_range > 0.003:  # >0.3% in 5 minutes
        # Widen bands for late volatility
        band_width *= 1.5

    # Add closing auction buffer:
    if hours_to_close < 0.5:  # After 3:30 PM
        # Last 30 min: add extra buffer for auction
        band_width *= 1.2
```

**Expected improvement:** 84.5% → 96%+ hit rate at 3:00 PM

---

#### 5. **Real-Time Regime Detection**

**Problem:** Predictions don't adapt when market regime changes mid-day

**Example scenario:**
- 10 AM: Calm market, narrow bands
- 12 PM: News breaks, volatility spikes
- Current prediction: Still using 10 AM narrow bands (MISS!)

**Proposed solution:**
```python
# Track regime changes in real-time:
class IntraDay RegimeDetector:
    def __init__(self):
        self.baseline_vol = None  # Set at market open
        self.current_vol = None

    def detect_regime_change(self, current_price, history_5min):
        # Compute rolling 5-min volatility
        self.current_vol = std(history_5min) / mean(history_5min)

        if self.current_vol > self.baseline_vol * 1.5:
            # Volatility spiked by 50%+
            return "ELEVATED"
        elif self.current_vol > self.baseline_vol * 2.0:
            # Volatility doubled
            return "HIGH"
        else:
            return "NORMAL"

    def adjust_bands(self, base_bands, regime):
        if regime == "ELEVATED":
            return base_bands * 1.3  # +30% width
        elif regime == "HIGH":
            return base_bands * 1.6  # +60% width
        else:
            return base_bands
```

**Expected improvement:** Prevent regime-shift misses (+2-3% hit rate)

---

#### 6. **Volume-Weighted Band Selection**

**Insight:** High volume = high conviction = tighter range

**Current:** Ignores volume entirely

**Proposed:**
```python
# Volume profile analysis:
current_volume = sum(volume_last_30min)
avg_volume_same_time = historical_avg_volume(time_of_day)

volume_ratio = current_volume / avg_volume_same_time

if volume_ratio > 1.5:  # 50% above average
    # High volume = high conviction = tighter bands
    band_width *= 0.9  # -10% width
elif volume_ratio < 0.7:  # 30% below average
    # Low volume = low conviction = wider bands
    band_width *= 1.1  # +10% width
```

**Expected improvement:** +1-2% hit rate, better capital efficiency

---

#### 7. **Similar Days Enhancement**

**Current:** Matches based on gap, intraday move, and some features

**Missing:**
- Sector performance correlation
- Market breadth (advance/decline)
- VIX term structure
- Options flow (if available)

**Proposed:**
```python
# Enhanced similarity scoring:
def enhanced_similarity(current_day, historical_day):
    base_score = current_similarity_score(current_day, historical_day)

    # Add market breadth matching:
    if available(advance_decline_data):
        breadth_similarity = abs(
            current_day.adv_dec_ratio - historical_day.adv_dec_ratio
        )
        base_score *= (1 - breadth_similarity * 0.1)

    # Add VIX term structure:
    if available(vix_term_structure):
        vix_slope_similarity = abs(
            current_day.vix_slope - historical_day.vix_slope
        )
        base_score *= (1 - vix_slope_similarity * 0.15)

    return base_score
```

**Expected improvement:** +2-3% hit rate through better pattern matching

---

### **For Multi-Day (Priority: MEDIUM)**

#### 1. **Fix 2DTE & 5DTE Performance**

**Problem:** 2DTE and 5DTE have 93.3% hit rate (below 99% target)

**Root cause:** These are "transition DTEs"
- Too far for 0DTE intraday logic
- Too close for long-term multi-day logic
- Fall through the cracks

**Proposed solution:**
```python
# Hybrid approach for short-term DTEs (1-7 days):
if dte <= 7:
    # Combine 0DTE intraday features with multi-day features
    features = {
        # Multi-day features:
        'position_vs_sma20': ...,
        'return_5d': ...,
        'vol_regime': ...,

        # PLUS 0DTE intraday features:
        'current_intraday_move': ...,
        'intraday_volatility': ...,
        'hours_to_close_today': ...,
        'gap_pct': ...,
    }

    # Use hybrid model that understands both timeframes
    prediction = hybrid_predictor.predict(features)
else:
    # Use standard multi-day approach
    prediction = multiday_predictor.predict(features)
```

**Expected improvement:** 93.3% → 99%+ for 2DTE & 5DTE

---

#### 2. **Regime-Aware Volatility Scaling**

**Current:** Time-aware features added (good!)
**Missing:** Regime transitions

**Problem:** If market regime changes between prediction time and target date, bands are miscalibrated

**Example:**
- Monday 10 AM: Predict Friday close (5DTE)
- Market is calm → narrow bands
- Wednesday: Major news → regime shifts to volatile
- Friday close: Misses due to narrow bands from Monday

**Proposed solution:**
```python
# Add forward-looking regime uncertainty:
if dte >= 3:
    # For multi-day predictions, add regime transition buffer
    regime_transition_prob = estimate_regime_transition_probability(dte)

    # Widen bands proportionally to transition probability
    band_width *= (1 + regime_transition_prob * 0.2)

    # Example: 5DTE has 30% chance of regime change
    # → widen bands by 6% (30% * 0.2)
```

**Expected improvement:** +1-2% hit rate on multi-day predictions

---

## 📊 Summary of Proposed Improvements

### **0DTE Improvements (Expected Impact)**

| Improvement | Current Hit% | Expected Hit% | Difficulty | Priority |
|-------------|-------------|---------------|------------|----------|
| Replace Statistical Model | 24.4% | 95%+ | Medium | **HIGH** |
| Dynamic Band Width | 92.4% | 97%+ | Medium | **HIGH** |
| Opening Gap Model | 88.3% (9:30) | 95%+ | Low | **HIGH** |
| Late-Day Volatility Model | 84.5% (3:00) | 96%+ | Low | **HIGH** |
| Real-Time Regime Detection | 92.4% | 94-95% | High | Medium |
| Volume-Weighted Bands | 92.4% | 93-94% | Low | Medium |
| Enhanced Similar Days | 92.4% | 94-95% | Medium | Medium |

**Combined Expected Improvement:** 92.4% → **98-99% hit rate**

### **Multi-Day Improvements (Expected Impact)**

| Improvement | Current Hit% | Expected Hit% | Difficulty | Priority |
|-------------|-------------|---------------|------------|----------|
| Fix 2DTE & 5DTE (Hybrid Model) | 93.3% | 99%+ | Medium | **HIGH** |
| Regime Transition Buffer | 93-100% | 99-100% | Low | Medium |

**Combined Expected Improvement:** 93.3% → **99%+ hit rate** for all DTEs

---

## 🎯 Recommended Implementation Priority

### **Phase 1: Critical 0DTE Fixes** (Est. 6-8 hours)

1. ✅ **Opening Gap Model** (2 hours)
   - Add gap detection and dynamic widening
   - Test at 9:30 AM predictions

2. ✅ **Late-Day Volatility Buffer** (1 hour)
   - Add 3:00-4:00 PM special handling
   - Test at 3:00 PM predictions

3. ✅ **Dynamic Band Width Adjustment** (3 hours)
   - Implement real-time intraday vol factor
   - Test across all time slots

**Expected Result:** 0DTE hit rate improves from 92.4% to ~96%

### **Phase 2: Statistical Model Replacement** (Est. 8-12 hours)

4. ✅ **Build Context-Aware 0DTE Ensemble** (8 hours)
   - Train LGBM model with 0DTE features
   - Implement similar-days weighting
   - Create ensemble combiner

5. ✅ **Backtest & Validate** (2 hours)
   - Run 60-day backtest
   - Verify 98%+ hit rate

**Expected Result:** 0DTE hit rate improves from ~96% to ~98-99%

### **Phase 3: Multi-Day Fixes** (Est. 4-6 hours)

6. ✅ **Hybrid Model for 2DTE & 5DTE** (4 hours)
   - Combine 0DTE + multi-day features
   - Train separate models for short DTEs

7. ✅ **Regime Transition Buffer** (1 hour)
   - Add forward-looking regime uncertainty
   - Widen bands for longer DTEs

**Expected Result:** Multi-day hit rate improves from 93.3% to 99%+ for all DTEs

### **Phase 4: Advanced Features** (Est. 6-10 hours)

8. ⚠️ **Real-Time Regime Detection** (4 hours)
9. ⚠️ **Volume-Weighted Band Selection** (2 hours)
10. ⚠️ **Enhanced Similar Days Matching** (3 hours)

**Expected Result:** Hit rate reaches 99-99.5% consistently

---

## 📁 Files to Create/Modify

### **New Files:**
1. `scripts/close_predictor/opening_gap_model.py` - Gap detection and adjustment
2. `scripts/close_predictor/late_day_volatility.py` - 3:00-4:00 PM handler
3. `scripts/close_predictor/regime_detector.py` - Real-time regime tracking
4. `scripts/close_predictor/0dte_ensemble.py` - Context-aware 0DTE model

### **Files to Modify:**
1. `scripts/close_predictor/prediction.py` - Integrate new models
2. `scripts/close_predictor/band_selector.py` - Dynamic band width logic
3. `scripts/close_predictor/features.py` - Add new features
4. `scripts/close_predictor/multi_day_predictor.py` - Hybrid model for 2-5 DTE

---

## 🧪 Testing Strategy

### **Unit Tests:**
- Test gap model at different gap sizes
- Test late-day buffer at different times
- Test regime detector with historical regime shifts

### **Backtests:**
- 60-day comprehensive backtest (all DTEs, all hours)
- 90-day validation (out-of-sample)
- Compare hit rates before/after each improvement

### **Live Testing:**
- Deploy improvements incrementally
- Monitor hit rates daily for first week
- Rollback if hit rate drops below 98%

---

## 📈 Expected Final Performance

### **After All Improvements:**

| DTE | Time | Current Hit% | Expected Hit% | Improvement |
|-----|------|-------------|---------------|-------------|
| 0DTE | 9:30 AM | 88.3% | 98%+ | **+9.7pp** |
| 0DTE | 3:00 PM | 84.5% | 98%+ | **+13.5pp** |
| 0DTE | Overall | 92.4% | 98-99% | **+6.6pp** |
| 1DTE | - | 100% | 100% | Maintained |
| 2DTE | - | 93.3% | 99%+ | **+5.7pp** |
| 5DTE | - | 93.3% | 99%+ | **+5.7pp** |
| 10DTE | - | 100% | 100% | Maintained |

**Overall System:** 92-100% → **98-99%+ consistently**

---

## 💰 Business Impact

### **Current State:**
- 0DTE: 92.4% hit rate → ~7.6% miss rate
- Misses mean losses or missed opportunities
- Low confidence in early morning and late afternoon

### **After Improvements:**
- 0DTE: 98-99% hit rate → ~1-2% miss rate
- **75% reduction in miss rate**
- High confidence throughout trading day
- Can safely trade earlier in day (9:30-10:00)
- Can safely hold positions into close (3:00-4:00)

### **Capital Efficiency:**
- Current: Must use wider P99 bands (95.2% hit rate)
- After: Can use tighter P97 bands (98%+ hit rate)
- **~20% reduction in capital requirements**

---

## 🚀 Next Steps

1. **Review and approve implementation plan**
2. **Prioritize improvements** (Phase 1 first, then 2, etc.)
3. **Allocate development time** (~20-30 hours total)
4. **Implement incrementally** (test after each phase)
5. **Backtest thoroughly** before deploying to live
6. **Monitor live performance** closely for first 2 weeks

**Ready to start implementation?** 🎯
