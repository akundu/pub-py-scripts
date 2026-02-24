# Quick Action Plan: Improve 0DTE & Multi-Day Predictions

**Date:** 2026-02-23
**Status:** 🚨 **URGENT - 0DTE Below Target**

---

## 🎯 **THE PROBLEM**

### **0DTE Performance: 92.4% (Target: 99%)**

**Worst Times:**
- 🔴 **9:30 AM: 88.3%** (opening gap issues)
- 🔴 **3:00 PM: 84.5%** (late-day volatility)
- 🔴 **Statistical model: 24.4%** (completely broken)

### **Multi-Day Issues:**
- 🟡 **2DTE: 93.3%** (transition zone problem)
- 🟡 **5DTE: 93.3%** (transition zone problem)

---

## 🚀 **TOP 3 QUICK WINS** (Est. 4 hours total)

### **1. Opening Gap Model** (2 hours)

**What:** Add dynamic band widening for market open (9:30-10:30 AM)

**Code:**
```python
# In band_selector.py or new opening_gap_model.py
def adjust_for_opening_gap(base_bands, current_price, prev_close, hour):
    if hour <= 10.5:  # Before 10:30 AM
        gap_pct = abs(current_price - prev_close) / prev_close

        if gap_pct > 0.01:  # >1% gap
            # Widen bands by 2x gap magnitude
            multiplier = 1 + (gap_pct * 2)
            return base_bands * multiplier

    return base_bands
```

**Expected Impact:** 88.3% → 95%+ at 9:30 AM

---

### **2. Late-Day Volatility Buffer** (1 hour)

**What:** Add special handling for 2:30-4:00 PM (closing auction period)

**Code:**
```python
# In band_selector.py
def adjust_for_late_day(base_bands, hours_to_close):
    if hours_to_close < 1.5:  # After 2:30 PM
        # Check recent 5-min volatility
        recent_vol = compute_5min_volatility()

        if recent_vol > threshold:
            # Widen for late volatility
            return base_bands * 1.5

        # Always add closing auction buffer
        if hours_to_close < 0.5:  # After 3:30 PM
            return base_bands * 1.2

    return base_bands
```

**Expected Impact:** 84.5% → 96%+ at 3:00 PM

---

### **3. Dynamic Intraday Vol Adjustment** (1 hour)

**What:** Real-time band width based on current day's volatility

**Code:**
```python
# In prediction.py or features.py
def compute_intraday_vol_factor(current_price, day_high, day_low, prev_close):
    intraday_range_pct = (day_high - day_low) / prev_close * 100

    if intraday_range_pct > 2.0:  # Volatile day
        factor = 1.0 + (intraday_range_pct - 1.5) / 10.0
        return min(1.5, factor)  # Cap at 1.5x

    return 1.0  # Normal day
```

**Expected Impact:** 92.4% → 95%+ overall

---

## 📋 **FULL IMPLEMENTATION ROADMAP**

### **Phase 1: Critical 0DTE Fixes** (4 hours)
- [ ] Opening Gap Model
- [ ] Late-Day Volatility Buffer
- [ ] Dynamic Intraday Vol Adjustment

**Expected Result:** 92.4% → ~96%

### **Phase 2: Replace Statistical Model** (8-12 hours)
- [ ] Build context-aware 0DTE ensemble
- [ ] Train LGBM model with 0DTE features
- [ ] Implement similar-days weighting
- [ ] Backtest & validate

**Expected Result:** ~96% → ~98-99%

### **Phase 3: Fix Multi-Day 2DTE & 5DTE** (4-6 hours)
- [ ] Create hybrid model (0DTE + multi-day features)
- [ ] Add regime transition buffer
- [ ] Backtest & validate

**Expected Result:** 93.3% → 99%+

### **Phase 4: Advanced Features** (6-10 hours)
- [ ] Real-time regime detection
- [ ] Volume-weighted band selection
- [ ] Enhanced similar days matching

**Expected Result:** 99% → 99.5%+

---

## 📊 **DETAILED ANALYSIS**

See full report: `COMPREHENSIVE_60DAY_ANALYSIS.md`

Key sections:
- 0DTE performance by time of day
- Multi-day performance by DTE
- Root cause analysis
- 10 specific improvement proposals
- Expected impact estimates
- Implementation priority

---

## 🧪 **TESTING BEFORE DEPLOYMENT**

After each improvement:
1. Run 60-day backtest
2. Verify hit rate ≥98%
3. Check all time slots
4. Deploy incrementally

**Command:**
```bash
python scripts/backtest_comprehensive.py --ticker NDX --test-days 60 --output-dir results/after_improvements
```

---

## 📈 **EXPECTED FINAL RESULTS**

| Metric | Before | After | Improvement |
|--------|--------|-------|-------------|
| 0DTE Overall | 92.4% | 98-99% | **+6.6pp** |
| 0DTE 9:30 AM | 88.3% | 98%+ | **+9.7pp** |
| 0DTE 3:00 PM | 84.5% | 98%+ | **+13.5pp** |
| 2DTE | 93.3% | 99%+ | **+5.7pp** |
| 5DTE | 93.3% | 99%+ | **+5.7pp** |

**Overall:** 92-100% → **98-99%+ consistently**

---

## 🎯 **START HERE**

**Most Critical (Do First):**
1. ✅ **Opening Gap Model** (9:30 AM fix)
2. ✅ **Late-Day Buffer** (3:00 PM fix)
3. ✅ **Intraday Vol Adjustment** (overall improvement)

**Time Required:** 4 hours
**Expected Impact:** +3-4% hit rate improvement

**Ready to implement?** Start with `opening_gap_model.py` 🚀
