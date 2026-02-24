# Opening Gap Model - Usage Guide

**Created:** 2026-02-24
**Status:** ✅ **IMPLEMENTED**

---

## 🎯 **Problem Solved**

**Before:** 0DTE predictions at 9:30 AM had only **88.3% hit rate** (target: 99%)

**Root Cause:** Opening gaps create uncertainty not captured by historical patterns
- Pre-market moves ignored
- Gap fill probability not estimated
- Opening volatility spike not modeled

**After:** Expected to improve 9:30 AM hit rate to **95%+**

---

## 📋 **How It Works**

### **1. Gap Detection**

Detects opening gap between current price and previous close:

```python
gap_pct = (current_price - prev_close) / prev_close * 100
```

**Gap Classifications:**
- **None:** <0.5% (no adjustment)
- **Small:** 0.5-1.0% (widen bands by 5-15%)
- **Medium:** 1.0-1.5% (widen bands by 15-30%)
- **Large:** 1.5-2.5% (widen bands by 30-60%)
- **Extreme:** >2.5% (widen bands by 60-100%)

### **2. Time Decay**

Gap adjustment decreases as trading day progresses:

| Time | Decay Factor | Reason |
|------|-------------|--------|
| 9:30 AM | 100% | Gap still very relevant |
| 10:00 AM | 100% | Still in opening volatility |
| 10:30 AM | 75% | Gap partially absorbed |
| 11:00 AM | 50% | Gap mostly absorbed |
| 11:30 AM | 25% | Gap nearly absorbed |
| 12:00 PM+ | 0% | Gap no longer relevant |

### **3. Band Adjustment Formula**

```python
# Calculate multiplier based on gap magnitude
if gap_abs < 0.5%:
    multiplier = 1.0  # No adjustment
elif gap_abs < 1.0%:
    multiplier = 1.0 + (gap_abs * 0.15)  # 5-15% wider
elif gap_abs < 1.5%:
    multiplier = 1.15 + (gap_abs - 1.0) * 0.30  # 15-30% wider
elif gap_abs < 2.5%:
    multiplier = 1.30 + (gap_abs - 1.5) * 0.30  # 30-60% wider
else:
    multiplier = 1.60 + min((gap_abs - 2.5) * 0.20, 0.40)  # 60-100% wider

# Apply time decay
effective_multiplier = 1.0 + (multiplier - 1.0) * time_decay_factor

# Widen bands (preserves midpoint)
new_half_width = original_half_width * effective_multiplier
```

---

## 🚀 **Usage**

### **Automatic (Default)**

The opening gap model is **automatically applied** in 0DTE predictions when:
- Time is between 9:30 AM and 11:30 AM
- Gap is ≥0.5%

No code changes needed - it's integrated into the prediction pipeline!

### **Manual Control**

To disable gap adjustment (for testing/comparison):

```python
from scripts.close_predictor.prediction import make_unified_prediction

prediction = make_unified_prediction(
    # ... other parameters ...
    use_gap_adjustment=False,  # Disable gap adjustment
)
```

### **Standalone Testing**

Test the model directly:

```bash
python scripts/close_predictor/opening_gap_model.py
```

**Example output:**
```
Current Price:   $25,000.00
Previous Close:  $24,630.00
Gap:             +1.50% (large)
Direction:       up
Recommended:     1.30x band width

Base Bands (P97): $24,500 - $25,500 (width: $1,000, 4.00%)

9:30 AM    | Decay: 100% | $24,350 - $25,650 | Width:  5.20%
10:00 AM   | Decay: 100% | $24,350 - $25,650 | Width:  5.20%
10:30 AM   | Decay:  75% | $24,387 - $25,613 | Width:  4.90%
11:00 AM   | Decay:  50% | $24,425 - $25,575 | Width:  4.60%
12:00 PM   | Decay:   0% | $24,500 - $25,500 | Width:  4.00%
```

---

## 📊 **Examples**

### **Example 1: Small Gap (0.8%)**

**Scenario:** NDX opens at $25,000, previous close $24,800

```python
from scripts.close_predictor.opening_gap_model import detect_opening_gap

gap = detect_opening_gap(25000, 24800)
# gap.gap_pct = +0.81%
# gap.gap_magnitude = 'small'
# gap.recommended_multiplier = 1.12x (12% wider bands)
```

**Impact at 9:30 AM:**
- Base P97 width: ±2.0% → ±2.24%
- Base P99 width: ±3.0% → ±3.36%

### **Example 2: Large Gap (2.0%)**

**Scenario:** NDX gaps down to $24,500, previous close $25,000

```python
gap = detect_opening_gap(24500, 25000)
# gap.gap_pct = -2.00%
# gap.gap_magnitude = 'large'
# gap.recommended_multiplier = 1.45x (45% wider bands)
```

**Impact at 9:30 AM:**
- Base P97 width: ±2.0% → ±2.90%
- Base P99 width: ±3.0% → ±4.35%

**Impact at 10:30 AM** (75% decay):
- Base P97 width: ±2.0% → ±2.68%
- Base P99 width: ±3.0% → ±4.01%

### **Example 3: Extreme Gap (3.5%)**

**Scenario:** Major news overnight, NDX gaps up to $26,000, previous close $25,100

```python
gap = detect_opening_gap(26000, 25100)
# gap.gap_pct = +3.59%
# gap.gap_magnitude = 'extreme'
# gap.recommended_multiplier = 1.82x (82% wider bands)
```

**Impact at 9:30 AM:**
- Base P97 width: ±2.0% → ±3.64%
- Base P99 width: ±3.0% → ±5.46%

---

## 🧪 **Testing Strategy**

### **Unit Tests**

Test gap detection and adjustment:

```python
from scripts.close_predictor.opening_gap_model import (
    detect_opening_gap,
    adjust_bands_for_gap,
    compute_time_decay_factor,
)

# Test gap detection
def test_gap_detection():
    # No gap
    gap = detect_opening_gap(25000, 25000)
    assert gap.gap_magnitude == 'none'
    assert gap.recommended_multiplier == 1.0

    # Small gap
    gap = detect_opening_gap(25100, 25000)
    assert gap.gap_magnitude == 'small'
    assert 1.0 < gap.recommended_multiplier <= 1.15

    # Large gap
    gap = detect_opening_gap(25500, 25000)
    assert gap.gap_magnitude == 'large'
    assert 1.30 < gap.recommended_multiplier <= 1.60

# Test time decay
def test_time_decay():
    assert compute_time_decay_factor(9.5) == 1.0   # 9:30 AM
    assert compute_time_decay_factor(10.5) == 0.75  # 10:30 AM
    assert compute_time_decay_factor(12.0) == 0.0   # 12:00 PM

# Test band adjustment
def test_band_adjustment():
    base_bands = {'P97': (24500, 25500)}
    adjusted = adjust_bands_for_gap(
        base_bands,
        current_price=25400,
        prev_close=25000,
        hour=9.5,
    )
    # Bands should be wider due to gap
    assert adjusted['P97'][1] - adjusted['P97'][0] > base_bands['P97'][1] - base_bands['P97'][0]
```

### **Integration Tests**

Test with actual prediction pipeline:

```python
# Make prediction at 9:30 AM with gap
prediction_with_gap = make_unified_prediction(
    # ... parameters ...
    current_price=25400,
    prev_close=25000,
    current_time=datetime(2026, 2, 24, 9, 30),
    use_gap_adjustment=True,
)

# Make same prediction without gap adjustment
prediction_without_gap = make_unified_prediction(
    # ... same parameters ...
    use_gap_adjustment=False,
)

# Bands should be wider with gap adjustment
assert prediction_with_gap.combined_bands['P97'].width_pct > \
       prediction_without_gap.combined_bands['P97'].width_pct
```

### **Backtesting**

Run comprehensive backtest to validate improvement:

```bash
# Backtest with gap adjustment (new)
python scripts/backtest_comprehensive.py --ticker NDX --test-days 60 \
  --output-dir results/with_gap_adjustment

# Compare with baseline (if needed)
python scripts/compare_backtest_results.py \
  --baseline results/comprehensive_60d \
  --improved results/with_gap_adjustment
```

**Expected Results:**
- 9:30 AM hit rate: 88.3% → 95%+
- 10:00 AM hit rate: 91.7% → 95%+
- Overall 0DTE hit rate: 92.4% → 94-95%

---

## 🔧 **Configuration**

### **Adjustable Parameters**

You can tune the gap model by modifying `opening_gap_model.py`:

**1. Gap Thresholds:**
```python
# In detect_opening_gap()
threshold_pct: float = 0.5  # Minimum gap to be significant (default 0.5%)
```

**2. Multiplier Formulas:**
```python
# In detect_opening_gap()
# Small gap multiplier
multiplier = 1.0 + (gap_abs * 0.15)  # Adjust 0.15 to change sensitivity

# Medium gap multiplier
multiplier = 1.15 + (gap_abs - 1.0) * 0.30  # Adjust 0.30 for scaling

# Large gap multiplier
multiplier = 1.30 + (gap_abs - 1.5) * 0.30

# Extreme gap multiplier
multiplier = 1.60 + min((gap_abs - 2.5) * 0.20, 0.40)
```

**3. Time Decay Schedule:**
```python
# In compute_time_decay_factor()
if hour <= 10.0:    return 1.0    # Full adjustment until 10:00 AM
elif hour <= 10.5:  return 0.75   # 75% until 10:30 AM
elif hour <= 11.0:  return 0.50   # 50% until 11:00 AM
elif hour <= 11.5:  return 0.25   # 25% until 11:30 AM
else:               return 0.0    # No adjustment after 11:30 AM
```

**4. Maximum Multiplier Cap:**
```python
# In detect_opening_gap()
multiplier = min(multiplier, 2.0)  # Cap at 2.0x (default)
```

---

## 📈 **Expected Impact**

### **Before Opening Gap Model**

| Time | Hit Rate (P97) | Band Width | Issue |
|------|---------------|------------|-------|
| 9:30 AM | 88.3% | 2.94% | Too narrow for gaps |
| 10:00 AM | 91.7% | 2.75% | Still too narrow |
| 10:30 AM | 93.3% | 2.29% | Improving |

### **After Opening Gap Model**

| Time | Expected Hit Rate | Expected Width | Improvement |
|------|------------------|----------------|-------------|
| 9:30 AM | **95%+** | 3.2-4.0% (adaptive) | **+6.7pp** |
| 10:00 AM | **95%+** | 3.0-3.8% (adaptive) | **+3.3pp** |
| 10:30 AM | **96%+** | 2.5-3.0% (adaptive) | **+2.7pp** |

**Key Improvements:**
- ✅ Adapts to gap size (larger gaps → wider bands)
- ✅ Decays over time (gap less relevant as day progresses)
- ✅ Preserves band midpoint (doesn't bias up/down)
- ✅ Maintains tight bands when no gap present

---

## 🚀 **Next Steps**

1. ✅ **Implemented:** Opening gap model created
2. ✅ **Integrated:** Added to prediction pipeline
3. ⏳ **Next:** Run backtest to validate improvement
4. ⏳ **Next:** Implement late-day volatility model (3:00 PM fix)
5. ⏳ **Next:** Implement dynamic intraday vol adjustment

**Command to validate:**
```bash
# Run 60-day backtest with gap adjustment
python scripts/backtest_comprehensive.py --ticker NDX --test-days 60 \
  --output-dir results/with_gap_model

# Check 9:30 AM performance specifically
grep "9:30" results/with_gap_model/0dte_summary_NDX.csv
```

**Expected validation outcome:**
- 9:30 AM P97 hit rate: 88.3% → 95%+
- Overall 0DTE P97 hit rate: 92.4% → 94-95%

---

## 📝 **Files Modified/Created**

### **New Files:**
- ✅ `scripts/close_predictor/opening_gap_model.py` - Core gap model

### **Modified Files:**
- ✅ `scripts/close_predictor/prediction.py` - Integrated gap adjustment

### **Documentation:**
- ✅ `OPENING_GAP_MODEL_USAGE.md` - This file

---

**Status:** ✅ **Ready for testing and validation**
