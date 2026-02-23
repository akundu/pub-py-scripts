# Why Conditional Shows Up (And How It Works)

## TL;DR

**✅ Conditional DOES show up in `/predictions/` - it's working correctly!**

Here's proof from a live test:
```
✅ Conditional (Feature-Weighted) ⭐ RECOMMENDED
   Has bands: P99 $23,414.78 - $27,285.99 (width: 15.48%)
```

## How To See Conditional in `/predictions/`

### For Multi-Day Predictions (1-20 DTE)

1. **Navigate to:** `http://localhost:8000/predictions/NDX`
2. **Select days ahead:** Use the slider or buttons (1D, 2D, 5D, 10D, 20D)
3. **Look for:** "🎯 Multi-Day Ensemble Predictions" section
4. **You'll see 4 methods:**
   - Baseline (Percentile) - Reference
   - **Conditional (Feature-Weighted) ⭐ RECOMMENDED** ← This is it!
   - Ensemble (LightGBM) - Too wide
   - Ensemble Combined - Conservative blend

### For 0DTE (Same Day)

**Important:** 0DTE doesn't have "Conditional" in the same way. Instead you see:
- Percentile (Historical) - Simple distribution
- LightGBM (Statistical) - ML model
- **Combined (Blended) ⭐ RECOMMENDED** - Best for 0DTE

---

## What "On-The-Fly" Means

**Conditional is NOT a saved model file** - it's computed dynamically each time you request a prediction.

### The Process (Step-by-Step)

#### Step 1: Compute Current Market Features (37 features)

```python
# When you request a 5-day prediction at 10:00 AM on 2026-02-23:

current_context = MarketContext(
    # Volatility (11 features)
    vix=14.2,                    # Current VIX
    vix_percentile=45.0,         # VIX rank 0-100
    realized_vol_5d=12.3,        # 5-day realized vol
    realized_vol_20d=13.8,       # 20-day realized vol
    iv_rank=52.0,                # IV percentile
    vix1d=0.75,                  # 1-day implied vol

    # Volume (3 features)
    volume_ratio=1.15,           # Today/20-day avg
    volume_spike=False,          # >2x volume?
    volume_trend=0.03,           # 5-day slope

    # Position (6 features)
    position_vs_sma20=1.2,       # % from 20-day SMA
    position_vs_sma50=2.8,       # % from 50-day SMA
    position_in_20d_range=58.0,  # Percentile in range

    # Momentum (6 features)
    return_1d=0.3,               # 1-day return %
    return_5d=1.8,               # 5-day return %
    return_20d=5.1,              # 20-day return %
    trend_strength=32.0,         # 0-100

    # Calendar (4 features)
    day_of_week=0,               # Monday
    is_opex_week=False,
    month=2,                     # February

    # Derived (3 features)
    is_trending=True,            # Trend strength > 25
    is_overbought=False,
    is_oversold=False,

    # Earnings (2 features)
    days_to_earnings=45,
    earnings_within_window=False,

    # Intraday (2 features)
    gap_size=0.12,               # % gap from prev close
    intraday_range=0.85,         # (H-L)/C %
)

# Total: 37 features computed from LIVE market data
```

#### Step 2: Compute Historical Features (For Last 250 Days)

```python
historical_contexts = []

for each day in last_250_trading_days:
    # Compute the SAME 37 features, but for that historical day
    hist_ctx = MarketContext(
        vix=vix_on_that_day,
        position_vs_sma20=position_on_that_day,
        return_5d=return_on_that_day,
        # ... all 37 features for that day
    )
    historical_contexts.append(hist_ctx)

# Now we have 250 historical "fingerprints" of market conditions
```

#### Step 3: Calculate Similarity Scores

```python
similarities = []

for i, hist_ctx in enumerate(historical_contexts):
    # How similar is that historical day to TODAY?
    similarity = compute_feature_similarity(
        current=current_context,      # Today's 37 features
        historical=hist_ctx            # That day's 37 features
    )

    # Similarity score 0.0-1.0
    # 1.0 = identical market conditions
    # 0.5 = somewhat similar
    # 0.1 = very different

    similarities.append(similarity)

# Example results:
# Day 1 (2026-01-15): VIX=14, trending → similarity = 0.92 (very similar!)
# Day 50 (2025-11-20): VIX=25, choppy → similarity = 0.15 (very different)
# Day 100 (2025-09-10): VIX=13, trending → similarity = 0.88 (very similar!)
```

#### Step 4: Weight Historical Returns

```python
weighted_returns = []

for i, historical_5day_return in enumerate(all_5day_returns):
    similarity = similarities[i]

    # High similarity → include this return many times
    # Low similarity → include it less

    if similarity > 0.8:
        weight = 1.0      # Include 100%
        repeat_count = 10
    elif similarity > 0.6:
        weight = 0.7      # Include 70%
        repeat_count = 7
    elif similarity > 0.4:
        weight = 0.4      # Include 40%
        repeat_count = 4
    else:
        weight = 0.1      # Include 10%
        repeat_count = 1

    # Add this return multiple times based on similarity
    for _ in range(repeat_count):
        weighted_returns.append(historical_5day_return)

# Now we have a weighted distribution:
# - Days similar to today appear more frequently
# - Days different from today appear less frequently
```

#### Step 5: Extract Percentiles

```python
# Create distribution from weighted returns
weighted_distribution = np.array(weighted_returns)

# Extract percentiles
conditional_bands = {
    'P95': (
        percentile(weighted_distribution, 2.5),   # Lower 2.5%
        percentile(weighted_distribution, 97.5)   # Upper 97.5%
    ),
    'P99': (
        percentile(weighted_distribution, 0.5),   # Lower 0.5%
        percentile(weighted_distribution, 99.5)   # Upper 99.5%
    ),
    # ... other bands
}

# Result:
# P99: -7.74% to +9.10% move from current price
# → $23,414.78 to $27,285.99 (current: $25,012.62)
```

---

## Why This Is "On-The-Fly"

**No saved model file exists because:**
1. Current market features change every second (price, volume, etc.)
2. Similarity scores must be recomputed for current conditions
3. Weighting is dynamic based on TODAY's market state
4. Each prediction is fresh and adapted to current regime

**Contrast with Ensemble (LightGBM):**
- Ensemble has a saved file: `lgbm_5dte.pkl`
- File contains learned patterns from training
- Patterns are FIXED until next retraining
- Doesn't adapt to current regime automatically

---

## Feature Similarity Calculation (The Secret Sauce)

```python
def compute_feature_similarity(current, historical):
    """
    Compares two MarketContext objects and returns similarity 0.0-1.0

    Weights:
    - Volatility: 40% (most important - VIX, realized vol)
    - Position: 30% (where price is vs SMAs)
    - Momentum: 20% (recent returns, trend)
    - Calendar: 10% (day of week, OPEX)
    """

    # Volatility similarity (40% weight)
    vix_diff = abs(current.vix - historical.vix)
    vol_sim = max(0, 1 - vix_diff / 10.0)
    # If VIX differs by <5 points → high similarity
    # If VIX differs by >10 points → low similarity

    # Position similarity (30% weight)
    pos_diff = abs(current.position_vs_sma20 - historical.position_vs_sma20)
    pos_sim = max(0, 1 - pos_diff / 5.0)
    # If both at SMA+1% → high similarity
    # If one at SMA+3%, other at SMA-2% → low similarity

    # Momentum similarity (20% weight)
    mom_diff = abs(current.return_5d - historical.return_5d)
    mom_sim = max(0, 1 - mom_diff / 5.0)
    # If both trending up 2% → high similarity
    # If one up 3%, other down 2% → low similarity

    # Calendar similarity (10% weight)
    cal_sim = 1.0 if current.day_of_week == historical.day_of_week else 0.5
    # Monday vs Monday → higher similarity
    # Monday vs Friday → lower similarity

    # Weighted average
    total_similarity = (
        0.40 * vol_sim +
        0.30 * pos_sim +
        0.20 * mom_sim +
        0.10 * cal_sim
    )

    return total_similarity

# Example:
# Today: VIX=14, +1% from SMA20, +2% 5-day return, Monday
# Historical Day: VIX=15, +0.8% from SMA20, +1.8% return, Monday
# → Similarity = 0.92 (very similar!)
#
# Another Historical Day: VIX=25, -3% from SMA20, -2% return, Friday
# → Similarity = 0.18 (very different!)
```

---

## Why Conditional Appears in `/predictions/`

### Backend Flow

```
1. User requests: GET /predictions/api/lazy/future/NDX/5

2. db_server.py → common/predictions.py → scripts/predict_close_now.py

3. predict_close_now.py executes:
   a. Compute current_context (37 features)
   b. Compute historical_contexts (250 days × 37 features)
   c. Call predict_with_conditional_distribution()
   d. Get conditional_bands

4. Add to response:
   pred.ensemble_methods = [
       {'method': 'Baseline', 'bands': baseline_bands},
       {'method': 'Conditional', 'bands': conditional_bands, 'recommended': True},
       {'method': 'Ensemble', 'bands': ensemble_bands},
   ]

5. common/predictions.py serializes:
   {
       'ensemble_methods': [...],
       'recommended_method': 'Conditional (Feature-Weighted)'
   }

6. Returns JSON to frontend
```

### Frontend Flow

```javascript
// db_server.py JavaScript code (line 5664-5782)

async function loadPrediction() {
    const response = await fetch(`/predictions/api/lazy/future/NDX/5`);
    const data = await response.json();

    // Check if ensemble_methods exists
    if (data.ensemble_methods) {
        // Display all 4 methods
        for (const method of data.ensemble_methods) {
            // Highlight recommended method with green border + ⭐
            if (method.recommended) {
                display_with_green_border(method);
            } else {
                display_with_gray_border(method);
            }

            // Show all bands (P95, P97, P98, P99)
            display_bands_table(method.bands);
        }
    }
}
```

---

## Why You Might Not See It

### Possible Reasons

1. **Looking at 0DTE instead of multi-day**
   - 0DTE doesn't have "Conditional" (different prediction method)
   - Multi-day (1-20 DTE) shows Conditional
   - **Solution:** Click "1D", "2D", "5D", "10D", or "20D" buttons

2. **Data computation failed silently**
   - Not enough historical data (need 50+ days)
   - Missing VIX data
   - Error during similarity calculation
   - **Solution:** Check browser console for errors

3. **Caching issue**
   - Old cached response without ensemble_methods
   - **Solution:** Hard refresh (Cmd+Shift+R or Ctrl+Shift+R)

4. **Server not running db_server.py**
   - Still using old version
   - **Solution:** Restart server

---

## How To Verify It's Working

### Test 1: Command Line

```bash
cd stocks/
python scripts/predict_close_now.py NDX --days-ahead 5
```

**Expected Output:**
```
================================================================================
🏆 Conditional (Feature-Weighted)
⭐ RECOMMENDED - 37% tighter bands, 97-99% hit rate
================================================================================
P95     $ 23,935.26 - $ 26,592.63   (± 1,329 pts, ± 5.31%)
P99     $ 23,414.78 - $ 27,285.99   (± 1,936 pts, ± 7.74%)
```

### Test 2: API Endpoint

```bash
curl http://localhost:8000/predictions/api/lazy/future/NDX/5 | jq '.ensemble_methods[] | select(.method | contains("Conditional"))'
```

**Expected Output:**
```json
{
  "method": "Conditional (Feature-Weighted)",
  "description": "⭐ RECOMMENDED - Best balance of tight bands and reliability",
  "bands": {
    "P99": {
      "lo_price": 23414.78,
      "hi_price": 27285.99,
      "width_pct": 15.48
    }
  },
  "recommended": true,
  "backtest_performance": "37-39% tighter bands, 97-99% hit rate"
}
```

### Test 3: Web UI

1. Navigate to: `http://localhost:8000/predictions/NDX`
2. Click "5D" button (5 days ahead)
3. Scroll down to "🎯 Multi-Day Ensemble Predictions"
4. Look for green box with ⭐ RECOMMENDED

**Expected Display:**
```
┌─────────────────────────────────────────────────────────┐
│ Conditional (Feature-Weighted) ⭐ RECOMMENDED           │ <- Green border
│ Best balance of tight bands and reliability            │
│                                                         │
│ Band   Lower           Upper           Width           │
│ P95    $23,935.26      $26,592.63      10.62%         │
│ P99    $23,414.78      $27,285.99      15.48%         │
└─────────────────────────────────────────────────────────┘
```

---

## Summary Table

| Question | Answer |
|----------|--------|
| **Does Conditional show up?** | ✅ YES - in `/predictions/` for multi-day (1-20 DTE) |
| **Is it a saved model file?** | ❌ NO - computed on-the-fly |
| **What features does it use?** | 37 features from MarketContext (same as Ensemble) |
| **How does it work?** | Weights historical returns by feature similarity |
| **Why is it better?** | Adapts to current regime, 37% tighter bands |
| **When to use it?** | All multi-day predictions (1-20 DTE) |
| **File location?** | `scripts/close_predictor/multi_day_predictor.py` |
| **Similarity code?** | `scripts/close_predictor/multi_day_features.py` |

---

## Key Takeaway

**Conditional IS in `/predictions/` and it's working!**

It's not a saved model because:
- It ADAPTS to current market conditions in real-time
- Similarity scores change every second
- Distribution is reweighted for each prediction
- This is WHY it outperforms Ensemble (which is fixed)

Think of it as:
- **Ensemble** = Pre-recorded music (static, doesn't change)
- **Conditional** = Live improvisation (adapts to the moment)

Both use the same instruments (37 features), but Conditional plays what sounds best RIGHT NOW, while Ensemble plays what sounded good during training.
