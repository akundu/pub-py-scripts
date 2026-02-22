# Ensemble vs Conditional - What Makes Them Different?

**Date:** February 22, 2026

---

## 🎯 Quick Answer

**Conditional** uses **feature-weighted historical distributions** (smart percentiles).
**Ensemble** uses **machine learning quantile regression** (LightGBM).

Both use the same input features, but process them completely differently.

---

## 📊 Comparison Table

| Aspect | Conditional (Feature-Weighted) | Ensemble (LightGBM) |
|--------|-------------------------------|-------------------|
| **Type** | Non-parametric, distribution-based | Parametric, ML regression |
| **Core Method** | Weighted historical percentiles | Gradient boosted quantile regression |
| **How it works** | Find similar past days → weight returns → compute percentiles | Train ML models → predict quantiles directly |
| **Data Usage** | Uses ALL historical data every time | Learns patterns during training, then applies |
| **Adaptation** | Adapts instantly to new regimes | Requires retraining for new regimes |
| **Overfitting Risk** | Low (uses raw data) | High (can memorize training patterns) |
| **Band Width** | Tighter (37-39% vs baseline) | Wider (24-58% vs baseline) |
| **Best For** | Multi-day predictions (1-20 DTE) | Research, risk management |

---

## 1️⃣ CONDITIONAL (Feature-Weighted Distribution)

### What It Does

**Think of it as "smart percentiles":**
1. Look at today's market features (VIX, volume, momentum, etc.)
2. Find similar historical days (weighted by feature similarity)
3. Take the weighted distribution of those returns
4. Compute percentiles from that distribution

### Example: Predicting 5DTE NDX

**Current Context:**
- VIX = 15
- Volume ratio = 1.1x
- 5-day momentum = +2%
- Price vs MA50 = +3%

**Step 1: Find Similar Days**
```python
# Weight historical days by similarity
for historical_day in past_250_days:
    similarity = calculate_feature_similarity(
        current_vix=15,
        historical_vix=historical_day.vix,
        current_volume=1.1,
        historical_volume=historical_day.volume,
        # ... other features
    )

    # Days with VIX=14-16, volume=1.0-1.2x get high weight
    # Days with VIX=25, volume=0.5x get low weight
    weights.append(similarity)
```

**Step 2: Weighted Distribution**
```python
# Create weighted distribution of historical 5-day returns
weighted_returns = []
for day, weight in zip(historical_days, weights):
    weighted_returns.append(day.return_5d * weight)

# Compute percentiles from weighted distribution
P95 = weighted_percentile(weighted_returns, 95)
P99 = weighted_percentile(weighted_returns, 99)
```

**Result:**
- P95: ±1.5% ($300 range on $20k NDX)
- P99: ±2.2% ($440 range)

### Why It's Tighter

**Conditional uses context-aware filtering:**
- When VIX is low (12-15), it only looks at low-VIX historical days
- When volume is high, it weights high-volume days more
- This creates tighter bands because it's not mixing different regimes

**Baseline (simple percentile) mixes everything:**
- Uses ALL historical days equally
- Includes high-VIX crashes mixed with low-VIX calm periods
- This creates wider bands to cover all scenarios

### Strengths
✅ Adapts instantly to regime changes (no retraining needed)
✅ Uses ALL historical data (larger effective sample size)
✅ Tighter bands = better capital efficiency
✅ Interpretable (you can see why it made a prediction)
✅ Low overfitting risk (based on actual distributions)

### Weaknesses
⚠️ Requires good feature engineering
⚠️ Can fail if current regime is unprecedented (no similar historical days)
⚠️ Slightly lower hit rate than Ensemble (97-99% vs 100%)

---

## 2️⃣ ENSEMBLE (LightGBM Quantile Regression)

### What It Does

**Think of it as "machine learning magic box":**
1. Train separate LightGBM models for each quantile (P95, P97, P98, P99)
2. Each model learns: "Given these features, what's the Xth percentile of returns?"
3. At prediction time: Feed in current features → get quantile predictions directly

### Example: Predicting 5DTE NDX

**Training (One-time):**
```python
# Train P99 model
train_data = []
for day in historical_training_period:
    features = [day.vix, day.volume, day.momentum, ...]
    target = day.actual_5d_return  # Actual outcome
    train_data.append((features, target))

# Train quantile regression model
model_p99 = LGBMRegressor(objective='quantile', alpha=0.99)
model_p99.fit(features, targets)

# Model learns: "When VIX=15, volume=1.1, predict P99 = 2.8%"
```

**Prediction (At runtime):**
```python
# Just run the model
current_features = [15, 1.1, 0.02, 0.03, ...]  # VIX, volume, momentum...
p99_prediction = model_p99.predict(current_features)

# Result: P99 = 2.8% (wider than Conditional's 2.2%)
```

### Why It's Wider

**Ensemble adds safety margins during training:**
1. **Overfitting Protection:** Adds conservative buffer to avoid underpredicting
2. **Loss Function:** Quantile loss is asymmetric - punishes under-prediction heavily
3. **Noise in Training:** Learns to be conservative when uncertain
4. **Regularization:** L1/L2 penalties push toward wider predictions

**Example:**
```
True P99 in training data: 2.2%
LightGBM learned: "Be safe, predict 3.0%"
    ↓
30% wider band than necessary
```

### Strengths
✅ Perfect 100% hit rates (very conservative)
✅ Learns complex non-linear patterns
✅ Handles feature interactions automatically
✅ Good for risk management (better safe than sorry)

### Weaknesses
❌ Bands 24-58% too wide for trading (over-conservative)
❌ Requires retraining when market regime changes
❌ Overfitting risk (memorizes training patterns)
❌ Black box (hard to explain predictions)
❌ Credit collection potential too low

---

## 3️⃣ INPUT FEATURES (Same for Both)

Both methods use identical market features:

### Tier 1: Core Features
```python
- vix                  # Current VIX level
- vix_percentile       # VIX rank vs 1-year history
- volume_ratio         # Today's volume / 20-day average
- momentum_5d          # 5-day price momentum
- price_vs_ma50        # Distance from 50-day moving average
```

### Tier 2: Context Features
```python
- iv_rank              # Implied volatility rank
- days_to_earnings     # Days until next earnings event
- intraday_range       # (High - Low) / Close
- gap_size             # Open vs previous close
```

### Tier 3: Technical Features
```python
- ma_trend             # Slope of moving averages
- volume_trend         # 5-day volume trend
- vix1d                # 1-day implied move
```

**Key Point:** Both models see the SAME inputs. The difference is HOW they process them.

---

## 4️⃣ COMPUTATION DIFFERENCES

### Conditional Computation Flow

```python
def predict_conditional(current_features, historical_data):
    """
    Conditional: Weight historical days by feature similarity
    """
    weights = []

    # Step 1: Calculate similarity scores
    for historical_day in historical_data:
        similarity = 0

        # VIX similarity (scaled)
        vix_diff = abs(current_features.vix - historical_day.vix)
        vix_weight = exp(-vix_diff / 5.0)  # Decay with distance

        # Volume similarity
        vol_diff = abs(current_features.volume - historical_day.volume)
        vol_weight = exp(-vol_diff / 0.5)

        # Combined similarity
        similarity = vix_weight * 0.4 + vol_weight * 0.3 + ...
        weights.append(similarity)

    # Step 2: Normalize weights
    weights = weights / sum(weights)

    # Step 3: Weighted percentiles
    returns = [day.future_return for day in historical_data]
    p99 = weighted_percentile(returns, weights, 99)

    return p99

# Time complexity: O(n) - scans all historical data
# Retraining: Never (uses raw data)
```

### Ensemble Computation Flow

```python
def predict_ensemble(current_features, trained_model):
    """
    Ensemble: Direct quantile prediction via LightGBM
    """
    # Model was already trained (offline)
    # Just run inference

    feature_vector = [
        current_features.vix,
        current_features.volume,
        current_features.momentum,
        # ... all features
    ]

    # LightGBM tree traversal
    p99 = trained_model.predict(feature_vector)

    return p99

# Time complexity: O(log n) - tree traversal
# Retraining: Monthly or when RMSE degrades
```

### Key Difference

**Conditional:** Processes all historical data every time (adaptive)
**Ensemble:** Uses pre-trained model (fixed until retrained)

---

## 5️⃣ WHY CONDITIONAL IS TIGHTER

### Conditional Adapts to Current Regime

**Example: Low VIX Period (VIX = 12)**

**Conditional:**
```python
# Only uses similar low-VIX days
similar_days = historical_data[vix between 10-14]
# 50 days out of 250 match
# These 50 days had narrow return distributions
# Result: P99 = ±1.8%
```

**Ensemble:**
```python
# Trained on ALL 250 days (VIX 10-30 range)
# Learned conservative pattern to cover all regimes
# Result: P99 = ±3.2% (78% wider!)
```

### Why Ensemble Over-Predicts

**Training Bias:**
```
Training data includes:
- 80% normal days (VIX 12-18): ±1.5-2.5% moves
- 15% volatile days (VIX 20-25): ±3.0-4.5% moves
- 5% crash days (VIX 30+): ±6.0-10.0% moves

LightGBM loss function (quantile loss):
- Heavily penalizes under-prediction
- Lightly penalizes over-prediction

Result: Model learns to be conservative
- Predicts ±3.5% even when current VIX = 12
- This covers the 5% crash scenarios
- But it's too wide for normal 95% of the time
```

---

## 6️⃣ REAL-WORLD EXAMPLE

### Scenario: Predict 5DTE NDX on Feb 22, 2026

**Current Market:**
- NDX = $20,000
- VIX = 14 (low, calm market)
- Volume = 1.1x average
- Momentum 5d = +1.5%

### Conditional Prediction

**Step 1: Find Similar Days**
```
Out of 250 historical days:
- 55 days had VIX 12-16 (similar)
- 45 days had volume 0.9-1.3x (similar)
- 30 days matched both criteria

These 30 days had 5-day returns:
[-1.2%, -0.8%, -0.5%, +0.3%, +0.8%, +1.1%, +1.5%, +1.8%, +2.0%, +2.3%, ...]
```

**Step 2: Weight and Compute**
```
Weighted P99 = 2.2%
Result: $20,000 ± $440 = $19,560 - $20,440
```

**Strike Selection:**
- Sell 19,600/20,400 iron condor
- Collect $150 credit
- 98% probability of profit

---

### Ensemble Prediction

**Model trained on all 250 days:**
```python
model_p99.predict([vix=14, volume=1.1, momentum=1.5])
# Returns: 3.0%

Result: $20,000 ± $600 = $19,400 - $20,600
```

**Strike Selection:**
- Sell 19,500/20,500 iron condor
- Collect $60 credit (strikes too far)
- 100% probability of profit (but credit too low to be worth it)

---

### Side-by-Side Comparison

| Method | P99 Range | Strikes | Credit | Win % | Expected Value |
|--------|-----------|---------|--------|-------|----------------|
| **Conditional** | **±$440** | **19,600/20,400** | **$150** | **98%** | **$147** |
| Ensemble | ±$600 | 19,500/20,500 | $60 | 100% | $60 |

**Conditional generates 2.5x more income with nearly identical safety.**

---

## 7️⃣ WHEN TO USE EACH METHOD

### Use Conditional When:
✅ Trading for income (credit spreads, iron condors)
✅ Capital efficiency matters
✅ Multi-day predictions (1-20 DTE)
✅ Market regime is well-represented in historical data
✅ You need interpretable predictions

### Use Ensemble When:
✅ Risk management (worst-case scenarios)
✅ Portfolio hedging (need conservative estimates)
✅ High-stakes decisions where safety > profit
✅ You have 100% accuracy requirement
✅ Research and backtesting (to stress-test strategies)

### Never Use Ensemble For:
❌ Income trading (bands too wide)
❌ 0DTE predictions (overfits badly)
❌ When you need tight strikes
❌ When commission costs matter

---

## 8️⃣ TRAINING DATA IMPLICATIONS

### How Training Window Affects Each Method

#### Conditional (Feature-Weighted)
```
3 months (60 days):  ❌ Too small, not enough similar days
6 months (125 days): ⚠️  Borderline, works in stable markets
1 year (250 days):   ✅ Optimal, good regime coverage
2 years (500 days):  ✅ Better, more robust to regime changes
```

**Why more data helps Conditional:**
- Larger pool of similar days to match against
- Better coverage of different VIX regimes
- More robust percentile estimates
- BUT: Very old data (2+ years) may be less relevant

---

#### Ensemble (LightGBM)
```
3 months (60 days):  ❌ Underfitting, not enough patterns
6 months (125 days): ⚠️  Borderline, prone to overfitting
1 year (250 days):   ✅ Good balance
2 years (500 days):  ⚠️  May include stale patterns, requires retraining more often
```

**Why Ensemble is sensitive to window size:**
- Too small: Overfits to recent noise
- Too large: Learns outdated patterns
- Optimal: 250-500 days (1-2 years)
- Requires periodic retraining (monthly)

---

### Optimal Training Window by Prediction Horizon

| Prediction Horizon | Conditional Optimal | Ensemble Optimal | Reasoning |
|-------------------|--------------------|-----------------|-----------|
| 0DTE (same day) | 90-180 days | N/A (don't use) | Recent data most relevant |
| 1-3 DTE | 180-250 days | 250 days | Balance recency and sample size |
| 5-10 DTE | 250-365 days | 250-365 days | Need full market cycle |
| 15-20 DTE | 365-500 days | 365-500 days | Longer horizon = more history |

---

## 9️⃣ SUMMARY

### Key Takeaways

1. **Same Inputs, Different Processing**
   - Both use identical market features (VIX, volume, momentum, etc.)
   - Conditional: Weights historical distributions
   - Ensemble: Learns patterns via ML

2. **Conditional is Tighter**
   - 37-39% narrower bands than baseline
   - Adapts to current regime instantly
   - Better for trading income

3. **Ensemble is Conservative**
   - 24-58% wider bands than baseline
   - Perfect 100% hit rate
   - Better for risk management

4. **Training Window Matters**
   - Too small: Overfitting / insufficient data
   - Too large: Stale patterns
   - Optimal: 250-500 days (1-2 years)
   - Conditional benefits more from larger windows

5. **Use Cases**
   - **Trading:** Conditional (tighter = more credit)
   - **Risk Management:** Ensemble (conservative = safer)

---

**Bottom Line:** Conditional is better for trading because it adapts to current market conditions and produces tighter bands, while Ensemble is too conservative (over-predicts risk) because it's trained to cover worst-case scenarios.

---

**Last Updated:** February 22, 2026
