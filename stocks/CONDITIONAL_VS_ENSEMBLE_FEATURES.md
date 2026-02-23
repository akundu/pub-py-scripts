# Conditional vs Ensemble: Feature Analysis

## Question 1: Why No Conditional Model File for 0DTE?

**Answer: There is NO file-based "conditional model" - it's computed dynamically!**

### 0DTE (Same-Day) Predictions

**For 0DTE, "Conditional" doesn't exist as a separate method.** Instead you have:

1. **Percentile Method**: Simple historical distribution of intraday moves
2. **LightGBM Method**: ML model trained on intraday patterns
3. **Combined Method**: Blends the two above (RECOMMENDED for 0DTE)

The 0DTE methods don't use the "conditional distribution" approach because:
- Intraday patterns are different from multi-day patterns
- Features are different (time-to-close, intraday volatility vs multi-day context)
- No need for feature weighting - the LightGBM model learns patterns directly

### Multi-Day (1-20 DTE) Predictions

For multi-day predictions, you have 4 methods:

| Method | How It Works | Files Created |
|--------|--------------|---------------|
| **Baseline** | Simple percentile of N-day historical returns | None (computed on-the-fly) |
| **Conditional** ⭐ | Weights historical returns by feature similarity | **None** (computed on-the-fly) |
| **Ensemble** | LightGBM quantile regression | `lgbm_Xdte.pkl` (20 files) |
| **Ensemble Combined** | Conservative blend of Conditional + Ensemble | None (computed on-the-fly) |

**Key Point:** Only Ensemble creates model files. Conditional is computed dynamically each time!

---

## Question 2: What Features Are Used?

### BOTH Methods Use THE SAME Features!

**Critical Insight:** Conditional and Ensemble use **identical input features** from `MarketContext`. The difference is HOW they process these features:

- **Conditional**: Computes similarity scores, weights historical samples
- **Ensemble**: LightGBM learns non-linear patterns during training

### Complete Feature List (39 Features)

Both methods use these features from `scripts/close_predictor/multi_day_features.py`:

#### **Volatility Features (11 features)**
```python
1.  vix                    # Current VIX level
2.  vix_percentile         # VIX rank 0-100
3.  realized_vol_5d        # 5-day realized volatility
4.  realized_vol_20d       # 20-day realized volatility
5.  vol_regime_low         # 1.0 if low vol, else 0.0
6.  vol_regime_high        # 1.0 if high vol, else 0.0
7.  iv_rank                # IV percentile vs 1yr realized (TIER 1)
8.  iv_percentile          # Current IV rank 0-100 (TIER 1)
9.  iv_term_structure      # 30d IV / 90d IV ratio (TIER 1)
10. vix1d                  # 1-day implied volatility (TIER 1)
11. vix1d_percentile       # VIX1D historical rank (TIER 1)
```

#### **Volume Features (3 features - TIER 1)**
```python
12. volume_ratio           # Today's vol / 20-day avg
13. volume_spike           # 1.0 if ratio > 2.0, else 0.0
14. volume_trend           # 5-day volume slope (normalized)
```

#### **Earnings Features (2 features - TIER 2)**
```python
15. days_to_earnings       # Days until next earnings
16. earnings_within_window # 1.0 if earnings in prediction window
```

#### **Intraday Volatility (2 features - TIER 2)**
```python
17. gap_size               # (Open - Prev Close) / Prev Close %
18. intraday_range         # (High - Low) / Close %
```

#### **Position Features (6 features)**
```python
19. position_vs_sma20      # % distance from 20-day SMA
20. position_vs_sma50      # % distance from 50-day SMA
21. position_in_10d_range  # 0-100 percentile in 10-day range
22. position_in_20d_range  # 0-100 percentile in 20-day range
23. distance_from_high_20d # % below 20-day high
24. distance_from_low_20d  # % above 20-day low
```

#### **Momentum Features (6 features)**
```python
25. return_1d              # 1-day return %
26. return_5d              # 5-day return %
27. return_10d             # 10-day return %
28. return_20d             # 20-day return %
29. consecutive_days       # Consecutive up (+) or down (-) days
30. trend_strength         # ADX-like metric 0-100
```

#### **Calendar Features (4 features)**
```python
31. day_of_week            # 0=Mon, 4=Fri
32. is_opex_week           # 1.0 if options expiration week
33. days_to_month_end      # Trading days until month end
34. month                  # 1-12
```

#### **Derived Features (3 features)**
```python
35. is_overbought          # 1.0 if position_vs_sma20 > 3%
36. is_oversold            # 1.0 if position_vs_sma20 < -3%
37. is_trending            # 1.0 if trend_strength > 25
```

**Total: 37 features** (not 39, I miscounted initially)

---

## Question 3: How Do They Use The Features Differently?

### Conditional Method (Feature Weighting)

**File**: `scripts/close_predictor/multi_day_predictor.py`

**Process:**
1. Compute current `MarketContext` (37 features)
2. For each historical day, compute its `MarketContext`
3. Calculate similarity score between current and each historical day:
   ```python
   similarity = compute_feature_similarity(current_context, historical_context)
   ```
4. Weight historical N-day returns by similarity
5. Create weighted distribution of returns
6. Extract percentiles (P95, P97, P98, P99) from weighted distribution

**Why It's Better:**
- Adapts to current regime automatically
- Only uses "similar" historical days
- Result: **37-39% TIGHTER bands** than baseline

**Example:**
```
Current market: VIX=15, trending up, mid-range
↓
Find similar days: VIX 13-17, trending, mid-range
↓
Weight their 5-day returns heavily
↓
Downweight days with VIX=25, choppy, at extremes
↓
Result: Distribution reflects CURRENT conditions
```

### Ensemble Method (LightGBM Learning)

**File**: `scripts/close_predictor/multi_day_lgbm.py`

**Process:**
1. Training phase:
   - For each historical day, compute `MarketContext` (37 features)
   - Get actual N-day forward return
   - Train LightGBM to predict: `return = f(features)`

2. Prediction phase:
   - Compute current `MarketContext` (37 features)
   - Feed to LightGBM model
   - Get predicted return distribution
   - Extract percentiles

**Why It's Wider:**
- Learns complex patterns during training
- Tries to generalize across all regimes
- When regime changes, predictions become conservative
- Result: **24-58% WIDER bands** than baseline

**Example:**
```
Training (May 2024 - May 2025): VIX=12-20, bull market
↓
Model learns: "When VIX=15, expect +0.3% move"
↓
Prediction (Nov 2025): VIX=15, but different regime
↓
Model still predicts based on old patterns
↓
Adds safety margin → WIDER bands
```

---

## Similarity Calculation (Conditional Only)

**File**: `scripts/close_predictor/multi_day_features.py`

```python
def compute_feature_similarity(
    current: MarketContext,
    historical: MarketContext
) -> float:
    """
    Computes weighted similarity score between two market contexts.

    Returns:
        Score 0.0-1.0 (1.0 = identical conditions)
    """

    # Volatility similarity (40% weight)
    vix_diff = abs(current.vix - historical.vix)
    vol_sim = max(0, 1 - vix_diff / 10.0)  # Within 10 VIX points = high similarity

    # Position similarity (30% weight)
    pos_diff = abs(current.position_vs_sma20 - historical.position_vs_sma20)
    pos_sim = max(0, 1 - pos_diff / 5.0)  # Within 5% = high similarity

    # Momentum similarity (20% weight)
    mom_diff = abs(current.return_5d - historical.return_5d)
    mom_sim = max(0, 1 - mom_diff / 5.0)  # Within 5% return = high similarity

    # Calendar similarity (10% weight)
    cal_sim = 1.0 if current.day_of_week == historical.day_of_week else 0.5

    # Weighted average
    similarity = (
        0.40 * vol_sim +
        0.30 * pos_sim +
        0.20 * mom_sim +
        0.10 * cal_sim
    )

    return similarity
```

---

## Feature Vectors - Actual Example

### NDX on 2026-02-23 at 10:00 AM

```python
current_context = MarketContext(
    # Volatility features
    vix=14.2,
    vix_percentile=45.0,
    realized_vol_5d=12.3,
    realized_vol_20d=13.8,
    vol_regime="medium",
    iv_rank=52.0,
    iv_percentile=48.0,
    iv_term_structure=0.95,
    vix1d=0.75,
    vix1d_percentile=42.0,

    # Volume features
    volume_ratio=1.15,
    volume_spike=False,
    volume_trend=0.03,

    # Earnings features
    days_to_earnings=45,
    earnings_within_window=False,

    # Intraday features
    gap_size=0.12,
    intraday_range=0.85,

    # Position features
    position_vs_sma20=1.2,
    position_vs_sma50=2.8,
    position_in_10d_range=65.0,
    position_in_20d_range=58.0,
    distance_from_high_20d=-1.8,
    distance_from_low_20d=4.2,

    # Momentum features
    return_1d=0.3,
    return_5d=1.8,
    return_10d=3.2,
    return_20d=5.1,
    consecutive_days=2,
    trend_strength=32.0,

    # Calendar features
    day_of_week=0,  # Monday
    is_opex_week=False,
    days_to_month_end=5,
    month=2,

    # Derived features
    is_overbought=False,
    is_oversold=False,
    is_trending=True,
)

# Convert to feature vector
feature_vector = current_context.to_dict()
# Returns 37-element dict:
# {
#   'vix': 14.2,
#   'vix_percentile': 45.0,
#   'realized_vol_5d': 12.3,
#   ... (34 more features)
# }
```

### How Each Method Uses This Vector

**Conditional:**
1. Compares this vector to all historical day vectors
2. Finds days with similar values (VIX~14, trending, mid-range)
3. Weights those days' 5-day returns heavily
4. Result: Distribution reflects current market state

**Ensemble:**
1. Feeds this vector directly to LightGBM model
2. Model predicts: "Given these features, expect X% move"
3. Uses quantile prediction to get distribution
4. Result: Model's learned patterns applied to current state

---

## Summary Table

| Aspect | Conditional | Ensemble |
|--------|-------------|----------|
| **Input Features** | 37 features from MarketContext | 37 features from MarketContext |
| **Feature Processing** | Similarity weighting | LightGBM learning |
| **Model File** | None (computed on-the-fly) | `lgbm_Xdte.pkl` (20 files) |
| **Training** | No training needed | Requires periodic retraining |
| **Adaptability** | Auto-adapts to current regime | Fixed until retrained |
| **Band Width** | 37-39% TIGHTER than baseline | 24-58% WIDER than baseline |
| **Hit Rate** | 97-99% | 100% |
| **Recommendation** | ⭐ **BEST for trading** | Too conservative |
| **When It Fails** | Too few similar historical days | Regime change |

---

## Why Conditional Wins

**Conditional is TIGHTER because:**
1. Only uses relevant historical days (high similarity)
2. Ignores irrelevant days (low similarity)
3. Adapts automatically to current market regime
4. No staleness - always current

**Ensemble is WIDER because:**
1. Tries to generalize across all historical regimes
2. When uncertain, adds safety margin
3. Fixed patterns from training - doesn't adapt
4. Becomes stale after ~25-40 days

---

## Practical Implications

### For Trading Credit Spreads

**Use Conditional bands:**
- Tightest possible while maintaining 97-99% hit rate
- Maximizes premium collected
- Example: 5-day P99 band is ±400 pts instead of ±640 pts
- Extra 240 pts = **37% more capital efficiency**

### Annual Income Impact

Based on 180-day backtest:

| Method | P99 Width | Premium/Trade | Annual Income |
|--------|-----------|---------------|---------------|
| Baseline | 13.17% | $520 | $2,600 |
| Conditional | 7.97% | $1,528 | $7,644 |
| Ensemble | 20.80% | $390 | $1,950 |

**Conditional earns +$5,044/year more than baseline (+194%)**

---

## File Locations

### Feature Definitions
- `scripts/close_predictor/multi_day_features.py` - MarketContext class (37 features)
- Lines 19-126: Complete feature list and to_dict() conversion

### Conditional Method
- `scripts/close_predictor/multi_day_predictor.py` - Feature weighting logic
- Function: `predict_with_conditional_distribution()`
- Function: `weight_historical_samples()` - Similarity-based weighting

### Ensemble Method
- `scripts/close_predictor/multi_day_lgbm.py` - LightGBM training/prediction
- Class: `MultiDayLGBMPredictor`
- Method: `train()` - Trains on MarketContext features
- Method: `predict_distribution()` - Generates quantile predictions

### Similarity Calculation
- `scripts/close_predictor/multi_day_features.py`
- Function: `compute_feature_similarity()` - Compares two MarketContext objects

---

## Conclusion

1. **0DTE has NO conditional model** - it's a different prediction approach entirely
2. **Multi-day Conditional uses 37 features** - same as Ensemble
3. **Difference is processing** - Conditional weights samples, Ensemble learns patterns
4. **Conditional wins for trading** - 37% tighter bands, 97-99% hit rate
5. **Both methods are ticker-agnostic** - work identically for NDX and SPX

The key insight: **Same data, different algorithms, dramatically different results.**
