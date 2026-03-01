"""
Directional momentum analysis and asymmetric prediction bands.

Given current momentum state (consecutive up/down days, multi-day returns),
computes:
  - P(up) vs P(down) for each DTE
  - Mean reversion likelihood after extended streaks
  - Asymmetric bands that skew wider in the likely direction
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

from .bands import weighted_percentile
from .models import UnifiedBand, UNIFIED_BAND_NAMES


@dataclass
class MomentumState:
    """Classification of current market momentum."""
    trend_label: str          # "strong_up", "up", "neutral", "down", "strong_down"
    consecutive_days: int     # signed: positive = up streak, negative = down streak
    return_5d: float          # 5-day return as decimal (e.g. 0.02 = 2%)
    is_extended_streak: bool  # abs(consecutive_days) >= 3


@dataclass
class DirectionalProbability:
    """Probability of up vs down move over the prediction horizon."""
    p_up: float               # probability of positive return
    p_down: float             # probability of negative return
    up_count: int             # number of similar periods that went up
    down_count: int           # number of similar periods that went down
    total_samples: int        # total similar samples found
    confidence: str           # "high" (>=50), "medium" (>=30), "low" (<30)
    mean_reversion_prob: float  # probability of reversal after current streak


@dataclass
class DirectionalAnalysis:
    """Complete directional analysis result."""
    momentum_state: MomentumState
    direction_probability: DirectionalProbability
    asymmetric_bands: Dict[str, UnifiedBand]


def classify_momentum(ctx: Any) -> MomentumState:
    """Classify current momentum from MarketContext fields.

    Args:
        ctx: MarketContext with consecutive_days, return_5d, etc.

    Returns:
        MomentumState with trend_label and streak info.
    """
    consec = getattr(ctx, 'consecutive_days', 0)
    ret_5d = getattr(ctx, 'return_5d', 0.0)

    if consec >= 3 and ret_5d > 0.01:
        trend_label = "strong_up"
    elif consec >= 1 and ret_5d > 0:
        trend_label = "up"
    elif consec <= -3 and ret_5d < -0.01:
        trend_label = "strong_down"
    elif consec <= -1 and ret_5d < 0:
        trend_label = "down"
    else:
        trend_label = "neutral"

    return MomentumState(
        trend_label=trend_label,
        consecutive_days=consec,
        return_5d=ret_5d,
        is_extended_streak=abs(consec) >= 3,
    )


def compute_directional_probability(
    n_day_returns: np.ndarray,
    historical_contexts: List[Any],
    current_context: Any,
    days_ahead: int,
) -> DirectionalProbability:
    """Compute P(up) vs P(down) using similar historical periods.

    Finds historical periods with similar market context (via
    compute_feature_similarity), then counts up/down outcomes.

    Args:
        n_day_returns: Array of historical N-day returns (decimals).
        historical_contexts: List of MarketContext for each historical period.
        current_context: Current MarketContext.
        days_ahead: Prediction horizon in days (for labeling only).

    Returns:
        DirectionalProbability with up/down probabilities and confidence.
    """
    from .multi_day_features import compute_feature_similarity

    n = min(len(n_day_returns), len(historical_contexts))
    if n == 0:
        return DirectionalProbability(
            p_up=0.5, p_down=0.5, up_count=0, down_count=0,
            total_samples=0, confidence="low", mean_reversion_prob=0.5,
        )

    returns = n_day_returns[:n]
    contexts = historical_contexts[:n]

    # Compute similarities and find matching periods
    similarities = np.array([
        compute_feature_similarity(current_context, ctx) for ctx in contexts
    ])

    # Progressive threshold relaxation to get enough samples
    for threshold in [0.5, 0.4, 0.3, 0.2, 0.0]:
        mask = similarities > threshold
        if mask.sum() >= 20 or threshold == 0.0:
            break

    matched_returns = returns[mask]
    total = len(matched_returns)

    if total == 0:
        return DirectionalProbability(
            p_up=0.5, p_down=0.5, up_count=0, down_count=0,
            total_samples=0, confidence="low", mean_reversion_prob=0.5,
        )

    up_count = int((matched_returns > 0).sum())
    down_count = int((matched_returns < 0).sum())
    p_up = up_count / total
    p_down = down_count / total

    # Confidence level
    if total >= 50:
        confidence = "high"
    elif total >= 30:
        confidence = "medium"
    else:
        confidence = "low"

    # Mean reversion probability for current streak
    mean_rev = compute_mean_reversion_probability(
        getattr(current_context, 'consecutive_days', 0),
        n_day_returns,
        historical_contexts,
        days_ahead,
    )

    return DirectionalProbability(
        p_up=round(p_up, 4),
        p_down=round(p_down, 4),
        up_count=up_count,
        down_count=down_count,
        total_samples=total,
        confidence=confidence,
        mean_reversion_prob=round(mean_rev, 4),
    )


def compute_mean_reversion_probability(
    consecutive_days: int,
    n_day_returns: np.ndarray,
    historical_contexts: List[Any],
    days_ahead: int,
) -> float:
    """Compute probability of reversal after a streak of consecutive days.

    Filters history to periods with similar streak length (within 1 day)
    and same direction, then returns fraction that reversed.

    Args:
        consecutive_days: Current signed streak (positive=up, negative=down).
        n_day_returns: Array of historical N-day returns.
        historical_contexts: Corresponding MarketContext list.
        days_ahead: Prediction horizon (unused, reserved).

    Returns:
        Float probability of reversal (0.0-1.0). Returns 0.5 if insufficient data.
    """
    if consecutive_days == 0:
        return 0.5

    n = min(len(n_day_returns), len(historical_contexts))
    if n < 5:
        return 0.5

    returns = n_day_returns[:n]
    streak_sign = 1 if consecutive_days > 0 else -1
    streak_abs = abs(consecutive_days)

    # Find historical periods with similar streak
    matched_returns = []
    for i, ctx in enumerate(historical_contexts[:n]):
        hist_consec = getattr(ctx, 'consecutive_days', 0)
        hist_sign = 1 if hist_consec > 0 else (-1 if hist_consec < 0 else 0)
        hist_abs = abs(hist_consec)

        # Same direction and similar magnitude (within 1 day)
        if hist_sign == streak_sign and abs(hist_abs - streak_abs) <= 1:
            matched_returns.append(returns[i])

    if len(matched_returns) < 5:
        return 0.5

    matched = np.array(matched_returns)
    # Reversal = return in opposite direction to the streak
    if streak_sign > 0:
        reversals = (matched < 0).sum()
    else:
        reversals = (matched > 0).sum()

    return float(reversals / len(matched))


def compute_asymmetric_bands(
    n_day_returns: np.ndarray,
    current_price: float,
    direction_prob: DirectionalProbability,
    weights: Optional[np.ndarray] = None,
) -> Dict[str, UnifiedBand]:
    """Compute asymmetric prediction bands skewed by directional probability.

    Splits returns into up/down distributions and weights the tail
    percentiles by P(up) and P(down). Always at least as wide as
    symmetric bands.

    Args:
        n_day_returns: Array of historical returns (as percentages, e.g. 1.5 = 1.5%).
        current_price: Current price for band computation.
        direction_prob: DirectionalProbability with p_up, p_down.
        weights: Optional sample weights for weighted_percentile.

    Returns:
        Dict of band_name -> UnifiedBand with source="directional".
    """
    if len(n_day_returns) < 10:
        return {}

    returns = np.asarray(n_day_returns, dtype=float)

    # Split into up and down returns
    up_mask = returns > 0
    down_mask = returns < 0
    up_returns = returns[up_mask] if up_mask.any() else returns
    down_returns = returns[down_mask] if down_mask.any() else returns

    # Symmetric band definitions: band_name -> (lo_pct, hi_pct)
    band_defs = {
        "P95": (2.5, 97.5),
        "P97": (1.5, 98.5),
        "P98": (1.0, 99.0),
        "P99": (0.5, 99.5),
        "P100": (0.0, 100.0),
    }

    use_weighted = weights is not None and len(weights) == len(returns)
    bands = {}

    for name, (lo_p, hi_p) in band_defs.items():
        # Symmetric baseline
        if use_weighted:
            sym_lo = weighted_percentile(returns, weights, lo_p) / 100.0
            sym_hi = weighted_percentile(returns, weights, hi_p) / 100.0
        else:
            sym_lo = np.percentile(returns, lo_p) / 100.0
            sym_hi = np.percentile(returns, hi_p) / 100.0

        # Directional adjustment:
        # - Higher P(up) -> wider upper band (more room for upside)
        # - Higher P(down) -> wider lower band (more room for downside)
        p_up = direction_prob.p_up
        p_down = direction_prob.p_down

        # Asymmetric scaling: at P(up)=0.5 -> scale=1.0 (symmetric)
        # at P(up)=0.7 -> upper_scale=1.4, lower_scale=0.6
        # We use 2*p as the scale factor (so 0.5 maps to 1.0)
        upper_scale = 2.0 * p_up
        lower_scale = 2.0 * p_down

        # Apply scaling to the distance from center
        center = (sym_lo + sym_hi) / 2.0
        asym_lo = center + (sym_lo - center) * lower_scale
        asym_hi = center + (sym_hi - center) * upper_scale

        # Safety: asymmetric bands must be at least as wide as symmetric
        final_lo = min(asym_lo, sym_lo)
        final_hi = max(asym_hi, sym_hi)

        lo_price = current_price * (1 + final_lo)
        hi_price = current_price * (1 + final_hi)
        width_pts = hi_price - lo_price
        width_pct = width_pts / current_price * 100.0 if current_price else 0.0

        bands[name] = UnifiedBand(
            name=name,
            lo_price=lo_price,
            hi_price=hi_price,
            lo_pct=final_lo * 100.0,
            hi_pct=final_hi * 100.0,
            width_pts=width_pts,
            width_pct=width_pct,
            source="directional",
        )

    return bands


def compute_directional_analysis(
    current_context: Any,
    current_price: float,
    n_day_returns: np.ndarray,
    historical_contexts: List[Any],
    days_ahead: int,
    weights: Optional[np.ndarray] = None,
) -> DirectionalAnalysis:
    """Top-level entry point for directional momentum analysis.

    Orchestrates momentum classification, directional probability,
    and asymmetric band computation.

    Args:
        current_context: Current MarketContext.
        current_price: Current price.
        n_day_returns: Array of historical N-day returns (as percentages).
        historical_contexts: List of historical MarketContext objects.
        days_ahead: Prediction horizon in days.
        weights: Optional sample weights.

    Returns:
        DirectionalAnalysis with momentum_state, direction_probability,
        and asymmetric_bands.
    """
    # Step 1: Classify momentum
    momentum = classify_momentum(current_context)

    # Step 2: Convert returns to decimals for probability computation
    returns_decimal = np.asarray(n_day_returns, dtype=float) / 100.0

    # Step 3: Compute directional probability
    dir_prob = compute_directional_probability(
        returns_decimal, historical_contexts, current_context, days_ahead,
    )

    # Step 4: Compute asymmetric bands (uses percentage returns)
    asym_bands = compute_asymmetric_bands(
        n_day_returns, current_price, dir_prob, weights,
    )

    return DirectionalAnalysis(
        momentum_state=momentum,
        direction_probability=dir_prob,
        asymmetric_bands=asym_bands,
    )
