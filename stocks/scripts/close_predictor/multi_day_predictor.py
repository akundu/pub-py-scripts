#!/usr/bin/env python3
"""
Multi-day ahead prediction using conditional distributions.

Instead of using all historical N-day returns equally, weights them by similarity
to current market conditions (volatility, position, momentum, calendar).
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass

from scripts.close_predictor.multi_day_features import MarketContext, compute_feature_similarity
from scripts.close_predictor.models import UnifiedBand
from scripts.close_predictor.bands import map_percentile_to_bands


@dataclass
class WeightedSample:
    """A historical N-day return sample with similarity weight."""
    return_pct: float          # N-day return %
    context: MarketContext     # Market context at start of period
    weight: float = 1.0        # Similarity weight to current conditions
    date: Optional[str] = None # Start date of period


def weight_historical_samples(
    n_day_returns: List[float],
    historical_contexts: List[MarketContext],
    current_context: MarketContext,
    min_samples: int = 30,
    top_k: Optional[int] = None,
) -> np.ndarray:
    """Weight historical N-day returns by similarity to current market conditions.

    Args:
        n_day_returns: List of historical N-day returns (% change)
        historical_contexts: List of MarketContext for each return's starting conditions
        current_context: Current market conditions
        min_samples: Minimum samples to include (even if low similarity)
        top_k: If set, only use top K most similar samples

    Returns:
        Weighted array of returns (may repeat samples based on weights)
    """
    if len(n_day_returns) != len(historical_contexts):
        # Fallback: return unweighted
        return np.array(n_day_returns)

    # Compute similarity scores
    samples = []
    for ret, ctx in zip(n_day_returns, historical_contexts):
        similarity = compute_feature_similarity(current_context, ctx)
        samples.append(WeightedSample(return_pct=ret, context=ctx, weight=similarity))

    # Sort by weight descending
    samples.sort(key=lambda s: s.weight, reverse=True)

    # Take top_k if specified
    if top_k and top_k < len(samples):
        samples = samples[:top_k]

    # Ensure minimum sample size
    if len(samples) < min_samples:
        samples = samples[:min_samples] if len(samples) >= min_samples else samples

    # Create weighted array by repeating samples proportional to weight
    # Normalize weights to sum to 1.0
    total_weight = sum(s.weight for s in samples)
    if total_weight == 0:
        return np.array([s.return_pct for s in samples])

    # Use multinomial resampling to create weighted distribution
    # Each sample gets floor(weight/total * N) + probabilistic extra
    target_size = max(len(samples), 100)  # Want at least 100 effective samples
    weighted_returns = []

    for s in samples:
        prob = s.weight / total_weight
        count = int(prob * target_size)
        weighted_returns.extend([s.return_pct] * count)

    # Fill remaining with highest-weight samples
    while len(weighted_returns) < target_size and samples:
        weighted_returns.append(samples[0].return_pct)

    return np.array(weighted_returns)


def segment_by_volatility_regime(
    n_day_returns: List[float],
    historical_contexts: List[MarketContext],
    current_context: MarketContext,
) -> np.ndarray:
    """Segment historical samples by volatility regime, return only matching regime.

    Args:
        n_day_returns: Historical N-day returns
        historical_contexts: Historical market contexts
        current_context: Current market conditions

    Returns:
        Array of returns from matching volatility regime
    """
    if len(n_day_returns) != len(historical_contexts):
        return np.array(n_day_returns)

    # Get current regime
    current_regime = current_context.vol_regime

    # Filter to matching regime
    regime_returns = [
        ret for ret, ctx in zip(n_day_returns, historical_contexts)
        if ctx.vol_regime == current_regime
    ]

    # If too few samples in exact regime, expand to include medium
    if len(regime_returns) < 30:
        regime_returns = [
            ret for ret, ctx in zip(n_day_returns, historical_contexts)
            if ctx.vol_regime in (current_regime, "medium")
        ]

    # Final fallback: use all
    if len(regime_returns) < 20:
        regime_returns = n_day_returns

    return np.array(regime_returns)


def apply_volatility_scaling(
    returns: np.ndarray,
    current_realized_vol: Optional[float],
    historical_realized_vols: Optional[List[float]],
) -> np.ndarray:
    """Scale historical returns by ratio of current to historical volatility.

    Similar to 0DTE vol_scale_moves but for multi-day periods.

    Args:
        returns: Historical N-day returns
        current_realized_vol: Current realized volatility (annualized %)
        historical_realized_vols: Historical realized vols (parallel to returns)

    Returns:
        Scaled returns array
    """
    if current_realized_vol is None or historical_realized_vols is None:
        return returns

    if len(returns) != len(historical_realized_vols):
        return returns

    # Scale each return by vol ratio
    scaled_returns = []
    for ret, hist_vol in zip(returns, historical_realized_vols):
        if hist_vol is None or hist_vol <= 0:
            scaled_returns.append(ret)
            continue

        vol_ratio = current_realized_vol / hist_vol
        # Cap scaling at 0.5x to 2.0x to avoid extremes
        vol_ratio = np.clip(vol_ratio, 0.5, 2.0)

        scaled_returns.append(ret * vol_ratio)

    return np.array(scaled_returns)


def predict_with_conditional_distribution(
    ticker: str,
    days_ahead: int,
    current_price: float,
    current_context: MarketContext,
    n_day_returns: List[float],
    historical_contexts: List[MarketContext],
    historical_realized_vols: Optional[List[float]] = None,
    use_weighting: bool = True,
    use_regime_filter: bool = True,
    use_vol_scaling: bool = True,
) -> Dict[str, UnifiedBand]:
    """Build percentile bands using conditional distribution.

    Args:
        ticker: Ticker symbol
        days_ahead: Number of trading days ahead
        current_price: Current price
        current_context: Current market conditions
        n_day_returns: Historical N-day returns
        historical_contexts: Historical contexts (parallel to returns)
        historical_realized_vols: Optional realized vols for scaling
        use_weighting: Whether to weight samples by similarity
        use_regime_filter: Whether to filter by volatility regime
        use_vol_scaling: Whether to scale by volatility ratio

    Returns:
        Dict of percentile bands (P95, P97, P98, P99, P100)
    """
    # Step 1: Filter by volatility regime if enabled
    if use_regime_filter:
        returns = segment_by_volatility_regime(
            n_day_returns, historical_contexts, current_context
        )
    else:
        returns = np.array(n_day_returns)

    # Step 2: Apply volatility scaling if enabled
    if use_vol_scaling and current_context.realized_vol_5d:
        returns = apply_volatility_scaling(
            returns,
            current_context.realized_vol_5d,
            historical_realized_vols,
        )

    # Step 3: Weight by similarity if enabled
    if use_weighting:
        # Rebuild contexts list to match filtered returns
        if use_regime_filter:
            filtered_contexts = [
                ctx for ctx in historical_contexts
                if ctx.vol_regime == current_context.vol_regime
                or (len(returns) < 30 and ctx.vol_regime == "medium")
            ]
            if len(filtered_contexts) < len(returns):
                filtered_contexts = historical_contexts[:len(returns)]
        else:
            filtered_contexts = historical_contexts

        returns = weight_historical_samples(
            list(returns),
            filtered_contexts[:len(returns)],
            current_context,
            min_samples=30,
        )

    # Step 4: Build percentile bands
    bands = map_percentile_to_bands(returns, current_price)

    return bands


def get_regime_stats(
    n_day_returns: List[float],
    historical_contexts: List[MarketContext],
) -> Dict[str, Dict]:
    """Get statistics for each volatility regime.

    Returns:
        Dict mapping regime -> stats dict
    """
    regimes = {}

    for regime in ["low", "medium", "high"]:
        regime_returns = [
            ret for ret, ctx in zip(n_day_returns, historical_contexts)
            if ctx.vol_regime == regime
        ]

        if regime_returns:
            regimes[regime] = {
                'count': len(regime_returns),
                'mean': np.mean(regime_returns),
                'median': np.median(regime_returns),
                'std': np.std(regime_returns),
                'min': np.min(regime_returns),
                'max': np.max(regime_returns),
                'p05': np.percentile(regime_returns, 5),
                'p95': np.percentile(regime_returns, 95),
            }

    return regimes
