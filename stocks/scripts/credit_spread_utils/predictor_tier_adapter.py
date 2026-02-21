"""
Adapter between close predictor confidence and time-allocated tiered strategy.

Translates prediction confidence/band width into ROI threshold multipliers
and budget scaling, enabling dynamic tier adjustment based on market conditions.

Three main functions:
- compute_predictor_adjustment(): Compute ROI multiplier and budget scale from prediction
- apply_adjustment_to_tiers(): Deep-copy config with adjusted thresholds
- validate_strikes_against_bands(): Filter positions whose short strikes fall inside predicted bands
"""

import copy
import logging
from dataclasses import dataclass
from typing import Dict, List, Optional

from .time_allocated_tiered_utils import (
    ClosePredictorIntegrationConfig,
    TierConfig,
    TimeAllocatedTieredConfig,
    WindowDeployment,
)


# Confidence levels ordered from highest to lowest
CONFIDENCE_ORDER = ["HIGH", "MEDIUM", "LOW", "VERY_LOW"]


@dataclass
class PredictorAdjustment:
    """Result of computing a predictor-based adjustment for a window."""
    roi_multiplier: float       # Multiplied against each tier's roi_threshold
    budget_scale: float         # Multiplied against window budget_pct
    annotation: str             # Human-readable description of the adjustment
    effective_confidence: str   # Confidence after time-of-day penalty


def _shift_confidence_down(confidence: str, steps: int) -> str:
    """Shift a confidence level down by a number of steps.

    Args:
        confidence: Starting confidence level (HIGH, MEDIUM, LOW, VERY_LOW)
        steps: Number of levels to shift down (0 = no shift)

    Returns:
        Shifted confidence level, clamped to VERY_LOW at minimum.
    """
    if confidence not in CONFIDENCE_ORDER:
        return confidence
    idx = CONFIDENCE_ORDER.index(confidence)
    new_idx = min(idx + steps, len(CONFIDENCE_ORDER) - 1)
    return CONFIDENCE_ORDER[new_idx]


def _penalty_to_steps(penalty: float) -> int:
    """Convert a time-of-day penalty value to confidence shift steps.

    penalty >= 0.15 -> 2 steps (e.g. HIGH -> LOW)
    penalty >= 0.05 -> 1 step  (e.g. HIGH -> MEDIUM)
    penalty < 0.05  -> 0 steps (no shift)
    """
    if penalty >= 0.15:
        return 2
    elif penalty >= 0.05:
        return 1
    return 0


def compute_predictor_adjustment(
    prediction,
    config: ClosePredictorIntegrationConfig,
    window_label: str,
    logger: Optional[logging.Logger] = None,
) -> PredictorAdjustment:
    """Compute ROI multiplier and budget scale from a close prediction.

    Takes the prediction's confidence level, applies time-of-day penalties
    to get an effective confidence, then looks up the ROI multiplier.
    Band width further scales the multiplier:
    - Narrow bands (<1% width) -> 0.90x (tighter = more confident)
    - Wide bands (>2% width)   -> 1.25x (wider = less confident)

    Args:
        prediction: UnifiedPrediction from make_unified_prediction()
        config: ClosePredictorIntegrationConfig with multipliers and penalties
        window_label: e.g. "6am", "7am" — used to look up time penalty
        logger: Optional logger

    Returns:
        PredictorAdjustment with roi_multiplier, budget_scale, annotation
    """
    confidence = prediction.confidence or "MEDIUM"

    # Apply time-of-day penalty
    penalty = config.time_of_day_penalties.get(window_label, 0.0)
    steps = _penalty_to_steps(penalty)
    effective_confidence = _shift_confidence_down(confidence, steps)

    if logger and steps > 0:
        logger.debug(
            f"  Predictor: {confidence} -> {effective_confidence} "
            f"(time penalty={penalty:.2f} for {window_label})"
        )

    # Look up base ROI multiplier from confidence
    roi_multiplier = config.confidence_roi_multipliers.get(effective_confidence, 1.0)

    # Factor in band width
    band = prediction.combined_bands.get(config.band_level)
    if band is None:
        band = prediction.percentile_bands.get(config.band_level)

    band_width_pct = 0.0
    if band is not None:
        band_width_pct = band.width_pct

    if band_width_pct > 0 and band_width_pct < 0.01:
        # Narrow bands -> slightly more aggressive
        roi_multiplier *= 0.90
    elif band_width_pct >= 0.02:
        # Wide bands -> more conservative
        roi_multiplier *= 1.25

    # Budget scale: inverse of ROI multiplier (more confident = deploy more)
    # HIGH confidence (roi_mult ~0.63) -> budget_scale ~1.3
    # LOW confidence (roi_mult ~1.375) -> budget_scale ~0.7
    budget_scale = 1.0 / roi_multiplier if roi_multiplier > 0 else 1.0

    # Clamp budget scale
    lo, hi = config.budget_scale_clamp
    budget_scale = max(lo, min(hi, budget_scale))

    annotation = (
        f"Predictor [{window_label}]: confidence={confidence}->{effective_confidence}, "
        f"roi_mult={roi_multiplier:.3f}, budget_scale={budget_scale:.3f}, "
        f"band_width={band_width_pct*100:.2f}%"
    )

    if logger:
        logger.info(f"  {annotation}")

    return PredictorAdjustment(
        roi_multiplier=roi_multiplier,
        budget_scale=budget_scale,
        annotation=annotation,
        effective_confidence=effective_confidence,
    )


def apply_adjustment_to_tiers(
    config: TimeAllocatedTieredConfig,
    adjustment: PredictorAdjustment,
    window_label: Optional[str] = None,
) -> TimeAllocatedTieredConfig:
    """Create a deep copy of the config with adjusted ROI thresholds and budgets.

    CRITICAL: Never mutate the original config — always return a new copy.
    This prevents state leakage across continuous mode iterations.

    Args:
        config: Original TimeAllocatedTieredConfig (NOT modified)
        adjustment: PredictorAdjustment with multipliers
        window_label: If provided, only adjust the budget for this specific window

    Returns:
        New TimeAllocatedTieredConfig with adjusted values
    """
    adjusted = copy.deepcopy(config)

    # Adjust ROI thresholds for all tiers
    for tier in adjusted.put_tiers:
        tier.roi_threshold *= adjustment.roi_multiplier
    for tier in adjusted.call_tiers:
        tier.roi_threshold *= adjustment.roi_multiplier

    # Adjust budget for the specified window (or all windows)
    lo, hi = config.close_predictor_config.budget_scale_clamp if config.close_predictor_config else (0.5, 1.5)
    for w in adjusted.hourly_windows:
        if window_label is None or w.label == window_label:
            new_budget = w.budget_pct * adjustment.budget_scale
            w.budget_pct = max(0.0, min(new_budget, 1.0))

    return adjusted


def validate_strikes_against_bands(
    deployment: WindowDeployment,
    prediction,
    band_level: str = "P95",
) -> WindowDeployment:
    """Filter positions from a WindowDeployment whose short strikes fall inside the predicted band.

    For PUTs: reject if short_strike >= band.lo_price (strike inside predicted range)
    For CALLs: reject if short_strike <= band.hi_price (strike inside predicted range)

    Args:
        deployment: WindowDeployment with deployed_positions
        prediction: UnifiedPrediction with combined_bands
        band_level: Which band level to validate against (default P95)

    Returns:
        New WindowDeployment with filtered positions (original not modified)
    """
    band = prediction.combined_bands.get(band_level)
    if band is None:
        band = prediction.percentile_bands.get(band_level)
    if band is None:
        return deployment

    filtered = copy.deepcopy(deployment)
    valid_positions = []

    for pos in filtered.deployed_positions:
        if pos.option_type.lower() == 'put':
            if pos.short_strike >= band.lo_price:
                # Strike inside predicted range — reject
                continue
        elif pos.option_type.lower() == 'call':
            if pos.short_strike <= band.hi_price:
                # Strike inside predicted range — reject
                continue
        valid_positions.append(pos)

    filtered.deployed_positions = valid_positions
    return filtered


# Tier-level scaling factors for dynamic percent_beyond:
# T3 (safest) uses 100% of band half-width, T2 uses 85%, T1 uses 70%
TIER_LEVEL_SCALE = {3: 1.0, 2: 0.85, 1: 0.70}


def get_window_band_level(
    config: ClosePredictorIntegrationConfig,
    window_label: str,
) -> str:
    """Get the band level for a specific window.

    Uses per_window_band_levels if configured, otherwise falls back
    to the global band_level.

    Args:
        config: ClosePredictorIntegrationConfig with band settings
        window_label: e.g. "6am", "7am"

    Returns:
        Band level string, e.g. "P99", "P98", "P97", "P95"
    """
    if config.per_window_band_levels:
        return config.per_window_band_levels.get(window_label, config.band_level)
    return config.band_level


def derive_dynamic_percent_beyond(
    prediction,
    band_level: str,
    current_price: float,
) -> Optional[float]:
    """Derive percent_beyond from a prediction's band width at the given level.

    The percent_beyond is half the band width as a fraction of current price.
    This represents the distance from center to edge of the predicted range.

    Args:
        prediction: UnifiedPrediction with combined_bands/percentile_bands
        band_level: Which band to use (P95/P97/P98/P99)
        current_price: Current underlying price for percentage calculation

    Returns:
        Half-width as a fraction (e.g., 0.028 for ±2.8%), or None if band unavailable.
    """
    band = prediction.combined_bands.get(band_level)
    if band is None:
        band = prediction.percentile_bands.get(band_level)
    if band is None or current_price <= 0:
        return None

    half_width = (band.hi_price - band.lo_price) / (2 * current_price)
    return half_width


def apply_dynamic_tiers(
    base_tiers: List[TierConfig],
    band_half_width: float,
    logger: Optional[logging.Logger] = None,
    window_label: str = "",
    tier_level_scale: Optional[Dict] = None,
) -> List[TierConfig]:
    """Create tier copies with percent_beyond derived from predicted band width.

    Each tier level gets a scaled fraction of the band half-width.
    Default scaling: T3=100%, T2=85%, T1=70%.
    Lower scaling values = strikes closer to ATM = more premium = more deployments.

    Args:
        base_tiers: Original tier configs (NOT modified)
        band_half_width: Half-width fraction from derive_dynamic_percent_beyond()
        logger: Optional logger
        window_label: For logging context
        tier_level_scale: Optional dict mapping tier level (str or int) to scale factor.
                         Overrides TIER_LEVEL_SCALE defaults.

    Returns:
        New list of TierConfig with dynamically derived percent_beyond values.
    """
    # Build effective scale map: config overrides > defaults
    effective_scale = dict(TIER_LEVEL_SCALE)
    if tier_level_scale:
        for k, v in tier_level_scale.items():
            effective_scale[int(k)] = float(v)

    adjusted = []
    for tier in base_tiers:
        new_tier = copy.deepcopy(tier)
        scale = effective_scale.get(tier.level, 0.85)
        new_tier.percent_beyond = band_half_width * scale

        if logger:
            logger.info(
                f"    Dynamic T{tier.level} [{window_label}]: "
                f"percent_beyond={new_tier.percent_beyond*100:.3f}% "
                f"(band_half={band_half_width*100:.3f}% × scale={scale:.2f})"
            )
        adjusted.append(new_tier)
    return adjusted
