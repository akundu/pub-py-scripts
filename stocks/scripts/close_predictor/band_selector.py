"""
Intelligent band selection for options trading based on market conditions.

Recommends which confidence band (P95/P97/P98/P99) to use based on:
- Volatility regime (VIX levels)
- Time to close
- Recent price action
- Trend strength
- Your risk tolerance

Goal: Be correct and bounded, but don't miss opportunities.
"""

from dataclasses import dataclass
from typing import Optional, Tuple
from enum import Enum


class RiskProfile(Enum):
    """Trading risk profile."""
    AGGRESSIVE = "aggressive"      # Max premium, accept more risk
    MODERATE = "moderate"          # Balanced risk/reward
    CONSERVATIVE = "conservative"  # Safety first, lower premium


class MarketRegime(Enum):
    """Current market volatility regime."""
    LOW_VOL = "low_vol"           # VIX < 12
    NORMAL = "normal"             # VIX 12-20
    ELEVATED = "elevated"         # VIX 20-30
    HIGH_VOL = "high_vol"         # VIX > 30


@dataclass
class BandRecommendation:
    """Recommended band selection with rationale."""
    recommended_band: str          # "P95", "P97", "P98", or "P99"
    confidence_level: float        # 0.0-1.0
    expected_hit_rate: float       # 0.80-0.99
    rationale: str                 # Why this band?
    alternative_band: str          # If you want more/less risk
    opportunity_score: float       # 0.0-1.0 (higher = better premium opportunity)


def get_market_regime(vix: float) -> MarketRegime:
    """Classify current volatility regime."""
    if vix < 12:
        return MarketRegime.LOW_VOL
    elif vix < 20:
        return MarketRegime.NORMAL
    elif vix < 30:
        return MarketRegime.ELEVATED
    else:
        return MarketRegime.HIGH_VOL


def calculate_trend_strength(
    current_price: float,
    prev_close: float,
    day_high: float,
    day_low: float,
) -> float:
    """
    Calculate trend strength (0.0-1.0).

    Higher = stronger directional move, less likely to reverse.
    """
    if day_high == day_low:
        return 0.5

    # Range position (0 = at low, 1 = at high)
    range_pos = (current_price - day_low) / (day_high - day_low)

    # Distance from previous close
    move_pct = abs(current_price - prev_close) / prev_close

    # Strong trend if:
    # 1. Price at extreme of range (< 0.2 or > 0.8)
    # 2. Significant move from prev close (> 0.5%)

    extreme_position = max(0, (abs(range_pos - 0.5) - 0.3) / 0.2)  # 0-1
    significant_move = min(1.0, move_pct / 0.01)  # 1% move = 1.0

    return (extreme_position + significant_move) / 2


def recommend_band(
    vix: float,
    hours_to_close: float,
    current_price: float,
    prev_close: float,
    day_high: float,
    day_low: float,
    risk_profile: RiskProfile = RiskProfile.MODERATE,
    realized_vol: Optional[float] = None,
    historical_avg_vol: Optional[float] = None,
) -> BandRecommendation:
    """
    Recommend optimal band based on market conditions and risk profile.

    Args:
        vix: Current VIX level
        hours_to_close: Hours until market close
        current_price: Current index price
        prev_close: Previous day's close
        day_high: Today's high
        day_low: Today's low
        risk_profile: Your trading risk profile
        realized_vol: Recent realized volatility
        historical_avg_vol: Long-term average volatility

    Returns:
        BandRecommendation with suggested band and rationale
    """
    regime = get_market_regime(vix)
    trend_strength = calculate_trend_strength(current_price, prev_close, day_high, day_low)

    # Calculate vol ratio (realized vs historical)
    vol_ratio = 1.0
    if realized_vol and historical_avg_vol and historical_avg_vol > 0:
        vol_ratio = realized_vol / historical_avg_vol

    # Base recommendation matrix
    # [regime][risk_profile] -> (band, confidence, expected_hit)
    recommendations = {
        MarketRegime.LOW_VOL: {
            RiskProfile.AGGRESSIVE: ("P95", 0.85, 0.80),
            RiskProfile.MODERATE: ("P97", 0.90, 0.90),
            RiskProfile.CONSERVATIVE: ("P98", 0.95, 0.95),
        },
        MarketRegime.NORMAL: {
            RiskProfile.AGGRESSIVE: ("P97", 0.85, 0.85),
            RiskProfile.MODERATE: ("P98", 0.90, 0.93),
            RiskProfile.CONSERVATIVE: ("P99", 0.95, 0.98),
        },
        MarketRegime.ELEVATED: {
            RiskProfile.AGGRESSIVE: ("P98", 0.80, 0.88),
            RiskProfile.MODERATE: ("P99", 0.90, 0.95),
            RiskProfile.CONSERVATIVE: ("P99", 0.95, 0.98),
        },
        MarketRegime.HIGH_VOL: {
            RiskProfile.AGGRESSIVE: ("P98", 0.75, 0.85),
            RiskProfile.MODERATE: ("P99", 0.85, 0.93),
            RiskProfile.CONSERVATIVE: ("P99", 0.90, 0.97),
        },
    }

    base_band, base_confidence, base_hit_rate = recommendations[regime][risk_profile]

    # Adjustments based on additional factors

    # 1. Time to close - less time = more certainty = can use tighter band
    if hours_to_close < 1.0:
        # Last hour - very predictable
        time_adjustment = -1  # Move to tighter band
        confidence_boost = 0.05
    elif hours_to_close < 2.0:
        # Last 2 hours - quite predictable
        time_adjustment = 0
        confidence_boost = 0.03
    elif hours_to_close > 5.0:
        # Early day - more uncertainty
        time_adjustment = 1  # Move to wider band
        confidence_boost = -0.05
    else:
        time_adjustment = 0
        confidence_boost = 0

    # 2. Trend strength - strong trend = less likely to reverse = tighter band
    if trend_strength > 0.7:
        # Strong directional move
        time_adjustment -= 1
        confidence_boost += 0.05
    elif trend_strength < 0.3:
        # Choppy, no clear direction
        time_adjustment += 1
        confidence_boost -= 0.03

    # 3. Vol ratio - if realized vol much higher than normal, use wider bands
    if vol_ratio > 1.5:
        # Vol spike - be more conservative
        time_adjustment += 1
        confidence_boost -= 0.05
    elif vol_ratio < 0.7:
        # Vol compression - can be more aggressive
        time_adjustment -= 1
        confidence_boost += 0.03

    # Apply adjustments
    band_names = ["P95", "P97", "P98", "P99"]
    base_idx = band_names.index(base_band)
    adjusted_idx = max(0, min(3, base_idx + time_adjustment))
    recommended_band = band_names[adjusted_idx]

    # Adjusted confidence and hit rate
    adjusted_confidence = min(1.0, base_confidence + confidence_boost)
    adjusted_hit_rate = base_hit_rate + (confidence_boost * 2)  # 2x effect on hit rate

    # Alternative band (one step different)
    if risk_profile == RiskProfile.AGGRESSIVE:
        # Suggest more conservative alternative
        alt_idx = min(3, adjusted_idx + 1)
    else:
        # Suggest more aggressive alternative
        alt_idx = max(0, adjusted_idx - 1)
    alternative_band = band_names[alt_idx]

    # Calculate opportunity score (inverse of conservatism)
    # P95 = high opportunity (100%), P99 = low opportunity (25%)
    opportunity_score = 1.0 - (adjusted_idx * 0.25)

    # Build rationale
    rationale_parts = []
    rationale_parts.append(f"{regime.value.replace('_', ' ').title()} volatility (VIX={vix:.1f})")

    if hours_to_close < 1.0:
        rationale_parts.append("Last hour (high confidence)")
    elif hours_to_close < 2.0:
        rationale_parts.append("Near close (good confidence)")
    elif hours_to_close > 5.0:
        rationale_parts.append("Early day (lower confidence)")

    if trend_strength > 0.7:
        rationale_parts.append("Strong trend (directional bias)")
    elif trend_strength < 0.3:
        rationale_parts.append("Choppy action (uncertain)")

    if vol_ratio > 1.5:
        rationale_parts.append("Vol spike detected (be cautious)")
    elif vol_ratio < 0.7:
        rationale_parts.append("Vol compression (stable)")

    rationale_parts.append(f"{risk_profile.value.title()} risk profile")

    rationale = " | ".join(rationale_parts)

    return BandRecommendation(
        recommended_band=recommended_band,
        confidence_level=adjusted_confidence,
        expected_hit_rate=adjusted_hit_rate,
        rationale=rationale,
        alternative_band=alternative_band,
        opportunity_score=opportunity_score,
    )


def format_recommendation(rec: BandRecommendation) -> str:
    """Format recommendation for display."""
    lines = []
    lines.append("="*80)
    lines.append("INTELLIGENT BAND RECOMMENDATION")
    lines.append("="*80)
    lines.append(f"\n✓ RECOMMENDED: {rec.recommended_band} Band")
    lines.append(f"  Confidence: {rec.confidence_level*100:.0f}%")
    lines.append(f"  Expected Hit Rate: {rec.expected_hit_rate*100:.0f}%")
    lines.append(f"  Opportunity Score: {rec.opportunity_score*100:.0f}/100")
    lines.append(f"\n  Rationale: {rec.rationale}")
    lines.append(f"\n  Alternative: {rec.alternative_band} (if you want {'more' if rec.alternative_band > rec.recommended_band else 'less'} risk)")
    lines.append("\n" + "="*80)
    return "\n".join(lines)
