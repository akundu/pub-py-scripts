#!/usr/bin/env python3
"""
Regime detection for multi-day prediction models.

Detects when market regime has changed significantly enough that
models should fall back to baseline/conditional methods instead of
using potentially stale ensemble predictions.
"""

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
from collections import deque

from scripts.close_predictor.multi_day_features import MarketContext


@dataclass
class RegimeState:
    """Current market regime state."""
    vix_regime: str  # 'low', 'medium', 'high', 'extreme'
    vix_level: float
    vix_1d: Optional[float]
    iv_rank: Optional[float]
    volume_regime: str  # 'low', 'normal', 'high'
    trend: str  # 'up', 'down', 'sideways'
    is_stable: bool  # True if regime has been stable
    confidence: float  # Confidence in current regime (0-1)


@dataclass
class RegimeChange:
    """Information about a detected regime change."""
    from_regime: RegimeState
    to_regime: RegimeState
    timestamp: datetime
    severity: float  # 0.0-1.0, higher = more severe change
    should_fallback: bool  # True if should use fallback methods


class RegimeDetector:
    """Detects market regime changes for model selection."""

    def __init__(
        self,
        stability_window: int = 5,
        vix_change_threshold: float = 5.0,
        volume_change_threshold: float = 0.5,
    ):
        """Initialize regime detector.

        Args:
            stability_window: Number of days to consider for regime stability
            vix_change_threshold: VIX change (absolute) to trigger regime change
            volume_change_threshold: Volume ratio change to trigger regime change
        """
        self.stability_window = stability_window
        self.vix_change_threshold = vix_change_threshold
        self.volume_change_threshold = volume_change_threshold

        # Track recent regime states
        self.recent_states: deque = deque(maxlen=stability_window)
        self.current_state: Optional[RegimeState] = None
        self.last_change: Optional[RegimeChange] = None

    def classify_vix_regime(self, vix: float) -> str:
        """Classify VIX level into regime.

        Args:
            vix: VIX level

        Returns:
            Regime string: 'low', 'medium', 'high', 'extreme'
        """
        if vix < 12:
            return 'low'
        elif vix < 20:
            return 'medium'
        elif vix < 30:
            return 'high'
        else:
            return 'extreme'

    def classify_volume_regime(self, volume_ratio: float) -> str:
        """Classify volume level into regime.

        Args:
            volume_ratio: Volume vs average

        Returns:
            Regime string: 'low', 'normal', 'high'
        """
        if volume_ratio < 0.7:
            return 'low'
        elif volume_ratio < 1.5:
            return 'normal'
        else:
            return 'high'

    def detect_trend(self, context: MarketContext) -> str:
        """Detect price trend from context.

        Args:
            context: Market context

        Returns:
            Trend: 'up', 'down', 'sideways'
        """
        # Use price momentum features if available
        momentum_5d = getattr(context, 'momentum_5d', None)
        if momentum_5d is not None:
            if momentum_5d > 1.5:
                return 'up'
            elif momentum_5d < -1.5:
                return 'down'
            else:
                return 'sideways'

        # Fallback to simple heuristic
        return 'sideways'

    def get_current_regime(self, context: MarketContext) -> RegimeState:
        """Determine current market regime from context.

        Args:
            context: Current market context

        Returns:
            Current regime state
        """
        vix = getattr(context, 'vix', 15.0)
        vix_1d = getattr(context, 'vix_1d', None) or getattr(context, 'vix1d', None)
        iv_rank = getattr(context, 'iv_rank', None)
        volume_ratio = getattr(context, 'volume_ratio', 1.0)

        vix_regime = self.classify_vix_regime(vix)
        volume_regime = self.classify_volume_regime(volume_ratio)
        trend = self.detect_trend(context)

        # Determine regime stability
        is_stable = self._is_regime_stable(vix_regime, volume_regime)

        # Calculate regime confidence
        confidence = self._calculate_regime_confidence(context)

        return RegimeState(
            vix_regime=vix_regime,
            vix_level=vix,
            vix_1d=vix_1d,
            iv_rank=iv_rank,
            volume_regime=volume_regime,
            trend=trend,
            is_stable=is_stable,
            confidence=confidence,
        )

    def _is_regime_stable(self, current_vix_regime: str, current_volume_regime: str) -> bool:
        """Check if regime has been stable recently.

        Args:
            current_vix_regime: Current VIX regime
            current_volume_regime: Current volume regime

        Returns:
            True if regime is stable
        """
        if len(self.recent_states) < self.stability_window:
            return False

        # Check if recent states are all the same regime
        for state in self.recent_states:
            if (state.vix_regime != current_vix_regime or
                state.volume_regime != current_volume_regime):
                return False

        return True

    def _calculate_regime_confidence(self, context: MarketContext) -> float:
        """Calculate confidence in current regime classification.

        Args:
            context: Market context

        Returns:
            Confidence score 0.0-1.0
        """
        confidence_factors = []

        # Factor 1: VIX clarity (extreme values = high confidence)
        vix = getattr(context, 'vix', 15.0)
        if vix < 10 or vix > 35:
            confidence_factors.append(0.9)  # Very clear regime
        elif 12 <= vix <= 18:
            confidence_factors.append(0.8)  # Clear low/medium regime
        else:
            confidence_factors.append(0.6)  # Transitional zone

        # Factor 2: Volume consistency
        volume_ratio = getattr(context, 'volume_ratio', 1.0)
        if 0.8 <= volume_ratio <= 1.2:
            confidence_factors.append(0.9)  # Normal, stable volume
        else:
            confidence_factors.append(0.7)  # Unusual volume

        # Factor 3: Stability
        if len(self.recent_states) >= self.stability_window:
            confidence_factors.append(0.9)  # Stable regime
        else:
            confidence_factors.append(0.5)  # New or changing regime

        return float(np.mean(confidence_factors))

    def detect_regime_change(
        self,
        context: MarketContext,
        timestamp: Optional[datetime] = None,
    ) -> Optional[RegimeChange]:
        """Detect if regime has changed significantly.

        Args:
            context: Current market context
            timestamp: Current timestamp (optional)

        Returns:
            RegimeChange if change detected, None otherwise
        """
        new_regime = self.get_current_regime(context)

        if self.current_state is None:
            # First time - initialize
            self.current_state = new_regime
            self.recent_states.append(new_regime)
            return None

        # Check for significant changes
        vix_change = abs(new_regime.vix_level - self.current_state.vix_level)
        regime_changed = (
            new_regime.vix_regime != self.current_state.vix_regime or
            new_regime.volume_regime != self.current_state.volume_regime
        )

        # Calculate change severity
        severity = 0.0

        if new_regime.vix_regime != self.current_state.vix_regime:
            # VIX regime changed
            severity += 0.5

        if vix_change > self.vix_change_threshold:
            # Large VIX spike
            severity += min(0.5, vix_change / 20.0)

        if new_regime.volume_regime != self.current_state.volume_regime:
            # Volume regime changed
            severity += 0.3

        # Extreme regimes = higher severity
        if new_regime.vix_regime == 'extreme':
            severity += 0.4

        severity = min(1.0, severity)

        # Determine if should fallback to baseline
        should_fallback = severity > 0.5 or new_regime.vix_regime == 'extreme'

        if regime_changed or severity > 0.3:
            change = RegimeChange(
                from_regime=self.current_state,
                to_regime=new_regime,
                timestamp=timestamp or datetime.now(),
                severity=severity,
                should_fallback=should_fallback,
            )

            # Update state
            self.current_state = new_regime
            self.recent_states.append(new_regime)
            self.last_change = change

            return change

        # No significant change
        self.recent_states.append(new_regime)
        return None

    def should_use_fallback(self, context: MarketContext) -> Tuple[bool, str]:
        """Determine if should use fallback methods instead of ensemble.

        Args:
            context: Current market context

        Returns:
            (should_fallback, reason)
        """
        current_regime = self.get_current_regime(context)

        # Extreme VIX = always fallback
        if current_regime.vix_regime == 'extreme':
            return True, f"Extreme VIX regime ({current_regime.vix_level:.1f})"

        # Very low VIX can be unpredictable
        if current_regime.vix_level < 10:
            return True, f"Very low VIX ({current_regime.vix_level:.1f})"

        # Recent regime change = fallback
        if self.last_change and self.last_change.should_fallback:
            return True, f"Recent regime change (severity: {self.last_change.severity:.2f})"

        # Unstable regime = use caution
        if not current_regime.is_stable:
            return True, "Regime is unstable"

        # Low confidence in regime = fallback
        if current_regime.confidence < 0.6:
            return True, f"Low regime confidence ({current_regime.confidence:.2f})"

        return False, "Regime is stable, use ensemble"

    def get_recommended_method(
        self,
        context: MarketContext,
        ensemble_confidence: float,
    ) -> Tuple[str, str]:
        """Get recommended prediction method.

        Args:
            context: Current market context
            ensemble_confidence: Ensemble model's confidence score

        Returns:
            (method, reason) where method is 'ensemble', 'conditional', or 'baseline'
        """
        should_fallback, regime_reason = self.should_use_fallback(context)

        if should_fallback:
            # Use baseline (most conservative)
            return 'baseline', f"Regime fallback: {regime_reason}"

        # Check ensemble confidence
        if ensemble_confidence < 0.5:
            return 'conditional', f"Low ensemble confidence ({ensemble_confidence:.2f})"
        elif ensemble_confidence < 0.7:
            return 'conditional', f"Moderate ensemble confidence ({ensemble_confidence:.2f})"
        else:
            return 'ensemble', f"High confidence ({ensemble_confidence:.2f}), stable regime"


def test_regime_detector():
    """Test regime detector."""
    from scripts.close_predictor.multi_day_features import MarketContext

    detector = RegimeDetector()

    # Test scenario 1: Stable low VIX
    print("=" * 60)
    print("Test 1: Stable low VIX regime")
    print("=" * 60)

    for i in range(10):
        ctx = MarketContext(vix=14.0 + i * 0.2, volume_ratio=1.0)
        change = detector.detect_regime_change(ctx)

        if change:
            print(f"Day {i}: REGIME CHANGE detected!")
            print(f"  From: {change.from_regime.vix_regime} (VIX {change.from_regime.vix_level:.1f})")
            print(f"  To: {change.to_regime.vix_regime} (VIX {change.to_regime.vix_level:.1f})")
            print(f"  Severity: {change.severity:.2f}")
            print(f"  Should fallback: {change.should_fallback}")

        should_fallback, reason = detector.should_use_fallback(ctx)
        method, method_reason = detector.get_recommended_method(ctx, ensemble_confidence=0.8)

        print(f"Day {i}: VIX={ctx.vix:.1f}, Regime={detector.current_state.vix_regime}")
        print(f"  Use fallback: {should_fallback} ({reason})")
        print(f"  Recommended: {method} ({method_reason})")

    # Test scenario 2: VIX spike
    print("\n" + "=" * 60)
    print("Test 2: VIX spike from 15 to 25")
    print("=" * 60)

    for i, vix in enumerate([15, 16, 18, 22, 25, 24, 23, 22, 20]):
        ctx = MarketContext(vix=vix, volume_ratio=1.5 if vix > 20 else 1.0)
        change = detector.detect_regime_change(ctx)

        if change:
            print(f"\nDay {i}: ⚠️  REGIME CHANGE!")
            print(f"  {change.from_regime.vix_regime} → {change.to_regime.vix_regime}")
            print(f"  Severity: {change.severity:.2f}, Should fallback: {change.should_fallback}")

        should_fallback, reason = detector.should_use_fallback(ctx)
        method, method_reason = detector.get_recommended_method(ctx, ensemble_confidence=0.8)

        print(f"Day {i}: VIX={vix:.1f}, Method: {method}")


if __name__ == '__main__':
    test_regime_detector()
