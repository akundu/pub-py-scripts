#!/usr/bin/env python3
"""
Adaptive multi-day predictor with regime detection and confidence-based method selection.

This integrates Phase 3 improvements:
- Regime detection
- Model confidence scoring
- Automatic fallback to conditional/baseline when appropriate
- Rolling error tracking and retraining detection

Usage:
    predictor = AdaptiveMultiDayPredictor()
    predictor.load_models('results/multi_day_walkforward/models')

    # Get prediction with automatic method selection
    bands, metadata = predictor.predict_adaptive(
        dte=5,
        context=market_context,
        current_price=20000.0,
    )

    print(f"Selected method: {metadata['selected_method']}")
    print(f"Reason: {metadata['selection_reason']}")
    print(f"Ensemble confidence: {metadata.get('ensemble_confidence', 0.0):.2f}")
"""

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict

from scripts.close_predictor.multi_day_lgbm import MultiDayEnsemble
from scripts.close_predictor.multi_day_features import MarketContext
from scripts.close_predictor.regime_detector import RegimeDetector
from scripts.close_predictor.models import UnifiedBand


@dataclass
class PredictionMetadata:
    """Metadata about adaptive prediction."""
    selected_method: str  # 'ensemble', 'conditional', 'baseline'
    selection_reason: str
    ensemble_confidence: Optional[float] = None
    regime_state: Optional[str] = None
    feature_drift: Optional[float] = None
    should_retrain: bool = False
    retraining_reason: Optional[str] = None


class AdaptiveMultiDayPredictor:
    """Adaptive predictor with regime-based method selection."""

    def __init__(
        self,
        confidence_threshold_ensemble: float = 0.7,
        confidence_threshold_conditional: float = 0.5,
        enable_regime_detection: bool = True,
        enable_confidence_scoring: bool = True,
        enable_retraining_detection: bool = True,
    ):
        """Initialize adaptive predictor.

        Args:
            confidence_threshold_ensemble: Min confidence to use ensemble
            confidence_threshold_conditional: Min confidence to use conditional
            enable_regime_detection: Enable regime-based fallback
            enable_confidence_scoring: Enable confidence-based method selection
            enable_retraining_detection: Enable retraining detection
        """
        self.confidence_threshold_ensemble = confidence_threshold_ensemble
        self.confidence_threshold_conditional = confidence_threshold_conditional
        self.enable_regime_detection = enable_regime_detection
        self.enable_confidence_scoring = enable_confidence_scoring
        self.enable_retraining_detection = enable_retraining_detection

        # Initialize predictors
        self.ensemble = MultiDayEnsemble()
        self.regime_detector = RegimeDetector() if enable_regime_detection else None

        # Baseline predictor (simple statistical)
        self.baseline_width_multipliers = {
            1: 0.055,
            2: 0.078,
            3: 0.100,
            5: 0.121,
            7: 0.145,
            10: 0.167,
            15: 0.215,
            20: 0.240,
        }

    def load_ensemble_models(self, models_dir: Path):
        """Load ensemble models from directory.

        Args:
            models_dir: Directory containing LGBM models
        """
        self.ensemble.load_all(models_dir)
        print(f"✓ Loaded {len(self.ensemble.models)} ensemble models")


    def predict_baseline(
        self,
        dte: int,
        current_price: float,
        context: MarketContext,
    ) -> Dict[str, UnifiedBand]:
        """Generate baseline prediction using simple statistical approach.

        Args:
            dte: Days to expiration
            current_price: Current price
            context: Market context (for VIX scaling)

        Returns:
            Dict of percentile bands
        """
        # Get base width multiplier
        if dte in self.baseline_width_multipliers:
            base_width_pct = self.baseline_width_multipliers[dte]
        else:
            # Interpolate for missing DTEs
            dtes = sorted(self.baseline_width_multipliers.keys())
            if dte < min(dtes):
                base_width_pct = self.baseline_width_multipliers[min(dtes)]
            elif dte > max(dtes):
                base_width_pct = self.baseline_width_multipliers[max(dtes)]
            else:
                # Linear interpolation
                for i in range(len(dtes) - 1):
                    if dtes[i] <= dte <= dtes[i + 1]:
                        t = (dte - dtes[i]) / (dtes[i + 1] - dtes[i])
                        base_width_pct = (
                            self.baseline_width_multipliers[dtes[i]] * (1 - t) +
                            self.baseline_width_multipliers[dtes[i + 1]] * t
                        )
                        break

        # VIX scaling
        vix = getattr(context, 'vix', 15.0)
        vix_scale = 1.0
        if vix < 12:
            vix_scale = 0.85
        elif vix > 20:
            vix_scale = 1.0 + (vix - 20) * 0.02

        # Build percentile bands
        band_defs = {
            'P95': 1.96,   # 2 std devs
            'P97': 2.17,
            'P98': 2.33,
            'P99': 2.58,
        }

        bands = {}
        for name, z_score in band_defs.items():
            width_pct = base_width_pct * z_score * vix_scale
            lo_price = current_price * (1 - width_pct / 2)
            hi_price = current_price * (1 + width_pct / 2)

            bands[name] = UnifiedBand(
                name=name,
                lo_price=lo_price,
                hi_price=hi_price,
                lo_pct=-(width_pct / 2) * 100,
                hi_pct=(width_pct / 2) * 100,
                width_pts=hi_price - lo_price,
                width_pct=width_pct * 100,
                source="baseline",
            )

        return bands

    def select_prediction_method(
        self,
        dte: int,
        context: MarketContext,
    ) -> Tuple[str, str, Dict]:
        """Select best prediction method based on regime and confidence.

        Args:
            dte: Days to expiration
            context: Market context

        Returns:
            (method, reason, debug_info)
        """
        debug_info = {}

        # Check 1: Regime detection
        if self.enable_regime_detection and self.regime_detector:
            should_fallback, regime_reason = self.regime_detector.should_use_fallback(context)
            debug_info['regime_fallback'] = should_fallback
            debug_info['regime_reason'] = regime_reason

            if should_fallback:
                return 'baseline', f"Regime fallback: {regime_reason}", debug_info

            # Get current regime state
            regime_state = self.regime_detector.get_current_regime(context)
            debug_info['regime_state'] = regime_state.vix_regime
            debug_info['regime_confidence'] = regime_state.confidence

        # Check 2: Ensemble confidence
        ensemble_confidence = None
        if self.enable_confidence_scoring and dte in self.ensemble.models:
            # Get feature drift score
            feature_drift = None
            if hasattr(self.ensemble.models[dte], 'monitor_feature_drift'):
                feature_drift = self.ensemble.models[dte].monitor_feature_drift(context)
                debug_info['feature_drift'] = feature_drift

            # Get ensemble confidence
            ensemble_confidence = self.ensemble.get_prediction_confidence(
                dte=dte,
                context=context,
                feature_drift_score=feature_drift,
            )
            debug_info['ensemble_confidence'] = ensemble_confidence

            # Use regime detector's recommendation if available
            if self.regime_detector:
                method, reason = self.regime_detector.get_recommended_method(
                    context=context,
                    ensemble_confidence=ensemble_confidence,
                )
                debug_info['recommended_method'] = method
                return method, reason, debug_info

            # Fallback to simple confidence thresholds
            if ensemble_confidence >= self.confidence_threshold_ensemble:
                return 'ensemble', f"High confidence ({ensemble_confidence:.2f})", debug_info
            elif ensemble_confidence >= self.confidence_threshold_conditional:
                return 'conditional', f"Moderate confidence ({ensemble_confidence:.2f})", debug_info
            else:
                return 'baseline', f"Low confidence ({ensemble_confidence:.2f})", debug_info

        # Check 3: Retraining detection
        if self.enable_retraining_detection and dte in self.ensemble.models:
            needs_retrain, retrain_reason = self.ensemble.needs_retraining(dte)
            debug_info['needs_retraining'] = needs_retrain
            debug_info['retraining_reason'] = retrain_reason

            if needs_retrain:
                return 'baseline', f"Retraining needed: {retrain_reason}", debug_info

        # Default: use ensemble if model exists
        if dte in self.ensemble.models:
            return 'ensemble', "Default (ensemble available)", debug_info
        else:
            return 'baseline', "Default (no ensemble available)", debug_info

    def predict_adaptive(
        self,
        dte: int,
        context: MarketContext,
        current_price: float,
    ) -> Tuple[Dict[str, UnifiedBand], PredictionMetadata]:
        """Generate adaptive prediction with automatic method selection.

        Args:
            dte: Days to expiration
            context: Market context
            current_price: Current price

        Returns:
            (bands, metadata)
        """
        # Select method
        method, reason, debug_info = self.select_prediction_method(dte, context)

        # Generate prediction using selected method
        if method == 'ensemble':
            bands = self.ensemble.predict(dte, context, current_price)
        else:
            # Both 'conditional' and 'baseline' use baseline for now
            # TODO: Implement proper conditional predictor
            bands = self.predict_baseline(dte, current_price, context)

        # Build metadata
        metadata = PredictionMetadata(
            selected_method=method,
            selection_reason=reason,
            ensemble_confidence=debug_info.get('ensemble_confidence'),
            regime_state=debug_info.get('regime_state'),
            feature_drift=debug_info.get('feature_drift'),
            should_retrain=debug_info.get('needs_retraining', False),
            retraining_reason=debug_info.get('retraining_reason'),
        )

        return bands, metadata

    def update_regime_detector(self, context: MarketContext):
        """Update regime detector with current context.

        Args:
            context: Current market context
        """
        if self.regime_detector:
            change = self.regime_detector.detect_regime_change(context)
            if change:
                print(f"⚠️  Regime change detected: {change.from_regime.vix_regime} → "
                      f"{change.to_regime.vix_regime} (severity: {change.severity:.2f})")

    def record_prediction_outcome(
        self,
        dte: int,
        predicted_return: float,
        actual_return: float,
    ):
        """Record prediction outcome for error tracking.

        Args:
            dte: Days ahead
            predicted_return: Predicted return %
            actual_return: Actual return %
        """
        self.ensemble.record_prediction_outcome(dte, predicted_return, actual_return)

    def get_health_report(self) -> Dict:
        """Get health report for all models.

        Returns:
            Dict with health metrics
        """
        return self.ensemble.get_ensemble_health_report()


def test_adaptive_predictor():
    """Test adaptive predictor."""
    from scripts.close_predictor.multi_day_features import MarketContext

    predictor = AdaptiveMultiDayPredictor()

    # Test scenarios
    scenarios = [
        ("Stable low VIX", MarketContext(vix=14.0, volume_ratio=1.0)),
        ("High VIX spike", MarketContext(vix=28.0, volume_ratio=1.8)),
        ("Extreme VIX", MarketContext(vix=35.0, volume_ratio=2.5)),
        ("Very low VIX", MarketContext(vix=9.5, volume_ratio=0.8)),
    ]

    print("=" * 70)
    print("ADAPTIVE PREDICTOR TEST")
    print("=" * 70)

    for scenario_name, context in scenarios:
        print(f"\n{scenario_name}:")
        print(f"  VIX: {context.vix:.1f}, Volume: {context.volume_ratio:.1f}x")

        # Test method selection
        method, reason, debug = predictor.select_prediction_method(5, context)
        print(f"  → Selected method: {method}")
        print(f"  → Reason: {reason}")

        if 'ensemble_confidence' in debug:
            print(f"  → Ensemble confidence: {debug['ensemble_confidence']:.2f}")
        if 'regime_state' in debug:
            print(f"  → Regime: {debug['regime_state']}")

        # Generate baseline prediction
        bands = predictor.predict_baseline(5, 20000.0, context)
        if 'P99' in bands:
            p99 = bands['P99']
            print(f"  → P99 band: ${p99.lo_price:.0f} - ${p99.hi_price:.0f} "
                  f"(width: {p99.width_pct:.2f}%)")


if __name__ == '__main__':
    test_adaptive_predictor()
