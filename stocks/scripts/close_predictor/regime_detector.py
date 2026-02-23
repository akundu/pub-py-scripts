#!/usr/bin/env python3
"""
Regime detection for multi-day predictions.

Monitors prediction accuracy over time to detect when market regime changes.
"""

import numpy as np
import json
from pathlib import Path
from datetime import datetime, date
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict


@dataclass
class RegimeState:
    """Current regime state for a ticker/DTE combination."""
    ticker: str
    days_ahead: int
    training_rmse: float = 0.0
    training_mae: float = 0.0
    recent_errors: List[float] = None
    rolling_rmse: float = 0.0
    rolling_mae: float = 0.0
    is_regime_changed: bool = False
    regime_change_confidence: float = 0.0
    rmse_ratio: float = 1.0
    last_updated: str = ""
    regime_change_detected_at: Optional[str] = None
    recommended_method: str = "ensemble"
    fallback_reason: Optional[str] = None

    def __post_init__(self):
        if self.recent_errors is None:
            self.recent_errors = []
        if not self.last_updated:
            self.last_updated = datetime.now().isoformat()

    def to_dict(self) -> Dict:
        return asdict(self)


class RegimeDetector:
    """Detects market regime changes by monitoring prediction accuracy."""

    def __init__(
        self,
        ticker: str,
        days_ahead: int,
        training_rmse: float,
        training_mae: float = 0.0,
        window_size: int = 30,
        regime_change_threshold: float = 1.5,
        cache_dir: Optional[Path] = None,
    ):
        self.ticker = ticker
        self.days_ahead = days_ahead
        self.window_size = window_size
        self.regime_change_threshold = regime_change_threshold
        self.cache_dir = cache_dir

        self.state = RegimeState(
            ticker=ticker,
            days_ahead=days_ahead,
            training_rmse=training_rmse,
            training_mae=training_mae,
            recent_errors=[],
        )

        if cache_dir:
            self._load_state()

    def add_prediction_error(
        self,
        predicted_return: float,
        actual_return: float,
        prediction_date: Optional[date] = None,
    ) -> Dict:
        """Record prediction error and update regime state."""
        error = predicted_return - actual_return

        self.state.recent_errors.append(error)
        if len(self.state.recent_errors) > self.window_size:
            self.state.recent_errors.pop(0)

        if len(self.state.recent_errors) >= 5:
            errors_array = np.array(self.state.recent_errors)
            self.state.rolling_rmse = float(np.sqrt(np.mean(errors_array ** 2)))
            self.state.rolling_mae = float(np.mean(np.abs(errors_array)))

            if self.state.training_rmse > 0:
                self.state.rmse_ratio = self.state.rolling_rmse / self.state.training_rmse
            else:
                self.state.rmse_ratio = 1.0

            self._detect_regime_change()

        self.state.last_updated = datetime.now().isoformat()

        if self.cache_dir:
            self._save_state()

        return self.get_status()

    def _detect_regime_change(self):
        """Detect if regime has changed based on error metrics."""
        rmse_degraded = self.state.rmse_ratio > self.regime_change_threshold

        mae_ratio = 1.0
        if self.state.training_mae > 0:
            mae_ratio = self.state.rolling_mae / self.state.training_mae
        mae_degraded = mae_ratio > self.regime_change_threshold

        if rmse_degraded:
            excess = (self.state.rmse_ratio - self.regime_change_threshold)
            confidence = min(1.0, excess / self.regime_change_threshold)

            if mae_degraded:
                confidence = min(1.0, confidence * 1.2)

            self.state.regime_change_confidence = confidence

            if confidence >= 0.5:
                if not self.state.is_regime_changed:
                    self.state.regime_change_detected_at = datetime.now().isoformat()
                self.state.is_regime_changed = True
            else:
                self.state.is_regime_changed = False
        else:
            self.state.is_regime_changed = False
            self.state.regime_change_confidence = 0.0
            self.state.regime_change_detected_at = None

        self._update_recommendation()

    def _update_recommendation(self):
        """Update recommended prediction method based on regime state."""
        if not self.state.is_regime_changed:
            self.state.recommended_method = "ensemble"
            self.state.fallback_reason = None

        elif self.state.regime_change_confidence < 0.7:
            self.state.recommended_method = "conditional"
            self.state.fallback_reason = (
                f"Rolling RMSE {self.state.rmse_ratio:.2f}x training baseline"
            )

        else:
            self.state.recommended_method = "baseline"
            self.state.fallback_reason = (
                f"Rolling RMSE {self.state.rmse_ratio:.2f}x training baseline "
                f"(confidence: {self.state.regime_change_confidence:.1%})"
            )

    def get_status(self) -> Dict:
        """Get current regime status with recommendations."""
        return {
            'ticker': self.state.ticker,
            'days_ahead': self.state.days_ahead,
            'is_regime_changed': self.state.is_regime_changed,
            'confidence': self.state.regime_change_confidence,
            'recommended_method': self.state.recommended_method,
            'rmse_ratio': self.state.rmse_ratio,
            'rolling_rmse': self.state.rolling_rmse,
            'training_rmse': self.state.training_rmse,
            'n_samples': len(self.state.recent_errors),
            'fallback_reason': self.state.fallback_reason,
            'last_updated': self.state.last_updated,
        }

    def reset(self):
        """Reset regime state (e.g., after model retraining)."""
        self.state.recent_errors = []
        self.state.rolling_rmse = 0.0
        self.state.rolling_mae = 0.0
        self.state.is_regime_changed = False
        self.state.regime_change_confidence = 0.0
        self.state.rmse_ratio = 1.0
        self.state.regime_change_detected_at = None
        self.state.recommended_method = "ensemble"
        self.state.fallback_reason = None
        self.state.last_updated = datetime.now().isoformat()

        if self.cache_dir:
            self._save_state()

    def _get_cache_path(self) -> Path:
        return self.cache_dir / f"regime_{self.ticker}_{self.days_ahead}dte.json"

    def _save_state(self):
        if not self.cache_dir:
            return

        self.cache_dir.mkdir(parents=True, exist_ok=True)
        cache_file = self._get_cache_path()

        with open(cache_file, 'w') as f:
            json.dump(self.state.to_dict(), f, indent=2)

    def _load_state(self) -> bool:
        if not self.cache_dir:
            return False

        cache_file = self._get_cache_path()
        if not cache_file.exists():
            return False

        try:
            with open(cache_file, 'r') as f:
                data = json.load(f)

            self.state.recent_errors = data.get('recent_errors', [])
            self.state.rolling_rmse = data.get('rolling_rmse', 0.0)
            self.state.rolling_mae = data.get('rolling_mae', 0.0)
            self.state.is_regime_changed = data.get('is_regime_changed', False)
            self.state.regime_change_confidence = data.get('regime_change_confidence', 0.0)
            self.state.rmse_ratio = data.get('rmse_ratio', 1.0)
            self.state.recommended_method = data.get('recommended_method', 'ensemble')
            self.state.fallback_reason = data.get('fallback_reason')
            self.state.last_updated = data.get('last_updated', '')
            self.state.regime_change_detected_at = data.get('regime_change_detected_at')

            return True

        except Exception as e:
            print(f"Warning: Could not load regime state: {e}")
            return False

    @classmethod
    def create_for_model(
        cls,
        ticker: str,
        days_ahead: int,
        model_metadata: Dict,
        cache_dir: Optional[Path] = None,
    ) -> 'RegimeDetector':
        training_rmse = model_metadata.get('validation_rmse', 3.0)
        training_mae = model_metadata.get('validation_mae', 2.0)

        return cls(
            ticker=ticker,
            days_ahead=days_ahead,
            training_rmse=training_rmse,
            training_mae=training_mae,
            cache_dir=cache_dir,
        )
