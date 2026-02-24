#!/usr/bin/env python3
"""
LightGBM ensemble model for multi-day ahead predictions.

Trains separate models for each DTE (1-20) to predict N-day forward returns
based on market context features.
"""

import numpy as np
import pandas as pd
import lightgbm as lgb
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import pickle
from datetime import datetime, timedelta
from collections import deque

from scripts.close_predictor.multi_day_features import MarketContext
from scripts.close_predictor.models import UnifiedBand


class MultiDayLGBMPredictor:
    """LightGBM model for multi-day return prediction."""

    def __init__(self, days_ahead: int):
        """Initialize predictor for specific DTE.

        Args:
            days_ahead: Number of trading days ahead to predict
        """
        self.days_ahead = days_ahead
        self.model: Optional[lgb.Booster] = None
        self.feature_names: List[str] = []
        self.training_stats: Dict = {}

    def train(
        self,
        contexts: List[MarketContext],
        forward_returns: List[float],
        n_estimators: int = 100,
        learning_rate: float = 0.05,
        max_depth: int = 5,
        min_child_samples: int = 20,
    ) -> Dict:
        """Train LightGBM model to predict N-day returns.

        Args:
            contexts: List of historical market contexts
            forward_returns: List of N-day forward returns (parallel to contexts)
            n_estimators: Number of boosting rounds
            learning_rate: Learning rate
            max_depth: Max tree depth
            min_child_samples: Min samples per leaf

        Returns:
            Training statistics dict
        """
        # Convert contexts to feature matrix
        X = []
        for ctx in contexts:
            X.append(ctx.to_dict())

        df = pd.DataFrame(X)
        self.feature_names = list(df.columns)

        y = np.array(forward_returns)

        # Train/validation split (80/20)
        split_idx = int(len(df) * 0.8)
        X_train, X_val = df.iloc[:split_idx], df.iloc[split_idx:]
        y_train, y_val = y[:split_idx], y[split_idx:]

        # Create LightGBM datasets
        train_data = lgb.Dataset(X_train, label=y_train, free_raw_data=False)
        val_data = lgb.Dataset(X_val, label=y_val, reference=train_data, free_raw_data=False)

        # Training parameters
        params = {
            'objective': 'regression',
            'metric': 'rmse',
            'boosting_type': 'gbdt',
            'num_leaves': 2 ** max_depth,
            'max_depth': max_depth,
            'learning_rate': learning_rate,
            'min_child_samples': min_child_samples,
            'feature_fraction': 0.8,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'verbose': -1,
        }

        # Train model
        self.model = lgb.train(
            params,
            train_data,
            num_boost_round=n_estimators,
            valid_sets=[train_data, val_data],
            valid_names=['train', 'valid'],
        )

        # Compute training statistics
        y_pred_train = self.model.predict(X_train)
        y_pred_val = self.model.predict(X_val)

        self.training_stats = {
            'days_ahead': self.days_ahead,
            'n_samples': len(df),
            'n_features': len(self.feature_names),
            'train_rmse': np.sqrt(np.mean((y_train - y_pred_train) ** 2)),
            'val_rmse': np.sqrt(np.mean((y_val - y_pred_val) ** 2)),
            'train_mae': np.mean(np.abs(y_train - y_pred_train)),
            'val_mae': np.mean(np.abs(y_val - y_pred_val)),
            'feature_importance': dict(zip(
                self.feature_names,
                self.model.feature_importance(importance_type='gain')
            )),
        }

        return self.training_stats

    def predict(self, context: MarketContext) -> float:
        """Predict N-day forward return for given market context.

        Args:
            context: Current market context

        Returns:
            Predicted return % (e.g., 2.5 for +2.5%)
        """
        if self.model is None:
            return 0.0

        features = context.to_dict()
        X = pd.DataFrame([features])[self.feature_names]
        return float(self.model.predict(X)[0])

    def predict_distribution(
        self,
        context: MarketContext,
        current_price: float,
        n_simulations: int = 1000,
        add_noise_std: float = 2.0,
        effective_days_ahead: Optional[float] = None,
        intraday_vol_factor: float = 1.0,
    ) -> Dict[str, UnifiedBand]:
        """Predict full distribution using model + noise.

        Uses model's point prediction as center, adds noise based on validation RMSE
        to generate a distribution, then compute percentile bands.

        Args:
            context: Current market context
            current_price: Current price
            n_simulations: Number of Monte Carlo samples
            add_noise_std: Noise std dev to add (% return)
            effective_days_ahead: Adjusted days ahead accounting for time decay (default: self.days_ahead)
            intraday_vol_factor: Scaling factor for intraday volatility (default: 1.0)

        Returns:
            Dict of percentile bands
        """
        if self.model is None:
            # Return default bands centered on current price
            return {}

        # Get point prediction
        mean_return = self.predict(context)

        # Add noise to create distribution
        # Noise std = validation RMSE + additional uncertainty
        val_rmse = self.training_stats.get('val_rmse', 3.0)
        total_std = np.sqrt(val_rmse ** 2 + add_noise_std ** 2)

        # Generate simulated returns
        simulated_returns = np.random.normal(mean_return, total_std, n_simulations)

        # Apply time decay and intraday volatility scaling
        if effective_days_ahead is not None and self.days_ahead > 0 and effective_days_ahead != self.days_ahead:
            time_decay_factor = effective_days_ahead / self.days_ahead
            simulated_returns = simulated_returns * time_decay_factor

        if intraday_vol_factor != 1.0:
            simulated_returns = simulated_returns * intraday_vol_factor

        # Build percentile bands
        band_defs = {
            'P95':  (2.5, 97.5),
            'P97':  (1.5, 98.5),
            'P98':  (1.0, 99.0),
            'P99':  (0.5, 99.5),
            'P100': (0.0, 100.0),
        }

        bands = {}
        for name, (lo_p, hi_p) in band_defs.items():
            lo_pct = np.percentile(simulated_returns, lo_p) / 100.0
            hi_pct = np.percentile(simulated_returns, hi_p) / 100.0
            lo_price = current_price * (1 + lo_pct)
            hi_price = current_price * (1 + hi_pct)
            width_pts = hi_price - lo_price
            width_pct = (hi_price - lo_price) / current_price * 100.0 if current_price else 0.0

            bands[name] = UnifiedBand(
                name=name,
                lo_price=lo_price,
                hi_price=hi_price,
                lo_pct=lo_pct * 100.0,
                hi_pct=hi_pct * 100.0,
                width_pts=width_pts,
                width_pct=width_pct,
                source="lgbm_ensemble",
            )

        return bands

    def get_prediction_confidence(
        self,
        context: MarketContext,
        recent_errors: Optional[List[float]] = None,
        feature_drift_score: Optional[float] = None,
    ) -> float:
        """Calculate prediction confidence score (0.0-1.0).

        Combines multiple factors:
        - Feature similarity to training data
        - Recent prediction errors
        - Feature drift from training distribution
        - VIX regime match

        Args:
            context: Current market context
            recent_errors: List of recent prediction errors (optional)
            feature_drift_score: Pre-computed feature drift score (optional)

        Returns:
            Confidence score from 0.0 (no confidence) to 1.0 (high confidence)
        """
        if self.model is None:
            return 0.0

        confidence_components = []

        # Component 1: Feature drift (if available)
        if feature_drift_score is not None:
            # Lower drift = higher confidence
            # drift of 0.0 = 1.0 confidence, drift of 1.0 = 0.0 confidence
            drift_confidence = max(0.0, 1.0 - feature_drift_score)
            confidence_components.append(drift_confidence)

        # Component 2: Recent error trend (if available)
        if recent_errors and len(recent_errors) > 0:
            # Compare recent errors to validation RMSE
            val_rmse = self.training_stats.get('val_rmse', 3.0)
            recent_rmse = np.sqrt(np.mean(np.array(recent_errors) ** 2))

            # If recent errors are within 150% of validation RMSE, high confidence
            error_ratio = recent_rmse / val_rmse if val_rmse > 0 else 2.0
            error_confidence = max(0.0, min(1.0, 1.5 - error_ratio))
            confidence_components.append(error_confidence)

        # Component 3: VIX regime appropriateness
        # Models trained in one VIX regime may not work well in another
        vix_level = getattr(context, 'vix', None)
        if vix_level is not None:
            # Penalize extreme VIX (< 10 or > 30)
            if vix_level < 10:
                vix_confidence = 0.6  # Low VIX can be unpredictable
            elif vix_level > 30:
                vix_confidence = 0.5  # High VIX = high uncertainty
            else:
                # Normal range (10-30), higher confidence
                vix_confidence = 1.0
            confidence_components.append(vix_confidence)

        # Combine components (average)
        if confidence_components:
            return float(np.mean(confidence_components))
        else:
            return 0.7  # Default moderate confidence if no components available

    def monitor_feature_drift(
        self,
        current_context: MarketContext,
        training_contexts: Optional[List[MarketContext]] = None,
    ) -> float:
        """Monitor feature drift from training distribution.

        Compares current feature values to training distribution to detect
        if we're predicting on data very different from training data.

        Args:
            current_context: Current market context
            training_contexts: Historical training contexts (optional)

        Returns:
            Drift score from 0.0 (no drift) to 1.0+ (significant drift)
        """
        if self.model is None:
            return 0.0

        current_features = current_context.to_dict()

        # If we don't have training contexts, use a simple heuristic
        # based on feature importance and value ranges
        if training_contexts is None or len(training_contexts) == 0:
            # Check for extreme values in key features
            drift_score = 0.0

            # Check VIX (expect 5-40 range)
            vix = current_features.get('vix', 15.0)
            if vix < 8 or vix > 35:
                drift_score += 0.3

            # Check volume ratio (expect 0.5-2.0 range)
            vol_ratio = current_features.get('volume_ratio', 1.0)
            if vol_ratio < 0.3 or vol_ratio > 3.0:
                drift_score += 0.2

            return min(1.0, drift_score)

        # Calculate drift using training data distribution
        training_df = pd.DataFrame([ctx.to_dict() for ctx in training_contexts])

        drift_scores = []
        for feature_name in self.feature_names:
            if feature_name not in current_features:
                continue

            current_val = current_features[feature_name]
            if feature_name in training_df.columns:
                train_mean = training_df[feature_name].mean()
                train_std = training_df[feature_name].std()

                if train_std > 0:
                    # Z-score: how many standard deviations from training mean
                    z_score = abs(current_val - train_mean) / train_std
                    # Convert to 0-1 score (z > 3 = significant drift)
                    feature_drift = min(1.0, z_score / 3.0)
                    drift_scores.append(feature_drift)

        if drift_scores:
            # Use max drift (most drifted feature determines overall drift)
            return float(np.max(drift_scores))
        else:
            return 0.0

    def save(self, filepath: Path):
        """Save model to disk."""
        model_data = {
            'days_ahead': self.days_ahead,
            'feature_names': self.feature_names,
            'training_stats': self.training_stats,
            'model': self.model,
        }
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)

    @classmethod
    def load(cls, filepath: Path) -> 'MultiDayLGBMPredictor':
        """Load model from disk."""
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)

        predictor = cls(days_ahead=model_data['days_ahead'])
        predictor.feature_names = model_data['feature_names']
        predictor.training_stats = model_data['training_stats']
        predictor.model = model_data['model']

        return predictor


class MultiDayEnsemble:
    """Ensemble of LightGBM models for DTE 1-20."""

    def __init__(self, error_tracking_window: int = 30):
        """Initialize ensemble.

        Args:
            error_tracking_window: Number of recent predictions to track for error monitoring
        """
        self.models: Dict[int, MultiDayLGBMPredictor] = {}
        self.error_tracking_window = error_tracking_window
        # Track recent errors for each DTE: {dte: deque of (actual, predicted) tuples}
        self.recent_errors: Dict[int, deque] = {}
        self.training_date: Optional[datetime] = None

    def train_all(
        self,
        all_dates: List[str],
        contexts_by_date: Dict[str, MarketContext],
        returns_by_dte: Dict[int, Dict[str, float]],
        max_dte: int = 20,
    ) -> Dict[int, Dict]:
        """Train models for all DTEs.

        Args:
            all_dates: List of dates in chronological order
            contexts_by_date: Dict mapping date -> MarketContext
            returns_by_dte: Dict mapping dte -> dict(date -> forward_return)
            max_dte: Maximum DTE to train

        Returns:
            Dict mapping dte -> training stats
        """
        stats = {}

        for dte in range(1, max_dte + 1):
            if dte not in returns_by_dte:
                continue

            # Build training data
            contexts = []
            forward_returns = []

            for date in all_dates:
                if date in contexts_by_date and date in returns_by_dte[dte]:
                    contexts.append(contexts_by_date[date])
                    forward_returns.append(returns_by_dte[dte][date])

            if len(contexts) < 100:
                print(f"⚠️  Skipping {dte}DTE: only {len(contexts)} samples (need 100+)")
                continue

            print(f"Training {dte}DTE model with {len(contexts)} samples...")
            model = MultiDayLGBMPredictor(days_ahead=dte)
            dte_stats = model.train(contexts, forward_returns)
            self.models[dte] = model
            stats[dte] = dte_stats

            print(f"  ✓ {dte}DTE: train_rmse={dte_stats['train_rmse']:.2f}%, "
                  f"val_rmse={dte_stats['val_rmse']:.2f}%")

        return stats

    def predict(self, dte: int, context: MarketContext, current_price: float) -> Dict[str, UnifiedBand]:
        """Predict using ensemble model for specific DTE.

        Args:
            dte: Days ahead
            context: Current market context
            current_price: Current price

        Returns:
            Dict of percentile bands
        """
        if dte not in self.models:
            return {}

        return self.models[dte].predict_distribution(context, current_price)

    def save_all(self, output_dir: Path):
        """Save all models to directory."""
        output_dir.mkdir(parents=True, exist_ok=True)

        for dte, model in self.models.items():
            filepath = output_dir / f"lgbm_{dte}dte.pkl"
            model.save(filepath)
            print(f"Saved {dte}DTE model → {filepath}")

    def load_all(self, input_dir: Path):
        """Load all models from directory."""
        for filepath in sorted(input_dir.glob("lgbm_*dte.pkl")):
            model = MultiDayLGBMPredictor.load(filepath)
            self.models[model.days_ahead] = model
            print(f"Loaded {model.days_ahead}DTE model from {filepath}")

    def record_prediction_outcome(
        self,
        dte: int,
        predicted_return: float,
        actual_return: float,
    ):
        """Record prediction outcome for error tracking.

        Args:
            dte: Days ahead that was predicted
            predicted_return: Model's predicted return %
            actual_return: Actual realized return %
        """
        if dte not in self.recent_errors:
            self.recent_errors[dte] = deque(maxlen=self.error_tracking_window)

        error = abs(actual_return - predicted_return)
        self.recent_errors[dte].append(error)

    def get_recent_rmse(self, dte: int) -> Optional[float]:
        """Get RMSE from recent predictions for a specific DTE.

        Args:
            dte: Days ahead

        Returns:
            Recent RMSE or None if insufficient data
        """
        if dte not in self.recent_errors or len(self.recent_errors[dte]) < 5:
            return None

        errors = list(self.recent_errors[dte])
        return float(np.sqrt(np.mean(np.array(errors) ** 2)))

    def needs_retraining(
        self,
        dte: int,
        error_threshold_multiplier: float = 1.5,
        days_since_training: Optional[int] = None,
        max_days_without_retraining: int = 30,
    ) -> Tuple[bool, str]:
        """Detect if ensemble needs retraining for specific DTE.

        Triggers retraining if:
        1. Recent RMSE > training validation RMSE * threshold
        2. More than max_days since last training

        Args:
            dte: Days ahead to check
            error_threshold_multiplier: Multiplier for validation RMSE threshold
            days_since_training: Days since last training (optional)
            max_days_without_retraining: Max days before forced retraining

        Returns:
            (needs_retraining, reason)
        """
        if dte not in self.models:
            return True, f"No model exists for {dte}DTE"

        model = self.models[dte]

        # Check 1: Error spike detection
        recent_rmse = self.get_recent_rmse(dte)
        if recent_rmse is not None:
            val_rmse = model.training_stats.get('val_rmse', 0.0)
            if val_rmse > 0:
                error_ratio = recent_rmse / val_rmse
                if error_ratio > error_threshold_multiplier:
                    return True, (
                        f"Recent RMSE ({recent_rmse:.2f}%) is {error_ratio:.1f}x "
                        f"validation RMSE ({val_rmse:.2f}%)"
                    )

        # Check 2: Time-based retraining
        if days_since_training is not None and days_since_training > max_days_without_retraining:
            return True, f"Last trained {days_since_training} days ago (max: {max_days_without_retraining})"

        return False, "Model performance is stable"

    def get_prediction_confidence(
        self,
        dte: int,
        context: MarketContext,
        feature_drift_score: Optional[float] = None,
    ) -> float:
        """Get prediction confidence for specific DTE.

        Args:
            dte: Days ahead
            context: Current market context
            feature_drift_score: Pre-computed drift score (optional)

        Returns:
            Confidence score 0.0-1.0
        """
        if dte not in self.models:
            return 0.0

        model = self.models[dte]

        # Get recent errors for this DTE
        recent_errors_list = None
        if dte in self.recent_errors and len(self.recent_errors[dte]) > 0:
            recent_errors_list = list(self.recent_errors[dte])

        return model.get_prediction_confidence(
            context=context,
            recent_errors=recent_errors_list,
            feature_drift_score=feature_drift_score,
        )

    def get_ensemble_health_report(self) -> Dict:
        """Generate health report for all ensemble models.

        Returns:
            Dict with health metrics for each DTE
        """
        report = {}

        for dte in sorted(self.models.keys()):
            model = self.models[dte]
            recent_rmse = self.get_recent_rmse(dte)
            val_rmse = model.training_stats.get('val_rmse', 0.0)

            error_ratio = None
            if recent_rmse is not None and val_rmse > 0:
                error_ratio = recent_rmse / val_rmse

            needs_retrain, reason = self.needs_retraining(dte)

            report[dte] = {
                'validation_rmse': val_rmse,
                'recent_rmse': recent_rmse,
                'error_ratio': error_ratio,
                'needs_retraining': needs_retrain,
                'reason': reason,
                'n_samples_tracked': len(self.recent_errors.get(dte, [])),
            }

        return report
