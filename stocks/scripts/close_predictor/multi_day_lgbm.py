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
    ) -> Dict[str, UnifiedBand]:
        """Predict full distribution using model + noise.

        Uses model's point prediction as center, adds noise based on validation RMSE
        to generate a distribution, then compute percentile bands.

        Args:
            context: Current market context
            current_price: Current price
            n_simulations: Number of Monte Carlo samples
            add_noise_std: Noise std dev to add (% return)

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

    def __init__(self):
        """Initialize ensemble."""
        self.models: Dict[int, MultiDayLGBMPredictor] = {}

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
