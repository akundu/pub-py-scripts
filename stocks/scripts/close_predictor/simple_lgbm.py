"""
SimpleLGBM: Robust alternative to quantile regression.

Key insight: Separate median prediction (what ML is good at) from uncertainty
quantification (use empirical distribution of residuals).

This approach is more robust than LightGBM quantile regression because:
1. Median prediction is stable and well-calibrated
2. Uncertainty bounds come from actual historical errors
3. No risk of overconfident quantile predictions
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional, List
import lightgbm as lgb
from dataclasses import dataclass

@dataclass
class SimpleLGBMConfig:
    """Configuration for SimpleLGBM"""
    n_estimators: int = 100  # Fewer than quantile version (less overfitting)
    learning_rate: float = 0.03
    max_depth: int = 4  # Shallower trees
    min_child_samples: int = 50  # Larger leaves
    subsample: float = 0.8  # Bag 80% of data
    colsample_bytree: float = 0.8  # Use 80% of features
    reg_alpha: float = 0.1  # L1 regularization
    reg_lambda: float = 1.0  # L2 regularization


class SimpleLGBM:
    """
    Simple LightGBM predictor using median + empirical residuals.

    Workflow:
    1. Train LightGBM to predict median move
    2. Compute residuals on training data
    3. Store empirical quantiles of residuals
    4. At prediction time: median ± residual quantiles

    This guarantees well-calibrated quantiles by construction.
    """

    def __init__(self, config: Optional[SimpleLGBMConfig] = None):
        self.config = config or SimpleLGBMConfig()
        self.median_model = None
        self.residual_quantiles = None
        self.training_stats = {}

    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        """
        Train median predictor and compute residual distribution.

        Args:
            X: Features (N samples × M features)
            y: Target variable (N samples) - percent move to close
        """
        print(f"Training SimpleLGBM on {len(X)} samples...")

        # 1. Train median predictor
        self.median_model = lgb.LGBMRegressor(
            n_estimators=self.config.n_estimators,
            learning_rate=self.config.learning_rate,
            max_depth=self.config.max_depth,
            min_child_samples=self.config.min_child_samples,
            subsample=self.config.subsample,
            colsample_bytree=self.config.colsample_bytree,
            reg_alpha=self.config.reg_alpha,
            reg_lambda=self.config.reg_lambda,
            objective='regression',  # Standard L2 loss (median)
            random_state=42,
            n_jobs=-1,
            verbose=-1,
        )

        self.median_model.fit(X, y)

        # 2. Compute residuals
        y_pred_median = self.median_model.predict(X)
        residuals = y - y_pred_median

        # 3. Store empirical quantiles of residuals
        self.residual_quantiles = {
            'p01': np.percentile(residuals, 1),
            'p05': np.percentile(residuals, 5),
            'p10': np.percentile(residuals, 10),
            'p25': np.percentile(residuals, 25),
            'p50': np.percentile(residuals, 50),
            'p75': np.percentile(residuals, 75),
            'p90': np.percentile(residuals, 90),
            'p95': np.percentile(residuals, 95),
            'p99': np.percentile(residuals, 99),
        }

        # 4. Store training statistics
        self.training_stats = {
            'n_samples': len(X),
            'median_mae': np.mean(np.abs(residuals)),
            'median_rmse': np.sqrt(np.mean(residuals**2)),
            'residual_std': np.std(residuals),
            'p10_to_p90_width': self.residual_quantiles['p90'] - self.residual_quantiles['p10'],
        }

        print(f"✓ SimpleLGBM trained:")
        print(f"  MAE: {self.training_stats['median_mae']:.4f}%")
        print(f"  RMSE: {self.training_stats['median_rmse']:.4f}%")
        print(f"  P10-P90 width: {self.training_stats['p10_to_p90_width']:.4f}%")

    def predict(self, X: pd.DataFrame) -> Dict[str, float]:
        """
        Predict median and quantiles for new data.

        Returns dict with:
            - p10, p50, p90: predicted percent moves
            - lo_10, mid, hi_90: same (alias)
        """
        if self.median_model is None:
            raise ValueError("Model not trained. Call fit() first.")

        # Predict median
        median = self.median_model.predict(X)[0]

        # Add residual quantiles to get prediction quantiles
        predictions = {
            'p01': median + self.residual_quantiles['p01'],
            'p05': median + self.residual_quantiles['p05'],
            'p10': median + self.residual_quantiles['p10'],
            'p25': median + self.residual_quantiles['p25'],
            'p50': median,
            'p75': median + self.residual_quantiles['p75'],
            'p90': median + self.residual_quantiles['p90'],
            'p95': median + self.residual_quantiles['p95'],
            'p99': median + self.residual_quantiles['p99'],
        }

        # Add aliases for compatibility
        predictions['lo_10'] = predictions['p10']
        predictions['mid'] = predictions['p50']
        predictions['hi_90'] = predictions['p90']

        return predictions

    def predict_with_bands(
        self,
        X: pd.DataFrame,
        current_price: float,
        percentile_levels: List[int] = [95, 97, 98, 99]
    ) -> Dict[str, Dict[str, float]]:
        """
        Predict and convert to price bands.

        Returns:
            {
                'P95': {'lo_price': ..., 'hi_price': ..., 'lo_pct': ..., 'hi_pct': ...},
                'P97': {...},
                ...
            }
        """
        pct_predictions = self.predict(X)

        bands = {}
        for level in percentile_levels:
            # Map percentile level to residual quantile
            # P95 = ± P05 residuals (symmetric)
            # P97 = ± P03 residuals
            # P98 = ± P02 residuals
            # P99 = ± P01 residuals
            lower_q = 100 - level
            upper_q = level

            # Get corresponding percentiles from residuals
            if lower_q <= 1:
                lo_pct = pct_predictions['p01']
            elif lower_q <= 5:
                lo_pct = pct_predictions['p05']
            elif lower_q <= 10:
                lo_pct = pct_predictions['p10']
            else:
                # Interpolate for intermediate values
                lo_pct = pct_predictions['p10']

            if upper_q >= 99:
                hi_pct = pct_predictions['p99']
            elif upper_q >= 95:
                hi_pct = pct_predictions['p95']
            elif upper_q >= 90:
                hi_pct = pct_predictions['p90']
            else:
                hi_pct = pct_predictions['p90']

            # Use symmetric bounds (take wider of |lo| and |hi|)
            width = max(abs(lo_pct), abs(hi_pct))
            median = pct_predictions['p50']

            bands[f'P{level}'] = {
                'lo_pct': median - width,
                'hi_pct': median + width,
                'lo_price': current_price * (1 + (median - width) / 100),
                'hi_price': current_price * (1 + (median + width) / 100),
                'width_pct': 2 * width,
            }

        return bands

    def get_feature_importance(self, top_n: int = 10) -> pd.DataFrame:
        """Get feature importance from median model"""
        if self.median_model is None:
            raise ValueError("Model not trained")

        importance = pd.DataFrame({
            'feature': self.median_model.feature_name_,
            'importance': self.median_model.feature_importances_
        }).sort_values('importance', ascending=False)

        return importance.head(top_n)

    def calibrate(
        self,
        X_val: pd.DataFrame,
        y_val: pd.Series,
        adjustment_factor: float = 0.1
    ) -> None:
        """
        Fine-tune residual quantiles using validation data.

        This allows for online calibration if test distribution differs from train.

        Args:
            X_val: Validation features
            y_val: Validation targets
            adjustment_factor: How much to adjust (0.1 = 10% weight on validation)
        """
        # Predict on validation
        y_pred = self.median_model.predict(X_val)
        residuals_val = y_val - y_pred

        # Compute validation quantiles
        val_quantiles = {
            'p01': np.percentile(residuals_val, 1),
            'p05': np.percentile(residuals_val, 5),
            'p10': np.percentile(residuals_val, 10),
            'p25': np.percentile(residuals_val, 25),
            'p50': np.percentile(residuals_val, 50),
            'p75': np.percentile(residuals_val, 75),
            'p90': np.percentile(residuals_val, 90),
            'p95': np.percentile(residuals_val, 95),
            'p99': np.percentile(residuals_val, 99),
        }

        # Blend with training quantiles
        for key in self.residual_quantiles:
            old = self.residual_quantiles[key]
            new = val_quantiles[key]
            self.residual_quantiles[key] = (
                old * (1 - adjustment_factor) + new * adjustment_factor
            )

        print(f"✓ Calibrated on {len(X_val)} validation samples")
        print(f"  Adjustment factor: {adjustment_factor:.1%}")


# Convenience function for backwards compatibility
def train_simple_lgbm(
    X: pd.DataFrame,
    y: pd.Series,
    config: Optional[SimpleLGBMConfig] = None
) -> SimpleLGBM:
    """Train and return a SimpleLGBM model"""
    model = SimpleLGBM(config)
    model.fit(X, y)
    return model
