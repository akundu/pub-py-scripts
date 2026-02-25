#!/usr/bin/env python3
"""
Diagnostic analysis for LightGBM predictor performance.

Analyzes why LightGBM has tighter bands but lower hit rate compared to Statistical.
"""

import sys
import numpy as np
import pandas as pd

from scripts.csv_prediction_backtest import (
    get_available_dates,
    build_training_data,
)
from scripts.strategy_utils.close_predictor import (
    LGBMClosePredictor,
    StatisticalClosePredictor,
)
from scripts.close_predictor.models import STAT_FEATURE_CONFIG


def main():
    """Diagnostic analysis."""
    print("="*80)
    print("LightGBM Predictor Diagnostic Analysis")
    print("="*80)

    ticker = 'NDX'
    dates = get_available_dates(ticker, 300)
    test_date = dates[-1]

    print(f"\nLoading training data for {ticker} (test date: {test_date})...")
    train_df = build_training_data(ticker, test_date, 250)

    print(f"  Training samples: {len(train_df)}")

    # Train both models
    print("\nTraining LightGBM...")
    lgbm = LGBMClosePredictor(
        n_estimators=150,
        learning_rate=0.05,
        max_depth=6,
        use_fallback=False,
    )
    lgbm.fit(train_df)

    print("\nTraining Statistical...")
    stat = StatisticalClosePredictor(min_samples=5, **STAT_FEATURE_CONFIG)
    stat.fit(train_df)

    # Analyze training data target distribution
    print("\n" + "="*80)
    print("Training Data Analysis")
    print("="*80)

    target = ((train_df['day_close'] - train_df['hour_price']) / train_df['hour_price']).values

    print(f"\nTarget (close_move_pct) distribution:")
    print(f"  Mean:   {np.mean(target)*100:+.3f}%")
    print(f"  Std:    {np.std(target)*100:.3f}%")
    print(f"  Min:    {np.min(target)*100:+.3f}%")
    print(f"  Max:    {np.max(target)*100:+.3f}%")

    percentiles = [5, 10, 25, 50, 75, 90, 95]
    print(f"\n  Percentiles:")
    for p in percentiles:
        val = np.percentile(target, p)
        print(f"    P{p:2d}: {val*100:+.3f}%")

    # LGBM validation metrics
    print("\n" + "="*80)
    print("LightGBM Model Analysis")
    print("="*80)

    print(f"\nValidation MAE:")
    for quantile, mae in lgbm.validation_mae.items():
        print(f"  {quantile.upper()}: {mae:.4f} ({mae*100:.2f}%)")

    print(f"\nFeature Importance (Top 10):")
    top_features = sorted(lgbm.feature_importance.items(), key=lambda x: x[1], reverse=True)[:10]
    for i, (feat, imp) in enumerate(top_features, 1):
        print(f"  {i:2d}. {feat:25s}: {imp:7.1f}")

    # Check for overfitting
    print("\n" + "="*80)
    print("Overfitting Check")
    print("="*80)

    # Split data
    split_idx = int(len(train_df) * 0.8)
    train_subset = train_df[:split_idx]
    val_subset = train_df[split_idx:]

    print(f"\nTraining on first 80% ({len(train_subset)} samples)...")
    print(f"Validating on last 20% ({len(val_subset)} samples)...")

    # Train on subset
    lgbm_subset = LGBMClosePredictor(
        n_estimators=150,
        learning_rate=0.05,
        max_depth=6,
        use_fallback=False,
    )
    lgbm_subset.fit(train_subset)

    # Predict on validation set
    from scripts.strategy_utils.close_predictor import LGBM_FEATURE_NAMES

    val_features = lgbm_subset._prepare_features(val_subset)
    val_target = ((val_subset['day_close'] - val_subset['hour_price']) / val_subset['hour_price']).values

    # Get predictions
    val_pred_p10 = lgbm_subset.quantile_models['p10'].predict(val_features)
    val_pred_p50 = lgbm_subset.quantile_models['p50'].predict(val_features)
    val_pred_p90 = lgbm_subset.quantile_models['p90'].predict(val_features)

    # Calculate hit rate
    within_bounds = (val_pred_p10 <= val_target) & (val_target <= val_pred_p90)
    hit_rate = within_bounds.mean() * 100

    print(f"\nValidation Results:")
    print(f"  Hit rate (P10-P90): {hit_rate:.1f}%")
    print(f"  P50 MAE:            {np.mean(np.abs(val_target - val_pred_p50)):.4f} ({np.mean(np.abs(val_target - val_pred_p50))*100:.2f}%)")

    # Band width analysis
    band_widths = val_pred_p90 - val_pred_p10
    print(f"\nBand Width Analysis:")
    print(f"  Mean:   {np.mean(band_widths)*100:.3f}%")
    print(f"  Median: {np.median(band_widths)*100:.3f}%")
    print(f"  Std:    {np.std(band_widths)*100:.3f}%")

    # Compare with empirical percentiles
    empirical_p10 = np.percentile(val_target, 10)
    empirical_p50 = np.percentile(val_target, 50)
    empirical_p90 = np.percentile(val_target, 90)
    empirical_width = empirical_p90 - empirical_p10

    print(f"\nEmpirical vs Predicted Percentiles:")
    print(f"  P10:  empirical={empirical_p10*100:+.3f}%, predicted={np.mean(val_pred_p10)*100:+.3f}%")
    print(f"  P50:  empirical={empirical_p50*100:+.3f}%, predicted={np.mean(val_pred_p50)*100:+.3f}%")
    print(f"  P90:  empirical={empirical_p90*100:+.3f}%, predicted={np.mean(val_pred_p90)*100:+.3f}%")
    print(f"  Width: empirical={empirical_width*100:.3f}%, predicted={np.mean(band_widths)*100:.3f}%")

    # Recommendations
    print("\n" + "="*80)
    print("Recommendations")
    print("="*80)

    if hit_rate < 80:
        print("\n⚠️  Hit rate is low (<80%). Possible issues:")
        print("  1. Bands may be too narrow")
        print("  2. Model may be overfitting to training data")
        print("  3. Volatility scaling may be too aggressive")

        if np.mean(band_widths) < empirical_width * 0.8:
            print("\n  → Bands are narrower than empirical width")
            print("  → Consider increasing regularization:")
            print("      - Increase min_child_samples (current: 20)")
            print("      - Decrease max_depth (current: 6)")
            print("      - Increase subsample rate")
            print("      - Add band width scaling factor (1.2-1.5x)")

    if lgbm.validation_mae['p50'] < 0.005:
        print("\n✓ Validation MAE is good (<0.5%)")

    print("\n" + "="*80)
    print("Analysis Complete")
    print("="*80)

    return 0


if __name__ == '__main__':
    sys.exit(main())
