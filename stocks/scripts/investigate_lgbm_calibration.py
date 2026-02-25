#!/usr/bin/env python3
"""
Investigate why LightGBM model was producing overconfident predictions.
Checks quantile calibration on training and test data.
"""

import os
import sys
import pickle
import numpy as np
import pandas as pd
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

def load_cached_model(ticker="NDX"):
    """Load the most recent cached LightGBM model"""
    cache_dir = project_root / "models" / "cache"

    # Find most recent model file
    model_files = list(cache_dir.glob(f"lgbm_close_model_{ticker}_*.pkl"))
    if not model_files:
        print(f"❌ No cached model found for {ticker}")
        return None

    latest_model = max(model_files, key=os.path.getmtime)
    print(f"Loading model: {latest_model.name}")

    with open(latest_model, 'rb') as f:
        model_data = pickle.load(f)

    return model_data

def check_quantile_calibration(model_data):
    """Check if predicted quantiles match actual coverage"""

    if not model_data or 'training_data' not in model_data:
        print("⚠️  No training data available in cached model")
        return

    train_features = model_data['training_data']['features']
    train_targets = model_data['training_data']['targets']

    print(f"\n{'='*80}")
    print("QUANTILE CALIBRATION CHECK")
    print(f"{'='*80}\n")
    print(f"Training samples: {len(train_targets)}")

    # Get model predictions on training data
    predictor = model_data.get('predictor')
    if not predictor:
        print("⚠️  No predictor object in cached model")
        return

    # Check if we can predict
    if not hasattr(predictor, 'predict_quantiles'):
        print("⚠️  Predictor doesn't have predict_quantiles method")
        return

    print("\nPredicting on training data...")

    # Get predictions for each sample
    predictions = []
    for idx in range(len(train_features)):
        features = train_features.iloc[[idx]]
        try:
            pred = predictor.predict_quantiles(features)
            predictions.append(pred)
        except Exception as e:
            print(f"Error predicting sample {idx}: {e}")
            continue

    if not predictions:
        print("❌ Could not generate predictions")
        return

    # Extract quantile predictions
    p10_preds = np.array([p['p10'] for p in predictions])
    p50_preds = np.array([p['p50'] for p in predictions])
    p90_preds = np.array([p['p90'] for p in predictions])

    # Calculate actual coverage
    p10_coverage = (train_targets < p10_preds).mean() * 100
    p50_coverage = (train_targets < p50_preds).mean() * 100
    p90_coverage = (train_targets < p90_preds).mean() * 100

    print("\n" + "="*80)
    print("QUANTILE COVERAGE ON TRAINING DATA")
    print("="*80)
    print(f"P10: {p10_coverage:.1f}% of actuals below prediction (target: 10%)")
    print(f"P50: {p50_coverage:.1f}% of actuals below prediction (target: 50%)")
    print(f"P90: {p90_coverage:.1f}% of actuals below prediction (target: 90%)")

    # Assess calibration
    print("\n" + "="*80)
    print("CALIBRATION ASSESSMENT")
    print("="*80)

    p10_error = abs(p10_coverage - 10.0)
    p50_error = abs(p50_coverage - 50.0)
    p90_error = abs(p90_coverage - 90.0)

    print(f"\nP10 error: {p10_error:.1f}pp {'✅' if p10_error < 5 else '⚠️' if p10_error < 10 else '❌'}")
    print(f"P50 error: {p50_error:.1f}pp {'✅' if p50_error < 5 else '⚠️' if p50_error < 10 else '❌'}")
    print(f"P90 error: {p90_error:.1f}pp {'✅' if p90_error < 5 else '⚠️' if p90_error < 10 else '❌'}")

    # Check prediction spreads
    print("\n" + "="*80)
    print("PREDICTION SPREAD ANALYSIS")
    print("="*80)

    p10_to_p90_spread = np.median(p90_preds - p10_preds)
    actual_p10_to_p90 = np.percentile(train_targets, 90) - np.percentile(train_targets, 10)

    print(f"\nMedian predicted P10-P90 spread: {p10_to_p90_spread:.4f}%")
    print(f"Actual P10-P90 spread in data:   {actual_p10_to_p90:.4f}%")
    print(f"Ratio (actual/predicted):        {actual_p10_to_p90/p10_to_p90_spread:.2f}x")

    if actual_p10_to_p90 / p10_to_p90_spread > 5:
        print("❌ Model predictions are MUCH narrower than reality (5x+)")
    elif actual_p10_to_p90 / p10_to_p90_spread > 2:
        print("⚠️  Model predictions are too narrow (2-5x)")
    elif actual_p10_to_p90 / p10_to_p90_spread > 1.5:
        print("⚠️  Model predictions are slightly narrow (1.5-2x)")
    else:
        print("✅ Model predictions are reasonably calibrated")

    # Root cause analysis
    print("\n" + "="*80)
    print("ROOT CAUSE ANALYSIS")
    print("="*80)

    if p10_error > 20 and p90_error > 20:
        print("\n❌ SEVERE MISCALIBRATION")
        print("   Both quantiles are far from target")
        print("   → LightGBM quantile loss function may not be working properly")
        print("   → Consider using SimpleLGBM approach (median + empirical residuals)")

    elif actual_p10_to_p90 / p10_to_p90_spread > 3:
        print("\n⚠️  OVERCONFIDENT PREDICTIONS")
        print("   Model is predicting much narrower ranges than actual data")
        print("   → Likely overfitting to training data")
        print("   → Need to reduce model complexity or increase regularization")
        print("   → OR use empirical residual distribution instead of quantile regression")

    elif p10_error < 10 and p50_error < 10 and p90_error < 10:
        print("\n✅ MODEL IS WELL-CALIBRATED ON TRAINING DATA")
        print("   Problem is likely in:")
        print("   → Distribution shift between train and test")
        print("   → Band width scaling parameter (LGBM_BAND_WIDTH_SCALE)")
        print("   → Need to test on out-of-sample data")

    else:
        print("\n⚠️  MODERATE CALIBRATION ISSUES")
        print("   Some quantiles are off but not severely")
        print("   → May benefit from quantile-specific calibration factors")
        print("   → Or switch to SimpleLGBM with empirical residuals")

def main():
    print("="*80)
    print("LightGBM CALIBRATION INVESTIGATION")
    print("="*80)

    model_data = load_cached_model("NDX")

    if model_data:
        check_quantile_calibration(model_data)
    else:
        print("\n❌ Could not load model for investigation")
        print("   Run: python scripts/predict_close.py NDX --retrain")
        return 1

    print("\n" + "="*80)
    print("INVESTIGATION COMPLETE")
    print("="*80)
    return 0

if __name__ == "__main__":
    sys.exit(main())
