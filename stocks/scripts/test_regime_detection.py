#!/usr/bin/env python3
"""
Test script for regime detection system.

Simulates prediction errors to demonstrate regime change detection.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.close_predictor.regime_detector import RegimeDetector

def main():
    print("="*80)
    print("REGIME DETECTION TEST")
    print("="*80)

    # Create detector for NDX 5DTE with training RMSE of 2.0%
    detector = RegimeDetector(
        ticker="NDX",
        days_ahead=5,
        training_rmse=2.0,
        training_mae=1.5,
        cache_dir=Path(__file__).parent.parent / "models" / "regime_cache",
    )

    print("\n1. NORMAL REGIME - Predictions within baseline")
    print("-" * 80)

    # Simulate good predictions (errors around ±2%)
    for i in range(10):
        predicted = 2.5
        actual = predicted + (i % 3 - 1) * 0.5  # Errors of -0.5, 0, +0.5
        status = detector.add_prediction_error(predicted, actual)

    print(f"Rolling RMSE: {status['rolling_rmse']:.2f}%")
    print(f"Training RMSE: {status['training_rmse']:.2f}%")
    print(f"RMSE Ratio: {status['rmse_ratio']:.2f}x")
    print(f"Regime Changed: {status['is_regime_changed']}")
    print(f"Recommended Method: {status['recommended_method']}")

    print("\n2. MODERATE REGIME CHANGE - Predictions getting worse")
    print("-" * 80)

    # Simulate moderate degradation (errors around ±3-4%)
    for i in range(15):
        predicted = 2.5
        actual = predicted + (i % 5 - 2) * 1.5  # Larger errors
        status = detector.add_prediction_error(predicted, actual)

    print(f"Rolling RMSE: {status['rolling_rmse']:.2f}%")
    print(f"Training RMSE: {status['training_rmse']:.2f}%")
    print(f"RMSE Ratio: {status['rmse_ratio']:.2f}x")
    print(f"Regime Changed: {status['is_regime_changed']}")
    print(f"Confidence: {status['confidence']:.1%}")
    print(f"Recommended Method: {status['recommended_method']}")
    print(f"Fallback Reason: {status['fallback_reason']}")

    print("\n3. SEVERE REGIME CHANGE - Predictions far off")
    print("-" * 80)

    # Simulate severe degradation (errors around ±5-7%)
    for i in range(20):
        predicted = 2.5
        actual = predicted + (i % 7 - 3) * 2.0  # Very large errors
        status = detector.add_prediction_error(predicted, actual)

    print(f"Rolling RMSE: {status['rolling_rmse']:.2f}%")
    print(f"Training RMSE: {status['training_rmse']:.2f}%")
    print(f"RMSE Ratio: {status['rmse_ratio']:.2f}x")
    print(f"Regime Changed: {status['is_regime_changed']}")
    print(f"Confidence: {status['confidence']:.1%}")
    print(f"Recommended Method: {status['recommended_method']}")
    print(f"Fallback Reason: {status['fallback_reason']}")

    print("\n4. RECOVERY - Predictions improving")
    print("-" * 80)

    # Simulate recovery (back to good predictions)
    for i in range(25):
        predicted = 2.5
        actual = predicted + (i % 3 - 1) * 0.4  # Small errors again
        status = detector.add_prediction_error(predicted, actual)

    print(f"Rolling RMSE: {status['rolling_rmse']:.2f}%")
    print(f"Training RMSE: {status['training_rmse']:.2f}%")
    print(f"RMSE Ratio: {status['rmse_ratio']:.2f}x")
    print(f"Regime Changed: {status['is_regime_changed']}")
    print(f"Confidence: {status['confidence']:.1%}")
    print(f"Recommended Method: {status['recommended_method']}")
    print(f"Fallback Reason: {status['fallback_reason']}")

    print("\n" + "="*80)
    print("SUMMARY: Method Selection Logic")
    print("="*80)
    print("• RMSE < 1.5x baseline → Use Ensemble (high confidence)")
    print("• RMSE 1.5-2.5x baseline → Use Conditional (moderate confidence)")  
    print("• RMSE > 2.5x baseline → Use Baseline (regime changed)")
    print("\nThis ensures graceful degradation when market conditions change!")


if __name__ == '__main__':
    main()
