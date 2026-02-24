#!/usr/bin/env python3
"""
Complete system test showing regime detection and smart fallback.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from scripts.close_predictor.regime_detector import RegimeDetector

def test_scenario(name, detector, error_multiplier, iterations):
    """Test a scenario with specific error magnitude"""
    print(f"\n{'='*80}")
    print(f"SCENARIO: {name}")
    print(f"{'='*80}")
    
    for i in range(iterations):
        predicted = 2.5
        actual = predicted + (i % 7 - 3) * error_multiplier
        detector.add_prediction_error(predicted, actual)
    
    status = detector.get_status()
    
    print(f"Prediction Samples: {status['n_samples']}")
    print(f"Rolling RMSE: {status['rolling_rmse']:.2f}%")
    print(f"Training RMSE: {status['training_rmse']:.2f}%")
    print(f"RMSE Ratio: {status['rmse_ratio']:.2f}x")
    print(f"Regime Changed: {status['is_regime_changed']}")
    print(f"Confidence: {status['confidence']:.1%}")
    print(f"\n→ RECOMMENDED METHOD: {status['recommended_method'].upper()}")
    if status['fallback_reason']:
        print(f"  Reason: {status['fallback_reason']}")
    
    return status

def main():
    print("\n" + "="*80)
    print("REGIME DETECTION & SMART FALLBACK SYSTEM TEST")
    print("="*80)
    
    # Create fresh detector
    cache_dir = Path(__file__).parent / "models" / "regime_cache"
    detector = RegimeDetector(
        ticker="NDX",
        days_ahead=5,
        training_rmse=1.54,  # From actual NDX model
        training_mae=1.2,
        cache_dir=cache_dir,
    )
    detector.reset()
    
    # Scenario 1: Normal regime (small errors)
    test_scenario(
        "Normal Market - Models Performing Well",
        detector,
        error_multiplier=0.3,  # Small errors
        iterations=15
    )
    
    # Scenario 2: Moderate degradation
    test_scenario(
        "Moderate Degradation - Some Drift Detected",
        detector,
        error_multiplier=1.2,  # Medium errors
        iterations=20
    )
    
    # Scenario 3: Severe regime change
    test_scenario(
        "Severe Regime Change - Market Conditions Changed",
        detector,
        error_multiplier=3.5,  # Large errors
        iterations=25
    )
    
    # Summary
    print(f"\n{'='*80}")
    print("SYSTEM BEHAVIOR SUMMARY")
    print(f"{'='*80}")
    print("""
The system automatically adapts based on prediction accuracy:

1. NORMAL REGIME (RMSE < 1.5x baseline)
   → Use ENSEMBLE for best predictions
   ✓ Models are performing as expected
   ✓ High confidence in ML predictions

2. MODERATE DRIFT (RMSE 1.5-2.5x baseline)
   → Use CONDITIONAL for safety
   ⚠️  Some degradation detected
   ⚠️  ML models may be getting stale

3. SEVERE CHANGE (RMSE > 2.5x baseline)
   → Use BASELINE for reliability
   🔴 Major regime change detected
   🔴 ML models no longer reliable
   
This ensures graceful degradation without manual intervention!
""")

if __name__ == '__main__':
    main()
