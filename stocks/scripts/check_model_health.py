#!/usr/bin/env python3
"""
Model Health Monitoring Script

Checks if multi-day prediction models need retraining based on:
1. Validation RMSE degradation
2. Model age (days since last training)
3. VIX regime changes

Usage:
    python scripts/check_model_health.py --ticker NDX
    python scripts/check_model_health.py --ticker SPX --alert-threshold 3.5
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import json
from datetime import datetime, timedelta
import pandas as pd
from typing import Dict, Optional


def load_model_metadata(ticker: str, model_dir: Path = None) -> Optional[Dict]:
    """Load metadata from latest production models."""
    if model_dir is None:
        model_dir = Path(__file__).parent.parent / "models" / "production" / f"{ticker}_latest"

    metadata_file = model_dir / "training_metadata.json"

    if not metadata_file.exists():
        print(f"⚠️  No model metadata found at: {metadata_file}")
        return None

    with open(metadata_file, 'r') as f:
        return json.load(f)


def check_model_age(metadata: Dict, max_age_days: int = 30) -> Dict:
    """Check if models are too old."""
    timestamp_str = metadata.get('timestamp', '')

    try:
        # Parse timestamp (format: YYYYMMDD_HHMMSS)
        train_date = datetime.strptime(timestamp_str, '%Y%m%d_%H%M%S')
        age_days = (datetime.now() - train_date).days

        status = "✓ OK" if age_days < max_age_days else "⚠️ RETRAIN NEEDED"

        return {
            'check': 'Model Age',
            'train_date': train_date.strftime('%Y-%m-%d'),
            'age_days': age_days,
            'threshold': max_age_days,
            'status': status,
            'needs_retrain': age_days >= max_age_days
        }
    except Exception as e:
        return {
            'check': 'Model Age',
            'status': f"⚠️ ERROR: {e}",
            'needs_retrain': True
        }


def check_validation_rmse(metadata: Dict, alert_threshold: float = 3.5) -> Dict:
    """Check if validation RMSE is too high."""
    avg_rmse = metadata.get('avg_val_rmse', 0)
    max_rmse = metadata.get('max_val_rmse', 0)

    # Check if any DTE has high error
    high_error_dtes = []
    validation_stats = metadata.get('validation_stats', {})

    for dte, stats in validation_stats.items():
        val_rmse = stats.get('val_rmse', 0)
        if val_rmse > alert_threshold:
            high_error_dtes.append((int(dte), val_rmse))

    needs_retrain = max_rmse > alert_threshold or len(high_error_dtes) > 5

    if max_rmse < 2.5:
        status = "✓ Excellent"
    elif max_rmse < 3.5:
        status = "✓ Good"
    elif max_rmse < 4.5:
        status = "⚠️ Acceptable"
    else:
        status = "⚠️ RETRAIN NEEDED"

    return {
        'check': 'Validation RMSE',
        'avg_rmse': round(avg_rmse, 2),
        'max_rmse': round(max_rmse, 2),
        'threshold': alert_threshold,
        'high_error_count': len(high_error_dtes),
        'high_error_dtes': high_error_dtes[:5],  # Show up to 5
        'status': status,
        'needs_retrain': needs_retrain
    }


def check_vix_regime_change(ticker: str) -> Dict:
    """Check if VIX has changed significantly since training."""
    try:
        from scripts.csv_prediction_backtest import get_available_dates, load_csv_data

        # Get recent VIX data
        recent_dates = get_available_dates('VIX', 60)[-60:]
        vix_values = []

        for date_str in recent_dates:
            df = load_csv_data('VIX', date_str)
            if df is not None and not df.empty:
                vix_values.append(df.iloc[-1]['close'])

        if len(vix_values) < 30:
            return {
                'check': 'VIX Regime',
                'status': '⚠️ Insufficient data',
                'needs_retrain': False
            }

        # Compare recent 10-day avg to 30-day avg
        recent_vix = sum(vix_values[-10:]) / len(vix_values[-10:])
        baseline_vix = sum(vix_values[-30:]) / len(vix_values[-30:])

        vix_change_pct = ((recent_vix - baseline_vix) / baseline_vix) * 100

        needs_retrain = abs(vix_change_pct) > 50  # 50% change triggers retrain

        if abs(vix_change_pct) < 20:
            status = "✓ Stable"
        elif abs(vix_change_pct) < 50:
            status = "⚠️ Moderate change"
        else:
            status = "⚠️ REGIME CHANGE"

        return {
            'check': 'VIX Regime',
            'current_vix_10d': round(recent_vix, 2),
            'baseline_vix_30d': round(baseline_vix, 2),
            'change_pct': round(vix_change_pct, 1),
            'threshold': 50,
            'status': status,
            'needs_retrain': needs_retrain
        }

    except Exception as e:
        return {
            'check': 'VIX Regime',
            'status': f'⚠️ ERROR: {e}',
            'needs_retrain': False
        }


def main():
    parser = argparse.ArgumentParser(description='Check multi-day model health')
    parser.add_argument('--ticker', type=str, default='NDX', help='Ticker symbol')
    parser.add_argument('--max-age-days', type=int, default=30, help='Max days since training')
    parser.add_argument('--alert-threshold', type=float, default=3.5, help='RMSE alert threshold (percent)')
    parser.add_argument('--model-dir', type=Path, help='Custom model directory')
    args = parser.parse_args()

    print(f"\n{'='*80}")
    print(f"MODEL HEALTH CHECK: {args.ticker}")
    print(f"{'='*80}\n")

    # Load model metadata
    metadata = load_model_metadata(args.ticker, args.model_dir)

    if metadata is None:
        print("❌ No production models found - need to run initial training")
        print(f"\nRun: python scripts/retrain_multi_day_models.py --ticker {args.ticker}")
        return 1

    # Run checks
    checks = [
        check_model_age(metadata, args.max_age_days),
        check_validation_rmse(metadata, args.alert_threshold),
        check_vix_regime_change(args.ticker),
    ]

    # Display results
    needs_retrain = False

    for check in checks:
        print(f"{'─'*80}")
        print(f"\n{check['check'].upper()}")
        print(f"{'─'*40}")

        for key, value in check.items():
            if key not in ['check', 'needs_retrain']:
                if isinstance(value, list):
                    print(f"  {key}: {value if value else 'None'}")
                else:
                    print(f"  {key}: {value}")

        print(f"\n  Status: {check['status']}")

        if check.get('needs_retrain', False):
            needs_retrain = True

    # Overall recommendation
    print(f"\n{'='*80}")
    print("RECOMMENDATION")
    print(f"{'='*80}\n")

    if needs_retrain:
        print("⚠️  RETRAINING RECOMMENDED")
        print(f"\nRun: python scripts/retrain_multi_day_models.py --ticker {args.ticker}")
    else:
        print("✓ Models are healthy - no immediate action needed")
        print(f"\nNext scheduled retrain: {metadata.get('train_period', 'Unknown').split(' to ')[1] if 'train_period' in metadata else 'N/A'}")

    print(f"\n{'='*80}\n")

    return 1 if needs_retrain else 0


if __name__ == '__main__':
    sys.exit(main())
