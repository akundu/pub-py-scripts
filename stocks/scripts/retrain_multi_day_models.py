#!/usr/bin/env python3
"""
Production Retraining Script for Multi-Day Prediction Models

Retrains LightGBM ensemble models on the most recent data and validates performance.
Run this monthly or when validation metrics degrade.

Usage:
    python scripts/retrain_multi_day_models.py --ticker NDX
    python scripts/retrain_multi_day_models.py --ticker SPX --train-days 250 --validate-days 30
"""

import sys
import os
from pathlib import Path
from datetime import datetime

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import numpy as np
import pandas as pd
from typing import Dict, List

from scripts.csv_prediction_backtest import get_available_dates, load_csv_data
from scripts.close_predictor.multi_day_features import compute_market_context, MarketContext
from scripts.close_predictor.multi_day_lgbm import MultiDayEnsemble


def load_vix_data(vix_ticker: str, all_dates: List[str]) -> Dict[str, pd.DataFrame]:
    """Load VIX/VIX1D historical data."""
    vix_data = {}
    for date_str in all_dates:
        df = load_csv_data(vix_ticker, date_str)
        if df is not None and not df.empty:
            vix_data[date_str] = df
    return vix_data


def main():
    parser = argparse.ArgumentParser(description='Retrain multi-day prediction models')
    parser.add_argument('--ticker', type=str, default='NDX', help='Ticker symbol')
    parser.add_argument('--train-days', type=int, default=250, help='Days of training data')
    parser.add_argument('--validate-days', type=int, default=30, help='Days for validation')
    parser.add_argument('--max-dte', type=int, default=20, help='Maximum DTE to train')
    parser.add_argument('--output-dir', type=Path, default=Path('models/production'),
                        help='Output directory for trained models')
    args = parser.parse_args()

    print(f"\n{'='*80}")
    print(f"MULTI-DAY MODEL RETRAINING")
    print(f"{'='*80}\n")
    print(f"Ticker:           {args.ticker}")
    print(f"Training window:  {args.train_days} days")
    print(f"Validation:       {args.validate_days} days")
    print(f"Max DTE:          {args.max_dte}")
    print(f"Timestamp:        {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    # Load historical data
    print("Loading historical data...")
    total_lookback = args.train_days + args.validate_days + args.max_dte + 20
    all_dates = get_available_dates(args.ticker, total_lookback)

    data_by_date = {}
    for date_str in all_dates:
        df = load_csv_data(args.ticker, date_str)
        if df is not None and not df.empty:
            data_by_date[date_str] = df

    print(f"✓ Loaded {len(all_dates)} trading days")

    # Load VIX and VIX1D data
    print("Loading VIX and VIX1D data...")
    vix_data_by_date = load_vix_data('VIX', all_dates)
    vix1d_data_by_date = load_vix_data('VIX1D', all_dates)
    print(f"✓ Loaded VIX data for {len(vix_data_by_date)} days, VIX1D for {len(vix1d_data_by_date)} days")

    # Determine train/validate split
    # Most recent validate_days for validation, previous train_days for training
    validate_start_idx = len(all_dates) - args.validate_days - args.max_dte
    train_end_idx = validate_start_idx
    train_start_idx = max(0, train_end_idx - args.train_days)

    train_dates = all_dates[train_start_idx:train_end_idx]
    validate_dates = all_dates[validate_start_idx:validate_start_idx + args.validate_days]

    print(f"\nTrain period:     {train_dates[0]} to {train_dates[-1]} ({len(train_dates)} days)")
    print(f"Validate period:  {validate_dates[0]} to {validate_dates[-1]} ({len(validate_dates)} days)")

    # Compute market contexts
    print("\nComputing market context features...")
    all_contexts = {}

    for i, date_str in enumerate(all_dates):
        if date_str not in data_by_date:
            continue

        lookback_idx = max(0, i - 60)
        history_dates = all_dates[lookback_idx:i+1]

        history_rows = []
        for d in history_dates:
            if d in data_by_date:
                df = data_by_date[d]
                history_rows.append({
                    'date': d,
                    'open': df.iloc[0]['open'] if 'open' in df.columns else df.iloc[0]['close'],
                    'close': df.iloc[-1]['close'],
                    'high': df['high'].max() if 'high' in df.columns else df.iloc[-1]['close'],
                    'low': df['low'].min() if 'low' in df.columns else df.iloc[-1]['close'],
                    'volume': df['volume'].sum() if 'volume' in df.columns else 0,
                })

        if not history_rows:
            continue

        price_history = pd.DataFrame(history_rows)
        current_price = price_history.iloc[-1]['close']
        current_date = datetime.strptime(date_str, '%Y-%m-%d').date()

        # Build VIX history
        vix_history = None
        if vix_data_by_date:
            vix_rows = []
            for d in history_dates:
                if d in vix_data_by_date:
                    vdf = vix_data_by_date[d]
                    if not vdf.empty:
                        vix_rows.append({'date': d, 'close': vdf.iloc[-1]['close']})
            if vix_rows:
                vix_history = pd.DataFrame(vix_rows)

        # Build VIX1D history
        vix1d_history = None
        if vix1d_data_by_date:
            vix1d_rows = []
            for d in history_dates:
                if d in vix1d_data_by_date:
                    v1df = vix1d_data_by_date[d]
                    if not v1df.empty:
                        vix1d_rows.append({'date': d, 'close': v1df.iloc[-1]['close']})
            if vix1d_rows:
                vix1d_history = pd.DataFrame(vix1d_rows)

        ctx = compute_market_context(
            ticker=args.ticker,
            current_price=current_price,
            current_date=current_date,
            price_history=price_history,
            vix_history=vix_history,
            vix1d_history=vix1d_history,
            iv_data=None,
        )
        all_contexts[date_str] = ctx

    print(f"✓ Computed contexts for {len(all_contexts)} dates")

    # Compute forward returns
    print(f"\nComputing {args.max_dte}-day forward returns...")
    returns_by_dte = {}

    for dte in range(1, args.max_dte + 1):
        returns_by_dte[dte] = {}
        for i, start_date in enumerate(all_dates):
            if i + dte >= len(all_dates):
                continue
            end_date = all_dates[i + dte]
            if start_date not in data_by_date or end_date not in data_by_date:
                continue

            start_close = data_by_date[start_date].iloc[-1]['close']
            end_close = data_by_date[end_date].iloc[-1]['close']
            forward_return = (end_close - start_close) / start_close * 100
            returns_by_dte[dte][start_date] = forward_return

    print(f"✓ Computed returns for DTE 1-{args.max_dte}")

    # Train ensemble
    print("\nTraining LightGBM ensemble models...")
    ensemble = MultiDayEnsemble()
    train_contexts_by_date = {d: all_contexts[d] for d in train_dates if d in all_contexts}

    ensemble_stats = ensemble.train_all(
        all_dates=train_dates,
        contexts_by_date=train_contexts_by_date,
        returns_by_dte=returns_by_dte,
        max_dte=args.max_dte,
    )

    # Save models
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    model_dir = args.output_dir / f"{args.ticker}_{timestamp}"
    ensemble.save_all(model_dir)

    # Also save to 'latest' directory (overwrite previous)
    latest_dir = args.output_dir / f"{args.ticker}_latest"
    ensemble.save_all(latest_dir)

    print(f"\n✓ Saved models to:")
    print(f"  - {model_dir} (timestamped)")
    print(f"  - {latest_dir} (latest)")

    # Validation summary
    print(f"\n{'='*80}")
    print("VALIDATION SUMMARY")
    print(f"{'='*80}\n")
    print(f"{'DTE':<6} {'Train RMSE':<12} {'Val RMSE':<12} {'Status':<20}")
    print(f"{'─'*60}")

    for dte in range(1, min(args.max_dte + 1, 21)):
        if dte in ensemble_stats:
            stats = ensemble_stats[dte]
            train_rmse = stats['train_rmse']
            val_rmse = stats['val_rmse']

            # Determine status
            if val_rmse < 2.0:
                status = "✓ Excellent"
            elif val_rmse < 3.0:
                status = "✓ Good"
            elif val_rmse < 4.0:
                status = "⚠ Acceptable"
            else:
                status = "⚠ High Error"

            print(f"{dte:<6} {train_rmse:>9.2f}%   {val_rmse:>9.2f}%   {status}")

    # Overall assessment
    print(f"\n{'─'*60}")
    avg_val_rmse = np.mean([ensemble_stats[dte]['val_rmse'] for dte in ensemble_stats])
    max_val_rmse = max([ensemble_stats[dte]['val_rmse'] for dte in ensemble_stats])

    print(f"\nAverage Validation RMSE: {avg_val_rmse:.2f}%")
    print(f"Maximum Validation RMSE: {max_val_rmse:.2f}%")

    if max_val_rmse < 3.0:
        print("\n✓ Models are PRODUCTION READY")
    elif max_val_rmse < 4.0:
        print("\n⚠ Models are acceptable but should be monitored")
    else:
        print("\n⚠ Models show high error - consider retraining with different parameters")

    print(f"\n{'='*80}\n")

    # Save training metadata
    metadata = {
        'ticker': args.ticker,
        'train_period': f"{train_dates[0]} to {train_dates[-1]}",
        'train_days': len(train_dates),
        'validate_period': f"{validate_dates[0]} to {validate_dates[-1]}",
        'timestamp': timestamp,
        'avg_val_rmse': float(avg_val_rmse),
        'max_val_rmse': float(max_val_rmse),
        'validation_stats': {dte: {
            'train_rmse': float(stats['train_rmse']),
            'val_rmse': float(stats['val_rmse']),
        } for dte, stats in ensemble_stats.items()}
    }

    # Save to timestamped directory
    metadata_file = model_dir / "training_metadata.json"
    import json
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)

    # Also save to latest directory
    latest_metadata_file = latest_dir / "training_metadata.json"
    with open(latest_metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"✓ Saved training metadata to:")
    print(f"  - {metadata_file}")
    print(f"  - {latest_metadata_file}")

    # Clear prediction cache to force refresh
    cache_dir = Path(__file__).parent.parent / ".prediction_cache"
    if cache_dir.exists():
        import glob
        cache_pattern = str(cache_dir / f"future_{args.ticker}_*.json")
        cleared_count = 0
        for cache_file in glob.glob(cache_pattern):
            try:
                Path(cache_file).unlink()
                cleared_count += 1
            except Exception:
                pass

        if cleared_count > 0:
            print(f"\n✓ Cleared {cleared_count} cached prediction(s) for {args.ticker}")


if __name__ == '__main__':
    main()
