#!/usr/bin/env python3
"""
Deprecated: use predict_close.py train --max-dte 20 instead.

This script is a backward-compatibility wrapper around predict_close.train_multi_day_models().
"""

import sys
import warnings
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

warnings.warn(
    "retrain_multi_day_models.py is deprecated. "
    "Use: python scripts/predict_close.py train --max-dte 20",
    DeprecationWarning,
    stacklevel=2,
)

import argparse
from scripts.predict_close import train_multi_day_models


def main():
    parser = argparse.ArgumentParser(
        description='Retrain multi-day prediction models (DEPRECATED: use predict_close.py train --max-dte N)',
    )
    parser.add_argument('--ticker', type=str, default='NDX', help='Ticker symbol')
    parser.add_argument('--train-days', type=int, default=250, help='Days of training data')
    parser.add_argument('--validate-days', type=int, default=30, help='Days for validation')
    parser.add_argument('--max-dte', type=int, default=20, help='Maximum DTE to train')
    parser.add_argument('--output-dir', type=Path, default=Path('models/production'),
                        help='Output directory for trained models')
    args = parser.parse_args()

    print("⚠️  This script is deprecated. Use instead:")
    print(f"   python scripts/predict_close.py train {args.ticker} --max-dte {args.max_dte} "
          f"--train-days {args.train_days} --validate-days {args.validate_days}\n")

    train_multi_day_models(
        ticker=args.ticker,
        train_days=args.train_days,
        validate_days=args.validate_days,
        max_dte=args.max_dte,
        output_dir=args.output_dir,
    )


if __name__ == '__main__':
    main()
