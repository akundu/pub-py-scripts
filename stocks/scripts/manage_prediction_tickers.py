#!/usr/bin/env python3
"""
Manage prediction tickers: list, add, remove, train, and check status.

Central management CLI for the prediction ticker config at
data/lists/prediction_tickers.yaml. Use this instead of manually editing
multiple files when adding or removing tickers from the prediction system.
"""

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from common.prediction_config import get_prediction_tickers, get_prediction_tickers_config_path

import yaml


def _load_config() -> dict:
    """Load the prediction tickers YAML config."""
    config_path = get_prediction_tickers_config_path()
    if not config_path.is_file():
        return {'type': 'prediction_tickers',
                'description': 'Tickers with close-price prediction models (0DTE + multi-day)',
                'symbols': []}
    with open(config_path, 'r') as f:
        return yaml.safe_load(f) or {}


def _save_config(data: dict) -> None:
    """Save the prediction tickers YAML config."""
    config_path = get_prediction_tickers_config_path()
    config_path.parent.mkdir(parents=True, exist_ok=True)
    with open(config_path, 'w') as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False)


def _get_model_info(ticker: str) -> dict:
    """Get info about a ticker's models (existence, age, validation metrics)."""
    info = {'has_0dte': False, 'has_multiday': False, 'model_age_days': None, 'val_rmse': None}

    # Check 0DTE model cache
    cache_dir = PROJECT_ROOT / '.cache'
    for p in cache_dir.glob(f'lgbm_model_{ticker}_*.pkl'):
        info['has_0dte'] = True
        mtime = datetime.fromtimestamp(p.stat().st_mtime)
        age = (datetime.now() - mtime).days
        info['model_age_days'] = age
        break

    # Check multi-day production models
    prod_dir = PROJECT_ROOT / 'models' / 'production' / ticker
    if prod_dir.is_dir():
        info['has_multiday'] = True
        meta_path = prod_dir / 'metadata.json'
        if meta_path.is_file():
            try:
                meta = json.loads(meta_path.read_text())
                info['val_rmse'] = meta.get('validation_rmse')
                retrained = meta.get('retrained_date', '')
                if retrained:
                    rt = datetime.strptime(retrained, '%Y%m%d')
                    info['model_age_days'] = (datetime.now() - rt).days
            except Exception:
                pass

    return info


def _check_csv_data(ticker: str) -> dict:
    """Check if CSV data exists for a ticker."""
    info = {'has_equity': False, 'has_options': False}

    equity_dir = PROJECT_ROOT / 'equities_output' / ticker
    if equity_dir.is_dir() and any(equity_dir.glob('*.csv')):
        info['has_equity'] = True

    for options_base in ['options_csv_output', 'options_csv_output_full']:
        options_dir = PROJECT_ROOT / options_base / ticker
        if options_dir.is_dir() and any(options_dir.glob('*.csv')):
            info['has_options'] = True
            break

    return info


def cmd_list(args):
    """Show all configured prediction tickers with model status."""
    tickers = get_prediction_tickers()
    config_path = get_prediction_tickers_config_path()

    print(f"Config: {config_path}")
    print(f"Tickers: {len(tickers)}")
    print()
    print(f"{'Ticker':<8} {'0DTE':<8} {'Multi-day':<12} {'Age (days)':<12} {'Val RMSE':<12} {'Equity CSV':<12} {'Options CSV'}")
    print('-' * 80)

    for ticker in tickers:
        model = _get_model_info(ticker)
        csv = _check_csv_data(ticker)

        dte_status = 'Yes' if model['has_0dte'] else 'No'
        multi_status = 'Yes' if model['has_multiday'] else 'No'
        age = str(model['model_age_days']) if model['model_age_days'] is not None else '-'
        rmse = f"{model['val_rmse']:.2f}%" if model['val_rmse'] is not None else '-'
        equity = 'Yes' if csv['has_equity'] else 'MISSING'
        options = 'Yes' if csv['has_options'] else 'MISSING'

        print(f"{ticker:<8} {dte_status:<8} {multi_status:<12} {age:<12} {rmse:<12} {equity:<12} {options}")


def cmd_add(args):
    """Add a ticker to the prediction config."""
    ticker = args.ticker.upper()
    data = _load_config()
    symbols = data.get('symbols', [])

    if ticker in symbols:
        print(f"{ticker} is already in the config.")
        return

    # Check CSV data availability
    csv = _check_csv_data(ticker)
    if not csv['has_equity']:
        print(f"WARNING: No equity CSV data found for {ticker} in equities_output/{ticker}/")
        if not args.force:
            print("Use --force to add anyway, or provide equity data first.")
            return

    symbols.append(ticker)
    data['symbols'] = symbols
    _save_config(data)
    print(f"Added {ticker} to prediction config.")

    # Train if requested
    if args.train:
        print(f"\nTraining 0DTE model for {ticker}...")
        cmd = [sys.executable, 'scripts/predict_close.py', 'train', ticker, '--clear-cache']
        subprocess.run(cmd, cwd=str(PROJECT_ROOT))

        if args.max_dte:
            print(f"\nTraining multi-day models (1-{args.max_dte} DTE) for {ticker}...")
            cmd = [sys.executable, 'scripts/predict_close.py', 'train', ticker,
                   '--max-dte', str(args.max_dte)]
            subprocess.run(cmd, cwd=str(PROJECT_ROOT))

    # Show updated config
    print()
    print("Updated tickers:", ', '.join(get_prediction_tickers()))
    print()
    print("Next steps:")
    print(f"  1. Train models:  python scripts/predict_close.py train {ticker} --max-dte 20")
    print(f"  2. Test predict:  python scripts/predict_close.py {ticker}")
    print(f"  3. Update cron:   python scripts/manage_prediction_tickers.py status")


def cmd_remove(args):
    """Remove a ticker from the prediction config (does not delete models)."""
    ticker = args.ticker.upper()
    data = _load_config()
    symbols = data.get('symbols', [])

    if ticker not in symbols:
        print(f"{ticker} is not in the config.")
        return

    symbols.remove(ticker)
    data['symbols'] = symbols
    _save_config(data)
    print(f"Removed {ticker} from prediction config.")
    print(f"Note: Models in models/production/{ticker}/ and .cache/ were NOT deleted.")
    print("Updated tickers:", ', '.join(get_prediction_tickers()))


def cmd_train(args):
    """Train models for one or all tickers."""
    if args.all:
        tickers = get_prediction_tickers()
    elif args.ticker:
        tickers = [args.ticker.upper()]
    else:
        print("Specify a ticker or --all")
        return

    for ticker in tickers:
        if ticker not in get_prediction_tickers():
            print(f"WARNING: {ticker} is not in prediction config. Add it first.")
            continue

        print(f"\n{'='*60}")
        print(f"Training {ticker}")
        print(f"{'='*60}")

        # 0DTE model
        print(f"\nTraining 0DTE model for {ticker}...")
        cmd = [sys.executable, 'scripts/predict_close.py', 'train', ticker, '--clear-cache']
        subprocess.run(cmd, cwd=str(PROJECT_ROOT))

        # Multi-day models
        max_dte = args.max_dte or 20
        print(f"\nTraining multi-day models (1-{max_dte} DTE) for {ticker}...")
        cmd = [sys.executable, 'scripts/predict_close.py', 'train', ticker,
               '--max-dte', str(max_dte)]
        subprocess.run(cmd, cwd=str(PROJECT_ROOT))


def cmd_status(args):
    """Show model health and cron instructions."""
    tickers = get_prediction_tickers()
    config_path = get_prediction_tickers_config_path()

    print("Prediction Ticker Status")
    print(f"Config: {config_path}")
    print()

    # Show model status
    cmd_list(args)

    # Show cron instructions
    ticker_list = ','.join(tickers)
    print()
    print("=" * 60)
    print("Crontab Instructions")
    print("=" * 60)
    print()
    print("# Monthly retraining (all tickers):")
    print(f'0 2 * * 6 [ $(date +\\%d) -le 7 ] && cd "$PROJECT_DIR" && ./scripts/retrain_models_auto.sh --all --force >> logs/retraining/cron_$(date +\\%Y\\%m).log 2>&1')
    print()
    print("# Prewarm predictions (weekdays before market open):")
    print(f'30 5 * * 1-5 cd "$PROJECT_DIR" && curl -s "http://localhost:8000/predictions/api/prewarm?ticker={ticker_list}" > /dev/null')
    print()
    print("# Or dynamically from config:")
    print('30 5 * * 1-5 cd "$PROJECT_DIR" && TICKERS=$(python3 -c "from common.prediction_config import get_prediction_tickers; print(\',\'.join(get_prediction_tickers()))") && curl -s "http://localhost:8000/predictions/api/prewarm?ticker=$TICKERS" > /dev/null')


def main():
    parser = argparse.ArgumentParser(
        description='''
Manage prediction tickers for the close-price prediction system.

All programs (db_server.py, predict_close.py, retrain_models_auto.sh)
read from the centralized config at data/lists/prediction_tickers.yaml.
Use this tool to add, remove, train, and check status of prediction tickers.
        ''',
        epilog='''
Examples:
  %(prog)s list
      Show all configured tickers with model status

  %(prog)s add TQQQ --train
      Add TQQQ and train its 0DTE model

  %(prog)s add QQQ --train --max-dte 20
      Add QQQ and train 0DTE + multi-day models

  %(prog)s remove TQQQ
      Remove TQQQ from config (keeps models)

  %(prog)s train NDX
      Retrain NDX models (0DTE + multi-day)

  %(prog)s train --all
      Retrain all configured tickers

  %(prog)s status
      Show model health and cron instructions
        ''',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    subparsers = parser.add_subparsers(dest='command', help='Command to run')

    # list
    sub_list = subparsers.add_parser('list', help='Show configured tickers with model status')

    # add
    sub_add = subparsers.add_parser('add', help='Add a ticker to the prediction config')
    sub_add.add_argument('ticker', help='Ticker symbol to add (e.g., QQQ)')
    sub_add.add_argument('--train', action='store_true', help='Train 0DTE model after adding')
    sub_add.add_argument('--max-dte', type=int, default=0, metavar='N',
                         help='Also train multi-day models up to N DTE (e.g., 20)')
    sub_add.add_argument('--force', action='store_true',
                         help='Add even if CSV data is missing')

    # remove
    sub_remove = subparsers.add_parser('remove', help='Remove a ticker from the prediction config')
    sub_remove.add_argument('ticker', help='Ticker symbol to remove')

    # train
    sub_train = subparsers.add_parser('train', help='Train models for a ticker')
    sub_train.add_argument('ticker', nargs='?', help='Ticker to train (or use --all)')
    sub_train.add_argument('--all', action='store_true', help='Train all configured tickers')
    sub_train.add_argument('--max-dte', type=int, default=20, metavar='N',
                           help='Max DTE for multi-day models (default: 20)')

    # status
    sub_status = subparsers.add_parser('status', help='Show model health and cron instructions')

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return

    commands = {
        'list': cmd_list,
        'add': cmd_add,
        'remove': cmd_remove,
        'train': cmd_train,
        'status': cmd_status,
    }

    commands[args.command](args)


if __name__ == '__main__':
    main()
