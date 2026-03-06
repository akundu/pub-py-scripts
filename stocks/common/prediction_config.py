#!/usr/bin/env python3
"""
Centralized prediction ticker configuration.

Single source of truth for which tickers have close-price prediction models.
All programs (db_server, predict_close, retrain_models_auto.sh, etc.) should
read from this module instead of hardcoding ticker lists.
"""

import os
from pathlib import Path
from typing import List

import yaml

# Default tickers if config file is missing
_DEFAULT_TICKERS = ['NDX', 'SPX']

# Config file path relative to project root
_CONFIG_FILENAME = 'data/lists/prediction_tickers.yaml'


def _find_project_root() -> Path:
    """Find the project root by looking for known markers."""
    # Try from this file's location: common/ -> stocks/
    candidate = Path(__file__).resolve().parent.parent
    if (candidate / 'data' / 'lists').is_dir():
        return candidate

    # Try from cwd
    cwd = Path.cwd()
    if (cwd / 'data' / 'lists').is_dir():
        return cwd

    # Walk up from cwd
    for parent in cwd.parents:
        if (parent / 'data' / 'lists').is_dir():
            return parent

    return cwd


def get_prediction_tickers() -> List[str]:
    """Load prediction tickers from data/lists/prediction_tickers.yaml.

    Returns:
        List of ticker symbols (e.g., ['NDX', 'SPX', 'TQQQ']).
        Falls back to ['NDX', 'SPX'] if config file is missing.
    """
    config_path = _find_project_root() / _CONFIG_FILENAME

    if not config_path.is_file():
        return list(_DEFAULT_TICKERS)

    try:
        with open(config_path, 'r') as f:
            data = yaml.safe_load(f)
        symbols = data.get('symbols', _DEFAULT_TICKERS)
        if not isinstance(symbols, list) or not symbols:
            return list(_DEFAULT_TICKERS)
        return [str(s).upper() for s in symbols]
    except Exception:
        return list(_DEFAULT_TICKERS)


def get_prediction_tickers_config_path() -> Path:
    """Return the path to the prediction tickers config file."""
    return _find_project_root() / _CONFIG_FILENAME
