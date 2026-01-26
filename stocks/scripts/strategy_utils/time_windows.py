"""
Time window configuration and CSV path generation.

Handles generating CSV paths for different analysis windows (1yr, 6mo, 3mo, 1mo, 1wk).
"""

import glob as glob_module
import os
from datetime import date, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Window definitions with their date offsets
WINDOW_DEFINITIONS = {
    '1yr': {'days': 365, 'label': '1 Year'},
    '6mo': {'days': 180, 'label': '6 Months'},
    '3mo': {'days': 90, 'label': '3 Months'},
    '1mo': {'days': 30, 'label': '1 Month'},
    '2wk': {'days': 14, 'label': '2 Weeks'},
    '1wk': {'days': 7, 'label': '1 Week'},
}

# Default weights for different regimes
DEFAULT_WEIGHTS = {
    'STABLE': {'6mo': 0.15, '3mo': 0.25, '1mo': 0.25, '2wk': 0.20, '1wk': 0.15},
    'LOW': {'6mo': 0.15, '3mo': 0.25, '1mo': 0.25, '2wk': 0.20, '1wk': 0.15},
    'MEDIUM': {'6mo': 0.12, '3mo': 0.23, '1mo': 0.25, '2wk': 0.22, '1wk': 0.18},
    'HIGH': {'6mo': 0.10, '3mo': 0.18, '1mo': 0.27, '2wk': 0.25, '1wk': 0.20},
    'EXTREME': {'6mo': 0.08, '3mo': 0.15, '1mo': 0.27, '2wk': 0.25, '1wk': 0.25},
}


def get_window_date_range(window: str, reference_date: Optional[date] = None) -> Tuple[date, date]:
    """
    Get the start and end dates for a time window.

    Args:
        window: Window identifier ('1yr', '6mo', '3mo', '1mo', '1wk')
        reference_date: End date for the window (defaults to today)

    Returns:
        Tuple of (start_date, end_date)
    """
    if reference_date is None:
        reference_date = date.today()

    if window not in WINDOW_DEFINITIONS:
        raise ValueError(f"Unknown window: {window}. Valid: {list(WINDOW_DEFINITIONS.keys())}")

    days = WINDOW_DEFINITIONS[window]['days']
    start_date = reference_date - timedelta(days=days)

    return start_date, reference_date


def get_time_window_paths(
    ticker: str,
    window: str,
    csv_dir: str = './options_csv_output',
    reference_date: Optional[date] = None
) -> List[str]:
    """
    Generate CSV paths for a specific time window.

    Args:
        ticker: Underlying ticker (e.g., 'NDX', 'SPX')
        window: Window identifier ('1yr', '6mo', '3mo', '1mo', '1wk')
        csv_dir: Base directory for CSV files
        reference_date: End date for the window (defaults to today)

    Returns:
        List of CSV file paths within the date range
    """
    start_date, end_date = get_window_date_range(window, reference_date)

    # Build glob pattern
    base_dir = os.path.join(csv_dir, ticker)
    pattern = os.path.join(base_dir, f"{ticker}_options_*.csv")

    # Expand and find matching files
    all_files = sorted(glob_module.glob(pattern))

    # Filter by date range
    filtered_files = []
    for filepath in all_files:
        # Extract date from filename: {TICKER}_options_{YYYY-MM-DD}.csv
        filename = os.path.basename(filepath)
        try:
            date_str = filename.replace(f"{ticker}_options_", "").replace(".csv", "")
            file_date = date.fromisoformat(date_str)
            if start_date <= file_date <= end_date:
                filtered_files.append(filepath)
        except ValueError:
            continue

    return filtered_files


def get_all_window_paths(
    ticker: str,
    csv_dir: str = './options_csv_output',
    reference_date: Optional[date] = None,
    windows: Optional[List[str]] = None
) -> Dict[str, List[str]]:
    """
    Get CSV paths for all time windows.

    Args:
        ticker: Underlying ticker
        csv_dir: Base directory for CSV files
        reference_date: End date for windows (defaults to today)
        windows: List of windows to include (defaults to all)

    Returns:
        Dict mapping window names to lists of CSV paths
    """
    if windows is None:
        windows = list(WINDOW_DEFINITIONS.keys())

    return {
        window: get_time_window_paths(ticker, window, csv_dir, reference_date)
        for window in windows
    }


def get_cached_results_path(
    ticker: str,
    window: str,
    output_dir: str = '.'
) -> Path:
    """
    Get the path for cached grid search results for a time window.

    Args:
        ticker: Underlying ticker
        window: Window identifier
        output_dir: Directory for results files

    Returns:
        Path to the results CSV file
    """
    return Path(output_dir) / f"{ticker.lower()}_{window}_results.csv"


def results_exist_and_recent(
    ticker: str,
    window: str,
    output_dir: str = '.',
    max_age_hours: int = 24
) -> bool:
    """
    Check if cached results exist and are recent enough.

    Args:
        ticker: Underlying ticker
        window: Window identifier
        output_dir: Directory for results files
        max_age_hours: Maximum age in hours before results are stale

    Returns:
        True if recent results exist
    """
    results_path = get_cached_results_path(ticker, window, output_dir)

    if not results_path.exists():
        return False

    # Check file age
    from datetime import datetime
    file_mtime = datetime.fromtimestamp(results_path.stat().st_mtime)
    age = datetime.now() - file_mtime

    return age.total_seconds() < max_age_hours * 3600


def get_window_label(window: str) -> str:
    """Get human-readable label for a window."""
    return WINDOW_DEFINITIONS.get(window, {}).get('label', window)


def get_weights_for_regime(regime: str) -> Dict[str, float]:
    """
    Get time window weights adjusted for VIX regime.

    Args:
        regime: VIX regime ('LOW', 'MEDIUM', 'HIGH', 'EXTREME', 'STABLE')

    Returns:
        Dict mapping window names to weights (sum to 1.0)
    """
    return DEFAULT_WEIGHTS.get(regime, DEFAULT_WEIGHTS['MEDIUM']).copy()
