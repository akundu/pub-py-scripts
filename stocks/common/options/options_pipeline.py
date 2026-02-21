"""
Analysis pipeline helpers for options analyzer.
"""

import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
from datetime import date, timedelta


def calculate_date_ranges(
    start_date: Optional[str],
    max_days: Optional[int],
    end_date: Optional[str],
    min_days: Optional[int] = None
) -> Tuple[str, Optional[str]]:
    """Calculate start and end dates for option expiration filtering.
    
    Default start_date is yesterday (allowing options that expired < 24hrs ago).
    
    Args:
        start_date: Explicit start date in YYYY-MM-DD format
        max_days: Maximum days from today (sets end_date)
        end_date: Explicit end date in YYYY-MM-DD format
        min_days: Minimum days from today (sets start_date, overrides start_date if provided)
    """
    # If min_days is set, calculate start_date from today (overrides explicit start_date)
    if min_days is not None:
        start_date = (date.today() + timedelta(days=min_days)).strftime('%Y-%m-%d')
    # Otherwise, default start_date to yesterday if not specified (allows options expired < 24hrs ago)
    elif start_date is None:
        start_date = (date.today() - timedelta(days=1)).strftime('%Y-%m-%d')
    
    # If max_days is set, calculate end_date from today (overrides explicit end_date)
    if max_days is not None:
        end_date = (date.today() + timedelta(days=max_days)).strftime('%Y-%m-%d')
    
    return start_date, end_date


def split_combined_options_by_date_range(
    options_df: pd.DataFrame,
    start_date: str,
    end_date: Optional[str],
    long_start_date: str,
    long_end_date: str
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split combined options DataFrame into short-term and long-term options.
    
    Returns:
        Tuple of (short_options_df, long_options_df)
    """
    if options_df.empty or 'expiration_date' not in options_df.columns:
        return options_df, pd.DataFrame()
    
    today_date = date.today()
    
    def get_days_to_expiry(exp_date):
        if pd.isna(exp_date):
            return None
        try:
            if isinstance(exp_date, pd.Timestamp):
                exp_dt = exp_date.date() if hasattr(exp_date, 'date') else pd.to_datetime(exp_date).date()
            else:
                exp_dt = pd.to_datetime(exp_date).date()
            return (exp_dt - today_date).days
        except:
            return None
    
    options_df = options_df.copy()
    options_df['_temp_days_to_expiry'] = options_df['expiration_date'].apply(get_days_to_expiry)
    
    # Short-term options: within the original short-term date range
    short_mask = options_df['_temp_days_to_expiry'].notna()
    if end_date:
        end_dt = pd.to_datetime(end_date).date()
        short_mask = short_mask & (
            options_df['expiration_date'].apply(
                lambda x: pd.to_datetime(x).date() if pd.notna(x) else None
            ) <= end_dt
        )
    if start_date:
        start_dt = pd.to_datetime(start_date).date()
        short_mask = short_mask & (
            options_df['expiration_date'].apply(
                lambda x: pd.to_datetime(x).date() if pd.notna(x) else None
            ) >= start_dt
        )
    
    # Long-term options: within the long-term date range
    long_mask = options_df['_temp_days_to_expiry'].notna()
    if long_start_date:
        long_start_dt = pd.to_datetime(long_start_date).date()
        long_mask = long_mask & (
            options_df['expiration_date'].apply(
                lambda x: pd.to_datetime(x).date() if pd.notna(x) else None
            ) >= long_start_dt
        )
    if long_end_date:
        long_end_dt = pd.to_datetime(long_end_date).date()
        long_mask = long_mask & (
            options_df['expiration_date'].apply(
                lambda x: pd.to_datetime(x).date() if pd.notna(x) else None
            ) <= long_end_dt
        )
    
    # Extract long options before filtering short options
    long_options_df = options_df[long_mask].copy()
    short_options_df = options_df[short_mask].copy()
    short_options_df = short_options_df.drop(columns=['_temp_days_to_expiry'], errors='ignore')
    
    return short_options_df, long_options_df


def log_spread_analysis_start(
    df_short: pd.DataFrame,
    spread_strike_tolerance: float,
    spread_long_days: int,
    spread_long_min_days: Optional[int],
    spread_long_days_tolerance: int,
    log_func
) -> None:
    """Log spread analysis start information."""
    log_func("INFO", f"\n=== Starting Spread Analysis ===")
    log_func("INFO", f"Short-term options found: {len(df_short)}")
    if not df_short.empty:
        log_func("INFO", f"Short-term tickers: {df_short['ticker'].unique().tolist()}")
        log_func("INFO", f"Short-term strike range: ${df_short['strike_price'].min():.2f} to ${df_short['strike_price'].max():.2f}")
        log_func("INFO", f"Short-term days to expiry: {df_short['days_to_expiry'].min()} to {df_short['days_to_expiry'].max()}")
    
    # Debug logging
    import sys
    print(f"DEBUG: Spread mode parameters:", file=sys.stderr)
    print(f"  spread_strike_tolerance: {spread_strike_tolerance}%", file=sys.stderr)
    print(f"  spread_long_days: {spread_long_days}", file=sys.stderr)
    print(f"  spread_long_min_days: {spread_long_min_days}", file=sys.stderr)
    print(f"  spread_long_days_tolerance: {spread_long_days_tolerance}", file=sys.stderr)


