"""
Data processing and formatting utilities.
"""

import pandas as pd
import numpy as np
import re
from datetime import datetime
from typing import Tuple, Optional
from .config import (
    COLUMN_VARIATIONS, DESIRED_COLUMN_ORDER, HIDDEN_COLUMNS,
    ALWAYS_HIDDEN_COLUMNS
)


def normalize_col_name(name: str) -> str:
    """Normalize column name for matching (lowercase, underscores, no spaces)."""
    return str(name).strip().lower().replace(' ', '_').replace('-', '_')


def format_age_seconds(age_seconds) -> str:
    """Format age in seconds to human-readable format.
    
    Args:
        age_seconds: Age in seconds (float, int, or string)
        
    Returns:
        Human-readable string like "5 secs", "2 mins", "3 hrs", "1.2 days"
    """
    if pd.isna(age_seconds) or age_seconds == '' or age_seconds is None:
        return ''
    
    try:
        # Convert to float
        if isinstance(age_seconds, str):
            match = re.search(r'-?\d+\.?\d*', age_seconds)
            if match:
                age_sec = float(match.group())
            else:
                return str(age_seconds)
        else:
            age_sec = float(age_seconds)
        
        # Handle negative or zero
        if age_sec < 0:
            return 'N/A'
        if age_sec == 0:
            return '0 secs'
        
        # Convert to appropriate unit
        if age_sec < 60:
            return f"{age_sec:.1f} secs" if age_sec < 10 else f"{int(age_sec)} secs"
        elif age_sec < 3600:
            mins = age_sec / 60
            return f"{mins:.1f} mins" if mins < 10 else f"{int(mins)} mins"
        elif age_sec < 86400:
            hrs = age_sec / 3600
            return f"{hrs:.1f} hrs" if hrs < 10 else f"{int(hrs)} hrs"
        else:
            days = age_sec / 86400
            return f"{days:.1f} days" if days < 10 else f"{int(days)} days"
    except (ValueError, TypeError, AttributeError):
        return str(age_seconds)


def format_numeric_value(value, col_name: str) -> str:
    """Format numeric value based on column type.
    
    Args:
        value: The value to format
        col_name: Column name to determine formatting
        
    Returns:
        Formatted string
    """
    if pd.isna(value) or value == '' or value is None:
        return ''
    
    normalized_col = normalize_col_name(col_name)
    
    # Currency columns
    if any(x in normalized_col for x in ['price', 'strike', 'premium', 'prem', 'cost', 'total']):
        try:
            num_val = float(value)
            return f"${num_val:,.2f}"
        except (ValueError, TypeError):
            return str(value)
    
    # Percentage columns
    if 'pct' in normalized_col or 'percent' in normalized_col or normalized_col.endswith('%'):
        try:
            num_val = float(value)
            return f"{num_val:.2f}%"
        except (ValueError, TypeError):
            return str(value)
    
    # Trade quality column - fixed decimal precision
    if normalized_col == 'trade_quality':
        try:
            num_val = float(value)
            return f"{num_val:.2f}"
        except (ValueError, TypeError):
            return str(value)
    
    # Integer columns
    if any(x in normalized_col for x in ['volume', 'contracts', 'cnt', 'days']):
        try:
            num_val = float(value)
            return f"{int(num_val):,}"
        except (ValueError, TypeError):
            return str(value)
    
    # Default: return as string
    return str(value)


def extract_numeric_value(value) -> Optional[float]:
    """Extract numeric value from formatted string.
    
    Args:
        value: Value that might be formatted (e.g., "$1,234.56", "12.34%")
        
    Returns:
        Numeric value or None
    """
    if pd.isna(value) or value == '' or value is None:
        return None
    
    try:
        if isinstance(value, (int, float)):
            return float(value)
        
        # Remove currency symbols, commas, percentages
        val_str = str(value).replace('$', '').replace(',', '').replace('%', '').strip()
        match = re.search(r'-?\d+\.?\d*', val_str)
        if match:
            return float(match.group())
    except (ValueError, TypeError, AttributeError):
        pass
    
    return None


def find_first_matching_column(df: pd.DataFrame, candidates) -> Optional[str]:
    """Find the first column in df that matches any of the candidate names (normalized)."""
    normalized_candidates = {c.replace('.', '').lower() for c in candidates}
    for col in df.columns:
        normalized = normalize_col_name(col).replace('.', '')
        if normalized in normalized_candidates:
            return col
    return None


def calculate_theta_percentages(df: pd.DataFrame) -> Tuple[Optional[pd.Series], Optional[pd.Series]]:
    """Calculate theta percentages for short and long term options.
    
    Args:
        df: DataFrame with theta and premium columns
        
    Returns:
        Tuple of (theta_pct_series, l_theta_pct_series)
    """
    theta_pct = None
    l_theta_pct = None
    
    # Identify premium columns (short & long)
    short_prem_col = find_first_matching_column(df, ['option_premium', 'opt_prem', 'opt_prem.'])
    long_prem_col = find_first_matching_column(df, ['l_opt_prem', 'l_prem', 'long_option_premium'])
    
    # Short theta percentage
    if 'theta' in df.columns and short_prem_col:
        theta_vals = df['theta'].apply(extract_numeric_value)
        prem_vals = df[short_prem_col].apply(extract_numeric_value)
        theta_pct = (theta_vals / prem_vals) * 100
        theta_pct = theta_pct.where((prem_vals != 0) & pd.notna(theta_vals) & pd.notna(prem_vals))
    
    # Long theta percentage
    long_theta_col = find_first_matching_column(df, ['l_theta', 'long_theta'])
    if long_theta_col and long_prem_col:
        l_theta_vals = df[long_theta_col].apply(extract_numeric_value)
        l_prem_vals = df[long_prem_col].apply(extract_numeric_value)
        l_theta_pct = (l_theta_vals / l_prem_vals) * 100
        l_theta_pct = l_theta_pct.where((l_prem_vals != 0) & pd.notna(l_theta_vals) & pd.notna(l_prem_vals))
    
    return theta_pct, l_theta_pct


def prepare_dataframe_for_display(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Prepare DataFrame for HTML display.
    
    Args:
        df: Raw DataFrame with options data
        
    Returns:
        Tuple of (formatted_df, raw_df) where:
        - formatted_df: DataFrame with formatted values for display
        - raw_df: DataFrame with raw values for filtering/sorting
    """
    if df.empty:
        return df.copy(), df.copy()
    
    df_display = df.copy()
    df_raw = df.copy()
    
    # Calculate theta percentages (used for sorting & display metadata)
    theta_pct_series, l_theta_pct_series = calculate_theta_percentages(df_raw)
    
    # Normalize column names (handle variations)
    for standard_name, variations in COLUMN_VARIATIONS.items():
        for col in df_display.columns:
            if col in variations and col != standard_name:
                df_display = df_display.rename(columns={col: standard_name})
                df_raw = df_raw.rename(columns={col: standard_name})
                break
    
    # Format date columns
    date_cols = ['expiration_date', 'l_expiration_date', 'long_expiration_date']
    for col in date_cols:
        if col in df_display.columns:
            df_display[col] = df_display[col].apply(
                lambda x: str(x)[:10] if pd.notna(x) else ''
            )
    
    # Rename latest_opt_ts to latest_option_writets for display (standardize column name)
    # The API returns latest_opt_ts, but we want to use latest_option_writets consistently
    if 'latest_opt_ts' in df_display.columns and 'latest_option_writets' not in df_display.columns:
        df_display = df_display.rename(columns={'latest_opt_ts': 'latest_option_writets'})
    if 'latest_opt_ts' in df_raw.columns and 'latest_option_writets' not in df_raw.columns:
        df_raw = df_raw.rename(columns={'latest_opt_ts': 'latest_option_writets'})
    
    # Format age columns (latest_option_writets)
    age_cols = ['latest_option_writets', 'latest_opt_ts']
    for col in age_cols:
        if col in df_display.columns:
            df_display[col] = df_display[col].apply(format_age_seconds)
    
    # Format numeric columns
    for col in df_display.columns:
        normalized_col = normalize_col_name(col)
        # Skip date and age columns (already formatted)
        if any(x in normalized_col for x in ['expiration', 'exp_date', 'writets', 'latest_opt']):
            continue
        
        # Format numeric values
        if df_display[col].dtype in [np.int64, np.float64, 'int64', 'float64']:
            df_display[col] = df_display[col].apply(
                lambda x: format_numeric_value(x, col) if pd.notna(x) else ''
            )
        elif df_display[col].dtype == 'object':
            # Try to format if it looks numeric
            sample_val = df_display[col].dropna().iloc[0] if len(df_display[col].dropna()) > 0 else None
            if sample_val is not None:
                try:
                    float(str(sample_val).replace('$', '').replace(',', '').replace('%', ''))
                    df_display[col] = df_display[col].apply(
                        lambda x: format_numeric_value(x, col) if pd.notna(x) and str(x).strip() else ''
                    )
                except (ValueError, TypeError):
                    pass
    
    # Replace current_price with price_with_change if both exist
    # This makes current_price show price with change (e.g., "$124.59 (+$0.50 (+0.40%))")
    if 'price_with_change' in df_display.columns and 'current_price' in df_display.columns:
        # Replace current_price with price_with_change so the price column shows the change
        df_display['current_price'] = df_display['price_with_change']
        # Also update df_raw so the raw data matches
        if 'price_with_change' in df_raw.columns:
            df_raw['current_price'] = df_raw['price_with_change']
    
    # Reorder columns (basic ordering)
    ordered_cols = []
    remaining_cols = set(df_display.columns)
    
    # Add desired columns in order
    for col in DESIRED_COLUMN_ORDER:
        if col in df_display.columns:
            ordered_cols.append(col)
            remaining_cols.discard(col)
    
    # Add remaining columns
    for col in df_display.columns:
        if col in remaining_cols:
            ordered_cols.append(col)
    
    df_display = df_display[ordered_cols]
    df_raw = df_raw[ordered_cols]
    
    # Store theta percentages in attrs for downstream renderers
    if theta_pct_series is not None:
        theta_pct_map = theta_pct_series.to_dict()
        df_display.attrs['theta_pct'] = theta_pct_map
        df_raw.attrs['theta_pct'] = theta_pct_map
    if l_theta_pct_series is not None:
        l_theta_pct_map = l_theta_pct_series.to_dict()
        df_display.attrs['l_theta_pct'] = l_theta_pct_map
        df_raw.attrs['l_theta_pct'] = l_theta_pct_map
    
    return df_display, df_raw


def split_calls_and_puts(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, bool, bool]:
    """Split DataFrame into calls and puts.
    
    Args:
        df: DataFrame with option_type column
        
    Returns:
        Tuple of (df_calls, df_puts, has_calls, has_puts)
    """
    if 'option_type' not in df.columns:
        # No option_type column - treat all as calls
        return df.copy(), pd.DataFrame(), True, False
    
    df_calls = df[df['option_type'].str.lower() == 'call'].copy()
    df_puts = df[df['option_type'].str.lower() == 'put'].copy()
    
    has_calls = len(df_calls) > 0
    has_puts = len(df_puts) > 0
    
    return df_calls, df_puts, has_calls, has_puts

