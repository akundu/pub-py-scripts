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
    
    # Format price_with_change column (combine price and change)
    if 'price_with_change' in df_display.columns and 'current_price' in df_display.columns:
        # If price_with_change is already formatted, keep it
        # Otherwise, combine current_price and change_pct
        pass  # Keep as-is for now
    
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

