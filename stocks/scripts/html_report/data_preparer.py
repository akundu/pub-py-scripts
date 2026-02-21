"""
Data preparation utilities for HTML report generation.
"""

import pandas as pd
import numpy as np
from datetime import datetime
from .formatters import format_age_seconds, format_numeric_value, normalize_col_name
from .constants import (
    COLUMN_VARIATIONS, DESIRED_COLUMN_ORDER, 
    HIDDEN_COLUMNS, ALWAYS_HIDDEN_COLUMNS
)

def prepare_dataframe_for_display(df: pd.DataFrame) -> tuple:
    """Prepare DataFrame for display by formatting columns and renaming.
    
    Args:
        df: Raw DataFrame to prepare
        
    Returns:
        Tuple of (df_display, df_raw) where df_display is formatted and df_raw is original
    """
    # Prepare data for HTML table
    df_display = df.copy()
    
    # Store raw values before formatting (for filtering) - keep original DataFrame
    df_raw = df.copy()
    
    # Format numeric columns for better display
    # Note: expiration_date columns are excluded - they will be formatted as dates separately
    numeric_cols = [
        'ticker','pe_ratio','market_cap_b','curr_price','strike_price','price_above_curr','opt_prem.','IV','delta','theta','days_to_expiry','s_prem_tot','s_day_prem','l_strike','l_prem','liv','l_delta','l_theta','l_days_to_expiry','l_prem_tot','l_cnt_avl','prem_diff','net_premium','net_daily_premi','volume','num_contracts','option_ticker','l_option_ticker',
        'spread_slippage','net_premium_after_spread','net_daily_premium_after_spread','spread_impact_pct','liquidity_score','assignment_risk','trade_quality'
    ]
    
    # Use imported normalize_col_name from formatters
    
    # Also check for common column name variations
    all_numeric_cols = set(numeric_cols)
    for col in df_display.columns:
        col_lower = col.lower()
        # Map common variations
        if 'pe' in col_lower and 'ratio' in col_lower:
            all_numeric_cols.add(col)
        elif 'market' in col_lower and 'cap' in col_lower:
            all_numeric_cols.add(col)
        elif 'current' in col_lower or 'curr' in col_lower:
            all_numeric_cols.add(col)
        elif 'strike' in col_lower:
            all_numeric_cols.add(col)
        elif 'premium' in col_lower:
            all_numeric_cols.add(col)
        elif 'delta' in col_lower or 'theta' in col_lower:
            all_numeric_cols.add(col)
        elif 'volume' in col_lower or 'contracts' in col_lower:
            all_numeric_cols.add(col)
        elif 'days' in col_lower:
            all_numeric_cols.add(col)
    
    # Format expiration date columns to show only date portion (no time)
    def format_date_value(x, col_name):
        """Format date values to show only date portion."""
        if pd.isna(x) or x == '' or x is None:
            return ''
        normalized_col = normalize_col_name(col_name)
        if normalized_col in ['expiration_date', 'l_expiration_date', 'long_expiration_date']:
            try:
                # First try to parse as datetime/timestamp
                if isinstance(x, pd.Timestamp):
                    return x.strftime('%Y-%m-%d')
                elif isinstance(x, (str, datetime)):
                    dt = pd.to_datetime(x, errors='coerce')
                    if pd.notna(dt):
                        return dt.strftime('%Y-%m-%d')
                
                # If that fails, try to parse as numeric (might be a year like 2025.00)
                x_str = str(x).strip()
                try:
                    # Try to parse as float first (handles cases like 2025.00)
                    num_val = float(x_str)
                    # If it's a reasonable year (1900-2100), it might be a year
                    if 1900 <= num_val <= 2100:
                        # Check if it's just a year (like 2025.00) - return as-is for now
                        # But try to parse the original value from df_raw if available
                        pass
                except (ValueError, TypeError):
                    pass
                
                # Try to extract date portion from string
                if ' ' in x_str:
                    return x_str.split(' ')[0]
                elif 'T' in x_str:
                    return x_str.split('T')[0]
                elif len(x_str) >= 10:
                    return x_str[:10]
                else:
                    # Return the full string if it's short enough
                    return x_str
            except (ValueError, TypeError, AttributeError):
                # If all parsing fails, return the string representation
                return str(x)
        elif normalized_col in ['latest_option_writets', 'latest_opt_ts']:
            # For latest_option_writets, it's stored as age in seconds (not a timestamp)
            # Format as human-readable age (secs, mins, hrs, days)
            try:
                # First check if it's already a timestamp (pd.Timestamp or datetime)
                if isinstance(x, pd.Timestamp):
                    # If it's a timestamp, calculate age from now
                    now = pd.Timestamp.now(tz='UTC') if x.tz else pd.Timestamp.now()
                    if x.tz:
                        age_sec = (now - x).total_seconds()
                    else:
                        age_sec = (now.tz_localize(None) - x).total_seconds()
                    return format_age_seconds(age_sec)
                elif isinstance(x, (str, datetime)):
                    # Try to parse as datetime first
                    dt = pd.to_datetime(x, errors='coerce')
                    if pd.notna(dt):
                        # It's a timestamp, calculate age
                        now = pd.Timestamp.now(tz='UTC') if dt.tz else pd.Timestamp.now()
                        if dt.tz:
                            age_sec = (now - dt).total_seconds()
                        else:
                            age_sec = (now.tz_localize(None) - dt).total_seconds()
                        return format_age_seconds(age_sec)
                
                # If not a timestamp, treat as age in seconds (numeric)
                x_str = str(x).strip()
                
                # Check if it looks like a timestamp string (has date-like patterns)
                if ' ' in x_str and len(x_str) >= 19:
                    # Looks like a timestamp string, try to parse
                    dt = pd.to_datetime(x_str, errors='coerce')
                    if pd.notna(dt):
                        now = pd.Timestamp.now(tz='UTC') if dt.tz else pd.Timestamp.now()
                        if dt.tz:
                            age_sec = (now - dt).total_seconds()
                        else:
                            age_sec = (now.tz_localize(None) - dt).total_seconds()
                        return format_age_seconds(age_sec)
                    # If parsing fails, return as-is
                    return x_str[:19]
                elif 'T' in x_str:
                    # ISO format timestamp
                    dt = pd.to_datetime(x_str, errors='coerce')
                    if pd.notna(dt):
                        now = pd.Timestamp.now(tz='UTC') if dt.tz else pd.Timestamp.now()
                        if dt.tz:
                            age_sec = (now - dt).total_seconds()
                        else:
                            age_sec = (now.tz_localize(None) - dt).total_seconds()
                        return format_age_seconds(age_sec)
                    return x_str.replace('T', ' ')[:19]
                else:
                    # Likely a numeric value (age in seconds)
                    # Try to parse as float
                    try:
                        age_sec = float(x_str)
                        # If it's a reasonable age value (0 to 1 year in seconds)
                        if 0 <= age_sec <= 31536000:  # 1 year in seconds
                            return format_age_seconds(age_sec)
                        # If it's a very large number, might be a timestamp in milliseconds
                        elif age_sec > 1000000000000:  # Timestamp in milliseconds
                            dt = pd.to_datetime(age_sec / 1000, unit='s', errors='coerce')
                            if pd.notna(dt):
                                now = pd.Timestamp.now(tz='UTC')
                                age_sec_calc = (now - dt.tz_localize('UTC')).total_seconds()
                                return format_age_seconds(age_sec_calc)
                    except (ValueError, TypeError):
                        pass
                    
                    # If all else fails, return as string
                    return x_str
            except (ValueError, TypeError, AttributeError):
                # If all parsing fails, return the string representation
                return str(x)
        return x
    
    # Format date columns first - use raw values from df_raw before any numeric formatting
    for col in df_display.columns:
        normalized_col = normalize_col_name(col)
        if normalized_col in ['expiration_date', 'l_expiration_date', 'long_expiration_date', 'latest_option_writets', 'latest_opt_ts']:
            # Use raw values from df_raw to get original date format before any numeric conversion
            if df_raw is not None and col in df_raw.columns:
                df_display[col] = df_raw[col].apply(lambda x: format_date_value(x, col))
            else:
                df_display[col] = df_display[col].apply(lambda x: format_date_value(x, col))
            # Remove from numeric columns so they don't get formatted as numbers
            all_numeric_cols.discard(col)
    
    for col in all_numeric_cols:
        if col in df_display.columns:
            df_display[col] = df_display[col].apply(lambda x: format_numeric_value(x, col))
    
    # Replace remaining NaN with empty strings for display
    df_display = df_display.fillna('')
    
    # Helper function to find column by partial name match (handles truncated variations)
    def find_column_by_partial_name(df, target_pattern):
        """Find a column in DataFrame by partial name match, handling variations."""
        target_pattern_normalized = target_pattern.lower().replace(' ', '_')
        for col in df.columns:
            col_normalized = normalize_col_name(col)
            # Check if the column starts with the pattern (handles truncation)
            if col_normalized.startswith(target_pattern_normalized):
                return col
            # Also check exact match
            if col_normalized == target_pattern_normalized:
                return col
        return None
    
    # Normalize/standardize column names first
    # Handle current_price variations (curr_price, current_price, cur_price)
    current_price_col = find_column_by_partial_name(df_display, 'curr')
    if current_price_col:
        # Check if it's actually a price column (not current_strike, etc.)
        if 'price' in current_price_col.lower() or current_price_col.lower() in ['curr_price', 'cur_price']:
            if current_price_col != 'current_price':
                df_display = df_display.rename(columns={current_price_col: 'current_price'})
                df_raw = df_raw.rename(columns={current_price_col: 'current_price'})
    
    # Handle price_with_change variations (price_with_chan, price_with_ch, price_with_change)
    price_with_change_col = find_column_by_partial_name(df_display, 'price_with_ch')
    if price_with_change_col and 'pct' not in price_with_change_col.lower():
        if price_with_change_col != 'price_with_change':
            df_display = df_display.rename(columns={price_with_change_col: 'price_with_change'})
            df_raw = df_raw.rename(columns={price_with_change_col: 'price_with_change'})
    
    # Handle price_change_pct variations (price_change_pc, price_change_pct)
    for col in df_display.columns:
        col_lower = col.lower()
        if 'price_change' in col_lower and 'pc' in col_lower and col != 'price_change_pct':
            df_display = df_display.rename(columns={col: 'price_change_pct'})
            df_raw = df_raw.rename(columns={col: 'price_change_pct'})
            break
    
    # Rename latest_opt_ts to latest_option_writets for display (before reordering)
    if 'latest_opt_ts' in df_display.columns:
        df_display = df_display.rename(columns={'latest_opt_ts': 'latest_option_writets'})
        if 'latest_opt_ts' in df_raw.columns:
            df_raw = df_raw.rename(columns={'latest_opt_ts': 'latest_option_writets'})
    
    # Reorder columns: ticker, pe_ratio, market_cap_b, current_price, price_with_change (renamed to change_pct)
    # Define desired column order (only reorder if columns exist)
    # Note: price_change_pct is kept for sorting but not shown in desired_order
    
    # Build a flexible mapping for column ordering that handles CSV column name variations
    def find_matching_column(desired_name):
        """Find a column that matches the desired name, handling variations."""
        # Exact match first
        if desired_name in df_display.columns:
            return desired_name
        
        # Handle common variations
        variations = {
            'current_price': ['curr_price', 'cur_price', 'current_price'],
            'price_with_change': ['price_with_change', 'price_with_chan', 'price_with_ch'],
            'option_premium': ['opt_prem.', 'opt_prem', 'option_premium'],
            'bid_ask': ['bid:ask', 'bid_ask'],
            'short_premium_total': ['s_prem_tot', 's_prem_total', 'short_premium_total'],
            'short_daily_premium': ['s_day_prem', 's_daily_prem', 'short_daily_premium'],
            'long_strike_price': ['l_strike', 'l_strike_price', 'long_strike_price'],
            'long_option_premium': ['l_opt_prem', 'l_prem', 'l_option_premium', 'long_option_premium'],
            'long_bid_ask': ['l_bid:ask', 'l_bid_ask', 'long_bid_ask'],
            'long_implied_volatility': ['liv', 'l_iv', 'long_implied_volatility'],
            'long_delta': ['l_delta', 'long_delta'],
            'long_theta': ['l_theta', 'long_theta'],
            'long_expiration_date': ['l_expiration_date', 'long_expiration_date'],
            'long_days_to_expiry': ['l_days_to_expiry', 'long_days_to_expiry'],
            'long_premium_total': ['l_prem_tot', 'l_premium_total', 'long_premium_total'],
            'long_contracts_available': ['l_cnt_avl', 'l_contracts_available', 'long_contracts_available'],
            'net_daily_premium': ['net_daily_premi', 'net_daily_premium'],
        }
        
        if desired_name in variations:
            for var in variations[desired_name]:
                if var in df_display.columns:
                    return var
        
        return None
    
    # Define desired order with standardized names
    desired_order_names = [
        'ticker', 'option_type', 'pe_ratio', 'market_cap_b', 'current_price', 'price_with_change',
        'strike_price', 'option_premium', 'expiration_date', 'bid_ask', 'delta', 'theta',
        'short_premium_total', 'short_daily_premium',
        'long_strike_price', 'long_option_premium', 'long_expiration_date', 'long_bid_ask', 'long_delta', 'long_theta',
        'long_implied_volatility', 'long_days_to_expiry', 
        'long_premium_total', 'long_contracts_available',
        'net_premium', 'net_daily_premium',
        'price_above_current', 'premium_above_diff_percentage',
        'implied_volatility', 'days_to_expiry',
        'potential_premium', 'daily_premium',
        'volume', 'num_contracts', 'option_ticker', 'long_option_ticker',
        'premium_diff',  # Will be hidden by default
        'spread_slippage', 'net_premium_after_spread', 'net_daily_premium_after_spread',
        'spread_impact_pct', 'liquidity_score', 'assignment_risk', 'trade_quality',
        'latest_option_writets'  # Latest option write timestamp (always visible, rightmost)
    ]
    
    # Get existing columns in desired order, handling variations
    ordered_cols = []
    for desired_name in desired_order_names:
        actual_col = find_matching_column(desired_name)
        if actual_col:
            ordered_cols.append(actual_col)
    
    # Remove latest_option_writets from ordered_cols if present (will be added at the end)
    # latest_option_writets should always be at the rightmost position
    if 'latest_option_writets' in ordered_cols:
        ordered_cols.remove('latest_option_writets')
    
    # Add any remaining columns (excluding price_change_pct and latest_option_writets from visible)
    remaining_cols = [col for col in df_display.columns 
                      if col not in ordered_cols 
                      and col != 'price_change_pct' 
                      and col != 'latest_option_writets']
    # Add price_change_pct before latest_option_writets (hidden, used for sorting)
    if 'price_change_pct' in df_display.columns:
        remaining_cols.append('price_change_pct')
    # Add latest_option_writets at the very end (rightmost position) if it exists
    if 'latest_option_writets' in df_display.columns:
        remaining_cols.append('latest_option_writets')
    df_display = df_display[ordered_cols + remaining_cols]
    
    # Rename price_with_change to change_pct for display
    if 'price_with_change' in df_display.columns:
        df_display = df_display.rename(columns={'price_with_change': 'change_pct'})
        # Also update df_raw for consistency
        if 'price_with_change' in df_raw.columns:
            df_raw = df_raw.rename(columns={'price_with_change': 'change_pct'})
    
    # Rename l_prem to l_opt_prem for display
    if 'l_prem' in df_display.columns:
        df_display = df_display.rename(columns={'l_prem': 'l_opt_prem'})
        if 'l_prem' in df_raw.columns:
            df_raw = df_raw.rename(columns={'l_prem': 'l_opt_prem'})
    
    # Rename l_prem_tot to buy_cost for display
    if 'l_prem_tot' in df_display.columns:
        df_display = df_display.rename(columns={'l_prem_tot': 'buy_cost'})
        if 'l_prem_tot' in df_raw.columns:
            df_raw = df_raw.rename(columns={'l_prem_tot': 'buy_cost'})
    
    return df_display, df_raw
