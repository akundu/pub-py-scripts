"""
Output formatting functions for options analysis results.
"""

import pandas as pd
import csv
from typing import Dict, List, Any, Optional
from common.common import (
    normalize_timestamp_to_utc,
    normalize_timestamp_for_display,
    normalize_expiration_date_to_utc,
    format_age_seconds
)


def format_dataframe_for_display(df: pd.DataFrame) -> pd.DataFrame:
    """Format DataFrame columns for table display (currency, percentages, etc.)."""
    df = df.copy()
    
    # Format currency columns
    # For large totals, use comma formatting; for smaller values, use 2 decimal places
    large_currency_cols = ['short_premium_total', 'short_daily_premium', 'long_premium_total', 
                          'net_premium', 'net_daily_premium', 'potential_premium', 'daily_premium']
    small_currency_cols = ['current_price', 'strike_price', 'price_above_current', 'option_premium', 
                          'long_strike_price', 'long_option_premium', 'premium_diff']
    
    for col in large_currency_cols:
        if col in df.columns:
            df[col] = df[col].apply(lambda x: f"${float(x):,.2f}" if pd.notna(x) else "N/A")
    
    for col in small_currency_cols:
        if col in df.columns:
            df[col] = df[col].apply(lambda x: f"${float(x):.2f}" if pd.notna(x) else "N/A")
    
    # Format numeric columns
    numeric_cols = ['pe_ratio', 'market_cap_b']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = df[col].apply(lambda x: f"{x:.2f}" if pd.notna(x) else "N/A")
    
    # Format percentage columns
    pct_cols = ['option_premium_percentage', 'premium_above_diff_percentage']
    for col in pct_cols:
        if col in df.columns:
            df[col] = df[col].apply(lambda x: f"{x:.2f}%" if pd.notna(x) else "N/A")
    
    # Format implied volatility columns (convert from decimal to percentage)
    iv_cols = ['implied_volatility', 'long_implied_volatility']
    for col in iv_cols:
        if col in df.columns:
            df[col] = df[col].apply(lambda x: f"{x * 100:.2f}%" if pd.notna(x) else "N/A")
    
    # Format integer columns
    if 'long_contracts_available' in df.columns:
        df['long_contracts_available'] = df['long_contracts_available'].apply(
            lambda x: f"{int(x)}" if pd.notna(x) else "N/A"
        )
    
    return df


def normalize_and_select_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize timestamps and select output columns."""
    df = df.copy()
    
    # Round numeric columns
    for col in ['bid', 'ask', 'delta', 'theta']:
        if col in df.columns:
            df[col] = df[col].round(2)
    
    if 'implied_volatility' in df.columns:
        df['implied_volatility'] = df['implied_volatility'].round(4)
    
    # Normalize timestamp columns
    for ts_col in ['last_quote_timestamp', 'write_timestamp']:
        if ts_col in df.columns:
            df[ts_col] = df[ts_col].apply(normalize_timestamp_to_utc)
    
    output_cols = [
        'ticker', 'current_price', 'price_with_change', 'price_change_pct', 'strike_price', 'price_above_current',
        'option_premium', 'bid_ask', 'implied_volatility', 'delta', 'theta', 'volume', 'num_contracts',
        'potential_premium', 'daily_premium', 'expiration_date', 'days_to_expiry',
        'last_quote_timestamp', 'option_ticker', 'latest_opt_ts'
    ]
    available_cols = [c for c in output_cols if c in df.columns]
    return df[available_cols].copy()


def create_compact_headers(df: pd.DataFrame) -> Dict[str, str]:
    """Create compact headers that are at most 4 characters longer than the data width."""
    compact_headers = {}
    
    # Define mapping for common columns to shorter names
    header_mapping = {
        'current_price': 'curr_price',
        'price_above_current': 'price_above_curr',
        'option_premium': 'opt_prem.',
        'bid_ask': 'bid:ask',
        'option_premium_percentage': 'opt_prem.%',
        'implied_volatility': 'iv',
        'long_strike_price': 'l_strike',
        'long_option_premium': 'l_prem',
        'long_bid_ask': 'l_bid:ask',
        'long_expiration_date': 'l_expiration_date',
        'long_days_to_expiry': 'l_days_to_expiry',
        'long_option_ticker': 'l_option_ticker',
        'long_delta': 'l_delta',
        'long_theta': 'l_theta',
        'long_implied_volatility': 'liv',
        'long_volume': 'l_volume',
        'long_contracts_available': 'l_cnt_avl',
        'premium_diff': 'prem_diff',
        'short_premium_total': 's_prem_tot',
        'short_daily_premium': 's_day_prem',
        'long_premium_total': 'l_prem_tot',
        'iv_recommendation': 'iv_rec',  # Prevent truncation
    }
    
    for col in df.columns:
        if col in header_mapping:
            compact_headers[col] = header_mapping[col]
        else:
            # For unknown columns, use the original name but truncate if too long
            compact_headers[col] = col[:15] if len(col) > 8 else col
    
    return compact_headers


def format_csv_output(
    df: pd.DataFrame, 
    delimiter: str = ',', 
    quoting: str = 'minimal', 
    group_by: str = 'overall',
    output_file: Optional[str] = None
) -> str:
    """Format DataFrame as CSV with proper formatting."""
    # Convert quoting string to csv module constant
    quoting_map = {
        'minimal': csv.QUOTE_MINIMAL,
        'all': csv.QUOTE_ALL,
        'none': csv.QUOTE_NONE,
        'nonnumeric': csv.QUOTE_NONNUMERIC
    }
    csv_quoting = quoting_map.get(quoting, csv.QUOTE_MINIMAL)
    
    # Create a copy for CSV formatting
    df_csv = df.copy()
    
    # Format numeric columns for CSV (remove $ symbols and % symbols for cleaner data)
    for col in ['current_price', 'strike_price', 'price_above_current', 'option_premium', 'potential_premium', 'daily_premium',
                'long_strike_price', 'long_option_premium', 'premium_diff', 'short_premium_total', 'short_daily_premium', 'long_premium_total', 'net_premium', 'net_daily_premium']:
        if col in df_csv.columns:
            df_csv[col] = df_csv[col].apply(lambda x: float(x.replace('$', '').replace(',', '')) if isinstance(x, str) and '$' in str(x) else x)
    
    for col in ['pe_ratio', 'market_cap_b']:
        if col in df_csv.columns:
            df_csv[col] = df_csv[col].apply(lambda x: float(x.replace(',', '')) if isinstance(x, str) and ',' in str(x) else x)
    
    for col in ['option_premium_percentage', 'premium_above_diff_percentage']:
        if col in df_csv.columns:
            df_csv[col] = df_csv[col].apply(lambda x: float(x.replace('%', '').replace(',', '')) if isinstance(x, str) and '%' in str(x) else x)
    
    # Handle grouping
    if group_by == 'ticker':
        # For CSV, we'll create a single CSV with all data but add a grouping column
        df_csv['group'] = df_csv['ticker']
        # Sort by ticker first, then by the original sort order
        df_csv = df_csv.sort_values(['ticker'])
    
    # Generate CSV content
    csv_content = df_csv.to_csv(
        index=False, 
        sep=delimiter, 
        quoting=csv_quoting,
        na_rep='',
        float_format='%.2f'
    )
    
    # Save to file if specified
    if output_file:
        with open(output_file, 'w', newline='', encoding='utf-8') as f:
            f.write(csv_content)
    
    return csv_content

