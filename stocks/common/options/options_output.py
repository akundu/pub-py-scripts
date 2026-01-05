"""
Output formatting helpers for options analyzer.
"""

import pandas as pd
from typing import Dict, List, Any, Optional
from tabulate import tabulate

from common.options.options_filters import FilterParser, FilterExpression
from common.common import (
    format_age_seconds,
    normalize_expiration_date_to_utc,
    normalize_timestamp_for_display
)
from common.options.options_formatting import format_dataframe_for_display


def enrich_dataframe_with_financial_data(
    df: pd.DataFrame,
    financial_data: Dict[str, Dict[str, Any]]
) -> pd.DataFrame:
    """Add financial information to the dataframe."""
    if 'ticker' not in df.columns:
        return df
    
    df_renamed = df.copy()
    df_renamed['pe_ratio'] = df_renamed['ticker'].map(
        lambda x: financial_data.get(x, {}).get('pe_ratio')
    )
    df_renamed['market_cap'] = df_renamed['ticker'].map(
        lambda x: financial_data.get(x, {}).get('market_cap')
    )
    df_renamed['market_cap_b'] = df_renamed['market_cap'].apply(
        lambda x: round(x / 1e9, 2) if pd.notna(x) and x is not None else None
    )
    
    # Add IV metrics from financial_data
    # Risk score (0-10) - check multiple locations
    df_renamed['risk_score'] = df_renamed['ticker'].map(
        lambda x: (
            financial_data.get(x, {}).get('risk_score')  # Root level (from options_analyzer)
            or financial_data.get(x, {}).get('iv_strategy', {}).get('risk_score')  # From iv_strategy (from stock info)
            or (financial_data.get(x, {}).get('iv_metrics', {}).get('risk_score') 
                if isinstance(financial_data.get(x, {}).get('iv_metrics'), dict) else None)
        )
    )
    
    # IV Rank 30-day
    df_renamed['iv_rank_30'] = df_renamed['ticker'].map(
        lambda x: financial_data.get(x, {}).get('iv_rank')
    )
    
    # IV Rank 90-day
    df_renamed['iv_rank_90'] = df_renamed['ticker'].map(
        lambda x: financial_data.get(x, {}).get('iv_90d_rank') 
        or financial_data.get(x, {}).get('iv_metrics', {}).get('rank_90d')
        if isinstance(financial_data.get(x, {}).get('iv_metrics'), dict)
        else None
    )
    
    # IV Recommendation - check multiple locations
    df_renamed['iv_recommendation'] = df_renamed['ticker'].map(
        lambda x: (
            financial_data.get(x, {}).get('iv_recommendation')  # Root level (from options_analyzer)
            or financial_data.get(x, {}).get('iv_strategy', {}).get('recommendation')  # From iv_strategy (from stock info)
            or (financial_data.get(x, {}).get('iv_metrics', {}).get('recommendation')
                if isinstance(financial_data.get(x, {}).get('iv_metrics'), dict) else None)
        )
    )
    
    # Roll Yield (percentage)
    def parse_roll_yield(value):
        """Parse roll_yield from string like '2.5%' to float, or return as-is if already numeric."""
        if value is None:
            return None
        if isinstance(value, (int, float)):
            return float(value)
        if isinstance(value, str):
            # Remove % sign and parse
            cleaned = value.rstrip('%').strip()
            try:
                return float(cleaned)
            except (ValueError, TypeError):
                return None
        return None
    
    df_renamed['roll_yield'] = df_renamed['ticker'].map(
        lambda x: parse_roll_yield(
            financial_data.get(x, {}).get('iv_metrics', {}).get('roll_yield')
            if isinstance(financial_data.get(x, {}).get('iv_metrics'), dict)
            else financial_data.get(x, {}).get('roll_yield')
        )
    )
    
    # Normalize numeric columns
    for col in ['implied_volatility', 'long_implied_volatility', 'long_contracts_available',
                'risk_score', 'iv_rank_30', 'iv_rank_90', 'roll_yield']:
        if col in df_renamed.columns:
            df_renamed[col] = pd.to_numeric(df_renamed[col], errors='coerce')
    
    return df_renamed


def add_derived_percentage_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Add derived percentage columns to dataframe."""
    df_renamed = df.copy()
    
    # Add option premium percentage calculation
    if 'option_premium' in df_renamed.columns and 'current_price' in df_renamed.columns:
        df_renamed['option_premium_percentage'] = (
            df_renamed['option_premium'] / df_renamed['current_price'] * 100
        ).round(2)
    
    # Add premium vs above difference percentage
    if all(col in df_renamed.columns for col in ['option_premium', 'price_above_current']):
        df_renamed['premium_above_diff_percentage'] = (
            (
                (df_renamed['option_premium'] - df_renamed['price_above_current']) 
                / df_renamed['price_above_current']
            ).where(df_renamed['price_above_current'] != 0) * 100
        ).round(2)
    
    return df_renamed


def format_timestamp_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Format timestamp columns for display."""
    df_renamed = df.copy()
    
    # Format latest_opt_ts FIRST as age in seconds (it's a float, not a timestamp)
    if 'latest_opt_ts' in df_renamed.columns:
        df_renamed['latest_opt_ts'] = df_renamed['latest_opt_ts'].apply(
            format_age_seconds
        ).astype(str)
    
    # Convert timestamps to EST/EDT for display
    try:
        if 'expiration_date' in df_renamed.columns:
            ser = df_renamed['expiration_date'].apply(normalize_expiration_date_to_utc)
            df_renamed['expiration_date'] = ser.dt.strftime('%Y-%m-%d %H:%M:%S')
        
        for ts_col in ['last_quote_timestamp', 'write_timestamp']:
            if ts_col in df_renamed.columns:
                ser = df_renamed[ts_col].apply(normalize_timestamp_for_display)
                df_renamed[ts_col] = ser.dt.strftime('%Y-%m-%d %H:%M:%S')
    except Exception:
        # If conversion fails, leave as-is
        pass
    
    return df_renamed


def get_display_columns(is_spread_mode: bool) -> List[str]:
    """Get the list of display columns based on mode."""
    # IV metrics columns to include
    iv_metrics_columns = ['risk_score', 'iv_rank_30', 'iv_rank_90', 'iv_recommendation', 'roll_yield']
    
    if is_spread_mode:
        return [
            'ticker', 'option_type', 'pe_ratio', 'market_cap_b', 'current_price', 
            'price_with_change', 'price_change_pct',
            # IV metrics (for tkr_info column)
            *iv_metrics_columns,
            # Short option details
            'strike_price', 'price_above_current', 'option_premium', 'bid_ask',
            'implied_volatility', 'delta', 'theta', 'expiration_date', 'days_to_expiry',
            'short_premium_total', 'short_daily_premium',
            # Long option details
            'long_strike_price', 'long_option_premium', 'long_bid_ask', 
            'long_implied_volatility', 'long_delta', 'long_theta',
            'long_expiration_date', 'long_days_to_expiry', 'long_premium_total', 
            'long_contracts_available',
            # Premium comparison
            'premium_diff',
            # Net spread calculations
            'net_premium', 'net_daily_premium',
            # Additional details
            'volume', 'num_contracts', 'option_ticker', 'long_option_ticker', 'latest_opt_ts'
        ]
    else:
        return [
            'ticker', 'option_type', 'pe_ratio', 'market_cap_b', 'current_price', 
            'price_with_change', 'price_change_pct',
            # IV metrics (for tkr_info column)
            *iv_metrics_columns,
            'strike_price',
            'price_above_current', 'option_premium', 'bid_ask', 
            'premium_above_diff_percentage',
            'implied_volatility', 'delta', 'theta',
            'potential_premium', 'daily_premium', 'expiration_date', 'days_to_expiry',
            'volume', 'num_contracts', 'option_ticker', 'latest_opt_ts'
        ]


def resolve_sort_key(
    sort_by: str,
    df: pd.DataFrame,
    header_reverse_map: Dict[str, str]
) -> Optional[str]:
    """Resolve sort key from user input to actual column name."""
    sort_key = sort_by
    
    # Check if it's a compact header name
    if sort_by in header_reverse_map:
        sort_key = header_reverse_map[sort_by]
    
    # Try case-insensitive substring match
    if sort_key not in df.columns:
        candidates = [
            c for c in df.columns 
            if sort_key.lower() in str(c).lower()
        ]
        if len(candidates) == 1:
            sort_key = candidates[0]
    
    return sort_key if sort_key in df.columns else None


def apply_sorting(
    df: pd.DataFrame,
    sort_by: Optional[str],
    compact_headers: Dict[str, str]
) -> pd.DataFrame:
    """Apply sorting to dataframe."""
    if not sort_by:
        return df
    
    header_reverse_map = {v: k for k, v in compact_headers.items()}
    sort_key = resolve_sort_key(sort_by, df, header_reverse_map)
    
    if sort_key:
        # Sort premium_diff ascending (lower is better for spreads), others descending
        ascending = (sort_key == 'premium_diff')
        return df.sort_values(by=sort_key, ascending=ascending)
    
    return df


def apply_top_n_filter(
    df: pd.DataFrame,
    top_n: Optional[int]
) -> pd.DataFrame:
    """Apply top-n filter per ticker-option_type combination."""
    if top_n is None or top_n <= 0:
        return df
    
    if 'ticker' not in df.columns:
        return df.head(top_n)
    
    if 'option_type' in df.columns:
        return df.groupby(['ticker', 'option_type'], group_keys=False).head(top_n)
    else:
        return df.groupby('ticker', group_keys=False).head(top_n)


def resolve_csv_columns(
    csv_columns: Optional[List[str]],
    df: pd.DataFrame,
    header_reverse_map: Dict[str, str]
) -> Optional[List[str]]:
    """Resolve CSV column names from user input."""
    if not csv_columns:
        return None
    
    resolved_columns = []
    for requested in csv_columns:
        # Direct match
        if requested in df.columns:
            resolved_columns.append(requested)
            continue
        
        # Compact header name
        if requested in header_reverse_map:
            resolved_columns.append(header_reverse_map[requested])
            continue
        
        # Case-insensitive match against compact headers
        matches = [
            header_reverse_map[h]
            for h in header_reverse_map
            if h.lower() == requested.lower()
        ]
        if len(matches) == 1:
            resolved_columns.append(matches[0])
            continue
        
        # Case-insensitive substring match on original columns
        substring_matches = [
            col for col in df.columns
            if requested.lower() in str(col).lower()
        ]
        if len(substring_matches) == 1:
            resolved_columns.append(substring_matches[0])
            continue
    
    return resolved_columns if resolved_columns else None


def format_table_output(
    df: pd.DataFrame,
    compact_headers: Dict[str, str],
    group_by: str = 'overall'
) -> str:
    """Format dataframe as table output."""
    df_formatted = format_dataframe_for_display(df)
    df_formatted = df_formatted.rename(columns=compact_headers)
    
    if group_by == 'ticker' and 'ticker' in df_formatted.columns:
        output_lines = []
        for ticker in sorted(df_formatted['ticker'].unique()):
            ticker_data = df_formatted[df_formatted['ticker'] == ticker]
            output_lines.append(f"\n--- {ticker} ---")
            output_lines.append(tabulate(
                ticker_data,
                headers='keys',
                tablefmt='grid',
                showindex=False
            ))
        return "\n".join(output_lines)
    else:
        return tabulate(
            df_formatted,
            headers='keys',
            tablefmt='grid',
            showindex=False
        )


