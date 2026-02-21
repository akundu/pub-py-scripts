#!/usr/bin/env python3
"""
Evaluate Covered Calls - Analyze covered call spread opportunities from CSV data.

This script reads CSV data containing covered call spread information and provides
detailed analysis including risk metrics, profitability metrics, and recommendations.

Usage:
    python scripts/evaluate_covered_calls.py --file results.csv
    python scripts/evaluate_covered_calls.py < results.csv
    cat results.csv | python scripts/evaluate_covered_calls.py
    python scripts/evaluate_covered_calls.py --file results.csv --html --output-dir results_html
"""

import pandas as pd
import numpy as np
import argparse
import sys
import os
import textwrap
import re
import logging
from pathlib import Path
from typing import Optional

# Set up logging - use INFO level by default, DEBUG for troubleshooting
logger = logging.getLogger(__name__)

# Add scripts directory to path for imports
scripts_dir = Path(__file__).parent
if str(scripts_dir) not in sys.path:
    sys.path.insert(0, str(scripts_dir))

# Import HTML report generator (using new v2 implementation)
#from html_report_generator import generate_html_output
from html_report_v2 import generate_html_output


def safe_to_numeric(series, col_name=None):
    """Convert series to numeric, handling malformed concatenated strings like '0.120.260.110.210.36'."""
    def extract_first_number(val):
        if pd.isna(val):
            return np.nan
        try:
            val_str = str(val)
            # Try direct conversion first
            try:
                return float(val_str)
            except (ValueError, TypeError):
                # Extract first valid number from string
                match = re.search(r'-?\d+\.?\d*', val_str)
                if match:
                    extracted = float(match.group())
                    if col_name and val_str != str(extracted):
                        logger.debug(f"Column '{col_name}': Extracted {extracted} from malformed string '{val_str}'")
                    return extracted
                if col_name:
                    logger.warning(f"Column '{col_name}': Could not extract number from '{val_str}', using NaN")
                return np.nan
        except (ValueError, TypeError, AttributeError) as e:
            if col_name:
                logger.warning(f"Column '{col_name}': Error processing value '{val}': {e}")
            return np.nan
    
    # Apply safe conversion and ensure numeric dtype
    logger.debug(f"Converting column '{col_name}' to numeric (dtype: {series.dtype})")
    result = series.apply(extract_first_number)
    # Ensure the result is actually numeric dtype (not object)
    result = pd.to_numeric(result, errors='coerce')
    logger.debug(f"Column '{col_name}' converted to dtype: {result.dtype}")
    return result


def wrap_dataframe_columns(df: pd.DataFrame, width: int = 15) -> pd.DataFrame:
    """Return a copy of the DataFrame with column headers wrapped to the specified width."""
    wrapped_df = df.copy()
    wrapped_columns = {}
    for col in wrapped_df.columns:
        col_name = str(col).replace('_', ' ')
        wrapped_lines = textwrap.wrap(
            col_name,
            width=width,
            break_long_words=True,
            break_on_hyphens=False
        )
        wrapped_columns[col] = "\n".join(line.strip() for line in wrapped_lines) if wrapped_lines else col_name
    wrapped_df = wrapped_df.rename(columns=wrapped_columns)
    return wrapped_df


def load_data(file_path: Optional[str] = None) -> pd.DataFrame:
    """Load CSV data from file or stdin.
    
    Args:
        file_path: Path to CSV file, or None to read from stdin
        
    Returns:
        DataFrame with loaded and cleaned data
    """
    if file_path is None or file_path == '-':
        # Read from stdin
        df = pd.read_csv(sys.stdin)
    else:
        # Read from file
        df = pd.read_csv(file_path)
    
    # Clean up the data - remove duplicate header rows
    df = df[df['ticker'] != 'ticker']
    
    # Convert numeric columns
    # Note: option_ticker, l_option_ticker, expiration_date, l_expiration_date, price_with_change are strings, not numeric
    numeric_cols = [
        'pe_ratio','market_cap_b','curr_price','current_price','strike_price','price_above_curr','opt_prem.','IV','delta','theta','days_to_expiry','s_prem_tot','s_day_prem','l_strike','l_prem','liv','l_delta','l_theta','l_days_to_expiry','l_prem_tot','l_cnt_avl','prem_diff','net_premium','net_daily_premi','volume','num_contracts','price_change_pct','premium_ratio_pct'
    ]
    
    for col in numeric_cols:
        if col in df.columns:
            logger.debug(f"Processing column '{col}' (current dtype: {df[col].dtype})")
            # Check for malformed values before conversion - even if dtype is numeric, there might be string values
            if df[col].dtype == 'object':
                sample_values = df[col].dropna().head(5).tolist()
                logger.debug(f"  Sample values: {sample_values}")
                # Check if any values look malformed (contain multiple dots in a row)
                malformed = df[col].astype(str).str.contains(r'\d+\.\d+\.', regex=True, na=False)
                if malformed.any():
                    malformed_values = df[col][malformed].unique()[:3]
                    logger.warning(f"  Found {malformed.sum()} malformed values in '{col}': {malformed_values.tolist()}")
            else:
                # Even if dtype is numeric, check for string values that might have slipped through
                string_mask = df[col].astype(str).str.contains(r'\d+\.\d+\.', regex=True, na=False)
                if string_mask.any():
                    logger.warning(f"  Found {string_mask.sum()} malformed string values in numeric column '{col}'")
                    malformed_values = df[col][string_mask].unique()[:3]
                    logger.warning(f"  Malformed values: {malformed_values.tolist()}")
                    # Force conversion to object first, then convert safely
                    df[col] = df[col].astype(str)
            
            # Use safe conversion for columns that might have malformed data
            if col in ['net_daily_premi', 'net_daily_premium', 'net_premium', 's_prem_tot', 's_day_prem', 'l_prem_tot']:
                try:
                    df[col] = safe_to_numeric(df[col], col_name=col)
                    # Double-check: ensure no string values remain
                    if df[col].dtype == 'object' or df[col].astype(str).str.contains(r'\d+\.\d+\.', regex=True, na=False).any():
                        logger.warning(f"  Column '{col}' still contains malformed values after conversion, retrying...")
                        df[col] = safe_to_numeric(df[col], col_name=col)
                except Exception as e:
                    logger.error(f"Error converting column '{col}' to numeric: {e}")
                    logger.error(f"  Column dtype: {df[col].dtype}")
                    logger.error(f"  Sample values: {df[col].dropna().head(3).tolist()}")
                    raise
            else:
                try:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                except Exception as e:
                    logger.error(f"Error converting column '{col}' to numeric: {e}")
                    logger.error(f"  Column dtype: {df[col].dtype}")
                    logger.error(f"  Sample values: {df[col].dropna().head(3).tolist()}")
                    # Try safe conversion as fallback
                    logger.info(f"  Attempting safe conversion for '{col}'")
                    df[col] = safe_to_numeric(df[col], col_name=col)
    
    # Final check: verify net_daily_premi is truly numeric
    if 'net_daily_premi' in df.columns:
        if df['net_daily_premi'].dtype == 'object':
            logger.error("net_daily_premi is still object dtype after conversion!")
            logger.error(f"  Sample values: {df['net_daily_premi'].dropna().head(5).tolist()}")
            # Force conversion one more time
            df['net_daily_premi'] = safe_to_numeric(df['net_daily_premi'], col_name='net_daily_premi')
        # Check for any remaining string values
        string_check = df['net_daily_premi'].astype(str).str.contains(r'\d+\.\d+\.', regex=True, na=False)
        if string_check.any():
            logger.error(f"Found {string_check.sum()} malformed values still in net_daily_premi after conversion!")
            logger.error(f"  Values: {df['net_daily_premi'][string_check].unique()[:5].tolist()}")
            # Force fix these values
            def fix_malformed(val):
                if pd.isna(val):
                    return np.nan
                val_str = str(val)
                if re.search(r'\d+\.\d+\.', val_str):
                    match = re.search(r'-?\d+\.?\d*', val_str)
                    if match:
                        return float(match.group())
                return val
            df['net_daily_premi'] = df['net_daily_premi'].apply(fix_malformed)
            df['net_daily_premi'] = pd.to_numeric(df['net_daily_premi'], errors='coerce')
    
    return df


def calculate_bid_ask_analysis(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate bid/ask spread analysis and scoring metrics.
    
    MATH EXPLANATIONS:
    
    1. SPREAD SLIPPAGE:
       - short_spread = short_ask - short_bid (per contract)
       - long_spread = long_ask - long_bid (per contract)
       - spread_slippage = (short_spread + long_spread) * num_contracts
       - This represents the total cost of bid/ask spreads across all contracts
    
    2. LIQUIDITY SCORE (0-10 points):
       - Very tight short spread (<3%): +2 points
       - Acceptable short spread (<10%): +1 point
       - Very tight long spread (<3%): +2 points
       - Acceptable long spread (<10%): +1 point
       - High volume (>1000): +2 points
       - Decent volume (>200): +1 point
       - Good open interest (>200): +1 point
       - Maximum: 10 points
    
    3. TRADE QUALITY SCORE:
       - Liquidity Score * 2 (0-20 points) - Liquidity is CRITICAL
       - (10 - Assignment Risk) (4-10 points) - Lower risk is better
       - (net_daily_premium_after_spread / 100) (variable points) - Premium contribution
       - Reasonable valuation bonus (P/E < 25): +2 points
    
    Args:
        df: DataFrame with loaded data
        
    Returns:
        DataFrame with additional bid/ask analysis columns
    """
    df = df.copy()
    
    # Parse bid/ask spreads for short position (temporary for calculations only)
    short_spread = pd.Series(0.0, index=df.index)
    short_spread_pct = pd.Series(np.nan, index=df.index)
    if 'bid:ask' in df.columns:
        bid_ask_split = df['bid:ask'].str.split(':', expand=True)
        if len(bid_ask_split.columns) >= 2:
            short_bid = pd.to_numeric(bid_ask_split[0], errors='coerce')
            short_ask = pd.to_numeric(bid_ask_split[1], errors='coerce')
            short_spread = short_ask - short_bid
            short_spread_pct = (short_spread / df['opt_prem.'] * 100).round(2)
            short_spread_pct = short_spread_pct.replace([np.inf, -np.inf], np.nan)
    
    # Parse bid/ask spreads for long position (temporary for calculations only)
    long_spread = pd.Series(0.0, index=df.index)
    long_spread_pct = pd.Series(np.nan, index=df.index)
    if 'l_bid:ask' in df.columns:
        l_bid_ask_split = df['l_bid:ask'].str.split(':', expand=True)
        if len(l_bid_ask_split.columns) >= 2:
            long_bid = pd.to_numeric(l_bid_ask_split[0], errors='coerce')
            long_ask = pd.to_numeric(l_bid_ask_split[1], errors='coerce')
            long_spread = long_ask - long_bid
            long_spread_pct = (long_spread / df['l_prem'] * 100).round(2)
            long_spread_pct = long_spread_pct.replace([np.inf, -np.inf], np.nan)
    
    # Calculate real costs after spreads
    # spread_slippage = (short_spread + long_spread) * num_contracts
    df['spread_slippage'] = ((short_spread.fillna(0) + long_spread.fillna(0)) * 
                             df.get('num_contracts', 0))
    df['net_premium_after_spread'] = df['net_premium'] - df['spread_slippage']
    # Safely calculate net_daily_premium_after_spread, ensuring numeric types
    try:
        net_prem_after = pd.to_numeric(df['net_premium_after_spread'], errors='coerce')
        days_exp = pd.to_numeric(df['days_to_expiry'], errors='coerce')
        # Handle different cases:
        # - If days < 0 (expired): daily premium is 0.0
        # - If days = 0 (0DTE): daily premium = net_premium_after_spread (full premium earned today)
        # - If days > 0: daily premium = net_premium_after_spread / days
        df['net_daily_premium_after_spread'] = pd.Series([
            0.0 if (pd.isna(days) or days < 0)  # Expired
            else (round(net_prem, 2) if pd.notna(net_prem) else 0.0) if days == 0  # 0DTE
            else round(net_prem / days, 2) if pd.notna(net_prem) else 0.0  # Future dates
            for net_prem, days in zip(net_prem_after, days_exp)
        ], index=df.index)
    except Exception as e:
        logger.error(f"Error calculating net_daily_premium_after_spread: {e}")
        logger.error(f"  net_premium_after_spread dtype: {df['net_premium_after_spread'].dtype}")
        logger.error(f"  days_to_expiry dtype: {df['days_to_expiry'].dtype}")
        raise
    df['net_daily_premium_after_spread'] = df['net_daily_premium_after_spread'].replace(
        [np.inf, -np.inf], np.nan)
    
    # Calculate spread impact percentage - ensure numeric types first
    try:
        spread_slip = pd.to_numeric(df['spread_slippage'], errors='coerce')
        net_prem = pd.to_numeric(df['net_premium'], errors='coerce')
        df['spread_impact_pct'] = (spread_slip / net_prem * 100).round(2)
        df['spread_impact_pct'] = df['spread_impact_pct'].replace([np.inf, -np.inf], np.nan)
        # Ensure the result is numeric
        df['spread_impact_pct'] = pd.to_numeric(df['spread_impact_pct'], errors='coerce')
    except Exception as e:
        logger.error(f"Error calculating spread_impact_pct: {e}")
        logger.error(f"  spread_slippage dtype: {df['spread_slippage'].dtype}")
        logger.error(f"  net_premium dtype: {df['net_premium'].dtype}")
        raise
    
    # Liquidity Scoring (0-10) - using temporary spread calculations
    # Ensure all values are numeric before comparisons
    short_spread_pct_num = pd.to_numeric(short_spread_pct, errors='coerce').fillna(999)
    long_spread_pct_num = pd.to_numeric(long_spread_pct, errors='coerce').fillna(999)
    volume_num = pd.to_numeric(df.get('volume', 0), errors='coerce').fillna(0)
    num_contracts_num = pd.to_numeric(df.get('num_contracts', 0), errors='coerce').fillna(0)
    
    df['liquidity_score'] = (
        (short_spread_pct_num < 3).astype(int) * 2 +    # Very tight short spread
        (short_spread_pct_num < 10).astype(int) * 1 +   # Acceptable short spread
        (long_spread_pct_num < 3).astype(int) * 2 +    # Very tight long spread
        (long_spread_pct_num < 10).astype(int) * 1 +   # Acceptable long spread
        (volume_num > 1000).astype(int) * 2 +            # High volume
        (volume_num > 200).astype(int) * 1 +             # Decent volume
        (num_contracts_num > 200).astype(int) * 1       # Good open interest
    )
    
    # Assignment Risk Scoring (0-6, lower is better)
    # Ensure all values are numeric before comparisons
    delta_num = pd.to_numeric(df.get('delta', 0), errors='coerce').fillna(0)
    price_above_num = pd.to_numeric(df.get('price_above_curr', 999), errors='coerce').fillna(999)
    pe_ratio_num = pd.to_numeric(df.get('pe_ratio', 0), errors='coerce').fillna(0)
    
    df['assignment_risk'] = (
        (delta_num > 0.4).astype(int) * 2 +              # High delta = higher assignment risk
        (price_above_num < 2).astype(int) * 2 +   # Close to strike = risky
        (pe_ratio_num > 40).astype(int) * 2               # Expensive valuation = risky
    )
    
    # Overall Trade Quality Score
    # Ensure all values are numeric before calculations
    liquidity_score_num = pd.to_numeric(df['liquidity_score'], errors='coerce').fillna(0)
    assignment_risk_num = pd.to_numeric(df['assignment_risk'], errors='coerce').fillna(0)
    net_daily_after_num = pd.to_numeric(df['net_daily_premium_after_spread'], errors='coerce').fillna(0)
    pe_ratio_quality_num = pd.to_numeric(df.get('pe_ratio', 999), errors='coerce').fillna(999)
    
    df['trade_quality'] = (
        liquidity_score_num * 2 +                                # Liquidity is CRITICAL (0-20 points)
        (10 - assignment_risk_num) +                             # Lower risk is better (4-10 points)
        (net_daily_after_num / 100) +   # Premium contribution (variable)
        (pe_ratio_quality_num < 25).astype(int) * 2              # Reasonable valuation bonus (0-2 points)
    )
    
    # Calculate premium ratio percentage: (short_premium / long_premium) * 100
    # This shows what percentage of the long hedge cost is covered by the short premium received
    # Lower values are better (less cost to hedge, more efficient spread)
    try:
        s_prem_tot_num = pd.to_numeric(df.get('s_prem_tot', 0), errors='coerce')
        l_prem_tot_num = pd.to_numeric(df.get('l_prem_tot', 0), errors='coerce')
        df['premium_ratio_pct'] = (s_prem_tot_num / l_prem_tot_num * 100).round(2)
        df['premium_ratio_pct'] = df['premium_ratio_pct'].replace([np.inf, -np.inf], np.nan)
        # Ensure the result is numeric
        df['premium_ratio_pct'] = pd.to_numeric(df['premium_ratio_pct'], errors='coerce')
    except Exception as e:
        logger.error(f"Error calculating premium_ratio_pct: {e}")
        logger.error(f"  s_prem_tot dtype: {df.get('s_prem_tot', pd.Series()).dtype if 's_prem_tot' in df.columns else 'N/A'}")
        logger.error(f"  l_prem_tot dtype: {df.get('l_prem_tot', pd.Series()).dtype if 'l_prem_tot' in df.columns else 'N/A'}")
        # Set to NaN if calculation fails
        df['premium_ratio_pct'] = np.nan
    
    return df


def print_top_20_analysis(df: pd.DataFrame) -> None:
    """Print analysis for top 20 performers by net daily premium."""
    # Ensure net_daily_premi is numeric before sorting
    if 'net_daily_premi' in df.columns:
        df = df.copy()
        if not pd.api.types.is_numeric_dtype(df['net_daily_premi']):
            df['_net_daily_premi_numeric'] = safe_to_numeric(df['net_daily_premi'])
            df_valid = df[df['_net_daily_premi_numeric'].notna()]
            if len(df_valid) > 0:
                top_20 = df_valid.nlargest(20, '_net_daily_premi_numeric').drop(columns=['_net_daily_premi_numeric'])
            else:
                top_20 = df.head(20).drop(columns=['_net_daily_premi_numeric'], errors='ignore')
        else:
            top_20 = df.nlargest(20, 'net_daily_premi')
    else:
        top_20 = df.head(20)
    
    print("=" * 120)
    print("DEEP DIVE: TOP 20 COVERED CALL SPREADS BY NET DAILY PREMIUM")
    print("=" * 120)
    
    # Create comprehensive analysis
    analysis = top_20[['ticker', 'curr_price', 'strike_price', 'net_daily_premi', 
                       'volume', 'pe_ratio', 'market_cap_b']].copy()
    
    # Calculate key risk metrics
    analysis['moneyness'] = ((top_20['strike_price'] - top_20['curr_price']) / top_20['curr_price'] * 100).round(2)
    analysis['short_delta'] = top_20['delta']
    analysis['l_delta'] = top_20['l_delta']
    analysis['delta_diff'] = (top_20['l_delta'] - top_20['delta']).round(3)
    analysis['theta_decay_daily'] = top_20['theta']
    # analysis['premium_capture_pct'] = top_20['option_premium_percentage']
    analysis['spread_width'] = top_20['l_strike'] - top_20['strike_price']
    analysis['net_premium'] = top_20['net_premium']
    analysis['roi_on_spread'] = ((top_20['net_premium'] / 100000) * 100).round(2)
    
    print("\n### RISK METRICS OVERVIEW ###\n")
    risk_df = analysis[['ticker', 'moneyness', 'short_delta', 'l_delta', 'delta_diff',
                        'theta_decay_daily', 'volume']]
    print(wrap_dataframe_columns(risk_df).to_string(index=False))
    
    print("\n\n### PROFITABILITY METRICS ###\n")
    profitability_df = analysis[['ticker', 'net_daily_premi', 'net_premium', 'roi_on_spread',
                                # 'premium_capture_pct', 
                                 'spread_width']]
    print(wrap_dataframe_columns(profitability_df).to_string(index=False))
    
    print("\n\n### FUNDAMENTAL QUALITY ###\n")
    fundamentals_df = analysis[['ticker', 'pe_ratio', 'market_cap_b', 'curr_price']]
    print(wrap_dataframe_columns(fundamentals_df).to_string(index=False))


def print_detailed_analysis(df: pd.DataFrame) -> None:
    """Print unified detailed analysis for top 10 picks incorporating all parameters."""
    print("\n\n" + "=" * 120)
    print("COMPREHENSIVE ANALYSIS: TOP 10 PICKS WITH OPTION TICKERS")
    print("=" * 120)
    
    # Ensure net_daily_premi is numeric before sorting
    if 'net_daily_premi' in df.columns:
        df = df.copy()
        if not pd.api.types.is_numeric_dtype(df['net_daily_premi']):
            df['_net_daily_premi_numeric'] = safe_to_numeric(df['net_daily_premi'])
            df_valid = df[df['_net_daily_premi_numeric'].notna()]
            if len(df_valid) > 0:
                top_10 = df_valid.nlargest(10, '_net_daily_premi_numeric').drop(columns=['_net_daily_premi_numeric'])
            else:
                top_10 = df.head(10).drop(columns=['_net_daily_premi_numeric'], errors='ignore')
        else:
            top_10 = df.nlargest(10, 'net_daily_premi')
    else:
        top_10 = df.head(10)
    
    for idx, row in top_10.iterrows():
        ticker = row['ticker']
        print(f"\n{'=' * 110}")
        print(f"#{top_10.index.get_loc(idx) + 1}: {ticker}")
        print(f"{'=' * 110}")
        
        # Position Details
        print(f"\n📊 POSITION STRUCTURE:")
        print(f"   Current Price: ${row['curr_price']:.2f}")
        print(f"   Short Strike: ${row['strike_price']:.2f} ({((row['strike_price']-row['curr_price'])/row['curr_price']*100):.2f}% OTM)")
        print(f"   Long Strike: ${row['l_strike']:.2f}")
        print(f"   Spread Width: ${row['l_strike'] - row['strike_price']:.2f}")
        
        # Option Tickers - HIGHLIGHTED
        print(f"\n🎯 OPTION TICKERS:")
        bid_ask_short = row.get('bid:ask', 'N/A:N/A')
        bid_ask_long = row.get('l_bid:ask', 'N/A:N/A')
        print(f"   ┌─ SHORT (SELL): {row['option_ticker']}")
        print(f"   │  Strike: ${row['strike_price']:.2f} | Expiry: {row['expiration_date'][:10]} ({int(row['days_to_expiry'])} DTE)")
        print(f"   │  Premium: ${row['opt_prem.']:.2f} per contract | Bid:Ask: {bid_ask_short}")
        print(f"   │  Total Credit: ${row['s_prem_tot']:,.0f} ({int(row['num_contracts'])} contracts)")
        print(f"   │")
        print(f"   └─ LONG (BUY):  {row['l_option_ticker']}")
        print(f"      Strike: ${row['l_strike']:.2f} | Expiry: {row['l_expiration_date'][:10]} ({int(row['l_days_to_expiry'])} DTE)")
        print(f"      Premium: ${row['l_prem']:.2f} per contract | Bid:Ask: {bid_ask_long}")
        print(f"      Total Debit: ${row['l_prem_tot']:,.0f} ({int(row['num_contracts'])} contracts)")
        
        # Premium Analysis
        print(f"\n💰 PREMIUM BREAKDOWN:")
        print(f"   Short Premium: ${row['s_prem_tot']:,.0f} (${row['s_day_prem']:,.0f}/day)")
        print(f"   Long Premium: ${row['l_prem_tot']:,.0f}")
        print(f"   Net Credit: ${row['net_premium']:,.0f}")
        print(f"   Daily Income: ${row['net_daily_premi']:,.2f}")
        print(f"   ROI on $100k: {(row['net_premium']/100000*100):.2f}%")
        
        # Bid/Ask & Spread Analysis (if available)
        if 'spread_impact_pct' in row and pd.notna(row.get('spread_impact_pct')):
            spread_slippage = row.get('spread_slippage', 0)
            net_after_spread = row.get('net_premium_after_spread', row['net_premium'])
            net_daily_after = row.get('net_daily_premium_after_spread', row['net_daily_premi'])
            spread_impact = row.get('spread_impact_pct', 0)
            
            if 'liquidity_score' in row and pd.notna(row.get('liquidity_score')):
                liquidity_score = row.get('liquidity_score', 0)
                assignment_risk = row.get('assignment_risk', 0)
                trade_quality = row.get('trade_quality', 0)
                print(f"\n💱 SPREAD & LIQUIDITY ANALYSIS:")
                print(f"   Spread Slippage: ${spread_slippage:,.0f}")
                print(f"   Net Premium After Spread: ${net_after_spread:,.0f}")
                print(f"   Daily Income After Spread: ${net_daily_after:,.2f}")
                print(f"   Spread Impact: {spread_impact:.2f}%")
                print(f"   Liquidity Score: {liquidity_score:.0f}/10")
                print(f"   Assignment Risk: {assignment_risk:.0f}/6")
                print(f"   Trade Quality Score: {trade_quality:.1f}")
        
        # Greeks & Risk
        print(f"\n📈 GREEKS & RISK:")
        print(f"   Short Delta: {row['delta']:.2f} | Long Delta: {row['l_delta']:.2f} | Net: {row['l_delta']-row['delta']:.3f}")
        print(f"   Short Theta: {row['theta']:.2f} | Long Theta: {row['l_theta']:.2f}")
        print(f"   Assignment Risk: {'LOW' if row['delta'] < 0.35 else 'MODERATE' if row['delta'] < 0.50 else 'HIGH'}")
        
        # Liquidity & Fundamentals
        print(f"\n🔄 LIQUIDITY & FUNDAMENTALS:")
        print(f"   Volume: {row['volume']:,.0f} contracts")
        print(f"   Num Contracts: {row['num_contracts']:.0f}")
        print(f"   P/E Ratio: {row['pe_ratio']:.2f}")
        print(f"   Market Cap: ${row['market_cap_b']:.2f}B")
        
        # Risk Assessment
        print(f"\n⚠️  RISK ASSESSMENT:")
        
        # Assignment risk
        if row['delta'] < 0.35:
            assignment_risk = "LOW - Strike is well OTM"
        elif row['delta'] < 0.50:
            assignment_risk = "MODERATE - Near ATM, watch closely"
        else:
            assignment_risk = "HIGH - ITM or very close, likely assignment"
        print(f"   Assignment Risk: {assignment_risk}")
        
        # Liquidity risk
        if row['volume'] > 1000:
            liquidity = "EXCELLENT - Very liquid"
        elif row['volume'] > 300:
            liquidity = "GOOD - Adequate liquidity"
        elif row['volume'] > 100:
            liquidity = "FAIR - May have wider spreads"
        else:
            liquidity = "POOR - Low liquidity, watch bid-ask"
        print(f"   Liquidity: {liquidity}")
        
        # Valuation
        if row['pe_ratio'] < 15:
            valuation = "ATTRACTIVE - Trading at discount"
        elif row['pe_ratio'] < 25:
            valuation = "FAIR - Reasonably valued"
        elif row['pe_ratio'] < 50:
            valuation = "ELEVATED - Premium valuation"
        else:
            valuation = "EXPENSIVE - Very high P/E"
        print(f"   Valuation: {valuation}")
        
        # Overall score
        score = 0
        if row['net_daily_premi'] > 10000: score += 3
        elif row['net_daily_premi'] > 7000: score += 2
        else: score += 1
        
        if row['volume'] > 1000: score += 3
        elif row['volume'] > 300: score += 2
        elif row['volume'] > 100: score += 1
        
        if row['delta'] < 0.35: score += 3
        elif row['delta'] < 0.50: score += 2
        else: score += 1
        
        if row['pe_ratio'] < 25: score += 2
        elif row['pe_ratio'] < 50: score += 1
        
        print(f"\n⭐ OVERALL SCORE: {score}/11")
        
        if score >= 9:
            recommendation = "STRONG BUY - Excellent risk/reward"
        elif score >= 7:
            recommendation = "BUY - Good opportunity"
        elif score >= 5:
            recommendation = "HOLD - Acceptable but monitor"
        else:
            recommendation = "PASS - Better opportunities available"
        
        print(f"   Recommendation: {recommendation}")


def print_bid_ask_analysis(df: pd.DataFrame) -> None:
    """Print summary statistics and top recommendations."""
    print("\n\n" + "=" * 120)
    print("SUMMARY STATISTICS & TOP RECOMMENDATIONS")
    print("=" * 120)
    
    # Check if required columns exist
    if 'spread_impact_pct' not in df.columns:
        print("\n⚠️  Spread analysis columns not found. Skipping summary statistics.")
        return
    
    print(f"\n📊 SPREAD IMPACT STATISTICS:")
    spread_impact_mean = df['spread_impact_pct'].mean()
    spread_impact_median = df['spread_impact_pct'].median()
    low_impact_count = (df['spread_impact_pct'] < 5).sum()
    high_impact_count = (df['spread_impact_pct'] > 10).sum()
    
    print(f"   Average spread impact: {spread_impact_mean:.2f}%")
    print(f"   Median spread impact: {spread_impact_median:.2f}%")
    print(f"   Trades with <5% impact: {low_impact_count}/{len(df)}")
    print(f"   Trades with >10% impact: {high_impact_count}/{len(df)} ⚠️")
    
    print(f"\n💧 LIQUIDITY DISTRIBUTION:")
    if 'liquidity_score' in df.columns:
        high_liquidity = (df['liquidity_score'] >= 7).sum()
        medium_liquidity = ((df['liquidity_score'] >= 4) & (df['liquidity_score'] < 7)).sum()
        low_liquidity = (df['liquidity_score'] < 4).sum()
        print(f"   High liquidity (7-10): {high_liquidity} trades")
        print(f"   Medium liquidity (4-6): {medium_liquidity} trades")
        print(f"   Low liquidity (<4): {low_liquidity} trades ⚠️")
    
    # Top 5 recommendations by trade quality
    print("\n" + "=" * 120)
    print("🏆 TOP 5 RECOMMENDED TRADES (By Trade Quality Score)")
    print("=" * 120)
    
    if 'trade_quality' in df.columns:
        # Ensure trade_quality is numeric before sorting
        if not pd.api.types.is_numeric_dtype(df['trade_quality']):
            logger.warning("trade_quality is not numeric, converting...")
            df['trade_quality'] = pd.to_numeric(df['trade_quality'], errors='coerce')
        top5 = df.nlargest(5, 'trade_quality')
    else:
        top5 = df.head(5)
        for i, (idx, row) in enumerate(top5.iterrows(), 1):
            print(f"\n#{i}. {row['ticker']} - Quality Score: {row['trade_quality']:.1f}")
            
            net_daily_after = row.get('net_daily_premium_after_spread', 0)
            spread_impact = row.get('spread_impact_pct', 0)
            liquidity = row.get('liquidity_score', 0)
            assignment_risk = row.get('assignment_risk', 0)
            
            print(f"   💰 Daily Premium: ${net_daily_after:.2f} (after {spread_impact:.1f}% spread cost)")
            print(f"   📊 Liquidity: {liquidity:.0f}/10 | Volume: {row.get('volume', 0):.0f} | OI: {row.get('num_contracts', 0):.0f}")
            print(f"   ⚠️  Risk: {assignment_risk:.0f}/6 | P/E: {row.get('pe_ratio', 0):.2f} | Delta: {row.get('delta', 0):.2f}")
            
            if 'option_ticker' in row:
                bid_ask_short = row.get('bid:ask', 'N/A:N/A')
                print(f"   📉 SHORT: {row['option_ticker']} @ ${row.get('strike_price', 0):.2f} | Bid:Ask: {bid_ask_short}")
            
            if 'l_option_ticker' in row:
                bid_ask_long = row.get('l_bid:ask', 'N/A:N/A')
                print(f"   📈 LONG:  {row['l_option_ticker']} @ ${row.get('l_strike', 0):.2f} | Bid:Ask: {bid_ask_long}")


def print_summary_rankings(df: pd.DataFrame) -> None:
    """Print summary rankings and top recommendations."""
    print("\n\n" + "=" * 120)
    print("SUMMARY RANKINGS")
    print("=" * 120)
    
    # Ensure net_daily_premi is numeric before sorting
    if 'net_daily_premi' in df.columns:
        df = df.copy()
        if not pd.api.types.is_numeric_dtype(df['net_daily_premi']):
            df['_net_daily_premi_numeric'] = safe_to_numeric(df['net_daily_premi'])
            df_valid = df[df['_net_daily_premi_numeric'].notna()]
            if len(df_valid) > 0:
                top_10 = df_valid.nlargest(10, '_net_daily_premi_numeric').drop(columns=['_net_daily_premi_numeric'])
            else:
                top_10 = df.head(10).drop(columns=['_net_daily_premi_numeric'], errors='ignore')
        else:
            top_10 = df.nlargest(10, 'net_daily_premi')
    else:
        top_10 = df.head(10)
    
    # Create final ranking
    ranking = top_10[['ticker', 'net_daily_premi', 'volume', 'delta', 'pe_ratio', 
                      'option_ticker', 'l_option_ticker']].copy()
    ranking['liquidity_score'] = ranking['volume'].apply(lambda x: 3 if x > 1000 else 2 if x > 300 else 1)
    ranking['delta_score'] = ranking['delta'].apply(lambda x: 3 if x < 0.35 else 2 if x < 0.50 else 1)
    ranking['premium_score'] = ranking['net_daily_premi'].apply(lambda x: 3 if x > 10000 else 2 if x > 7000 else 1)
    ranking['pe_score'] = ranking['pe_ratio'].apply(lambda x: 2 if x < 25 else 1 if x < 50 else 0)
    ranking['total_score'] = ranking['liquidity_score'] + ranking['delta_score'] + ranking['premium_score'] + ranking['pe_score']
    
    ranking = ranking.sort_values('total_score', ascending=False)
    
    print("\nFINAL RANKINGS (by composite score):\n")
    ranking_df = ranking[['ticker', 'net_daily_premi', 'volume', 'delta', 'total_score']]
    print(wrap_dataframe_columns(ranking_df).to_string(index=False))
    
    print("\n\n" + "=" * 120)
    print("🏆 TOP 3 RECOMMENDATIONS WITH TRADE DETAILS")
    print("=" * 120)
    
    top_3 = ranking.head(3)
    for i, (idx, row) in enumerate(top_3.iterrows(), 1):
        ticker_data = df[df['ticker'] == row['ticker']].iloc[0]
        
        print(f"\n{'─' * 110}")
        print(f"#{i}. {row['ticker']} - Score: {row['total_score']}/11 | Daily Premium: ${row['net_daily_premi']:,.2f} | Volume: {row['volume']:,.0f}")
        print(f"{'─' * 110}")
        print(f"\n   📍 CURRENT PRICE: ${ticker_data['curr_price']:.2f}")
        bid_ask_short = ticker_data.get('bid:ask', 'N/A:N/A')
        bid_ask_long = ticker_data.get('l_bid:ask', 'N/A:N/A')
        print(f"\n   🔴 SELL (SHORT):  {row['option_ticker']}")
        print(f"      Strike: ${ticker_data['strike_price']:.2f} | Exp: {ticker_data['expiration_date'][:10]} | Premium: ${ticker_data['opt_prem.']:.2f} | Bid:Ask: {bid_ask_short}")
        print(f"      Contracts: {int(ticker_data['num_contracts'])} | Total Credit: ${ticker_data['s_prem_tot']:,.0f}")
        print(f"\n   🟢 BUY (LONG):    {row['l_option_ticker']}")
        print(f"      Strike: ${ticker_data['l_strike']:.2f} | Exp: {ticker_data['l_expiration_date'][:10]} | Premium: ${ticker_data['l_prem']:.2f} | Bid:Ask: {bid_ask_long}")
        print(f"      Contracts: {int(ticker_data['num_contracts'])} | Total Debit: ${ticker_data['l_prem_tot']:,.0f}")
        print(f"\n   💵 NET POSITION: ${ticker_data['net_premium']:,.0f} credit ({(ticker_data['net_premium']/100000*100):.1f}% ROI)")
        print(f"   📊 RISK: Delta {ticker_data['delta']:.2f} | Volume {ticker_data['volume']:,.0f} | P/E {ticker_data['pe_ratio']:.1f}")
    
    print("\n\n" + "=" * 120)
    print("📋 QUICK REFERENCE: ALL TOP 10 OPTION TICKERS")
    print("=" * 120)
    
    print("\n{:<8} {:<30} {:<30} {:<12}".format("TICKER", "SHORT (SELL)", "LONG (BUY)", "DAILY $"))
    print("─" * 110)
    
    for idx, row in ranking.iterrows():
        ticker_data = df[df['ticker'] == row['ticker']].iloc[0]
        print("{:<8} {:<30} {:<30} ${:>10,.2f}".format(
            row['ticker'],
            row['option_ticker'],
            row['l_option_ticker'],
            row['net_daily_premi']
        ))
    
    print("\n" + "=" * 120)


def main():
    """Main function to run the covered calls evaluation."""
    parser = argparse.ArgumentParser(
        description="Evaluate covered call spread opportunities from CSV data",
        epilog="""
Examples:
  python scripts/evaluate_covered_calls.py --file results.csv
  python scripts/evaluate_covered_calls.py < results.csv
  cat results.csv | python scripts/evaluate_covered_calls.py
  python scripts/evaluate_covered_calls.py --file -
  python scripts/evaluate_covered_calls.py --file results.csv --html --output-dir results_html
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        '--file', '-f',
        type=str,
        default=None,
        help="Path to CSV file to process. Use '-' or omit to read from stdin (default: stdin)"
    )
    
    parser.add_argument(
        '--html',
        action='store_true',
        help="Generate HTML output with sortable table"
    )
    
    parser.add_argument(
        '--output-dir', '-o',
        type=str,
        default='html_output',
        help="Directory path for HTML output (default: html_output)"
    )
    
    
    parser.add_argument(
        '--debug',
        action='store_true',
        help="Enable debug logging"
    )
    
    args = parser.parse_args()
    
    # Set up logging based on debug flag
    if args.debug:
        logging.basicConfig(level=logging.DEBUG, format='%(levelname)s: %(message)s')
    else:
        logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    
    # Determine input source
    file_path = args.file
    
    try:
        # Load data
        df = load_data(file_path)
        
        if df.empty:
            print("Error: No data loaded. Check your input file or stdin.", file=sys.stderr)
            sys.exit(1)
        
        # Calculate bid/ask analysis
        df = calculate_bid_ask_analysis(df)
        
        # Generate HTML output if requested
        if args.html:
            try:
                logger.info("Generating HTML output...")
                
                # Determine CSV source (file path or URL)
                # Use the file path if provided, otherwise default location
                csv_source = file_path if file_path and file_path != '-' else str(Path.home() / "Downloads" / "results.csv")
                
                generate_html_output(
                    df, 
                    args.output_dir, 
                    csv_source=csv_source
                )
                logger.info(f"HTML output generated successfully in {args.output_dir}")
            except Exception as e:
                logger.error(f"Error generating HTML output: {e}")
                logger.error(f"  Error type: {type(e).__name__}")
                import traceback
                logger.error(f"  Traceback:\n{traceback.format_exc()}")
                raise
        else:
            # Print analyses to stdout
            print_top_20_analysis(df)
            print_detailed_analysis(df)
            print_bid_ask_analysis(df)
            print_summary_rankings(df)
        
    except FileNotFoundError:
        print(f"Error: File not found: {file_path}", file=sys.stderr)
        sys.exit(1)
    except pd.errors.EmptyDataError:
        print("Error: Input data is empty.", file=sys.stderr)
        sys.exit(1)
    except KeyError as e:
        print(f"Error: Required column missing: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
