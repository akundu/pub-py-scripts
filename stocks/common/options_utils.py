"""
Options-specific utility functions for processing and analyzing options data.
"""

import sys
import pandas as pd
import math
from typing import Dict, List, Any, Optional, Tuple
from datetime import timedelta

# Import common utilities
from common.common import (
    extract_ticker_from_option_ticker,
    calculate_option_premium,
    format_bid_ask,
    format_price_with_change,
    calculate_days_to_expiry,
    normalize_timestamp_to_utc,
    normalize_expiration_date_to_utc,
    fetch_latest_option_timestamp_standalone
)


def ensure_ticker_column(options_df: pd.DataFrame) -> pd.DataFrame:
    """Ensure ticker column exists in options DataFrame, extracting from option_ticker if needed."""
    if 'ticker' not in options_df.columns and 'option_ticker' in options_df.columns:
        options_df = options_df.copy()
        options_df['ticker'] = options_df['option_ticker'].apply(extract_ticker_from_option_ticker)
    return options_df


def calculate_option_metrics(
    df: pd.DataFrame,
    position_size: float,
    days_to_expiry: Optional[int] = None
) -> pd.DataFrame:
    """Calculate option metrics: premium, contracts, potential premium, daily premium."""
    df = df.copy()
    df['strike_price'] = df['strike_price'].round(2)
    df['price_above_current'] = (df['strike_price'] - df['current_price']).round(2)
    df['option_premium'] = df.apply(calculate_option_premium, axis=1)
    df['bid_ask'] = df.apply(format_bid_ask, axis=1)
    
    # Normalize expiration_date and calculate days_to_expiry
    df['expiration_date'] = df['expiration_date'].apply(normalize_expiration_date_to_utc)
    # Use current time (not normalized) so we can check if we're before market close on expiration day
    today = pd.Timestamp.now(tz='UTC')
    df['days_to_expiry'] = df['expiration_date'].apply(lambda x: calculate_days_to_expiry(x, today))
    
    # Normalize implied_volatility
    if 'implied_volatility' in df.columns:
        df['implied_volatility'] = pd.to_numeric(df['implied_volatility'], errors='coerce').round(4)
    else:
        df['implied_volatility'] = pd.Series([float('nan')] * len(df), index=df.index)
    
    # Apply days_to_expiry filter
    if days_to_expiry is not None:
        df = df[
            (df['days_to_expiry'] >= days_to_expiry - 1) &
            (df['days_to_expiry'] <= days_to_expiry + 1)
        ].copy()
    
    # Calculate num_contracts and premiums
    # For calls (covered calls): num_contracts based on current_price (stock purchase cost)
    # For puts (cash-secured puts): num_contracts based on strike_price (cash required to secure assignment)
    df['num_contracts'] = df.apply(
        lambda row: (
            # For puts: use strike_price (cash-secured puts require cash = strike_price * 100 * contracts)
            math.floor(position_size / (row.get('strike_price') * 100))
            if str(row.get('option_type', 'call')).lower() == 'put' 
               and pd.notna(row.get('strike_price')) and row.get('strike_price') > 0
            # For calls: use current_price (covered calls require stock purchase = current_price * 100 * contracts)
            else (0 if pd.isna(row.get('current_price')) or row.get('current_price') <= 0 
                  else math.floor(position_size / (row.get('current_price') * 100)))
        ),
        axis=1
    )
    df['potential_premium'] = (df['num_contracts'] * (df['option_premium'] * 100)).round(2)
    df['daily_premium'] = df.apply(
        lambda row: 0 if row['days_to_expiry'] <= 0 
                  else round(row['potential_premium'] / row['days_to_expiry'], 2),
        axis=1
    )
    
    return df


def apply_basic_filters(
    df: pd.DataFrame,
    min_volume: int = 0,
    min_premium: float = 0.0,
    min_write_timestamp: Optional[str] = None,
    debug: bool = False,
    ticker: Optional[str] = None
) -> pd.DataFrame:
    """Apply basic filters: volume, premium, write timestamp."""
    if min_volume > 0:
        df = df[df['volume'] >= min_volume].copy()
    
    if min_premium > 0.0:
        df = df[df['potential_premium'] >= min_premium].copy()
    
    if min_write_timestamp:
        try:
            prefix = f"{ticker} - " if ticker else ""
            import pytz
            est = pytz.timezone('America/New_York')
            min_ts = pd.to_datetime(min_write_timestamp)
            if min_ts.tz is None:
                min_ts = est.localize(min_ts)
            min_ts_utc = min_ts.astimezone(pytz.UTC)
            if 'write_timestamp' in df.columns:
                before_normalize = len(df)
                df['write_timestamp'] = df['write_timestamp'].apply(normalize_timestamp_to_utc)
                
                # Debug: Check for any null timestamps after normalization
                null_count = df['write_timestamp'].isna().sum()
                if debug and null_count > 0:
                    import sys
                    print(f"DEBUG [apply_basic_filters]: {prefix}Warning - {null_count} records have null write_timestamp after normalization (out of {before_normalize} total)", file=sys.stderr)
                
                # Debug: Show timestamp range before filtering
                if debug and df['write_timestamp'].notna().any():
                    min_actual_ts = df['write_timestamp'].min()
                    max_actual_ts = df['write_timestamp'].max()
                    import sys
                    print(f"DEBUG [apply_basic_filters]: {prefix}Timestamp range: {min_actual_ts} to {max_actual_ts}, filter threshold: {min_ts_utc}", file=sys.stderr)
                    print(f"DEBUG [apply_basic_filters]: {prefix}Records before filter: {len(df)}, threshold: {min_ts_utc}", file=sys.stderr)
                
                df = df[df['write_timestamp'] >= min_ts_utc].copy()
                
                # Debug: Show filtering result
                filtered_out = before_normalize - len(df)
                if debug and filtered_out > 0:
                    import sys
                    print(f"DEBUG [apply_basic_filters]: {prefix}Filtered out {filtered_out} records (kept {len(df)}), threshold was {min_ts_utc}", file=sys.stderr)
        except Exception as e:
            if debug:
                import sys
                print(f"DEBUG [apply_basic_filters]: {prefix}Error applying timestamp filter: {e}", file=sys.stderr)
                import traceback
                traceback.print_exc(file=sys.stderr)
            pass  # Ignore timestamp filter errors
    
    return df


async def get_previous_close_for_date(
    db: Any,
    ticker: str,
    reference_date: pd.Timestamp,
    debug: bool = False
) -> Optional[float]:
    """
    Get the previous close price for a ticker based on a reference date.
    Returns the close price from the day before the reference date.
    
    Uses StockDataService.get() to fetch daily prices for a date range and picks the relevant price.
    
    Args:
        db: Database connection object (must have get_stock_data method)
        ticker: Ticker symbol
        reference_date: The date to use as reference (will find close price before this date)
        debug: Whether to print debug messages
        
    Returns:
        Previous close price (float) or None if not found
    """
    try:
        # Convert reference_date to date in EST timezone
        import pytz
        from datetime import timedelta
        
        if reference_date.tz is None:
            reference_date = reference_date.tz_localize('UTC')
        else:
            reference_date = reference_date.tz_convert('UTC')
        
        # Convert to EST for date comparison
        est = pytz.timezone('America/New_York')
        reference_date_est = reference_date.tz_convert(est)
        reference_date_only = reference_date_est.date()
        
        # Calculate date range: fetch last 10 days before reference date to ensure we get the previous close
        # This handles weekends/holidays where there might be gaps
        end_date = reference_date_only - timedelta(days=1)  # Day before reference date
        start_date = end_date - timedelta(days=10)  # Go back 10 days to ensure we get data
        
        # Use get_stock_data to fetch daily prices for the date range
        if hasattr(db, 'get_stock_data'):
            df = await db.get_stock_data(
                ticker=ticker,
                start_date=start_date.strftime('%Y-%m-%d'),
                end_date=end_date.strftime('%Y-%m-%d'),
                interval='daily'
            )
            
            if df is not None and not df.empty:
                # Get the most recent close price (last row, since data is ordered by date)
                # The date index should be the date column
                if isinstance(df.index, pd.DatetimeIndex):
                    # Filter to only dates before the reference date
                    df_filtered = df[df.index.date < reference_date_only]
                    if not df_filtered.empty:
                        # Get the last row (most recent date before reference)
                        latest_row = df_filtered.iloc[-1]
                        prev_close = float(latest_row['close'])
                        prev_date = df_filtered.index[-1].date() if hasattr(df_filtered.index[-1], 'date') else df_filtered.index[-1]
                        
                        if debug:
                            print(f"DEBUG: {ticker} - Found previous close ${prev_close:.2f} for date {prev_date} (before reference date {reference_date_only})", file=sys.stderr)
                        return prev_close
                else:
                    # If index is not DatetimeIndex, check if there's a 'date' column
                    if 'date' in df.columns:
                        df['date'] = pd.to_datetime(df['date'])
                        df_filtered = df[df['date'].dt.date < reference_date_only]
                        if not df_filtered.empty:
                            latest_row = df_filtered.iloc[-1]
                            prev_close = float(latest_row['close'])
                            prev_date = latest_row['date'].date() if hasattr(latest_row['date'], 'date') else latest_row['date']
                            
                            if debug:
                                print(f"DEBUG: {ticker} - Found previous close ${prev_close:.2f} for date {prev_date} (before reference date {reference_date_only})", file=sys.stderr)
                            return prev_close
        
        if debug:
            print(f"DEBUG: {ticker} - No previous close found before {reference_date_only}", file=sys.stderr)
        return None
    except Exception as e:
        if debug:
            print(f"DEBUG: {ticker} - Error getting previous close for date {reference_date}: {e}", file=sys.stderr)
        return None


async def attach_price_data(
    options_df: pd.DataFrame,
    db: Any,
    ticker: str,
    use_market_time: bool,
    redis_client: Optional[Any] = None,
    debug: bool = False
) -> pd.DataFrame:
    """Attach current price and price change data to options DataFrame."""
    price_data = await db.get_latest_price_with_data(ticker, use_market_time=use_market_time)
    if not price_data or not price_data.get('price'):
        return pd.DataFrame()
    
    current_price = price_data['price']
    price_timestamp = price_data.get('timestamp')
    price_source = price_data.get('source', 'unknown')
    options_df = options_df.copy()
    options_df['current_price'] = current_price
    
    # Get previous close price based on the current price's date
    # If market is open: compare realtime price to previous close
    # If market is closed: compare current close price to previous close (day before current price's date)
    prev_close = None
    if price_timestamp:
        # Use the current price's timestamp to get the previous close
        if isinstance(price_timestamp, pd.Timestamp):
            prev_close = await get_previous_close_for_date(db, ticker, price_timestamp, debug=debug)
        else:
            # Convert to Timestamp if needed
            try:
                ts = pd.to_datetime(price_timestamp, utc=True)
                prev_close = await get_previous_close_for_date(db, ticker, ts, debug=debug)
            except Exception as e:
                if debug:
                    print(f"DEBUG: {ticker} - Could not parse timestamp {price_timestamp}: {e}", file=sys.stderr)
    
    # Fallback to standard method if we couldn't get previous close based on date
    if prev_close is None:
        prev_close_prices = await db.get_previous_close_prices([ticker])
        prev_close = prev_close_prices.get(ticker) if prev_close_prices else None
    
    if debug:
        prev_close_str = f"{prev_close:.2f}" if prev_close is not None else "None"
        print(f"DEBUG: {ticker} - current_price=${current_price:.2f}, prev_close=${prev_close_str}, source={price_source}, timestamp={price_timestamp}", file=sys.stderr)
        if prev_close is None:
            print(f"DEBUG: {ticker} - WARNING: prev_close is None, cannot calculate price change", file=sys.stderr)
        elif prev_close <= 0:
            print(f"DEBUG: {ticker} - WARNING: prev_close={prev_close} is <= 0, cannot calculate price change", file=sys.stderr)
        elif abs(current_price - prev_close) < 0.01:
            print(f"DEBUG: {ticker} - WARNING: current_price ({current_price:.2f}) equals prev_close ({prev_close:.2f}), change will be 0", file=sys.stderr)
    
    # Format price with change
    price_with_change, price_change_pct = format_price_with_change(current_price, prev_close)
    options_df['price_with_change'] = price_with_change
    options_df['price_change_pct'] = price_change_pct
    
    # Fetch latest option timestamp (with Redis caching)
    latest_opt_ts = await fetch_latest_option_timestamp_standalone(db, ticker, redis_client=redis_client, debug=debug)
    options_df['latest_opt_ts'] = latest_opt_ts
    
    # Filter out rows without price
    options_df = options_df[options_df['current_price'].notna()].copy()
    
    return options_df

