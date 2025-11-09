#!/usr/bin/env python3
"""
Quick diagnostic script to check what option expiration dates exist in the database.
"""
import sys
import os
import asyncio
import traceback
import time
from pathlib import Path
from datetime import datetime, timezone
import pandas as pd

# Add project root to path
CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from common.stock_db import get_stock_db

def format_timestamp_local(ts):
    """Convert timestamp to local timezone and format for display."""
    if ts is None:
        return None
    
    try:
        # Get local timezone from system
        # Use UTC-aware datetime to get local timezone
        local_tz = datetime.now(timezone.utc).astimezone().tzinfo
        
        # Handle pandas Timestamp
        if isinstance(ts, pd.Timestamp):
            # Convert to datetime object
            dt = ts.to_pydatetime()
            # If timezone-naive, assume UTC
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            # Convert to local timezone
            dt_local = dt.astimezone(local_tz)
            return dt_local.strftime('%Y-%m-%d %H:%M:%S')
        
        # Handle datetime objects
        if isinstance(ts, datetime):
            # If timezone-naive, assume UTC
            if ts.tzinfo is None:
                ts = ts.replace(tzinfo=timezone.utc)
            # Convert to local timezone
            ts_local = ts.astimezone(local_tz)
            return ts_local.strftime('%Y-%m-%d %H:%M:%S')
    except Exception:
        # If timezone conversion fails, return as string
        return str(ts)
    
    # Fallback: return as string
    return str(ts)

async def print_closest_itm_options(db, ticker: str, current_price: float, debug: bool = False):
    """Print options closest to in-the-money for calls and puts."""
    try:
        # For calls: find strike_price <= current_price, closest to current_price (highest strike <= price)
        # For puts: find strike_price >= current_price, closest to current_price (lowest strike >= price)
        
        # Get the latest expiration date to find current options
        latest_exp_query = f"""
        SELECT MAX(expiration_date) as latest_exp
        FROM options_data
        WHERE ticker = '{ticker}'
        """
        
        latest_exp_df = await db.execute_select_sql(latest_exp_query)
        if latest_exp_df.empty:
            if debug:
                print(f"  [DEBUG] No expiration dates found for closest ITM options")
            return
        
        # Normalize column names
        if not latest_exp_df.empty:
            if len(latest_exp_df.columns) > 0 and str(latest_exp_df.columns[0]).isdigit():
                latest_exp_df.columns = ['latest_exp']
            else:
                latest_exp_df.columns = [str(c).lower() for c in latest_exp_df.columns]
        
        latest_exp = latest_exp_df.iloc[0]['latest_exp']
        if latest_exp is None:
            if debug:
                print(f"  [DEBUG] Latest expiration date is None")
            return
        
        # Query for closest call option (strike <= current_price, highest strike)
        call_query = f"""
        SELECT 
            option_ticker,
            expiration_date,
            strike_price,
            option_type,
            price,
            bid,
            ask,
            delta,
            gamma,
            theta,
            vega,
            implied_volatility,
            volume,
            open_interest,
            write_timestamp
        FROM (
            SELECT *, ROW_NUMBER() OVER (
                PARTITION BY option_ticker, expiration_date, strike_price, option_type 
                ORDER BY write_timestamp DESC
            ) as rn
            FROM options_data
            WHERE ticker = '{ticker}'
            AND option_type = 'call'
            AND strike_price <= {current_price}
            AND expiration_date = '{latest_exp}'
        ) WHERE rn = 1
        ORDER BY strike_price DESC
        LIMIT 1
        """
        
        # Query for closest put option (strike >= current_price, lowest strike)
        put_query = f"""
        SELECT 
            option_ticker,
            expiration_date,
            strike_price,
            option_type,
            price,
            bid,
            ask,
            delta,
            gamma,
            theta,
            vega,
            implied_volatility,
            volume,
            open_interest,
            write_timestamp
        FROM (
            SELECT *, ROW_NUMBER() OVER (
                PARTITION BY option_ticker, expiration_date, strike_price, option_type 
                ORDER BY write_timestamp DESC
            ) as rn
            FROM options_data
            WHERE ticker = '{ticker}'
            AND option_type = 'put'
            AND strike_price >= {current_price}
            AND expiration_date = '{latest_exp}'
        ) WHERE rn = 1
        ORDER BY strike_price ASC
        LIMIT 1
        """
        
        if debug:
            print(f"  [DEBUG] Querying for closest ITM options with current_price={current_price}")
            print(f"  [DEBUG] Latest expiration date: {latest_exp}")
        
        # Time the queries (these bypass cache)
        call_start = time.time()
        call_df = await db.execute_select_sql(call_query)
        call_time = time.time() - call_start
        
        put_start = time.time()
        put_df = await db.execute_select_sql(put_query)
        put_time = time.time() - put_start
        
        if debug:
            print(f"  [DEBUG] Call query took {call_time:.3f}s (direct DB query, cache bypassed)")
            print(f"  [DEBUG] Put query took {put_time:.3f}s (direct DB query, cache bypassed)")
        
        # Normalize column names
        if not call_df.empty:
            if len(call_df.columns) > 0 and str(call_df.columns[0]).isdigit():
                call_df.columns = ['option_ticker', 'expiration_date', 'strike_price', 'option_type', 
                                  'price', 'bid', 'ask', 'delta', 'gamma', 'theta', 'vega', 
                                  'implied_volatility', 'volume', 'open_interest', 'write_timestamp']
            else:
                call_df.columns = [str(c).lower() for c in call_df.columns]
        
        if not put_df.empty:
            if len(put_df.columns) > 0 and str(put_df.columns[0]).isdigit():
                put_df.columns = ['option_ticker', 'expiration_date', 'strike_price', 'option_type', 
                                 'price', 'bid', 'ask', 'delta', 'gamma', 'theta', 'vega', 
                                 'implied_volatility', 'volume', 'open_interest', 'write_timestamp']
            else:
                put_df.columns = [str(c).lower() for c in put_df.columns]
        
        print(f"  Closest to in-the-money options (latest expiration: {latest_exp}):")
        
        # Print call option
        if not call_df.empty:
            call = call_df.iloc[0]
            moneyness = ((current_price - call['strike_price']) / call['strike_price']) * 100 if call['strike_price'] > 0 else 0
            print(f"    CALL:")
            print(f"      Option: {call['option_ticker']}")
            print(f"      Strike: ${call['strike_price']:.2f}")
            print(f"      Moneyness: {moneyness:.2f}% ITM")
            price_val = call.get('price')
            bid_val = call.get('bid')
            ask_val = call.get('ask')
            if price_val is not None and not pd.isna(price_val):
                print(f"      Price: ${price_val:.2f}")
            else:
                print(f"      Price: N/A")
            if bid_val is not None and ask_val is not None and not pd.isna(bid_val) and not pd.isna(ask_val):
                print(f"      Bid/Ask: ${bid_val:.2f} / ${ask_val:.2f}")
            else:
                print(f"      Bid/Ask: N/A")
            if call.get('delta') is not None:
                print(f"      Delta: {call['delta']:.4f}")
            if call.get('gamma') is not None:
                print(f"      Gamma: {call['gamma']:.4f}")
            if call.get('theta') is not None:
                print(f"      Theta: {call['theta']:.4f}")
            if call.get('vega') is not None:
                print(f"      Vega: {call['vega']:.4f}")
            if call.get('implied_volatility') is not None:
                print(f"      IV: {call['implied_volatility']:.2%}")
            if call.get('volume') is not None:
                print(f"      Volume: {call['volume']}")
            if call.get('open_interest') is not None:
                print(f"      Open Interest: {call['open_interest']}")
        else:
            print(f"    CALL: No call option found with strike <= ${current_price:.2f}")
        
        # Print put option
        if not put_df.empty:
            put = put_df.iloc[0]
            moneyness = ((put['strike_price'] - current_price) / put['strike_price']) * 100 if put['strike_price'] > 0 else 0
            print(f"    PUT:")
            print(f"      Option: {put['option_ticker']}")
            print(f"      Strike: ${put['strike_price']:.2f}")
            print(f"      Moneyness: {moneyness:.2f}% ITM")
            price_val = put.get('price')
            bid_val = put.get('bid')
            ask_val = put.get('ask')
            if price_val is not None and not pd.isna(price_val):
                print(f"      Price: ${price_val:.2f}")
            else:
                print(f"      Price: N/A")
            if bid_val is not None and ask_val is not None and not pd.isna(bid_val) and not pd.isna(ask_val):
                print(f"      Bid/Ask: ${bid_val:.2f} / ${ask_val:.2f}")
            else:
                print(f"      Bid/Ask: N/A")
            if put.get('delta') is not None:
                print(f"      Delta: {put['delta']:.4f}")
            if put.get('gamma') is not None:
                print(f"      Gamma: {put['gamma']:.4f}")
            if put.get('theta') is not None:
                print(f"      Theta: {put['theta']:.4f}")
            if put.get('vega') is not None:
                print(f"      Vega: {put['vega']:.4f}")
            if put.get('implied_volatility') is not None:
                print(f"      IV: {put['implied_volatility']:.2%}")
            if put.get('volume') is not None:
                print(f"      Volume: {put['volume']}")
            if put.get('open_interest') is not None:
                print(f"      Open Interest: {put['open_interest']}")
        else:
            print(f"    PUT: No put option found with strike >= ${current_price:.2f}")
            
    except Exception as e:
        if debug:
            print(f"  Error finding closest ITM options: {e}")
            print(f"  Traceback: {traceback.format_exc()}")
        else:
            print(f"  Error finding closest ITM options: {e}")

async def check_dates(db_conn: str, tickers: list, enable_cache: bool = True, debug: bool = False):
    """Check available option expiration dates in the database."""
    redis_url = os.getenv('REDIS_URL', 'redis://localhost:6379/0') if enable_cache else None
    db = get_stock_db('questdb', db_config=db_conn, enable_cache=enable_cache, redis_url=redis_url)
    
    # Get initial cache statistics
    initial_cache_stats = {}
    if hasattr(db, 'get_cache_statistics'):
        initial_cache_stats = db.get_cache_statistics()
    
    # Check cache status
    cache_enabled = False
    if hasattr(db, 'cache'):
        cache_enabled = getattr(db.cache, 'enable_cache', False)
    
    if debug:
        print(f"[DEBUG] Cache enabled: {cache_enabled}")
        print(f"[DEBUG] enable_cache parameter: {enable_cache}")
        if hasattr(db, 'cache') and hasattr(db.cache, 'redis_url'):
            print(f"[DEBUG] Redis URL: {db.cache.redis_url if cache_enabled else 'N/A'}")
        print(f"[DEBUG] NOTE: execute_select_sql() bypasses the cache layer.")
        print(f"[DEBUG] Cache is only used by service layer methods (e.g., options_service.get()).")
        print(f"[DEBUG] Initial cache stats: hits={initial_cache_stats.get('hits', 0)}, misses={initial_cache_stats.get('misses', 0)}")
        print()
    
    print(f"Checking options data for: {', '.join(tickers)}\n")
    
    for ticker in tickers:
        try:
            # Query for distinct expiration dates with their last update time
            query = f"""
            SELECT 
                expiration_date, 
                option_type,
                MAX(write_timestamp) as last_update
            FROM options_data
            WHERE ticker = '{ticker}'
            AND option_type = 'call'
            GROUP BY expiration_date, option_type
            ORDER BY expiration_date
            """
            
            # Time the query to help distinguish cache vs DB (though execute_select_sql bypasses cache)
            query_start = time.time()
            df = await db.execute_select_sql(query)
            query_time = time.time() - query_start
            
            if debug:
                print(f"[DEBUG] Expiration dates query took {query_time:.3f}s (direct DB query, cache bypassed)")
            
            # QuestDB may return columns as numeric indices, so map them to proper names
            if not df.empty:
                # Check if columns are numeric indices (QuestDB behavior)
                if len(df.columns) > 0 and str(df.columns[0]).isdigit():
                    # Map based on SELECT order: expiration_date, option_type, last_update
                    df.columns = ['expiration_date', 'option_type', 'last_update']
                else:
                    # Normalize column names to lowercase for QuestDB compatibility
                    df.columns = [str(c).lower() for c in df.columns]
            
            # Also get the overall last update time for this ticker
            overall_query = f"""
            SELECT MAX(write_timestamp) as last_update
            FROM options_data
            WHERE ticker = '{ticker}'
            AND option_type = 'call'
            """
            
            overall_start = time.time()
            overall_df = await db.execute_select_sql(overall_query)
            overall_time = time.time() - overall_start
            
            if debug:
                print(f"[DEBUG] Overall last update query took {overall_time:.3f}s (direct DB query, cache bypassed)")
            
            # QuestDB may return columns as numeric indices, so map them to proper names
            if not overall_df.empty:
                # Check if columns are numeric indices (QuestDB behavior)
                if len(overall_df.columns) > 0 and str(overall_df.columns[0]).isdigit():
                    # Map based on SELECT order: last_update
                    overall_df.columns = ['last_update']
                else:
                    # Normalize column names to lowercase for QuestDB compatibility
                    overall_df.columns = [str(c).lower() for c in overall_df.columns]
            
            if df.empty:
                print(f"{ticker}: No options data found")
            else:
                print(f"{ticker}: Found {len(df)} unique expiration dates")
                print(f"  Earliest: {df.iloc[0]['expiration_date']}")
                print(f"  Latest:   {df.iloc[-1]['expiration_date']}")
                
                # Print overall last update time
                if not overall_df.empty and 'last_update' in overall_df.columns:
                    last_update = overall_df.iloc[0]['last_update']
                    if last_update is not None:
                        last_update_local = format_timestamp_local(last_update)
                        print(f"  Last overall update: {last_update_local}")
                
                print(f"  All dates with last update times:")
                for idx, row in df.iterrows():
                    expiration_date = row['expiration_date']
                    last_update = row.get('last_update')
                    if last_update is not None:
                        last_update_local = format_timestamp_local(last_update)
                        print(f"    - {expiration_date} (last updated: {last_update_local})")
                    else:
                        print(f"    - {expiration_date} (last updated: N/A)")
                
                # Get current stock price (this uses the service layer which may use cache)
                try:
                    price_start = time.time()
                    current_price = await db.get_latest_price(ticker, use_market_time=True)
                    price_time = time.time() - price_start
                    
                    if debug:
                        print(f"[DEBUG] get_latest_price() took {price_time:.3f}s (may use cache)")
                    
                    if current_price is None:
                        print(f"  Current stock price: N/A (unable to fetch)")
                    else:
                        print(f"  Current stock price: ${current_price:.2f}")
                        
                        # Find options closest to in-the-money
                        await print_closest_itm_options(db, ticker, current_price, debug)
                except Exception as e:
                    if debug:
                        print(f"  Error getting current price: {e}")
                        print(f"  Traceback: {traceback.format_exc()}")
                    else:
                        print(f"  Current stock price: N/A (error: {e})")
            print()
            
        except Exception as e:
            print(f"Error checking {ticker}: {e}")
            print(f"Traceback: {traceback.format_exc()}\n")
    
    # Print final cache statistics (only if debug is active)
    if debug and hasattr(db, 'get_cache_statistics'):
        final_cache_stats = db.get_cache_statistics()
        print("\n" + "=" * 80)
        print("Cache Statistics")
        print("=" * 80)
        print(f"Cache Enabled: {final_cache_stats.get('enabled', False)}")
        print(f"Total Requests: {final_cache_stats.get('total_requests', 0)}")
        print(f"Cache Hits: {final_cache_stats.get('hits', 0)}")
        print(f"Cache Misses: {final_cache_stats.get('misses', 0)}")
        print(f"Cache Sets: {final_cache_stats.get('sets', 0)}")
        print(f"Cache Invalidations: {final_cache_stats.get('invalidations', 0)}")
        print(f"Cache Errors: {final_cache_stats.get('errors', 0)}")
        
        hit_rate = final_cache_stats.get('hit_rate', 0.0)
        if final_cache_stats.get('total_requests', 0) > 0:
            print(f"Hit Rate: {hit_rate:.2%}")
        
        print("\nNOTE: Direct SQL queries (execute_select_sql) bypass the cache.")
        print("Cache is only used by service layer methods (e.g., get_latest_price).")
        print("=" * 80)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Check available option expiration dates")
    parser.add_argument('--db-conn', required=True, help="Database connection string")
    parser.add_argument('--symbols', nargs='+', required=True, help="Ticker symbols to check")
    parser.add_argument('--debug', action='store_true', help="Enable debug output to see cache usage")
    parser.add_argument('--no-cache', action='store_true', help="Disable caching to see behavior from DB")
    
    args = parser.parse_args()
    
    enable_cache = not args.no_cache
    
    asyncio.run(check_dates(args.db_conn, args.symbols, enable_cache=enable_cache, debug=args.debug))









