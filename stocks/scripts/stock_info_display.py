#!/usr/bin/env python3
"""
Display comprehensive stock information including price, options, and financial ratios.

This script combines functionality from fetch_symbol_data.py and fetch_options.py
to provide a unified view of stock information.
"""

import asyncio
import argparse
import sys
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import List, Dict, Any, Optional
import pandas as pd
import logging

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from fetch_symbol_data import (
    get_current_price,
    get_financial_ratios,
    get_latest_news,
    get_latest_iv,
    process_symbol_data,
    _format_price_block,
    _normalize_timezone_string,
    _get_et_now,
    get_default_db_path,
    StockDBBase,
)
from scripts.fetch_options import HistoricalDataFetcher
from common.stock_db import get_stock_db

logger = logging.getLogger(__name__)

# Try to import Polygon client
try:
    from polygon.rest import RESTClient as PolygonRESTClient
    POLYGON_AVAILABLE = True
except ImportError:
    POLYGON_AVAILABLE = False
    print("Warning: polygon-api-client not installed. Polygon.io data source will not be available.", file=sys.stderr)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Display comprehensive stock information (price, options, financial ratios)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Display latest info for a single symbol (default behavior)
  python stock_info_display.py AAPL

  # Explicitly request latest price only
  python stock_info_display.py AAPL --latest

  # Display info for multiple symbols
  python stock_info_display.py AAPL MSFT GOOGL

  # Display with date range for price data
  python stock_info_display.py AAPL --start-date 2024-01-01 --end-date 2024-12-31

  # Display options for next 90 days (instead of default 180)
  python stock_info_display.py AAPL --options-days 90

  # Force fetch from API (bypass DB/cache)
  python stock_info_display.py AAPL --force-fetch

  # Use specific database
  python stock_info_display.py AAPL --db-path questdb://localhost:9000
        """
    )
    
    parser.add_argument(
        "symbols",
        nargs="+",
        help="Stock symbol(s) to display (e.g., AAPL MSFT GOOGL)"
    )
    
    parser.add_argument(
        "--latest",
        action="store_true",
        help="Show only the latest price information (overrides --start-date and --end-date)"
    )
    
    parser.add_argument(
        "--start-date",
        type=str,
        default=None,
        help="Start date for price data (YYYY-MM-DD). Default: latest only. Ignored if --latest is set."
    )
    
    parser.add_argument(
        "--end-date",
        type=str,
        default=None,
        help="End date for price data (YYYY-MM-DD). Default: today. Ignored if --latest is set."
    )
    
    parser.add_argument(
        "--options-days",
        type=int,
        default=180,
        help="Number of days ahead to fetch options data (default: 180)"
    )
    
    parser.add_argument(
        "--force-fetch",
        action="store_true",
        help="Force fetch all data from API (bypass DB/cache)"
    )
    
    parser.add_argument(
        "--db-type",
        type=str,
        default="questdb",
        choices=["sqlite", "duckdb", "questdb", "postgresql"],
        help="Database type (default: questdb)"
    )
    
    parser.add_argument(
        "--db-path",
        type=str,
        default=None,
        help="Database path or connection string (e.g., questdb://localhost:9000, postgresql://user:pass@host:port/db)"
    )
    
    parser.add_argument(
        "--data-source",
        choices=["polygon", "alpaca"],
        default="polygon",
        help="Data source for fetching (default: polygon)"
    )
    
    parser.add_argument(
        "--timezone",
        type=str,
        default="America/New_York",
        help="Timezone for displaying timestamps (default: America/New_York)"
    )
    
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Logging level (default: INFO)"
    )
    
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Disable Redis caching for QuestDB operations"
    )
    
    parser.add_argument(
        "--show-price-history",
        action="store_true",
        help="Show price history table (only if date range specified)"
    )
    
    parser.add_argument(
        "--options-type",
        choices=["all", "call", "put"],
        default="all",
        help="Filter options by type (default: all)"
    )
    
    parser.add_argument(
        "--strike-range-percent",
        type=int,
        default=None,
        help="Filter options by strike range (±percent from stock price, e.g., 20 for ±20%%)"
    )
    
    parser.add_argument(
        "--max-options-per-expiry",
        type=int,
        default=10,
        help="Maximum number of options to show per expiration date (default: 10)"
    )
    
    parser.add_argument(
        "--show-news",
        action="store_true",
        help="Show latest news articles for the symbol"
    )
    
    parser.add_argument(
        "--show-iv",
        action="store_true",
        help="Show latest implied volatility statistics"
    )
    
    return parser.parse_args()


async def get_price_info(
    symbol: str,
    db_instance: StockDBBase,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    force_fetch: bool = False,
    data_source: str = "polygon",
    timezone_str: Optional[str] = None,
    latest_only: bool = False
) -> Dict[str, Any]:
    """Get price information for a symbol.
    
    Args:
        latest_only: If True, only fetch latest price and skip historical data
    """
    result = {
        "symbol": symbol,
        "current_price": None,
        "price_data": None,
        "error": None
    }
    
    try:
        # Get cache instance if available
        cache_instance = None
        if hasattr(db_instance, 'cache') and db_instance.cache:
            cache_instance = db_instance.cache
        
        # Check cache for latest price data first (if not force_fetch)
        cached_price_data = None
        if not force_fetch and cache_instance:
            try:
                from common.redis_cache import CacheKeyGenerator
                cache_key = CacheKeyGenerator.latest_price_data(symbol)
                cached_df = await cache_instance.get(cache_key)
                if cached_df is not None and not cached_df.empty:
                    # Convert DataFrame back to dict
                    cached_price_data = cached_df.iloc[0].to_dict()
                    # Restore nested structures if any
                    if 'realtime_df' in cached_price_data and isinstance(cached_price_data['realtime_df'], str):
                        import json
                        try:
                            cached_price_data['realtime_df'] = json.loads(cached_price_data['realtime_df'])
                        except:
                            cached_price_data['realtime_df'] = None
                    logger.debug(f"Found cached latest price data for {symbol}")
            except Exception as e:
                logger.debug(f"Cache check failed for price data {symbol}: {e}")
        
        # Get current/latest price
        if cached_price_data and not force_fetch:
            # Use cached data
            price_info = {
                'symbol': symbol,
                'price': cached_price_data.get('price'),
                'bid_price': cached_price_data.get('bid_price'),
                'ask_price': cached_price_data.get('ask_price'),
                'timestamp': cached_price_data.get('timestamp'),
                'source': cached_price_data.get('source', 'cache'),
                'data_source': data_source
            }
        elif force_fetch:
            # Force fetch from API
            price_info = await get_current_price(
                symbol,
                data_source=data_source,
                stock_db_instance=db_instance,
                max_age_seconds=0  # Force fresh fetch
            )
        else:
            # Try DB first, then API if needed
            price_info = await get_current_price(
                symbol,
                data_source=data_source,
                stock_db_instance=db_instance,
                max_age_seconds=600  # 10 minutes
            )
        
        if price_info:
            result["current_price"] = price_info
            
            # Cache the price data if we have a cache instance and it's not from cache
            if cache_instance and price_info.get('source') != 'cache':
                try:
                    from common.redis_cache import CacheKeyGenerator
                    import json
                    import pandas as pd
                    cache_key = CacheKeyGenerator.latest_price_data(symbol)
                    # Convert price_info dict to DataFrame for caching
                    cache_dict = price_info.copy()
                    # Serialize nested structures
                    if 'realtime_df' in cache_dict and cache_dict['realtime_df'] is not None:
                        # Store as JSON string if it's a DataFrame
                        if isinstance(cache_dict['realtime_df'], pd.DataFrame):
                            cache_dict['realtime_df'] = cache_dict['realtime_df'].to_json(orient='records')
                    cache_df = pd.DataFrame([cache_dict])
                    await cache_instance.set(cache_key, cache_df, ttl=300)  # 5 minutes TTL
                    logger.debug(f"Cached latest price data for {symbol}")
                except Exception as e:
                    logger.debug(f"Failed to cache price data for {symbol}: {e}")
        
        # Get historical price data if date range specified and not latest_only
        if not latest_only and (start_date or end_date):
            price_df = await process_symbol_data(
                symbol=symbol,
                timeframe="daily",
                start_date=start_date,
                end_date=end_date,
                stock_db_instance=db_instance,
                force_fetch=force_fetch,
                query_only=not force_fetch,
                data_source=data_source,
                log_level="ERROR"  # Suppress verbose logging
            )
            
            if not price_df.empty:
                result["price_data"] = price_df
        
    except Exception as e:
        result["error"] = str(e)
        logger.error(f"Error getting price info for {symbol}: {e}")
    
    return result


async def get_options_info(
    symbol: str,
    db_instance: StockDBBase,
    options_days: int = 180,
    force_fetch: bool = False,
    data_source: str = "polygon",
    option_type: str = "all",
    strike_range_percent: Optional[int] = None,
    max_options_per_expiry: int = 10,
    enable_cache: bool = True,
    redis_url: Optional[str] = None
) -> Dict[str, Any]:
    """Get options information for a symbol."""
    import time
    fetch_start = time.time()
    
    result = {
        "symbol": symbol,
        "options_data": None,
        "error": None,
        "source": None,
        "fetch_time_ms": None
    }
    
    try:
        # Note: Caching is handled by OptionsDataService.get_latest() internally
        # HistoricalDataFetcher.get_active_options_for_date() calls db.get_latest_options_data()
        # which uses the service method that handles caching
        
        # Get current stock price for strike range calculation
        stock_price = None
        try:
            price_info = await get_current_price(
                symbol,
                data_source=data_source,
                stock_db_instance=db_instance,
                max_age_seconds=3600  # 1 hour is fine for options
            )
            if price_info and price_info.get("price"):
                stock_price = price_info["price"]
        except Exception:
            pass  # Continue without stock price
        
        # Calculate date range for options
        today = datetime.now().date()
        end_date = today + timedelta(days=options_days)
        target_date_str = today.strftime("%Y-%m-%d")
        
        # Check if we should fetch from API or use DB
        db_check_start = time.time()
        if force_fetch:
            # Force fetch from Polygon API
            if not POLYGON_AVAILABLE:
                result["error"] = "Polygon API client not available"
                return result
            
            api_key = os.getenv("POLYGON_API_KEY")
            if not api_key:
                result["error"] = "POLYGON_API_KEY environment variable not set"
                return result
            
            api_fetch_start = time.time()
            fetcher = HistoricalDataFetcher(api_key, quiet=True)
            options_result = await fetcher.get_active_options_for_date(
                symbol=symbol,
                target_date_str=target_date_str,
                option_type=option_type,
                stock_close_price=stock_price,
                strike_range_percent=strike_range_percent,
                max_days_to_expiry=options_days,
                include_expired=False,
                use_db=False,
                force_fresh=True
            )
            api_fetch_time = (time.time() - api_fetch_start) * 1000
            fetch_time = (time.time() - fetch_start) * 1000
            
            result["options_data"] = options_result
            result["source"] = "api"
            result["fetch_time_ms"] = fetch_time
            logger.info(f"[OPTIONS API FETCH] Options for {symbol} (api_fetch: {api_fetch_time:.1f}ms, total: {fetch_time:.1f}ms)")
        else:
            # Try DB first
            try:
                # Get options from database
                db_conn = None
                if hasattr(db_instance, "db_config"):
                    db_conn = db_instance.db_config
                elif hasattr(db_instance, "connection_string"):
                    db_conn = db_instance.connection_string
                
                if db_conn:
                    # Use HistoricalDataFetcher to query DB
                    api_key = os.getenv("POLYGON_API_KEY", "")  # Not used when use_db=True
                    fetcher = HistoricalDataFetcher(api_key, quiet=True)
                    
                    db_fetch_start = time.time()
                    options_result = await fetcher.get_active_options_for_date(
                        symbol=symbol,
                        target_date_str=target_date_str,
                        option_type=option_type,
                        stock_close_price=stock_price,
                        strike_range_percent=strike_range_percent,
                        max_days_to_expiry=options_days,
                        include_expired=False,
                        use_db=True,
                        db_conn=db_conn,
                        force_fresh=False,
                        enable_cache=enable_cache,
                        redis_url=redis_url
                    )
                    db_fetch_time = (time.time() - db_fetch_start) * 1000
                    fetch_time = (time.time() - fetch_start) * 1000
                    
                    result["options_data"] = options_result
                    result["source"] = "database"
                    result["fetch_time_ms"] = fetch_time
                    logger.info(f"[OPTIONS DB HIT] Options for {symbol} (db_fetch: {db_fetch_time:.1f}ms, total: {fetch_time:.1f}ms)")
                    
                    # Note: Caching is handled by OptionsDataService.get_latest() internally
                    # HistoricalDataFetcher.get_active_options_for_date() calls db.get_latest_options_data()
                    # which uses the service method that handles caching
                else:
                    # Fallback to API if no DB connection
                    if POLYGON_AVAILABLE:
                        api_key = os.getenv("POLYGON_API_KEY")
                        if api_key:
                            api_fetch_start = time.time()
                            fetcher = HistoricalDataFetcher(api_key, quiet=True)
                            options_result = await fetcher.get_active_options_for_date(
                                symbol=symbol,
                                target_date_str=target_date_str,
                                option_type=option_type,
                                stock_close_price=stock_price,
                                strike_range_percent=strike_range_percent,
                                max_days_to_expiry=options_days,
                                include_expired=False,
                                use_db=False,
                                force_fresh=False
                            )
                            api_fetch_time = (time.time() - api_fetch_start) * 1000
                            fetch_time = (time.time() - fetch_start) * 1000
                            
                            result["options_data"] = options_result
                            result["source"] = "api"
                            result["fetch_time_ms"] = fetch_time
                            logger.info(f"[OPTIONS API FETCH] Options for {symbol} (api_fetch: {api_fetch_time:.1f}ms, total: {fetch_time:.1f}ms)")
                            
                            # Note: For API results, caching should happen when data is saved to DB
                            # The service method will handle caching when the data is stored
                        else:
                            result["error"] = "POLYGON_API_KEY not set and no DB connection available"
                    else:
                        result["error"] = "No database connection and Polygon API not available"
            except Exception as e:
                logger.warning(f"[OPTIONS DB ERROR] Error getting options from DB for {symbol}: {e}, trying API...")
                # Fallback to API
                if POLYGON_AVAILABLE:
                    api_key = os.getenv("POLYGON_API_KEY")
                    if api_key:
                        api_fetch_start = time.time()
                        fetcher = HistoricalDataFetcher(api_key, quiet=True)
                        options_result = await fetcher.get_active_options_for_date(
                            symbol=symbol,
                            target_date_str=target_date_str,
                            option_type=option_type,
                            stock_close_price=stock_price,
                            strike_range_percent=strike_range_percent,
                            max_days_to_expiry=options_days,
                            include_expired=False,
                            use_db=False,
                            force_fresh=False
                        )
                        api_fetch_time = (time.time() - api_fetch_start) * 1000
                        fetch_time = (time.time() - fetch_start) * 1000
                        
                        result["options_data"] = options_result
                        result["source"] = "api"
                        result["fetch_time_ms"] = fetch_time
                        logger.info(f"[OPTIONS API FETCH] Options for {symbol} (api_fetch: {api_fetch_time:.1f}ms, total: {fetch_time:.1f}ms)")
                        
                        # Note: For API results, caching should happen when data is saved to DB
                        # The service method will handle caching when the data is stored
    
    except Exception as e:
        result["error"] = str(e)
        logger.error(f"Error getting options info for {symbol}: {e}")
    
    return result


async def get_financial_info(
    symbol: str,
    db_instance: StockDBBase,
    force_fetch: bool = False
) -> Dict[str, Any]:
    """Get financial ratios/information for a symbol."""
    import time
    fetch_start = time.time()
    
    result = {
        "symbol": symbol,
        "financial_data": None,
        "error": None,
        "source": None,
        "fetch_time_ms": None
    }
    
    try:
        # Get cache instance if available
        cache_instance = None
        if hasattr(db_instance, 'cache') and db_instance.cache:
            cache_instance = db_instance.cache
        
        # Check cache first (if not force_fetch)
        if not force_fetch and cache_instance:
            try:
                from common.redis_cache import CacheKeyGenerator
                cache_key = CacheKeyGenerator.financial_info(symbol)
                cache_check_start = time.time()
                cached_df = await cache_instance.get(cache_key)
                cache_check_time = (time.time() - cache_check_start) * 1000
                
                if cached_df is not None and not cached_df.empty:
                    # Convert DataFrame back to dict
                    cached_financial = cached_df.iloc[0].to_dict()
                    fetch_time = (time.time() - fetch_start) * 1000
                    result["financial_data"] = cached_financial
                    result["source"] = "cache"
                    result["fetch_time_ms"] = fetch_time
                    logger.info(f"[FINANCIAL CACHE HIT] Financial data for {symbol} (cache_check: {cache_check_time:.1f}ms, total: {fetch_time:.1f}ms)")
                    return result
                else:
                    logger.debug(f"[FINANCIAL CACHE MISS] No cached financial data for {symbol} (cache_check: {cache_check_time:.1f}ms)")
            except Exception as e:
                logger.debug(f"[FINANCIAL CACHE ERROR] Cache check failed for {symbol}: {e}")
        
        # Try DB first if not forcing fetch
        if not force_fetch:
            try:
                db_check_start = time.time()
                financial_df = await db_instance.get_financial_info(symbol)
                db_check_time = (time.time() - db_check_start) * 1000
                
                if not financial_df.empty:
                    # Get the most recent entry
                    latest = financial_df.iloc[-1].to_dict()
                    fetch_time = (time.time() - fetch_start) * 1000
                    result["financial_data"] = latest
                    result["source"] = "database"
                    result["fetch_time_ms"] = fetch_time
                    logger.info(f"[FINANCIAL DB HIT] Financial data for {symbol} (db_check: {db_check_time:.1f}ms, total: {fetch_time:.1f}ms)")
                    
                    # Cache the result
                    if cache_instance:
                        try:
                            from common.redis_cache import CacheKeyGenerator
                            import pandas as pd
                            cache_key = CacheKeyGenerator.financial_info(symbol)
                            cache_df = pd.DataFrame([latest])
                            cache_set_start = time.time()
                            await cache_instance.set(cache_key, cache_df, ttl=3600)  # 1 hour TTL
                            cache_set_time = (time.time() - cache_set_start) * 1000
                            logger.info(f"[FINANCIAL CACHE SET] Cached financial data for {symbol} (set_time: {cache_set_time:.1f}ms, ttl: 3600s)")
                        except Exception as e:
                            logger.debug(f"[FINANCIAL CACHE ERROR] Failed to cache financial data for {symbol}: {e}")
                    
                    return result
            except Exception as e:
                logger.debug(f"[FINANCIAL DB ERROR] Error getting financial info from DB for {symbol}: {e}")
        
        # Fetch from API
        api_key = os.getenv("POLYGON_API_KEY")
        if not api_key:
            result["error"] = "POLYGON_API_KEY environment variable not set"
            return result
        
        api_fetch_start = time.time()
        ratios = await get_financial_ratios(symbol, api_key)
        api_fetch_time = (time.time() - api_fetch_start) * 1000
        fetch_time = (time.time() - fetch_start) * 1000
        
        if ratios:
            result["financial_data"] = ratios
            result["source"] = "api"
            result["fetch_time_ms"] = fetch_time
            logger.info(f"[FINANCIAL API FETCH] Financial data for {symbol} (api_fetch: {api_fetch_time:.1f}ms, total: {fetch_time:.1f}ms)")
            
            # Save to DB if we have a DB instance
            try:
                await db_instance.save_financial_info(symbol, ratios)
            except Exception as e:
                logger.debug(f"[FINANCIAL DB SAVE ERROR] Error saving financial info to DB for {symbol}: {e}")
            
            # Cache the result
            if cache_instance:
                try:
                    from common.redis_cache import CacheKeyGenerator
                    import pandas as pd
                    cache_key = CacheKeyGenerator.financial_info(symbol)
                    cache_df = pd.DataFrame([ratios])
                    cache_set_start = time.time()
                    await cache_instance.set(cache_key, cache_df, ttl=3600)  # 1 hour TTL
                    cache_set_time = (time.time() - cache_set_start) * 1000
                    logger.info(f"[FINANCIAL CACHE SET] Cached financial data for {symbol} (set_time: {cache_set_time:.1f}ms, ttl: 3600s)")
                except Exception as e:
                    logger.debug(f"[FINANCIAL CACHE ERROR] Failed to cache financial data for {symbol}: {e}")
        else:
            result["error"] = "No financial ratios data available"
    
    except Exception as e:
        result["error"] = str(e)
        logger.error(f"[FINANCIAL ERROR] Error getting financial info for {symbol}: {e}")
    
    return result

async def get_news_info(
    symbol: str,
    db_instance: StockDBBase,
    force_fetch: bool = False,
    enable_cache: bool = True
) -> Dict[str, Any]:
    """Get latest news for a symbol."""
    import time
    fetch_start = time.time()
    
    result = {
        "symbol": symbol,
        "news_data": None,
        "error": None,
        "freshness": None
    }
    
    try:
        api_key = os.getenv("POLYGON_API_KEY")
        if not api_key:
            result["error"] = "POLYGON_API_KEY environment variable not set"
            return result
        
        # Get cache instance if available
        cache_instance = None
        if enable_cache and hasattr(db_instance, 'cache') and db_instance.cache:
            cache_instance = db_instance.cache
        
        news_data = await get_latest_news(
            symbol,
            api_key,
            max_items=10,
            cache_instance=cache_instance if not force_fetch else None,
            cache_ttl=3600  # 1 hour TTL
        )
        
        fetch_time = (time.time() - fetch_start) * 1000
        
        if news_data:
            logger.info(f"[NEWS] Fetched {news_data.get('count', 0)} news articles for {symbol} (fetch_time: {fetch_time:.1f}ms, cached: {not force_fetch and cache_instance is not None})")
            result["news_data"] = news_data
            
            # Calculate freshness
            if news_data.get('fetched_at'):
                try:
                    fetched_dt = datetime.fromisoformat(news_data['fetched_at'].replace('Z', '+00:00'))
                    age_seconds = (datetime.now(timezone.utc) - fetched_dt).total_seconds()
                    result["freshness"] = {
                        "age_seconds": age_seconds,
                        "age_minutes": age_seconds / 60,
                        "is_fresh": age_seconds < 3600,  # Fresh if less than 1 hour old
                        "needs_refetch": age_seconds > 7200  # Needs refetch if older than 2 hours
                    }
                except Exception:
                    pass
        else:
            result["error"] = "No news data available"
    
    except Exception as e:
        result["error"] = str(e)
        logger.error(f"Error getting news info for {symbol}: {e}")
    
    return result

async def get_iv_info(
    symbol: str,
    db_instance: StockDBBase,
    force_fetch: bool = False,
    enable_cache: bool = True
) -> Dict[str, Any]:
    """Get latest IV information for a symbol."""
    result = {
        "symbol": symbol,
        "iv_data": None,
        "error": None,
        "freshness": None
    }
    
    try:
        # Get cache instance if available
        cache_instance = None
        if enable_cache and hasattr(db_instance, 'cache') and db_instance.cache:
            cache_instance = db_instance.cache
        
        import time
        iv_fetch_start = time.time()
        iv_data = await get_latest_iv(
            symbol,
            db_instance=db_instance,
            cache_instance=cache_instance if not force_fetch else None,
            cache_ttl=300  # 5 minutes TTL
        )
        iv_fetch_time = (time.time() - iv_fetch_start) * 1000
        
        if iv_data:
            iv_source = iv_data.get('source', 'unknown')
            logger.info(f"[IV] Fetched IV data for {symbol} from {iv_source} (count: {iv_data.get('statistics', {}).get('count', 0)}, fetch_time: {iv_fetch_time:.1f}ms)")
            result["iv_data"] = iv_data
            result["source"] = iv_source
            result["fetch_time_ms"] = iv_fetch_time
            
            # Calculate freshness
            if iv_data.get('fetched_at'):
                try:
                    fetched_dt = datetime.fromisoformat(iv_data['fetched_at'].replace('Z', '+00:00'))
                    age_seconds = (datetime.now(timezone.utc) - fetched_dt).total_seconds()
                    result["freshness"] = {
                        "age_seconds": age_seconds,
                        "age_minutes": age_seconds / 60,
                        "is_fresh": age_seconds < 300,  # Fresh if less than 5 minutes old
                        "needs_refetch": age_seconds > 600  # Needs refetch if older than 10 minutes
                    }
                except Exception:
                    pass
        else:
            result["error"] = "No IV data available (options data may not be available)"
    
    except Exception as e:
        result["error"] = str(e)
        logger.error(f"Error getting IV info for {symbol}: {e}")
    
    return result


def format_price_display(price_info: Dict[str, Any], timezone_str: Optional[str] = None, latest_only: bool = False) -> List[str]:
    """Format price information for display."""
    lines = []
    
    symbol = price_info.get("symbol", "N/A")
    current_price = price_info.get("current_price")
    price_data = price_info.get("price_data")
    error = price_info.get("error")
    
    lines.append(f"\n{'='*80}")
    lines.append(f"PRICE INFORMATION: {symbol}")
    if latest_only:
        lines.append("(Latest price only - no historical data)")
    lines.append(f"{'='*80}")
    
    if error:
        lines.append(f"Error: {error}")
        return lines
    
    # Display current/latest price
    if current_price:
        lines.append("\nCurrent/Latest Price:")
        price_lines = _format_price_block(current_price, timezone_str or "America/New_York")
        lines.extend(price_lines)
        
        # Show data source and cache info
        source = current_price.get("source", "N/A")
        cache_hit = current_price.get("cache_hit", False)
        fetch_time = current_price.get("fetch_time_ms")
        
        source_display = source
        if cache_hit:
            source_display += " (CACHED)"
        elif source == "database":
            source_display += " (DB)"
        elif source in ["polygon_quote", "polygon_trade", "polygon_daily"]:
            source_display += " (API)"
        
        lines.append(f"Source: {source_display}")
        
        # Show timing info if available
        if fetch_time is not None:
            timing_info = f"Fetch time: {fetch_time:.1f}ms"
            if current_price.get("cache_check_time_ms"):
                timing_info += f" (cache check: {current_price.get('cache_check_time_ms'):.1f}ms)"
            if current_price.get("db_check_time_ms"):
                timing_info += f" (db check: {current_price.get('db_check_time_ms'):.1f}ms)"
            if current_price.get("api_fetch_time_ms"):
                timing_info += f" (api: {current_price.get('api_fetch_time_ms'):.1f}ms)"
            lines.append(timing_info)
    
    # Display historical price data if available
    if price_data is not None and not price_data.empty:
        lines.append(f"\nHistorical Price Data ({len(price_data)} rows):")
        display_cols = ['open', 'high', 'low', 'close']
        if 'volume' in price_data.columns:
            display_cols.append('volume')
        
        available_cols = [col for col in display_cols if col in price_data.columns]
        if available_cols:
            # Show summary stats
            lines.append(f"\nDate Range: {price_data.index.min()} to {price_data.index.max()}")
            lines.append(f"\nSummary Statistics:")
            summary = price_data[available_cols].describe()
            lines.append(str(summary))
            
            # Show first and last few rows
            lines.append(f"\nFirst 5 rows:")
            lines.append(str(price_data[available_cols].head()))
            lines.append(f"\nLast 5 rows:")
            lines.append(str(price_data[available_cols].tail()))
    
    return lines


def format_options_display(options_info: Dict[str, Any], max_options_per_expiry: int = 10) -> List[str]:
    """Format options information for display."""
    lines = []
    
    symbol = options_info.get("symbol", "N/A")
    options_data = options_info.get("options_data")
    error = options_info.get("error")
    source = options_info.get("source", "unknown")
    fetch_time = options_info.get("fetch_time_ms")
    
    lines.append(f"\n{'='*80}")
    lines.append(f"OPTIONS INFORMATION: {symbol}")
    lines.append(f"{'='*80}")
    
    # Show source and timing
    if source:
        source_display = source.upper()
        if source == "cache":
            source_display += " (CACHED)"
        elif source == "database":
            source_display += " (DB)"
        elif source == "api":
            source_display += " (API)"
        lines.append(f"Source: {source_display}")
    if fetch_time is not None:
        lines.append(f"Fetch time: {fetch_time:.1f}ms")
    if source or fetch_time is not None:
        lines.append("")
    
    if error:
        lines.append(f"Error: {error}")
        return lines
    
    if not options_data:
        lines.append("No options data available")
        return lines
    
    if not options_data.get("success", False):
        error_msg = options_data.get("error", "Unknown error")
        lines.append(f"Error fetching options: {error_msg}")
        return lines
    
    contracts = options_data.get("data", {}).get("contracts", [])
    
    if not contracts:
        lines.append("No options contracts found")
        return lines
    
    lines.append(f"\nFound {len(contracts)} options contracts")
    
    # Group by expiration date
    by_expiry = {}
    for contract in contracts:
        exp = contract.get("expiration", "Unknown")
        if exp not in by_expiry:
            by_expiry[exp] = []
        by_expiry[exp].append(contract)
    
    # Display by expiration
    for exp_date in sorted(by_expiry.keys())[:20]:  # Limit to first 20 expirations
        contracts_for_exp = by_expiry[exp_date]
        lines.append(f"\n--- Expiration: {exp_date} ({len(contracts_for_exp)} contracts) ---")
        
        # Sort by type and strike
        contracts_for_exp.sort(key=lambda x: (x.get("type", ""), x.get("strike", 0)))
        
        # Limit to max_options_per_expiry
        display_contracts = contracts_for_exp[:max_options_per_expiry]
        
        # Create table
        table_data = []
        for c in display_contracts:
            table_data.append([
                c.get("ticker", "N/A"),
                c.get("type", "N/A").upper(),
                f"${c.get('strike', 0):.2f}",
                f"${c.get('bid', 0):.2f}" if c.get("bid") is not None else "N/A",
                f"${c.get('ask', 0):.2f}" if c.get("ask") is not None else "N/A",
                f"${c.get('day_close', 0):.2f}" if c.get("day_close") is not None else "N/A",
                f"{c.get('delta', 0):.3f}" if c.get("delta") is not None else "N/A",
                f"{c.get('gamma', 0):.3f}" if c.get("gamma") is not None else "N/A",
                f"{c.get('theta', 0):.3f}" if c.get("theta") is not None else "N/A",
                f"{c.get('vega', 0):.3f}" if c.get("vega") is not None else "N/A",
                f"{c.get('implied_volatility', 0):.3f}" if c.get("implied_volatility") is not None else "N/A",
            ])
        
        if table_data:
            try:
                from tabulate import tabulate
                lines.append(tabulate(
                    table_data,
                    headers=["Ticker", "Type", "Strike", "Bid", "Ask", "Close", "Delta", "Gamma", "Theta", "Vega", "IV"],
                    tablefmt="grid"
                ))
            except ImportError:
                # Fallback if tabulate not available
                lines.append("Ticker | Type | Strike | Bid | Ask | Close | Delta | Gamma | Theta | Vega | IV")
                lines.append("-" * 80)
                for row in table_data:
                    lines.append(" | ".join(str(x) for x in row))
        
        if len(contracts_for_exp) > max_options_per_expiry:
            lines.append(f"... and {len(contracts_for_exp) - max_options_per_expiry} more contracts for this expiration")
    
    if len(by_expiry) > 20:
        lines.append(f"\n... and {len(by_expiry) - 20} more expiration dates")
    
    return lines


def format_financial_display(financial_info: Dict[str, Any]) -> List[str]:
    """Format financial ratios information for display."""
    lines = []
    
    symbol = financial_info.get("symbol", "N/A")
    financial_data = financial_info.get("financial_data")
    error = financial_info.get("error")
    source = financial_info.get("source", "unknown")
    fetch_time = financial_info.get("fetch_time_ms")
    
    lines.append(f"\n{'='*80}")
    lines.append(f"FINANCIAL RATIOS: {symbol}")
    lines.append(f"{'='*80}")
    
    # Show source and timing
    source_display = source.upper() if source else "UNKNOWN"
    if source == "cache":
        source_display += " (CACHED)"
    elif source == "database":
        source_display += " (DB)"
    elif source == "api":
        source_display += " (API)"
    lines.append(f"Source: {source_display}")
    if fetch_time is not None:
        lines.append(f"Fetch time: {fetch_time:.1f}ms")
    lines.append("")
    
    if error:
        lines.append(f"Error: {error}")
        return lines
    
    if not financial_data:
        lines.append("No financial ratios data available")
        return lines
    
    # Display key ratios
    key_ratios = [
        ("P/E Ratio", "price_to_earnings"),
        ("P/B Ratio", "price_to_book"),
        ("P/S Ratio", "price_to_sales"),
        ("PEG Ratio", "peg_ratio"),
        ("Debt-to-Equity", "debt_to_equity"),
        ("Return on Equity", "return_on_equity"),
        ("Return on Assets", "return_on_assets"),
        ("Current Ratio", "current"),
        ("Quick Ratio", "quick"),
        ("Cash Ratio", "cash"),
        ("Dividend Yield", "dividend_yield"),
        ("Market Cap", "market_cap"),
        ("Enterprise Value", "enterprise_value"),
        ("Free Cash Flow", "free_cash_flow"),
        ("EV to Sales", "ev_to_sales"),
        ("EV to EBITDA", "ev_to_ebitda"),
        ("Price to Cash Flow", "price_to_cash_flow"),
        ("Price to Free Cash Flow", "price_to_free_cash_flow"),
    ]
    
    table_data = []
    for label, key in key_ratios:
        value = financial_data.get(key)
        if value is not None:
            if isinstance(value, (int, float)):
                if abs(value) >= 1e9:
                    value_str = f"${value/1e9:.2f}B"
                elif abs(value) >= 1e6:
                    value_str = f"${value/1e6:.2f}M"
                elif abs(value) >= 1e3:
                    value_str = f"${value/1e3:.2f}K"
                else:
                    value_str = f"{value:.2f}"
            else:
                value_str = str(value)
        else:
            value_str = "N/A"
        table_data.append([label, value_str])
    
    if table_data:
        try:
            from tabulate import tabulate
            lines.append(tabulate(table_data, headers=["Metric", "Value"], tablefmt="grid"))
        except ImportError:
            # Fallback if tabulate not available
            lines.append("Metric | Value")
            lines.append("-" * 80)
            for row in table_data:
                lines.append(" | ".join(str(x) for x in row))
    
    return lines

def format_news_display(news_info: Dict[str, Any]) -> List[str]:
    """Format news information for display."""
    lines = []
    
    symbol = news_info.get("symbol", "N/A")
    news_data = news_info.get("news_data")
    error = news_info.get("error")
    freshness = news_info.get("freshness")
    
    lines.append(f"\n{'='*80}")
    lines.append(f"LATEST NEWS: {symbol}")
    lines.append(f"{'='*80}")
    
    if error:
        lines.append(f"Error: {error}")
        return lines
    
    if not news_data or not news_data.get("articles"):
        lines.append("No news articles available")
        return lines
    
    # Display freshness info
    if freshness:
        age_min = freshness.get("age_minutes", 0)
        status = "🟢 FRESH" if freshness.get("is_fresh") else "🟡 STALE" if not freshness.get("needs_refetch") else "🔴 NEEDS REFETCH"
        lines.append(f"Data Status: {status} (Age: {age_min:.1f} minutes)")
        if freshness.get("needs_refetch"):
            lines.append("⚠️  Recommendation: Refetch from source (data is >2 hours old)")
        lines.append("")
    
    articles = news_data.get("articles", [])
    lines.append(f"Found {len(articles)} news articles (from {news_data.get('date_range', {}).get('start', 'N/A')} to {news_data.get('date_range', {}).get('end', 'N/A')})")
    lines.append(f"Fetched at: {news_data.get('fetched_at', 'N/A')}")
    lines.append("")
    
    # Display top articles
    for i, article in enumerate(articles[:10], 1):  # Show top 10
        lines.append(f"{i}. {article.get('title', 'No title')}")
        if article.get('published_utc'):
            lines.append(f"   Published: {article['published_utc']}")
        if article.get('publisher', {}).get('name'):
            lines.append(f"   Source: {article['publisher']['name']}")
        if article.get('description'):
            desc = article['description'][:200] + "..." if len(article.get('description', '')) > 200 else article.get('description', '')
            lines.append(f"   {desc}")
        if article.get('article_url'):
            lines.append(f"   URL: {article['article_url']}")
        lines.append("")
    
    if len(articles) > 10:
        lines.append(f"... and {len(articles) - 10} more articles")
    
    return lines

def format_iv_display(iv_info: Dict[str, Any]) -> List[str]:
    """Format IV information for display."""
    lines = []
    
    symbol = iv_info.get("symbol", "N/A")
    iv_data = iv_info.get("iv_data")
    error = iv_info.get("error")
    freshness = iv_info.get("freshness")
    
    lines.append(f"\n{'='*80}")
    lines.append(f"LATEST IMPLIED VOLATILITY: {symbol}")
    lines.append(f"{'='*80}")
    
    if error:
        lines.append(f"Error: {error}")
        return lines
    
    if not iv_data:
        lines.append("No IV data available (options data may not be available)")
        return lines
    
    # Display freshness info
    if freshness:
        age_min = freshness.get("age_minutes", 0)
        status = "🟢 FRESH" if freshness.get("is_fresh") else "🟡 STALE" if not freshness.get("needs_refetch") else "🔴 NEEDS REFETCH"
        lines.append(f"Data Status: {status} (Age: {age_min:.1f} minutes)")
        if freshness.get("needs_refetch"):
            lines.append("⚠️  Recommendation: Refetch from source (data is >10 minutes old)")
        lines.append("")
    
    stats = iv_data.get("statistics", {})
    lines.append(f"Data timestamp: {iv_data.get('data_timestamp', 'N/A')}")
    lines.append(f"Fetched at: {iv_data.get('fetched_at', 'N/A')}")
    if iv_data.get('current_price'):
        lines.append(f"Current price: ${iv_data['current_price']:.2f}")
    lines.append("")
    
    lines.append("IV Statistics:")
    lines.append(f"  Count: {stats.get('count', 'N/A')}")
    if stats.get('mean') is not None:
        lines.append(f"  Mean IV: {stats['mean']:.4f} ({stats['mean']*100:.2f}%)")
    if stats.get('median') is not None:
        lines.append(f"  Median IV: {stats['median']:.4f} ({stats['median']*100:.2f}%)")
    if stats.get('min') is not None:
        lines.append(f"  Min IV: {stats['min']:.4f} ({stats['min']*100:.2f}%)")
    if stats.get('max') is not None:
        lines.append(f"  Max IV: {stats['max']:.4f} ({stats['max']*100:.2f}%)")
    if stats.get('std') is not None:
        lines.append(f"  Std Dev: {stats['std']:.4f}")
    
    if 'atm_iv' in iv_data:
        atm = iv_data['atm_iv']
        if atm.get('mean') is not None:
            lines.append("")
            lines.append("ATM IV (within 5% of current price):")
            lines.append(f"  Mean: {atm['mean']:.4f} ({atm['mean']*100:.2f}%)")
            lines.append(f"  Count: {atm.get('count', 'N/A')}")
    
    if 'call_iv' in iv_data:
        call = iv_data['call_iv']
        if call.get('mean') is not None:
            lines.append("")
            lines.append("Call Options IV:")
            lines.append(f"  Mean: {call['mean']:.4f} ({call['mean']*100:.2f}%)")
            lines.append(f"  Count: {call.get('count', 'N/A')}")
    
    if 'put_iv' in iv_data:
        put = iv_data['put_iv']
        if put.get('mean') is not None:
            lines.append("")
            lines.append("Put Options IV:")
            lines.append(f"  Mean: {put['mean']:.4f} ({put['mean']*100:.2f}%)")
            lines.append(f"  Count: {put.get('count', 'N/A')}")
    
    return lines


async def process_symbol(
    symbol: str,
    db_instance: StockDBBase,
    args: argparse.Namespace
) -> Dict[str, Any]:
    """Process a single symbol and gather all information."""
    result = {
        "symbol": symbol,
        "price_info": None,
        "options_info": None,
        "financial_info": None,
        "news_info": None,
        "iv_info": None
    }
    
    # If --latest is set, clear start_date and end_date
    start_date = None if args.latest else args.start_date
    end_date = None if args.latest else args.end_date
    
    # Gather all information in parallel
    # Create all tasks first to ensure they start simultaneously
    import time
    parallel_start = time.time()
    
    tasks = []
    task_indices = {}
    
    # Price info (always needed)
    tasks.append(get_price_info(
        symbol,
        db_instance,
        start_date=start_date,
        end_date=end_date,
        force_fetch=args.force_fetch,
        data_source=args.data_source,
        timezone_str=args.timezone,
        latest_only=args.latest
    ))
    task_indices['price'] = len(tasks) - 1
    
    # Options info (always needed for display)
    tasks.append(get_options_info(
        symbol,
        db_instance,
        options_days=args.options_days,
        force_fetch=args.force_fetch,
        data_source=args.data_source,
        option_type=args.options_type,
        strike_range_percent=args.strike_range_percent,
        max_options_per_expiry=args.max_options_per_expiry,
        enable_cache=not args.no_cache,
        redis_url=os.getenv("REDIS_URL") if not args.no_cache else None
    ))
    task_indices['options'] = len(tasks) - 1
    
    # Financial info (always needed)
    tasks.append(get_financial_info(
        symbol,
        db_instance,
        force_fetch=args.force_fetch
    ))
    task_indices['financial'] = len(tasks) - 1
    
    # Add news and IV tasks if requested
    if args.show_news:
        tasks.append(get_news_info(
            symbol,
            db_instance,
            force_fetch=args.force_fetch,
            enable_cache=not args.no_cache
        ))
        task_indices['news'] = len(tasks) - 1
    else:
        # Create a completed future that returns None immediately
        async def return_none_news():
            return None
        tasks.append(return_none_news())
        task_indices['news'] = len(tasks) - 1
    
    if args.show_iv:
        tasks.append(get_iv_info(
            symbol,
            db_instance,
            force_fetch=args.force_fetch,
            enable_cache=not args.no_cache
        ))
        task_indices['iv'] = len(tasks) - 1
    else:
        # Create a completed future that returns None immediately
        async def return_none_iv():
            return None
        tasks.append(return_none_iv())
        task_indices['iv'] = len(tasks) - 1
    
    # Execute all tasks in parallel - asyncio.gather runs them concurrently
    logger.debug(f"[PARALLEL] Starting {len(tasks)} parallel data fetches for {symbol}")
    results = await asyncio.gather(*tasks, return_exceptions=True)
    parallel_time = (time.time() - parallel_start) * 1000
    logger.debug(f"[PARALLEL] Completed all {len(tasks)} parallel fetches for {symbol} in {parallel_time:.1f}ms")
    
    # Handle exceptions and extract results
    result["price_info"] = results[task_indices['price']] if not isinstance(results[task_indices['price']], Exception) else {"error": str(results[task_indices['price']])}
    result["options_info"] = results[task_indices['options']] if not isinstance(results[task_indices['options']], Exception) else {"error": str(results[task_indices['options']])}
    result["financial_info"] = results[task_indices['financial']] if not isinstance(results[task_indices['financial']], Exception) else {"error": str(results[task_indices['financial']])}
    result["news_info"] = results[task_indices['news']] if len(results) > task_indices['news'] and not isinstance(results[task_indices['news']], Exception) and results[task_indices['news']] else None
    result["iv_info"] = results[task_indices['iv']] if len(results) > task_indices['iv'] and not isinstance(results[task_indices['iv']], Exception) and results[task_indices['iv']] else None
    
    return result


async def main():
    """Main entry point."""
    global args
    args = parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format='%(asctime)s [%(levelname)s] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Check Polygon availability if needed
    if args.data_source == "polygon" and not POLYGON_AVAILABLE:
        print("Error: Polygon.io data source selected but polygon-api-client is not installed.", file=sys.stderr)
        print("Install with: pip install polygon-api-client", file=sys.stderr)
        sys.exit(1)
    
    # Initialize database instance
    enable_cache = not args.no_cache
    db_instance = None
    
    try:
        if args.db_path and ':' in args.db_path:
            if args.db_path.startswith('questdb://'):
                db_instance = get_stock_db(
                    "questdb",
                    args.db_path,
                    log_level=args.log_level,
                    enable_cache=enable_cache,
                    redis_url=os.getenv('REDIS_URL', 'redis://localhost:6379/0') if enable_cache else None
                )
            elif args.db_path.startswith('postgresql://'):
                db_instance = get_stock_db("postgresql", args.db_path, log_level=args.log_level)
            else:
                db_instance = get_stock_db("remote", args.db_path, log_level=args.log_level)
        else:
            actual_db_path = args.db_path or (get_default_db_path("duckdb") if args.db_type == 'duckdb' else get_default_db_path("db"))
            db_instance = get_stock_db(args.db_type, actual_db_path, log_level=args.log_level)
        
        # Process all symbols
        import time
        overall_start = time.time()
        results = []
        for symbol in args.symbols:
            symbol_start = time.time()
            logger.info(f"[TIMING] Starting data fetch for {symbol}")
            symbol_result = await process_symbol(symbol, db_instance, args)
            symbol_time = (time.time() - symbol_start) * 1000
            logger.info(f"[TIMING] Completed data fetch for {symbol} in {symbol_time:.1f}ms")
            results.append(symbol_result)
        
        overall_time = (time.time() - overall_start) * 1000
        logger.info(f"[TIMING] Total time for all symbols: {overall_time:.1f}ms")
        
        # Display results
        for result in results:
            symbol = result["symbol"]
            
            # Price information
            if result["price_info"]:
                price_lines = format_price_display(result["price_info"], args.timezone, latest_only=args.latest)
                print("\n".join(price_lines))
            
            # Options information
            if result["options_info"]:
                options_lines = format_options_display(result["options_info"], args.max_options_per_expiry)
                print("\n".join(options_lines))
            
            # Financial information
            if result["financial_info"]:
                financial_lines = format_financial_display(result["financial_info"])
                print("\n".join(financial_lines))
            
            # News information
            if result["news_info"]:
                news_lines = format_news_display(result["news_info"])
                print("\n".join(news_lines))
            
            # IV information
            if result["iv_info"]:
                iv_lines = format_iv_display(result["iv_info"])
                print("\n".join(iv_lines))
            
            print()  # Spacing between symbols
        
        # Print cache statistics if available
        if db_instance and hasattr(db_instance, 'get_cache_statistics'):
            try:
                cache_stats = db_instance.get_cache_statistics()
                if cache_stats and (cache_stats.get('hits', 0) > 0 or cache_stats.get('misses', 0) > 0):
                    print("\n" + "=" * 80)
                    print("Cache Statistics")
                    print("=" * 80)
                    print(f"Hits:        {cache_stats.get('hits', 0)}")
                    print(f"Misses:      {cache_stats.get('misses', 0)}")
                    print(f"Sets:        {cache_stats.get('sets', 0)}")
                    print(f"Invalidations: {cache_stats.get('invalidations', 0)}")
                    print(f"Errors:      {cache_stats.get('errors', 0)}")
                    total = cache_stats.get('hits', 0) + cache_stats.get('misses', 0)
                    if total > 0:
                        hit_rate = (cache_stats.get('hits', 0) / total) * 100
                        print(f"Hit Rate:    {hit_rate:.2f}%")
                    print("=" * 80 + "\n")
            except Exception as e:
                if args.log_level == "DEBUG":
                    print(f"Error getting cache statistics: {e}", file=sys.stderr)
    
    finally:
        # Cleanup
        if db_instance and hasattr(db_instance, 'cache') and hasattr(db_instance.cache, 'wait_for_pending_writes'):
            try:
                await db_instance.cache.wait_for_pending_writes(timeout=10.0)
            except Exception:
                pass
        
        if db_instance and hasattr(db_instance, 'close_session') and callable(db_instance.close_session):
            try:
                await db_instance.close_session()
            except Exception as e:
                logger.debug(f"Error closing database session: {e}")


if __name__ == "__main__":
    asyncio.run(main())

