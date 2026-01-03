"""
Financial Data Module

This module provides functionality for:
- Fetching financial ratios from Polygon API
- Caching financial data (Redis and database)
- Integrating IV analysis when syncing
- Parallel execution support (async and multiprocessing)
"""

import os
import json
import time
import random
import threading
import logging
import asyncio
import aiohttp
from datetime import datetime, timezone, date
from typing import Optional, Dict, Any, Tuple
import pandas as pd

from .stock_db import StockDBBase
from .redis_cache import CacheKeyGenerator
from .market_hours import is_market_hours, is_market_preopen, is_market_postclose
from .logging_utils import get_logger
from .iv_analysis import IVAnalyzer

logger = get_logger(__name__)

# Background fetch tracking
_active_background_fetches: set = set()  # Set of (symbol, data_type) tuples
_background_fetch_lock = threading.Lock()  # Thread lock for thread-safe set operations

# Market hours thresholds (in seconds)
MARKET_DEFAULT_THRESHOLD = 12 * 60 * 60  # 12 hours
MARKET_CLOSE_THRESHOLD = MARKET_DEFAULT_THRESHOLD
MARKET_OPEN_THRESHOLD = 30 * 60  # 30 minutes
MARKET_PREOPEN_THRESHOLD = 2 * 60 * 60  # 2 hours
MARKET_POSTCLOSE_THRESHOLD = MARKET_PREOPEN_THRESHOLD


def _jitter_threshold(base_threshold: int | float) -> float:
    """Apply a ±10% randomization to the threshold so that checks are slightly
    staggered instead of all firing exactly at the same threshold boundary."""
    if base_threshold <= 0:
        return base_threshold
    # Random factor in [0.9, 1.1]
    factor = random.uniform(0.9, 1.1)
    return base_threshold * factor


def _should_trigger_background_fetch(
    last_save_time: Optional[datetime], 
    data_type: str = "price", 
    symbol: str = ""
) -> bool:
    """Check if background fetch should be triggered based on last save time and market hours.
    
    Args:
        last_save_time: Last time data was saved (UTC datetime or None)
        data_type: Type of data ("price", "options", "financial", "news", "iv")
        symbol: Stock symbol (used to prevent duplicate background fetches)
    
    Returns:
        True if background fetch should be triggered, False otherwise
    """
    # Check if a background fetch is already in progress for this symbol/data_type
    if symbol:
        fetch_key = (symbol.upper(), data_type)
        with _background_fetch_lock:
            if fetch_key in _active_background_fetches:
                logger.debug(f"[BACKGROUND FETCH] Skipping background fetch for {symbol} {data_type} - already in progress")
                return False
    
    if last_save_time is None:
        return True  # No data cached, should fetch
    
    # Ensure last_save_time is timezone-aware UTC
    if isinstance(last_save_time, str):
        last_save_time = datetime.fromisoformat(last_save_time.replace('Z', '+00:00'))
    if last_save_time.tzinfo is None:
        last_save_time = last_save_time.replace(tzinfo=timezone.utc)
    elif last_save_time.tzinfo != timezone.utc:
        last_save_time = last_save_time.astimezone(timezone.utc)
    
    now_utc = datetime.now(timezone.utc)
    age_seconds = (now_utc - last_save_time).total_seconds()

    effective_threshold = _jitter_threshold(MARKET_DEFAULT_THRESHOLD)
    if is_market_hours():  # Check if market is open
        effective_threshold = _jitter_threshold(MARKET_OPEN_THRESHOLD)
    elif is_market_preopen():
        effective_threshold = _jitter_threshold(MARKET_PREOPEN_THRESHOLD)
    elif is_market_postclose():
        effective_threshold = _jitter_threshold(MARKET_POSTCLOSE_THRESHOLD)

    return age_seconds > effective_threshold


def _get_last_save_time_from_cache(cached_df: Optional[pd.DataFrame]) -> Optional[datetime]:
    """Extract last_save_time from cached DataFrame.
    
    Args:
        cached_df: Cached DataFrame that may contain last_save_time column
    
    Returns:
        last_save_time as datetime or None
    """
    if cached_df is None or cached_df.empty:
        return None
    
    try:
        # Check if last_save_time is in the DataFrame
        if 'last_save_time' in cached_df.columns:
            last_save = cached_df.iloc[0]['last_save_time']
            if pd.isna(last_save) or last_save is None:
                return None
            # Convert to datetime if it's a string
            if isinstance(last_save, str):
                return datetime.fromisoformat(last_save.replace('Z', '+00:00'))
            elif isinstance(last_save, pd.Timestamp):
                return last_save.to_pydatetime()
            elif isinstance(last_save, datetime):
                return last_save
        return None
    except Exception:
        return None


async def _trigger_background_fetch(
    symbol: str,
    db_instance: StockDBBase,
    data_type: str,
    fetch_func,
    *args,
    **kwargs
) -> None:
    """Trigger a background fetch for data.
    
    Args:
        symbol: Stock symbol
        db_instance: Database instance
        data_type: Type of data being fetched
        fetch_func: Async function to call for fetching
        *args, **kwargs: Arguments to pass to fetch_func
    """
    # Check if already in progress
    fetch_key = (symbol.upper(), data_type)
    with _background_fetch_lock:
        if fetch_key in _active_background_fetches:
            logger.debug(f"[BACKGROUND FETCH] Skipping duplicate background fetch for {symbol} {data_type}")
            return
        _active_background_fetches.add(fetch_key)
    
    try:
        # Create background task (fire-and-forget)
        async def _background_fetch():
            try:
                logger.debug(f"[BACKGROUND FETCH] Starting background fetch for {data_type} data: {symbol}")
                await fetch_func(*args, **kwargs)
                logger.debug(f"[BACKGROUND FETCH] Completed background fetch for {data_type} data: {symbol}")
            except Exception as e:
                logger.warning(f"[BACKGROUND FETCH] Error in background fetch for {data_type} data {symbol}: {e}")
            finally:
                # Remove from active fetches when done
                with _background_fetch_lock:
                    _active_background_fetches.discard(fetch_key)
        
        # Create task without awaiting (fire-and-forget)
        asyncio.create_task(_background_fetch())
        logger.debug(f"[BACKGROUND FETCH] Triggered background fetch task for {data_type} data: {symbol}")
    except Exception as e:
        logger.debug(f"[BACKGROUND FETCH] Failed to trigger background fetch for {data_type} data {symbol}: {e}")
        with _background_fetch_lock:
            _active_background_fetches.discard(fetch_key)


async def get_financial_ratios(ticker: str, api_key: str) -> Optional[Dict[str, Any]]:
    """Fetch financial ratios (P/E, P/B, etc.) from Polygon.io for a given ticker.
    
    Args:
        ticker: Stock ticker symbol
        api_key: Polygon.io API key
    
    Returns:
        Dictionary with financial ratios or None on error
    """
    try:
        url = f"https://api.polygon.io/stocks/financials/v1/ratios"
        params = {
            "ticker": ticker,
            "apiKey": api_key
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    if data.get("status") == "OK" and data.get("results"):
                        results = data["results"]
                        # Handle case where results is a list - take the first item
                        if isinstance(results, list) and len(results) > 0:
                            return results[0]
                        elif isinstance(results, dict):
                            return results
                        else:
                            logger.warning(f"Unexpected results format for {ticker}: {type(results)}")
                            return None
                    else:
                        # More detailed error logging
                        status = data.get("status", "UNKNOWN")
                        message = data.get("message", "")
                        results_count = len(data.get("results", [])) if isinstance(data.get("results"), (list, dict)) else 0
                        logger.debug(f"Polygon API returned status={status}, message={message}, results_count={results_count} for {ticker}")
                        return None
                else:
                    error_text = await response.text()
                    logger.warning(f"Polygon API returned status {response.status} for {ticker}: {error_text}")
                    return None
    except Exception as e:
        logger.error(f"Error fetching financial ratios for {ticker}: {e}")
        return None


async def _calculate_iv_analysis(
    symbol: str,
    db_instance: StockDBBase,
    calendar_days: int = 90,
    polygon_api_key: Optional[str] = None,
    redis_url: Optional[str] = None,
    server_url: Optional[str] = None,
    data_dir: str = "data"
) -> Optional[Dict[str, Any]]:
    """Calculate IV analysis for a symbol.
    
    Args:
        symbol: Stock ticker symbol
        db_instance: Database instance for fetching price history
        calendar_days: Days ahead to check for earnings
        polygon_api_key: Polygon API key (for IV data only)
        redis_url: Redis connection URL
        server_url: Local db_server URL for price history (fallback if db_instance fails)
        data_dir: Data directory for caching
    
    Returns:
        IV analysis result dictionary or None on error
    
    Note:
        Price history is fetched from database or HTTP server, NOT from Polygon API.
        Polygon API is only used for IV/options data.
    """
    try:
        if not polygon_api_key:
            polygon_api_key = os.getenv("POLYGON_API_KEY")
            if not polygon_api_key:
                logger.warning(f"No Polygon API key available for IV analysis of {symbol}")
                return None
        
        analyzer = IVAnalyzer(
            polygon_api_key=polygon_api_key,
            data_dir=data_dir,
            redis_url=redis_url,
            server_url=server_url,
            db_instance=db_instance,  # Pass database instance for direct access
            use_polygon=False,  # Never use Polygon for price history
            logger=logger
        )
        
        result, _ = await analyzer.get_iv_analysis(
            ticker=symbol,
            calendar_days=calendar_days,
            force_refresh=True  # Always force refresh when syncing
        )
        
        return result
    except Exception as e:
        logger.error(f"Error calculating IV analysis for {symbol}: {e}")
        return None


async def get_financial_info(
    symbol: str,
    db_instance: StockDBBase,
    force_fetch: bool = False,
    include_iv_analysis: bool = False,
    iv_calendar_days: int = 90,
    iv_server_url: Optional[str] = None,
    iv_use_polygon: bool = False,
    iv_data_dir: str = "data"
) -> Dict[str, Any]:
    """
    Get financial ratios/information for a symbol.
    
    When force_fetch=True and include_iv_analysis=True, both financial ratios
    and IV analysis are calculated together and saved to the database.
    
    Args:
        symbol: Stock ticker symbol
        db_instance: Database instance
        force_fetch: Force API refresh, bypassing cache
        include_iv_analysis: Include IV analysis when syncing (force_fetch=True)
        iv_calendar_days: Days ahead for IV earnings check
        iv_server_url: Local server URL for IV price history
        iv_use_polygon: Force using Polygon for IV price history
        iv_data_dir: Data directory for IV caching
    
    Returns:
        Dictionary with financial_data, source, error, and fetch_time_ms
    """
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
        cached_financial = None
        last_save_time = None
        if not force_fetch and cache_instance:
            try:
                cache_key = CacheKeyGenerator.financial_info(symbol)
                cache_check_start = time.time()
                cached_df = await cache_instance.get(cache_key)
                cache_check_time = (time.time() - cache_check_start) * 1000
                
                if cached_df is not None and not cached_df.empty:
                    # Convert DataFrame back to dict
                    cached_financial = cached_df.iloc[0].to_dict()
                    
                    # Parse IV analysis JSON if present
                    if 'iv_analysis_json' in cached_financial and cached_financial.get('iv_analysis_json'):
                        try:
                            iv_analysis = json.loads(cached_financial['iv_analysis_json'])
                            # Merge IV analysis data into financial_data for easy access
                            if 'metrics' in iv_analysis:
                                cached_financial['iv_metrics'] = iv_analysis['metrics']
                            if 'strategy' in iv_analysis:
                                cached_financial['iv_strategy'] = iv_analysis['strategy']
                            # Use relative_rank from JSON if database column is None
                            if cached_financial.get('relative_rank') is None and 'relative_rank' in iv_analysis:
                                cached_financial['relative_rank'] = iv_analysis['relative_rank']
                            # Also keep the full JSON for reference
                            cached_financial['iv_analysis'] = iv_analysis
                        except (json.JSONDecodeError, TypeError) as e:
                            logger.debug(f"[FINANCIAL IV PARSE] Could not parse IV analysis JSON from cache for {symbol}: {e}")
                    
                    # Get last_save_time
                    last_save_time = _get_last_save_time_from_cache(cached_df)
                    fetch_time = (time.time() - fetch_start) * 1000
                    result["financial_data"] = cached_financial
                    result["source"] = "cache"
                    result["fetch_time_ms"] = fetch_time
                    logger.info(f"[FINANCIAL CACHE HIT] Financial data for {symbol} (cache_check: {cache_check_time:.1f}ms, total: {fetch_time:.1f}ms)")
                    
                    # Check if background fetch should be triggered
                    if _should_trigger_background_fetch(last_save_time, "financial", symbol):
                        # Trigger background fetch but return cached data immediately
                        async def _fetch_financial_background():
                            try:
                                # Re-fetch financial data - use force_fetch=True to bypass cache
                                financial_result = await get_financial_info(
                                    symbol, db_instance, force_fetch=True,
                                    include_iv_analysis=include_iv_analysis,
                                    iv_calendar_days=iv_calendar_days,
                                    iv_server_url=iv_server_url,
                                    iv_use_polygon=iv_use_polygon,
                                    iv_data_dir=iv_data_dir
                                )
                                return financial_result
                            except Exception as e:
                                logger.warning(f"Background financial fetch failed for {symbol}: {e}")
                                return None
                        
                        # Fire-and-forget: create task without awaiting
                        await _trigger_background_fetch(
                            symbol, db_instance, "financial", _fetch_financial_background
                        )
                    
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
                    
                    # Parse IV analysis JSON if present
                    if 'iv_analysis_json' in latest and latest.get('iv_analysis_json'):
                        try:
                            iv_analysis = json.loads(latest['iv_analysis_json'])
                            # Merge IV analysis data into financial_data for easy access
                            if 'metrics' in iv_analysis:
                                latest['iv_metrics'] = iv_analysis['metrics']
                            if 'strategy' in iv_analysis:
                                latest['iv_strategy'] = iv_analysis['strategy']
                            # Use relative_rank from JSON if database column is None
                            if latest.get('relative_rank') is None and 'relative_rank' in iv_analysis:
                                latest['relative_rank'] = iv_analysis['relative_rank']
                            # Also keep the full JSON for reference
                            latest['iv_analysis'] = iv_analysis
                        except (json.JSONDecodeError, TypeError) as e:
                            logger.debug(f"[FINANCIAL IV PARSE] Could not parse IV analysis JSON for {symbol}: {e}")
                    
                    fetch_time = (time.time() - fetch_start) * 1000
                    result["financial_data"] = latest
                    result["source"] = "database"
                    result["fetch_time_ms"] = fetch_time
                    logger.info(f"[FINANCIAL DB HIT] Financial data for {symbol} (db_check: {db_check_time:.1f}ms, total: {fetch_time:.1f}ms)")
                    
                    # Cache the result
                    if cache_instance:
                        try:
                            cache_key = CacheKeyGenerator.financial_info(symbol)
                            # Add last_save_time to cached data
                            latest_with_time = latest.copy()
                            latest_with_time['last_save_time'] = datetime.now(timezone.utc).isoformat()
                            cache_df = pd.DataFrame([latest_with_time])
                            cache_set_start = time.time()
                            await cache_instance.set(cache_key, cache_df, ttl=None)  # No TTL (infinite cache)
                            cache_set_time = (time.time() - cache_set_start) * 1000
                            logger.info(f"[FINANCIAL CACHE SET] Cached financial data for {symbol} (set_time: {cache_set_time:.1f}ms, no TTL)")
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
        
        # Calculate IV analysis if requested and syncing
        iv_analysis_result = None
        if force_fetch and include_iv_analysis:
            iv_fetch_start = time.time()
            iv_analysis_result = await _calculate_iv_analysis(
                symbol=symbol,
                db_instance=db_instance,  # Pass database instance for direct access
                calendar_days=iv_calendar_days,
                polygon_api_key=api_key,
                redis_url=os.getenv("REDIS_URL"),
                server_url=iv_server_url,
                data_dir=iv_data_dir
            )
            iv_fetch_time = (time.time() - iv_fetch_start) * 1000
            if iv_analysis_result:
                logger.info(f"[FINANCIAL IV] Calculated IV analysis for {symbol} (iv_fetch: {iv_fetch_time:.1f}ms)")
        
        fetch_time = (time.time() - fetch_start) * 1000
        
        if ratios:
            # Merge IV analysis into financial data if available
            if iv_analysis_result:
                # Extract IV metrics for database columns
                metrics = iv_analysis_result.get("metrics", {})
                strategy = iv_analysis_result.get("strategy", {})
                
                # Parse IV_30d from percentage string (e.g., "40.41%")
                iv_30d_str = metrics.get("iv_30d", "")
                iv_30d = None
                if iv_30d_str and iv_30d_str.endswith("%"):
                    try:
                        iv_30d = float(iv_30d_str.rstrip("%")) / 100.0
                    except (ValueError, AttributeError):
                        pass
                
                # Get rank values
                iv_rank = metrics.get("rank")
                relative_rank = iv_analysis_result.get("relative_rank")
                
                # Add IV columns to ratios
                ratios['iv_30d'] = iv_30d
                ratios['iv_rank'] = float(iv_rank) if iv_rank is not None else None
                ratios['relative_rank'] = float(relative_rank) if relative_rank is not None else None
                ratios['iv_analysis_json'] = json.dumps(iv_analysis_result)
                ratios['iv_analysis_spare'] = None  # Spare column for future use
            
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
                    cache_key = CacheKeyGenerator.financial_info(symbol)
                    # Add last_save_time to cached data
                    ratios_with_time = ratios.copy()
                    ratios_with_time['last_save_time'] = datetime.now(timezone.utc).isoformat()
                    cache_df = pd.DataFrame([ratios_with_time])
                    cache_set_start = time.time()
                    await cache_instance.set(cache_key, cache_df, ttl=None)  # No TTL (infinite cache)
                    cache_set_time = (time.time() - cache_set_start) * 1000
                    logger.info(f"[FINANCIAL CACHE SET] Cached financial data for {symbol} (set_time: {cache_set_time:.1f}ms, no TTL)")
                except Exception as e:
                    logger.debug(f"[FINANCIAL CACHE ERROR] Failed to cache financial data for {symbol}: {e}")
        else:
            result["error"] = "No financial ratios data available"
    
    except Exception as e:
        result["error"] = str(e)
        logger.error(f"[FINANCIAL ERROR] Error getting financial info for {symbol}: {e}")
    
    return result


# Multiprocessing support - synchronous wrapper for ProcessPoolExecutor
def get_financial_info_worker(
    symbol: str,
    db_config: str,
    db_type: str = "questdb",
    force_fetch: bool = False,
    include_iv_analysis: bool = False,
    iv_calendar_days: int = 90,
    iv_server_url: Optional[str] = None,
    iv_use_polygon: bool = False,
    iv_data_dir: str = "data",
    redis_url: Optional[str] = None,
    log_level: str = "ERROR"
) -> Dict[str, Any]:
    """
    Worker function for multiprocessing execution.
    Creates its own event loop and database connection.
    
    Args:
        symbol: Stock ticker symbol
        db_config: Database connection string
        db_type: Database type ("questdb", "postgres", etc.)
        force_fetch: Force API refresh
        include_iv_analysis: Include IV analysis when syncing
        iv_calendar_days: Days ahead for IV earnings check
        iv_server_url: Local server URL for IV price history
        iv_use_polygon: Force using Polygon for IV price history
        iv_data_dir: Data directory for IV caching
        redis_url: Redis connection URL
        log_level: Logging level
    
    Returns:
        Dictionary with financial_data, source, error, and fetch_time_ms
    """
    # Use asyncio.run() which properly handles event loop creation/cleanup
    # This is the recommended way for running async code from sync context
    # In multiprocessing, each worker process has a clean state, so this works well
    async def _async_worker():
        from .stock_db import get_stock_db
        
        # Initialize database connection
        db = get_stock_db(
            db_type, 
            db_config=db_config, 
            enable_cache=True,
            redis_url=redis_url,
            log_level=log_level,
            auto_init=False
        )
        
        try:
            # Initialize database
            await db._init_db()
            
            # Call async function
            result = await get_financial_info(
                symbol=symbol,
                db_instance=db,
                force_fetch=force_fetch,
                include_iv_analysis=include_iv_analysis,
                iv_calendar_days=iv_calendar_days,
                iv_server_url=iv_server_url,
                iv_use_polygon=iv_use_polygon,
                iv_data_dir=iv_data_dir
            )
            
            return result
        finally:
            # Close database connection
            await db.close()
    
    try:
        # Check if we're in a context with an existing event loop
        # This can happen in tests or when called from async context
        # In multiprocessing, each process has a clean state, so this shouldn't occur
        try:
            loop = asyncio.get_running_loop()
            # If we get here, there's a running loop - we can't use asyncio.run()
            # This means we're being called from an async context (like a test)
            # Return an error indicating this worker is only for multiprocessing
            error_msg = (
                f"Cannot use asyncio.run() when event loop is already running for {symbol}. "
                "This worker function is designed for multiprocessing contexts only. "
                "For async contexts, use get_financial_info() directly."
            )
            logger.error(error_msg)
            return {
                "symbol": symbol,
                "financial_data": None,
                "error": error_msg,
                "source": None,
                "fetch_time_ms": 0
            }
        except RuntimeError as e:
            if "no running event loop" in str(e).lower():
                # No running loop - safe to use asyncio.run()
                # This is the expected case for multiprocessing workers
                return asyncio.run(_async_worker())
            else:
                # Some other RuntimeError - re-raise
                raise
        
    except Exception as e:
        logger.error(f"Error in financial_info worker for {symbol}: {e}", exc_info=True)
        return {
            "symbol": symbol,
            "financial_data": None,
            "error": str(e),
            "source": None,
            "fetch_time_ms": 0  # Use 0 instead of None to avoid format errors
        }
