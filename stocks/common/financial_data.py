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
import numpy as np

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
    data_dir: str = "data",
    benchmark_ticker: str = "VOO"  # Changed from SPY to VOO
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
        benchmark_ticker: Benchmark ticker for relative rank calculation (default: VOO)
    
    Returns:
        IV analysis result dictionary with relative_rank included, or None on error
    
    Note:
        Price history is fetched from database or HTTP server, NOT from Polygon API.
        Polygon API is only used for IV/options data.
        Relative rank is calculated as: ticker_rank / benchmark_rank (ratio)
        A value of 1.0 means equal IV rank, >1.0 means ticker has higher IV rank, <1.0 means lower
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
        
        # Calculate IV analysis for the symbol
        result, _ = await analyzer.get_iv_analysis(
            ticker=symbol,
            calendar_days=calendar_days,
            force_refresh=True  # Always force refresh when syncing
        )
        
        # Calculate benchmark (VOO) IV rank for relative ranking
        # If symbol is the benchmark, set relative_rank to 1.0 (ratio of 1.0 = equal)
        if result:
            if symbol.upper() == benchmark_ticker.upper():
                # Symbol is the benchmark - relative rank is always 1.0 (ratio)
                result["relative_rank"] = 1.0
                logger.debug(f"[IV] Symbol {symbol} is the benchmark ({benchmark_ticker}), setting relative_rank to 1.0")
            else:
                # Calculate relative rank vs benchmark
                try:
                    benchmark_result, _ = await analyzer.get_iv_analysis(
                        ticker=benchmark_ticker,
                        calendar_days=calendar_days,
                        force_refresh=False  # Use cache for benchmark if available
                    )
                    
                    if benchmark_result and "metrics" in benchmark_result:
                        benchmark_rank = benchmark_result["metrics"].get("rank")
                        ticker_rank = result.get("metrics", {}).get("rank")
                        
                        if benchmark_rank is not None and ticker_rank is not None:
                            if benchmark_rank > 0:
                                # Calculate relative rank as ratio (ticker_rank / benchmark_rank)
                                # This shows how many times higher/lower the ticker's IV rank is vs benchmark
                                # Example: 1.46 means ticker is 46% higher, 0.5 means ticker is 50% lower
                                relative_rank = round(ticker_rank / benchmark_rank, 2)
                                result["relative_rank"] = relative_rank
                                logger.debug(f"[IV] Calculated relative_rank for {symbol}: {relative_rank} (ticker_rank={ticker_rank}, {benchmark_ticker}_rank={benchmark_rank})")
                            elif benchmark_rank == 0 and ticker_rank == 0:
                                # Both are 0, so they're equal
                                result["relative_rank"] = 1.0
                                logger.debug(f"[IV] Both {symbol} and {benchmark_ticker} have IV rank 0, setting relative_rank to 1.0")
                            else:
                                # Benchmark is 0 but ticker is not - can't calculate ratio meaningfully
                                # Set to None or a large number to indicate ticker is much higher
                                result["relative_rank"] = None
                                logger.debug(f"[IV] Cannot calculate relative_rank: benchmark_rank={benchmark_rank} (zero), ticker_rank={ticker_rank}")
                        else:
                            logger.debug(f"[IV] Could not calculate relative_rank: ticker_rank={ticker_rank}, benchmark_rank={benchmark_rank}")
                    else:
                        logger.debug(f"[IV] Could not get benchmark ({benchmark_ticker}) IV analysis for relative ranking")
                except Exception as benchmark_error:
                    logger.warning(f"[IV] Error calculating benchmark ({benchmark_ticker}) IV rank: {benchmark_error}")
                    # Continue without relative_rank if benchmark calculation fails
        
        return result
    except Exception as e:
        logger.error(f"Error calculating IV analysis for {symbol}: {e}")
        return None


async def get_financial_info(
    symbol: str,
    db_instance: StockDBBase,
    force_fetch: bool = False,
    cache_only: bool = False,  # If True, only serve from cache, never fetch from API
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
    
    # Initialize cache_instance early so it's available in all code paths
    cache_instance = None
    if hasattr(db_instance, 'cache') and db_instance.cache:
        cache_instance = db_instance.cache
    
    try:
        # Get financial data from database (uses FinancialDataService.get() which has its own cache)
        # No need for separate cache layer here - let FinancialDataService handle caching
        if cache_only:
            # Cache-only mode: only check database/cache, never fetch from API
            logger.debug(f"[FINANCIAL CACHE ONLY] Only checking database/cache for {symbol}, not fetching from API")
            force_fetch = False  # Override force_fetch in cache-only mode
            # Fall through to the existing database check logic below
        if not force_fetch:
            try:
                db_check_start = time.time()
                financial_df = await db_instance.get_financial_info(symbol)
                db_check_time = (time.time() - db_check_start) * 1000
                
                if not financial_df.empty:
                    # Get the most recent entry
                    latest = financial_df.iloc[-1].to_dict()
                    
                    # Ensure write_timestamp is properly formatted if present
                    if 'write_timestamp' in latest and latest['write_timestamp'] is not None:
                        from datetime import datetime
                        import pandas as pd
                        ts = latest['write_timestamp']
                        # Convert pandas Timestamp or datetime to ISO string
                        if isinstance(ts, pd.Timestamp):
                            latest['write_timestamp'] = ts.isoformat()
                        elif isinstance(ts, datetime):
                            latest['write_timestamp'] = ts.isoformat()
                        elif not isinstance(ts, str):
                            # Try to convert to string
                            latest['write_timestamp'] = str(ts)
                    
                    # Debug: Log what columns we got from the database
                    iv_related_cols = [k for k in latest.keys() if 'iv' in k.lower()]
                    logger.info(f"[FINANCIAL DB] {symbol}: IV-related columns from DB: {iv_related_cols}")
                    if 'iv_analysis_json' in latest:
                        json_val = latest.get('iv_analysis_json')
                        logger.info(f"[FINANCIAL DB] {symbol}: iv_analysis_json present: {json_val is not None}, type: {type(json_val)}, length: {len(str(json_val)) if json_val else 0}")
                    if 'write_timestamp' in latest:
                        logger.debug(f"[FINANCIAL DB] {symbol}: write_timestamp present: {latest.get('write_timestamp')}")
                    
                    # Check if database result has valid iv_analysis_json
                    # If not, query repository directly to bypass service layer cache
                    db_has_iv_json = (
                        'iv_analysis_json' in latest and 
                        latest.get('iv_analysis_json') is not None and
                        str(latest.get('iv_analysis_json')).strip() != ''
                    )
                    
                    if not db_has_iv_json:
                        logger.info(f"[FINANCIAL DB] Service layer cache for {symbol} missing iv_analysis_json, querying repository directly...")
                        try:
                            repo_check_start = time.time()
                            # Query repository directly to bypass service layer cache
                            if hasattr(db_instance, 'financial_service') and hasattr(db_instance.financial_service, 'financial_repo'):
                                logger.debug(f"[FINANCIAL DB] Querying repository directly for {symbol} to bypass service cache")
                                repo_df = await db_instance.financial_service.financial_repo.get(symbol)
                            elif hasattr(db_instance, 'financial_repo'):
                                logger.debug(f"[FINANCIAL DB] Querying financial_repo directly for {symbol}")
                                repo_df = await db_instance.financial_repo.get(symbol)
                            else:
                                repo_df = None
                                logger.warning(f"[FINANCIAL DB] Cannot access repository directly for {symbol}")
                            
                            repo_check_time = (time.time() - repo_check_start) * 1000
                            
                            if repo_df is not None and not repo_df.empty:
                                latest_repo = repo_df.iloc[-1].to_dict()
                                # If repository has iv_analysis_json, merge it into latest
                                if 'iv_analysis_json' in latest_repo and latest_repo.get('iv_analysis_json') and str(latest_repo.get('iv_analysis_json')).strip():
                                    logger.info(f"[FINANCIAL DB] Repository has iv_analysis_json for {symbol}, merging into result (repo_check: {repo_check_time:.1f}ms)")
                                    latest['iv_analysis_json'] = latest_repo['iv_analysis_json']
                                    db_has_iv_json = True
                                    # Also merge other IV-related fields
                                    for iv_field in ['iv_30d', 'iv_90d', 'iv_rank', 'iv_90d_rank', 'iv_rank_diff', 'relative_rank']:
                                        if iv_field in latest_repo and (iv_field not in latest or latest.get(iv_field) is None):
                                            latest[iv_field] = latest_repo[iv_field]
                                    
                                    # Parse the merged JSON
                                    try:
                                        iv_analysis_json_str = latest['iv_analysis_json']
                                        if not isinstance(iv_analysis_json_str, str):
                                            iv_analysis_json_str = str(iv_analysis_json_str)
                                        iv_analysis = json.loads(iv_analysis_json_str)
                                        if 'metrics' in iv_analysis:
                                            latest['iv_metrics'] = iv_analysis['metrics']
                                        if 'strategy' in iv_analysis:
                                            latest['iv_strategy'] = iv_analysis['strategy']
                                        if latest.get('relative_rank') is None and 'relative_rank' in iv_analysis:
                                            latest['relative_rank'] = iv_analysis['relative_rank']
                                        latest['iv_analysis'] = iv_analysis
                                        logger.info(f"[FINANCIAL DB] Successfully parsed and merged IV data from repository for {symbol}")
                                    except (json.JSONDecodeError, TypeError) as parse_e:
                                        logger.warning(f"[FINANCIAL DB] Could not parse merged IV JSON for {symbol}: {parse_e}")
                                else:
                                    logger.info(f"[FINANCIAL DB] Repository also missing or empty iv_analysis_json for {symbol}")
                        except Exception as repo_error:
                            logger.warning(f"[FINANCIAL DB] Error querying repository for {symbol}: {repo_error}")
                    
                    # Parse IV analysis JSON if present
                    # IMPORTANT: Keep the original iv_analysis_json string for frontend parsing
                    # Also add parsed versions for convenience
                    if 'iv_analysis_json' in latest and latest.get('iv_analysis_json'):
                        try:
                            iv_analysis_json_str = latest['iv_analysis_json']
                            # Ensure it's a string (might be bytes or other type from database)
                            if not isinstance(iv_analysis_json_str, str):
                                iv_analysis_json_str = str(iv_analysis_json_str)
                            
                            # Keep the original JSON string - DO NOT REMOVE IT
                            latest['iv_analysis_json'] = iv_analysis_json_str
                            
                            # Parse it for convenience
                            iv_analysis = json.loads(iv_analysis_json_str)
                            logger.info(f"[FINANCIAL DB] {symbol}: Successfully parsed iv_analysis_json, keys: {list(iv_analysis.keys())}")
                            
                            # Merge IV analysis data into financial_data for easy access
                            if 'metrics' in iv_analysis:
                                latest['iv_metrics'] = iv_analysis['metrics']
                                logger.info(f"[FINANCIAL DB] {symbol}: Added iv_metrics with keys: {list(iv_analysis['metrics'].keys())}")
                            if 'strategy' in iv_analysis:
                                latest['iv_strategy'] = iv_analysis['strategy']
                                logger.info(f"[FINANCIAL DB] {symbol}: Added iv_strategy with keys: {list(iv_analysis['strategy'].keys())}")
                            # Use relative_rank from JSON if database column is None
                            if latest.get('relative_rank') is None and 'relative_rank' in iv_analysis:
                                latest['relative_rank'] = iv_analysis['relative_rank']
                            # Also keep the full parsed JSON object for reference
                            latest['iv_analysis'] = iv_analysis
                            
                            logger.info(f"[FINANCIAL DB] {symbol}: Preserved iv_analysis_json string ({len(iv_analysis_json_str)} chars) and added parsed objects")
                        except (json.JSONDecodeError, TypeError) as e:
                            logger.warning(f"[FINANCIAL IV PARSE] Could not parse IV analysis JSON for {symbol}: {e}")
                            # Even if parsing fails, keep the original JSON string
                            if 'iv_analysis_json' not in latest:
                                logger.warning(f"[FINANCIAL IV PARSE] {symbol}: iv_analysis_json was removed during parsing, this should not happen")
                    else:
                        logger.info(f"[FINANCIAL DB] {symbol}: No iv_analysis_json found in database record")
                    
                    fetch_time = (time.time() - fetch_start) * 1000
                    result["financial_data"] = latest
                    result["source"] = "database"
                    result["fetch_time_ms"] = fetch_time
                    logger.info(f"[FINANCIAL DB HIT] Financial data for {symbol} (db_check: {db_check_time:.1f}ms, total: {fetch_time:.1f}ms)")
                    
                    # Cache the result (FinancialDataService already caches, but we can also cache here if needed)
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
                else:
                    logger.info(f"[FINANCIAL DB] No financial data found in database for {symbol}")
            except Exception as db_error:
                logger.warning(f"[FINANCIAL DB] Error fetching financial data from database for {symbol}: {db_error}")
            except Exception as e:
                logger.debug(f"[FINANCIAL DB ERROR] Error getting financial info from DB for {symbol}: {e}")
        
        # Fetch from API (unless cache_only mode)
        if cache_only:
            # In cache_only mode, if we reach here, it means no cached data was found
            result["error"] = "No cached financial data available (cache_only mode)"
            result["fetch_time_ms"] = (time.time() - fetch_start) * 1000
            logger.debug(f"[FINANCIAL CACHE ONLY] No cached data for {symbol}, returning error (cache_only=True)")
            return result
        
        # Check if this is an index symbol (I: prefix) - Polygon doesn't support indexes
        is_index = symbol.startswith("I:") or symbol.startswith("^")
        if is_index:
            logger.info(f"[FINANCIAL API] Skipping Polygon API call for index symbol {symbol} (Polygon doesn't support indexes)")
            result["error"] = f"Financial ratios not available for index symbols (Polygon API doesn't support indexes)"
            result["fetch_time_ms"] = (time.time() - fetch_start) * 1000
            return result
        
        api_key = os.getenv("POLYGON_API_KEY")
        if not api_key:
            result["error"] = "POLYGON_API_KEY environment variable not set"
            return result
        
        api_fetch_start = time.time()
        ratios = await get_financial_ratios(symbol, api_key)
        api_fetch_time = (time.time() - api_fetch_start) * 1000
        
        # Log what we got from Polygon API - use print to ensure it shows up
        if ratios:
            logger.debug(f"[FINANCIAL API] Fetched ratios from Polygon for {symbol}: {len(ratios)} keys")
        else:
            logger.warning(f"[FINANCIAL API] No ratios returned from Polygon API for {symbol}")
        
        # Calculate IV analysis if requested and syncing
        iv_analysis_result = None
        iv_fetch_time = None
        if force_fetch and include_iv_analysis:
            # Check if this is an index symbol - IV analysis doesn't work for indexes
            if is_index:
                logger.info(f"[FINANCIAL IV] Skipping IV analysis for index symbol {symbol} (IV analysis not available for indexes)")
            else:
                logger.info(f"[FINANCIAL IV] Starting IV analysis calculation for {symbol}")
                iv_fetch_start = time.time()
                try:
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
                    else:
                        logger.warning(f"[FINANCIAL IV] IV analysis calculation returned None for {symbol}")
                except Exception as iv_error:
                    iv_fetch_time = (time.time() - iv_fetch_start) * 1000
                    logger.error(f"[FINANCIAL IV] Error calculating IV analysis for {symbol}: {iv_error}")
                    import traceback
                    logger.debug(f"[FINANCIAL IV] IV analysis error traceback: {traceback.format_exc()}")
        elif include_iv_analysis and not force_fetch:
            logger.debug(f"[FINANCIAL IV] IV analysis requested for {symbol} but force_fetch=False, skipping calculation")
        
        fetch_time = (time.time() - fetch_start) * 1000
        
        # Prepare financial data dict - start with ratios if available, otherwise empty dict
        # If we have existing data in DB and ratios is None, try to preserve existing ratios
        financial_data = {}
        
        # If force_fetch and we have ratios, use them; otherwise try to preserve existing data
        if ratios:
            financial_data = ratios.copy()
            logger.info(f"[FINANCIAL] Using ratios from API for {symbol} ({len(ratios)} keys)")
            logger.info(f"[FINANCIAL] Ratio keys: {list(ratios.keys())[:20]}...")  # Show first 20 keys
            # Log some sample values to verify structure
            sample_keys = ['price_to_earnings', 'price_to_book', 'market_cap', 'current_ratio', 'current', 'quick_ratio', 'quick']
            sample_values = {k: ratios.get(k) for k in sample_keys if k in ratios}
            logger.info(f"[FINANCIAL] Sample ratio values from Polygon: {sample_values}")
            
            # Map Polygon API field names to database column names
            # Database save function expects 'current', 'quick', 'cash' but Polygon may return 'current_ratio', etc.
            field_mapping = {
                'current_ratio': 'current',
                'quick_ratio': 'quick',
                'cash_ratio': 'cash',
                # Also handle if they come as 'current', 'quick', 'cash' already
            }
            for polygon_key, db_key in field_mapping.items():
                if polygon_key in financial_data and db_key not in financial_data:
                    financial_data[db_key] = financial_data[polygon_key]
                    logger.debug(f"[FINANCIAL] Mapped {polygon_key} -> {db_key} for {symbol}")
            logger.debug(f"[FINANCIAL] After mapping, financial_data keys: {list(financial_data.keys())[:15]}...")
            
            # ALWAYS use today's date when saving to ensure we update the latest record
            # This prevents creating multiple records with different dates
            financial_data['date'] = date.today().isoformat()
            logger.debug(f"[FINANCIAL] Set date to today for {symbol}: {financial_data['date']}")
        elif not force_fetch:
            # If not forcing fetch and no ratios, try to get existing data from DB to preserve it
            try:
                existing_df = await db_instance.get_financial_info(symbol)
                if not existing_df.empty:
                    existing = existing_df.iloc[-1].to_dict()
                    # Copy existing ratio fields to preserve them
                    ratio_fields = ['price', 'market_cap', 'earnings_per_share', 'price_to_earnings', 
                                   'price_to_book', 'price_to_sales', 'price_to_cash_flow', 
                                   'price_to_free_cash_flow', 'dividend_yield', 'return_on_assets',
                                   'return_on_equity', 'debt_to_equity', 'current', 'quick', 'cash',
                                   'ev_to_sales', 'ev_to_ebitda', 'enterprise_value', 'free_cash_flow']
                    for field in ratio_fields:
                        if field in existing and pd.notna(existing[field]):
                            financial_data[field] = existing[field]
                    logger.debug(f"[FINANCIAL] Preserved existing ratios from DB for {symbol}")
            except Exception as preserve_error:
                logger.debug(f"[FINANCIAL] Could not preserve existing ratios: {preserve_error}")
        
        # Merge IV analysis into financial data if available (even if ratios is None)
        if iv_analysis_result:
            logger.debug(f"[FINANCIAL IV] Merging IV analysis into financial_data for {symbol}")
            
            # If we don't have ratios in financial_data but are forcing fetch, 
            # try to preserve existing ratios from DB before adding IV analysis
            if not ratios and force_fetch and not financial_data:
                try:
                    existing_df = await db_instance.get_financial_info(symbol)
                    if not existing_df.empty:
                        existing = existing_df.iloc[-1].to_dict()
                        # Preserve existing ratio fields that aren't None
                        # Map database column names (current_ratio) to expected field names (current)
                        ratio_field_mapping = {
                            'current_ratio': 'current',
                            'quick_ratio': 'quick',
                            'cash_ratio': 'cash'
                        }
                        ratio_fields = ['price', 'market_cap', 'earnings_per_share', 'price_to_earnings', 
                                       'price_to_book', 'price_to_sales', 'price_to_cash_flow', 
                                       'price_to_free_cash_flow', 'dividend_yield', 'return_on_assets',
                                       'return_on_equity', 'debt_to_equity', 'ev_to_sales', 
                                       'ev_to_ebitda', 'enterprise_value', 'free_cash_flow']
                        for field in ratio_fields:
                            if field in existing and pd.notna(existing[field]) and field not in financial_data:
                                financial_data[field] = existing[field]
                        # Handle mapped fields
                        for db_field, expected_field in ratio_field_mapping.items():
                            if db_field in existing and pd.notna(existing[db_field]) and expected_field not in financial_data:
                                financial_data[expected_field] = existing[db_field]
                        logger.debug(f"[FINANCIAL IV] Preserved existing ratios from DB for {symbol} before merging IV")
                except Exception as preserve_error:
                    logger.debug(f"[FINANCIAL IV] Could not preserve existing ratios when merging IV: {preserve_error}")
            
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
            
            # Parse IV_90d from percentage string if available
            iv_90d_str = metrics.get("iv_90d", "")
            iv_90d = None
            if iv_90d_str and iv_90d_str.endswith("%"):
                try:
                    iv_90d = float(iv_90d_str.rstrip("%")) / 100.0
                except (ValueError, AttributeError):
                    pass
            
            # Get rank values
            iv_rank = metrics.get("rank")  # 30-day IV rank
            iv_rank_90d = metrics.get("rank_90d")  # 90-day IV rank
            rank_diff = metrics.get("rank_diff")  # 30-day rank / 90-day rank (ratio)
            relative_rank = iv_analysis_result.get("relative_rank")
            
            # Add IV columns to financial_data (works whether ratios exists or not)
            financial_data['iv_30d'] = iv_30d
            financial_data['iv_rank'] = float(iv_rank) if iv_rank is not None else None
            if iv_90d is not None:
                financial_data['iv_90d'] = iv_90d
            if iv_rank_90d is not None:
                financial_data['iv_90d_rank'] = float(iv_rank_90d)
            if rank_diff is not None:
                financial_data['iv_rank_diff'] = float(rank_diff)
            financial_data['relative_rank'] = float(relative_rank) if relative_rank is not None else None
            financial_data['iv_analysis_json'] = json.dumps(iv_analysis_result)
            financial_data['iv_analysis_spare'] = None  # Spare column for future use
            
            # Ensure date field exists (required by database) - ALWAYS use today's date to ensure we update the latest record
            # This prevents creating multiple records with different dates
            financial_data['date'] = date.today().isoformat()
            logger.debug(f"[FINANCIAL IV] Set date field to today for {symbol}: {financial_data['date']}")
            
            logger.debug(f"[FINANCIAL IV] financial_data keys after IV merge: {list(financial_data.keys())}")
        
        # Only set error if we have neither ratios nor IV analysis
        if not ratios and not iv_analysis_result:
            result["error"] = "No financial ratios data available and IV analysis could not be calculated"
            logger.warning(f"[FINANCIAL] No data available for {symbol}: ratios={ratios is not None}, iv_analysis={iv_analysis_result is not None}")
        elif financial_data:
            # We have at least ratios or IV analysis (or both)
            result["financial_data"] = financial_data
            result["source"] = "api"
            result["fetch_time_ms"] = fetch_time
            
            if ratios:
                logger.info(f"[FINANCIAL API FETCH] Financial data for {symbol} (api_fetch: {api_fetch_time:.1f}ms, total: {fetch_time:.1f}ms)")
            elif iv_analysis_result:
                iv_time_str = f"{iv_fetch_time:.1f}ms" if iv_fetch_time is not None else "N/A"
                logger.info(f"[FINANCIAL IV] IV analysis for {symbol} (iv_fetch: {iv_time_str}, total: {fetch_time:.1f}ms)")
            
            # Calculate 52-week high/low from price history if not already in financial_data
            has_week52_low = 'week_52_low' in financial_data and financial_data.get('week_52_low') is not None
            has_week52_high = 'week_52_high' in financial_data and financial_data.get('week_52_high') is not None
            logger.debug(f"[FINANCIAL 52-WEEK] {symbol}: Checking 52-week calculation - has_low={has_week52_low}, has_high={has_week52_high}, db_instance={db_instance is not None}")
            
            if db_instance and (not has_week52_low or not has_week52_high):
                try:
                    from datetime import datetime, timedelta
                    import pandas as pd
                    
                    logger.debug(f"[FINANCIAL 52-WEEK] {symbol}: Starting 52-week range calculation from price history")
                    # Get merged price series for last 365 days
                    merged_df = await db_instance.get_merged_price_series(symbol)
                    logger.debug(f"[FINANCIAL 52-WEEK] {symbol}: Got merged_df - is_none={merged_df is None}, is_df={isinstance(merged_df, pd.DataFrame) if merged_df is not None else False}, empty={merged_df.empty if isinstance(merged_df, pd.DataFrame) else 'N/A'}")
                    
                    if merged_df is not None and isinstance(merged_df, pd.DataFrame) and not merged_df.empty:
                        # Get close column
                        close_col = None
                        if 'close' in merged_df.columns:
                            close_col = merged_df['close']
                        elif 'price' in merged_df.columns:
                            close_col = merged_df['price']
                        
                        logger.debug(f"[FINANCIAL 52-WEEK] {symbol}: close_col available={close_col is not None}, columns={list(merged_df.columns)}")
                        
                        if close_col is not None:
                            # Filter to last 365 days
                            if isinstance(merged_df.index, pd.DatetimeIndex):
                                one_year_ago = pd.Timestamp.now() - pd.Timedelta(days=365)
                                recent_df = merged_df[merged_df.index >= one_year_ago]
                                if not recent_df.empty:
                                    if 'close' in recent_df.columns:
                                        valid_prices = recent_df['close'].dropna()
                                    elif 'price' in recent_df.columns:
                                        valid_prices = recent_df['price'].dropna()
                                    else:
                                        valid_prices = close_col.dropna()
                                else:
                                    valid_prices = close_col.dropna()
                            else:
                                valid_prices = close_col.dropna()
                            
                            logger.debug(f"[FINANCIAL 52-WEEK] {symbol}: Found {len(valid_prices)} valid prices for 52-week calculation")
                            
                            if len(valid_prices) > 0:
                                if not has_week52_low:
                                    financial_data['week_52_low'] = float(valid_prices.min())
                                    logger.info(f"[FINANCIAL 52-WEEK] {symbol}: Calculated week_52_low={financial_data.get('week_52_low')}")
                                if not has_week52_high:
                                    financial_data['week_52_high'] = float(valid_prices.max())
                                    logger.info(f"[FINANCIAL 52-WEEK] {symbol}: Calculated week_52_high={financial_data.get('week_52_high')}")
                            else:
                                logger.warning(f"[FINANCIAL 52-WEEK] {symbol}: No valid prices found for 52-week calculation")
                        else:
                            logger.warning(f"[FINANCIAL 52-WEEK] {symbol}: No close/price column found in merged_df")
                    else:
                        logger.warning(f"[FINANCIAL 52-WEEK] {symbol}: merged_df is None, empty, or not a DataFrame - cannot calculate 52-week range")
                except NotImplementedError:
                    logger.debug(f"[FINANCIAL 52-WEEK] {symbol}: Backend does not support get_merged_price_series - skipping 52-week range calculation")
                except Exception as e:
                    logger.warning(f"[FINANCIAL 52-WEEK] {symbol}: Error calculating 52-week range: {e}", exc_info=True)
            else:
                if has_week52_low and has_week52_high:
                    logger.debug(f"[FINANCIAL 52-WEEK] {symbol}: 52-week range already exists in financial_data - skipping calculation")
                elif not db_instance:
                    logger.debug(f"[FINANCIAL 52-WEEK] {symbol}: No db_instance available - cannot calculate 52-week range")
            
            # Calculate 52-week range BEFORE saving if not already calculated
            # This ensures it's calculated even if we didn't fetch from API (e.g., updating existing record)
            if db_instance and financial_data:
                has_week52_low = 'week_52_low' in financial_data and financial_data.get('week_52_low') is not None
                has_week52_high = 'week_52_high' in financial_data and financial_data.get('week_52_high') is not None
                
                if not has_week52_low or not has_week52_high:
                    try:
                        from datetime import datetime, timedelta
                        import pandas as pd
                        
                        logger.debug(f"[FINANCIAL 52-WEEK] {symbol}: Calculating 52-week range before save - has_low={has_week52_low}, has_high={has_week52_high}")
                        # Get merged price series for last 365 days
                        merged_df = await db_instance.get_merged_price_series(symbol)
                        logger.debug(f"[FINANCIAL 52-WEEK] {symbol}: Got merged_df - is_none={merged_df is None}, is_df={isinstance(merged_df, pd.DataFrame) if merged_df is not None else False}, empty={merged_df.empty if isinstance(merged_df, pd.DataFrame) else 'N/A'}")
                        
                        if merged_df is not None and isinstance(merged_df, pd.DataFrame) and not merged_df.empty:
                            # Get close column
                            close_col = None
                            if 'close' in merged_df.columns:
                                close_col = merged_df['close']
                            elif 'price' in merged_df.columns:
                                close_col = merged_df['price']
                            
                            logger.debug(f"[FINANCIAL 52-WEEK] {symbol}: close_col available={close_col is not None}, columns={list(merged_df.columns)}")
                            
                            if close_col is not None:
                                # Filter to last 365 days
                                if isinstance(merged_df.index, pd.DatetimeIndex):
                                    one_year_ago = pd.Timestamp.now() - pd.Timedelta(days=365)
                                    recent_df = merged_df[merged_df.index >= one_year_ago]
                                    if not recent_df.empty:
                                        if 'close' in recent_df.columns:
                                            valid_prices = recent_df['close'].dropna()
                                        elif 'price' in recent_df.columns:
                                            valid_prices = recent_df['price'].dropna()
                                        else:
                                            valid_prices = close_col.dropna()
                                    else:
                                        valid_prices = close_col.dropna()
                                else:
                                    valid_prices = close_col.dropna()
                                
                                logger.debug(f"[FINANCIAL 52-WEEK] {symbol}: Found {len(valid_prices)} valid prices for 52-week calculation")
                                
                                if len(valid_prices) > 0:
                                    if not has_week52_low:
                                        financial_data['week_52_low'] = float(valid_prices.min())
                                        logger.info(f"[FINANCIAL 52-WEEK] {symbol}: Calculated week_52_low={financial_data.get('week_52_low')}")
                                    if not has_week52_high:
                                        financial_data['week_52_high'] = float(valid_prices.max())
                                        logger.info(f"[FINANCIAL 52-WEEK] {symbol}: Calculated week_52_high={financial_data.get('week_52_high')}")
                                else:
                                    logger.warning(f"[FINANCIAL 52-WEEK] {symbol}: No valid prices found for 52-week calculation")
                            else:
                                logger.warning(f"[FINANCIAL 52-WEEK] {symbol}: No close/price column found in merged_df")
                        else:
                            logger.warning(f"[FINANCIAL 52-WEEK] {symbol}: merged_df is None, empty, or not a DataFrame - cannot calculate 52-week range")
                    except NotImplementedError:
                        logger.debug(f"[FINANCIAL 52-WEEK] {symbol}: Backend does not support get_merged_price_series - skipping 52-week range before save")
                    except Exception as e:
                        logger.warning(f"[FINANCIAL 52-WEEK] {symbol}: Error calculating 52-week range before save: {e}", exc_info=True)
            
            # Save to DB if we have a DB instance and financial_data is not empty
            if financial_data:
                # Ensure date is always set to today before saving
                financial_data['date'] = date.today().isoformat()
                logger.debug(f"[FINANCIAL DB SAVE] Saving financial_data for {symbol} with date: {financial_data['date']}, week_52_low={financial_data.get('week_52_low')}, week_52_high={financial_data.get('week_52_high')}")
                try:
                    await db_instance.save_financial_info(symbol, financial_data)
                    logger.debug(f"[FINANCIAL DB SAVE] Successfully saved financial_info for {symbol}")
                    # Clear cache after saving to ensure fresh data on next read
                    if cache_instance:
                        try:
                            cache_key = CacheKeyGenerator.financial_info(symbol)
                            await cache_instance.delete(cache_key)
                            logger.debug(f"[FINANCIAL DB SAVE] Cleared cache for {symbol} after saving IV analysis")
                        except Exception as cache_del_error:
                            logger.debug(f"[FINANCIAL DB SAVE] Could not clear cache: {cache_del_error}")
                    if not ratios and iv_analysis_result:
                        logger.info(f"[FINANCIAL DB SAVE] Saved IV analysis (without ratios) for {symbol} to database")
                    elif ratios and iv_analysis_result:
                        logger.info(f"[FINANCIAL DB SAVE] Saved financial data with IV analysis for {symbol} to database")
                    elif ratios:
                        logger.info(f"[FINANCIAL DB SAVE] Saved financial ratios for {symbol} to database")
                except Exception as e:
                    logger.error(f"[FINANCIAL DB SAVE ERROR] Error saving financial info to DB for {symbol}: {e}")
                    import traceback
                    logger.debug(f"[FINANCIAL DB SAVE ERROR] Traceback: {traceback.format_exc()}")
            else:
                logger.warning(f"[FINANCIAL DB SAVE] financial_data is empty for {symbol}, skipping save")
            
            # Cache the result
            if cache_instance:
                try:
                    cache_key = CacheKeyGenerator.financial_info(symbol)
                    # Add last_save_time to cached data
                    financial_data_with_time = financial_data.copy()
                    financial_data_with_time['last_save_time'] = datetime.now(timezone.utc).isoformat()
                    cache_df = pd.DataFrame([financial_data_with_time])
                    cache_set_start = time.time()
                    await cache_instance.set(cache_key, cache_df, ttl=None)  # No TTL (infinite cache)
                    cache_set_time = (time.time() - cache_set_start) * 1000
                    logger.info(f"[FINANCIAL CACHE SET] Cached financial data for {symbol} (set_time: {cache_set_time:.1f}ms, no TTL)")
                except Exception as e:
                    logger.debug(f"[FINANCIAL CACHE ERROR] Failed to cache financial data for {symbol}: {e}")
    
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
