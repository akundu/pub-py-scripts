#!/usr/bin/env python3
"""
Options Analyzer - Covered Call Premium Analysis Tool

This program analyzes covered call opportunities across all strike prices and tickers
in a QuestDB options database. It calculates potential premiums for $1M stock positions
and provides flexible filtering, sorting, and output options.

Usage:
    export POLYGON_API_KEY=YOUR_API_KEY  # Optional, for symbol lists
    python options_analyzer.py --db-conn questdb://user:pass@host:8812/db
    python options_analyzer.py --symbols AAPL,MSFT,GOOGL --days 14 --output csv
    python options_analyzer.py --types sp-500 --sort daily_premium --group-by ticker
    python options_analyzer.py --min-volume 1000 --max-days 30 --output results.csv
"""

import os
import sys
import argparse
import asyncio
import pandas as pd
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Any, Optional, Tuple, Union
from tabulate import tabulate
from pathlib import Path
import json
import re
import math
from concurrent.futures import ProcessPoolExecutor
import functools
import subprocess
import multiprocessing

# Ensure project root is on sys.path so `common` can be imported when running from any cwd
CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Import common functions
from common.common import (
    black_scholes_call,
    check_tickers_for_refresh as common_check_tickers_for_refresh,
    fetch_latest_option_timestamp_standalone as common_fetch_latest_option_timestamp_standalone,
    get_redis_client_for_refresh,
    check_redis_refresh_pending,
    set_redis_refresh_pending,
    clear_redis_refresh_pending,
    set_redis_last_write_timestamp,
    REDIS_AVAILABLE,
    extract_ticker_from_option_ticker,
    calculate_option_premium,
    format_bid_ask,
    format_price_with_change,
    calculate_days_to_expiry,
    format_age_seconds,
    normalize_timestamp_to_utc,
    normalize_expiration_date_to_utc,
    normalize_timestamp_for_display
)

# Import common symbol loading functions
from common.symbol_loader import add_symbol_arguments, fetch_lists_data
from common.stock_db import get_stock_db
from common.market_hours import is_market_hours as common_is_market_hours

# Import options utilities
from common.options_utils import (
    ensure_ticker_column,
    calculate_option_metrics,
    apply_basic_filters,
    get_previous_close_for_date,
    attach_price_data
)

# Import filter system
from common.options.options_filters import FilterExpression, FilterParser

# Import worker functions
from common.options.options_workers import (
    process_ticker_analysis,
    process_spread_match,
    setup_worker_imports,
    import_filter_classes
)

# Import spread analysis
from common.options.options_spread import (
    calculate_long_options_date_range,
    calculate_combined_date_range,
    fetch_long_term_options,
    filter_and_prepare_long_options,
    prepare_spread_matching_data,
    execute_spread_matching
)

# Import formatting
from common.options.options_formatting import (
    format_dataframe_for_display,
    normalize_and_select_columns,
    create_compact_headers,
    format_csv_output
)

# Import refresh functionality
from common.options.options_refresh import (
    process_refresh_batch,
    calculate_refresh_date_ranges
)

# Import new helper modules
from common.options.options_args import (
    ARGUMENT_EXAMPLES,
    add_database_arguments,
    add_analysis_arguments,
    add_spread_arguments,
    add_performance_arguments,
    add_filter_arguments,
    add_output_arguments,
    log_parsed_arguments
)
from common.options.options_output import (
    enrich_dataframe_with_financial_data,
    add_derived_percentage_columns,
    format_timestamp_columns,
    get_display_columns,
    apply_sorting,
    apply_top_n_filter,
    resolve_csv_columns,
    format_table_output
)
from common.options.options_pipeline import (
    calculate_date_ranges,
    split_combined_options_by_date_range,
    log_spread_analysis_start
)

# Import fetch_options for refresh functionality
try:
    from scripts.fetch_options import HistoricalDataFetcher
    POLYGON_AVAILABLE = True
except ImportError:
    # If fetch_options is not available, refresh feature will be disabled
    HistoricalDataFetcher = None
    POLYGON_AVAILABLE = False


# ============================================================================
# OptionsAnalyzer Class
# ============================================================================

# Module-level cache for timestamp lookups (used by standalone functions)
_timestamp_cache_per_process: Dict[str, pd.Timestamp] = {}

# Wrapper function for timestamp fetching (used by OptionsAnalyzer class methods)
async def _fetch_latest_option_timestamp_standalone(
    db,
    ticker: str,
    cache: Optional[Dict[str, pd.Timestamp]] = None,
    redis_client: Optional[Any] = None,
    debug: bool = False
) -> Optional[float]:
    """
    Standalone function to fetch latest option write timestamp for a single ticker.
    Returns the age in seconds (difference between now and the timestamp).
    Can be used in multiprocessing workers or regular code paths.
    """
    if cache is None:
        cache = _timestamp_cache_per_process
    return await common_fetch_latest_option_timestamp_standalone(db, ticker, cache, redis_client=redis_client, debug=debug)


# Import from options_workers module for backward compatibility
from common.options.options_workers import (
    process_ticker_analysis as _process_ticker_analysis,
    process_ticker_spread_analysis as _process_ticker_spread_analysis,
    process_spread_match as _process_spread_match
)

# Create aliases with underscores for backward compatibility
_extract_ticker_from_option_ticker = extract_ticker_from_option_ticker
_get_previous_close_for_date = get_previous_close_for_date
_format_price_with_change = format_price_with_change
_calculate_option_metrics = calculate_option_metrics
_apply_basic_filters = apply_basic_filters
_normalize_and_select_columns = normalize_and_select_columns
_normalize_to_utc = normalize_timestamp_to_utc
_safe_days_calc = calculate_days_to_expiry
_format_age_seconds = format_age_seconds
_normalize_expiration_date_for_display = normalize_expiration_date_to_utc
_normalize_timestamp_for_display = normalize_timestamp_for_display
_format_dataframe_for_display = format_dataframe_for_display


class OptionsAnalyzer:
    """Analyzes covered call opportunities across all strike prices and tickers."""
    
    def __init__(self, db_conn: str, log_level: str = "INFO", debug: bool = False, enable_cache: bool = True, redis_url: str | None = None, risk_free_rate: float = 0.05):
        """Initialize the options analyzer with database connection."""
        self.db_conn = db_conn
        self.log_level = log_level.upper()  # Normalize to uppercase
        self.debug = debug or (self.log_level == "DEBUG")
        self.enable_cache = enable_cache
        self.redis_url = redis_url
        self.db = None
        self.risk_free_rate = risk_free_rate  # Annual risk-free rate (default 5%)
        self._timestamp_cache: Dict[str, pd.Timestamp] = {}  # Cache for latest option timestamps per ticker
        self._redis_client = None  # Redis client for timestamp caching
    
    def _should_log(self, level: str) -> bool:
        """Check if a log level should be printed based on current log_level setting."""
        levels = {"DEBUG": 0, "INFO": 1, "WARNING": 2, "ERROR": 3}
        current_level = levels.get(self.log_level, 1)
        message_level = levels.get(level.upper(), 1)
        return message_level >= current_level
    
    def _log(self, level: str, message: str, file=sys.stderr):
        """Log a message if the level is appropriate."""
        if self._should_log(level):
            print(message, file=file)
    
    def _infer_ticker_label(
        self,
        df: Optional[pd.DataFrame] = None,
        tickers: Optional[List[str]] = None
    ) -> Optional[str]:
        """Generate a compact ticker label for debug logs."""
        values: List[str] = []
        if df is not None and 'ticker' in df.columns:
            try:
                values = df['ticker'].dropna().astype(str).tolist()
            except Exception:
                values = [str(v) for v in df['ticker'].dropna().tolist()]
        elif tickers:
            values = [str(t) for t in tickers if t]
        if not values:
            return None
        seen: List[str] = []
        for val in values:
            if val and val not in seen:
                seen.append(val)
        if not seen:
            return None
        if len(seen) <= 3:
            return ",".join(seen)
        return ",".join(seen[:3]) + f"+{len(seen)-3}"
    
    async def _fetch_latest_option_timestamps(
        self,
        tickers: List[str],
        cache: Optional[Dict[str, pd.Timestamp]] = None
    ) -> Dict[str, Optional[float]]:
        """
        Fetch latest option write timestamps for multiple tickers.
        Returns ages in seconds (difference between now and the timestamp).
        
        Args:
            tickers: List of ticker symbols to fetch timestamps for
            cache: Optional dictionary to use/update as cache (avoids duplicate fetches)
            
        Returns:
            Dictionary mapping ticker -> age in seconds (float) or None if no timestamp found
        """
        # Use instance cache if no cache provided
        if cache is None:
            cache = self._timestamp_cache
        
        # Fetch ages for all tickers using the standalone function (it handles caching internally)
        result: Dict[str, Optional[float]] = {}
        for ticker in tickers:
            age_seconds = await _fetch_latest_option_timestamp_standalone(
                self.db, ticker, cache=cache, redis_client=self._redis_client, debug=self.debug
            )
            result[ticker] = age_seconds
        
        return result
    
    @staticmethod
    def _extract_ticker_from_option_ticker(option_ticker):
        """Extract ticker symbol from option_ticker (e.g., 'AAPL250117C00150000' -> 'AAPL')."""
        return _extract_ticker_from_option_ticker(option_ticker)
    
    def _black_scholes_call(self, S: float, K: float, T: float, r: float, sigma: float) -> float:
        """Calculate Black-Scholes call option price (delegates to common function)."""
        return black_scholes_call(S, K, T, r, sigma)
    
    def _create_compact_headers(self, df: pd.DataFrame) -> Dict[str, str]:
        """Create compact headers that are at most 4 characters longer than the data width."""
        compact_headers = {}

        'current_price', 'strike_price', 'volume', 'pe_ratio', 'market_cap_b'
        
        # Define mapping for common columns to shorter names
        header_mapping = {
            # 'ticker': 'ticker',
            'current_price': 'curr_price',
            # 'pe_ratio': 'pe_ratio',
            # 'market_cap': 'market_cap',
            # 'market_cap_b': 'market_cap_b',
            # 'strike_price': 'strike_price',
            'price_above_current': 'price_above_curr',
            'option_premium': 'opt_prem.',
            'bid_ask': 'bid:ask',
            'option_premium_percentage': 'opt_prem.%',
            # 'premium_above_diff_percentage': 'DIFF%',
            'implied_volatility': 'iv',
            # 'delta': 'delta',
            # 'theta': 'theta',
            # 'volume': 'volume',
            # 'num_contracts': 'CNT',
            # 'potential_premium': 'POT_PREM',
            # 'daily_premium': 'DAILY_PREM',
            # 'expiration_date': 'EXP (UTC)',
            # 'days_to_expiry': 'DAYS',
            # 'last_quote_timestamp': 'LQUOTE_TS',
            # 'write_timestamp': 'WRITE_TS (EST)',
            # 'option_ticker': 'OPT_TKR',
            # # Spread-related columns
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
            # 'net_premium': 'net_premium',
            # 'net_daily_premium': 'net_daily_premium'
        }
        
        for col in df.columns:
            if col in header_mapping:
                compact_headers[col] = header_mapping[col]
            else:
                # For unknown columns, use the original name but truncate if too long
                compact_headers[col] = col[:15] if len(col) > 8 else col
        
        return compact_headers
    
    def _format_csv_output(
        self, 
        df: pd.DataFrame, 
        delimiter: str = ',', 
        quoting: str = 'minimal', 
        group_by: str = 'overall',
        output_file: Optional[str] = None
    ) -> str:
        """Format DataFrame as CSV with proper formatting."""
        import csv
        
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
            self._log("INFO", f"CSV results saved to {output_file}")
        
        return csv_content
        
    async def initialize(self):
        """Initialize database connection."""
        try:
            # Use INFO level for database to reduce cache message verbosity, even when analyzer is in DEBUG mode
            # This allows us to see ticker exclusion debug messages without being flooded by cache hits/misses
            db_log_level = "INFO" if self.log_level == "DEBUG" else self.log_level
            self.db = get_stock_db('questdb', db_config=self.db_conn, enable_cache=self.enable_cache, redis_url=self.redis_url, log_level=db_log_level)
            cache_status = "enabled" if self.enable_cache else "disabled"
            self._log("INFO", f"Database connection established successfully (cache: {cache_status}).")
            if self.log_level == "DEBUG" and db_log_level == "INFO":
                self._log("DEBUG", "Database log level set to INFO to reduce cache message verbosity (analyzer remains in DEBUG mode)")
            
            # Initialize Redis client for timestamp caching
            if self.enable_cache and self.redis_url and REDIS_AVAILABLE:
                self._redis_client = get_redis_client_for_refresh(self.redis_url)
        except Exception as e:
            print(f"Error connecting to database: {e}", file=sys.stderr)
            sys.exit(1)
    
    async def get_financial_info(self, tickers: List[str], max_workers: int = 4) -> Dict[str, Dict[str, Any]]:
        """Get financial information (P/E, market_cap) for the given tickers using multiprocessing."""
        financial_data = {}
        
        if self.debug:
            print(f"DEBUG: Fetching financial info for {len(tickers)} tickers using {max_workers} processes", file=sys.stderr)
        
        # Track cache statistics
        initial_cache_stats = None
        if self.enable_cache and hasattr(self.db, 'get_cache_statistics'):
            try:
                initial_cache_stats = self.db.get_cache_statistics()
            except:
                pass
        
        total = len(tickers)
        
        # Prepare arguments for each ticker
        process_args = []
        for ticker in tickers:
            args = (
                ticker,
                self.db_conn,
                self.enable_cache,
                self.redis_url,
                self.log_level,
                self.debug
            )
            process_args.append(args)
        
        # Execute in parallel using ProcessPoolExecutor
        loop = asyncio.get_event_loop()
        
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = [
                loop.run_in_executor(executor, _process_ticker_financial_info, args)
                for args in process_args
            ]
            
            # Wait for all results
            results = await asyncio.gather(*futures, return_exceptions=True)
        
        # Process results and build financial_data dictionary
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                ticker = tickers[i]
                self._log("WARNING", f"Error processing financial info for {ticker}: {result}")
                financial_data[ticker] = {
                    'pe_ratio': None,
                    'market_cap': None,
                    'price': None
                }
            else:
                ticker, data = result
                financial_data[ticker] = data
        
        # Log final cache stats if debug is enabled
        if self.debug:
            cache_info = ""
            if initial_cache_stats is not None and self.enable_cache and hasattr(self.db, 'get_cache_statistics'):
                try:
                    current_cache_stats = self.db.get_cache_statistics()
                    hits_diff = current_cache_stats.get('hits', 0) - initial_cache_stats.get('hits', 0)
                    misses_diff = current_cache_stats.get('misses', 0) - initial_cache_stats.get('misses', 0)
                    total_requests = hits_diff + misses_diff
                    if total_requests > 0:
                        hit_rate = (hits_diff / total_requests * 100) if total_requests > 0 else 0.0
                        cache_info = f" [cache: {hits_diff} hits, {misses_diff} misses, {hit_rate:.1f}% hit rate]"
                except:
                    pass
            print(f"DEBUG: Fetched financial info for {total}/{total} tickers (100%){cache_info}", file=sys.stderr)
        
        return financial_data
    
    async def _analyze_options_multiprocess(
        self,
        tickers_upper: List[str],
        start_date: str,
        end_date: Optional[str],
        timestamp_lookback_days: int,
        position_size: float,
        days_to_expiry: Optional[int],
        min_volume: int,
        min_premium: float,
        min_write_timestamp: Optional[str],
        use_market_time: bool,
        filters: Optional[List[FilterExpression]],
        filter_logic: str,
        option_type: str,
        max_workers: int,
        sensible_price: float = 1.0,
        max_bid_ask_spread: float = 2.0
    ) -> pd.DataFrame:
        """
        Analyze options using multiprocessing where each ticker is processed end-to-end in a separate process.
        All processing (fetching, filtering, metrics, filters) happens in worker processes.
        Main process only aggregates, sorts, and presents results.
        """
        from concurrent.futures import ProcessPoolExecutor
        
        self._log("INFO", f"Using multiprocessing mode: processing {len(tickers_upper)} tickers with {max_workers} workers")
        self._log("INFO", "Each ticker will be processed end-to-end in a separate process (fetch, filter, metrics, filters)")
        
        # Prepare arguments for each ticker
        process_args = []
        for ticker in tickers_upper:
            args = (
                ticker,
                self.db_conn,
                start_date,
                end_date,
                timestamp_lookback_days,
                position_size,
                days_to_expiry,
                min_volume,
                min_premium,
                min_write_timestamp,
                use_market_time,
                filters,  # FilterExpression objects should be picklable
                filter_logic,
                option_type,
                self.enable_cache,
                self.redis_url,
                self.log_level,
                self.debug,
                sensible_price,
                max_bid_ask_spread
            )
            process_args.append(args)
        
        # Execute in parallel using ProcessPoolExecutor
        loop = asyncio.get_event_loop()
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = [
                loop.run_in_executor(executor, _process_ticker_analysis, args)
                for args in process_args
            ]
            results = await asyncio.gather(*futures, return_exceptions=True)
        
        # Collect results and handle errors
        dfs = []
        errors = []
        for i, result in enumerate(results):
            ticker = tickers_upper[i]
            if isinstance(result, Exception):
                error_msg = f"Error processing {ticker}: {result}"
                errors.append(error_msg)
                self._log("ERROR", error_msg)
                if self.debug:
                    import traceback
                    traceback.print_exc()
            elif isinstance(result, tuple) and len(result) == 2:
                df, error = result
                if error:
                    errors.append(error)
                    self._log("WARNING", error)
                if not df.empty:
                    dfs.append(df)
            else:
                self._log("WARNING", f"Unexpected result type for {ticker}: {type(result)}")
        
        if errors and self.debug:
            self._log("INFO", f"Encountered {len(errors)} error(s) during processing")
        
        # Concatenate all results
        if not dfs:
            self._log("INFO", "No results from any ticker")
            return pd.DataFrame()
        
        # Filter out empty DataFrames before concatenation to avoid FutureWarning
        non_empty_dfs = [df for df in dfs if not df.empty]
        if not non_empty_dfs:
            self._log("INFO", "No results from any ticker (all DataFrames were empty)")
            return pd.DataFrame()
        
        combined_df = pd.concat(non_empty_dfs, ignore_index=True)
        self._log("INFO", f"Combined results from {len(dfs)} tickers: {len(combined_df)} total options")
        
        return combined_df
    
    async def _analyze_spread_multiprocess(
        self,
        tickers_upper: List[str],
        start_date: str,
        end_date: Optional[str],
        long_start_date: str,
        long_end_date: str,
        timestamp_lookback_days: int,
        position_size: float,
        days_to_expiry: Optional[int],
        min_volume: int,
        min_premium: float,
        min_write_timestamp: Optional[str],
        use_market_time: bool,
        filters: Optional[List[FilterExpression]],
        filter_logic: str,
        option_type: str,
        spread_strike_tolerance: float,
        spread_long_days: int,
        spread_long_days_tolerance: int,
        spread_long_min_days: Optional[int],
        max_workers: int,
        sensible_price: float = 0.01,
        max_bid_ask_spread: float = 2.0,
        max_bid_ask_spread_long: float = 2.0
    ) -> pd.DataFrame:
        """
        Analyze spread options using multiprocessing where each ticker is processed end-to-end in a separate process.
        All processing (fetching short-term, processing short-term, fetching long-term, matching, spread calculations)
        happens in worker processes. Main process only aggregates results.
        """
        from concurrent.futures import ProcessPoolExecutor
        
        self._log("INFO", f"Using multiprocessing mode for spread analysis: processing {len(tickers_upper)} tickers with {max_workers} workers")
        self._log("INFO", "Each ticker will be processed end-to-end in a separate process (short-term fetch/process, long-term fetch, matching, spread calculations)")
        
        # Prepare arguments for each ticker
        process_args = []
        for ticker in tickers_upper:
            args = (
                ticker,
                self.db_conn,
                start_date,
                end_date,
                long_start_date,
                long_end_date,
                timestamp_lookback_days,
                position_size,
                days_to_expiry,
                min_volume,
                min_premium,
                min_write_timestamp,
                use_market_time,
                filters,
                filter_logic,
                option_type,
                spread_strike_tolerance,
                spread_long_days,
                spread_long_days_tolerance,
                spread_long_min_days,
                self.risk_free_rate,
                self.enable_cache,
                self.redis_url,
                self.log_level,
                self.debug,
                sensible_price,
                max_bid_ask_spread,
                max_bid_ask_spread_long
            )
            process_args.append(args)
        
        # Execute in parallel using ProcessPoolExecutor
        loop = asyncio.get_event_loop()
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = [
                loop.run_in_executor(executor, _process_ticker_spread_analysis, args)
                for args in process_args
            ]
            results = await asyncio.gather(*futures, return_exceptions=True)
        
        # Collect results and handle errors
        dfs = []
        errors = []
        for i, result in enumerate(results):
            ticker = tickers_upper[i]
            if isinstance(result, Exception):
                error_msg = f"Error processing spread analysis for {ticker}: {result}"
                errors.append(error_msg)
                self._log("ERROR", error_msg)
                if self.debug:
                    import traceback
                    traceback.print_exc()
            elif isinstance(result, tuple) and len(result) == 2:
                df, error = result
                if error:
                    errors.append(error)
                    self._log("WARNING", error)
                if not df.empty:
                    dfs.append(df)
            else:
                self._log("WARNING", f"Unexpected result type for {ticker}: {type(result)}")
        
        if errors and self.debug:
            self._log("INFO", f"Encountered {len(errors)} error(s) during spread processing")
        
        # Concatenate all results
        if not dfs:
            self._log("INFO", "No spread results from any ticker")
            return pd.DataFrame()
        
        # Filter out empty DataFrames before concatenation to avoid FutureWarning
        non_empty_dfs = [df for df in dfs if not df.empty]
        if not non_empty_dfs:
            self._log("INFO", "No spread results from any ticker (all DataFrames were empty)")
            return pd.DataFrame()
        
        combined_df = pd.concat(non_empty_dfs, ignore_index=True)
        self._log("INFO", f"Combined spread results from {len(dfs)} tickers: {len(combined_df)} total spread opportunities")
        
        return combined_df
    
    async def analyze_options(
        self,
        tickers: List[str],
        option_type: str = 'call',
        days_to_expiry: Optional[int] = None,
        min_volume: int = 0,
        max_days: Optional[int] = None,
        min_days: Optional[int] = None,
        min_premium: float = 0.0,
        position_size: float = 100000.0,
        filters: Optional[List[FilterExpression]] = None,
        filter_logic: str = 'AND',
        use_market_time: bool = True,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        max_concurrent: int = 10,
        batch_size: int = 50,
        timestamp_lookback_days: int = 7,
        max_workers: int = 4,
        spread_mode: bool = False,
        spread_strike_tolerance: float = 0.0,
        spread_long_days: int = 90,
        spread_long_days_tolerance: int = 14,
        spread_long_min_days: Optional[int] = None,
        min_write_timestamp: Optional[str] = None,
        sensible_price: float = 1.0,
        max_bid_ask_spread: float = 2.0,
        max_bid_ask_spread_long: float = 2.0
    ) -> pd.DataFrame:
        """
        Analyze options opportunities for the given tickers.
        
        Args:
            tickers: List of ticker symbols to analyze
            option_type: Type of options to analyze ('call', 'put', or 'both'). Default: 'call'
            days_to_expiry: Number of days to expiry (if None, analyze all available)
            min_volume: Minimum volume filter
            max_days: Maximum days from today for expiration (convenience param that sets end_date, overrides end_date if both provided)
            min_days: Minimum days from today for expiration (convenience param that sets start_date, overrides start_date if both provided)
            min_premium: Minimum potential premium filter
            position_size: Position size in dollars for calculations
            filters: List of FilterExpression objects to apply
            filter_logic: Logic to combine filters ('AND' or 'OR')
            use_market_time: Whether to use market hours logic for price fetching
            start_date: Start date for option expiration filtering (YYYY-MM-DD format, defaults to today)
            end_date: End date for option expiration filtering (YYYY-MM-DD format, defaults to None, overridden by max_days)
            max_concurrent: Maximum concurrent queries per batch (lower = less memory)
            batch_size: Number of tickers per batch (lower = less memory)
            timestamp_lookback_days: Days to look back for option timestamp data (default: 7, lower = less memory but may miss data)
            max_workers: Number of worker processes for multiprocessing (default: 4, typically CPU count)
            spread_mode: Enable calendar spread analysis (sell short-term, buy long-term)
            spread_strike_tolerance: Percentage tolerance for matching strike prices (e.g., 5.0 for ±5%)
            spread_long_days: Maximum/target days to expiry for long-term options to buy
            spread_long_days_tolerance: Days tolerance for long option expiration window (default: 14, searches ±14 days around target, ignored if spread_long_min_days is set)
            spread_long_min_days: Minimum days to expiry for long options (if set, searches from min to spread_long_days instead of using tolerance)
            min_write_timestamp: Minimum write timestamp for options (EST format like '2025-11-05 10:00:00', filters out older options)
            max_bid_ask_spread: Maximum bid-ask spread ratio for short options (default: 2.0). Formula: (ask - bid) / bid <= max_spread. Set to 0 to disable.
            max_bid_ask_spread_long: Maximum bid-ask spread ratio for long options in spread mode (default: 2.0). Formula: (ask - bid) / bid <= max_spread. Set to 0 to disable.
            
        Returns:
            DataFrame with analysis results
        """
        if not tickers:
            return pd.DataFrame()
        
        # Convert tickers to uppercase for database compatibility
        tickers_upper = [ticker.upper() for ticker in tickers]
        
        # Calculate date ranges
        start_date, end_date = calculate_date_ranges(start_date, max_days, end_date, min_days)
        if self.debug and max_days is not None:
            print(f"DEBUG: Using max_days={max_days}: filtering options expiring through {end_date}", file=sys.stderr)
        if self.debug and min_days is not None:
            print(f"DEBUG: Using min_days={min_days}: filtering options expiring from {start_date}", file=sys.stderr)
        
        # Use memory-efficient batch fetching instead of single large query
        try:
            # If using multiprocessing, process each ticker in a separate process
            if max_workers > 1:
                return await self._analyze_with_multiprocessing(
                    tickers_upper=tickers_upper,
                    start_date=start_date,
                    end_date=end_date,
                    spread_mode=spread_mode,
                    spread_long_days=spread_long_days,
                    spread_long_days_tolerance=spread_long_days_tolerance,
                    spread_long_min_days=spread_long_min_days,
                    spread_strike_tolerance=spread_strike_tolerance,
                    timestamp_lookback_days=timestamp_lookback_days,
                    position_size=position_size,
                    days_to_expiry=days_to_expiry,
                    min_volume=min_volume,
                    min_premium=min_premium,
                    min_write_timestamp=min_write_timestamp,
                    use_market_time=use_market_time,
                    filters=filters,
                    filter_logic=filter_logic,
                    option_type=option_type,
                    max_workers=max_workers,
                    sensible_price=sensible_price,
                    max_bid_ask_spread=max_bid_ask_spread,
                    max_bid_ask_spread_long=max_bid_ask_spread_long
                )
            
            # 1) Fetch options universe (combined short and long if spread mode, otherwise just short)
            options_df, long_options_df = await self._fetch_options_universe(
                spread_mode=spread_mode,
                tickers_upper=tickers_upper,
                start_date=start_date,
                end_date=end_date,
                spread_long_days=spread_long_days,
                spread_long_days_tolerance=spread_long_days_tolerance,
                spread_long_min_days=spread_long_min_days,
                timestamp_lookback_days=timestamp_lookback_days,
                max_workers=max_workers,
                max_concurrent=max_concurrent,
                batch_size=batch_size,
                min_write_timestamp=min_write_timestamp,
                option_type=option_type
            )
            if options_df.empty:
                return pd.DataFrame()

            # Process options through the pipeline
            df = await self._process_options_pipeline(
                options_df=options_df,
                tickers_upper=tickers_upper,
                option_type=option_type,
                use_market_time=use_market_time,
                position_size=position_size,
                days_to_expiry=days_to_expiry,
                min_volume=min_volume,
                min_premium=min_premium,
                min_write_timestamp=min_write_timestamp,
                sensible_price=sensible_price,
                spread_mode=spread_mode,
                spread_strike_tolerance=spread_strike_tolerance,
                spread_long_days=spread_long_days,
                spread_long_days_tolerance=spread_long_days_tolerance,
                spread_long_min_days=spread_long_min_days,
                start_date=start_date,
                end_date=end_date,
                max_concurrent=max_concurrent,
                batch_size=batch_size,
                timestamp_lookback_days=timestamp_lookback_days,
                max_workers=max_workers,
                long_options_df=long_options_df
            )
            
            if self.debug:
                print(f"DEBUG: Final options count before spread/filtering: {len(df)}", file=sys.stderr)
            
            return df
        except Exception as e:
            self._log("ERROR", f"Error analyzing options: {e}")
            import traceback
            if self.debug:
                traceback.print_exc()
            return pd.DataFrame()

    # ===== Helper methods for analyze_options =====
    async def _analyze_with_multiprocessing(
        self,
        tickers_upper: List[str],
        start_date: str,
        end_date: Optional[str],
        spread_mode: bool,
        spread_long_days: int,
        spread_long_days_tolerance: int,
        spread_long_min_days: Optional[int],
        spread_strike_tolerance: float,
        timestamp_lookback_days: int,
        position_size: float,
        days_to_expiry: Optional[int],
        min_volume: int,
        min_premium: float,
        min_write_timestamp: Optional[str],
        use_market_time: bool,
        filters: Optional[List[FilterExpression]],
        filter_logic: str,
        option_type: str,
        max_workers: int,
        sensible_price: float,
        max_bid_ask_spread: float = 2.0,
        max_bid_ask_spread_long: float = 2.0
    ) -> pd.DataFrame:
        """Analyze options using multiprocessing."""
        if spread_mode:
            # Calculate long-term date range for spread mode
            long_start_date, long_end_date = self._calculate_long_options_date_range(
                spread_long_days, spread_long_days_tolerance, spread_long_min_days
            )
            return await self._analyze_spread_multiprocess(
                tickers_upper=tickers_upper,
                start_date=start_date,
                end_date=end_date,
                long_start_date=long_start_date,
                long_end_date=long_end_date,
                timestamp_lookback_days=timestamp_lookback_days,
                position_size=position_size,
                days_to_expiry=days_to_expiry,
                min_volume=min_volume,
                min_premium=min_premium,
                min_write_timestamp=min_write_timestamp,
                use_market_time=use_market_time,
                filters=filters,
                filter_logic=filter_logic,
                option_type=option_type,
                spread_strike_tolerance=spread_strike_tolerance,
                spread_long_days=spread_long_days,
                spread_long_days_tolerance=spread_long_days_tolerance,
                spread_long_min_days=spread_long_min_days,
                max_workers=max_workers,
                sensible_price=sensible_price,
                max_bid_ask_spread=max_bid_ask_spread,
                max_bid_ask_spread_long=max_bid_ask_spread_long
            )
        else:
            return await self._analyze_options_multiprocess(
                tickers_upper=tickers_upper,
                start_date=start_date,
                end_date=end_date,
                timestamp_lookback_days=timestamp_lookback_days,
                position_size=position_size,
                days_to_expiry=days_to_expiry,
                min_volume=min_volume,
                min_premium=min_premium,
                min_write_timestamp=min_write_timestamp,
                use_market_time=use_market_time,
                filters=filters,
                filter_logic=filter_logic,
                option_type=option_type,
                max_workers=max_workers,
                sensible_price=sensible_price,
                max_bid_ask_spread=max_bid_ask_spread
            )
    
    async def _process_options_pipeline(
        self,
        options_df: pd.DataFrame,
        tickers_upper: List[str],
        option_type: str,
        use_market_time: bool,
        position_size: float,
        days_to_expiry: Optional[int],
        min_volume: int,
        min_premium: float,
        min_write_timestamp: Optional[str],
        sensible_price: float,
        spread_mode: bool,
        spread_strike_tolerance: float,
        spread_long_days: int,
        spread_long_days_tolerance: int,
        spread_long_min_days: Optional[int],
        start_date: str,
        end_date: Optional[str],
        max_concurrent: int,
        batch_size: int,
        timestamp_lookback_days: int,
        max_workers: int,
        long_options_df: Optional[pd.DataFrame]
    ) -> pd.DataFrame:
        """Process options through the analysis pipeline."""
        # 2) Filter by option type
        options_df = self._filter_options_by_type(options_df, option_type)
        if options_df.empty:
            option_type_label = option_type if option_type != 'both' else 'call/put'
            self._log("INFO", f"No {option_type_label} options found after filtering")
            return pd.DataFrame()

        # 3) Attach latest prices (market-time aware)
        df = await self._attach_latest_prices(options_df, tickers_upper, use_market_time)
        if df.empty:
            return pd.DataFrame()

        # 4) Derive metrics and apply filters
        df = self._derive_and_filter_short_metrics(
            df=df,
            position_size=position_size,
            days_to_expiry=days_to_expiry,
            min_volume=min_volume,
            min_premium=min_premium,
            min_write_timestamp=min_write_timestamp,
            option_type=option_type,
            sensible_price=sensible_price,
        )
        if df.empty:
            return pd.DataFrame()

        # 5) Normalize timestamps and select columns
        df = self._normalize_short_timestamps_and_select(df)
        
        # If spread mode is enabled, match short-term options with long-term options
        if spread_mode:
            log_spread_analysis_start(
                df_short=df,
                spread_strike_tolerance=spread_strike_tolerance,
                spread_long_days=spread_long_days,
                spread_long_min_days=spread_long_min_days,
                spread_long_days_tolerance=spread_long_days_tolerance,
                log_func=self._log
            )
            
            df = await self._create_spread_analysis(
                df_short=df,
                tickers=tickers_upper,
                spread_strike_tolerance=spread_strike_tolerance,
                spread_long_days=spread_long_days,
                spread_long_days_tolerance=spread_long_days_tolerance,
                spread_long_min_days=spread_long_min_days,
                start_date=start_date,
                end_date=end_date,
                max_concurrent=max_concurrent,
                batch_size=batch_size,
                timestamp_lookback_days=timestamp_lookback_days,
                max_workers=max_workers,
                position_size=position_size,
                min_write_timestamp=min_write_timestamp,
                option_type=option_type,
                long_options_df=long_options_df
            )
        
        return df
    
    async def _fetch_options_universe(
        self,
        spread_mode: bool,
        tickers_upper: List[str],
        start_date: str,
        end_date: Optional[str],
        spread_long_days: int,
        spread_long_days_tolerance: int,
        spread_long_min_days: Optional[int],
        timestamp_lookback_days: int,
        max_workers: int,
        max_concurrent: int,
        batch_size: int,
        min_write_timestamp: Optional[str],
        option_type: str
    ) -> Tuple[pd.DataFrame, Optional[pd.DataFrame]]:
        """
        Fetch options universe (combined short and long if spread mode, otherwise just short).
        
        Returns:
            Tuple of (options_df, long_options_df). long_options_df is None if not in spread mode.
        """
        long_options_df = None
        
        if spread_mode:
            # Compute long-term date window upfront
            long_start_date, long_end_date = self._calculate_long_options_date_range(
                spread_long_days, spread_long_days_tolerance, spread_long_min_days
            )
            
            # Calculate combined date range for single fetch
            combined_start_date, combined_end_date = self._calculate_combined_date_range(
                start_date, end_date, long_start_date, long_end_date
            )
            
            # Use larger timestamp lookback for combined fetch (long-term options may be older)
            combined_timestamp_lookback = max(timestamp_lookback_days, 180)
            
            self._log("INFO", f"Fetching options for {len(tickers_upper)} tickers (combined short and long-term range: {combined_start_date} to {combined_end_date})...")
            
            # Single fetch covering both short and long term ranges
            options_df = await self._fetch_combined_options_window(
                tickers_upper=tickers_upper,
                start_date=combined_start_date,
                end_date=combined_end_date,
                timestamp_lookback_days=combined_timestamp_lookback,
                max_workers=max_workers,
                max_concurrent=max_concurrent,
                batch_size=batch_size,
            )
            
            self._log("INFO", f"Fetched {len(options_df)} total options (combined short and long-term)")
            
            # Split the combined results into short and long term
            if not options_df.empty:
                options_df, long_options_df = split_combined_options_by_date_range(
                    options_df=options_df,
                    start_date=start_date,
                    end_date=end_date,
                    long_start_date=long_start_date,
                    long_end_date=long_end_date
                )
                
                if not long_options_df.empty:
                    long_options_df = self._filter_and_prepare_long_options(
                        long_options_df=long_options_df,
                        min_write_timestamp=min_write_timestamp,
                        long_start_date=long_start_date,
                        long_end_date=long_end_date,
                        tickers=tickers_upper,
                        option_type=option_type
                    )
                    self._log("INFO", f"Split into {len(options_df)} short-term and {len(long_options_df)} long-term options")
        else:
            self._log("INFO", f"Fetching options for {len(tickers_upper)} tickers (date range: {start_date} to {end_date or 'unlimited'})...")
            options_df = await self._fetch_short_options_window(
                tickers_upper=tickers_upper,
                start_date=start_date,
                end_date=end_date,
                timestamp_lookback_days=timestamp_lookback_days,
                max_workers=max_workers,
                max_concurrent=max_concurrent,
                batch_size=batch_size,
            )
            self._log("INFO", f"Fetched {len(options_df)} options")
        
        return options_df, long_options_df
    
    async def _fetch_short_options_window(
        self,
        tickers_upper: List[str],
        start_date: str,
        end_date: Optional[str],
        timestamp_lookback_days: int,
        max_workers: int,
        max_concurrent: int,
        batch_size: int
    ) -> pd.DataFrame:
        self._log("INFO", f"Starting options fetch for {len(tickers_upper)} tickers (date range: {start_date} to {end_date or 'unlimited'})...")
        if self.debug:
            print(f"DEBUG: Starting options fetch for {len(tickers_upper)} tickers", file=sys.stderr)
            print(f"DEBUG: Date range: {start_date} to {end_date}", file=sys.stderr)
            print(f"DEBUG: Tickers: {tickers_upper[:10]}{'...' if len(tickers_upper) > 10 else ''}", file=sys.stderr)

        if max_workers > 1:
            if self.debug:
                print(f"DEBUG: Using multiprocess mode with {max_workers} workers", file=sys.stderr)
            options_df = await self.db.get_latest_options_data_batch_multiprocess(
                tickers=tickers_upper,
                start_datetime=start_date,
                end_datetime=end_date,
                batch_size=batch_size,
                max_workers=max_workers,
                timestamp_lookback_days=timestamp_lookback_days
            )
        else:
            if self.debug:
                print("DEBUG: Using single-process mode", file=sys.stderr)
            options_df = await self.db.get_latest_options_data_batch(
                tickers=tickers_upper,
                start_datetime=start_date,
                end_datetime=end_date,
                max_concurrent=max_concurrent,
                batch_size=batch_size,
                timestamp_lookback_days=timestamp_lookback_days
            )

        if self.debug:
            print(f"DEBUG: Fetched {len(options_df)} total options from database", file=sys.stderr)
            if not options_df.empty:
                print(f"DEBUG: Options columns: {list(options_df.columns)}", file=sys.stderr)
                if 'ticker' in options_df.columns:
                    unique_tickers = options_df['ticker'].unique().tolist()
                    print(f"DEBUG: Options tickers: {unique_tickers[:10]}{'...' if len(unique_tickers) > 10 else ''}", file=sys.stderr)
                if 'option_type' in options_df.columns:
                    option_types = options_df['option_type'].unique().tolist()
                    print(f"DEBUG: Option types found: {option_types}", file=sys.stderr)
            else:
                print("DEBUG: No options data returned from database", file=sys.stderr)
        
        # Ensure ticker column exists - extract from option_ticker if missing
        if not options_df.empty and 'ticker' not in options_df.columns:
            if 'option_ticker' in options_df.columns:
                if self.debug:
                    print("DEBUG: ticker column missing, extracting from option_ticker", file=sys.stderr)
                options_df['ticker'] = options_df['option_ticker'].apply(self._extract_ticker_from_option_ticker)
                
                if self.debug:
                    extracted_tickers = options_df['ticker'].dropna().unique().tolist()
                    print(f"DEBUG: Extracted tickers from option_ticker: {extracted_tickers[:10]}{'...' if len(extracted_tickers) > 10 else ''}", file=sys.stderr)
            else:
                # If we have the list of tickers being queried, we could try to match, but it's safer to fail
                if self.debug:
                    print("DEBUG: Warning: Neither 'ticker' nor 'option_ticker' column found in options DataFrame", file=sys.stderr)
        
        self._log("INFO", f"Finished options fetch: {len(options_df)} options retrieved for {len(tickers_upper)} tickers")
        return options_df

    def _filter_options_by_type(self, options_df: pd.DataFrame, option_type: str = 'call') -> pd.DataFrame:
        before_filter = len(options_df)
        if 'option_type' in options_df.columns:
            if option_type == 'both':
                # Don't filter, keep both calls and puts
                self._log("INFO", f"Keeping both call and put options: {len(options_df)} options")
                if self.debug:
                    print(f"DEBUG: Keeping both call and put options: {len(options_df)} options", file=sys.stderr)
            else:
                options_df = options_df[options_df['option_type'] == option_type]
                self._log("INFO", f"After {option_type} filter: {len(options_df)} options (was {before_filter})")
                if self.debug:
                    print(f"DEBUG: After {option_type} filter: {len(options_df)} options (was {before_filter})", file=sys.stderr)
        else:
            self._log("WARNING", "'option_type' column not found in options DataFrame")
            if self.debug:
                print("DEBUG: Warning: 'option_type' column not found in options DataFrame", file=sys.stderr)
        return options_df

    async def _attach_latest_prices(
        self,
        options_df: pd.DataFrame,
        tickers_upper: List[str],
        use_market_time: bool,
        timestamp_cache: Optional[Dict[str, pd.Timestamp]] = None
    ) -> pd.DataFrame:
        stock_prices: Dict[str, float] = {}
        price_sources: Dict[str, str] = {}
        price_timestamps: Dict[str, Any] = {}

        async def fetch_price_for_ticker(ticker):
            try:
                price_data = await self.db.get_latest_price_with_data(ticker, use_market_time=use_market_time)
                if price_data and price_data.get('price'):
                    price = price_data['price']
                    source = price_data.get('source', 'unknown')
                    timestamp = price_data.get('timestamp')
                    if self.debug:
                        if source == 'realtime':
                            market_status = "OPEN (using latest realtime price)"
                        elif source == 'daily':
                            market_status = "CLOSED (using last close price from daily data)"
                        elif source == 'hourly':
                            market_status = "CLOSED (using hourly close price as fallback)"
                        else:
                            market_status = f"UNKNOWN (source: {source})"
                        print(f"DEBUG: {ticker}: ${price:.2f} from {source} - Market {market_status}", file=sys.stderr)
                    return ticker, price, source, timestamp
                return ticker, None, None, None
            except Exception as e:
                self._log("WARNING", f"Could not fetch price for {ticker}: {e}")
                return ticker, None, None, None

        price_tasks = [fetch_price_for_ticker(ticker) for ticker in tickers_upper]
        price_results = await asyncio.gather(*price_tasks)
        for ticker, price, source, timestamp in price_results:
            if price is not None:
                stock_prices[ticker] = price
                price_sources[ticker] = source
                if timestamp is not None:
                    price_timestamps[ticker] = timestamp

        # Fetch previous close prices based on each ticker's current price date
        # If market is open: compare realtime price to previous close
        # If market is closed: compare current close price to previous close (day before current price's date)
        prev_close_prices: Dict[str, float] = {}
        try:
            # First, try to get previous close based on each ticker's price timestamp
            for ticker in stock_prices.keys():
                timestamp = price_timestamps.get(ticker)
                if timestamp:
                    try:
                        if isinstance(timestamp, pd.Timestamp):
                            prev_close = await _get_previous_close_for_date(self.db, ticker, timestamp, debug=self.debug)
                        else:
                            ts = pd.to_datetime(timestamp, utc=True)
                            prev_close = await _get_previous_close_for_date(self.db, ticker, ts, debug=self.debug)
                        if prev_close is not None:
                            prev_close_prices[ticker] = prev_close
                    except Exception as e:
                        if self.debug:
                            print(f"DEBUG: {ticker} - Error getting previous close for date: {e}", file=sys.stderr)
            
            # Fallback: for any tickers we couldn't get previous close for, use standard method
            missing_tickers = [t for t in stock_prices.keys() if t not in prev_close_prices]
            if missing_tickers:
                fallback_prev_closes = await self.db.get_previous_close_prices(missing_tickers)
                prev_close_prices.update(fallback_prev_closes)
            
            if self.debug:
                print(f"DEBUG: Fetched previous close prices for {len(prev_close_prices)} tickers", file=sys.stderr)
                # Debug each ticker's prices
                for ticker in list(stock_prices.keys())[:5]:  # Show first 5 for debugging
                    current = stock_prices.get(ticker)
                    prev = prev_close_prices.get(ticker)
                    timestamp = price_timestamps.get(ticker)
                    if current and prev:
                        change = current - prev
                        change_pct = (change / prev) * 100 if prev > 0 else 0
                        print(f"DEBUG: {ticker} - current=${current:.2f}, prev_close=${prev:.2f}, change=${change:.2f} ({change_pct:.2f}%), timestamp={timestamp}", file=sys.stderr)
                    elif current:
                        print(f"DEBUG: {ticker} - current=${current:.2f}, prev_close=None (cannot calculate change), timestamp={timestamp}", file=sys.stderr)
        except Exception as e:
            if self.debug:
                print(f"DEBUG: Could not fetch previous close prices: {e}", file=sys.stderr)

        # Fetch latest option timestamps for each ticker (same as refresh check)
        # Use the reusable method with caching (use instance cache if no cache provided)
        cache_to_use = timestamp_cache if timestamp_cache is not None else self._timestamp_cache
        latest_opt_timestamps = await self._fetch_latest_option_timestamps(list(stock_prices.keys()), cache=cache_to_use)

        self._log("INFO", f"Fetched stock prices for {len(stock_prices)}/{len(tickers_upper)} tickers")
        if self.debug:
            print(f"DEBUG: Fetched stock prices for {len(stock_prices)}/{len(tickers_upper)} tickers", file=sys.stderr)
            if stock_prices and price_sources:
                source_counts: Dict[str, int] = {}
                for _, source in price_sources.items():
                    source_counts[source] = source_counts.get(source, 0) + 1
                print(f"DEBUG: Price sources: {source_counts}", file=sys.stderr)
                if self.debug:
                    sample_prices = [(t, stock_prices[t], price_sources.get(t, 'unknown')) for t in list(stock_prices.keys())[:5]]
                    print(f"DEBUG: Sample prices (ticker, price, source): {sample_prices}", file=sys.stderr)

        if not stock_prices:
            self._log("WARNING", "No stock prices fetched. Cannot calculate option metrics.")
            return pd.DataFrame()

        df = options_df.copy()
        if 'ticker' not in df.columns:
            self._log("ERROR", f"DataFrame missing 'ticker' column. Available columns: {list(df.columns)}")
            return pd.DataFrame()
        df['current_price'] = df['ticker'].map(stock_prices)
        df['latest_opt_ts'] = df['ticker'].map(latest_opt_timestamps)
        
        # Calculate price change and format as single column using helper function
        # Store percentage change separately for sorting
        def format_price_with_change_wrapper(row):
            ticker = row['ticker']
            current = row['current_price']
            prev_close = prev_close_prices.get(ticker) if prev_close_prices else None
            return _format_price_with_change(current, prev_close)
        
        # Apply formatting and store both display value and sort value
        result = df.apply(format_price_with_change_wrapper, axis=1, result_type='expand')
        df['price_with_change'] = result[0]
        df['price_change_pct'] = result[1]  # Store percentage for sorting
        
        before_price_filter = len(df)
        df = df[df['current_price'].notna()]
        self._log("INFO", f"After price mapping: {len(df)} options (was {before_price_filter})")
        if self.debug:
            print(f"DEBUG: After price mapping: {len(df)} options (was {before_price_filter})", file=sys.stderr)
        return df

    def _derive_and_filter_short_metrics(
        self,
        df: pd.DataFrame,
        position_size: float,
        days_to_expiry: Optional[int],
        min_volume: int,
        min_premium: float,
        min_write_timestamp: Optional[str],
        option_type: str = 'call',
        sensible_price: float = 1.0
    ) -> pd.DataFrame:
        """Derive metrics and apply filters for short options (uses helper functions)."""
        # Use helper function to calculate metrics
        df = _calculate_option_metrics(df, position_size, days_to_expiry)
        
        # Apply sensible price filter (strike price relative to current price as percentage multiplier)
        if sensible_price > 0 and not df.empty and 'current_price' in df.columns and 'strike_price' in df.columns and 'option_type' in df.columns:
            before_sensible_filter = len(df)
            if option_type == 'call' or option_type == 'both':
                # For calls: only show strikes > current_price * (1 + sensible_price) (OTM calls)
                # Example: if current_price=100 and sensible_price=0.05, show strikes > 105
                call_mask = (df['option_type'] == 'call') & (df['strike_price'] > df['current_price'] * (1 + sensible_price))
                if option_type == 'call':
                    df = df[call_mask].copy()
                else:  # both
                    # Keep calls that meet the filter, and all puts (puts will be filtered separately)
                    put_mask = df['option_type'] == 'put'
                    df = df[call_mask | put_mask].copy()
            
            if option_type == 'put' or option_type == 'both':
                # For puts: only show strikes < current_price * (1 - sensible_price) (OTM puts)
                # Example: if current_price=100 and sensible_price=0.05, show strikes < 95
                put_mask = (df['option_type'] == 'put') & (df['strike_price'] < df['current_price'] * (1 - sensible_price))
                if option_type == 'put':
                    df = df[put_mask].copy()
                else:  # both
                    # Keep puts that meet the filter, and all calls (calls already filtered above)
                    call_mask = df['option_type'] == 'call'
                    df = df[call_mask | put_mask].copy()
            
            if before_sensible_filter != len(df):
                self._log("INFO", f"After sensible_price filter ({sensible_price*100:.1f}%): {len(df)} options (was {before_sensible_filter})")
                if self.debug:
                    print(f"DEBUG: After sensible_price filter ({sensible_price*100:.1f}%): {len(df)} options (was {before_sensible_filter})", file=sys.stderr)
        
        if days_to_expiry is not None and not df.empty:
            before_days_filter = len(df)
            self._log("INFO", f"After days_to_expiry filter ({days_to_expiry}): {len(df)} options (was {before_days_filter})")
            if self.debug:
                print(f"DEBUG: After days_to_expiry filter ({days_to_expiry}): {len(df)} options (was {before_days_filter})", file=sys.stderr)

        # Apply basic filters using helper function
            before_volume_filter = len(df)
        ticker_label = self._infer_ticker_label(df)
        df = _apply_basic_filters(
            df,
            min_volume,
            min_premium,
            min_write_timestamp,
            debug=self.debug,
            ticker=ticker_label
        )
        
        if min_volume > 0 and before_volume_filter != len(df):
            self._log("INFO", f"After min_volume filter ({min_volume}): {len(df)} options (was {before_volume_filter})")
            if self.debug:
                print(f"DEBUG: After min_volume filter ({min_volume}): {len(df)} options (was {before_volume_filter})", file=sys.stderr)

        if min_premium > 0.0 and before_volume_filter != len(df):
            self._log("INFO", f"After min_premium filter ({min_premium}): {len(df)} options")
            if self.debug:
                print(f"DEBUG: After min_premium filter ({min_premium}): {len(df)} options", file=sys.stderr)

        if min_write_timestamp:
            try:
                import pytz
                est = pytz.timezone('America/New_York')
                min_ts = pd.to_datetime(min_write_timestamp)
                if min_ts.tz is None:
                    min_ts = est.localize(min_ts)
                min_ts_utc = min_ts.astimezone(pytz.UTC)
                self._log("INFO", f"Filtered options to those written after {min_write_timestamp} EST ({min_ts_utc} UTC)")
            except Exception as e:
                self._log("WARNING", f"Could not apply write timestamp filter: {e}")
        
        return df

    def _normalize_short_timestamps_and_select(self, df: pd.DataFrame) -> pd.DataFrame:
        """Normalize timestamps and select output columns (uses helper function)."""
        return _normalize_and_select_columns(df)
    
    def _calculate_long_options_date_range(
        self,
        spread_long_days: int,
        spread_long_days_tolerance: int,
        spread_long_min_days: Optional[int]
    ) -> Tuple[str, str]:
        """
        Calculate the date range for fetching long-term options.
        
        Returns:
            Tuple of (start_date, end_date) as strings in YYYY-MM-DD format
        """
        from datetime import date, timedelta
        today = date.today()
        
        if spread_long_min_days is not None:
            long_start_date = (today + timedelta(days=spread_long_min_days)).strftime('%Y-%m-%d')
            long_end_date = (today + timedelta(days=spread_long_days)).strftime('%Y-%m-%d')
            self._log("INFO", f"Long-term options expiring between {spread_long_min_days} and {spread_long_days} days")
            self._log("INFO", f"  Date range: {long_start_date} to {long_end_date}")
            if self.debug:
                print(f"DEBUG: Calculated long-term date range: {long_start_date} to {long_end_date} (from today {today})", file=sys.stderr)
        else:
            long_start_date = (today + timedelta(days=spread_long_days - spread_long_days_tolerance)).strftime('%Y-%m-%d')
            long_end_date = (today + timedelta(days=spread_long_days + spread_long_days_tolerance)).strftime('%Y-%m-%d')
            self._log("INFO", f"Long-term options expiring around {spread_long_days} days (±{spread_long_days_tolerance} days)")
            self._log("INFO", f"  Date range: {long_start_date} to {long_end_date}")
            if self.debug:
                print(f"DEBUG: Calculated long-term date range: {long_start_date} to {long_end_date} (from today {today})", file=sys.stderr)
        
        return long_start_date, long_end_date
    
    def _calculate_combined_date_range(
        self,
        short_start_date: str,
        short_end_date: Optional[str],
        long_start_date: str,
        long_end_date: str
    ) -> Tuple[str, Optional[str]]:
        """
        Calculate the combined date range covering both short and long term options.
        
        Returns:
            Tuple of (combined_start_date, combined_end_date) as strings in YYYY-MM-DD format
        """
        from datetime import datetime
        
        # Convert all dates to date objects for comparison
        short_start_dt = datetime.strptime(short_start_date, '%Y-%m-%d').date()
        short_end_dt = datetime.strptime(short_end_date, '%Y-%m-%d').date() if short_end_date else None
        long_start_dt = datetime.strptime(long_start_date, '%Y-%m-%d').date()
        long_end_dt = datetime.strptime(long_end_date, '%Y-%m-%d').date()
        
        # Find the overall min and max dates
        all_dates = [short_start_dt, long_start_dt, long_end_dt]
        if short_end_dt:
            all_dates.append(short_end_dt)
        
        combined_start = min(all_dates)
        combined_end = max(all_dates)
        
        return combined_start.strftime('%Y-%m-%d'), combined_end.strftime('%Y-%m-%d')
    
    async def _fetch_combined_options_window(
        self,
        tickers_upper: List[str],
        start_date: str,
        end_date: Optional[str],
        timestamp_lookback_days: int,
        max_workers: int,
        max_concurrent: int,
        batch_size: int
    ) -> pd.DataFrame:
        """Fetch options for combined date range (used in spread mode)."""
        self._log("INFO", f"Starting combined options fetch for {len(tickers_upper)} tickers (combined date range: {start_date} to {end_date or 'unlimited'})...")
        if self.debug:
            print(f"DEBUG: Starting combined options fetch for {len(tickers_upper)} tickers", file=sys.stderr)
            print(f"DEBUG: Combined date range: {start_date} to {end_date}", file=sys.stderr)
            print(f"DEBUG: Tickers: {tickers_upper[:10]}{'...' if len(tickers_upper) > 10 else ''}", file=sys.stderr)

        if max_workers > 1:
            if self.debug:
                print(f"DEBUG: Using multiprocess mode with {max_workers} workers", file=sys.stderr)
            options_df = await self.db.get_latest_options_data_batch_multiprocess(
                tickers=tickers_upper,
                start_datetime=start_date,
                end_datetime=end_date,
                batch_size=batch_size,
                max_workers=max_workers,
                timestamp_lookback_days=timestamp_lookback_days
            )
        else:
            if self.debug:
                print("DEBUG: Using single-process mode", file=sys.stderr)
            options_df = await self.db.get_latest_options_data_batch(
                tickers=tickers_upper,
                start_datetime=start_date,
                end_datetime=end_date,
                max_concurrent=max_concurrent,
                batch_size=batch_size,
                timestamp_lookback_days=timestamp_lookback_days
            )

        if self.debug:
            print(f"DEBUG: Fetched {len(options_df)} total options from database", file=sys.stderr)
            if not options_df.empty:
                print(f"DEBUG: Options columns: {list(options_df.columns)}", file=sys.stderr)
                if 'ticker' in options_df.columns:
                    unique_tickers = options_df['ticker'].unique().tolist()
                    print(f"DEBUG: Options tickers: {unique_tickers[:10]}{'...' if len(unique_tickers) > 10 else ''}", file=sys.stderr)
                if 'option_type' in options_df.columns:
                    option_types = options_df['option_type'].unique().tolist()
                    print(f"DEBUG: Option types found: {option_types}", file=sys.stderr)
            else:
                print("DEBUG: No options data returned from database", file=sys.stderr)
        
        # Ensure ticker column exists - extract from option_ticker if missing
        if not options_df.empty and 'ticker' not in options_df.columns:
            if 'option_ticker' in options_df.columns:
                if self.debug:
                    print("DEBUG: ticker column missing, extracting from option_ticker", file=sys.stderr)
                options_df['ticker'] = options_df['option_ticker'].apply(self._extract_ticker_from_option_ticker)
                
                if self.debug:
                    extracted_tickers = options_df['ticker'].dropna().unique().tolist()
                    print(f"DEBUG: Extracted tickers from option_ticker: {extracted_tickers[:10]}{'...' if len(extracted_tickers) > 10 else ''}", file=sys.stderr)
            else:
                # If we have the list of tickers being queried, we could try to match, but it's safer to fail
                if self.debug:
                    print("DEBUG: Warning: Neither 'ticker' nor 'option_ticker' column found in options DataFrame", file=sys.stderr)
        
        self._log("INFO", f"Finished combined options fetch: {len(options_df)} options retrieved for {len(tickers_upper)} tickers")
        return options_df
    
    async def _fetch_long_term_options(
        self,
        tickers: List[str],
        long_start_date: str,
        long_end_date: str,
        timestamp_lookback_days: int,
        max_workers: int,
        max_concurrent: int,
        batch_size: int
    ) -> pd.DataFrame:
        """
        Fetch long-term options from the database.
        
        Returns:
            DataFrame with long-term options data
        """
        # Use a much larger timestamp lookback for long-term options since they may have been
        # written weeks or months ago but are still valid. Use at least 180 days to catch options
        # that were written when they were first listed (which could be months before expiration)
        long_timestamp_lookback_days = max(timestamp_lookback_days, 180)
        if self.debug:
            print(f"DEBUG: Using timestamp_lookback_days={long_timestamp_lookback_days} for long-term options (vs {timestamp_lookback_days} for short-term)", file=sys.stderr)
        
        try:
            if self.debug:
                print(f"DEBUG: Calling get_latest_options_data_batch with:", file=sys.stderr)
                print(f"  tickers: {tickers}", file=sys.stderr)
                print(f"  start_datetime: {long_start_date}", file=sys.stderr)
                print(f"  end_datetime: {long_end_date}", file=sys.stderr)
                print(f"  timestamp_lookback_days: {long_timestamp_lookback_days}", file=sys.stderr)
            
            if max_workers > 1:
                long_options_df = await self.db.get_latest_options_data_batch_multiprocess(
                    tickers=tickers,
                    start_datetime=long_start_date,
                    end_datetime=long_end_date,
                    batch_size=batch_size,
                    max_workers=max_workers,
                    timestamp_lookback_days=long_timestamp_lookback_days
                )
            else:
                long_options_df = await self.db.get_latest_options_data_batch(
                    tickers=tickers,
                    start_datetime=long_start_date,
                    end_datetime=long_end_date,
                    max_concurrent=max_concurrent,
                    batch_size=batch_size,
                    timestamp_lookback_days=long_timestamp_lookback_days
                )
            
            if self.debug:
                print(f"DEBUG: Batch fetch returned {len(long_options_df)} rows", file=sys.stderr)
            
            # Ensure ticker column exists - extract from option_ticker if missing
            if not long_options_df.empty and 'ticker' not in long_options_df.columns:
                if 'option_ticker' in long_options_df.columns:
                    if self.debug:
                        print("DEBUG: ticker column missing in long options, extracting from option_ticker", file=sys.stderr)
                    long_options_df['ticker'] = long_options_df['option_ticker'].apply(self._extract_ticker_from_option_ticker)
                    
                    if self.debug:
                        extracted_tickers = long_options_df['ticker'].dropna().unique().tolist()
                        print(f"DEBUG: Extracted tickers from long option_ticker: {extracted_tickers[:10]}{'...' if len(extracted_tickers) > 10 else ''}", file=sys.stderr)
                else:
                    if self.debug:
                        print("DEBUG: Warning: Neither 'ticker' nor 'option_ticker' column found in long options DataFrame", file=sys.stderr)
            
            return long_options_df
        except Exception as e:
            self._log("ERROR", f"Error fetching long-term options: {e}")
            import traceback
            if self.debug:
                traceback.print_exc()
            raise
    
    def _filter_and_prepare_long_options(
        self,
        long_options_df: pd.DataFrame,
        min_write_timestamp: Optional[str],
        long_start_date: str,
        long_end_date: str,
        tickers: List[str],
        option_type: str = 'call'
    ) -> pd.DataFrame:
        """
        Filter and prepare long options DataFrame (filter by option type, write timestamp filter, etc.).
        
        Args:
            option_type: Type of options to filter ('call', 'put', or 'both'). Default: 'call'
        
        Returns:
            Filtered and prepared DataFrame
        """
        if self.debug:
            print(f"DEBUG: Fetched {len(long_options_df)} total options from database")
            if not long_options_df.empty:
                print(f"DEBUG: Long options columns: {list(long_options_df.columns)}")
                if 'ticker' in long_options_df.columns:
                    unique_tickers = long_options_df['ticker'].unique().tolist()
                    print(f"DEBUG: Long options tickers: {unique_tickers}")
                    for ticker in unique_tickers:
                        count = len(long_options_df[long_options_df['ticker'] == ticker])
                        print(f"DEBUG:   {ticker}: {count} options")
                if 'option_type' in long_options_df.columns:
                    print(f"DEBUG: Option types: {long_options_df['option_type'].unique().tolist()}")
            else:
                print(f"DEBUG: No long-term options found in database for date range {long_start_date} to {long_end_date}")
        
        # Apply write timestamp filter to long options if specified
        if min_write_timestamp and not long_options_df.empty:
            before_count = len(long_options_df)
            ticker_label = self._infer_ticker_label(long_options_df, tickers)
            long_options_df = _apply_basic_filters(
                long_options_df,
                0,
                0.0,
                min_write_timestamp,
                debug=self.debug,
                ticker=ticker_label
            )
            after_count = len(long_options_df)
            if before_count != after_count:
                self._log("INFO", f"Filtered long options by write timestamp: {before_count} -> {after_count} options")
                if self.debug:
                    import pytz
                    est = pytz.timezone('America/New_York')
                    min_ts = pd.to_datetime(min_write_timestamp)
                    if min_ts.tz is None:
                        min_ts = est.localize(min_ts)
                    min_ts_utc = min_ts.astimezone(pytz.UTC)
                    print(f"DEBUG: Applied write timestamp filter >= {min_ts_utc} UTC")
        
        if long_options_df.empty:
            self._log("WARNING", "No long-term options found for spread analysis.")
            if self.debug:
                print("DEBUG: This could mean:", file=sys.stderr)
                print("  1. No options data in database for the specified date range", file=sys.stderr)
                print("  2. Options exist but not in the target expiration window", file=sys.stderr)
                print(f"  3. Check database for tickers: {tickers}", file=sys.stderr)
            return pd.DataFrame()
        
        # Filter by option type
        # Note: When option_type='both', we keep both calls and puts in long_options_df.
        # The actual matching (in process_spread_match) will ensure that each short option
        # only matches with long options of the same type (puts with puts, calls with calls).
        if 'option_type' in long_options_df.columns:
            if option_type == 'both':
                # Keep both calls and puts (matching will be done by process_spread_match based on each short option's type)
                if self.debug:
                    print(f"DEBUG: Keeping both call and put long options: {len(long_options_df)} options")
            else:
                long_options_df = long_options_df[long_options_df['option_type'] == option_type].copy()
                if self.debug:
                    print(f"DEBUG: After filtering for {option_type} long options: {len(long_options_df)} {option_type} options")
        
        if long_options_df.empty:
            option_type_label = option_type if option_type != 'both' else 'call/put'
            self._log("WARNING", f"No long-term {option_type_label} options found for spread analysis.")
            return pd.DataFrame()
        
        # Ensure we have a copy before modifying
        long_options_df = long_options_df.copy()
        if 'implied_volatility' in long_options_df.columns:
            long_options_df['implied_volatility'] = pd.to_numeric(long_options_df['implied_volatility'], errors='coerce').round(4)
        else:
            long_options_df['implied_volatility'] = pd.Series([float('nan')] * len(long_options_df), index=long_options_df.index)
        
        # Calculate days to expiry for long options using utility functions
        long_options_df['expiration_date'] = long_options_df['expiration_date'].apply(_normalize_to_utc)
        today_ts = pd.Timestamp.now(tz='UTC').normalize()
        long_options_df['days_to_expiry'] = long_options_df['expiration_date'].apply(lambda x: _safe_days_calc(x, today_ts))
        
        if self.debug:
            print(f"DEBUG: Long options days to expiry range: {long_options_df['days_to_expiry'].min()} to {long_options_df['days_to_expiry'].max()}")
            print(f"DEBUG: Long options strike price range: {long_options_df['strike_price'].min()} to {long_options_df['strike_price'].max()}")
        
        return long_options_df
    
    def _prepare_spread_matching_data(
        self,
        df_short: pd.DataFrame,
        long_options_df: pd.DataFrame
    ) -> Tuple[List[Dict], Dict[str, List[Dict]]]:
        """
        Prepare data structures for multiprocessing spread matching.
        
        Returns:
            Tuple of (short_rows_list, long_options_dict)
        """
        # Reset index and deduplicate
        df_short = df_short.reset_index(drop=True)
        if 'option_ticker' in df_short.columns:
            before_dedup = len(df_short)
            df_short = df_short.drop_duplicates(subset=['option_ticker'], keep='first')
            if self.debug and len(df_short) < before_dedup:
                print(f"DEBUG: Deduplicated df_short: {before_dedup} -> {len(df_short)} rows (removed {before_dedup - len(df_short)} duplicates)", file=sys.stderr)
        
        # Convert df_short rows to dictionaries (serializable)
        short_rows_list = [row.to_dict() for _, row in df_short.iterrows()]
        
        # Convert long_options_df to a dictionary structure (ticker -> list of option dicts)
        long_options_dict = {}
        for ticker in long_options_df['ticker'].unique():
            ticker_options = long_options_df[long_options_df['ticker'] == ticker]
            long_options_dict[ticker] = []
            for _, row in ticker_options.iterrows():
                row_dict = row.to_dict()
                # Timestamps are preserved as-is (they're pickleable)
                if 'expiration_date' in row_dict and isinstance(row_dict['expiration_date'], pd.Timestamp):
                    row_dict['expiration_date'] = row_dict['expiration_date']
                long_options_dict[ticker].append(row_dict)
        
        return short_rows_list, long_options_dict
    
    async def _execute_spread_matching(
        self,
        short_rows_list: List[Dict],
        long_options_dict: Dict[str, List[Dict]],
        spread_strike_tolerance: float,
        spread_long_days: int,
        spread_long_days_tolerance: int,
        spread_long_min_days: Optional[int],
        position_size: float,
        max_workers: int
    ) -> List[Dict]:
        """
        Execute spread matching using multiprocessing or sequential processing.
        
        Returns:
            List of spread result dictionaries
        """
        # Prepare arguments for multiprocessing
        process_args = [
            (
                short_row_dict,
                long_options_dict,
                spread_strike_tolerance,
                spread_long_days,
                spread_long_days_tolerance,
                spread_long_min_days,
                position_size,
                self.risk_free_rate,
                self.debug
            )
            for short_row_dict in short_rows_list
        ]
        
        # Use multiprocessing to process spread matches in parallel
        if max_workers > 1 and len(short_rows_list) > 0:
            if self.debug:
                print(f"DEBUG: Processing {len(short_rows_list)} spread matches using {max_workers} CPU workers", file=sys.stderr)
            
            loop = asyncio.get_event_loop()
            with ProcessPoolExecutor(max_workers=max_workers) as executor:
                futures = [loop.run_in_executor(executor, _process_spread_match, args) for args in process_args]
                results = await asyncio.gather(*futures)
            
            # Filter out None results (no matches)
            return [r for r in results if r is not None]
        else:
            # Fallback to sequential processing
            if self.debug and max_workers <= 1:
                print(f"DEBUG: Using sequential processing (max_workers={max_workers})", file=sys.stderr)
            spread_results = []
            for args in process_args:
                result = _process_spread_match(args)
                if result is not None:
                    spread_results.append(result)
            return spread_results
    
    async def _create_spread_analysis(
        self,
        df_short: pd.DataFrame,
        tickers: List[str],
        spread_strike_tolerance: float,
        spread_long_days: int,
        spread_long_days_tolerance: int,
            spread_long_min_days: Optional[int],
            start_date: Optional[str],
            end_date: Optional[str],
            max_concurrent: int,
            batch_size: int,
            timestamp_lookback_days: int,
            max_workers: int,
            position_size: float,
            min_write_timestamp: Optional[str],
            option_type: str = 'call',
            long_options_df: Optional[pd.DataFrame] = None
        ) -> pd.DataFrame:
        """
        Match short-term options with long-term options to create calendar spread analysis.
        
        Args:
            df_short: DataFrame with short-term option analysis
            tickers: List of tickers to analyze
            spread_strike_tolerance: Percentage tolerance for strike matching
            spread_long_days: Maximum/target days to expiry for long options
            spread_long_days_tolerance: Days tolerance for long option expiration window (e.g., ±14 days, ignored if spread_long_min_days is set)
            spread_long_min_days: Minimum days to expiry for long options (if set, searches from min to max instead of using tolerance)
            long_options_df: Optional pre-fetched long-term options DataFrame. If provided, skips fetching.
            Other args: Same as analyze_options
            
        Returns:
            DataFrame with spread analysis including long option details and net calculations
        """
        if df_short.empty:
            return df_short
        
        try:
            # Calculate date range for long options
            long_start_date, long_end_date = self._calculate_long_options_date_range(
                spread_long_days, spread_long_days_tolerance, spread_long_min_days
            )
            
            if self.debug:
                print(f"DEBUG: Tickers to fetch: {tickers}", file=sys.stderr)
                print(f"DEBUG: Short-term options count: {len(df_short)}", file=sys.stderr)
                print(f"DEBUG: Short-term date range in df_short:", file=sys.stderr)
                if not df_short.empty and 'expiration_date' in df_short.columns:
                    print(f"  Min expiration: {df_short['expiration_date'].min()}", file=sys.stderr)
                    print(f"  Max expiration: {df_short['expiration_date'].max()}", file=sys.stderr)
            
            # Fetch long-term options if not already provided
            if long_options_df is None:
                self._log("INFO", f"Fetching long-term options...")
                long_options_df = await self._fetch_long_term_options(
                    tickers=tickers,
                    long_start_date=long_start_date,
                    long_end_date=long_end_date,
                    timestamp_lookback_days=timestamp_lookback_days,
                    max_workers=max_workers,
                    max_concurrent=max_concurrent,
                    batch_size=batch_size
                )
            else:
                self._log("INFO", f"Using pre-fetched long-term options ({len(long_options_df)} rows)")
                if self.debug:
                    print(f"DEBUG: Using pre-fetched long-term options DataFrame with {len(long_options_df)} rows", file=sys.stderr)
                long_options_df = long_options_df.copy()
            
            # Filter and prepare long options
            long_options_df = self._filter_and_prepare_long_options(
                long_options_df=long_options_df,
                min_write_timestamp=min_write_timestamp,
                long_start_date=long_start_date,
                long_end_date=long_end_date,
                tickers=tickers,
                option_type=option_type
            )
            
            if long_options_df.empty:
                return pd.DataFrame()
            
            # Prepare data for spread matching
            short_rows_list, long_options_dict = self._prepare_spread_matching_data(
                df_short=df_short,
                long_options_df=long_options_df
            )
            
            # Execute spread matching
            spread_results = await self._execute_spread_matching(
                short_rows_list=short_rows_list,
                long_options_dict=long_options_dict,
                spread_strike_tolerance=spread_strike_tolerance,
                spread_long_days=spread_long_days,
                spread_long_days_tolerance=spread_long_days_tolerance,
                spread_long_min_days=spread_long_min_days,
                position_size=position_size,
                max_workers=max_workers
            )
            
            if not spread_results:
                self._log("WARNING", "No matching spread opportunities found within strike tolerance.")
                if self.debug:
                    print(f"DEBUG: Summary - Processed {len(short_rows_list)} short options, but none matched with long options", file=sys.stderr)
                    print("DEBUG: Possible reasons:", file=sys.stderr)
                    print("  1. Strike prices don't overlap between short and long options", file=sys.stderr)
                    print("  2. Strike tolerance is too strict (try increasing --spread-strike-tolerance)", file=sys.stderr)
                    print(f"  3. Long options not available in the {spread_long_days}±{spread_long_days_tolerance} day window", file=sys.stderr)
                    print(f"  4. Try increasing --spread-long-days-tolerance to widen the search window", file=sys.stderr)
                return pd.DataFrame()
            
            # Create spread DataFrame
            df_spread = pd.DataFrame(spread_results)
            
            self._log("INFO", f"✓ Found {len(df_spread)} spread opportunities (matched short and long options).")
            
            return df_spread
            
        except Exception as e:
            self._log("ERROR", f"Error creating spread analysis: {e}")
            import traceback
            if self.debug:
                traceback.print_exc()
            return pd.DataFrame()
    
    def format_output(
        self,
        df: pd.DataFrame,
        financial_data: Dict[str, Dict[str, Any]],
        output_format: str = 'table',
        group_by: str = 'overall',
        output_file: Optional[str] = None,
        sort_by: Optional[str] = None,
        filters: Optional[List[FilterExpression]] = None,
        filter_logic: str = 'AND',
        csv_delimiter: str = ',',
        csv_quoting: str = 'minimal',
        csv_columns: Optional[List[str]] = None,
        top_n: Optional[int] = 1,
        spread_mode: bool = False
    ) -> str:
        """Format the analysis results for output."""
        if self.debug:
            print(f"DEBUG: format_output called with {len(df)} rows", file=sys.stderr)
        if df.empty:
            self._log("INFO", "DataFrame is empty in format_output")
            return "No options data found matching the criteria."
        
        if 'ticker' not in df.columns:
            return "Error: DataFrame missing 'ticker' column"
        
        # Enrich dataframe with financial data and derived columns
        df_renamed = enrich_dataframe_with_financial_data(df, financial_data)
        df_renamed = add_derived_percentage_columns(df_renamed)
        df_renamed = format_timestamp_columns(df_renamed)
        
        # Select display columns
        is_spread_mode = 'net_premium' in df_renamed.columns
        display_columns = get_display_columns(is_spread_mode)
        available_columns = [col for col in display_columns if col in df_renamed.columns]
        df_display = df_renamed[available_columns].copy()
        
        # Apply filters if provided
        if filters:
            before_filter_count = len(df_display)
            df_display = FilterParser.apply_filters(df_display, filters, filter_logic)
            self._log("INFO", f"After filter application: {len(df_display)} rows (was {before_filter_count})")
            if self.debug:
                print(f"DEBUG: After filter application: {len(df_display)} rows (was {before_filter_count})", file=sys.stderr)
        
        # Apply sorting
        compact_headers = self._create_compact_headers(df_display)
        df_display = apply_sorting(df_display, sort_by, compact_headers)
        
        # Apply top-n filter with grouping
        before_topn = len(df_display)
        if top_n and top_n > 0:
            # Determine grouping columns based on mode
            if spread_mode:
                group_cols = ['ticker', 'expiration_date', 'long_expiration_date']
            else:
                group_cols = ['ticker', 'expiration_date']
            
            # Only use columns that exist in the dataframe
            group_cols = [col for col in group_cols if col in df_display.columns]
            
            if group_cols:
                # Group by the appropriate columns and take top N within each group
                df_display = df_display.groupby(group_cols, group_keys=False).head(top_n)
                if self.debug:
                    grouping_desc = " + ".join(group_cols)
                    print(f"DEBUG: Applied top-n filter ({top_n}) grouped by ({grouping_desc}): {len(df_display)} rows (was {before_topn})", file=sys.stderr)
            else:
                # Fallback to global top-n if no grouping columns available
                df_display = apply_top_n_filter(df_display, top_n)
                if self.debug:
                    print(f"DEBUG: Applied global top-n filter ({top_n}): {len(df_display)} rows (was {before_topn})", file=sys.stderr)
            
            if before_topn != len(df_display):
                self._log("INFO", f"After top-n filter ({top_n}): {len(df_display)} rows (was {before_topn})")
        else:
            if self.debug:
                print(f"DEBUG: Skipping top-n filter (top_n={top_n})", file=sys.stderr)
        
        # Handle CSV formatting
        if output_format == 'csv':
            return self._format_csv_output_with_columns(
                df_display, compact_headers, csv_columns, csv_delimiter, 
                csv_quoting, group_by, output_file
            )
        
        # Recreate compact_headers for table output (may have been modified)
        compact_headers = self._create_compact_headers(df_display)
        
        # Format table output
        result = format_table_output(df_display, compact_headers, group_by)
        
        # Save to file if specified
        if output_file:
            with open(output_file, 'w') as f:
                f.write(result)
            self._log("INFO", f"Results saved to {output_file}")
        
        return result
    
    def _format_csv_output_with_columns(
        self,
        df: pd.DataFrame,
        compact_headers: Dict[str, str],
        csv_columns: Optional[List[str]],
        csv_delimiter: str,
        csv_quoting: str,
        group_by: str,
        output_file: Optional[str]
    ) -> str:
        """Format CSV output with column resolution."""
        df_csv = df.copy()
        header_reverse_map = {v: k for k, v in compact_headers.items()}
        
        if csv_columns:
            resolved_columns = resolve_csv_columns(csv_columns, df_csv, header_reverse_map)
            if resolved_columns:
                df_csv = df_csv[resolved_columns]
                compact_headers = {
                    col: compact_headers[col] for col in resolved_columns if col in compact_headers
                }
        
        df_csv = df_csv.rename(columns=compact_headers)
        return self._format_csv_output(df_csv, csv_delimiter, csv_quoting, group_by, output_file)


# ============================================================================
# Helper functions for main() - modularized for reusability
# ============================================================================

# Alias for backward compatibility
_get_redis_client_for_refresh = get_redis_client_for_refresh
_check_redis_refresh_pending = check_redis_refresh_pending
_set_redis_refresh_pending = set_redis_refresh_pending
_clear_redis_refresh_pending = clear_redis_refresh_pending


def _build_analysis_args(args, tickers: List[str], filters: List[FilterExpression]) -> Dict[str, Any]:
    """Build arguments dictionary for analyze_options call. Reusable for both initial and refresh analysis."""
    return {
        'tickers': tickers,
        'option_type': getattr(args, 'option_type', 'call'),
        'days_to_expiry': args.days,
        'min_volume': args.min_volume,
        'max_days': args.max_days,
        'min_days': getattr(args, 'min_days', None),
        'batch_size': args.batch_size,
        'min_premium': args.min_premium,
        'position_size': args.position_size,
        'filters': filters,
        'filter_logic': args.filter_logic,
        'use_market_time': not args.no_market_time,
        'start_date': args.start_date,
        'end_date': args.end_date,
        'timestamp_lookback_days': args.timestamp_lookback_days,
        'max_workers': args.max_workers,
        'spread_mode': args.spread,
        'spread_strike_tolerance': args.spread_strike_tolerance,
        'spread_long_days': args.spread_long_days,
        'spread_long_days_tolerance': args.spread_long_days_tolerance,
        'spread_long_min_days': args.spread_long_min_days,
        'min_write_timestamp': args.min_write_timestamp,
        'sensible_price': getattr(args, 'sensible_price', 0.01),
        'max_bid_ask_spread': getattr(args, 'max_bid_ask_spread', 2.0),
        'max_bid_ask_spread_long': getattr(args, 'max_bid_ask_spread_long', 2.0)
    }


def _get_arg_value(args: Union[argparse.Namespace, Dict[str, Any]], key: str, default=None):
    """Safely retrieve an argument value from either argparse Namespace or dict."""
    if isinstance(args, dict):
        return args.get(key, default)
    return getattr(args, key, default)


async def _check_tickers_for_refresh(
    analyzer: OptionsAnalyzer,
    tickers: List[str],
    refresh_threshold_seconds: int,
    redis_client: Optional[Any] = None,
    timestamp_cache: Optional[Dict[str, pd.Timestamp]] = None,
    min_write_timestamp: Optional[str] = None
) -> List[str]:
    """
    Check which tickers need refresh based on their latest write_timestamp.
    Also includes tickers that don't meet the min_write_timestamp criteria.
    
    Args:
        analyzer: OptionsAnalyzer instance
        tickers: List of ticker symbols to check
        refresh_threshold_seconds: Age threshold in seconds for refresh
        redis_client: Optional Redis client for deduplication
        timestamp_cache: Optional cache dictionary to reuse previously fetched timestamps
        min_write_timestamp: Optional minimum write timestamp (EST format) - tickers with data older than this will be refreshed
        
    Returns:
        List of ticker symbols that need refresh
    """
    # Use analyzer's method to fetch timestamps
    async def fetch_timestamps(tickers_list: List[str], cache: Optional[Dict]) -> Dict[str, Optional[float]]:
        """Wrapper to use analyzer's timestamp fetching method."""
        return await analyzer._fetch_latest_option_timestamps(tickers_list, cache=cache)
    
    return await common_check_tickers_for_refresh(
        db=analyzer.db,
        tickers=tickers,
        refresh_threshold_seconds=refresh_threshold_seconds,
        fetch_timestamp_func=fetch_timestamps,
        redis_client=redis_client,
        timestamp_cache=timestamp_cache,
        min_write_timestamp=min_write_timestamp,
        debug=analyzer.debug
    )


def _calculate_refresh_date_ranges(
    analyzer: OptionsAnalyzer,
    args,
    today_str: str,
    today_date
) -> Tuple[str, Optional[str], int]:
    """
    Calculate date ranges and max_days for refresh fetch.
    For refresh, we always use 30 days max expiration.
    
    Returns:
        Tuple of (short_start_date, max_end_date, combined_max_days)
    """
    from datetime import date
    
    # Short-term date range (from original analysis)
    start_date = _get_arg_value(args, 'start_date', None)
    short_start_date = start_date if start_date else today_str
    short_end_date = _get_arg_value(args, 'end_date', None)
    
    # Calculate long-term date range if in spread mode
    long_start_date = None
    long_end_date = None
    if _get_arg_value(args, 'spread', False):
        long_start_date, long_end_date = analyzer._calculate_long_options_date_range(
            _get_arg_value(args, 'spread_long_days', 90),
            _get_arg_value(args, 'spread_long_days_tolerance', 10),
            _get_arg_value(args, 'spread_long_min_days', None)
        )
    
    # Determine the maximum end date for combined fetch (if spread mode) or display
    max_end_date = short_end_date
    if _get_arg_value(args, 'spread', False) and long_end_date:
        # Convert to date objects for comparison
        short_end_dt = datetime.strptime(short_end_date, '%Y-%m-%d').date() if short_end_date else None
        long_end_dt = datetime.strptime(long_end_date, '%Y-%m-%d').date()
        
        if short_end_dt:
            max_end_date = max(short_end_dt, long_end_dt).strftime('%Y-%m-%d')
        else:
            max_end_date = long_end_date
    
    # For refresh, always use 30 days max expiration
    combined_max_days = 30
    
    return short_start_date, max_end_date, combined_max_days


def _process_ticker_financial_info(args_tuple):
    """
    Process financial info for a single ticker in a separate process.
    
    Args:
        args_tuple: Tuple containing:
            - ticker: Ticker symbol
            - db_conn: Database connection string
            - enable_cache: Whether caching is enabled
            - redis_url: Redis URL for caching
            - log_level: Logging level
            - debug: Whether debug output is enabled
    
    Returns:
        Tuple of (ticker, financial_data_dict)
    """
    import asyncio
    import sys
    import os
    import pandas as pd
    from pathlib import Path
    from typing import Dict, Any
    
    # Unpack arguments
    (ticker, db_conn, enable_cache, redis_url, log_level, debug) = args_tuple
    
    # Re-import needed modules in worker process
    CURRENT_DIR = Path(__file__).resolve().parent
    PROJECT_ROOT = CURRENT_DIR.parent
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    
    from common.stock_db import get_stock_db
    
    async def _async_process():
        # Create database connection in worker process
        # Use INFO level for database to reduce cache message verbosity, even when analyzer is in DEBUG mode
        db_log_level = "INFO" if log_level == "DEBUG" else log_level
        db = get_stock_db('questdb', db_config=db_conn, enable_cache=enable_cache, 
                        redis_url=redis_url, log_level=db_log_level)
        
        try:
            async with db:
                # Fetch financial info
                df = await db.get_financial_info(ticker)
                
                if not df.empty:
                    # Map column names to expected fields
                    row = df.iloc[0]
                    # Try different possible column names
                    pe_ratio = None
                    if 'price_to_earnings' in df.columns:
                        pe_ratio = row['price_to_earnings']
                    elif 'pe_ratio' in df.columns:
                        pe_ratio = row['pe_ratio']
                    
                    market_cap = row['market_cap'] if 'market_cap' in df.columns else None
                    price = row['price'] if 'price' in df.columns else None
                    
                    # Extract IV metrics from financial info
                    financial_data_dict = {
                        'pe_ratio': pe_ratio,
                        'market_cap': market_cap,
                        'price': price
                    }
                    
                    # Extract IV rank (30-day)
                    if 'iv_rank' in df.columns:
                        financial_data_dict['iv_rank'] = row['iv_rank']
                    
                    # Extract IV rank (90-day)
                    if 'iv_90d_rank' in df.columns:
                        financial_data_dict['iv_90d_rank'] = row['iv_90d_rank']
                    
                    # Parse IV analysis JSON if present
                    if 'iv_analysis_json' in df.columns and pd.notna(row.get('iv_analysis_json')):
                        import json
                        try:
                            iv_analysis = json.loads(row['iv_analysis_json'])
                            # Store full iv_metrics dict for enrich_dataframe_with_financial_data
                            if 'metrics' in iv_analysis:
                                financial_data_dict['iv_metrics'] = iv_analysis['metrics']
                            
                            # Extract individual metrics for easier access
                            metrics = iv_analysis.get('metrics', {})
                            strategy = iv_analysis.get('strategy', {})
                            
                            # Store strategy data for compatibility with stock info page format
                            if strategy:
                                financial_data_dict['iv_strategy'] = strategy
                            
                            # Risk score is in strategy, not metrics - store at root level for backward compatibility
                            if 'risk_score' in strategy:
                                financial_data_dict['risk_score'] = strategy['risk_score']
                            
                            # IV rank 90-day is in metrics as rank_90d
                            if 'rank_90d' in metrics:
                                financial_data_dict['iv_90d_rank'] = metrics['rank_90d']
                            
                            # Roll yield is in metrics as a string like "2.5%"
                            if 'roll_yield' in metrics:
                                roll_yield_str = metrics['roll_yield']
                                if isinstance(roll_yield_str, str) and roll_yield_str.endswith('%'):
                                    financial_data_dict['roll_yield'] = float(roll_yield_str.rstrip('%'))
                                else:
                                    financial_data_dict['roll_yield'] = roll_yield_str
                            
                            # Recommendation is in strategy - store at root level for backward compatibility
                            if 'recommendation' in strategy:
                                financial_data_dict['iv_recommendation'] = strategy['recommendation']
                        except (json.JSONDecodeError, TypeError, ValueError) as e:
                            if debug:
                                print(f"DEBUG: Could not parse IV analysis JSON for {ticker}: {e}", file=sys.stderr)
                    
                    return (ticker, financial_data_dict)
                else:
                    return (ticker, {
                        'pe_ratio': None,
                        'market_cap': None,
                        'price': None,
                        'iv_rank': None,
                        'iv_90d_rank': None,
                        'risk_score': None,
                        'iv_recommendation': None,
                        'roll_yield': None
                    })
        except Exception as e:
            if debug:
                print(f"WARNING [PID {os.getpid()}]: Could not fetch financial info for {ticker}: {e}", file=sys.stderr)
            return (ticker, {
                'pe_ratio': None,
                'market_cap': None,
                'price': None
            })
    
    # Run async function in worker process
    return asyncio.run(_async_process())


def _process_refresh_batch(args_tuple):
    """
    Process a batch of tickers for refresh in a separate process.
    
    Args:
        args_tuple: Tuple containing:
            - tickers: List of ticker symbols to refresh
            - db_conn: Database connection string
            - api_key: Polygon API key
            - data_dir: Data directory
            - today_str: Today's date string
            - enable_cache: Whether caching is enabled
            - redis_url: Redis URL for caching
            - log_level: Logging level
            - debug: Whether debug output is enabled
    
    Returns:
        List of result dictionaries, one per ticker
    """
    import asyncio
    import sys
    import os
    import pandas as pd
    from pathlib import Path
    
    # Unpack arguments
    (tickers, db_conn, api_key, data_dir, today_str, enable_cache, redis_url, log_level, debug) = args_tuple
    
    # Re-import needed modules in worker process
    CURRENT_DIR = Path(__file__).resolve().parent
    PROJECT_ROOT = CURRENT_DIR.parent
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    
    from common.stock_db import get_stock_db
    from common.common import get_redis_client_for_refresh, set_redis_last_write_timestamp, REDIS_AVAILABLE
    from scripts.fetch_options import HistoricalDataFetcher
    
    async def _async_process():
        # Get process ID for logging
        import multiprocessing
        process_id = os.getpid()
        try:
            process_name = multiprocessing.current_process().name
        except:
            process_name = 'unknown'
        ticker_list = ', '.join(tickers)
        print(f"INFO [PID {process_id}, Process: {process_name}]: Starting refresh batch - Processing {len(tickers)} ticker(s): {ticker_list}", file=sys.stderr)
        
        # Create database connection in worker process
        # Use INFO level for database to reduce cache message verbosity, even when analyzer is in DEBUG mode
        db_log_level = "INFO" if log_level == "DEBUG" else log_level
        db = get_stock_db('questdb', db_config=db_conn, enable_cache=enable_cache, 
                        redis_url=redis_url, log_level=db_log_level)
        
        # Create fetcher instance
        fetcher = HistoricalDataFetcher(
            api_key,
            data_dir,
            verbose=False,  # Suppress fetch_options progress in workers (quiet=True means verbose=False)
            snapshot_max_concurrent=0
        )
        
        results = []
        
        try:
            async with db:
                for ticker in tickers:
                    try:
                        # Get stock price for the ticker
                        stock_result = await fetcher.get_stock_price_for_date(ticker, today_str)
                        stock_close_price = stock_result['data'].get('close') if stock_result.get('success') else None
                        
                        # Fetch options - always use 30 days max expiration for refresh
                        options_result = await fetcher.get_active_options_for_date(
                            symbol=ticker,
                            target_date_str=today_str,
                            option_type='call',
                            stock_close_price=stock_close_price,
                            strike_range_percent=None,
                            max_days_to_expiry=30,  # Always use 30 days for refresh
                            include_expired=False,
                            use_cache=False,
                            save_to_csv=False,
                            use_db=False,
                            db_conn=None,
                            force_fresh=True,
                            enable_cache=enable_cache,
                            redis_url=redis_url
                        )
                        
                        if options_result.get('success'):
                            contracts = options_result['data'].get('contracts', [])
                            if contracts:
                                # Convert contracts to DataFrame and save to database
                                contracts_df = pd.DataFrame.from_records(contracts)
                                if not contracts_df.empty:
                                    # Map columns to match DB schema
                                    if 'ticker' in contracts_df.columns and 'option_ticker' not in contracts_df.columns:
                                        contracts_df = contracts_df.rename(columns={'ticker': 'option_ticker'})
                                    
                                    column_mapping = {
                                        'expiration': 'expiration_date',
                                        'strike': 'strike_price',
                                        'type': 'option_type',
                                    }
                                    for old_name, new_name in column_mapping.items():
                                        if old_name in contracts_df.columns:
                                            contracts_df = contracts_df.rename(columns={old_name: new_name})
                                    
                                    # Save to database
                                    await db.save_options_data(df=contracts_df, ticker=ticker)
                                    
                                    # Update Redis cache with the current timestamp
                                    if enable_cache and redis_url:
                                        redis_client = get_redis_client_for_refresh(redis_url) if redis_url else None
                                        if redis_client:
                                            from datetime import datetime, timezone
                                            now_utc = datetime.now(timezone.utc)
                                            set_redis_last_write_timestamp(redis_client, ticker, now_utc, ttl_seconds=86400)
                            
                            results.append({'ticker': ticker, 'success': True, 'contracts': len(contracts)})
                        else:
                            results.append({'ticker': ticker, 'success': False, 'error': options_result.get('error', 'Unknown error')})
                    except Exception as e:
                        results.append({'ticker': ticker, 'success': False, 'error': str(e)})
        except Exception as e:
            # If database connection fails, return errors for all tickers
            for ticker in tickers:
                results.append({'ticker': ticker, 'success': False, 'error': f"Database error: {str(e)}"})
        
        # Log completion with process ID
        successful_count = sum(1 for r in results if r.get('success'))
        print(f"INFO [PID {process_id}]: Completed refresh batch - {successful_count}/{len(tickers)} ticker(s) successful", file=sys.stderr)
        
        return results
    
    # Run async function in worker process
    return asyncio.run(_async_process())


async def _fetch_and_save_refresh_options(
    fetcher: Any,
    ticker: str,
    today_str: str,
    combined_max_days: int,
    analyzer: OptionsAnalyzer,
    enable_cache: bool,
    redis_client: Optional[Any] = None
) -> Dict[str, Any]:
    """Fetch and save options data for a single ticker during refresh."""
    try:
        # Get stock price for the ticker (needed for fetch_options)
        stock_result = await fetcher.get_stock_price_for_date(ticker, today_str)
        stock_close_price = stock_result['data'].get('close') if stock_result.get('success') else None
        
        # Fetch options - always use 30 days max expiration for refresh
        redis_url = os.getenv('REDIS_URL', 'redis://localhost:6379/0') if enable_cache else None
        
        options_result = await fetcher.get_active_options_for_date(
            symbol=ticker,
            target_date_str=today_str,
            option_type='call',  # Only calls for covered call analysis
            stock_close_price=stock_close_price,
            strike_range_percent=None,  # Fetch all strikes
            max_days_to_expiry=30,  # Always use 30 days for refresh
            include_expired=False,
            use_cache=False,  # Don't use CSV cache
            save_to_csv=False,  # Don't save to CSV
            use_db=False,  # We'll save manually
            db_conn=None,
            force_fresh=True,  # Force fresh fetch
            enable_cache=enable_cache,
            redis_url=redis_url
        )
        
        if options_result.get('success'):
            contracts = options_result['data'].get('contracts', [])
            if contracts:
                # Convert contracts to DataFrame and save to database
                contracts_df = pd.DataFrame.from_records(contracts)
                if not contracts_df.empty:
                    # Map columns to match DB schema
                    if 'ticker' in contracts_df.columns and 'option_ticker' not in contracts_df.columns:
                        contracts_df = contracts_df.rename(columns={'ticker': 'option_ticker'})
                    
                    column_mapping = {
                        'expiration': 'expiration_date',
                        'strike': 'strike_price',
                        'type': 'option_type',
                    }
                    for old_name, new_name in column_mapping.items():
                        if old_name in contracts_df.columns:
                            contracts_df = contracts_df.rename(columns={old_name: new_name})
                    
                    # Save to database using analyzer's db connection
                    await analyzer.db.save_options_data(df=contracts_df, ticker=ticker)
                    
                    # Update Redis cache with the current timestamp
                    if redis_client:
                        from datetime import datetime, timezone
                        now_utc = datetime.now(timezone.utc)
                        set_redis_last_write_timestamp(redis_client, ticker, now_utc, ttl_seconds=86400)
                        _clear_redis_refresh_pending(redis_client, ticker)
            
            analyzer._log("INFO", f"  ✓ Fetched and saved {len(contracts)} options contracts for {ticker}")
            return {'ticker': ticker, 'success': True, 'contracts': len(contracts)}
        else:
            # Clear Redis flag on failure
            if redis_client:
                _clear_redis_refresh_pending(redis_client, ticker)
            analyzer._log("WARNING", f"  ✗ Failed to fetch options for {ticker}: {options_result.get('error', 'Unknown error')}")
            return {'ticker': ticker, 'success': False, 'error': options_result.get('error')}
    except Exception as e:
        # Clear Redis flag on error
        if redis_client:
            _clear_redis_refresh_pending(redis_client, ticker)
        analyzer._log("ERROR", f"  ✗ Error fetching options for {ticker}: {e}")
        return {'ticker': ticker, 'success': False, 'error': str(e)}


async def _run_refresh_analysis(
    analyzer: OptionsAnalyzer,
    args,
    df: pd.DataFrame,
    filters: List[FilterExpression],
    refresh_threshold_seconds: int,
    redis_client: Optional[Any] = None,
    timestamp_cache: Optional[Dict[str, pd.Timestamp]] = None
) -> pd.DataFrame:
    """
    Run the refresh analysis: check timestamps, fetch fresh data, and re-analyze.
    
    Returns:
        Updated DataFrame with refreshed results, or original if refresh fails/skipped
    """
    original_df = df.copy()
    
    if not POLYGON_AVAILABLE or HistoricalDataFetcher is None:
        analyzer._log("WARNING", "Warning: --refresh-results requires fetch_options module. Refresh skipped.")
        return original_df
    
    api_key = os.getenv('POLYGON_API_KEY')
    if not api_key:
        analyzer._log("WARNING", "Warning: POLYGON_API_KEY environment variable not set. Refresh skipped.")
        return original_df
    
    # Check if market is open
    now_utc = datetime.now(timezone.utc)
    is_market_open = common_is_market_hours(now_utc, "America/New_York")
    
    if not is_market_open:
        analyzer._log("INFO", "Market is closed. Skipping refresh (option prices don't change during non-market hours).")
        return original_df
    
    # Extract unique tickers from results
    result_tickers = df['ticker'].unique().tolist() if 'ticker' in df.columns else []
    
    if not result_tickers:
        analyzer._log("INFO", "No tickers found in results. Skipping refresh.")
        return original_df
    
    # Check which tickers need refresh (reuse timestamp cache if provided)
    min_write_timestamp = getattr(args, 'min_write_timestamp', None)
    tickers_to_refresh = await _check_tickers_for_refresh(
        analyzer, result_tickers, refresh_threshold_seconds, redis_client, timestamp_cache, min_write_timestamp
    )
    
    if not tickers_to_refresh:
        analyzer._log("INFO", f"\nAll tickers have fresh data (within {refresh_threshold_seconds}s threshold). No refresh needed.")
        return original_df
    
    # Calculate percentage
    refresh_percentage = (len(tickers_to_refresh) / len(result_tickers) * 100) if result_tickers else 0
    
    # Print summary to stderr so it's always visible
    print(f"\n=== Refresh Summary ===", file=sys.stderr)
    print(f"Total tickers in results: {len(result_tickers)}", file=sys.stderr)
    print(f"Tickers being refreshed: {len(tickers_to_refresh)} ({refresh_percentage:.1f}%)", file=sys.stderr)
    print(f"\nAll tickers in results: {', '.join(sorted(result_tickers))}", file=sys.stderr)
    print(f"\nTickers being refreshed: {', '.join(sorted(tickers_to_refresh))}", file=sys.stderr)
    print("", file=sys.stderr)
    
    analyzer._log("INFO", f"\n=== Refreshing options data for {len(tickers_to_refresh)} ticker(s) ===")
    analyzer._log("INFO", f"Tickers to refresh: {', '.join(tickers_to_refresh)}")
    
    try:
        # Calculate date ranges
        today_str = datetime.now().strftime('%Y-%m-%d')
        from datetime import date
        today_date = date.today()
        
        short_start_date, max_end_date, combined_max_days = _calculate_refresh_date_ranges(
            analyzer, args, today_str, today_date
        )
        
        # Set Redis flags for pending refreshes (only during market hours)
        if redis_client:
            for ticker in tickers_to_refresh:
                _set_redis_refresh_pending(redis_client, ticker, ttl_seconds=900)
        
        # Use multiprocessing with max_workers/2 processes
        enable_cache = not args.no_cache
        redis_url = os.getenv('REDIS_URL', 'redis://localhost:6379/0') if enable_cache else None
        max_workers = getattr(args, 'max_workers', 4)
        refresh_workers = max(1, max_workers // 2)  # Use half of max_workers, minimum 1
        
        analyzer._log("INFO", f"Using {refresh_workers} processes for refresh (max_workers={max_workers})")
        
        # Split tickers across processes
        tickers_per_process = max(1, len(tickers_to_refresh) // refresh_workers)
        ticker_batches = []
        for i in range(0, len(tickers_to_refresh), tickers_per_process):
            batch = tickers_to_refresh[i:i + tickers_per_process]
            if batch:
                ticker_batches.append(batch)
        
        # Ensure we don't have more batches than workers
        if len(ticker_batches) > refresh_workers:
            ticker_batches = ticker_batches[:refresh_workers]
        
        analyzer._log("INFO", f"Split {len(tickers_to_refresh)} tickers into {len(ticker_batches)} batches")
        
        # Log ticker distribution per batch (print to stderr so it's always visible)
        print(f"\nRefresh multiprocess ticker distribution ({refresh_workers} processes):", file=sys.stderr)
        for i, batch in enumerate(ticker_batches, 1):
            ticker_list = ', '.join(batch)
            print(f"  Process {i}: {len(batch)} ticker(s) - {ticker_list}", file=sys.stderr)
        print("", file=sys.stderr)  # Empty line for readability
        
        # Prepare arguments for each batch
        process_args = []
        for batch in ticker_batches:
            args_tuple = (
                batch,
                analyzer.db_conn,
                api_key,
                args.data_dir,
                today_str,
                enable_cache,
                redis_url,
                analyzer.log_level,
                analyzer.debug
            )
            process_args.append(args_tuple)
        
        # Execute in parallel using ProcessPoolExecutor
        from concurrent.futures import ProcessPoolExecutor
        loop = asyncio.get_event_loop()
        start_time = datetime.now()
        
        with ProcessPoolExecutor(max_workers=refresh_workers) as executor:
            futures = [
                loop.run_in_executor(executor, _process_refresh_batch, args)
                for args in process_args
            ]
            batch_results = await asyncio.gather(*futures, return_exceptions=True)
        
        # Flatten results from all batches
        refresh_results = []
        for i, batch_result in enumerate(batch_results):
            if isinstance(batch_result, Exception):
                # If a batch failed, mark all tickers in that batch as failed
                batch = ticker_batches[i]
                for ticker in batch:
                    refresh_results.append({'ticker': ticker, 'success': False, 'error': str(batch_result)})
            else:
                refresh_results.extend(batch_result)
        
        # Clear Redis flags for successful refreshes
        if redis_client:
            for result in refresh_results:
                if result.get('success'):
                    _clear_redis_refresh_pending(redis_client, result['ticker'])
        
        # Log progress
            elapsed = (datetime.now() - start_time).total_seconds()
        successful = sum(1 for r in refresh_results if r.get('success'))
        analyzer._log("INFO", f"\nRefresh complete: {successful}/{len(tickers_to_refresh)} tickers successful ({elapsed:.1f}s elapsed)")
        
        # Log individual results
        for result in refresh_results:
            ticker = result.get('ticker', 'unknown')
            if result.get('success'):
                contracts = result.get('contracts', 0)
                analyzer._log("INFO", f"  ✓ {ticker}: {contracts} contracts")
            else:
                error = result.get('error', 'Unknown error')
                analyzer._log("WARNING", f"  ✗ {ticker}: {error}")
        
        # Small delay to ensure database commits are visible before re-analysis
        if successful > 0:
            await asyncio.sleep(0.5)  # 500ms delay to ensure DB commits are visible
        
        analyzer._log("INFO", f"Re-analyzing options for refreshed tickers...")
        
        # Re-run analysis on refreshed tickers only
        analysis_args = _build_analysis_args(args, tickers_to_refresh, filters)
        df = await analyzer.analyze_options(**analysis_args)
        
        if df.empty:
            analyzer._log("WARNING", "Warning: No results after refresh. Using original results.")
            return original_df
        else:
            analyzer._log("INFO", f"✓ Re-analysis complete: {len(df)} options found after refresh")
            return df
    
    except Exception as e:
        analyzer._log("ERROR", f"Error during refresh: {e}")
        if args.debug:
            import traceback
            traceback.print_exc()
        analyzer._log("WARNING", "Using original analysis results due to refresh error.")
        return original_df


def _run_background_refresh_worker_subprocess(
    db_conn: str,
    tickers: List[str],
    refresh_threshold_seconds: int,
    redis_url: Optional[str],
    log_level: str,
    debug: bool,
    enable_cache: bool,
    args_dict: Dict[str, Any]
) -> None:
    """
    Worker function for subprocess-based background refresh.
    This version doesn't use multiprocessing.Queue since it's called via subprocess.
    """
    _run_background_refresh_worker_internal(
        db_conn, tickers, refresh_threshold_seconds, redis_url,
        log_level, debug, enable_cache, args_dict, status_queue=None
    )


def _run_background_refresh_worker(
    db_conn: str,
    tickers: List[str],
    refresh_threshold_seconds: int,
    redis_url: Optional[str],
    log_level: str,
    debug: bool,
    enable_cache: bool,
    args_dict: Dict[str, Any],
    status_queue: Optional[Any] = None
) -> None:
    """
    Worker function to run refresh in background process (multiprocessing version).
    """
    _run_background_refresh_worker_internal(
        db_conn, tickers, refresh_threshold_seconds, redis_url,
        log_level, debug, enable_cache, args_dict, status_queue
    )


def _run_background_refresh_worker_internal(
    db_conn: str,
    tickers: List[str],
    refresh_threshold_seconds: int,
    redis_url: Optional[str],
    log_level: str,
    debug: bool,
    enable_cache: bool,
    args_dict: Dict[str, Any],
    status_queue: Optional[Any] = None
) -> None:
    """
    Worker function to run refresh in background process.
    
    Args:
        status_queue: Optional multiprocessing.Queue to send status updates to parent process
    """
    import asyncio
    import sys
    import os
    from pathlib import Path
    
    # Immediately send status to parent if queue is provided
    if status_queue is not None:
        try:
            import multiprocessing
            worker_pid = os.getpid()
            try:
                worker_name = multiprocessing.current_process().name
            except:
                worker_name = 'unknown'
            status_queue.put({
                'pid': worker_pid,
                'name': worker_name,
                'tickers': tickers,
                'ticker_count': len(tickers)
            })
        except Exception as e:
            # If queue communication fails, continue anyway
            print(f"Warning: Failed to send status to parent: {e}", file=sys.stderr)
    
    # Re-import needed modules in the worker process
    CURRENT_DIR = Path(__file__).resolve().parent
    PROJECT_ROOT = CURRENT_DIR.parent
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    
    async def _worker():
        try:
            # Log process info
            worker_pid = os.getpid()
            try:
                import multiprocessing
                worker_name = multiprocessing.current_process().name
            except:
                worker_name = 'unknown'
            print(f"Background refresh worker started [PID {worker_pid}, Process: {worker_name}]", file=sys.stderr)
            
            # Create analyzer in worker process
            analyzer = OptionsAnalyzer(db_conn, log_level=log_level, debug=debug, enable_cache=enable_cache, redis_url=redis_url)
            await analyzer.initialize()
            
            # Get Redis client
            redis_client = None
            if redis_url and REDIS_AVAILABLE:
                redis_client = _get_redis_client_for_refresh(redis_url)
            
            # Check market hours
            from common.market_hours import is_market_hours
            now_utc = datetime.now(timezone.utc)
            is_market_open = is_market_hours(now_utc, "America/New_York")
            
            if not is_market_open:
                analyzer._log("INFO", "Background refresh: Market is closed. Skipping refresh.")
                return
            
            # Check which tickers need refresh
            min_write_timestamp = args_dict.get('min_write_timestamp', None)
            tickers_to_refresh = await _check_tickers_for_refresh(
                analyzer, tickers, refresh_threshold_seconds, redis_client, None, min_write_timestamp
            )
            
            if not tickers_to_refresh:
                analyzer._log("INFO", "Background refresh: No tickers need refresh.")
                return
            
            analyzer._log("INFO", f"Background refresh: Refreshing {len(tickers_to_refresh)} ticker(s): {', '.join(tickers_to_refresh)}")
            
            # Import fetch_options in worker
            try:
                from scripts.fetch_options import HistoricalDataFetcher
            except ImportError:
                analyzer._log("ERROR", "Background refresh: fetch_options module not available.")
                return
            
            api_key = os.getenv('POLYGON_API_KEY')
            if not api_key:
                analyzer._log("ERROR", "Background refresh: POLYGON_API_KEY not set.")
                return
            
            # Calculate date ranges
            from datetime import date
            today_str = datetime.now().strftime('%Y-%m-%d')
            today_date = date.today()
            
            short_start_date, max_end_date, combined_max_days = _calculate_refresh_date_ranges(
                analyzer, args_dict, today_str, today_date
            )
            
            # Set Redis flags
            if redis_client:
                for ticker in tickers_to_refresh:
                    _set_redis_refresh_pending(redis_client, ticker, ttl_seconds=1800)
            
            # Use multiprocessing with max_workers/2 processes
            max_workers = args_dict.get('max_workers', 4)
            refresh_workers = max(1, max_workers // 2)  # Use half of max_workers, minimum 1
            
            analyzer._log("INFO", f"Background refresh: Using {refresh_workers} processes (max_workers={max_workers})")
            
            # Split tickers across processes
            tickers_per_process = max(1, len(tickers_to_refresh) // refresh_workers)
            ticker_batches = []
            for i in range(0, len(tickers_to_refresh), tickers_per_process):
                batch = tickers_to_refresh[i:i + tickers_per_process]
                if batch:
                    ticker_batches.append(batch)
            
            # Ensure we don't have more batches than workers
            if len(ticker_batches) > refresh_workers:
                ticker_batches = ticker_batches[:refresh_workers]
            
            analyzer._log("INFO", f"Background refresh: Split {len(tickers_to_refresh)} tickers into {len(ticker_batches)} batches")
            
            # Log ticker distribution per batch (print to stderr so it's always visible)
            print(f"\nBackground refresh multiprocess ticker distribution ({refresh_workers} processes):", file=sys.stderr)
            for i, batch in enumerate(ticker_batches, 1):
                ticker_list = ', '.join(batch)
                print(f"  Process {i}: {len(batch)} ticker(s) - {ticker_list}", file=sys.stderr)
            print("", file=sys.stderr)  # Empty line for readability
            
            # Prepare arguments for each batch
            process_args = []
            for batch in ticker_batches:
                args_tuple = (
                    batch,
                    db_conn,
                    api_key,
                    args_dict.get('data_dir', './data'),
                    today_str,
                    enable_cache,
                    redis_url,
                    log_level,
                    debug
                )
                process_args.append(args_tuple)
            
            # Execute in parallel using ProcessPoolExecutor
            from concurrent.futures import ProcessPoolExecutor
            loop = asyncio.get_event_loop()
            
            with ProcessPoolExecutor(max_workers=refresh_workers) as executor:
                futures = [
                    loop.run_in_executor(executor, _process_refresh_batch, args)
                    for args in process_args
                ]
                batch_results = await asyncio.gather(*futures, return_exceptions=True)
            
            # Flatten results from all batches
            refresh_results = []
            for i, batch_result in enumerate(batch_results):
                if isinstance(batch_result, Exception):
                    # If a batch failed, mark all tickers in that batch as failed
                    batch = ticker_batches[i]
                    for ticker in batch:
                        refresh_results.append({'ticker': ticker, 'success': False, 'error': str(batch_result)})
                else:
                    refresh_results.extend(batch_result)
            
            # Clear Redis flags and log results
            for result in refresh_results:
                ticker = result.get('ticker', 'unknown')
                if result.get('success'):
                    if redis_client:
                        _clear_redis_refresh_pending(redis_client, ticker)
                    contracts = result.get('contracts', 0)
                    analyzer._log("INFO", f"Background refresh: ✓ {ticker} - {contracts} contracts")
                else:
                    if redis_client:
                        _clear_redis_refresh_pending(redis_client, ticker)
                    error = result.get('error', 'Unknown error')
                    analyzer._log("WARNING", f"Background refresh: ✗ {ticker} - {error}")
            
            analyzer._log("INFO", "Background refresh: Complete")
            
        except Exception as e:
            print(f"Background refresh worker error: {e}", file=sys.stderr)
            import traceback
            traceback.print_exc()
        finally:
            if analyzer and analyzer.db:
                await analyzer.db.close()
    
    # Run the async worker
    asyncio.run(_worker())


async def _run_background_refresh(
    analyzer: OptionsAnalyzer,
    args,
    df: pd.DataFrame,
    filters: List[FilterExpression],
    refresh_threshold_seconds: int,
    redis_client: Optional[Any],
    timestamp_cache: Optional[Dict[str, pd.Timestamp]] = None,
    all_tickers: Optional[List[str]] = None
) -> None:
    """
    Run refresh in a background process without waiting.
    Main process continues and shows existing results.
    
    Args:
        all_tickers: Optional list of all loaded tickers to check. If provided, checks all tickers
                     instead of just those in results. This allows refreshing data for tickers that
                     may not have passed filters yet.
    """
    if not POLYGON_AVAILABLE or HistoricalDataFetcher is None:
        analyzer._log("WARNING", "Background refresh: fetch_options module not available. Skipping.")
        return
    
    api_key = os.getenv('POLYGON_API_KEY')
    if not api_key:
        analyzer._log("WARNING", "Background refresh: POLYGON_API_KEY not set. Skipping.")
        return
    
    # Check if market is open
    now_utc = datetime.now(timezone.utc)
    is_market_open = common_is_market_hours(now_utc, "America/New_York")
    
    if not is_market_open:
        analyzer._log("INFO", "Background refresh: Market is closed. Skipping.")
        return
    
    # Use all_tickers if provided, otherwise fall back to tickers from results
    if all_tickers:
        tickers_to_check = all_tickers
        analyzer._log("INFO", f"Background refresh: Checking freshness for all {len(tickers_to_check)} loaded tickers")
    else:
        # Extract unique tickers from results
        tickers_to_check = df['ticker'].unique().tolist() if 'ticker' in df.columns else []
        if not tickers_to_check:
            analyzer._log("INFO", "Background refresh: No tickers found in results. Skipping.")
            return
    
    # Check which tickers need refresh (with Redis deduplication, reuse timestamp cache if provided)
    min_write_timestamp = getattr(args, 'min_write_timestamp', None)
    tickers_to_refresh = await _check_tickers_for_refresh(
        analyzer, tickers_to_check, refresh_threshold_seconds, redis_client, timestamp_cache, min_write_timestamp
    )
    
    if not tickers_to_refresh:
        analyzer._log("INFO", "Background refresh: No tickers need refresh.")
        return
    
    # Calculate percentage
    refresh_percentage = (len(tickers_to_refresh) / len(tickers_to_check) * 100) if tickers_to_check else 0
    
    # Print summary to stderr so it's always visible
    print(f"\n=== Background Refresh Summary ===", file=sys.stderr)
    print(f"Total tickers checked: {len(tickers_to_check)}", file=sys.stderr)
    print(f"Tickers being refreshed: {len(tickers_to_refresh)} ({refresh_percentage:.1f}%)", file=sys.stderr)
    if len(tickers_to_check) <= 20:
        print(f"\nAll tickers checked: {', '.join(sorted(tickers_to_check))}", file=sys.stderr)
    else:
        print(f"\nAll tickers checked: {', '.join(sorted(tickers_to_check)[:10])} ... ({len(tickers_to_check)} total)", file=sys.stderr)
    if len(tickers_to_refresh) <= 20:
        print(f"\nTickers being refreshed: {', '.join(sorted(tickers_to_refresh))}", file=sys.stderr)
    else:
        print(f"\nTickers being refreshed: {', '.join(sorted(tickers_to_refresh)[:10])} ... ({len(tickers_to_refresh)} total)", file=sys.stderr)
    print("", file=sys.stderr)
    
    tickers_display = ', '.join(tickers_to_refresh[:10])
    if len(tickers_to_refresh) > 10:
        tickers_display += f" ... ({len(tickers_to_refresh)} total)"
    analyzer._log("WARNING", f"Background refresh initiated for {len(tickers_to_refresh)} ticker(s): {tickers_display}")
    analyzer._log("INFO", f"Starting background refresh for {len(tickers_to_refresh)} ticker(s): {tickers_display}")
    
    # Calculate refresh workers and ticker distribution in main process for visibility
    max_workers = getattr(args, 'max_workers', 4)
    refresh_workers = max(1, max_workers // 2)  # Use half of max_workers, minimum 1
    
    # Split tickers across processes to show distribution
    tickers_per_process = max(1, len(tickers_to_refresh) // refresh_workers)
    ticker_batches = []
    for i in range(0, len(tickers_to_refresh), tickers_per_process):
        batch = tickers_to_refresh[i:i + tickers_per_process]
        if batch:
            ticker_batches.append(batch)
    
    # Ensure we don't have more batches than workers
    if len(ticker_batches) > refresh_workers:
        ticker_batches = ticker_batches[:refresh_workers]
    
    # Print ticker distribution to stderr so it's always visible
    print(f"\nBackground refresh multiprocess ticker distribution ({refresh_workers} processes):", file=sys.stderr)
    for i, batch in enumerate(ticker_batches, 1):
        ticker_list = ', '.join(batch)
        print(f"  Process {i}: {len(batch)} ticker(s) - {ticker_list}", file=sys.stderr)
    print("", file=sys.stderr)  # Empty line for readability
    
    # Prepare arguments for worker process
    args_dict = {
        'data_dir': getattr(args, 'data_dir', './data'),
        'spread': getattr(args, 'spread', False),
        'start_date': getattr(args, 'start_date', None),
        'end_date': getattr(args, 'end_date', None),
        'max_days': getattr(args, 'max_days', None),
        'spread_long_days': getattr(args, 'spread_long_days', 90),
        'spread_long_days_tolerance': getattr(args, 'spread_long_days_tolerance', 10),
        'spread_long_min_days': getattr(args, 'spread_long_min_days', None),
        'min_write_timestamp': min_write_timestamp,
        'max_workers': max_workers,
    }
    
    # Spawn background process using subprocess so it can survive parent exit
    try:
        import subprocess
        import json
        import tempfile
        
        # Create a temporary file to pass arguments (since subprocess needs serializable args)
        # We'll pass the arguments as JSON in environment variables and a temp file
        temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False)
        worker_args = {
            'db_conn': analyzer.db_conn,
            'tickers': tickers_to_refresh,
            'refresh_threshold_seconds': refresh_threshold_seconds,
            'redis_url': analyzer.redis_url,
            'log_level': analyzer.log_level,
            'debug': analyzer.debug,
            'enable_cache': analyzer.enable_cache,
            'args_dict': args_dict
        }
        json.dump(worker_args, temp_file)
        temp_file.close()
        args_file = temp_file.name
        
        # Get the script directory for proper path resolution
        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(script_dir)
        
        # Create a wrapper script that will run the worker
        # We need to import the module and call the worker function
        script_content = f"""
import sys
import os
import json
import asyncio
from pathlib import Path

# Add project to path
PROJECT_ROOT = r'{project_root}'
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Import the worker function
from scripts.options_analyzer import _run_background_refresh_worker_subprocess

# Read arguments from file
with open(r'{args_file}', 'r') as f:
    args = json.load(f)

# Run the worker
_run_background_refresh_worker_subprocess(**args)
"""
        
        # Write wrapper script to temp file
        script_file = tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False)
        script_file.write(script_content)
        script_file.close()
        script_path = script_file.name
        
        # Make script executable
        os.chmod(script_path, 0o755)
        
        # Start subprocess with proper detachment
        # Use start_new_session=True to create a new process group
        # Redirect stdout/stderr to files so we can see output
        log_file = tempfile.NamedTemporaryFile(mode='w', suffix='.log', delete=False)
        log_file.close()
        
        process = subprocess.Popen(
            [sys.executable, script_path],
            stdout=open(log_file.name, 'w'),
            stderr=subprocess.STDOUT,
            start_new_session=True,  # Create new session so process survives parent exit
            cwd=os.getcwd()
        )
        
        # Give it a moment to start and send initial status
        import time
        time.sleep(0.5)
        
        # Check if process is still running
        if process.poll() is None:
            print(f"\nBackground refresh process started and running:", file=sys.stderr)
            print(f"  Worker PID: {process.pid}", file=sys.stderr)
            print(f"  Processing {len(tickers_to_refresh)} ticker(s)", file=sys.stderr)
            print(f"  Main process PID: {os.getpid()}", file=sys.stderr)
            print(f"  Tickers: {', '.join(tickers_to_refresh[:10])}{'...' if len(tickers_to_refresh) > 10 else ''}", file=sys.stderr)
            print(f"  Log file: {log_file.name}", file=sys.stderr)
            print(f"\nTo check background processes: ps auxww | grep -E 'options_analyzer|{process.pid}'", file=sys.stderr)
            print(f"To view logs: tail -f {log_file.name}", file=sys.stderr)
            print("", file=sys.stderr)
            analyzer._log("INFO", f"Background refresh process started (PID: {process.pid}). Main process continuing with existing results.")
        else:
            # Process exited immediately, check the log
            print(f"\nBackground refresh process exited immediately (exit code: {process.returncode})", file=sys.stderr)
            print(f"Check log file for details: {log_file.name}", file=sys.stderr)
            try:
                with open(log_file.name, 'r') as f:
                    log_content = f.read()
                    if log_content:
                        print(f"Log content:\n{log_content}", file=sys.stderr)
            except:
                pass
            analyzer._log("WARNING", f"Background refresh process exited immediately (PID: {process.pid}, exit code: {process.returncode})")
    except Exception as e:
        print(f"ERROR: Failed to start background refresh process: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc(file=sys.stderr)
        analyzer._log("ERROR", f"Failed to start background refresh process: {e}")


def _print_statistics(analyzer: OptionsAnalyzer, args):
    """Print multiprocess and cache statistics."""
    # Print multiprocess statistics if using multiprocessing
    if args.max_workers > 1 and hasattr(analyzer.db, 'print_process_statistics'):
        # Use log_level to determine if we should print (INFO or lower)
        quiet = not analyzer._should_log("INFO")
        analyzer.db.print_process_statistics(quiet=quiet)
    
    # Print cache statistics (always print at INFO level or lower)
    if analyzer._should_log("INFO") and hasattr(analyzer.db, 'get_cache_statistics'):
        cache_stats = analyzer.db.get_cache_statistics()
        print("\n=== Cache Statistics ===", file=sys.stderr)
        if cache_stats.get('enabled', False):
            print(f"Cache Status: ENABLED", file=sys.stderr)
            print(f"Total Requests: {cache_stats.get('total_requests', 0)}", file=sys.stderr)
            print(f"Cache Hits: {cache_stats.get('hits', 0)}", file=sys.stderr)
            print(f"Cache Misses: {cache_stats.get('misses', 0)}", file=sys.stderr)
            hit_rate = cache_stats.get('hit_rate', 0.0)
            print(f"Hit Rate: {hit_rate:.2%}", file=sys.stderr)
            negative_hits = cache_stats.get('negative_hits', 0)
            negative_sets = cache_stats.get('negative_sets', 0)
            print(f"Negative Cache Hits: {negative_hits}", file=sys.stderr)
            print(f"Negative Cache Sets: {negative_sets}", file=sys.stderr)
            print(f"Cache Sets: {cache_stats.get('sets', 0)}", file=sys.stderr)
            print(f"Cache Invalidations: {cache_stats.get('invalidations', 0)}", file=sys.stderr)
            print(f"Cache Errors: {cache_stats.get('errors', 0)}", file=sys.stderr)
        else:
            print(f"Cache Status: DISABLED", file=sys.stderr)
        # Database query statistics (if available)
        db_query_count = cache_stats.get('db_query_count')
        if db_query_count is not None:
            print(f"\n=== Database Query Statistics ===", file=sys.stderr)
            print(f"Total Database Queries: {db_query_count}", file=sys.stderr)
            print("===================================\n", file=sys.stderr)
        else:
            print("===================================\n", file=sys.stderr)


def _parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments for the options analyzer."""
    parser = argparse.ArgumentParser(
        description="Analyze covered call opportunities across all strike prices and tickers.",
        epilog=ARGUMENT_EXAMPLES
    )
    
    # Add symbol input arguments using common library
    add_symbol_arguments(parser, required=True, allow_positional=False)
    
    # Add all argument groups
    add_database_arguments(parser)
    add_analysis_arguments(parser)
    add_spread_arguments(parser)
    add_performance_arguments(parser)
    add_filter_arguments(parser)
    add_output_arguments(parser)
    
    # If help is requested, print and exit early to avoid running any analysis code
    if any(flag in sys.argv for flag in ("-h", "--help")):
        parser.print_help()
        sys.exit(0)

    args = parser.parse_args()
    log_parsed_arguments(args)
    
    return args


def _build_analysis_args(args: argparse.Namespace, symbols_list: List[str], filters: List) -> dict:
    """Build arguments dictionary for analyze_options from parsed args."""
    return {
        'tickers': symbols_list,
        'option_type': getattr(args, 'option_type', 'call'),
        'days_to_expiry': args.days,
        'min_volume': args.min_volume,
        'max_days': args.max_days,
        'min_days': getattr(args, 'min_days', None),
        'min_premium': args.min_premium,
        'position_size': args.position_size,
        'filters': filters,
        'filter_logic': args.filter_logic,
        'use_market_time': not args.no_market_time,
        'start_date': args.start_date,
        'end_date': args.end_date,
        'max_concurrent': getattr(args, 'max_concurrent', 10),  # Default to 10 if not specified
        'batch_size': args.batch_size,
        'timestamp_lookback_days': args.timestamp_lookback_days,
        'max_workers': args.max_workers,
        'spread_mode': args.spread,
        'spread_strike_tolerance': args.spread_strike_tolerance,
        'spread_long_days': args.spread_long_days,
        'spread_long_days_tolerance': args.spread_long_days_tolerance,
        'spread_long_min_days': args.spread_long_min_days,
        'min_write_timestamp': args.min_write_timestamp,
        'sensible_price': getattr(args, 'sensible_price', 0.01),
        'max_bid_ask_spread': getattr(args, 'max_bid_ask_spread', 2.0),
        'max_bid_ask_spread_long': getattr(args, 'max_bid_ask_spread_long', 2.0)
    }


def _print_statistics(analyzer: 'OptionsAnalyzer', args: argparse.Namespace) -> None:
    """Print multiprocess and cache statistics."""
    # Print multiprocess statistics if using multiprocessing
    if args.max_workers > 1 and hasattr(analyzer.db, 'print_process_statistics'):
        quiet = not analyzer._should_log("INFO")
        analyzer.db.print_process_statistics(quiet=quiet)
    
    # Print cache statistics (always print at INFO level or lower)
    if analyzer._should_log("INFO") and hasattr(analyzer.db, 'get_cache_statistics'):
        cache_stats = analyzer.db.get_cache_statistics()
        print("\n=== Cache Statistics ===", file=sys.stderr)
        if cache_stats.get('enabled', False):
            print(f"Cache Status: ENABLED", file=sys.stderr)
            print(f"Total Requests: {cache_stats.get('total_requests', 0)}", file=sys.stderr)
            print(f"Cache Hits: {cache_stats.get('hits', 0)}", file=sys.stderr)
            print(f"Cache Misses: {cache_stats.get('misses', 0)}", file=sys.stderr)
            hit_rate = cache_stats.get('hit_rate', 0.0)
            print(f"Hit Rate: {hit_rate:.2%}", file=sys.stderr)
            negative_hits = cache_stats.get('negative_hits', 0)
            negative_sets = cache_stats.get('negative_sets', 0)
            print(f"Negative Cache Hits: {negative_hits}", file=sys.stderr)
            print(f"Negative Cache Sets: {negative_sets}", file=sys.stderr)
            print(f"Cache Sets: {cache_stats.get('sets', 0)}", file=sys.stderr)
            print(f"Cache Invalidations: {cache_stats.get('invalidations', 0)}", file=sys.stderr)
            print(f"Cache Errors: {cache_stats.get('errors', 0)}", file=sys.stderr)
        else:
            print(f"Cache Status: DISABLED", file=sys.stderr)
        # Database query statistics (if available)
        db_query_count = cache_stats.get('db_query_count')
        if db_query_count is not None:
            print(f"\n=== Database Query Statistics ===", file=sys.stderr)
            print(f"Total Database Queries: {db_query_count}", file=sys.stderr)
            print("===================================\n", file=sys.stderr)
        else:
            print("===================================\n", file=sys.stderr)
    
    # Also check for args.stats flag (legacy support)
    if args.stats and hasattr(analyzer.db, 'get_cache_stats'):
        stats = analyzer.db.get_cache_stats()
        if stats:
            print("\n===================================", file=sys.stderr)
            print("Cache Statistics:", file=sys.stderr)
            print("===================================", file=sys.stderr)
            for key, value in stats.items():
                print(f"{key}: {value}", file=sys.stderr)
            print("===================================\n", file=sys.stderr)


async def main():
    """Main function to run the options analyzer."""
    args = _parse_arguments()
    
    # Initialize analyzer
    enable_cache = not args.no_cache
    redis_url = os.getenv('REDIS_URL', 'redis://localhost:6379/0') if enable_cache else None
    log_level = args.log_level if hasattr(args, 'log_level') else ("DEBUG" if args.debug else "INFO")
    
    analyzer = OptionsAnalyzer(args.db_conn, log_level=log_level, debug=args.debug, enable_cache=enable_cache, redis_url=redis_url)
    await analyzer.initialize()
    
    # Use async context manager to ensure close() is always called
    async with analyzer.db:
        # Get symbols list using common library
        quiet = not analyzer._should_log("INFO")
        symbols_list = await fetch_lists_data(args, quiet)
        if not symbols_list:
            print("No symbols specified or found. Exiting.", file=sys.stderr)
            sys.exit(1)
        
        analyzer._log("INFO", f"Analyzing {len(symbols_list)} tickers...")
        
        if args.debug:
            print(f"DEBUG: Symbols list: {symbols_list[:10]}{'...' if len(symbols_list) > 10 else ''}", file=sys.stderr)
        
        # Get financial information
        max_workers = getattr(args, 'max_workers', 4)
        financial_data = await analyzer.get_financial_info(symbols_list, max_workers=max_workers)
        
        # Parse filters
        filters = []
        if hasattr(args, 'filter') and args.filter:
            try:
                normalized_filters = [' '.join(f.split()) for f in args.filter]
                filters = FilterParser.parse_filters(normalized_filters)
                if filters:
                    analyzer._log("INFO", f"Applied {len(filters)} filter(s) with {args.filter_logic} logic:")
                    for i, f in enumerate(filters, 1):
                        analyzer._log("INFO", f"  {i}. {f}")
            except Exception as e:
                print(f"Error parsing filters: {e}", file=sys.stderr)
                sys.exit(1)
        
        # Analyze options
        analysis_args = _build_analysis_args(args, symbols_list, filters)
        df = await analyzer.analyze_options(**analysis_args)
        
        if df.empty:
            analyzer._log("INFO", "DataFrame is empty after analysis. Check debug output above for details.")
            print("No options data found matching the criteria.")
            return
        
        # Get Redis client for refresh deduplication (if available)
        redis_url = os.getenv('REDIS_URL', 'redis://localhost:6379/0') if enable_cache else None
        redis_client = None
        if redis_url and REDIS_AVAILABLE:
            redis_client = _get_redis_client_for_refresh(redis_url)
        
        # Use the analyzer's instance-level timestamp cache
        timestamp_cache = analyzer._timestamp_cache
        
        # Refresh results if requested and market is open
        if args.refresh_results is not None:
            refresh_threshold_seconds = args.refresh_results
            df = await _run_refresh_analysis(
                analyzer, args, df, filters, refresh_threshold_seconds, redis_client, timestamp_cache
            )
        elif args.refresh_results_background is not None:
            refresh_threshold_seconds = args.refresh_results_background
            await _run_background_refresh(
                analyzer, args, df, filters, refresh_threshold_seconds, redis_client, timestamp_cache, all_tickers=symbols_list
            )
        
        # Print statistics
        _print_statistics(analyzer, args)
        
        # Determine output format and file
        output_format = 'table'
        output_file = None
        
        if args.output.lower() == 'csv':
            output_format = 'csv'
        elif args.output.lower() != 'table':
            output_file = args.output
            if args.output.endswith('.csv'):
                output_format = 'csv'
            else:
                output_format = 'table'
        
        # Normalize sort input
        import re as _re
        sort_arg = _re.sub(r"\s+", "", args.sort) if hasattr(args, 'sort') and args.sort else None
        
        # If in spread mode and user didn't specify a sort, default to net_daily_premium
        if args.spread and args.sort == 'daily_premium':
            sort_arg = 'net_daily_premium'

        # Parse CSV columns if specified
        csv_columns = None
        if hasattr(args, 'csv_columns') and args.csv_columns:
            csv_columns = [col.strip() for col in args.csv_columns.split(',')]
        else:
            csv_columns = None

        # Format and display results
        result = analyzer.format_output(
            df=df,
            financial_data=financial_data,
            output_format=output_format,
            group_by=args.group_by,
            output_file=output_file,
            sort_by=sort_arg,
            filters=filters,
            filter_logic=args.filter_logic,
            csv_delimiter=args.csv_delimiter,
            csv_quoting=args.csv_quoting,
            csv_columns=csv_columns,
            top_n=args.top_n,
            spread_mode=args.spread
        )
        
        if analyzer._should_log("INFO") or output_file is None:
            print(result)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nOperation cancelled by user.")
    except Exception as e:
        print(f"\nAn unexpected error occurred: {e}", file=sys.stderr)
        sys.exit(1)

