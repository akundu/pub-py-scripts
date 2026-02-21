#!/usr/bin/env python3
"""
Historical Stock and Options Data Fetcher for a Specific Date using Polygon API

This program fetches the stock price and all active options contracts for a given
symbol on a specific historical date.

Usage:
    export POLYGON_API_KEY=YOUR_API_KEY
    python historical_stock_options.py AAPL 2024-06-01
    python historical_stock_options.py --symbols AAPL MSFT GOOGL --date 2024-06-01
    python historical_stock_options.py --symbols-list symbols.yaml --date 2024-06-01
    python historical_stock_options.py --types sp-500 --date 2024-06-01
"""

import os
import sys
import argparse
import asyncio
import time
import csv
import signal
import pandas as pd
import yaml
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Any
from tabulate import tabulate
from pathlib import Path
from zoneinfo import ZoneInfo

# Ensure project root is on sys.path so `common` can be imported when running from any cwd
CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Import common symbol loading functions
from common.symbol_loader import add_symbol_arguments, fetch_lists_data
from common.stock_db import get_stock_db
from common.common import (
    run_iteration_in_subprocess,
    check_tickers_for_refresh,
    fetch_latest_option_timestamp_standalone,
    get_redis_client_for_refresh,
    check_redis_refresh_pending,
    set_redis_last_write_timestamp
)
from common.market_hours import is_market_hours as common_is_market_hours, compute_market_transition_times as common_compute_market_transition_times

try:
    from polygon import RESTClient
    POLYGON_AVAILABLE = True
except ImportError:
    POLYGON_AVAILABLE = False

def _convert_index_symbol_for_polygon(symbol: str) -> tuple[str, bool]:
    """
    Convert index symbol format (I:SPX) to Polygon API format (SPX).
    
    For Polygon options API, index symbols need to be in format "SPX" not "I:SPX".
    This function extracts the base symbol for API calls while preserving the original
    for display/logging purposes.
    
    Args:
        symbol: Input symbol (e.g., "I:SPX", "SPX", "AAPL")
        
    Returns:
        tuple: (polygon_symbol, is_index)
            - polygon_symbol: Symbol to use for Polygon API calls (e.g., "SPX" for "I:SPX")
            - is_index: True if this is an index symbol, False otherwise
    """
    # Handle case-insensitive matching for "I:" prefix
    symbol_upper = symbol.upper()
    if symbol_upper.startswith("I:"):
        # Extract base symbol (e.g., "I:SPX" -> "SPX", "i:spx" -> "SPX")
        polygon_symbol = symbol[2:].upper()
        return polygon_symbol, True
    return symbol.upper(), False


class HistoricalDataFetcher:
    """Fetches historical stock and options data from Polygon.io."""
    CACHE_DURATION_MINUTES = {
        'market_open': 20,
        'market_closed': 360,
        'post_market': 60,
    }

    @staticmethod
    def _compute_market_transition_times(now_utc: datetime, tz_name: str = "America/New_York") -> tuple[float | None, float | None]:
        """Compute time in seconds to next regular market open and close from now.

        - Market hours assumed: 09:30–16:00 local (Mon–Fri). Holidays not considered.
        - Returns (seconds_to_open, seconds_to_close). Either may be None if not applicable.
        """
        try:
            tz = ZoneInfo(tz_name)
        except Exception:
            tz = ZoneInfo("America/New_York")

        now_local = now_utc.astimezone(tz)

        # Build today's open/close
        today_open = now_local.replace(hour=9, minute=30, second=0, microsecond=0)
        today_close = now_local.replace(hour=16, minute=0, second=0, microsecond=0)

        # Helper to advance to next weekday (Mon-Fri)
        def next_weekday(dt: datetime) -> datetime:
            d = dt
            while d.weekday() >= 5:  # 5=Sat, 6=Sun
                d = d + timedelta(days=1)
            return d

        seconds_to_open: float | None = None
        seconds_to_close: float | None = None

        # Compute seconds to next open
        if now_local < today_open:
            seconds_to_open = (today_open - now_local).total_seconds()
        else:
            # Find next trading day's open
            next_day = next_weekday((now_local + timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0))
            next_open = next_day.replace(hour=9, minute=30, second=0, microsecond=0)
            seconds_to_open = (next_open - now_local).total_seconds()

        # Compute seconds to close if currently before today's close and it's a weekday
        if now_local.weekday() < 5 and now_local < today_close:
            seconds_to_close = (today_close - now_local).total_seconds()
        else:
            seconds_to_close = None

        return (seconds_to_open, seconds_to_close)

    def __init__(self, api_key: str, data_dir: str = "data", verbose: bool = False, snapshot_max_concurrent: int = 0, refresh_threshold_seconds: int | None = None):
        if not api_key:
            raise ValueError("Polygon API key is required.")
        self.client = RESTClient(api_key)
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)
        self.quiet = not verbose  # quiet is inverse of verbose
        # 0 disables per-contract snapshot concurrency; otherwise bounded parallelism
        self.snapshot_max_concurrent = max(0, int(snapshot_max_concurrent))
        self.refresh_threshold_seconds = refresh_threshold_seconds

    @staticmethod
    def _is_market_open(dt: datetime | None = None) -> bool:
        # Delegate to shared implementation (America/New_York)
        return common_is_market_hours(dt, "America/New_York")

    def _get_cache_duration_minutes(self) -> int:
        """Get cache duration in minutes based on market status."""
        now_utc = datetime.now(timezone.utc)
        
        if self._is_market_open(now_utc):
            return HistoricalDataFetcher.CACHE_DURATION_MINUTES['market_open']
        
        # Market is closed: choose the smaller of (time until next open) vs (closed cadence)
        seconds_to_open, _seconds_to_close = common_compute_market_transition_times(now_utc, "America/New_York")
        closed_cadence_secs = HistoricalDataFetcher.CACHE_DURATION_MINUTES['market_closed'] * 60
        if seconds_to_open is None:
            sleep_secs = closed_cadence_secs
        else:
            sleep_secs = min(max(seconds_to_open, 0), closed_cadence_secs)
        # Return minutes, rounded up to avoid waking too early, minimum 1 minute
        minutes = max(1, int((sleep_secs + 59) // 60))
        return minutes

    def _get_csv_path(self, symbol: str, expiration_date: str) -> Path:
        """Get the CSV file path for a specific symbol and expiration date."""
        # Normalize and guard against empty symbol resulting in flat writes
        norm_symbol = (symbol or "").strip().upper()
        if not norm_symbol:
            norm_symbol = "UNKNOWN"
        options_dir = self.data_dir / "options"
        symbol_dir = options_dir / norm_symbol
        symbol_dir.mkdir(parents=True, exist_ok=True)
        return symbol_dir / f"{expiration_date}.csv"

    def _should_fetch_fresh_data(self, csv_path: Path) -> bool:
        """Check if we should fetch fresh data based on cache duration or refresh threshold."""
        if not csv_path.exists():
            return True
        
        file_mtime = datetime.fromtimestamp(csv_path.stat().st_mtime)
        age_seconds = (datetime.now() - file_mtime).total_seconds()
        
        # If refresh_threshold_seconds is set, use it; otherwise use cache duration
        if self.refresh_threshold_seconds is not None:
            return age_seconds > self.refresh_threshold_seconds
        
        cache_duration = self._get_cache_duration_minutes()
        return age_seconds > (cache_duration * 60)

    def _save_options_to_csv(self, symbol: str, options_data: Dict[str, Any]) -> None:
        """Save options data to CSV files organized by expiration date."""
        # Handle both data structures: direct contracts or nested in 'data'
        contracts = options_data.get('contracts', [])
        if not contracts and 'data' in options_data:
            contracts = options_data['data'].get('contracts', [])
        
        if not contracts:
            return
        current_time = datetime.now().isoformat()
        
        # Group contracts by expiration date
        by_expiration = {}
        for contract in contracts:
            exp_date = contract.get('expiration', 'unknown')
            if exp_date not in by_expiration:
                by_expiration[exp_date] = []
            by_expiration[exp_date].append(contract)
        
        # Save each expiration date to its own CSV file
        for exp_date, contracts_list in by_expiration.items():
            if exp_date == 'unknown':
                continue
                
            csv_path = self._get_csv_path(symbol, exp_date)
            # Ensure directory exists regardless of prior steps
            csv_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Prepare data for CSV
            csv_data = []
            for contract in contracts_list:
                csv_data.append({
                    'timestamp': current_time,
                    'ticker': contract.get('ticker', ''),
                    'type': contract.get('type', ''),
                    'strike': contract.get('strike', ''),
                    'expiration': contract.get('expiration', ''),
                    'bid': contract.get('bid', ''),
                    'ask': contract.get('ask', ''),
                    'day_close': contract.get('day_close', ''),
                    'fmv': contract.get('fmv', ''),
                    'delta': contract.get('delta', ''),
                    'gamma': contract.get('gamma', ''),
                    'theta': contract.get('theta', ''),
                    'vega': contract.get('vega', ''),
                    'implied_volatility': contract.get('implied_volatility', ''),
                    'volume': contract.get('volume', ''),
                })
            
            # Write to CSV (append mode)
            file_exists = csv_path.exists()
            with open(csv_path, 'a', newline='', encoding='utf-8') as csvfile:
                fieldnames = ['timestamp', 'ticker', 'type', 'strike', 'expiration', 
                             'bid', 'ask', 'day_close', 'fmv', 'delta', 'gamma', 'theta', 'vega', 'implied_volatility', 'volume']
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                
                if not file_exists:
                    writer.writeheader()
                writer.writerows(csv_data)
            
            if not self.quiet:
                print(f"Saved {len(csv_data)} contracts for {symbol} expiration {exp_date} to {csv_path}")
            else:
                # In quiet mode, still emit a minimal trace for debugging wrong path issues
                print(str(csv_path))

    def _load_options_from_csv(self, symbol: str, expiration_date: str) -> List[Dict[str, Any]]:
        """Load the most recent options data from CSV file."""
        csv_path = self._get_csv_path(symbol, expiration_date)
        
        if not csv_path.exists():
            return []
        
        try:
            df = pd.read_csv(csv_path)
            if df.empty:
                return []
            
            # Get the most recent data (last timestamp)
            latest_timestamp = df['timestamp'].max()
            latest_data = df[df['timestamp'] == latest_timestamp]
            
            # Convert back to list of dicts
            contracts = []
            for _, row in latest_data.iterrows():
                contract = {
                    'ticker': row['ticker'],
                    'type': row['type'],
                    'strike': row['strike'],
                    'expiration': row['expiration'],
                    'bid': row['bid'] if pd.notna(row['bid']) else None,
                    'ask': row['ask'] if pd.notna(row['ask']) else None,
                    'day_close': row['day_close'] if pd.notna(row['day_close']) else None,
                    'fmv': row['fmv'] if pd.notna(row['fmv']) else None,
                    'delta': row['delta'] if pd.notna(row['delta']) else None,
                    'gamma': row['gamma'] if pd.notna(row['gamma']) else None,
                    'theta': row['theta'] if pd.notna(row['theta']) else None,
                    'vega': row['vega'] if pd.notna(row['vega']) else None,
                    'implied_volatility': row['implied_volatility'] if 'implied_volatility' in row and pd.notna(row['implied_volatility']) else None,
                    'volume': row['volume'] if 'volume' in row and pd.notna(row['volume']) else None,
                    'last_quote_timestamp': row['last_quote_timestamp'] if 'last_quote_timestamp' in row and pd.notna(row['last_quote_timestamp']) else None,
                }
                contracts.append(contract)
            
            return contracts
        except Exception as e:
            print(f"Error loading CSV data: {e}")
            return []

    def _handle_api_error(self, error: Exception, data_type: str, symbol: str | None = None) -> Dict[str, Any]:
        """Handles API errors gracefully."""
        if symbol:
            error_msg = f"Error fetching {data_type} for {symbol}: {str(error)}"
        else:
            error_msg = f"Error fetching {data_type}: {str(error)}"
        print(f"Warning: {error_msg}", file=sys.stderr)
        return {"error": error_msg, "data": None}

    async def get_stock_price_for_date(self, symbol: str, target_date_str: str) -> Dict[str, Any]:
        """
        Fetches the historical stock price (OHLCV) for the requested date.
        If the date is a non-trading day, it finds the most recent previous trading day.
        """
        target_date_dt = datetime.strptime(target_date_str, '%Y-%m-%d')
        
        # Convert index symbol for Polygon API (I:SPX -> SPX)
        polygon_symbol, is_index = _convert_index_symbol_for_polygon(symbol)
        
        # We look at a 7-day window ending on the target date to ensure we find a trading day.
        search_start = (target_date_dt - timedelta(days=7)).strftime('%Y-%m-%d')
        search_end = target_date_str
        
        if not self.quiet:
            if is_index:
                print(f"Fetching historical price for {symbol} (index, using Polygon symbol {polygon_symbol}) on or before {target_date_str}...", flush=True)
            else:
                print(f"Fetching historical price for {symbol} on or before {target_date_str}...", flush=True)

        try:
            # Sort descending and limit to 1 to get the latest trading day on or before the target date
            aggs = self.client.get_aggs(
                ticker=polygon_symbol,
                multiplier=1,
                timespan="day",
                from_=search_start,
                to=search_end,
                adjusted=True,
                sort="desc", 
                limit=1
            )
            
            if not aggs:
                return self._handle_api_error(Exception(f"No trading data found on or before {target_date_str} in the last 7 days"), "stock price", symbol)

            # The first (and only) result is the one we want
            bar = aggs[0]
            trading_date = datetime.fromtimestamp(bar.timestamp / 1000).strftime('%Y-%m-%d')
            
            if not self.quiet:
                print(f"Found data for trading day: {trading_date} (requested: {target_date_str})")
            
            return {"success": True, "data": {
                'target_date': target_date_str,
                'trading_date': trading_date,
                'open': bar.open,
                'high': bar.high,
                'low': bar.low,
                'close': bar.close,
                'volume': bar.volume,
            }}

        except Exception as e:
            return self._handle_api_error(e, "stock price", symbol)

    async def get_active_options_for_date(
        self,
        symbol: str,
        target_date_str: str,
        option_type: str,
        stock_close_price: float | None,
        strike_range_percent: int | None,
        max_days_to_expiry: int | None,
        include_expired: bool,
        use_cache: bool = False,
        save_to_csv: bool = False,
        use_db: bool = False,
        db_conn: str | None = None,
        force_fresh: bool = False,
        enable_cache: bool = True,
        redis_url: str | None = None,
        db_timeout: float = 60.0,
        db_only: bool = False
    ) -> Dict[str, Any]:
        """
        Fetches all options contracts that were active on a specific date and gets their
        current snapshot data (including prices and Greeks).
        Applies filters for option type and strike price range.
        """
        # Convert index symbol for Polygon API (I:SPX -> SPX)
        polygon_symbol, is_index = _convert_index_symbol_for_polygon(symbol)
        
        options_data = {"contracts": []}
        target_date_dt = datetime.strptime(target_date_str, '%Y-%m-%d')
        
        if not self.quiet:
            if is_index:
                print(f"Fetching options for {symbol} (index, using Polygon symbol {polygon_symbol}) expiring around {target_date_str}...")
            else:
                print(f"Fetching options for {symbol} expiring around {target_date_str}...")
        overall_start_time = time.time()
        
        # 1) Prefer DB when enabled: load latest options from database and return if available (unless force_fresh)
        if use_db and db_conn and not force_fresh:
            try:
                # Determine database type based on connection string
                db_type = 'questdb'  # default
                db_config = db_conn
                if db_conn.startswith('http://') or db_conn.startswith('https://'):
                    db_type = 'remote'
                    # StockDBClient expects "host:port" format, not "http://host:port"
                    db_config = db_conn.replace('http://', '').replace('https://', '')
                elif db_conn.startswith('postgresql://'):
                    db_type = 'postgresql'
                
                # Use db_timeout parameter for remote connections (default 60 seconds)
                db = get_stock_db(db_type, db_config=db_config, enable_cache=enable_cache, redis_url=redis_url, timeout=db_timeout if db_type == 'remote' else None)
                # Use polygon_symbol for database queries (I:SPX -> SPX)
                latest_df = await db.get_latest_options_data(ticker=polygon_symbol)
                if latest_df is not None and not latest_df.empty:
                    # Map DB dataframe to contracts list in script format
                    contracts_from_db = []
                    for _, row in latest_df.iterrows():
                        contracts_from_db.append({
                            'ticker': row.get('option_ticker'),
                            'type': (row.get('option_type') or '').lower(),
                            'strike': row.get('strike_price'),
                            'expiration': row.get('expiration_date'),
                            'bid': row.get('bid'),
                            'ask': row.get('ask'),
                            'day_close': row.get('day_close'),
                            'fmv': row.get('fmv'),
                            'delta': row.get('delta'),
                            'gamma': row.get('gamma'),
                            'theta': row.get('theta'),
                            'vega': row.get('vega'),
                            'rho': row.get('rho'),
                            'implied_volatility': row.get('implied_volatility'),
                            'volume': row.get('volume'),
                            'open_interest': row.get('open_interest'),
                            'last_quote_timestamp': row.get('last_quote_timestamp'),
                        })

                    # Apply filters locally to DB-loaded data to match CLI options
                    filtered_contracts = contracts_from_db
                    # 1) option type
                    if option_type != 'all':
                        filtered_contracts = [c for c in filtered_contracts if (c.get('type') or '').lower() == option_type]
                    # 2) strike range
                    if strike_range_percent is not None and stock_close_price is not None:
                        contracts_before_strike_filter = len(filtered_contracts)
                        min_strike = stock_close_price * (1 - strike_range_percent / 100)
                        max_strike = stock_close_price * (1 + strike_range_percent / 100)
                        filtered_contracts = [c for c in filtered_contracts if min_strike <= (c.get('strike') or -1) <= max_strike]
                        contracts_filtered_out = contracts_before_strike_filter - len(filtered_contracts)
                        if contracts_filtered_out > 0:
                            print(f"Filtered out {contracts_filtered_out} options that did not meet the {strike_range_percent}% strike range limit (kept {len(filtered_contracts)} out of {contracts_before_strike_filter})")
                    # 3) expiration date window (±max_days_to_expiry around target_date)
                    if max_days_to_expiry is not None:
                        min_date = (target_date_dt - timedelta(days=max_days_to_expiry)).date()
                        max_date = (target_date_dt + timedelta(days=max_days_to_expiry)).date()

                        def _parse_expiration(exp_value):
                            # Handles date strings like 'YYYY-MM-DD' or datetime/date objects
                            try:
                                if exp_value is None:
                                    return None
                                if hasattr(exp_value, 'date'):
                                    # datetime or pandas Timestamp
                                    return exp_value.date() if hasattr(exp_value, 'hour') else exp_value
                                if isinstance(exp_value, str):
                                    return datetime.strptime(exp_value[:10], '%Y-%m-%d').date()
                            except Exception:
                                return None
                            return None

                        filtered_contracts = [
                            c for c in filtered_contracts
                            if (lambda d: d is not None and min_date <= d <= max_date)(_parse_expiration(c.get('expiration')))
                        ]

                    if filtered_contracts:
                        options_data['contracts'] = filtered_contracts
                        # When loading from DB, we don't have the original downloaded count, so use filtered count
                        options_data['total_downloaded'] = len(filtered_contracts)
                        return {"success": True, "data": options_data}
            except Exception as _db_e:
                # Fall back to CSV/API silently on DB issues
                pass

        # 2) Check if we should use cached CSV data (only when explicitly enabled and not force_fresh)
        if use_cache and not force_fresh:
            # Try to load from cache for each potential expiration date
            cache_hit = False
            if max_days_to_expiry is not None:
                # Check cache for dates within the range
                for days_offset in range(-max_days_to_expiry, max_days_to_expiry + 1):
                    check_date = target_date_dt + timedelta(days=days_offset)
                    exp_date_str = check_date.strftime('%Y-%m-%d')
                    csv_path = self._get_csv_path(symbol, exp_date_str)
                    
                    if not self._should_fetch_fresh_data(csv_path):
                        cached_contracts = self._load_options_from_csv(symbol, exp_date_str)
                        if cached_contracts:
                            options_data["contracts"].extend(cached_contracts)
                            cache_hit = True
                            if not self.quiet:
                                print(f"Loaded {len(cached_contracts)} contracts from cache for {exp_date_str}")
            else:
                # Check cache for the target date
                csv_path = self._get_csv_path(symbol, target_date_str)
                if not self._should_fetch_fresh_data(csv_path):
                    cached_contracts = self._load_options_from_csv(symbol, target_date_str)
                    if cached_contracts:
                        options_data["contracts"] = cached_contracts
                        cache_hit = True
                        if not self.quiet:
                            print(f"Loaded {len(cached_contracts)} contracts from cache for {target_date_str}")
            
            if cache_hit:
                if not self.quiet:
                    print(f"Using cached data. Cache duration: {self._get_cache_duration_minutes()} minutes")
                return {"success": True, "data": options_data}
        
        # If db_only is enabled and we didn't find data in DB or cache, return error
        if db_only:
            error_msg = f"No data found in database or cache for {symbol} on {target_date_str}. Use --force-fresh to fetch from API."
            if not self.quiet:
                print(f"Error: {error_msg}", file=sys.stderr)
            return {"success": False, "error": error_msg, "data": options_data}
        
        try:
            # --- Build API query parameters for efficient filtering ---
            # Default: show options active on or after the target date
            expiration_date_gte = target_date_str
            expiration_date_lte = None

            # If max_days_to_expiry is provided, create a +/- window around the target date
            if max_days_to_expiry is not None:
                min_date = target_date_dt - timedelta(days=max_days_to_expiry)
                expiration_date_gte = min_date.strftime('%Y-%m-%d')
                
                max_date = target_date_dt + timedelta(days=max_days_to_expiry)
                expiration_date_lte = max_date.strftime('%Y-%m-%d')
                if not self.quiet:
                    print(f"  ...searching for expirations between {expiration_date_gte} and {expiration_date_lte}", flush=True)
            else:
                if not self.quiet:
                    print(f"  ...searching for expirations on or after {expiration_date_gte}", flush=True)


            # Fetch active contracts that meet the date criteria
            if not self.quiet:
                print("  ...fetching active contracts within date range...", flush=True)
            fetch_active_start = time.time()
            active_contracts_generator = self.client.list_options_contracts(
                underlying_ticker=polygon_symbol,
                expiration_date_gte=expiration_date_gte,
                expiration_date_lte=expiration_date_lte,
                limit=1000,
                expired=False
            )
            
            # --- Manual iteration for debugging ---
            all_contracts = []
            page_num = 1
            if not self.quiet:
                print("  Iterating through active contract pages from API...", flush=True)
            page_start_time = time.time()
            try:
                for contract in active_contracts_generator:
                    all_contracts.append(contract)
                    if len(all_contracts) % 250 == 0 and not self.quiet: # Print progress every 250 contracts
                        page_end_time = time.time()
                        print(f"    ... fetched page {page_num} ({len(all_contracts)} total), took {page_end_time - page_start_time:.2f}s", flush=True)
                        page_start_time = time.time()
                        page_num += 1
            except Exception as e:
                print(f"ERROR: Exception during active contract iteration: {e}", file=sys.stderr)
            # --- End Manual iteration ---

            fetch_active_end = time.time()
            if not self.quiet:
                print(f"  [TIMER] Finished iterating active contracts. Total took {fetch_active_end - fetch_active_start:.2f} seconds.")


            # Conditionally fetch expired contracts if requested
            if include_expired:
                if not self.quiet:
                    print("  ...fetching EXPIRED contracts (this may be slow)...", flush=True)
                fetch_expired_start = time.time()
                expired_contracts_generator = self.client.list_options_contracts(
                    underlying_ticker=polygon_symbol,
                    expiration_date_gte=expiration_date_gte,
                    expiration_date_lte=expiration_date_lte,
                    limit=1000,
                    expired=True
                )
                
                # Manual iteration for debugging
                expired_contracts_list = []
                page_num = 1
                if not self.quiet:
                    print("  Iterating through EXPIRED contract pages from API...", flush=True)
                page_start_time = time.time()
                try:
                    for contract in expired_contracts_generator:
                        expired_contracts_list.append(contract)
                        if len(expired_contracts_list) % 250 == 0 and not self.quiet:
                            page_end_time = time.time()
                            print(f"    ... fetched EXPIRED page {page_num} ({len(expired_contracts_list)} total), took {page_end_time - page_start_time:.2f}s", flush=True)
                            page_start_time = time.time()
                            page_num += 1
                except Exception as e:
                    print(f"ERROR: Exception during expired contract iteration: {e}", file=sys.stderr)
                all_contracts.extend(expired_contracts_list)
                # --- End Manual iteration ---

                fetch_expired_end = time.time()
                if not self.quiet:
                    print(f"  [TIMER] Finished iterating expired contracts. Total took {fetch_expired_end - fetch_expired_start:.2f} seconds.")

            if not self.quiet:
                print(f"Found a total of {len(all_contracts)} contracts from API.", flush=True)

            # Store total downloaded count for summary
            total_downloaded_from_api = len(all_contracts)

            # The API has already filtered by date, so all returned contracts are relevant.
            # No further local date filtering is required.
            
            active_contracts = all_contracts
            
            # --- Apply Filters ---
            filter_start = time.time()
            filtered_contracts = active_contracts

            # 1. Filter by option type
            if option_type != 'all':
                if not self.quiet:
                    print(f"Filtering for '{option_type}' options...")
                filtered_contracts = [
                    c for c in filtered_contracts
                    if getattr(c, 'contract_type', '').lower() == option_type
                ]

            # 2. Filter by strike price range
            contracts_before_strike_filter = len(filtered_contracts)
            if strike_range_percent is not None and stock_close_price is not None:
                if not self.quiet:
                    print(f"Filtering for strikes within {strike_range_percent}% of close price ${stock_close_price:.2f}...")
                min_strike = stock_close_price * (1 - strike_range_percent / 100)
                max_strike = stock_close_price * (1 + strike_range_percent / 100)
                
                filtered_contracts = [
                    c for c in filtered_contracts
                    if min_strike <= getattr(c, 'strike_price', -1) <= max_strike
                ]
                contracts_filtered_out = contracts_before_strike_filter - len(filtered_contracts)
                if contracts_filtered_out > 0:
                    print(f"Filtered out {contracts_filtered_out} options that did not meet the {strike_range_percent}% strike range limit (kept {len(filtered_contracts)} out of {contracts_before_strike_filter})")
            else:
                # No strike filtering - show summary
                if not self.quiet and contracts_before_strike_filter > 0:
                    print(f"No strike-range-percent constraint: keeping all {contracts_before_strike_filter} contracts (no filtering applied)")

            filter_end = time.time()
            if not self.quiet:
                print(f"  [TIMER] Local filtering took {filter_end - filter_start:.2f} seconds.")

            # Store total downloaded count in options_data for later use
            options_data["total_downloaded"] = total_downloaded_from_api
            
            if not self.quiet:
                total_downloaded = len(all_contracts)
                total_after_filtering = len(filtered_contracts)
                print(f"Summary: Downloaded {total_downloaded} contracts from API, {total_after_filtering} contracts after filtering ({total_downloaded - total_after_filtering} filtered out).", flush=True)
                print(f"Fetching snapshot data for all {len(filtered_contracts)} contracts...", flush=True)

            # --- Fetch snapshot data only for the contracts we will display ---
            processing_start = time.time()
            # Helper to fetch snapshot for a single contract
            def _fetch_snapshot(contract_obj, index_in_list: int) -> dict:
                contract_ticker_local = getattr(contract_obj, 'ticker', None)
                if not contract_ticker_local:
                    return {}
                details = {
                    'ticker': contract_ticker_local,
                    'type': getattr(contract_obj, 'contract_type', 'N/A'),
                    'strike': getattr(contract_obj, 'strike_price', 'N/A'),
                    'expiration': getattr(contract_obj, 'expiration_date', 'N/A'),
                    'bid': None,
                    'ask': None,
                    'day_close': None,
                    'fmv': None,
                    'last_quote_timestamp': None,
                    'implied_volatility': None,
                    'volume': None,
                }
                # Check if IV is available on the contract object itself
                for iv_field in ['implied_volatility', 'iv', 'implied_vol', 'volatility']:
                    iv_value = getattr(contract_obj, iv_field, None)
                    if iv_value is not None:
                        if hasattr(iv_value, 'value'):
                            details['implied_volatility'] = iv_value.value
                        elif isinstance(iv_value, (int, float)):
                            details['implied_volatility'] = iv_value
                        break
                try:
                    snapshot_local = self.client.get_snapshot_option(polygon_symbol, contract_ticker_local)
                    
                    # Also try to get quote separately using get_last_quote
                    quote_local = None
                    try:
                        quote_local = self.client.get_last_quote(ticker=contract_ticker_local)
                    except Exception:
                        # get_last_quote may not be available for all plans or contracts
                        pass
                    
                    if snapshot_local:
                        # First, try to get bid/ask from last_quote (primary source)
                        # Also check snapshot object itself for bid/ask as fallback
                        snapshot_bid = getattr(snapshot_local, 'bid', None)
                        snapshot_ask = getattr(snapshot_local, 'ask', None)
                        
                        # Check if last_quote exists and is not None
                        has_last_quote = hasattr(snapshot_local, 'last_quote') and snapshot_local.last_quote is not None
                        
                        if has_last_quote:
                            # Get bid and ask separately - try multiple possible field names
                            quote_bid = None
                            quote_ask = None
                            
                            # Try standard field names first
                            quote_bid = getattr(snapshot_local.last_quote, 'bid', None)
                            quote_ask = getattr(snapshot_local.last_quote, 'ask', None)
                            
                            # Try alternative field names if standard ones are None
                            if quote_bid is None:
                                quote_bid = getattr(snapshot_local.last_quote, 'bid_price', None)
                            if quote_ask is None:
                                quote_ask = getattr(snapshot_local.last_quote, 'ask_price', None)
                            
                            # Try even more alternative names (bp, ap, etc.)
                            if quote_bid is None:
                                quote_bid = getattr(snapshot_local.last_quote, 'bp', None)
                            if quote_ask is None:
                                quote_ask = getattr(snapshot_local.last_quote, 'ap', None)
                            
                            # Only set if they're actually provided (not None)
                            if quote_bid is not None:
                                details['bid'] = quote_bid
                            if quote_ask is not None:
                                details['ask'] = quote_ask
                            
                            # Extract timestamp from last_quote - try multiple possible field names
                            quote_timestamp = None
                            for field_name in ['t', 'timestamp', 'ts', 'time']:
                                quote_timestamp = getattr(snapshot_local.last_quote, field_name, None)
                                if quote_timestamp:
                                    break
                            
                            if quote_timestamp:
                                # Convert from nanoseconds to datetime
                                from datetime import datetime, timezone
                                details['last_quote_timestamp'] = datetime.fromtimestamp(quote_timestamp / 1000000000, tz=timezone.utc)
                            
                            # Fallback: if no timestamp found, use current time
                            if not details.get('last_quote_timestamp'):
                                from datetime import datetime, timezone
                                details['last_quote_timestamp'] = datetime.now(tz=timezone.utc)
                        
                        # Fallback: if last_quote didn't provide bid/ask, check details object
                        if details['bid'] is None or details['ask'] is None:
                            if hasattr(snapshot_local, 'details') and snapshot_local.details:
                                details_bid = getattr(snapshot_local.details, 'bid', None)
                                details_ask = getattr(snapshot_local.details, 'ask', None)
                                if details_bid is None:
                                    details_bid = getattr(snapshot_local.details, 'bid_price', None)
                                if details_ask is None:
                                    details_ask = getattr(snapshot_local.details, 'ask_price', None)
                                if details['bid'] is None and details_bid is not None:
                                    details['bid'] = details_bid
                                if details['ask'] is None and details_ask is not None:
                                    details['ask'] = details_ask
                        
                        # Fallback: if last_quote didn't provide bid/ask, try get_last_quote result
                        if (details['bid'] is None or details['ask'] is None) and quote_local:
                            quote_bid = getattr(quote_local, 'bid_price', None)
                            quote_ask = getattr(quote_local, 'ask_price', None)
                            if quote_bid is None:
                                quote_bid = getattr(quote_local, 'bid', None)
                            if quote_ask is None:
                                quote_ask = getattr(quote_local, 'ask', None)
                            if details['bid'] is None and quote_bid is not None:
                                details['bid'] = quote_bid
                            if details['ask'] is None and quote_ask is not None:
                                details['ask'] = quote_ask
                        
                        # Fallback: if last_quote didn't provide bid/ask, check snapshot object itself
                        if details['bid'] is None and snapshot_bid is not None:
                            details['bid'] = snapshot_bid
                        if details['ask'] is None and snapshot_ask is not None:
                            details['ask'] = snapshot_ask
                        
                        # Final fallback: if no timestamp was set anywhere, use current time
                        if not details['last_quote_timestamp']:
                            from datetime import datetime, timezone
                            details['last_quote_timestamp'] = datetime.now(tz=timezone.utc)
                        
                        # Get day_close and volume (but don't use as bid/ask fallback)
                        if hasattr(snapshot_local, 'day') and snapshot_local.day:
                            day_close = getattr(snapshot_local.day, 'close', None)
                            if day_close:
                                details['day_close'] = day_close
                            details['volume'] = getattr(snapshot_local.day, 'volume', None)
                            
                            # Check if day object has bid/ask
                            day_bid = getattr(snapshot_local.day, 'bid', None)
                            day_ask = getattr(snapshot_local.day, 'ask', None)
                            if day_bid is None:
                                day_bid = getattr(snapshot_local.day, 'bid_price', None)
                            if day_ask is None:
                                day_ask = getattr(snapshot_local.day, 'ask_price', None)
                            
                            # Use day bid/ask as fallback if we don't have them yet
                            if details['bid'] is None and day_bid is not None:
                                details['bid'] = day_bid
                            if details['ask'] is None and day_ask is not None:
                                details['ask'] = day_ask
                        
                        # Get FMV (but don't use as bid/ask fallback)
                        if hasattr(snapshot_local, 'fair_market_value') and snapshot_local.fair_market_value:
                            fmv = getattr(snapshot_local.fair_market_value, 'value', None)
                            if fmv:
                                details['fmv'] = fmv
                        
                        # Only use last_trade.price as fallback if BOTH bid and ask are missing
                        # And only set it to one of them, not both (to avoid identical values)
                        if hasattr(snapshot_local, 'last_trade') and snapshot_local.last_trade:
                            last_price = getattr(snapshot_local.last_trade, 'price', None)
                            if last_price:
                                # Only use as fallback if both are missing
                                if details['bid'] is None and details['ask'] is None:
                                    # Use last_price as a mid-point estimate, but don't set both to same value
                                    # Instead, leave them as None or use a small spread estimate
                                    # For now, we'll leave them as None to indicate missing data
                                    pass
                                # If only one is missing, don't fill it with last_price to avoid confusion
                        if hasattr(snapshot_local, 'greeks') and snapshot_local.greeks:
                            details['delta'] = getattr(snapshot_local.greeks, 'delta', None)
                            details['gamma'] = getattr(snapshot_local.greeks, 'gamma', None)
                            details['theta'] = getattr(snapshot_local.greeks, 'theta', None)
                            details['vega'] = getattr(snapshot_local.greeks, 'vega', None)
                            # Try to get IV from greeks object first
                            details['implied_volatility'] = getattr(snapshot_local.greeks, 'implied_volatility', None)
                        
                        # Also check for IV directly on snapshot object (alternative location)
                        if details['implied_volatility'] is None:
                            if hasattr(snapshot_local, 'implied_volatility') and snapshot_local.implied_volatility:
                                iv_value = snapshot_local.implied_volatility
                                # Handle both direct value and nested value attribute
                                if hasattr(iv_value, 'value'):
                                    details['implied_volatility'] = iv_value.value
                                elif isinstance(iv_value, (int, float)):
                                    details['implied_volatility'] = iv_value
                                else:
                                    details['implied_volatility'] = getattr(iv_value, 'implied_volatility', None)
                        
                        # Try alternative field names for IV
                        if details['implied_volatility'] is None:
                            for iv_field in ['iv', 'impliedVolatility', 'implied_vol', 'volatility']:
                                iv_value = getattr(snapshot_local, iv_field, None)
                                if iv_value is not None:
                                    if hasattr(iv_value, 'value'):
                                        details['implied_volatility'] = iv_value.value
                                    elif isinstance(iv_value, (int, float)):
                                        details['implied_volatility'] = iv_value
                                    break
                except Exception as e_local:
                    # If snapshot fetch fails entirely, we can't get bid/ask
                    pass
                return details

            if self.snapshot_max_concurrent > 0:
                from concurrent.futures import ThreadPoolExecutor, as_completed
                max_workers = min(self.snapshot_max_concurrent, 32)
                with ThreadPoolExecutor(max_workers=max_workers) as pool:
                    futures = []
                    for idx, c in enumerate(filtered_contracts):
                        futures.append(pool.submit(_fetch_snapshot, c, idx))
                    processed = 0
                    for fut in as_completed(futures):
                        res = fut.result() or {}
                        if res:
                            options_data["contracts"].append(res)
                        processed += 1
                        if processed % 10 == 0 and not self.quiet:
                            ticker = res.get('ticker', 'unknown') if res else 'unknown'
                            print(f"  ...processed {processed} of {len(filtered_contracts)} contracts (latest: {ticker})...", flush=True)
            else:
                for i, contract in enumerate(filtered_contracts):
                    if i > 0 and i % 10 == 0 and not self.quiet:
                        contract_ticker = getattr(contract, 'ticker', 'unknown')
                        print(f"  ...processed {i} of {len(filtered_contracts)} contracts (latest: {contract_ticker})...", flush=True)
                    res = _fetch_snapshot(contract, i)
                    if res:
                        options_data["contracts"].append(res)

            processing_end = time.time()
            if not self.quiet:
                print(f"  [TIMER] Processing and fetching snapshots for {len(filtered_contracts)} contracts took {processing_end - processing_start:.2f} seconds.")
            
            overall_end_time = time.time()
            if not self.quiet:
                print(f"  [TIMER] Total time for get_active_options_for_date: {overall_end_time - overall_start_time:.2f} seconds.")

        except Exception as e:
            return self._handle_api_error(e, "options data", symbol)
        
        # Save to CSV files only when explicitly requested
        if save_to_csv and options_data["contracts"]:
            self._save_options_to_csv(symbol, options_data)
        
        # Ensure total_downloaded is set (defaults to 0 if not set during processing)
        if "total_downloaded" not in options_data:
            options_data["total_downloaded"] = len(options_data.get("contracts", []))
        
        return {"success": True, "data": options_data}

    def format_output(
        self,
        symbol: str,
        target_date: str,
        stock_result: Dict[str, Any],
        options_result: Dict[str, Any],
        option_type: str,
        strike_range_percent: int | None,
        options_per_expiry: int,
        max_days_to_expiry: int | None
    ):
        """Formats the fetched data into readable tables."""
        output = []

        # --- Stock Price ---
        output.append(f"\n--- Stock Price for {symbol} ---")
        if stock_result.get('success'):
            data = stock_result['data']
            stock_table = [
                ['Requested Date', data.get('target_date')],
                ['Trading Day', data.get('trading_date')],
                ['Open', f"${data.get('open'):.2f}"],
                ['High', f"${data.get('high'):.2f}"],
                ['Low', f"${data.get('low'):.2f}"],
                ['Close', f"${data.get('close'):.2f}"],
                ['Volume', f"{data.get('volume'):,}"],
            ]
            output.append(tabulate(stock_table, headers=['Metric', 'Value'], tablefmt='grid'))
        else:
            output.append(f"Could not fetch stock price: {stock_result.get('error', 'Unknown error')}")

        # --- Options Data ---
        output.append(f"\n--- Options for {symbol} on {target_date} ---")
        output.append("Note: Bid/Ask are only available with Polygon plans that include real-time quotes for options. If your plan doesn't include this, bid/ask will show as N/A. Day Close = daily close price. FMV = Fair Market Value. Greeks, IV (Implied Volatility), and Volume from current market data.")

        # Add a note about the filters
        filter_notes = []
        if option_type != 'all':
            filter_notes.append(f"Type: {option_type.title()}")
        if strike_range_percent is not None:
             if stock_result.get('success'):
                close_price = stock_result['data']['close']
                min_strike = close_price * (1 - strike_range_percent / 100)
                max_strike = close_price * (1 + strike_range_percent / 100)
                filter_notes.append(f"Strike Range: ±{strike_range_percent}% of ${close_price:.2f} (strikes from ${min_strike:.2f} to ${max_strike:.2f})")
        if max_days_to_expiry is not None:
            filter_notes.append(f"Max Expiry: ±{max_days_to_expiry} days around {target_date}")

        if filter_notes:
            output.append(f"Filters Applied: {', '.join(filter_notes)}")
            
        if options_result.get('success'):
            contracts = options_result['data']['contracts']
            stock_close_price = None
            if stock_result.get('success'):
                stock_close_price = stock_result['data'].get('close')

            if not contracts:
                output.append(f"No active options contracts found for {symbol} on this date with the specified filters.")
            else:
                # Group by expiration
                options_by_expiry = {}
                for c in contracts:
                    exp = c['expiration']
                    if exp not in options_by_expiry:
                        options_by_expiry[exp] = []
                    options_by_expiry[exp].append(c)
                
                # Debug: Print all available expirations
                print(f"\nDEBUG: Found {len(options_by_expiry)} unique expirations for {symbol}:")
                for exp in sorted(options_by_expiry.keys()):
                    print(f"  {exp}: {len(options_by_expiry[exp])} contracts")
                
                for exp_date in sorted(options_by_expiry.keys())[:20]: # Show first 20 expirations
                    output.append(f"\nExpiration: {exp_date} (ticker: {symbol})")
                    options_table = []
                    
                    options_for_this_expiry = options_by_expiry[exp_date]
                    
                    # --- New logic to select options around the stock price ---
                    selected_options = []
                    if stock_close_price:
                        calls = [c for c in options_for_this_expiry if c.get('type','').lower() == 'call']
                        puts = [p for p in options_for_this_expiry if p.get('type','').lower() == 'put']

                        # Select calls around the money
                        below_price_calls = [c for c in calls if c.get('strike', -1) <= stock_close_price]
                        above_price_calls = [c for c in calls if c.get('strike', -1) > stock_close_price]
                        selected_options.extend(below_price_calls[-options_per_expiry:])
                        selected_options.extend(above_price_calls[:options_per_expiry])

                        # Select puts around the money
                        below_price_puts = [p for p in puts if p.get('strike', -1) <= stock_close_price]
                        above_price_puts = [p for p in puts if p.get('strike', -1) > stock_close_price]
                        selected_options.extend(below_price_puts[-options_per_expiry:])
                        selected_options.extend(above_price_puts[:options_per_expiry])
                        
                        # Sort final list for a clean display
                        selected_options.sort(key=lambda x: (x.get('type',''), x.get('strike',0)))
                    else:
                        # Fallback if no stock price is available: show first N contracts
                        selected_options = options_for_this_expiry[:options_per_expiry]

                    for contract in selected_options:
                        options_table.append([
                            contract.get('ticker', 'N/A'),
                            contract.get('type', 'N/A').title(),
                            f"${contract.get('strike'):.2f}",
                            f"${contract.get('bid'):.2f}" if contract.get('bid') is not None else 'N/A',
                            f"${contract.get('ask'):.2f}" if contract.get('ask') is not None else 'N/A',
                            f"${contract.get('day_close'):.2f}" if contract.get('day_close') is not None else 'N/A',
                            f"${contract.get('fmv'):.2f}" if contract.get('fmv') is not None else 'N/A',
                            f"{contract.get('delta'):.3f}" if contract.get('delta') is not None else 'N/A',
                            f"{contract.get('gamma'):.3f}" if contract.get('gamma') is not None else 'N/A',
                            f"{contract.get('theta'):.3f}" if contract.get('theta') is not None else 'N/A',
                            f"{contract.get('vega'):.3f}" if contract.get('vega') is not None else 'N/A',
                            f"{contract.get('implied_volatility'):.3f}" if contract.get('implied_volatility') is not None else 'N/A',
                            f"{contract.get('volume'):,}" if contract.get('volume') is not None else 'N/A',
                        ])
                    output.append(tabulate(options_table, headers=[f'Ticker ({symbol})', 'Type', 'Strike', 'Bid', 'Ask', 'Day Close', 'FMV', 'Delta', 'Gamma', 'Theta', 'Vega', 'IV', 'Volume'], tablefmt='grid'))
        else:
            output.append(f"Could not fetch options data: {options_result.get('error', 'Unknown error')}")
            
        rendered = "\n".join(output)
        return rendered

    def format_straddle_output(
        self,
        symbol: str,
        target_date: str,
        stock_result: Dict[str, Any],
        options_result: Dict[str, Any],
    ):
        """Formats options data into a straddle view showing matching calls and puts at the same strike."""
        output = []
        
        # --- Stock Price ---
        output.append(f"\n--- Stock Price for {symbol} ---")
        stock_close_price = None
        if stock_result.get('success'):
            data = stock_result['data']
            stock_close_price = data.get('close')
            stock_table = [
                ['Requested Date', data.get('target_date')],
                ['Trading Day', data.get('trading_date')],
                ['Open', f"${data.get('open'):.2f}"],
                ['High', f"${data.get('high'):.2f}"],
                ['Low', f"${data.get('low'):.2f}"],
                ['Close', f"${data.get('close'):.2f}"],
                ['Volume', f"{data.get('volume'):,}"],
            ]
            output.append(tabulate(stock_table, headers=['Metric', 'Value'], tablefmt='grid'))
        else:
            output.append(f"Could not fetch stock price: {stock_result.get('error', 'Unknown error')}")
        
        # --- Straddle View ---
        output.append(f"\n--- Straddle View for {symbol} on {target_date} ---")
        output.append("Straddles show matching Call and Put options at the same strike price and expiration.")
        output.append("Mid = (Bid + Ask) / 2. Total Cost = Call Mid + Put Mid.")
        
        if not options_result.get('success'):
            output.append(f"Could not fetch options data: {options_result.get('error', 'Unknown error')}")
            return "\n".join(output)
        
        contracts = options_result['data']['contracts']
        if not contracts:
            output.append(f"No options contracts found for {symbol}.")
            return "\n".join(output)
        
        # Group by expiration
        options_by_expiry = {}
        for c in contracts:
            exp = c.get('expiration')
            if exp:
                if exp not in options_by_expiry:
                    options_by_expiry[exp] = {'calls': {}, 'puts': {}}
                option_type = c.get('type', '').lower()
                strike = c.get('strike')
                if strike is not None:
                    strike_key = float(strike)
                    if option_type == 'call':
                        options_by_expiry[exp]['calls'][strike_key] = c
                    elif option_type == 'put':
                        options_by_expiry[exp]['puts'][strike_key] = c
        
        # Find straddles (matching strikes with both call and put)
        for exp_date in sorted(options_by_expiry.keys())[:20]:  # Show first 20 expirations
            expiry_data = options_by_expiry[exp_date]
            calls = expiry_data['calls']
            puts = expiry_data['puts']
            
            # Find common strikes
            common_strikes = sorted(set(calls.keys()) & set(puts.keys()))
            
            if not common_strikes:
                continue
            
            output.append(f"\nExpiration: {exp_date} (ticker: {symbol})")
            output.append(f"Found {len(common_strikes)} straddles (strikes with both call and put)")
            
            straddle_table = []
            for strike in common_strikes:
                call = calls[strike]
                put = puts[strike]
                
                # Calculate mid prices
                call_bid = call.get('bid')
                call_ask = call.get('ask')
                call_mid = (call_bid + call_ask) / 2 if (call_bid is not None and call_ask is not None) else None
                
                put_bid = put.get('bid')
                put_ask = put.get('ask')
                put_mid = (put_bid + put_ask) / 2 if (put_bid is not None and put_ask is not None) else None
                
                # Total cost
                total_cost = (call_mid + put_mid) if (call_mid is not None and put_mid is not None) else None
                
                # Helper function to safely extract numeric values (handles None, NaN, etc.)
                def safe_get_numeric(d, *keys):
                    for key in keys:
                        val = d.get(key)
                        if val is not None:
                            # Check for NaN (works for both numpy and pandas NaN)
                            try:
                                import math
                                if isinstance(val, float) and math.isnan(val):
                                    continue
                            except (TypeError, ImportError):
                                pass
                            # Check for pandas NA
                            try:
                                import pandas as pd
                                if pd.isna(val):
                                    continue
                            except (TypeError, ImportError, AttributeError):
                                pass
                            if isinstance(val, (int, float)):
                                return val
                    return None
                
                # Extract Greeks for Call - try multiple possible keys
                call_delta = safe_get_numeric(call, 'delta', 'greek_delta')
                call_gamma = safe_get_numeric(call, 'gamma', 'greek_gamma')
                call_theta = safe_get_numeric(call, 'theta', 'greek_theta')
                call_vega = safe_get_numeric(call, 'vega', 'greek_vega')
                call_iv = safe_get_numeric(call, 'implied_volatility', 'iv', 'implied_vol')
                
                # Extract Greeks for Put - try multiple possible keys
                put_delta = safe_get_numeric(put, 'delta', 'greek_delta')
                put_gamma = safe_get_numeric(put, 'gamma', 'greek_gamma')
                put_theta = safe_get_numeric(put, 'theta', 'greek_theta')
                put_vega = safe_get_numeric(put, 'vega', 'greek_vega')
                put_iv = safe_get_numeric(put, 'implied_volatility', 'iv', 'implied_vol')
                
                # Distance from stock price
                distance = None
                distance_pct = None
                if stock_close_price is not None:
                    distance = strike - stock_close_price
                    distance_pct = (distance / stock_close_price * 100) if stock_close_price > 0 else None
                
                # Build row: Strike, Dist, Dist %, then Call data (Bid, Ask, Mid, Delta, Gamma, Theta, Vega, IV), then Put data (Bid, Ask, Mid, Delta, Gamma, Theta, Vega, IV), then Total Cost
                straddle_table.append([
                    f"${strike:.2f}",
                    f"{distance:+.2f}" if distance is not None else 'N/A',
                    f"{distance_pct:+.2f}%" if distance_pct is not None else 'N/A',
                    # Call data
                    f"${call_bid:.2f}" if call_bid is not None else 'N/A',
                    f"${call_ask:.2f}" if call_ask is not None else 'N/A',
                    f"${call_mid:.2f}" if call_mid is not None else 'N/A',
                    f"{call_delta:+.3f}" if call_delta is not None else 'N/A',
                    f"{call_gamma:.3f}" if call_gamma is not None else 'N/A',
                    f"{call_theta:.3f}" if call_theta is not None else 'N/A',
                    f"{call_vega:.3f}" if call_vega is not None else 'N/A',
                    f"{call_iv:.3f}" if call_iv is not None else 'N/A',
                    # Put data
                    f"${put_bid:.2f}" if put_bid is not None else 'N/A',
                    f"${put_ask:.2f}" if put_ask is not None else 'N/A',
                    f"${put_mid:.2f}" if put_mid is not None else 'N/A',
                    f"{put_delta:+.3f}" if put_delta is not None else 'N/A',
                    f"{put_gamma:.3f}" if put_gamma is not None else 'N/A',
                    f"{put_theta:.3f}" if put_theta is not None else 'N/A',
                    f"{put_vega:.3f}" if put_vega is not None else 'N/A',
                    f"{put_iv:.3f}" if put_iv is not None else 'N/A',
                    # Total cost
                    f"${total_cost:.2f}" if total_cost is not None else 'N/A',
                ])
            
            if straddle_table:
                output.append(tabulate(
                    straddle_table,
                    headers=[
                        'Strike', 'Dist', 'Dist %',
                        'Call Bid', 'Call Ask', 'Call Mid', 'Call Δ', 'Call Γ', 'Call Θ', 'Call ν', 'Call IV',
                        'Put Bid', 'Put Ask', 'Put Mid', 'Put Δ', 'Put Γ', 'Put Θ', 'Put ν', 'Put IV',
                        'Total Cost'
                    ],
                    tablefmt='grid'
                ))
        
        rendered = "\n".join(output)
        return rendered


async def display_and_save_saved_options(
    symbols_list: list[str],
    args: argparse.Namespace,
    api_key: str
) -> None:
    """Display and/or save saved options from the database after fetching (non-continuous mode only)."""
    # Auto-enable save_saved_to_csv if save_saved_to_csv_file is specified
    if getattr(args, 'save_saved_to_csv_file', None) and not getattr(args, 'save_saved_to_csv', False):
        args.save_saved_to_csv = True
    
    if not getattr(args, 'display_saved', False) and not getattr(args, 'save_saved_to_csv', False):
        return
    
    if not getattr(args, 'use_db', None):
        if getattr(args, 'verbose', False):
            print("Warning: --display-saved and --save-saved-to-csv require --use-db to be set.", file=sys.stderr)
        return
    
    db_conn = getattr(args, 'use_db', None)
    if not db_conn:
        if getattr(args, 'verbose', False):
            print("Warning: Database connection not available for retrieving saved options.", file=sys.stderr)
        return
    
    # Determine database type
    db_type = 'questdb'  # default
    db_config = db_conn
    if db_conn.startswith('http://') or db_conn.startswith('https://'):
        db_type = 'remote'
        db_config = db_conn.replace('http://', '').replace('https://', '')
    elif db_conn.startswith('postgresql://'):
        db_type = 'postgresql'
    
    enable_cache = not getattr(args, 'no_cache', False)
    redis_url = os.getenv('REDIS_URL', 'redis://localhost:6379/0') if enable_cache else None
    db_timeout = getattr(args, 'db_timeout', 60.0)
    
    fetcher = HistoricalDataFetcher(
        api_key,
        getattr(args, 'data_dir', None),
        getattr(args, 'verbose', False),
        getattr(args, 'snapshot_max_concurrent', 0),
        getattr(args, 'refresh_threshold_seconds', None)
    )
    
    for symbol in symbols_list:
        try:
            # Get the target date(s) - use start_date if provided, otherwise date
            target_date = getattr(args, 'start_date', None) or args.date
            end_date = getattr(args, 'end_date', None)
            
            # Calculate max_days_to_expiry based on the date range that was actually fetched
            # This ensures we only show options from the date range that was requested
            if end_date:
                start_dt = datetime.strptime(target_date, '%Y-%m-%d')
                end_dt = datetime.strptime(end_date, '%Y-%m-%d')
                days_diff = (end_dt - start_dt).days
                max_days_to_expiry = max(1, days_diff)  # At least 1 day
            else:
                # If no end_date, use the original max_days_to_expiry or default to 30
                max_days_to_expiry = getattr(args, 'max_days_to_expiry', 30)
            
            # Retrieve saved options from database
            # Always show messages when --display-saved or --save-saved-to-csv is used
            show_messages = getattr(args, 'display_saved', False) or getattr(args, 'save_saved_to_csv', False) or getattr(args, 'verbose', False)
            if show_messages:
                date_range_msg = f"{target_date} to {end_date}" if end_date else target_date
                print(f"\n--- Retrieving saved options for {symbol} from database (date range: {date_range_msg}, max_days_to_expiry: {max_days_to_expiry}) ---")
            
            # Use the existing get_active_options_for_date with use_db=True to retrieve from DB
            # Respect the same date range constraints that were used when fetching
            options_result = await fetcher.get_active_options_for_date(
                symbol=symbol,
                target_date_str=target_date,
                option_type='all',  # Show all option types when displaying saved options
                stock_close_price=None,  # Don't filter by stock price for saved options display
                strike_range_percent=None,  # Don't filter by strike range for saved options display
                max_days_to_expiry=max_days_to_expiry,  # Respect the date range that was fetched
                include_expired=getattr(args, 'include_expired', False),
                use_db=True,
                db_conn=db_conn,
                force_fresh=False,
                enable_cache=enable_cache,
                redis_url=redis_url,
                db_timeout=db_timeout,
                db_only=getattr(args, 'db_only', False)
            )
            
            if not options_result.get('success'):
                if show_messages:
                    print(f"Could not retrieve saved options for {symbol}: {options_result.get('error', 'Unknown error')}")
                continue
            
            contracts = options_result['data'].get('contracts', [])
            if not contracts:
                if show_messages:
                    print(f"No saved options found for {symbol} in the database.")
                continue
            
            if show_messages:
                print(f"Found {len(contracts)} saved options for {symbol}")
            
            # Get stock price for display
            stock_result = await fetcher.get_stock_price_for_date(symbol, target_date)
            
            # Display options if requested (always show when --display-saved is used, regardless of verbose)
            if getattr(args, 'display_saved', False):
                # Check if straddle view is requested
                if getattr(args, 'straddle_view', False):
                    # Show straddle view
                    output = fetcher.format_straddle_output(
                        symbol=symbol,
                        target_date=target_date,
                        stock_result=stock_result,
                        options_result=options_result
                    )
                    print(output)
                else:
                    # Use a very large number for options_per_expiry to show all saved options
                    output = fetcher.format_output(
                        symbol=symbol,
                        target_date=target_date,
                        stock_result=stock_result,
                        options_result=options_result,
                        option_type=getattr(args, 'option_type', 'all'),
                        strike_range_percent=None,  # Don't filter for display
                        options_per_expiry=999999,  # Show all saved options
                        max_days_to_expiry=getattr(args, 'max_days_to_expiry', 30)
                    )
                    print(output)
            
            # Save to CSV if requested
            if getattr(args, 'save_saved_to_csv', False):
                csv_filename = getattr(args, 'save_saved_to_csv_file', None)
                if csv_filename:
                    # Save to a single specified file
                    csv_path = Path(csv_filename)
                    csv_path.parent.mkdir(parents=True, exist_ok=True)
                    
                    # Prepare data for CSV
                    current_time = datetime.now().isoformat()
                    csv_data = []
                    for contract in contracts:
                        csv_data.append({
                            'timestamp': current_time,
                            'ticker': contract.get('ticker', ''),
                            'type': contract.get('type', ''),
                            'strike': contract.get('strike', ''),
                            'expiration': contract.get('expiration', ''),
                            'bid': contract.get('bid', ''),
                            'ask': contract.get('ask', ''),
                            'day_close': contract.get('day_close', ''),
                            'fmv': contract.get('fmv', ''),
                            'delta': contract.get('delta', ''),
                            'gamma': contract.get('gamma', ''),
                            'theta': contract.get('theta', ''),
                            'vega': contract.get('vega', ''),
                            'implied_volatility': contract.get('implied_volatility', ''),
                            'volume': contract.get('volume', ''),
                        })
                    
                    # Write to CSV (append mode)
                    file_exists = csv_path.exists()
                    with open(csv_path, 'a', newline='', encoding='utf-8') as csvfile:
                        fieldnames = ['timestamp', 'ticker', 'type', 'strike', 'expiration', 
                                     'bid', 'ask', 'day_close', 'fmv', 'delta', 'gamma', 'theta', 'vega', 'implied_volatility', 'volume']
                        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                        
                        if not file_exists:
                            writer.writeheader()
                        writer.writerows(csv_data)
                    
                    if show_messages:
                        print(f"Saved {len(contracts)} options for {symbol} to {csv_path}")
                else:
                    # Use default behavior: save to separate files by expiration date
                    options_data = {
                        'contracts': contracts,
                        'total_downloaded': len(contracts)
                    }
                    fetcher._save_options_to_csv(symbol, options_data)
                    if show_messages:
                        print(f"Saved {len(contracts)} options for {symbol} to CSV files (organized by expiration date).")
        
        except Exception as e:
            if getattr(args, 'verbose', False):
                print(f"Error retrieving saved options for {symbol}: {e}", file=sys.stderr)


def split_tickers_into_chunks(tickers: list[str], chunk_size: int = 250) -> list[list[str]]:
    """Split ticker list into chunks of maximum chunk_size."""
    chunks = []
    for i in range(0, len(tickers), chunk_size):
        chunks.append(tickers[i:i + chunk_size])
    return chunks


def generate_month_ranges(start_date_str: str, num_months: int, end_date_str: str | None = None) -> list[tuple[str, str]]:
    """Generate 30-day date ranges starting from start_date.
    
    Returns list of (start_date, end_date) tuples in YYYY-MM-DD format.
    If end_date_str is provided, stops generating ranges once the end date is reached.
    """
    start_date = datetime.strptime(start_date_str, '%Y-%m-%d')
    end_date = datetime.strptime(end_date_str, '%Y-%m-%d') if end_date_str else None
    ranges = []
    for i in range(num_months):
        month_start = start_date + timedelta(days=i * 30)
        month_end = month_start + timedelta(days=30) - timedelta(days=1)
        
        # If end_date is specified, cap the month_end to not exceed it
        if end_date and month_end > end_date:
            if month_start > end_date:
                # This month range starts after the end date, stop generating
                break
            month_end = end_date
        
        ranges.append((
            month_start.strftime('%Y-%m-%d'),
            month_end.strftime('%Y-%m-%d')
        ))
        
        # If we've reached the end date, stop generating more ranges
        if end_date and month_end >= end_date:
            break
    return ranges


def generate_day_ranges(start_date_str: str, num_days: int, end_date_str: str | None = None) -> list[tuple[str, str]]:
    """Generate single-day date ranges starting from start_date.
    
    Returns list of (start_date, end_date) tuples in YYYY-MM-DD format.
    Each range represents a single day.
    If end_date_str is provided, stops generating ranges once the end date is reached.
    """
    start_date = datetime.strptime(start_date_str, '%Y-%m-%d')
    end_date = datetime.strptime(end_date_str, '%Y-%m-%d') if end_date_str else None
    ranges = []
    for i in range(num_days):
        day_start = start_date + timedelta(days=i)
        day_end = day_start  # Single day range
        
        # If end_date is specified, cap the day_end to not exceed it
        if end_date and day_end > end_date:
            if day_start > end_date:
                # This day range starts after the end date, stop generating
                break
            day_end = end_date
        
        ranges.append((
            day_start.strftime('%Y-%m-%d'),
            day_end.strftime('%Y-%m-%d')
        ))
        
        # If we've reached the end date, stop generating more ranges
        if end_date and day_end >= end_date:
            break
    return ranges


def allocate_processes_by_proximity(total_processes: int, num_months: int) -> list[int]:
    """Allocate processes to months using 2.5x exponential decay per month.
    
    Month 1 gets 2.5x month 2, month 2 gets 2.5x month 3, etc.
    Each month gets at least 1 process.
    Remaining processes after allocation go to month 1 (most recent).
    """
    if num_months == 0:
        return []
    if total_processes < num_months:
        # Not enough processes - give 1 to each
        return [1] * num_months
    
    # Calculate base allocations with 2.5x decay per month
    # Month 0 (most recent) should get the most, with each subsequent month getting 1/2.5
    # We'll work backwards: start with month n-1 getting 1, month n-2 getting 2.5, etc.
    allocations = []
    for i in range(num_months):
        # Month i gets 2.5^(num_months - 1 - i) relative weight
        # Month 0 gets 2.5^(num_months-1), month n-1 gets 2.5^0 = 1
        weight = 2.5 ** (num_months - 1 - i)
        allocations.append(weight)
    
    # Normalize to use available processes
    total_weight = sum(allocations)
    
    # First pass: allocate based on weights, but ensure at least 1 per month
    # We'll allocate proportionally, then floor at 1
    allocated = 0
    for i in range(num_months):
        # Calculate proportional allocation
        proportional = int(total_processes * allocations[i] / total_weight)
        # Ensure at least 1
        allocations[i] = max(1, proportional)
        allocated += allocations[i]
    
    # Second pass: if we allocated too many, reduce from farthest months first
    if allocated > total_processes:
        excess = allocated - total_processes
        for i in range(num_months - 1, -1, -1):  # Start from farthest month
            if allocations[i] > 1 and excess > 0:
                reduction = min(excess, allocations[i] - 1)
                allocations[i] -= reduction
                excess -= reduction
                if excess == 0:
                    break
    
    # Third pass: if we have remaining processes, give them all to month 1 (most recent)
    allocated = sum(allocations)
    if allocated < total_processes:
        remaining = total_processes - allocated
        allocations[0] += remaining
    
    return allocations


def _dry_run_worker_standalone(chunk: list[str], target_date: str, worker_id: int, month_idx: int, month_start: str, month_end: str, use_db: bool) -> dict:
    """Standalone dry-run worker function (must be at module level for ProcessPoolExecutor pickling)."""
    import os
    pid = os.getpid()
    print(f"[DRY-RUN Month {month_idx + 1} Worker {worker_id} PID {pid}] Would process chunk with {len(chunk)} tickers: {chunk[:5]}{'...' if len(chunk) > 5 else ''}")
    print(f"[DRY-RUN Month {month_idx + 1} Worker {worker_id} PID {pid}] Would fetch options for date range: {month_start} to {month_end}")
    print(f"[DRY-RUN Month {month_idx + 1} Worker {worker_id} PID {pid}] Would save to DB: {use_db}")
    print(f"[DRY-RUN Month {month_idx + 1} Worker {worker_id} PID {pid}] All executions would be successful (simulated)")
    return {
        'symbols_processed': len(chunk),
        'save_success_count': len(chunk) if use_db else 0,
        'save_failure_count': 0,
        'errors': []
    }


def _run_for_ticker_chunk(ticker_chunk: list[str], target_date_str: str, args_namespace: argparse.Namespace, api_key: str) -> dict:
    """Worker task: runs fetch for a chunk of tickers for a specific date."""
    results = {
        'symbols_processed': 0,
        'save_success_count': 0,
        'save_failure_count': 0,
        'errors': []
    }
    
    for symbol in ticker_chunk:
        try:
            result = _run_for_symbol(symbol, args_namespace, api_key)
            if isinstance(result, dict) and 'save_success' in result:
                results['symbols_processed'] += 1
                if result.get('save_success'):
                    results['save_success_count'] += 1
                else:
                    results['save_failure_count'] += 1
            else:
                results['errors'].append(f"{symbol}: {result}")
        except Exception as e:
            results['errors'].append(f"{symbol}: {str(e)}")
    
    return results


def _run_for_symbol(symbol: str, args_namespace: argparse.Namespace, api_key: str) -> str:
    """Worker task: runs fetch for a single symbol, saves to DB in worker process, and returns formatted output string."""
    # Convert index symbol for database operations (I:SPX -> SPX)
    # The fetcher functions handle conversion for API calls, but we need it here for DB saves
    polygon_symbol, _ = _convert_index_symbol_for_polygon(symbol)
    
    try:
        fetcher = HistoricalDataFetcher(
            api_key,
            args_namespace.data_dir,
            getattr(args_namespace, 'verbose', False),
            getattr(args_namespace, 'snapshot_max_concurrent', 0),
            getattr(args_namespace, 'refresh_threshold_seconds', None)
        )
        
        async def _inner():
            stock_result = await fetcher.get_stock_price_for_date(symbol, args_namespace.date)
            stock_close_price = stock_result['data'].get('close') if stock_result.get('success') else None
            # Determine cache settings
            enable_cache = not getattr(args_namespace, 'no_cache', False)
            redis_url = os.getenv('REDIS_URL', 'redis://localhost:6379/0') if enable_cache else None
            options_result = await fetcher.get_active_options_for_date(
                symbol=symbol,
                target_date_str=args_namespace.date,
                option_type=args_namespace.option_type,
                stock_close_price=stock_close_price,
                strike_range_percent=args_namespace.strike_range_percent,
                max_days_to_expiry=args_namespace.max_days_to_expiry,
                include_expired=args_namespace.include_expired,
                # If --force-fresh is set, bypass CSV cache reads even if --use-csv is enabled
                use_cache=(getattr(args_namespace, 'use_csv', False) and not getattr(args_namespace, 'force_fresh', False)),
                save_to_csv=getattr(args_namespace, 'use_csv', False),
                use_db=bool(getattr(args_namespace, 'use_db', None)),
                db_conn=getattr(args_namespace, 'use_db', None),
                force_fresh=getattr(args_namespace, 'force_fresh', False),
                enable_cache=enable_cache,
                redis_url=redis_url,
                db_timeout=getattr(args_namespace, 'db_timeout', 60.0),
                db_only=getattr(args_namespace, 'db_only', False)
            )
            
            # Save to database in worker process (multi-processed saves)
            save_success = False
            if getattr(args_namespace, 'use_db', None) and options_result.get('success'):
                contracts = options_result['data'].get('contracts') or []
                total_downloaded = options_result['data'].get('total_downloaded', len(contracts))
                if contracts:
                    import pandas as _pd
                    # Build DataFrame in expected shape for DB layer
                    df = _pd.DataFrame.from_records(contracts)
                    if not df.empty:
                        # Map columns to expected names: option_ticker from 'ticker'
                        if 'ticker' in df.columns and 'option_ticker' not in df.columns:
                            df = df.rename(columns={'ticker': 'option_ticker'})
                        
                        # Map column names to match DB schema
                        column_mapping = {
                            'expiration': 'expiration_date',
                            'strike': 'strike_price', 
                            'type': 'option_type',
                            'bid': 'bid',
                            'ask': 'ask',
                            'day_close': 'day_close',
                            'price': 'price',
                            'delta': 'delta',
                            'gamma': 'gamma', 
                            'theta': 'theta',
                            'vega': 'vega',
                            'rho': 'rho',
                            'implied_volatility': 'implied_volatility',
                            'volume': 'volume',
                            'open_interest': 'open_interest',
                            'last_quote_timestamp': 'last_quote_timestamp'
                        }
                        
                        # Debug: Check if expiration column exists with different names
                        expiration_candidates = ['expiration', 'expiration_date', 'expiry', 'expiry_date']
                        found_expiration = None
                        for candidate in expiration_candidates:
                            if candidate in df.columns:
                                found_expiration = candidate
                                break
                        
                        if found_expiration and found_expiration != 'expiration_date':
                            if getattr(args_namespace, 'verbose', False) and getattr(args_namespace, 'debug', False):
                                print(f"DEBUG: Found expiration column as '{found_expiration}', mapping to 'expiration_date'")
                            df = df.rename(columns={found_expiration: 'expiration_date'})
                        
                        # Rename columns that exist
                        for old_name, new_name in column_mapping.items():
                            if old_name in df.columns:
                                df = df.rename(columns={old_name: new_name})
                        
                        # Ensure required columns exist; fill missing if necessary
                        required_cols = ['option_ticker', 'expiration_date', 'strike_price', 'option_type']
                        for req_col in required_cols:
                            if req_col not in df.columns:
                                df[req_col] = _pd.NA
                        
                        # Save in worker process - each worker has its own DB connection
                        db_config = getattr(args_namespace, 'use_db', None)
                        if db_config:
                            # Initialize database connection in worker process
                            if db_config.startswith('http://') or db_config.startswith('https://'):
                                # HTTP server - save via HTTP in worker process
                                import aiohttp
                                async with aiohttp.ClientSession() as session:
                                    db_save_task = {
                                        'symbol': polygon_symbol,  # Use converted symbol for database
                                        'df': df,
                                        'contracts_count': len(contracts)
                                    }
                                    await save_single_symbol_via_http(db_save_task, db_config, session, args_namespace)
                                    save_success = True
                            else:
                                # Direct database connection - create DB instance in worker
                                from common.stock_db import get_stock_db
                                
                                # Determine database type
                                db_type = 'questdb'  # default
                                if db_config.startswith('postgresql://'):
                                    db_type = 'postgresql'
                                
                                # Create DB instance in worker process - each worker has its own connection
                                worker_db_instance = get_stock_db(
                                    db_type, 
                                    db_config=db_config, 
                                    enable_cache=enable_cache, 
                                    redis_url=redis_url
                                )
                                
                                # Use async context manager to ensure proper cleanup
                                async with worker_db_instance:
                                    await worker_db_instance.save_options_data(df=df, ticker=polygon_symbol)  # Use converted symbol for database
                                    
                                    # Update Redis cache
                                    if enable_cache:
                                        redis_client = get_redis_client_for_refresh(redis_url) if redis_url else None
                                        if redis_client:
                                            from datetime import datetime, timezone
                                            now_utc = datetime.now(timezone.utc)
                                            set_redis_last_write_timestamp(redis_client, symbol, now_utc, ttl_seconds=86400)
                                    
                                    save_success = True
                                    if getattr(args_namespace, 'verbose', False):
                                        saved_count = len(contracts)
                                        print(f"[SAVE] {symbol}: Successfully saved {saved_count} contracts out of {total_downloaded} downloaded (worker PID: {os.getpid()})")
            
            formatted_output = fetcher.format_output(
                symbol=symbol,
                target_date=args_namespace.date,
                stock_result=stock_result,
                options_result=options_result,
                option_type=args_namespace.option_type,
                strike_range_percent=args_namespace.strike_range_percent,
                options_per_expiry=args_namespace.options_per_expiry,
                max_days_to_expiry=args_namespace.max_days_to_expiry
            )
            
            # Return formatted output (save already happened in worker)
            return {
                'formatted_output': formatted_output,
                'db_save_data': None,  # Already saved in worker, no need to return
                'save_success': save_success
            }
        
        return asyncio.run(_inner())
    except Exception as e:
        import traceback
        error_msg = f"Error processing {symbol}: {e}"
        if getattr(args_namespace, 'debug', False):
            error_msg += f"\n{traceback.format_exc()}"
        return error_msg

async def save_single_symbol_via_http(db_save_task: dict, http_url: str, session, args) -> None:
    """Save a single symbol's options data via HTTP request (for incremental saves)."""
    import aiohttp
    import json
    
    symbol = db_save_task['symbol']
    df = db_save_task['df']
    contracts_count = db_save_task['contracts_count']
    
    # Get batch size from args, default to 100
    batch_size = getattr(args, 'db_batch_size', 100)
    
    try:
        # Convert DataFrame to records for HTTP transmission
        df_for_http = df.copy()
        for col_name in df_for_http.select_dtypes(include=['datetime64[ns]', 'datetime64[ns, UTC]']).columns:
            df_for_http[col_name] = df_for_http[col_name].dt.strftime('%Y-%m-%dT%H:%M:%S.%fZ')
        
        data_records = df_for_http.to_dict(orient='records')
        total_records = len(data_records)
        
        if total_records <= batch_size:
            # Single request for small datasets
            if getattr(args, 'verbose', False):
                print(f"[SAVE] {symbol}: Sending {total_records} records to database...")
            
            payload = {
                "command": "save_options_data",
                "params": {
                    "ticker": symbol,
                    "data": data_records,
                    "index_col": "expiration_date"
                }
            }
            
            async with session.post(
                f"{http_url}/db_command",
                json=payload,
                headers={"Content-Type": "application/json"}
            ) as response:
                if response.status == 200:
                    if getattr(args, 'verbose', False):
                        print(f"[SAVE] {symbol}: Successfully saved {contracts_count} contracts")
                    
                    # Update Redis cache
                    enable_cache = not getattr(args, 'no_cache', False)
                    if enable_cache:
                        redis_url = os.getenv('REDIS_URL', 'redis://localhost:6379/0')
                        redis_client = get_redis_client_for_refresh(redis_url) if redis_url else None
                        if redis_client:
                            from datetime import datetime, timezone
                            now_utc = datetime.now(timezone.utc)
                            set_redis_last_write_timestamp(redis_client, symbol, now_utc, ttl_seconds=86400)
                else:
                    error_text = await response.text()
                    if getattr(args, 'verbose', False):
                        print(f"[SAVE] {symbol}: Warning - HTTP {response.status} - {error_text}")
        else:
            # Split into batches
            num_batches = (total_records + batch_size - 1) // batch_size
            if getattr(args, 'verbose', False):
                print(f"[SAVE] {symbol}: Splitting {total_records} records into {num_batches} batches...")
            
            successful_batches = 0
            failed_batches = 0
            
            for batch_num in range(num_batches):
                start_idx = batch_num * batch_size
                end_idx = min(start_idx + batch_size, total_records)
                batch_records = data_records[start_idx:end_idx]
                
                payload = {
                    "command": "save_options_data",
                    "params": {
                        "ticker": symbol,
                        "data": batch_records,
                        "index_col": "expiration_date"
                    }
                }
                
                try:
                    async with session.post(
                        f"{http_url}/db_command",
                        json=payload,
                        headers={"Content-Type": "application/json"}
                    ) as response:
                        if response.status == 200:
                            successful_batches += 1
                            if getattr(args, 'verbose', False) and (batch_num + 1) % 10 == 0:
                                print(f"[SAVE] {symbol}: Batch {batch_num + 1}/{num_batches} saved")
                            
                            # Update Redis cache after last batch
                            if batch_num == num_batches - 1:
                                enable_cache = not getattr(args, 'no_cache', False)
                                if enable_cache:
                                    redis_url = os.getenv('REDIS_URL', 'redis://localhost:6379/0')
                                    redis_client = get_redis_client_for_refresh(redis_url) if redis_url else None
                                    if redis_client:
                                        from datetime import datetime, timezone
                                        now_utc = datetime.now(timezone.utc)
                                        set_redis_last_write_timestamp(redis_client, symbol, now_utc, ttl_seconds=86400)
                        else:
                            error_text = await response.text()
                            failed_batches += 1
                            if getattr(args, 'verbose', False):
                                print(f"[SAVE] {symbol}: Warning - batch {batch_num + 1} failed: HTTP {response.status}")
                except Exception as batch_e:
                    failed_batches += 1
                    if getattr(args, 'verbose', False):
                        print(f"[SAVE] {symbol}: Warning - batch {batch_num + 1} error: {batch_e}")
            
            if getattr(args, 'verbose', False):
                if failed_batches == 0:
                    print(f"[SAVE] {symbol}: Successfully saved all {num_batches} batches ({contracts_count} total contracts)")
                else:
                    print(f"[SAVE] {symbol}: Completed - {successful_batches} successful, {failed_batches} failed batches")
                    
    except Exception as e:
        if getattr(args, 'verbose', False):
            print(f"[SAVE] {symbol}: Error saving to DB: {e}")

async def save_single_symbol_via_direct_db(db_save_task: dict, db_instance, args) -> None:
    """Save a single symbol's options data via direct database connection (for incremental saves)."""
    symbol = db_save_task['symbol']
    df = db_save_task['df']
    contracts_count = db_save_task['contracts_count']
    
    try:
        if getattr(args, 'verbose', False):
            print(f"[SAVE] {symbol}: Saving {contracts_count} contracts to database...")
        
        await db_instance.save_options_data(df=df, ticker=symbol)
        
        if getattr(args, 'verbose', False):
            print(f"[SAVE] {symbol}: Successfully saved {contracts_count} contracts")
        
        # Update Redis cache
        enable_cache = not getattr(args, 'no_cache', False)
        if enable_cache:
            redis_url = os.getenv('REDIS_URL', 'redis://localhost:6379/0')
            redis_client = get_redis_client_for_refresh(redis_url) if redis_url else None
            if redis_client:
                from datetime import datetime, timezone
                now_utc = datetime.now(timezone.utc)
                set_redis_last_write_timestamp(redis_client, symbol, now_utc, ttl_seconds=86400)
    except Exception as e:
        if getattr(args, 'verbose', False):
            print(f"[SAVE] {symbol}: Warning - failed to save: {e}")

async def save_options_to_database(db_save_tasks: list, args) -> None:
    """Save options data to database in main thread (batch mode - for backward compatibility)."""
    if not db_save_tasks:
        return
    
    db_config = getattr(args, 'use_db', None)
    if getattr(args, 'verbose', False):
        print(f"Connecting to database: {db_config}")
    
    # Check if this is an HTTP server connection
    if db_config and db_config.startswith('http://'):
        # Use HTTP requests to save data via db_server.py
        await save_options_via_http(db_save_tasks, db_config, args)
    else:
        # Use direct database connection
        await save_options_via_direct_db(db_save_tasks, db_config, args)

async def save_options_via_http(db_save_tasks: list, http_url: str, args) -> None:
    """Save options data via HTTP requests to db_server.py with automatic batching for large datasets."""
    import aiohttp
    import json
    
    # Get batch size from args, default to 100
    batch_size = getattr(args, 'db_batch_size', 100)
    
    try:
        async with aiohttp.ClientSession() as session:
            for task in db_save_tasks:
                symbol = task['symbol']
                df = task['df']
                contracts_count = task['contracts_count']
                
                try:
                    # Convert DataFrame to records for HTTP transmission
                    # First, convert datetime columns to ISO format strings
                    df_for_http = df.copy()
                    for col_name in df_for_http.select_dtypes(include=['datetime64[ns]', 'datetime64[ns, UTC]']).columns:
                        df_for_http[col_name] = df_for_http[col_name].dt.strftime('%Y-%m-%dT%H:%M:%S.%fZ')
                    
                    # Convert to records
                    data_records = df_for_http.to_dict(orient='records')
                    total_records = len(data_records)
                    
                    if total_records <= batch_size:
                        # Single request for small datasets
                        if getattr(args, 'verbose', False):
                            print(f"Sending HTTP request to {http_url}/db_command with command: save_options_data ({total_records} records)")
                        
                        # Prepare the HTTP request payload
                        payload = {
                            "command": "save_options_data",
                            "params": {
                                "ticker": symbol,
                                "data": data_records,
                                "index_col": "expiration_date"  # Use expiration_date as index
                            }
                        }
                        
                        # Send HTTP POST request to db_server.py
                        async with session.post(
                            f"{http_url}/db_command",
                            json=payload,
                            headers={"Content-Type": "application/json"}
                        ) as response:
                            if response.status == 200:
                                result = await response.json()
                                if getattr(args, 'verbose', False):
                                    print(f"Successfully saved {contracts_count} options contracts to database for {symbol}")
                                
                                # Update Redis cache with the current timestamp
                                enable_cache = not getattr(args, 'no_cache', False)
                                if enable_cache:
                                    redis_url = os.getenv('REDIS_URL', 'redis://localhost:6379/0')
                                    redis_client = get_redis_client_for_refresh(redis_url) if redis_url else None
                                    if redis_client:
                                        from datetime import datetime, timezone
                                        now_utc = datetime.now(timezone.utc)
                                        set_redis_last_write_timestamp(redis_client, symbol, now_utc, ttl_seconds=86400)
                            else:
                                error_text = await response.text()
                                if getattr(args, 'verbose', False):
                                    print(f"Warning: failed to save options to DB for {symbol}: HTTP {response.status} - {error_text}")
                    else:
                        # Split large datasets into batches
                        num_batches = (total_records + batch_size - 1) // batch_size  # Ceiling division
                        if getattr(args, 'verbose', False):
                            print(f"Splitting {total_records} records for {symbol} into {num_batches} batches of max {batch_size} records each")
                        
                        successful_batches = 0
                        failed_batches = 0
                        
                        for batch_num in range(num_batches):
                            start_idx = batch_num * batch_size
                            end_idx = min(start_idx + batch_size, total_records)
                            batch_records = data_records[start_idx:end_idx]
                            
                            if getattr(args, 'verbose', False):
                                print(f"  Sending batch {batch_num + 1}/{num_batches} for {symbol} ({len(batch_records)} records)")
                            
                            # Prepare the HTTP request payload for this batch
                            payload = {
                                "command": "save_options_data",
                                "params": {
                                    "ticker": symbol,
                                    "data": batch_records,
                                    "index_col": "expiration_date"  # Use expiration_date as index
                                }
                            }
                            
                            try:
                                # Send HTTP POST request to db_server.py
                                async with session.post(
                                    f"{http_url}/db_command",
                                    json=payload,
                                    headers={"Content-Type": "application/json"}
                                ) as response:
                                    if response.status == 200:
                                        result = await response.json()
                                        successful_batches += 1
                                        if getattr(args, 'verbose', False):
                                            print(f"    Batch {batch_num + 1} saved successfully")
                                        # Update Redis cache after last batch
                                        if batch_num == num_batches - 1:
                                            enable_cache = not getattr(args, 'no_cache', False)
                                            if enable_cache:
                                                redis_url = os.getenv('REDIS_URL', 'redis://localhost:6379/0')
                                                redis_client = get_redis_client_for_refresh(redis_url) if redis_url else None
                                                if redis_client:
                                                    from datetime import datetime, timezone
                                                    now_utc = datetime.now(timezone.utc)
                                                    set_redis_last_write_timestamp(redis_client, symbol, now_utc, ttl_seconds=86400)
                                    else:
                                        error_text = await response.text()
                                        failed_batches += 1
                                        if getattr(args, 'verbose', False):
                                            print(f"    Warning: failed to save batch {batch_num + 1} for {symbol}: HTTP {response.status} - {error_text}")
                            except Exception as batch_e:
                                failed_batches += 1
                                if getattr(args, 'verbose', False):
                                    print(f"    Warning: failed to save batch {batch_num + 1} for {symbol}: {batch_e}")
                        
                        # Summary for batched requests
                        if getattr(args, 'verbose', False):
                            if failed_batches == 0:
                                print(f"Successfully saved all {num_batches} batches ({contracts_count} total contracts) to database for {symbol}")
                            else:
                                print(f"Completed batch processing for {symbol}: {successful_batches} successful, {failed_batches} failed batches")
                            
                except Exception as e:
                    if getattr(args, 'verbose', False):
                        print(f"Warning: failed to save options to DB for {symbol}: {e}")
                        
    except Exception as e:
        if getattr(args, 'verbose', False):
            print(f"Error connecting to HTTP database server: {e}")

async def save_options_via_direct_db(db_save_tasks: list, db_config: str, args) -> None:
    """Save options data via direct database connection."""
    db_instance = None
    try:
        # Create database instance directly to avoid async initialization issues
        from common.questdb_db import StockQuestDB
        enable_cache = not getattr(args, 'no_cache', False)
        redis_url = os.getenv('REDIS_URL', 'redis://localhost:6379/0') if enable_cache else None
        db_instance = StockQuestDB(db_config, auto_init=False, enable_cache=enable_cache, redis_url=redis_url)
        
        # Manually initialize the database connection with retry logic
        max_retries = 3
        for attempt in range(max_retries):
            try:
                if getattr(args, 'verbose', False):
                    print(f"Initializing database connection (attempt {attempt + 1}/{max_retries})...")
                await db_instance._init_db()
                if getattr(args, 'verbose', False):
                    print("Database connection initialized successfully")
                break
            except Exception as e:
                if attempt < max_retries - 1:
                    if getattr(args, 'verbose', False):
                        print(f"Database initialization attempt {attempt + 1} failed: {e}, retrying...")
                    await asyncio.sleep(2)  # Wait 2 seconds before retry
                else:
                    if getattr(args, 'verbose', False):
                        print(f"Database initialization failed after {max_retries} attempts: {e}")
                    raise e
        
        for task in db_save_tasks:
            symbol = task['symbol']
            df = task['df']
            contracts_count = task['contracts_count']
            
            try:
                await db_instance.save_options_data(df=df, ticker=symbol)
                if getattr(args, 'verbose', False):
                    print(f"Successfully saved {contracts_count} options contracts to database for {symbol}")
                
                # Update Redis cache with the current timestamp
                enable_cache = not getattr(args, 'no_cache', False)
                if enable_cache:
                    redis_url = os.getenv('REDIS_URL', 'redis://localhost:6379/0')
                    redis_client = get_redis_client_for_refresh(redis_url) if redis_url else None
                    if redis_client:
                        from datetime import datetime, timezone
                        now_utc = datetime.now(timezone.utc)
                        set_redis_last_write_timestamp(redis_client, symbol, now_utc, ttl_seconds=86400)
            except Exception as e:
                if getattr(args, 'verbose', False):
                    print(f"Warning: failed to save options to DB for {symbol}: {e}")
                    
    except Exception as e:
        if getattr(args, 'verbose', False):
            print(f"Error connecting to database: {e}")
    finally:
        # Properly close database connection
        if db_instance and hasattr(db_instance, 'close_session') and callable(db_instance.close_session):
            try:
                await db_instance.close_session()
            except Exception as close_e:
                if getattr(args, 'verbose', False):
                    print(f"Warning: error closing DB connection: {close_e}")


def _cancel_executor_futures(futures_map):
    """Cancel any outstanding futures."""
    for fut in list(futures_map.keys()):
        try:
            fut.cancel()
        except Exception:
            pass


def _terminate_process_pool(executor):
    """Forcefully terminate worker processes for a ProcessPoolExecutor."""
    processes = getattr(executor, "_processes", None)
    if not processes:
        return
    for proc in processes.values():
        if proc.is_alive():
            try:
                proc.terminate()
            except Exception:
                pass


async def _execute_month_cluster(
    month_index: int,
    month_start_date: str,
    month_end_date: str,
    ticker_chunks: list[list[str]],
    num_processes: int,
    args: argparse.Namespace,
    api_key: str,
    all_pools: list,
) -> dict:
    """Execute options fetching for one month cluster with allocated processes."""
    from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
    
    dry_run = getattr(args, 'dry_run', False)
    
    # Respect the user's executor type preference
    user_executor_type = getattr(args, 'executor_type', 'thread')
    use_threads = (user_executor_type == 'thread') or (num_processes == 0)
    
    # Determine max workers based on executor type
    if use_threads:
        max_workers = max(1, len(ticker_chunks))
    else:
        max_workers = max(1, num_processes)
    
    executor_type_name = 'thread' if use_threads else 'process'
    
    if getattr(args, 'verbose', False):
        print(f"[Month {month_index + 1}] Processing {len(ticker_chunks)} ticker chunks "
              f"for date range {month_start_date} to {month_end_date} "
              f"using {executor_type_name} executor with {max_workers} workers")
    
    # DRY-RUN MODE: Actually fork processes to show what they would do
    if dry_run:
        total_symbols = sum(len(chunk) for chunk in ticker_chunks)
        if getattr(args, 'verbose', False):
            print(f"[DRY-RUN Month {month_index + 1}] Creating {executor_type_name} executor with {max_workers} workers")
            print(f"[DRY-RUN Month {month_index + 1}] Will process {len(ticker_chunks)} chunks ({total_symbols} total symbols)")
        
        # Actually create the executor and fork processes in dry-run mode
        ExecutorCls = ThreadPoolExecutor if use_threads else ProcessPoolExecutor
        pool = ExecutorCls(max_workers=max_workers)
        all_pools.append(pool)
        
        # Note: For ProcessPoolExecutor, we need to use module-level function for pickling
        use_db_flag = bool(getattr(args, 'use_db', None))
        
        # Submit tasks to actually fork processes
        if getattr(args, 'verbose', False):
            print(f"[DRY-RUN Month {month_index + 1}] Submitting {len(ticker_chunks)} tasks to executor (will fork processes)...")
        
        futures_map = {}
        for chunk_idx, chunk in enumerate(ticker_chunks):
            future = pool.submit(_dry_run_worker_standalone, chunk, month_start_date, chunk_idx + 1, month_index, month_start_date, month_end_date, use_db_flag)
            futures_map[future] = chunk
        
        # Give processes a moment to start and show their PIDs
        if getattr(args, 'verbose', False) and ExecutorCls == ProcessPoolExecutor:
            import time
            time.sleep(0.3)  # Brief pause to let processes start
            if hasattr(pool, '_processes') and pool._processes:
                num_workers = len(pool._processes)
                worker_pids = [p.pid for p in pool._processes.values() if hasattr(p, 'pid')]
                print(f"[DRY-RUN Month {month_index + 1}] ProcessPoolExecutor has {num_workers} worker processes active")
                if worker_pids:
                    print(f"[DRY-RUN Month {month_index + 1}] Worker PIDs: {worker_pids}")
        
        # Wait for all workers to report
        cluster_results = {
            'symbols_processed': 0,
            'save_success_count': 0,
            'save_failure_count': 0,
            'errors': []
        }
        
        for future in as_completed(futures_map):
            try:
                result = future.result()
                cluster_results['symbols_processed'] += result.get('symbols_processed', 0)
                cluster_results['save_success_count'] += result.get('save_success_count', 0)
                cluster_results['save_failure_count'] += result.get('save_failure_count', 0)
            except Exception as e:
                chunk = futures_map[future]
                error_msg = f"Chunk {chunk[:3] if chunk else 'unknown'}...: {str(e)}"
                if getattr(args, 'verbose', False):
                    print(f"[DRY-RUN Month {month_index + 1}] ERROR: {error_msg}", file=sys.stderr)
                cluster_results['errors'].append(error_msg)
        
        # Shutdown pool (processes will exit)
        if pool in all_pools:
            all_pools.remove(pool)
        pool.shutdown(wait=True, cancel_futures=False)
        
        if getattr(args, 'verbose', False):
            print(f"[DRY-RUN Month {month_index + 1}] All workers completed and exited")
        
        return cluster_results
    
    # Create executor
    ExecutorCls = ThreadPoolExecutor if use_threads else ProcessPoolExecutor
    pool = ExecutorCls(max_workers=max_workers)
    all_pools.append(pool)
    
    if getattr(args, 'verbose', False):
        executor_type_display = "ProcessPoolExecutor" if ExecutorCls == ProcessPoolExecutor else "ThreadPoolExecutor"
        print(f"[Month {month_index + 1}] Created {executor_type_display} with max_workers={max_workers} (Main PID: {os.getpid()})")
        if ExecutorCls == ProcessPoolExecutor:
            print(f"[Month {month_index + 1}] Worker processes will be created lazily when tasks are submitted")
            print(f"[Month {month_index + 1}] Up to {max_workers} worker processes will be created for this month")
    
    # Update args with month date range
    month_args = argparse.Namespace(**vars(args))
    month_args.date = month_start_date
    # Adjust max_days_to_expiry to cover the month range
    month_start_dt = datetime.strptime(month_start_date, '%Y-%m-%d')
    month_end_dt = datetime.strptime(month_end_date, '%Y-%m-%d')
    days_in_period = (month_end_dt - month_start_dt).days
    
    # Check if we're in days_ahead mode - if so, limit max_days_to_expiry to the period length
    days_ahead = getattr(args, 'days_ahead', None)
    use_days_mode = days_ahead is not None  # 0 is a valid value meaning "1 day"
    
    # If end_date is provided, limit max_days_to_expiry to the end_date
    end_date = getattr(args, 'end_date', None)
    if end_date:
        end_dt = datetime.strptime(end_date, '%Y-%m-%d')
        # Calculate days from month_start_date to end_date
        days_to_end = (end_dt - month_start_dt).days
        # Limit max_days_to_expiry to not exceed end_date
        # This ensures options expiring after end_date are not fetched
        # max_days_to_expiry creates a +/- window, so we use days_to_end directly
        month_args.max_days_to_expiry = max(1, days_to_end)
        if getattr(args, 'verbose', False):
            print(f"[Month {month_index + 1}] Limiting max_days_to_expiry to {month_args.max_days_to_expiry} (end_date: {end_date}, target_date: {month_start_date}, days_to_end: {days_to_end})", file=sys.stderr)
    elif use_days_mode:
        # In days_ahead mode, limit max_days_to_expiry to the period length (or 1 if single day)
        # This ensures we only fetch options expiring within the requested day range
        month_args.max_days_to_expiry = max(1, days_in_period + 1)  # +1 to include the end day
        if getattr(args, 'verbose', False):
            print(f"[Day {month_index + 1}] Limiting max_days_to_expiry to {month_args.max_days_to_expiry} (days_ahead mode, period length: {days_in_period + 1} days)", file=sys.stderr)
    else:
        month_args.max_days_to_expiry = max(days_in_period + 1, getattr(args, 'max_days_to_expiry', 30))
    
    futures_map = {}
    cluster_results = {
        'symbols_processed': 0,
        'save_success_count': 0,
        'save_failure_count': 0,
        'errors': []
    }
    
    try:
        # Submit all ticker chunks
        if getattr(args, 'verbose', False):
            print(f"[Month {month_index + 1}] Submitting {len(ticker_chunks)} ticker chunk tasks to executor...")
        for chunk in ticker_chunks:
            future = pool.submit(_run_for_ticker_chunk, chunk, month_start_date, month_args, api_key)
            futures_map[future] = chunk
        
        if getattr(args, 'verbose', False) and ExecutorCls == ProcessPoolExecutor:
            # ProcessPoolExecutor creates workers lazily, so they should be created now
            import time
            time.sleep(1.0)  # Give processes more time to start
            if hasattr(pool, '_processes') and pool._processes:
                num_workers = len(pool._processes)
                worker_pids = [p.pid for p in pool._processes.values() if hasattr(p, 'pid')]
                print(f"[Month {month_index + 1}] ProcessPoolExecutor has {num_workers} worker processes active (expected: up to {max_workers}, tasks: {len(ticker_chunks)})")
                if worker_pids:
                    print(f"[Month {month_index + 1}] Worker PIDs: {worker_pids}")
                if num_workers < min(max_workers, len(ticker_chunks)):
                    print(f"[Month {month_index + 1}] NOTE: ProcessPoolExecutor creates workers lazily. Only {num_workers} workers created for {len(ticker_chunks)} tasks (max_workers={max_workers})")
            else:
                print(f"[Month {month_index + 1}] WARNING: ProcessPoolExecutor worker processes not yet visible (they're created lazily)")
        
        # Process results as they complete
        if getattr(args, 'verbose', False):
            print(f"[Month {month_index + 1}] Processing results as tasks complete...")
        for future in as_completed(futures_map):
            try:
                result = future.result()
                cluster_results['symbols_processed'] += result.get('symbols_processed', 0)
                cluster_results['save_success_count'] += result.get('save_success_count', 0)
                cluster_results['save_failure_count'] += result.get('save_failure_count', 0)
                cluster_results['errors'].extend(result.get('errors', []))
            except Exception as e:
                chunk = futures_map[future]
                cluster_results['errors'].append(f"Chunk {chunk[:3]}...: {str(e)}")
    except KeyboardInterrupt:
        if getattr(args, 'verbose', False):
            print(f"\n[Month {month_index + 1}] KeyboardInterrupt received, cancelling tasks...", file=sys.stderr)
        for future in futures_map:
            future.cancel()
        pool.shutdown(wait=False, cancel_futures=True)
        raise
    finally:
        if pool in all_pools:
            all_pools.remove(pool)
        pool.shutdown(wait=True, cancel_futures=False)
    
    if getattr(args, 'verbose', False):
        print(f"[Month {month_index + 1}] Completed: {cluster_results['symbols_processed']} symbols, "
              f"{cluster_results['save_success_count']} saves successful, "
              f"{cluster_results['save_failure_count']} saves failed")
    
    return cluster_results


async def _execute_options_iteration(
    symbols_list: list[str],
    args: argparse.Namespace,
    api_key: str,
    all_pools: list | None = None,
) -> dict:
    """
    Execute one iteration of options fetching (without sleeping) and return summary data.
    Checks if tickers were recently fetched and skips them if data is fresh.
    Supports multi-month processing with weighted process allocation.
    """
    from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
    
    if all_pools is None:
        all_pools = []
    
    dry_run = getattr(args, 'dry_run', False)
    days_ahead = getattr(args, 'days_ahead', None)
    months_ahead = getattr(args, 'months_ahead', 0)
    ticker_chunk_size = getattr(args, 'ticker_chunk_size', 250)
    
    # Check if days_ahead is provided (takes precedence over months_ahead)
    # days_ahead=0 means "only the start date" (1 day), days_ahead=None means "not specified"
    use_days_mode = days_ahead is not None  # 0 is a valid value meaning "1 day"
    
    # If days_ahead is not specified and months_ahead is 0 or not set, use single-date mode (backward compatibility)
    if not use_days_mode and months_ahead <= 0:
        return await _execute_options_iteration_single_date(symbols_list, args, api_key, all_pools)
    
    # Multi-period mode: split by time and tickers
    start_date = getattr(args, 'start_date', None) or args.date
    end_date = getattr(args, 'end_date', None)
    
    if use_days_mode:
        # If days_ahead is 0, treat it as 1 (only the start date)
        if days_ahead == 0:
            days_ahead = 1
        # Days-ahead mode
        # If end_date is provided, calculate days_ahead from start_date to end_date
        if end_date:
            start_dt = datetime.strptime(start_date, '%Y-%m-%d')
            end_dt = datetime.strptime(end_date, '%Y-%m-%d')
            days_diff = (end_dt - start_dt).days + 1  # +1 to include both start and end dates
            if getattr(args, 'verbose', False):
                print(f"--end-date specified ({end_date}), calculating days_ahead: {days_diff} (from {start_date} to {end_date})", file=sys.stderr)
            days_ahead = days_diff
        
        date_ranges = generate_day_ranges(start_date, days_ahead, end_date)
        num_periods = days_ahead
        period_label = "day"
        period_label_plural = "days"
    else:
        # Months-ahead mode (original behavior)
        # If end_date is provided, calculate months_ahead from start_date to end_date
        if end_date:
            start_dt = datetime.strptime(start_date, '%Y-%m-%d')
            end_dt = datetime.strptime(end_date, '%Y-%m-%d')
            days_diff = (end_dt - start_dt).days
            # Calculate number of 30-day periods needed, rounding up
            calculated_months_ahead = max(1, (days_diff + 29) // 30)  # +29 to round up
            if getattr(args, 'verbose', False):
                print(f"--end-date specified ({end_date}), calculating months_ahead: {calculated_months_ahead} (from {start_date} to {end_date}, {days_diff} days)", file=sys.stderr)
            months_ahead = calculated_months_ahead
        
        date_ranges = generate_month_ranges(start_date, months_ahead, end_date)
        num_periods = months_ahead
        period_label = "month"
        period_label_plural = "months"
    
    # Allocate processes to months first to determine optimal chunk size
    if args.max_concurrent and args.max_concurrent > 0:
        total_processes = args.max_concurrent
    else:
        total_processes = (os.cpu_count() or 1) if args.executor_type == 'process' else (os.cpu_count() or 1) * 5
    
    process_allocations = allocate_processes_by_proximity(total_processes, num_periods)
    
    # Adjust chunk size to ensure we have enough chunks to utilize all allocated processes
    # Each period needs at least as many chunks as it has allocated processes
    max_processes_per_period = max(process_allocations) if process_allocations else 1
    # Ensure we have at least max_processes_per_period chunks, but respect user's max chunk size
    optimal_chunk_size = min(ticker_chunk_size, max(1, len(symbols_list) // max(max_processes_per_period, 1)))
    
    # Split tickers into chunks
    ticker_chunks = split_tickers_into_chunks(symbols_list, optimal_chunk_size)
    
    # Warn if we had to adjust chunk size
    if optimal_chunk_size < ticker_chunk_size and getattr(args, 'verbose', False):
        period_name = period_label.capitalize() if num_periods == 1 else period_label_plural.capitalize()
        print(f"  NOTE: Adjusted chunk size from {ticker_chunk_size} to {optimal_chunk_size} to create {len(ticker_chunks)} chunks (needed for {max_processes_per_period} processes in {period_name} 1)")
    
    if getattr(args, 'verbose', False):
        mode_label = f"MULTI-{period_label_plural.upper()}" if num_periods > 1 else f"SINGLE-{period_label.upper()}"
        print(f"\n[{mode_label} MODE] Processing {len(symbols_list)} tickers across {num_periods} {period_label_plural if num_periods > 1 else period_label}")
        print(f"  Ticker chunks: {len(ticker_chunks)} (max {ticker_chunk_size} per chunk)")
        print(f"  Total processes: {total_processes} (executor type: {args.executor_type})")
        print(f"  Process allocation by {period_label} (all {period_label_plural} run in parallel):")
        total_allocated = 0
        max_actual_workers = 0
        for i, (date_range, proc_count) in enumerate(zip(date_ranges, process_allocations)):
            # ProcessPoolExecutor will only create as many workers as there are tasks
            actual_workers = min(proc_count, len(ticker_chunks))
            max_actual_workers += actual_workers
            period_name = f"{period_label.capitalize()} {i+1}" if num_periods > 1 else period_label.capitalize()
            print(f"    {period_name} ({date_range[0]} to {date_range[1]}): {proc_count} processes allocated, {actual_workers} actual workers (limited by {len(ticker_chunks)} chunks)")
            total_allocated += proc_count
        print(f"  Total processes allocated: {total_allocated} (should equal {total_processes})")
        print(f"  Maximum actual worker processes: {max_actual_workers} (limited by {len(ticker_chunks)} chunks per {period_label})")
        if args.executor_type == 'process':
            print(f"  Note: ProcessPoolExecutor creates worker processes lazily when tasks are submitted.")
            print(f"  Since there are only {len(ticker_chunks)} chunks per {period_label}, each {period_label} can use at most {len(ticker_chunks)} workers.")
            print(f"  To use more processes, increase --ticker-chunk-size to create more chunks, or reduce --max-concurrent.")
            print(f"  You should see up to {max_actual_workers} worker processes in 'ps aux' once tasks start.")
    
    # Execute each month cluster in parallel
    overall_results = {
        'symbols_processed': 0,
        'save_success_count': 0,
        'save_failure_count': 0,
        'errors': []
    }
    
    async def run_period_cluster(period_index: int, period_start: str, period_end: str) -> dict:
        """Wrapper to run a period cluster and handle errors."""
        try:
            return await _execute_month_cluster(
                month_index=period_index,
                month_start_date=period_start,
                month_end_date=period_end,
                ticker_chunks=ticker_chunks,
                num_processes=process_allocations[period_index],
                args=args,
                api_key=api_key,
                all_pools=all_pools
            )
        except KeyboardInterrupt:
            if getattr(args, 'verbose', False):
                period_name = f"{period_label.capitalize()} {period_index + 1}" if num_periods > 1 else period_label.capitalize()
                print(f"\n[MULTI-{period_label_plural.upper()}] KeyboardInterrupt at {period_name}, stopping...", file=sys.stderr)
            raise
        except Exception as e:
            if getattr(args, 'verbose', False):
                period_name = f"{period_label.capitalize()} {period_index + 1}" if num_periods > 1 else period_label.capitalize()
                print(f"\n[MULTI-{period_label_plural.upper()}] Error in {period_name}: {e}", file=sys.stderr)
            period_name = f"{period_label.capitalize()} {period_index + 1}" if num_periods > 1 else period_label.capitalize()
            return {
                'symbols_processed': 0,
                'save_success_count': 0,
                'save_failure_count': 0,
                'errors': [f"{period_name}: {str(e)}"]
            }
    
    # Run all period clusters in parallel (truly concurrent)
    if getattr(args, 'verbose', False):
        print(f"  Running all {num_periods} {period_label_plural} clusters in parallel...")
        print(f"  Main process PID: {os.getpid()}")
        print(f"  All {period_label_plural} clusters will start simultaneously via asyncio.gather()")
    
    # Create all tasks first, then await them all together to ensure true parallelism
    period_tasks = [
        asyncio.create_task(run_period_cluster(period_index, period_start, period_end))
        for period_index, (period_start, period_end) in enumerate(date_ranges)
    ]
    
    if getattr(args, 'verbose', False):
        print(f"  Created {len(period_tasks)} async tasks - all will execute concurrently")
        print(f"  Expected total worker processes: {sum(process_allocations)} (across all {period_label_plural})")
        print(f"  To verify parallel execution, run: ps aux | grep -E 'Python|python' | grep fetch_options")
        print(f"  Or check all Python processes: ps aux | grep -E 'Python|python' | grep -v grep")
        print(f"  You should see 1 main process + up to {sum(process_allocations)} worker processes")
        print(f"  Note: ProcessPoolExecutor creates workers lazily when tasks are submitted")
        print(f"  All {period_label_plural} will start simultaneously, but workers may appear gradually")
    
    # Wait for all periods to complete
    try:
        period_results_list = await asyncio.gather(*period_tasks, return_exceptions=True)
        
        # Aggregate results
        for period_results in period_results_list:
            if isinstance(period_results, Exception):
                overall_results['errors'].append(f"{period_label.capitalize()} cluster error: {str(period_results)}")
            else:
                overall_results['symbols_processed'] += period_results.get('symbols_processed', 0)
                overall_results['save_success_count'] += period_results.get('save_success_count', 0)
                overall_results['save_failure_count'] += period_results.get('save_failure_count', 0)
                overall_results['errors'].extend(period_results.get('errors', []))
    except KeyboardInterrupt:
        if getattr(args, 'verbose', False):
            print(f"\n[MULTI-{period_label_plural.upper()}] KeyboardInterrupt: Cancelling all {period_label_plural} clusters...", file=sys.stderr)
        for task in period_tasks:
            task.cancel()
        await asyncio.gather(*month_tasks, return_exceptions=True)
        raise
    
    if getattr(args, 'verbose', False):
        print(f"\n[MULTI-MONTH SUMMARY] Total: {overall_results['symbols_processed']} symbols processed, "
              f"{overall_results['save_success_count']} saves successful, "
              f"{overall_results['save_failure_count']} saves failed")
        if overall_results['errors']:
            print(f"  Errors: {len(overall_results['errors'])}")
            if getattr(args, 'debug', False) or len(overall_results['errors']) > 0:
                # Show first few errors for debugging
                for i, error in enumerate(overall_results['errors'][:10]):
                    print(f"    Error {i+1}: {error}")
                if len(overall_results['errors']) > 10:
                    print(f"    ... and {len(overall_results['errors']) - 10} more errors")
    
    return {
        "symbols_processed": overall_results['symbols_processed'],
        "db_tasks": overall_results['save_success_count'],
    }


async def _execute_options_iteration_single_date(
    symbols_list: list[str],
    args: argparse.Namespace,
    api_key: str,
    all_pools: list | None = None,
) -> dict:
    """Execute one iteration for a single date (original behavior, backward compatible)."""
    from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
    
    if all_pools is None:
        all_pools = []
    
    dry_run = getattr(args, 'dry_run', False)
    
    # Check if we should skip recently fetched tickers
    symbols_to_fetch = symbols_list
    refresh_threshold_seconds = getattr(args, 'refresh_threshold_seconds', None)
    
    if refresh_threshold_seconds is None:
        # Calculate refresh threshold based on next fetch interval (if in continuous mode)
        if getattr(args, 'continuous', False):
            now_utc = datetime.now(timezone.utc)
            is_market_open = HistoricalDataFetcher._is_market_open(now_utc)
            if is_market_open:
                # Market open: use market_open cache duration (20 minutes)
                refresh_threshold_seconds = HistoricalDataFetcher.CACHE_DURATION_MINUTES['market_open'] * 60
            else:
                # Market closed: use market_closed cache duration (360 minutes)
                refresh_threshold_seconds = HistoricalDataFetcher.CACHE_DURATION_MINUTES['market_closed'] * 60
        else:
            # Single run: use market_open cache duration as default (20 minutes)
            refresh_threshold_seconds = HistoricalDataFetcher.CACHE_DURATION_MINUTES['market_open'] * 60
    elif getattr(args, 'verbose', False):
        print(f"Using user-specified refresh threshold: {refresh_threshold_seconds}s")
    
    # Check if tickers need refresh (only if using database and not forcing fresh fetch)
    if getattr(args, 'use_db', None) and refresh_threshold_seconds and not getattr(args, 'force_fresh', False):
        try:
            enable_cache = not getattr(args, 'no_cache', False)
            redis_url = os.getenv('REDIS_URL', 'redis://localhost:6379/0') if enable_cache else None
            redis_client = get_redis_client_for_refresh(redis_url) if redis_url else None
            
            # Determine database type based on connection string
            db_type = 'questdb'  # default
            db_config = args.use_db
            if args.use_db.startswith('http://') or args.use_db.startswith('https://'):
                db_type = 'remote'
                # StockDBClient expects "host:port" format, not "http://host:port"
                db_config = args.use_db.replace('http://', '').replace('https://', '')
            elif args.use_db.startswith('postgresql://'):
                db_type = 'postgresql'
            
            # Create database connection for timestamp checking
            db = get_stock_db(db_type, db_config=db_config, enable_cache=enable_cache, redis_url=redis_url)
            async with db:
                # Create a wrapper function to fetch timestamps for multiple tickers
                async def fetch_timestamps(tickers: list[str], cache: dict) -> dict:
                    """Fetch latest option timestamps for multiple tickers."""
                    result = {}
                    debug_mode = getattr(args, 'debug', False)
                    for ticker in tickers:
                        age = await fetch_latest_option_timestamp_standalone(db, ticker, cache, redis_client=redis_client, debug=debug_mode)
                        result[ticker] = age
                        if debug_mode:
                            from datetime import datetime, timezone
                            now_utc = datetime.now(timezone.utc)
                            cached_ts = cache.get(ticker)
                            print(f"DEBUG [fetch_timestamps]: {ticker} - age={age}, now_utc={now_utc}, cached_ts={cached_ts}", file=sys.stderr)
                    return result
                
                # Check which tickers need refresh
                # Enable debug for timestamp checking if --debug flag is set
                debug_mode = getattr(args, 'debug', False)
                symbols_to_fetch = await check_tickers_for_refresh(
                    db=db,
                    tickers=symbols_list,
                    refresh_threshold_seconds=refresh_threshold_seconds,
                    fetch_timestamp_func=fetch_timestamps,
                    redis_client=redis_client,
                    timestamp_cache=None,
                    min_write_timestamp=None,
                    debug=debug_mode
                )
                
                if getattr(args, 'verbose', False) and len(symbols_to_fetch) < len(symbols_list):
                    skipped = len(symbols_list) - len(symbols_to_fetch)
                    print(f"Skipping {skipped} ticker(s) with fresh data (threshold: {refresh_threshold_seconds}s)")
        except Exception as e:
            # If checking fails, fetch all tickers
            if getattr(args, 'verbose', False):
                import traceback
                print(f"Warning: Could not check ticker freshness ({e}). Fetching all tickers.", file=sys.stderr)
                if getattr(args, 'debug', False):
                    traceback.print_exc(file=sys.stderr)
            symbols_to_fetch = symbols_list
    elif getattr(args, 'force_fresh', False):
        # Force fresh is enabled, skip refresh check and fetch all symbols
        symbols_to_fetch = symbols_list
        if getattr(args, 'verbose', False):
            print("--force-fresh enabled: Skipping refresh check, will fetch all symbols from Polygon API")
    
    if not symbols_to_fetch:
        if getattr(args, 'verbose', False):
            print("All tickers have fresh data. Skipping fetch.")
        return {
            "symbols_processed": 0,
            "db_tasks": 0,
        }

    exec_type = args.executor_type
    if args.max_concurrent and args.max_concurrent > 0:
        max_workers = args.max_concurrent
    else:
        max_workers = (os.cpu_count() or 1) if exec_type == 'process' else (os.cpu_count() or 1) * 5

    # DRY-RUN MODE: Actually fork processes to show what they would do
    if dry_run:
        if getattr(args, 'verbose', False):
            print(f"\n[DRY-RUN] Creating {exec_type} executor with {max_workers} workers")
            print(f"[DRY-RUN] Will process {len(symbols_to_fetch)} symbols for date: {args.date}")
        
        # Actually create the executor and fork processes in dry-run mode
        ExecutorCls = ProcessPoolExecutor if exec_type == 'process' else ThreadPoolExecutor
        pool = ExecutorCls(max_workers=max_workers)
        if all_pools is not None:
            all_pools.append(pool)
        
        # Create a dry-run worker function that just reports what it would do
        def _dry_run_symbol_worker(symbol: str, target_date: str) -> dict:
            """Dry-run worker that reports what it would process."""
            import os
            pid = os.getpid()
            print(f"[DRY-RUN Worker PID {pid}] Would process symbol: {symbol}")
            print(f"[DRY-RUN Worker PID {pid}] Would fetch options for date: {target_date}")
            print(f"[DRY-RUN Worker PID {pid}] Would save to DB: {bool(getattr(args, 'use_db', None))}")
            print(f"[DRY-RUN Worker PID {pid}] All executions would be successful (simulated)")
            return {
                'formatted_output': f"[DRY-RUN] {symbol} - simulated success",
                'db_save_data': None,
                'save_success': True if getattr(args, 'use_db', None) else False
            }
        
        # Submit tasks to actually fork processes
        futures_map = {}
        for symbol in symbols_to_fetch:
            future = pool.submit(_dry_run_symbol_worker, symbol, args.date)
            futures_map[future] = symbol
        
        # Wait for all workers to report
        save_count = 0
        for future in as_completed(futures_map):
            try:
                result = future.result()
                if result.get('save_success'):
                    save_count += 1
            except Exception as e:
                symbol = futures_map[future]
                if getattr(args, 'verbose', False):
                    print(f"[DRY-RUN] Error for {symbol}: {e}")
        
        # Shutdown pool (processes will exit)
        pool.shutdown(wait=True, cancel_futures=False)
        if all_pools is not None and pool in all_pools:
            all_pools.remove(pool)
        
        if getattr(args, 'verbose', False):
            print(f"[DRY-RUN] All workers completed and exited")
        
        return {
            "symbols_processed": len(symbols_to_fetch),
            "db_tasks": save_count,
        }

    if getattr(args, 'verbose', False):
        print(
            f"Starting concurrent fetch for {len(symbols_to_fetch)} symbols "
            f"using {exec_type} executor with max_workers={max_workers}"
        )

    ExecutorCls = ProcessPoolExecutor if exec_type == 'process' else ThreadPoolExecutor
    
    # Track save results (saves now happen in worker processes, so we just track success/failure)
    save_count = 0
    save_failures = 0
    
    pool = ExecutorCls(max_workers=max_workers)
    if all_pools is not None:
        all_pools.append(pool)
    shutdown_called = False
    futures_map = {}
    try:
        for symbol in symbols_to_fetch:
            futures_map[pool.submit(_run_for_symbol, symbol, args, api_key)] = symbol
        for fut in as_completed(futures_map):
            result = fut.result()
            if isinstance(result, dict) and 'formatted_output' in result:
                if getattr(args, 'verbose', False) and result['formatted_output']:
                    print(result['formatted_output'])
                
                # Track save results (saves already happened in worker process)
                if result.get('save_success'):
                    save_count += 1
                elif getattr(args, 'use_db', None) and result.get('save_success') is False:
                    save_failures += 1
            elif getattr(args, 'verbose', False) and result:
                print(result)
    except KeyboardInterrupt:
        if getattr(args, 'verbose', False):
            print("\nKeyboardInterrupt received. Attempting to cancel outstanding option fetch tasks...", file=sys.stderr)
        _cancel_executor_futures(futures_map)
        pool.shutdown(wait=False, cancel_futures=True)
        shutdown_called = True
        if ExecutorCls is ProcessPoolExecutor:
            _terminate_process_pool(pool)
        raise
    finally:
        if not shutdown_called:
            pool.shutdown(wait=True, cancel_futures=False)
        if all_pools is not None and pool in all_pools:
            all_pools.remove(pool)
    
    if getattr(args, 'verbose', False) and getattr(args, 'use_db', None):
        print(f"\n[SUMMARY] Database saves: {save_count} successful, {save_failures} failed (saves performed in worker processes)")

    return {
        "symbols_processed": len(symbols_to_fetch),
        "db_tasks": save_count,  # Count successful saves
    }


def _options_iteration_worker(
    symbols_list: list[str],
    args_dict: dict,
    api_key: str,
) -> dict:
    """
    Wrapper suitable for running an options iteration inside a subprocess.
    """
    iteration_args = argparse.Namespace(**args_dict)

    async def _runner() -> dict:
        return await _execute_options_iteration(symbols_list, iteration_args, api_key, all_pools=[])

    return asyncio.run(_runner())


async def main():
    """Main function to run the data fetcher."""
    if not POLYGON_AVAILABLE:
        print("Error: polygon-api-client is not installed.", file=sys.stderr)
        print("Please install it with: pip install polygon-api-client", file=sys.stderr)
        sys.exit(1)

    parser = argparse.ArgumentParser(
        description="Fetch historical stock and options data for a specific date from Polygon.io.",
        epilog="""
Examples:
  # Get data for a specific date (single symbol)
  python historical_stock_options.py AAPL --date 2024-06-05

  # Get data for multiple symbols
  python historical_stock_options.py --symbols AAPL MSFT GOOGL --date 2024-06-05

  # Get data from YAML file
  python historical_stock_options.py --symbols-list symbols.yaml --date 2024-06-05

  # Get data for S&P 500 stocks
  python historical_stock_options.py --types sp-500 --date 2024-06-05

  # Get data for today (default) and show only calls
  python historical_stock_options.py TSLA --option-type call

  # Show puts within 10% of the close price, expiring within 90 days
  python historical_stock_options.py GOOGL --date 2024-05-01 --option-type put --strike-range-percent 10 --max-days-to-expiry 90

  # Get historical data for a past date, including expired contracts (can be slow)
  python historical_stock_options.py TQQQ --date 2024-05-01 --max-days-to-expiry 14 --include-expired

  # Save to QuestDB database
  python historical_stock_options.py AAPL --date 2024-06-05 --db-path questdb://user:password@localhost:8812/stock_data

  # Save to HTTP database server (db_server.py)
  python historical_stock_options.py MSFT --date 2024-06-05 --db-path localhost:9002
  
  # Save to direct QuestDB connection (explicit protocol)
  python historical_stock_options.py MSFT --date 2024-06-05 --db-path questdb://user:password@localhost:8812/stock_data

  # Save with custom batch size for large datasets (default: 100)
  python historical_stock_options.py SPX --date 2024-06-05 --db-path localhost:9002 --db-batch-size 50

  # Force fresh fetch from Polygon API and save to database (bypasses cache)
  python historical_stock_options.py AAPL --date 2024-06-05 --force-fresh --use-db questdb://user:pass@localhost:8812/db

  # Quiet mode - suppress output but still save CSV files
  python historical_stock_options.py --symbols AAPL MSFT --date 2024-06-05 --quiet

  # Fetch once before waiting for market open (useful since option prices don't change during non-market hours)
  python historical_stock_options.py AAPL --fetch-once-before-wait --continuous
"""
    )
    
    # Add symbol input arguments using common library (disable positional symbols)
    add_symbol_arguments(parser, required=True, allow_positional=False)
    
    parser.add_argument(
        '--date',
        default=datetime.now().strftime('%Y-%m-%d'),
        help="Fetch options for a specific date in YYYY-MM-DD format (default: today). When --months-ahead is 0, fetches data for this single date. When --months-ahead > 0, uses this as the start date for multi-month mode."
    )
    parser.add_argument(
        '--start-date',
        type=str,
        default=None,
        help="Explicit start date in YYYY-MM-DD format. Overrides --date if provided. Primarily used for multi-month mode."
    )
    parser.add_argument(
        '--end-date',
        type=str,
        default=None,
        help="Stop fetching at this date in YYYY-MM-DD format. When provided, limits the date range regardless of --months-ahead."
    )
    parser.add_argument(
        '--months-ahead',
        type=int,
        default=6,
        help="Number of 30-day periods to fetch ahead from start date (default: 6). Set to 0 for single-date mode. Ignored if --end-date is provided. Ignored if --days-ahead is provided."
    )
    parser.add_argument(
        '--days-ahead',
        type=int,
        default=None,
        help="Number of days to fetch ahead from start date. If provided, this takes precedence over --months-ahead. Ignored if --end-date is provided."
    )
    parser.add_argument(
        '--ticker-chunk-size',
        type=int,
        default=250,
        help="Maximum number of tickers per chunk when processing in multi-month mode (default: 250)."
    )
    parser.add_argument(
        '--option-type',
        choices=['call', 'put', 'all'],
        default='all',
        help="Type of options to display (default: all)."
    )
    parser.add_argument(
        '--strike-range-percent',
        type=int,
        default=None,
        help="Show options with strikes within this percentage of the stock's closing price (e.g., 10 for ±10%%)."
    )
    parser.add_argument(
        '--options-per-expiry',
        type=int,
        default=5,
        help="Number of options to show on each side of the stock price (default: 5)."
    )
    parser.add_argument(
        '--max-days-to-expiry',
        type=int,
        default=30,
        help="Creates a +/- window around the target date for option expirations to show (e.g., 30 for ±30 days)."
    )
    parser.add_argument(
        '--include-expired',
        action='store_true',
        help="Include expired options in the search (can be much slower)."
    )
    parser.add_argument(
        '--data-dir',
        default='data',
        help="Directory to store CSV data files (default: data)."
    )
    # CSV read/write control
    parser.add_argument(
        '--use-csv',
        action='store_true',
        help="Enable CSV cache: read fresh CSV if present and write new snapshots to CSV."
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        default=False,
        help="Enable verbose output (default: quiet mode)."
    )
    # Remove old CSV flags (backward compatibility shim)
    parser.add_argument(
        '--use-csv-cache',
        action='store_true',
        help=argparse.SUPPRESS
    )
    parser.add_argument(
        '--save-to-csv',
        action='store_true',
        help=argparse.SUPPRESS
    )
    parser.add_argument(
        '--display-saved',
        action='store_true',
        help="Display all saved options from the database after fetching (non-continuous mode only)."
    )
    parser.add_argument(
        '--straddle-view',
        action='store_true',
        help="When used with --display-saved, show options in straddle format (matching calls and puts at same strike)."
    )
    parser.add_argument(
        '--save-saved-to-csv',
        action='store_true',
        help="Save all saved options from the database to CSV files after fetching (non-continuous mode only)."
    )
    parser.add_argument(
        '--save-saved-to-csv-file',
        type=str,
        default=None,
        help="When used with --save-saved-to-csv, save all options to a single CSV file with this name. If not specified, saves to separate files by expiration date."
    )
    parser.add_argument(
        '--continuous',
        action='store_true',
        help="Continuously fetch in a loop, sleeping based on cache duration."
    )
    parser.add_argument(
        '--continous',
        action='store_true',
        dest='continuous',
        help="Alias for --continuous (common typo)."
    )
    parser.add_argument(
        '--interval-multiplier',
        type=float,
        default=1.0,
        help="Multiplier for cadence-based intervals (e.g., 0.5 twice as fast, 2.0 half as often)."
    )
    parser.add_argument(
        '--refresh-threshold-seconds',
        type=int,
        default=None,
        help="Override the automatic ticker refresh threshold (seconds) when deciding whether to re-fetch data."
    )
    parser.add_argument(
        '--continuous-max-runs',
        type=int,
        default=None,
        help="Maximum number of continuous runs before stopping (default: unlimited)."
    )
    parser.add_argument(
        '--max-concurrent',
        type=int,
        default=2,
        help="Max number of symbols to process concurrently (default: CPU count for processes, CPU*5 for threads)."
    )
    parser.add_argument(
        '--executor-type',
        choices=['thread', 'process'],
        default='thread',
        help="Executor type for per-symbol concurrency (default: thread)."
    )
    parser.add_argument(
        '--snapshot-max-concurrent',
        type=int,
        default=5,
        help="Max concurrent per-contract snapshot requests within a symbol (0 disables)."
    )
    parser.add_argument(
        '--use-db',
        type=str,
        default=None,
        help="QuestDB connection string to enable DB read/write (e.g., questdb://user:pass@host:8812/db)."
    )
    parser.add_argument(
        '--db-path',
        type=str,
        nargs='+',
        default=None,
        help="Path to the local database file (SQLite/DuckDB) or remote server address (host:port). Type is inferred from format. Host:port format connects to HTTP server (db_server.py), questdb:// or postgresql:// connects directly to database. Can specify multiple databases. Overrides --use-db if provided."
    )
    parser.add_argument(
        '--db-batch-size',
        type=int,
        default=500,
        help="Maximum number of records to send in a single database request when using HTTP server (default: 500). Large datasets will be automatically split into batches."
    )
    parser.add_argument(
        '--fetch-once-before-wait',
        action='store_true',
        help="If market is closed, fetch once immediately before waiting for market open. Useful since option prices don't change during non-market hours."
    )
    parser.add_argument(
        '--force-fresh',
        action='store_true',
        help="Force fresh fetch from Polygon API, bypassing both database cache and CSV cache. Use with --use-db to save to database."
    )
    parser.add_argument(
        '--db-only',
        action='store_true',
        help="Only serve data from database and/or cache. Do not make external API calls. Returns an error if data is not found in DB/cache."
    )
    parser.add_argument(
        '--no-cache',
        action='store_true',
        help="Disable Redis caching for QuestDB operations (default: cache enabled)"
    )
    parser.add_argument(
        '--db-timeout',
        type=float,
        default=60.0,
        help="Timeout in seconds for database requests (default: 60.0). Increase for large queries or slow connections."
    )
    parser.add_argument(
        '--debug',
        action='store_true',
        help="Enable debug output with detailed information about data fetching and timestamp checking."
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help="Dry-run mode: actually fork processes and log what would be executed without actually running. Shows process allocation, ticker chunks, and execution sequence."
    )
    # Backward-compatibility (hidden): legacy --db-conn maps to --use-db
    parser.add_argument(
        '--db-conn',
        type=str,
        default=None,
        help=argparse.SUPPRESS
    )
    
    # Use parse_known_args to handle --types with subtraction (e.g., -stocks_to_track)
    # which argparse might interpret as a flag
    args, unknown = parser.parse_known_args()
    
    # Post-process to merge unknown args that are part of --types
    if hasattr(args, 'types') and args.types:
        from common.symbol_loader import post_process_types_argument
        post_process_types_argument(args, parser, unknown)

    # Handle start_date: if provided, override date
    if getattr(args, 'start_date', None):
        print(f"--start-date specified ({args.start_date}), overriding --date", file=sys.stderr)
        args.date = args.start_date
    elif getattr(args, 'date', None) and args.months_ahead == 0:
        print(f"--date specified ({args.date}) with --months-ahead 0 (single-date mode)", file=sys.stderr)
    elif getattr(args, 'date', None):
        print(f"--date specified ({args.date}) as start date for multi-month mode", file=sys.stderr)

    # Validate conflicting arguments
    if getattr(args, 'db_only', False) and getattr(args, 'force_fresh', False):
        parser.error("--db-only and --force-fresh cannot be used together. --db-only prevents API calls, while --force-fresh forces API calls.")
    
    # Backward-compatibility: map deprecated flags to new consolidated flags
    if getattr(args, 'use_csv_cache', False) or getattr(args, 'save_to_csv', False):
        args.use_csv = True
    if getattr(args, 'db_conn', None) and not getattr(args, 'use_db', None):
        args.use_db = args.db_conn
    
    # Process --db-path argument (similar to fetch_all_data.py)
    if args.db_path:
        # Use the first database path if multiple are provided
        db_path = args.db_path[0]
        if ':' in db_path:
            # Check if it's a QuestDB connection string
            if db_path.startswith('questdb://'):
                # QuestDB database - use questdb type
                args.use_db = db_path
                print(f"Using QuestDB database at: {db_path}")
            # Check if it's a PostgreSQL connection string
            elif db_path.startswith('postgresql://'):
                # PostgreSQL database - use postgresql type
                args.use_db = db_path
                print(f"Using PostgreSQL database at: {db_path}")
            else:
                # Remote database (host:port format) - assume HTTP server unless explicitly QuestDB/PostgreSQL
                if not db_path.startswith(('questdb://', 'postgresql://')):
                    # Default to HTTP server connection
                    if ':' in db_path:
                        host, port = db_path.split(':', 1)
                        args.use_db = f"http://{host}:{port}"
                        print(f"Using HTTP database server at: {db_path}")
                    else:
                        # Just host, assume HTTP server on default port
                        args.use_db = f"http://{db_path}:9002"
                        print(f"Using HTTP database server at: {db_path}:9002")
                else:
                    # Explicit QuestDB or PostgreSQL connection string
                    args.use_db = db_path
                    print(f"Using direct database at: {db_path}")
        else:
            # Local database - not supported for options data, warn user
            print(f"Warning: Local database path '{db_path}' is not supported for options data. Options data requires QuestDB or PostgreSQL.", file=sys.stderr)
            args.use_db = None
    
    api_key = os.getenv('POLYGON_API_KEY')
    if not api_key:
        print("Error: POLYGON_API_KEY environment variable not set.", file=sys.stderr)
        sys.exit(1)
        
    try:
        datetime.strptime(args.date, '%Y-%m-%d')
    except ValueError:
        print(f"Error: Invalid date format '{args.date}'. Please use YYYY-MM-DD.", file=sys.stderr)
        sys.exit(1)

    # Get symbols list using common library
    symbols_list = await fetch_lists_data(args, not getattr(args, 'verbose', False))
    if not symbols_list:
        print("No symbols specified or found. Exiting.", file=sys.stderr)
        sys.exit(1)

    if args.continuous:
        # Check market status and wait if needed (only in continuous mode)
        now_utc = datetime.now(timezone.utc)
        is_market_open = common_is_market_hours(now_utc, "America/New_York")
        seconds_to_open, _ = common_compute_market_transition_times(now_utc, "America/New_York")
        
        if not is_market_open and seconds_to_open is not None:
            # Market is closed - handle based on fetch-once-before-wait flag
            if getattr(args, 'fetch_once_before_wait', False):
                # Fetch once immediately before waiting
                if getattr(args, 'verbose', False):
                    print(f"Market is closed. Fetching once immediately before waiting for market open...")
                
                # Use the same iteration function which now handles incremental saves
                if getattr(args, 'verbose', False):
                    print(f"Fetching data for {len(symbols_list)} symbols (one-time fetch before waiting)...")
                
                all_pools = []
                await _execute_options_iteration(symbols_list, args, api_key, all_pools)
                
                # Now wait for market open
                if getattr(args, 'verbose', False):
                    hours_to_wait = seconds_to_open / 3600
                    print(f"One-time fetch completed. Waiting {hours_to_wait:.2f} hours ({seconds_to_open:.0f} seconds) until market opens...")
                
                await asyncio.sleep(seconds_to_open)
                
                # Re-check market status after waiting
                now_utc = datetime.now(timezone.utc)
                is_market_open = common_is_market_hours(now_utc, "America/New_York")
                if getattr(args, 'verbose', False):
                    if is_market_open:
                        print("Market is now open. Proceeding with normal operation...")
                    else:
                        print("Warning: Market is still not open after waiting. Proceeding anyway...")
            else:
                # Wait for market open before starting
                if getattr(args, 'verbose', False):
                    hours_to_wait = seconds_to_open / 3600
                    print(f"Market is closed. Waiting {hours_to_wait:.2f} hours ({seconds_to_open:.0f} seconds) until market opens before starting...")
                
                await asyncio.sleep(seconds_to_open)
                
                # Re-check market status after waiting
                now_utc = datetime.now(timezone.utc)
                is_market_open = common_is_market_hours(now_utc, "America/New_York")
                if getattr(args, 'verbose', False):
                    if is_market_open:
                        print("Market is now open. Starting data fetch...")
                    else:
                        print("Warning: Market is still not open after waiting. Proceeding anyway...")

    # Track all pools for graceful shutdown
    all_pools = []
    
    if args.continuous:
        days_ahead = getattr(args, 'days_ahead', None)
        months_ahead = getattr(args, 'months_ahead', 6)  # Default is 6 from argument parser
        
        # Check if days_ahead is provided (takes precedence over months_ahead)
        # days_ahead=0 means "only the start date" (1 day), days_ahead=None means "not specified"
        use_days_mode = days_ahead is not None  # 0 is a valid value meaning "1 day"
        
        # If days_ahead is 0, treat it as 1 (only the start date)
        if use_days_mode and days_ahead == 0:
            days_ahead = 1
        
        # Multi-period continuous mode
        if use_days_mode or months_ahead > 0:
            start_date = getattr(args, 'start_date', None) or args.date
            end_date = getattr(args, 'end_date', None)
            
            if use_days_mode:
                # Days-ahead mode
                # If end_date is provided, calculate days_ahead from start_date to end_date
                if end_date:
                    start_dt = datetime.strptime(start_date, '%Y-%m-%d')
                    end_dt = datetime.strptime(end_date, '%Y-%m-%d')
                    days_diff = (end_dt - start_dt).days + 1  # +1 to include both start and end dates
                    if getattr(args, 'verbose', False):
                        print(f"--end-date specified ({end_date}), calculating days_ahead: {days_diff} (from {start_date} to {end_date})", file=sys.stderr)
                    days_ahead = days_diff
                
                date_ranges = generate_day_ranges(start_date, days_ahead, end_date)
                num_periods = days_ahead
                period_label = "day"
                period_label_plural = "days"
            else:
                # Months-ahead mode (original behavior)
                # If end_date is provided, calculate months_ahead from start_date to end_date
                if end_date:
                    start_dt = datetime.strptime(start_date, '%Y-%m-%d')
                    end_dt = datetime.strptime(end_date, '%Y-%m-%d')
                    days_diff = (end_dt - start_dt).days
                    # Calculate number of 30-day periods needed, rounding up
                    calculated_months_ahead = max(1, (days_diff + 29) // 30)  # +29 to round up
                    if getattr(args, 'verbose', False):
                        print(f"--end-date specified ({end_date}), calculating months_ahead: {calculated_months_ahead} (from {start_date} to {end_date}, {days_diff} days)", file=sys.stderr)
                    months_ahead = calculated_months_ahead
                
                date_ranges = generate_month_ranges(start_date, months_ahead, end_date)
                num_periods = months_ahead
                period_label = "month"
                period_label_plural = "months"
            
            # Calculate exponential intervals for each period (smaller for near periods, larger for far periods)
            # Base interval: 20 minutes for period 1, exponentially increasing up to 60 minutes max
            base_interval_minutes = HistoricalDataFetcher.CACHE_DURATION_MINUTES['market_open']
            max_interval_minutes = 60.0
            period_intervals = []
            for i in range(num_periods):
                # Exponential growth: period 0 gets base, period n-1 gets max
                if num_periods == 1:
                    interval = base_interval_minutes * 60
                else:
                    # Exponential: base * (max/base)^(i/(n-1))
                    ratio = (max_interval_minutes / base_interval_minutes) ** (i / (num_periods - 1))
                    interval = base_interval_minutes * ratio * 60
                period_intervals.append(interval)
            
            if getattr(args, 'verbose', False):
                mode_label = f"CONTINUOUS MULTI-{period_label_plural.upper()}"
                print(f"\n[{mode_label} MODE] Starting {num_periods} {period_label_plural} clusters with exponential intervals:")
                for i, (date_range, interval) in enumerate(zip(date_ranges, period_intervals)):
                    period_name = f"{period_label.capitalize()} {i+1}" if num_periods > 1 else period_label.capitalize()
                    print(f"  {period_name} ({date_range[0]} to {date_range[1]}): {interval/60:.1f} min interval")
            
            # Create async tasks for each period cluster
            # Each period cluster runs independently - when its sleep expires, it starts a new iteration
            # regardless of what other period clusters are doing
            async def period_cluster_loop(period_idx: int, period_start: str, period_end: str, sleep_interval: float):
                """Continuous loop for a single period cluster with market hours awareness.
                
                This function runs independently for each period. When its sleep interval expires,
                it will start a new iteration even if other period clusters are still running.
                Each iteration creates new process pools that are independent of other periods.
                """
                run_num = 0
                market_open_delay_seconds = 2 * 60  # 2 minutes after market opens
                
                # Each period cluster maintains its own pool tracking (local to this async function)
                # This ensures independence - one period's processes don't affect another
                period_pools = []
                
                def get_seconds_until_market_open_plus_delay(now_utc: datetime) -> float | None:
                    """Calculate seconds until market opens + 2 minutes delay."""
                    try:
                        tz = ZoneInfo("America/New_York")
                        now_local = now_utc.astimezone(tz)
                        today_open = now_local.replace(hour=9, minute=30, second=0, microsecond=0)
                        
                        def next_weekday(dt: datetime) -> datetime:
                            d = dt
                            while d.weekday() >= 5:
                                d = d + timedelta(days=1)
                            return d
                        
                        if now_local < today_open:
                            next_open = today_open
                        else:
                            next_day = next_weekday((now_local + timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0))
                            next_open = next_day.replace(hour=9, minute=30, second=0, microsecond=0)
                        
                        target_time = next_open + timedelta(seconds=market_open_delay_seconds)
                        seconds_until = (target_time - now_local).total_seconds()
                        return max(0, seconds_until)
                    except Exception:
                        return None
                
                def get_seconds_since_market_opened(now_utc: datetime) -> float | None:
                    """Calculate seconds since market opened today (if market is open)."""
                    try:
                        tz = ZoneInfo("America/New_York")
                        now_local = now_utc.astimezone(tz)
                        today_open = now_local.replace(hour=9, minute=30, second=0, microsecond=0)
                        if now_local < today_open:
                            return None
                        if now_local.weekday() < 5:
                            seconds_since = (now_local - today_open).total_seconds()
                            return seconds_since
                        return None
                    except Exception:
                        return None
                
                while True:
                    # Check market status before each iteration
                    now_utc = datetime.now(timezone.utc)
                    is_market_open = common_is_market_hours(now_utc, "America/New_York")
                    
                    # If market is closed, wait until market opens + 2 minutes
                    if not is_market_open:
                        seconds_until_wake = get_seconds_until_market_open_plus_delay(now_utc)
                        if seconds_until_wake is not None:
                            if getattr(args, 'verbose', False):
                                hours_to_wait = seconds_until_wake / 3600
                                mins_to_wait = seconds_until_wake / 60
                                period_name = f"{period_label.capitalize()} {period_idx + 1}" if num_periods > 1 else period_label.capitalize()
                                if hours_to_wait >= 1:
                                    print(f"[{period_name}] Market closed. Waiting {hours_to_wait:.2f} hours until market opens + 2 mins...")
                                else:
                                    print(f"[{period_name}] Market closed. Waiting {mins_to_wait:.1f} minutes until market opens + 2 mins...")
                            
                            # Sleep in chunks to check for shutdown
                            sleep_chunk = min(60.0, seconds_until_wake)
                            elapsed = 0.0
                            while elapsed < seconds_until_wake:
                                await asyncio.sleep(sleep_chunk)
                                elapsed += sleep_chunk
                                if elapsed + sleep_chunk > seconds_until_wake:
                                    sleep_chunk = seconds_until_wake - elapsed
                            
                            # Re-check market status after waiting
                            now_utc = datetime.now(timezone.utc)
                            is_market_open = common_is_market_hours(now_utc, "America/New_York")
                            continue
                    
                    # Market is open - check if we need to wait until 2 minutes after open
                    seconds_since_open = get_seconds_since_market_opened(now_utc)
                    if seconds_since_open is not None and seconds_since_open < market_open_delay_seconds:
                        wait_time = market_open_delay_seconds - seconds_since_open
                        if getattr(args, 'verbose', False):
                            period_name = f"{period_label.capitalize()} {period_idx + 1}" if num_periods > 1 else period_label.capitalize()
                            print(f"[{period_name}] Market just opened. Waiting {wait_time:.0f} seconds until 2 mins after open...")
                        
                        sleep_chunk = min(10.0, wait_time)
                        elapsed = 0.0
                        while elapsed < wait_time:
                            await asyncio.sleep(sleep_chunk)
                            elapsed += sleep_chunk
                            if elapsed + sleep_chunk > wait_time:
                                sleep_chunk = wait_time - elapsed
                        
                        now_utc = datetime.now(timezone.utc)
                        is_market_open = common_is_market_hours(now_utc, "America/New_York")
                    
                    # Market is open and at least 2 minutes have passed - proceed with fetch
                    run_num += 1
                    if getattr(args, 'verbose', False):
                        period_name = f"{period_label.capitalize()} {period_idx + 1}" if num_periods > 1 else period_label.capitalize()
                        print(f"\n[{period_name}] Continuous run #{run_num} at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                    
                    try:
                        # Create period-specific args
                        period_args = argparse.Namespace(**vars(args))
                        period_args.date = period_start
                        # Set both to 0 to use single-date mode for this period
                        period_args.months_ahead = 0
                        period_args.days_ahead = None
                        period_start_dt = datetime.strptime(period_start, '%Y-%m-%d')
                        period_end_dt = datetime.strptime(period_end, '%Y-%m-%d')
                        days_in_period = (period_end_dt - period_start_dt).days + 1  # +1 to include both start and end
                        
                        # If end_date is provided, limit max_days_to_expiry to the end_date
                        end_date = getattr(args, 'end_date', None)
                        if end_date:
                            end_dt = datetime.strptime(end_date, '%Y-%m-%d')
                            # Calculate days from period_start to end_date
                            days_to_end = (end_dt - period_start_dt).days + 1  # +1 to include both start and end
                            # Limit max_days_to_expiry to not exceed end_date
                            # max_days_to_expiry creates a +/- window, so we use days_to_end directly
                            period_args.max_days_to_expiry = max(1, days_to_end)
                        elif use_days_mode:
                            # In days_ahead mode, limit max_days_to_expiry to the period length (or 1 if single day)
                            # This ensures we only fetch options expiring within the requested day range
                            period_args.max_days_to_expiry = max(1, days_in_period + 1)  # +1 to include the end day
                        else:
                            period_args.max_days_to_expiry = max(days_in_period + 1, getattr(args, 'max_days_to_expiry', 30))
                        
                        # Execute this period's iteration (processes will exit after completion)
                        # Each period cluster uses its own pool list to ensure independence
                        # When this iteration completes, processes exit, and a new iteration can start
                        # even if other period clusters are still running
                        await _execute_options_iteration(symbols_list, period_args, api_key, period_pools)
                        
                        if getattr(args, 'verbose', False):
                            period_name = f"{period_label.capitalize()} {period_idx + 1}" if num_periods > 1 else period_label.capitalize()
                            print(f"[{period_name}] Iteration #{run_num} completed. Processes have exited. Next iteration will start after sleep interval.")
                    except KeyboardInterrupt:
                        if getattr(args, 'verbose', False):
                            period_name = f"{period_label.capitalize()} {period_idx + 1}" if num_periods > 1 else period_label.capitalize()
                            print(f"[{period_name}] Interrupted", file=sys.stderr)
                        break
                    except Exception as e:
                        if getattr(args, 'verbose', False):
                            period_name = f"{period_label.capitalize()} {period_idx + 1}" if num_periods > 1 else period_label.capitalize()
                            print(f"[{period_name}] Error: {e}", file=sys.stderr)
                    
                    # After fetch, sleep for this period's interval
                    # This sleep is independent - when it expires, this period cluster will start
                    # a new iteration even if other period clusters are still running or sleeping
                    adjusted_sleep = sleep_interval * (args.interval_multiplier if getattr(args, 'interval_multiplier', None) else 1.0)
                    if getattr(args, 'verbose', False):
                        period_name = f"{period_label.capitalize()} {period_idx + 1}" if num_periods > 1 else period_label.capitalize()
                        print(f"[{period_name}] Sleeping for {adjusted_sleep/60:.1f} minutes. Will start iteration #{run_num + 1} independently when sleep expires.")
                    
                    # Sleep in chunks to allow other period clusters to proceed independently
                    # Each period cluster's sleep timer is independent
                    sleep_chunk = min(60.0, adjusted_sleep)
                    elapsed = 0.0
                    while elapsed < adjusted_sleep:
                        await asyncio.sleep(sleep_chunk)
                        elapsed += sleep_chunk
                        if elapsed + sleep_chunk > adjusted_sleep:
                            sleep_chunk = adjusted_sleep - elapsed
                    
                    # Sleep expired - this period cluster will now start a new iteration
                    # This happens independently of other period clusters' status
                    if getattr(args, 'verbose', False):
                        period_name = f"{period_label.capitalize()} {period_idx + 1}" if num_periods > 1 else period_label.capitalize()
                        print(f"[{period_name}] Sleep interval expired. Starting new iteration (independent of other {period_label_plural}).")
            
            # Run all period clusters concurrently
            tasks = []
            for period_idx, (period_start, period_end) in enumerate(date_ranges):
                task = asyncio.create_task(period_cluster_loop(period_idx, period_start, period_end, period_intervals[period_idx]))
                tasks.append(task)
            
            try:
                await asyncio.gather(*tasks)
            except KeyboardInterrupt:
                if getattr(args, 'verbose', False):
                    print(f"\nKeyboardInterrupt: Cancelling all {period_label_plural} clusters...", file=sys.stderr)
                for task in tasks:
                    task.cancel()
                await asyncio.gather(*tasks, return_exceptions=True)
        else:
            # Single-date continuous mode (original behavior)
            run_num = 0
            was_market_open = None  # Track previous market state to detect transitions
            while True:
                run_num += 1
                
                # Check for market transition from open to closed
                now_utc = datetime.now(timezone.utc)
                is_market_open_start = HistoricalDataFetcher._is_market_open(now_utc)
                
                if was_market_open is True and not is_market_open_start:
                    if getattr(args, 'verbose', False):
                        print(f"\n--- MARKET TRANSITION DETECTED: OPEN → CLOSED at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ---")
                        print(f"Performing final fetch after market close to capture EOD data...")
                
                if getattr(args, 'verbose', False):
                    print(f"\n--- Continuous run #{run_num} at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ---")

                iteration_args_dict = vars(args).copy()
                payload = await asyncio.to_thread(
                    run_iteration_in_subprocess,
                    _options_iteration_worker,
                    symbols_list,
                    iteration_args_dict,
                    api_key,
                )

                if payload.get("status") != "ok":
                    error_text = payload.get("error", "Unknown error in options iteration subprocess")
                    print(
                        f"Error during options iteration (process exit {payload.get('exitcode')}): {error_text}",
                        file=sys.stderr,
                    )

                # Intelligent sleep based on market transitions
                now_utc = datetime.now(timezone.utc)
                seconds_to_open, seconds_to_close = common_compute_market_transition_times(now_utc, "America/New_York")
                is_market_open = HistoricalDataFetcher._is_market_open(now_utc)
                
                # Check if we just transitioned from open to closed
                just_closed = (was_market_open is True and not is_market_open)

                sleep_seconds = HistoricalDataFetcher.CACHE_DURATION_MINUTES['market_closed'] * 60
                if is_market_open:
                    base_sleep = HistoricalDataFetcher.CACHE_DURATION_MINUTES['market_open'] * 60
                    if seconds_to_close is not None:
                        sleep_seconds = max(min(base_sleep, seconds_to_close), 5)
                        if getattr(args, 'verbose', False):
                            print(
                                f"Next run in {sleep_seconds:.0f}s (market open, "
                                f"{HistoricalDataFetcher.CACHE_DURATION_MINUTES['market_open']}min interval; "
                                f"{seconds_to_close:.0f}s until close) [MARKET OPEN]"
                            )
                    else:
                        sleep_seconds = base_sleep
                        if getattr(args, 'verbose', False):
                            print(
                                f"Next run in {sleep_seconds:.0f}s (market open, "
                                f"{HistoricalDataFetcher.CACHE_DURATION_MINUTES['market_open']}min interval) [MARKET OPEN]"
                            )
                else:
                    # Market is closed
                    if just_closed and getattr(args, 'verbose', False):
                        print(f"Post-close fetch completed. Entering extended sleep until next market open.")
                    
                    opening_soon_threshold = HistoricalDataFetcher.CACHE_DURATION_MINUTES['market_open'] * 60
                    if seconds_to_open is not None:
                        if seconds_to_open <= opening_soon_threshold:
                            sleep_seconds = seconds_to_open
                            if getattr(args, 'verbose', False):
                                print(
                                    f"Next run in {sleep_seconds:.0f}s (sleeping until market open in "
                                    f"{seconds_to_open:.0f}s) [MARKET CLOSED→OPEN]"
                                )
                        else:
                            sleep_seconds = seconds_to_open - opening_soon_threshold
                            if getattr(args, 'verbose', False):
                                print(
                                    f"Next run in {sleep_seconds:.0f}s (markets closed, will wake "
                                    f"{opening_soon_threshold/60:.0f}min before market open in {seconds_to_open:.0f}s) [MARKET CLOSED]"
                                )
                    else:
                        if getattr(args, 'verbose', False):
                            print(
                                f"Next run in {sleep_seconds:.0f}s (markets closed, "
                                f"{HistoricalDataFetcher.CACHE_DURATION_MINUTES['market_closed']}min interval) [MARKET CLOSED]"
                            )
                
                # Update the market state tracker for next iteration
                was_market_open = is_market_open

                adjusted_sleep = sleep_seconds * (args.interval_multiplier if getattr(args, 'interval_multiplier', None) else 1.0)
                if getattr(args, 'verbose', False):
                    print(f"Next run in {adjusted_sleep:.0f}s")
                await asyncio.sleep(adjusted_sleep)
                
                # After waking up, check if market transitioned from open to closed during sleep
                # If so, perform one more fetch to capture EOD data before long sleep
                if was_market_open is True:
                    current_market_state = HistoricalDataFetcher._is_market_open(datetime.now(timezone.utc))
                    if not current_market_state:
                        # Market closed while we were sleeping - fetch once more for EOD data
                        if getattr(args, 'verbose', False):
                            print(f"\n--- MARKET CLOSED DURING SLEEP - Performing post-close fetch ---")
                        run_num += 1
                        
                        try:
                            # Run the post-close fetch
                            iteration_args_dict = vars(args).copy()
                            payload = await asyncio.to_thread(
                                run_iteration_in_subprocess,
                                _options_iteration_worker,
                                symbols_list,
                                iteration_args_dict,
                                api_key,
                            )
                            
                            if payload.get("status") != "ok":
                                error_text = payload.get("error", "Unknown error in post-close options iteration subprocess")
                                print(
                                    f"Error during post-close fetch (process exit {payload.get('exitcode')}): {error_text}",
                                    file=sys.stderr,
                                )
                            elif getattr(args, 'verbose', False):
                                print(f"Post-close fetch #{run_num} completed successfully")
                            
                            # Check if we should stop
                            if args.continuous_max_runs and run_num >= args.continuous_max_runs:
                                if getattr(args, 'verbose', False):
                                    print("Reached maximum runs, stopping continuous mode.")
                                break
                            
                            # Update market state tracker
                            was_market_open = False
                            
                        except Exception as e:
                            if getattr(args, 'verbose', False):
                                print(f"Error during post-close fetch: {e}", file=sys.stderr)
                            was_market_open = False

                if args.continuous_max_runs and run_num >= args.continuous_max_runs:
                    if getattr(args, 'verbose', False):
                        print("Reached maximum runs, stopping continuous mode.")
                    break
        return
    else:
        # Single run (no sleep)
        await _execute_options_iteration(symbols_list, args, api_key, all_pools)
        
        # Display and/or save saved options if requested (non-continuous mode only)
        # Auto-enable save_saved_to_csv if save_saved_to_csv_file is specified
        if getattr(args, 'save_saved_to_csv_file', None) and not getattr(args, 'save_saved_to_csv', False):
            args.save_saved_to_csv = True
        if getattr(args, 'display_saved', False) or getattr(args, 'save_saved_to_csv', False):
            await display_and_save_saved_options(symbols_list, args, api_key)
        
        return

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nOperation cancelled by user.")
    except Exception as e:
        print(f"\nAn unexpected error occurred: {e}", file=sys.stderr)
        sys.exit(1) 