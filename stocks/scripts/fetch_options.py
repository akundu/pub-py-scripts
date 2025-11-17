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

    def __init__(self, api_key: str, data_dir: str = "data", quiet: bool = False, snapshot_max_concurrent: int = 0):
        if not api_key:
            raise ValueError("Polygon API key is required.")
        self.client = RESTClient(api_key)
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)
        self.quiet = quiet
        # 0 disables per-contract snapshot concurrency; otherwise bounded parallelism
        self.snapshot_max_concurrent = max(0, int(snapshot_max_concurrent))

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
        """Check if we should fetch fresh data based on cache duration."""
        if not csv_path.exists():
            return True
        
        file_mtime = datetime.fromtimestamp(csv_path.stat().st_mtime)
        cache_duration = self._get_cache_duration_minutes()
        
        return datetime.now() - file_mtime > timedelta(minutes=cache_duration)

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

    def _handle_api_error(self, error: Exception, data_type: str) -> Dict[str, Any]:
        """Handles API errors gracefully."""
        error_msg = f"Error fetching {data_type}: {str(error)}"
        print(f"Warning: {error_msg}", file=sys.stderr)
        return {"error": error_msg, "data": None}

    async def get_stock_price_for_date(self, symbol: str, target_date_str: str) -> Dict[str, Any]:
        """
        Fetches the historical stock price (OHLCV) for the requested date.
        If the date is a non-trading day, it finds the most recent previous trading day.
        """
        target_date_dt = datetime.strptime(target_date_str, '%Y-%m-%d')
        
        # We look at a 7-day window ending on the target date to ensure we find a trading day.
        search_start = (target_date_dt - timedelta(days=7)).strftime('%Y-%m-%d')
        search_end = target_date_str
        
        if not self.quiet:
            print(f"Fetching historical price for {symbol} on or before {target_date_str}...", flush=True)

        try:
            # Sort descending and limit to 1 to get the latest trading day on or before the target date
            aggs = self.client.get_aggs(
                ticker=symbol,
                multiplier=1,
                timespan="day",
                from_=search_start,
                to=search_end,
                adjusted=True,
                sort="desc", 
                limit=1
            )
            
            if not aggs:
                return self._handle_api_error(Exception(f"No trading data found on or before {target_date_str} in the last 7 days"), "stock price")

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
            return self._handle_api_error(e, "stock price")

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
        redis_url: str | None = None
    ) -> Dict[str, Any]:
        """
        Fetches all options contracts that were active on a specific date and gets their
        current snapshot data (including prices and Greeks).
        Applies filters for option type and strike price range.
        """
        options_data = {"contracts": []}
        target_date_dt = datetime.strptime(target_date_str, '%Y-%m-%d')
        
        if not self.quiet:
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
                
                db = get_stock_db(db_type, db_config=db_config, enable_cache=enable_cache, redis_url=redis_url)
                latest_df = await db.get_latest_options_data(ticker=symbol)
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
                        min_strike = stock_close_price * (1 - strike_range_percent / 100)
                        max_strike = stock_close_price * (1 + strike_range_percent / 100)
                        filtered_contracts = [c for c in filtered_contracts if min_strike <= (c.get('strike') or -1) <= max_strike]
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
                underlying_ticker=symbol,
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
                    underlying_ticker=symbol,
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
            if strike_range_percent is not None and stock_close_price is not None:
                if not self.quiet:
                    print(f"Filtering for strikes within {strike_range_percent}% of close price ${stock_close_price:.2f}...")
                min_strike = stock_close_price * (1 - strike_range_percent / 100)
                max_strike = stock_close_price * (1 + strike_range_percent / 100)
                
                filtered_contracts = [
                    c for c in filtered_contracts
                    if min_strike <= getattr(c, 'strike_price', -1) <= max_strike
                ]

            filter_end = time.time()
            if not self.quiet:
                print(f"  [TIMER] Local filtering took {filter_end - filter_start:.2f} seconds.")

            if not self.quiet:
                print(f"Found {len(filtered_contracts)} contracts after filtering. Fetching snapshot data for all {len(filtered_contracts)} contracts...", flush=True)

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
                    snapshot_local = self.client.get_snapshot_option(symbol, contract_ticker_local)
                    
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
            return self._handle_api_error(e, "options data")
        
        # Save to CSV files only when explicitly requested
        if save_to_csv and options_data["contracts"]:
            self._save_options_to_csv(symbol, options_data)
        
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

def _run_for_symbol(symbol: str, args_namespace: argparse.Namespace, api_key: str) -> str:
    """Worker task: runs fetch for a single symbol and returns formatted output string."""
    try:
        fetcher = HistoricalDataFetcher(
            api_key,
            args_namespace.data_dir,
            args_namespace.quiet,
            getattr(args_namespace, 'snapshot_max_concurrent', 0)
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
                redis_url=redis_url
            )
            # Return data for database saving in main thread
            db_save_data = None
            if getattr(args_namespace, 'use_db', None) and options_result.get('success'):
                contracts = options_result['data'].get('contracts') or []
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
                        
                        db_save_data = {
                            'symbol': symbol,
                            'df': df,
                            'contracts_count': len(contracts)
                        }
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
            
            # Return both formatted output and database save data
            return {
                'formatted_output': formatted_output,
                'db_save_data': db_save_data
            }
        return asyncio.run(_inner())
    except Exception as e:
        return f"Error processing {symbol}: {e}"

async def save_options_to_database(db_save_tasks: list, args) -> None:
    """Save options data to database in main thread."""
    if not db_save_tasks:
        return
    
    db_config = getattr(args, 'use_db', None)
    if not args.quiet:
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
                        if not args.quiet:
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
                                if not args.quiet:
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
                                if not args.quiet:
                                    print(f"Warning: failed to save options to DB for {symbol}: HTTP {response.status} - {error_text}")
                    else:
                        # Split large datasets into batches
                        num_batches = (total_records + batch_size - 1) // batch_size  # Ceiling division
                        if not args.quiet:
                            print(f"Splitting {total_records} records for {symbol} into {num_batches} batches of max {batch_size} records each")
                        
                        successful_batches = 0
                        failed_batches = 0
                        
                        for batch_num in range(num_batches):
                            start_idx = batch_num * batch_size
                            end_idx = min(start_idx + batch_size, total_records)
                            batch_records = data_records[start_idx:end_idx]
                            
                            if not args.quiet:
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
                                        if not args.quiet:
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
                                        if not args.quiet:
                                            print(f"    Warning: failed to save batch {batch_num + 1} for {symbol}: HTTP {response.status} - {error_text}")
                            except Exception as batch_e:
                                failed_batches += 1
                                if not args.quiet:
                                    print(f"    Warning: failed to save batch {batch_num + 1} for {symbol}: {batch_e}")
                        
                        # Summary for batched requests
                        if not args.quiet:
                            if failed_batches == 0:
                                print(f"Successfully saved all {num_batches} batches ({contracts_count} total contracts) to database for {symbol}")
                            else:
                                print(f"Completed batch processing for {symbol}: {successful_batches} successful, {failed_batches} failed batches")
                            
                except Exception as e:
                    if not args.quiet:
                        print(f"Warning: failed to save options to DB for {symbol}: {e}")
                        
    except Exception as e:
        if not args.quiet:
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
                if not args.quiet:
                    print(f"Initializing database connection (attempt {attempt + 1}/{max_retries})...")
                await db_instance._init_db()
                if not args.quiet:
                    print("Database connection initialized successfully")
                break
            except Exception as e:
                if attempt < max_retries - 1:
                    if not args.quiet:
                        print(f"Database initialization attempt {attempt + 1} failed: {e}, retrying...")
                    await asyncio.sleep(2)  # Wait 2 seconds before retry
                else:
                    if not args.quiet:
                        print(f"Database initialization failed after {max_retries} attempts: {e}")
                    raise e
        
        for task in db_save_tasks:
            symbol = task['symbol']
            df = task['df']
            contracts_count = task['contracts_count']
            
            try:
                await db_instance.save_options_data(df=df, ticker=symbol)
                if not args.quiet:
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
                if not args.quiet:
                    print(f"Warning: failed to save options to DB for {symbol}: {e}")
                    
    except Exception as e:
        if not args.quiet:
            print(f"Error connecting to database: {e}")
    finally:
        # Properly close database connection
        if db_instance and hasattr(db_instance, 'close_session') and callable(db_instance.close_session):
            try:
                await db_instance.close_session()
            except Exception as close_e:
                if not args.quiet:
                    print(f"Warning: error closing DB connection: {close_e}")


async def _execute_options_iteration(
    symbols_list: list[str],
    args: argparse.Namespace,
    api_key: str,
) -> dict:
    """
    Execute one iteration of options fetching (without sleeping) and return summary data.
    Checks if tickers were recently fetched and skips them if data is fresh.
    """
    from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed

    # Check if we should skip recently fetched tickers
    symbols_to_fetch = symbols_list
    refresh_threshold_seconds = None
    
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
                
                if not args.quiet and len(symbols_to_fetch) < len(symbols_list):
                    skipped = len(symbols_list) - len(symbols_to_fetch)
                    print(f"Skipping {skipped} ticker(s) with fresh data (threshold: {refresh_threshold_seconds}s)")
        except Exception as e:
            # If checking fails, fetch all tickers
            if not args.quiet:
                import traceback
                print(f"Warning: Could not check ticker freshness ({e}). Fetching all tickers.", file=sys.stderr)
                if getattr(args, 'debug', False):
                    traceback.print_exc(file=sys.stderr)
            symbols_to_fetch = symbols_list
    elif getattr(args, 'force_fresh', False):
        # Force fresh is enabled, skip refresh check and fetch all symbols
        symbols_to_fetch = symbols_list
        if not args.quiet:
            print("--force-fresh enabled: Skipping refresh check, will fetch all symbols from Polygon API")
    
    if not symbols_to_fetch:
        if not args.quiet:
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

    if not args.quiet:
        print(
            f"Starting concurrent fetch for {len(symbols_to_fetch)} symbols "
            f"using {exec_type} executor with max_workers={max_workers}"
        )

    ExecutorCls = ProcessPoolExecutor if exec_type == 'process' else ThreadPoolExecutor
    db_save_tasks = []
    with ExecutorCls(max_workers=max_workers) as pool:
        futures_map = {pool.submit(_run_for_symbol, symbol, args, api_key): symbol for symbol in symbols_to_fetch}
        for fut in as_completed(futures_map):
            result = fut.result()
            if isinstance(result, dict) and 'formatted_output' in result:
                if not args.quiet and result['formatted_output']:
                    print(result['formatted_output'])
                if result.get('db_save_data'):
                    db_save_tasks.append(result['db_save_data'])
            elif not args.quiet and result:
                print(result)

    if db_save_tasks and getattr(args, 'use_db', None):
        await save_options_to_database(db_save_tasks, args)

    return {
        "symbols_processed": len(symbols_to_fetch),
        "db_tasks": len(db_save_tasks),
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
        return await _execute_options_iteration(symbols_list, iteration_args, api_key)

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
  python historical_stock_options.py AAPL --date 2024-06-05 --db-path questdb://stock_user:stock_password@localhost:8812/stock_data

  # Save to HTTP database server (db_server.py)
  python historical_stock_options.py MSFT --date 2024-06-05 --db-path localhost:9002
  
  # Save to direct QuestDB connection (explicit protocol)
  python historical_stock_options.py MSFT --date 2024-06-05 --db-path questdb://stock_user:stock_password@localhost:8812/stock_data

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
    
    # Add symbol input arguments using common library
    add_symbol_arguments(parser, required=True)
    
    parser.add_argument(
        '--date',
        default=datetime.now().strftime('%Y-%m-%d'),
        help="The historical date in YYYY-MM-DD format (default: today)."
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
        '--quiet',
        action='store_true',
        help="Suppress output but still save CSV files."
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
        '--continuous',
        action='store_true',
        help="Continuously fetch in a loop, sleeping based on cache duration."
    )
    parser.add_argument(
        '--interval-multiplier',
        type=float,
        default=1.0,
        help="Multiplier for cadence-based intervals (e.g., 0.5 twice as fast, 2.0 half as often)."
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
        '--no-cache',
        action='store_true',
        help="Disable Redis caching for QuestDB operations (default: cache enabled)"
    )
    parser.add_argument(
        '--debug',
        action='store_true',
        help="Enable debug output with detailed information about data fetching and timestamp checking."
    )
    # Backward-compatibility (hidden): legacy --db-conn maps to --use-db
    parser.add_argument(
        '--db-conn',
        type=str,
        default=None,
        help=argparse.SUPPRESS
    )
    
    args = parser.parse_args()

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
    symbols_list = await fetch_lists_data(args, args.quiet)
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
                if not args.quiet:
                    print(f"Market is closed. Fetching once immediately before waiting for market open...")
                
                # Run a single fetch
                from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
                exec_type = args.executor_type
                if args.max_concurrent and args.max_concurrent > 0:
                    max_workers = args.max_concurrent
                else:
                    max_workers = (os.cpu_count() or 1) if exec_type == 'process' else (os.cpu_count() or 1) * 5
                
                if not args.quiet:
                    print(f"Fetching data for {len(symbols_list)} symbols (one-time fetch before waiting)...")
                
                ExecutorCls = ProcessPoolExecutor if exec_type == 'process' else ThreadPoolExecutor
                with ExecutorCls(max_workers=max_workers) as pool:
                    futures_map = {pool.submit(_run_for_symbol, symbol, args, api_key): symbol for symbol in symbols_list}
                    db_save_tasks = []
                    for fut in as_completed(futures_map):
                        result = fut.result()
                        if isinstance(result, dict) and 'formatted_output' in result:
                            if not args.quiet and result['formatted_output']:
                                print(result['formatted_output'])
                            if result.get('db_save_data'):
                                db_save_tasks.append(result['db_save_data'])
                        elif not args.quiet and result:
                            print(result)
                    
                    # Save all database data
                    if db_save_tasks and getattr(args, 'use_db', None):
                        await save_options_to_database(db_save_tasks, args)
                
                # Now wait for market open
                if not args.quiet:
                    hours_to_wait = seconds_to_open / 3600
                    print(f"One-time fetch completed. Waiting {hours_to_wait:.2f} hours ({seconds_to_open:.0f} seconds) until market opens...")
                
                await asyncio.sleep(seconds_to_open)
                
                # Re-check market status after waiting
                now_utc = datetime.now(timezone.utc)
                is_market_open = common_is_market_hours(now_utc, "America/New_York")
                if not args.quiet:
                    if is_market_open:
                        print("Market is now open. Proceeding with normal operation...")
                    else:
                        print("Warning: Market is still not open after waiting. Proceeding anyway...")
            else:
                # Wait for market open before starting
                if not args.quiet:
                    hours_to_wait = seconds_to_open / 3600
                    print(f"Market is closed. Waiting {hours_to_wait:.2f} hours ({seconds_to_open:.0f} seconds) until market opens before starting...")
                
                await asyncio.sleep(seconds_to_open)
                
                # Re-check market status after waiting
                now_utc = datetime.now(timezone.utc)
                is_market_open = common_is_market_hours(now_utc, "America/New_York")
                if not args.quiet:
                    if is_market_open:
                        print("Market is now open. Starting data fetch...")
                    else:
                        print("Warning: Market is still not open after waiting. Proceeding anyway...")

    if args.continuous:
        run_num = 0
        while True:
            run_num += 1
            if not args.quiet:
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

            # Intelligent sleep based on market transitions (performed in parent)
            now_utc = datetime.now(timezone.utc)
            seconds_to_open, seconds_to_close = common_compute_market_transition_times(now_utc, "America/New_York")
            is_market_open = HistoricalDataFetcher._is_market_open(now_utc)

            sleep_seconds = HistoricalDataFetcher.CACHE_DURATION_MINUTES['market_closed'] * 60
            if is_market_open:
                base_sleep = HistoricalDataFetcher.CACHE_DURATION_MINUTES['market_open'] * 60
                if seconds_to_close is not None:
                    sleep_seconds = max(min(base_sleep, seconds_to_close), 5)
                    if not args.quiet:
                        print(
                            f"Next run in {sleep_seconds:.0f}s (market open, "
                            f"{HistoricalDataFetcher.CACHE_DURATION_MINUTES['market_open']}min interval; "
                            f"{seconds_to_close:.0f}s until close) [MARKET OPEN]"
                        )
                else:
                    sleep_seconds = base_sleep
                    if not args.quiet:
                        print(
                            f"Next run in {sleep_seconds:.0f}s (market open, "
                            f"{HistoricalDataFetcher.CACHE_DURATION_MINUTES['market_open']}min interval) [MARKET OPEN]"
                        )
            else:
                opening_soon_threshold = HistoricalDataFetcher.CACHE_DURATION_MINUTES['market_open'] * 60
                if not args.quiet:
                    print(
                        f"opening_soon_threshold: {opening_soon_threshold}, seconds_to_open: {seconds_to_open}",
                        file=sys.stderr,
                    )
                # If we know when market opens next, sleep until shortly before it opens
                # (or until it opens if it's very soon)
                if seconds_to_open is not None:
                    if seconds_to_open <= opening_soon_threshold:
                        # Market opens very soon - sleep until it opens
                        sleep_seconds = seconds_to_open
                        if not args.quiet:
                            print(
                                f"Next run in {sleep_seconds:.0f}s (sleeping until market open in "
                                f"{seconds_to_open:.0f}s) [MARKET CLOSED→OPEN]"
                            )
                    else:
                        # Market opens later - sleep until shortly before it opens
                        # Wake up 20 minutes before market open to be ready
                        sleep_seconds = seconds_to_open - opening_soon_threshold
                        if not args.quiet:
                            print(
                                f"Next run in {sleep_seconds:.0f}s (markets closed, will wake "
                                f"{opening_soon_threshold/60:.0f}min before market open in {seconds_to_open:.0f}s) [MARKET CLOSED]"
                            )
                else:
                    # Don't know when market opens - use default closed interval
                    if not args.quiet:
                        print(
                            f"Next run in {sleep_seconds:.0f}s (markets closed, "
                            f"{HistoricalDataFetcher.CACHE_DURATION_MINUTES['market_closed']}min interval) [MARKET CLOSED]"
                        )

            adjusted_sleep = sleep_seconds * (args.interval_multiplier if getattr(args, 'interval_multiplier', None) else 1.0)
            if not args.quiet:
                print(f"Next run in {adjusted_sleep:.0f}s")
            await asyncio.sleep(adjusted_sleep)

            if args.continuous_max_runs and run_num >= args.continuous_max_runs:
                if not args.quiet:
                    print("Reached maximum runs, stopping continuous mode.")
                break
        return
    else:
        # Single run (no sleep)
        await _execute_options_iteration(symbols_list, args, api_key)
        return

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nOperation cancelled by user.")
    except Exception as e:
        print(f"\nAn unexpected error occurred: {e}", file=sys.stderr)
        sys.exit(1) 