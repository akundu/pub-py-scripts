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

try:
    from polygon import RESTClient
    POLYGON_AVAILABLE = True
except ImportError:
    POLYGON_AVAILABLE = False

class HistoricalDataFetcher:
    """Fetches historical stock and options data from Polygon.io."""
    CACHE_DURATION_MINUTES = {
        'market_open': 30,
        'market_closed': 360,
        'post_market': 90,
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
    def _is_market_open(dt: datetime = None) -> bool:
        """Check if market is currently open (9:30 AM - 4:00 PM ET, Mon-Fri)."""
        if dt is None:
            dt = datetime.now()
        
        # Convert to ET (assuming system is in ET or adjust as needed)
        # For simplicity, assuming system timezone is ET
        weekday = dt.weekday()  # 0 = Monday, 6 = Sunday
        if weekday >= 5:  # Saturday or Sunday
            return False
        
        market_open = dt.replace(hour=9, minute=30, second=0, microsecond=0)
        market_close = dt.replace(hour=16, minute=0, second=0, microsecond=0)
        
        return market_open <= dt <= market_close

    def _get_cache_duration_minutes(self) -> int:
        """Get cache duration in minutes based on market status."""
        now = datetime.now()
        
        if self._is_market_open(now):
            return HistoricalDataFetcher.CACHE_DURATION_MINUTES['market_open']
        elif now.hour < 9 or now.hour >= 20:  # Before 9 AM or after 8 PM
            return HistoricalDataFetcher.CACHE_DURATION_MINUTES['market_closed']
        else:
            return HistoricalDataFetcher.CACHE_DURATION_MINUTES['post_market']

    def _get_csv_path(self, symbol: str, expiration_date: str) -> Path:
        """Get the CSV file path for a specific symbol and expiration date."""
        options_dir = self.data_dir / "options"
        symbol_dir = options_dir / symbol.upper()
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
                })
            
            # Write to CSV (append mode)
            file_exists = csv_path.exists()
            with open(csv_path, 'a', newline='', encoding='utf-8') as csvfile:
                fieldnames = ['timestamp', 'ticker', 'type', 'strike', 'expiration', 
                             'bid', 'ask', 'day_close', 'fmv', 'delta', 'gamma', 'theta', 'vega']
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                
                if not file_exists:
                    writer.writeheader()
                writer.writerows(csv_data)
            
            if not self.quiet:
                print(f"Saved {len(csv_data)} contracts for {symbol} expiration {exp_date} to {csv_path}")

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
        use_cache: bool = True
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
        
        # Check if we should use cached data
        if use_cache:
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
                }
                try:
                    snapshot_local = self.client.get_snapshot_option(symbol, contract_ticker_local)
                    if snapshot_local:
                        if hasattr(snapshot_local, 'last_quote') and snapshot_local.last_quote:
                            details['bid'] = getattr(snapshot_local.last_quote, 'bid', None)
                            details['ask'] = getattr(snapshot_local.last_quote, 'ask', None)
                        if hasattr(snapshot_local, 'last_trade') and snapshot_local.last_trade:
                            last_price = getattr(snapshot_local.last_trade, 'price', None)
                            if last_price and not details['bid']:
                                details['bid'] = last_price
                                details['ask'] = last_price
                        if hasattr(snapshot_local, 'day') and snapshot_local.day:
                            day_close = getattr(snapshot_local.day, 'close', None)
                            if day_close:
                                details['day_close'] = day_close
                                if not details['bid']:
                                    details['bid'] = day_close
                                    details['ask'] = day_close
                        if hasattr(snapshot_local, 'fair_market_value') and snapshot_local.fair_market_value:
                            fmv = getattr(snapshot_local.fair_market_value, 'value', None)
                            if fmv:
                                details['fmv'] = fmv
                                if not details['bid']:
                                    details['bid'] = fmv
                                    details['ask'] = fmv
                        if hasattr(snapshot_local, 'greeks') and snapshot_local.greeks:
                            details['delta'] = getattr(snapshot_local.greeks, 'delta', None)
                            details['gamma'] = getattr(snapshot_local.greeks, 'gamma', None)
                            details['theta'] = getattr(snapshot_local.greeks, 'theta', None)
                            details['vega'] = getattr(snapshot_local.greeks, 'vega', None)
                        if index_in_list < 3 and not self.quiet:
                            print(f"    DEBUG: Snapshot for {contract_ticker_local}: bid={details.get('bid')}, ask={details.get('ask')}")
                except Exception as e_local:
                    if index_in_list < 3 and not self.quiet:
                        print(f"    DEBUG: Snapshot error for {contract_ticker_local}: {e_local}")
                    # Fallback to historical for first few only
                    try:
                        if index_in_list < 3:
                            if not self.quiet:
                                print(f"    DEBUG: Trying historical data for {contract_ticker_local}...")
                            historical_data_local = self.client.get_aggs(
                                ticker=contract_ticker_local,
                                multiplier=1,
                                timespan="day",
                                from_=target_date_str,
                                to=target_date_str,
                                adjusted=True,
                                sort="desc",
                                limit=1
                            )
                            if historical_data_local:
                                bar_local = historical_data_local[0]
                                details['bid'] = bar_local.close
                                details['ask'] = bar_local.close
                                if not self.quiet:
                                    print(f"    DEBUG: Historical price for {contract_ticker_local}: ${bar_local.close:.2f}")
                    except Exception as hist_e_local:
                        if index_in_list < 3 and not self.quiet:
                            print(f"    DEBUG: Historical data also failed for {contract_ticker_local}: {hist_e_local}")
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
        
        # Save to CSV files
        if options_data["contracts"]:
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
        output.append("Note: Bid/Ask from real-time data if available, otherwise from day close. Day Close = daily close price. FMV = Fair Market Value. Greeks from current market data.")

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
                        ])
                    output.append(tabulate(options_table, headers=[f'Ticker ({symbol})', 'Type', 'Strike', 'Bid', 'Ask', 'Day Close', 'FMV', 'Delta', 'Gamma', 'Theta', 'Vega'], tablefmt='grid'))
        else:
            output.append(f"Could not fetch options data: {options_result.get('error', 'Unknown error')}")
            
        rendered = "\n".join(output)
        if not self.quiet:
            print(rendered)
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
            options_result = await fetcher.get_active_options_for_date(
                symbol=symbol,
                target_date_str=args_namespace.date,
                option_type=args_namespace.option_type,
                stock_close_price=stock_close_price,
                strike_range_percent=args_namespace.strike_range_percent,
                max_days_to_expiry=args_namespace.max_days_to_expiry,
                include_expired=args_namespace.include_expired,
                use_cache=not args_namespace.no_cache
            )
            return fetcher.format_output(
                symbol=symbol,
                target_date=args_namespace.date,
                stock_result=stock_result,
                options_result=options_result,
                option_type=args_namespace.option_type,
                strike_range_percent=args_namespace.strike_range_percent,
                options_per_expiry=args_namespace.options_per_expiry,
                max_days_to_expiry=args_namespace.max_days_to_expiry
            )
        return asyncio.run(_inner())
    except Exception as e:
        return f"Error processing {symbol}: {e}"

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

  # Quiet mode - suppress output but still save CSV files
  python historical_stock_options.py --symbols AAPL MSFT --date 2024-06-05 --quiet
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
        default=None,
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
    parser.add_argument(
        '--no-cache',
        action='store_true',
        help="Force fresh data fetch, bypassing cache."
    )
    parser.add_argument(
        '--quiet',
        action='store_true',
        help="Suppress output but still save CSV files."
    )
    parser.add_argument(
        '--continuous',
        action='store_true',
        help="Continuously fetch in a loop, sleeping based on cache duration."
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
        default=None,
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
        default=0,
        help="Max concurrent per-contract snapshot requests within a symbol (0 disables)."
    )
    
    args = parser.parse_args()
    
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

    fetcher = HistoricalDataFetcher(api_key, args.data_dir, args.quiet)
    
    async def run_one_batch_and_sleep_once() -> bool:
        # returns True if should continue, False to stop
        from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
        exec_type = args.executor_type
        if args.max_concurrent and args.max_concurrent > 0:
            max_workers = args.max_concurrent
        else:
            max_workers = (os.cpu_count() or 1) if exec_type == 'process' else (os.cpu_count() or 1) * 5

        if not args.quiet:
            print(f"Starting concurrent fetch for {len(symbols_list)} symbols using {exec_type} executor with max_workers={max_workers}")

        ExecutorCls = ProcessPoolExecutor if exec_type == 'process' else ThreadPoolExecutor
        with ExecutorCls(max_workers=max_workers) as pool:
            futures_map = {pool.submit(_run_for_symbol, symbol, args, api_key): symbol for symbol in symbols_list}
            for fut in as_completed(futures_map):
                out = fut.result()
                if not args.quiet and out:
                    print(out)

        # Intelligent sleep based on market transitions
        now_utc = datetime.now(timezone.utc)
        seconds_to_open, seconds_to_close = HistoricalDataFetcher._compute_market_transition_times(now_utc, "America/New_York")
        is_market_open = HistoricalDataFetcher._is_market_open(now_utc.astimezone(ZoneInfo("America/New_York")))
        
        if is_market_open:
            # Market is open: prefer staying on open cadence; do not sleep past close
            base_sleep = HistoricalDataFetcher.CACHE_DURATION_MINUTES['market_open'] * 60
            if seconds_to_close is not None:
                sleep_seconds = max(min(base_sleep, seconds_to_close), 5)
                if not args.quiet:
                    print(f"Next run in {sleep_seconds:.0f}s (market open, {HistoricalDataFetcher.CACHE_DURATION_MINUTES['market_open']}min interval; {seconds_to_close:.0f}s until close) [MARKET OPEN]")
            else:
                sleep_seconds = base_sleep
                if not args.quiet:
                    print(f"Next run in {sleep_seconds:.0f}s (market open, {HistoricalDataFetcher.CACHE_DURATION_MINUTES['market_open']}min interval) [MARKET OPEN]")
        else:
            # Market is closed: if opening soon, sleep to open; otherwise use closed cadence
            opening_soon_threshold = HistoricalDataFetcher.CACHE_DURATION_MINUTES['market_open'] * 60
            if seconds_to_open is not None and seconds_to_open <= opening_soon_threshold:
                sleep_seconds = max(seconds_to_open, 5)
                if not args.quiet:
                    print(f"Next run in {sleep_seconds:.0f}s (sleeping until market open in {seconds_to_open:.0f}s) [MARKET CLOSED→OPEN]")
            else:
                sleep_seconds = HistoricalDataFetcher.CACHE_DURATION_MINUTES['market_closed'] * 60
                msg_extra = f"; {seconds_to_open:.0f}s until open" if seconds_to_open is not None else ""
                if not args.quiet:
                    print(f"Next run in {sleep_seconds:.0f}s (markets closed, {HistoricalDataFetcher.CACHE_DURATION_MINUTES['market_closed']}min interval{msg_extra}) [MARKET CLOSED]")
        
        await asyncio.sleep(sleep_seconds)
        return True

    if args.continuous:
        run_num = 0
        while True:
            run_num += 1
            if not args.quiet:
                print(f"\n--- Continuous run #{run_num} at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ---")
            should_continue = await run_one_batch_and_sleep_once()
            if args.continuous_max_runs and run_num >= args.continuous_max_runs:
                if not args.quiet:
                    print("Reached maximum runs, stopping continuous mode.")
                break
            if not should_continue:
                break
        return
    else:
        # Single run (no sleep)
        from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
        exec_type = args.executor_type
        if args.max_concurrent and args.max_concurrent > 0:
            max_workers = args.max_concurrent
        else:
            max_workers = (os.cpu_count() or 1) if exec_type == 'process' else (os.cpu_count() or 1) * 5
        if not args.quiet:
            print(f"Starting concurrent fetch for {len(symbols_list)} symbols using {exec_type} executor with max_workers={max_workers}")
        ExecutorCls = ProcessPoolExecutor if exec_type == 'process' else ThreadPoolExecutor
        with ExecutorCls(max_workers=max_workers) as pool:
            futures_map = {pool.submit(_run_for_symbol, symbol, args, api_key): symbol for symbol in symbols_list}
            for fut in as_completed(futures_map):
                out = fut.result()
                if not args.quiet and out:
                    print(out)
        return

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nOperation cancelled by user.")
    except Exception as e:
        print(f"\nAn unexpected error occurred: {e}", file=sys.stderr)
        sys.exit(1) 