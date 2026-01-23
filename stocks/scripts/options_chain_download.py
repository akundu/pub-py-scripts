#!/usr/bin/env python3
"""
Download options chain data from Polygon.io API.

This script fetches options chain data for a given ticker over a specified time window.
It attempts to fetch 5-minute data (ideal) and falls back to hourly data if needed.
Data is fetched at least every hour, ideally every 5 minutes during trading hours.
"""

import os
import sys
import argparse
from datetime import datetime, timedelta
from polygon.rest import RESTClient
import pandas as pd
from pathlib import Path
import multiprocessing
from multiprocessing import Pool, cpu_count
from functools import partial
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

def _fetch_contract_aggregates_threaded_range(args_tuple):
    """
    Worker function for threaded fetching - fetches aggregates for a single contract over a date range.
    
    Args:
        args_tuple: Tuple of (api_key, contract_ticker, contract_data, start_date, end_date, multiplier, timespan)
    
    Returns:
        Tuple of (contract_ticker, contract_data, raw_aggs) or None on error
    """
    api_key, contract_ticker, contract_data, start_date, end_date, multiplier, timespan = args_tuple
    
    try:
        # Create client in worker thread
        client = RESTClient(api_key)
        
        # Fetch aggregates
        aggs = []
        for agg in client.list_aggs(
            ticker=contract_ticker,
            multiplier=multiplier,
            timespan=timespan,
            from_=start_date,
            to=end_date,
            limit=50000
        ):
            aggs.append(agg)
        
        if not aggs:
            return None  # No data for this contract
        
        return (contract_ticker, contract_data, aggs)
        
    except Exception as e:
        # Return error info
        return {'error': str(e), 'ticker': contract_ticker}


def get_options_chain_data(
    client: RESTClient,
    underlying: str,
    start_date: str,
    end_date: str,
    interval: str = "5min",
    max_contracts: int = None,
    include_expired: bool = True,
    num_processes: int = None,
    max_connections: int = 20
):
    """
    Fetch options chain data for a given underlying ticker.
    
    Args:
        client: Polygon RESTClient instance
        underlying: Stock ticker symbol (e.g., "SPY", "AAPL")
        start_date: Start date in YYYY-MM-DD format
        end_date: End date in YYYY-MM-DD format
        interval: Data interval - "5min" (ideal) or "hour" (fallback)
        max_contracts: Maximum number of contracts to fetch (None = all)
        include_expired: Whether to include expired contracts
    
    Returns:
        Tuple of (pandas.DataFrame with options chain data, dict mapping contract ticker to contract object)
    """
    print(f"Fetching options chain data for {underlying}")
    print(f"Date range: {start_date} to {end_date}")
    print(f"Interval: {interval}")
    print()
    
    all_data = []
    
    # 1. Get all options contracts for the underlying
    print(f"Fetching options contracts for {underlying}...")
    contracts = []
    
    # Try different symbol formats for indices
    symbol_variants = [underlying.upper()]
    if not underlying.upper().startswith(('I:', 'O:')):
        # Try with I: prefix for indices
        symbol_variants.append(f"I:{underlying.upper()}")
    
    for symbol_variant in symbol_variants:
        try:
            # Fetch contracts - include expired if requested
            contract_params = {
                "underlying_ticker": symbol_variant,
                "limit": 1000
            }
            
            if not include_expired:
                contract_params["expired"] = False
            
            variant_count = 0
            for c in client.list_options_contracts(**contract_params):
                # Avoid duplicates
                if c.ticker not in [existing.ticker for existing in contracts]:
                    contracts.append(c)
                    variant_count += 1
                    if max_contracts and len(contracts) >= max_contracts:
                        break
            
            if variant_count > 0:
                print(f"  Found {variant_count} contracts for {symbol_variant}")
            
            if max_contracts and len(contracts) >= max_contracts:
                break
                
        except Exception as e:
            print(f"  WARNING: Failed to fetch contracts for {symbol_variant}: {e}", file=sys.stderr)
            continue
    
    print(f"Found {len(contracts)} total contracts. Fetching {interval} data...")
    print()
    
    if not contracts:
        print(f"ERROR: No options contracts found for {underlying}", file=sys.stderr)
        print(f"  Tried symbols: {symbol_variants}", file=sys.stderr)
        return pd.DataFrame(), {}
    
    if not contracts:
        print(f"WARNING: No options contracts found for {underlying}", file=sys.stderr)
        return pd.DataFrame(), {}
    
    # Create contracts dictionary
    contracts_dict = {c.ticker: c for c in contracts}
    
    # 2. Set up timespan parameters based on interval
    if interval == "5min":
        multiplier = 5
        timespan = "minute"
    elif interval == "hour":
        multiplier = 1
        timespan = "hour"
    else:
        print(f"ERROR: Unsupported interval '{interval}'. Use '5min' or 'hour'", file=sys.stderr)
        return pd.DataFrame()
    
    # 3. Fetch aggregates for each contract using multiple connections (parallel fetching)
    print(f"Fetching aggregates for {len(contracts)} contracts using {max_connections} connections...", file=sys.stderr)
    
    # Get API key for worker threads
    api_key = os.getenv('POLYGON_API_KEY')
    if not api_key:
        print("ERROR: POLYGON_API_KEY environment variable not set", file=sys.stderr)
        sys.exit(1)
    
    # Prepare contract data for threaded fetching
    fetch_args = []
    for contract in contracts:
        contract_data = {
            'expiration_date': contract.expiration_date,
            'strike_price': contract.strike_price,
            'contract_type': contract.contract_type
        }
        fetch_args.append((
            api_key,
            contract.ticker,
            contract_data,
            start_date,
            end_date,
            multiplier,
            timespan
        ))
    
    # Fetch all raw aggregates data in parallel using ThreadPoolExecutor
    raw_aggregates_data = []  # List of (contract_ticker, contract_data, raw_aggs) tuples
    
    successful_fetches = 0
    failed_fetches = 0
    completed = 0
    
    # Use ThreadPoolExecutor for I/O-bound fetching (API calls)
    with ThreadPoolExecutor(max_workers=max_connections) as executor:
        # Submit all fetch tasks
        future_to_contract = {
            executor.submit(_fetch_contract_aggregates_threaded_range, args): args[1] 
            for args in fetch_args
        }
        
        # Process completed tasks as they finish
        for future in as_completed(future_to_contract):
            completed += 1
            contract_ticker = future_to_contract[future]
            
            try:
                result = future.result()
                
                if result is None:
                    # No data for this contract
                    continue
                elif isinstance(result, dict) and 'error' in result:
                    failed_fetches += 1
                    if failed_fetches <= 5:
                        print(f"WARNING: Failed to fetch data for {contract_ticker}: {result.get('error', 'Unknown error')}", file=sys.stderr)
                else:
                    # Success - add data
                    raw_aggregates_data.append(result)
                    successful_fetches += 1
                
                # Progress indicator
                if completed % 10 == 0 or completed == len(fetch_args):
                    print(f"Fetched {completed}/{len(fetch_args)} contracts... ({successful_fetches} successful, {failed_fetches} failed)", file=sys.stderr)
                    
            except Exception as e:
                failed_fetches += 1
                if failed_fetches <= 5:
                    print(f"WARNING: Exception processing result for {contract_ticker}: {e}", file=sys.stderr)
    
    print(f"\nFetching completed: {successful_fetches} successful, {failed_fetches} failed", file=sys.stderr)
    
    # 4. Process raw aggregates data in parallel (if multiprocessing enabled)
    if num_processes is None:
        num_processes = 1  # Default to sequential
    
    use_multiprocessing = len(raw_aggregates_data) > 1 and num_processes > 1
    
    if use_multiprocessing:
        # Limit processes to number of contracts and available CPUs
        num_processes = min(len(raw_aggregates_data), num_processes, cpu_count())
        
        print(f"Processing {len(raw_aggregates_data)} contracts using {num_processes} processes...", file=sys.stderr)
        
        # Prepare arguments for processing workers
        process_args = []
        for contract_ticker, contract_data, raw_aggs in raw_aggregates_data:
            process_args.append((
                contract_ticker,
                contract_data,
                underlying,
                raw_aggs
            ))
        
        # Use multiprocessing Pool to process data in parallel
        with Pool(processes=num_processes) as pool:
            results = pool.map(_process_aggregates_worker, process_args)
        
        # Collect processed results
        successful_contracts = 0
        failed_contracts = 0
        
        for result in results:
            if result is None:
                continue  # No data for this contract
            elif isinstance(result, dict) and 'error' in result:
                failed_contracts += 1
                if failed_contracts <= 5:
                    print(f"WARNING: Failed to process data for {result.get('ticker', 'unknown')}: {result.get('error', 'Unknown error')}", file=sys.stderr)
            else:
                # Success - add data
                all_data.extend(result)
                successful_contracts += 1
        
        print(f"Processing completed: {successful_contracts} successful, {failed_contracts} failed", file=sys.stderr)
    else:
        # Sequential processing
        print(f"Processing {len(raw_aggregates_data)} contracts sequentially...", file=sys.stderr)
        
        successful_contracts = 0
        for contract_ticker, contract_data, raw_aggs in raw_aggregates_data:
            result = _process_aggregates_worker((contract_ticker, contract_data, underlying, raw_aggs))
            if result and not (isinstance(result, dict) and 'error' in result):
                all_data.extend(result)
                successful_contracts += 1
        
        print(f"Processing completed: {successful_contracts} successful", file=sys.stderr)
    
    if not all_data:
        print("WARNING: No data retrieved for any contracts", file=sys.stderr)
        return pd.DataFrame()
    
    # Convert to DataFrame
    df = pd.DataFrame(all_data)
    
    # Sort by timestamp
    df = df.sort_values('timestamp').reset_index(drop=True)
    
    print(f"\nRetrieved {len(df)} total records")
    if not df.empty:
        print(f"Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
        print(f"Unique contracts: {df['ticker'].nunique()}")
    
    return df, contracts_dict


def _fetch_contract_aggregates_threaded(args_tuple):
    """
    Worker function for threaded fetching - fetches aggregates for a single contract.
    This function must be at module level to be picklable.
    
    Args:
        args_tuple: Tuple of (api_key, contract_ticker, contract_data, target_date, multiplier, timespan)
    
    Returns:
        Tuple of (contract_ticker, contract_data, raw_aggs) or None on error
    """
    api_key, contract_ticker, contract_data, target_date, multiplier, timespan = args_tuple
    
    try:
        # Create client in worker thread
        client = RESTClient(api_key)
        
        # Fetch aggregates
        aggs = []
        for agg in client.list_aggs(
            ticker=contract_ticker,
            multiplier=multiplier,
            timespan=timespan,
            from_=target_date,
            to=target_date,
            limit=50000
        ):
            aggs.append(agg)
        
        if not aggs:
            return None  # No data for this contract
        
        return (contract_ticker, contract_data, aggs)
        
    except Exception as e:
        # Return error info
        return {'error': str(e), 'ticker': contract_ticker}


def get_0dte_options_for_date(
    client: RESTClient,
    underlying: str,
    target_date: str,
    interval: str = "5min",
    num_processes: int = None,
    max_connections: int = 20
):
    """
    Fetch all options expiring on a specific date (0DTE) and their prices throughout that day.
    
    Args:
        client: Polygon RESTClient instance
        underlying: Stock ticker symbol (e.g., "SPY", "AAPL")
        target_date: Target date in YYYY-MM-DD format (expiration date)
        interval: Data interval - "5min" (ideal) or "hour" (fallback)
        num_processes: Number of parallel processes for data processing
    
    Returns:
        Tuple of (pandas.DataFrame with options chain data, dict mapping contract ticker to contract object)
    """
    print(f"Fetching 0DTE options for {underlying} expiring on {target_date}")
    print(f"Interval: {interval}")
    print()
    
    all_data = []
    
    # 1. Get all options contracts expiring on the target date
    # For historical data, we need to include expired contracts
    print(f"Fetching options contracts expiring on {target_date}...")
    contracts = []
    
    # Determine if this is a historical date (in the past)
    target_dt = datetime.strptime(target_date, '%Y-%m-%d')
    today_dt = datetime.now()
    is_historical = target_dt.date() < today_dt.date()
    
    if is_historical:
        print(f"Note: {target_date} is in the past - will fetch expired contracts", file=sys.stderr)
    
    # Try different symbol formats for indices
    symbol_variants = [underlying.upper()]
    if not underlying.upper().startswith(('I:', 'O:')):
        # Try with I: prefix for indices
        symbol_variants.append(f"I:{underlying.upper()}")
    
    all_contracts_found = False
    
    for symbol_variant in symbol_variants:
        try:
            # First try with expired=True (for historical data)
            if is_historical:
                print(f"  Trying {symbol_variant} with expired=True...", file=sys.stderr)
                contract_params_expired = {
                    "underlying_ticker": symbol_variant,
                    "expiration_date_gte": target_date,
                    "expiration_date_lte": target_date,
                    "limit": 1000,
                    "expired": True
                }
                
                for c in client.list_options_contracts(**contract_params_expired):
                    contracts.append(c)
                
                if contracts:
                    print(f"  Found {len(contracts)} expired contracts for {symbol_variant}", file=sys.stderr)
                    all_contracts_found = True
                    break
            
            # Also try with expired=False (in case some are still active)
            print(f"  Trying {symbol_variant} with expired=False...", file=sys.stderr)
            contract_params_active = {
                "underlying_ticker": symbol_variant,
                "expiration_date_gte": target_date,
                "expiration_date_lte": target_date,
                "limit": 1000,
                "expired": False
            }
            
            active_count = 0
            for c in client.list_options_contracts(**contract_params_active):
                # Avoid duplicates
                if c.ticker not in [existing.ticker for existing in contracts]:
                    contracts.append(c)
                    active_count += 1
            
            if active_count > 0:
                print(f"  Found {active_count} active contracts for {symbol_variant}", file=sys.stderr)
            
            if contracts:
                all_contracts_found = True
                break
                
        except Exception as e:
            print(f"  WARNING: Failed to fetch contracts for {symbol_variant}: {e}", file=sys.stderr)
            continue
    
    if not contracts:
        print(f"ERROR: No options contracts found expiring on {target_date}", file=sys.stderr)
        print(f"  Tried symbols: {symbol_variants}", file=sys.stderr)
        print(f"  Note: For historical dates, contracts must be expired. Make sure the date is correct.", file=sys.stderr)
        return pd.DataFrame(), {}
    
    print(f"Found {len(contracts)} total contracts expiring on {target_date}")
    print()
    
    # Create contracts dictionary
    contracts_dict = {c.ticker: c for c in contracts}
    
    # 2. Set up timespan parameters based on interval
    if interval == "5min":
        multiplier = 5
        timespan = "minute"
    elif interval == "hour":
        multiplier = 1
        timespan = "hour"
    else:
        print(f"ERROR: Unsupported interval '{interval}'. Use '5min' or 'hour'", file=sys.stderr)
        return pd.DataFrame(), {}
    
    # 3. Fetch aggregates for the target date using multiple connections (parallel fetching)
    print(f"Fetching {interval} aggregates for {len(contracts)} contracts on {target_date} using {max_connections} connections...", file=sys.stderr)
    
    # Get API key for worker threads
    api_key = os.getenv('POLYGON_API_KEY')
    if not api_key:
        print("ERROR: POLYGON_API_KEY environment variable not set", file=sys.stderr)
        sys.exit(1)
    
    # Prepare contract data for threaded fetching
    fetch_args = []
    for contract in contracts:
        contract_data = {
            'expiration_date': contract.expiration_date,
            'strike_price': contract.strike_price,
            'contract_type': contract.contract_type
        }
        fetch_args.append((
            api_key,
            contract.ticker,
            contract_data,
            target_date,
            multiplier,
            timespan
        ))
    
    # Fetch all raw aggregates data in parallel using ThreadPoolExecutor
    raw_aggregates_data = []  # List of (contract_ticker, contract_data, raw_aggs) tuples
    
    successful_fetches = 0
    failed_fetches = 0
    completed = 0
    
    # Use ThreadPoolExecutor for I/O-bound fetching (API calls)
    with ThreadPoolExecutor(max_workers=max_connections) as executor:
        # Submit all fetch tasks
        future_to_contract = {
            executor.submit(_fetch_contract_aggregates_threaded, args): args[1] 
            for args in fetch_args
        }
        
        # Process completed tasks as they finish
        for future in as_completed(future_to_contract):
            completed += 1
            contract_ticker = future_to_contract[future]
            
            try:
                result = future.result()
                
                if result is None:
                    # No data for this contract
                    continue
                elif isinstance(result, dict) and 'error' in result:
                    failed_fetches += 1
                    if failed_fetches <= 5:
                        print(f"WARNING: Failed to fetch data for {contract_ticker}: {result.get('error', 'Unknown error')}", file=sys.stderr)
                else:
                    # Success - add data
                    raw_aggregates_data.append(result)
                    successful_fetches += 1
                
                # Progress indicator
                if completed % 10 == 0 or completed == len(fetch_args):
                    print(f"Fetched {completed}/{len(fetch_args)} contracts... ({successful_fetches} successful, {failed_fetches} failed)", file=sys.stderr)
                    
            except Exception as e:
                failed_fetches += 1
                if failed_fetches <= 5:
                    print(f"WARNING: Exception processing result for {contract_ticker}: {e}", file=sys.stderr)
    
    print(f"\nFetching completed: {successful_fetches} successful, {failed_fetches} failed", file=sys.stderr)
    
    # 4. Process raw aggregates data in parallel (if multiprocessing enabled)
    if num_processes is None:
        num_processes = 1  # Default to sequential
    
    use_multiprocessing = len(raw_aggregates_data) > 1 and num_processes > 1
    
    if use_multiprocessing:
        # Limit processes to number of contracts and available CPUs
        num_processes = min(len(raw_aggregates_data), num_processes, cpu_count())
        
        print(f"Processing {len(raw_aggregates_data)} contracts using {num_processes} processes...", file=sys.stderr)
        
        # Prepare arguments for processing workers
        process_args = []
        for contract_ticker, contract_data, raw_aggs in raw_aggregates_data:
            process_args.append((
                contract_ticker,
                contract_data,
                underlying,
                raw_aggs
            ))
        
        # Use multiprocessing Pool to process data in parallel
        with Pool(processes=num_processes) as pool:
            results = pool.map(_process_aggregates_worker, process_args)
        
        # Collect processed results
        successful_contracts = 0
        failed_contracts = 0
        
        for result in results:
            if result is None:
                continue  # No data for this contract
            elif isinstance(result, dict) and 'error' in result:
                failed_contracts += 1
                if failed_contracts <= 5:
                    print(f"WARNING: Failed to process data for {result.get('ticker', 'unknown')}: {result.get('error', 'Unknown error')}", file=sys.stderr)
            else:
                # Success - add data
                all_data.extend(result)
                successful_contracts += 1
        
        print(f"Processing completed: {successful_contracts} successful, {failed_contracts} failed", file=sys.stderr)
    else:
        # Sequential processing
        print(f"Processing {len(raw_aggregates_data)} contracts sequentially...", file=sys.stderr)
        
        successful_contracts = 0
        for contract_ticker, contract_data, raw_aggs in raw_aggregates_data:
            result = _process_aggregates_worker((contract_ticker, contract_data, underlying, raw_aggs))
            if result and not (isinstance(result, dict) and 'error' in result):
                all_data.extend(result)
                successful_contracts += 1
        
        print(f"Processing completed: {successful_contracts} successful", file=sys.stderr)
    
    if not all_data:
        print("WARNING: No data retrieved for any contracts", file=sys.stderr)
        return pd.DataFrame(), {}
    
    # Convert to DataFrame
    df = pd.DataFrame(all_data)
    
    # Sort by timestamp
    df = df.sort_values('timestamp').reset_index(drop=True)
    
    print(f"\nRetrieved {len(df)} total records for {target_date}")
    if not df.empty:
        print(f"Time range: {df['timestamp'].min()} to {df['timestamp'].max()}")
        print(f"Unique contracts: {df['ticker'].nunique()}")
    
    return df, contracts_dict


def _process_aggregates_worker(args_tuple):
    """
    Worker function for multiprocessing - processes raw aggregates data.
    This function must be at module level to be picklable.
    
    Args:
        args_tuple: Tuple of (contract_ticker, contract_data, underlying, raw_aggs)
    
    Returns:
        List of data dictionaries or None on error
    """
    contract_ticker, contract_data, underlying, raw_aggs = args_tuple
    
    try:
        if not raw_aggs:
            return None  # No data for this contract
        
        # Convert raw aggregates to data rows
        all_data = []
        for bar in raw_aggs:
            all_data.append({
                "ticker": contract_ticker,
                "underlying": underlying.upper(),
                "expiry": contract_data.get('expiration_date', ''),
                "strike": contract_data.get('strike_price', ''),
                "type": contract_data.get('contract_type', ''),
                "timestamp": pd.to_datetime(bar.timestamp, unit='ms', utc=True),
                "open": bar.open,
                "high": bar.high,
                "low": bar.low,
                "close": bar.close,
                "volume": bar.volume,
                "vwap": getattr(bar, 'vwap', None),
                "transactions": getattr(bar, 'transactions', None)
            })
        
        return all_data
        
    except Exception as e:
        # Return error info instead of None to track failures
        return {'error': str(e), 'ticker': contract_ticker}


def get_realtime_options_chain(
    client: RESTClient,
    underlying: str,
    expiration_date: str = None
) -> pd.DataFrame:
    """
    Fetch real-time options chain with bid/ask quotes for all options.
    This is for CURRENT time only, not historical.
    
    Args:
        client: Polygon RESTClient instance
        underlying: Underlying ticker symbol
        expiration_date: Optional expiration date filter (YYYY-MM-DD)
    
    Returns:
        DataFrame with current bid/ask, Greeks, etc. for all options
    """
    print(f"Fetching real-time options chain for {underlying}...")
    
    # Get all snapshots at once
    snapshots = get_all_options_snapshot(client, underlying)
    
    if not snapshots:
        print(f"WARNING: No snapshots found for {underlying}", file=sys.stderr)
        return pd.DataFrame()
    
    # Convert to DataFrame
    rows = []
    for contract_ticker, snapshot in snapshots.items():
        # Parse contract ticker to extract details
        # Format: O:SPXW260121C02800000
        try:
            # Extract expiry from ticker
            ticker_part = contract_ticker.replace('O:', '')
            # Find the expiry date portion (after underlying, before C/P)
            for i, char in enumerate(ticker_part):
                if char in ('C', 'P') and i > 6:  # At least 6 chars for underlying + date
                    # Everything before C/P includes underlying + 6-digit date
                    date_start = i - 6
                    expiry_str = ticker_part[date_start:i]
                    contract_type = 'call' if char == 'C' else 'put'
                    strike_str = ticker_part[i+1:]
                    # Convert strike (last 8 digits, divide by 1000)
                    strike = int(strike_str) / 1000 if strike_str.isdigit() else None
                    # Convert expiry (YYMMDD)
                    expiry = f"20{expiry_str[:2]}-{expiry_str[2:4]}-{expiry_str[4:6]}"
                    break
            else:
                continue  # Couldn't parse ticker
        except Exception:
            continue
        
        # Filter by expiration if specified
        if expiration_date and expiry != expiration_date:
            continue
        
        rows.append({
            'timestamp': snapshot.get('last_quote_timestamp') or pd.Timestamp.now(tz='UTC'),
            'ticker': contract_ticker,
            'type': contract_type,
            'strike': strike,
            'expiration': expiry,
            'bid': snapshot.get('bid'),
            'ask': snapshot.get('ask'),
            'mid': (snapshot.get('bid', 0) + snapshot.get('ask', 0)) / 2 if snapshot.get('bid') and snapshot.get('ask') else None,
            'fmv': snapshot.get('fmv'),
            'delta': snapshot.get('delta'),
            'gamma': snapshot.get('gamma'),
            'theta': snapshot.get('theta'),
            'vega': snapshot.get('vega'),
            'implied_volatility': snapshot.get('implied_volatility'),
            'underlying_price': snapshot.get('underlying_price'),
        })
    
    df = pd.DataFrame(rows)
    
    if not df.empty:
        df = df.sort_values(['expiration', 'strike', 'type']).reset_index(drop=True)
        print(f"Retrieved {len(df)} options contracts with bid/ask")
        if expiration_date:
            print(f"Filtered to expiration: {expiration_date}")
    
    return df


def fetch_historical_quotes(api_key: str, contract_ticker: str, target_date: str, limit: int = 50000) -> list:
    """
    Fetch historical quotes (bid/ask) for an options contract on a specific date.
    
    Args:
        api_key: Polygon API key
        contract_ticker: Options contract ticker (e.g., "O:SPXW260121C02800000")
        target_date: Target date in YYYY-MM-DD format
        limit: Maximum quotes to fetch
    
    Returns:
        List of quote dictionaries with bid, ask, timestamp
    """
    import requests
    
    quotes = []
    
    try:
        # Use Polygon REST API directly for options quotes
        url = f"https://api.polygon.io/v3/quotes/{contract_ticker}"
        params = {
            "timestamp.gte": f"{target_date}T00:00:00Z",
            "timestamp.lte": f"{target_date}T23:59:59Z",
            "limit": limit,
            "sort": "timestamp",
            "order": "asc",
            "apiKey": api_key
        }
        
        response = requests.get(url, params=params, timeout=30)
        
        if response.status_code == 200:
            data = response.json()
            results = data.get('results', [])
            
            for q in results:
                quotes.append({
                    'timestamp': pd.to_datetime(q.get('sip_timestamp'), unit='ns', utc=True) if q.get('sip_timestamp') else None,
                    'bid': q.get('bid_price'),
                    'ask': q.get('ask_price'),
                    'bid_size': q.get('bid_size'),
                    'ask_size': q.get('ask_size'),
                })
        elif response.status_code == 403:
            # Quotes endpoint requires higher tier subscription
            pass
        
    except Exception as e:
        pass
    
    return quotes


def get_all_options_snapshot(client: RESTClient, underlying: str) -> dict:
    """
    Fetch snapshot data (bid/ask, Greeks) for ALL options contracts at once.
    This is much faster than fetching individual snapshots.
    
    Args:
        client: Polygon RESTClient instance
        underlying: Underlying ticker symbol
    
    Returns:
        Dictionary mapping contract ticker to snapshot data
    """
    snapshots = {}
    
    try:
        # Try different symbol formats for indices
        symbol_variants = [underlying.upper()]
        if not underlying.upper().startswith(('I:', 'O:')):
            symbol_variants.append(f"I:{underlying.upper()}")
        
        for symbol_variant in symbol_variants:
            try:
                # Get all options snapshots at once
                for snapshot in client.list_snapshot_options_chain(symbol_variant):
                    contract_ticker = snapshot.details.ticker if hasattr(snapshot, 'details') and snapshot.details else None
                    if not contract_ticker:
                        continue
                    
                    snapshot_data = {
                        'bid': None,
                        'ask': None,
                        'fmv': None,
                        'delta': None,
                        'gamma': None,
                        'theta': None,
                        'vega': None,
                        'implied_volatility': None,
                        'underlying_price': None,
                        'last_quote_timestamp': None
                    }
                    
                    # Get bid/ask from last_quote
                    if hasattr(snapshot, 'last_quote') and snapshot.last_quote:
                        quote_bid = getattr(snapshot.last_quote, 'bid', None)
                        quote_ask = getattr(snapshot.last_quote, 'ask', None)
                        if quote_bid is None:
                            quote_bid = getattr(snapshot.last_quote, 'bid_price', None)
                        if quote_ask is None:
                            quote_ask = getattr(snapshot.last_quote, 'ask_price', None)
                        
                        if quote_bid is not None:
                            snapshot_data['bid'] = quote_bid
                        if quote_ask is not None:
                            snapshot_data['ask'] = quote_ask
                        
                        # Get quote timestamp
                        quote_ts = getattr(snapshot.last_quote, 'sip_timestamp', None)
                        if quote_ts:
                            snapshot_data['last_quote_timestamp'] = pd.to_datetime(quote_ts, unit='ns', utc=True)
                    
                    # Get FMV
                    if hasattr(snapshot, 'fair_market_value') and snapshot.fair_market_value:
                        fmv = getattr(snapshot.fair_market_value, 'value', None)
                        if fmv:
                            snapshot_data['fmv'] = fmv
                    
                    # Get Greeks
                    if hasattr(snapshot, 'greeks') and snapshot.greeks:
                        snapshot_data['delta'] = getattr(snapshot.greeks, 'delta', None)
                        snapshot_data['gamma'] = getattr(snapshot.greeks, 'gamma', None)
                        snapshot_data['theta'] = getattr(snapshot.greeks, 'theta', None)
                        snapshot_data['vega'] = getattr(snapshot.greeks, 'vega', None)
                        snapshot_data['implied_volatility'] = getattr(snapshot.greeks, 'implied_volatility', None)
                    
                    # Get underlying price
                    if hasattr(snapshot, 'underlying_asset') and snapshot.underlying_asset:
                        snapshot_data['underlying_price'] = getattr(snapshot.underlying_asset, 'price', None)
                    
                    snapshots[contract_ticker] = snapshot_data
                
                if snapshots:
                    break  # Found snapshots, no need to try other variants
                    
            except Exception as e:
                continue
                
    except Exception as e:
        print(f"WARNING: Failed to fetch all options snapshots: {e}", file=sys.stderr)
    
    return snapshots


def fetch_snapshot_data(client, underlying: str, contract_ticker: str) -> dict:
    """
    Fetch snapshot data (bid/ask, Greeks, FMV) for an options contract.
    
    Args:
        client: Polygon RESTClient instance
        underlying: Underlying ticker symbol
        contract_ticker: Options contract ticker (e.g., "O:SPXW260121C02800000")
    
    Returns:
        Dictionary with bid, ask, fmv, delta, gamma, theta, vega, implied_volatility
    """
    snapshot_data = {
        'bid': None,
        'ask': None,
        'fmv': None,
        'delta': None,
        'gamma': None,
        'theta': None,
        'vega': None,
        'implied_volatility': None
    }
    
    try:
        snapshot = client.get_snapshot_option(underlying.upper(), contract_ticker)
        
        if snapshot:
            # Get bid/ask from last_quote
            if hasattr(snapshot, 'last_quote') and snapshot.last_quote:
                quote_bid = getattr(snapshot.last_quote, 'bid', None)
                quote_ask = getattr(snapshot.last_quote, 'ask', None)
                if quote_bid is None:
                    quote_bid = getattr(snapshot.last_quote, 'bid_price', None)
                if quote_ask is None:
                    quote_ask = getattr(snapshot.last_quote, 'ask_price', None)
                if quote_bid is None:
                    quote_bid = getattr(snapshot.last_quote, 'bp', None)
                if quote_ask is None:
                    quote_ask = getattr(snapshot.last_quote, 'ap', None)
                
                if quote_bid is not None:
                    snapshot_data['bid'] = quote_bid
                if quote_ask is not None:
                    snapshot_data['ask'] = quote_ask
            
            # Get FMV
            if hasattr(snapshot, 'fair_market_value') and snapshot.fair_market_value:
                fmv = getattr(snapshot.fair_market_value, 'value', None)
                if fmv:
                    snapshot_data['fmv'] = fmv
            
            # Get Greeks
            if hasattr(snapshot, 'greeks') and snapshot.greeks:
                snapshot_data['delta'] = getattr(snapshot.greeks, 'delta', None)
                snapshot_data['gamma'] = getattr(snapshot.greeks, 'gamma', None)
                snapshot_data['theta'] = getattr(snapshot.greeks, 'theta', None)
                snapshot_data['vega'] = getattr(snapshot.greeks, 'vega', None)
                snapshot_data['implied_volatility'] = getattr(snapshot.greeks, 'implied_volatility', None)
            
            # Try alternative IV location
            if snapshot_data['implied_volatility'] is None:
                if hasattr(snapshot, 'implied_volatility') and snapshot.implied_volatility:
                    iv_value = snapshot.implied_volatility
                    if hasattr(iv_value, 'value'):
                        snapshot_data['implied_volatility'] = iv_value.value
                    elif isinstance(iv_value, (int, float)):
                        snapshot_data['implied_volatility'] = iv_value
    except Exception:
        # Snapshot fetch failed - return empty data
        pass
    
    return snapshot_data


def format_chain_csv(df: pd.DataFrame, client, underlying: str, contracts_dict: dict, output_dir: str = None) -> None:
    """
    Format options chain data into the specified CSV format and output by trading day.
    
    For historical data, bid/ask are estimated from aggregate high/low prices:
    - bid = low price (conservative estimate of where buyers were)
    - ask = high price (conservative estimate of where sellers were)
    
    For current day data, real-time snapshots are fetched for actual bid/ask.
    
    Args:
        df: DataFrame with options chain data from aggregates
        client: Polygon RESTClient instance
        underlying: Underlying ticker symbol
        contracts_dict: Dictionary mapping contract ticker to contract object
        output_dir: Directory to save CSV files (None = current directory). If provided, creates subdirectory per ticker.
    """
    if df.empty:
        print("WARNING: No data to format", file=sys.stderr)
        return
    
    # Create output directory if specified
    if output_dir:
        # Create ticker-specific subdirectory
        ticker_dir = Path(output_dir) / underlying.upper()
        
        # Check if path exists and is a file (not a directory)
        if ticker_dir.exists() and not ticker_dir.is_dir():
            print(f"WARNING: {ticker_dir} exists as a file, removing it to create directory", file=sys.stderr)
            ticker_dir.unlink()  # Remove the file
        
        # Create directory (and parent directories if needed)
        ticker_dir.mkdir(parents=True, exist_ok=True)
        base_path = ticker_dir
    else:
        base_path = Path('.')
    
    # Convert timestamp to datetime if needed
    if not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
        df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # Group by trading day (date only, ignoring timezone for grouping)
    df['trading_day'] = df['timestamp'].dt.date
    
    # Get unique trading days
    trading_days = sorted(df['trading_day'].unique())
    
    print(f"\nFormatting data for {len(trading_days)} trading days...", file=sys.stderr)
    
    # Determine if we're processing historical or current data
    today = datetime.now().date()
    
    # Cache snapshot data to avoid repeated API calls for the same contract
    snapshot_cache = {}
    
    # Process each trading day
    for trading_day in trading_days:
        day_df = df[df['trading_day'] == trading_day].copy()
        is_historical = trading_day < today
        
        # Prepare output data
        output_rows = []
        
        # Get unique contracts for this day
        unique_contracts = day_df['ticker'].unique()
        
        # Only fetch snapshots for current/recent data (not historical)
        # Historical snapshots are stale and won't have relevant bid/ask
        if not is_historical:
            contracts_to_fetch = [ct for ct in unique_contracts if ct not in snapshot_cache]
            
            if contracts_to_fetch:
                # Fetch all snapshots sequentially using one client connection
                print(f"  Fetching {len(contracts_to_fetch)} snapshots for current day...", file=sys.stderr)
                for contract_ticker in contracts_to_fetch:
                    snapshot_cache[contract_ticker] = fetch_snapshot_data(client, underlying, contract_ticker)
        else:
            print(f"  {trading_day}: Using aggregate high/low as bid/ask estimates (historical data)", file=sys.stderr)
        
        # Process each row in the day's data (each aggregate bar)
        for _, row in day_df.iterrows():
            contract_ticker = row['ticker']
            
            # For historical data, use aggregate high/low as bid/ask proxy
            # For current data, use snapshot bid/ask
            if is_historical:
                # Use low as bid estimate, high as ask estimate
                # This represents the price range during that interval
                snapshot = {
                    'bid': row.get('low') if pd.notna(row.get('low')) else None,
                    'ask': row.get('high') if pd.notna(row.get('high')) else None,
                    'fmv': None,
                    'delta': None,
                    'gamma': None,
                    'theta': None,
                    'vega': None,
                    'implied_volatility': None
                }
            else:
                snapshot = snapshot_cache.get(contract_ticker, {})
            
            # Get contract info
            contract = contracts_dict.get(contract_ticker)
            if not contract:
                # Fallback to row data if contract not in dict
                contract_type = row.get('type', '').lower()
                strike = row.get('strike', '')
                expiry = row.get('expiry', '')
            else:
                contract_type = contract.contract_type.lower() if hasattr(contract, 'contract_type') else row.get('type', '').lower()
                strike = contract.strike_price if hasattr(contract, 'strike_price') else row.get('strike', '')
                expiry = contract.expiration_date if hasattr(contract, 'expiration_date') else row.get('expiry', '')
            
            # Use close price from aggregates as day_close
            day_close = row.get('close', None)
            
            # Format output row - match exact format from user's example
            # Handle volume - convert to int if valid, otherwise empty string
            volume_val = row.get('volume', '')
            if pd.notna(volume_val) and volume_val != '':
                try:
                    volume_val = int(volume_val)
                except (ValueError, TypeError):
                    volume_val = ''
            else:
                volume_val = ''
            
            # Get vwap from row if available
            vwap_val = row.get('vwap', '')
            if pd.notna(vwap_val) and vwap_val != '':
                try:
                    vwap_val = float(vwap_val)
                except (ValueError, TypeError):
                    vwap_val = ''
            else:
                vwap_val = ''
            
            output_row = {
                'timestamp': row['timestamp'].isoformat(),
                'ticker': contract_ticker,
                'type': contract_type,
                'strike': strike,
                'expiration': expiry,
                'bid': snapshot.get('bid') if snapshot.get('bid') is not None else '',
                'ask': snapshot.get('ask') if snapshot.get('ask') is not None else '',
                'day_close': day_close if day_close is not None and pd.notna(day_close) else '',
                'vwap': vwap_val,
                'fmv': snapshot.get('fmv') if snapshot.get('fmv') is not None else '',
                'delta': snapshot.get('delta') if snapshot.get('delta') is not None else '',
                'gamma': snapshot.get('gamma') if snapshot.get('gamma') is not None else '',
                'theta': snapshot.get('theta') if snapshot.get('theta') is not None else '',
                'vega': snapshot.get('vega') if snapshot.get('vega') is not None else '',
                'implied_volatility': snapshot.get('implied_volatility') if snapshot.get('implied_volatility') is not None else '',
                'volume': volume_val
            }
            
            output_rows.append(output_row)
        
        # Create DataFrame for this day
        if output_rows:
            day_output_df = pd.DataFrame(output_rows)
            
            # Sort by timestamp, then by strike, then by type
            day_output_df = day_output_df.sort_values(['timestamp', 'strike', 'type']).reset_index(drop=True)
            
            # Generate filename: {ticker}_options_{YYYY-MM-DD}.csv
            filename = f"{underlying.upper()}_options_{trading_day.strftime('%Y-%m-%d')}.csv"
            filepath = base_path / filename
            
            # Write CSV with exact column order
            columns_order = ['timestamp', 'ticker', 'type', 'strike', 'expiration', 'bid', 'ask', 
                           'day_close', 'vwap', 'fmv', 'delta', 'gamma', 'theta', 'vega', 'implied_volatility', 'volume']
            day_output_df = day_output_df[columns_order]
            day_output_df.to_csv(filepath, index=False)
            print(f"  Saved {len(day_output_df)} records to {filepath}", file=sys.stderr)
    
    print(f"\nCompleted formatting. Output {len(trading_days)} CSV files.", file=sys.stderr)


def process_single_ticker(
    ticker: str,
    args,
    zero_dte_mode: str,
    api_key: str,
    num_processes: int,
    max_connections: int
) -> tuple:
    """
    Process a single ticker - fetch data and return results.
    
    Args:
        ticker: Ticker symbol to process
        args: Parsed command-line arguments
        zero_dte_mode: Mode for 0DTE ('single', 'range', or None)
        api_key: Polygon API key
        num_processes: Number of parallel processes for data processing
        max_connections: Maximum number of concurrent connections
    
    Returns:
        Tuple of (ticker, df, contracts_dict, success) where success is a boolean
    """
    try:
        print(f"\n{'='*80}", file=sys.stderr)
        print(f"Processing ticker: {ticker}", file=sys.stderr)
        print(f"{'='*80}", file=sys.stderr)
        
        client = RESTClient(api_key)
        
        # Fetch data - check if 0DTE mode
        if zero_dte_mode == 'single':
            # 0DTE mode: fetch all options expiring on specific date
            df, contracts_dict = get_0dte_options_for_date(
                client=client,
                underlying=ticker,
                target_date=args.zero_dte_date,
                interval=args.interval,
                num_processes=num_processes,
                max_connections=max_connections
            )
        elif zero_dte_mode == 'range':
            # 0DTE mode: fetch all options expiring on each date in range
            start_dt = datetime.strptime(args.zero_dte_date_start, '%Y-%m-%d')
            end_dt = datetime.strptime(args.zero_dte_date_end, '%Y-%m-%d')
            
            # Generate list of dates (inclusive)
            date_list = []
            current_dt = start_dt
            while current_dt <= end_dt:
                date_list.append(current_dt.strftime('%Y-%m-%d'))
                current_dt += timedelta(days=1)
            
            print(f"Fetching 0DTE options for {len(date_list)} dates: {args.zero_dte_date_start} to {args.zero_dte_date_end}", file=sys.stderr)
            print()
            
            # Fetch data for each date and combine
            all_dfs = []
            all_contracts_dict = {}
            
            for i, target_date in enumerate(date_list, 1):
                print(f"\n  Processing date {i}/{len(date_list)}: {target_date}", file=sys.stderr)
                
                date_df, date_contracts_dict = get_0dte_options_for_date(
                    client=client,
                    underlying=ticker,
                    target_date=target_date,
                    interval=args.interval,
                    num_processes=num_processes,
                    max_connections=max_connections
                )
                
                if not date_df.empty:
                    all_dfs.append(date_df)
                    all_contracts_dict.update(date_contracts_dict)
            
            # Combine all DataFrames
            if all_dfs:
                df = pd.concat(all_dfs, ignore_index=True)
                df = df.sort_values('timestamp').reset_index(drop=True)
                contracts_dict = all_contracts_dict
                print(f"\n  Combined data from {len(date_list)} dates: {len(df)} total records", file=sys.stderr)
            else:
                df = pd.DataFrame()
                contracts_dict = {}
                print(f"\n  No data retrieved for any of the {len(date_list)} dates", file=sys.stderr)
        else:
            # Regular date range mode
            df, contracts_dict = get_options_chain_data(
                client=client,
                underlying=ticker,
                start_date=args.start,
                end_date=args.end,
                interval=args.interval,
                max_contracts=args.max_contracts,
                include_expired=not args.exclude_expired,
                num_processes=num_processes,
                max_connections=max_connections
            )
        
        if df.empty:
            print(f"WARNING: No data retrieved for {ticker}", file=sys.stderr)
            return (ticker, pd.DataFrame(), {}, False)
        
        return (ticker, df, contracts_dict, True)
        
    except Exception as e:
        print(f"ERROR: Failed to process {ticker}: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return (ticker, pd.DataFrame(), {}, False)


def main():
    parser = argparse.ArgumentParser(
        description="Download options chain data from Polygon.io API",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Fetch options for a date range
  %(prog)s SPY --start 2025-01-01 --end 2025-01-31
  %(prog)s AAPL --start 2025-01-01 --end 2025-01-31 --interval hour
  
  # Fetch multiple tickers simultaneously
  %(prog)s SPY AAPL QQQ --start 2025-01-01 --end 2025-01-31 --format-chain-csv --output-dir ./output
  %(prog)s SPY AAPL --zero-dte-date 2026-01-10 --format-chain-csv --output-dir ./output
  
  # Fetch all 0DTE options expiring on a specific date
  %(prog)s SPY --zero-dte-date 2026-01-10
  %(prog)s SPY --zero-dte-date 2026-01-10 --format-chain-csv --output-dir ./output
  %(prog)s SPY --zero-dte-date 2026-01-10 --max-connections 30 --num-processes 8
  
  # Fetch all 0DTE options for a date range (each date in the range)
  %(prog)s SPY --zero-dte-date-start 2026-01-10 --zero-dte-date-end 2026-01-15
  %(prog)s SPY --zero-dte-date-start 2026-01-10 --zero-dte-date-end 2026-01-15 --format-chain-csv --output-dir ./output
  %(prog)s SPY --zero-dte-date-start 2026-01-10 --zero-dte-date-end 2026-01-15 --max-connections 30 --num-processes 8
  
  # Other options
  %(prog)s SPY --start 2025-01-01 --end 2025-01-31 --max-contracts 100 --output spy_options.csv
  %(prog)s SPY --start 2025-01-01 --end 2025-01-31 --format-chain-csv --output-dir ./output
  %(prog)s SPY --start 2025-01-01 --end 2025-01-31 --num-processes 8 --max-connections 20
        """
    )
    
    parser.add_argument(
        'ticker',
        type=str,
        nargs='+',
        help='Underlying ticker symbol(s) (e.g., SPY, AAPL). Can specify multiple tickers to download simultaneously.'
    )
    
    parser.add_argument(
        '--start',
        type=str,
        default=None,
        help='Start date in YYYY-MM-DD format (required unless --0dte-date is used)'
    )
    
    parser.add_argument(
        '--end',
        type=str,
        default=None,
        help='End date in YYYY-MM-DD format (required unless --0dte-date is used)'
    )
    
    parser.add_argument(
        '--zero-dte-date',
        '--0dte-date',
        type=str,
        default=None,
        metavar='DATE',
        dest='zero_dte_date',
        help='Fetch all 0DTE options expiring on this date (YYYY-MM-DD). When specified, fetches only options expiring on this exact date and their prices throughout that day. Mutually exclusive with --start/--end and --zero-dte-date-start/--zero-dte-date-end.'
    )
    
    parser.add_argument(
        '--zero-dte-date-start',
        type=str,
        default=None,
        metavar='DATE',
        dest='zero_dte_date_start',
        help='Start date for 0DTE options range (YYYY-MM-DD). Fetches 0DTE options for each date from start to end (inclusive). Must be used with --zero-dte-date-end. Mutually exclusive with --start/--end and --zero-dte-date.'
    )
    
    parser.add_argument(
        '--zero-dte-date-end',
        type=str,
        default=None,
        metavar='DATE',
        dest='zero_dte_date_end',
        help='End date for 0DTE options range (YYYY-MM-DD). Fetches 0DTE options for each date from start to end (inclusive). Must be used with --zero-dte-date-start. Mutually exclusive with --start/--end and --zero-dte-date.'
    )
    
    parser.add_argument(
        '--interval',
        type=str,
        choices=['5min', 'hour'],
        default='5min',
        help='Data interval: 5min (ideal, every 5 minutes) or hour (fallback, every hour). Default: 5min'
    )
    
    parser.add_argument(
        '--max-contracts',
        type=int,
        default=None,
        help='Maximum number of contracts to fetch (default: all available)'
    )
    
    parser.add_argument(
        '--exclude-expired',
        action='store_true',
        help='Exclude expired contracts (default: include expired)'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Output CSV file path (default: print to stdout)'
    )
    
    parser.add_argument(
        '--format-chain-csv',
        action='store_true',
        help='Output formatted CSV files (one per trading day) with bid/ask, Greeks, etc. Format: timestamp,ticker,type,strike,expiration,bid,ask,day_close,fmv,delta,gamma,theta,vega,implied_volatility,volume'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default=None,
        help='Output directory for formatted CSV files (default: current directory). Only used with --format-chain-csv'
    )
    
    parser.add_argument(
        '--num-processes',
        type=int,
        default=None,
        help=f'Number of parallel processes to use for data processing (default: {max(1, cpu_count() - 1)}). Set to 1 to disable multiprocessing.'
    )
    
    parser.add_argument(
        '--max-connections',
        type=int,
        default=20,
        help='Maximum number of concurrent connections for fetching data (default: 20). Higher values may speed up fetching but could hit rate limits.'
    )
    
    args = parser.parse_args()
    
    # Validate arguments
    zero_dte_mode = None  # 'single', 'range', or None
    
    if args.zero_dte_date:
        zero_dte_mode = 'single'
        try:
            datetime.strptime(args.zero_dte_date, '%Y-%m-%d')
        except ValueError:
            print("ERROR: --zero-dte-date must be in YYYY-MM-DD format", file=sys.stderr)
            sys.exit(1)
        
        if args.start or args.end:
            print("ERROR: --zero-dte-date cannot be used with --start/--end", file=sys.stderr)
            sys.exit(1)
        
        if args.zero_dte_date_start or args.zero_dte_date_end:
            print("ERROR: --zero-dte-date cannot be used with --zero-dte-date-start/--zero-dte-date-end", file=sys.stderr)
            sys.exit(1)
    
    elif args.zero_dte_date_start or args.zero_dte_date_end:
        zero_dte_mode = 'range'
        
        if not args.zero_dte_date_start or not args.zero_dte_date_end:
            print("ERROR: Both --zero-dte-date-start and --zero-dte-date-end must be specified", file=sys.stderr)
            sys.exit(1)
        
        try:
            start_dt = datetime.strptime(args.zero_dte_date_start, '%Y-%m-%d')
            end_dt = datetime.strptime(args.zero_dte_date_end, '%Y-%m-%d')
        except ValueError:
            print("ERROR: --zero-dte-date-start and --zero-dte-date-end must be in YYYY-MM-DD format", file=sys.stderr)
            sys.exit(1)
        
        if start_dt > end_dt:
            print("ERROR: --zero-dte-date-start must be before or equal to --zero-dte-date-end", file=sys.stderr)
            sys.exit(1)
        
        if args.start or args.end:
            print("ERROR: --zero-dte-date-start/--zero-dte-date-end cannot be used with --start/--end", file=sys.stderr)
            sys.exit(1)
    
    else:
        # Regular date range mode
        if not args.start or not args.end:
            print("ERROR: Either --start/--end, --zero-dte-date, or --zero-dte-date-start/--zero-dte-date-end must be specified", file=sys.stderr)
            sys.exit(1)
        
        try:
            datetime.strptime(args.start, '%Y-%m-%d')
            datetime.strptime(args.end, '%Y-%m-%d')
        except ValueError:
            print("ERROR: Dates must be in YYYY-MM-DD format", file=sys.stderr)
            sys.exit(1)
    
    # Get API key for client creation
    api_key = os.getenv('POLYGON_API_KEY')
    if not api_key:
        print("ERROR: POLYGON_API_KEY environment variable not set", file=sys.stderr)
        sys.exit(1)
    
    # Determine number of processes
    if args.num_processes is None:
        num_processes = max(1, cpu_count() - 1)  # Default to CPU count - 1
    else:
        num_processes = max(1, args.num_processes)  # At least 1
    
    # Validate max_connections
    max_connections = max(1, args.max_connections)  # At least 1 connection
    
    if num_processes > 1:
        print(f"Using {num_processes} parallel processes for data processing", file=sys.stderr)
    
    print(f"Using {max_connections} concurrent connections for fetching", file=sys.stderr)
    
    # Get list of tickers (support multiple)
    tickers = args.ticker if isinstance(args.ticker, list) else [args.ticker]
    
    print(f"\nProcessing {len(tickers)} ticker(s): {', '.join(tickers)}", file=sys.stderr)
    
    # Process multiple tickers in parallel
    if len(tickers) > 1:
        print(f"Processing {len(tickers)} tickers simultaneously...", file=sys.stderr)
        
        # Use ThreadPoolExecutor to process multiple tickers in parallel
        # Limit to reasonable number to avoid overwhelming the API
        max_ticker_workers = min(len(tickers), 5)  # Process up to 5 tickers at once
        
        results = []
        with ThreadPoolExecutor(max_workers=max_ticker_workers) as executor:
            # Submit all ticker processing tasks
            future_to_ticker = {
                executor.submit(process_single_ticker, ticker, args, zero_dte_mode, api_key, num_processes, max_connections): ticker
                for ticker in tickers
            }
            
            # Collect results as they complete
            for future in as_completed(future_to_ticker):
                ticker = future_to_ticker[future]
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    print(f"ERROR: Exception processing {ticker}: {e}", file=sys.stderr)
                    results.append((ticker, pd.DataFrame(), {}, False))
    else:
        # Single ticker - process directly
        result = process_single_ticker(tickers[0], args, zero_dte_mode, api_key, num_processes, max_connections)
        results = [result]
    
    # Process results for each ticker
    all_successful = True
    for ticker, df, contracts_dict, success in results:
        if not success:
            all_successful = False
            continue
        
        if df.empty:
            print(f"WARNING: No data retrieved for {ticker}", file=sys.stderr)
            all_successful = False
            continue
        
        # Output results for this ticker
        if args.format_chain_csv:
            # Output formatted CSV files by trading day
            # Create a client for this ticker (needed for snapshot fetching)
            client = RESTClient(api_key)
            format_chain_csv(df, client, ticker, contracts_dict, args.output_dir)
        elif args.output:
            # For multiple tickers with --output, append ticker to filename
            if len(tickers) > 1:
                output_path = Path(args.output)
                ticker_output = output_path.parent / f"{output_path.stem}_{ticker}{output_path.suffix}"
                df.to_csv(ticker_output, index=False)
                print(f"\nData for {ticker} saved to {ticker_output}", file=sys.stderr)
            else:
                df.to_csv(args.output, index=False)
                print(f"\nData saved to {args.output}")
        else:
            # Print to stdout
            print(f"\n{'='*80}")
            print(f"Options Chain Data for {ticker}:")
            print("=" * 80)
            print(df.to_string())
            print("=" * 80)
            
            # Also show summary statistics
            print(f"\nSummary for {ticker}:")
            print(f"  Total records: {len(df)}")
            print(f"  Unique contracts: {df['ticker'].nunique()}")
            print(f"  Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
            print(f"  Expiry dates: {df['expiry'].nunique()} unique")
            print(f"  Contract types: {df['type'].value_counts().to_dict()}")
    
    if not all_successful:
        print("\nWARNING: Some tickers failed to process or returned no data.", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    # Windows multiprocessing guard
    if sys.platform == 'win32':
        # On Windows, multiprocessing requires this guard
        multiprocessing.freeze_support()
    main()
