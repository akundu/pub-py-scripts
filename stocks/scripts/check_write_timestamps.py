#!/usr/bin/env python3
"""
Diagnostic script to check write_timestamp values for options data.
Helps identify why options might be filtered out by min_write_timestamp.
"""

import sys
import os
import asyncio
import logging
from datetime import datetime, timezone
from pathlib import Path

# Add project root to path
CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Import after path setup
import pandas as pd
import pytz
from common.stock_db import get_stock_db

LOG_FORMAT = "%(asctime)s [%(levelname)s] %(name)s - %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
LOGGER = logging.getLogger("scripts.check_write_timestamps")


def configure_logging(log_level: str) -> int:
    """Configure root logging with timestamps."""
    level = getattr(logging, log_level.upper(), logging.INFO)
    root_logger = logging.getLogger()
    root_logger.setLevel(level)

    formatter = logging.Formatter(LOG_FORMAT, DATE_FORMAT)
    if not root_logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(formatter)
        root_logger.addHandler(handler)
    else:
        for handler in root_logger.handlers:
            handler.setFormatter(formatter)

    logging.getLogger("asyncio").setLevel(logging.WARNING)
    logging.getLogger("aioredis").setLevel(logging.WARNING)
    return level


def _log_print(*args, **kwargs):
    """Replacement for print that routes through logging with timestamps."""
    level = kwargs.pop("level", logging.INFO)
    sep = kwargs.pop("sep", " ")
    end = kwargs.pop("end", "\n")
    kwargs.pop("file", None)
    kwargs.pop("flush", None)
    message = sep.join(str(arg) for arg in args)
    if end and end != "\n":
        message = f"{message}{end}"
    LOGGER.log(level, message)


# Override print inside this module so existing calls include timestamps
print = _log_print  # type: ignore


async def check_write_timestamps(
    ticker: str,
    db_conn: str,
    min_write_timestamp: str = None,
    enable_cache: bool = False,
    log_level: str = "INFO",
):
    """Check write_timestamp values for a specific ticker."""
    
    log_level = (log_level or "INFO").upper()
    print(f"\n{'='*80}")
    print(f"Checking write_timestamps for {ticker}")
    print(f"{'='*80}\n")
    
    # Initialize database
    cache_status = "ENABLED" if enable_cache else "DISABLED"
    print(f"Cache: {cache_status}\n")
    db = get_stock_db('questdb', db_config=db_conn, enable_cache=enable_cache, log_level=log_level)
    try:
        # Initialize database connection
        if hasattr(db, 'initialize'):
            await db.initialize()
        elif hasattr(db, '_init_db'):
            await db._init_db()
        else:
            raise RuntimeError("Database object has no initialize or _init_db method")
        # Get current time
        now_utc = datetime.now(timezone.utc)
        now_est = now_utc.astimezone(pytz.timezone('America/New_York'))
        print(f"Current time (UTC): {now_utc}")
        print(f"Current time (EST): {now_est}\n")
        
        # Parse min_write_timestamp if provided
        min_ts_utc = None
        if min_write_timestamp:
            est = pytz.timezone('America/New_York')
            min_ts = pd.to_datetime(min_write_timestamp)
            if min_ts.tz is None:
                min_ts = est.localize(min_ts)
            min_ts_utc = min_ts.astimezone(pytz.UTC)
            print(f"Min write timestamp (EST): {min_write_timestamp}")
            print(f"Min write timestamp (UTC): {min_ts_utc}")
            print(f"Age threshold: {(now_utc - min_ts_utc).total_seconds() / 3600:.1f} hours\n")
        
        # Fetch options data for the ticker
        print(f"Fetching options data for {ticker}...")
        options_df = await db.get_latest_options_data_batch(
            tickers=[ticker],
            start_datetime=None,
            end_datetime=None,
            max_concurrent=1,
            batch_size=1,
            timestamp_lookback_days=30  # Look back 30 days
        )
        
        # Also fetch short-term options (expiring within 21 days) to match the analysis
        from datetime import timedelta
        today = datetime.now(timezone.utc).date()
        short_end_date = (today + timedelta(days=21)).strftime('%Y-%m-%d')
        print(f"\nFetching short-term options (expiring by {short_end_date})...")
        short_term_df = await db.get_latest_options_data_batch(
            tickers=[ticker],
            start_datetime=None,
            end_datetime=short_end_date,
            max_concurrent=1,
            batch_size=1,
            timestamp_lookback_days=30
        )
        
        if options_df.empty:
            print(f"❌ No options data found for {ticker} in the database")
            return
        
        print(f"✅ Found {len(options_df)} total options records for {ticker}")
        if not short_term_df.empty:
            print(f"✅ Found {len(short_term_df)} short-term options (expiring within 21 days)\n")
        else:
            print(f"⚠️  No short-term options found (expiring within 21 days)\n")
        
        # Use short-term options for detailed analysis if available, otherwise use all
        analysis_df = short_term_df if not short_term_df.empty else options_df
        analysis_label = "short-term" if not short_term_df.empty else "all"
        
        # Check if write_timestamp column exists
        if 'write_timestamp' not in analysis_df.columns:
            print("❌ WARNING: 'write_timestamp' column not found in options data!")
            print(f"Available columns: {list(analysis_df.columns)}")
            return
        
        # Convert write_timestamp to UTC if needed
        if analysis_df['write_timestamp'].dtype == 'object':
            analysis_df['write_timestamp'] = pd.to_datetime(analysis_df['write_timestamp'], errors='coerce')
        
        # Normalize to UTC
        from common.common import normalize_timestamp_to_utc
        analysis_df['write_timestamp_utc'] = analysis_df['write_timestamp'].apply(normalize_timestamp_to_utc)
        
        # Calculate age in hours
        analysis_df['age_hours'] = (now_utc - analysis_df['write_timestamp_utc']).dt.total_seconds() / 3600
        
        # Also analyze all options for comparison
        if not short_term_df.empty and len(options_df) != len(short_term_df):
            if 'write_timestamp' in options_df.columns:
                if options_df['write_timestamp'].dtype == 'object':
                    options_df['write_timestamp'] = pd.to_datetime(options_df['write_timestamp'], errors='coerce')
                options_df['write_timestamp_utc'] = options_df['write_timestamp'].apply(normalize_timestamp_to_utc)
                options_df['age_hours'] = (now_utc - options_df['write_timestamp_utc']).dt.total_seconds() / 3600
        
        # Statistics
        print(f"{'='*80}")
        print(f"WRITE_TIMESTAMP STATISTICS ({analysis_label.upper()} OPTIONS)")
        print(f"{'='*80}")
        print(f"Total records: {len(analysis_df)}")
        print(f"Records with valid write_timestamp: {analysis_df['write_timestamp_utc'].notna().sum()}")
        print(f"Records with null write_timestamp: {analysis_df['write_timestamp_utc'].isna().sum()}\n")
        
        if analysis_df['write_timestamp_utc'].notna().any():
            latest_ts = analysis_df['write_timestamp_utc'].max()
            oldest_ts = analysis_df['write_timestamp_utc'].min()
            latest_age_hours = (now_utc - latest_ts).total_seconds() / 3600
            oldest_age_hours = (now_utc - oldest_ts).total_seconds() / 3600
            
            print(f"Latest write_timestamp (UTC): {latest_ts}")
            print(f"Latest write_timestamp age: {latest_age_hours:.1f} hours ({latest_age_hours/24:.1f} days)")
            print(f"\nOldest write_timestamp (UTC): {oldest_ts}")
            print(f"Oldest write_timestamp age: {oldest_age_hours:.1f} hours ({oldest_age_hours/24:.1f} days)\n")
            
            # Compare with all options if we're analyzing short-term
            if not short_term_df.empty and len(options_df) != len(short_term_df) and 'write_timestamp_utc' in options_df.columns:
                all_latest_ts = options_df['write_timestamp_utc'].max()
                all_oldest_ts = options_df['write_timestamp_utc'].min()
                print(f"COMPARISON WITH ALL OPTIONS:")
                print(f"  All options latest: {all_latest_ts} (age: {(now_utc - all_latest_ts).total_seconds() / 3600:.1f}h)")
                print(f"  All options oldest: {all_oldest_ts} (age: {(now_utc - all_oldest_ts).total_seconds() / 3600:.1f}h)")
                print(f"  Short-term latest:  {latest_ts} (age: {latest_age_hours:.1f}h)")
                print(f"  Short-term oldest:  {oldest_ts} (age: {oldest_age_hours:.1f}h)\n")
            
            # Show distribution by age
            print(f"{'='*80}")
            print(f"AGE DISTRIBUTION ({analysis_label.upper()} OPTIONS)")
            print(f"{'='*80}")
            age_bins = [0, 1, 6, 12, 24, 48, 72, 168, float('inf')]  # hours
            age_labels = ['<1h', '1-6h', '6-12h', '12-24h', '1-2d', '2-3d', '3-7d', '>7d']
            analysis_df['age_bin'] = pd.cut(analysis_df['age_hours'], bins=age_bins, labels=age_labels, right=False)
            age_counts = analysis_df['age_bin'].value_counts().sort_index()
            for bin_label, count in age_counts.items():
                pct = (count / len(analysis_df)) * 100
                print(f"  {bin_label:>8}: {count:>6} records ({pct:>5.1f}%)")
            print()
            
            # Filter analysis - test the actual filter logic used in the code
            if min_ts_utc:
                print(f"{'='*80}")
                print(f"FILTER ANALYSIS ({analysis_label.upper()} OPTIONS)")
                print(f"{'='*80}")
                before_count = len(analysis_df)
                
                # Apply the same filter logic as apply_basic_filters
                from common.common import normalize_timestamp_to_utc
                filtered_df = analysis_df.copy()
                if 'write_timestamp' in filtered_df.columns:
                    filtered_df['write_timestamp'] = filtered_df['write_timestamp'].apply(normalize_timestamp_to_utc)
                    filtered_df = filtered_df[filtered_df['write_timestamp'] >= min_ts_utc].copy()
                
                after_count = len(filtered_df)
                filtered_out = before_count - after_count
                
                print(f"Records before filter: {before_count}")
                print(f"Records after filter (>= {min_ts_utc} UTC): {after_count}")
                print(f"Records filtered out: {filtered_out} ({(filtered_out/before_count*100):.1f}%)\n")
                
                if filtered_out > 0:
                    print(f"❌ All {filtered_out} records were filtered out because their write_timestamp is older than {min_write_timestamp} EST")
                    print(f"   This means no options data has been downloaded for {ticker} in the last {(now_utc - min_ts_utc).total_seconds() / 3600:.1f} hours\n")
                    
                    # Show some examples of filtered records
                    print("Sample of filtered records (oldest timestamps):")
                    oldest_filtered = options_df[options_df['write_timestamp_utc'] < min_ts_utc].nlargest(5, 'write_timestamp_utc')
                    for idx, row in oldest_filtered.iterrows():
                        print(f"  - {row.get('option_ticker', 'N/A')}: write_timestamp={row['write_timestamp_utc']} (age: {row['age_hours']:.1f}h)")
                else:
                    print(f"✅ All records pass the filter\n")
            
            # Show recent records
            print(f"{'='*80}")
            print(f"MOST RECENT RECORDS - {analysis_label.upper()} OPTIONS (Top 10)")
            print(f"{'='*80}")
            recent = analysis_df.nlargest(10, 'write_timestamp_utc')[
                ['option_ticker', 'write_timestamp_utc', 'age_hours', 'expiration_date', 'strike_price', 'option_type']
            ]
            print(recent.to_string(index=False))
            print()
            
            # If we filtered out records, show some examples
            if min_ts_utc and filtered_out > 0:
                print(f"{'='*80}")
                print(f"FILTERED OUT RECORDS (Sample of oldest)")
                print(f"{'='*80}")
                filtered_out_df = analysis_df[analysis_df['write_timestamp_utc'] < min_ts_utc]
                if not filtered_out_df.empty:
                    oldest_filtered = filtered_out_df.nlargest(10, 'write_timestamp_utc')[
                        ['option_ticker', 'write_timestamp_utc', 'age_hours', 'expiration_date', 'strike_price', 'option_type']
                    ]
                    print(oldest_filtered.to_string(index=False))
                    print()
        
    finally:
        await db.close()


async def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Check write_timestamp values for options data')
    parser.add_argument('ticker', help='Ticker symbol to check (e.g., GOOG)')
    parser.add_argument('--db-conn', 
                       default='questdb://user:password@localhost:8812/stock_data',
                       help='Database connection string')
    parser.add_argument('--min-write-timestamp',
                       help='Minimum write timestamp in EST format (e.g., "2025-11-14 17:52:38")')
    parser.add_argument('--enable-cache',
                       action='store_true',
                       help='Enable database cache (default: cache is disabled)')
    parser.add_argument('--log-level',
                       default='INFO',
                       choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
                       help='Log level for diagnostic output (default: INFO)')
    
    args = parser.parse_args()
    
    configure_logging(args.log_level)
    await check_write_timestamps(
        args.ticker,
        args.db_conn,
        args.min_write_timestamp,
        args.enable_cache,
        args.log_level,
    )


if __name__ == '__main__':
    asyncio.run(main())

