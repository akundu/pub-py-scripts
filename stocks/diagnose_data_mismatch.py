#!/usr/bin/env python3
"""
Diagnostic script to investigate data fetch/replay mismatch.

Usage:
    python diagnose_data_mismatch.py --symbol I:NDX --start-date 2026-02-05 --end-date 2026-02-08
"""

import asyncio
import pandas as pd
from datetime import datetime, timezone
import os
import sys
import argparse

# Add the project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from common.stock_db import get_stock_db
from common.fetcher import FetcherFactory
from common.symbol_utils import get_yfinance_symbol, is_index_symbol


async def diagnose_data_mismatch(symbol: str, start_date: str, end_date: str, db_path: str):
    """Diagnose data mismatch between fetch and replay."""

    print("=" * 80)
    print(f"DIAGNOSTIC REPORT: Data Mismatch for {symbol}")
    print(f"Date Range: {start_date} to {end_date}")
    print("=" * 80)

    # Step 1: Fetch fresh data from Yahoo Finance
    print("\n[STEP 1] Fetching fresh data from Yahoo Finance...")
    print("-" * 80)

    is_index = is_index_symbol(symbol)
    yf_symbol = get_yfinance_symbol(symbol) if is_index else symbol

    fetcher = FetcherFactory.create_fetcher("yahoo", symbol=yf_symbol, log_level="INFO")
    result = await fetcher.fetch_historical_data(
        symbol=yf_symbol,
        timeframe="daily",
        start_date=start_date,
        end_date=end_date
    )

    if not result.success:
        print(f"ERROR: Failed to fetch data: {result.error}")
        return

    fresh_df = result.data
    print(f"\nFetched {len(fresh_df)} rows from Yahoo Finance:")
    print(f"  Index timezone: {fresh_df.index.tz}")
    print(f"  Index dtype: {fresh_df.index.dtype}")
    print("\nFirst 3 rows:")
    print(fresh_df.head(3))

    # Step 2: Query database
    print("\n[STEP 2] Querying QuestDB database...")
    print("-" * 80)

    db_ticker = symbol.split(':')[1] if ':' in symbol else symbol

    # Initialize database
    enable_cache = not os.getenv('NO_CACHE', False)
    redis_url = os.getenv('REDIS_URL', 'redis://localhost:6379/0') if enable_cache else None

    db = get_stock_db('questdb', db_config=db_path, enable_cache=enable_cache, redis_url=redis_url, log_level='DEBUG')
    await db._init_db()

    # Query with exact dates
    db_df = await db.get_stock_data(
        ticker=db_ticker,
        start_date=start_date,
        end_date=end_date,
        interval="daily"
    )

    print(f"\nQueried {len(db_df)} rows from database:")
    print(f"  Index timezone: {db_df.index.tz}")
    print(f"  Index dtype: {db_df.index.dtype}")
    print("\nFirst 3 rows from DB:")
    print(db_df.head(3))

    # Step 3: Compare timestamps
    print("\n[STEP 3] Timestamp Analysis")
    print("-" * 80)

    if not fresh_df.empty and not db_df.empty:
        print("\nFresh data timestamps (first 3):")
        for idx in fresh_df.index[:3]:
            print(f"  {idx} (UTC: {idx.tz_convert(timezone.utc) if idx.tz else 'naive'})")

        print("\nDatabase timestamps (first 3):")
        for idx in db_df.index[:3]:
            utc_idx = idx.tz_localize(timezone.utc) if idx.tz is None else idx.tz_convert(timezone.utc)
            print(f"  {idx} (UTC: {utc_idx})")

    # Step 4: Compare prices
    print("\n[STEP 4] Price Comparison")
    print("-" * 80)

    if not fresh_df.empty and not db_df.empty:
        # Normalize indices for comparison
        fresh_df_utc = fresh_df.copy()
        fresh_df_utc.index = fresh_df_utc.index.tz_convert(timezone.utc)
        fresh_df_utc.index = fresh_df_utc.index.tz_localize(None)  # Make naive for comparison

        db_df_utc = db_df.copy()
        if db_df_utc.index.tz is not None:
            db_df_utc.index = db_df_utc.index.tz_convert(timezone.utc)
            db_df_utc.index = db_df_utc.index.tz_localize(None)

        # Find matching dates
        common_dates = fresh_df_utc.index.intersection(db_df_utc.index)

        if len(common_dates) > 0:
            print(f"\nFound {len(common_dates)} matching dates:")
            for date in common_dates[:5]:  # Show first 5
                fresh_close = fresh_df_utc.loc[date, 'close']
                db_close = db_df_utc.loc[date, 'close']
                diff = abs(fresh_close - db_close)
                match = "✓ MATCH" if diff < 0.01 else f"✗ MISMATCH (diff: {diff:.2f})"

                print(f"\n  Date: {date}")
                print(f"    Fresh close: {fresh_close:.6f}")
                print(f"    DB close:    {db_close:.6f}")
                print(f"    {match}")
        else:
            print("\n✗ NO MATCHING DATES FOUND!")
            print("\nFresh data dates (UTC naive):")
            for idx in fresh_df_utc.index[:5]:
                print(f"  {idx}")
            print("\nDatabase dates (UTC naive):")
            for idx in db_df_utc.index[:5]:
                print(f"  {idx}")

    # Step 5: Raw database query
    print("\n[STEP 5] Raw Database Query (bypassing cache)")
    print("-" * 80)

    # Direct SQL query to see exactly what's in the database
    query = f"""
        SELECT date, open, high, low, close, volume, write_timestamp
        FROM daily_prices
        WHERE ticker = '{db_ticker}'
        AND date >= '{start_date}'
        AND date < '{end_date}'
        ORDER BY date
    """

    print(f"\nExecuting raw SQL query:")
    print(query)

    # Get a connection and run the query
    async with db.connection.get_connection() as conn:
        rows = await conn.fetch(query)

        if rows:
            print(f"\nFound {len(rows)} rows in database:")
            for row in rows:
                print(f"\n  Date: {row['date']}")
                print(f"    Type: {type(row['date'])}, TZ: {getattr(row['date'], 'tzinfo', 'N/A')}")
                print(f"    Open: {row['open']}, High: {row['high']}, Low: {row['low']}")
                print(f"    Close: {row['close']}, Volume: {row['volume']}")
                print(f"    Write timestamp: {row['write_timestamp']}")
        else:
            print("\n✗ NO ROWS FOUND IN DATABASE!")

    await db.close()

    print("\n" + "=" * 80)
    print("DIAGNOSTIC COMPLETE")
    print("=" * 80)


async def main():
    parser = argparse.ArgumentParser(description="Diagnose data fetch/replay mismatch")
    parser.add_argument('--symbol', required=True, help='Symbol to diagnose (e.g., I:NDX)')
    parser.add_argument('--start-date', required=True, help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end-date', required=True, help='End date (YYYY-MM-DD)')
    parser.add_argument('--db-path', default=os.getenv('QUEST_DB_STRING'),
                       help='QuestDB connection string')

    args = parser.parse_args()

    if not args.db_path:
        print("ERROR: --db-path required or set QUEST_DB_STRING environment variable")
        sys.exit(1)

    await diagnose_data_mismatch(args.symbol, args.start_date, args.end_date, args.db_path)


if __name__ == '__main__':
    asyncio.run(main())
