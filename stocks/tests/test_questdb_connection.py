#!/usr/bin/env python3
"""Test QuestDB connection and check for today's realtime data."""

import sys
import os
import asyncpg
import asyncio
from datetime import datetime

async def check_realtime_data():
    quest_str = os.getenv('QUEST_DB_STRING', 'questdb://stock_user:stock_password@lin1.kundu.dev:8812/stock_data')
    db_config = quest_str.replace('questdb://', 'postgresql://')

    print(f'Connecting to: {db_config.split("@")[1]}')

    try:
        conn = await asyncpg.connect(db_config)
        print('✅ Connected successfully!\n')

        # Check table schema first
        print('=== realtime_data table schema ===')
        schema = await conn.fetch('''
            SELECT column_name, data_type
            FROM information_schema.columns
            WHERE table_name = 'realtime_data'
            ORDER BY ordinal_position
        ''')

        for row in schema:
            print(f'  {row["column_name"]}: {row["data_type"]}')

        # Check what tickers exist
        print('\n=== Available tickers in realtime_data ===')
        tickers = await conn.fetch('''
            SELECT ticker, COUNT(*) as count
            FROM realtime_data
            GROUP BY ticker
            ORDER BY count DESC
            LIMIT 10
        ''')

        if tickers:
            for row in tickers:
                print(f'  {row["ticker"]}: {row["count"]:,} rows')
        else:
            print('  ❌ No data in table at all!')

        # Count total rows
        print('\n=== Total rows in realtime_data ===')
        total_count = await conn.fetchval("SELECT COUNT(*) FROM realtime_data")
        print(f'  Total: {total_count:,} rows')

        # Check latest NDX data (without I: prefix)
        print(f'\n=== Latest NDX data (last 10 ticks) ===')
        ndx_latest = await conn.fetch('''
            SELECT timestamp, ticker, price, size, ask_price
            FROM realtime_data
            WHERE ticker = 'NDX'
            ORDER BY timestamp DESC
            LIMIT 10
        ''')

        if ndx_latest:
            for row in ndx_latest:
                ts = row['timestamp']
                price = row['price']
                ask = row['ask_price'] if row['ask_price'] else 0
                print(f'  {ts}: Price=${price:.2f}, Ask=${ask:.2f}')

            # Show date range
            print(f'\n=== NDX data date range ===')
            date_range = await conn.fetchrow('''
                SELECT
                    MIN(timestamp) as first_ts,
                    MAX(timestamp) as last_ts
                FROM realtime_data
                WHERE ticker = 'NDX'
            ''')

            print(f'  First: {date_range["first_ts"]}')
            print(f'  Last:  {date_range["last_ts"]}')
        else:
            print('  ❌ No NDX data found')

        await conn.close()

    except Exception as e:
        print(f'❌ Error: {e}')
        import traceback
        traceback.print_exc()
        return 1

    return 0

if __name__ == "__main__":
    sys.exit(asyncio.run(check_realtime_data()))
