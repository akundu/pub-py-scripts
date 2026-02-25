#!/usr/bin/env python3
"""
One-time cleanup: remove daily_prices rows with non-midnight timestamps.

After the normalization fix in questdb_db.py, new daily rows are written at midnight UTC.
But old rows may still have non-midnight timestamps (e.g. 05:00, 06:00 from yfinance UTC offsets).
These old rows sort higher than midnight and appear as "latest" in queries.

This script deletes daily_prices rows where the time component is not midnight,
so only the normalized midnight rows remain.

Usage:
    python cleanup_daily_timestamps.py              # dry-run (show what would be deleted)
    python cleanup_daily_timestamps.py --execute     # actually delete
    python cleanup_daily_timestamps.py --ticker NDX  # only for a specific ticker

Examples:
    python cleanup_daily_timestamps.py
        Show non-midnight daily rows (dry run)

    python cleanup_daily_timestamps.py --execute
        Delete all non-midnight daily rows

    python cleanup_daily_timestamps.py --ticker NDX --execute
        Delete non-midnight daily rows for NDX only
"""

import asyncio
import argparse
import os
import sys


async def main():
    parser = argparse.ArgumentParser(
        description='''
Remove daily_prices rows with non-midnight timestamps from QuestDB.

Old fetches stored daily bars at non-midnight UTC times (e.g. 05:00, 06:00)
from yfinance timezone offsets. After the normalization fix, new data is stored
at midnight UTC. This script cleans up the old non-midnight rows so they don't
shadow the corrected data.
        ''',
        epilog='''
Examples:
    %(prog)s
        Dry run: show non-midnight rows that would be deleted

    %(prog)s --execute
        Delete all non-midnight daily rows

    %(prog)s --ticker NDX --execute
        Delete non-midnight daily rows for NDX only

    %(prog)s --ticker NDX --ticker SPX
        Dry run for NDX and SPX
        ''',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument('--execute', action='store_true',
                        help='Actually delete rows. Without this flag, only shows what would be deleted (dry run).')
    parser.add_argument('--ticker', action='append', default=None,
                        help='Limit to specific ticker(s). Can be specified multiple times.')
    args = parser.parse_args()

    db_config = (
        os.getenv('QUEST_DB_STRING')
        or os.getenv('QUESTDB_CONNECTION_STRING')
        or os.getenv('QUESTDB_URL')
    )
    if not db_config:
        print("Error: Set QUEST_DB_STRING environment variable.", file=sys.stderr)
        sys.exit(1)
    if db_config.startswith('questdb://'):
        db_config = db_config.replace('questdb://', 'postgresql://', 1)

    try:
        import asyncpg
    except ImportError:
        print("Error: asyncpg not installed. pip install asyncpg", file=sys.stderr)
        sys.exit(1)

    conn = await asyncpg.connect(db_config)

    try:
        # Find non-midnight daily rows
        ticker_filter = ""
        query_params = []
        if args.ticker:
            placeholders = ", ".join(f"${i+1}" for i in range(len(args.ticker)))
            ticker_filter = f" AND ticker IN ({placeholders})"
            query_params = args.ticker

        # Count affected rows
        count_query = f"""
            SELECT ticker, count(*) as cnt,
                   min(date) as min_date, max(date) as max_date
            FROM daily_prices
            WHERE (hour(date) != 0 OR minute(date) != 0 OR second(date) != 0)
            {ticker_filter}
            GROUP BY ticker
            ORDER BY ticker
        """
        rows = await conn.fetch(count_query, *query_params)

        if not rows:
            print("No non-midnight daily_prices rows found. Database is clean.")
            await conn.close()
            return

        print(f"{'DRY RUN - ' if not args.execute else ''}Non-midnight daily_prices rows:\n")
        total = 0
        for r in rows:
            print(f"  {r['ticker']:>10s}: {r['cnt']:>5d} rows  (date range: {r['min_date']} to {r['max_date']})")
            total += r['cnt']
        print(f"\n  {'Total':>10s}: {total:>5d} rows")

        # Show sample rows
        sample_query = f"""
            SELECT ticker, date, open, high, low, close, write_timestamp
            FROM daily_prices
            WHERE (hour(date) != 0 OR minute(date) != 0 OR second(date) != 0)
            {ticker_filter}
            ORDER BY date DESC
            LIMIT 10
        """
        samples = await conn.fetch(sample_query, *query_params)
        if samples:
            print(f"\n  Sample rows (showing up to 10):")
            for s in samples:
                print(f"    {s['ticker']:>6s}  date={s['date']}  C={s['close']:.2f}  written={s['write_timestamp']}")

        if not args.execute:
            print(f"\nDry run complete. Use --execute to delete these {total} rows.")
        else:
            # Delete non-midnight rows
            delete_query = f"""
                DELETE FROM daily_prices
                WHERE (hour(date) != 0 OR minute(date) != 0 OR second(date) != 0)
                {ticker_filter}
            """
            print(f"\nDeleting {total} non-midnight rows...")
            await conn.execute(delete_query, *query_params)
            print(f"Deleted {total} rows.")

            # Also clear Redis cache for affected tickers so stale cache doesn't mask the fix
            redis_url = os.getenv('REDIS_URL', 'redis://localhost:6379/0')
            try:
                import redis
                r = redis.from_url(redis_url)
                ticker_list = args.ticker if args.ticker else [row['ticker'] for row in rows]
                cleared = 0
                for t in ticker_list:
                    pattern = f"stocks:daily_prices:{t}:*"
                    keys = r.keys(pattern)
                    if keys:
                        r.delete(*keys)
                        cleared += len(keys)
                print(f"Cleared {cleared} Redis cache keys for affected tickers.")
            except Exception as e:
                print(f"Warning: Could not clear Redis cache: {e}")
                print("You may want to manually flush relevant cache keys.")

            # Verify
            verify = await conn.fetch(count_query, *query_params)
            if not verify:
                print("Verification: All non-midnight rows successfully removed.")
            else:
                remaining = sum(r['cnt'] for r in verify)
                print(f"Warning: {remaining} non-midnight rows still remain (QuestDB WAL may need time to process).")

    finally:
        await conn.close()


if __name__ == '__main__':
    asyncio.run(main())
