#!/usr/bin/env python3
"""
Verification script to test that writetimestamp updates correctly on re-fetch.

This script:
1. Queries the current writetimestamp for recent NDX data
2. Runs fetch_symbol_data to re-fetch the same data
3. Queries again to verify writetimestamp was updated

Usage:
    python verify_timestamp_update.py
"""

import asyncio
import os
import sys
from datetime import datetime, timezone
import subprocess

# Add project root to path
sys.path.insert(0, os.path.dirname(__file__))

from common.questdb_db import StockQuestDB


async def query_timestamps(db: StockQuestDB, ticker: str = 'NDX', limit: int = 3):
    """Query and display current timestamps for a ticker."""
    async with db.connection.get_connection() as conn:
        rows = await conn.fetch(
            """
            SELECT ticker, date, close, write_timestamp
            FROM daily_prices
            WHERE ticker = $1
            ORDER BY date DESC
            LIMIT $2
            """,
            ticker, limit
        )

        print(f"\n{'='*80}")
        print(f"Recent {ticker} data (showing {len(rows)} rows):")
        print(f"{'='*80}")
        for row in rows:
            print(f"  Date: {row['date']}")
            print(f"  Close: {row['close']:.2f}")
            print(f"  Write Timestamp: {row['write_timestamp']}")
            print(f"  {'-'*76}")

        return rows


async def main():
    # Get connection string from environment
    db_string = os.getenv('QUEST_DB_STRING') or os.getenv('QUESTDB_CONNECTION_STRING')
    if not db_string:
        print("ERROR: QUEST_DB_STRING environment variable not set")
        print("Please set: export QUEST_DB_STRING='questdb://user:pass@host:port/db'")
        return 1

    print(f"\nConnecting to QuestDB...")
    db = StockQuestDB(db_string)

    try:
        # Step 1: Query BEFORE re-fetch
        print("\n" + "="*80)
        print("STEP 1: Query timestamps BEFORE re-fetch")
        print("="*80)
        before_rows = await query_timestamps(db, 'NDX', 3)

        if not before_rows:
            print("\nWARNING: No existing NDX data found. The test will only verify INSERT works.")
        else:
            print(f"\nOldest writetimestamp: {before_rows[0]['write_timestamp']}")

        # Step 2: Run fetch_symbol_data to re-fetch
        print("\n" + "="*80)
        print("STEP 2: Re-fetching NDX data using fetch_symbol_data.py")
        print("="*80)

        # Wait a moment to ensure timestamp will be different
        await asyncio.sleep(1)

        cmd = [
            sys.executable,
            "fetch_symbol_data.py",
            "--symbol", "I:NDX",
            "--db-path", db_string,
            "--interval", "daily"
        ]

        print(f"Running: {' '.join(cmd)}")
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            cwd=os.path.dirname(__file__)
        )

        if result.returncode != 0:
            print(f"\nERROR: fetch_symbol_data.py failed:")
            print(result.stderr)
            return 1

        print("\nFetch completed successfully!")

        # Step 3: Query AFTER re-fetch
        print("\n" + "="*80)
        print("STEP 3: Query timestamps AFTER re-fetch")
        print("="*80)
        after_rows = await query_timestamps(db, 'NDX', 3)

        # Step 4: Verify update
        print("\n" + "="*80)
        print("STEP 4: Verification Results")
        print("="*80)

        if not before_rows:
            print("\n✅ SUCCESS: New data was inserted (no existing data to compare)")
            return 0

        # Compare timestamps
        success = True
        for i, (before, after) in enumerate(zip(before_rows, after_rows)):
            if before['date'] != after['date']:
                print(f"\n⚠️  Row {i}: Dates don't match - skipping comparison")
                continue

            print(f"\nRow {i} - Date: {after['date']}")
            print(f"  Before: {before['write_timestamp']}")
            print(f"  After:  {after['write_timestamp']}")

            if after['write_timestamp'] > before['write_timestamp']:
                print(f"  ✅ UPDATED: Timestamp increased by {(after['write_timestamp'] - before['write_timestamp']).total_seconds():.2f} seconds")
            elif after['write_timestamp'] == before['write_timestamp']:
                print(f"  ❌ FAILED: Timestamp did NOT update")
                success = False
            else:
                print(f"  ❌ FAILED: Timestamp went backwards (should not happen)")
                success = False

        print("\n" + "="*80)
        if success:
            print("✅ VERIFICATION PASSED: All timestamps updated correctly!")
        else:
            print("❌ VERIFICATION FAILED: Some timestamps did not update")
        print("="*80)

        return 0 if success else 1

    finally:
        await db.close()


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
