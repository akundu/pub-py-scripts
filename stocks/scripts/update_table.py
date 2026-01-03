#!/usr/bin/env python3
import sys
from pathlib import Path
# Add the parent directory to Python path so 'common' module can be found
sys.path.append(str(Path(__file__).parent.parent))

import asyncio
from common.questdb_db import StockQuestDB

async def add_write_timestamp_columns():
    """Add write_timestamp columns to existing QuestDB tables."""
    db_path = "questdb://user:password@localhost:8812/stock_data"  # Your connection string
    db = StockQuestDB(db_path, pool_max_size=1, connection_timeout_seconds=30)
    
    async with db.get_connection() as conn:
        try:
            # Add to daily_prices
            await conn.execute("ALTER TABLE daily_prices ADD COLUMN write_timestamp TIMESTAMP")
            print("✓ Added write_timestamp to daily_prices")
        except Exception as e:
            print(f"Error adding to daily_prices: {e}")
            
        try:
            # Add to hourly_prices
            await conn.execute("ALTER TABLE hourly_prices ADD COLUMN write_timestamp TIMESTAMP")
            print("✓ Added write_timestamp to hourly_prices")
        except Exception as e:
            print(f"Error adding to hourly_prices: {e}")

if __name__ == "__main__":
    asyncio.run(add_write_timestamp_columns())