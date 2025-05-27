#!/usr/bin/env python3
"""
Test script for moving average and exponential moving average functionality
in the stock database implementations.
"""

import pandas as pd
from common.stock_db import get_stock_db
from datetime import datetime, timedelta
import asyncio
import os


async def test_ma_ema():
    """Test the MA and EMA functionality with both SQLite and DuckDB."""

    # Create test data
    dates = pd.date_range(start="2024-01-01", periods=50, freq="D")
    prices = [100 + i + (i % 5) for i in range(50)]  # Simple price pattern

    test_df = pd.DataFrame(
        {
            "open": [p - 1 for p in prices],
            "high": [p + 2 for p in prices],
            "low": [p - 2 for p in prices],
            "close": prices,
            "volume": [1000000] * 50,
        },
        index=dates,
    )

    print("Test DataFrame created with", len(test_df), "rows")
    print("Date range:", test_df.index.min(), "to", test_df.index.max())
    print("Sample data:")
    print(test_df.head())

    # Test SQLite implementation
    print("\n=== Testing SQLite Implementation ===")
    sqlite_db = get_stock_db("sqlite", "./test_ma_ema_sqlite.db")

    # Save data with default MA and EMA periods
    await sqlite_db.save_stock_data(test_df, "TEST", "daily")

    # Retrieve data to see if MA and EMA were calculated
    result_df = await sqlite_db.get_stock_data("TEST")
    print("Retrieved data columns:", result_df.columns.tolist())
    print("Sample with MA/EMA:")
    ma_ema_cols = [col for col in result_df.columns if col.startswith(("ma_", "ema_"))]
    display_cols = ["close"] + ma_ema_cols
    print(result_df[display_cols].tail(10))

    # Test DuckDB implementation
    print("\n=== Testing DuckDB Implementation ===")
    duckdb_db = get_stock_db("duckdb", "./test_ma_ema_duckdb.duckdb")

    # Save data with custom MA and EMA periods
    custom_ma_periods = [5, 20]
    custom_ema_periods = [12, 26]
    await duckdb_db.save_stock_data(
        test_df,
        "TEST",
        "daily",
        ma_periods=custom_ma_periods,
        ema_periods=custom_ema_periods,
    )

    # Retrieve data to see if custom MA and EMA were calculated
    result_df_duck = await duckdb_db.get_stock_data("TEST")
    print("Retrieved data columns:", result_df_duck.columns.tolist())
    print("Sample with custom MA/EMA:")
    ma_ema_cols_duck = [
        col for col in result_df_duck.columns if col.startswith(("ma_", "ema_"))
    ]
    display_cols_duck = ["close"] + ma_ema_cols_duck
    print(result_df_duck[display_cols_duck].tail(10))

    # Test incremental data addition
    print("\n=== Testing Incremental Data Addition ===")
    # Add more data to test incremental calculation
    new_dates = pd.date_range(start="2024-02-20", periods=10, freq="D")
    new_prices = [150 + i for i in range(10)]

    new_test_df = pd.DataFrame(
        {
            "open": [p - 1 for p in new_prices],
            "high": [p + 2 for p in new_prices],
            "low": [p - 2 for p in new_prices],
            "close": new_prices,
            "volume": [1000000] * 10,
        },
        index=new_dates,
    )

    await sqlite_db.save_stock_data(new_test_df, "TEST", "daily")

    # Retrieve updated data
    updated_result = await sqlite_db.get_stock_data("TEST")
    print("Updated data length:", len(updated_result))
    print("Latest data with MA/EMA:")
    print(updated_result[display_cols].tail(5))

    print("\n=== Test completed successfully! ===")


if __name__ == "__main__":
    # Clean up any existing test databases
    for db_file in ["./test_ma_ema_sqlite.db", "./test_ma_ema_duckdb.duckdb"]:
        if os.path.exists(db_file):
            os.remove(db_file)

    asyncio.run(test_ma_ema())
