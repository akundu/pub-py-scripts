import asyncio
import argparse
from datetime import datetime

import pandas as pd

from common.stock_db import get_stock_db


async def main() -> None:
    parser = argparse.ArgumentParser(
        description="Inspect merged price series (realtime + hourly + daily) for a symbol."
    )
    parser.add_argument("symbol", help="Ticker symbol, e.g. CVNA")
    parser.add_argument(
        "--db-type",
        default="questdb",
        help="Database type for get_stock_db (default: questdb)",
    )
    parser.add_argument(
        "--db-config",
        default=None,
        help="DB config/URL to pass to get_stock_db (defaults to server's usual config)",
    )
    parser.add_argument(
        "--lookback-days",
        type=int,
        default=365,
        help="Total daily lookback window (default: 365)",
    )
    parser.add_argument(
        "--hourly-days",
        type=int,
        default=7,
        help="Number of days of hourly history before realtime window (default: 7)",
    )
    parser.add_argument(
        "--realtime-hours",
        type=int,
        default=24,
        help="Number of hours of realtime quotes at the front of the series (default: 24)",
    )
    args = parser.parse_args()

    db = get_stock_db(
        args.db_type,
        db_config=args.db_config,
        enable_cache=True,
    )

    async with db:
        df = await db.get_merged_price_series(
            args.symbol,
            lookback_days=args.lookback_days,
            hourly_days=args.hourly_days,
            realtime_hours=args.realtime_hours,
        )

    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        print(f"No merged series data available for {args.symbol}")
        return

    # Basic summary
    print(f"Symbol: {args.symbol}")
    print(f"Rows:   {len(df)}")
    if isinstance(df.index, pd.DatetimeIndex):
        print(f"From:   {df.index.min()}  ->  To: {df.index.max()}")

    # Count by source
    if "source" in df.columns:
        print("\nCounts by source:")
        print(df["source"].value_counts())

    # Show a few head/tail rows
    print("\nHead (5):")
    print(df.head(5))
    print("\nTail (5):")
    print(df.tail(5))


if __name__ == "__main__":
    asyncio.run(main())



