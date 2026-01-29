#!/usr/bin/env python3
"""
Download equity (stock/index) bar data from Polygon.io API.

This script fetches 5-minute (or hourly) aggregate bars for given tickers
over a specified date range and saves to CSV files or a directory.
Follows the same CLI and output model as options_chain_download.py.
"""

import os
import sys
import argparse
from datetime import datetime, timedelta
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
from polygon.rest import RESTClient


def get_equity_aggregates(
    client: RESTClient,
    ticker: str,
    start_date: str,
    end_date: str,
    interval: str = "5min",
) -> pd.DataFrame:
    """
    Fetch equity aggregate bars for a single ticker over a date range.

    Args:
        client: Polygon RESTClient instance
        ticker: Ticker symbol (e.g. SPY, AAPL, I:SPX)
        start_date: Start date YYYY-MM-DD
        end_date: End date YYYY-MM-DD
        interval: "5min" or "hour"

    Returns:
        DataFrame with columns: timestamp, ticker, open, high, low, close, volume, vwap, transactions
    """
    if interval == "5min":
        multiplier = 5
        timespan = "minute"
    elif interval == "hour":
        multiplier = 1
        timespan = "hour"
    else:
        raise ValueError(f"Unsupported interval '{interval}'. Use '5min' or 'hour'.")

    rows = []
    for agg in client.list_aggs(
        ticker=ticker.upper(),
        multiplier=multiplier,
        timespan=timespan,
        from_=start_date,
        to=end_date,
        limit=50000,
    ):
        rows.append({
            "timestamp": pd.to_datetime(agg.timestamp, unit="ms", utc=True),
            "ticker": ticker.upper(),
            "open": agg.open,
            "high": agg.high,
            "low": agg.low,
            "close": agg.close,
            "volume": agg.volume if agg.volume is not None else 0,
            "vwap": getattr(agg, "vwap", None),
            "transactions": getattr(agg, "transactions", None),
        })

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    df = df.sort_values("timestamp").reset_index(drop=True)
    return df


def _process_single_ticker(args_tuple):
    """
    Worker for parallel ticker processing: fetch equity data for one ticker.

    Args:
        args_tuple: (api_key, ticker, start_date, end_date, interval)

    Returns:
        (ticker, df, success)
    """
    api_key, ticker, start_date, end_date, interval = args_tuple
    try:
        client = RESTClient(api_key)
        df = get_equity_aggregates(
            client=client,
            ticker=ticker,
            start_date=start_date,
            end_date=end_date,
            interval=interval,
        )
        return (ticker, df, True)
    except Exception as e:
        print(f"ERROR: Failed to fetch {ticker}: {e}", file=sys.stderr)
        return (ticker, pd.DataFrame(), False)


def format_equities_by_day(
    df: pd.DataFrame,
    ticker: str,
    output_dir: str,
) -> None:
    """
    Write equity data to one CSV per trading day under output_dir/TICKER/.

    Args:
        df: DataFrame with timestamp, open, high, low, close, volume, vwap, transactions
        ticker: Ticker symbol (used for subdir and filename)
        output_dir: Base directory; creates output_dir/TICKER/
    """
    if df.empty:
        return

    ticker_dir = Path(output_dir) / ticker.upper()
    if ticker_dir.exists() and not ticker_dir.is_dir():
        ticker_dir.unlink()
    ticker_dir.mkdir(parents=True, exist_ok=True)

    if not pd.api.types.is_datetime64_any_dtype(df["timestamp"]):
        df = df.copy()
        df["timestamp"] = pd.to_datetime(df["timestamp"])

    df = df.copy()
    df["trading_day"] = df["timestamp"].dt.date
    trading_days = sorted(df["trading_day"].unique())

    cols = ["timestamp", "ticker", "open", "high", "low", "close", "volume", "vwap", "transactions"]

    for day in trading_days:
        day_df = df[df["trading_day"] == day].drop(columns=["trading_day"])
        day_df = day_df[cols]
        path = ticker_dir / f"{ticker.upper()}_equities_{day.strftime('%Y-%m-%d')}.csv"
        day_df.to_csv(path, index=False)
        print(f"  Saved {len(day_df)} rows to {path}", file=sys.stderr)

    print(f"Wrote {len(trading_days)} day(s) for {ticker} under {ticker_dir}", file=sys.stderr)


def main():
    parser = argparse.ArgumentParser(
        description="Download equity bar data from Polygon.io API (5-min or hourly)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single ticker, date range, 5-min (default)
  %(prog)s SPY --start 2025-01-01 --end 2025-01-31
  %(prog)s AAPL --start 2025-01-01 --end 2025-01-31 --interval hour

  # Save to a single CSV file
  %(prog)s SPY --start 2025-01-01 --end 2025-01-31 --output spy_equities.csv

  # Save one CSV per trading day per ticker under a directory
  %(prog)s SPY AAPL --start 2025-01-01 --end 2025-01-31 --output-dir ./equities_output

  # Multiple tickers in parallel
  %(prog)s SPY QQQ IWM --start 2025-01-01 --end 2025-01-31 --output-dir ./equities_output --max-connections 10
        """,
    )

    parser.add_argument(
        "ticker",
        type=str,
        nargs="+",
        help="Ticker symbol(s) (e.g. SPY, AAPL). For indices use I:SPX, I:NDX. Multiple tickers are downloaded in parallel.",
    )
    parser.add_argument(
        "--start",
        type=str,
        required=True,
        help="Start date YYYY-MM-DD",
    )
    parser.add_argument(
        "--end",
        type=str,
        required=True,
        help="End date YYYY-MM-DD",
    )
    parser.add_argument(
        "--interval",
        type=str,
        choices=["5min", "hour"],
        default="5min",
        help="Bar interval: 5min (default) or hour",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output CSV path. For multiple tickers, filename becomes stem_TICKER.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory to save one CSV per trading day per ticker (output_dir/TICKER/TICKER_equities_YYYY-MM-DD.csv)",
    )
    parser.add_argument(
        "--max-connections",
        type=int,
        default=20,
        help="Max concurrent connections when fetching multiple tickers (default: 20)",
    )

    args = parser.parse_args()

    try:
        start_dt = datetime.strptime(args.start, "%Y-%m-%d")
        end_dt = datetime.strptime(args.end, "%Y-%m-%d")
    except ValueError:
        print("ERROR: --start and --end must be YYYY-MM-DD", file=sys.stderr)
        sys.exit(1)

    if start_dt > end_dt:
        print("ERROR: --start must be before or equal to --end", file=sys.stderr)
        sys.exit(1)

    api_key = os.getenv("POLYGON_API_KEY")
    if not api_key:
        print("ERROR: POLYGON_API_KEY environment variable not set", file=sys.stderr)
        sys.exit(1)

    tickers = list(args.ticker)
    max_workers = min(len(tickers), max(1, args.max_connections))

    if len(tickers) > 1:
        fetch_args = [
            (api_key, t, args.start, args.end, args.interval)
            for t in tickers
        ]
        results = []
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_ticker = {
                executor.submit(_process_single_ticker, a): a[1]
                for a in fetch_args
            }
            for future in as_completed(future_to_ticker):
                results.append(future.result())
    else:
        results = [
            _process_single_ticker((api_key, tickers[0], args.start, args.end, args.interval))
        ]

    all_ok = True
    for ticker, df, success in results:
        if not success:
            all_ok = False
            continue
        if df.empty:
            print(f"WARNING: No data for {ticker}", file=sys.stderr)
            all_ok = False
            continue

        print(f"\n{ticker}: {len(df)} bars from {df['timestamp'].min()} to {df['timestamp'].max()}", file=sys.stderr)

        if args.output_dir:
            format_equities_by_day(df, ticker, args.output_dir)
        elif args.output:
            out_path = Path(args.output)
            if len(tickers) > 1:
                out_path = out_path.parent / f"{out_path.stem}_{ticker}{out_path.suffix}"
            df.to_csv(out_path, index=False)
            print(f"Saved to {out_path}", file=sys.stderr)
        else:
            print(f"\n--- {ticker} ---")
            print(df.to_string())
            print("---")

    if not all_ok:
        sys.exit(1)


if __name__ == "__main__":
    main()
