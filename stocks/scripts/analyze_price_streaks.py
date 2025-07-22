import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse
import asyncio
from collections import Counter, defaultdict
from datetime import datetime, timedelta
import pandas as pd
from common.stock_db import StockDBClient

def parse_args():
    parser = argparse.ArgumentParser(description="Analyze up/down streaks for a stock using StockDBClient.")
    parser.add_argument("symbol", help="Stock symbol (e.g. AAPL)")
    parser.add_argument("--start", required=False, help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", required=False, help="End date (YYYY-MM-DD). Defaults to today if not provided.")
    parser.add_argument("--days-back", type=int, help="If provided, analyze the last N days ending at --end (or today if --end is not provided)")
    parser.add_argument("--port", type=int, required=True, help="Port for StockDBClient (e.g. 8080)")
    parser.add_argument("--host", default="localhost", help="Host for StockDBClient (default: localhost)")
    parser.add_argument("--interval", default="daily", choices=["daily", "hourly"], help="Data interval (default: daily)")
    parser.add_argument("--debug", action="store_true", help="Print detailed streak date ranges and lengths.")
    args = parser.parse_args()
    today = datetime.today().strftime('%Y-%m-%d')
    if args.days_back is not None:
        # Determine the end date (use --end if provided, else today)
        if args.end:
            try:
                end_date = datetime.strptime(args.end, '%Y-%m-%d')
            except ValueError:
                parser.error("--end must be in YYYY-MM-DD format if provided.")
        else:
            end_date = datetime.today()
            args.end = end_date.strftime('%Y-%m-%d')
        start_date = end_date - timedelta(days=args.days_back)
        args.start = start_date.strftime('%Y-%m-%d')
        print(f"Using date range: {args.start} to {args.end} (last {args.days_back} days)")
    else:
        if not args.end:
            args.end = today
        if not args.start:
            parser.error("--start is required if --days-back is not provided.")
    return args

def compute_streaks(df: pd.DataFrame):
    """
    Returns:
        up_streaks: list of dicts {start_date, end_date, length}
        down_streaks: list of dicts {start_date, end_date, length}
    """
    if df.empty or 'close' not in df.columns:
        return [], []
    closes = df['close'].values
    dates = df.index.to_list()
    up_streaks = []
    down_streaks = []
    streak_type = None
    streak_start = 0
    streak_len = 0
    for i in range(1, len(closes)):
        if closes[i] > closes[i-1]:
            if streak_type == 'up':
                streak_len += 1
            else:
                if streak_type == 'down' and streak_len > 0:
                    down_streaks.append({
                        'start_date': dates[streak_start],
                        'end_date': dates[i-1],
                        'length': streak_len
                    })
                streak_type = 'up'
                streak_start = i-1
                streak_len = 1
        elif closes[i] < closes[i-1]:
            if streak_type == 'down':
                streak_len += 1
            else:
                if streak_type == 'up' and streak_len > 0:
                    up_streaks.append({
                        'start_date': dates[streak_start],
                        'end_date': dates[i-1],
                        'length': streak_len
                    })
                streak_type = 'down'
                streak_start = i-1
                streak_len = 1
        else:
            # Flat day, treat as streak break
            if streak_type == 'up' and streak_len > 0:
                up_streaks.append({
                    'start_date': dates[streak_start],
                    'end_date': dates[i-1],
                    'length': streak_len
                })
            elif streak_type == 'down' and streak_len > 0:
                down_streaks.append({
                    'start_date': dates[streak_start],
                    'end_date': dates[i-1],
                    'length': streak_len
                })
            streak_type = None
            streak_len = 0
            streak_start = i
    # Add last streak
    if streak_type == 'up' and streak_len > 0:
        up_streaks.append({
            'start_date': dates[streak_start],
            'end_date': dates[len(closes)-1],
            'length': streak_len
        })
    elif streak_type == 'down' and streak_len > 0:
        down_streaks.append({
            'start_date': dates[streak_start],
            'end_date': dates[len(closes)-1],
            'length': streak_len
        })
    return up_streaks, down_streaks

def print_streaks(streaks, label):
    if not streaks:
        print(f"No {label} streaks found.")
        return
    print(f"\n{label.capitalize()} streaks (date ranges and lengths):")
    for s in streaks:
        print(f"  {s['start_date'].strftime('%Y-%m-%d')} to {s['end_date'].strftime('%Y-%m-%d')}: {s['length']} days")

def print_histogram(streaks, label):
    freq = Counter(s['length'] for s in streaks)
    if not freq:
        print(f"No {label} streaks to show in histogram.")
        return
    print(f"\n{label.capitalize()} streaks histogram:")
    for length in sorted(freq):
        bar = '#' * freq[length]
        print(f"  {length} days: {bar} ({freq[length]})")

async def main():
    args = parse_args()
    server_addr = f"{args.host}:{args.port}"
    client = StockDBClient(server_addr)
    try:
        df = await client.get_stock_data(
            args.symbol,
            start_date=args.start,
            end_date=args.end,
            interval=args.interval
        )
        if df.empty:
            print(f"No data found for {args.symbol} between {args.start} and {args.end}.")
            return
        df = df.sort_index()
        up_streaks, down_streaks = compute_streaks(df)
        if args.debug:
            print_streaks(up_streaks, "up")
        print_histogram(up_streaks, "up")
        if args.debug:
            print_streaks(down_streaks, "down")
        print_histogram(down_streaks, "down")
    finally:
        await client.close_session()

if __name__ == "__main__":
    asyncio.run(main()) 