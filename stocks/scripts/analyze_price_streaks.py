import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse
import asyncio
from collections import Counter
from datetime import datetime, timedelta
import pandas as pd
from common.stock_db import StockDBClient

# Color codes for terminal output
try:
    from colorama import init, Fore, Style
    init()  # Initialize colorama for cross-platform support
    GREEN = Fore.GREEN
    RED = Fore.RED
    RESET = Style.RESET_ALL
except ImportError:
    # Fallback if colorama is not available
    GREEN = ""
    RED = ""
    RESET = ""

def parse_args():
    parser = argparse.ArgumentParser(description='Analyze price streaks for a stock symbol')
    parser.add_argument("symbol", help="Stock symbol to analyze")
    parser.add_argument("--start", required=False, help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", required=False, help="End date (YYYY-MM-DD). Defaults to today if not provided.")
    parser.add_argument("--days-back", type=int, help="If provided, analyze the last N days ending at --end (or today if --end is not provided)")
    parser.add_argument("--port", type=int, default=9001, help="Port for StockDBClient (default: 9001)")
    parser.add_argument("--interval", choices=['daily', 'hourly'], default='daily', help="Data interval to analyze (default: daily)")
    parser.add_argument("--debug", action="store_true", help="Print detailed streak date ranges and lengths.")
    parser.add_argument("--raw", action="store_true", help="Show raw price data downloaded from server.")
    parser.add_argument("--remove-outliers", type=float, nargs='?', const=10.0, metavar='PERCENT', 
                       help="Remove outliers: remove PERCENT%% from bottom and top when calculating averages (default: 10%%)")
    parser.add_argument("--print-avg", action="store_true", help="Print average values in output.")
    
    args = parser.parse_args()
    
    today = datetime.today().strftime('%Y-%m-%d')
    if args.days_back is not None:
        if args.end:
            end_date = datetime.strptime(args.end, '%Y-%m-%d')
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
        up_streaks: list of dicts {start_date, end_date, length, avg_movement, start_price, end_price}
        down_streaks: list of dicts {start_date, end_date, length, avg_movement, start_price, end_price}
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
                    # Calculate total movement from start to end of streak
                    start_price = closes[streak_start]
                    end_price = closes[i-1]
                    total_movement = ((end_price - start_price) / start_price * 100) if start_price != 0 else 0
                    down_streaks.append({
                        'start_date': dates[streak_start],
                        'end_date': dates[i-1],
                        'length': streak_len,
                        'avg_movement': total_movement,
                        'start_price': start_price,
                        'end_price': end_price
                    })
                streak_type = 'up'
                streak_start = i-1
                streak_len = 1
        elif closes[i] < closes[i-1]:
            if streak_type == 'down':
                streak_len += 1
            else:
                if streak_type == 'up' and streak_len > 0:
                    # Calculate total movement from start to end of streak
                    start_price = closes[streak_start]
                    end_price = closes[i-1]
                    total_movement = ((end_price - start_price) / start_price * 100) if start_price != 0 else 0
                    up_streaks.append({
                        'start_date': dates[streak_start],
                        'end_date': dates[i-1],
                        'length': streak_len,
                        'avg_movement': total_movement,
                        'start_price': start_price,
                        'end_price': end_price
                    })
                streak_type = 'down'
                streak_start = i-1
                streak_len = 1
        else:
            # Flat day, treat as continuation only if a streak is ongoing
            if streak_type in ('up', 'down'):
                streak_len += 1
            # else: do nothing, just move past the day 
    # Add last streak
    if streak_type == 'up' and streak_len > 0:
        start_price = closes[streak_start]
        end_price = closes[len(closes)-1]
        total_movement = ((end_price - start_price) / start_price * 100) if start_price != 0 else 0
        up_streaks.append({
            'start_date': dates[streak_start],
            'end_date': dates[len(closes)-1],
            'length': streak_len,
            'avg_movement': total_movement,
            'start_price': start_price,
            'end_price': end_price
        })
    elif streak_type == 'down' and streak_len > 0:
        start_price = closes[streak_start]
        end_price = closes[len(closes)-1]
        total_movement = ((end_price - start_price) / start_price * 100) if start_price != 0 else 0
        down_streaks.append({
            'start_date': dates[streak_start],
            'end_date': dates[len(closes)-1],
            'length': streak_len,
            'avg_movement': total_movement,
            'start_price': start_price,
            'end_price': end_price
        })
    return up_streaks, down_streaks

def print_streaks(streaks, label, print_avg=True):
    if not streaks:
        print(f"No {label} streaks found.")
        return
    print(f"\n{label.capitalize()} streaks (date ranges, lengths, and total movements):")
    for s in streaks:
        if print_avg:
            print(f"  {s['start_date'].strftime('%Y-%m-%d')} to {s['end_date'].strftime('%Y-%m-%d')}: {s['length']} days, total {s['avg_movement']:+6.2f}%")
        else:
            print(f"  {s['start_date'].strftime('%Y-%m-%d')} to {s['end_date'].strftime('%Y-%m-%d')}: {s['length']} days")

def print_histogram(streaks, label, total_days=None, debug=False, remove_outliers=False, outlier_percent=10.0, print_avg=True):
    freq = Counter(s['length'] for s in streaks)
    if not freq:
        print(f"No {label} streaks to show in histogram.")
        return
    print(f"\n{label.capitalize()} streaks histogram:")
    total = sum(freq.values()) if total_days is None else total_days
    
    # Calculate average movement, with outlier removal if requested
    length_avg_movements = {}
    length_streak_details = {}  # Store full streak details for debug
    for streak in streaks:
        length = streak['length']
        if length not in length_avg_movements:
            length_avg_movements[length] = []
            length_streak_details[length] = []
        length_avg_movements[length].append(streak['avg_movement'])
        length_streak_details[length].append(streak)
    
    for length in sorted(freq):
        count = freq[length]
        percent = (count / total * 100) if total else 0
        bar = '#' * count
        
        # Calculate average movement, with outlier removal if requested
        movements = length_avg_movements[length]
        if remove_outliers and len(movements) > 4:  # Need at least 5 data points to remove outliers
            movements_sorted = sorted(movements)
            n = len(movements_sorted)
            # Remove bottom and top outlier_percent%
            bottom_cutoff = int(n * (outlier_percent / 100.0))
            top_cutoff = int(n * (1.0 - outlier_percent / 100.0))
            filtered_movements = movements_sorted[bottom_cutoff:top_cutoff]
            avg_movement = sum(filtered_movements) / len(filtered_movements) if filtered_movements else 0
            outlier_info = f" (outliers removed: {len(movements) - len(filtered_movements)} of {len(movements)} at {outlier_percent:.1f}%)"
        else:
            avg_movement = sum(movements) / len(movements) if movements else 0
            outlier_info = ""
        
        if print_avg:
            print(f"  {length} days: {count:4d} {percent:6.2f}% avg {avg_movement:+6.2f}%{outlier_info}  {bar}")
        else:
            print(f"  {length} days: {count:4d} {percent:6.2f}% {bar}")
        
        if debug:
            print(f"    DEBUG - Individual movements for {length}-day {label} streaks:")
            for i, streak in enumerate(length_streak_details[length]):
                # Get the actual prices from the DataFrame for this streak
                start_date = streak['start_date']
                end_date = streak['end_date']
                start_price = streak['start_price']
                end_price = streak['end_price']
                if print_avg:
                    print(f"      Streak {i+1}: {streak['avg_movement']:+6.2f}% ({start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}) ${start_price:.2f} → ${end_price:.2f}")
                else:
                    print(f"      Streak {i+1}: ({start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}) ${start_price:.2f} → ${end_price:.2f}")

def print_summary(up_streaks, down_streaks):
    up_days = sum(s['length'] for s in up_streaks)
    down_days = sum(s['length'] for s in down_streaks)
    total = up_days + down_days
    up_pct = (up_days / total * 100) if total else 0
    down_pct = (down_days / total * 100) if total else 0
    print("\nSummary:")
    print(f"  Up days:   {up_days} ({up_pct:.1f}%)")
    print(f"  Down days: {down_days} ({down_pct:.1f}%)")

def analyze_days_of_week(df):
    # Only consider rows with a valid close and sorted by date
    df = df.copy()
    df = df.sort_index()
    df = df[df['close'].notna()]
    if df.empty:
        return {}
    # Compute up/down for each day
    df['prev_close'] = df['close'].shift(1)
    df['up'] = df['close'] > df['prev_close']
    df['down'] = df['close'] < df['prev_close']
    df['pct_change'] = ((df['close'] - df['prev_close']) / df['prev_close'] * 100)
    df['weekday'] = df.index.day_name()
    result = {}
    for day in ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']:
        day_df = df[df['weekday'] == day]
        up = day_df['up'].sum()
        down = day_df['down'].sum()
        total = up + down
        up_pct = (up / total * 100) if total else 0
        down_pct = (down / total * 100) if total else 0
        avg_up_pct = day_df[day_df['up']]['pct_change'].mean() if up > 0 else 0
        avg_down_pct = day_df[day_df['down']]['pct_change'].mean() if down > 0 else 0
        
        # Calculate total movement from start to end of each day
        day_movements = []
        for _, group in day_df.groupby(day_df.index.date):
            if len(group) > 1:
                start_price = group.iloc[0]['close']
                end_price = group.iloc[-1]['close']
                movement = ((end_price - start_price) / start_price * 100) if start_price != 0 else 0
                day_movements.append(movement)
        
        avg_day_movement = sum(day_movements) / len(day_movements) if day_movements else 0
        
        result[day] = {
            'up': int(up), 'down': int(down), 'up_pct': up_pct, 'down_pct': down_pct, 
            'total': total, 'avg_up_pct': avg_up_pct, 'avg_down_pct': avg_down_pct,
            'avg_day_movement': avg_day_movement
        }
    return result

def analyze_weeks(df):
    # Resample to weekly (using last close of each week)
    df = df.copy()
    df = df.sort_index()
    df = df[df['close'].notna()]
    if df.empty:
        return {'up': 0, 'down': 0, 'up_pct': 0, 'down_pct': 0, 'total': 0, 'avg_up_pct': 0, 'avg_down_pct': 0, 'avg_week_movement': 0}
    weekly = df['close'].resample('W-FRI').last()
    weekly = weekly.dropna()
    if len(weekly) < 2:
        return {'up': 0, 'down': 0, 'up_pct': 0, 'down_pct': 0, 'total': 0, 'avg_up_pct': 0, 'avg_down_pct': 0, 'avg_week_movement': 0}
    prev = weekly.shift(1)
    up_mask = (weekly > prev)
    down_mask = (weekly < prev)
    up = up_mask.sum()
    down = down_mask.sum()
    total = up + down
    up_pct = (up / total * 100) if total else 0
    down_pct = (down / total * 100) if total else 0
    pct_changes = ((weekly - prev) / prev * 100)
    avg_up_pct = pct_changes[up_mask].mean() if up > 0 else 0
    avg_down_pct = pct_changes[down_mask].mean() if down > 0 else 0
    
    # Calculate average week movement (from start to end of week)
    # Convert to timezone-naive to avoid UserWarning
    df_tz_naive = df.copy()
    if df_tz_naive.index.tz is not None:
        df_tz_naive.index = df_tz_naive.index.tz_localize(None)
    
    week_movements = []
    for _, group in df_tz_naive.groupby(df_tz_naive.index.to_period('W')):
        if len(group) > 1:
            start_price = group.iloc[0]['close']
            end_price = group.iloc[-1]['close']
            movement = ((end_price - start_price) / start_price * 100) if start_price != 0 else 0
            week_movements.append(movement)
    
    avg_week_movement = sum(week_movements) / len(week_movements) if week_movements else 0
    
    return {
        'up': int(up), 'down': int(down), 'up_pct': up_pct, 'down_pct': down_pct, 
        'total': total, 'avg_up_pct': avg_up_pct, 'avg_down_pct': avg_down_pct,
        'avg_week_movement': avg_week_movement
    }

def analyze_months(df):
    # Resample to monthly (using last close of each month)
    df = df.copy()
    df = df.sort_index()
    df = df[df['close'].notna()]
    if df.empty:
        return {'up': 0, 'down': 0, 'up_pct': 0, 'down_pct': 0, 'total': 0, 'up_months': [], 'down_months': [], 'avg_up_pct': 0, 'avg_down_pct': 0, 'avg_month_movement': 0}
    # Use 'ME' for month-end to avoid FutureWarning
    monthly = df['close'].resample('ME').last()
    monthly = monthly.dropna()
    if len(monthly) < 2:
        return {'up': 0, 'down': 0, 'up_pct': 0, 'down_pct': 0, 'total': 0, 'up_months': [], 'down_months': [], 'avg_up_pct': 0, 'avg_down_pct': 0, 'avg_month_movement': 0}
    prev = monthly.shift(1)
    up_mask = (monthly > prev)
    down_mask = (monthly < prev)
    up = up_mask.sum()
    down = down_mask.sum()
    total = up + down
    up_pct = (up / total * 100) if total else 0
    down_pct = (down / total * 100) if total else 0
    # Get month names for up and down months
    up_months = [idx.strftime('%B %Y') for idx, is_up in zip(monthly.index, up_mask) if is_up]
    down_months = [idx.strftime('%B %Y') for idx, is_down in zip(monthly.index, down_mask) if is_down]
    pct_changes = ((monthly - prev) / prev * 100)
    avg_up_pct = pct_changes[up_mask].mean() if up > 0 else 0
    avg_down_pct = pct_changes[down_mask].mean() if down > 0 else 0
    
    # Calculate average month movement (from start to end of month)
    # Convert to timezone-naive to avoid UserWarning
    df_tz_naive = df.copy()
    if df_tz_naive.index.tz is not None:
        df_tz_naive.index = df_tz_naive.index.tz_localize(None)
    
    month_movements = []
    for _, group in df_tz_naive.groupby(df_tz_naive.index.to_period('M')):
        if len(group) > 1:
            start_price = group.iloc[0]['close']
            end_price = group.iloc[-1]['close']
            movement = ((end_price - start_price) / start_price * 100) if start_price != 0 else 0
            month_movements.append(movement)
    
    avg_month_movement = sum(month_movements) / len(month_movements) if month_movements else 0
    
    return {
        'up': int(up), 'down': int(down), 'up_pct': up_pct, 'down_pct': down_pct,
        'total': total, 'up_months': up_months, 'down_months': down_months,
        'avg_up_pct': avg_up_pct, 'avg_down_pct': avg_down_pct,
        'avg_month_movement': avg_month_movement
    }

def analyze_hours_of_day(df):
    # Only consider rows with a valid close and sorted by date
    df = df.copy()
    df = df.sort_index()
    df = df[df['close'].notna()]
    if df.empty:
        return {}
    # Compute up/down for each hour
    df['prev_close'] = df['close'].shift(1)
    df['up'] = df['close'] > df['prev_close']
    df['down'] = df['close'] < df['prev_close']
    df['pct_change'] = ((df['close'] - df['prev_close']) / df['prev_close'] * 100)
    df['hour'] = df.index.hour
    result = {}
    # Market hours are typically 9-16 (9 AM to 4 PM)
    for hour in range(9, 17):
        hour_df = df[df['hour'] == hour]
        up = hour_df['up'].sum()
        down = hour_df['down'].sum()
        total = up + down
        up_pct = (up / total * 100) if total else 0
        down_pct = (down / total * 100) if total else 0
        avg_up_pct = hour_df[hour_df['up']]['pct_change'].mean() if up > 0 else 0
        avg_down_pct = hour_df[hour_df['down']]['pct_change'].mean() if down > 0 else 0
        
        # Calculate total movement from start to end of each hour
        hour_movements = []
        for _, group in hour_df.groupby([hour_df.index.date, hour_df.index.hour]):
            if len(group) > 1:
                start_price = group.iloc[0]['close']
                end_price = group.iloc[-1]['close']
                movement = ((end_price - start_price) / start_price * 100) if start_price != 0 else 0
                hour_movements.append(movement)
        
        avg_hour_movement = sum(hour_movements) / len(hour_movements) if hour_movements else 0
        
        result[hour] = {
            'up': int(up), 'down': int(down), 'up_pct': up_pct, 'down_pct': down_pct, 
            'total': total, 'avg_up_pct': avg_up_pct, 'avg_down_pct': avg_down_pct,
            'avg_hour_movement': avg_hour_movement
        }
    return result

def analyze_hourly_days_of_week(df):
    # Only consider rows with a valid close and sorted by date
    df = df.copy()
    df = df.sort_index()
    df = df[df['close'].notna()]
    if df.empty:
        return {}
    # Compute up/down for each day
    df['prev_close'] = df['close'].shift(1)
    df['up'] = df['close'] > df['prev_close']
    df['down'] = df['close'] < df['prev_close']
    df['pct_change'] = ((df['close'] - df['prev_close']) / df['prev_close'] * 100)
    df['weekday'] = df.index.day_name()
    result = {}
    for day in ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']:
        day_df = df[df['weekday'] == day]
        up = day_df['up'].sum()
        down = day_df['down'].sum()
        total = up + down
        up_pct = (up / total * 100) if total else 0
        down_pct = (down / total * 100) if total else 0
        avg_up_pct = day_df[day_df['up']]['pct_change'].mean() if up > 0 else 0
        avg_down_pct = day_df[day_df['down']]['pct_change'].mean() if down > 0 else 0
        
        # Calculate total movement from start to end of each day
        day_movements = []
        for _, group in day_df.groupby(day_df.index.date):
            if len(group) > 1:
                start_price = group.iloc[0]['close']
                end_price = group.iloc[-1]['close']
                movement = ((end_price - start_price) / start_price * 100) if start_price != 0 else 0
                day_movements.append(movement)
        
        avg_day_movement = sum(day_movements) / len(day_movements) if day_movements else 0
        
        result[day] = {
            'up': int(up), 'down': int(down), 'up_pct': up_pct, 'down_pct': down_pct, 
            'total': total, 'avg_up_pct': avg_up_pct, 'avg_down_pct': avg_down_pct,
            'avg_day_movement': avg_day_movement
        }
    return result

def analyze_hourly_streaks(df):
    # Analyze hourly streaks (consecutive up/down hours)
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
                    start_price = closes[streak_start]
                    end_price = closes[i-1]
                    total_movement = ((end_price - start_price) / start_price * 100) if start_price != 0 else 0
                    down_streaks.append({
                        'start_date': dates[streak_start],
                        'end_date': dates[i-1],
                        'length': streak_len,
                        'avg_movement': total_movement,
                        'start_price': start_price,
                        'end_price': end_price
                    })
                streak_type = 'up'
                streak_start = i-1
                streak_len = 1
        elif closes[i] < closes[i-1]:
            if streak_type == 'down':
                streak_len += 1
            else:
                if streak_type == 'up' and streak_len > 0:
                    start_price = closes[streak_start]
                    end_price = closes[i-1]
                    total_movement = ((end_price - start_price) / start_price * 100) if start_price != 0 else 0
                    up_streaks.append({
                        'start_date': dates[streak_start],
                        'end_date': dates[i-1],
                        'length': streak_len,
                        'avg_movement': total_movement,
                        'start_price': start_price,
                        'end_price': end_price
                    })
                streak_type = 'down'
                streak_start = i-1
                streak_len = 1
        else:
            # Flat hour, treat as continuation only if a streak is ongoing
            if streak_type in ('up', 'down'):
                streak_len += 1
            # else: do nothing, just move past the hour
    # Add last streak
    if streak_type == 'up' and streak_len > 0:
        start_price = closes[streak_start]
        end_price = closes[len(closes)-1]
        total_movement = ((end_price - start_price) / start_price * 100) if start_price != 0 else 0
        up_streaks.append({
            'start_date': dates[streak_start],
            'end_date': dates[len(closes)-1],
            'length': streak_len,
            'avg_movement': total_movement,
            'start_price': start_price,
            'end_price': end_price
        })
    elif streak_type == 'down' and streak_len > 0:
        start_price = closes[streak_start]
        end_price = closes[len(closes)-1]
        total_movement = ((end_price - start_price) / start_price * 100) if start_price != 0 else 0
        down_streaks.append({
            'start_date': dates[streak_start],
            'end_date': dates[len(closes)-1],
            'length': streak_len,
            'avg_movement': total_movement,
            'start_price': start_price,
            'end_price': end_price
        })
    return up_streaks, down_streaks

def print_timeframe_analysis(df, interval="daily", print_avg=True):
    if interval == "hourly":
        print("\nHour-of-day up/down analysis (market hours 9-16):")
        hour_results = analyze_hours_of_day(df)
        for hour in range(9, 17):
            res = hour_results.get(hour, {'up': 0, 'down': 0, 'up_pct': 0, 'down_pct': 0, 'total': 0, 'avg_up_pct': 0, 'avg_down_pct': 0, 'avg_hour_movement': 0})
            if print_avg:
                print(f"  {hour:2d}:00    : {GREEN}↑{RESET} {res['up']:3d} ({res['up_pct']:5.1f}%) avg +{res['avg_up_pct']:5.2f}%  {RED}↓{RESET} {res['down']:3d} ({res['down_pct']:5.1f}%) avg {res['avg_down_pct']:6.2f}%  Total {res['total']:3d}  Avg hour: {res['avg_hour_movement']:+6.2f}%")
            else:
                print(f"  {hour:2d}:00    : {GREEN}↑{RESET} {res['up']:3d} ({res['up_pct']:5.1f}%)  {RED}↓{RESET} {res['down']:3d} ({res['down_pct']:5.1f}%)  Total {res['total']:3d}")
        
        print("\nHourly day-of-week up/down analysis:")
        day_results = analyze_hourly_days_of_week(df)
        for day in ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']:
            res = day_results.get(day, {'up': 0, 'down': 0, 'up_pct': 0, 'down_pct': 0, 'total': 0, 'avg_up_pct': 0, 'avg_down_pct': 0, 'avg_day_movement': 0})
            if print_avg:
                print(f"  {day:9}: {GREEN}↑{RESET} {res['up']:3d} ({res['up_pct']:5.1f}%) avg +{res['avg_up_pct']:5.2f}%  {RED}↓{RESET} {res['down']:3d} ({res['down_pct']:5.1f}%) avg {res['avg_down_pct']:6.2f}%  Total {res['total']:3d}  Avg day: {res['avg_day_movement']:+6.2f}%")
            else:
                print(f"  {day:9}: {GREEN}↑{RESET} {res['up']:3d} ({res['up_pct']:5.1f}%)  {RED}↓{RESET} {res['down']:3d} ({res['down_pct']:5.1f}%)  Total {res['total']:3d}")
    else:
        print("\nDay-of-week up/down analysis:")
        day_results = analyze_days_of_week(df)
        for day in ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']:
            res = day_results.get(day, {'up': 0, 'down': 0, 'up_pct': 0, 'down_pct': 0, 'total': 0, 'avg_up_pct': 0, 'avg_down_pct': 0, 'avg_day_movement': 0})
            if print_avg:
                print(f"  {day:9}: {GREEN}↑{RESET} {res['up']:3d} ({res['up_pct']:5.1f}%) avg +{res['avg_up_pct']:5.2f}%  {RED}↓{RESET} {res['down']:3d} ({res['down_pct']:5.1f}%) avg {res['avg_down_pct']:6.2f}%  Total {res['total']:3d}  Avg day: {res['avg_day_movement']:+6.2f}%")
            else:
                print(f"  {day:9}: {GREEN}↑{RESET} {res['up']:3d} ({res['up_pct']:5.1f}%)  {RED}↓{RESET} {res['down']:3d} ({res['down_pct']:5.1f}%)  Total {res['total']:3d}")
    
    print("\nWeekly up/down analysis:")
    week_res = analyze_weeks(df)
    if print_avg:
        print(f"  {GREEN}↑{RESET} weeks:   {week_res['up']:3d} ({week_res['up_pct']:5.1f}%) avg +{week_res['avg_up_pct']:5.2f}%  {RED}↓{RESET} weeks: {week_res['down']:3d} ({week_res['down_pct']:5.1f}%) avg {week_res['avg_down_pct']:6.2f}%  Total: {week_res['total']:3d}  Avg week: {week_res['avg_week_movement']:+6.2f}%")
    else:
        print(f"  {GREEN}↑{RESET} weeks:   {week_res['up']:3d} ({week_res['up_pct']:5.1f}%)  {RED}↓{RESET} weeks: {week_res['down']:3d} ({week_res['down_pct']:5.1f}%)  Total: {week_res['total']:3d}")
    print("\nMonthly up/down analysis:")
    month_res = analyze_months(df)
    if print_avg:
        print(f"  {GREEN}↑{RESET} months:  {month_res['up']:3d} ({month_res['up_pct']:5.1f}%) avg +{month_res['avg_up_pct']:5.2f}%  {RED}↓{RESET} months: {month_res['down']:3d} ({month_res['down_pct']:5.1f}%) avg {month_res['avg_down_pct']:6.2f}%  Total: {month_res['total']:3d}  Avg month: {month_res['avg_month_movement']:+6.2f}%")
    else:
        print(f"  {GREEN}↑{RESET} months:  {month_res['up']:3d} ({month_res['up_pct']:5.1f}%)  {RED}↓{RESET} months: {month_res['down']:3d} ({month_res['down_pct']:5.1f}%)  Total: {month_res['total']:3d}")
    if month_res['up_months']:
        print(f"    {GREEN}↑{RESET} months:   {', '.join(month_res['up_months'])}")
    if month_res['down_months']:
        print(f"    {RED}↓{RESET} months: {', '.join(month_res['down_months'])}")

async def main():
    args = parse_args()
    server_addr = f"localhost:{args.port}" # Default to localhost for now
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
        
        if args.raw:
            print(f"\nRaw Price Data for {args.symbol} ({args.start} to {args.end}):")
            print("=" * 80)
            print(f"{'Date':<20} {'Open':<10} {'High':<10} {'Low':<10} {'Close':<10} {'Volume':<12}")
            print("-" * 80)
            for date, row in df.iterrows():
                date_str = date.strftime('%Y-%m-%d %H:%M') if args.interval == 'hourly' else date.strftime('%Y-%m-%d')
                print(f"{date_str:<20} {row['open']:<10.2f} {row['high']:<10.2f} {row['low']:<10.2f} {row['close']:<10.2f} {row.get('volume', 0):<12.0f}")
            print("=" * 80)
            print(f"Total records: {len(df)}")
            print()
        
        if args.interval == "hourly":
            up_streaks, down_streaks = analyze_hourly_streaks(df)
            if args.debug:
                print_streaks(up_streaks, "up", print_avg=args.print_avg)
            print_histogram(up_streaks, "up", total_days=sum(s['length'] for s in up_streaks), debug=args.debug, remove_outliers=args.remove_outliers is not None, outlier_percent=args.remove_outliers or 10.0, print_avg=args.print_avg)
            if args.debug:
                print_streaks(down_streaks, "down", print_avg=args.print_avg)
            print_histogram(down_streaks, "down", total_days=sum(s['length'] for s in down_streaks), debug=args.debug, remove_outliers=args.remove_outliers is not None, outlier_percent=args.remove_outliers or 10.0, print_avg=args.print_avg)
            print_summary(up_streaks, down_streaks)
        else:
            up_streaks, down_streaks = compute_streaks(df)
            if args.debug:
                print_streaks(up_streaks, "up", print_avg=args.print_avg)
            print_histogram(up_streaks, "up", total_days=sum(s['length'] for s in up_streaks), debug=args.debug, remove_outliers=args.remove_outliers is not None, outlier_percent=args.remove_outliers or 10.0, print_avg=args.print_avg)
            if args.debug:
                print_streaks(down_streaks, "down", print_avg=args.print_avg)
            print_histogram(down_streaks, "down", total_days=sum(s['length'] for s in down_streaks), debug=args.debug, remove_outliers=args.remove_outliers is not None, outlier_percent=args.remove_outliers or 10.0, print_avg=args.print_avg)
            print_summary(up_streaks, down_streaks)
        
        print_timeframe_analysis(df, args.interval, print_avg=args.print_avg)
    finally:
        await client.close_session()

if __name__ == "__main__":
    asyncio.run(main()) 