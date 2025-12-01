import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse
import asyncio
from collections import Counter
from datetime import datetime, timedelta
import pandas as pd
from common.stock_db import StockDBClient
from common.streak_analyzer import StreakAnalyzer, compute_streaks, analyze_hourly_streaks

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
    parser.add_argument("--port", type=int, default=9100, help="Port for StockDBClient (default: 9100)")
    parser.add_argument("--interval", choices=['daily', 'hourly'], default='daily', help="Data interval to analyze (default: daily)")
    parser.add_argument("--debug", action="store_true", help="Print detailed streak date ranges and lengths.")
    parser.add_argument("--raw", action="store_true", help="Show raw price data downloaded from server.")
    parser.add_argument("--remove-outliers", type=float, nargs='?', const=10.0, metavar='PERCENT', 
                       help="Remove outliers: remove PERCENT%% from bottom and top when calculating averages (default: 10%%)")
    parser.add_argument("--print-avg", action="store_true", help="Print average values in output.")
    parser.add_argument("--no-cache", action="store_true", help="Disable caching and fetch fresh data from database.")
    
    # Filter arguments
    parser.add_argument("--filter-change", type=float, metavar='PERCENT', 
                       help="Filter data to show behavior after stock has made a +/- change of PERCENT%%")
    parser.add_argument("--filter-periods", type=int, default=1, metavar='PERIODS',
                       help="Number of periods (days/hours) for the filter change (default: 1)")
    parser.add_argument("--filter-direction", choices=['up', 'down', 'either'], default='either',
                       help="Direction of the filter change: up, down, or either (default: either)")
    parser.add_argument("--filter-mode", choices=['first', 'all', 'segments', 'trigger_periods', 'before_triggers', 'after_all_triggers'], default='after_all_triggers',
                       help="Filter mode: 'first' (after first trigger), 'all' (after any trigger), 'segments' (separate analysis for each trigger), 'trigger_periods' (only trigger periods), 'before_triggers' (periods leading to triggers), 'after_all_triggers' (aggregate all after periods)")
    parser.add_argument("--exclude-trigger-dates", action="store_true", default=True,
                       help="Exclude trigger dates from analysis (default: True)")
    
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

def filter_data_by_change(df: pd.DataFrame, change_percent: float, periods: int = 1, direction: str = 'either', mode: str = 'first', exclude_trigger_dates: bool = True) -> pd.DataFrame:
    """
    Filter data to only include periods after the stock has made a specified change.
    
    Args:
        df: DataFrame with price data
        change_percent: The percentage change to filter by (positive or negative)
        periods: Number of periods (days/hours) for the change
        direction: 'up', 'down', or 'either' for the direction of change
        mode: 'first' (after first trigger), 'all' (after any trigger), 'segments' (separate analysis for each trigger), 'after_all_triggers' (aggregate all after periods)
    
    Returns:
        Filtered DataFrame containing only data after the specified changes
    """
    if df.empty or 'close' not in df.columns:
        return df
    
    df = df.copy()
    df = df.sort_index()
    
    # Calculate percentage change over the specified number of periods
    df['pct_change'] = df['close'].pct_change(periods=periods) * 100
    
    # Create mask based on direction and change percentage
    if direction == 'up':
        mask = df['pct_change'] >= change_percent
    elif direction == 'down':
        mask = df['pct_change'] <= -change_percent
    else:  # 'either'
        mask = (df['pct_change'] >= change_percent) | (df['pct_change'] <= -change_percent)
    
    # Find indices where the condition is met
    trigger_indices = df[mask].index
    
    if len(trigger_indices) == 0:
        print(f"No periods found where stock made {direction} change of {change_percent}% over {periods} period(s)")
        return pd.DataFrame()  # Return empty DataFrame
    
    print(f"Found {len(trigger_indices)} trigger events where stock made {direction} change of {change_percent}% over {periods} period(s)")
    print(f"Trigger dates: {[idx.strftime('%Y-%m-%d') for idx in trigger_indices[:10]]}{'...' if len(trigger_indices) > 10 else ''}")
    
    if mode == 'first':
        # Show data after the first trigger only
        after_trigger_mask = df.index >= trigger_indices[0]
        filtered_df = df[after_trigger_mask].copy()
        print(f"Filtered data: {len(df)} -> {len(filtered_df)} periods (after first trigger)")
        
    elif mode == 'all':
        # Show data after any trigger (this is the same as 'first' for now, but could be enhanced)
        after_trigger_mask = df.index >= trigger_indices[0]
        filtered_df = df[after_trigger_mask].copy()
        print(f"Filtered data: {len(df)} -> {len(filtered_df)} periods (after any trigger)")
        
    elif mode == 'after_all_triggers':
        # Aggregate all periods that come after any trigger (excluding trigger periods themselves)
        after_mask = pd.Series(False, index=df.index)
        
        print(f"\nDebug: Trigger analysis:")
        for i, trigger_idx in enumerate(trigger_indices):
            # For each trigger, include ALL data after that trigger (not just until the next trigger)
            if exclude_trigger_dates:
                after_start = df.index.get_loc(trigger_idx) + 1  # Exclude trigger date
                print(f"  Trigger {i+1}: {trigger_idx.strftime('%Y-%m-%d')} -> After: {df.index[after_start].strftime('%Y-%m-%d')} to end (excluded trigger date)")
            else:
                after_start = df.index.get_loc(trigger_idx)  # Include trigger date
                print(f"  Trigger {i+1}: {trigger_idx.strftime('%Y-%m-%d')} -> After: {df.index[after_start].strftime('%Y-%m-%d')} to end (included trigger date)")
            after_mask.iloc[after_start:] = True
        
        filtered_df = df[after_mask].copy()
        print(f"Filtered data: {len(df)} -> {len(filtered_df)} periods (aggregated after all triggers)")
        print(f"Showing behavior after {len(trigger_indices)} trigger events (excluding trigger periods themselves)")
        print(f"Note: This includes overlapping periods after each trigger")
        
        # Show what percentage of original data is included
        original_days = len(df)
        filtered_days = len(filtered_df)
        percentage = (filtered_days / original_days) * 100
        print(f"Data coverage: {filtered_days}/{original_days} days ({percentage:.1f}%)")
        
    elif mode == 'segments':
        # For segments mode, we'll return the original data but mark the triggers
        # The analysis will be done separately for each segment
        filtered_df = df.copy()
        print(f"Using segments mode: will analyze each {periods}-period segment after triggers")
        # Add trigger information to the DataFrame for later use
        filtered_df['is_trigger'] = mask
        filtered_df['trigger_number'] = 0
        for i, trigger_idx in enumerate(trigger_indices):
            filtered_df.loc[trigger_idx, 'trigger_number'] = i + 1
    
    elif mode == 'trigger_periods':
        # Show only the periods where triggers occur
        filtered_df = df[mask].copy()
        print(f"Filtered data: {len(df)} -> {len(filtered_df)} periods (only trigger periods)")
        print(f"Showing only the {len(trigger_indices)} periods where stock made {direction} change of {change_percent}% over {periods} period(s)")
    
    elif mode == 'before_triggers':
        # Show periods leading up to triggers (the periods that caused the triggers)
        # For each trigger, include the periods that led to it
        before_mask = pd.Series(False, index=df.index)
        for trigger_idx in trigger_indices:
            # Include the trigger period and the periods leading up to it
            start_idx = df.index.get_loc(trigger_idx) - periods + 1
            if start_idx >= 0:
                before_mask.iloc[start_idx:df.index.get_loc(trigger_idx) + 1] = True
        
        filtered_df = df[before_mask].copy()
        print(f"Filtered data: {len(df)} -> {len(filtered_df)} periods (periods leading to triggers)")
        print(f"Showing the {periods}-period segments that led to {len(trigger_indices)} trigger events")
    
    return filtered_df

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
        
        # Debug: Print DataFrame info if raw mode
        if args.raw and args.debug:
            print(f"\nDEBUG: DataFrame info after get_stock_data:")
            print(f"  Index type: {type(df.index)}, dtype: {df.index.dtype}")
            print(f"  Index name: {df.index.name}")
            print(f"  Columns: {df.columns.tolist()}")
            if len(df) > 0:
                print(f"  First index value: {df.index[0]} (type: {type(df.index[0])})")
                if 'date' in df.columns:
                    print(f"  First 'date' column value: {df['date'].iloc[0]} (type: {type(df['date'].iloc[0])})")
                if 'datetime' in df.columns:
                    print(f"  First 'datetime' column value: {df['datetime'].iloc[0]} (type: {type(df['datetime'].iloc[0])})")
        
        # Check if index is DatetimeIndex but with wrong values (all 1970-01-01)
        if pd.api.types.is_datetime64_any_dtype(df.index) and len(df) > 0:
            first_date = df.index[0]
            if isinstance(first_date, pd.Timestamp):
                if first_date.year == 1970 and first_date.month == 1 and first_date.day == 1:
                    # Dates are incorrectly parsed - need to re-parse from original column
                    if args.debug:
                        print(f"\nDEBUG: Index is DatetimeIndex but dates are wrong (1970-01-01)")
                        print(f"  Trying to find original date column to re-parse...")
                    # Try to find the original date column
                    date_col = None
                    for col_name in ['date', 'datetime', 'timestamp']:
                        if col_name in df.columns:
                            date_col = col_name
                            break
                    if date_col and date_col in df.columns:
                        # Re-parse the date column
                        date_series = df[date_col]
                        if date_series.dtype == 'object' or pd.api.types.is_string_dtype(date_series):
                            # Remove 'Z' timezone indicator if present
                            date_series_clean = date_series.str.replace('Z$', '', regex=True) if hasattr(date_series, 'str') else date_series
                            df[date_col] = pd.to_datetime(date_series_clean, format='%Y-%m-%dT%H:%M:%S.%f', errors='coerce')
                            if df[date_col].isna().all():
                                df[date_col] = pd.to_datetime(date_series_clean, format='%Y-%m-%dT%H:%M:%S', errors='coerce')
                            if df[date_col].isna().all():
                                df[date_col] = pd.to_datetime(date_series, errors='coerce')
                            df.set_index(date_col, inplace=True)
                            df = df[df.index.notna()]
        
        # Ensure the DataFrame has a DatetimeIndex
        if not pd.api.types.is_datetime64_any_dtype(df.index):
            # Try to find a date/datetime column - check all possible column names
            date_col = None
            
            # First, check if index name matches a column (most common case after reset_index)
            if df.index.name and df.index.name in df.columns:
                date_col = df.index.name
            else:
                # Check standard column names
                for col_name in ['date', 'datetime', 'timestamp']:
                    if col_name in df.columns:
                        date_col = col_name
                        break
                
                # If still not found, check all columns for datetime-like content
                if date_col is None:
                    for col_name in df.columns:
                        # Check if column contains date-like strings
                        if df[col_name].dtype == 'object':
                            sample_val = df[col_name].dropna().iloc[0] if not df[col_name].dropna().empty else None
                            if sample_val and isinstance(sample_val, str):
                                # Check if it looks like a date string
                                if 'T' in str(sample_val) or '-' in str(sample_val):
                                    # Try to parse it
                                    try:
                                        test_parse = pd.to_datetime(sample_val, errors='coerce')
                                        if pd.notna(test_parse):
                                            date_col = col_name
                                            break
                                    except:
                                        pass
            
            if date_col and date_col in df.columns:
                # Parse the date column - try multiple formats
                date_series = df[date_col]
                
                # Debug: Check what we're working with
                if args.debug:
                    print(f"\nDEBUG: Found date column '{date_col}'")
                    print(f"  Column dtype: {date_series.dtype}")
                    if len(date_series) > 0:
                        print(f"  First value: {date_series.iloc[0]} (type: {type(date_series.iloc[0])})")
                        print(f"  Sample values: {date_series.head(3).tolist()}")
                
                # Check if dates are already parsed but incorrectly (showing as 1970-01-01)
                if pd.api.types.is_datetime64_any_dtype(date_series):
                    # Check if all dates are epoch (1970-01-01)
                    first_date = date_series.iloc[0] if len(date_series) > 0 else None
                    if first_date and isinstance(first_date, pd.Timestamp):
                        if first_date.year == 1970 and first_date.month == 1 and first_date.day == 1:
                            if args.debug:
                                print(f"  WARNING: Dates appear to be incorrectly parsed (all showing 1970-01-01)")
                            # Dates are already parsed but wrong - we need to get original values
                            # This shouldn't happen, but if it does, we can't fix it here
                            # The issue is in _parse_df_from_response
                
                # If it's already a string, try to parse it
                if date_series.dtype == 'object' or pd.api.types.is_string_dtype(date_series):
                    # First, try to parse ISO format strings (from server: '2025-05-13T00:00:00.000000Z')
                    # Remove 'Z' timezone indicator if present, as pandas format doesn't handle it well
                    date_series_clean = date_series.str.replace('Z$', '', regex=True) if hasattr(date_series, 'str') else date_series
                    
                    # Try specific ISO format patterns
                    try:
                        # Try with microseconds
                        df[date_col] = pd.to_datetime(date_series_clean, format='%Y-%m-%dT%H:%M:%S.%f', errors='coerce')
                    except (ValueError, TypeError):
                        try:
                            # Try without microseconds
                            df[date_col] = pd.to_datetime(date_series_clean, format='%Y-%m-%dT%H:%M:%S', errors='coerce')
                        except (ValueError, TypeError):
                            try:
                                # Try date only format
                                df[date_col] = pd.to_datetime(date_series_clean, format='%Y-%m-%d', errors='coerce')
                            except (ValueError, TypeError):
                                # Fall back to flexible parsing (pandas can handle ISO8601 automatically)
                                df[date_col] = pd.to_datetime(date_series, errors='coerce')
                    
                    # If all values are still NaT, try flexible parsing
                    if df[date_col].isna().all():
                        df[date_col] = pd.to_datetime(date_series, errors='coerce')
                else:
                    # If it's numeric, might be a timestamp
                    if pd.api.types.is_numeric_dtype(date_series):
                        first_val = date_series.iloc[0] if len(date_series) > 0 else 0
                        if first_val > 1e10:  # Likely milliseconds
                            df[date_col] = pd.to_datetime(date_series, unit='ms', errors='coerce')
                        elif first_val > 1e9:  # Likely seconds
                            df[date_col] = pd.to_datetime(date_series, unit='s', errors='coerce')
                        else:
                            df[date_col] = pd.to_datetime(date_series, errors='coerce')
                    else:
                        df[date_col] = pd.to_datetime(date_series, errors='coerce')
                
                df.set_index(date_col, inplace=True)
                df = df[df.index.notna()]
            else:
                # Try to convert the index itself
                try:
                    if pd.api.types.is_numeric_dtype(df.index):
                        first_val = df.index[0] if len(df) > 0 else 0
                        if first_val > 1e10:  # Likely milliseconds
                            df.index = pd.to_datetime(df.index, unit='ms', errors='coerce')
                        elif first_val > 1e9:  # Likely seconds
                            df.index = pd.to_datetime(df.index, unit='s', errors='coerce')
                        else:
                            df.index = pd.to_datetime(df.index, errors='coerce')
                    else:
                        df.index = pd.to_datetime(df.index, format='ISO8601', errors='coerce')
                        if df.index.isna().any():
                            df.index = pd.to_datetime(df.index, errors='coerce')
                    df = df[df.index.notna()]
                except Exception as e:
                    if args.debug:
                        print(f"\nDEBUG: Error converting index: {e}")
                        print(f"  Index type: {type(df.index)}, dtype: {df.index.dtype}")
                        print(f"  Columns: {df.columns.tolist()}")
                    raise ValueError(f"Could not convert DataFrame index to DatetimeIndex. Index type: {type(df.index)}, columns: {df.columns.tolist()}, error: {e}")
        
        # Final check - ensure index is datetime
        if not pd.api.types.is_datetime64_any_dtype(df.index):
            if args.debug:
                print(f"\nDEBUG: Index still not datetime after conversion attempts")
                print(f"  Index type: {type(df.index)}, dtype: {df.index.dtype}")
                print(f"  First index value: {df.index[0] if len(df) > 0 else 'N/A'}")
            raise ValueError(f"Failed to convert DataFrame index to DatetimeIndex. Index type: {type(df.index)}, dtype: {df.index.dtype}")
        
        df = df.sort_index()
        
        # Apply filter if specified
        if args.filter_change is not None:
            print(f"\nApplying filter: {args.filter_direction} change of {args.filter_change}% over {args.filter_periods} period(s) (mode: {args.filter_mode})")
            df = filter_data_by_change(df, args.filter_change, args.filter_periods, args.filter_direction, args.filter_mode, args.exclude_trigger_dates)
            if df.empty:
                print("No data remaining after applying filter.")
                return
        
        if args.raw:
            print(f"\nRaw Price Data for {args.symbol} ({args.start} to {args.end}):")
            print("=" * 80)
            print(f"{'Date':<20} {'Open':<10} {'High':<10} {'Low':<10} {'Close':<10} {'Volume':<12}")
            print("-" * 80)
            for date, row in df.iterrows():
                # Convert to datetime if it's a pandas Timestamp
                if hasattr(date, 'to_pydatetime'):
                    date = date.to_pydatetime()
                elif isinstance(date, pd.Timestamp):
                    date = date.to_pydatetime()
                # Format the date string
                date_str = date.strftime('%Y-%m-%d %H:%M') if args.interval == 'hourly' else date.strftime('%Y-%m-%d')
                print(f"{date_str:<20} {row['open']:<10.2f} {row['high']:<10.2f} {row['low']:<10.2f} {row['close']:<10.2f} {row.get('volume', 0):<12.0f}")
            print("=" * 80)
            print(f"Total records: {len(df)}")
            print()
        
        # Add debugging information about data coverage
        if args.debug:
            print(f"\nData Analysis Summary:")
            print(f"Total records in DataFrame: {len(df)}")
            print(f"Date range: {df.index.min()} to {df.index.max()}")
            
            # Count actual trading days vs calendar days
            if args.interval == 'daily':
                calendar_days = (df.index.max() - df.index.min()).days + 1
                trading_days = len(df)
                print(f"Calendar days in range: {calendar_days}")
                print(f"Trading days with data: {trading_days}")
                print(f"Missing trading days: {calendar_days - trading_days}")
                
                # Check for flat days (no price change)
                df['prev_close'] = df['close'].shift(1)
                flat_days = ((df['close'] == df['prev_close']) & (df['prev_close'].notna())).sum()
                print(f"Days with no price change (flat): {flat_days}")
                
                # Check for weekends/holidays
                weekend_days = df[df.index.dayofweek >= 5].shape[0]
                print(f"Weekend days in data: {weekend_days}")
        
        if args.interval == "hourly":
            up_streaks, down_streaks = analyze_hourly_streaks(df)
            
            # Debug: Show streak coverage
            if args.debug:
                total_up_days = sum(s['length'] for s in up_streaks)
                total_down_days = sum(s['length'] for s in down_streaks)
                total_streak_days = total_up_days + total_down_days
                print(f"\nStreak Coverage Analysis:")
                print(f"Total days in DataFrame: {len(df)}")
                print(f"Days covered by up streaks: {total_up_days}")
                print(f"Days covered by down streaks: {total_down_days}")
                print(f"Total days covered by streaks: {total_streak_days}")
                print(f"Days NOT covered by streaks: {len(df) - total_streak_days}")
            
            if args.debug:
                print_streaks(up_streaks, "up", print_avg=args.print_avg)
            print_histogram(up_streaks, "up", total_days=sum(s['length'] for s in up_streaks), debug=args.debug, remove_outliers=args.remove_outliers is not None, outlier_percent=args.remove_outliers or 10.0, print_avg=args.print_avg)
            if args.debug:
                print_streaks(down_streaks, "down", print_avg=args.print_avg)
            print_histogram(down_streaks, "down", total_days=sum(s['length'] for s in down_streaks), debug=args.debug, remove_outliers=args.remove_outliers is not None, outlier_percent=args.remove_outliers or 10.0, print_avg=args.print_avg)
            print_summary(up_streaks, down_streaks)
        else:
            up_streaks, down_streaks = compute_streaks(df)
            
            # Debug: Show streak coverage
            if args.debug:
                total_up_days = sum(s['length'] for s in up_streaks)
                total_down_days = sum(s['length'] for s in down_streaks)
                total_streak_days = total_up_days + total_down_days
                print(f"\nStreak Coverage Analysis:")
                print(f"Total days in DataFrame: {len(df)}")
                print(f"Days covered by up streaks: {total_up_days}")
                print(f"Days covered by down streaks: {total_down_days}")
                print(f"Total days covered by streaks: {total_streak_days}")
                print(f"Days NOT covered by streaks: {len(df) - total_streak_days}")
            
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
