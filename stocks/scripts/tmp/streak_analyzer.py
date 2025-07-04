import sqlite3
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import argparse
from datetime import datetime, timedelta
import numpy as np

# ANSI Color codes
class Colors:
    GREEN = '\033[92m'
    RED = '\033[91m'
    RESET = '\033[0m'
    BOLD = '\033[1m'
    BLUE = '\033[94m'
    YELLOW = '\033[93m'

def color_number(value, format_str="{:.2%}", use_sign=True):
    """
    Color a number based on its value (green for positive, red for negative).
    
    Args:
        value: The numeric value to format
        format_str: Format string for the number (default: "{:.2%}")
        use_sign: Whether to include + sign for positive numbers
    
    Returns:
        str: Colored formatted string
    """
    if value > 0:
        sign = "+" if use_sign else ""
        return f"{Colors.GREEN}{sign}{format_str.format(value)}{Colors.RESET}"
    elif value < 0:
        return f"{Colors.RED}{format_str.format(value)}{Colors.RESET}"
    else:
        return format_str.format(value)

def color_price(value, format_str="${:.2f}"):
    """
    Color a price value (neutral coloring).
    
    Args:
        value: The price value to format
        format_str: Format string for the price (default: "${:.2f}")
    
    Returns:
        str: Formatted string
    """
    return format_str.format(value)

def analyze_streaks(db_path, ticker, threshold, lookback_days=None, return_data=False, analyze_performance=False, output_format='image', comparison_method='hourly', debug=False, event_type='both'):
    """
    Analyzes consecutive up/down day streaks following a significant price move.
    Uses hourly data to detect threshold crossings during normal trading hours.

    Args:
        db_path (str): The FULL, ABSOLUTE path to the SQLite database file.
        ticker (str): The stock ticker symbol to analyze.
        threshold (float): The percentage move (e.g., 0.01 for 1%) to trigger analysis.
        lookback_days (int, optional): Number of days to look back from the most recent date. 
                                     If None, uses all available data.
        return_data (bool): If True, returns the data for further analysis. If False, only prints and plots.
        analyze_performance (bool): If True, analyzes market performance after streaks end.
        output_format (str): Output format for histograms - 'image' or 'text'.
        comparison_method (str): 'close' for close-to-close comparison, 'hourly' for hourly data comparison.
        debug (bool): If True, prints debug information about streak calculations.
        event_type (str): 'both' for both spikes and dips, 'spikes' for spikes only, 'dips' for dips only.

    Returns:
        If return_data is True: A tuple containing (spike_counts_df, dip_counts_df, figure)
        If return_data is False: None (just prints results and shows plots)
    """
    if not os.path.exists(db_path):
        print(f"--- ERROR ---")
        print(f"Database file not found at the path: {db_path}")
        print("Please provide the correct full path using the --db-path argument.")
        if return_data:
            return None, None, None
        return

    try:
        # 1. Connect to DB and fetch daily data
        con = sqlite3.connect(db_path)
        daily_query = f"SELECT date, close FROM daily_prices WHERE ticker = '{ticker}' ORDER BY date"
        df = pd.read_sql_query(daily_query, con)
        
        if df.empty:
            print(f"No daily data found for {ticker} in the database.")
            con.close()
            if return_data:
                return None, None, None
            return

        # 2. Prepare the daily data
        df['date'] = pd.to_datetime(df['date'])
        df.sort_values('date', inplace=True)
        df.reset_index(drop=True, inplace=True)

        # 3. Apply lookback window if specified
        if lookback_days is not None:
            max_date = df['date'].max()
            cutoff_date = max_date - timedelta(days=lookback_days)
            df = df[df['date'] >= cutoff_date].copy()
            df.reset_index(drop=True, inplace=True)
            print(f"Analyzing data from {cutoff_date.strftime('%Y-%m-%d')} to {max_date.strftime('%Y-%m-%d')} ({lookback_days} days lookback)")
        else:
            print(f"Analyzing all available data from {df['date'].min().strftime('%Y-%m-%d')} to {df['date'].max().strftime('%Y-%m-%d')}")

        if df.empty:
            print(f"No data found for {ticker} within the specified lookback window.")
            con.close()
            if return_data:
                return None, None, None
            return

        # 4. Choose comparison method
        if comparison_method == 'close':
            print(f"Using close-to-close comparison method")
            con.close()
            return _analyze_streaks_daily_fallback(df, ticker, threshold, lookback_days, return_data, analyze_performance, output_format, debug, event_type)
        
        # 5. Fetch hourly data for the same time period (hourly method)
        start_date = df['date'].min().strftime('%Y-%m-%d')
        end_date = df['date'].max().strftime('%Y-%m-%d')
        
        hourly_query = f"""
        SELECT datetime, open, high, low, close, volume 
        FROM hourly_prices 
        WHERE ticker = '{ticker}' 
        AND date(datetime) BETWEEN '{start_date}' AND '{end_date}'
        ORDER BY datetime
        """
        hourly_df = pd.read_sql_query(hourly_query, con)
        con.close()
        
        if hourly_df.empty:
            print(f"No hourly data found for {ticker} in the database for the specified period.")
            print("Falling back to daily close comparison method.")
            return _analyze_streaks_daily_fallback(df, ticker, threshold, lookback_days, return_data, analyze_performance, output_format, debug, event_type)
        
        # 6. Prepare hourly data and filter for trading hours
        hourly_df['datetime'] = pd.to_datetime(hourly_df['datetime'])
        hourly_df['date'] = hourly_df['datetime'].dt.date
        hourly_df['hour'] = hourly_df['datetime'].dt.hour
        hourly_df['minute'] = hourly_df['datetime'].dt.minute
        
        # Filter for normal trading hours (9:30 AM to 4:01 PM ET)
        # Assuming the data is already in ET timezone
        trading_hours_df = hourly_df[
            ((hourly_df['hour'] == 9) & (hourly_df['minute'] >= 30)) |
            ((hourly_df['hour'] >= 10) & (hourly_df['hour'] <= 15)) |
            ((hourly_df['hour'] == 16) & (hourly_df['minute'] <= 1))
        ].copy()
        
        print(f"Using hourly data for threshold crossing detection during trading hours (9:30 AM - 4:01 PM ET)")
        
        # Print event type filter
        if event_type == 'spikes':
            print(f"Filtering for SPIKE events only")
        elif event_type == 'dips':
            print(f"Filtering for DIP events only")
        else:
            print(f"Analyzing both SPIKE and DIP events")
        
        # 7. Calculate daily percentage changes for trigger identification
        df['pct_change'] = df['close'].pct_change()
        
        # 8. Identify streaks after trigger events using hourly threshold crossings
        streaks_after_spike = []
        streaks_after_dip = []
        
        # For performance analysis, we need to track more details about each streak
        spike_streak_details = []  # List of (streak_length, start_date, end_date, start_price, end_price)
        dip_streak_details = []

        for i in range(1, len(df)):
            trigger_pct_change = df.loc[i, 'pct_change']
            
            # Process spike events
            if trigger_pct_change > threshold and event_type in ['both', 'spikes']: # Spike Trigger
                if debug:
                    trigger_date = df.loc[i, 'date']
                    trigger_price = df.loc[i, 'close']
                    # Find the hour when the threshold was crossed on the trigger day
                    trigger_hour_info = _find_trigger_crossing_hour(df, trading_hours_df, i, threshold, 'up')
                    print(f"\n🔥 SPIKE TRIGGER: {trigger_date.strftime('%Y-%m-%d')} {trigger_hour_info} - {color_number(trigger_pct_change)} change ({color_price(trigger_price)})")
                
                streak_len = _count_hourly_streak(df, trading_hours_df, i, threshold, 'up', debug)
                streaks_after_spike.append(streak_len)
                
                if debug:
                    print(f"   → Resulting streak length: {streak_len}")
                
                # Store details for performance analysis
                if analyze_performance:
                    # For performance analysis, we analyze from the trigger event date
                    trigger_date = df.loc[i, 'date']
                    trigger_price = df.loc[i, 'close']
                    
                    if trigger_date and trigger_price:
                        spike_streak_details.append((streak_len, pd.to_datetime(trigger_date), pd.to_datetime(trigger_date), trigger_price, trigger_price))

            # Process dip events
            elif trigger_pct_change < -threshold and event_type in ['both', 'dips']: # Dip Trigger
                if debug:
                    trigger_date = df.loc[i, 'date']
                    trigger_price = df.loc[i, 'close']
                    # Find the hour when the threshold was crossed on the trigger day
                    trigger_hour_info = _find_trigger_crossing_hour(df, trading_hours_df, i, threshold, 'down')
                    print(f"\n📉 DIP TRIGGER: {trigger_date.strftime('%Y-%m-%d')} {trigger_hour_info} - {color_number(trigger_pct_change)} change ({color_price(trigger_price)})")
                
                streak_len = _count_hourly_streak(df, trading_hours_df, i, threshold, 'down', debug)
                streaks_after_dip.append(streak_len)
                
                if debug:
                    print(f"   → Resulting streak length: {streak_len}")
                
                # Store details for performance analysis
                if analyze_performance:
                    # For performance analysis, we analyze from the trigger event date
                    trigger_date = df.loc[i, 'date']
                    trigger_price = df.loc[i, 'close']
                    
                    if trigger_date and trigger_price:
                        dip_streak_details.append((streak_len, pd.to_datetime(trigger_date), pd.to_datetime(trigger_date), trigger_price, trigger_price))

        # 9. Generate the histogram data
        spike_counts = pd.Series(streaks_after_spike).value_counts().sort_index()
        dip_counts = pd.Series(streaks_after_dip).value_counts().sort_index()

        # Print results
        event_description = {
            'both': 'Using Hourly Threshold Crossings',
            'spikes': 'SPIKE Events Only (Using Hourly Threshold Crossings)',
            'dips': 'DIP Events Only (Using Hourly Threshold Crossings)'
        }
        
        print(f"--- Streak Analysis for {ticker} ({event_description[event_type]}) ---")
        print(f"Found {len(streaks_after_spike)} spike events and {len(streaks_after_dip)} dip events")
        
        if event_type in ['both', 'spikes'] and len(streaks_after_spike) > 0:
            print(f"\n### Histogram for Consecutive Days with >{threshold*100:.0f}% Threshold Crossings UP (Trading Hours) AND Close UP After a SPIKE ###")
            print("Streak Length | Number of Occurrences")
            print("-------------------------------------")
            for length, count in spike_counts.items():
                print(f"{length:<13} | {count}")

        if event_type in ['both', 'dips'] and len(streaks_after_dip) > 0:
            print(f"\n### Histogram for Consecutive Days with >{threshold*100:.0f}% Threshold Crossings DOWN (Trading Hours) AND Close DOWN After a DIP ###")
            print("Streak Length | Number of Occurrences")
            print("-------------------------------------")
            for length, count in dip_counts.items():
                print(f"{length:<13} | {count}")

        # 10. Performance Analysis (if requested)
        if analyze_performance:
            print(f"\n{'='*80}")
            print(f"📊 MARKET PERFORMANCE ANALYSIS AFTER STREAKS")
            print(f"{'='*80}")
            
            if event_type in ['both', 'spikes'] and len(spike_streak_details) > 0:
                _analyze_streak_performance(df, spike_streak_details, "UP THRESHOLD CROSSINGS (After Spike)", ticker)
            
            if event_type in ['both', 'dips'] and len(dip_streak_details) > 0:
                _analyze_streak_performance(df, dip_streak_details, "DOWN THRESHOLD CROSSINGS (After Dip)", ticker)

        # 11. Generate output based on format
        fig = None  # Initialize fig variable
        
        if output_format == 'image':
            # Create visual histograms
            sns.set_style("whitegrid")
            
            # Determine subplot configuration based on event type
            if event_type == 'spikes':
                fig, ax = plt.subplots(1, 1, figsize=(8, 6))
                axes = [ax]
            elif event_type == 'dips':
                fig, ax = plt.subplots(1, 1, figsize=(8, 6))
                axes = [ax]
            else:
                fig, axes = plt.subplots(1, 2, figsize=(16, 6), sharey=True)
            
            # Create title with lookback info
            title_suffix = f" (Last {lookback_days} Days)" if lookback_days else " (All Available Data)"
            event_title = {
                'both': f'Histogram of Consecutive Days with >{threshold*100:.0f}% Threshold Crossings (Trading Hours) AND Close Direction for {ticker}{title_suffix}',
                'spikes': f'Histogram of Consecutive Days with >{threshold*100:.0f}% UP Threshold Crossings (Trading Hours) for {ticker}{title_suffix}',
                'dips': f'Histogram of Consecutive Days with >{threshold*100:.0f}% DOWN Threshold Crossings (Trading Hours) for {ticker}{title_suffix}'
            }
            fig.suptitle(event_title[event_type], fontsize=16)

            if event_type in ['both', 'spikes'] and len(spike_counts) > 0:
                ax_idx = 0 if event_type == 'spikes' else 0
                sns.barplot(x=spike_counts.index, y=spike_counts.values, ax=axes[ax_idx], palette='Greens_d')
                axes[ax_idx].set_title(f'Days with >{threshold*100:.0f}% UP Crossings (Trading Hours) AND Close UP Following a SPIKE')
                axes[ax_idx].set_xlabel('Length of Consecutive Days')
                axes[ax_idx].set_ylabel('Number of Occurrences')
                
                for p in axes[ax_idx].patches:
                    axes[ax_idx].annotate(f'{int(p.get_height())}', (p.get_x() + p.get_width() / 2., p.get_height()), ha='center', va='center', xytext=(0, 9), textcoords='offset points')

            if event_type in ['both', 'dips'] and len(dip_counts) > 0:
                ax_idx = 0 if event_type == 'dips' else 1
                sns.barplot(x=dip_counts.index, y=dip_counts.values, ax=axes[ax_idx], palette='Reds_d')
                axes[ax_idx].set_title(f'Days with >{threshold*100:.0f}% DOWN Crossings (Trading Hours) AND Close DOWN Following a DIP')
                axes[ax_idx].set_xlabel('Length of Consecutive Days')
                if event_type == 'both':
                    pass  # ylabel already set for first subplot
                else:
                    axes[ax_idx].set_ylabel('Number of Occurrences')
                
                for p in axes[ax_idx].patches:
                    axes[ax_idx].annotate(f'{int(p.get_height())}', (p.get_x() + p.get_width() / 2., p.get_height()), ha='center', va='center', xytext=(0, 9), textcoords='offset points')

            plt.tight_layout(rect=[0, 0.03, 1, 0.95])
            
            if not return_data:
                # For command-line usage, save and show plot
                plot_filename = f'{ticker}_streak_histograms_{event_type}.png'
                plt.savefig(plot_filename)
                print(f"\nVisual histogram saved to '{plot_filename}'")
                plt.show()
        
        elif output_format == 'text':
            # Create text-based ASCII histograms
            print(f"\n{'='*80}")
            print(f"📊 ASCII HISTOGRAM VISUALIZATION")
            print(f"{'='*80}")
            
            if event_type in ['both', 'spikes'] and len(spike_counts) > 0:
                _print_ascii_histogram(spike_counts, f"UP Threshold Crossings (Trading Hours) AND Close UP After >{threshold*100:.0f}% SPIKE")
            
            if event_type in ['both', 'dips'] and len(dip_counts) > 0:
                _print_ascii_histogram(dip_counts, f"DOWN Threshold Crossings (Trading Hours) AND Close DOWN After >{threshold*100:.0f}% DIP")
        
        elif output_format == 'none':
            # Skip histogram output
            print(f"\nSkipping histogram output (output format set to 'none')")
        
        # Return data if requested (for notebook usage)
        if return_data:
            # Convert to DataFrame format for notebook display
            spike_counts_df = spike_counts.reset_index() if len(spike_counts) > 0 else pd.DataFrame(columns=['Streak Length', 'Number of Occurrences'])
            spike_counts_df.columns = ['Streak Length', 'Number of Occurrences']
            
            dip_counts_df = dip_counts.reset_index() if len(dip_counts) > 0 else pd.DataFrame(columns=['Streak Length', 'Number of Occurrences'])
            dip_counts_df.columns = ['Streak Length', 'Number of Occurrences']
            
            # Only return figure if image format is used
            fig = fig if output_format == 'image' else None
            return spike_counts_df, dip_counts_df, fig
        
        return None

    except Exception as e:
        print(f"Error analyzing streaks: {str(e)}")
        if return_data:
            return None, None, None
        return None

def _count_hourly_streak(df, trading_hours_df, trigger_index, threshold, direction, debug=False):
    """
    Count consecutive days where the price crosses the threshold during trading hours
    AND the close price moves in the expected direction.
    
    Args:
        df: Daily price DataFrame
        trading_hours_df: Hourly price DataFrame filtered for trading hours
        trigger_index: Index of the trigger event in the daily DataFrame
        threshold: Percentage threshold
        direction: 'up' or 'down'
        debug: If True, prints debug information
    
    Returns:
        int: Length of the streak
    """
    streak_len = 0
    
    # Start checking from the day after the trigger
    for day_offset in range(1, len(df) - trigger_index):
        current_day_index = trigger_index + day_offset
        if current_day_index >= len(df):
            break
            
        current_date = df.loc[current_day_index, 'date']
        previous_close = df.loc[current_day_index - 1, 'close']
        current_close = df.loc[current_day_index, 'close']
        
        # Convert pandas Timestamp to date for comparison
        current_date_only = current_date.date()
        
        # Get hourly data for this specific day
        day_hourly_data = trading_hours_df[trading_hours_df['date'] == current_date_only]
        
        if day_hourly_data.empty:
            # No hourly data for this day, break the streak
            if debug:
                print(f"     ❌ {current_date.strftime('%Y-%m-%d')}: No hourly data - streak broken")
            break
        
        # Check if threshold was crossed at any point during trading hours
        threshold_crossed = False
        crossing_info = None
        crossing_hour = None
        
        for _, hour_row in day_hourly_data.iterrows():
            hour_time = hour_row['datetime']
            if direction == 'up':
                # Check if any price (open, high, low, close) crossed the up threshold
                price_change_open = (hour_row['open'] - previous_close) / previous_close
                price_change_high = (hour_row['high'] - previous_close) / previous_close
                price_change_low = (hour_row['low'] - previous_close) / previous_close
                price_change_close = (hour_row['close'] - previous_close) / previous_close
                
                if price_change_open > threshold:
                    threshold_crossed = True
                    crossing_hour = hour_time.strftime('%H:%M')
                    crossing_info = f"OPEN at {crossing_hour} - {color_number(price_change_open)} ({color_price(hour_row['open'])})"
                    break
                elif price_change_high > threshold:
                    threshold_crossed = True
                    crossing_hour = hour_time.strftime('%H:%M')
                    crossing_info = f"HIGH at {crossing_hour} - {color_number(price_change_high)} ({color_price(hour_row['high'])})"
                    break
                elif price_change_low > threshold:
                    threshold_crossed = True
                    crossing_hour = hour_time.strftime('%H:%M')
                    crossing_info = f"LOW at {crossing_hour} - {color_number(price_change_low)} ({color_price(hour_row['low'])})"
                    break
                elif price_change_close > threshold:
                    threshold_crossed = True
                    crossing_hour = hour_time.strftime('%H:%M')
                    crossing_info = f"CLOSE at {crossing_hour} - {color_number(price_change_close)} ({color_price(hour_row['close'])})"
                    break
            else:  # direction == 'down'
                # Check if any price (open, high, low, close) crossed the down threshold
                price_change_open = (hour_row['open'] - previous_close) / previous_close
                price_change_high = (hour_row['high'] - previous_close) / previous_close
                price_change_low = (hour_row['low'] - previous_close) / previous_close
                price_change_close = (hour_row['close'] - previous_close) / previous_close
                
                if price_change_open < -threshold:
                    threshold_crossed = True
                    crossing_hour = hour_time.strftime('%H:%M')
                    crossing_info = f"OPEN at {crossing_hour} - {color_number(price_change_open)} ({color_price(hour_row['open'])})"
                    break
                elif price_change_high < -threshold:
                    threshold_crossed = True
                    crossing_hour = hour_time.strftime('%H:%M')
                    crossing_info = f"HIGH at {crossing_hour} - {color_number(price_change_high)} ({color_price(hour_row['high'])})"
                    break
                elif price_change_low < -threshold:
                    threshold_crossed = True
                    crossing_hour = hour_time.strftime('%H:%M')
                    crossing_info = f"LOW at {crossing_hour} - {color_number(price_change_low)} ({color_price(hour_row['low'])})"
                    break
                elif price_change_close < -threshold:
                    threshold_crossed = True
                    crossing_hour = hour_time.strftime('%H:%M')
                    crossing_info = f"CLOSE at {crossing_hour} - {color_number(price_change_close)} ({color_price(hour_row['close'])})"
                    break
        
        # For a day to be part of the streak, both conditions must be met:
        # 1. Threshold was crossed during trading hours
        # 2. Close price moved in the expected direction
        daily_close_change = (current_close - previous_close) / previous_close
        
        if threshold_crossed:
            if direction == 'up':
                # For UP streaks, close must be higher than previous close
                if current_close > previous_close:
                    streak_len += 1
                    if debug:
                        print(f"     ✅ {current_date.strftime('%Y-%m-%d')} [{crossing_hour}]: Threshold crossed [{crossing_info}] AND close UP {color_number(daily_close_change)} ({color_price(current_close)}) - streak continues")
                else:
                    # Streak is broken - threshold crossed but close didn't move up
                    if debug:
                        print(f"     ❌ {current_date.strftime('%Y-%m-%d')} [{crossing_hour}]: Threshold crossed [{crossing_info}] BUT close DOWN {color_number(daily_close_change)} ({color_price(current_close)}) - streak broken")
                    break
            else:  # direction == 'down'
                # For DOWN streaks, close must be lower than previous close
                if current_close < previous_close:
                    streak_len += 1
                    if debug:
                        print(f"     ✅ {current_date.strftime('%Y-%m-%d')} [{crossing_hour}]: Threshold crossed [{crossing_info}] AND close DOWN {color_number(daily_close_change)} ({color_price(current_close)}) - streak continues")
                else:
                    # Streak is broken - threshold crossed but close didn't move down
                    if debug:
                        print(f"     ❌ {current_date.strftime('%Y-%m-%d')} [{crossing_hour}]: Threshold crossed [{crossing_info}] BUT close UP {color_number(daily_close_change)} ({color_price(current_close)}) - streak broken")
                    break
        else:
            # Streak is broken - threshold not crossed
            if debug:
                print(f"     ❌ {current_date.strftime('%Y-%m-%d')}: Threshold NOT crossed (close change: {color_number(daily_close_change)}, close: {color_price(current_close)}) - streak broken")
            break
    
    return streak_len

def _print_ascii_histogram(counts, title):
    """
    Print an ASCII histogram for the given counts.
    
    Args:
        counts: pandas Series with streak lengths as index and counts as values
        title: Title for the histogram
    """
    if counts.empty:
        print(f"\n{title}")
        print("No data to display")
        return
    
    print(f"\n{title}")
    print("=" * len(title))
    
    max_count = counts.max()
    max_bar_length = 50  # Maximum length of bars in characters
    
    # Scale factor to fit bars within max_bar_length
    scale_factor = max_bar_length / max_count if max_count > 0 else 1
    
    if max_count > max_bar_length:
        scale_info = f"Histogram (each █ ≈ {max_count/max_bar_length:.1f})"
    else:
        scale_info = "Histogram"
    
    print(f"{'Length':<8} | {'Count':<6} | {scale_info}")
    print("-" * 70)
    
    for length in sorted(counts.index):
        count = counts[length]
        bar_length = int(count * scale_factor)
        bar = "█" * bar_length
        print(f"{length:<8} | {count:<6} | {bar}")
    
    print(f"\nTotal events: {counts.sum()}")
    print(f"Most frequent streak length: {counts.idxmax()} (occurred {counts.max()} times)")

def _analyze_streaks_daily_fallback(df, ticker, threshold, lookback_days, return_data, analyze_performance, output_format='image', debug=False, event_type='both'):
    """
    Fallback to the original daily close comparison method when hourly data is not available.
    """
    print("Using daily close comparison method (fallback).")
    
    # Print event type filter
    if event_type == 'spikes':
        print(f"Filtering for SPIKE events only")
    elif event_type == 'dips':
        print(f"Filtering for DIP events only")
    else:
        print(f"Analyzing both SPIKE and DIP events")
    
    # This is the original logic from the previous implementation
    df['pct_change'] = df['close'].pct_change()
    df['direction'] = 'Neutral'
    df.loc[df['pct_change'] > 0, 'direction'] = 'Up'
    df.loc[df['pct_change'] < 0, 'direction'] = 'Down'

    # Identify streaks after trigger events
    streaks_after_spike = []
    streaks_after_dip = []
    
    # For performance analysis, we need to track more details about each streak
    spike_streak_details = []  # List of (streak_length, start_date, end_date, start_price, end_price)
    dip_streak_details = []

    for i in range(1, len(df)):
        trigger_pct_change = df.loc[i, 'pct_change']
        
        if trigger_pct_change > threshold and event_type in ['both', 'spikes']: # Spike Trigger
            if debug:
                trigger_date = df.loc[i, 'date'].strftime('%Y-%m-%d')
                trigger_price = df.loc[i, 'close']
                print(f"\n🔥 SPIKE TRIGGER: {trigger_date} - {color_number(trigger_pct_change)} change ({color_price(trigger_price)})")
            
            streak_len = 0
            streak_start_idx = i + 1
            streak_end_idx = i + 1
            
            for j in range(i + 1, len(df)):
                current_direction = df.loc[j, 'direction']
                current_date = df.loc[j, 'date'].strftime('%Y-%m-%d')
                current_pct_change = df.loc[j, 'pct_change']
                current_close = df.loc[j, 'close']
                
                if current_direction == 'Up':
                    streak_len += 1
                    streak_end_idx = j
                    if debug:
                        print(f"     ✅ {current_date}: UP day {color_number(current_pct_change)} ({color_price(current_close)}) - streak continues")
                else:
                    if debug:
                        print(f"     ❌ {current_date}: NOT UP ({current_direction}) {color_number(current_pct_change)} ({color_price(current_close)}) - streak broken")
                    break
            
            streaks_after_spike.append(streak_len)
            
            if debug:
                print(f"   → Resulting streak length: {streak_len}")
            
            # Store details for performance analysis
            if analyze_performance:
                # For performance analysis, we analyze from the trigger event date
                trigger_date = df.loc[i, 'date']
                trigger_price = df.loc[i, 'close']
                
                if trigger_date and trigger_price:
                    spike_streak_details.append((streak_len, trigger_date, trigger_date, trigger_price, trigger_price))

        elif trigger_pct_change < -threshold and event_type in ['both', 'dips']: # Dip Trigger
            if debug:
                trigger_date = df.loc[i, 'date'].strftime('%Y-%m-%d')
                trigger_price = df.loc[i, 'close']
                print(f"\n📉 DIP TRIGGER: {trigger_date} - {color_number(trigger_pct_change)} change ({color_price(trigger_price)})")
            
            streak_len = 0
            streak_start_idx = i + 1
            streak_end_idx = i + 1
            
            for j in range(i + 1, len(df)):
                current_direction = df.loc[j, 'direction']
                current_date = df.loc[j, 'date'].strftime('%Y-%m-%d')
                current_pct_change = df.loc[j, 'pct_change']
                current_close = df.loc[j, 'close']
                
                if current_direction == 'Down':
                    streak_len += 1
                    streak_end_idx = j
                    if debug:
                        print(f"     ✅ {current_date}: DOWN day {color_number(current_pct_change)} ({color_price(current_close)}) - streak continues")
                else:
                    if debug:
                        print(f"     ❌ {current_date}: NOT DOWN ({current_direction}) {color_number(current_pct_change)} ({color_price(current_close)}) - streak broken")
                    break
            
            streaks_after_dip.append(streak_len)
            
            if debug:
                print(f"   → Resulting streak length: {streak_len}")
            
            # Store details for performance analysis
            if analyze_performance:
                # For performance analysis, we analyze from the trigger event date
                trigger_date = df.loc[i, 'date']
                trigger_price = df.loc[i, 'close']
                
                if trigger_date and trigger_price:
                    dip_streak_details.append((streak_len, trigger_date, trigger_date, trigger_price, trigger_price))

    # Generate the histogram data
    spike_counts = pd.Series(streaks_after_spike).value_counts().sort_index()
    dip_counts = pd.Series(streaks_after_dip).value_counts().sort_index()

    # Print results
    event_description = {
        'both': 'Daily Close Method',
        'spikes': 'SPIKE Events Only (Daily Close Method)',
        'dips': 'DIP Events Only (Daily Close Method)'
    }
    
    print(f"--- Streak Analysis for {ticker} ({event_description[event_type]}) ---")
    print(f"Found {len(streaks_after_spike)} spike events and {len(streaks_after_dip)} dip events")
    
    if event_type in ['both', 'spikes'] and len(streaks_after_spike) > 0:
        print(f"\n### Histogram for Consecutive UP Days After a >{threshold*100:.0f}% SPIKE ###")
        print("Streak Length | Number of Occurrences")
        print("-------------------------------------")
        for length, count in spike_counts.items():
            print(f"{length:<13} | {count}")

    if event_type in ['both', 'dips'] and len(streaks_after_dip) > 0:
        print(f"\n### Histogram for Consecutive DOWN Days After a >{threshold*100:.0f}% DIP ###")
        print("Streak Length | Number of Occurrences")
        print("-------------------------------------")
        for length, count in dip_counts.items():
            print(f"{length:<13} | {count}")

    # Performance Analysis (if requested)
    if analyze_performance:
        print(f"\n{'='*80}")
        print(f"📊 MARKET PERFORMANCE ANALYSIS AFTER STREAKS")
        print(f"{'='*80}")
        
        if event_type in ['both', 'spikes'] and len(spike_streak_details) > 0:
            _analyze_streak_performance(df, spike_streak_details, "UP STREAKS (After Spike)", ticker)
        
        if event_type in ['both', 'dips'] and len(dip_streak_details) > 0:
            _analyze_streak_performance(df, dip_streak_details, "DOWN STREAKS (After Dip)", ticker)

    # Generate output based on format
    fig = None  # Initialize fig variable
    
    if output_format == 'image':
        # Create visual histograms
        sns.set_style("whitegrid")
        
        # Determine subplot configuration based on event type
        if event_type == 'spikes':
            fig, ax = plt.subplots(1, 1, figsize=(8, 6))
            axes = [ax]
        elif event_type == 'dips':
            fig, ax = plt.subplots(1, 1, figsize=(8, 6))
            axes = [ax]
        else:
            fig, axes = plt.subplots(1, 2, figsize=(16, 6), sharey=True)
        
        # Create title with lookback info
        title_suffix = f" (Last {lookback_days} Days)" if lookback_days else " (All Available Data)"
        event_title = {
            'both': f'Histogram of Consecutive Trading Day Streaks for {ticker} After a >{threshold*100:.0f}% Move{title_suffix}',
            'spikes': f'Histogram of Consecutive UP Day Streaks for {ticker} After a >{threshold*100:.0f}% SPIKE{title_suffix}',
            'dips': f'Histogram of Consecutive DOWN Day Streaks for {ticker} After a >{threshold*100:.0f}% DIP{title_suffix}'
        }
        fig.suptitle(event_title[event_type], fontsize=16)

        if event_type in ['both', 'spikes'] and len(spike_counts) > 0:
            ax_idx = 0 if event_type == 'spikes' else 0
            sns.barplot(x=spike_counts.index, y=spike_counts.values, ax=axes[ax_idx], palette='Greens_d')
            axes[ax_idx].set_title(f'Streaks Following a >{threshold*100:.0f}% UP Day')
            axes[ax_idx].set_xlabel('Length of Consecutive UP Streak')
            axes[ax_idx].set_ylabel('Number of Occurrences')
            
            for p in axes[ax_idx].patches:
                axes[ax_idx].annotate(f'{int(p.get_height())}', (p.get_x() + p.get_width() / 2., p.get_height()), ha='center', va='center', xytext=(0, 9), textcoords='offset points')

        if event_type in ['both', 'dips'] and len(dip_counts) > 0:
            ax_idx = 0 if event_type == 'dips' else 1
            sns.barplot(x=dip_counts.index, y=dip_counts.values, ax=axes[ax_idx], palette='Reds_d')
            axes[ax_idx].set_title(f'Streaks Following a >{threshold*100:.0f}% DOWN Day')
            axes[ax_idx].set_xlabel('Length of Consecutive DOWN Streak')
            if event_type == 'both':
                pass  # ylabel already set for first subplot
            else:
                axes[ax_idx].set_ylabel('Number of Occurrences')
            
            for p in axes[ax_idx].patches:
                axes[ax_idx].annotate(f'{int(p.get_height())}', (p.get_x() + p.get_width() / 2., p.get_height()), ha='center', va='center', xytext=(0, 9), textcoords='offset points')

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        
        if not return_data:
            # For command-line usage, save and show plot
            plot_filename = f'{ticker}_streak_histograms_{event_type}.png'
            plt.savefig(plot_filename)
            print(f"\nVisual histogram saved to '{plot_filename}'")
            plt.show()
    
    elif output_format == 'text':
        # Create text-based ASCII histograms
        print(f"\n{'='*80}")
        print(f"📊 ASCII HISTOGRAM VISUALIZATION")
        print(f"{'='*80}")
        
        if event_type in ['both', 'spikes'] and len(spike_counts) > 0:
            _print_ascii_histogram(spike_counts, f"UP Streaks After >{threshold*100:.0f}% SPIKE")
        
        if event_type in ['both', 'dips'] and len(dip_counts) > 0:
            _print_ascii_histogram(dip_counts, f"DOWN Streaks After >{threshold*100:.0f}% DIP")
    
    # elif output_format == 'none':
    #     # Skip histogram output
    #     print(f"\nSkipping histogram output (output format set to 'none')")
    
    # Return data if requested (for notebook usage)
    if return_data:
        # Convert to DataFrame format for notebook display
        spike_counts_df = spike_counts.reset_index() if len(spike_counts) > 0 else pd.DataFrame(columns=['Streak Length', 'Number of Occurrences'])
        spike_counts_df.columns = ['Streak Length', 'Number of Occurrences']
        
        dip_counts_df = dip_counts.reset_index() if len(dip_counts) > 0 else pd.DataFrame(columns=['Streak Length', 'Number of Occurrences'])
        dip_counts_df.columns = ['Streak Length', 'Number of Occurrences']
        
        # Only return figure if image format is used
        fig = fig if output_format == 'image' else None
        return spike_counts_df, dip_counts_df, fig
    
    return None

def _analyze_streak_performance(df, streak_details, streak_type, ticker):
    """
    Analyzes market performance after streaks end.
    
    Args:
        df: DataFrame with price data
        streak_details: List of (streak_length, start_date, end_date, start_price, end_price)
        streak_type: String describing the type of streak
        ticker: Stock ticker symbol
    """
    if not streak_details:
        print(f"\n🔍 {streak_type}: No streak data available for performance analysis")
        return
    
    print(f"\n🔍 {streak_type} - Market Performance Analysis")
    print(f"{'='*60}")
    
    # Convert to DataFrame for easier manipulation
    df_details = pd.DataFrame(streak_details, columns=['streak_length', 'start_date', 'end_date', 'start_price', 'end_price'])
    
    # Group by streak length
    performance_by_length = {}
    
    for streak_len in sorted(df_details['streak_length'].unique()):
        streak_data = df_details[df_details['streak_length'] == streak_len]
        
        performances = {
            '1_week': [],
            '2_weeks': [],
            '1_month': [],
            '2_months': [],
            '3_months': []
        }
        
        for _, row in streak_data.iterrows():
            trigger_date = row['start_date']  # This is now the trigger event date
            trigger_price = row['start_price']  # This is now the trigger event price
            
            # Calculate performance for different time periods
            for period, days in [('1_week', 7), ('2_weeks', 14), ('1_month', 30), ('2_months', 60), ('3_months', 90)]:
                future_date = trigger_date + timedelta(days=days)
                
                # Find the closest available price after the target date
                future_prices = df[df['date'] >= future_date]
                if not future_prices.empty:
                    future_price = future_prices.iloc[0]['close']
                    performance = ((future_price - trigger_price) / trigger_price) * 100
                    performances[period].append(performance)
        
        # Calculate averages for this streak length
        avg_performances = {}
        for period in performances:
            if performances[period]:
                avg_performances[period] = np.mean(performances[period])
            else:
                avg_performances[period] = None
        
        performance_by_length[streak_len] = avg_performances
    
    # Print results
    print(f"\nAverage Market Performance After {streak_type} Trigger Events:")
    print(f"{'Streak Length':<15} | {'1 Week':<8} | {'2 Weeks':<8} | {'1 Month':<8} | {'2 Months':<9} | {'3 Months':<9}")
    print(f"{'-' * 75}")
    
    for streak_len in sorted(performance_by_length.keys()):
        perfs = performance_by_length[streak_len]
        
        def format_perf(perf):
            if perf is None:
                return "N/A"
            else:
                if perf > 0:
                    return f"{Colors.GREEN}+{perf:.2f}%{Colors.RESET}"
                elif perf < 0:
                    return f"{Colors.RED}{perf:.2f}%{Colors.RESET}"
                else:
                    return f"{perf:.2f}%"
        
        # Note: colored text makes formatting tricky, so we'll print without perfect alignment
        print(f"{streak_len:<15} | {format_perf(perfs['1_week'])} | {format_perf(perfs['2_weeks'])} | {format_perf(perfs['1_month'])} | {format_perf(perfs['2_months'])} | {format_perf(perfs['3_months'])}")

def _find_trigger_crossing_hour(df, trading_hours_df, trigger_index, threshold, direction):
    """
    Finds the hour when the threshold was crossed on the trigger day.
    
    Args:
        df: Daily price DataFrame
        trading_hours_df: Hourly price DataFrame filtered for trading hours
        trigger_index: Index of the trigger event in the daily DataFrame
        threshold: Percentage threshold
        direction: 'up' or 'down'
    
    Returns:
        str: Hour information about the threshold crossing
    """
    trigger_date = df.loc[trigger_index, 'date']
    previous_close = df.loc[trigger_index - 1, 'close']  # Previous day's close
    
    # Get hourly data for the trigger day
    trigger_day_hourly_data = trading_hours_df[trading_hours_df['date'] == trigger_date.date()]
    
    if trigger_day_hourly_data.empty:
        return "[No hourly data]"
    
    for _, hour_row in trigger_day_hourly_data.iterrows():
        hour_time = hour_row['datetime']
        if direction == 'up':
            # Check if any price (open, high, low, close) crossed the up threshold from previous day's close
            price_change_open = (hour_row['open'] - previous_close) / previous_close
            price_change_high = (hour_row['high'] - previous_close) / previous_close
            price_change_low = (hour_row['low'] - previous_close) / previous_close
            price_change_close = (hour_row['close'] - previous_close) / previous_close
            
            if price_change_open > threshold:
                return f"[OPEN at {hour_time.strftime('%H:%M')}]"
            elif price_change_high > threshold:
                return f"[HIGH at {hour_time.strftime('%H:%M')}]"
            elif price_change_low > threshold:
                return f"[LOW at {hour_time.strftime('%H:%M')}]"
            elif price_change_close > threshold:
                return f"[CLOSE at {hour_time.strftime('%H:%M')}]"
        else:  # direction == 'down'
            # Check if any price (open, high, low, close) crossed the down threshold from previous day's close
            price_change_open = (hour_row['open'] - previous_close) / previous_close
            price_change_high = (hour_row['high'] - previous_close) / previous_close
            price_change_low = (hour_row['low'] - previous_close) / previous_close
            price_change_close = (hour_row['close'] - previous_close) / previous_close
            
            if price_change_open < -threshold:
                return f"[OPEN at {hour_time.strftime('%H:%M')}]"
            elif price_change_high < -threshold:
                return f"[HIGH at {hour_time.strftime('%H:%M')}]"
            elif price_change_low < -threshold:
                return f"[LOW at {hour_time.strftime('%H:%M')}]"
            elif price_change_close < -threshold:
                return f"[CLOSE at {hour_time.strftime('%H:%M')}]"
    
    return "[Threshold not crossed during trading hours]"

# --- Run the analysis ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze consecutive up/down day streaks for a stock.")
    parser.add_argument("--db-path", required=True, help="The FULL, ABSOLUTE path to the SQLite database file.")
    parser.add_argument("--ticker", default="QQQ", help="Stock ticker symbol to analyze, e.g., SPY.")
    parser.add_argument("--threshold", type=float, default=0.01, help="The percentage move to trigger a streak analysis, e.g., 0.01 for 1 percent.")
    parser.add_argument("--lookback-days", type=int, help="Number of days to look back from the most recent date. If not specified, uses all available data.")
    parser.add_argument("--analyze-performance", action="store_true", help="Analyze market performance after streaks end for 1 week, 2 weeks, 1 month, 2 months, 3 months.")
    parser.add_argument("--output-format", type=str, default='image', choices=['image', 'text', 'none'], help="Output format for histograms - image, text, or none.")
    parser.add_argument("--comparison-method", type=str, default='hourly', choices=['close', 'hourly'], help="Comparison method: 'close' for close-to-close comparison, 'hourly' for hourly data comparison.")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode to print detailed information about streak calculations.")
    parser.add_argument("--event-type", type=str, default='both', choices=['both', 'spikes', 'dips'], help="Filter for spikes, dips, or both.")
    
    args = parser.parse_args()
    
    analyze_streaks(
        db_path=args.db_path, 
        ticker=args.ticker, 
        threshold=args.threshold, 
        lookback_days=args.lookback_days,
        analyze_performance=args.analyze_performance,
        output_format=args.output_format,
        comparison_method=args.comparison_method,
        debug=args.debug,
        event_type=args.event_type
    )
