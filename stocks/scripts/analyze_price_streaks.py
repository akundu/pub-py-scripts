import argparse
import pandas as pd
from collections import Counter

def analyze_streaks(df: pd.DataFrame, price_column: str) -> tuple[Counter, Counter]:
    """
    Analyzes the DataFrame for consecutive price increases and decreases.
    Returns two Counters: one for up-streaks and one for down-streaks.
    Each Counter maps streak length to its frequency.
    """
    up_streak_counts = Counter()
    down_streak_counts = Counter()

    if len(df) < 2:
        return up_streak_counts, down_streak_counts

    # Calculate price differences
    # Ensure price column is numeric for diff()
    df[price_column] = pd.to_numeric(df[price_column], errors='coerce')
    df = df.dropna(subset=[price_column]) # Remove rows where price could not be converted
    
    if len(df) < 2: # Check again after potential dropna
        return up_streak_counts, down_streak_counts
        
    price_diffs = df[price_column].diff()

    current_up_streak = 0
    current_down_streak = 0

    for diff in price_diffs.iloc[1:]: # Skip the first NaN diff
        if diff > 0: # Price increased
            if current_down_streak > 0:
                down_streak_counts[current_down_streak] += 1
            current_down_streak = 0
            current_up_streak += 1
        elif diff < 0: # Price decreased
            if current_up_streak > 0:
                up_streak_counts[current_up_streak] += 1
            current_up_streak = 0
            current_down_streak += 1
        else: # Price stayed the same
            if current_up_streak > 0:
                up_streak_counts[current_up_streak] += 1
            if current_down_streak > 0:
                down_streak_counts[current_down_streak] += 1
            current_up_streak = 0
            current_down_streak = 0

    # Account for the last streak
    if current_up_streak > 0:
        up_streak_counts[current_up_streak] += 1
    if current_down_streak > 0:
        down_streak_counts[current_down_streak] += 1

    return up_streak_counts, down_streak_counts

def get_all_streaks_detailed(df: pd.DataFrame, price_column: str) -> list[dict]:
    """
    Identifies all consecutive price streaks (up or down) and returns a list of dicts,
    each containing details about a streak: type, length, start_price, end_price, start_index.
    """
    detailed_streaks = []
    if len(df) < 2:
        return detailed_streaks

    # Ensure price column is numeric and handle NaNs
    df[price_column] = pd.to_numeric(df[price_column], errors='coerce')
    df = df.dropna(subset=[price_column])
    if len(df) < 2:
        return detailed_streaks

    price_diffs = df[price_column].diff()
    
    current_streak_type = None # 'up' or 'down'
    current_streak_length = 0
    streak_start_price = 0
    streak_start_index = 0 # Original index in df

    # Iterate from the first actual difference
    for i in range(1, len(price_diffs)):
        diff = price_diffs.iloc[i]
        current_price = df[price_column].iloc[i]
        prev_price = df[price_column].iloc[i-1]

        if diff > 0: # Price increased
            if current_streak_type == 'up':
                current_streak_length += 1
            else: # New up-streak or end of down-streak
                if current_streak_type == 'down' and current_streak_length > 0:
                    detailed_streaks.append({
                        'type': 'down',
                        'length': current_streak_length,
                        'start_price': streak_start_price,
                        'end_price': prev_price, # prev_price is the end of the down streak
                        'start_df_index': streak_start_index,
                        'end_df_index': i -1
                    })
                current_streak_type = 'up'
                current_streak_length = 1
                streak_start_price = prev_price # Streak starts from the price before the first increase
                streak_start_index = i - 1
        elif diff < 0: # Price decreased
            if current_streak_type == 'down':
                current_streak_length += 1
            else: # New down-streak or end of up-streak
                if current_streak_type == 'up' and current_streak_length > 0:
                    detailed_streaks.append({
                        'type': 'up',
                        'length': current_streak_length,
                        'start_price': streak_start_price,
                        'end_price': prev_price, # prev_price is the end of the up streak
                        'start_df_index': streak_start_index,
                        'end_df_index': i -1 
                    })
                current_streak_type = 'down'
                current_streak_length = 1
                streak_start_price = prev_price # Streak starts from the price before the first decrease
                streak_start_index = i - 1
        else: # Price stayed the same (diff == 0)
            if current_streak_length > 0:
                # End the current streak
                detailed_streaks.append({
                    'type': current_streak_type,
                    'length': current_streak_length,
                    'start_price': streak_start_price,
                    'end_price': prev_price, # price before it became flat
                    'start_df_index': streak_start_index,
                    'end_df_index': i - 1
                })
            current_streak_type = None
            current_streak_length = 0
            streak_start_price = 0
            streak_start_index = 0

    # Account for the very last streak in the data
    if current_streak_length > 0 and current_streak_type is not None:
        detailed_streaks.append({
            'type': current_streak_type,
            'length': current_streak_length,
            'start_price': streak_start_price,
            'end_price': df[price_column].iloc[-1], # Last price of the data
            'start_df_index': streak_start_index,
            'end_df_index': len(df) - 1
        })
    
    return detailed_streaks

def perform_conditional_analysis(detailed_streaks: list[dict], 
                                 initial_streak_type: str, 
                                 initial_streak_length: int, 
                                 price_column_name: str) -> tuple[Counter, dict[int, list[float]]]:
    """
    Analyzes streaks that follow a specific initial streak.
    Returns a Counter for lengths of subsequent streaks and a dict for their avg profit/loss percentages.
    The dict maps subsequent streak length to a list of profit/loss percentages.
    """
    subsequent_streak_lengths = Counter()
    subsequent_streak_profit_details = {} # length -> list of profit_pcts

    for i in range(len(detailed_streaks) - 1):
        current_streak = detailed_streaks[i]
        next_streak = detailed_streaks[i+1]

        if current_streak['type'] == initial_streak_type and current_streak['length'] == initial_streak_length:
            # We found the initial target streak. Now look at the next_streak.
            # Ensure the next_streak is of the opposite type for this analysis.
            if (initial_streak_type == 'down' and next_streak['type'] == 'up') or \
               (initial_streak_type == 'up' and next_streak['type'] == 'down'):
                
                subsequent_streak_lengths[next_streak['length']] += 1
                
                # Calculate profit/loss percentage for the subsequent streak
                # For an up-streak after down: (next_end - next_start) / next_start
                # For a down-streak after up: (next_end - next_start) / next_start (will be negative)
                start_p = next_streak['start_price']
                end_p = next_streak['end_price']
                
                if start_p != 0: # Avoid division by zero
                    profit_pct = ((end_p - start_p) / start_p) * 100
                else:
                    profit_pct = 0

                if next_streak['length'] not in subsequent_streak_profit_details:
                    subsequent_streak_profit_details[next_streak['length']] = []
                subsequent_streak_profit_details[next_streak['length']].append(profit_pct)
                
    return subsequent_streak_lengths, subsequent_streak_profit_details

def print_conditional_results(condition_desc: str, 
                              subsequent_desc: str, 
                              lengths_counter: Counter, 
                              profit_details: dict[int, list[float]]):
    print(f"\n{condition_desc}:")
    if not lengths_counter:
        print(f"  No specific {subsequent_desc.lower()} found matching the criteria.")
        return

    for length, count in sorted(lengths_counter.items()):
        plural = "s" if length > 1 else ""
        avg_profit = 0
        if length in profit_details and profit_details[length]:
            avg_profit = sum(profit_details[length]) / len(profit_details[length])
        
        print(f"  - {length} consecutive {subsequent_desc.lower()[:-1]}{plural}: {count} time(s), Avg P/L: {avg_profit:.2f}%")

def main():
    parser = argparse.ArgumentParser(description="Analyze consecutive price streaks from a CSV file.")
    parser.add_argument("--input-file", type=str, required=True,
                        help="Path to the input CSV file containing quote data.")
    parser.add_argument("--price-column", type=str, default="price",
                        help="Name of the column containing the price data (default: price).")
    parser.add_argument("--timestamp-column", type=str, default="timestamp",
                        help="Name of the column containing timestamp data for sorting (default: timestamp).")
    parser.add_argument("--follow-down", type=int, metavar='N',
                        help="Analyze up-streaks immediately following exactly N consecutive down-ticks.")
    parser.add_argument("--follow-up", type=int, metavar='M',
                        help="Analyze down-streaks immediately following exactly M consecutive up-ticks.")
    parser.add_argument("--analyze-per-minute", action="store_true",
                        help="Enable analysis of price streaks on a per-minute basis (using last price of each minute).")
    
    args = parser.parse_args()

    try:
        df = pd.read_csv(args.input_file)
    except FileNotFoundError:
        print(f"Error: Input file not found at {args.input_file}")
        return
    except Exception as e:
        print(f"Error reading CSV file: {e}")
        return

    if args.timestamp_column not in df.columns:
        print(f"Error: Timestamp column '{args.timestamp_column}' not found in the CSV.")
        return
    if args.price_column not in df.columns:
        print(f"Error: Price column '{args.price_column}' not found in the CSV.")
        return

    try:
        # Attempt to convert to datetime, handling potential errors
        df[args.timestamp_column] = pd.to_datetime(df[args.timestamp_column], errors='coerce')
        df = df.dropna(subset=[args.timestamp_column]) # Remove rows where timestamp could not be converted
        df.sort_values(by=args.timestamp_column, inplace=True)
    except Exception as e:
        print(f"Error processing timestamp column '{args.timestamp_column}': {e}")
        return
        
    if df.empty:
        print("No data to analyze after initial processing.")
        return

    up_streaks, down_streaks = analyze_streaks(df, args.price_column)

    print(f"Analysis for price column: '{args.price_column}' in file: '{args.input_file}' (Tick-by-Tick)")
    
    print("\n--- Distribution of Consecutive Price Increases (Tick-by-Tick) ---")
    if up_streaks:
        for length, count in sorted(up_streaks.items()):
            plural = "s" if length > 1 else ""
            print(f"{length} consecutive increase{plural}: {count} time(s)")
    else:
        print("No consecutive price increases found (Tick-by-Tick).")

    print("\n--- Distribution of Consecutive Price Decreases (Tick-by-Tick) ---")
    if down_streaks:
        for length, count in sorted(down_streaks.items()):
            plural = "s" if length > 1 else ""
            print(f"{length} consecutive decrease{plural}: {count} time(s)")
    else:
        print("No consecutive price decreases found (Tick-by-Tick).")

    # --- Conditional Subsequent Streak Analysis ---
    if args.follow_down is not None or args.follow_up is not None:
        print("\n--- Conditional Subsequent Streak Analysis (Tick-by-Tick) ---")
        detailed_streaks = get_all_streaks_detailed(df.copy(), args.price_column) # Use a copy to avoid modifying original df
        if not detailed_streaks:
            print("No detailed streaks found to perform conditional analysis.")
        else:
            if args.follow_down is not None:
                up_after_down_lengths, up_after_down_avg_pct = perform_conditional_analysis(
                    detailed_streaks, 'down', args.follow_down, args.price_column
                )
                print_conditional_results(
                    f"After {args.follow_down} consecutive DECREASE(S)", 
                    "subsequent INCREASES", 
                    up_after_down_lengths, 
                    up_after_down_avg_pct
                )

            if args.follow_up is not None:
                down_after_up_lengths, down_after_up_avg_pct = perform_conditional_analysis(
                    detailed_streaks, 'up', args.follow_up, args.price_column
                )
                print_conditional_results(
                    f"After {args.follow_up} consecutive INCREASE(S)", 
                    "subsequent DECREASES", 
                    down_after_up_lengths, 
                    down_after_up_avg_pct
                )

    # --- Per-Minute Streak Analysis ---
    if args.analyze_per_minute:
        print("\n--- Per-Minute Streak Analysis ---")
        # Ensure the DataFrame is sorted by timestamp before resampling
        df_sorted_for_minute = df.sort_values(by=args.timestamp_column)
        
        # Resample to 1-minute intervals, taking the last price in each minute
        minute_price_series = df_sorted_for_minute.set_index(args.timestamp_column)[args.price_column].resample('T').last().dropna()
        if minute_price_series.empty:
            print("No data available after resampling to 1-minute intervals.")
        else:
            minute_df = minute_price_series.reset_index()
            # The price column in minute_df is now args.price_column (due to .last()), timestamp is args.timestamp_column
            # No, reset_index will make the price column likely '0' or its original name. We need to rename it or ensure analyze_streaks uses it.
            # Let's rename the price column after reset_index if it's not already the expected name.
            # If original price col was 'price', series name is 'price'. reset_index(name=...) is better.
            minute_df = minute_price_series.reset_index(name=args.price_column)
            
            print(f"Analyzing {len(minute_df)} one-minute intervals.")
            min_up_streaks, min_down_streaks = analyze_streaks(minute_df, args.price_column)
            
            print("\n--- Distribution of Consecutive Per-Minute Price Increases ---")
            if min_up_streaks:
                for length, count in sorted(min_up_streaks.items()):
                    plural = "s" if length > 1 else ""
                    print(f"{length} consecutive minute increase{plural}: {count} time(s)")
            else:
                print("No consecutive per-minute price increases found.")

            print("\n--- Distribution of Consecutive Per-Minute Price Decreases ---")
            if min_down_streaks:
                for length, count in sorted(min_down_streaks.items()):
                    plural = "s" if length > 1 else ""
                    print(f"{length} consecutive minute decrease{plural}: {count} time(s)")
            else:
                print("No consecutive per-minute price decreases found.")

if __name__ == "__main__":
    main() 