# in @s1.py create a stock trading strategy (using backtesting) with the following info managed and passed in
# - take in amt to invest
# - how many days worth of data to build the learnings from
# - an input of the stock symbol to look at
# - is it hourly or daily data that we'd like to learn from

# note the data of the symbol's csv is in data/hourly|daily/symbol_[hourly|daily].csv.
# the directory of the csv should be allowed to be passed in as an argument

# the csv looks like the following for example
# datetime,close,high,low,n,open,volume,vw
# 2023-05-22 08:00:00+00:00,318.71,318.84,318.42,272,318.45,9569,318.643344
# 2023-05-22 09:00:00+00:00,318.69,318.74,318.24,363,318.42,15632,318.445798
# 2023-05-22 10:00:00+00:00,318.6,318.75,318.2,247,318.4,10823,318.528768


# strategy to implement
# - take a look at days worth of data to figure out how many times to invest in one hour and selling later
# - you can invest more before cashing out the previous investments
# - all investments should end in the same day (in market trading hours)
# - can invest in the same hour multiple times
# - dont have to invest all the money in one iteration
# - take in a parameter to provide the minimum number of hours to invest


# the program should output based on the week of the day for the hour of the day how much to invest when to invest and when to sell to make it profitable w/ high probability
# given the amt being invested and the probability to succeed, can you also print out in the col how much to invest of the total amount in that hour. 
# i'd like to invest as much based on strategies passed in of aim_for_growth or minimize_loss.

#take a parameter to only incorporate data when the volume in that hour is within some % of the mean 

import argparse
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta
import pytz # For timezone handling

def load_data(data_dir: str, symbol: str, timeframe: str, learning_days: int, volume_filter_pct: int | None) -> pd.DataFrame:
    """Loads and prepares stock data from a CSV file."""
    file_path = Path(data_dir) / timeframe / f"{symbol.upper()}_{timeframe}.csv"
    if not file_path.exists():
        print(f"Error: Data file not found at {file_path}")
        return pd.DataFrame()

    df = pd.read_csv(file_path)
    
    # Ensure required columns exist
    required_cols = ['datetime', 'open', 'close', 'high', 'low']
    if not all(col in df.columns for col in required_cols):
        print(f"Error: CSV file must contain columns: {', '.join(required_cols)}")
        return pd.DataFrame()

    # Convert to datetime and localize to US/Eastern
    try:
        df['datetime'] = pd.to_datetime(df['datetime'])
        if df['datetime'].dt.tz is None:
            df['datetime'] = df['datetime'].dt.tz_localize('UTC') # Assume UTC if naive
        df['datetime_et'] = df['datetime'].dt.tz_convert('US/Eastern')
    except Exception as e:
        print(f"Error processing datetime column: {e}")
        return pd.DataFrame()

    df['date_et'] = df['datetime_et'].dt.date
    df['day_of_week_et'] = df['datetime_et'].dt.day_name()
    df['hour_et'] = df['datetime_et'].dt.hour

    # Filter for the learning period (most recent N unique trading days)
    if not df.empty:
        unique_trading_days = sorted(df['date_et'].unique(), reverse=True)
        if len(unique_trading_days) < learning_days:
            print(f"Warning: Requested {learning_days} learning days, but only {len(unique_trading_days)} unique trading days are available in the data.")
            cutoff_date = unique_trading_days[-1] if unique_trading_days else datetime.now(pytz.timezone('US/Eastern')).date() - timedelta(days=1)
        else:
            cutoff_date = unique_trading_days[learning_days - 1]
        
        df = df[df['date_et'] >= cutoff_date].copy()

    # Apply volume filter if specified
    if not df.empty and volume_filter_pct is not None and volume_filter_pct > 0:
        if 'volume' not in df.columns:
            print("Warning: 'volume' column not found in data. Cannot apply volume filter.")
            return df
        
        print(f"Applying volume filter: +/- {volume_filter_pct}% of mean volume for each hour...")
        # Calculate mean volume for each hour of the day
        mean_volume_per_hour = df.groupby('hour_et')['volume'].mean()
        
        # Create columns for lower and upper volume bounds based on the mean for that hour
        df['mean_hourly_volume'] = df['hour_et'].map(mean_volume_per_hour)
        df['volume_lower_bound'] = df['mean_hourly_volume'] * (1 - volume_filter_pct / 100.0)
        df['volume_upper_bound'] = df['mean_hourly_volume'] * (1 + volume_filter_pct / 100.0)
        
        # Filter the DataFrame
        original_rows = len(df)
        df = df[(df['volume'] >= df['volume_lower_bound']) & (df['volume'] <= df['volume_upper_bound'])].copy()
        filtered_rows = len(df)
        print(f"Volume filter removed {original_rows - filtered_rows} rows. {filtered_rows} rows remaining.")
        
        # Clean up temporary columns
        df.drop(columns=['mean_hourly_volume', 'volume_lower_bound', 'volume_upper_bound'], inplace=True)

    return df

def run_backtest(df: pd.DataFrame, investment_amount: float, min_hold_hours: int) -> pd.DataFrame:
    """
    Runs the backtesting strategy.
    Strategy: Buy at open of an hour, sell at close of same or subsequent hour within the same day.
    Market hours for entry: 9 AM - 3 PM ET (inclusive, representing 09:00 to 15:00 bars).
    Trades must close by the end of the 3 PM ET hour (i.e., by 15:59:59 ET).
    """
    trades = []
    
    # Define market hours (e.g., 9 for 9:00-9:59, 15 for 15:00-15:59)
    market_open_hour = 9
    market_close_hour = 15 # Last hour for entry and exit

    if df.empty:
        return pd.DataFrame()

    # Iterate over each unique trading day in the filtered data
    for trading_date in sorted(df['date_et'].unique()):
        day_df = df[df['date_et'] == trading_date].sort_values(by='datetime_et')

        for entry_hour in range(market_open_hour, market_close_hour + 1):
            entry_candle = day_df[day_df['hour_et'] == entry_hour]
            if entry_candle.empty:
                continue
            
            entry_price = entry_candle.iloc[0]['open']
            entry_datetime_et = entry_candle.iloc[0]['datetime_et']
            day_of_week = entry_datetime_et.day_name()

            # Adjust the starting point for exit_hour based on min_hold_hours
            start_exit_hour = entry_hour + min_hold_hours

            for exit_hour in range(start_exit_hour, market_close_hour + 1):
                # Ensure we don't try to exit before the market opens if min_hold_hours is large
                # and pushes start_exit_hour beyond market_open_hour for a later day (though this strategy is intraday)
                # More importantly, ensure exit_hour is within the current trading day's market hours.
                if exit_hour > market_close_hour:
                    continue # Should not happen with the loop range, but good for safety

                exit_candle = day_df[day_df['hour_et'] == exit_hour]
                if exit_candle.empty:
                    continue

                exit_price = exit_candle.iloc[0]['close']
                exit_datetime_et = exit_candle.iloc[0]['datetime_et']
                
                if entry_price == 0: # Avoid division by zero
                    profit_pct = 0
                else:
                    profit_pct = ((exit_price - entry_price) / entry_price) * 100
                
                trades.append({
                    'entry_datetime_et': entry_datetime_et,
                    'exit_datetime_et': exit_datetime_et,
                    'entry_price': entry_price,
                    'exit_price': exit_price,
                    'profit_pct': profit_pct,
                    'day_of_week': day_of_week,
                    'entry_hour': entry_hour,
                    'exit_hour': exit_hour,
                })
                
    return pd.DataFrame(trades)

def analyze_trades(trades_df: pd.DataFrame) -> pd.DataFrame:
    """Analyzes trades to find profitable patterns."""
    if trades_df.empty:
        return pd.DataFrame()

    analysis = trades_df.groupby(['day_of_week', 'entry_hour', 'exit_hour']).agg(
        num_trades=('profit_pct', 'count'),
        avg_profit_pct=('profit_pct', 'mean'),
        win_rate=('profit_pct', lambda x: (x > 0).mean() * 100)
    ).reset_index()
    
    analysis.sort_values(by=['win_rate', 'avg_profit_pct'], ascending=[False, False], inplace=True)
    return analysis

def calculate_investment_allocation(df: pd.DataFrame, total_amount: float, strategy: str) -> pd.DataFrame:
    """
    Calculates investment allocation based on the specified strategy.
    """
    if df.empty:
        return df # Return the empty DataFrame directly

    # Ensure avg_profit_pct is not zero for aim_for_growth to prevent division by zero if all profits are tiny
    # and to ensure patterns with 0 profit don't get allocation if they somehow pass other filters.
    if strategy == 'aim_for_growth':
        # Score emphasizes both win rate and magnitude of average profit.
        # Add a small epsilon to avg_profit_pct to handle cases where it might be zero but win_rate is high,
        # though our filter (avg_profit_pct > 0) should prevent this.
        df['score'] = (df['win_rate'] / 100.0) * (df['avg_profit_pct'] + 1e-6) # Ensure positive avg_profit for scoring
    elif strategy == 'minimize_loss':
        # Score emphasizes higher win rates. avg_profit_pct is used as a tie-breaker or secondary factor.
        # We already filter for avg_profit_pct > 0.
        df['score'] = (df['win_rate'] / 100.0) * (df['win_rate'] / 100.0) # Weight win_rate more heavily
        # Optionally, add a smaller component of profit: + (df['avg_profit_pct'] / 100.0) * 0.1 
    else:
        print(f"Warning: Unknown allocation strategy: {strategy}. Defaulting to equal allocation.")
        df['score'] = 1 # Assign a uniform score for equal allocation

    # Handle cases where all scores are zero (e.g., if all avg_profit_pct were somehow <=0 and not filtered out)
    if df['score'].sum() == 0:
        if not df.empty:
            print("Warning: All pattern scores are zero. Allocating equally among promising patterns if any.")
            df['score'] = 1.0 / len(df) if len(df) > 0 else 0
        else:
            df['suggested_investment'] = 0.0
            return df.drop(columns=['score'], errors='ignore')
            
    total_score = df['score'].sum()
    
    if total_score > 0:
        df['allocation_ratio'] = df['score'] / total_score
        df['suggested_investment'] = df['allocation_ratio'] * total_amount
    else: # Fallback if total_score is still somehow zero (e.g. only one pattern with score 0)
        if not df.empty:
            print("Warning: Total score is zero. Allocating equally among promising patterns if any.")
            df['suggested_investment'] = total_amount / len(df) if len(df) > 0 else 0.0
        else:
            df['suggested_investment'] = 0.0

    return df.drop(columns=['score', 'allocation_ratio'], errors='ignore')

def main():
    parser = argparse.ArgumentParser(description="Stock Trading Strategy S1 - Backtesting")
    parser.add_argument("--amount", type=float, required=True, help="Total amount to invest.")
    parser.add_argument("--learning-days", type=int, required=True, help="Number of past days of data to learn from.")
    parser.add_argument("--symbol", type=str, required=True, help="Stock symbol (e.g., AAPL).")
    parser.add_argument("--timeframe", type=str, choices=['hourly', 'daily'], default='hourly', help="Timeframe of the data (hourly or daily). Currently, strategy is optimized for hourly.")
    parser.add_argument("--data-dir", type=str, default="./data", help="Directory where CSV data is stored.")
    parser.add_argument("--min-hold-hours", type=int, default=0, help="Minimum number of hours to hold an investment before selling (default: 0, can sell in the same hour).")
    parser.add_argument("--allocation-strategy", type=str, choices=['aim_for_growth', 'minimize_loss'], default='aim_for_growth', help="Strategy for allocating investment amount across promising patterns (default: aim_for_growth).")
    parser.add_argument("--volume-filter-pct", type=int, default=None, help="Filter data to include only hours where volume is within this +/- percentage of the mean volume for that specific hour of the day (e.g., 20 for +/-20%%). Default: None (no filter).")
    parser.add_argument("--sort-by", type=str, nargs='+', default=['win_rate', 'avg_profit_pct'], help="One or more columns to sort the promising patterns table by (e.g., win_rate avg_profit_pct num_trades suggested_investment). Default: win_rate avg_profit_pct. All sorting is descending.")
    
    args = parser.parse_args()

    if args.timeframe == 'daily':
        print("Warning: This strategy is primarily designed for hourly data. Running with daily data might not yield meaningful intraday results.")
        # For daily, the concept of entry/exit hour as defined might not apply well.
        # Consider adapting strategy or exiting if daily is chosen. For now, it will proceed.

    print(f"Starting S1 backtesting for {args.symbol}...")
    print(f"Investment Amount: ${args.amount:,.2f}")
    print(f"Learning from last {args.learning_days} trading days of {args.timeframe} data.")
    print(f"Data directory: {Path(args.data_dir).resolve()}")
    
    # Load data
    data_df = load_data(args.data_dir, args.symbol, args.timeframe, args.learning_days, args.volume_filter_pct)
    if data_df.empty:
        print("Could not load data. Exiting.")
        return

    # Run backtest
    print(f"\nRunning backtest on {len(data_df)} data points...")
    trades_df = run_backtest(data_df, args.amount, args.min_hold_hours)
    if trades_df.empty:
        print("No trades were generated. This could be due to insufficient data or no valid trading opportunities based on the strategy.")
        return
    
    print(f"Generated {len(trades_df)} simulated trades.")

    # Analyze trades
    analysis_df = analyze_trades(trades_df)
    
    if analysis_df.empty:
        print("No trade patterns found after analysis.")
        return

    # Output promising patterns
    promising_patterns = analysis_df[
        (analysis_df['win_rate'] >= 60) & (analysis_df['avg_profit_pct'] > 0)
    ].copy() # Use .copy() to avoid SettingWithCopyWarning when adding new column

    print("\n--- Historically Profitable Trading Patterns (Win Rate >= 60%, Avg. Profit > 0%) ---")
    if not promising_patterns.empty:
        # Calculate and add suggested investment amounts
        promising_patterns = calculate_investment_allocation(promising_patterns, args.amount, args.allocation_strategy)

        # Validate and apply custom sorting
        valid_sort_columns = [col for col in args.sort_by if col in promising_patterns.columns]
        if len(valid_sort_columns) != len(args.sort_by):
            invalid_cols = set(args.sort_by) - set(valid_sort_columns)
            print(f"Warning: Invalid column(s) provided for --sort-by: {invalid_cols}. Using default sort order.")
            sort_by_columns = ['win_rate', 'avg_profit_pct'] # Default sort
        else:
            sort_by_columns = valid_sort_columns
        
        # All specified columns will be sorted in descending order
        ascending_order = [False] * len(sort_by_columns)
        promising_patterns.sort_values(by=sort_by_columns, ascending=ascending_order, inplace=True)

        pd.set_option('display.max_rows', None)
        pd.set_option('display.width', 1000)
        # Ensure the new column is in the list of columns to print
        columns_to_display = ['day_of_week', 'entry_hour', 'exit_hour', 'num_trades', 'avg_profit_pct', 'win_rate', 'suggested_investment']
        print(promising_patterns[columns_to_display].to_string(index=False))
        
        print("\nInvestment Suggestion:")
        print(f"Based on these historical patterns for {args.symbol} over the last {args.learning_days} days:")
        print("- Consider these day/hour combinations for potential buy (at open) and sell (at close) signals.")
        print("- The 'avg_profit_pct' and 'win_rate' indicate historical performance of these specific hourly patterns.")
        print(f"- You could allocate portions of your ${args.amount:,.2f} investment amount to trades that match these patterns.")
        print("- Always consider risk management. Past performance is not indicative of future results.")
    else:
        print("No trading patterns met the criteria of >= 60% win rate and > 0% average profit.")
        print("Consider adjusting the learning period, symbol, or analyzing raw trade results for insights.")

    # print("\n--- All Analyzed Trade Patterns ---")
    # print(analysis_df.to_string(index=False))

if __name__ == "__main__":
    main()