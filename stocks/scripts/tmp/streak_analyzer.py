import sqlite3
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import argparse
from datetime import datetime, timedelta
import numpy as np

def analyze_streaks(db_path, ticker, threshold, lookback_days=None, return_data=False, analyze_performance=False):
    """
    Analyzes consecutive up/down day streaks following a significant price move.

    Args:
        db_path (str): The FULL, ABSOLUTE path to the SQLite database file.
        ticker (str): The stock ticker symbol to analyze.
        threshold (float): The percentage move (e.g., 0.01 for 1%) to trigger analysis.
        lookback_days (int, optional): Number of days to look back from the most recent date. 
                                     If None, uses all available data.
        return_data (bool): If True, returns the data for further analysis. If False, only prints and plots.
        analyze_performance (bool): If True, analyzes market performance after streaks end.

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
        # 1. Connect to DB and fetch data
        con = sqlite3.connect(db_path)
        query = f"SELECT date, close FROM daily_prices WHERE ticker = '{ticker}' ORDER BY date"
        df = pd.read_sql_query(query, con)
        con.close()

        if df.empty:
            print(f"No data found for {ticker} in the database.")
            if return_data:
                return None, None, None
            return

        # 2. Prepare the data
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
            if return_data:
                return None, None, None
            return

        df['pct_change'] = df['close'].pct_change()
        df['direction'] = 'Neutral'
        df.loc[df['pct_change'] > 0, 'direction'] = 'Up'
        df.loc[df['pct_change'] < 0, 'direction'] = 'Down'

        # 4. Identify streaks after trigger events
        streaks_after_spike = []
        streaks_after_dip = []
        
        # For performance analysis, we need to track more details about each streak
        spike_streak_details = []  # List of (streak_length, start_date, end_date, start_price, end_price)
        dip_streak_details = []

        for i in range(1, len(df)):
            trigger_pct_change = df.loc[i, 'pct_change']
            
            if trigger_pct_change > threshold: # Spike Trigger
                streak_len = 0
                streak_start_idx = i + 1
                streak_end_idx = i + 1
                
                for j in range(i + 1, len(df)):
                    if df.loc[j, 'direction'] == 'Up':
                        streak_len += 1
                        streak_end_idx = j
                    else:
                        break
                
                streaks_after_spike.append(streak_len)
                
                # Store details for performance analysis
                if analyze_performance and streak_start_idx < len(df):
                    start_date = df.loc[streak_start_idx, 'date'] if streak_start_idx < len(df) else None
                    end_date = df.loc[streak_end_idx, 'date'] if streak_end_idx < len(df) else None
                    start_price = df.loc[streak_start_idx, 'close'] if streak_start_idx < len(df) else None
                    end_price = df.loc[streak_end_idx, 'close'] if streak_end_idx < len(df) else None
                    
                    if start_date and end_date and start_price and end_price:
                        spike_streak_details.append((streak_len, start_date, end_date, start_price, end_price))

            elif trigger_pct_change < -threshold: # Dip Trigger
                streak_len = 0
                streak_start_idx = i + 1
                streak_end_idx = i + 1
                
                for j in range(i + 1, len(df)):
                    if df.loc[j, 'direction'] == 'Down':
                        streak_len += 1
                        streak_end_idx = j
                    else:
                        break
                
                streaks_after_dip.append(streak_len)
                
                # Store details for performance analysis
                if analyze_performance and streak_start_idx < len(df):
                    start_date = df.loc[streak_start_idx, 'date'] if streak_start_idx < len(df) else None
                    end_date = df.loc[streak_end_idx, 'date'] if streak_end_idx < len(df) else None
                    start_price = df.loc[streak_start_idx, 'close'] if streak_start_idx < len(df) else None
                    end_price = df.loc[streak_end_idx, 'close'] if streak_end_idx < len(df) else None
                    
                    if start_date and end_date and start_price and end_price:
                        dip_streak_details.append((streak_len, start_date, end_date, start_price, end_price))

        # 5. Generate the histogram data
        spike_counts = pd.Series(streaks_after_spike).value_counts().sort_index()
        dip_counts = pd.Series(streaks_after_dip).value_counts().sort_index()

        # Print results
        print(f"--- Streak Analysis for {ticker} ---")
        print(f"Found {len(streaks_after_spike)} spike events and {len(streaks_after_dip)} dip events")
        print(f"\n### Histogram for Consecutive UP Days After a >{threshold*100:.0f}% SPIKE ###")
        print("Streak Length | Number of Occurrences")
        print("-------------------------------------")
        for length, count in spike_counts.items():
            print(f"{length:<13} | {count}")

        print(f"\n### Histogram for Consecutive DOWN Days After a >{threshold*100:.0f}% DIP ###")
        print("Streak Length | Number of Occurrences")
        print("-------------------------------------")
        for length, count in dip_counts.items():
            print(f"{length:<13} | {count}")

        # 6. Performance Analysis (if requested)
        if analyze_performance:
            print(f"\n{'='*80}")
            print(f"📊 MARKET PERFORMANCE ANALYSIS AFTER STREAKS")
            print(f"{'='*80}")
            
            _analyze_streak_performance(df, spike_streak_details, "UP STREAKS (After Spike)", ticker)
            _analyze_streak_performance(df, dip_streak_details, "DOWN STREAKS (After Dip)", ticker)

        # 7. Plot the histograms
        sns.set_style("whitegrid")
        fig, axes = plt.subplots(1, 2, figsize=(16, 6), sharey=True)
        
        # Create title with lookback info
        title_suffix = f" (Last {lookback_days} Days)" if lookback_days else " (All Available Data)"
        fig.suptitle(f'Histogram of Consecutive Trading Day Streaks for {ticker} After a >{threshold*100:.0f}% Move{title_suffix}', fontsize=16)

        sns.barplot(x=spike_counts.index, y=spike_counts.values, ax=axes[0], palette='Greens_d')
        axes[0].set_title(f'Streaks Following a >{threshold*100:.0f}% UP Day')
        axes[0].set_xlabel('Length of Consecutive UP Streak')
        axes[0].set_ylabel('Number of Occurrences')

        sns.barplot(x=dip_counts.index, y=dip_counts.values, ax=axes[1], palette='Reds_d')
        axes[1].set_title(f'Streaks Following a >{threshold*100:.0f}% DOWN Day')
        axes[1].set_xlabel('Length of Consecutive DOWN Streak')
        
        for ax in axes:
            for p in ax.patches:
                ax.annotate(f'{int(p.get_height())}', (p.get_x() + p.get_width() / 2., p.get_height()), ha='center', va='center', xytext=(0, 9), textcoords='offset points')

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        
        if not return_data:
            # For command-line usage, save and show plot
            plot_filename = f'{ticker}_streak_histograms.png'
            plt.savefig(plot_filename)
            print(f"\nVisual histogram saved to '{plot_filename}'")
            plt.show()
        
        # Return data if requested (for notebook usage)
        if return_data:
            # Convert to DataFrame format for notebook display
            spike_counts_df = spike_counts.reset_index()
            spike_counts_df.columns = ['Streak Length', 'Number of Occurrences']
            
            dip_counts_df = dip_counts.reset_index()
            dip_counts_df.columns = ['Streak Length', 'Number of Occurrences']
            
            return spike_counts_df, dip_counts_df, fig
        
        return None

    except Exception as e:
        print(f"An error occurred: {e}")
        if return_data:
            return None, None, None
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
            end_date = row['end_date']
            end_price = row['end_price']
            
            # Calculate performance for different time periods
            for period, days in [('1_week', 7), ('2_weeks', 14), ('1_month', 30), ('2_months', 60), ('3_months', 90)]:
                future_date = end_date + timedelta(days=days)
                
                # Find the closest available price after the target date
                future_prices = df[df['date'] >= future_date]
                if not future_prices.empty:
                    future_price = future_prices.iloc[0]['close']
                    performance = ((future_price - end_price) / end_price) * 100
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
    print(f"\nAverage Market Performance After {streak_type} End:")
    print(f"{'Streak Length':<15} | {'1 Week':<8} | {'2 Weeks':<8} | {'1 Month':<8} | {'2 Months':<9} | {'3 Months':<9}")
    print(f"{'-' * 75}")
    
    for streak_len in sorted(performance_by_length.keys()):
        perfs = performance_by_length[streak_len]
        
        def format_perf(perf):
            if perf is None:
                return "N/A"
            else:
                return f"{perf:+6.2f}%"
        
        print(f"{streak_len:<15} | {format_perf(perfs['1_week']):<8} | {format_perf(perfs['2_weeks']):<8} | {format_perf(perfs['1_month']):<8} | {format_perf(perfs['2_months']):<9} | {format_perf(perfs['3_months']):<9}")

# --- Run the analysis ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze consecutive up/down day streaks for a stock.")
    parser.add_argument("--db-path", required=True, help="The FULL, ABSOLUTE path to the SQLite database file.")
    parser.add_argument("--ticker", default="QQQ", help="Stock ticker symbol to analyze (e.g., 'SPY').")
    parser.add_argument("--threshold", type=float, default=0.01, help="The percentage move to trigger a streak analysis (e.g., 0.01 for 1%).")
    parser.add_argument("--lookback-days", type=int, help="Number of days to look back from the most recent date. If not specified, uses all available data.")
    parser.add_argument("--analyze-performance", action="store_true", help="Analyze market performance after streaks end (1 week, 2 weeks, 1 month, 2 months, 3 months).")
    
    args = parser.parse_args()
    
    analyze_streaks(
        db_path=args.db_path, 
        ticker=args.ticker, 
        threshold=args.threshold, 
        lookback_days=args.lookback_days,
        analyze_performance=args.analyze_performance
    )
