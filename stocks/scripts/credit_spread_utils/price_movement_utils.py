"""
Price movement analysis utilities.

Analyzes price movements from a given time of day to market close,
or close-to-close if no time specified. Outputs statistical summary
with percentiles and optional histogram visualization.

Extracted from scripts/analyze_price_movements.py for reuse via
analyze_credit_spread_intervals.py --mode price-movements.
"""

import sys
from datetime import datetime, time, timedelta
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from zoneinfo import ZoneInfo

# Timezone mapping
TIMEZONE_MAP = {
    'PST': 'America/Los_Angeles',
    'PDT': 'America/Los_Angeles',
    'EST': 'America/New_York',
    'EDT': 'America/New_York',
    'UTC': 'UTC',
}

# Market close time in Eastern Time
MARKET_CLOSE_ET = time(16, 0)  # 4:00 PM ET


def load_ticker_data(data_dir: str, ticker: str,
                     start_date: Optional[str] = None,
                     end_date: Optional[str] = None) -> pd.DataFrame:
    """
    Load all CSV files for a ticker within the date range.

    Returns a DataFrame with all data sorted by timestamp.
    """
    ticker_dir = Path(data_dir) / ticker

    if not ticker_dir.exists():
        raise FileNotFoundError(f"Ticker directory not found: {ticker_dir}")

    # Parse date filters
    start_dt = datetime.strptime(start_date, '%Y-%m-%d').date() if start_date else None
    end_dt = datetime.strptime(end_date, '%Y-%m-%d').date() if end_date else None

    # Find all CSV files
    csv_files = sorted(ticker_dir.glob(f'{ticker}_equities_*.csv'))

    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {ticker_dir}")

    # Filter files by date range
    filtered_files = []
    for f in csv_files:
        # Extract date from filename: TICKER_equities_YYYY-MM-DD.csv
        date_str = f.stem.split('_')[-1]
        try:
            file_date = datetime.strptime(date_str, '%Y-%m-%d').date()
            if start_dt and file_date < start_dt:
                continue
            if end_dt and file_date > end_dt:
                continue
            filtered_files.append(f)
        except ValueError:
            continue

    if not filtered_files:
        raise ValueError(f"No data files found for {ticker} in specified date range")

    # Load and concatenate all files
    dfs = []
    for f in filtered_files:
        try:
            df = pd.read_csv(f)
            if not df.empty:
                dfs.append(df)
        except Exception as e:
            print(f"Warning: Error loading {f}: {e}", file=sys.stderr)

    if not dfs:
        raise ValueError(f"No valid data loaded for {ticker}")

    combined = pd.concat(dfs, ignore_index=True)
    combined['timestamp'] = pd.to_datetime(combined['timestamp'], utc=True)
    combined = combined.sort_values('timestamp').reset_index(drop=True)

    return combined


def get_market_close_utc(date: datetime.date) -> datetime:
    """
    Get market close time in UTC for a given date.
    Market closes at 4:00 PM Eastern Time.
    """
    et_tz = ZoneInfo('America/New_York')
    close_et = datetime.combine(date, MARKET_CLOSE_ET, tzinfo=et_tz)
    return close_et.astimezone(ZoneInfo('UTC'))


def convert_time_to_utc(from_time: str, date: datetime.date, timezone: str) -> datetime:
    """
    Convert a time string (HH:MM) in the given timezone to UTC datetime for a specific date.
    """
    tz_name = TIMEZONE_MAP.get(timezone, timezone)
    tz = ZoneInfo(tz_name)

    hour, minute = map(int, from_time.split(':'))
    local_dt = datetime.combine(date, time(hour, minute), tzinfo=tz)
    return local_dt.astimezone(ZoneInfo('UTC'))


def get_daily_data(df: pd.DataFrame) -> dict:
    """
    Group data by trading day and return dict of date -> DataFrame.

    A trading day is defined by the market calendar (data belongs to the day
    the market was open, based on the close time in ET).
    """
    et_tz = ZoneInfo('America/New_York')

    daily_data = {}

    # Group by calendar date in ET
    df['date_et'] = df['timestamp'].dt.tz_convert(et_tz).dt.date

    for date, group in df.groupby('date_et'):
        # Skip days with very little data (less than 60 minutes of trading)
        if len(group) < 12:  # 12 * 5min = 60 minutes minimum
            continue
        daily_data[date] = group.copy()

    return daily_data


def calculate_movements(df: pd.DataFrame, from_time: Optional[str] = None,
                       timezone: str = 'PST',
                       day_direction: Optional[str] = None) -> List[Tuple[datetime.date, float]]:
    """
    Calculate price movements for each trading day.

    For each trading day:
    - If from_time specified: find price at that time, calculate % change to close
    - If no from_time: calculate % change from prior day's close to current close

    Args:
        day_direction: If 'up', only include days that closed higher than prior day.
                      If 'down', only include days that closed lower than prior day.

    Returns list of (date, pct_change) tuples.
    """
    daily_data = get_daily_data(df)
    dates = sorted(daily_data.keys())

    movements = []
    prev_close = None

    for date in dates:
        day_df = daily_data[date]

        # Get market close time and find the closest data point
        market_close_utc = get_market_close_utc(date)

        # Find data points before or at market close
        close_data = day_df[day_df['timestamp'] <= market_close_utc]
        if close_data.empty:
            # Use all data if nothing is before close (extended hours only)
            close_data = day_df

        # Get the last price as close price
        close_price = close_data.iloc[-1]['close']

        # Determine if this is an up or down day (vs prior close)
        is_up_day = None
        if prev_close is not None and prev_close > 0:
            is_up_day = close_price > prev_close

        # Check day_direction filter
        if day_direction:
            if prev_close is None:
                # Can't determine direction for first day
                prev_close = close_price
                continue
            if day_direction == 'up' and not is_up_day:
                prev_close = close_price
                continue
            if day_direction == 'down' and is_up_day:
                prev_close = close_price
                continue

        if from_time:
            # Calculate movement from specified time to close
            target_time_utc = convert_time_to_utc(from_time, date, timezone)

            # Find data point closest to (but not after) target time
            # Allow a small window (within 5 minutes before target)
            window_start = target_time_utc - timedelta(minutes=5)
            candidates = day_df[(day_df['timestamp'] >= window_start) &
                               (day_df['timestamp'] <= target_time_utc)]

            if candidates.empty:
                # Try to find closest point after target time (within 10 min)
                window_end = target_time_utc + timedelta(minutes=10)
                candidates = day_df[(day_df['timestamp'] > target_time_utc) &
                                   (day_df['timestamp'] <= window_end)]

            if candidates.empty:
                prev_close = close_price
                continue  # Skip this day - no data at target time

            # Use the closest data point
            time_diffs = abs(candidates['timestamp'] - target_time_utc)
            start_idx = time_diffs.idxmin()
            start_price = day_df.loc[start_idx, 'close']

            if start_price > 0:
                pct_change = ((close_price - start_price) / start_price) * 100
                movements.append((date, pct_change))
        else:
            # Calculate close-to-close movement
            if prev_close is not None and prev_close > 0:
                pct_change = ((close_price - prev_close) / prev_close) * 100
                movements.append((date, pct_change))

        prev_close = close_price

    return movements


def calculate_statistics(movements: List[Tuple[datetime.date, float]]) -> dict:
    """Calculate statistical summary of movements.

    Percentiles are computed by magnitude (absolute value), so:
    - 5th percentile = smallest magnitude (closest to 0)
    - 95th/99th percentile = largest magnitude (most extreme moves)

    This makes percentiles intuitive for both up and down day analysis:
    - Down days: 95th percentile = worst losses (most negative)
    - Up days: 95th percentile = best gains (most positive)
    """
    if not movements:
        return None

    pct_changes = [m[1] for m in movements]
    arr = np.array(pct_changes)

    positive_days = sum(1 for p in pct_changes if p > 0)
    negative_days = sum(1 for p in pct_changes if p < 0)
    zero_days = sum(1 for p in pct_changes if p == 0)

    # Sort by magnitude (absolute value) for percentile calculation
    sorted_by_magnitude = sorted(pct_changes, key=lambda x: abs(x))
    n = len(sorted_by_magnitude)

    def magnitude_percentile(p: int) -> float:
        """Get value at the p-th percentile by magnitude."""
        if p == 100:
            return sorted_by_magnitude[-1]
        # Index for percentile (0-based)
        idx = int((p / 100.0) * (n - 1))
        return sorted_by_magnitude[idx]

    return {
        'count': len(pct_changes),
        'mean': np.mean(arr),
        'std': np.std(arr),
        'min': np.min(arr),
        'max': np.max(arr),
        'median': np.median(arr),
        'percentiles': {
            5: magnitude_percentile(5),
            10: magnitude_percentile(10),
            25: magnitude_percentile(25),
            50: magnitude_percentile(50),
            75: magnitude_percentile(75),
            90: magnitude_percentile(90),
            95: magnitude_percentile(95),
            98: magnitude_percentile(98),
            99: magnitude_percentile(99),
            100: magnitude_percentile(100),
        },
        'positive_days': positive_days,
        'negative_days': negative_days,
        'zero_days': zero_days,
        'positive_pct': (positive_days / len(pct_changes)) * 100 if pct_changes else 0,
        'negative_pct': (negative_days / len(pct_changes)) * 100 if pct_changes else 0,
    }


def print_statistics(stats: dict, ticker: str, start_date: str, end_date: str,
                    from_time: Optional[str], timezone: str,
                    day_direction: Optional[str] = None):
    """Print formatted statistics to console."""
    measurement = f"From {from_time} {timezone} to Close" if from_time else "Close-to-Close"

    print()
    print("=" * 40)
    print("PRICE MOVEMENT STATISTICS")
    print("=" * 40)
    print(f"Ticker:        {ticker}")
    print(f"Period:        {start_date} to {end_date}")
    print(f"Measurement:   {measurement}")
    if day_direction:
        print(f"Day Filter:    {day_direction.upper()} days only")
    print(f"Days Analyzed: {stats['count']}")
    print()
    print(f"Mean:          {stats['mean']:+.2f}%")
    print(f"Std Dev:       {stats['std']:.2f}%")
    print(f"Min:           {stats['min']:+.2f}%")
    print(f"Max:           {stats['max']:+.2f}%")
    print(f"Median:        {stats['median']:+.2f}%")
    print()
    print("Percentiles:")
    for p, val in stats['percentiles'].items():
        print(f"  {p:3d}th:       {val:+.2f}%")
    print()
    print(f"Positive Days: {stats['positive_days']} ({stats['positive_pct']:.1f}%)")
    print(f"Negative Days: {stats['negative_days']} ({stats['negative_pct']:.1f}%)")
    if stats['zero_days'] > 0:
        zero_pct = (stats['zero_days'] / stats['count']) * 100
        print(f"Zero Days:     {stats['zero_days']} ({zero_pct:.1f}%)")
    print("=" * 40)
    print()


def generate_histogram(movements: List[Tuple[datetime.date, float]],
                      stats: dict, ticker: str, start_date: str, end_date: str,
                      from_time: Optional[str], timezone: str, output_path: str):
    """Generate histogram visualization."""
    try:
        import matplotlib.pyplot as plt
        from scipy import stats as scipy_stats
    except ImportError:
        print("Warning: matplotlib or scipy not available. Skipping histogram.", file=sys.stderr)
        return

    pct_changes = [m[1] for m in movements]

    fig, ax = plt.subplots(figsize=(12, 7))

    # Create histogram
    n, bins, patches = ax.hist(pct_changes, bins=50, density=True, alpha=0.7,
                                color='steelblue', edgecolor='black', linewidth=0.5)

    # Add normal distribution overlay
    x = np.linspace(min(pct_changes), max(pct_changes), 100)
    mu, std = stats['mean'], stats['std']
    normal_curve = scipy_stats.norm.pdf(x, mu, std)
    ax.plot(x, normal_curve, 'r-', linewidth=2, label='Normal Distribution')

    # Add percentile lines
    percentile_colors = {5: 'red', 25: 'orange', 50: 'green', 75: 'orange', 95: 'red'}
    percentile_styles = {5: '--', 25: '-.', 50: '-', 75: '-.', 95: '--'}

    for p, val in stats['percentiles'].items():
        if p in percentile_colors:
            ax.axvline(x=val, color=percentile_colors[p], linestyle=percentile_styles[p],
                      linewidth=1.5, label=f'{p}th percentile: {val:+.2f}%')

    # Labels and title
    measurement = f"From {from_time} {timezone} to Close" if from_time else "Close-to-Close"
    ax.set_xlabel('Price Change (%)', fontsize=12)
    ax.set_ylabel('Density', fontsize=12)
    ax.set_title(f'{ticker} Daily Price Movements\n{measurement}\n{start_date} to {end_date} ({stats["count"]} days)',
                fontsize=14)

    # Legend
    ax.legend(loc='upper right', fontsize=9)

    # Grid
    ax.grid(True, alpha=0.3)

    # Add statistics text box
    textstr = f'Mean: {mu:+.2f}%\nStd: {std:.2f}%\nMedian: {stats["median"]:+.2f}%'
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=props)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Histogram saved to: {output_path}")


def run_price_movement_analysis(args):
    """
    Run price movement analysis using parsed args.

    This is the top-level orchestrator called from analyze_credit_spread_intervals.py
    --mode price-movements or from the thin wrapper script.
    """
    # Validate from-time format
    if args.from_time:
        try:
            parts = args.from_time.split(':')
            if len(parts) != 2:
                raise ValueError()
            hour, minute = int(parts[0]), int(parts[1])
            if not (0 <= hour <= 23 and 0 <= minute <= 59):
                raise ValueError()
        except ValueError:
            print(f"Error: Invalid time format '{args.from_time}'. Use HH:MM format.", file=sys.stderr)
            return 1

    # Resolve data_dir — use args.data_dir if available, otherwise default
    data_dir = getattr(args, 'data_dir', 'equities_output') or 'equities_output'

    # Resolve timezone — use args.pm_timezone if available (from --mode), else args.timezone
    tz = getattr(args, 'pm_timezone', None) or getattr(args, 'timezone', 'PST') or 'PST'

    # Load data
    ticker = getattr(args, 'underlying_ticker', None) or getattr(args, 'ticker', None)
    if not ticker:
        print("Error: --ticker is required for price-movements mode", file=sys.stderr)
        return 1

    print(f"Loading data for {ticker} from {data_dir}/...")
    try:
        df = load_ticker_data(data_dir, ticker,
                              getattr(args, 'start_date', None),
                              getattr(args, 'end_date', None))
    except (FileNotFoundError, ValueError) as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    # Determine actual date range from loaded data
    dates = pd.to_datetime(df['timestamp']).dt.date
    actual_start = dates.min()
    actual_end = dates.max()

    print(f"Found data from {actual_start} to {actual_end}")

    from_time = getattr(args, 'from_time', None)
    day_direction = getattr(args, 'day_direction', None)

    # Calculate movements
    if from_time:
        print(f"Analyzing movements from {from_time} {tz} to close...")
    else:
        print("Analyzing close-to-close movements...")

    movements = calculate_movements(df, from_time, tz, day_direction)

    if not movements:
        print("Error: No valid movements calculated. Check data availability.", file=sys.stderr)
        return 1

    # Get date range from movements
    move_dates = [m[0] for m in movements]
    start_date_str = str(min(move_dates))
    end_date_str = str(max(move_dates))

    print(f"Analyzed {len(movements)} trading days")

    # Calculate and print statistics
    stats = calculate_statistics(movements)
    print_statistics(stats, ticker, start_date_str, end_date_str,
                    from_time, tz, day_direction)

    # Generate histogram if not disabled
    no_plot = getattr(args, 'no_plot', False)
    plot_output = getattr(args, 'plot_output', None) or getattr(args, 'output', 'price_movements.png')

    if not no_plot:
        generate_histogram(movements, stats, ticker, start_date_str, end_date_str,
                          from_time, tz, plot_output)

    return 0
