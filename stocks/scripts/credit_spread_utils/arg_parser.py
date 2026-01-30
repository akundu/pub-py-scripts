"""
Command line argument parsing for analyze_credit_spread_intervals.py

Organized into logical groups for better maintainability.
"""

import argparse


def _add_input_args(parser: argparse.ArgumentParser):
    """Add input-related arguments."""
    parser.add_argument(
        "--csv-path",
        required=False,
        nargs='+',
        help="Path(s) to CSV file(s) with options data (timestamps in PST). Can provide multiple files for aggregate analysis. Not required if --csv-dir is provided."
    )
    parser.add_argument(
        "--csv-dir",
        type=str,
        default=None,
        help="Directory containing CSV files organized by ticker (e.g., csv_dir/TICKER/TICKER_options_YYYY-MM-DD.csv). Automatically appends ticker subdirectory. Requires --ticker or --underlying-ticker."
    )
    parser.add_argument(
        "--start-date",
        type=str,
        default=None,
        help="Start date for filtering CSV files (YYYY-MM-DD). Only processes files with dates >= start-date. If --end-date is not provided, processes all files from start-date to today."
    )
    parser.add_argument(
        "--end-date",
        type=str,
        default=None,
        help="End date for filtering CSV files (YYYY-MM-DD). Only processes files with dates <= end-date. Requires --start-date or --csv-dir."
    )
    parser.add_argument(
        "--underlying-ticker",
        dest="underlying_ticker",
        help="Underlying stock ticker (e.g., SPX, I:SPX). If not provided, will be extracted from option tickers in CSV"
    )
    parser.add_argument(
        "--ticker",
        dest="underlying_ticker",
        help="Alias for --underlying-ticker"
    )


def _add_filter_args(parser: argparse.ArgumentParser):
    """Add filtering-related arguments."""
    parser.add_argument(
        "--option-type",
        choices=["call", "put", "both"],
        default="both",
        help="Option type: call, put, or both (default: both)"
    )
    parser.add_argument(
        "--percent-beyond",
        type=str,
        required=False,
        help="Percentage beyond previous day's closing price. Can be a single value (e.g., 0.05 for 5%%) or two values separated by colon for puts:calls (e.g., 0.03:0.05 means 3%% for puts, 5%% for calls). Required unless --grid-config is used."
    )
    parser.add_argument(
        "--risk-cap",
        type=float,
        default=None,
        help="Maximum risk amount to cap the spread at. Optional if --max-spread-width is provided."
    )
    parser.add_argument(
        "--min-spread-width",
        type=float,
        default=5.0,
        help="Minimum spread width (strike difference)"
    )
    parser.add_argument(
        "--max-spread-width",
        type=str,
        default="200.0",
        help="Maximum spread width (strike difference). Can be a single value (e.g., 100) or two values for puts:calls (e.g., 50:100)"
    )
    parser.add_argument(
        "--min-contract-price",
        type=float,
        default=0.0,
        help="Minimum price for a contract to be included. Contracts at or below this price will be excluded. Default: 0.0"
    )
    parser.add_argument(
        "--max-credit-width-ratio",
        type=float,
        default=0.60,
        help="Maximum ratio of credit to spread width (default: 0.60 = 60%%). Filters out unrealistic spreads where credit is too close to width, which typically indicates stale pricing or deep ITM/OTM options. Use 1.0 to disable this filter."
    )
    parser.add_argument(
        "--max-strike-distance-pct",
        type=float,
        default=None,
        help="Maximum distance of short strike from previous close, as percentage (e.g., 0.05 = 5%%). Filters out deep ITM/OTM options with poor liquidity. Example: --max-strike-distance-pct 0.03 only allows strikes within 3%% of previous close."
    )
    parser.add_argument(
        "--min-premium-diff",
        type=str,
        default=None,
        help="Minimum premium price difference between short and long side (net credit). Can be a single value (e.g., 0.50) or two values for puts:calls (e.g., 0.30:0.50). Filters out spreads with insufficient premium difference. Example: --min-premium-diff 0.50 requires at least $0.50 net credit per share."
    )


def _add_trading_args(parser: argparse.ArgumentParser):
    """Add trading hours and capital management arguments."""
    parser.add_argument(
        "--min-trading-hour",
        type=int,
        default=None,
        help="Minimum hour of the day (in the timezone specified by --output-timezone) before which no new positions can be added. Default: None (no minimum). This allows you to start counting transactions only after a certain hour. Example: --min-trading-hour 9 starts trading only after 9:00 AM. Uses same timezone as --output-timezone."
    )
    parser.add_argument(
        "--max-trading-hour",
        type=int,
        default=15,
        help="Maximum hour of the day (in the timezone specified by --output-timezone) after which no new positions can be added. Default: 15 (3:00 PM). This prevents trading in the last hour when volatility can be extreme. Example: --max-trading-hour 14 stops trading after 2:00 PM. Uses same timezone as --output-timezone."
    )
    parser.add_argument(
        "--profit-target-pct",
        type=float,
        default=None,
        help="Profit target as percentage of max credit. If spread reaches this profit level, it is considered closed early. Example: --profit-target-pct 0.50 means exit when 50%% of max profit is reached. This simulates taking profits early rather than holding to expiration."
    )
    parser.add_argument(
        "--max-live-capital",
        type=float,
        default=None,
        help="Maximum dollar amount of active capital (max loss exposure) allowed per calendar day. Positions that would exceed this limit are skipped. Example: --max-live-capital 5000 limits total max loss exposure to $5000 per day. Uses calendar date in output timezone."
    )
    parser.add_argument(
        "--force-close-hour",
        type=int,
        default=None,
        help="Hour at which all trades must be closed (in the timezone specified by --output-timezone). Default: None (hold to expiration). Example: --force-close-hour 15 closes all positions at 3:00 PM. P&L is calculated based on the underlying price at this time. Use this to avoid holding positions through market close."
    )


def _add_output_args(parser: argparse.ArgumentParser):
    """Add output and display arguments."""
    parser.add_argument(
        "--summary",
        action="store_true",
        help="Print a one-line summarized view: date, max credit, num contracts, price diff %%"
    )
    parser.add_argument(
        "--summary-only",
        action="store_true",
        help="Print only the final one-line summary (no individual interval lines)"
    )
    parser.add_argument(
        "--output-timezone",
        default="America/Los_Angeles",
        help="Timezone for displayed timestamps (e.g., America/Los_Angeles, America/New_York, UTC, PST, PDT, EST, EDT). Default: America/Los_Angeles"
    )
    parser.add_argument(
        "--most-recent",
        action="store_true",
        help="Only show the best result(s) for the most recent timestamp(s). Useful for identifying current investment opportunities."
    )
    parser.add_argument(
        "--best-only",
        action="store_true",
        help="When used with --most-recent, show only the single best spread (call or put) from the latest data. Use this to get the one actionable investment opportunity right now. Requires --most-recent."
    )
    parser.add_argument(
        "--continuous",
        type=float,
        nargs='?',
        const=10.0,
        default=None,
        help="Continuously run analysis in a loop. Optionally specify wait time in seconds between runs (default: 10 seconds). Use with --most-recent --best-only for current investment opportunities. Similar to continuous mode in fetch_all_data.py."
    )
    parser.add_argument(
        "--run-once-before-wait",
        action="store_true",
        help="If market is closed, run once immediately before waiting for market open. Only effective with --continuous. Useful since options data doesn't change during non-market hours."
    )
    parser.add_argument(
        "--use-market-hours",
        action="store_true",
        help="Use market hours awareness to adjust run intervals (longer intervals when markets are closed). Only effective with --continuous. Off by default."
    )
    parser.add_argument(
        "--continuous-max-runs",
        type=int,
        default=None,
        help="Maximum number of continuous runs before stopping (default: run indefinitely). Only effective with --continuous."
    )
    parser.add_argument(
        "--curr-price",
        action="store_true",
        help="Use current/latest price instead of previous trading day's close price. Only effective with --continuous. Fetches the most recent price from the database (realtime -> hourly -> daily)."
    )
    parser.add_argument(
        "--histogram",
        action="store_true",
        help="Generate histogram of hourly performance when multiple CSV files are provided. Shows success/failure rates and total counts by hour."
    )
    parser.add_argument(
        "--histogram-output",
        default="credit_spread_hourly_analysis.png",
        help="Output filename for histogram (default: credit_spread_hourly_analysis.png)"
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=None,
        help="Only show top N spreads per day (ranked by max credit). Useful for realistic backtest scenarios where you only take the best opportunities. Example: --top-n 3 shows only the 3 best spreads each day."
    )


def _add_advanced_args(parser: argparse.ArgumentParser):
    """Add advanced configuration arguments."""
    parser.add_argument(
        "--use-mid-price",
        action="store_true",
        help="Use mid-price (bid+ask)/2 instead of bid/ask"
    )
    parser.add_argument(
        "--processes",
        type=int,
        default=1,
        help="Number of parallel processes to use for processing multiple files. Default: 1 (sequential). Use 0 to auto-detect CPU count. Only effective when processing multiple CSV files."
    )
    parser.add_argument(
        "--db-path",
        dest="db_path",
        help="QuestDB connection string (default: from environment)"
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Disable Redis cache"
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        help="Logging level"
    )
    parser.add_argument(
        "--no-data-cache",
        action="store_true",
        help="Disable binary data cache (always load from CSVs)"
    )
    parser.add_argument(
        "--cache-dir",
        default=".options_cache",
        help="Directory for binary data cache (default: .options_cache)"
    )
    parser.add_argument(
        "--clear-cache",
        action="store_true",
        help="Clear the binary data cache and exit"
    )
    parser.add_argument(
        "--grid-config",
        type=str,
        default=None,
        help="Path to YAML file with grid search configuration. When provided, runs grid search instead of normal analysis."
    )
    parser.add_argument(
        "--grid-output",
        type=str,
        default="grid_results.csv",
        help="Output CSV file for grid search results (default: grid_results.csv)"
    )
    parser.add_argument(
        "--grid-resume",
        action="store_true",
        help="Resume grid search from existing results file (skips already-computed combinations)"
    )
    parser.add_argument(
        "--grid-sort",
        type=str,
        default="net_pnl",
        choices=["net_pnl", "profit_factor", "win_rate", "total_trades"],
        help="Sort key for grid search results (default: net_pnl)"
    )
    parser.add_argument(
        "--grid-top-n",
        type=int,
        default=20,
        help="Number of top results to display from grid search (default: 20)"
    )
    parser.add_argument(
        "--grid-dry-run",
        action="store_true",
        help="Show grid search configuration without running the backtests"
    )


def _add_rate_limit_args(parser: argparse.ArgumentParser):
    """Add rate limiting arguments."""
    parser.add_argument(
        "--rate-limit-max",
        type=int,
        default=0,
        help="Max transactions allowed in the rate limit window. 0 = disabled (default). "
             "Example: --rate-limit-max 100 --rate-limit-window 60 = 100 per minute."
    )
    parser.add_argument(
        "--rate-limit-window",
        type=float,
        default=0,
        help="Time window in seconds for rate limiting. 0 = disabled (default). "
             "Example: --rate-limit-max 100 --rate-limit-window 60 = 100 per minute."
    )


def parse_args() -> argparse.Namespace:
    """Parse command line arguments.

    Returns:
        Parsed arguments namespace
    """
    parser = argparse.ArgumentParser(
        description="Analyze credit spreads at 15-minute intervals from CSV options data"
    )

    # Add argument groups
    _add_input_args(parser)
    _add_filter_args(parser)
    _add_trading_args(parser)
    _add_output_args(parser)
    _add_advanced_args(parser)
    _add_rate_limit_args(parser)

    return parser.parse_args()
