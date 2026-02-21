"""
Command-line argument parsing helpers for options analyzer.
"""

import argparse
import sys
from typing import List


# Examples text for help
ARGUMENT_EXAMPLES = """
Examples:
  # Analyze all available tickers
  python options_analyzer.py --db-conn questdb://user:pass@host:8812/db

  # Analyze specific symbols with 14-day expiry window
  python options_analyzer.py --symbols AAPL MSFT GOOGL --days 14 --output csv

  # Analyze S&P 500 stocks sorted by daily premium
  python options_analyzer.py --types sp-500 --sort daily_premium --group-by ticker

  # Filter by volume and max days (options expiring within 30 days), save to file
  python options_analyzer.py --symbols AAPL MSFT --min-volume 1000 --max-days 30 --output results.csv

  # CSV with custom formatting
  python options_analyzer.py --symbols AAPL --output results.csv --csv-delimiter ";" --csv-quoting all

  # CSV with specific columns only
  python options_analyzer.py --symbols AAPL --output results.csv --csv-columns "ticker,current_price,strike_price,potential_premium,daily_premium"

  # Show only high-premium opportunities
  python options_analyzer.py --min-premium 5000 --sort potential_premium

  # Filter by expiration date range (show only options expiring in January 2024)
  python options_analyzer.py --start-date 2024-01-01 --end-date 2024-01-31
  
  # Show only options expiring within 30 days from today
  python options_analyzer.py --symbols AAPL --max-days 30
  
  # Show only options expiring between 7 and 30 days from today
  python options_analyzer.py --symbols AAPL --min-days 7 --max-days 30
  
  # Show only options expiring today or later (default behavior)
  python options_analyzer.py --symbols AAPL
  
  # Show options expiring from a specific date onwards
  python options_analyzer.py --start-date 2024-02-15
  
  # Show all options including those already expired
  python options_analyzer.py --start-date 2020-01-01
  
  # max-days overrides end-date (shows options expiring within 60 days, not through 2024-12-31)
  python options_analyzer.py --symbols AAPL --end-date 2024-12-31 --max-days 60
  
  # Filter options by write timestamp (only show options written after specified time in EST)
  python options_analyzer.py --symbols AAPL --min-write-timestamp "2025-11-05 10:00:00"
  
  # Use multiprocessing with 8 workers (automatically enabled when max-workers > 1)
  python options_analyzer.py --symbols AAPL --max-workers 8

  # Filter by P/E ratio and market cap (using B/M suffixes)
  python options_analyzer.py --filter "pe_ratio > 20" --filter "market_cap < 1B"

  # Filter with OR logic
  python options_analyzer.py --filter "pe_ratio > 30" --filter "market_cap > 5B" --filter-logic OR

  # Filter for options with volume data
  python options_analyzer.py --filter "volume exists" --filter "pe_ratio exists"

  # Market cap filtering with different formats
  python options_analyzer.py --filter "market_cap > 500M" --filter "market_cap < 3.5B"

  # Field-to-field comparisons
  python options_analyzer.py --filter "num_contracts > volume" --filter "potential_premium > daily_premium"

  # Mathematical expressions in filters
  python options_analyzer.py --filter "num_contracts*0.1 > volume" --filter "potential_premium+1000 > daily_premium"

  # Percentage fields (derived)
  # option_premium_percentage = (option_premium / current_price) * 100
  # premium_above_diff_percentage = ((option_premium - price_above_current) / price_above_current) * 100
  python options_analyzer.py --filter "option_premium_percentage >= 10" --filter "premium_above_diff_percentage > 0"

  # Calendar spread analysis (sell short-term, buy long-term)
  # Exact strike match, 90-day long options
  python options_analyzer.py --symbols AAPL --spread --max-days 30 --spread-long-days 90

  # Spread with 5% strike tolerance (allows ±5% difference in strikes)
  python options_analyzer.py --symbols AAPL MSFT --spread --spread-strike-tolerance 5.0 --spread-long-days 120

  # Spread with explicit min/max days range
  python options_analyzer.py --symbols AAPL --spread --spread-long-min-days 60 --spread-long-days 120

  # Spread with filters on net premium
  python options_analyzer.py --symbols AAPL --spread --filter "net_daily_premium > 100" --filter "net_premium > 1000"

  # Spread sorted by net daily premium
  python options_analyzer.py --symbols AAPL GOOGL --spread --sort net_daily_premium --max-days 14

  # Limit to top 5 options per ticker-option_type combination
  # If a ticker has both calls and puts, shows top 5 calls AND top 5 puts
  python options_analyzer.py --symbols AAPL MSFT GOOGL --top-n 5 --sort daily_premium

  # Spread mode with top 3 spreads per ticker-option_type combination
  python options_analyzer.py --symbols AAPL MSFT --spread --top-n 3 --sort net_daily_premium
"""


def add_database_arguments(parser: argparse.ArgumentParser) -> None:
    """Add database connection arguments."""
    parser.add_argument(
        '--db-conn',
        type=str,
        required=True,
        help="QuestDB connection string (e.g. questdb://user:pass@host:8812/db)."
    )


def add_analysis_arguments(parser: argparse.ArgumentParser) -> None:
    """Add analysis parameter arguments."""
    parser.add_argument(
        '--days',
        type=int,
        default=None,
        help="Number of days to expiry window (e.g. 14 for ±14 days around target). If not specified analyze all available expirations."
    )
    parser.add_argument(
        '--max-days',
        type=int,
        default=None,
        help="Maximum days from today for option expiration (convenience parameter that sets end-date to today + max-days, overrides --end-date if both are provided)."
    )
    parser.add_argument(
        '--min-days',
        type=int,
        default=None,
        help="Minimum days from today for option expiration (convenience parameter that sets start-date to today + min-days, overrides --start-date if both are provided)."
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=300,
        help="Number of tickers per batch when fetching options in multiprocessing mode (default: 300). Lower uses less memory."
    )
    parser.add_argument(
        '--start-date',
        type=str,
        default=None,
        help="Start date for option expiration filtering in YYYY-MM-DD format (defaults to today to show only options expiring today or later)."
    )
    parser.add_argument(
        '--end-date',
        type=str,
        default=None,
        help="End date for option expiration filtering in YYYY-MM-DD format (defaults to None for no upper bound, overridden by --max-days if both are provided)."
    )
    parser.add_argument(
        '--min-volume',
        type=int,
        default=0,
        help="Minimum volume filter for options (default: 0)."
    )
    parser.add_argument(
        '--min-premium',
        type=float,
        default=0.0,
        help="Minimum potential premium filter (default: 0.0)."
    )
    parser.add_argument(
        '--position-size',
        type=float,
        default=100000.0,
        help="Position size in dollars for premium calculations (default: 100000.0 = $100K)."
    )
    parser.add_argument(
        '--data-dir',
        default='data',
        help="Directory to store data files (default: data)."
    )
    parser.add_argument(
        '--no-market-time',
        action='store_true',
        help="Disable market-hours logic (gets latest stock price from any source regardless of market open/closed).",
    )
    parser.add_argument(
        '--no-cache',
        action='store_true',
        help="Disable Redis caching for QuestDB operations (default: cache enabled)"
    )
    parser.add_argument(
        '--min-write-timestamp',
        type=str,
        default=None,
        help="Minimum write timestamp for options in EST format (e.g., '2025-11-05 10:00:00'). Filters out options written before this time. Useful for getting only fresh options data.",
    )
    parser.add_argument(
        '--option-type',
        type=str,
        choices=['call', 'put', 'both'],
        default='call',
        help="Type of options to analyze: 'call' (default, covered calls), 'put' (cash-secured puts), or 'both' (analyze both calls and puts).",
    )
    parser.add_argument(
        '--sensible-price',
        type=float,
        default=0.01,
        help="Filter strikes based on current price as a percentage multiplier. For calls: only show strikes > current_price * (1 + sensible_price). For puts: only show strikes < current_price * (1 - sensible_price). Default: 0.01 (1%%). Set to 0 to disable. Example: 0.05 means 5%% above/below current price.",
    )
    parser.add_argument(
        '--max-bid-ask-spread',
        type=float,
        default=2.0,
        help="Maximum bid-ask spread as a ratio of bid price for short options. Formula: (ask - bid) / bid <= max_spread. Default: 2.0 (200%% spread). Set to 0 to disable. Example: 1.0 means ask can be at most 2x the bid.",
    )
    parser.add_argument(
        '--max-bid-ask-spread-long',
        type=float,
        default=2.0,
        help="Maximum bid-ask spread as a ratio of bid price for long options (spread mode only). Formula: (ask - bid) / bid <= max_spread. Default: 2.0 (200%% spread). Set to 0 to disable.",
    )


def add_spread_arguments(parser: argparse.ArgumentParser) -> None:
    """Add spread analysis arguments."""
    parser.add_argument(
        '--spread',
        action='store_true',
        help="Enable calendar spread analysis mode (sell short-term calls, buy long-term calls at similar strikes).",
    )
    parser.add_argument(
        '--spread-strike-tolerance',
        type=float,
        default=0.0,
        help="Percentage tolerance for matching strike prices in spread mode (e.g., 5.0 for ±5%%). Default: 0.0 (exact match).",
    )
    parser.add_argument(
        '--spread-long-days',
        type=int,
        default=90,
        help="Target days to expiry for long-term options to buy in spread mode (default: 90).",
    )
    parser.add_argument(
        '--spread-long-days-tolerance',
        type=int,
        default=14,
        help="Days tolerance for long option expiration window in spread mode (default: 14, searches ±14 days around target). Ignored if --spread-long-min-days is specified.",
    )
    parser.add_argument(
        '--spread-long-min-days',
        type=int,
        default=None,
        help="Minimum days to expiry for long options in spread mode. If set, searches from this min to --spread-long-days (ignores tolerance). Example: --spread-long-min-days 60 --spread-long-days 120 searches 60-120 day window.",
    )


def add_performance_arguments(parser: argparse.ArgumentParser) -> None:
    """Add performance tuning arguments."""
    parser.add_argument(
        '--timestamp-lookback-days',
        type=int,
        default=7,
        help="Number of days to look back for option timestamp data (default: 7). Lower values use less memory but may miss older data. Increase if you see missing options."
    )
    parser.add_argument(
        '--max-workers',
        type=int,
        default=4,
        help="Number of worker processes for multiprocessing (default: 4, typically set to CPU count). Multiprocessing is automatically enabled when max-workers > 1."
    )


def add_filter_arguments(parser: argparse.ArgumentParser) -> None:
    """Add filter arguments."""
    filter_help = (
        "Filter expressions (can be used multiple times). Format: 'field operator value' or 'field operator field' or 'field exists/not_exists' or 'field*multiplier operator value'. "
        "Supported operators: > >= < <= == != exists not_exists. "
        "Mathematical operations: + - * / (e.g. 'num_contracts*0.1 > volume' 'potential_premium+1000 > daily_premium'). "
        "Field-to-field comparisons: 'num_contracts > volume' or 'potential_premium > daily_premium'. "
        "Market cap values support T (trillion) B (billion) and M (million) suffixes (e.g. 'market_cap < 3.5T'). "
        "Multiple expressions in one --filter can be comma-separated. "
        "STANDARD FIELDS (always available): "
        "Financial: pe_ratio (float), market_cap (float, supports T/B/M suffixes). "
        "Pricing: current_price (float), strike_price (float), price_above_current (float), option_premium (float). "
        "Derived percentages: option_premium_percentage (float, = option_premium/current_price*100), "
        "premium_above_diff_percentage (float, = (option_premium-price_above_current)/price_above_current*100). "
        "Option Greeks: delta (float), theta (float), implied_volatility (float). "
        "Volume/Contracts: volume (int), num_contracts (int). "
        "Premium calculations: potential_premium (float, = num_contracts*option_premium*100), "
        "daily_premium (float, = potential_premium/days_to_expiry). "
        "Time: days_to_expiry (int). "
        "SPREAD MODE FIELDS (only when --spread is enabled): "
        "Long option details: long_strike_price (float), long_option_premium (float), long_days_to_expiry (int), "
        "long_delta (float), long_theta (float), long_implied_volatility (float), long_expiration_date (str), "
        "long_option_ticker (str), long_volume (int), long_contracts_available (int, open interest). "
        "Spread calculations: premium_diff (float, = long_premium - short_premium per contract), "
        "short_premium_total (float, = num_contracts*short_premium*100), "
        "short_daily_premium (float, = short_premium_total/short_days_to_expiry), "
        "long_premium_total (float, = num_contracts*long_premium*100), "
        "net_premium (float, = short_premium_total - (long_premium_total - estimated_long_premium_at_short_expiry_total), "
        "uses Black-Scholes to estimate long option value at short expiration), "
        "net_daily_premium (float, = net_premium/short_days_to_expiry). "
        "Note: In spread mode, num_contracts = floor(position_size / (long_premium * 100)). "
        "Examples: 'pe_ratio > 20', 'market_cap < 3.5T', 'num_contracts > volume', 'num_contracts*0.1 > volume', "
        "'potential_premium > daily_premium', 'volume exists', 'option_premium_percentage >= 10', "
        "'premium_above_diff_percentage > 0', 'net_daily_premium > 100', 'net_premium > 1000'."
    )
    
    parser.add_argument(
        '--filter',
        action='append',
        type=str,
        help=filter_help
    )
    parser.add_argument(
        '--filter-logic',
        choices=['AND', 'OR'],
        default='AND',
        help="Logic to combine multiple filter expressions (default: AND)."
    )


def add_output_arguments(parser: argparse.ArgumentParser) -> None:
    """Add output formatting arguments."""
    parser.add_argument(
        '--output',
        type=str,
        default='table',
        help="Output format: 'table' 'csv' or filename (e.g. 'results.csv')."
    )
    parser.add_argument(
        '--group-by',
        choices=['ticker', 'overall'],
        default='overall',
        help="Group results by ticker or show overall ranking (default: overall)."
    )
    parser.add_argument(
        '--sort',
        type=str,
        default='daily_premium',
        help=(
            "Sort by any displayed field (full or abbreviated header). "
            "Examples: daily_premium potential_premium ticker days_to_expiry option_premium_percentage premium_above_diff_percentage "
            "net_daily_premium net_premium premium_diff short_premium_total short_daily_premium long_premium_total long_strike_price long_days_to_expiry "
            "or abbreviations like TKR PRC STRK PREM%% DIFF%% POT_PREM DAY_PREM NET_DAY NET_PREM PREM_DIFF S_PREM_TOT S_DAY_PREM L_PREM_TOT L_STRK L_DAYS."
        )
    )
    parser.add_argument(
        '--log-level',
        type=str,
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        default='INFO',
        help="Set logging level: DEBUG (most verbose), INFO (default), WARNING, ERROR (least verbose)."
    )
    parser.add_argument(
        '--debug',
        action='store_true',
        help="Enable debug output with detailed information about data fetching and matching."
    )
    parser.add_argument(
        '--top-n',
        type=int,
        default=1,
        help="Limit results to top N options per ticker-option_type combination (based on sort order). Example: --top-n 10 shows the best 10 calls AND the best 10 puts for each ticker (if both types exist). Applied after sorting and filtering. Default: 1."
    )
    parser.add_argument(
        '--refresh-results',
        type=int,
        nargs='?',
        const=300,
        default=None,
        help="If market is open, refresh options data for tickers in results and re-analyze if the most recent write_timestamp is older than the specified threshold (in seconds). Default: 300 seconds. Requires POLYGON_API_KEY environment variable. Example: --refresh-results 300 or --refresh-results (uses default 300)."
    )
    parser.add_argument(
        '--refresh-results-background',
        type=int,
        nargs='?',
        const=300,
        default=None,
        help="Like --refresh-results, but runs refresh in a background process without waiting. Main process shows existing analysis results immediately. Requires POLYGON_API_KEY and Redis (for deduplication). Example: --refresh-results-background 300"
    )
    parser.add_argument(
        '--csv-delimiter',
        type=str,
        default=',',
        help="CSV delimiter character (default: ',')."
    )
    parser.add_argument(
        '--csv-quoting',
        choices=['minimal', 'all', 'none', 'nonnumeric'],
        default='minimal',
        help="CSV quoting style: minimal (default), all, none, nonnumeric."
    )
    parser.add_argument(
        '--csv-columns',
        type=str,
        help="Comma-separated list of columns to include in CSV output. If not specified, all columns are included."
    )
    parser.add_argument(
        '--stats',
        action='store_true',
        help="Display cache statistics at the end of the output (cache hit rate, etc.)."
    )


def log_parsed_arguments(args: argparse.Namespace) -> None:
    """Log parsed arguments for debugging."""
    if args.debug:
        print("DEBUG: Parsed arguments:", file=sys.stderr)
        print(f"  symbols: {getattr(args, 'symbols', None)}", file=sys.stderr)
        print(f"  types: {getattr(args, 'types', None)}", file=sys.stderr)
        print(f"  max_days: {args.max_days}", file=sys.stderr)
        print(f"  start_date: {args.start_date}", file=sys.stderr)
        print(f"  end_date: {args.end_date}", file=sys.stderr)
        print(f"  spread: {args.spread}", file=sys.stderr)
        print(f"  spread_strike_tolerance: {args.spread_strike_tolerance}", file=sys.stderr)
        print(f"  spread_long_days: {args.spread_long_days}", file=sys.stderr)
        print(f"  spread_long_min_days: {args.spread_long_min_days}", file=sys.stderr)
        print(f"  spread_long_days_tolerance: {args.spread_long_days_tolerance}", file=sys.stderr)
        print(f"  position_size: {args.position_size}", file=sys.stderr)
        print(f"  top_n: {args.top_n}", file=sys.stderr)
        print(f"  sort: {args.sort}", file=sys.stderr)
        print(f"  timestamp_lookback_days: {args.timestamp_lookback_days}", file=sys.stderr)
        print(f"  max_workers: {args.max_workers}", file=sys.stderr)


