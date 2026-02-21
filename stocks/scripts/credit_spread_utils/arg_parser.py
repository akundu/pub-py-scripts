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
    parser.add_argument(
        "--dynamic-spread-width",
        type=str,
        default=None,
        help="Enable dynamic spread width based on strike distance from previous close. "
             "Can be a JSON string or path to JSON config file. "
             "Example: '{\"mode\": \"linear\", \"base_width\": 20, \"slope_factor\": 1000}' "
             "Modes: 'linear' (width = base + distance_pct * slope), "
             "'stepped' (lookup table with thresholds), 'formula' (custom expression). "
             "When enabled, max-spread-width becomes the ceiling. "
             "Wider spreads for further OTM positions capture more premium at safer distances."
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
        choices=["net_pnl", "profit_factor", "win_rate", "total_trades", "annualized_roi"],
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
    parser.add_argument(
        "--rate-limit-blocks",
        type=str,
        default=None,
        help="Time-block based rate limiting with different limits per time block. "
             "Format: 'HH:MM-HH:MM=N,HH:MM-HH:MM=N,...' where N is max transactions per block. "
             "Times are in ET (Eastern Time). Transactions are evenly spaced within each block. "
             "Example: '09:30-10:00=3,10:00-11:00=4,11:00-12:00=2' means 3 trades in first 30min, "
             "4 trades in the next hour, 2 trades in the following hour. "
             "Takes precedence over --rate-limit-max/--rate-limit-window if both are specified."
    )


def _add_scale_in_args(parser: argparse.ArgumentParser):
    """Add scale-in strategy arguments."""
    parser.add_argument(
        "--scale-in-config",
        type=str,
        default=None,
        help="Path to JSON config file for scale-in on breach strategy. "
             "The config defines layered entry levels that trigger when previous layers are breached. "
             "Example: scale_in_config_ndx.json"
    )
    parser.add_argument(
        "--scale-in-enabled",
        action="store_true",
        help="Enable scale-in on breach strategy. Requires --scale-in-config. "
             "When enabled, positions are entered in layers: L1 at entry, L2 when L1 breached, etc. "
             "This strategy reduces average losses by 23-68%% compared to single entry."
    )
    parser.add_argument(
        "--scale-in-summary-only",
        action="store_true",
        help="When using scale-in strategy, only show aggregate summary statistics "
             "instead of individual layer details."
    )


def _add_tiered_args(parser: argparse.ArgumentParser):
    """Add tiered investment strategy arguments."""
    parser.add_argument(
        "--tiered-config",
        type=str,
        default=None,
        help="Path to JSON config file for tiered investment strategy. "
             "The config defines multiple concurrent tiers with per-tier contract count (N) "
             "and spread width (M). Example: tiered_config_ndx.json"
    )
    parser.add_argument(
        "--tiered-enabled",
        action="store_true",
        help="Enable tiered investment strategy. Requires --tiered-config. "
             "When enabled, multiple concurrent positions enter at different distances "
             "from close, each with its own N contracts and M width. "
             "Tiers activate when framework constraints are satisfied."
    )
    parser.add_argument(
        "--tiered-summary-only",
        action="store_true",
        help="When using tiered strategy, only show aggregate summary statistics "
             "instead of individual trade details."
    )


def _add_delta_filter_args(parser: argparse.ArgumentParser):
    """Add delta-based filtering arguments."""
    parser.add_argument(
        "--max-short-delta",
        type=float,
        default=None,
        help="Maximum absolute delta for short leg (e.g., 0.15 = 15 delta). "
             "Filters out spreads where short leg delta exceeds this value. "
             "Lower delta = further OTM = lower probability of being ITM at expiration."
    )
    parser.add_argument(
        "--min-short-delta",
        type=float,
        default=None,
        help="Minimum absolute delta for short leg (e.g., 0.05 = 5 delta). "
             "Filters out spreads where short leg delta is below this value. "
             "Ensures spreads have sufficient premium (very low delta = low premium)."
    )
    parser.add_argument(
        "--max-long-delta",
        type=float,
        default=None,
        help="Maximum absolute delta for long leg. Rarely needed since long leg "
             "is typically further OTM than short leg."
    )
    parser.add_argument(
        "--min-long-delta",
        type=float,
        default=None,
        help="Minimum absolute delta for long leg. Rarely needed."
    )
    parser.add_argument(
        "--delta-range",
        type=str,
        default=None,
        help="Shorthand for min/max short delta range. Format: 'MIN-MAX' (e.g., '0.05-0.20' for 5-20 delta). "
             "Can also specify just max: '0.15' means max 15 delta. "
             "Overrides --min-short-delta and --max-short-delta if both are specified."
    )
    parser.add_argument(
        "--require-delta",
        action="store_true",
        help="Skip spreads where delta cannot be determined. By default, spreads "
             "with unknown delta are included. Use this to require delta calculation."
    )
    parser.add_argument(
        "--delta-default-iv",
        type=float,
        default=0.20,
        help="Default IV for Black-Scholes delta calculation when option IV is unavailable "
             "and VIX1D is not used. Default: 0.20 (20%%). "
             "This is the fallback IV used for delta estimation."
    )
    parser.add_argument(
        "--vix1d-dir",
        type=str,
        default="../equities_output/I:VIX1D",
        help="Directory containing VIX1D CSV files for IV lookup. "
             "Files should be named I:VIX1D_equities_YYYY-MM-DD.csv. "
             "Default: ../equities_output/I:VIX1D"
    )
    parser.add_argument(
        "--use-vix1d",
        action="store_true",
        help="Use VIX1D for IV in Black-Scholes delta calculation instead of default IV. "
             "VIX1D provides 1-day expected volatility which is more accurate for 0DTE options. "
             "Requires VIX1D data files in --vix1d-dir."
    )


def parse_args() -> argparse.Namespace:
    """Parse command line arguments.

    Returns:
        Parsed arguments namespace
    """
    parser = argparse.ArgumentParser(
        description="Analyze credit spreads at 15-minute intervals from CSV options data",
        epilog="""
================================================================================
DAILY WORKFLOW COMMANDS
================================================================================

These are the primary commands for daily trading analysis. Run from the project
root directory (stocks/).

--- BACKTESTING (historical analysis) ---

1. Standard backtest - NDX, last 3 months:

   python scripts/analyze_credit_spread_intervals.py \\
       --csv-dir options_csv_output --ticker NDX \\
       --start-date 2025-11-01 --end-date 2026-02-07 \\
       --percent-beyond 0.005:0.015 --max-spread-width 20:30 \\
       --risk-cap 500000 --profit-target-pct 0.80 \\
       --min-trading-hour 6 --max-trading-hour 12 \\
       --output-timezone America/Los_Angeles --summary

2. Backtest with dynamic spread widths (Linear-500 config):

   python scripts/analyze_credit_spread_intervals.py \\
       --csv-dir options_csv_output --ticker NDX \\
       --start-date 2025-11-01 \\
       --percent-beyond 0.005:0.015 --max-spread-width 50 \\
       --dynamic-spread-width '{"mode":"linear","base_width":15,"slope_factor":500,"min_width":10,"max_width":50}' \\
       --risk-cap 500000 --profit-target-pct 0.80 --summary

3. Backtest with scale-in strategy:

   python scripts/analyze_credit_spread_intervals.py \\
       --csv-dir options_csv_output --ticker NDX \\
       --start-date 2025-11-01 \\
       --percent-beyond 0.025:0.026 --max-spread-width 25 \\
       --scale-in-enabled --scale-in-config scripts/scale_in_config_ndx.json \\
       --risk-cap 500000 --summary

4. Backtest with tiered investment strategy:

   python scripts/analyze_credit_spread_intervals.py \\
       --csv-dir options_csv_output --ticker NDX \\
       --start-date 2025-11-01 \\
       --percent-beyond 0.02 --max-spread-width 50 \\
       --tiered-enabled --tiered-config scripts/tiered_config_ndx.json \\
       --summary

5. Backtest with delta filtering (VIX1D-based IV):

   python scripts/analyze_credit_spread_intervals.py \\
       --csv-dir options_csv_output --ticker NDX \\
       --start-date 2025-11-01 \\
       --percent-beyond 0.005 --max-spread-width 50 \\
       --max-short-delta 0.15 --use-vix1d \\
       --option-type put --summary

--- GRID SEARCH (parameter optimization) ---

6. Multi-timeframe grid search (run each timeframe separately):

   # 1-week
   python scripts/analyze_credit_spread_intervals.py \\
       --grid-config scripts/grid_config_ndx_1wk_100pct.json \\
       --grid-output scripts/ndx_1wk_100pct_results.csv \\
       --grid-sort net_pnl --grid-top-n 20 --log-level WARNING

   # 1-month
   python scripts/analyze_credit_spread_intervals.py \\
       --grid-config scripts/grid_config_ndx_1mo_100pct.json \\
       --grid-output scripts/ndx_1mo_100pct_results.csv \\
       --grid-sort net_pnl --grid-top-n 20 --log-level WARNING

   # 3-month, 6-month: same pattern with different grid configs

7. Delta grid search (NDX puts, delta 0.01-0.20):

   python scripts/analyze_credit_spread_intervals.py \\
       --grid-config scripts/ndx_optimal_puts_config.json \\
       --grid-output scripts/ndx_delta_puts_results.csv \\
       --grid-sort net_pnl --grid-top-n 20 --log-level WARNING --no-data-cache

8. Resume an interrupted grid search:

   python scripts/analyze_credit_spread_intervals.py \\
       --grid-config scripts/grid_config_ndx_1mo_100pct.json \\
       --grid-output scripts/ndx_1mo_100pct_results.csv \\
       --grid-resume --grid-sort net_pnl --grid-top-n 20

9. Grid search with strategy framework:

   python scripts/analyze_credit_spread_intervals.py \\
       --grid-config scripts/grid_config_with_strategy.json \\
       --grid-output scripts/strategy_grid_results.csv \\
       --grid-sort net_pnl --grid-top-n 20

   Grid config with strategy section:
   {
     "strategy": {"name": "tiered", "config_file": "tiered_config_ndx.json"},
     "fixed_params": { ... },
     "grid_params": {
       "strategy.feature_flags.greedy_t3_first": [true, false],
       "percent_beyond": [0.015, 0.020, 0.025]
     }
   }

--- CONTINUOUS MODE (live monitoring) ---

10. Live monitoring during market hours:

   python scripts/analyze_credit_spread_intervals.py \\
       --csv-dir options_csv_output --ticker NDX \\
       --percent-beyond 0.005:0.015 --max-spread-width 20:30 \\
       --risk-cap 500000 --profit-target-pct 0.80 \\
       --continuous 10 --most-recent --best-only \\
       --curr-price --use-market-hours \\
       --output-timezone America/Los_Angeles

11. Continuous with market hours and auto-stop:

   python scripts/analyze_credit_spread_intervals.py \\
       --csv-dir options_csv_output --ticker NDX \\
       --percent-beyond 0.005:0.015 --max-spread-width 20:30 \\
       --continuous 15 --most-recent --best-only \\
       --curr-price --use-market-hours \\
       --run-once-before-wait --continuous-max-runs 100

--- SUPPORTING SCRIPTS (separate from main analyzer) ---

12. Price movement analysis (close-to-close, intraday extremes):
   python scripts/fetch_index_prices.py --ticker NDX --period 6mo

13. Risk gradient analysis:
   python scripts/ndx_risk_gradient_analysis.py

================================================================================
GRID CONFIG FILE FORMAT
================================================================================

{
  "fixed_params": {
    "csv_dir": "options_csv_output",
    "underlying_ticker": "NDX",
    "percent_beyond": "0.005",
    "max_spread_width": "50",
    "risk_cap": 5000,
    "max_credit_width_ratio": 0.60,
    "max_trading_hour": 15,
    "min_trading_hour": 7,
    "option_type": "put",
    "use_vix1d": true,
    "vix1d_dir": "equities_output/I:VIX1D",
    "start_date": "2025-11-05",
    "end_date": "2026-02-02"
  },
  "grid_params": {
    "max_short_delta": [0.01, 0.02, 0.05, 0.10, 0.15, 0.20]
  }
}

For both puts and calls, move option_type to grid_params:
  "grid_params": {
    "option_type": ["put", "call"],
    "max_short_delta": [0.01, 0.02, ..., 0.20]
  }

OUTPUT CSV COLUMNS:
  rank, max_short_delta, total_trades, win_rate, total_credits,
  total_gains, total_losses, net_pnl, profit_factor, roi

================================================================================
ARCHITECTURE: See docs/CREDIT_SPREAD_STRATEGIES.md for full architecture,
module map, and strategy framework documentation.
================================================================================
""",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    # Add argument groups
    _add_mode_args(parser)
    _add_input_args(parser)
    _add_filter_args(parser)
    _add_trading_args(parser)
    _add_output_args(parser)
    _add_advanced_args(parser)
    _add_rate_limit_args(parser)
    _add_scale_in_args(parser)
    _add_tiered_args(parser)
    _add_delta_filter_args(parser)
    _add_strategy_args(parser)
    _add_close_predictor_args(parser)
    _add_dte_comparison_args(parser)
    _add_price_movement_args(parser)
    _add_max_move_args(parser)
    _add_risk_gradient_args(parser)

    return parser.parse_args()


def _add_mode_args(parser: argparse.ArgumentParser):
    """Add analysis mode selection."""
    parser.add_argument(
        '--mode',
        choices=['credit-spread', 'price-movements', 'max-move', 'risk-gradient', 'dte-comparison'],
        default='credit-spread',
        help='Analysis mode. credit-spread (default): existing credit spread analysis. '
             'price-movements: close-to-close or time-to-close price movement statistics. '
             'max-move: intraday extreme movement tables by 30-min time slots. '
             'risk-gradient: risk gradient analysis from historical safe points. '
             'dte-comparison: compare credit spreads across DTE buckets (0, 3, 5, 10 DTE).'
    )


def _add_close_predictor_args(parser: argparse.ArgumentParser):
    """Add close predictor risk gate arguments."""
    group = parser.add_argument_group(
        'Close Predictor Gate',
        'Use the unified close predictor as a risk gate to filter unsafe spreads'
    )
    group.add_argument(
        '--close-predictor',
        action='store_true',
        help='Enable close prediction risk gate. Uses the unified close predictor '
             'to check whether the predicted close band suggests the short strike is safe.'
    )
    group.add_argument(
        '--close-predictor-level',
        type=str,
        default='P95',
        choices=['P95', 'P98', 'P99', 'P100'],
        help='Band level to check (default: P95). Higher levels are wider/safer.'
    )
    group.add_argument(
        '--close-predictor-buffer',
        type=str,
        default='0',
        help='Minimum buffer between band edge and short strike. '
             'Accepts points (e.g., 50) or percentage (e.g., 0.5%%). '
             'Default: 0 (no buffer).'
    )
    group.add_argument(
        '--close-predictor-mode',
        type=str,
        default='gate',
        choices=['gate', 'annotate'],
        help='gate = skip unsafe spreads (default), annotate = add info but don\'t filter.'
    )
    group.add_argument(
        '--close-predictor-lookback',
        type=int,
        default=250,
        help='Training lookback in trading days for the close predictor models (default: 250).'
    )


def _add_price_movement_args(parser: argparse.ArgumentParser):
    """Add price movement analysis arguments (used with --mode price-movements)."""
    group = parser.add_argument_group(
        'Price Movement Mode',
        'Arguments for --mode price-movements (close-to-close or time-to-close analysis)'
    )
    group.add_argument(
        '--data-dir',
        default='equities_output',
        help='Directory containing equity data (default: equities_output). '
             'Used with --mode price-movements.'
    )
    group.add_argument(
        '--from-time',
        type=str,
        default=None,
        help='Time of day to measure FROM (HH:MM). If not provided, uses prior day close. '
             'Used with --mode price-movements.'
    )
    group.add_argument(
        '--pm-timezone',
        type=str,
        default='PST',
        choices=['PST', 'PDT', 'EST', 'EDT', 'UTC'],
        help='Timezone for --from-time input (default: PST). '
             'Named --pm-timezone to avoid conflict with --output-timezone. '
             'Used with --mode price-movements.'
    )
    group.add_argument(
        '--no-plot',
        action='store_true',
        help='Skip histogram generation, print stats only. '
             'Used with --mode price-movements.'
    )
    group.add_argument(
        '--plot-output',
        type=str,
        default='price_movements.png',
        help='Output path for histogram image (default: price_movements.png). '
             'Used with --mode price-movements.'
    )
    group.add_argument(
        '--day-direction',
        type=str,
        default=None,
        choices=['up', 'down'],
        help='Filter to only days that closed up or down vs prior day. '
             'Used with --mode price-movements.'
    )


def _add_max_move_args(parser: argparse.ArgumentParser):
    """Add max move analysis arguments (used with --mode max-move)."""
    group = parser.add_argument_group(
        'Max Move Mode',
        'Arguments for --mode max-move (intraday extreme movement tables)'
    )
    group.add_argument(
        '--days',
        type=int,
        default=125,
        help='Number of trading days to analyze (default: 125, ~6 months). '
             'Used with --mode max-move.'
    )


def _add_risk_gradient_args(parser: argparse.ArgumentParser):
    """Add risk gradient analysis arguments (used with --mode risk-gradient)."""
    group = parser.add_argument_group(
        'Risk Gradient Mode',
        'Arguments for --mode risk-gradient (risk gradient from historical safe points)'
    )
    group.add_argument(
        '--lookback-days',
        type=int,
        nargs='+',
        default=[90, 180],
        help='Lookback periods in days (default: 90 180). '
             'Used with --mode risk-gradient.'
    )
    group.add_argument(
        '--gradient-steps',
        type=int,
        default=7,
        help='Number of gradient steps from safe point (default: 7). '
             'Used with --mode risk-gradient.'
    )
    group.add_argument(
        '--step-size',
        type=float,
        default=0.0025,
        help='Step size in decimal (0.0025 = 0.25%%) (default: 0.0025). '
             'Used with --mode risk-gradient.'
    )
    group.add_argument(
        '--generate-config-only',
        action='store_true',
        help='Only generate config files, do not run backtest. '
             'Used with --mode risk-gradient.'
    )
    group.add_argument(
        '--run-backtest',
        action='store_true',
        help='Run backtest after generating configs. '
             'Used with --mode risk-gradient.'
    )
    group.add_argument(
        '--detailed-output',
        action='store_true',
        help='Enable detailed output with hourly and 10-minute block breakdowns. '
             'Used with --mode risk-gradient.'
    )
    group.add_argument(
        '--time-periods',
        type=str,
        nargs='+',
        default=['3mo', '1mo', 'week1', 'week2', 'week3', 'week4'],
        help='Time periods to analyze (default: 3mo 1mo week1 week2 week3 week4). '
             'Used with --mode risk-gradient.'
    )
    group.add_argument(
        '--time-analysis-config',
        type=str,
        default=None,
        help='Path to a config file to run time-of-day analysis. '
             'Used with --mode risk-gradient.'
    )
    group.add_argument(
        '--output-dir',
        default=None,
        help='Output directory for configs and results (default: scripts/). '
             'Used with --mode risk-gradient.'
    )


def _add_dte_comparison_args(parser: argparse.ArgumentParser):
    """Add DTE comparison analysis arguments (used with --mode dte-comparison)."""
    group = parser.add_argument_group(
        'DTE Comparison Mode',
        'Arguments for --mode dte-comparison (compare credit spreads across DTE buckets)'
    )
    group.add_argument(
        '--dte-buckets',
        type=str,
        default='0,3,5,10',
        help='Comma-separated DTE buckets to compare (default: 0,3,5,10). '
             'Used with --mode dte-comparison.'
    )
    group.add_argument(
        '--dte-tolerance',
        type=int,
        default=1,
        help='Calendar days tolerance for DTE bucket matching (default: 1). '
             'DTE=4 maps to bucket 3 or 5 depending on which is closer. '
             'Used with --mode dte-comparison.'
    )
    group.add_argument(
        '--multi-dte-dir',
        type=str,
        default='options_csv_output_full',
        help='Directory with multi-day options CSVs for >0DTE data '
             '(default: options_csv_output_full). '
             'Used with --mode dte-comparison.'
    )
    group.add_argument(
        '--zero-dte-dir',
        type=str,
        default='options_csv_output',
        help='Directory with 0DTE options CSVs (default: options_csv_output). '
             'When provided, 0DTE rows are loaded exclusively from this directory '
             'and >0DTE rows come from --multi-dte-dir. '
             'Used with --mode dte-comparison.'
    )
    group.add_argument(
        '--min-volume',
        type=int,
        default=5,
        help='Minimum volume for option legs (default: 5). '
             'Filters out illiquid contracts. '
             'Used with --mode dte-comparison.'
    )
    group.add_argument(
        '--hold-max-days',
        type=int,
        default=None,
        help='Max days to hold underwater position before forced exit (default: same as DTE). '
             'Used with --mode dte-comparison.'
    )
    group.add_argument(
        '--exit-profit-pcts',
        type=str,
        default='50,60,70,80,90',
        help='Comma-separated profit target percentages to test (default: 50,60,70,80,90). '
             'Grid search sweeps these to find optimal exit %%. '
             'Used with --mode dte-comparison.'
    )
    group.add_argument(
        '--percent-beyond-percentile',
        type=str,
        default=None,
        help='Adaptive percent_beyond from historical price move percentiles. '
             'Comma-separated percentile values (e.g., "90,95,99"). '
             'Computes N-day return distributions and uses the Nth percentile as '
             'percent_beyond, scaled by DTE. Overrides --percent-beyond. '
             'Used with --mode dte-comparison.'
    )
    group.add_argument(
        '--percentile-lookback',
        type=int,
        default=180,
        help='Lookback days for percentile computation (default: 180). '
             'Used with --percent-beyond-percentile. '
             'Used with --mode dte-comparison.'
    )
    group.add_argument(
        '--exit-dte',
        type=str,
        default=None,
        help='DTE-based exit: close position when N calendar days remain before expiration. '
             'Comma-separated values (e.g., "1,2,3"). '
             'Used with --mode dte-comparison --two-phase.'
    )
    group.add_argument(
        '--min-vix1d',
        type=float,
        default=None,
        help='Minimum VIX1D to enter a trade (filters low-vol days). '
             'Only enter trades when VIX1D >= this threshold. '
             'Used with --mode dte-comparison --two-phase.'
    )
    group.add_argument(
        '--max-vix1d',
        type=float,
        default=None,
        help='Maximum VIX1D to enter a trade (filters extreme-vol days). '
             'Only enter trades when VIX1D <= this threshold. '
             'Used with --mode dte-comparison --two-phase.'
    )
    group.add_argument(
        '--two-phase',
        action='store_true',
        help='Enable two-phase analysis with Phase A (raw trade building with caching) '
             'and Phase B (fast exit strategy evaluation). '
             'Phase A holds everything to expiration with daily mark-to-market. '
             'Phase B applies exit rules (profit target, DTE exit, IV filter) post-hoc. '
             'Used with --mode dte-comparison.'
    )
    group.add_argument(
        '--num-processes',
        type=int,
        default=0,
        help='Number of parallel processes for Phase A. '
             '0 = auto (cpu_count). 1 = single-process (no fork overhead). '
             'Used with --mode dte-comparison --two-phase.'
    )
    group.add_argument(
        '--strike-selection',
        type=str,
        choices=['max_credit', 'boundary'],
        default='max_credit',
        help='Strike selection mode. max_credit (default): select spread with highest credit '
             '(closest to the money). boundary: select spread whose short strike is closest '
             'to the percentile boundary target price. '
             'Used with --mode dte-comparison --two-phase.'
    )
    group.add_argument(
        '--iron-condor',
        action='store_true',
        help='Build iron condors (combined put + call spread). '
             'When enabled, forces --option-type both implicitly. '
             'Collects premium from both sides simultaneously. '
             'Used with --mode dte-comparison --two-phase.'
    )
    group.add_argument(
        '--stop-loss-multiple',
        type=float,
        default=None,
        help='Stop-loss at N× credit collected (e.g., 2.0 = exit when loss >= 2x credit). '
             'Checked before profit target in Phase B evaluation. '
             'Used with --mode dte-comparison --two-phase.'
    )
    group.add_argument(
        '--flow-filter',
        type=str,
        choices=['with', 'against'],
        default=None,
        help='Only trade with-flow or against-flow directions. '
             'with = put spread on up days, call spread on down days. '
             'against = opposite. None (default) = trade both. '
             'Applied as Phase B filter. '
             'Used with --mode dte-comparison --two-phase.'
    )


def _add_strategy_args(parser):
    """Add strategy framework arguments."""
    group = parser.add_argument_group('Strategy Framework',
                                     'Select and configure trading strategies')
    group.add_argument(
        '--strategy',
        choices=['single_entry', 'scale_in', 'tiered', 'time_allocated_tiered'],
        default=None,
        help='Strategy to use. When specified, uses the strategy framework. '
             'Default behavior (None) uses legacy code paths. '
             'single_entry=default best-spread-per-interval, '
             'scale_in=layered entry on breach, '
             'tiered=multi-tier position management.'
    )
    group.add_argument(
        '--strategy-config',
        type=str,
        default=None,
        help='Path to JSON strategy configuration file with feature flags. '
             'Format: {"strategy": "tiered", "enabled": true, '
             '"feature_flags": {"greedy_t3_first": true}, '
             '"config_file": "tiered_config_ndx.json"}'
    )
