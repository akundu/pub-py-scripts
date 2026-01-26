#!/usr/bin/env python3
"""
Multi-Timeframe Credit Spread Strategy Recommender.

Analyzes credit spread performance across multiple time windows (1yr, 6mo, 3mo, 1mo, 1wk),
incorporates VIX regime analysis, and produces actionable recommendations.

Usage:
    python scripts/strategy_recommender.py --ticker NDX \\
        --max-live-capital 300000 \\
        --risk-cap 50000 \\
        --output-timezone PDT

Output:
    - Console report with parameter recommendations and confidence scores
    - Optional JSON export to {ticker}_recommendation.json
"""

import argparse
import asyncio
import json
import os
import sys
import tempfile
from datetime import date, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

# Add scripts dir to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import local modules
from analyze_credit_spread_intervals import run_grid_search
from strategy_utils.time_windows import (
    WINDOW_DEFINITIONS,
    get_time_window_paths,
    get_cached_results_path,
    results_exist_and_recent,
    get_weights_for_regime,
    get_window_label,
)
from strategy_utils.convergence import (
    TRACKED_PARAMS,
    analyze_convergence,
    get_convergence_summary,
    detect_trend,
)
from strategy_utils.vix_regime import (
    get_vix_regime,
    get_mock_vix_regime,
)
from strategy_utils.report import (
    generate_console_report,
    generate_json_report,
    save_json_report,
)

# Add common path for QuestDB
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common.questdb_db import StockQuestDB
from common.logging_utils import get_logger


# Standard grid parameters for all timeframe analyses
STANDARD_GRID = {
    "option_type": ["both"],
    "percent_beyond_put": [0.005, 0.0075, 0.01, 0.0125],
    "percent_beyond_call": [0.015, 0.02, 0.025],
    "max_spread_width_put": [10, 20, 30],
    "max_spread_width_call": [10, 20, 30],
    "min_trading_hour": [6, 7, 8, 9, 10, 11, 12],
    "max_trading_hour": [7, 8, 9, 10, 11, 12, 13],
}


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Multi-Timeframe Credit Spread Strategy Recommender",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Basic usage with NDX
    python strategy_recommender.py --ticker NDX --max-live-capital 300000 --risk-cap 50000

    # Use cached results only (no auto-run)
    python strategy_recommender.py --ticker SPX --no-auto-run

    # Use existing result files with custom names
    python strategy_recommender.py --ticker NDX --no-auto-run --windows 3mo 1wk \\
        --results-map "3mo=ndx_hourly_90d_pdt_results.csv" "1wk=ndx_hourly_pdt_results.csv"

    # Export to JSON
    python strategy_recommender.py --ticker NDX --json-output ndx_recommendation.json

    # Dry run to see what would be analyzed
    python strategy_recommender.py --ticker NDX --dry-run
""",
    )

    # Required arguments
    parser.add_argument(
        "--ticker",
        required=True,
        help="Underlying ticker symbol (e.g., NDX, SPX)",
    )

    # Capital parameters
    parser.add_argument(
        "--max-live-capital",
        type=float,
        default=300000,
        help="Maximum concurrent capital exposure (default: 300000)",
    )
    parser.add_argument(
        "--risk-cap",
        type=float,
        default=50000,
        help="Maximum risk per single trade (default: 50000)",
    )

    # Time window options
    parser.add_argument(
        "--windows",
        nargs="+",
        default=["1yr", "6mo", "3mo", "1mo", "1wk"],
        help="Time windows to analyze (default: 1yr 6mo 3mo 1mo 1wk)",
    )
    parser.add_argument(
        "--no-auto-run",
        action="store_true",
        help="Don't auto-run grid searches for missing timeframes",
    )
    parser.add_argument(
        "--force-refresh",
        action="store_true",
        help="Force re-run of grid searches even if recent results exist",
    )

    # Paths
    parser.add_argument(
        "--csv-dir",
        default="./options_csv_output",
        help="Directory containing options CSV data (default: ./options_csv_output)",
    )
    parser.add_argument(
        "--results-dir",
        default=".",
        help="Directory for grid search results CSVs (default: current dir)",
    )
    parser.add_argument(
        "--cache-dir",
        default=".options_cache",
        help="Binary cache directory (default: .options_cache)",
    )

    # Database
    parser.add_argument(
        "--db-path",
        default=None,
        help="QuestDB connection string (default: $QUEST_DB_STRING env var)",
    )

    # Output options
    parser.add_argument(
        "--output-timezone",
        default="PDT",
        help="Output timezone for trading hours (default: PDT)",
    )
    parser.add_argument(
        "--json-output",
        default=None,
        help="Path to save JSON report (default: {ticker}_recommendation.json)",
    )
    parser.add_argument(
        "--no-json",
        action="store_true",
        help="Don't save JSON output",
    )

    # Analysis options
    parser.add_argument(
        "--top-n",
        type=int,
        default=10,
        help="Number of top results to consider per window (default: 10)",
    )
    parser.add_argument(
        "--profit-target-pct",
        type=float,
        default=80,
        help="Profit target percentage (default: 80)",
    )
    parser.add_argument(
        "--min-contract-price",
        type=float,
        default=0.5,
        help="Minimum option contract price (default: 0.5)",
    )

    # Execution options
    parser.add_argument(
        "--processes",
        type=int,
        default=0,
        help="Number of parallel processes for grid search (0=auto)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be analyzed without running",
    )
    parser.add_argument(
        "--no-vix",
        action="store_true",
        help="Skip VIX regime analysis (use default weights)",
    )

    # Logging
    parser.add_argument(
        "--log-level",
        default="WARNING",
        help="Log level (default: WARNING)",
    )

    # Existing results mapping
    parser.add_argument(
        "--results-map",
        nargs="+",
        metavar="WINDOW=PATH",
        help="Map windows to existing results files (e.g., 3mo=ndx_hourly_90d_pdt_results.csv)",
    )

    return parser.parse_args()


def load_results_csv(path: Path, logger) -> Optional[pd.DataFrame]:
    """Load results CSV and return as DataFrame."""
    if not path.exists():
        logger.debug(f"Results file not found: {path}")
        return None

    try:
        df = pd.read_csv(path)
        if df.empty:
            logger.warning(f"Results file is empty: {path}")
            return None
        logger.info(f"Loaded {len(df)} results from {path}")
        return df
    except Exception as e:
        logger.warning(f"Failed to load results from {path}: {e}")
        return None


def create_grid_config_for_window(
    ticker: str,
    window: str,
    csv_paths: List[str],
    args,
) -> dict:
    """Create grid search config for a specific time window."""
    db_path = args.db_path
    if db_path is None:
        db_path = "$QUEST_DB_STRING"

    return {
        "fixed_params": {
            "csv_path": csv_paths,
            "db_path": db_path,
            "underlying_ticker": ticker,
            "risk_cap": args.risk_cap,
            "max_live_capital": args.max_live_capital,
            "top_n": 5,
            "min_contract_price": args.min_contract_price,
            "output_timezone": args.output_timezone,
            "profit_target_pct": args.profit_target_pct,
            "no_cache": False,
        },
        "grid_params": STANDARD_GRID,
    }


async def run_grid_search_for_window(
    ticker: str,
    window: str,
    csv_paths: List[str],
    output_path: Path,
    args,
    logger,
) -> bool:
    """Run grid search for a specific time window."""
    logger.info(f"Running grid search for {ticker} {window} window...")

    # Create config
    config = create_grid_config_for_window(ticker, window, csv_paths, args)

    # Write temp config file
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".json", delete=False
    ) as f:
        json.dump(config, f)
        config_path = f.name

    try:
        # Create args for run_grid_search
        class GridArgs:
            pass

        grid_args = GridArgs()
        grid_args.grid_config = config_path
        grid_args.grid_output = str(output_path)
        grid_args.grid_top_n = 20
        grid_args.grid_sort = "net_pnl"
        grid_args.grid_dry_run = False
        grid_args.grid_resume = False
        grid_args.cache_dir = args.cache_dir
        grid_args.no_data_cache = False
        grid_args.log_level = args.log_level
        grid_args.processes = args.processes

        # Run grid search
        result = await run_grid_search(grid_args)
        return result == 0

    finally:
        # Clean up temp config
        os.unlink(config_path)


def parse_results_map(results_map: Optional[List[str]]) -> Dict[str, str]:
    """Parse results-map argument into a dictionary."""
    if not results_map:
        return {}

    mapping = {}
    for item in results_map:
        if "=" in item:
            window, path = item.split("=", 1)
            mapping[window.strip()] = path.strip()
    return mapping


async def gather_results_for_windows(
    ticker: str,
    windows: List[str],
    csv_dir: str,
    results_dir: str,
    args,
    logger,
) -> Dict[str, pd.DataFrame]:
    """
    Gather grid search results for all windows.
    Runs grid searches for missing windows if auto-run is enabled.
    """
    results = {}
    results_map = parse_results_map(getattr(args, "results_map", None))

    for window in windows:
        # Check for explicit mapping first
        if window in results_map:
            explicit_path = Path(results_map[window])
            df = load_results_csv(explicit_path, logger)
            if df is not None:
                results[window] = df
                logger.info(f"{window}: Loaded from explicit mapping {explicit_path}")
                continue
            else:
                logger.warning(f"{window}: Could not load from mapped path {explicit_path}")

        results_path = get_cached_results_path(ticker, window, results_dir)

        # Check if we need to run grid search
        need_run = False

        if args.force_refresh:
            need_run = True
            logger.info(f"{window}: Force refresh requested")
        elif not results_path.exists():
            need_run = True
            logger.info(f"{window}: No existing results found")
        elif not results_exist_and_recent(ticker, window, results_dir, max_age_hours=168):  # 1 week
            need_run = True
            logger.info(f"{window}: Results are stale")

        if need_run and not args.no_auto_run:
            # Get CSV paths for this window
            csv_paths = get_time_window_paths(ticker, window, csv_dir)

            if not csv_paths:
                logger.warning(f"{window}: No CSV files found, skipping")
                continue

            logger.info(f"{window}: Found {len(csv_paths)} CSV files")

            if not args.dry_run:
                success = await run_grid_search_for_window(
                    ticker, window, csv_paths, results_path, args, logger
                )
                if not success:
                    logger.warning(f"{window}: Grid search failed")
                    continue

        # Load results
        df = load_results_csv(results_path, logger)
        if df is not None:
            results[window] = df
            logger.info(f"{window}: Loaded {len(df)} results")
        elif not args.no_auto_run and not need_run:
            logger.warning(f"{window}: Could not load results")

    return results


def build_recommendation(
    convergence: Dict[str, Dict],
    vix_regime: Dict[str, Any],
    args,
) -> Dict[str, Any]:
    """
    Build final recommendation from convergence analysis.
    """
    recommendation = {
        "percent_beyond_put": convergence.get("percent_beyond_put", {}).get(
            "recommended", 0.0075
        ),
        "percent_beyond_call": convergence.get("percent_beyond_call", {}).get(
            "recommended", 0.02
        ),
        "max_spread_width_put": convergence.get("max_spread_width_put", {}).get(
            "recommended", 20
        ),
        "max_spread_width_call": convergence.get("max_spread_width_call", {}).get(
            "recommended", 20
        ),
        "min_trading_hour": convergence.get("min_trading_hour", {}).get(
            "recommended", 6
        ),
        "max_trading_hour": convergence.get("max_trading_hour", {}).get(
            "recommended", 9
        ),
        "profit_target_pct": args.profit_target_pct,
        "risk_cap": args.risk_cap,
        "max_live_capital": args.max_live_capital,
        "option_type": "both",
    }

    # Ensure trading hours are valid
    if recommendation["min_trading_hour"] > recommendation["max_trading_hour"]:
        recommendation["max_trading_hour"] = recommendation["min_trading_hour"] + 3

    return recommendation


async def main():
    """Main entry point."""
    args = parse_args()
    logger = get_logger("strategy_recommender", level=args.log_level)

    ticker = args.ticker.upper()
    windows = args.windows

    print(f"\n{'=' * 65}")
    print(f" Strategy Recommender - {ticker}")
    print(f"{'=' * 65}\n")

    # Dry run mode
    if args.dry_run:
        print("DRY RUN MODE - No grid searches will be executed\n")

        print(f"Configuration:")
        print(f"  Ticker: {ticker}")
        print(f"  Windows: {', '.join(windows)}")
        print(f"  Max Live Capital: ${args.max_live_capital:,.0f}")
        print(f"  Risk Cap: ${args.risk_cap:,.0f}")
        print(f"  CSV Directory: {args.csv_dir}")
        print(f"  Results Directory: {args.results_dir}")
        print()

        # Check data availability
        print("Data availability:")
        for window in windows:
            csv_paths = get_time_window_paths(ticker, window, args.csv_dir)
            results_path = get_cached_results_path(ticker, window, args.results_dir)
            has_results = results_path.exists()

            status = "CACHED" if has_results else "NEEDS RUN" if csv_paths else "NO DATA"
            print(f"  {window:>4}: {len(csv_paths):>3} CSV files, results: {status}")

        return 0

    # Initialize database for VIX data
    db = None
    vix_regime = None

    if not args.no_vix:
        try:
            db_path = args.db_path
            if db_path is None:
                db_path = os.environ.get("QUEST_DB_STRING")

            if db_path:
                db = StockQuestDB(db_path, enable_cache=True, logger=logger)
                vix_regime = await get_vix_regime(db, logger)
                logger.info(f"VIX regime: {vix_regime['regime']}")
        except Exception as e:
            logger.warning(f"Failed to fetch VIX data: {e}")

    if vix_regime is None:
        vix_regime = get_mock_vix_regime()
        logger.info("Using mock VIX data")

    # Get weights based on regime
    weights = get_weights_for_regime(vix_regime["regime"])

    # Gather results for all windows
    print(f"Analyzing {len(windows)} time windows...")
    results = await gather_results_for_windows(
        ticker,
        windows,
        args.csv_dir,
        args.results_dir,
        args,
        logger,
    )

    if not results:
        print("\nError: No results available for any time window.")
        print("Please run grid searches manually or check data availability.")
        if db:
            await db.close()
        return 1

    print(f"\nLoaded results for {len(results)} windows: {', '.join(results.keys())}")

    # Analyze convergence
    convergence = analyze_convergence(results, weights, top_n=args.top_n)
    summary = get_convergence_summary(convergence)

    # Build recommendation
    recommendation = build_recommendation(convergence, vix_regime, args)

    # Generate console report
    report = generate_console_report(
        ticker=ticker,
        vix_regime=vix_regime,
        convergence=convergence,
        recommendation=recommendation,
        summary=summary,
        max_live_capital=args.max_live_capital,
        risk_cap=args.risk_cap,
        output_timezone=args.output_timezone,
        windows=list(results.keys()),
    )
    print(report)

    # Generate and save JSON report
    if not args.no_json:
        json_report = generate_json_report(
            ticker=ticker,
            vix_regime=vix_regime,
            convergence=convergence,
            recommendation=recommendation,
            summary=summary,
            max_live_capital=args.max_live_capital,
            risk_cap=args.risk_cap,
            output_timezone=args.output_timezone,
            windows=list(results.keys()),
        )

        json_path = args.json_output
        if json_path is None:
            json_path = f"{ticker.lower()}_recommendation.json"

        save_json_report(json_report, json_path)
        print(f"\nJSON report saved to: {json_path}")

    # Cleanup
    if db:
        await db.close()

    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
