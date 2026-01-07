#!/usr/bin/env python3
"""
Port of the covered_call_generation shell script to native Python.
"""

from __future__ import annotations

import argparse
import glob
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import List
from zoneinfo import ZoneInfo

BASE_DIR = Path(__file__).resolve().parents[1]
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from common.market_hours import compute_market_transition_times, is_market_hours as common_is_market_hours  # noqa: E402
from common.gemini_analysis import run_gemini_analysis_on_file, DEFAULT_GEMINI_INSTRUCTION  # noqa: E402
from common.cache_warmup import warmup_stock_info_cache  # noqa: E402
import pandas as pd  # noqa: E402
TMP_DIR = Path("/tmp")

STORE_DIR_DEFAULT = Path("/Users/akundu/programs/http-proxy/static/")
OUTPUT_DIR_NAME = "stocks_to_buy"
ANALYSIS_FILE_BASENAME = "analysis"
DOWNLOAD_LOC_DEFAULT = Path.home() / "Downloads" / "results.csv"
GEMINI_PROG = BASE_DIR / "tests" / "gemini_test.py"
QUERY_LOC = "questdb://user:password@localhost:8812/stock_data"

TYPE_FLAG = "types"
TYPE_INPUT = "all"

MAX_WORKERS = 4
BATCH_SIZE = 300
MAX_DAYS = 30
POSITION_SIZE = 100000
GEMINI_COOLDOWN_SECONDS = 3600
LAST_GEMINI_RUN_FILE = TMP_DIR / "covered_call_last_gemini_run_epoch"

MARKET_HOURS_LOOKBACK_SECONDS = 3600  # 1 hours default
OFF_HOURS_LOOKBACK_SECONDS = 259200  # 72 hours default

MIN_PE = 1
MIN_VOL = 10
MIN_LONG_PREMIUM = 0.5
MIN_PREMIUM = 0.25
MIN_NET_PREMIUM = 1000
CURR_PRICE_MULT = 1.01
SENSIBLE_PRICE = 0.001

SPREAD_STRIKE_TOLERANCE = 5
SPREAD_LONG_DAYS = 120
SPREAD_LONG_MIN_DAYS = None  # Set to None to use tolerance-based range, or set to a specific minimum
SPREAD_LONG_DAYS_TOLERANCE = 60

REFRESH_RESULTS_BACKGROUND = 600

LOG_LEVEL = "WARNING"
OPTION_TYPE = "both"
SORT = "potential_premium"
DEFAULT_SLEEP_SECONDS = 120
MAX_OFF_HOURS_SLEEP = 3600
MIN_OFF_HOURS_SLEEP = 3600

# Use the default instruction from common module
GEMINI_INSTRUCTION = DEFAULT_GEMINI_INSTRUCTION


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate covered call recommendations.")
    parser.add_argument("--force-gemini", action="store_true", help="Force Gemini to run.")
    parser.add_argument("--gemini-only", action="store_true", help="Run only the Gemini analysis.")
    parser.add_argument("--gemini-input-file", type=str, default="", help="Path to Gemini input CSV.")
    parser.add_argument("--gemini-output-dir", type=str, default="", help="Gemini output directory name.")
    parser.add_argument("--gemini-store-dir", type=str, default="", help="Directory to store Gemini outputs.")
    parser.add_argument("--no-cache", action="store_true", help="Forward --no-cache to options_analyzer.")
    parser.add_argument("--db-server-host", type=str, default="mm.kundu.dev", help="Database server hostname for cache warmup (default: mm.kundu.dev).")
    parser.add_argument("--db-server-port", type=int, default=9100, help="Database server port for cache warmup (default: 9100).")
    parser.add_argument("--output-file", type=str, default="", help=f"Path to output CSV file (default: {DOWNLOAD_LOC_DEFAULT}).")
    parser.add_argument("--html-output-dir", type=str, default="", help=f"Directory for HTML output (default: {TMP_DIR / OUTPUT_DIR_NAME}).")
    
    execution_group = parser.add_mutually_exclusive_group()
    execution_group.add_argument(
        "--once",
        action="store_true",
        help="Run only once and exit (default: run continuously)."
    )
    execution_group.add_argument(
        "--iterations",
        type=int,
        metavar="N",
        default=None,
        help="Run N times and exit (default: run continuously)."
    )
    
    # Options analyzer parameters
    parser.add_argument("--type-flag", type=str, choices=["symbols", "types"], default=TYPE_FLAG, help=f"Type flag: 'symbols' or 'types' (default: {TYPE_FLAG}).")
    parser.add_argument("--type-input", type=str, default=TYPE_INPUT, help=f"Type input value (default: {TYPE_INPUT}).")
    parser.add_argument("--db-conn", type=str, default=QUERY_LOC, help=f"Database connection string (default: {QUERY_LOC}).")
    parser.add_argument("--max-days", type=int, default=MAX_DAYS, help=f"Maximum days to expiry (default: {MAX_DAYS}).")
    parser.add_argument("--min-days", type=int, default=None, help="Minimum days to expiry for short-term options (default: None, uses yesterday).")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE, help=f"Batch size (default: {BATCH_SIZE}).")
    parser.add_argument("--max-workers", type=int, default=MAX_WORKERS, help=f"Maximum workers (default: {MAX_WORKERS}).")
    parser.add_argument("--position-size", type=int, default=POSITION_SIZE, help=f"Position size (default: {POSITION_SIZE}).")
    parser.add_argument("--no-spread", action="store_true", help="Disable spread analysis (default: spread enabled).")
    parser.add_argument("--spread-strike-tolerance", type=int, default=SPREAD_STRIKE_TOLERANCE, help=f"Spread strike tolerance (default: {SPREAD_STRIKE_TOLERANCE}).")
    parser.add_argument("--spread-long-days", type=int, default=SPREAD_LONG_DAYS, help=f"Spread long days (default: {SPREAD_LONG_DAYS}).")
    parser.add_argument("--spread-long-min-days", type=int, default=SPREAD_LONG_MIN_DAYS, help=f"Spread long minimum days (default: {SPREAD_LONG_MIN_DAYS}).")
    parser.add_argument("--spread-long-days-tolerance", type=int, default=SPREAD_LONG_DAYS_TOLERANCE, help=f"Spread long days tolerance (default: {SPREAD_LONG_DAYS_TOLERANCE}).")
    parser.add_argument("--log-level", type=str, default=LOG_LEVEL, help=f"Log level (default: {LOG_LEVEL}).")
    parser.add_argument("--option-type", type=str, default=OPTION_TYPE, help=f"Option type: call, put, or both (default: {OPTION_TYPE}).")
    parser.add_argument("--sensible-price", type=float, default=SENSIBLE_PRICE, help=f"Sensible price threshold (default: {SENSIBLE_PRICE}).")
    parser.add_argument("--min-vol", type=int, default=MIN_VOL, help=f"Minimum volume filter (default: {MIN_VOL}).")
    parser.add_argument("--sort", type=str, default=SORT, help=f"Sort field (default: {SORT}).")
    parser.add_argument("--top-n", type=int, default=5, help="Top N results (default: 5).")
    parser.add_argument("--no-stats", action="store_true", help="Disable stats output (default: stats enabled).")
    parser.add_argument("--market-hours-lookback-seconds", type=int, default=MARKET_HOURS_LOOKBACK_SECONDS, help=f"Seconds to look back during market hours (default: {MARKET_HOURS_LOOKBACK_SECONDS}).")
    parser.add_argument("--off-hours-lookback-seconds", type=int, default=OFF_HOURS_LOOKBACK_SECONDS, help=f"Seconds to look back outside market hours (default: {OFF_HOURS_LOOKBACK_SECONDS}).")
    parser.add_argument("--max-bid-ask-spread", type=float, default=2.0, help="Maximum bid-ask spread ratio for short options. Formula: (ask - bid) / bid <= max_spread. Default: 2.0 (200%% spread). Set to 0 to disable.")
    parser.add_argument("--max-bid-ask-spread-long", type=float, default=2.0, help="Maximum bid-ask spread ratio for long options (spread mode). Formula: (ask - bid) / bid <= max_spread. Default: 2.0 (200%% spread). Set to 0 to disable.")
    
    return parser.parse_args()


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def format_sleep_duration(seconds: int) -> str:
    """Format sleep duration in a human-readable format."""
    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    secs = seconds % 60
    
    parts = []
    if hours > 0:
        parts.append(f"{hours} hour{'s' if hours != 1 else ''}")
    if minutes > 0:
        parts.append(f"{minutes} minute{'s' if minutes != 1 else ''}")
    if secs > 0 or not parts:
        parts.append(f"{secs} second{'s' if secs != 1 else ''}")
    
    return ", ".join(parts) + f" ({seconds} seconds)"


def write_subset_file(source: Path, dest: Path, token: str) -> None:
    with source.open("r", encoding="utf-8") as src, dest.open("w", encoding="utf-8") as dst:
        for line in src:
            if token in line:
                dst.write(line)


def run_gemini_analysis(
    input_file: Path,
    output_dir_name: str,
    store_dir: Path | None,
    force_run: bool,
    market_hours: bool,
    cooldown_seconds: int,
    last_run_file: Path,
) -> None:
    """Run Gemini analysis using the common module.
    
    This is a wrapper around the common.gemini_analysis.run_gemini_analysis_on_file function
    that maintains backward compatibility with the existing interface.
    """
    print("Running Gemini analysis...", file=sys.stderr, flush=True)
    
    # Determine output directory
    if store_dir and output_dir_name:
        output_dir = store_dir / output_dir_name
    else:
        output_dir = TMP_DIR
    
    # Use the common function
    analysis_outputs = run_gemini_analysis_on_file(
        input_file=input_file,
        output_dir=output_dir,
        instruction=GEMINI_INSTRUCTION,
        gemini_prog=GEMINI_PROG,
        base_dir=BASE_DIR,
        force_run=force_run,
        market_hours=market_hours,
        cooldown_seconds=cooldown_seconds,
        last_run_file=last_run_file,
    )
    
    # Copy outputs to final destination if needed
    if store_dir and output_dir_name:
        dest_dir = store_dir / output_dir_name
        ensure_dir(dest_dir)
        for opt_type, output_path in analysis_outputs.items():
            if output_path.exists():
                final_name = f"{ANALYSIS_FILE_BASENAME}.{opt_type}.html"
                shutil.copy(output_path, dest_dir / final_name)


def build_options_analyzer_command(
    download_loc: Path,
    db_conn: str,
    no_cache: bool,
    type_flag: str,
    type_input: str,
    max_days: int,
    min_days: int | None,
    batch_size: int,
    max_workers: int,
    position_size: int,
    spread: bool,
    spread_strike_tolerance: int,
    spread_long_days: int,
    spread_long_min_days: int,
    spread_long_days_tolerance: int,
    log_level: str,
    option_type: str,
    sensible_price: float,
    min_vol: int,
    sort: str,
    top_n: int,
    stats: bool,
    min_write_timestamp: str | None = None,
    max_bid_ask_spread: float = 2.0,
    max_bid_ask_spread_long: float = 2.0,
) -> List[str]:
    """Build command line arguments for options_analyzer.py using argparse."""
    # Import the argument parser functions
    from common.options.options_args import (
        add_database_arguments,
        add_analysis_arguments,
        add_spread_arguments,
        add_performance_arguments,
        add_filter_arguments,
        add_output_arguments,
    )
    from common.symbol_loader import add_symbol_arguments

    # Create a parser to get argument definitions
    parser = argparse.ArgumentParser()
    add_symbol_arguments(parser, required=True, allow_positional=False)
    add_database_arguments(parser)
    add_analysis_arguments(parser)
    add_spread_arguments(parser)
    add_performance_arguments(parser)
    add_filter_arguments(parser)
    add_output_arguments(parser)

    # Build command starting with script path
    cmd: List[str] = [sys.executable, str(BASE_DIR / "scripts" / "options_analyzer.py")]

    # Helper to get option string from action
    def get_option_string(action: argparse.Action) -> str:
        """Get the option string (e.g., '--db-conn') from an action."""
        if action.option_strings:
            return action.option_strings[0]  # Use the first/longest option
        return f"--{action.dest.replace('_', '-')}"

    # Map of argument values we want to set
    arg_values = {
        type_flag: type_input,
        "db_conn": db_conn,
        "max_days": max_days,
        "min_days": min_days,
        "batch_size": batch_size,
        "max_workers": max_workers,
        "position_size": position_size,
        "spread": spread,
        "spread_strike_tolerance": spread_strike_tolerance,
        "spread_long_days": spread_long_days,
        "spread_long_min_days": spread_long_min_days,
        "spread_long_days_tolerance": spread_long_days_tolerance,
        "output": str(download_loc),
        "log_level": log_level,
        "option_type": option_type,
        "sensible_price": sensible_price,
        "min_volume": min_vol,
        "sort": sort,
        "top_n": top_n,
        "stats": stats,
        "no_cache": no_cache,
        "min_write_timestamp": min_write_timestamp,
        "max_bid_ask_spread": max_bid_ask_spread,
        "max_bid_ask_spread_long": max_bid_ask_spread_long,
    }

    # Build command line arguments using parser actions
    for action in parser._actions:
        if action.dest == "help":
            continue

        value = arg_values.get(action.dest)
        if value is None:
            continue

        option_str = get_option_string(action)

        # Handle boolean flags (store_true/store_false)
        if isinstance(action, (argparse._StoreTrueAction, argparse._StoreFalseAction)):
            if value:
                cmd.append(option_str)
        # Handle append actions (like --filter which can be used multiple times)
        elif isinstance(action, argparse._AppendAction):
            for item in value:
                cmd.extend([option_str, str(item)])
        # Handle regular arguments
        else:
            cmd.extend([option_str, str(value)])

    return cmd


def copy_analysis_files(store_dir: Path, output_dir_name: str) -> None:
    dest_dir = store_dir / output_dir_name
    ensure_dir(dest_dir)
    pattern = str(TMP_DIR / f"{ANALYSIS_FILE_BASENAME}.*")
    for src_path in glob.glob(pattern):
        try:
            shutil.copy(src_path, dest_dir)
        except OSError:
            pass


def run_loop(
    *,
    force_gemini: bool,
    gemini_only: bool,
    gemini_input_file: Path,
    gemini_output_dir: str,
    gemini_store_dir: Path,
    no_cache: bool,
    run_once: bool = False,
    iterations: int | None = None,
    type_flag: str,
    type_input: str,
    db_conn: str,
    max_days: int,
    min_days: int | None,
    batch_size: int,
    max_workers: int,
    position_size: int,
    spread: bool,
    spread_strike_tolerance: int,
    spread_long_days: int,
    spread_long_min_days: int,
    spread_long_days_tolerance: int,
    log_level: str,
    option_type: str,
    sensible_price: float,
    min_vol: int,
    sort: str,
    top_n: int,
    stats: bool,
    market_hours_lookback_seconds: int,
    off_hours_lookback_seconds: int,
    db_server_host: str,
    db_server_port: int,
    max_bid_ask_spread: float,
    max_bid_ask_spread_long: float,
    output_file: Path | None = None,
    html_output_dir: Path | None = None,
) -> None:
    download_loc = output_file if output_file else DOWNLOAD_LOC_DEFAULT
    html_output_path = html_output_dir if html_output_dir else (TMP_DIR / OUTPUT_DIR_NAME)
    store_dir = gemini_store_dir

    if gemini_only:
        if not gemini_input_file.exists():
            print(f"Error: Input file not found: {gemini_input_file}", file=sys.stderr, flush=True)
            sys.exit(1)
        run_gemini_analysis(
            gemini_input_file,
            gemini_output_dir,
            store_dir,
            force_run=True,
            market_hours=True,
            cooldown_seconds=GEMINI_COOLDOWN_SECONDS,
            last_run_file=LAST_GEMINI_RUN_FILE,
        )
        sys.exit(0)

    # Determine iteration limit
    max_iterations: int | None = None
    if run_once:
        max_iterations = 1
    elif iterations is not None:
        max_iterations = iterations

    iteration_count = 0

    while True:
        iteration_count += 1
        now_utc = datetime.now(timezone.utc)
        market_hours = common_is_market_hours(now_utc)

        # Calculate min_write_timestamp based on market hours and lookback seconds
        if market_hours:
            # During market hours: use market hours lookback
            lookback_seconds = market_hours_lookback_seconds
            sleep_time = DEFAULT_SLEEP_SECONDS
        else:
            # Outside market hours: use off-hours lookback
            lookback_seconds = off_hours_lookback_seconds
            # Calculate time until next market open
            seconds_to_open, _ = compute_market_transition_times(now_utc)
            if seconds_to_open is None or seconds_to_open <= 0:
                # Fallback to max sleep if calculation fails
                sleep_time = MAX_OFF_HOURS_SLEEP
            else:
                # Sleep for min of 1 hour (3600s) or time until market opens
                # This ensures we wake up before market opens, but sleep at most 1 hour
                # If market opens in less than 1 hour, we'll wake up then
                # If market opens in more than 1 hour, we'll sleep for 1 hour max
                sleep_time = min(MAX_OFF_HOURS_SLEEP, int(seconds_to_open))
            market_hours = False

        # Calculate min_write_timestamp in Eastern time (EST/EDT depending on DST)
        lookback_time = now_utc - timedelta(seconds=lookback_seconds)
        # Convert to America/New_York timezone (handles DST automatically)
        eastern_tz = ZoneInfo("America/New_York")
        lookback_time_eastern = lookback_time.astimezone(eastern_tz)
        # Format as naive datetime string (without timezone indicator) as expected by options_analyzer
        min_write_timestamp = lookback_time_eastern.strftime("%Y-%m-%d %H:%M:%S")

        print(datetime.now(), file=sys.stderr, flush=True)
        print(f"Market hours: {market_hours}, Lookback: {lookback_seconds}s, Min write timestamp (Eastern): {min_write_timestamp}", file=sys.stderr, flush=True)
        command = build_options_analyzer_command(
            download_loc,
            db_conn,
            no_cache,
            type_flag,
            type_input,
            max_days,
            min_days,
            batch_size,
            max_workers,
            position_size,
            spread,
            spread_strike_tolerance,
            spread_long_days,
            spread_long_min_days,
            spread_long_days_tolerance,
            log_level,
            option_type,
            sensible_price,
            min_vol,
            sort,
            top_n,
            stats,
            min_write_timestamp,
            max_bid_ask_spread,
            max_bid_ask_spread_long,
        )
        print(f"Executing: {' '.join(command)}", file=sys.stderr, flush=True)
        start_time = time.time()
        result = subprocess.run(command, cwd=str(BASE_DIR))
        elapsed = int(time.time() - start_time)
        print(f"Elapsed: {elapsed} seconds with result = {result.returncode}", file=sys.stderr, flush=True)

        # Cache warmup: Load results CSV and warm up cache for all tickers
        if result.returncode == 0 and download_loc.exists():
            try:
                print("Loading results CSV for cache warmup...", file=sys.stderr, flush=True)
                df = pd.read_csv(download_loc)
                
                # Clean up duplicate header rows if present
                if not df.empty and 'ticker' in df.columns:
                    df = df[df['ticker'] != 'ticker']
                
                if not df.empty:
                    # Calculate TTL as 1/2 of the sleep time (in seconds)
                    ttl_seconds = sleep_time / 2.0
                    print(
                        f"Starting cache warmup for tickers in results (TTL: {ttl_seconds:.0f}s = 1/2 of sleep interval)",
                        file=sys.stderr,
                        flush=True
                    )
                    # Fire-and-forget warmup (wait_timeout=None)
                    warmup_stock_info_cache(
                        df,
                        host=db_server_host,
                        port=db_server_port,
                        ttl_seconds=ttl_seconds,
                        wait_timeout=None  # Fire-and-forget
                    )
                    print("Cache warmup initiated", file=sys.stderr, flush=True)
                else:
                    print("Results CSV is empty, skipping cache warmup", file=sys.stderr, flush=True)
            except Exception as e:
                print(f"Error during cache warmup (non-fatal): {e}", file=sys.stderr, flush=True)
                # Don't fail the whole process if warmup fails

        # Run stock analysis after options analysis completes successfully
        if result.returncode == 0:
            try:
                stock_analysis_csv = Path.home() / "Downloads" / "stock_analysis.csv"
                symbols_dir = os.path.expanduser("~/programs/var/US-Stock-Symbols")
                
                print("Running stock analysis...", file=sys.stderr, flush=True)
                analyze_cmd = [
                    sys.executable,
                    str(BASE_DIR / "scripts" / "analyze_stocks.py"),
                    "--db-path", db_conn,
                    "--symbols-dir", symbols_dir,
                    "--csv", str(stock_analysis_csv),
                    "--top-n", "10",
                ]
                # Don't display the command output -- ensure no output shown
                analyze_start_time = time.time()
                with open(os.devnull, 'w') as devnull:
                    analyze_result = subprocess.run(
                        analyze_cmd,
                        cwd=str(BASE_DIR),
                        stdout=devnull,
                        stderr=devnull,
                    )
                analyze_elapsed = int(time.time() - analyze_start_time)
                print(f"Stock analysis elapsed: {analyze_elapsed} seconds with result = {analyze_result.returncode}", file=sys.stderr, flush=True)
                
                if analyze_result.returncode == 0 and stock_analysis_csv.exists():
                    print(f"Stock analysis CSV generated: {stock_analysis_csv}", file=sys.stderr, flush=True)
                else:
                    print(f"Stock analysis failed or CSV not found (non-fatal)", file=sys.stderr, flush=True)
            except Exception as e:
                print(f"Error during stock analysis (non-fatal): {e}", file=sys.stderr, flush=True)
                # Don't fail the whole process if stock analysis fails

        # don't need to generate this any more as it should be generated once on a change if necessary
        # if result.returncode == 0:
        #     evaluate_cmd = [
        #         sys.executable,
        #         str(BASE_DIR / "scripts" / "evaluate_covered_calls.py"),
        #         "--file",
        #         str(download_loc),
        #         "--html",
        #         "--output-dir",
        #         str(html_output_path),
        #         "--db-server-host",
        #         str(db_server_host),
        #         "--db-server-port",
        #         str(db_server_port),
        #     ]
        #     print(f"Executing: {' '.join(evaluate_cmd)}", file=sys.stderr, flush=True)
        #     eval_result = subprocess.run(evaluate_cmd, cwd=str(BASE_DIR))
        #     print(f"evaluate_covered_calls.py completed with result = {eval_result.returncode}", file=sys.stderr, flush=True)
        #     if eval_result.returncode == 0:
        #         dest_dir = store_dir / OUTPUT_DIR_NAME
        #         shutil.rmtree(dest_dir, ignore_errors=True)
        #         try:
        #             shutil.move(str(html_output_path), dest_dir)
        #             print(f"Moved results to {dest_dir}", file=sys.stderr, flush=True)
        #         except FileNotFoundError:
        #             print("Temporary output directory missing; skipping move.", file=sys.stderr, flush=True)
        #     else:
        #         print("evaluate_covered_calls.py failed; skipping result move.", file=sys.stderr, flush=True)
        #
        #     run_gemini_analysis(
        #         download_loc,
        #         gemini_output_dir,
        #         store_dir,
        #         force_run=force_gemini,
        #         market_hours=market_hours,
        #         cooldown_seconds=GEMINI_COOLDOWN_SECONDS,
        #         last_run_file=LAST_GEMINI_RUN_FILE,
        #     )

        # try:
        #     copy_analysis_files(store_dir, gemini_output_dir)
        # except Exception as exc:  # noqa: BLE001
        #     print(f"Failed to copy analysis files: {exc}", file=sys.stderr, flush=True)

        # Exit if we've reached the iteration limit
        if max_iterations is not None and iteration_count >= max_iterations:
            if max_iterations == 1:
                print("Completed single execution. Exiting.", file=sys.stderr, flush=True)
            else:
                print(f"Completed {iteration_count} iteration(s). Exiting.", file=sys.stderr, flush=True)
            break

        sleep_duration_str = format_sleep_duration(sleep_time)
        print(f"Sleeping for {sleep_duration_str}...", file=sys.stderr, flush=True)
        try:
            time.sleep(sleep_time)
        except KeyboardInterrupt:
            print("Interrupted; exiting loop.", file=sys.stderr, flush=True)
            break


def main() -> None:
    args = parse_args()
    gemini_input_file = Path(args.gemini_input_file or DOWNLOAD_LOC_DEFAULT)
    gemini_output_dir = args.gemini_output_dir or OUTPUT_DIR_NAME
    gemini_store_dir = Path(os.path.expanduser(args.gemini_store_dir or STORE_DIR_DEFAULT))
    output_file = Path(args.output_file) if args.output_file else None
    html_output_dir = Path(args.html_output_dir) if args.html_output_dir else None

    ensure_dir(gemini_store_dir)

    run_loop(
        force_gemini=args.force_gemini,
        gemini_only=args.gemini_only,
        gemini_input_file=gemini_input_file,
        gemini_output_dir=gemini_output_dir,
        gemini_store_dir=gemini_store_dir,
        no_cache=args.no_cache,
        run_once=args.once,
        iterations=args.iterations,
        type_flag=args.type_flag,
        type_input=args.type_input,
        db_conn=args.db_conn,
        max_days=args.max_days,
        min_days=args.min_days,
        batch_size=args.batch_size,
        max_workers=args.max_workers,
        position_size=args.position_size,
        spread=not args.no_spread,
        spread_strike_tolerance=args.spread_strike_tolerance,
        spread_long_days=args.spread_long_days,
        spread_long_min_days=args.spread_long_min_days,
        spread_long_days_tolerance=args.spread_long_days_tolerance,
        log_level=args.log_level,
        option_type=args.option_type,
        sensible_price=args.sensible_price,
        min_vol=args.min_vol,
        sort=args.sort,
        top_n=args.top_n,
        stats=not args.no_stats,
        market_hours_lookback_seconds=args.market_hours_lookback_seconds,
        off_hours_lookback_seconds=args.off_hours_lookback_seconds,
        db_server_host=args.db_server_host,
        db_server_port=args.db_server_port,
        max_bid_ask_spread=args.max_bid_ask_spread,
        max_bid_ask_spread_long=args.max_bid_ask_spread_long,
        output_file=output_file,
        html_output_dir=html_output_dir,
    )


if __name__ == "__main__":
    main()

