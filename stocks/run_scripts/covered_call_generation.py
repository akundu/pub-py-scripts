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

BASE_DIR = Path(__file__).resolve().parents[1]
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from common.market_hours import compute_market_transition_times, is_market_hours as common_is_market_hours  # noqa: E402
TMP_DIR = Path("/tmp")

STORE_DIR_DEFAULT = Path("/Users/akundu/programs/http-proxy/static/")
OUTPUT_DIR_NAME = "stocks_to_buy"
ANALYSIS_FILE_BASENAME = "analysis"
DOWNLOAD_LOC_DEFAULT = Path.home() / "Downloads" / "results.csv"
GEMINI_PROG = BASE_DIR / "tests" / "gemini_test.py"
QUERY_LOC = "questdb://stock_user:stock_password@localhost:8812/stock_data"

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
SPREAD_LONG_MIN_DAYS = 45
SPREAD_LONG_DAYS_TOLERANCE = 60

REFRESH_RESULTS_BACKGROUND = 600

LOG_LEVEL = "WARNING"
OPTION_TYPE = "both"
SORT = "potential_premium"
DEFAULT_SLEEP_SECONDS = 300
MAX_OFF_HOURS_SLEEP = 3600
MIN_OFF_HOURS_SLEEP = 3600

GEMINI_INSTRUCTION = (
    "given the provided file of spread option trades possible, choose the 5 best set "
    "(based on realism of possibility of it happening) of dealing with risk and being "
    "aggressive and being conservative. focus on the intrinsic characteristics of each "
    "spread (strike prices, premiums, days to expiry and theta and delta), the underlying "
    "stock's volatility and market cap, and the reported net_daily_premi as an indicator "
    "of potential theta gain/loss. assume these represent **calendar spreads**, where you "
    "sell the shorter-dated option and buy the longer-dated option of the same type. the "
    "net cost (debit) per share is generally (long leg premium - short leg premium). a "
    "positive net_daily_premi suggests a theoretical daily gain from time decay. also, "
    "use the short_daily_premium in the analysis. make sure to only pick realistic "
    "situations of being able to procure those things. also, give me 3 examples of risky "
    "and 3 examples of conservative choices. Write the responses in a HTML form that I "
    "can save to a .html file. make sure to cover the examples of 3 per put spread and "
    "call spread."
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate covered call recommendations.")
    parser.add_argument("--force-gemini", action="store_true", help="Force Gemini to run.")
    parser.add_argument("--gemini-only", action="store_true", help="Run only the Gemini analysis.")
    parser.add_argument("--gemini-input-file", type=str, default="", help="Path to Gemini input CSV.")
    parser.add_argument("--gemini-output-dir", type=str, default="", help="Gemini output directory name.")
    parser.add_argument("--gemini-store-dir", type=str, default="", help="Directory to store Gemini outputs.")
    parser.add_argument("--no-cache", action="store_true", help="Forward --no-cache to options_analyzer.")
    
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
    print("Running Gemini analysis...", file=sys.stderr, flush=True)
    if not market_hours and not force_run:
        print("Skipping Gemini analysis: outside market hours", file=sys.stderr, flush=True)
        return

    current_epoch = int(time.time())
    last_epoch = 0
    if last_run_file.exists():
        try:
            last_epoch = int(last_run_file.read_text().strip() or "0")
        except ValueError:
            last_epoch = 0

    if not force_run and current_epoch - last_epoch < cooldown_seconds:
        remaining = cooldown_seconds - (current_epoch - last_epoch)
        print(f"Skipping Gemini analysis: ran recently ({remaining}s remaining on cooldown)", file=sys.stderr, flush=True)
        return

    analysis_outputs: list[Path] = []
    for opt_type in ("call", "put"):
        subset_file = input_file.with_suffix(input_file.suffix + f".{opt_type}")
        try:
            write_subset_file(input_file, subset_file, opt_type)
        except FileNotFoundError:
            print(f"Input file not found for Gemini analysis: {input_file}", file=sys.stderr, flush=True)
            return

        html_output = TMP_DIR / f"{ANALYSIS_FILE_BASENAME}.{opt_type}.html"
        analysis_outputs.append(html_output)
        command = [
            sys.executable,
            str(GEMINI_PROG),
            "--instruction",
            GEMINI_INSTRUCTION,
            "--file",
            str(subset_file),
        ]
        print(f"Executing: {' '.join(command)}", file=sys.stderr, flush=True)
        with html_output.open("w", encoding="utf-8") as handle:
            result = subprocess.run(command, stdout=handle, stderr=subprocess.STDOUT, cwd=str(BASE_DIR))
        print(f"Gemini analysis ({opt_type}) completed with result = {result.returncode}", file=sys.stderr, flush=True)

    last_run_file.write_text(str(current_epoch), encoding="utf-8")

    if store_dir and output_dir_name:
        dest_dir = store_dir / output_dir_name
        ensure_dir(dest_dir)
        for output in analysis_outputs:
            if output.exists():
                shutil.copy(output, dest_dir / output.name)


def build_options_analyzer_command(
    download_loc: Path,
    db_conn: str,
    no_cache: bool,
    type_flag: str,
    type_input: str,
    max_days: int,
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
        "filter": [f"volume > {min_vol}"],
        "sort": sort,
        "top_n": top_n,
        "stats": stats,
        "no_cache": no_cache,
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
) -> None:
    download_loc = DOWNLOAD_LOC_DEFAULT
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

        if market_hours:
            # During market hours: sleep for short interval
            sleep_time = DEFAULT_SLEEP_SECONDS
        else:
            # Outside market hours: calculate time until next market open
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

        print(datetime.now(), file=sys.stderr, flush=True)
        command = build_options_analyzer_command(
            download_loc,
            db_conn,
            no_cache,
            type_flag,
            type_input,
            max_days,
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
        )
        print(f"Executing: {' '.join(command)}", file=sys.stderr, flush=True)
        start_time = time.time()
        result = subprocess.run(command, cwd=str(BASE_DIR))
        elapsed = int(time.time() - start_time)
        print(f"Elapsed: {elapsed} seconds with result = {result.returncode}", file=sys.stderr, flush=True)

        if result.returncode == 0:
            evaluate_cmd = [
                sys.executable,
                str(BASE_DIR / "scripts" / "evaluate_covered_calls.py"),
                "--file",
                str(download_loc),
                "--html",
                "--output-dir",
                str(TMP_DIR / OUTPUT_DIR_NAME),
            ]
            print(f"Executing: {' '.join(evaluate_cmd)}", file=sys.stderr, flush=True)
            eval_result = subprocess.run(evaluate_cmd, cwd=str(BASE_DIR))
            print(f"evaluate_covered_calls.py completed with result = {eval_result.returncode}", file=sys.stderr, flush=True)
            if eval_result.returncode == 0:
                dest_dir = store_dir / OUTPUT_DIR_NAME
                shutil.rmtree(dest_dir, ignore_errors=True)
                try:
                    shutil.move(str(TMP_DIR / OUTPUT_DIR_NAME), dest_dir)
                    print(f"Moved results to {dest_dir}", file=sys.stderr, flush=True)
                except FileNotFoundError:
                    print("Temporary output directory missing; skipping move.", file=sys.stderr, flush=True)
            else:
                print("evaluate_covered_calls.py failed; skipping result move.", file=sys.stderr, flush=True)

            run_gemini_analysis(
                download_loc,
                gemini_output_dir,
                store_dir,
                force_run=force_gemini,
                market_hours=market_hours,
                cooldown_seconds=GEMINI_COOLDOWN_SECONDS,
                last_run_file=LAST_GEMINI_RUN_FILE,
            )

        try:
            copy_analysis_files(store_dir, gemini_output_dir)
        except Exception as exc:  # noqa: BLE001
            print(f"Failed to copy analysis files: {exc}", file=sys.stderr, flush=True)

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
    )


if __name__ == "__main__":
    main()

