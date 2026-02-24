#!/usr/bin/env python3
"""
Close-to-close return percentiles by direction (up vs down days).

Compares prices across a configurable window (default: 1 day = consecutive days).
When the closing price at end of window is lower than at start, shows percentiles
of the % drop. When higher, shows percentiles of the % gain.
Uses QuestDB daily data; supports I:SPX, I:NDX, SPX, AAPL, etc.

Usage:
    python scripts/daily_range_percentiles.py --ticker SPX
    python scripts/daily_range_percentiles.py --ticker I:NDX:25000 SPX:5000   # manual close per ticker
    python scripts/daily_range_percentiles.py --ticker I:NDX SPX AAPL --lookback 63
    python scripts/daily_range_percentiles.py --ticker SPX --window 5  # 5-day window
"""

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from common.range_percentiles import (
    compute_range_percentiles_multi,
    compute_range_percentiles_multi_window_batch,
    parse_windows_arg,
    DEFAULT_LOOKBACK,
    DEFAULT_PERCENTILES,
    MIN_DAYS_DEFAULT,
    MIN_DIRECTION_DAYS_DEFAULT,
    DEFAULT_WINDOW,
)
from common.range_percentiles_formatter import format_as_text_table, format_multi_window_as_text_table

def parse_ticker_arg(token: str) -> tuple[str, float | None]:
    """
    Parse a single --ticker token into (ticker_symbol, override_close or None).
    Examples: 'SPX' -> ('SPX', None); 'SPX:5000' -> ('SPX', 5000.0); 'I:NDX:25000' -> ('I:NDX', 25000.0).
    """
    if ":" not in token:
        return (token.strip(), None)
    symbol, suffix = token.rsplit(":", 1)
    symbol = symbol.strip()
    suffix = suffix.strip()
    if not suffix:
        return (symbol, None)
    try:
        return (symbol, float(suffix))
    except ValueError:
        return (token.strip(), None)


def parse_percentiles(s: str) -> list[int]:
    """Parse comma- or space-separated integers (e.g. '75,90,95' or '75 90 95')."""
    out = []
    for part in s.replace(",", " ").split():
        part = part.strip()
        if not part:
            continue
        try:
            p = int(part)
            if 0 <= p <= 100:
                out.append(p)
            else:
                raise ValueError(f"Percentile must be 0-100: {p}")
        except ValueError as e:
            raise argparse.ArgumentTypeError(str(e))
    return sorted(set(out)) if out else DEFAULT_PERCENTILES.copy()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Daily range percentile predictions from QuestDB daily prices (supports I:* tickers)."
    )
    parser.add_argument(
        "--ticker",
        type=str,
        required=True,
        nargs="+",
        help="Ticker(s), optionally with manual close: SYMBOL or SYMBOL:CLOSE (e.g. SPX I:NDX:25000 SPX:5000).",
    )
    parser.add_argument(
        "--lookback",
        type=int,
        default=DEFAULT_LOOKBACK,
        help=f"Number of trading days to look back (default: {DEFAULT_LOOKBACK} ~ 6 months).",
    )
    parser.add_argument(
        "--window",
        type=int,
        default=DEFAULT_WINDOW,
        help=f"Window size in trading days for price comparison (default: {DEFAULT_WINDOW}). "
             f"1=consecutive days, 5=compare each day to 5 days prior, etc.",
    )
    parser.add_argument(
        "--windows",
        type=str,
        default=None,
        metavar="W",
        help="Multi-window mode: comma-separated (e.g., '1,5,10') or '*' for default "
             f"[1,3,5,10,15,20]. Overrides --window if specified.",
    )
    parser.add_argument(
        "--percentiles",
        type=parse_percentiles,
        default=None,
        metavar="P",
        help=f"Comma- or space-separated percentiles (default: {','.join(map(str, DEFAULT_PERCENTILES))}).",
    )
    parser.add_argument(
        "--min-days",
        type=int,
        default=MIN_DAYS_DEFAULT,
        help=f"Minimum number of days required to compute percentiles (default: {MIN_DAYS_DEFAULT}).",
    )
    parser.add_argument(
        "--min-direction-days",
        type=int,
        default=MIN_DIRECTION_DAYS_DEFAULT,
        help=f"Minimum days in each up/down subset to show that set (default: {MIN_DIRECTION_DAYS_DEFAULT}).",
    )
    parser.add_argument(
        "--db",
        type=str,
        default=None,
        help="QuestDB connection string. Default: QUEST_DB_STRING, QUESTDB_CONNECTION_STRING, or QUESTDB_URL env.",
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Disable Redis cache for DB reads.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output machine-readable JSON.",
    )
    parser.add_argument(
        "--ensure-tables",
        action="store_true",
        help="Ensure QuestDB tables exist on startup.",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="WARNING",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
        help="Logging level (default: WARNING).",
    )
    args = parser.parse_args()
    if args.percentiles is None:
        args.percentiles = DEFAULT_PERCENTILES.copy()
    if args.window < 1:
        parser.error("--window must be at least 1")
    return args




def main() -> int:
    args = parse_args()
    raw_tickers = args.ticker if isinstance(args.ticker, list) else [args.ticker]
    ticker_specs = [parse_ticker_arg(t) for t in raw_tickers]
    db_config = (
        args.db
        or os.getenv("QUEST_DB_STRING", "")
        or os.getenv("QUESTDB_CONNECTION_STRING", "")
        or os.getenv("QUESTDB_URL", "")
    )
    if not db_config.strip():
        print("ERROR: No QuestDB connection. Set --db or QUEST_DB_STRING (or QUESTDB_CONNECTION_STRING or QUESTDB_URL).", file=sys.stderr)
        return 1

    try:
        # Detect multi-window mode
        if args.windows is not None:
            # Multi-window analysis
            windows = parse_windows_arg(args.windows)

            results = asyncio.run(
                compute_range_percentiles_multi_window_batch(
                    ticker_specs=ticker_specs,
                    windows=windows,
                    lookback=args.lookback,
                    percentiles=args.percentiles,
                    min_days=args.min_days,
                    min_direction_days=args.min_direction_days,
                    db_config=db_config,
                    enable_cache=not args.no_cache,
                    ensure_tables=args.ensure_tables,
                    log_level=args.log_level,
                )
            )

            if args.json:
                print(json.dumps(results if len(results) != 1 else results[0], indent=2))
            else:
                for i, out in enumerate(results):
                    if i > 0:
                        print("\n" + "=" * 80 + "\n")
                    print(format_multi_window_as_text_table(out))
            return 0
        else:
            # Single-window mode (existing behavior)
            results = asyncio.run(
                compute_range_percentiles_multi(
                    ticker_specs=ticker_specs,
                    lookback=args.lookback,
                    percentiles=args.percentiles,
                    min_days=args.min_days,
                    min_direction_days=args.min_direction_days,
                    db_config=db_config,
                    enable_cache=not args.no_cache,
                    ensure_tables=args.ensure_tables,
                    log_level=args.log_level,
                    window=args.window,
                )
            )
            if args.json:
                print(json.dumps(results if len(results) != 1 else results[0], indent=2))
            else:
                for i, out in enumerate(results):
                    if i > 0:
                        print("\n" + "=" * 60 + "\n")
                    print(format_as_text_table(out))
            return 0
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        if args.json:
            print(json.dumps({"error": str(e)}, indent=2))
        return 1


if __name__ == "__main__":
    sys.exit(main())
