#!/usr/bin/env python3
"""
Auto-calibrate recommended percentiles for credit spread strike selection.

Runs a rolling backtest for each ticker, measures hit rates at each
percentile level, and selects the tightest percentile that meets the
target hit rate. Results are written to a JSON config file that the
web endpoints read for ★ row highlighting.

Usage:
    python -m scripts.calibrate_recommendations --days 90 --target 92.5
    python -m scripts.calibrate_recommendations --days 30 --tickers NDX  # quick test
    python -m scripts.calibrate_recommendations --help

Examples:
    # Full calibration (nightly cron)
    python -m scripts.calibrate_recommendations --days 90 --target 92.5 \\
        --tickers NDX,SPX,RUT --output results/calibration/recommended_percentiles.json

    # Quick single-ticker test
    python -m scripts.calibrate_recommendations --days 30 --tickers NDX

    # Check if tomorrow is a trading day (exit 0 = yes, exit 1 = no)
    python -m scripts.calibrate_recommendations --check-trading-day
"""

import argparse
import json
import sys
import time
from datetime import datetime, timedelta
from multiprocessing import Pool
from pathlib import Path

# Ensure project root is on path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DEFAULT_OUTPUT = str(PROJECT_ROOT / "results" / "calibration" / "recommended_percentiles.json")
DEFAULT_TICKERS = ["NDX", "SPX", "RUT"]
DEFAULT_DAYS = 90
DEFAULT_TARGET = 94.0
# Band levels tested by backtest_band_accuracy.py
BAND_LEVELS = [95, 97, 98, 99, 100]
# All percentile levels (including derived for intraday/max-move)
ALL_PERCENTILE_LEVELS = [75, 80, 85, 90, 95, 97, 98, 99, 100]


def is_next_trading_day() -> bool:
    """Check if tomorrow is a trading day (weekday)."""
    tomorrow = datetime.now().date() + timedelta(days=1)
    return tomorrow.weekday() < 5  # Mon=0 through Fri=4


def run_single_ticker(args):
    """Run backtest for a single ticker (called by multiprocessing pool)."""
    ticker, num_days, lookback = args

    # Each subprocess needs its own imports
    import sys
    import io
    import contextlib
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

    from scripts.backtest_band_accuracy import run_band_backtest

    # Suppress verbose output from model training
    with contextlib.redirect_stdout(io.StringIO()):
        results = run_band_backtest(ticker, num_days=num_days, lookback=lookback, verbose=False)

    if not results:
        return ticker, None

    # Compute hit rates and avg widths per band level
    hit_rates = {}
    avg_widths = {}
    for bn in BAND_LEVELS:
        band_name = f"P{bn}"
        hits = sum(1 for r in results if r.band_hit.get(band_name, False))
        total = sum(1 for r in results if band_name in r.band_hit)
        if total > 0:
            hit_rates[f"p{bn}"] = round(hits / total * 100, 1)
            widths = [r.band_widths[band_name] for r in results if band_name in r.band_widths]
            avg_widths[f"p{bn}"] = round(sum(widths) / len(widths), 2) if widths else 0.0
        else:
            hit_rates[f"p{bn}"] = 0.0
            avg_widths[f"p{bn}"] = 0.0

    # Midpoint error
    comb_errors = [r.combined_error_pct for r in results if r.combined_error_pct is not None]
    mid_error = round(sum(abs(e) for e in comb_errors) / len(comb_errors), 3) if comb_errors else 0.0

    return ticker, {
        "hit_rates": hit_rates,
        "avg_widths": avg_widths,
        "mid_error": mid_error,
        "test_days": len(results),
    }


def select_recommended(hit_rates: dict, target: float) -> int:
    """Select the tightest percentile that meets the target hit rate.

    Iterates from tightest (P75) to widest (P100) and returns the first
    percentile where hit_rate >= target. Falls back to P100 if none meet target.
    """
    for bn in BAND_LEVELS:
        key = f"p{bn}"
        if hit_rates.get(key, 0) >= target:
            return bn
    return 100  # fallback


def calibrate(tickers: list, num_days: int, target: float, lookback: int = 250,
              output_path: str = DEFAULT_OUTPUT) -> dict:
    """Run calibration for all tickers and write results."""
    print(f"Calibrating recommendations: {len(tickers)} tickers, {num_days}-day window, "
          f"target={target}% hit rate")

    start = time.time()

    # Run backtests in parallel
    pool_args = [(t, num_days, lookback) for t in tickers]
    with Pool(processes=min(len(tickers), 3)) as pool:
        raw_results = pool.map(run_single_ticker, pool_args)

    result = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "target_hit_rate": target,
        "backtest_days": num_days,
        "lookback": lookback,
        "tickers": {},
    }

    for ticker, data in raw_results:
        if data is None:
            print(f"  {ticker}: FAILED (no backtest results)")
            continue

        hr = data["hit_rates"]

        # Close-to-close: use the backtest results directly
        # The backtest tests symmetric Combined bands. Empirical analysis (120-day
        # and 250-day windows) consistently shows UP days have wider tails than
        # DOWN days across SPX/NDX/RUT — so CALL side needs wider bands.
        c2c_base = select_recommended(hr, target)
        c2c_put = c2c_base  # put (down) side: use base
        # Call (up) side: one step wider (up tails are 15-30% fatter than down)
        wider = [p for p in BAND_LEVELS if p > c2c_base]
        c2c_call = wider[0] if wider else c2c_base

        # Intraday: one level tighter (less time remaining, European settlement)
        tighter = [p for p in ALL_PERCENTILE_LEVELS if p < c2c_put]
        intra_put = tighter[-1] if tighter else c2c_put
        tighter = [p for p in ALL_PERCENTILE_LEVELS if p < c2c_call]
        intra_call = tighter[-1] if tighter else c2c_call

        # Max-move: one more level tighter (close determines P&L, not excursion)
        tighter = [p for p in ALL_PERCENTILE_LEVELS if p < intra_put]
        mm_put = tighter[-1] if tighter else intra_put
        tighter = [p for p in ALL_PERCENTILE_LEVELS if p < intra_call]
        mm_call = tighter[-1] if tighter else intra_call

        ticker_result = {
            "close_to_close": {"put": c2c_put, "call": c2c_call},
            "intraday": {"put": intra_put, "call": intra_call},
            "max_move": {"put": mm_put, "call": mm_call},
            "hit_rates": hr,
            "avg_widths": data["avg_widths"],
            "mid_error": data["mid_error"],
            "test_days": data["test_days"],
        }
        result["tickers"][ticker] = ticker_result

        print(f"  {ticker}: c2c put=P{c2c_put} call=P{c2c_call} "
              f"(base P{c2c_base} at {hr.get(f'p{c2c_base}', 0):.1f}% hit rate) "
              f"mid_err={data['mid_error']:.3f}%")

    elapsed = time.time() - start
    result["calibration_time_seconds"] = round(elapsed, 1)

    # Write to JSON
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nWrote recommendations to {output} ({elapsed:.0f}s)")

    return result


def main():
    parser = argparse.ArgumentParser(
        description='''
Auto-calibrate recommended percentiles for credit spread strike selection.

Runs a rolling backtest, measures hit rates at each percentile level,
and selects the tightest percentile meeting the target hit rate.
Results are written to a JSON file read by the web endpoints.
        ''',
        epilog='''
Examples:
  %(prog)s --days 90 --target 92.5
      Full calibration with 90-day window and 92.5%% target

  %(prog)s --days 30 --tickers NDX
      Quick test for NDX only

  %(prog)s --check-trading-day
      Exit 0 if tomorrow is a trading day, exit 1 otherwise
        ''',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--days", type=int, default=DEFAULT_DAYS,
                        help=f"Backtest window in trading days (default: {DEFAULT_DAYS})")
    parser.add_argument("--target", type=float, default=DEFAULT_TARGET,
                        help=f"Target hit rate %% (default: {DEFAULT_TARGET})")
    parser.add_argument("--tickers", type=str, default=",".join(DEFAULT_TICKERS),
                        help=f"Comma-separated tickers (default: {','.join(DEFAULT_TICKERS)})")
    parser.add_argument("--lookback", type=int, default=250,
                        help="Training lookback days for LightGBM (default: 250)")
    parser.add_argument("--output", type=str, default=DEFAULT_OUTPUT,
                        help=f"Output JSON path (default: {DEFAULT_OUTPUT})")
    parser.add_argument("--check-trading-day", action="store_true",
                        help="Check if tomorrow is a trading day and exit")

    args = parser.parse_args()

    if args.check_trading_day:
        if is_next_trading_day():
            print("Tomorrow is a trading day")
            sys.exit(0)
        else:
            print("Tomorrow is NOT a trading day")
            sys.exit(1)

    tickers = [t.strip().upper() for t in args.tickers.split(",") if t.strip()]
    calibrate(tickers, args.days, args.target, args.lookback, args.output)


if __name__ == "__main__":
    main()
