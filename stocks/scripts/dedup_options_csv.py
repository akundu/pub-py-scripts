#!/usr/bin/env python3
"""One-time dedup pass for per-trading-date options CSVs.

The cron's `fetch_options.py --csv-layout per-trading-date` step used to
append unconditionally on every run. After the dedup fix in
`_save_options_to_csv_by_trading_date`, future runs are idempotent — but
existing files on disk still carry months of duplicated rows from prior
unguarded runs.

This script walks `<data-dir>/<TICKER>/<TICKER>_options_<trading_date>.csv`
and rewrites each file in place, keeping only the first occurrence of each
(ticker, expiration, timestamp) tuple. Bid/ask within a 15-min NBBO bar are
deterministic medians, so dropping later duplicates is lossless.

Optional `--max-days-from-trading-date N` also drops rows whose expiration
sits outside ±N days of the filename's trading_date — useful if old files
were written with a wider `--max-days-to-expiry` than current cron settings
and you want to shed that legacy bloat.
"""

import argparse
import csv
import os
import sys
import tempfile
from datetime import datetime, timedelta
from pathlib import Path


def dedup_one(
    path: Path,
    max_days_from_trading_date: int | None,
    dry_run: bool,
) -> tuple[int, int, int]:
    """Returns (rows_in, rows_out, rows_dropped_by_window)."""
    trading_date_str = path.stem.split("_options_")[-1]
    try:
        trading_date_dt = datetime.strptime(trading_date_str, "%Y-%m-%d").date()
    except ValueError:
        trading_date_dt = None

    if max_days_from_trading_date is not None and trading_date_dt is None:
        print(
            f"  ! cannot parse trading_date from {path.name}, skipping window filter",
            file=sys.stderr,
        )

    seen: set[tuple[str, str, str]] = set()
    out_rows: list[dict] = []
    rows_in = 0
    rows_window_dropped = 0
    fieldnames: list[str] | None = None

    with open(path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        if not fieldnames:
            return (0, 0, 0)
        for r in reader:
            rows_in += 1
            if max_days_from_trading_date is not None and trading_date_dt is not None:
                exp_str = (r.get("expiration") or "").strip()
                if exp_str:
                    try:
                        exp_dt = datetime.strptime(exp_str, "%Y-%m-%d").date()
                        delta_days = abs((exp_dt - trading_date_dt).days)
                        if delta_days > max_days_from_trading_date:
                            rows_window_dropped += 1
                            continue
                    except ValueError:
                        pass
            key = (
                r.get("ticker", "") or "",
                r.get("expiration", "") or "",
                r.get("timestamp", "") or "",
            )
            if key in seen:
                continue
            seen.add(key)
            out_rows.append(r)

    rows_out = len(out_rows)
    if dry_run or rows_out == rows_in:
        return (rows_in, rows_out, rows_window_dropped)

    # Write atomically — the cron may be running concurrently.
    tmp_fd, tmp_name = tempfile.mkstemp(
        prefix=f".{path.name}.dedup.", dir=str(path.parent)
    )
    try:
        with os.fdopen(tmp_fd, "w", newline="", encoding="utf-8") as f_out:
            writer = csv.DictWriter(f_out, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(out_rows)
        os.replace(tmp_name, path)
    except Exception:
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        raise

    return (rows_in, rows_out, rows_window_dropped)


def main() -> int:
    p = argparse.ArgumentParser(
        description=(
            "Dedup per-trading-date options CSVs in place. Keeps the first "
            "occurrence of each (ticker, expiration, timestamp) tuple."
        ),
        epilog=(
            "Examples:\n"
            "  %(prog)s --data-dir options_csv_output_full\n"
            "      Dedup every <TICKER>/<TICKER>_options_*.csv under the dir.\n\n"
            "  %(prog)s --data-dir options_csv_output_full --tickers RUT SPX NDX\n"
            "      Restrict to specific ticker subdirs.\n\n"
            "  %(prog)s --data-dir options_csv_output_full \\\n"
            "      --max-days-from-trading-date 5\n"
            "      Also drop rows whose expiration is more than ±5 days from\n"
            "      the file's trading_date — matches current cron settings.\n\n"
            "  %(prog)s --data-dir options_csv_output_full --dry-run\n"
            "      Report what would be dropped without rewriting any file.\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--data-dir", required=True,
        help="Top-level data dir containing <TICKER>/<TICKER>_options_*.csv files",
    )
    p.add_argument(
        "--tickers", nargs="+", default=None,
        help="Optional list of ticker subdirs to limit to (default: all)",
    )
    p.add_argument(
        "--max-days-from-trading-date", type=int, default=None,
        help=(
            "If set, also drop rows whose expiration is more than N days from "
            "the file's trading_date. Use to shed legacy rows from runs that "
            "had a wider --max-days-to-expiry."
        ),
    )
    p.add_argument(
        "--dry-run", action="store_true",
        help="Report stats without rewriting files",
    )
    args = p.parse_args()

    data_dir = Path(args.data_dir)
    if not data_dir.is_dir():
        print(f"ERROR: not a directory: {data_dir}", file=sys.stderr)
        return 2

    if args.tickers:
        ticker_dirs = [data_dir / t for t in args.tickers]
    else:
        ticker_dirs = [d for d in sorted(data_dir.iterdir()) if d.is_dir()]

    grand_in = 0
    grand_out = 0
    grand_window = 0
    files_touched = 0
    files_seen = 0

    for tdir in ticker_dirs:
        if not tdir.is_dir():
            continue
        files = sorted(tdir.glob(f"{tdir.name}_options_*.csv"))
        if not files:
            continue
        print(f"\n[{tdir.name}] {len(files)} file(s)")
        for f in files:
            files_seen += 1
            try:
                rin, rout, rwin = dedup_one(
                    f, args.max_days_from_trading_date, args.dry_run,
                )
            except Exception as e:
                print(f"  ! {f.name}: {e}", file=sys.stderr)
                continue
            grand_in += rin
            grand_out += rout
            grand_window += rwin
            if rin != rout:
                files_touched += 1
                pct = 100.0 * (rin - rout) / rin if rin else 0.0
                window_note = f", window-dropped {rwin}" if rwin else ""
                print(
                    f"  {f.name}: {rin:>8,} -> {rout:>8,} "
                    f"(-{rin - rout:,}, -{pct:.1f}%{window_note})"
                )

    action = "would touch" if args.dry_run else "rewrote"
    print(
        f"\nDone. Saw {files_seen:,} files; {action} {files_touched:,}. "
        f"Rows: {grand_in:,} -> {grand_out:,} "
        f"(-{grand_in - grand_out:,}, "
        f"window-dropped {grand_window:,})"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
