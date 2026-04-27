#!/usr/bin/env python3
"""
Export QuestDB hourly_prices into the 5-min CSV layout the calibration uses.

Reads `hourly_prices` for a ticker, expands each hourly bar into 12 5-min
bars via linear interpolation, and writes one CSV per trading day under
`equities_output/<TICKER>/`. Intended for backfilling tickers where Polygon
doesn't have intraday history (e.g. RUT before 2024-09-09) but Yahoo
Finance does — see `fetch_symbol_data.py --timeframe hourly --data-source
yfinance` for the upstream fetcher.

Caveats:
- Within each hour, the 5-min OHLC values are interpolated, not real
  microstructure. The percentile model (which uses prices at 30-min ET
  boundaries) gets meaningful signal; intraday-momentum features in the
  LightGBM path are smoothed and weaker.
- The DB normalizes hourly timestamps to the hour boundary (9:30 ET →
  9:00 ET), but yfinance's hourly bars actually start at 9:30 ET. We
  reconstruct the true 9:30-anchored period when laying out the 5-min bars.
- `--skip-existing` is on by default to avoid overwriting real Polygon
  5-min CSVs; pass `--overwrite` if you want to replace them.

Usage:
    # Fill RUT 5-min CSVs for the days where Polygon has no intraday data
    python -m scripts.export_hourly_to_5min_csv RUT \\
        --start 2024-05-06 --end 2024-09-08

    # Inspect what would be written, don't touch disk
    python -m scripts.export_hourly_to_5min_csv RUT \\
        --start 2024-05-06 --end 2024-09-08 --dry-run
"""

import argparse
import asyncio
import csv
import os
import sys
from datetime import datetime, timedelta, time
from pathlib import Path
from typing import List, Optional, Tuple
from zoneinfo import ZoneInfo

import asyncpg

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

ET_TZ = ZoneInfo("America/New_York")
UTC_TZ = ZoneInfo("UTC")
MARKET_OPEN_ET = time(9, 30)
MARKET_CLOSE_ET = time(16, 0)
BAR_INTERVAL_MIN = 5
# 9:30 → 16:00 ET inclusive at 5-min = 79 bars (matches Polygon CSVs).
BARS_PER_DAY = 79
# yfinance hourly bars are anchored on the half-hour (9:30, 10:30, ..., 15:30 ET).
# QuestDB's `save_stock_data` normalizes minutes to 0 for dedup, so DB rows
# read back at 9:00, 10:00, ..., 15:00 ET. We add this offset to recover the
# original anchor when interpreting bar coverage.
DB_TIMESTAMP_OFFSET = timedelta(minutes=30)


def _bar_times_et(date_str: str) -> list[datetime]:
    """Return BARS_PER_DAY ET-aware datetimes from 9:30 to 16:00 inclusive."""
    base = datetime.strptime(date_str, "%Y-%m-%d").date()
    start = datetime.combine(base, MARKET_OPEN_ET, tzinfo=ET_TZ)
    return [start + timedelta(minutes=i * BAR_INTERVAL_MIN) for i in range(BARS_PER_DAY)]


def _interpolate_price(
    target_et: datetime,
    bars: List[Tuple[datetime, float, float, float, float]],
) -> float:
    """Linearly interpolate a price at `target_et` from sorted hourly bars.

    Each bar tuple: (anchor_et, open, high, low, close) where anchor_et is
    the bar's ANCHOR time in ET (e.g. 9:30, 10:30, ...). The bar's open is
    the price at anchor_et; close is the price at anchor_et + 1h.
    """
    if not bars:
        raise ValueError("No bars to interpolate from")

    # Build a piecewise-linear price curve from (anchor_et → open) and
    # (anchor_et + 1h → close) anchor points.
    points: List[Tuple[datetime, float]] = []
    for (anchor, o, _h, _l, c) in bars:
        points.append((anchor, o))
        points.append((anchor + timedelta(hours=1), c))
    points.sort(key=lambda p: p[0])

    # Clamp to endpoints if outside range.
    if target_et <= points[0][0]:
        return points[0][1]
    if target_et >= points[-1][0]:
        return points[-1][1]

    # Find bracketing pair.
    for i in range(len(points) - 1):
        t0, p0 = points[i]
        t1, p1 = points[i + 1]
        if t0 <= target_et <= t1:
            if t1 == t0:
                return p0
            frac = (target_et - t0).total_seconds() / (t1 - t0).total_seconds()
            return p0 + (p1 - p0) * frac
    return points[-1][1]


def _build_5min_rows(
    date_str: str, csv_ticker: str,
    db_bars: list[dict],
) -> list[tuple]:
    """Build BARS_PER_DAY 5-min rows for a single trading day.

    db_bars: rows from hourly_prices ordered by datetime ASC. Each row is a
    dict with keys datetime (naive UTC), open, high, low, close, volume.
    """
    if not db_bars:
        return []

    # Reconstruct the true ET anchor for each hourly bar (DB stores normalized
    # to the hour, original yfinance anchor was on the half-hour).
    anchored: List[Tuple[datetime, float, float, float, float]] = []
    for r in db_bars:
        ts_utc = r["datetime"]
        if ts_utc.tzinfo is None:
            ts_utc = ts_utc.replace(tzinfo=UTC_TZ)
        anchor_utc = ts_utc + DB_TIMESTAMP_OFFSET
        anchor_et = anchor_utc.astimezone(ET_TZ)
        anchored.append((anchor_et, float(r["open"]), float(r["high"]),
                         float(r["low"]), float(r["close"])))
    anchored.sort(key=lambda b: b[0])

    # Day-level high/low for pinning into the 5-min stream.
    day_high = max(b[2] for b in anchored)
    day_low = min(b[3] for b in anchored)
    day_open = anchored[0][1]
    day_close = anchored[-1][4]

    # Identify which 5-min bar to pin H and L into. We pick the bar nearest
    # the midpoint of the hourly bar that contains the day's high/low.
    high_anchor_et = next((b[0] for b in anchored if b[2] == day_high), anchored[0][0])
    low_anchor_et = next((b[0] for b in anchored if b[3] == day_low), anchored[-1][0])
    high_pin_et = high_anchor_et + timedelta(minutes=30)
    low_pin_et = low_anchor_et + timedelta(minutes=30)

    bar_times_et = _bar_times_et(date_str)
    bar_times_utc = [t.astimezone(UTC_TZ) for t in bar_times_et]

    # Find indices of the bars closest to high_pin and low_pin times.
    def _closest_idx(target):
        return min(range(BARS_PER_DAY),
                   key=lambda i: abs((bar_times_et[i] - target).total_seconds()))
    high_idx = _closest_idx(high_pin_et)
    low_idx = _closest_idx(low_pin_et)
    if high_idx == low_idx:
        # Avoid both being the same bar — bump low one slot if possible
        low_idx = (low_idx + 1) % BARS_PER_DAY

    rows: list[tuple] = []
    prev_close: Optional[float] = None
    for i, ts_et in enumerate(bar_times_et):
        c = _interpolate_price(ts_et, anchored)
        o = prev_close if prev_close is not None else day_open
        h = max(o, c)
        l = min(o, c)
        if i == high_idx:
            h = max(h, day_high)
        if i == low_idx:
            l = min(l, day_low)
        # First/last bar anchor exactly at day_open / day_close so downstream
        # consumers that read iloc[0]['open'] and iloc[-1]['close'] see the
        # values they expect.
        if i == 0:
            o = day_open
            h = max(h, o)
            l = min(l, o)
        if i == BARS_PER_DAY - 1:
            c = day_close
            h = max(h, c)
            l = min(l, c)

        ts_str = bar_times_utc[i].strftime("%Y-%m-%d %H:%M:%S+00:00")
        rows.append((ts_str, csv_ticker, o, h, l, c, 0, "", ""))
        prev_close = c

    return rows


async def _fetch_days(ticker: str, start: str, end: str) -> dict[str, list[dict]]:
    """Fetch hourly bars for ticker in [start, end], grouped by ET trading date.

    Tickers are stored under the bare symbol in hourly_prices (e.g. 'RUT' not 'I:RUT').
    """
    cs = os.environ.get("QUEST_DB_STRING") or os.environ.get("QUESTDB_CONNECTION_STRING") \
        or "questdb://stock_user:stock_password@lin1.kundu.dev:8812/stock_data"
    cs = cs.replace("questdb://", "postgresql://")

    db_ticker = ticker.replace("I:", "").upper()
    start_dt = datetime.strptime(start, "%Y-%m-%d")
    end_dt = datetime.strptime(end, "%Y-%m-%d")
    conn = await asyncpg.connect(cs)
    try:
        rows = await conn.fetch(
            """
            SELECT datetime, open, high, low, close, volume
            FROM hourly_prices
            WHERE ticker = $1 AND datetime >= $2 AND datetime < $3
            ORDER BY datetime
            """,
            db_ticker, start_dt, end_dt,
        )
    finally:
        await conn.close()

    by_date: dict[str, list[dict]] = {}
    for r in rows:
        ts_utc = r["datetime"]
        if ts_utc.tzinfo is None:
            ts_utc = ts_utc.replace(tzinfo=UTC_TZ)
        # Use the reconstructed ET anchor (add the 30-min offset) so a bar
        # captured before midnight UTC still groups into the right ET date.
        anchor_et = (ts_utc + DB_TIMESTAMP_OFFSET).astimezone(ET_TZ)
        # Skip rows outside regular market hours (9:30 → 16:00 ET window).
        if anchor_et.time() < MARKET_OPEN_ET or anchor_et.time() >= MARKET_CLOSE_ET:
            continue
        date_str = anchor_et.strftime("%Y-%m-%d")
        by_date.setdefault(date_str, []).append(dict(r))
    return by_date


def _csv_path(ticker: str, output_dir: Path, date_str: str) -> Path:
    csv_ticker = ticker if ticker.startswith("I:") else f"I:{ticker.upper()}"
    return output_dir / csv_ticker / f"{csv_ticker}_equities_{date_str}.csv"


def _write_csv(path: Path, rows: list[tuple]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as fp:
        w = csv.writer(fp)
        w.writerow(["timestamp", "ticker", "open", "high", "low", "close",
                    "volume", "vwap", "transactions"])
        w.writerows(rows)


async def main_async(args) -> int:
    by_date = await _fetch_days(args.ticker, args.start, args.end)
    if not by_date:
        print(f"No hourly_prices rows for {args.ticker} in [{args.start}, {args.end})", file=sys.stderr)
        return 1

    csv_ticker = args.ticker if args.ticker.startswith("I:") else f"I:{args.ticker.upper()}"
    output_dir = Path(args.output_dir).resolve()

    written = 0
    skipped_existing = 0
    skipped_short = 0
    for date_str in sorted(by_date.keys()):
        bars = by_date[date_str]
        # Need at least a few bars to interpolate a usable trading day.
        if len(bars) < 3:
            skipped_short += 1
            continue

        out_path = _csv_path(args.ticker, output_dir, date_str)
        if out_path.exists() and not args.overwrite:
            skipped_existing += 1
            continue

        rows = _build_5min_rows(date_str, csv_ticker, bars)
        if not rows:
            continue
        if args.dry_run:
            print(f"  [dry-run] would write {len(rows)} bars → {out_path.name}")
        else:
            _write_csv(out_path, rows)
        written += 1

    print(f"\n{'Simulated' if args.dry_run else 'Wrote'} {written} CSV files to {output_dir}/{csv_ticker}/")
    if skipped_existing:
        print(f"Skipped {skipped_existing} dates whose CSV already existed (use --overwrite to replace)")
    if skipped_short:
        print(f"Skipped {skipped_short} dates with < 3 hourly bars (insufficient to interpolate)")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s RUT --start 2024-05-06 --end 2024-09-09
      Fill RUT 5-min CSVs for days before Polygon coverage starts.

  %(prog)s RUT --start 2024-05-06 --end 2024-09-09 --dry-run
      Preview without writing.

  %(prog)s RUT --start 2024-05-06 --end 2024-09-09 --overwrite
      Replace existing CSVs (default behavior preserves real Polygon files).
        """,
    )
    parser.add_argument("ticker", help="Ticker (RUT, NDX, SPX, ...). DB symbol is the bare name.")
    parser.add_argument("--start", required=True, help="Start date YYYY-MM-DD (inclusive)")
    parser.add_argument("--end", required=True, help="End date YYYY-MM-DD (exclusive)")
    parser.add_argument("--output-dir", default="./equities_output",
                        help="Root output directory (default: ./equities_output)")
    parser.add_argument("--overwrite", action="store_true",
                        help="Overwrite existing CSVs (default: skip them)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show what would be written, don't touch disk")

    return asyncio.run(main_async(parser.parse_args()))


if __name__ == "__main__":
    sys.exit(main())
