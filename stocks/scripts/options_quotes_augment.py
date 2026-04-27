#!/usr/bin/env python3
"""Augment per-trading-day options CSVs with quote-derived bars.

Status (2026-04-26): BACKFILL TOOL ONLY.
    `scripts/fetch_options.py` now does the same NBBO-quote-bars fetch
    natively when `--historical-mode auto` (the default) detects a past
    date. Going forward, the cron-driven nightly fetch produces complete
    chains in one pass — running this script is unnecessary for new dates.

    Keep this script around for two cases:
      1. Backfilling chains that were downloaded BEFORE the
         `fetch_options.py` historical-mode integration landed.
      2. Refreshing stale quote data for specific (ticker, date) windows
         without re-running the full fetcher.

Why this exists (original motivation):
    The existing options_chain_download.py uses trade aggregates to populate
    bid/ask. Strikes with no trades on a given day get no row, even though
    Polygon has continuous NBBO quotes for them. This is especially bad for
    NDX/RUT where many OTM strikes are quoted but rarely traded.

What this does:
    For each (ticker, trading_date) in the requested range:
      1. Read the existing CSV (if any).
      2. List Polygon options contracts whose strike falls in a price band
         around the underlying spot for that day (default: spot ± 5%).
      3. For each contract NOT already in the CSV (or any with a 0 bid in
         the CSV when --refresh-zero-bids is set), pull historical quotes
         for the trading day, resample to N-minute bars (median bid/ask
         within each bar), and emit rows in the existing CSV schema.
      4. Append new rows to the existing CSV. Re-sort and dedupe so the
         file remains canonical.

Designed to be re-runnable: skips strikes already covered, only filling
genuine gaps.

Example:
    python scripts/options_quotes_augment.py \\
        --ticker NDX --start 2026-03-01 --end 2026-04-23 \\
        --otm-low 5 --otm-high 1 --max-connections 8

    python scripts/options_quotes_augment.py \\
        --ticker SPX --start 2025-01-02 --end 2026-04-23 --interval 15 \\
        --max-expirations 8
"""
from __future__ import annotations

import argparse
import os
import sys
import time
import textwrap
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
import requests

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "scripts"))

# Reuse equity-close loader from the analyzer
import nroi_drift_analysis as nda  # noqa: E402

POLYGON_BASE = "https://api.polygon.io"
TRADING_DAY_START_UTC = "T13:30:00Z"
TRADING_DAY_END_UTC = "T20:30:00Z"
CSV_COLUMNS = [
    "timestamp", "ticker", "type", "strike", "expiration",
    "bid", "ask", "day_close", "vwap", "fmv",
    "delta", "gamma", "theta", "vega", "implied_volatility", "volume",
]


@dataclass
class AugmentArgs:
    ticker: str
    start: date
    end: date
    otm_low_pct: float       # e.g. 5.0 → spot * 0.95
    otm_high_pct: float      # e.g. 1.0 → spot * 1.01
    interval_minutes: int
    max_expirations: int
    max_connections: int
    side: str                # "put", "call", or "both"
    refresh_zero_bids: bool
    output_dir: Path
    equities_dir: Path
    api_key: str
    dry_run: bool
    verbose: bool


# ──────────────────────────────────────────────────────────────────────
# Polygon HTTP helpers
# ──────────────────────────────────────────────────────────────────────


def _get(url: str, params: Optional[Dict] = None,
         api_key: Optional[str] = None, timeout: int = 30) -> Optional[Dict]:
    """GET with simple retry; returns parsed JSON or None on failure."""
    if params is not None and api_key:
        params = {**params, "apiKey": api_key}
    elif api_key and "apiKey=" not in url:
        sep = "&" if "?" in url else "?"
        url = f"{url}{sep}apiKey={api_key}"
    for attempt in range(3):
        try:
            r = requests.get(url, params=params, timeout=timeout)
            if r.status_code == 200:
                return r.json()
            if r.status_code in (429, 502, 503, 504):
                time.sleep(2 ** attempt)
                continue
            return None
        except requests.RequestException:
            time.sleep(2 ** attempt)
    return None


def list_contracts(ticker: str, exp_start: date, exp_end: date,
                   side: str, strike_low: float, strike_high: float,
                   api_key: str, max_per_page: int = 1000) -> List[Dict]:
    """List Polygon options contracts matching a ticker + expiration range +
    strike band + side (put/call/both). Paginates."""
    contract_types = ["put", "call"] if side == "both" else [side]
    out: List[Dict] = []
    for ct in contract_types:
        url = f"{POLYGON_BASE}/v3/reference/options/contracts"
        params = {
            "underlying_ticker": ticker,
            "expiration_date.gte": exp_start.isoformat(),
            "expiration_date.lte": exp_end.isoformat(),
            "contract_type": ct,
            "strike_price.gte": strike_low,
            "strike_price.lte": strike_high,
            "limit": max_per_page,
            "expired": "true",
            "order": "asc",
            "sort": "strike_price",
        }
        next_url = None
        while True:
            data = _get(url, params=None if next_url else params,
                        api_key=api_key) if next_url else _get(url, params=params, api_key=api_key)
            if not data:
                break
            out.extend(data.get("results", []))
            next_url = data.get("next_url")
            if not next_url:
                break
            url = next_url
            params = None
    return out


PER_CONTRACT_TIMEOUT_SEC = 30.0  # bail on a single contract after this many seconds
MAX_PAGES_PER_CONTRACT = 6       # cap pagination depth to avoid 1.5M-quote tails


def fetch_quote_bars(contract_ticker: str, target_date: date,
                     interval_minutes: int, api_key: str) -> List[Dict]:
    """Pull all quotes for a contract on a date, resample to N-min bars (median
    bid/ask within each bar). Returns list of dicts ready for CSV insertion.

    Bails early if a single contract is taking longer than
    PER_CONTRACT_TIMEOUT_SEC (default 30s) — without this cap, one slow
    paginating contract can hold up an entire day's worker pool."""
    target_str = target_date.isoformat()
    url = f"{POLYGON_BASE}/v3/quotes/{contract_ticker}"
    params = {
        "timestamp.gte": f"{target_str}{TRADING_DAY_START_UTC}",
        "timestamp.lte": f"{target_str}{TRADING_DAY_END_UTC}",
        "limit": 50000,
        "order": "asc",
    }
    quotes: List[Dict] = []
    next_url = None
    pages = 0
    t0 = time.time()
    while True:
        if time.time() - t0 > PER_CONTRACT_TIMEOUT_SEC:
            break  # take what we have so far
        data = _get(url if not next_url else next_url,
                    params=None if next_url else params,
                    api_key=api_key, timeout=15)  # shorter per-request timeout too
        if not data:
            break
        for q in data.get("results", []):
            quotes.append({
                "ts_ns": q.get("sip_timestamp"),
                "bid_price": q.get("bid_price"),
                "ask_price": q.get("ask_price"),
            })
        next_url = data.get("next_url")
        pages += 1
        if not next_url or pages > MAX_PAGES_PER_CONTRACT:
            break
    if not quotes:
        return []
    df = pd.DataFrame(quotes)
    df["ts"] = pd.to_datetime(df["ts_ns"], unit="ns", utc=True)
    df = df.set_index("ts").sort_index()
    bars = (df.resample(f"{interval_minutes}min")
              .agg({"bid_price": "median", "ask_price": "median"})
              .dropna(subset=["bid_price", "ask_price"], how="all"))
    return [
        {"timestamp": ts.strftime("%Y-%m-%dT%H:%M:%S+00:00"),
         "bid": row["bid_price"],
         "ask": row["ask_price"]}
        for ts, row in bars.iterrows()
    ]


# ──────────────────────────────────────────────────────────────────────
# Core augment logic
# ──────────────────────────────────────────────────────────────────────


def existing_keys(df: pd.DataFrame) -> set:
    """Set of (timestamp, ticker, strike, type) tuples already in CSV.
    Used to dedupe. Strike normalized to int when possible."""
    if df is None or df.empty:
        return set()
    keys = set()
    for _, r in df.iterrows():
        try:
            strike = float(r.get("strike"))
        except (TypeError, ValueError):
            strike = r.get("strike")
        keys.add((str(r.get("timestamp")), str(r.get("ticker")),
                  strike, str(r.get("type"))))
    return keys


def existing_zero_bid_strikes(df: pd.DataFrame, side: str) -> set:
    """Strikes whose CSV rows ALL have bid<=0 — candidates for refresh."""
    if df is None or df.empty:
        return set()
    sub = df[df["type"].astype(str).str.lower() == side]
    if sub.empty: return set()
    sub = sub.copy()
    sub["bid_num"] = pd.to_numeric(sub["bid"], errors="coerce").fillna(0)
    by_strike = sub.groupby("strike")["bid_num"].max()
    return set(by_strike[by_strike <= 0].index)


def get_spot(ticker: str, d: date, equities_dir: Path) -> Optional[float]:
    closes = nda.load_daily_closes(ticker, d, equities_dir)
    if len(closes) < 2:
        return None
    return float(closes.iloc[-2])


def augment_one_date(args: AugmentArgs, d: date) -> Tuple[int, int, int]:
    """Augment one (ticker, date). Returns (rows_added, contracts_processed,
    contracts_skipped)."""
    ticker = args.ticker
    csv_path = args.output_dir / ticker / f"{ticker}_options_{d.isoformat()}.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    existing = pd.DataFrame()
    if csv_path.exists():
        try:
            existing = pd.read_csv(csv_path, low_memory=False)
        except Exception as e:
            if args.verbose:
                print(f"  [{d}] read fail: {e}", file=sys.stderr)

    spot = get_spot(ticker, d, args.equities_dir)
    if spot is None or spot <= 0:
        return 0, 0, 0
    strike_low  = spot * (1 - args.otm_low_pct / 100)
    strike_high = spot * (1 + args.otm_high_pct / 100)

    # List contracts: expirations from `d` through the next ~10 calendar days
    # (covers DTE 0..7 with a safety buffer)
    contracts = list_contracts(
        ticker=ticker,
        exp_start=d,
        exp_end=d + timedelta(days=12),
        side=args.side,
        strike_low=strike_low,
        strike_high=strike_high,
        api_key=args.api_key,
    )
    if not contracts:
        return 0, 0, 0

    # Cap to top-N expirations (to bound API calls)
    if args.max_expirations and args.max_expirations > 0:
        unique_exps = sorted(set(c["expiration_date"] for c in contracts))[:args.max_expirations]
        contracts = [c for c in contracts if c["expiration_date"] in unique_exps]

    # Determine which contracts to fetch
    existing_tickers = set()
    if not existing.empty:
        existing_tickers = set(existing["ticker"].astype(str).unique())
    zero_bid_strikes = (
        existing_zero_bid_strikes(existing, args.side) if args.refresh_zero_bids and args.side != "both"
        else set()
    )

    todo: List[Dict] = []
    for c in contracts:
        ct = c["ticker"]
        s = c["strike_price"]
        if ct not in existing_tickers:
            todo.append(c)
        elif args.refresh_zero_bids and s in zero_bid_strikes:
            todo.append(c)

    if not todo:
        return 0, 0, len(contracts)

    if args.dry_run:
        if args.verbose:
            print(f"  [{d}] would fetch {len(todo)} contracts "
                  f"(strike band {strike_low:.0f}-{strike_high:.0f})", file=sys.stderr)
        return 0, len(todo), len(contracts) - len(todo)

    # Parallel quote fetches
    rows: List[Dict] = []
    def _work(c):
        bars = fetch_quote_bars(c["ticker"], d, args.interval_minutes, args.api_key)
        return c, bars

    with ThreadPoolExecutor(max_workers=args.max_connections) as ex:
        futures = {ex.submit(_work, c): c for c in todo}
        for fut in as_completed(futures):
            try:
                c, bars = fut.result()
            except Exception:
                continue
            for b in bars:
                rows.append({
                    "timestamp": b["timestamp"],
                    "ticker": c["ticker"],
                    "type": c["contract_type"],
                    "strike": c["strike_price"],
                    "expiration": c["expiration_date"],
                    "bid": b["bid"],
                    "ask": b["ask"],
                    "day_close": "",
                    "vwap": "",
                    "fmv": "",
                    "delta": "",
                    "gamma": "",
                    "theta": "",
                    "vega": "",
                    "implied_volatility": "",
                    "volume": "",
                })

    if not rows:
        return 0, len(todo), len(contracts) - len(todo)

    new_df = pd.DataFrame(rows, columns=CSV_COLUMNS)
    if existing.empty:
        merged = new_df
    else:
        # Align column shape, then concat + dedupe
        for col in CSV_COLUMNS:
            if col not in existing.columns:
                existing[col] = ""
        existing = existing[CSV_COLUMNS]
        merged = pd.concat([existing, new_df], ignore_index=True)
        merged = merged.drop_duplicates(
            subset=["timestamp", "ticker", "type", "strike", "expiration"],
            keep="last",  # prefer newly-fetched (quote-derived) rows
        )
    merged = merged.sort_values(["timestamp", "strike", "type"]).reset_index(drop=True)
    merged.to_csv(csv_path, index=False)

    return len(rows), len(todo), len(contracts) - len(todo)


# ──────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────


def parse_args() -> AugmentArgs:
    api_key = os.getenv("POLYGON_API_KEY")
    parser = argparse.ArgumentParser(
        description=textwrap.dedent("""
            Augment options CSVs with quote-derived bars for strikes that
            were never traded but had continuous NBBO quotes. Fixes the
            sparsity caused by the existing trade-aggregate-only download.

            For each trading day, lists Polygon contracts in a strike band
            around spot, fetches quotes for those NOT already in the CSV,
            resamples to N-minute bars (median bid/ask), and appends rows.
        """).strip(),
        epilog=textwrap.dedent("""
            Examples:
              %(prog)s --ticker NDX --start 2026-03-01 --end 2026-04-23
                  Last 8 weeks of NDX, default 5%% OTM put band.

              %(prog)s --ticker SPX --start 2025-01-02 --end 2026-04-23 \\
                       --otm-low 5 --otm-high 1 --interval 15 --max-connections 8
                  Full 16-month SPX augmentation.

              %(prog)s --ticker NDX --start 2026-04-21 --end 2026-04-21 --dry-run
                  Show what would be fetched without making API calls.
        """).strip(),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--ticker", required=True,
                        help="Underlying symbol (SPX, RUT, NDX).")
    parser.add_argument("--start", required=True, type=date.fromisoformat,
                        help="Start trading date (YYYY-MM-DD, inclusive).")
    parser.add_argument("--end", required=True, type=date.fromisoformat,
                        help="End trading date (YYYY-MM-DD, inclusive).")
    parser.add_argument("--otm-low", type=float, default=5.0,
                        help="Lower bound: strikes >= spot * (1 - X/100). Default 5.")
    parser.add_argument("--otm-high", type=float, default=1.0,
                        help="Upper bound: strikes <= spot * (1 + X/100). Default 1.")
    parser.add_argument("--interval", type=int, default=15,
                        dest="interval_minutes",
                        help="Bar interval in minutes. Default 15.")
    parser.add_argument("--max-expirations", type=int, default=8,
                        help="Max distinct expirations per trading day. Default 8.")
    parser.add_argument("--max-connections", type=int, default=8,
                        help="Parallel API connections. Default 8.")
    parser.add_argument("--side", default="put", choices=["put", "call", "both"],
                        help="Option side. Default put.")
    parser.add_argument("--refresh-zero-bids", action="store_true",
                        help="Also re-fetch strikes whose existing rows have bid<=0.")
    parser.add_argument("--output-dir", type=Path,
                        default=_REPO_ROOT / "options_csv_output_full",
                        help="Output dir. Default options_csv_output_full.")
    parser.add_argument("--equities-dir", type=Path,
                        default=_REPO_ROOT / "equities_output",
                        help="Equities dir. Default equities_output.")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("-v", "--verbose", action="store_true")
    ns = parser.parse_args()

    if not api_key:
        parser.error("POLYGON_API_KEY environment variable is required.")

    return AugmentArgs(
        ticker=ns.ticker.upper(),
        start=ns.start, end=ns.end,
        otm_low_pct=ns.otm_low, otm_high_pct=ns.otm_high,
        interval_minutes=ns.interval_minutes,
        max_expirations=ns.max_expirations,
        max_connections=ns.max_connections,
        side=ns.side,
        refresh_zero_bids=ns.refresh_zero_bids,
        output_dir=ns.output_dir,
        equities_dir=ns.equities_dir,
        api_key=api_key,
        dry_run=ns.dry_run,
        verbose=ns.verbose,
    )


def main() -> int:
    args = parse_args()
    print(f"Augmenting {args.ticker}  ·  {args.start} → {args.end}  "
          f"·  band {args.otm_low_pct}% OTM (low) / {args.otm_high_pct}% OTM (high)")
    print(f"Output: {args.output_dir}/{args.ticker}/")
    if args.dry_run:
        print("DRY RUN — no quotes will be fetched.\n")

    dates = pd.bdate_range(args.start, args.end).date.tolist()
    total_added = total_fetched = total_skipped = 0
    t_start = time.time()
    for i, d in enumerate(dates):
        try:
            added, fetched, skipped = augment_one_date(args, d)
        except KeyboardInterrupt:
            print("\nInterrupted.")
            break
        except Exception as e:
            print(f"  [{d}] ERROR: {e}", file=sys.stderr)
            continue
        total_added += added; total_fetched += fetched; total_skipped += skipped
        elapsed = time.time() - t_start
        rate = (i + 1) / elapsed if elapsed > 0 else 0
        eta = (len(dates) - i - 1) / rate if rate > 0 else 0
        print(f"  [{d}] +{added:>4d} rows  ({fetched} fetched, "
              f"{skipped} already-covered) | "
              f"{i+1}/{len(dates)}  rate {rate*60:.1f}/min  ETA {eta/60:.0f} min")

    print(f"\nDone. Added {total_added:,} rows across {len(dates)} dates "
          f"({total_fetched} contracts fetched, {total_skipped} skipped).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
