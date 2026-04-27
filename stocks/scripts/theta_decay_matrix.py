"""Theta-decay matrix: for each (ticker, side, DTE), report the cumulative
$/contract kept and % of entry credit captured at end-of-each-day-of-life.

Uses entries from results/nroi_drift_*/raw/records.parquet and re-prices the
same (short, long, expiration) spread at the 15:30 ET snapshot of every
calendar day from entry day through expiration day.

See `docs/strategies/nroi_theta_decay_playbook.md` for the full playbook —
data sources, regime classification, trade rules, and non-obvious gotchas.

Quick reference — D0 capture % (DTE 1 puts, moderate, 12-mo median):
    RUT 56% → close at D0 EOD       (fast same-day decay)
    SPX 43% → hold to D1 close      (symmetric, predictable)
    NDX  7% → hold to expiration    (slowest; don't scalp)

Reproduce:
    # Calls sweep aligned to puts' window first
    python3 scripts/nroi_drift_analysis.py \\
        --start 2025-04-23 --end 2026-04-23 \\
        --tickers SPX:25,RUT:25,NDX:60 \\
        --dtes 0,1,2,3,5 --tiers aggressive,moderate,conservative \\
        --sides call --workers 8 --primary-source full_dir \\
        --output-dir results/nroi_drift_calls_12mo

    # Then matrix per tier
    python3 scripts/theta_decay_matrix.py \\
        --dtes 0,1,2,5 --start 2025-04-23 --end 2026-04-23 \\
        --tier moderate \\
        --calls-parquet results/nroi_drift_calls_12mo/raw/records.parquet
"""
from __future__ import annotations

import argparse
from datetime import date, timedelta
from pathlib import Path
import pandas as pd
import sys

REPO = Path(__file__).resolve().parent.parent
EOD_UTC = "19:30:00"  # 15:30 ET ≈ 19:30 UTC during DST window of interest
ENTRY_UTC = "13:45:00"  # 09:45 ET — kept for symmetry; we already have entry credits


def load_entries(parquets: list[Path], dtes: list[int], sides: list[str],
                 tier: str = "moderate") -> pd.DataFrame:
    frames = [pd.read_parquet(p) for p in parquets if p.exists()]
    df = pd.concat(frames, ignore_index=True)
    df = df[(df["tier"] == tier)
            & (df["hour_et"] == 9)
            & df["short_strike"].notna()
            & df["dte"].isin(dtes)
            & df["side"].isin(sides)
            & df["net_credit"].notna()
            & (df["net_credit"] > 0)].copy()
    df["date"] = pd.to_datetime(df["date"]).dt.date
    df["expiration"] = df.apply(
        lambda r: r["date"] + timedelta(days=int(r["dte"])), axis=1
    )
    return df.reset_index(drop=True)


def load_chain(ticker: str, day: date, full_dir: Path) -> pd.DataFrame | None:
    p = full_dir / ticker / f"{ticker}_options_{day.isoformat()}.csv"
    if not p.exists():
        return None
    df = pd.read_csv(p, low_memory=False, usecols=[
        "timestamp", "type", "strike", "expiration", "bid", "ask"
    ])
    df["ts"] = pd.to_datetime(df["timestamp"], errors="coerce", utc=True)
    df = df.dropna(subset=["ts"])
    # Bound-check: drop out-of-range datetimes that pandas can't .date() on
    df = df[(df["ts"] >= "2020-01-01") & (df["ts"] <= "2030-12-31")].copy()
    df["minute_of_day"] = df["ts"].dt.hour * 60 + df["ts"].dt.minute
    return df


def snap_value(chain: pd.DataFrame, target_utc: str, side: str,
               short_k: float, long_k: float, exp: date) -> float | None:
    """Compute close-out cost of the spread at the given UTC snapshot.
    Returns ask_short - bid_long.
    """
    if chain is None or chain.empty:
        return None
    target_h, target_m, _ = target_utc.split(":")
    minute_of_day = int(target_h) * 60 + int(target_m)
    exp_str = exp.isoformat()
    sub = chain[(chain["expiration"] == exp_str) & (chain["type"] == side)]
    if sub.empty:
        return None
    sub = sub.copy()
    sub["abs_dt"] = (sub["minute_of_day"] - minute_of_day).abs()
    short_rows = sub[sub["strike"] == short_k].sort_values("abs_dt")
    long_rows  = sub[sub["strike"] == long_k].sort_values("abs_dt")
    if short_rows.empty or long_rows.empty:
        return None
    s = short_rows.iloc[0]
    l = long_rows.iloc[0]
    s_ask = s["ask"] if pd.notna(s["ask"]) and s["ask"] > 0 else (
        s["bid"] if pd.notna(s["bid"]) else None)
    l_bid = l["bid"] if pd.notna(l["bid"]) and l["bid"] > 0 else (
        l["ask"] if pd.notna(l["ask"]) else None)
    if s_ask is None or l_bid is None:
        return None
    return float(s_ask) - float(l_bid)


def settlement_loss(side: str, short_k: float, long_k: float,
                    underlying_close: float | None, width: float) -> float | None:
    """At expiration, intrinsic value of debt to close (short - long)."""
    if underlying_close is None:
        return None
    if side == "put":
        s = max(0.0, short_k - underlying_close)
        l = max(0.0, long_k  - underlying_close)
    else:  # call
        s = max(0.0, underlying_close - short_k)
        l = max(0.0, underlying_close - long_k)
    return min(s - l, width)  # bounded by width


def underlying_close(ticker: str, day: date, equities_dir: Path) -> float | None:
    p = equities_dir / ticker / f"{ticker}_equities_{day.isoformat()}.csv"
    if not p.exists():
        # try with I: prefix
        p2 = equities_dir / f"I:{ticker}" / f"I:{ticker}_equities_{day.isoformat()}.csv"
        if p2.exists():
            p = p2
        else:
            return None
    try:
        df = pd.read_csv(p, low_memory=False)
        if "close" not in df.columns:
            return None
        # Last row of the day
        return float(df["close"].dropna().iloc[-1])
    except Exception:
        return None


def build_matrix(entries: pd.DataFrame, full_dir: Path, equities_dir: Path,
                 verbose: bool = False) -> pd.DataFrame:
    """Returns long-form: ticker, side, dte, day_offset, n, mean_kept_dollars,
    mean_pct_captured, mean_entry_credit."""
    rows = []
    chain_cache: dict[tuple[str, str], pd.DataFrame | None] = {}

    for idx, e in entries.iterrows():
        if verbose and idx % 500 == 0:
            print(f"  entry {idx}/{len(entries)}", file=sys.stderr)
        tk = e["ticker"]
        side = e["side"]
        dte = int(e["dte"])
        entry_credit = float(e["net_credit"])
        width = float(e["width"])
        sk = float(e["short_strike"])
        lk = float(e["long_strike"])
        exp = e["expiration"]
        # Day offsets: 0 .. dte
        for offset in range(dte + 1):
            d = e["date"] + timedelta(days=offset)
            # Skip weekends — markets closed
            if d.weekday() >= 5:
                continue
            cache_key = (tk, d.isoformat())
            if cache_key not in chain_cache:
                chain_cache[cache_key] = load_chain(tk, d, full_dir)
            chain = chain_cache[cache_key]
            close_cost = snap_value(chain, EOD_UTC, side, sk, lk, exp)
            # If this is expiration day and we have no chain quote, fall back
            # to settlement value vs underlying close
            if close_cost is None and offset == dte:
                u = underlying_close(tk, d, equities_dir)
                close_cost = settlement_loss(side, sk, lk, u, width)
            if close_cost is None:
                continue
            kept = entry_credit - close_cost
            # bound below at -width (worst case if breached deep ITM)
            kept = max(kept, -width)
            pct = kept / entry_credit * 100.0 if entry_credit > 0 else 0.0
            rows.append({
                "ticker": tk, "side": side, "dte": dte,
                "day_offset": offset,
                "entry_credit": entry_credit,
                "kept": kept,
                "pct": pct,
                "is_expiration": (offset == dte),
            })
    return pd.DataFrame(rows)


def render_matrix(df: pd.DataFrame, dtes: list[int]):
    """Print the matrix grouped by ticker × side."""
    if df.empty:
        print("No data to render.")
        return
    tickers = sorted(df["ticker"].unique())
    sides = ["put", "call"]
    for tk in tickers:
        print(f"\n══════ {tk} ══════")
        for side in sides:
            sub = df[(df["ticker"] == tk) & (df["side"] == side)]
            if sub.empty:
                print(f"  ── {side.upper()}S: no records")
                continue
            print(f"  ── {side.upper()}S ──")
            # Per-DTE summary rows
            n_per_dte = (sub.groupby('dte')['day_offset'].apply(
                lambda s: int(sub.loc[s.index].pipe(lambda d: d[d['day_offset']==0]).shape[0])
            ))
            # Build header up to max DTE in scope
            max_dte = max(dtes)
            header = f"  {'DTE':>3} | {'n':>4} | {'entry $':>8} |"
            for d in range(max_dte + 1):
                header += f"   {'D'+str(d):>14} |"
            print(header)
            print("  " + "-" * (len(header) - 2))
            for dte in dtes:
                sd = sub[sub["dte"] == dte]
                if sd.empty:
                    continue
                n_entries = sd[sd["day_offset"] == 0]["kept"].count()
                if n_entries == 0:
                    n_entries = sd[sd["day_offset"] == sd["day_offset"].min()]["kept"].count()
                entry_credit_med = sd["entry_credit"].median()
                line = f"  {dte:>3} | {int(n_entries):>4} | $ {entry_credit_med:>5.3f} |"
                for d in range(max_dte + 1):
                    if d > dte:
                        line += f"   {'—':>14} |"
                        continue
                    cell = sd[sd["day_offset"] == d]
                    if cell.empty:
                        line += f"   {'—':>14} |"
                        continue
                    kept_dollars = cell["kept"].median() * 100  # $/contract
                    pct = cell["pct"].median()
                    star = "*" if d == dte else " "
                    line += f"   $ {kept_dollars:>5.0f} ({pct:>4.0f}%){star}|"
                print(line)
            print()
    print("  '*' = expiration day (last full day held).")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--puts-parquets", nargs="+", default=[
        "results/nroi_drift_16mo/raw/records.parquet",
        "results/nroi_drift_dte3/raw/records.parquet",
    ])
    p.add_argument("--calls-parquet", default="results/nroi_drift_calls/raw/records.parquet")
    p.add_argument("--full-dir", default="options_csv_output_full")
    p.add_argument("--equities-dir", default="equities_output")
    p.add_argument("--dtes", default="1,2", help="Comma-separated DTEs to analyze.")
    p.add_argument("--start", default=None, help="Optional entry-date floor.")
    p.add_argument("--end", default=None, help="Optional entry-date ceiling.")
    p.add_argument("--tier", default="moderate", choices=["aggressive", "moderate", "conservative"])
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args()

    dtes = sorted(int(x) for x in args.dtes.split(","))
    full_dir = REPO / args.full_dir
    equities_dir = REPO / args.equities_dir

    # Puts: load from put parquets (16-mo + dte3)
    put_paths = [REPO / x for x in args.puts_parquets]
    puts = load_entries(put_paths, dtes, ["put"], tier=args.tier)
    # Calls: load from calls parquet (2-mo only)
    call_paths = [REPO / args.calls_parquet]
    calls = load_entries(call_paths, dtes, ["call"], tier=args.tier)

    if args.start:
        s = pd.to_datetime(args.start).date()
        puts = puts[puts["date"] >= s]
        calls = calls[calls["date"] >= s]
    if args.end:
        e = pd.to_datetime(args.end).date()
        puts = puts[puts["date"] <= e]
        calls = calls[calls["date"] <= e]

    print(f"Puts entries:  {len(puts)} (window {puts['date'].min() if len(puts) else '—'} → {puts['date'].max() if len(puts) else '—'})")
    print(f"Calls entries: {len(calls)} (window {calls['date'].min() if len(calls) else '—'} → {calls['date'].max() if len(calls) else '—'})")

    df_puts = build_matrix(puts, full_dir, equities_dir, args.verbose)
    df_calls = build_matrix(calls, full_dir, equities_dir, args.verbose)
    df = pd.concat([df_puts, df_calls], ignore_index=True)

    print("=" * 90)
    print("CUMULATIVE VALUE PROCURED — same spread held through end-of-day on each day of life")
    print("Per-cell metrics: $/contract kept (cumulative) and % of entry credit captured")
    print("=" * 90)
    render_matrix(df, dtes)


if __name__ == "__main__":
    main()
