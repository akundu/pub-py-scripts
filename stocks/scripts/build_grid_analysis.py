#!/usr/bin/env python3
"""
Post-process comprehensive_backtest.py grid_summary.csv into grid_analysis_<suffix>.csv.

Steps:
  1. Read grid_summary.csv
  2. Apply max_width_cap filter (per-band per-DTE cap, matching comprehensive_backtest.py)
  3. Annotate with band descriptions, time buckets, n_contracts, ROI, scaled P&L
  4. Add pass/fail flags and composite `successful` column
  5. Write full grid_analysis CSV
  6. Write successful-only CSV (successful=True rows)

Usage:
  python scripts/build_grid_analysis.py \
      --in-dir  results/backtest_tight \
      --suffix  tight \
      --invest  30000 \
      [--min-win-rate 85] \
      [--min-roi 5] \
      [--min-trades 10]
"""

import argparse
import math
import sys
from pathlib import Path

import pandas as pd

# ── Band config (mirrors comprehensive_backtest.py) ───────────────────────────
BAND_DESCRIPTIONS = {
    'P95':  '2.5/97.5 pct of 100d moves',
    'P97':  '1.5/98.5 pct of 100d moves',
    'P98':  '1.0/99.0 pct of 100d moves',
    'P99':  '0.5/99.5 pct of 100d moves',
    'P100': '0.0/100.0 pct (min/max of 100d moves)',
}

BASE_MAX_WIDTH = {'P95': 30, 'P97': 30, 'P98': 40, 'P99': 50, 'P100': 50}


def max_width_for(band_name: str, dte: int) -> int:
    """Per-band max spread width, scaled by DTE (20% per day, capped at 3 DTE)."""
    base  = BASE_MAX_WIDTH.get(band_name, 30)
    scale = min(1.20 ** dte, 1.20 ** 3)   # cap scaling at 3 DTE
    return int(round(base * scale))


def get_time_bucket(time_et: str):
    """Map ET entry time to bucket ID and label."""
    h, m = map(int, time_et.split(':'))
    t = h * 60 + m
    if t <= 9 * 60 + 50:   # 09:30–09:50 ET = 06:30–06:50 PST
        return 'A', 'A: 06:30-06:50 PST / 09:30-09:50 ET (10-min)'
    elif t <= 11 * 60 + 45:  # 10:00–11:45 ET = 07:00–08:45 PST
        return 'B', 'B: 07:00-08:45 PST / 10:00-11:45 ET (15-min)'
    else:                    # 12:00+ ET = 09:00+ PST
        return 'C', 'C: 09:00-12:30 PST / 12:00-15:30 ET (30-min)'


def build_analysis(
    in_dir: Path,
    suffix: str,
    invest: float,
    min_win_rate: float,
    min_roi: float,
    min_trades: int,
) -> None:
    grid_path = in_dir / 'grid_summary.csv'
    if not grid_path.exists():
        print(f"ERROR: {grid_path} not found.", file=sys.stderr)
        sys.exit(1)

    print(f"Reading {grid_path} …")
    df = pd.read_csv(grid_path)
    print(f"  Loaded {len(df):,} rows.")

    # ── 1. Add band description ───────────────────────────────────────────────
    df['band_description'] = df['band'].map(BAND_DESCRIPTIONS).fillna('')

    # ── 2. Add max_width_cap and filter ──────────────────────────────────────
    df['max_width_cap'] = df.apply(
        lambda r: max_width_for(r['band'], r['dte']), axis=1
    )
    before = len(df)
    df = df[df['spread_width'] <= df['max_width_cap']].copy()
    print(f"  After max_width_cap filter: {len(df):,} rows (dropped {before - len(df):,}).")

    # ── 3. Time bucket ────────────────────────────────────────────────────────
    buckets = df['time_et'].apply(get_time_bucket)
    df['time_bucket_id'] = [b[0] for b in buckets]
    df['time_bucket']    = [b[1] for b in buckets]

    # ── 4. Contract sizing and scaled P&L ────────────────────────────────────
    df['n_contracts']    = (invest / df['avg_max_risk']).apply(math.floor)
    df['avg_credit_30k'] = (df['avg_credit'] * df['n_contracts']).round(2)
    df['roi_pct']        = (df['avg_credit'] / df['avg_max_risk'] * 100).round(2)
    df['avg_pnl_30k']    = (df['avg_pnl']    * df['n_contracts']).round(2)
    df['total_pnl_30k']  = (df['avg_pnl_30k'] * df['n_trades']).round(1)

    # ── 5. Pass/fail flags ────────────────────────────────────────────────────
    df['pass_win_rate']   = df['win_rate_pct'] >= min_win_rate
    df['pass_roi']        = df['roi_pct']       >= min_roi
    df['pass_min_trades'] = df['n_trades']      >= min_trades
    df['pass_sharpe']     = df['sharpe']        > 0
    df['pass_avg_pnl']    = df['avg_pnl']       > 0
    df['successful']      = (
        df['pass_win_rate'] & df['pass_roi'] & df['pass_min_trades'] &
        df['pass_sharpe']   & df['pass_avg_pnl']
    )

    # ── 6. Reorder columns (match existing grid_analysis format) ─────────────
    cols = [
        'dte', 'band', 'band_description', 'max_width_cap',
        'time_pst', 'time_et', 'time_bucket_id', 'time_bucket',
        'spread_type', 'flow_mode', 'spread_width',
        'n_trades', 'win_rate_pct', 'avg_credit', 'avg_max_risk',
        'n_contracts', 'avg_credit_30k', 'roi_pct',
        'avg_pnl', 'avg_pnl_30k', 'total_pnl_30k',
        'median_pnl', 'min_pnl', 'max_pnl', 'pnl_std', 'sharpe', 'avg_hold_days',
        'pass_win_rate', 'pass_roi', 'pass_min_trades', 'pass_sharpe', 'pass_avg_pnl',
        'successful',
    ]
    # Keep only columns that exist (in case grid_summary has extra/fewer cols)
    cols = [c for c in cols if c in df.columns]
    df = df[cols]

    # ── 7. Write full annotated CSV ───────────────────────────────────────────
    out_full = in_dir / f'grid_analysis_{suffix}.csv'
    df.to_csv(out_full, index=False)
    print(f"  Wrote {len(df):,} rows → {out_full}")

    # ── 8. Write successful-only CSV ──────────────────────────────────────────
    successful_df = df[df['successful']].copy()
    out_success = in_dir / f'grid_analysis_{suffix}_successful.csv'
    successful_df.to_csv(out_success, index=False)
    n_total = len(df)
    n_ok    = len(successful_df)
    print(f"  Wrote {n_ok:,}/{n_total:,} successful rows ({n_ok/n_total*100:.1f}%) → {out_success}")

    # ── 9. Quick summary ──────────────────────────────────────────────────────
    print()
    print("=== Success Criteria ===")
    print(f"  win_rate_pct >= {min_win_rate}%   pass: {df['pass_win_rate'].sum():,}")
    print(f"  roi_pct      >= {min_roi}%    pass: {df['pass_roi'].sum():,}")
    print(f"  n_trades     >= {min_trades}       pass: {df['pass_min_trades'].sum():,}")
    print(f"  sharpe       >  0           pass: {df['pass_sharpe'].sum():,}")
    print(f"  avg_pnl      >  0           pass: {df['pass_avg_pnl'].sum():,}")
    print(f"  ALL criteria met (successful): {n_ok:,} ({n_ok/n_total*100:.1f}%)")


def main():
    ap = argparse.ArgumentParser(description='Build grid_analysis CSV from grid_summary.csv')
    ap.add_argument('--in-dir',       type=Path, default=Path('results/backtest_tight'),
                    help='Directory containing grid_summary.csv (default: results/backtest_tight)')
    ap.add_argument('--suffix',       type=str,  default='tight',
                    help='Suffix for output files: grid_analysis_<suffix>.csv (default: tight)')
    ap.add_argument('--invest',       type=float, default=30_000,
                    help='$ deployed per trade for n_contracts / scaled P&L (default: 30000)')
    ap.add_argument('--min-win-rate', type=float, default=85.0,
                    help='Min win_rate_pct to pass (default: 85)')
    ap.add_argument('--min-roi',      type=float, default=5.0,
                    help='Min roi_pct to pass (default: 5)')
    ap.add_argument('--min-trades',   type=int,   default=10,
                    help='Min n_trades to pass (default: 10)')
    args = ap.parse_args()

    build_analysis(
        in_dir       = args.in_dir,
        suffix       = args.suffix,
        invest       = args.invest,
        min_win_rate = args.min_win_rate,
        min_roi      = args.min_roi,
        min_trades   = args.min_trades,
    )


if __name__ == '__main__':
    main()
