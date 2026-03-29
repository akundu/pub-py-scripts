"""VMaxMin Layer — Single-Day Simulator.

Simulates the full vMaxMin layer strategy for a single date, showing
every decision point in real-time order: entries, layers, EOD scans, rolls,
and expirations. Use this to validate against backtest results or to
preview what the strategy would do on a specific day.

Usage:
  python scripts/backtesting/scripts/run_vmaxmin_simulate.py \
      --ticker RUT --date 2026-03-20 --num-contracts 40 -v

  python scripts/backtesting/scripts/run_vmaxmin_simulate.py \
      --ticker RUT --date 2026-03-20 --num-contracts 1 --carries carries.json

Examples:
  # Simulate a single day with full detail
  %(prog)s --ticker RUT --date 2026-03-20 --num-contracts 40 -v

  # Simulate with carried positions from a prior day
  %(prog)s --ticker RUT --date 2026-03-13 --num-contracts 40 \
      --carry "put:2495/2490:5:8:2026-03-13:2026-03-10:100:0:3" -v

  # Quick check with 1 contract
  %(prog)s --ticker RUT --date 2026-03-18 --num-contracts 1
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from scripts.backtesting.scripts.vmaxmin_engine import (
    DEFAULT_CONFIG,
    TICKER_START_DATES,
    DayResult,
    RolledPosition,
    VMaxMinEngine,
    get_prev_close,
    get_trading_dates,
    load_0dte_options,
    load_equity_bars_df,
    _time_to_mins,
)


def parse_carry(s):
    """Parse a carry string: dir:short/long:width:contracts:exp:origin:orig_cr:cum_cost:roll_count"""
    parts = s.split(":")
    if len(parts) != 9:
        raise ValueError(f"Carry must have 9 colon-separated fields, got {len(parts)}: {s}")
    direction = parts[0]
    short_s, long_s = parts[1].split("/")
    return RolledPosition(
        direction=direction,
        short_strike=float(short_s),
        long_strike=float(long_s),
        width=float(parts[2]),
        num_contracts=int(parts[3]),
        expiration_date=parts[4],
        original_entry_date=parts[5],
        original_credit=float(parts[6]),
        cumulative_roll_cost=float(parts[7]),
        roll_count=int(parts[8]),
        credit_per_share=1.0,
    )


def main():
    parser = argparse.ArgumentParser(
        description="Simulate vMaxMin Layer strategy for a single day",
        epilog=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--ticker", default="RUT", help="Ticker (default: RUT)")
    parser.add_argument("--date", required=True, help="Trade date YYYY-MM-DD")
    parser.add_argument("--num-contracts", type=int, default=40, help="Contracts per position (default: 40)")
    parser.add_argument("--equity-dir", default="equities_output")
    parser.add_argument("--options-dir", default=None,
                        help="0DTE options dir (default: auto-detect)")
    parser.add_argument("--dte1-dir", default="csv_exports/options")
    parser.add_argument("--carry", action="append", default=[],
                        help="Carried position: dir:short/long:width:ctrs:exp:origin:orig_cr:cum_cost:roll#")
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    ticker = args.ticker
    trade_date = args.date

    # Auto-detect options dir
    if args.options_dir:
        opts_dir = args.options_dir
    else:
        # Try full_5/RUT/RUT first, then full_5/RUT, then full/RUT
        for candidate in [
            f"options_csv_output_full_5/{ticker}/{ticker}",
            f"options_csv_output_full_5/{ticker}",
            f"options_csv_output_full/{ticker}",
        ]:
            test_path = os.path.join(candidate, f"{ticker}_options_{trade_date}.csv")
            if os.path.exists(test_path):
                opts_dir = candidate
                break
        else:
            opts_dir = f"options_csv_output_full/{ticker}"

    config = {
        **DEFAULT_CONFIG,
        "strategy_mode": "layer",
        "num_contracts": args.num_contracts,
        "layer_dual_entry": True,
        "layer_entry_directions": "both",
        "max_roll_count": 99,
        "roll_recovery_threshold": 99.0,
        "roll_match_contracts": False,
        "roll_max_chain_contracts": None,
        "call_track_unlimited_budget": True,
        "call_track_check_times_pacific": ["08:35", "10:35"],
        "call_track_leg_placement": "best_roi",
        "call_track_depth_pct": 0.003,
        "layer_entry_window_start": "06:30",
        "layer_entry_window_end": "06:45",
        "layer_eod_scan_start": "12:50",
        "layer_eod_scan_end": "13:00",
        "layer_eod_proximity": 0.002,
        "equity_dir": args.equity_dir,
        "options_0dte_dir": os.path.dirname(opts_dir) if "/" in opts_dir else opts_dir,
        "options_dte1_dir": args.dte1_dir,
    }

    # Patch load functions if using nested dir
    import scripts.backtesting.scripts.vmaxmin_engine as eng
    _orig_load_0dte = eng.load_0dte_options
    _orig_load_dte1 = eng.load_dte1_options

    def patched_load_0dte(t, dt, options_dir):
        # Try nested path first
        import pandas as pd
        path = os.path.join(opts_dir, f"{t}_options_{dt}.csv")
        if os.path.exists(path):
            df = pd.read_csv(path)
            if not df.empty:
                df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
                from scripts.backtesting.scripts.vmaxmin_engine import _utc_to_pacific
                df["time_pacific"] = df["timestamp"].apply(_utc_to_pacific)
                for col in ["bid", "ask", "strike", "volume"]:
                    if col in df.columns:
                        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
                return df
        return _orig_load_0dte(t, dt, options_dir)

    def patched_load_dte1(t, exp_date, snap_date, csv_dir):
        result = _orig_load_dte1(t, exp_date, snap_date, csv_dir)
        if result is not None:
            return result
        # Fallback: next day's 0DTE
        return patched_load_0dte(t, exp_date, opts_dir)

    eng.load_0dte_options = patched_load_0dte
    eng.load_dte1_options = patched_load_dte1

    engine = VMaxMinEngine(config)
    all_dates = get_trading_dates(ticker, args.equity_dir, "2024-09-09", "2026-12-31")

    # Load data
    equity_df = load_equity_bars_df(ticker, trade_date, args.equity_dir)
    equity_prices = {}
    if not equity_df.empty:
        for _, row in equity_df.iterrows():
            equity_prices[row["time_pacific"]] = float(row["close"])

    options_0dte = patched_load_0dte(ticker, trade_date, opts_dir)
    prev_close = get_prev_close(ticker, trade_date, all_dates, args.equity_dir)

    # Parse carries
    carried = [parse_carry(c) for c in args.carry]

    # Run
    result, new_carries = engine.run_single_day(
        ticker, trade_date, equity_df, equity_prices,
        options_0dte, all_dates, prev_close,
        carried_positions=carried)

    # Restore
    eng.load_0dte_options = _orig_load_0dte
    eng.load_dte1_options = _orig_load_dte1

    # ── Output ──
    print(f"\n{'='*100}")
    print(f"  vMaxMin Layer Simulation — {ticker} {trade_date}")
    print(f"{'='*100}")
    print(f"  Open: {result.open_price:.1f}  Close: {result.close_price:.1f}  "
          f"HOD: {result.hod:.1f}  LOD: {result.lod:.1f}  "
          f"Range: {result.hod - result.lod:.1f} pts")
    print(f"  Prev close: {prev_close:.1f}" if prev_close else "  Prev close: N/A")
    print(f"  Contracts/position: {args.num_contracts}")
    if carried:
        print(f"  Carried in: {len(carried)} positions, "
              f"{sum(c.num_contracts for c in carried)} contracts")
    if result.failure_reason:
        print(f"  FAILURE: {result.failure_reason}")
    print()

    # Trade-by-trade timeline
    print(f"  {'Time':>5s}  {'Event':<20s}  {'Dir':>4s}  {'Strikes':>13s}  {'W':>3s}  "
          f"{'Ctrs':>4s}  {'Cr/Db':>10s}  {'Notes'}")
    print(f"  {'-'*90}")

    day_cr = 0
    for t in result.trades:
        cr_str = f"{'+'if t.credit_or_debit>=0 else ''}${t.credit_or_debit:,.0f}"
        print(f"  {t.time_pacific:>5s}  {t.event:<20s}  {t.direction:>4s}  "
              f"{t.short_strike:>6.0f}/{t.long_strike:>6.0f}  {t.width:>3.0f}  "
              f"x{t.num_contracts:>3d}  {cr_str:>10s}  {t.notes}")
        if t.event in ("entry", "layer_add"):
            day_cr += t.credit_or_debit

    # Summary
    print(f"\n  {'─'*90}")
    pnl = result.net_pnl
    print(f"  Credits: ${result.total_credits:>10,.0f}")
    print(f"  Debits:  ${result.total_debits:>10,.0f}")
    print(f"  Comms:   ${result.total_commissions:>10,.0f}")
    print(f"  Acct P&L:{'+'if pnl>=0 else ''}${pnl:>10,.0f}")
    print(f"  True P&L (entry+layer cr - comm): +${day_cr - result.total_commissions:>9,.0f}")
    print(f"  Rolls: {result.num_rolls}")

    if new_carries:
        print(f"\n  New carries → next day:")
        for rp in new_carries:
            print(f"    {rp.direction} {rp.short_strike}/{rp.long_strike} w=${rp.width:.0f} "
                  f"x{rp.num_contracts} exp={rp.expiration_date} "
                  f"roll#{rp.roll_count} cum_cost=${rp.cumulative_roll_cost:.0f}")
        total_exp = sum(rp.width * 100 * rp.num_contracts for rp in new_carries)
        print(f"    Total: {sum(rp.num_contracts for rp in new_carries)} contracts, "
              f"${total_exp:,.0f} exposure")
    else:
        print(f"\n  No carries — all positions settled")


if __name__ == "__main__":
    main()
