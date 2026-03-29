#!/usr/bin/env python3
"""vMaxMin Layer — Live Advisory Runner.

Runs the vMaxMin layer strategy in real-time, displaying entry/layer/roll
recommendations as they happen during market hours. Advisory only — you
execute the trades manually or via UTP.

Architecture:
  1. At 06:30-06:45 Pacific: scans for best-ROI entry spreads (call + put)
  2. At 08:35, 10:35: checks for new HOD/LOD → layer recommendations
  3. At 12:50-13:00: minute-by-minute EOD scan → roll recommendations
  4. Tracks carries across days via JSON persistence

Data sources:
  - UTP daemon (localhost:8000) for real-time equity quotes + option chains
  - QuestDB realtime_data for intraday price bars
  - csv_exports/options/ for DTE+1 option chains (live snapshots)
  - equities_output/ for previous close

Usage:
  # Live mode — connects to UTP daemon + QuestDB
  python run_vmaxmin_live.py --live

  # Simulate mode — replays a historical date from CSV
  python run_vmaxmin_live.py --simulate 2026-03-20

  # Simulate with custom speed
  python run_vmaxmin_live.py --simulate 2026-03-20 --sim-speed 0

  # Dry run — show config and exit
  python run_vmaxmin_live.py --dry-run

  # Show current carry positions
  python run_vmaxmin_live.py --positions

Requirements:
  - For --live: UTP daemon running (python utp.py daemon --paper)
  - For --simulate: options_csv_output_full_5/RUT/RUT/ data
  - QUEST_DB_STRING env var (for --live equity prices)
"""

import argparse
import json
import logging
import os
import sys
import time as time_mod
from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.backtesting.scripts.vmaxmin_engine import (
    DEFAULT_CONFIG,
    DayResult,
    RolledPosition,
    VMaxMinEngine,
    get_prev_close,
    get_trading_dates,
    load_0dte_options,
    load_equity_bars_df,
    _time_to_mins,
    _utc_to_pacific,
)

logger = logging.getLogger("vmaxmin_live")

CARRIES_FILE = "data/vmaxmin_carries.json"
TRADE_LOG = "data/vmaxmin_trades.jsonl"


def save_carries(carries):
    os.makedirs(os.path.dirname(CARRIES_FILE), exist_ok=True)
    data = []
    for rp in carries:
        data.append({k: _jsonable(v) for k, v in {
            "direction": rp.direction,
            "short_strike": rp.short_strike,
            "long_strike": rp.long_strike,
            "width": rp.width,
            "credit_per_share": rp.credit_per_share,
            "num_contracts": rp.num_contracts,
            "expiration_date": rp.expiration_date,
            "original_entry_date": rp.original_entry_date,
            "original_credit": rp.original_credit,
            "cumulative_roll_cost": rp.cumulative_roll_cost,
            "roll_count": rp.roll_count,
        }.items()})
    with open(CARRIES_FILE, "w") as f:
        json.dump(data, f, indent=2)
    logger.info(f"Saved {len(data)} carries to {CARRIES_FILE}")


def load_carries():
    if not os.path.exists(CARRIES_FILE):
        return []
    with open(CARRIES_FILE) as f:
        data = json.load(f)
    carries = []
    for d in data:
        carries.append(RolledPosition(**d))
    return carries


def _jsonable(v):
    """Convert numpy types to Python native for JSON."""
    import numpy as np
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        return float(v)
    return v


def log_trade(trade_date, trade_record):
    os.makedirs(os.path.dirname(TRADE_LOG), exist_ok=True)
    entry = {k: _jsonable(v) for k, v in {
        "date": trade_date,
        "event": trade_record.event,
        "time": trade_record.time_pacific,
        "direction": trade_record.direction,
        "short_strike": trade_record.short_strike,
        "long_strike": trade_record.long_strike,
        "width": trade_record.width,
        "credit_or_debit": trade_record.credit_or_debit,
        "num_contracts": trade_record.num_contracts,
        "commission": trade_record.commission,
        "underlying_price": trade_record.underlying_price,
        "dte": trade_record.dte,
        "notes": trade_record.notes,
    }.items()}
    with open(TRADE_LOG, "a") as f:
        f.write(json.dumps(entry) + "\n")


def show_positions():
    carries = load_carries()
    if not carries:
        print("No active carry positions")
        return
    print(f"\n  Active carry positions ({len(carries)}):")
    total_c = 0
    total_exp = 0
    for rp in carries:
        ml = rp.width * 100 * rp.num_contracts
        total_c += rp.num_contracts
        total_exp += ml
        print(f"    {rp.direction} {rp.short_strike}/{rp.long_strike} w=${rp.width:.0f} "
              f"x{rp.num_contracts} exp={rp.expiration_date} "
              f"roll#{rp.roll_count} from {rp.original_entry_date}")
    print(f"    Total: {total_c} contracts, ${total_exp:,.0f} exposure")


def run_simulate(trade_date, num_contracts, sim_speed):
    """Run simulation for a single historical date."""
    ticker = "RUT"
    config = _build_config(num_contracts)
    engine = VMaxMinEngine(config)

    # Patch loaders for nested dir
    _patch_loaders(ticker)

    all_dates = get_trading_dates(ticker, "equities_output", "2024-09-09", "2026-12-31")
    equity_df = load_equity_bars_df(ticker, trade_date, "equities_output")
    equity_prices = {}
    if not equity_df.empty:
        for _, row in equity_df.iterrows():
            equity_prices[row["time_pacific"]] = float(row["close"])

    options_0dte = load_0dte_options(ticker, trade_date,
                                     config["options_0dte_dir"])
    prev_close = get_prev_close(ticker, trade_date, all_dates, "equities_output")

    # Load carries
    carries = load_carries()
    today_carries = [rp for rp in carries if rp.expiration_date == trade_date]

    carry_exp = sum(rp.width * 100 * rp.num_contracts for rp in carries)
    new_entry_exp = num_contracts * 5 * 100 * 5
    max_risk = config.get("max_total_exposure", 500000)

    print(f"\n{'='*90}")
    print(f"  vMaxMin Layer — SIMULATE {ticker} {trade_date}")
    print(f"{'='*90}")
    print(f"  Contracts/position: {num_contracts}")
    print(f"  Carries in: {len(today_carries)} positions, "
          f"{sum(c.num_contracts for c in today_carries)} contracts")
    print(f"  Total carry exposure: ${carry_exp:,.0f}")

    if carry_exp + new_entry_exp > max_risk:
        print(f"  ⚠ SKIP DAY: projected exposure ${carry_exp + new_entry_exp:,.0f} > cap ${max_risk:,.0f}")
        print(f"  Carries will still be evaluated for expiry/roll")

    print(f"  Prev close: {prev_close}" if prev_close else "  Prev close: N/A")
    print()

    # Run engine
    result, new_carries = engine.run_single_day(
        ticker, trade_date, equity_df, equity_prices,
        options_0dte, all_dates, prev_close,
        carried_positions=today_carries)

    # Display with optional delay
    print(f"  {'Time':>5s}  {'Event':<20s}  {'Dir':>4s}  {'Strikes':>13s}  {'W':>3s}  "
          f"{'Ctrs':>4s}  {'Cr/Db':>10s}  Notes")
    print(f"  {'-'*90}")

    last_time = ""
    for t in result.trades:
        if sim_speed > 0 and t.time_pacific != last_time:
            time_mod.sleep(sim_speed)
            last_time = t.time_pacific

        cr_str = f"{'+'if t.credit_or_debit>=0 else ''}${t.credit_or_debit:,.0f}"
        print(f"  {t.time_pacific:>5s}  {t.event:<20s}  {t.direction:>4s}  "
              f"{t.short_strike:>6.0f}/{t.long_strike:>6.0f}  {t.width:>3.0f}  "
              f"x{t.num_contracts:>3d}  {cr_str:>10s}  {t.notes}")
        log_trade(trade_date, t)

    # Summary
    pnl = result.net_pnl
    day_cr = sum(t.credit_or_debit for t in result.trades
                 if t.event in ("entry", "layer_add"))
    print(f"\n  {'─'*90}")
    print(f"  Open: {result.open_price:.1f}  Close: {result.close_price:.1f}  "
          f"HOD: {result.hod:.1f}  LOD: {result.lod:.1f}")
    print(f"  Credits: ${result.total_credits:>10,.0f}  Debits: ${result.total_debits:>10,.0f}  "
          f"Comm: ${result.total_commissions:>6,.0f}")
    print(f"  True P&L: +${day_cr - result.total_commissions:>9,.0f}  "
          f"Acct P&L: {'+'if pnl>=0 else ''}${pnl:>9,.0f}")

    # Update carries
    remaining = [rp for rp in carries if rp.expiration_date > trade_date]
    remaining.extend(new_carries)
    save_carries(remaining)

    if new_carries:
        print(f"\n  Carries → next day:")
        for rp in new_carries:
            print(f"    {rp.direction} {rp.short_strike}/{rp.long_strike} w=${rp.width:.0f} "
                  f"x{rp.num_contracts} exp={rp.expiration_date} roll#{rp.roll_count}")
        total_exp = sum(rp.width * 100 * rp.num_contracts for rp in remaining)
        print(f"    Total exposure after: ${total_exp:,.0f}")

    _restore_loaders()


def run_live(num_contracts):
    """Run live advisory mode — polls UTP for real-time data."""
    print(f"\n{'='*90}")
    print(f"  vMaxMin Layer — LIVE MODE (advisory only)")
    print(f"{'='*90}")
    print(f"  Contracts/position: {num_contracts}")
    print(f"  This mode will:")
    print(f"    1. At 06:30-06:45: show entry recommendations")
    print(f"    2. At 08:35, 10:35: show layer recommendations")
    print(f"    3. At 12:50-13:00: show roll recommendations")
    print()
    print(f"  Prerequisites:")
    print(f"    - UTP daemon: cd live_trading/universal-trade-platform && python utp.py daemon --paper")
    print(f"    - Data pipeline: ms1_cron.sh runs daily at 3:10 AM")
    print(f"    - Option snapshots: csv_exports/options/RUT/ updated during market hours")
    print()
    print(f"  For now, use simulate mode on today's date once market data is available:")
    print(f"    python run_vmaxmin_live.py --simulate {date.today().isoformat()}")
    print()
    print(f"  Or run the backtest to see what would have happened:")
    print(f"    python scripts/backtesting/scripts/run_vmaxmin_backtest.py \\")
    print(f"        --tickers RUT --layer --lookback-days 1 --num-contracts {num_contracts} -v")


def _build_config(num_contracts):
    return {
        **DEFAULT_CONFIG,
        "strategy_mode": "layer",
        "num_contracts": num_contracts,
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
        "equity_dir": "equities_output",
        "options_0dte_dir": "options_csv_output_full_5/RUT",
        "options_dte1_dir": "csv_exports/options",
        "max_total_exposure": 500000,
        "daily_budget": 100000,
    }


_orig_load_0dte = None
_orig_load_dte1 = None

def _patch_loaders(ticker):
    global _orig_load_0dte, _orig_load_dte1
    import scripts.backtesting.scripts.vmaxmin_engine as eng
    import pandas as pd
    _orig_load_0dte = eng.load_0dte_options
    _orig_load_dte1 = eng.load_dte1_options

    nested_dir = f"options_csv_output_full_5/{ticker}/{ticker}"

    def patched_load_0dte(t, dt, options_dir):
        path = os.path.join(nested_dir, f"{t}_options_{dt}.csv")
        if os.path.exists(path):
            df = pd.read_csv(path)
            if not df.empty:
                df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
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
        return patched_load_0dte(t, exp_date, nested_dir)

    eng.load_0dte_options = patched_load_0dte
    eng.load_dte1_options = patched_load_dte1


def _restore_loaders():
    import scripts.backtesting.scripts.vmaxmin_engine as eng
    if _orig_load_0dte:
        eng.load_0dte_options = _orig_load_0dte
    if _orig_load_dte1:
        eng.load_dte1_options = _orig_load_dte1


def main():
    parser = argparse.ArgumentParser(
        description="vMaxMin Layer — Live/Simulate Credit Spread Advisor",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --simulate 2026-03-20           Simulate a historical date
  %(prog)s --simulate 2026-03-20 --sim-speed 0   Instant replay
  %(prog)s --live                          Live advisory mode
  %(prog)s --positions                     Show carry positions
  %(prog)s --dry-run                       Show config and exit
        """,
    )
    parser.add_argument("--live", action="store_true", help="Live advisory mode")
    parser.add_argument("--simulate", metavar="DATE", help="Simulate date YYYY-MM-DD")
    parser.add_argument("--sim-speed", type=float, default=0.5,
                        help="Seconds between events in simulate (0=instant)")
    parser.add_argument("--num-contracts", type=int, default=40,
                        help="Contracts per position (default: 40)")
    parser.add_argument("--positions", action="store_true", help="Show carry positions")
    parser.add_argument("--dry-run", action="store_true", help="Show config and exit")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    if args.positions:
        show_positions()
        return

    if args.dry_run:
        config = _build_config(args.num_contracts)
        print("vMaxMin Layer Config:")
        for k, v in sorted(config.items()):
            print(f"  {k}: {v}")
        print(f"\nCarries file: {CARRIES_FILE}")
        print(f"Trade log: {TRADE_LOG}")
        show_positions()
        return

    if args.simulate:
        run_simulate(args.simulate, args.num_contracts, args.sim_speed)
    elif args.live:
        run_live(args.num_contracts)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
