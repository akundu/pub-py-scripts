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


def _print_trade(time_str, event, direction, short_strike, long_strike,
                  width, credit, num_contracts, notes):
    """Print a formatted trade line to the console."""
    cr_str = f"{'+'if credit>=0 else ''}${credit:,.0f}"
    print(f"  {time_str:>5s}  {event:<20s}  {direction:>4s}  "
          f"{short_strike:>6.0f}/{long_strike:>6.0f}  {width:>3.0f}  "
          f"x{num_contracts:>3d}  {cr_str:>10s}  {notes}")


def _make_trade_record(event, time_pacific, direction, spread, num_contracts,
                       commission, price, dte, notes):
    """Create a TradeRecord-like object for log_trade."""
    from scripts.backtesting.scripts.vmaxmin_engine import TradeRecord
    credit_total = spread["credit"] * 100 * num_contracts
    return TradeRecord(
        event=event, time_pacific=time_pacific, direction=direction,
        short_strike=spread["short_strike"], long_strike=spread["long_strike"],
        width=spread["width"], credit_or_debit=credit_total,
        num_contracts=num_contracts, commission=commission,
        underlying_price=price, dte=dte, notes=notes,
    )


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
    max_risk = config.get("max_total_exposure", 500000)

    # Compute effective contracts for display
    if num_contracts:
        eff_contracts = num_contracts
        sizing_note = ""
    else:
        min_step = {"SPX": 5, "NDX": 10, "RUT": 5}.get(ticker, 5)
        daily_budget = config.get("daily_budget", 100000)
        check_times = config.get("call_track_check_times_pacific", [])
        max_positions = 2 + len(check_times) * 2
        eff_contracts = max(1, int(daily_budget / (max_positions * min_step * 100)))
        sizing_note = (f" (auto: ${daily_budget:,.0f} / "
                       f"({max_positions} positions × ${min_step} width × 100))")

    new_entry_exp = eff_contracts * 5 * 100 * 5

    print(f"\n{'='*90}")
    print(f"  vMaxMin Layer — SIMULATE {ticker} {trade_date}")
    print(f"{'='*90}")
    print(f"  Contracts/position: {eff_contracts}{sizing_note}")
    print(f"  Carries in: {len(today_carries)} positions, "
          f"{sum(c.num_contracts for c in today_carries)} contracts")
    print(f"  Total carry exposure: ${carry_exp:,.0f}")

    if carry_exp + new_entry_exp > max_risk:
        print(f"  ⚠ SKIP DAY: projected exposure ${carry_exp + new_entry_exp:,.0f} > cap ${max_risk:,.0f}")
        print(f"  Carries will still be evaluated for expiry/roll")

    print(f"  Prev close: {prev_close}" if prev_close else "  Prev close: N/A")

    # Data source display
    nested_dir = f"options_csv_output_full_5/{ticker}/{ticker}"
    eq_path = f"equities_output/I:{ticker}/I:{ticker}_equities_{trade_date}.csv"
    opt_path = f"{nested_dir}/{ticker}_options_{trade_date}.csv"
    print(f"  Data sources:")
    print(f"    Equity:  {eq_path}" + (" (found)" if os.path.exists(eq_path) else " (MISSING)"))
    print(f"    Options: {opt_path}" + (" (found)" if os.path.exists(opt_path) else " (MISSING)"))
    if os.path.exists(opt_path):
        import pandas as _pd
        _exp_df = _pd.read_csv(opt_path, usecols=["expiration"], nrows=5000)
        _exps = sorted(_exp_df["expiration"].unique())
        print(f"             Expirations in file: {', '.join(_exps)}")
        _dte0 = [e for e in _exps if e == trade_date]
        _dte1 = [e for e in _exps if e > trade_date]
        print(f"             0DTE: {_dte0[0] if _dte0 else 'none'}  "
              f"DTE+1: {_dte1[0] if _dte1 else 'none (will use next-day file)'}")
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


def run_simulate_range(start_date, end_date, num_contracts, sim_speed):
    """Run simulation across a date range, carries flow between days."""
    ticker = "RUT"
    all_dates = get_trading_dates(ticker, "equities_output", "2024-09-09", "2026-12-31")
    range_dates = [d for d in all_dates if start_date <= d <= end_date]

    if not range_dates:
        print(f"No trading dates found between {start_date} and {end_date}")
        return

    # Clear carries at start of range
    save_carries([])

    print(f"\n{'='*90}")
    print(f"  vMaxMin Layer — MULTI-DAY SIMULATE")
    print(f"  Range: {start_date} to {end_date} ({len(range_dates)} trading days)")
    print(f"  Contracts/position: {num_contracts}")
    print(f"{'='*90}\n")

    for dt in range_dates:
        run_simulate(dt, num_contracts, sim_speed)
        print()

    # Final summary
    carries = load_carries()
    print(f"\n{'='*90}")
    print(f"  Multi-Day Summary: {start_date} to {end_date} ({len(range_dates)} days)")
    print(f"{'='*90}")
    if carries:
        total_contracts = sum(c.num_contracts for c in carries)
        total_exp = sum(c.width * 100 * c.num_contracts for c in carries)
        print(f"  Final carries: {len(carries)} positions, "
              f"{total_contracts} contracts, ${total_exp:,.0f} exposure")
        for rp in carries:
            print(f"    {rp.direction} {rp.short_strike}/{rp.long_strike} w=${rp.width:.0f} "
                  f"x{rp.num_contracts} exp={rp.expiration_date} "
                  f"roll#{rp.roll_count} from {rp.original_entry_date}")
    else:
        print(f"  No carry positions remaining.")

    # Read trade log for P&L summary
    if os.path.exists(TRADE_LOG):
        total_credits = 0
        total_debits = 0
        with open(TRADE_LOG) as f:
            for line in f:
                try:
                    entry = json.loads(line.strip())
                    dt = entry.get("date", "")
                    if start_date <= dt <= end_date:
                        cd = entry.get("credit_or_debit", 0)
                        if cd > 0:
                            total_credits += cd
                        else:
                            total_debits += abs(cd)
                except (json.JSONDecodeError, KeyError):
                    continue
        print(f"  Total credits: ${total_credits:,.0f}  Total debits: ${total_debits:,.0f}")
        print(f"  Net: {'+'if total_credits >= total_debits else ''}${total_credits - total_debits:,.0f}")
    print()


def run_live(num_contracts, ticker="RUT"):
    """Run live advisory mode — polls UTP for real-time equity + option data.

    Phases (all times Pacific):
      06:30-06:45: Entry window — scan for best-ROI entry spreads (call + put)
      08:35, 10:35: Layer checks — new HOD/LOD → recommend additional spreads
      12:50-13:00: EOD scan — check proximity → recommend rolls for ITM positions
      After 13:00: Summary

    Between phases, polls every 60s to track price / HOD / LOD.
    """
    import signal as signal_mod
    import pytz

    from scripts.live_trading.providers.utp_provider import (
        UtpEquityProvider, UtpOptionsProvider,
    )
    from scripts.backtesting.scripts.vmaxmin_engine import (
        find_credit_spread, close_spread_cost, SpreadPosition, filter_valid_quotes,
    )

    config = _build_config(num_contracts)
    PT = pytz.timezone("US/Pacific")
    base_url = "http://localhost:8000"

    # ── Connectivity check ──────────────────────────────────────────────
    print(f"\n{'='*90}")
    print(f"  vMaxMin Layer — LIVE MODE (advisory only)")
    print(f"{'='*90}")
    # Auto-size contracts if not specified
    check_times_cfg = config.get("call_track_check_times_pacific", [])
    min_step = {"SPX": 5, "NDX": 10, "RUT": 5}.get(ticker, 5)
    if num_contracts:
        sizing_note = ""
    else:
        daily_budget = config.get("daily_budget", 100000)
        max_positions = 2 + len(check_times_cfg) * 2
        num_contracts = max(1, int(daily_budget / (max_positions * min_step * 100)))
        sizing_note = (f" (auto: ${daily_budget:,.0f} / "
                       f"({max_positions} pos × ${min_step} w × 100))")

    print(f"  Ticker: {ticker}  Contracts/position: {num_contracts}{sizing_note}")
    print(f"  UTP daemon: {base_url}")

    if not UtpEquityProvider.check_connection(base_url):
        print(f"\n  ERROR: Cannot reach UTP daemon at {base_url}")
        print(f"  Start it with: cd live_trading/universal-trade-platform && python utp.py daemon --paper")
        return
    print(f"  UTP connected.")

    # ── Initialize providers ────────────────────────────────────────────
    equity_prov = UtpEquityProvider()
    equity_prov.initialize({"utp_base_url": base_url, "csv_dir": "equities_output"})

    options_prov = UtpOptionsProvider()
    options_prov.initialize({
        "utp_base_url": base_url,
        "dte_buckets": [0, 1],
        "strike_range_pct": 0.05,
        "cache_ttl": 10,  # 10s cache — we want fresh data at decision points
    })

    # Get previous close
    today = date.today()
    trade_date = today.isoformat()
    prev_close = equity_prov.get_previous_close(ticker, today)
    print(f"  Prev close: {prev_close}" if prev_close else "  Prev close: N/A")

    # Load carries
    carries = load_carries()
    today_carries = [rp for rp in carries if rp.expiration_date == trade_date]
    carry_exp = sum(rp.width * 100 * rp.num_contracts for rp in carries)
    print(f"  Carries in: {len(today_carries)} positions, "
          f"{sum(c.num_contracts for c in today_carries)} contracts")
    print(f"  Total carry exposure: ${carry_exp:,.0f}")

    # ── Config ──────────────────────────────────────────────────────────
    window_start = _time_to_mins(config.get("layer_entry_window_start", "06:30"))
    window_end = _time_to_mins(config.get("layer_entry_window_end", "06:45"))
    eod_scan_start = _time_to_mins(config.get("layer_eod_scan_start", "12:50"))
    eod_scan_end = _time_to_mins(config.get("layer_eod_scan_end", "13:00"))
    eod_proximity = config.get("layer_eod_proximity", 0.002)
    min_roi_start = config.get("layer_entry_min_roi", 0.50)
    min_roi_floor = config.get("layer_entry_min_roi_floor", 0.0)
    window_range = max(window_end - window_start, 1)
    commission = config["commission_per_transaction"]

    eff_min_width = config.get("min_spread_width") or min_step
    max_width = min_step * config.get("max_width_multiplier", 10)

    print(f"\n  Schedule (Pacific):")
    print(f"    Entry window:  {config.get('layer_entry_window_start', '06:30')}"
          f"-{config.get('layer_entry_window_end', '06:45')}")
    print(f"    Layer checks:  {', '.join(check_times_cfg)}")
    print(f"    EOD scan:      {config.get('layer_eod_scan_start', '12:50')}"
          f"-{config.get('layer_eod_scan_end', '13:00')}")
    print(f"    Adaptive ROI:  {min_roi_start*100:.0f}% → {min_roi_floor*100:.0f}%")
    print(f"\n  Waiting for market... (Ctrl+C to quit)\n")

    # ── State ───────────────────────────────────────────────────────────
    positions = []          # SpreadPosition list (today's open positions)
    carried_positions = []  # SpreadPositions from carries
    hod = 0.0
    lod = float("inf")
    prev_hod = 0.0
    prev_lod = float("inf")
    entry_accepted = {"call": False, "put": False}  # adaptive ROI accepted
    best_fallback = {}      # dir → (spread, price, roi)
    trades = []             # (time_str, event, direction, short, long, width, credit, contracts, notes)
    rolled_set = set()      # position ids already rolled at EOD
    check_times_done = set()
    last_eod_scan_min = eod_scan_start - 1

    # Load carried positions into tracking
    for rp in today_carries:
        pos = SpreadPosition(
            direction=rp.direction, short_strike=rp.short_strike,
            long_strike=rp.long_strike, width=rp.width,
            credit_per_share=rp.credit_per_share, num_contracts=rp.num_contracts,
            entry_time="carried", entry_price=0, dte=1,
        )
        carried_positions.append(pos)
        positions.append(pos)
        _print_trade("carried", "carry_in", rp.direction, rp.short_strike,
                     rp.long_strike, rp.width, 0, rp.num_contracts,
                     f"From {rp.original_entry_date}, roll#{rp.roll_count}")

    stop = False
    def _sig_handler(sig, frame):
        nonlocal stop
        stop = True
    signal_mod.signal(signal_mod.SIGINT, _sig_handler)

    def _fetch_price():
        bars = equity_prov.get_bars(ticker, today)
        if bars is not None and not bars.empty:
            return float(bars["close"].iloc[-1])
        return None

    def _fetch_options(dte_list):
        """Fetch options chain from UTP, add time_pacific column."""
        price = _fetch_price()
        if price:
            options_prov.set_current_price(ticker, price)
        df = options_prov.get_options_chain(ticker, today, dte_buckets=dte_list)
        if df is not None and not df.empty:
            now_pt = datetime.now(PT)
            df["time_pacific"] = f"{now_pt.hour:02d}:{now_pt.minute:02d}"
        return df

    def _find_spread(snap, price, direction):
        """Try best_roi then nearest."""
        spread = find_credit_spread(snap, price, direction,
                                    eff_min_width, max_width,
                                    leg_placement="best_roi", depth_pct=0.003)
        if spread is None:
            spread = find_credit_spread(snap, price, direction,
                                        eff_min_width, max_width,
                                        leg_placement="nearest")
        return spread

    def _spread_roi(spread):
        if spread["width"] > spread["credit"]:
            return spread["credit"] / (spread["width"] - spread["credit"])
        return 0

    # ── Main loop ───────────────────────────────────────────────────────
    while not stop:
        now_pt = datetime.now(PT)
        now_mins = now_pt.hour * 60 + now_pt.minute
        now_str = f"{now_pt.hour:02d}:{now_pt.minute:02d}"

        # Before market open: sleep
        if now_mins < window_start:
            wait = (window_start - now_mins) * 60 - now_pt.second
            if wait > 60:
                print(f"  [{now_str}] Market opens in {wait // 60}m. Sleeping...")
                time_mod.sleep(min(wait, 60))
                continue
            time_mod.sleep(max(1, wait))
            continue

        # After market close: done
        if now_mins > eod_scan_end + 2:
            break

        # ── ENTRY PHASE (window_start to window_end) ───────────────────
        if window_start <= now_mins <= window_end:
            price = _fetch_price()
            if price is None:
                print(f"  [{now_str}] No price from UTP, retrying in 15s...")
                time_mod.sleep(15)
                continue

            # Update HOD/LOD
            if hod == 0:
                hod = price
                lod = price
                prev_hod = price
                prev_lod = price
            hod = max(hod, price)
            lod = min(lod, price)

            snap = _fetch_options([0])
            if snap is None or snap.empty:
                print(f"  [{now_str}] No 0DTE options from UTP, retrying in 30s...")
                time_mod.sleep(30)
                continue

            progress = (now_mins - window_start) / window_range
            dynamic_min_roi = min_roi_start - (min_roi_start - min_roi_floor) * progress

            for entry_dir in ["call", "put"]:
                if entry_accepted[entry_dir]:
                    continue
                spread = _find_spread(snap, price, entry_dir)
                if spread is None:
                    continue

                roi = _spread_roi(spread)

                # Track best fallback
                prev_fb = best_fallback.get(entry_dir)
                if prev_fb is None or roi > prev_fb[2]:
                    best_fallback[entry_dir] = (spread, price, roi)

                # Accept if meets threshold
                if roi >= dynamic_min_roi:
                    entry_accepted[entry_dir] = True
                    pos = SpreadPosition(
                        direction=entry_dir,
                        short_strike=spread["short_strike"],
                        long_strike=spread["long_strike"],
                        width=spread["width"],
                        credit_per_share=spread["credit"],
                        num_contracts=num_contracts,
                        entry_time=now_str, entry_price=price, dte=0,
                    )
                    positions.append(pos)
                    notes = f"ROI={roi*100:.0f}%, threshold={dynamic_min_roi*100:.0f}%"
                    _print_trade(now_str, "ENTRY", entry_dir,
                                 spread["short_strike"], spread["long_strike"],
                                 spread["width"], pos.total_credit, num_contracts, notes)
                    log_trade(trade_date, _make_trade_record(
                        "entry", now_str, entry_dir, spread, num_contracts,
                        commission, price, 0, f"Initial {entry_dir} spread ({notes})"))

            print(f"  [{now_str}] Price={price:.1f}  ROI threshold={dynamic_min_roi*100:.0f}%  "
                  f"Accepted: call={'Y' if entry_accepted['call'] else 'N'} "
                  f"put={'Y' if entry_accepted['put'] else 'N'}")

            # At window end: use fallback for any direction not yet accepted
            if now_mins >= window_end:
                for entry_dir in ["call", "put"]:
                    if entry_accepted[entry_dir]:
                        continue
                    if entry_dir in best_fallback:
                        spread, fb_price, fb_roi = best_fallback[entry_dir]
                        pos = SpreadPosition(
                            direction=entry_dir,
                            short_strike=spread["short_strike"],
                            long_strike=spread["long_strike"],
                            width=spread["width"],
                            credit_per_share=spread["credit"],
                            num_contracts=num_contracts,
                            entry_time=now_str, entry_price=fb_price, dte=0,
                        )
                        positions.append(pos)
                        notes = f"ROI={fb_roi*100:.0f}%, fallback best-seen"
                        _print_trade(now_str, "ENTRY (fallback)", entry_dir,
                                     spread["short_strike"], spread["long_strike"],
                                     spread["width"], pos.total_credit, num_contracts, notes)
                        log_trade(trade_date, _make_trade_record(
                            "entry", now_str, entry_dir, spread, num_contracts,
                            commission, fb_price, 0, f"Initial {entry_dir} spread ({notes})"))
                        entry_accepted[entry_dir] = True

            # Wait until next minute
            time_mod.sleep(max(1, 60 - now_pt.second))
            continue

        # ── INTRADAY: track HOD/LOD and check layer times ──────────────
        if now_mins < eod_scan_start:
            price = _fetch_price()
            if price:
                if hod == 0:
                    hod = price
                    lod = price
                    prev_hod = price
                    prev_lod = price
                hod = max(hod, price)
                lod = min(lod, price)

            # Layer check at configured times
            for check_time in check_times_cfg:
                ct_mins = _time_to_mins(check_time)
                if check_time in check_times_done:
                    continue
                if now_mins < ct_mins:
                    continue
                if now_mins > ct_mins + 2:  # missed window
                    check_times_done.add(check_time)
                    continue

                check_times_done.add(check_time)
                if price is None:
                    print(f"  [{now_str}] Layer check at {check_time}: no price")
                    continue

                breach_min = config.get("layer_breach_min_points") or eff_min_width
                new_hod = hod >= prev_hod + breach_min
                new_lod = lod <= prev_lod - breach_min

                if not new_hod and not new_lod:
                    print(f"  [{now_str}] Layer check: HOD={hod:.1f} LOD={lod:.1f} "
                          f"(no breach >= {breach_min:.0f}pts)")
                    continue

                snap = _fetch_options([0])
                if snap is None or snap.empty:
                    print(f"  [{now_str}] Layer check: no options data")
                    continue

                if new_hod:
                    call_spread = _find_spread(snap, hod, "call")
                    if call_spread:
                        pos = SpreadPosition(
                            direction="call",
                            short_strike=call_spread["short_strike"],
                            long_strike=call_spread["long_strike"],
                            width=call_spread["width"],
                            credit_per_share=call_spread["credit"],
                            num_contracts=num_contracts,
                            entry_time=now_str, entry_price=price, dte=0,
                        )
                        positions.append(pos)
                        notes = f"New HOD {hod:.0f} (prev {prev_hod:.0f}, breach>={breach_min:.0f})"
                        _print_trade(now_str, "LAYER", "call",
                                     call_spread["short_strike"], call_spread["long_strike"],
                                     call_spread["width"], pos.total_credit, num_contracts, notes)
                        log_trade(trade_date, _make_trade_record(
                            "layer_add", now_str, "call", call_spread, num_contracts,
                            commission, price, 0, notes))
                        prev_hod = hod

                if new_lod:
                    put_spread = _find_spread(snap, lod, "put")
                    if put_spread:
                        pos = SpreadPosition(
                            direction="put",
                            short_strike=put_spread["short_strike"],
                            long_strike=put_spread["long_strike"],
                            width=put_spread["width"],
                            credit_per_share=put_spread["credit"],
                            num_contracts=num_contracts,
                            entry_time=now_str, entry_price=price, dte=0,
                        )
                        positions.append(pos)
                        notes = f"New LOD {lod:.0f} (prev {prev_lod:.0f}, breach>={breach_min:.0f})"
                        _print_trade(now_str, "LAYER", "put",
                                     put_spread["short_strike"], put_spread["long_strike"],
                                     put_spread["width"], pos.total_credit, num_contracts, notes)
                        log_trade(trade_date, _make_trade_record(
                            "layer_add", now_str, "put", put_spread, num_contracts,
                            commission, price, 0, notes))
                        prev_lod = lod

            # Status every 5 minutes
            if now_pt.minute % 5 == 0 and now_pt.second < 30 and price:
                n_pos = len(positions)
                total_cr = sum(p.total_credit for p in positions if p.entry_time != "carried")
                print(f"  [{now_str}] Price={price:.1f}  HOD={hod:.1f}  LOD={lod:.1f}  "
                      f"Positions={n_pos}  Credits=${total_cr:,.0f}")

            time_mod.sleep(30)
            continue

        # ── EOD PHASE (eod_scan_start to eod_scan_end) ─────────────────
        if eod_scan_start <= now_mins <= eod_scan_end:
            # Scan each minute we haven't scanned yet
            if now_mins <= last_eod_scan_min:
                time_mod.sleep(10)
                continue
            last_eod_scan_min = now_mins

            price = _fetch_price()
            if price is None:
                print(f"  [{now_str}] EOD scan: no price")
                time_mod.sleep(30)
                continue

            hod = max(hod, price)
            lod = min(lod, price)

            snap_0dte = _fetch_options([0])
            snap_dte1 = _fetch_options([1])

            for pos in positions:
                pos_id = id(pos)
                if pos_id in rolled_set:
                    continue

                # Check ITM or within proximity
                if pos.direction == "call":
                    is_itm = price >= pos.short_strike
                else:
                    is_itm = price <= pos.short_strike

                if not is_itm and eod_proximity > 0:
                    dist_pct = abs(price - pos.short_strike) / price if price > 0 else 1
                    is_threatened = dist_pct <= eod_proximity
                else:
                    is_threatened = is_itm

                if not is_threatened:
                    continue

                # Position is ITM or threatened — recommend close + roll
                close_cost = None
                if snap_0dte is not None and not snap_0dte.empty:
                    close_cost = close_spread_cost(snap_0dte, pos)

                # Find DTE+1 replacement
                roll_spread = None
                if snap_dte1 is not None and not snap_dte1.empty:
                    roll_spread = _find_spread(snap_dte1, price, pos.direction)

                status = "ITM" if is_itm else f"within {eod_proximity*100:.1f}%"
                close_str = f"${close_cost*100*pos.num_contracts:,.0f}" if close_cost else "N/A"

                print(f"\n  {'!'*60}")
                print(f"  [{now_str}] EOD ROLL RECOMMENDATION — {pos.direction.upper()} {status}")
                print(f"    Current: {pos.direction} {pos.short_strike}/{pos.long_strike} "
                      f"w=${pos.width:.0f} x{pos.num_contracts}")
                print(f"    Price: {price:.1f}  Close cost: {close_str}")

                if roll_spread:
                    roll_credit = roll_spread["credit"] * 100 * pos.num_contracts
                    net = roll_credit - (close_cost * 100 * pos.num_contracts if close_cost else 0)
                    print(f"    Roll to DTE+1: {pos.direction} "
                          f"{roll_spread['short_strike']}/{roll_spread['long_strike']} "
                          f"w=${roll_spread['width']:.0f}  credit=${roll_credit:,.0f}")
                    print(f"    Net credit/debit: {'+'if net>=0 else ''}${net:,.0f}")

                    # Build carry for next day
                    exp_date_obj = today + timedelta(days=1)
                    # Skip weekends
                    while exp_date_obj.weekday() >= 5:
                        exp_date_obj += timedelta(days=1)

                    rp_orig = None
                    for c in today_carries:
                        if (c.direction == pos.direction and
                                c.short_strike == pos.short_strike):
                            rp_orig = c
                            break

                    notes = (f"Roll {status}: {pos.short_strike}/{pos.long_strike} → "
                             f"{roll_spread['short_strike']}/{roll_spread['long_strike']}")
                    log_trade(trade_date, _make_trade_record(
                        "eod_roll", now_str, pos.direction, roll_spread,
                        pos.num_contracts, commission, price, 1, notes))
                else:
                    print(f"    No DTE+1 options available — MANUAL INTERVENTION NEEDED")

                print(f"  {'!'*60}")
                rolled_set.add(pos_id)

            # Wait for next minute
            time_mod.sleep(max(1, 60 - now_pt.second))
            continue

        # Past EOD
        time_mod.sleep(5)

    # ── Summary ─────────────────────────────────────────────────────────
    print(f"\n{'='*90}")
    print(f"  vMaxMin Layer — LIVE SESSION SUMMARY")
    print(f"{'='*90}")
    n_entries = sum(1 for p in positions if p.entry_time != "carried" and p.dte == 0)
    total_cr = sum(p.total_credit for p in positions if p.entry_time != "carried")
    print(f"  Entries: {n_entries}  Layers: {len(positions) - n_entries - len(carried_positions)}")
    print(f"  Total credits collected: ${total_cr:,.0f}")
    print(f"  HOD: {hod:.1f}  LOD: {lod:.1f}")
    print(f"  Positions rolled at EOD: {len(rolled_set)}")

    # Cleanup
    equity_prov.close()
    options_prov.close()


def _build_config(num_contracts, overrides=None):
    config = {
        **DEFAULT_CONFIG,
        "strategy_mode": "layer",
        "layer_dual_entry": True,
        "layer_entry_directions": "both",
        "max_roll_count": 1,
        "roll_recovery_threshold": 99.0,
        "roll_match_contracts": True,
        "roll_max_chain_contracts": None,
        "call_track_unlimited_budget": True,
        "call_track_leg_placement": "best_roi",
        "call_track_depth_pct": 0.003,
        "layer_entry_window_start": "06:30",
        "layer_entry_window_end": "06:45",
        "layer_eod_scan_start": "12:50",
        "layer_eod_scan_end": "13:00",
        "layer_eod_proximity": 0.002,
        "layer_breach_min_points": 10,
        "call_track_check_times_pacific": [
            "07:05", "07:35", "08:05", "08:35",
        ],
        "equity_dir": "equities_output",
        "options_0dte_dir": "options_csv_output_full_5/RUT",
        "options_dte1_dir": "csv_exports/options",
        "max_total_exposure": 500000,
        "daily_budget": 100000,
    }
    if num_contracts:
        config["num_contracts"] = num_contracts
    if overrides:
        config.update(overrides)
    # else: engine auto-sizes from daily_budget / (max_positions × width)
    return config


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
  %(prog)s --simulate-range 2026-03-18 2026-03-23  Multi-day simulation
  %(prog)s --live                          Live advisory mode (UTP)
  %(prog)s --live --ticker NDX             Live mode for NDX
  %(prog)s --positions                     Show carry positions
  %(prog)s --dry-run                       Show config and exit
        """,
    )
    parser.add_argument("--live", action="store_true", help="Live advisory mode (connects to UTP daemon)")
    parser.add_argument("--simulate", metavar="DATE", help="Simulate date YYYY-MM-DD")
    parser.add_argument("--simulate-range", nargs=2, metavar=("START", "END"),
                        help="Simulate date range YYYY-MM-DD YYYY-MM-DD")
    parser.add_argument("--sim-speed", type=float, default=0.5,
                        help="Seconds between events in simulate (0=instant)")
    parser.add_argument("--num-contracts", type=int, default=None,
                        help="Contracts per position (default: auto from daily_budget)")
    parser.add_argument("--ticker", default="RUT",
                        help="Ticker symbol (default: RUT)")
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
    elif args.simulate_range:
        run_simulate_range(args.simulate_range[0], args.simulate_range[1],
                           args.num_contracts, args.sim_speed)
    elif args.live:
        run_live(args.num_contracts, ticker=args.ticker)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
