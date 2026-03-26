#!/usr/bin/env python3
"""Live Trading Advisor — Generic profile-based credit spread recommendations.

Loads an advisor profile YAML and evaluates tiers against real-time market data
(QuestDB + live option chain snapshots), displaying actionable entry/exit/roll
recommendations. Advisory only — you execute the trades.

Each profile defines its own tiers, risk limits, directional modes, and strategy
parameters. Profiles live in scripts/live_trading/advisor/profiles/<name>.yaml.

Requires:
  - QUEST_DB_STRING env var (for realtime equity prices)
  - csv_exports/options/{TICKER}/ (live option chain snapshots)
  - equities_output/{TICKER}/ (historical bars for signal computation)
  - options_csv_output_full/{TICKER}/ (historical options fallback)

Usage:
  python run_live_advisor.py --profile tiered_v2          # 9-tier NDX setup
  python run_live_advisor.py --profile single_p90_dte2    # Single-tier P90/DTE2
  python run_live_advisor.py --profile tiered_v2 --dry-run
  python run_live_advisor.py --profile tiered_v2 --ticker SPX
  python run_live_advisor.py --list-profiles              # Show available profiles
  python run_live_advisor.py --profile tiered_v2 --positions
  python run_live_advisor.py --profile tiered_v2 --summary
  python run_live_advisor.py --profile ./my_custom.yaml   # Load from path
"""

import argparse
import logging
import os
import queue
import signal
import sys
import threading
import time as time_mod
from datetime import date, datetime, time, timezone
from pathlib import Path

# Ensure project root is in path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.live_trading.advisor.profile_loader import (
    AdvisorProfile,
    list_profiles,
    load_profile,
)
from scripts.live_trading.advisor.position_tracker import PositionTracker
from scripts.live_trading.advisor.tier_evaluator import TierEvaluator, Recommendation
from scripts.live_trading.advisor.advisor_display import AdvisorDisplay, C

logger = logging.getLogger("live_advisor")

# Market hours (UTC)
MARKET_OPEN_UTC = time(13, 30)    # 9:30 AM ET
MARKET_CLOSE_UTC = time(20, 10)   # 1:10 PM PT / 4:10 PM ET
PRE_MARKET_UTC = time(13, 0)      # 9:00 AM ET


def _is_market_hours(now: datetime) -> bool:
    """Check if current time is within extended market hours."""
    t = now.time() if hasattr(now, "time") else now
    return PRE_MARKET_UTC <= t <= MARKET_CLOSE_UTC


def _is_trading_hours(now: datetime) -> bool:
    """Check if current time is within entry window (any tier)."""
    t = now.time() if hasattr(now, "time") else now
    return MARKET_OPEN_UTC <= t <= MARKET_CLOSE_UTC


def _stdin_reader(q: queue.Queue, stop_event: threading.Event) -> None:
    """Background thread that reads stdin lines and puts them in the queue."""
    while not stop_event.is_set():
        try:
            line = sys.stdin.readline()
            if not line:
                break
            line = line.strip()
            if line:
                q.put(line)
        except EOFError:
            break
        except Exception:
            break


def _print_rejection_summary(evaluators: dict) -> None:
    """Print aggregated rejection summary, collapsing parametric reasons."""
    import re

    print(f"\n  Rejection Summary:")
    raw: dict = {}
    for ev in evaluators.values():
        for reason, count in ev.rejection_counts.items():
            raw[reason] = raw.get(reason, 0) + count

    if not raw:
        print(f"    (none)")
        return

    # Group parametric rejections: "dte0_p92:total_credit_below_min($470<$500)"
    # → base key "dte0_p92:total_credit_below_min", with values extracted
    grouped: dict = {}  # base_key → {"count": N, "values": [float, ...]}
    simple: dict = {}   # non-parametric → count

    paren_re = re.compile(r'^(.+)\((.+)\)$')
    for reason, count in raw.items():
        # Split on last '(' to find parametric part
        m = paren_re.match(reason)
        if m and '$' in m.group(2):
            base = m.group(1)
            # Extract the dollar value (first number)
            nums = re.findall(r'\$([0-9,.]+)', m.group(2))
            val = float(nums[0].replace(',', '')) if nums else 0
            if base not in grouped:
                grouped[base] = {"count": 0, "values": []}
            grouped[base]["count"] += count
            grouped[base]["values"].extend([val] * count)
        else:
            simple[reason] = simple.get(reason, 0) + count

    # Print simple rejections
    all_lines = []
    for reason, count in simple.items():
        all_lines.append((count, reason))

    # Print grouped rejections with distribution
    for base, info in grouped.items():
        count = info["count"]
        vals = sorted(info["values"])
        if vals:
            mn, mx = vals[0], vals[-1]
            med = vals[len(vals) // 2]
            all_lines.append((count, f"{base} — {count}x, range ${mn:,.0f}-${mx:,.0f}, median ${med:,.0f}"))
        else:
            all_lines.append((count, base))

    for count, line in sorted(all_lines, key=lambda x: -x[0]):
        print(f"    {count:>5}x  {line}")


def _handle_command(
    cmd: str,
    entries: list,
    exits: list,
    tracker: PositionTracker,
    evaluator: TierEvaluator,
    display: AdvisorDisplay,
) -> bool:
    """Process a user command. Returns True to quit."""
    parts = cmd.strip().split()
    if not parts:
        return False

    action = parts[0].lower()

    # Merge entries + all_entries for lookup by order_id
    all_recs = entries  # primary list; caller can pass all_entries via entries param

    # Log raw input
    available_ids = [r.order_id for r in all_recs if r.order_id] if all_recs else []
    logger.info(f"Command received: {cmd!r} | action={action} | "
                f"{len(all_recs)} recs available, IDs={available_ids[:5]}")

    if action == "q":
        display.log_input(cmd, "quit")
        return True

    elif action == "p":
        display.log_input(cmd, "show positions")
        display.print_positions_detail(tracker, evaluator.get_current_price())

    elif action == "s":
        display.log_input(cmd, "show summary")
        display.print_summary(tracker)

    elif action == "buy":
        # Confirm entry by order ID: 'buy RUT_P2420_D0'
        if len(parts) < 2:
            display.log_input(cmd, "ERROR: missing order_id")
            display.print_error("Usage: buy <order_id>")
            return False
        target_oid = parts[1].upper()
        # Search all entries (including rejected) for this order ID
        rec = None
        for r in all_recs:
            if r.order_id and r.order_id.upper() == target_oid:
                rec = r
                break
        if rec is None:
            # Partial match
            matches = [r for r in all_recs if r.order_id and target_oid in r.order_id.upper()]
            if len(matches) == 1:
                rec = matches[0]
            elif len(matches) > 1:
                display.log_input(cmd, f"ERROR: ambiguous ({len(matches)} matches)")
                display.print_error(f"Ambiguous: {', '.join(r.order_id for r in matches)}")
                return False
            else:
                display.log_input(cmd, f"ERROR: not found. Available: {available_ids[:8]}")
                display.print_error(
                    f"Order {target_oid} not found. "
                    f"Available: {', '.join(available_ids[:8])}"
                )
                return False
        pos = tracker.add_position(
            tier_label=rec.tier_label,
            priority=rec.priority,
            direction=rec.direction,
            short_strike=rec.short_strike,
            long_strike=rec.long_strike,
            credit=rec.credit,
            num_contracts=rec.num_contracts,
            dte=rec.dte,
            entry_price=rec.entry_price,
            ticker=rec.ticker,
        )
        msg = (
            f"BOUGHT: [{rec.order_id}] → pos {pos.pos_id} | "
            f"{rec.ticker} {rec.direction.upper()} "
            f"{rec.short_strike:.0f}/{rec.long_strike:.0f} "
            f"x{rec.num_contracts} @ ${rec.credit:.2f} "
            f"(sell@${rec.short_price:.2f} buy@${rec.long_price:.2f})"
        )
        display.log_input(cmd, msg)
        display.print_success(f"  {msg}")

    elif action == "y":
        # Legacy: confirm entry by priority: 'y 1' or 'y 1 3'
        priorities = []
        for p in parts[1:]:
            try:
                priorities.append(int(p))
            except ValueError:
                display.print_error(f"Invalid priority: {p}")

        rec_map = {r.priority: r for r in entries}
        for pri in priorities:
            rec = rec_map.get(pri)
            if rec is None:
                display.print_error(f"No entry recommendation with priority {pri}")
                continue
            pos = tracker.add_position(
                tier_label=rec.tier_label,
                priority=rec.priority,
                direction=rec.direction,
                short_strike=rec.short_strike,
                long_strike=rec.long_strike,
                credit=rec.credit,
                num_contracts=rec.num_contracts,
                dte=rec.dte,
                entry_price=rec.entry_price,
            )
            display.print_success(
                f"  BOUGHT: [{rec.order_id}] → position {pos.pos_id} | "
                f"{rec.ticker} {rec.direction.upper()} "
                f"{rec.short_strike:.0f}/{rec.long_strike:.0f} "
                f"x{rec.num_contracts} @ ${rec.credit:.2f}"
            )

    elif action == "flush":
        # Close ALL open positions at once
        open_positions = tracker.get_open_positions()
        if not open_positions:
            display.log_input(cmd, "no open positions to flush")
            display.print_info("  No open positions to flush")
        else:
            for pos in open_positions:
                tracker.close_position(pos.pos_id, reason="flushed")
            msg = f"Flushed {len(open_positions)} positions"
            display.log_input(cmd, msg)
            display.print_success(f"  {msg}")

    elif action in ("x", "close"):
        # Close position: 'x <id>' or 'close <id> [exit_price]'
        if len(parts) < 2:
            display.print_error("Usage: close <position_id> [exit_price]")
            return False
        pos_id = parts[1]
        exit_price = 0.0
        if len(parts) >= 3:
            try:
                exit_price = float(parts[2])
            except ValueError:
                display.print_error(f"Invalid price: {parts[2]}")
                return False
        pos = tracker.close_position(pos_id, reason="manual_close", exit_price=exit_price)
        if pos:
            pnl_str = f" P&L: ${pos.realized_pnl:,.0f}" if exit_price > 0 else ""
            display.print_success(f"  Closed: {pos_id}{pnl_str}")
        else:
            display.print_error(f"Position {pos_id} not found or already closed")

    elif action == "r":
        # Confirm roll: 'r <id>'
        if len(parts) < 2:
            display.print_error("Usage: r <position_id>")
            return False
        pos_id = parts[1]
        current_price = evaluator.get_current_price() or 0
        pos = tracker.close_position(pos_id, reason="rolled", exit_price=current_price)
        if pos:
            display.print_success(f"  Rolled out: {pos_id} (closed old leg)")
            display.print_info("  Enter the new roll position via the next entry recommendation.")
        else:
            display.print_error(f"Position {pos_id} not found or already closed")

    else:
        display.print_error(f"Unknown command: {action}. Try: buy, close, p, s, q")

    return False


def _run_live_utp_mode(profile: AdvisorProfile, tracker, args) -> None:
    """Phase 1: display-only mode using UTP for live market data.

    Fetches quotes and option chains for ALL tickers in the profile
    (or a single --ticker override) and displays them in one view.
    """
    from datetime import date as _date

    from scripts.live_trading.providers.utp_provider import (
        UtpEquityProvider, UtpOptionsProvider,
    )
    from scripts.live_trading.advisor.utp_display import (
        TickerSnapshot, UtpDataDisplay,
    )

    base_url = profile.providers.utp_base_url

    # Check UTP connectivity
    print(f"{C.CYAN}Checking UTP connectivity at {base_url}...{C.RESET}")
    if not UtpEquityProvider.check_connection(base_url):
        print(f"{C.RED}Cannot reach UTP daemon at {base_url}{C.RESET}")
        print(f"{C.DIM}Start it with: cd live_trading/universal-trade-platform && python utp.py daemon{C.RESET}")
        return

    # Determine which tickers to display.
    # --ticker overrides to a single ticker; otherwise use all from profile.
    if args.ticker:
        tickers = [args.ticker]
    else:
        tickers = profile.tickers if profile.tickers else [profile.ticker]

    print(f"{C.GREEN}UTP daemon connected. Tickers: {', '.join(tickers)}{C.RESET}")

    # One shared equity provider (session reuse), one options provider
    equity_prov = UtpEquityProvider()
    equity_prov.initialize({
        "utp_base_url": base_url,
        "csv_dir": profile.providers.equity_csv_dir,
    })

    options_prov = UtpOptionsProvider()
    options_prov.initialize({
        "utp_base_url": base_url,
        "dte_buckets": profile.providers.dte_buckets,
    })

    utp_display = UtpDataDisplay(
        profile_name=profile.name,
        tickers=tickers,
    )

    # Pre-fetch previous close for each ticker (one-time)
    prev_closes = {}
    for ticker in tickers:
        prev_closes[ticker] = equity_prov.get_previous_close(ticker, _date.today())

    # Non-blocking input
    input_queue = queue.Queue()
    stop_event = threading.Event()
    input_thread = threading.Thread(
        target=_stdin_reader, args=(input_queue, stop_event), daemon=True
    )
    input_thread.start()

    def _signal_handler(sig, frame):
        stop_event.set()
    signal.signal(signal.SIGINT, _signal_handler)
    signal.signal(signal.SIGTERM, _signal_handler)

    interval = args.interval

    while not stop_event.is_set():
        now = datetime.now(timezone.utc)
        today = _date.today()

        # Fetch data for every ticker
        snapshots = []
        for ticker in tickers:
            bars = equity_prov.get_bars(ticker, today)
            price = float(bars["close"].iloc[-1]) if bars is not None and not bars.empty else None

            # Pass current price so options provider can compute strike range
            if price:
                options_prov.set_current_price(ticker, price)

            options_df = options_prov.get_options_chain(
                ticker, today,
                dte_buckets=profile.providers.dte_buckets,
            )

            snapshots.append(TickerSnapshot(
                ticker=ticker,
                price=price,
                prev_close=prev_closes.get(ticker),
                options_df=options_df,
                quote_cache_age=equity_prov._cache_age(f"quote_{ticker}"),
            ))

        # Aggregate cache stats
        cache_stats = {
            "quotes": {"age": min(
                (s.quote_cache_age for s in snapshots if s.quote_cache_age is not None),
                default=None,
            )},
            "chains": {
                "hits": options_prov.cache_stats["hits"],
                "misses": options_prov.cache_stats["misses"],
            },
        }

        utp_display.print_multi_ticker_snapshot(
            snapshots=snapshots,
            now=now,
            cache_stats=cache_stats,
            next_refresh_secs=interval,
        )

        # Wait for input or timeout
        try:
            cmd = input_queue.get(timeout=interval)
            if cmd and cmd.strip().lower() == "q":
                break
        except queue.Empty:
            pass

    stop_event.set()
    equity_prov.close()
    options_prov.close()
    print(f"\n{C.CYAN}UTP display stopped.{C.RESET}")


def _run_simulation(profile, tracker, args, display) -> None:
    """Simulate a trading day by replaying CSV data at accelerated speed.

    Uses the same TierEvaluator and display as live mode, but feeds it
    historical equity bars and option chains from disk instead of UTP/QuestDB.
    Each 5-min interval is replayed with --sim-speed seconds of wall-clock delay.
    """
    from datetime import date as _date, timedelta
    import glob
    import pandas as pd

    sim_date_str = args.simulate
    sim_speed = args.sim_speed

    try:
        sim_date = _date.fromisoformat(sim_date_str)
    except ValueError:
        display.print_error(f"Invalid date: {sim_date_str}. Use YYYY-MM-DD format.")
        return

    if args.ticker:
        sim_tickers = [args.ticker]
    else:
        sim_tickers = profile.tickers if profile.tickers else [profile.ticker]

    display.print_info(f"SIMULATION MODE: {sim_date_str} | tickers={','.join(sim_tickers)} | "
                       f"speed={sim_speed}s per 5-min interval")

    # Build evaluators using CSV-only providers (no QuestDB, no UTP)
    from copy import deepcopy
    import scripts.backtesting.providers.csv_equity_provider  # noqa: register
    import scripts.backtesting.providers.csv_exports_options_provider  # noqa: register
    from scripts.backtesting.providers.csv_equity_provider import CSVEquityProvider
    from scripts.backtesting.providers.csv_exports_options_provider import CSVExportsOptionsProvider
    from scripts.backtesting.instruments.credit_spread import CreditSpreadInstrument  # noqa
    from scripts.backtesting.instruments.factory import InstrumentFactory
    from scripts.backtesting.signals.percentile_range import PercentileRangeSignal

    evaluators: dict[str, TierEvaluator] = {}
    for eticker in sim_tickers:
        tp = deepcopy(profile)
        tp.ticker = eticker
        ev = TierEvaluator(tp, tracker, use_utp=False)

        # Wire up CSV-only providers (skip QuestDB)
        prov = tp.providers
        eq_prov = CSVEquityProvider()
        eq_prov.initialize({"csv_dir": prov.equity_csv_dir})
        ev._equity_provider = eq_prov

        # Use csv_exports/options/ which has multi-DTE data with timestamps
        opts_prov = CSVExportsOptionsProvider()
        opts_prov.initialize({
            "csv_dir": prov.options_csv_dir,  # csv_exports/options
            "fallback_csv_dir": prov.options_fallback_csv_dir,  # options_csv_output_full
            "dte_buckets": prov.dte_buckets,
        })
        ev._options_provider = opts_prov

        # Signal generator
        sig = tp.signal
        if sig.name and sig.name != "none":
            sig_gen = PercentileRangeSignal()
            sig_params = dict(sig.params)
            if "percentiles" not in sig_params:
                sig_params["percentiles"] = tp.all_percentiles
            if "dte_windows" not in sig_params:
                sig_params["dte_windows"] = tp.all_dtes
            sig_gen.setup(eq_prov, sig_params)
            ev._signal_gen = sig_gen

        # Instrument
        ev._instrument = InstrumentFactory.create(tp.instrument)

        ev._sim_date = sim_date
        evaluators[eticker] = ev
        display.print_success(f"  {eticker}: ready (CSV-only)")

    if not evaluators:
        display.print_error("No tickers initialized")
        return

    first_ev = next(iter(evaluators.values()))

    # Initialize day for all evaluators — bypass on_market_open() date issue
    # by computing prev_close from CSV directly and setting day state manually
    for eticker, ev in evaluators.items():
        # Get previous close from the day before sim_date
        equity_dir = profile.providers.equity_csv_dir
        for prefix in [f"I:{eticker}", eticker]:
            sorted_files = sorted(glob.glob(
                os.path.join(equity_dir, prefix, f"{prefix}_equities_*.csv")
            ))
            # Find file for date before sim_date
            prev_file = None
            for f in sorted_files:
                fdate = os.path.basename(f).split("_equities_")[1].replace(".csv", "")
                if fdate < sim_date_str:
                    prev_file = f
            if prev_file:
                try:
                    pf = pd.read_csv(prev_file, usecols=["close"])
                    ev._prev_close = float(pf["close"].iloc[-1])
                except Exception:
                    pass
            break

        if ev._prev_close is None:
            display.print_error(f"  {eticker}: no prev_close for {sim_date_str}")
            continue

        # Generate signals using the prev_close
        ev._day_initialized = True
        if ev._signal_gen is not None:
            from scripts.backtesting.strategies.base import DayContext
            bars = ev._equity_provider.get_bars(eticker, sim_date)
            if bars is None or bars.empty:
                bars = pd.DataFrame({"close": [ev._prev_close]})
            day_ctx = DayContext(
                ticker=eticker, trading_date=sim_date,
                equity_bars=bars, options_data=pd.DataFrame(),
                prev_close=ev._prev_close, signals={},
            )
            ev._today_signals = ev._signal_gen.generate(day_ctx)

        ev._load_adaptive_baseline()
        display.print_success(
            f"  {eticker}: prev_close={ev._prev_close:.2f}, "
            f"strikes={len(ev._today_signals.get('strikes', {}))} DTEs"
        )

    # Load equity bars for the simulation day to get timestamps
    equity_dir = profile.providers.equity_csv_dir
    timestamps = []
    for eticker in sim_tickers:
        for prefix in [f"I:{eticker}", eticker]:
            csv_path = os.path.join(equity_dir, prefix, f"{prefix}_equities_{sim_date_str}.csv")
            if os.path.exists(csv_path):
                bars = pd.read_csv(csv_path)
                if "timestamp" in bars.columns and not bars.empty:
                    bars["timestamp"] = pd.to_datetime(bars["timestamp"], utc=True)
                    timestamps.extend(bars["timestamp"].tolist())
                break

    if not timestamps:
        display.print_error(f"No equity data found for {sim_date_str}")
        for ev in evaluators.values():
            ev.close()
        return

    timestamps = sorted(set(timestamps))
    display.print_info(f"Replaying {len(timestamps)} intervals: "
                       f"{timestamps[0].strftime('%H:%M')} to {timestamps[-1].strftime('%H:%M')} UTC")

    # Load options data for the day (once, for all tickers)
    options_dir = profile.providers.options_fallback_csv_dir
    ticker_options: dict[str, pd.DataFrame] = {}
    for eticker in sim_tickers:
        opts_path = os.path.join(options_dir, eticker, f"{eticker}_options_{sim_date_str}.csv")
        if os.path.exists(opts_path):
            try:
                opts_df = pd.read_csv(opts_path)
                opts_df["timestamp"] = pd.to_datetime(opts_df["timestamp"], utc=True)
                ticker_options[eticker] = opts_df
                display.print_info(f"  {eticker}: {len(opts_df):,} option rows loaded")
            except Exception as e:
                display.print_warning(f"  {eticker}: options load failed: {e}")

    # Replay loop with input handling
    stop = threading.Event()
    def _sig(s, f):
        stop.set()
    signal.signal(signal.SIGINT, _sig)

    # Input thread for simulation
    input_queue = queue.Queue()
    input_thread = threading.Thread(
        target=_stdin_reader, args=(input_queue, stop), daemon=True
    )
    input_thread.start()

    # Use first evaluator for _handle_command compatibility
    first_ev_for_cmd = next(iter(evaluators.values()))
    all_entries: list = []
    last_exits: list = []

    for i, ts in enumerate(timestamps):
        if stop.is_set():
            break

        # --- Check for commands before evaluation ---
        try:
            cmd = input_queue.get_nowait()
            if cmd:
                logger.info(f"Sim input: {cmd!r}")
                should_quit = _handle_command(
                    cmd, all_entries, last_exits, tracker,
                    first_ev_for_cmd, display
                )
                if should_quit:
                    break
        except queue.Empty:
            pass

        # Get price at this timestamp for each ticker
        all_entries = []
        last_exits = []
        ticker_prices: dict = {}

        for eticker, ev in evaluators.items():
            bars = ev._equity_provider.get_bars(eticker, sim_date)
            if bars is None or bars.empty:
                continue
            if "timestamp" in bars.columns:
                bars["timestamp"] = pd.to_datetime(bars["timestamp"], utc=True)
                mask = bars["timestamp"] <= ts
                if mask.any():
                    price = float(bars.loc[mask, "close"].iloc[-1])
                else:
                    price = float(bars["close"].iloc[0])
            else:
                price = float(bars["close"].iloc[-1])

            ev._current_price = price
            ticker_prices[eticker] = {"price": price, "prev_close": ev.prev_close}

            if hasattr(ev._options_provider, "set_current_price"):
                ev._options_provider.set_current_price(eticker, price)
            if hasattr(ev._options_provider, "set_current_time"):
                ev._options_provider.set_current_time(ts)

            exits = ev.evaluate_exits(price, ts)
            last_exits.extend(exits)

            entries = ev.evaluate_entries(price, ts)
            all_entries.extend(entries)

        # Sort by normalized ROI (best opportunity first), pick top 3
        for rec in all_entries:
            w = rec.spread_width or 1
            cr = rec.credit
            ml = w - cr
            rec._sort_roi = (cr / ml * 100 / (rec.dte + 1)) if ml > 0 else 0
        all_entries.sort(key=lambda r: getattr(r, '_sort_roi', 0), reverse=True)
        last_entries = all_entries[:3]

        current_price = next(
            (v["price"] for v in ticker_prices.values()), None
        )
        if current_price is None:
            continue

        display.refresh(
            current_price=current_price,
            prev_close=first_ev.prev_close,
            entries=last_entries,
            exits=last_exits,
            tracker=tracker,
            now=ts,
            ticker_prices=ticker_prices,
            all_entries=all_entries,
        )

        pct = (i + 1) / len(timestamps) * 100
        print(f"{C.DIM}  [{i+1}/{len(timestamps)} {pct:.0f}%] "
              f"Sim time: {ts.to_pydatetime().astimezone().strftime('%H:%M %Z')} | "
              f"Next in {sim_speed:.1f}s (Ctrl+C to stop){C.RESET}")

        # --- Wait for sim_speed, checking for input every 0.5s ---
        wait_chunks = max(1, int(sim_speed / 0.5))
        for _ in range(wait_chunks):
            if stop.is_set():
                break
            try:
                cmd = input_queue.get(timeout=0.5)
                if cmd:
                    logger.info(f"Sim input during wait: {cmd!r}")
                    should_quit = _handle_command(
                        cmd, all_entries, last_exits, tracker,
                        first_ev_for_cmd, display
                    )
                    if should_quit:
                        stop.set()
                    break  # refresh immediately after command
            except queue.Empty:
                pass

    # --- End of Day Summary ---
    stop.set()
    print(f"\n{'='*80}")
    print(f"  END OF DAY SUMMARY — {sim_date_str}")
    print(f"{'='*80}")

    # Positions & P&L
    all_positions = tracker.get_open_positions() + tracker.get_closed_positions()
    open_pos = tracker.get_open_positions()
    closed_pos = tracker.get_closed_positions()
    print(f"\n  Positions: {len(all_positions)} total "
          f"({len(open_pos)} open, {len(closed_pos)} closed)")

    total_credit = 0
    total_risk = 0
    for pos in all_positions:
        total_credit += pos.total_credit
        total_risk += pos.max_loss
        status = "OPEN" if pos.status == "open" else f"CLOSED ({pos.close_reason})"
        # Compute unrealized P&L for open positions
        pnl_str = ""
        if pos.status == "open":
            tp = ticker_prices.get(pos.ticker, {})
            p = tp.get("price", 0)
            if p and pos.direction == "put":
                intr = max(0, pos.short_strike - p)
            elif p:
                intr = max(0, p - pos.short_strike)
            else:
                intr = 0
            unrealized = pos.total_credit - intr * pos.num_contracts * 100
            pnl_str = f"  unrealized P&L: ${unrealized:+,.0f}"
        elif pos.realized_pnl:
            pnl_str = f"  realized P&L: ${pos.realized_pnl:+,.0f}"
        print(f"    {pos.pos_id} {pos.ticker:<4} {pos.direction.upper():<4} "
              f"{pos.short_strike:.0f}/{pos.long_strike:.0f} "
              f"x{pos.num_contracts} cr=${pos.credit:.2f} "
              f"({status}){pnl_str}")

    if total_credit > 0:
        print(f"\n  Total credit collected: ${total_credit:,.0f}")
        print(f"  Total risk deployed:   ${total_risk:,.0f}")

    # Rejection summary
    _print_rejection_summary(evaluators)

    print(f"{'='*80}\n")

    for ev in evaluators.values():
        ev.close()
    display.print_info(f"Simulation of {sim_date_str} complete.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description='''
Live Trading Advisor — Generic profile-based credit spread recommendations.

Loads an advisor profile YAML and evaluates tiers against real-time data,
displaying entry/exit/roll recommendations. Advisory only — you execute
the trades.
        ''',
        epilog='''
Examples:
  %(prog)s --profile tiered_v2
      Run the 9-tier NDX advisor (default tiered_portfolio_v2 backtest)

  %(prog)s --profile single_p90_dte2
      Run a single-tier P90 DTE2 advisor

  %(prog)s --profile tiered_v2 --ticker SPX
      Override the profile's ticker to SPX

  %(prog)s --profile ./custom_profile.yaml
      Load a profile from an arbitrary path

  %(prog)s --list-profiles
      List available profile names

  %(prog)s --profile tiered_v2 --dry-run
      Show tier configuration without connecting to data sources

  %(prog)s --profile tiered_v2 --positions
      Show currently tracked positions and exit

  %(prog)s --profile tiered_v2 --summary
      Show daily performance summary and exit

  %(prog)s --profile tiered_v2 --interval 30
      Check for signals every 30 seconds (default: 60)

  %(prog)s --profile tiered_v2 --no-interactive
      Run in log mode without interactive prompts (for piping/logging)

  %(prog)s --profile v5_smart_roll --live
      Display live market data from UTP/IBKR (Phase 1: data only, no signals)

  %(prog)s --profile v5_smart_roll --live --interval 120
      UTP display with 2-minute refresh cycle

  %(prog)s --profile adaptive_v5 --simulate 2026-03-10
      Simulate March 10th from CSV data (3s per 5-min interval)

  %(prog)s --profile adaptive_v5 --simulate 2026-03-10 --sim-speed 0.5 --ticker RUT
      Fast simulation of RUT only
        ''',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--profile",
        help="Profile name (from profiles/ dir) or path to a YAML file"
    )
    parser.add_argument(
        "--list-profiles", action="store_true",
        help="List available profiles and exit"
    )
    parser.add_argument(
        "--ticker",
        help="Override the profile's ticker symbol"
    )
    parser.add_argument(
        "--interval", type=int, default=60,
        help="Seconds between evaluation cycles (default: 60)"
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Show configuration and exit without running"
    )
    parser.add_argument(
        "--positions", action="store_true",
        help="Show tracked positions and exit"
    )
    parser.add_argument(
        "--summary", action="store_true",
        help="Show daily summary and exit"
    )
    parser.add_argument(
        "--no-interactive", action="store_true",
        help="Run in log mode without interactive prompts"
    )
    parser.add_argument(
        "--live", action="store_true",
        help="Use UTP/IBKR for live market data"
    )
    parser.add_argument(
        "--simulate", metavar="DATE",
        help="Simulate a trading day from CSV data (e.g., 2026-03-10). "
             "Replays 5-min intervals at --sim-speed rate."
    )
    parser.add_argument(
        "--sim-speed", type=float, default=3.0,
        help="Seconds of wall-clock per 5-min simulated interval (default: 3.0)"
    )
    parser.add_argument(
        "--debug", action="store_true",
        help="Enable debug logging"
    )
    args = parser.parse_args()

    # Logging
    log_level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    # List profiles mode
    if args.list_profiles:
        profiles = list_profiles()
        if not profiles:
            print("No profiles found in scripts/live_trading/advisor/profiles/")
        else:
            print("Available profiles:")
            for p in profiles:
                print(f"  {p}")
        return

    # Profile is required for all other modes
    if not args.profile:
        parser.error("--profile is required (or use --list-profiles)")

    # Determine mode and load profile with mode-specific overrides
    if args.simulate:
        profile_mode = "simulate"
    elif args.live:
        profile_mode = "live"
    else:
        profile_mode = "live"

    try:
        profile = load_profile(args.profile, mode=profile_mode)
    except (FileNotFoundError, ValueError) as e:
        print(f"Error loading profile: {e}", file=sys.stderr)
        sys.exit(1)

    # Ticker override
    if args.ticker:
        profile.ticker = args.ticker

    interactive = not args.no_interactive
    display = AdvisorDisplay(profile, interactive=interactive)

    if not interactive:
        C.disable()

    # Dry run: show config and exit
    if args.dry_run:
        display.print_dry_run_config()
        return

    tracker = PositionTracker(profile_name=profile.name)

    # Positions-only mode
    if args.positions:
        display.print_positions_detail(tracker, None)
        return

    # Summary-only mode
    if args.summary:
        display.print_summary(tracker)
        return

    # Simulation mode: replay a historical day from CSV
    if args.simulate:
        _run_simulation(profile, tracker, args, display)
        return

    # Build evaluators — one per ticker for multi-ticker support
    use_utp = args.live
    if args.ticker:
        eval_tickers = [args.ticker]
    else:
        eval_tickers = profile.tickers if profile.tickers else [profile.ticker]

    evaluators: dict[str, TierEvaluator] = {}
    display.print_info(f"Initializing {'UTP/IBKR' if use_utp else 'CSV/QuestDB'} "
                       f"providers for {', '.join(eval_tickers)}...")
    for eticker in eval_tickers:
        # Clone profile with this ticker
        from copy import deepcopy
        tp = deepcopy(profile)
        tp.ticker = eticker
        ev = TierEvaluator(tp, tracker, use_utp=use_utp)
        try:
            ev.setup()
            evaluators[eticker] = ev
            display.print_success(f"  {eticker}: ready")
        except Exception as e:
            display.print_error(f"  {eticker}: setup failed: {e}")
            logger.exception(f"Setup error for {eticker}")

    if not evaluators:
        display.print_error("No tickers initialized successfully")
        return

    # For backwards compat: single evaluator reference (first ticker)
    evaluator = next(iter(evaluators.values()))

    # Non-blocking input via background thread
    input_queue = queue.Queue()
    stop_event = threading.Event()

    if interactive:
        input_thread = threading.Thread(
            target=_stdin_reader, args=(input_queue, stop_event), daemon=True
        )
        input_thread.start()

    # Graceful shutdown — first Ctrl+C sets stop flag, second force-exits
    _ctrl_c_count = [0]

    def _signal_handler(sig, frame):
        _ctrl_c_count[0] += 1
        stop_event.set()
        if _ctrl_c_count[0] >= 2:
            display.print_info("\nForce exit.")
            # Print summary before dying
            for ev in evaluators.values():
                ev.close()
            os._exit(0)
        display.print_info("\nShutting down... (Ctrl+C again to force)")

    signal.signal(signal.SIGINT, _signal_handler)
    signal.signal(signal.SIGTERM, _signal_handler)

    day_initialized = False
    last_entries: list = []
    last_exits: list = []
    ticker_list = list(evaluators.keys())

    display.print_info(
        f"Advisor running: profile={profile.name} tickers={','.join(ticker_list)} "
        f"interval={args.interval}s {'(UTP/IBKR)' if use_utp else '(CSV/QuestDB)'}"
    )

    while not stop_event.is_set():
        now = datetime.now(timezone.utc)

        # Outside market hours
        if not _is_market_hours(now):
            if day_initialized:
                for eticker, ev in evaluators.items():
                    price = ev.get_current_price()
                    if price:
                        ev.compute_eod_signal(price)
                day_initialized = False
                display.print_info("Market closed. EOD signals computed.")

            next_open_hr = MARKET_OPEN_UTC.hour
            next_open_min = MARKET_OPEN_UTC.minute
            display.print_waiting(
                f"Market closed. Next open: {next_open_hr:02d}:{next_open_min:02d} UTC  "
            )
            try:
                cmd = input_queue.get(timeout=30)
                if cmd and cmd.strip().lower() == "q":
                    break
            except queue.Empty:
                pass
            continue

        # Market open initialization (all tickers in parallel)
        if not day_initialized:
            display.print_info("Market opening — initializing day signals (parallel)...")
            all_ok = True
            init_results: dict = {}

            def _init_ticker(eticker_ev):
                etk, evl = eticker_ev
                try:
                    ok = evl.on_market_open()
                    return (etk, ok, evl.prev_close)
                except Exception as e:
                    return (etk, False, str(e))

            from concurrent.futures import ThreadPoolExecutor
            with ThreadPoolExecutor(max_workers=len(evaluators)) as pool:
                futures = {pool.submit(_init_ticker, item): item[0]
                           for item in evaluators.items()}
                for future in futures:
                    eticker, ok, prev = future.result()
                    if ok:
                        display.print_success(f"  {eticker}: prev_close={prev:.2f}")
                    else:
                        display.print_error(f"  {eticker}: failed ({prev})")
                        all_ok = False

            # Replace the old sequential block
            if all_ok or any(ev.day_initialized for ev in evaluators.values()):
                day_initialized = True
            else:
                display.print_error("No tickers initialized — retrying in 30s")
                time_mod.sleep(30)
                continue

        # --- Check for user commands FIRST (non-blocking) ---
        try:
            cmd = input_queue.get_nowait()
            if cmd:
                logger.info(f"Input queue received: {cmd!r}")
                should_quit = _handle_command(
                    cmd, all_entries, last_exits, tracker, evaluator, display
                )
                if should_quit:
                    break
        except queue.Empty:
            pass

        # --- Evaluate all tickers ---
        all_entries = []
        last_exits = []
        current_price = None
        ticker_prices: dict = {}

        for eticker, ev in evaluators.items():
            if stop_event.is_set():
                break

            try:
                price = ev.get_current_price()
            except Exception as e:
                logger.warning(f"Price fetch failed for {eticker}: {e}")
                price = None

            # Get quote timestamp if available
            quote_ts = None
            if price is not None:
                try:
                    bars = ev._equity_provider.get_bars(eticker, date.today())
                    if bars is not None and not bars.empty and "timestamp" in bars.columns:
                        quote_ts = bars["timestamp"].iloc[-1]
                except Exception:
                    pass

            # Always add to ticker_prices so header shows all tickers
            ticker_prices[eticker] = {
                "price": price,
                "prev_close": ev.prev_close,
                "quote_ts": quote_ts,
            }

            if price is None:
                continue
            if current_price is None:
                current_price = price

            if stop_event.is_set():
                break

            exits = ev.evaluate_exits(price, now)
            last_exits.extend(exits)

            if stop_event.is_set():
                break

            if _is_trading_hours(now):
                entries = ev.evaluate_entries(price, now)
                all_entries.extend(entries)

        # Sort by normalized ROI (best opportunity first), pick top 3
        for rec in all_entries:
            w = rec.spread_width or 1
            cr = rec.credit
            ml = w - cr
            rec._sort_roi = (cr / ml * 100 / (rec.dte + 1)) if ml > 0 else 0
        all_entries.sort(key=lambda r: getattr(r, '_sort_roi', 0), reverse=True)
        last_entries = all_entries[:3]

        if current_price is None:
            display.print_waiting("Waiting for price data...")
            time_mod.sleep(10)
            continue

        # --- Refresh display ---
        display.refresh(
            current_price=current_price,
            prev_close=evaluator.prev_close,
            entries=last_entries,
            exits=last_exits,
            tracker=tracker,
            now=now,
            ticker_prices=ticker_prices,
            all_entries=all_entries,
        )

        # --- Wait for next cycle, checking for input every second ---
        wait_time = max(1, args.interval)
        for _ in range(wait_time):
            if stop_event.is_set():
                break
            try:
                cmd = input_queue.get(timeout=1)
                if cmd:
                    logger.info(f"Input during wait: {cmd!r}")
                    should_quit = _handle_command(
                        cmd, all_entries, last_exits, tracker, evaluator, display
                    )
                    if should_quit:
                        stop_event.set()
                    break  # break wait loop to refresh immediately
            except queue.Empty:
                pass

    # --- End of Session Summary ---
    stop_event.set()
    print(f"\n{'='*80}")
    print(f"  SESSION SUMMARY")
    print(f"{'='*80}")

    all_positions = tracker.get_open_positions() + tracker.get_closed_positions()
    open_pos = tracker.get_open_positions()
    closed_pos = tracker.get_closed_positions()
    print(f"\n  Positions: {len(all_positions)} total "
          f"({len(open_pos)} open, {len(closed_pos)} closed)")

    total_credit = 0
    total_risk = 0
    for pos in all_positions:
        total_credit += pos.total_credit
        total_risk += pos.max_loss
        status = "OPEN" if pos.status == "open" else f"CLOSED ({pos.close_reason})"
        print(f"    {pos.pos_id} {getattr(pos, 'ticker', '?'):<4} "
              f"{pos.direction.upper():<4} "
              f"{pos.short_strike:.0f}/{pos.long_strike:.0f} "
              f"x{pos.num_contracts} cr=${pos.credit:.2f} ({status})")

    if total_credit > 0:
        print(f"\n  Total credit collected: ${total_credit:,.0f}")
        print(f"  Total risk deployed:   ${total_risk:,.0f}")

    _print_rejection_summary(evaluators)

    print(f"{'='*80}\n")

    for ev in evaluators.values():
        ev.close()
    display.print_info("Advisor stopped.")


if __name__ == "__main__":
    main()
