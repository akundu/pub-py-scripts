#!/usr/bin/env python3
"""Live Trading Advisor v2 — Multi-tier credit spread recommendations.

This is the backwards-compatible entry point that uses the tiered_v2 profile.
For the generic profile-based advisor, use run_live_advisor.py instead.

Usage:
  python run_live_advisor_v2.py                  # Run advisor (NDX)
  python run_live_advisor_v2.py --ticker SPX     # Different ticker
  python run_live_advisor_v2.py --dry-run        # Show config, don't run
  python run_live_advisor_v2.py --positions      # Show tracked positions
  python run_live_advisor_v2.py --summary        # Show daily summary
  python run_live_advisor_v2.py --interval 30    # Check every 30 seconds
  python run_live_advisor_v2.py --no-interactive # Log mode (no prompts)
"""

import argparse
import logging
import queue
import signal
import sys
import threading
import time as time_mod
from datetime import datetime, time, timezone
from pathlib import Path

# Ensure project root is in path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.live_trading.advisor.profile_loader import load_profile
from scripts.live_trading.advisor.position_tracker import PositionTracker
from scripts.live_trading.advisor.tier_evaluator import TierEvaluator, Recommendation
from scripts.live_trading.advisor.advisor_display import AdvisorDisplay, C

logger = logging.getLogger("live_advisor_v2")

# Market hours (UTC)
MARKET_OPEN_UTC = time(13, 30)    # 9:30 AM ET
MARKET_CLOSE_UTC = time(20, 0)    # 4:00 PM ET
PRE_MARKET_UTC = time(13, 0)      # 9:00 AM ET


def _is_market_hours(now: datetime) -> bool:
    t = now.time() if hasattr(now, "time") else now
    return PRE_MARKET_UTC <= t <= MARKET_CLOSE_UTC


def _is_trading_hours(now: datetime) -> bool:
    t = now.time() if hasattr(now, "time") else now
    return MARKET_OPEN_UTC <= t <= MARKET_CLOSE_UTC


def _stdin_reader(q: queue.Queue, stop_event: threading.Event) -> None:
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


def _handle_command(
    cmd: str, entries: list, exits: list,
    tracker: PositionTracker, evaluator: TierEvaluator, display: AdvisorDisplay,
) -> bool:
    parts = cmd.strip().split()
    if not parts:
        return False
    action = parts[0].lower()
    if action == "q":
        return True
    elif action == "p":
        display.print_positions_detail(tracker, evaluator.get_current_price())
    elif action == "s":
        display.print_summary(tracker)
    elif action == "y":
        rec_map = {r.priority: r for r in entries}
        for p in parts[1:]:
            try:
                pri = int(p)
            except ValueError:
                display.print_error(f"Invalid priority: {p}")
                continue
            rec = rec_map.get(pri)
            if rec is None:
                display.print_error(f"No entry recommendation with priority {pri}")
                continue
            pos = tracker.add_position(
                tier_label=rec.tier_label, priority=rec.priority,
                direction=rec.direction, short_strike=rec.short_strike,
                long_strike=rec.long_strike, credit=rec.credit,
                num_contracts=rec.num_contracts, dte=rec.dte,
                entry_price=rec.entry_price,
            )
            display.print_success(
                f"  Confirmed: {pos.pos_id} {rec.tier_label} "
                f"{rec.direction.upper()} {rec.short_strike:.0f}/{rec.long_strike:.0f} "
                f"x{rec.num_contracts} @ ${rec.credit:.2f}"
            )
    elif action == "x":
        if len(parts) < 2:
            display.print_error("Usage: x <position_id> [exit_price]")
            return False
        pos_id = parts[1]
        exit_price = float(parts[2]) if len(parts) >= 3 else 0.0
        pos = tracker.close_position(pos_id, reason="manual_close", exit_price=exit_price)
        if pos:
            pnl_str = f" P&L: ${pos.realized_pnl:,.0f}" if exit_price > 0 else ""
            display.print_success(f"  Closed: {pos_id}{pnl_str}")
        else:
            display.print_error(f"Position {pos_id} not found or already closed")
    elif action == "r":
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
        display.print_error(f"Unknown command: {action}")
    return False


def main() -> None:
    parser = argparse.ArgumentParser(
        description="""
Live Trading Advisor v2 — Multi-tier credit spread recommendations.

Evaluates 9 DTE tiers from the tiered_portfolio_v2 backtest against real-time
data and displays entry/exit/roll recommendations. Advisory only.

Note: This is the backwards-compatible entry point. For the generic
profile-based advisor, use run_live_advisor.py --profile tiered_v2.
        """,
        epilog="""
Examples:
  %(prog)s
      Run the advisor for NDX (default) during market hours

  %(prog)s --ticker SPX
      Run the advisor for SPX

  %(prog)s --dry-run
      Show tier configuration without connecting to data sources

  %(prog)s --positions
      Show currently tracked positions and exit

  %(prog)s --summary
      Show daily performance summary and exit

  %(prog)s --interval 30
      Check for signals every 30 seconds (default: 60)

  %(prog)s --no-interactive
      Run in log mode without interactive prompts (for piping/logging)
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--ticker", default="NDX", help="Ticker symbol (default: NDX)")
    parser.add_argument("--interval", type=int, default=60, help="Seconds between cycles (default: 60)")
    parser.add_argument("--dry-run", action="store_true", help="Show config and exit")
    parser.add_argument("--positions", action="store_true", help="Show positions and exit")
    parser.add_argument("--summary", action="store_true", help="Show summary and exit")
    parser.add_argument("--no-interactive", action="store_true", help="Log mode (no prompts)")
    parser.add_argument("--debug", action="store_true", help="Debug logging")
    args = parser.parse_args()

    log_level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    # Load the tiered_v2 profile
    profile = load_profile("tiered_v2")
    if args.ticker != "NDX":
        profile.ticker = args.ticker

    interactive = not args.no_interactive
    display = AdvisorDisplay(profile, interactive=interactive)
    if not interactive:
        C.disable()

    if args.dry_run:
        display.print_dry_run_config()
        return

    tracker = PositionTracker(profile_name=profile.name)

    if args.positions:
        display.print_positions_detail(tracker, None)
        return

    if args.summary:
        display.print_summary(tracker)
        return

    evaluator = TierEvaluator(profile, tracker)
    display.print_info("Initializing providers and signal generators...")
    try:
        evaluator.setup()
    except Exception as e:
        display.print_error(f"Setup failed: {e}")
        logger.exception("Setup error")
        return

    display.print_success("Setup complete.")

    input_queue = queue.Queue()
    stop_event = threading.Event()
    if interactive:
        input_thread = threading.Thread(
            target=_stdin_reader, args=(input_queue, stop_event), daemon=True
        )
        input_thread.start()

    def _signal_handler(sig, frame):
        stop_event.set()
        display.print_info("\nShutting down...")

    signal.signal(signal.SIGINT, _signal_handler)
    signal.signal(signal.SIGTERM, _signal_handler)

    day_initialized = False
    last_entries: list = []
    last_exits: list = []
    display.print_info(f"Advisor running for {profile.ticker} | interval={args.interval}s")

    while not stop_event.is_set():
        now = datetime.now(timezone.utc)
        if not _is_market_hours(now):
            if day_initialized:
                price = evaluator.get_current_price()
                if price:
                    evaluator.compute_eod_signal(price)
                day_initialized = False
                display.print_info("Market closed. EOD signal computed.")
            display.print_waiting(
                f"Market closed. Next open: {MARKET_OPEN_UTC.hour:02d}:{MARKET_OPEN_UTC.minute:02d} UTC  "
            )
            try:
                cmd = input_queue.get(timeout=30)
                if cmd and cmd.strip().lower() == "q":
                    break
            except queue.Empty:
                pass
            continue

        if not day_initialized:
            display.print_info("Market opening — initializing day signals...")
            if evaluator.on_market_open():
                day_initialized = True
                display.print_success(f"Day initialized: prev_close={evaluator.prev_close:.2f}")
            else:
                display.print_error("Failed to initialize — retrying in 30s")
                time_mod.sleep(30)
                continue

        current_price = evaluator.get_current_price()
        if current_price is None:
            display.print_waiting("Waiting for price data...")
            time_mod.sleep(10)
            continue

        last_exits = evaluator.evaluate_exits(current_price, now)
        if _is_trading_hours(now):
            last_entries = evaluator.evaluate_entries(current_price, now)
        else:
            last_entries = []

        display.refresh(
            current_price=current_price, prev_close=evaluator.prev_close,
            entries=last_entries, exits=last_exits, tracker=tracker, now=now,
        )

        wait_time = max(1, args.interval - 5)
        try:
            cmd = input_queue.get(timeout=wait_time)
            if cmd:
                if _handle_command(cmd, last_entries, last_exits, tracker, evaluator, display):
                    break
        except queue.Empty:
            pass

    stop_event.set()
    evaluator.close()
    display.print_info("Advisor stopped.")


if __name__ == "__main__":
    main()
