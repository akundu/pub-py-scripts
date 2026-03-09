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
MARKET_CLOSE_UTC = time(20, 0)    # 4:00 PM ET
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

    if action == "q":
        return True

    elif action == "p":
        display.print_positions_detail(tracker, evaluator.get_current_price())

    elif action == "s":
        display.print_summary(tracker)

    elif action == "y":
        # Confirm entry: 'y 1' or 'y 1 3'
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
                f"  Confirmed: {pos.pos_id} {rec.tier_label} "
                f"{rec.direction.upper()} {rec.short_strike:.0f}/{rec.long_strike:.0f} "
                f"x{rec.num_contracts} @ ${rec.credit:.2f}"
            )

    elif action == "x":
        # Close position: 'x <id>' or 'x <id> <exit_price>'
        if len(parts) < 2:
            display.print_error("Usage: x <position_id> [exit_price]")
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
        display.print_error(f"Unknown command: {action}")

    return False


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

    # Load profile
    try:
        profile = load_profile(args.profile)
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

    # Initialize evaluator
    evaluator = TierEvaluator(profile, tracker)
    display.print_info(f"Initializing providers and signal generators for '{profile.name}'...")
    try:
        evaluator.setup()
    except Exception as e:
        display.print_error(f"Setup failed: {e}")
        logger.exception("Setup error")
        return

    display.print_success("Setup complete.")

    # Non-blocking input via background thread
    input_queue = queue.Queue()
    stop_event = threading.Event()

    if interactive:
        input_thread = threading.Thread(
            target=_stdin_reader, args=(input_queue, stop_event), daemon=True
        )
        input_thread.start()

    # Graceful shutdown
    def _signal_handler(sig, frame):
        stop_event.set()
        display.print_info("\nShutting down...")

    signal.signal(signal.SIGINT, _signal_handler)
    signal.signal(signal.SIGTERM, _signal_handler)

    day_initialized = False
    last_entries: list = []
    last_exits: list = []

    display.print_info(
        f"Advisor running: profile={profile.name} ticker={profile.ticker} "
        f"interval={args.interval}s"
    )

    while not stop_event.is_set():
        now = datetime.now(timezone.utc)

        # Outside market hours
        if not _is_market_hours(now):
            # Compute EOD signal at close if we were initialized today
            if day_initialized:
                price = evaluator.get_current_price()
                if price:
                    evaluator.compute_eod_signal(price)
                day_initialized = False
                display.print_info("Market closed. EOD signal computed.")

            next_open_hr = MARKET_OPEN_UTC.hour
            next_open_min = MARKET_OPEN_UTC.minute
            display.print_waiting(
                f"Market closed. Next open: {next_open_hr:02d}:{next_open_min:02d} UTC  "
            )
            # Check for quit command
            try:
                cmd = input_queue.get(timeout=30)
                if cmd and cmd.strip().lower() == "q":
                    break
            except queue.Empty:
                pass
            continue

        # Market open initialization
        if not day_initialized:
            display.print_info("Market opening — initializing day signals...")
            if evaluator.on_market_open():
                day_initialized = True
                display.print_success(
                    f"Day initialized: prev_close={evaluator.prev_close:.2f}"
                )
            else:
                display.print_error("Failed to initialize — retrying in 30s")
                time_mod.sleep(30)
                continue

        # Get current price
        current_price = evaluator.get_current_price()
        if current_price is None:
            display.print_waiting("Waiting for price data...")
            time_mod.sleep(10)
            continue

        # Evaluate exits (every cycle)
        last_exits = evaluator.evaluate_exits(current_price, now)

        # Evaluate entries (only during trading hours)
        if _is_trading_hours(now):
            last_entries = evaluator.evaluate_entries(current_price, now)
        else:
            last_entries = []

        # Refresh display
        display.refresh(
            current_price=current_price,
            prev_close=evaluator.prev_close,
            entries=last_entries,
            exits=last_exits,
            tracker=tracker,
            now=now,
        )

        # Wait for input or timeout
        wait_time = max(1, args.interval - 5)
        try:
            cmd = input_queue.get(timeout=wait_time)
            if cmd:
                should_quit = _handle_command(
                    cmd, last_entries, last_exits, tracker, evaluator, display
                )
                if should_quit:
                    break
        except queue.Empty:
            pass

    # Cleanup
    stop_event.set()
    evaluator.close()
    display.print_info("Advisor stopped.")


if __name__ == "__main__":
    main()
