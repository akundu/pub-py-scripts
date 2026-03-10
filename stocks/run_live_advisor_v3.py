#!/usr/bin/env python3
"""Live Trading Advisor v3 — Cross-Ticker Selection with Volume Awareness.

Runs the same tiered_v2 profile across multiple tickers simultaneously,
comparing opportunities at each evaluation cycle and recommending the best
ticker per tier based on credit/risk ratio, volume adequacy, and bid-ask
tightness.

This is the live counterpart to run_tiered_backtest_v3.py — both use the
same scoring logic and volume-adjusted contract sizing.

All configuration comes from profiles/tiered_v2.yaml (single source of truth).

Requires:
  - QUEST_DB_STRING env var (for realtime equity prices)
  - csv_exports/options/{TICKER}/ (live option chain snapshots)
  - equities_output/{TICKER}/ (historical bars for signal computation)
  - options_csv_output_full/{TICKER}/ (historical options fallback)

Usage:
  python run_live_advisor_v3.py                           # All tickers (NDX, SPX, RUT)
  python run_live_advisor_v3.py --dry-run                 # Show config, no data needed
  python run_live_advisor_v3.py --tickers NDX SPX         # Specific tickers only
  python run_live_advisor_v3.py --volume-fill-pct 0.50    # Allow 50% of volume
  python run_live_advisor_v3.py --weights 0.5,0.25,0.25   # Custom scoring weights
  python run_live_advisor_v3.py --no-volume-cap            # Disable volume caps
  python run_live_advisor_v3.py --positions                # Show tracked positions
  python run_live_advisor_v3.py --summary                  # Show daily summary
  python run_live_advisor_v3.py --interval 30              # Check every 30 seconds
  python run_live_advisor_v3.py --no-interactive           # Non-interactive mode

Interactive commands (during market hours):
  y <priority>   Confirm entry (e.g., 'y 1' or 'y 1 3')
  x <id> [price] Close position (e.g., 'x pos_001 2.50')
  r <id>         Roll position (e.g., 'r pos_001')
  p              Show position details
  s              Show daily summary
  q              Quit
"""

import argparse
import logging
import math
import queue
import signal
import sys
import threading
import time as time_mod
from datetime import date, datetime, time, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

# Ensure project root is in path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.live_trading.advisor.profile_loader import (
    AdvisorProfile,
    load_profile,
)
from scripts.live_trading.advisor.position_tracker import PositionTracker
from scripts.live_trading.advisor.tier_evaluator import TierEvaluator, Recommendation
from scripts.live_trading.advisor.advisor_display import AdvisorDisplay, C

logger = logging.getLogger("live_advisor_v3")

# Market hours (UTC)
MARKET_OPEN_UTC = time(13, 30)   # 9:30 AM ET
MARKET_CLOSE_UTC = time(20, 0)   # 4:00 PM ET
PRE_MARKET_UTC = time(13, 0)     # 9:00 AM ET

# Ticker display colors (for terminal output)
TICKER_ANSI = {"NDX": C.RED, "SPX": C.BLUE, "RUT": C.GREEN}


# ---------------------------------------------------------------------------
# Scoring (same logic as run_tiered_backtest_v3.py)
# ---------------------------------------------------------------------------

def score_recommendation(
    rec: Recommendation,
    options_df: pd.DataFrame,
    weights: Tuple[float, float, float] = (0.40, 0.30, 0.30),
) -> Tuple[float, dict]:
    """Score a live recommendation for cross-ticker comparison.

    Looks up the actual volume and bid-ask from the options chain for the
    recommended strikes.

    Returns: (score, detail_dict)
    """
    w_credit, w_volume, w_bidask = weights

    # Credit/risk component
    max_loss = abs(rec.max_loss) if rec.max_loss > 0 else 1
    credit_risk = rec.total_credit / max_loss if max_loss > 0 else 0
    cr_score = min(1.0, max(0.0, (credit_risk - 0.05) / 0.45))

    # Volume component: look up actual volume at strikes
    short_vol, long_vol, ba_pct = _get_strike_liquidity_live(
        options_df, rec.short_strike, rec.long_strike, rec.direction
    )
    min_vol = min(short_vol, long_vol)
    vol_ratio = min_vol / max(rec.num_contracts, 1)

    if vol_ratio <= 0:
        vol_score = 0.0
    else:
        vol_score = min(1.0, math.log(1 + vol_ratio) / math.log(6))

    # Bid-ask component
    ba_score = max(0.0, min(1.0, (0.25 - ba_pct) / 0.24))

    score = w_credit * cr_score + w_volume * vol_score + w_bidask * ba_score

    detail = {
        "score": score,
        "cr_score": cr_score,
        "vol_score": vol_score,
        "ba_score": ba_score,
        "credit_risk_ratio": credit_risk,
        "min_leg_volume": min_vol,
        "short_volume": short_vol,
        "long_volume": long_vol,
        "bid_ask_pct": ba_pct,
    }
    return score, detail


def _get_strike_liquidity_live(
    options_df: pd.DataFrame,
    short_strike: float,
    long_strike: float,
    direction: str,
) -> Tuple[int, int, float]:
    """Get volume and bid-ask for both legs from a live options chain.

    Returns: (short_volume, long_volume, avg_bid_ask_spread_pct)
    """
    if options_df is None or options_df.empty:
        return 0, 0, 1.0

    def _lookup(strike):
        mask = (options_df["strike"] == strike) & (options_df["type"] == direction)
        rows = options_df[mask]
        if rows.empty:
            mask_near = (abs(options_df["strike"] - strike) <= 1) & (options_df["type"] == direction)
            rows = options_df[mask_near]
        if rows.empty:
            return 0, 1.0

        if "volume" in rows.columns:
            vol = int(pd.to_numeric(rows["volume"], errors="coerce").fillna(0).max())
        else:
            vol = 0
        best = rows.iloc[0]
        bid = float(best.get("bid", 0))
        ask = float(best.get("ask", 0))
        mid = (bid + ask) / 2 if (bid + ask) > 0 else 1
        ba = (ask - bid) / mid if mid > 0 else 1.0
        return vol, ba

    short_vol, short_ba = _lookup(short_strike)
    long_vol, long_ba = _lookup(long_strike)
    avg_ba = (short_ba + long_ba) / 2
    return short_vol, long_vol, avg_ba


def volume_adjusted_contracts(
    requested: int, min_leg_volume: int, volume_fill_pct: float = 0.25
) -> int:
    """Cap contracts at volume_fill_pct of available volume."""
    if min_leg_volume <= 0:
        return 0
    max_from_volume = max(1, int(min_leg_volume * volume_fill_pct))
    return min(requested, max_from_volume)


# ---------------------------------------------------------------------------
# Cross-ticker evaluator wrapper
# ---------------------------------------------------------------------------

class CrossTickerEvaluator:
    """Wraps multiple TierEvaluators (one per ticker) and cross-compares."""

    def __init__(
        self,
        profiles: Dict[str, AdvisorProfile],
        tracker: PositionTracker,
        weights: Tuple[float, float, float] = (0.40, 0.30, 0.30),
        volume_fill_pct: float = 0.25,
        apply_volume_cap: bool = True,
    ):
        self._profiles = profiles
        self._tracker = tracker
        self._evaluators: Dict[str, TierEvaluator] = {}
        self._weights = weights
        self._volume_fill_pct = volume_fill_pct
        self._apply_volume_cap = apply_volume_cap
        self._options_cache: Dict[str, pd.DataFrame] = {}  # ticker -> last options df
        self.prev_closes: Dict[str, float] = {}

    def setup(self) -> None:
        """Initialize all per-ticker evaluators."""
        for ticker, profile in self._profiles.items():
            ev = TierEvaluator(profile, self._tracker)
            ev.setup()
            self._evaluators[ticker] = ev
            logger.info(f"  {ticker} evaluator initialized")

    def on_market_open(self) -> Dict[str, bool]:
        """Initialize all tickers for the day. Returns {ticker: success}."""
        results = {}
        for ticker, ev in self._evaluators.items():
            ok = ev.on_market_open()
            results[ticker] = ok
            if ok:
                self.prev_closes[ticker] = ev.prev_close
        return results

    def get_current_prices(self) -> Dict[str, Optional[float]]:
        """Get latest prices for all tickers."""
        prices = {}
        for ticker, ev in self._evaluators.items():
            prices[ticker] = ev.get_current_price()
        return prices

    def evaluate_entries_cross_ticker(
        self, prices: Dict[str, float], now: datetime
    ) -> List[dict]:
        """Evaluate all tickers, score, and return best per tier.

        Returns list of dicts with:
          - recommendation: Recommendation object
          - ticker: str
          - score: float
          - detail: dict (volume, bid-ask, etc.)
          - adjusted_contracts: int
          - competing: list of (ticker, score) tuples
        """
        # Gather recommendations from all tickers
        all_recs: Dict[str, List[Recommendation]] = {}
        all_options: Dict[str, pd.DataFrame] = {}

        for ticker, ev in self._evaluators.items():
            price = prices.get(ticker)
            if price is None:
                continue
            recs = ev.evaluate_entries(price, now)
            all_recs[ticker] = recs

            # Get the options chain for volume lookup
            try:
                today = date.today()
                dte_buckets = self._profiles[ticker].providers.dte_buckets
                options_prov = ev._options_provider
                if options_prov:
                    opts = options_prov.get_options_chain(ticker, today, dte_buckets=dte_buckets)
                    if opts is not None:
                        all_options[ticker] = opts
            except Exception:
                pass

        # Group by tier label across tickers
        tier_candidates: Dict[str, List[Tuple[str, Recommendation, float, dict]]] = {}
        for ticker, recs in all_recs.items():
            options_df = all_options.get(ticker, pd.DataFrame())
            for rec in recs:
                score, detail = score_recommendation(rec, options_df, self._weights)
                key = rec.tier_label
                if key not in tier_candidates:
                    tier_candidates[key] = []
                tier_candidates[key].append((ticker, rec, score, detail))

        # Select best per tier
        results = []
        for tier_label, candidates in tier_candidates.items():
            # Sort by score descending
            candidates.sort(key=lambda x: x[2], reverse=True)
            competing = [(t, s) for t, _, s, _ in candidates]

            for ticker, rec, score, detail in candidates:
                min_vol = detail["min_leg_volume"]

                # Volume-adjusted sizing
                if self._apply_volume_cap:
                    adj = volume_adjusted_contracts(
                        rec.num_contracts, min_vol, self._volume_fill_pct
                    )
                else:
                    adj = rec.num_contracts

                if adj == 0:
                    continue  # Try next ticker

                results.append({
                    "recommendation": rec,
                    "ticker": ticker,
                    "score": score,
                    "detail": detail,
                    "adjusted_contracts": adj,
                    "competing": competing,
                })
                break  # Only take winner per tier

        # Sort by priority
        results.sort(key=lambda r: r["recommendation"].priority)
        return results

    def evaluate_exits_cross_ticker(
        self, prices: Dict[str, float], now: datetime
    ) -> List[Tuple[str, Recommendation]]:
        """Evaluate exits across all tickers."""
        all_exits = []
        for ticker, ev in self._evaluators.items():
            price = prices.get(ticker)
            if price is None:
                continue
            exits = ev.evaluate_exits(price, now)
            for rec in exits:
                all_exits.append((ticker, rec))
        return all_exits

    def close(self) -> None:
        for ev in self._evaluators.values():
            ev.close()


# ---------------------------------------------------------------------------
# Display helpers
# ---------------------------------------------------------------------------

def _print_header(profile_name: str, tickers: list, weights: tuple,
                  volume_fill_pct: float, apply_volume_cap: bool):
    """Print startup banner."""
    print(f"\n{C.BOLD}{C.CYAN}{'='*80}{C.RESET}")
    print(f"{C.BOLD}{C.CYAN}  Live Advisor v3 — Cross-Ticker Selection{C.RESET}")
    print(f"{C.BOLD}{C.CYAN}{'='*80}{C.RESET}")
    print(f"  Profile:          {profile_name}")
    print(f"  Tickers:          {', '.join(tickers)}")
    print(f"  Scoring weights:  credit/risk={weights[0]}, volume={weights[1]}, bid-ask={weights[2]}")
    print(f"  Volume cap:       {'enabled' if apply_volume_cap else 'disabled'} ({volume_fill_pct*100:.0f}% of available)")
    print(f"  Source:           profiles/tiered_v2.yaml")
    print(f"{C.CYAN}{'='*80}{C.RESET}\n")


def _print_cross_ticker_entries(entries: list, prices: Dict[str, float],
                                prev_closes: Dict[str, float]):
    """Print cross-ticker entry recommendations."""
    if not entries:
        print(f"  {C.DIM}No entry signals at this interval{C.RESET}")
        return

    print(f"\n{C.BOLD}{C.CYAN}  ENTRY RECOMMENDATIONS (cross-ticker best){C.RESET}")
    print(f"  {'Pri':>3} {'Tier':<16} {'Ticker':<6} {'Dir':<5} {'Short':>8} {'Long':>8} "
          f"{'Credit':>8} {'x Adj':>6} {'Total$':>10} {'Score':>6} {'Vol':>5} {'Competing'}")
    print(f"  {'─'*3} {'─'*16} {'─'*6} {'─'*5} {'─'*8} {'─'*8} "
          f"{'─'*8} {'─'*6} {'─'*10} {'─'*6} {'─'*5} {'─'*20}")

    for entry in entries:
        rec = entry["recommendation"]
        ticker = entry["ticker"]
        score = entry["score"]
        detail = entry["detail"]
        adj_contracts = entry["adjusted_contracts"]
        competing = entry["competing"]

        tc = TICKER_ANSI.get(ticker, C.WHITE)
        dc = C.GREEN if rec.direction == "put" else C.RED
        adj_total = rec.credit * adj_contracts * 100

        comp_str = ", ".join(f"{t}({s:.2f})" for t, s in competing)

        print(f"  {rec.priority:>3} {rec.tier_label:<16} "
              f"{tc}{ticker:<6}{C.RESET} "
              f"{dc}{rec.direction:<5}{C.RESET} "
              f"{rec.short_strike:>8.0f} {rec.long_strike:>8.0f} "
              f"${rec.credit:>6.2f} "
              f"x{adj_contracts:>4} "
              f"${adj_total:>8,.0f} "
              f"{score:>5.3f} "
              f"{detail['min_leg_volume']:>5} "
              f"{C.DIM}{comp_str}{C.RESET}")

    print()


def _print_cross_ticker_exits(exits: list):
    """Print cross-ticker exit recommendations."""
    if not exits:
        return

    print(f"\n{C.BOLD}{C.RED}  EXIT/ROLL SIGNALS{C.RESET}")
    for ticker, rec in exits:
        tc = TICKER_ANSI.get(ticker, C.WHITE)
        ac = C.RED if rec.action == "EXIT" else C.YELLOW
        print(f"  {ac}{rec.action}{C.RESET} {tc}{ticker}{C.RESET} "
              f"P{rec.priority} {rec.tier_label} "
              f"{rec.direction} {rec.short_strike:.0f}/{rec.long_strike:.0f} "
              f"x{rec.num_contracts} — {rec.reason}")
    print()


def _print_prices(prices: Dict[str, float], prev_closes: Dict[str, float]):
    """Print current prices with change from close."""
    parts = []
    for ticker in sorted(prices.keys()):
        price = prices[ticker]
        if price is None:
            continue
        tc = TICKER_ANSI.get(ticker, C.WHITE)
        prev = prev_closes.get(ticker)
        if prev and prev > 0:
            chg = (price - prev) / prev * 100
            chg_color = C.GREEN if chg >= 0 else C.RED
            parts.append(f"{tc}{ticker}{C.RESET} {price:,.2f} ({chg_color}{chg:+.2f}%{C.RESET})")
        else:
            parts.append(f"{tc}{ticker}{C.RESET} {price:,.2f}")
    print(f"  {' | '.join(parts)}")


# ---------------------------------------------------------------------------
# Input handling
# ---------------------------------------------------------------------------

def _stdin_reader(q: queue.Queue, stop_event: threading.Event) -> None:
    while not stop_event.is_set():
        try:
            line = sys.stdin.readline()
            if not line:
                break
            line = line.strip()
            if line:
                q.put(line)
        except (EOFError, Exception):
            break


def _handle_command(
    cmd: str,
    entries: list,
    exits: list,
    tracker: PositionTracker,
    evaluators: Dict[str, TierEvaluator],
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
        # Show positions across all tickers
        for ticker, ev in evaluators.items():
            price = ev.get_current_price()
            display.print_positions_detail(tracker, price)
        return False

    elif action == "s":
        display.print_summary(tracker)
        return False

    elif action == "y":
        # Confirm entry: 'y 1' or 'y 1 3'
        priorities = []
        for p in parts[1:]:
            try:
                priorities.append(int(p))
            except ValueError:
                display.print_error(f"Invalid priority: {p}")

        rec_map = {e["recommendation"].priority: e for e in entries}
        for pri in priorities:
            entry = rec_map.get(pri)
            if entry is None:
                display.print_error(f"No entry recommendation with priority {pri}")
                continue
            rec = entry["recommendation"]
            adj = entry["adjusted_contracts"]
            ticker = entry["ticker"]
            pos = tracker.add_position(
                tier_label=f"{ticker}_{rec.tier_label}",
                priority=rec.priority,
                direction=rec.direction,
                short_strike=rec.short_strike,
                long_strike=rec.long_strike,
                credit=rec.credit,
                num_contracts=adj,
                dte=rec.dte,
                entry_price=rec.entry_price,
            )
            tc = TICKER_ANSI.get(ticker, "")
            display.print_success(
                f"  Confirmed: {pos.pos_id} {tc}{ticker}{C.RESET} {rec.tier_label} "
                f"{rec.direction.upper()} {rec.short_strike:.0f}/{rec.long_strike:.0f} "
                f"x{adj} @ ${rec.credit:.2f}"
            )
        return False

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
            display.print_error(f"Position {pos_id} not found")
        return False

    elif action == "r":
        if len(parts) < 2:
            display.print_error("Usage: r <position_id>")
            return False
        pos_id = parts[1]
        # Find which ticker this position belongs to
        pos = tracker.close_position(pos_id, reason="rolled", exit_price=0)
        if pos:
            display.print_success(f"  Rolled out: {pos_id} (closed old leg)")
        else:
            display.print_error(f"Position {pos_id} not found")
        return False

    else:
        display.print_error(f"Unknown command: {action}")

    return False


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _is_market_hours(now: datetime) -> bool:
    t = now.time() if hasattr(now, "time") else now
    return PRE_MARKET_UTC <= t <= MARKET_CLOSE_UTC


def _is_trading_hours(now: datetime) -> bool:
    t = now.time() if hasattr(now, "time") else now
    return MARKET_OPEN_UTC <= t <= MARKET_CLOSE_UTC


def main() -> None:
    parser = argparse.ArgumentParser(
        description='''
Live Trading Advisor v3 — Cross-Ticker Selection.

Runs the tiered_v2 profile across multiple tickers simultaneously, comparing
opportunities at each cycle and recommending the best ticker per tier based on
credit/risk ratio, volume adequacy, and bid-ask tightness.

Single unified daily budget shared across all tickers. Volume-adjusted contract
sizing ensures realistic fills.

All configuration from profiles/tiered_v2.yaml (single source of truth).
        ''',
        epilog='''
Examples:
  %(prog)s
      Run all tickers (NDX, SPX, RUT) with default settings

  %(prog)s --tickers NDX SPX
      Run only NDX and SPX

  %(prog)s --dry-run
      Show configuration and exit (no data sources needed)

  %(prog)s --volume-fill-pct 0.50
      Allow filling up to 50%% of available volume (default: 25%%)

  %(prog)s --weights 0.5,0.25,0.25
      Custom scoring: credit_risk=50%%, volume=25%%, bid-ask=25%%

  %(prog)s --no-volume-cap
      Disable volume caps (v2-style sizing)

  %(prog)s --positions
      Show tracked positions and exit

  %(prog)s --summary
      Show daily summary and exit

Interactive commands during market hours:
  y <priority>     Confirm entry (e.g., 'y 1' or 'y 1 3')
  x <id> [price]   Close position (e.g., 'x pos_001 2.50')
  r <id>           Roll position
  p                Show positions
  s                Show summary
  q                Quit
        ''',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--tickers", nargs="+",
                        help="Tickers to run (default: all from profile)")
    parser.add_argument("--interval", type=int, default=60,
                        help="Seconds between evaluation cycles (default: 60)")
    parser.add_argument("--volume-fill-pct", type=float, default=0.25,
                        help="Max fraction of available volume per trade (default: 0.25)")
    parser.add_argument("--weights", type=str, default="0.40,0.30,0.30",
                        help="Scoring weights: credit,volume,bidask (default: 0.40,0.30,0.30)")
    parser.add_argument("--no-volume-cap", action="store_true",
                        help="Disable volume-based contract capping")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show configuration and exit")
    parser.add_argument("--positions", action="store_true",
                        help="Show tracked positions and exit")
    parser.add_argument("--summary", action="store_true",
                        help="Show daily summary and exit")
    parser.add_argument("--no-interactive", action="store_true",
                        help="Run in log mode without interactive prompts")
    parser.add_argument("--debug", action="store_true",
                        help="Enable debug logging")

    args = parser.parse_args()

    # Logging
    log_level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    # Parse weights
    weights = tuple(float(w) for w in args.weights.split(","))
    if len(weights) != 3 or abs(sum(weights) - 1.0) > 0.01:
        print(f"ERROR: weights must sum to 1.0, got {sum(weights):.3f}")
        sys.exit(1)

    # Load base profile
    try:
        base_profile = load_profile("tiered_v2")
    except (FileNotFoundError, ValueError) as e:
        print(f"Error loading profile: {e}", file=sys.stderr)
        sys.exit(1)

    # Determine tickers
    tickers = args.tickers or base_profile.tickers or ["NDX"]

    # Build one profile per ticker (clone with ticker-specific params)
    profiles: Dict[str, AdvisorProfile] = {}
    for ticker in tickers:
        from copy import deepcopy
        p = deepcopy(base_profile)
        p.ticker = ticker
        # Apply per-ticker spread_width overrides if available
        tp = p.ticker_params.get(ticker, {})
        if tp.get("min_credit"):
            p.strategy_defaults["min_credit"] = tp["min_credit"]
        if tp.get("max_move_cap"):
            p.exit_rules.max_move_cap = tp["max_move_cap"]
        profiles[ticker] = p

    _print_header(base_profile.name, tickers, weights,
                  args.volume_fill_pct, not args.no_volume_cap)

    # Dry run
    if args.dry_run:
        print("  Tier configuration:")
        for t in base_profile.tiers:
            print(f"    P{t.priority} {t.label:<16} DTE={t.dte} P{t.percentile} "
                  f"width={t.spread_width}pt {t.directional}")
        print(f"\n  Per-ticker spread scaling:")
        for ticker in tickers:
            tp = base_profile.ticker_params.get(ticker, {})
            sw_map = tp.get("spread_width_map", {})
            print(f"    {ticker}: {sw_map}, min_credit=${tp.get('min_credit', 0.75)}, "
                  f"max_move_cap={tp.get('max_move_cap', 150)}")
        print("\n  Volume/liquidity characteristics (from backtest analysis):")
        print("    SPX: median vol ~95/strike, 49% >= 100 — deepest liquidity")
        print("    NDX: median vol ~9/strike, 4% >= 100 — moderate")
        print("    RUT: median vol ~8/strike, 1% >= 100 — thinnest")
        return

    tracker = PositionTracker(profile_name="tiered_v3_cross_ticker")

    # Positions-only mode
    if args.positions:
        display = AdvisorDisplay(base_profile, interactive=True)
        display.print_positions_detail(tracker, None)
        return

    # Summary-only mode
    if args.summary:
        display = AdvisorDisplay(base_profile, interactive=True)
        display.print_summary(tracker)
        return

    # Initialize cross-ticker evaluator
    cross_eval = CrossTickerEvaluator(
        profiles=profiles,
        tracker=tracker,
        weights=weights,
        volume_fill_pct=args.volume_fill_pct,
        apply_volume_cap=not args.no_volume_cap,
    )

    interactive = not args.no_interactive
    display = AdvisorDisplay(base_profile, interactive=interactive)
    if not interactive:
        C.disable()

    print(f"  Initializing evaluators for {len(tickers)} tickers...")
    try:
        cross_eval.setup()
    except Exception as e:
        display.print_error(f"Setup failed: {e}")
        logger.exception("Setup error")
        return
    display.print_success("All evaluators initialized.")

    # Non-blocking input
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

    print(f"  Advisor running: interval={args.interval}s, tickers={','.join(tickers)}")
    print(f"  Commands: y <pri> = confirm | x <id> = close | p = positions | s = summary | q = quit\n")

    while not stop_event.is_set():
        now = datetime.now(timezone.utc)

        # Outside market hours
        if not _is_market_hours(now):
            if day_initialized:
                day_initialized = False
                display.print_info("Market closed.")

            display.print_waiting(
                f"Market closed. Next open: {MARKET_OPEN_UTC.hour:02d}:{MARKET_OPEN_UTC.minute:02d} UTC"
            )
            try:
                cmd = input_queue.get(timeout=30)
                if cmd and cmd.strip().lower() == "q":
                    break
            except queue.Empty:
                pass
            continue

        # Market open init
        if not day_initialized:
            display.print_info("Market opening — initializing all tickers...")
            results = cross_eval.on_market_open()
            ok_count = sum(1 for v in results.values() if v)
            if ok_count > 0:
                day_initialized = True
                for t, ok in results.items():
                    status = f"{C.GREEN}OK{C.RESET}" if ok else f"{C.RED}FAILED{C.RESET}"
                    pc = cross_eval.prev_closes.get(t, 0)
                    print(f"    {t}: {status} (prev_close={pc:.2f})")
            else:
                display.print_error("All tickers failed to initialize — retrying in 30s")
                time_mod.sleep(30)
                continue

        # Get current prices
        prices = cross_eval.get_current_prices()
        active_prices = {t: p for t, p in prices.items() if p is not None}
        if not active_prices:
            display.print_waiting("Waiting for price data...")
            time_mod.sleep(10)
            continue

        _print_prices(active_prices, cross_eval.prev_closes)

        # Evaluate exits (every cycle)
        last_exits = cross_eval.evaluate_exits_cross_ticker(active_prices, now)
        _print_cross_ticker_exits(last_exits)

        # Evaluate entries (only during trading hours)
        if _is_trading_hours(now):
            last_entries = cross_eval.evaluate_entries_cross_ticker(active_prices, now)
            _print_cross_ticker_entries(last_entries, active_prices, cross_eval.prev_closes)
        else:
            last_entries = []

        # Wait for input or timeout
        wait_time = max(1, args.interval - 5)
        try:
            cmd = input_queue.get(timeout=wait_time)
            if cmd:
                should_quit = _handle_command(
                    cmd, last_entries, last_exits, tracker,
                    cross_eval._evaluators, display
                )
                if should_quit:
                    break
        except queue.Empty:
            pass

    # Cleanup
    stop_event.set()
    cross_eval.close()
    display.print_info("Advisor stopped.")


if __name__ == "__main__":
    main()
