#!/usr/bin/env python3
"""Autonomous credit spread trader — CLI client for the auto-trader engine.

Sends strategy configuration to the utp_voice auto-trader engine, triggers
simulation runs, and displays results. For autoresearch, an agent modifies
the constants below.

Three execution modes:
  Historical Sim:  --date 2026-04-02               (CSV sim daemon, instant or streamed)
  Shadow Live:     --shadow                         (real IBKR prices, fake fills)
  Live Trading:    --live                           (real IBKR prices, real fills)

Usage:
    # Single day simulation (streaming display, 10s per bar)
    python sim_trader.py --date 2026-04-01

    # Single day simulation (instant backtest, no stream)
    python sim_trader.py --date 2026-04-01 --sim-speed 0

    # Multi-day backtest (no streaming)
    python sim_trader.py --start-date 2026-03-01 --end-date 2026-04-17

    # Shadow mode (real prices, fake fills)
    python sim_trader.py --shadow --interval 30

    # Live mode + follow stream
    python sim_trader.py --live --follow

Examples:
    # Quick single-day test with fast replay
    python sim_trader.py --date 2026-04-01 --sim-speed 5

    # Full Q1 backtest
    python sim_trader.py --start-date 2026-01-02 --end-date 2026-03-31

    # Show current engine config
    python sim_trader.py --show-config
"""

from __future__ import annotations

import argparse
import json
import sys

import requests

# ── Strategy Parameters (agent modifies these) ──────────────────────────────
# Optimized from auto-research sweep (162 combos, Mar-Apr 2026):
#   #1 val_score=4.38: SPX+RUT PUT DTE[0,1,2] 1.5% OTM W15, early entry
#   DTE[0,1,2] >> DTE[0] (140% more P&L), 1.5% OTM sweet spot
#   Early entry (6:30-7:30 PT) best: val=4.38, Sharpe=27, $9.5K/day
TICKERS = ["SPX", "RUT"]
OPTION_TYPES = ["put"]
MAX_TRADES_PER_DAY = 5
MIN_OTM_PCT = 0.015
SPREAD_WIDTH = 15
MIN_CREDIT = 0.25
NUM_CONTRACTS = 10
DTE = [0, 1, 2]
MAX_LOSS_PER_TRADE = 15000
MAX_LOSS_PER_DAY = 75000
PROFIT_TARGET_PCT = 0.50
STOP_LOSS_MULT = 2.0
ENTRY_START_ET = "09:30"  # 6:30 PT — best premiums at open
ENTRY_END_ET = "10:30"    # 7:30 PT — cutoff for new entries
# ─────────────────────────────────────────────────────────────────────────────

DEFAULT_VOICE_URL = "http://localhost:8801"


def build_config() -> dict:
    """Build config dict from the constants above."""
    return {
        "tickers": TICKERS,
        "option_types": OPTION_TYPES,
        "max_trades_per_day": MAX_TRADES_PER_DAY,
        "min_otm_pct": MIN_OTM_PCT,
        "spread_width": SPREAD_WIDTH,
        "min_credit": MIN_CREDIT,
        "num_contracts": NUM_CONTRACTS,
        "dte": DTE,
        "max_loss_per_trade": MAX_LOSS_PER_TRADE,
        "max_loss_per_day": MAX_LOSS_PER_DAY,
        "profit_target_pct": PROFIT_TARGET_PCT,
        "stop_loss_mult": STOP_LOSS_MULT,
        "entry_start_et": ENTRY_START_ET,
        "entry_end_et": ENTRY_END_ET,
    }


def print_day_result(result: dict) -> None:
    """Pretty-print a single day's result."""
    date = result.get("date", "?")
    trades = result.get("trades_taken", 0)
    wins = result.get("wins", 0)
    net_pnl = result.get("net_pnl", 0)
    total_credit = result.get("total_credit", 0)
    total_risk = result.get("total_risk", 0)
    win_rate = result.get("win_rate", 0)

    print(f"\n{'='*60}")
    print(f"  Date: {date}")
    print(f"  Trades: {trades}  |  Wins: {wins}  |  Win Rate: {win_rate:.1%}")
    print(f"  Net P&L: ${net_pnl:,.2f}")
    print(f"  Total Credit: ${total_credit:,.2f}  |  Total Risk: ${total_risk:,.2f}")
    print(f"{'='*60}")

    for t in result.get("trades", []):
        pnl = t.get("realized_pnl", 0)
        pnl_str = f"${pnl:+,.2f}"
        print(f"  {t.get('ticker','?'):>4} {t.get('option_type','?'):>4} "
              f"{t.get('short_strike',0):>8.1f}/{t.get('long_strike',0):<8.1f} "
              f"cr=${t.get('credit',0):.2f}  {t.get('exit_reason','?'):<18} {pnl_str}")


def print_aggregate(result: dict) -> None:
    """Pretty-print aggregate results from a multi-day run."""
    print(f"\n{'='*70}")
    print(f"  AGGREGATE RESULTS: {result.get('start_date')} → {result.get('end_date')}")
    print(f"{'='*70}")
    print(f"  Days traded:  {result.get('days_traded', 0):>6}  |  Days skipped: {result.get('days_skipped', 0)}")
    print(f"  Total trades: {result.get('total_trades', 0):>6}  |  Wins: {result.get('total_wins', 0)}  Losses: {result.get('total_losses', 0)}")
    print(f"  Win rate:     {result.get('win_rate', 0):>6.1%}")
    print(f"  Total P&L:    ${result.get('total_pnl', 0):>12,.2f}")
    print(f"  Peak risk:    ${result.get('peak_risk', 0):>12,.2f}")
    print(f"  Max drawdown: ${result.get('max_drawdown', 0):>12,.2f}")
    print(f"  Profit factor: {result.get('profit_factor', 0):>5.2f}")
    print(f"  Sharpe:        {result.get('sharpe', 0):>5.2f}")
    print(f"  val_score:     {result.get('val_score', 0):>10.6f}")
    print(f"{'='*70}")

    # Per-day summary table
    daily = result.get("daily_results", [])
    if daily:
        print(f"\n  {'Date':<12} {'Trades':>6} {'Wins':>5} {'Net P&L':>12} {'Risk':>10}")
        print(f"  {'-'*12} {'-'*6} {'-'*5} {'-'*12} {'-'*10}")
        for d in daily:
            print(f"  {d.get('date','?'):<12} {d.get('trades_taken',0):>6} "
                  f"{d.get('wins',0):>5} ${d.get('net_pnl',0):>10,.2f} "
                  f"${d.get('total_risk',0):>8,.0f}")


def stream_events(url: str) -> None:
    """Consume SSE stream from the given URL and render tick events to terminal."""
    try:
        with requests.get(url, stream=True, timeout=7200) as resp:
            resp.raise_for_status()
            for line in resp.iter_lines(decode_unicode=True):
                if not line:
                    continue
                if line.startswith(":"):
                    continue  # keepalive
                if not line.startswith("data: "):
                    continue
                try:
                    data = json.loads(line[6:])
                except json.JSONDecodeError:
                    continue

                event_type = data.get("event", "")

                if event_type == "tick":
                    time_pt = data.get("time_pt", data.get("time_et", "??:??"))
                    bar = data.get("bar", "?")
                    total = data.get("total_bars", "?")
                    prices = data.get("prices", {})
                    price_str = "  ".join(f"{t}: {p:,.2f}" for t, p in prices.items() if p)
                    actions = data.get("actions", [])
                    risk = data.get("risk", {})
                    pos_count = risk.get("open_count", 0)
                    daily_pnl = data.get("daily_pnl", 0)
                    used = risk.get("used", 0)
                    cap = risk.get("cap", 0)

                    bar_label = f"Bar {bar}/{total} | " if total != "?" else ""
                    print(f"\n[{time_pt} PT] {bar_label}{price_str}")

                    for action in actions:
                        prefix = "   >>> " if "ENTRY" in action else "   <<< "
                        print(f"{prefix}{action}")

                    print(f"           Positions: {pos_count} | P&L: ${daily_pnl:,.2f} "
                          f"| Risk: ${used:,.0f}/${cap:,.0f}")

                elif event_type == "summary":
                    print("\n" + "=" * 60)
                    print(f"  SUMMARY: {data.get('date', '?')}")
                    print(f"  Trades: {data.get('trades_taken', 0)}  |  "
                          f"Wins: {data.get('wins', 0)}  |  "
                          f"Win Rate: {data.get('win_rate', 0):.1%}")
                    print(f"  Net P&L: ${data.get('net_pnl', 0):,.2f}")
                    print(f"  Total Risk: ${data.get('total_risk', 0):,.2f}")
                    print("=" * 60)

                    for t in data.get("trades", []):
                        pnl = t.get("realized_pnl", 0)
                        print(f"  {t.get('ticker','?'):>4} {t.get('option_type','?'):>4} "
                              f"{t.get('short_strike',0):>8.1f}/{t.get('long_strike',0):<8.1f} "
                              f"cr=${t.get('credit',0):.2f}  "
                              f"{t.get('exit_reason','?'):<18} ${pnl:+,.2f}")

    except KeyboardInterrupt:
        print("\nStream interrupted.")
    except requests.exceptions.ConnectionError:
        print("Connection lost.", file=sys.stderr)


def main():
    parser = argparse.ArgumentParser(
        description="Autonomous credit spread trader — CLI client for the auto-trader engine.",
        epilog="""
Examples:
  %(prog)s --date 2026-04-01
      Run strategy for a single simulated day (streaming, 10s/bar)

  %(prog)s --date 2026-04-01 --sim-speed 0
      Instant backtest (no streaming display)

  %(prog)s --date 2026-04-01 --sim-speed 30
      Slow replay (30s per bar)

  %(prog)s --start-date 2026-03-01 --end-date 2026-04-17
      Multi-day backtest across date range

  %(prog)s --shadow
      Shadow mode: real IBKR prices, fake fills

  %(prog)s --shadow --interval 30
      Shadow mode checking every 30s

  %(prog)s --live
      Start live auto-trading (real prices, real trades)

  %(prog)s --live --follow
      Start live + tail the SSE stream

  %(prog)s --show-config
      Display current engine configuration
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--voice-url", default=DEFAULT_VOICE_URL,
                        help=f"UTP Voice server URL (default: {DEFAULT_VOICE_URL})")
    parser.add_argument("--date", help="Single simulation date (YYYY-MM-DD)")
    parser.add_argument("--start-date", help="Start date for multi-day backtest")
    parser.add_argument("--end-date", help="End date for multi-day backtest")
    parser.add_argument("--live", action="store_true",
                        help="Start live auto-trading (real prices, real trades)")
    parser.add_argument("--shadow", action="store_true",
                        help="Shadow mode: real IBKR prices, no execution")
    parser.add_argument("--follow", action="store_true",
                        help="Tail the SSE stream after starting live/shadow")
    parser.add_argument("--sim-speed", type=float, default=10,
                        help="Seconds between bars in streaming sim (0=instant backtest, default: 10)")
    parser.add_argument("--interval", type=int, default=60,
                        help="Seconds between checks in live/shadow mode (default: 60)")
    parser.add_argument("--no-stream", action="store_true",
                        help="Use non-streaming run-day endpoint (old behavior)")
    parser.add_argument("--show-config", action="store_true",
                        help="Show current engine configuration and exit")
    args = parser.parse_args()

    voice_url = args.voice_url.rstrip("/")
    config = build_config()

    if args.show_config:
        print(json.dumps(config, indent=2))
        return

    # Push config to engine
    try:
        resp = requests.post(f"{voice_url}/api/auto-trader/config", json=config, timeout=10)
        resp.raise_for_status()
    except Exception as e:
        print(f"Error connecting to voice server at {voice_url}: {e}", file=sys.stderr)
        sys.exit(1)

    if args.date:
        if args.no_stream or args.sim_speed == 0:
            # Non-streaming: instant backtest
            print(f"Running strategy for {args.date}...")
            resp = requests.post(f"{voice_url}/api/auto-trader/run-day", json={}, timeout=300)
            resp.raise_for_status()
            result = resp.json()
            print_day_result(result)
        else:
            # Streaming: real-time SSE display
            print(f"Streaming simulation for {args.date} ({args.sim_speed}s/bar)...")
            # POST to start the stream with sim_speed as query param
            stream_url = (
                f"{voice_url}/api/auto-trader/run-day-stream"
                f"?sim_speed={args.sim_speed}"
            )
            stream_events(stream_url)

    elif args.start_date and args.end_date:
        # Multi-day backtest (no streaming)
        print(f"Running backtest: {args.start_date} -> {args.end_date}...")
        resp = requests.post(
            f"{voice_url}/api/auto-trader/run-range",
            json={"start_date": args.start_date, "end_date": args.end_date},
            timeout=3600,
        )
        resp.raise_for_status()
        result = resp.json()
        print_aggregate(result)

    elif args.shadow:
        # Shadow mode: real prices, fake fills
        print("Starting shadow mode (real prices, no execution)...")
        print(f"  Tickers: {config['tickers']}")
        print(f"  Interval: {args.interval}s")
        resp = requests.post(
            f"{voice_url}/api/auto-trader/start-shadow",
            json={**config, "interval_seconds": args.interval},
            timeout=30,
        )
        resp.raise_for_status()
        result = resp.json()
        print(f"  Shadow engine started.")
        print(f"  Stop: curl -X POST {voice_url}/api/auto-trader/stop-shadow")

        # Always follow shadow stream
        print(f"  Tailing shadow stream...\n")
        stream_events(f"{voice_url}/api/auto-trader/shadow-stream")

    elif args.live:
        # Start live auto-trader engine loop
        print("Starting live auto-trading (engine mode)...")
        print(f"  Tickers: {config['tickers']}")
        print(f"  DTE: {config['dte']}")
        print(f"  OTM: {config['min_otm_pct']:.1%}  Width: {config['spread_width']}")
        print(f"  Entry window: {config['entry_start_et']}-{config['entry_end_et']} ET "
              f"({int(config['entry_start_et'].split(':')[0])-3}:{config['entry_start_et'].split(':')[1]}"
              f"-{int(config['entry_end_et'].split(':')[0])-3}:{config['entry_end_et'].split(':')[1]} PT)")
        print(f"  Max trades/day: {config['max_trades_per_day']}")
        resp = requests.post(
            f"{voice_url}/api/auto-trader/start-live",
            json={**config, "interval_seconds": args.interval},
            timeout=30,
        )
        resp.raise_for_status()
        result = resp.json()
        print(f"\n  Engine started. Checking every {result.get('interval_seconds', 60)}s.")
        print(f"  Monitor: curl {voice_url}/api/auto-trader/config")
        print(f"  Stop:    curl -X POST {voice_url}/api/auto-trader/stop-live")

        if args.follow:
            print(f"  Tailing live stream...\n")
            stream_events(f"{voice_url}/api/auto-trader/live-stream")

    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
