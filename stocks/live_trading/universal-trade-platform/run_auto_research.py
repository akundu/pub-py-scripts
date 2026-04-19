#!/usr/bin/env python3
"""Auto-research sweep — tests all parameter combinations via the auto-trader engine.

Drives the utp_voice auto-trader engine via HTTP, iterating over DTE combos,
tickers, option types, OTM%, and spread widths. Saves full results (including
every losing trade) and produces a ranked summary.

Usage:
    # Full sweep (4,050 combos)
    python run_auto_research.py \\
        --voice-url http://localhost:8801 \\
        --start 2026-01-02 --end 2026-04-17

    # Quick sweep (~135 combos)
    python run_auto_research.py --quick \\
        --voice-url http://localhost:8801 \\
        --start 2026-01-02 --end 2026-04-17

    # Resume interrupted run
    python run_auto_research.py --resume results/auto_research/research_20260418_1200.json \\
        --voice-url http://localhost:8801 \\
        --start 2026-01-02 --end 2026-04-17

Examples:
    python run_auto_research.py --quick --start 2026-03-01 --end 2026-04-01
        Quick sweep over ~135 parameter combos

    python run_auto_research.py --start 2026-01-02 --end 2026-04-17 --top-n 20
        Full sweep, show top 20 results
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import requests

# ── Sweep Dimensions ──────────────────────────────────────────────────────────

FULL_DTE_COMBOS = [
    [0], [1], [2], [3],
    [0, 1], [0, 1, 2], [0, 1, 2, 3],
    [1, 2], [2, 3],
]

FULL_TICKERS = [
    ["SPX"], ["RUT"], ["NDX"],
    ["SPX", "RUT"], ["SPX", "NDX"],
    ["SPX", "RUT", "NDX"],
]

FULL_OPTION_TYPES = [["put"], ["call"], ["put", "call"]]

FULL_OTM_PCTS = [0.01, 0.015, 0.02, 0.025, 0.03]

FULL_WIDTHS = [10, 15, 20, 25, 30]

# Quick mode: reduced dimensions
QUICK_DTE_COMBOS = [[0], [0, 1], [0, 1, 2]]
QUICK_TICKERS = [["SPX"], ["RUT"], ["NDX"]]
QUICK_OPTION_TYPES = [["put"], ["call"], ["put", "call"]]
QUICK_OTM_PCTS = [0.015, 0.02, 0.03]
QUICK_WIDTHS = [15, 25]

DEFAULT_VOICE_URL = "http://localhost:8801"


def build_combos(quick: bool = False) -> list[dict]:
    """Build all parameter combinations for the sweep."""
    if quick:
        dte_combos = QUICK_DTE_COMBOS
        tickers = QUICK_TICKERS
        option_types = QUICK_OPTION_TYPES
        otm_pcts = QUICK_OTM_PCTS
        widths = QUICK_WIDTHS
    else:
        dte_combos = FULL_DTE_COMBOS
        tickers = FULL_TICKERS
        option_types = FULL_OPTION_TYPES
        otm_pcts = FULL_OTM_PCTS
        widths = FULL_WIDTHS

    combos = []
    for dte, tkr, otype, otm, w in itertools.product(
        dte_combos, tickers, option_types, otm_pcts, widths,
    ):
        combos.append({
            "dte": dte,
            "tickers": tkr,
            "option_types": otype,
            "min_otm_pct": otm,
            "spread_width": w,
        })
    return combos


def make_label(combo: dict) -> str:
    """Short label for a combo."""
    dte_s = "+".join(str(d) for d in combo["dte"])
    tkr_s = "+".join(combo["tickers"])
    otype_s = "+".join(combo["option_types"])
    return f"DTE{dte_s}_{tkr_s}_{otype_s}_OTM{combo['min_otm_pct']}_W{combo['spread_width']}"


def run_combo(
    voice_url: str, combo: dict, start_date: str, end_date: str,
) -> dict:
    """Run a single parameter combo through the engine and return results."""
    # max_loss_per_trade must accommodate width * 100 * num_contracts
    num_contracts = 10
    width = combo["spread_width"]
    max_loss_per_trade = width * 100 * num_contracts  # worst case = full width
    config = {
        "tickers": combo["tickers"],
        "option_types": combo["option_types"],
        "min_otm_pct": combo["min_otm_pct"],
        "spread_width": width,
        "dte": combo["dte"],
        "max_trades_per_day": 5,
        "min_credit": 0.25,
        "num_contracts": num_contracts,
        "max_loss_per_trade": max_loss_per_trade,
        "max_loss_per_day": max_loss_per_trade * 5,
        "profit_target_pct": 0.50,
        "stop_loss_mult": 2.0,
        "entry_start_et": "09:45",
        "entry_end_et": "15:00",
    }

    # Push config
    resp = requests.post(f"{voice_url}/api/auto-trader/config", json=config, timeout=10)
    resp.raise_for_status()

    # Run range
    resp = requests.post(
        f"{voice_url}/api/auto-trader/run-range",
        json={"start_date": start_date, "end_date": end_date},
        timeout=7200,  # 2h timeout for long runs
    )
    resp.raise_for_status()
    result = resp.json()

    # Extract losing trades detail
    losing_trades = []
    worst_day_pnl = 0.0
    worst_day_date = ""
    max_consecutive_losses = 0
    current_streak = 0

    for day in result.get("daily_results", []):
        day_pnl = day.get("net_pnl", 0)
        if day_pnl < worst_day_pnl:
            worst_day_pnl = day_pnl
            worst_day_date = day.get("date", "")

        for t in day.get("trades", []):
            rpnl = t.get("realized_pnl", 0)
            if rpnl < 0:
                current_streak += 1
                max_consecutive_losses = max(max_consecutive_losses, current_streak)
                losing_trades.append({
                    "date": day.get("date", ""),
                    "ticker": t.get("ticker", ""),
                    "option_type": t.get("option_type", ""),
                    "short_strike": t.get("short_strike", 0),
                    "long_strike": t.get("long_strike", 0),
                    "credit": t.get("credit", 0),
                    "exit_reason": t.get("exit_reason", ""),
                    "realized_pnl": rpnl,
                    "dte": t.get("dte", 0),
                    "expiration": t.get("expiration", ""),
                })
            else:
                current_streak = 0

    return {
        "label": make_label(combo),
        "combo": combo,
        "val_score": result.get("val_score", 0),
        "total_pnl": result.get("total_pnl", 0),
        "win_rate": result.get("win_rate", 0),
        "sharpe": result.get("sharpe", 0),
        "max_drawdown": result.get("max_drawdown", 0),
        "profit_factor": result.get("profit_factor", 0),
        "total_trades": result.get("total_trades", 0),
        "total_wins": result.get("total_wins", 0),
        "total_losses": result.get("total_losses", 0),
        "days_traded": result.get("days_traded", 0),
        "peak_risk": result.get("peak_risk", 0),
        "losing_trades": losing_trades,
        "worst_day": {"date": worst_day_date, "pnl": worst_day_pnl},
        "max_consecutive_losses": max_consecutive_losses,
    }


def print_top_n(results: list[dict], n: int = 10) -> None:
    """Print top-N ranked results table."""
    ranked = sorted(results, key=lambda r: r.get("val_score", 0), reverse=True)
    print(f"\n{'='*100}")
    print(f"  TOP {n} RESULTS (ranked by val_score)")
    print(f"{'='*100}")
    print(f"  {'#':>3}  {'Label':<50} {'val_score':>10} {'P&L':>12} {'WR':>6} {'Sharpe':>7} {'MaxDD':>10} {'PF':>6} {'Trades':>6}")
    print(f"  {'-'*3}  {'-'*50} {'-'*10} {'-'*12} {'-'*6} {'-'*7} {'-'*10} {'-'*6} {'-'*6}")

    for i, r in enumerate(ranked[:n], 1):
        print(
            f"  {i:>3}  {r['label']:<50} "
            f"{r['val_score']:>10.6f} "
            f"${r['total_pnl']:>10,.2f} "
            f"{r['win_rate']:>5.1%} "
            f"{r['sharpe']:>7.2f} "
            f"${r['max_drawdown']:>8,.2f} "
            f"{r['profit_factor']:>6.2f} "
            f"{r['total_trades']:>6}"
        )

    # Failure analysis for top configs
    print(f"\n{'='*100}")
    print(f"  FAILURE ANALYSIS — Top {min(n, 5)} Configs")
    print(f"{'='*100}")
    for i, r in enumerate(ranked[:min(n, 5)], 1):
        lt = r.get("losing_trades", [])
        wd = r.get("worst_day", {})
        print(f"\n  #{i} {r['label']}")
        print(f"      Losing trades: {len(lt)}, Max consecutive losses: {r.get('max_consecutive_losses', 0)}")
        if wd.get("date"):
            print(f"      Worst day: {wd['date']} → ${wd['pnl']:,.2f}")
        if lt:
            for loss in lt[:5]:
                print(
                    f"      {loss['date']} {loss['ticker']:>4} {loss['option_type']:>4} "
                    f"{loss['short_strike']}/{loss['long_strike']} "
                    f"cr={loss['credit']:.2f} {loss['exit_reason']:<18} ${loss['realized_pnl']:+,.2f}"
                )
            if len(lt) > 5:
                print(f"      ... and {len(lt) - 5} more")


def main():
    parser = argparse.ArgumentParser(
        description="Auto-research sweep — test all parameter combinations via auto-trader engine.",
        epilog="""
Examples:
  %(prog)s --quick --start 2026-03-01 --end 2026-04-01
      Quick sweep (~135 combos)

  %(prog)s --start 2026-01-02 --end 2026-04-17 --top-n 20
      Full sweep (4,050 combos), show top 20

  %(prog)s --resume results/auto_research/research_20260418.json \\
      --start 2026-01-02 --end 2026-04-17
      Resume an interrupted run
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--voice-url", default=DEFAULT_VOICE_URL,
                        help=f"UTP Voice server URL (default: {DEFAULT_VOICE_URL})")
    parser.add_argument("--start", required=True, help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", required=True, help="End date (YYYY-MM-DD)")
    parser.add_argument("--quick", action="store_true",
                        help="Reduced sweep (~135 combos instead of 4,050)")
    parser.add_argument("--output-dir", default="results/auto_research",
                        help="Output directory (default: results/auto_research)")
    parser.add_argument("--top-n", type=int, default=10,
                        help="Number of top results to display (default: 10)")
    parser.add_argument("--resume", help="Resume from a previous research JSON file")
    args = parser.parse_args()

    voice_url = args.voice_url.rstrip("/")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    results_path = output_dir / f"research_{timestamp}.json"
    summary_path = output_dir / f"summary_{timestamp}.csv"

    # Build combos
    combos = build_combos(quick=args.quick)
    print(f"Sweep: {len(combos)} parameter combinations")
    print(f"Date range: {args.start} → {args.end}")
    print(f"Output: {results_path}")

    # Load previous results for resume
    completed: dict[str, dict] = {}
    if args.resume:
        resume_path = Path(args.resume)
        if resume_path.exists():
            with open(resume_path) as f:
                prev = json.load(f)
            for r in prev.get("results", []):
                completed[r["label"]] = r
            print(f"Resuming: {len(completed)} combos already completed")

    # Verify connection
    try:
        resp = requests.get(f"{voice_url}/api/auto-trader/config", timeout=5)
        resp.raise_for_status()
    except Exception as e:
        print(f"Error: Cannot connect to voice server at {voice_url}: {e}", file=sys.stderr)
        sys.exit(1)

    results: list[dict] = list(completed.values())
    start_time = time.time()

    for i, combo in enumerate(combos, 1):
        label = make_label(combo)
        if label in completed:
            continue

        elapsed = time.time() - start_time
        rate = (i - len(completed)) / max(elapsed, 1) if i > len(completed) else 0
        remaining = (len(combos) - i) / rate if rate > 0 else 0
        print(
            f"[{i}/{len(combos)}] {label}  "
            f"({len(results)} done, ~{remaining/60:.0f}m remaining)",
            end="... ",
            flush=True,
        )

        try:
            result = run_combo(voice_url, combo, args.start, args.end)
            results.append(result)
            completed[label] = result
            print(
                f"val={result['val_score']:.4f} P&L=${result['total_pnl']:,.0f} "
                f"WR={result['win_rate']:.0%} trades={result['total_trades']}"
            )
        except Exception as e:
            print(f"FAILED: {e}")
            results.append({
                "label": label,
                "combo": combo,
                "val_score": 0,
                "total_pnl": 0,
                "error": str(e),
            })

        # Incremental save every 5 combos
        if len(results) % 5 == 0:
            with open(results_path, "w") as f:
                json.dump({"results": results, "start": args.start, "end": args.end}, f)

    # Final save
    with open(results_path, "w") as f:
        json.dump(
            {"results": results, "start": args.start, "end": args.end,
             "quick": args.quick, "total_combos": len(combos)},
            f, indent=2,
        )

    # Write CSV summary
    csv_fields = [
        "label", "val_score", "total_pnl", "win_rate", "sharpe",
        "max_drawdown", "profit_factor", "total_trades", "total_wins",
        "total_losses", "peak_risk", "max_consecutive_losses",
    ]
    ranked = sorted(results, key=lambda r: r.get("val_score", 0), reverse=True)
    with open(summary_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=csv_fields, extrasaction="ignore")
        writer.writeheader()
        for r in ranked:
            writer.writerow(r)

    elapsed = time.time() - start_time
    print(f"\nCompleted {len(results)} combos in {elapsed/60:.1f} minutes")
    print(f"Results: {results_path}")
    print(f"Summary: {summary_path}")

    print_top_n(results, args.top_n)


if __name__ == "__main__":
    main()
