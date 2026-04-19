#!/usr/bin/env python3
"""Comprehensive auto-trader parameter sweep with diversity toggle and parallelism.

Expands on run_auto_research.py by sweeping ALL tunable parameters:
  - Tickers (SPX, RUT, NDX, combos)
  - Option types (put, call, both)
  - DTE combos (0DTE-only vs multi-DTE)
  - OTM% (distance from spot)
  - Spread widths
  - Number of contracts
  - Max trades per day
  - Entry windows (start/end times)
  - Profit target %
  - Stop loss multiplier
  - Diversity on/off

Parallelism:
  --workers N spawns N independent (daemon, voice) pairs on consecutive ports.
  Each worker runs combos concurrently via ThreadPoolExecutor. Each combo takes
  ~110s serially; with 8 workers a 432-combo sweep finishes in ~1 hour.

Two phases:
  Phase 1 — Core sweep: tickers x types x DTE x OTM x width x contracts x diversity
  Phase 2 — Fine-tune: top-N configs re-run with entry window / exit rule variations

Output:
  results/full_sweep/sweep_{timestamp}.json   — all results + losing trades
  results/full_sweep/summary_{timestamp}.csv  — ranked by val_score

Usage:
    # Parallel quick sweep (8 workers, ~1 hour)
    python run_full_sweep.py --quick --workers 8 \\
        --start 2026-02-18 --end 2026-04-17

    # Single worker (sequential, ~6 hours)
    python run_full_sweep.py --quick \\
        --start 2026-02-18 --end 2026-04-17

    # Full phase 1 + phase 2 fine-tune
    python run_full_sweep.py --quick --fine-tune --workers 8 \\
        --start 2026-02-18 --end 2026-04-17

    # Resume interrupted run
    python run_full_sweep.py --quick --workers 8 \\
        --start 2026-02-18 --end 2026-04-17 \\
        --resume results/full_sweep/sweep_20260419_1200.json

    # Use pre-started voice servers (skip auto-spawn)
    python run_full_sweep.py --quick \\
        --voice-urls http://localhost:8801,http://localhost:8802

Examples:
    python run_full_sweep.py --quick --workers 8 --start 2026-02-18 --end 2026-04-17
    python run_full_sweep.py --quick --workers 8 --fine-tune --start 2026-02-18 --end 2026-04-17
"""

from __future__ import annotations

import argparse
import atexit
import csv
import itertools
import json
import os
import signal
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

import requests

# ── Phase 1: Core Sweep Dimensions ──────────────────────────────────────────

FULL_DTE_COMBOS = [
    [0], [0, 1], [0, 1, 2], [1], [1, 2],
]
FULL_TICKERS = [
    ["SPX"], ["RUT"], ["NDX"], ["SPX", "RUT"], ["SPX", "RUT", "NDX"],
]
FULL_OPTION_TYPES = [["put"], ["call"], ["put", "call"]]
FULL_OTM_PCTS = [0.01, 0.015, 0.02, 0.025, 0.03]
FULL_WIDTHS = [10, 15, 20, 25, 30, 50]
FULL_NUM_CONTRACTS = [10, 20, 40]
FULL_MAX_TRADES = [5, 8]
FULL_DIVERSITY = [True, False]

# Quick mode
QUICK_DTE_COMBOS = [[0], [0, 1, 2], [1, 2]]
QUICK_TICKERS = [["SPX"], ["RUT"], ["NDX"], ["SPX", "RUT"], ["SPX", "RUT", "NDX"]]
QUICK_OPTION_TYPES = [["put"], ["put", "call"]]
QUICK_OTM_PCTS = [0.015, 0.02, 0.03]
QUICK_WIDTHS = [15, 25]
QUICK_NUM_CONTRACTS = [10, 20]
QUICK_MAX_TRADES = [5]
QUICK_DIVERSITY = [True, False]

# Phase 2: Fine-tune
FINE_ENTRY_STARTS = ["09:30", "09:45", "10:00", "10:30"]
FINE_ENTRY_ENDS = ["10:30", "11:00", "12:00", "13:00", "15:00"]
FINE_PROFIT_TARGETS = [0.30, 0.40, 0.50, 0.60, 0.70]
FINE_STOP_LOSSES = [1.5, 2.0, 2.5, 3.0]

# ── Worker Management ───────────────────────────────────────────────────────

_spawned_procs: list[subprocess.Popen] = []


def _cleanup_workers():
    """Kill all spawned daemon/voice processes."""
    for p in _spawned_procs:
        try:
            p.terminate()
        except Exception:
            pass
    # Wait briefly then force kill
    time.sleep(1)
    for p in _spawned_procs:
        try:
            p.kill()
        except Exception:
            pass


atexit.register(_cleanup_workers)


def _wait_for_port(url: str, timeout: int = 120) -> bool:
    """Poll until the URL responds or timeout."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            resp = requests.get(url, timeout=3)
            if resp.status_code < 500:
                return True
        except Exception:
            pass
        time.sleep(1)
    return False


def spawn_workers(
    n: int,
    base_daemon_port: int,
    base_voice_port: int,
    options_dir: str,
    equities_dir: str,
    tickers: str,
) -> list[str]:
    """Spawn N (daemon, voice) pairs and return list of voice URLs.

    Each daemon gets its own port and data dir for isolation.
    Each voice server connects to its paired daemon.
    """
    voice_urls = []
    script_dir = Path(__file__).parent

    for i in range(n):
        daemon_port = base_daemon_port + i
        voice_port = base_voice_port + i
        data_dir = f"data/utp/sweep_worker_{i}"

        # Check if daemon port already in use
        try:
            resp = requests.get(f"http://localhost:{daemon_port}/sim/status", timeout=2)
            if resp.status_code == 200:
                print(f"  Worker {i}: reusing existing daemon on port {daemon_port}")
                # Still need to spawn voice if not running
                try:
                    resp2 = requests.get(f"http://localhost:{voice_port}/api/auto-trader/config", timeout=2)
                    if resp2.status_code == 200:
                        print(f"  Worker {i}: reusing existing voice on port {voice_port}")
                        voice_urls.append(f"http://localhost:{voice_port}")
                        continue
                except Exception:
                    pass
                # Spawn voice only
                env = dict(os.environ)
                env["UTP_DAEMON_URL"] = f"http://localhost:{daemon_port}"
                env["UTP_VOICE_JWT_SECRET"] = "sweep-worker"
                voice_proc = subprocess.Popen(
                    [sys.executable, str(script_dir / "utp_voice.py"),
                     "serve", "--port", str(voice_port), "--public"],
                    env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                )
                _spawned_procs.append(voice_proc)
                voice_urls.append(f"http://localhost:{voice_port}")
                continue
        except Exception:
            pass

        # Spawn daemon
        daemon_cmd = [
            sys.executable, str(script_dir / "utp.py"), "daemon",
            "--sim-date", "2026-04-01",  # placeholder, load-date changes it
            "--tickers", tickers,
            "--options-dir", options_dir,
            "--equities-dir", equities_dir,
            "--server-port", str(daemon_port),
            "--data-dir", data_dir,
        ]
        daemon_proc = subprocess.Popen(
            daemon_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        _spawned_procs.append(daemon_proc)

        # Spawn voice
        env = dict(os.environ)
        env["UTP_DAEMON_URL"] = f"http://localhost:{daemon_port}"
        env["UTP_VOICE_JWT_SECRET"] = "sweep-worker"
        voice_proc = subprocess.Popen(
            [sys.executable, str(script_dir / "utp_voice.py"),
             "serve", "--port", str(voice_port), "--public"],
            env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        _spawned_procs.append(voice_proc)
        voice_urls.append(f"http://localhost:{voice_port}")

    # Wait for all to be ready
    print(f"  Waiting for {n} workers to start...")
    for i, url in enumerate(voice_urls):
        if not _wait_for_port(f"{url}/api/auto-trader/config", timeout=120):
            print(f"  WARNING: Worker {i} ({url}) failed to start", file=sys.stderr)
        else:
            print(f"  Worker {i}: ready at {url}")

    return voice_urls


# ── Combo Building ──────────────────────────────────────────────────────────

def build_phase1_combos(quick: bool = False) -> list[dict]:
    """Build all phase 1 parameter combinations."""
    if quick:
        dte_combos, tickers, option_types = QUICK_DTE_COMBOS, QUICK_TICKERS, QUICK_OPTION_TYPES
        otm_pcts, widths = QUICK_OTM_PCTS, QUICK_WIDTHS
        contracts, max_trades_list, diversity_list = QUICK_NUM_CONTRACTS, QUICK_MAX_TRADES, QUICK_DIVERSITY
    else:
        dte_combos, tickers, option_types = FULL_DTE_COMBOS, FULL_TICKERS, FULL_OPTION_TYPES
        otm_pcts, widths = FULL_OTM_PCTS, FULL_WIDTHS
        contracts, max_trades_list, diversity_list = FULL_NUM_CONTRACTS, FULL_MAX_TRADES, FULL_DIVERSITY

    combos = []
    for dte, tkr, otype, otm, w, nc, mt, div in itertools.product(
        dte_combos, tickers, option_types, otm_pcts, widths,
        contracts, max_trades_list, diversity_list,
    ):
        max_loss_per_trade = w * 100 * nc
        if max_loss_per_trade > 50_000:
            continue
        max_loss_per_day = min(max_loss_per_trade * mt, 500_000)
        combos.append({
            "dte": dte, "tickers": tkr, "option_types": otype,
            "min_otm_pct": otm, "spread_width": w, "num_contracts": nc,
            "max_trades_per_day": mt, "diversity_enabled": div,
            "entry_start_et": "09:30", "entry_end_et": "15:00",
            "profit_target_pct": 0.50, "stop_loss_mult": 2.0,
            "min_credit": 0.25,
            "max_loss_per_trade": min(max_loss_per_trade, 50_000),
            "max_loss_per_day": max_loss_per_day,
        })
    return combos


def build_phase2_combos(top_results: list[dict], n: int = 10) -> list[dict]:
    """Build phase 2 fine-tune combos from top-N phase 1 results."""
    ranked = sorted(top_results, key=lambda r: r.get("val_score", 0), reverse=True)
    combos = []
    for base in ranked[:n]:
        base_combo = base["combo"]
        for es, ee, pt, sl in itertools.product(
            FINE_ENTRY_STARTS, FINE_ENTRY_ENDS, FINE_PROFIT_TARGETS, FINE_STOP_LOSSES,
        ):
            if es >= ee:
                continue
            combo = dict(base_combo)
            combo["entry_start_et"] = es
            combo["entry_end_et"] = ee
            combo["profit_target_pct"] = pt
            combo["stop_loss_mult"] = sl
            combos.append(combo)
    return combos


def make_label(combo: dict) -> str:
    """Short label for a combo."""
    dte_s = "+".join(str(d) for d in combo["dte"])
    tkr_s = "+".join(combo["tickers"])
    otype_s = "+".join(combo["option_types"])
    nc = combo.get("num_contracts", 10)
    mt = combo.get("max_trades_per_day", 5)
    div = "div" if combo.get("diversity_enabled", True) else "nodiv"
    es = combo.get("entry_start_et", "09:30")
    ee = combo.get("entry_end_et", "15:00")
    pt = combo.get("profit_target_pct", 0.50)
    sl = combo.get("stop_loss_mult", 2.0)
    return (
        f"DTE{dte_s}_{tkr_s}_{otype_s}_OTM{combo['min_otm_pct']}_W{combo['spread_width']}"
        f"_C{nc}_T{mt}_{div}_E{es}-{ee}_PT{pt}_SL{sl}"
    )


# ── Single Combo Runner ────────────────────────────────────────────────────

def run_combo(voice_url: str, combo: dict, start_date: str, end_date: str) -> dict:
    """Run a single parameter combo through the engine and return results.

    Passes config inline in the run-range body to avoid global config races
    when multiple workers share a voice server.
    """
    config = {
        "tickers": combo["tickers"], "option_types": combo["option_types"],
        "min_otm_pct": combo["min_otm_pct"], "spread_width": combo["spread_width"],
        "dte": combo["dte"], "num_contracts": combo.get("num_contracts", 10),
        "max_trades_per_day": combo.get("max_trades_per_day", 5),
        "min_credit": combo.get("min_credit", 0.25),
        "max_loss_per_trade": combo.get("max_loss_per_trade", 15000),
        "max_loss_per_day": combo.get("max_loss_per_day", 75000),
        "profit_target_pct": combo.get("profit_target_pct", 0.50),
        "stop_loss_mult": combo.get("stop_loss_mult", 2.0),
        "entry_start_et": combo.get("entry_start_et", "09:30"),
        "entry_end_et": combo.get("entry_end_et", "15:00"),
        "diversity_enabled": combo.get("diversity_enabled", True),
    }

    # Pass config inline in run-range body (thread-safe, no global race)
    resp = requests.post(
        f"{voice_url}/api/auto-trader/run-range",
        json={"start_date": start_date, "end_date": end_date, "config": config},
        timeout=7200,
    )
    resp.raise_for_status()
    result = resp.json()

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
                    "date": day.get("date", ""), "ticker": t.get("ticker", ""),
                    "option_type": t.get("option_type", ""),
                    "short_strike": t.get("short_strike", 0),
                    "long_strike": t.get("long_strike", 0),
                    "credit": t.get("credit", 0),
                    "exit_reason": t.get("exit_reason", ""),
                    "realized_pnl": rpnl, "dte": t.get("dte", 0),
                    "expiration": t.get("expiration", ""),
                })
            else:
                current_streak = 0

    return {
        "label": make_label(combo), "combo": combo,
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
        "num_contracts": combo.get("num_contracts", 10),
        "diversity_enabled": combo.get("diversity_enabled", True),
    }


# ── Parallel Phase Runner ──────────────────────────────────────────────────

def run_phase_parallel(
    voice_urls: list[str],
    combos: list[dict],
    start: str, end: str,
    completed: dict[str, dict],
    results: list[dict],
    results_path: Path,
    phase_name: str,
) -> list[dict]:
    """Run combos in parallel across multiple voice workers.

    Each voice URL gets exclusive access via a per-worker queue. Combos are
    distributed round-robin so each daemon processes one combo at a time
    (preventing sim clock races).
    """
    pending = [(i, c) for i, c in enumerate(combos) if make_label(c) not in completed]
    if not pending:
        print(f"  All {len(combos)} combos already completed.")
        return results

    n_workers = len(voice_urls)
    print(f"  {len(pending)} combos to run across {n_workers} workers")

    # Partition pending combos into per-worker queues (round-robin)
    worker_queues: list[list] = [[] for _ in range(n_workers)]
    for seq, (idx, combo) in enumerate(pending):
        worker_queues[seq % n_workers].append((idx, combo))

    lock = threading.Lock()
    start_time = time.time()
    done_count = [0]
    save_counter = [0]

    def _run_worker(worker_id: int):
        """Each worker processes its queue sequentially (one combo at a time)."""
        voice_url = voice_urls[worker_id]
        for idx, combo in worker_queues[worker_id]:
            label = make_label(combo)

            try:
                result = run_combo(voice_url, combo, start, end)
            except Exception as e:
                result = {
                    "label": label, "combo": combo, "val_score": 0,
                    "total_pnl": 0, "error": str(e),
                    "num_contracts": combo.get("num_contracts", 10),
                    "diversity_enabled": combo.get("diversity_enabled", True),
                }

            with lock:
                results.append(result)
                completed[label] = result
                done_count[0] += 1
                save_counter[0] += 1
                elapsed = time.time() - start_time
                rate = done_count[0] / max(elapsed, 1)
                remaining = (len(pending) - done_count[0]) / rate if rate > 0 else 0
                vs = result.get("val_score", 0)
                pnl = result.get("total_pnl", 0)
                wr = result.get("win_rate", 0)
                tr = result.get("total_trades", 0)
                err = result.get("error", "")
                if err:
                    status = f"FAILED: {err[:40]}"
                else:
                    status = f"val={vs:.4f} P&L=${pnl:,.0f} WR={wr:.0%} trades={tr}"
                print(
                    f"[{phase_name} {done_count[0]}/{len(pending)}] W{worker_id} {label[:70]}  {status}  "
                    f"(~{remaining/60:.0f}m left)",
                    flush=True,
                )

                if save_counter[0] >= 10:
                    save_counter[0] = 0
                    try:
                        with open(results_path, "w") as f:
                            json.dump({"results": results, "start": start, "end": end}, f)
                    except Exception:
                        pass

    with ThreadPoolExecutor(max_workers=n_workers, thread_name_prefix="sweep") as pool:
        futures = [pool.submit(_run_worker, i) for i in range(n_workers)]
        try:
            for future in as_completed(futures):
                future.result()
        except KeyboardInterrupt:
            print("\n  Interrupted! Saving progress...")
            pool.shutdown(wait=False, cancel_futures=True)

    with open(results_path, "w") as f:
        json.dump({"results": results, "start": start, "end": end}, f)

    return results


# ── Display ─────────────────────────────────────────────────────────────────

def print_top_n(results: list[dict], n: int = 10) -> None:
    """Print top-N ranked results table."""
    valid = [r for r in results if r.get("val_score", 0) > 0]
    ranked = sorted(valid, key=lambda r: r["val_score"], reverse=True)
    print(f"\n{'='*120}")
    print(f"  TOP {n} RESULTS (ranked by val_score) — {len(valid)} valid of {len(results)} total")
    print(f"{'='*120}")
    hdr = (
        f"  {'#':>3}  {'Label':<70} {'val':>8} {'P&L':>12} "
        f"{'WR':>5} {'Sharpe':>7} {'Trades':>6} {'Div':>4}"
    )
    print(hdr)
    print(f"  {'-'*3}  {'-'*70} {'-'*8} {'-'*12} {'-'*5} {'-'*7} {'-'*6} {'-'*4}")

    for i, r in enumerate(ranked[:n], 1):
        div_str = "Y" if r.get("diversity_enabled", r.get("combo", {}).get("diversity_enabled", True)) else "N"
        label = r["label"][:70]
        print(
            f"  {i:>3}  {label:<70} "
            f"{r['val_score']:>8.4f} "
            f"${r['total_pnl']:>10,.0f} "
            f"{r['win_rate']:>4.0%} "
            f"{r['sharpe']:>7.2f} "
            f"{r['total_trades']:>6} "
            f"{div_str:>4}"
        )

    # Diversity comparison
    div_best = [r for r in ranked if r.get("diversity_enabled", r.get("combo", {}).get("diversity_enabled", True))]
    nodiv_best = [r for r in ranked if not r.get("diversity_enabled", r.get("combo", {}).get("diversity_enabled", True))]
    if div_best and nodiv_best:
        print(f"\n  DIVERSITY COMPARISON:")
        print(f"    Best WITH:    val={div_best[0]['val_score']:.4f}  P&L=${div_best[0]['total_pnl']:,.0f}  WR={div_best[0]['win_rate']:.0%}")
        print(f"    Best WITHOUT: val={nodiv_best[0]['val_score']:.4f}  P&L=${nodiv_best[0]['total_pnl']:,.0f}  WR={nodiv_best[0]['win_rate']:.0%}")

    # Failure analysis
    print(f"\n{'='*120}")
    print(f"  FAILURE ANALYSIS — Top {min(n, 5)} Configs")
    print(f"{'='*120}")
    for i, r in enumerate(ranked[:min(n, 5)], 1):
        lt = r.get("losing_trades", [])
        wd = r.get("worst_day", {})
        print(f"\n  #{i} {r['label']}")
        print(f"      Losing trades: {len(lt)}, Max consecutive: {r.get('max_consecutive_losses', 0)}")
        if wd.get("date"):
            print(f"      Worst day: {wd['date']} -> ${wd['pnl']:,.2f}")
        for loss in lt[:5]:
            print(
                f"      {loss['date']} {loss['ticker']:>4} {loss['option_type']:>4} "
                f"{loss['short_strike']}/{loss['long_strike']} "
                f"cr={loss['credit']:.2f} {loss['exit_reason']:<18} ${loss['realized_pnl']:+,.2f}"
            )
        if len(lt) > 5:
            print(f"      ... and {len(lt) - 5} more")


def save_results(results, results_path, summary_path, start, end, phase, quick, total_combos):
    """Save results to JSON and CSV."""
    with open(results_path, "w") as f:
        json.dump(
            {"results": results, "start": start, "end": end,
             "phase": phase, "quick": quick, "total_combos": total_combos},
            f, indent=2,
        )
    csv_fields = [
        "label", "val_score", "total_pnl", "win_rate", "sharpe",
        "max_drawdown", "profit_factor", "total_trades", "total_wins",
        "total_losses", "peak_risk", "max_consecutive_losses",
        "num_contracts", "diversity_enabled",
    ]
    ranked = sorted(results, key=lambda r: r.get("val_score", 0), reverse=True)
    with open(summary_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=csv_fields, extrasaction="ignore")
        writer.writeheader()
        for r in ranked:
            writer.writerow(r)


# ── Main ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Comprehensive auto-trader parameter sweep with parallelism.",
        epilog="""
Examples:
  %(prog)s --quick --workers 8 --start 2026-02-18 --end 2026-04-17
      Parallel quick sweep (~432 combos, ~1 hour with 8 workers)

  %(prog)s --quick --workers 8 --fine-tune --start 2026-02-18 --end 2026-04-17
      Phase 1 + phase 2 fine-tuning

  %(prog)s --quick --workers 8 --start 2026-02-18 --end 2026-04-17 \\
      --resume results/full_sweep/sweep_20260419.json
      Resume an interrupted run

  %(prog)s --quick --voice-urls http://localhost:8801,http://localhost:8802
      Use pre-started voice servers (no auto-spawn)
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--start", required=True, help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", required=True, help="End date (YYYY-MM-DD)")
    parser.add_argument("--quick", action="store_true", help="Reduced sweep (~432 combos)")
    parser.add_argument("--workers", type=int, default=1,
                        help="Number of parallel workers (each gets daemon+voice pair, default: 1)")
    parser.add_argument("--voice-urls",
                        help="Comma-separated voice URLs (skip auto-spawn, e.g. http://localhost:8801,http://localhost:8802)")
    parser.add_argument("--base-daemon-port", type=int, default=8100,
                        help="Starting port for auto-spawned daemons (default: 8100)")
    parser.add_argument("--base-voice-port", type=int, default=8801,
                        help="Starting port for auto-spawned voice servers (default: 8801)")
    parser.add_argument("--options-dir", default="../../options_csv_output_full",
                        help="Options CSV directory (default: ../../options_csv_output_full)")
    parser.add_argument("--equities-dir", default="../../equities_output",
                        help="Equities CSV directory (default: ../../equities_output)")
    parser.add_argument("--tickers", default="SPX,RUT,NDX",
                        help="Tickers for sim daemons (default: SPX,RUT,NDX)")
    parser.add_argument("--fine-tune", action="store_true",
                        help="Run phase 2 fine-tuning on top-N phase 1 results")
    parser.add_argument("--fine-tune-top", type=int, default=10,
                        help="Number of top configs to fine-tune (default: 10)")
    parser.add_argument("--output-dir", default="results/full_sweep",
                        help="Output directory (default: results/full_sweep)")
    parser.add_argument("--top-n", type=int, default=20,
                        help="Number of top results to display (default: 20)")
    parser.add_argument("--resume", help="Resume from a previous sweep JSON file")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    results_path = output_dir / f"sweep_{timestamp}.json"
    summary_path = output_dir / f"summary_{timestamp}.csv"

    # ── Resolve voice URLs ──
    if args.voice_urls:
        voice_urls = [u.strip().rstrip("/") for u in args.voice_urls.split(",")]
        n_workers = len(voice_urls)
        print(f"Using {n_workers} pre-started voice servers: {voice_urls}")
    elif args.workers > 1:
        n_workers = args.workers
        print(f"Spawning {n_workers} worker pairs (daemon+voice)...")
        voice_urls = spawn_workers(
            n_workers, args.base_daemon_port, args.base_voice_port,
            args.options_dir, args.equities_dir, args.tickers,
        )
        ready = sum(1 for u in voice_urls if _wait_for_port(f"{u}/api/auto-trader/config", timeout=5))
        print(f"  {ready}/{n_workers} workers ready")
        if ready == 0:
            print("ERROR: No workers started. Exiting.", file=sys.stderr)
            sys.exit(1)
        # Filter to only ready URLs
        voice_urls = [u for u in voice_urls if _wait_for_port(f"{u}/api/auto-trader/config", timeout=3)]
        n_workers = len(voice_urls)
    else:
        # Single worker, use default or first voice URL
        voice_urls = [f"http://localhost:{args.base_voice_port}"]
        n_workers = 1
        try:
            resp = requests.get(f"{voice_urls[0]}/api/auto-trader/config", timeout=5)
            resp.raise_for_status()
        except Exception as e:
            print(f"Error: Cannot connect to voice at {voice_urls[0]}: {e}", file=sys.stderr)
            sys.exit(1)

    # Build combos
    p1_combos = build_phase1_combos(quick=args.quick)
    print(f"\nPhase 1: {len(p1_combos)} combos | Workers: {n_workers} | "
          f"~{len(p1_combos) * 110 / n_workers / 60:.0f}m ETA")
    print(f"Date range: {args.start} -> {args.end}")
    print(f"Risk bounds: max $50K/trade, max $500K/day")
    print(f"Output: {results_path}")

    # Resume
    completed: dict[str, dict] = {}
    if args.resume:
        resume_path = Path(args.resume)
        if resume_path.exists():
            with open(resume_path) as f:
                prev = json.load(f)
            for r in prev.get("results", []):
                completed[r["label"]] = r
            print(f"Resuming: {len(completed)} combos already completed")

    results: list[dict] = list(completed.values())
    total_start = time.time()

    # ── Phase 1 ──
    print(f"\n{'='*80}")
    print(f"  PHASE 1: Core Parameter Sweep ({len(p1_combos)} combos, {n_workers} workers)")
    print(f"{'='*80}\n")

    results = run_phase_parallel(
        voice_urls, p1_combos, args.start, args.end,
        completed, results, results_path, "P1",
    )
    save_results(results, results_path, summary_path,
                 args.start, args.end, "phase1", args.quick, len(p1_combos))
    print(f"\nPhase 1 complete: {len(results)} combos")
    print_top_n(results, args.top_n)

    # ── Phase 2 (optional) ──
    if args.fine_tune:
        p2_combos = build_phase2_combos(results, n=args.fine_tune_top)
        print(f"\n{'='*80}")
        print(f"  PHASE 2: Fine-Tune Top {args.fine_tune_top} ({len(p2_combos)} combos, {n_workers} workers)")
        print(f"{'='*80}\n")
        results = run_phase_parallel(
            voice_urls, p2_combos, args.start, args.end,
            completed, results, results_path, "P2",
        )
        save_results(results, results_path, summary_path,
                     args.start, args.end, "phase1+2", args.quick,
                     len(p1_combos) + len(p2_combos))
        print(f"\nPhase 2 complete: {len(results)} total combos")
        print_top_n(results, args.top_n)

    elapsed = time.time() - total_start
    print(f"\nTotal: {len(results)} combos in {elapsed/60:.1f} min ({elapsed/3600:.1f} hr)")
    print(f"Results: {results_path}")
    print(f"Summary: {summary_path}")

    # Cleanup spawned workers
    if _spawned_procs:
        print("Stopping spawned workers...")
        _cleanup_workers()


if __name__ == "__main__":
    main()
