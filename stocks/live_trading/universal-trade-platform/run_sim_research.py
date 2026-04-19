#!/usr/bin/env python3
"""Evaluation harness for sim_trader autoresearch. DO NOT MODIFY.

This script is the immutable evaluation target. The autoresearch agent
modifies constants in sim_trader.py, then runs this script to get
val_score. Higher is better.

Usage:
    python run_sim_research.py
    python run_sim_research.py --voice-url http://localhost:8801
    python run_sim_research.py --start 2026-01-02 --end 2026-04-17
"""

from __future__ import annotations

import argparse
import sys

import requests

VOICE_URL = "http://localhost:8801"
EVAL_START = "2026-01-02"
EVAL_END = "2026-04-17"


def main():
    parser = argparse.ArgumentParser(description="Autoresearch evaluation harness")
    parser.add_argument("--voice-url", default=VOICE_URL)
    parser.add_argument("--start", default=EVAL_START)
    parser.add_argument("--end", default=EVAL_END)
    args = parser.parse_args()

    voice_url = args.voice_url.rstrip("/")

    # Import and push config from sim_trader
    import sim_trader
    config = sim_trader.build_config()

    try:
        resp = requests.post(f"{voice_url}/api/auto-trader/config", json=config, timeout=10)
        resp.raise_for_status()
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    print(f"Running evaluation: {args.start} → {args.end}")
    print(f"Config: tickers={config['tickers']}, width={config['spread_width']}, "
          f"otm={config['min_otm_pct']}, contracts={config['num_contracts']}")

    resp = requests.post(
        f"{voice_url}/api/auto-trader/run-range",
        json={"start_date": args.start, "end_date": args.end},
        timeout=7200,
    )
    resp.raise_for_status()
    result = resp.json()

    print(f"\nval_score: {result.get('val_score', 0):.6f}")
    print(f"total_pnl: {result.get('total_pnl', 0):.2f}")
    print(f"peak_risk: {result.get('peak_risk', 0):.0f}")
    print(f"win_rate: {result.get('win_rate', 0):.1%}")
    print(f"sharpe: {result.get('sharpe', 0):.2f}")
    print(f"profit_factor: {result.get('profit_factor', 0):.2f}")
    print(f"days_traded: {result.get('days_traded', 0)}")
    print(f"total_trades: {result.get('total_trades', 0)}")


if __name__ == "__main__":
    main()
