#!/usr/bin/env python3
"""
Evaluate Dynamic Risk Levels Across Timeframes.

Tests all risk levels (1-10) against historical data and generates
recommendations for each risk tier.

Usage:
    python scripts/evaluate_risk_levels.py --ticker NDX --processes 12

  # Basic usage
  python scripts/evaluate_risk_levels.py --ticker NDX --processes 12

  # With JSON output
  python scripts/evaluate_risk_levels.py --ticker NDX --processes 12 --json-output ndx_risk_levels.json

  # With custom base parameters
  python scripts/evaluate_risk_levels.py --ticker NDX --processes 12 \
    --base-pb 0.0075 \
    --base-pc 0.015 \
    --base-msw-put 20 \
    --base-msw-call 20

  # Full options
  python scripts/evaluate_risk_levels.py --help


"""

import argparse
import asyncio
import json
import os
import sys
import tempfile
from datetime import date, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import subprocess
import glob as glob_module

import pandas as pd

# Add scripts dir to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from strategy_utils.dynamic_risk import (
    BaseParams,
    DynamicRiskAdjuster,
    get_params_for_all_risk_levels,
    format_risk_level_summary,
    AdjustedParams,
)
from common.logging_utils import get_logger


# Window definitions for this analysis
ANALYSIS_WINDOWS = {
    '3mo': 90,
    '2mo': 60,
    '1mo': 30,
    '2wk': 14,
    '1wk': 7,
}


def get_csv_files_for_window(
    ticker: str,
    days: int,
    csv_dir: str,
) -> List[str]:
    """Get CSV files for a time window."""
    reference_date = date.today()
    start_date = reference_date - timedelta(days=days)

    base_dir = os.path.join(csv_dir, ticker)
    pattern = os.path.join(base_dir, f"{ticker}_options_*.csv")
    all_files = sorted(glob_module.glob(pattern))

    csv_files = []
    for filepath in all_files:
        filename = os.path.basename(filepath)
        try:
            date_str = filename.replace(f"{ticker}_options_", "").replace(".csv", "")
            file_date = date.fromisoformat(date_str)
            if start_date <= file_date <= reference_date:
                csv_files.append(filepath)
        except ValueError:
            continue

    return csv_files


def run_single_analysis(
    ticker: str,
    risk_level: int,
    params: AdjustedParams,
    csv_files: List[str],
    output_file: str,
    cache_dir: str,
    processes: int = 1,
) -> Optional[Dict[str, Any]]:
    """
    Run credit spread analysis for a single risk level.
    """
    # Build config for the grid search - csv_path must be a list, not comma-separated
    config = {
        "fixed_params": {
            "csv_path": csv_files,  # Pass as list, not comma-separated string
            "underlying_ticker": ticker,
            "db_path": "$QUEST_DB_STRING",
            "risk_cap": 50000,
            "max_live_capital": 300000,
            "profit_target_pct": 80,
            "min_contract_price": 0.5,
            "output_timezone": "PDT",
            "cache_dir": cache_dir,
        },
        "grid_params": {
            "option_type": ["both"],
            "percent_beyond_put": [params.percent_beyond_put],
            "percent_beyond_call": [params.percent_beyond_call],
            "max_spread_width_put": [params.max_spread_width_put],
            "max_spread_width_call": [params.max_spread_width_call],
            "min_trading_hour": [6],
            "max_trading_hour": [13],
        },
        "sort_by": "net_pnl",
    }

    # Write config to temp file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(config, f)
        config_file = f.name

    try:
        # Build command
        cmd = [
            sys.executable,
            os.path.join(os.path.dirname(__file__), "analyze_credit_spread_intervals.py"),
            "--grid-config", config_file,
            "--grid-output", output_file,
            "--processes", str(processes),
        ]

        # Run the analysis
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=3600,
        )

        if result.returncode != 0:
            # Check for actual error vs just no trades
            if "Error" in result.stderr or "error" in result.stderr.lower():
                return None

        # Read results
        if os.path.exists(output_file):
            df = pd.read_csv(output_file)
            if len(df) > 0:
                top = df.iloc[0]
                return {
                    'risk_level': risk_level,
                    'net_pnl': float(top.get('net_pnl', 0)),
                    'profit_factor': float(top.get('profit_factor', 0)),
                    'win_rate': float(top.get('win_rate', 0)),
                    'total_trades': int(top.get('total_trades', 0)),
                    'avg_win': float(top.get('avg_win', 0)),
                    'avg_loss': float(top.get('avg_loss', 0)),
                    'max_drawdown': float(top.get('max_drawdown', 0)),
                    'params': params.to_dict(),
                }

        return None

    except Exception as e:
        print(f"Error running analysis for risk level {risk_level}: {e}")
        return None
    finally:
        if os.path.exists(config_file):
            os.unlink(config_file)


def run_analysis_for_window(
    ticker: str,
    window: str,
    days: int,
    csv_dir: str,
    cache_dir: str,
    output_dir: str,
    processes: int,
    base_params: BaseParams,
) -> Dict[int, Dict[str, Any]]:
    """
    Run analysis for all risk levels for a specific time window.
    """
    print(f"\n{'='*60}")
    print(f"Analyzing {window} window ({days} days) for {ticker}")
    print(f"{'='*60}")

    csv_files = get_csv_files_for_window(ticker, days, csv_dir)

    if not csv_files:
        print(f"  No CSV files found for {window} window")
        return {}

    print(f"  Found {len(csv_files)} CSV files")

    # Get parameters for all risk levels
    adjuster = DynamicRiskAdjuster(base_params)
    params_by_level = {
        level: adjuster.get_static_params_for_risk_level(level)
        for level in range(1, 11)
    }

    results = {}

    # Run analysis for each risk level
    for risk_level in range(1, 11):
        params = params_by_level[risk_level]
        output_file = os.path.join(
            output_dir,
            f"{ticker.lower()}_{window}_risk{risk_level}_results.csv"
        )

        print(f"  Risk Level {risk_level}: pb={params.percent_beyond_put:.4f}, "
              f"pc={params.percent_beyond_call:.4f}, "
              f"msw={params.max_spread_width_put}:{params.max_spread_width_call}",
              end="", flush=True)

        result = run_single_analysis(
            ticker=ticker,
            risk_level=risk_level,
            params=params,
            csv_files=csv_files,
            output_file=output_file,
            cache_dir=cache_dir,
            processes=processes,
        )

        if result:
            results[risk_level] = result
            pnl = result['net_pnl']
            wr = result['win_rate']
            trades = result['total_trades']
            pf = result['profit_factor']
            print(f" -> P&L: ${pnl:,.0f}, WR: {wr:.0%}, Trades: {trades}, PF: {pf:.2f}")
        else:
            print(f" -> No results")

    return results


def run_all_windows(
    ticker: str,
    windows: Dict[str, int],
    csv_dir: str,
    cache_dir: str,
    output_dir: str,
    processes: int,
    base_params: BaseParams,
) -> Dict[str, Dict[int, Dict[str, Any]]]:
    """Run analysis for all windows."""
    all_results = {}

    for window, days in windows.items():
        results = run_analysis_for_window(
            ticker=ticker,
            window=window,
            days=days,
            csv_dir=csv_dir,
            cache_dir=cache_dir,
            output_dir=output_dir,
            processes=processes,
            base_params=base_params,
        )
        all_results[window] = results

    return all_results


def generate_summary_report(
    ticker: str,
    results: Dict[str, Dict[int, Dict[str, Any]]],
    base_params: BaseParams,
) -> str:
    """Generate a comprehensive summary report."""
    lines = []

    lines.append("\n" + "=" * 80)
    lines.append(f" {ticker} DYNAMIC RISK LEVEL ANALYSIS RESULTS")
    lines.append("=" * 80)

    lines.append(f"\nBase Parameters (Risk Level 5):")
    lines.append(f"  percent_beyond_put: {base_params.percent_beyond_put}")
    lines.append(f"  percent_beyond_call: {base_params.percent_beyond_call}")
    lines.append(f"  max_spread_width_put: {base_params.max_spread_width_put}")
    lines.append(f"  max_spread_width_call: {base_params.max_spread_width_call}")

    params_by_level = get_params_for_all_risk_levels(base_params)
    lines.append("\n" + format_risk_level_summary(params_by_level))

    # Results by window
    for window in ['3mo', '2mo', '1mo', '2wk', '1wk']:
        if window not in results or not results[window]:
            continue

        window_results = results[window]
        lines.append(f"\n{'='*80}")
        lines.append(f" {window.upper()} WINDOW RESULTS")
        lines.append(f"{'='*80}")

        lines.append(f"\n{'Risk':<6} {'Net P&L':>12} {'Win Rate':>10} {'Trades':>8} {'PF':>8} {'Avg Win':>10} {'Avg Loss':>10}")
        lines.append("-" * 70)

        sorted_levels = sorted(
            window_results.keys(),
            key=lambda x: window_results[x].get('net_pnl', 0),
            reverse=True
        )

        for level in sorted_levels:
            r = window_results[level]
            pnl = r.get('net_pnl', 0)
            wr = r.get('win_rate', 0)
            trades = r.get('total_trades', 0)
            pf = r.get('profit_factor', 0)
            avg_win = r.get('avg_win', 0)
            avg_loss = r.get('avg_loss', 0)

            lines.append(
                f"{level:<6} ${pnl:>11,.0f} {wr:>9.1%} {trades:>8} {pf:>8.2f} "
                f"${avg_win:>9,.0f} ${avg_loss:>9,.0f}"
            )

        if sorted_levels:
            best = sorted_levels[0]
            lines.append(f"\n  Best performing: Risk Level {best} "
                        f"(${window_results[best]['net_pnl']:,.0f} P&L)")

    # Cross-window analysis
    lines.append(f"\n{'='*80}")
    lines.append(f" CROSS-WINDOW ANALYSIS")
    lines.append(f"{'='*80}")

    avg_by_level = {}
    for level in range(1, 11):
        pnls = []
        wrs = []
        pfs = []
        for window, window_results in results.items():
            if level in window_results:
                pnls.append(window_results[level].get('net_pnl', 0))
                wrs.append(window_results[level].get('win_rate', 0))
                pfs.append(window_results[level].get('profit_factor', 0))
        if pnls:
            avg_by_level[level] = {
                'avg_pnl': sum(pnls) / len(pnls),
                'avg_wr': sum(wrs) / len(wrs),
                'avg_pf': sum(pfs) / len(pfs),
                'consistency': len([p for p in pnls if p > 0]) / len(pnls),
                'total_pnl': sum(pnls),
            }

    lines.append(f"\n{'Risk':<6} {'Avg P&L':>12} {'Total P&L':>14} {'Avg WR':>10} {'Avg PF':>8} {'Consistency':>12}")
    lines.append("-" * 70)

    for level in range(1, 11):
        if level in avg_by_level:
            a = avg_by_level[level]
            lines.append(
                f"{level:<6} ${a['avg_pnl']:>11,.0f} ${a['total_pnl']:>13,.0f} "
                f"{a['avg_wr']:>9.1%} {a['avg_pf']:>8.2f} {a['consistency']:>11.0%}"
            )

    return "\n".join(lines)


def generate_recommendations(
    ticker: str,
    results: Dict[str, Dict[int, Dict[str, Any]]],
    base_params: BaseParams,
) -> str:
    """Generate trading recommendations for tomorrow."""
    lines = []

    lines.append("\n" + "=" * 80)
    lines.append(f" {ticker} RECOMMENDATIONS FOR TOMORROW")
    lines.append("=" * 80)

    params_by_level = get_params_for_all_risk_levels(base_params)

    tiers = {
        'Conservative (1-3)': [1, 2, 3],
        'Neutral (4-6)': [4, 5, 6],
        'Aggressive (7-10)': [7, 8, 9, 10],
    }

    recent_windows = ['1wk', '2wk', '1mo']

    for tier_name, levels in tiers.items():
        lines.append(f"\n{'-'*60}")
        lines.append(f" {tier_name}")
        lines.append(f"{'-'*60}")

        best_level = None
        best_score = float('-inf')

        for level in levels:
            score = 0
            weight = 3
            for window in recent_windows:
                if window in results and level in results[window]:
                    pnl = results[window][level].get('net_pnl', 0)
                    wr = results[window][level].get('win_rate', 0)
                    pf = results[window][level].get('profit_factor', 0)
                    score += weight * (pnl / 100000 + wr * 50 + pf * 10)
                weight -= 1

            if score > best_score:
                best_score = score
                best_level = level

        if best_level is None:
            best_level = levels[len(levels) // 2]

        p = params_by_level[best_level]

        lines.append(f"\n  Recommended Risk Level: {best_level}")
        lines.append(f"\n  Parameters:")
        lines.append(f"    percent_beyond_put:  {p.percent_beyond_put:.4f}")
        lines.append(f"    percent_beyond_call: {p.percent_beyond_call:.4f}")
        lines.append(f"    max_spread_width_put:  {p.max_spread_width_put}")
        lines.append(f"    max_spread_width_call: {p.max_spread_width_call}")

        lines.append(f"\n  Recent Performance:")
        for window in recent_windows:
            if window in results and best_level in results[window]:
                r = results[window][best_level]
                lines.append(
                    f"    {window:>4}: P&L ${r['net_pnl']:>10,.0f}, "
                    f"WR {r['win_rate']:.0%}, PF {r['profit_factor']:.2f}, "
                    f"{r['total_trades']} trades"
                )

        lines.append(f"\n  Trading Instructions:")
        lines.append(f"    Entry Window: 6:30 AM - 10:00 AM PDT")
        lines.append(f"    Exit: At 80% profit target or by 1:00 PM PDT")
        lines.append(f"    Max Concurrent Capital: $300,000")
        lines.append(f"    Risk per Trade: $50,000")

    # Overall recommendation
    lines.append(f"\n{'='*80}")
    lines.append(f" OVERALL RECOMMENDATION")
    lines.append(f"{'='*80}")

    best_overall = None
    best_overall_score = float('-inf')

    for level in range(1, 11):
        score = 0
        for window, window_results in results.items():
            if level in window_results:
                pnl = window_results[level].get('net_pnl', 0)
                wr = window_results[level].get('win_rate', 0)
                pf = window_results[level].get('profit_factor', 0)
                consistency = 1 if pnl > 0 else 0
                score += pnl / 100000 + wr * 50 + pf * 10 + consistency * 20

        if score > best_overall_score:
            best_overall_score = score
            best_overall = level

    if best_overall:
        p = params_by_level[best_overall]
        lines.append(f"\n  Best Overall Risk Level: {best_overall}")
        lines.append(f"\n  Use these parameters for balanced risk/reward:")
        lines.append(f"    percent_beyond_put:  {p.percent_beyond_put:.4f}")
        lines.append(f"    percent_beyond_call: {p.percent_beyond_call:.4f}")
        lines.append(f"    max_spread_width_put:  {p.max_spread_width_put}")
        lines.append(f"    max_spread_width_call: {p.max_spread_width_call}")

    return "\n".join(lines)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Evaluate dynamic risk levels across timeframes"
    )

    parser.add_argument("--ticker", required=True, help="Ticker symbol")
    parser.add_argument("--csv-dir", default="./options_csv_output", help="CSV directory")
    parser.add_argument("--cache-dir", default="./.options_cache", help="Cache directory")
    parser.add_argument("--output-dir", default="./risk_level_results", help="Output directory")
    parser.add_argument("--processes", type=int, default=12, help="Parallel processes")
    parser.add_argument("--base-pb", type=float, default=0.0075, help="Base percent_beyond_put")
    parser.add_argument("--base-pc", type=float, default=0.015, help="Base percent_beyond_call")
    parser.add_argument("--base-msw-put", type=int, default=20, help="Base max_spread_width_put")
    parser.add_argument("--base-msw-call", type=int, default=20, help="Base max_spread_width_call")
    parser.add_argument("--json-output", default=None, help="JSON output path")

    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()

    ticker = args.ticker.upper()
    os.makedirs(args.output_dir, exist_ok=True)

    base_params = BaseParams(
        percent_beyond_put=args.base_pb,
        percent_beyond_call=args.base_pc,
        max_spread_width_put=args.base_msw_put,
        max_spread_width_call=args.base_msw_call,
    )

    print(f"\n{'='*60}")
    print(f" Dynamic Risk Level Evaluation - {ticker}")
    print(f"{'='*60}")
    print(f"\nBase Parameters:")
    print(f"  percent_beyond_put: {base_params.percent_beyond_put}")
    print(f"  percent_beyond_call: {base_params.percent_beyond_call}")
    print(f"  max_spread_width_put: {base_params.max_spread_width_put}")
    print(f"  max_spread_width_call: {base_params.max_spread_width_call}")
    print(f"\nProcesses: {args.processes}")

    params_by_level = get_params_for_all_risk_levels(base_params)
    print("\n" + format_risk_level_summary(params_by_level))

    results = run_all_windows(
        ticker=ticker,
        windows=ANALYSIS_WINDOWS,
        csv_dir=args.csv_dir,
        cache_dir=args.cache_dir,
        output_dir=args.output_dir,
        processes=args.processes,
        base_params=base_params,
    )

    summary = generate_summary_report(ticker, results, base_params)
    print(summary)

    recommendations = generate_recommendations(ticker, results, base_params)
    print(recommendations)

    if args.json_output:
        json_data = {
            'ticker': ticker,
            'base_params': {
                'percent_beyond_put': base_params.percent_beyond_put,
                'percent_beyond_call': base_params.percent_beyond_call,
                'max_spread_width_put': base_params.max_spread_width_put,
                'max_spread_width_call': base_params.max_spread_width_call,
            },
            'params_by_level': {
                str(level): p.to_dict()
                for level, p in params_by_level.items()
            },
            'results': {
                window: {
                    str(level): data
                    for level, data in window_results.items()
                }
                for window, window_results in results.items()
            },
        }
        with open(args.json_output, 'w') as f:
            json.dump(json_data, f, indent=2, default=str)
        print(f"\nJSON results saved to: {args.json_output}")

    report_file = os.path.join(args.output_dir, f"{ticker.lower()}_risk_level_report.txt")
    with open(report_file, 'w') as f:
        f.write(summary)
        f.write("\n\n")
        f.write(recommendations)
    print(f"\nReport saved to: {report_file}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
