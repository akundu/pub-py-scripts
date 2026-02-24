#!/usr/bin/env python3
"""
Compare baseline vs improved multi-day backtest results.

Analyzes performance across multiple timeframes (30/60/90/180 days)
and provides deployment recommendation.

Usage:
    python scripts/compare_backtest_results.py \\
        --baseline results/baseline_30d/ results/baseline_60d/ ... \\
        --improved results/improved_30d/ results/improved_60d/ ... \\
        --output comparison_report.md
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple
import pandas as pd
import numpy as np


def load_backtest_results(result_dir: Path) -> Dict:
    """Load backtest results from directory.

    Args:
        result_dir: Directory containing backtest results

    Returns:
        Dict with hit_rate, rmse, band_width, and other metrics
    """
    # Try to load summary JSON
    summary_file = result_dir / "summary.json"
    if summary_file.exists():
        with open(summary_file, 'r') as f:
            return json.load(f)

    # Try to load CSV results
    csv_file = result_dir / "results.csv"
    if not csv_file.exists():
        # Try alternative names
        for fname in result_dir.glob("*.csv"):
            if "result" in fname.name.lower():
                csv_file = fname
                break

    if csv_file.exists():
        df = pd.read_csv(csv_file)

        # Compute metrics from CSV
        metrics = {}

        # Hit rate (P99 bands)
        if 'hit_p99' in df.columns:
            metrics['hit_rate'] = df['hit_p99'].mean() * 100
        elif 'inside_p99' in df.columns:
            metrics['hit_rate'] = df['inside_p99'].mean() * 100

        # RMSE (if error column exists)
        if 'error_pct' in df.columns:
            metrics['rmse'] = np.sqrt((df['error_pct'] ** 2).mean())
        elif 'abs_error' in df.columns:
            metrics['rmse'] = np.sqrt((df['abs_error'] ** 2).mean())

        # Band width (P99)
        if 'p99_width_pct' in df.columns:
            metrics['band_width'] = df['p99_width_pct'].mean()
        elif 'width_pct' in df.columns:
            metrics['band_width'] = df['width_pct'].mean()

        # Number of samples
        metrics['n_samples'] = len(df)

        return metrics

    raise ValueError(f"Could not find results in {result_dir}")


def compare_metrics(baseline: Dict, improved: Dict) -> Tuple[str, str]:
    """Compare two sets of metrics and determine status.

    Returns:
        (change_str, status_emoji)
    """
    # Hit rate comparison
    if 'hit_rate' in baseline and 'hit_rate' in improved:
        hit_rate_change = improved['hit_rate'] - baseline['hit_rate']
        if improved['hit_rate'] >= 99.0:
            hit_status = "✅"
        else:
            hit_status = "❌"
    else:
        hit_rate_change = 0.0
        hit_status = "⚠️"

    # RMSE comparison
    if 'rmse' in baseline and 'rmse' in improved:
        rmse_change = ((improved['rmse'] - baseline['rmse']) / baseline['rmse']) * 100
        if rmse_change < 0:  # Lower is better
            rmse_status = "✅"
        elif rmse_change < 2.0:  # Tolerance
            rmse_status = "⚠️"
        else:
            rmse_status = "❌"
    else:
        rmse_change = 0.0
        rmse_status = "⚠️"

    # Band width comparison
    if 'band_width' in baseline and 'band_width' in improved:
        width_change = ((improved['band_width'] - baseline['band_width']) / baseline['band_width']) * 100
        if width_change < 0:  # Tighter is better
            width_status = "✅"
        else:
            width_status = "⚠️"
    else:
        width_change = 0.0
        width_status = "⚠️"

    # Overall status
    if hit_status == "✅" and (rmse_status == "✅" or width_status == "✅"):
        overall_status = "✅ Better"
    elif hit_status == "❌":
        overall_status = "❌ Worse (hit rate too low)"
    elif rmse_status == "❌":
        overall_status = "❌ Worse (RMSE degraded)"
    else:
        overall_status = "⚠️ Mixed"

    return f"{hit_rate_change:+.2f}% / {rmse_change:+.1f}% / {width_change:+.1f}%", overall_status


def generate_comparison_report(baseline_dirs: List[Path], improved_dirs: List[Path], output_file: Path):
    """Generate comparison report.

    Args:
        baseline_dirs: List of baseline result directories
        improved_dirs: List of improved result directories
        output_file: Output markdown file path
    """
    # Load all results
    baseline_results = {}
    improved_results = {}

    for bdir in baseline_dirs:
        timeframe = bdir.name.replace('baseline_', '').replace('_', ' ')
        baseline_results[timeframe] = load_backtest_results(bdir)

    for idir in improved_dirs:
        timeframe = idir.name.replace('improved_', '').replace('_', ' ')
        improved_results[timeframe] = load_backtest_results(idir)

    # Generate report
    report_lines = [
        "# Multi-Day Prediction: Baseline vs Improved Comparison",
        "",
        f"**Generated:** {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "## Summary Table",
        "",
        "| Timeframe | Metric | Baseline | Improved | Change | Status |",
        "|-----------|--------|----------|----------|--------|--------|",
    ]

    all_pass = True

    for timeframe in sorted(baseline_results.keys()):
        if timeframe not in improved_results:
            continue

        baseline = baseline_results[timeframe]
        improved = improved_results[timeframe]

        # Hit Rate
        if 'hit_rate' in baseline and 'hit_rate' in improved:
            change = improved['hit_rate'] - baseline['hit_rate']
            status = "✅" if improved['hit_rate'] >= 99.0 else "❌"
            report_lines.append(
                f"| {timeframe} | Hit Rate | {baseline['hit_rate']:.2f}% | {improved['hit_rate']:.2f}% | {change:+.2f}% | {status} |"
            )
            if status == "❌":
                all_pass = False

        # RMSE
        if 'rmse' in baseline and 'rmse' in improved:
            change_pct = ((improved['rmse'] - baseline['rmse']) / baseline['rmse']) * 100
            status = "✅" if change_pct < 0 else ("⚠️" if change_pct < 2.0 else "❌")
            report_lines.append(
                f"| {timeframe} | RMSE | {baseline['rmse']:.2f}% | {improved['rmse']:.2f}% | {change_pct:+.1f}% | {status} |"
            )
            if status == "❌":
                all_pass = False

        # Band Width
        if 'band_width' in baseline and 'band_width' in improved:
            change_pct = ((improved['band_width'] - baseline['band_width']) / baseline['band_width']) * 100
            status = "✅" if change_pct < 0 else "⚠️"
            report_lines.append(
                f"| {timeframe} | Band Width | {baseline['band_width']:.2f}% | {improved['band_width']:.2f}% | {change_pct:+.1f}% | {status} |"
            )

    report_lines.extend([
        "",
        "## Deployment Recommendation",
        "",
    ])

    if all_pass:
        report_lines.extend([
            "### ✅ DEPLOY RECOMMENDED",
            "",
            "**Criteria Met:**",
            "- All timeframes have hit rate ≥99%",
            "- RMSE improved or stable (<2% degradation) across all timeframes",
            "",
            "**Action:** Enable time-aware multi-day predictions by default",
        ])
    else:
        report_lines.extend([
            "### ❌ ROLLBACK RECOMMENDED",
            "",
            "**Issues Found:**",
            "- One or more timeframes have hit rate <99%",
            "- OR RMSE degraded by >2% on one or more timeframes",
            "",
            "**Action:** Keep old behavior as default, investigate issues",
        ])

    report_lines.extend([
        "",
        "## Detailed Metrics",
        "",
    ])

    for timeframe in sorted(baseline_results.keys()):
        if timeframe not in improved_results:
            continue

        report_lines.extend([
            f"### {timeframe.title()}",
            "",
            "**Baseline:**",
            f"```json",
            json.dumps(baseline_results[timeframe], indent=2),
            "```",
            "",
            "**Improved:**",
            f"```json",
            json.dumps(improved_results[timeframe], indent=2),
            "```",
            "",
        ])

    # Write report
    with open(output_file, 'w') as f:
        f.write('\n'.join(report_lines))

    print(f"✓ Comparison report written to: {output_file}")

    if all_pass:
        print("✅ DEPLOY RECOMMENDED")
        sys.exit(0)
    else:
        print("❌ ROLLBACK RECOMMENDED")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description="Compare baseline vs improved multi-day backtest results",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument('--baseline', nargs='+', required=True,
                        help='Baseline result directories')
    parser.add_argument('--improved', nargs='+', required=True,
                        help='Improved result directories')
    parser.add_argument('--output', type=str, default='comparison_report.md',
                        help='Output markdown file (default: comparison_report.md)')

    args = parser.parse_args()

    # Convert to Path objects
    baseline_dirs = [Path(d) for d in args.baseline]
    improved_dirs = [Path(d) for d in args.improved]
    output_file = Path(args.output)

    # Validate directories exist
    for d in baseline_dirs + improved_dirs:
        if not d.exists():
            print(f"❌ Directory not found: {d}")
            sys.exit(1)

    # Generate report
    generate_comparison_report(baseline_dirs, improved_dirs, output_file)


if __name__ == '__main__':
    main()
