"""Analyze grid sweep results."""

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from scripts.backtesting.results.reporters.grid_reporter import GridReporter


def main():
    parser = argparse.ArgumentParser(
        description="""
Analyze and rank grid sweep results from a JSON results file.
        """,
        epilog="""
Examples:
  %(prog)s results/grid_sweep/backtest_results.json
      Analyze grid sweep results

  %(prog)s results/grid_sweep/backtest_results.json --sort-by roi --top 20
      Show top 20 configs sorted by ROI
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("file", help="JSON grid sweep results file")
    parser.add_argument("--sort-by", default="roi", help="Metric to sort by (default: roi)")
    parser.add_argument("--top", type=int, default=10, help="Number of top results to show")

    args = parser.parse_args()

    with open(args.file) as f:
        data = json.load(f)

    grid_results = data.get("grid_results", [])
    if not grid_results:
        print("No grid results found in file.")
        return

    df = GridReporter.format_grid_results(grid_results)
    if args.sort_by in df.columns:
        df = df.sort_values(args.sort_by, ascending=False)

    print(f"\nTop {args.top} configurations by {args.sort_by}:")
    print(df.head(args.top).to_string(index=False))


if __name__ == "__main__":
    main()
