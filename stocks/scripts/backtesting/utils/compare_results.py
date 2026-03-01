"""Compare results across multiple backtest runs."""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from scripts.backtesting.results.reporters.comparison import ComparisonReporter


def main():
    parser = argparse.ArgumentParser(
        description="""
Compare results from multiple backtest runs side-by-side.
        """,
        epilog="""
Examples:
  %(prog)s results/run1/backtest_results.json results/run2/backtest_results.json
      Compare two backtest runs

  %(prog)s results/*/backtest_results.json
      Compare all runs in results/
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("files", nargs="+", help="JSON result files to compare")

    args = parser.parse_args()

    summaries = []
    labels = []
    for f in args.files:
        with open(f) as fh:
            summaries.append(json.load(fh))
        labels.append(Path(f).parent.name)

    ComparisonReporter.compare(summaries, labels)


if __name__ == "__main__":
    main()
