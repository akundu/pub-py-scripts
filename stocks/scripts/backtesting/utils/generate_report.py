"""Generate formatted reports from backtest JSON results."""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from scripts.backtesting.results.reporters import get_reporter


def main():
    parser = argparse.ArgumentParser(
        description="""
Generate formatted reports from backtest JSON result files.
        """,
        epilog="""
Examples:
  %(prog)s results/backtest_results.json --format console
      Print results to console

  %(prog)s results/backtest_results.json --format csv --output-dir reports/
      Export results to CSV
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("file", help="JSON backtest results file")
    parser.add_argument("--format", default="console",
                        choices=["console", "csv", "json"],
                        help="Output format (default: console)")
    parser.add_argument("--output-dir", default="results/reports",
                        help="Output directory for file-based reports")

    args = parser.parse_args()

    with open(args.file) as f:
        summary = json.load(f)

    # Create a mock config for output_dir
    class MockConfig:
        class infra:
            output_dir = args.output_dir
            ticker = summary.get("metrics", {}).get("ticker", "UNKNOWN")
            start_date = None
            end_date = None
        class strategy:
            name = "loaded_from_file"

    reporter = get_reporter(args.format)
    reporter.generate(summary, MockConfig)


if __name__ == "__main__":
    main()
