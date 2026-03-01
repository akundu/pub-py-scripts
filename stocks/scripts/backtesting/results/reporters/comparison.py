"""ComparisonReporter -- cross-run comparison of backtest results."""

import json
import os
from typing import Any, Dict, List

from .base import ReportGenerator


class ComparisonReporter(ReportGenerator):
    """Compares results across multiple backtest runs."""

    def generate(self, summary: Dict[str, Any], config: Any) -> None:
        # This reporter is used differently -- called with multiple summaries
        print("Use ComparisonReporter.compare() for cross-run comparison.")

    @staticmethod
    def compare(summaries: List[Dict[str, Any]], labels: List[str]) -> None:
        """Compare metrics across multiple backtest runs."""
        print("\n" + "=" * 100)
        print("BACKTEST COMPARISON")
        print("=" * 100)

        header = f"{'Metric':<20}"
        for label in labels:
            header += f" {label:<18}"
        print(header)
        print("-" * (20 + 18 * len(labels)))

        metric_keys = [
            "total_trades", "win_rate", "net_pnl", "roi",
            "profit_factor", "sharpe", "max_drawdown",
        ]

        for key in metric_keys:
            row = f"{key:<20}"
            for s in summaries:
                val = s.get("metrics", {}).get(key, "N/A")
                if isinstance(val, float):
                    row += f" {val:>16,.2f}"
                else:
                    row += f" {str(val):>16}"
            print(row)

        print("=" * 100)
