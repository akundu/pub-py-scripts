"""CSV reporter -- writes results to CSV files."""

import csv
import os
from typing import Any, Dict

from .base import ReportGenerator


class CSVReporter(ReportGenerator):
    """Writes trade results and summary metrics to CSV files."""

    def generate(self, summary: Dict[str, Any], config: Any) -> None:
        output_dir = "results/backtest"
        if hasattr(config, "infra"):
            output_dir = config.infra.output_dir

        os.makedirs(output_dir, exist_ok=True)

        # Write trades
        results = summary.get("results", [])
        if results:
            trades_path = os.path.join(output_dir, "trades.csv")
            fieldnames = sorted(results[0].keys()) if results else []
            with open(trades_path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(results)
            print(f"Trades written to {trades_path}")

        # Write summary metrics
        metrics = summary.get("metrics", {})
        if metrics:
            metrics_path = os.path.join(output_dir, "metrics.csv")
            with open(metrics_path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=sorted(metrics.keys()))
                writer.writeheader()
                writer.writerow(metrics)
            print(f"Metrics written to {metrics_path}")
