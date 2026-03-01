"""GridReporter -- report for grid sweep results."""

from typing import Any, Dict, List

import pandas as pd

from .base import ReportGenerator


class GridReporter(ReportGenerator):
    """Reports results from grid sweep parameter optimization."""

    def generate(self, summary: Dict[str, Any], config: Any) -> None:
        grid_results = summary.get("grid_results", [])
        if not grid_results:
            print("No grid sweep results to report.")
            return

        print("\n" + "=" * 100)
        print("GRID SWEEP RESULTS")
        print("=" * 100)
        print(f"Total configurations tested: {len(grid_results)}")

        # Convert to DataFrame for easy display
        rows = []
        for gr in grid_results:
            row = dict(gr.get("params", {}))
            row.update(gr.get("metrics", {}))
            rows.append(row)

        df = pd.DataFrame(rows)

        if "roi" in df.columns:
            df = df.sort_values("roi", ascending=False)

        # Show top 10
        print(f"\nTop 10 configurations by ROI:")
        print(df.head(10).to_string(index=False))
        print("=" * 100)

    @staticmethod
    def format_grid_results(grid_results: List[Dict]) -> pd.DataFrame:
        """Convert grid results to a sorted DataFrame."""
        rows = []
        for gr in grid_results:
            row = dict(gr.get("params", {}))
            row.update(gr.get("metrics", {}))
            rows.append(row)

        df = pd.DataFrame(rows)
        if "roi" in df.columns:
            df = df.sort_values("roi", ascending=False)
        return df
