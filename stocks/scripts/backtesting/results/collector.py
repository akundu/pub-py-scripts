"""ResultCollector -- aggregates trade results across the backtest."""

from collections import defaultdict
from datetime import date
from typing import Any, Dict, List, Optional

from .metrics import StandardMetrics


class ResultCollector:
    """Collects individual trade results and produces summary statistics."""

    def __init__(self):
        self._results: List[Dict[str, Any]] = []

    def add(self, result: Dict[str, Any]) -> None:
        self._results.append(result)

    def add_batch(self, results: List[Dict[str, Any]]) -> None:
        self._results.extend(results)

    @property
    def results(self) -> List[Dict[str, Any]]:
        return list(self._results)

    @property
    def count(self) -> int:
        return len(self._results)

    def summarize(self) -> Dict[str, Any]:
        """Generate summary with metrics."""
        metrics = StandardMetrics.compute(self._results)
        daily = self._daily_breakdown()

        return {
            "metrics": metrics,
            "total_trades": len(self._results),
            "daily_breakdown": daily,
            "results": self._results,
        }

    def _daily_breakdown(self) -> Dict[str, Any]:
        """Group results by date and compute per-day stats."""
        by_date: Dict[date, List[Dict]] = defaultdict(list)
        for r in self._results:
            trading_date = r.get("trading_date")
            if trading_date:
                by_date[trading_date].append(r)

        daily = {}
        for d, day_results in sorted(by_date.items()):
            daily[str(d)] = StandardMetrics.compute(day_results)

        return daily

    def clear(self) -> None:
        self._results.clear()
