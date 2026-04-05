"""Baseline comparison computations for orchestrated backtesting.

Compares orchestrated results against:
1. Standalone: Each algo's full P&L from Phase 1 (no selection)
2. Equal-weight: Accept all trades with budget-only filtering (no scoring)
3. v3 cross-ticker: Load from v3 results directory
"""

import logging
import os
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


def compute_standalone_baseline(
    per_instance_results: Dict[str, Dict[str, Any]],
) -> Dict[str, Dict[str, Any]]:
    """Each algo's standalone P&L from Phase 1 (no orchestration).

    Args:
        per_instance_results: Dict of instance_id -> {metrics, total_trades, ...}

    Returns:
        Dict of instance_id -> {metrics, total_trades, source}
    """
    baselines = {}
    for iid, data in per_instance_results.items():
        baselines[iid] = {
            "metrics": data.get("metrics", {}),
            "total_trades": data.get("total_trades", 0),
            "source": "standalone_phase1",
        }
    return baselines


def compute_equal_weight_baseline(
    all_trades: List[Dict[str, Any]],
    daily_budget: float = 200000,
) -> Dict[str, Any]:
    """Accept all trades with budget-only filtering (no scoring/selection).

    Simulates what would happen if every algo's trade were accepted,
    subject only to daily budget constraints.
    """
    from collections import defaultdict

    by_date: Dict[str, List[Dict]] = defaultdict(list)
    for trade in all_trades:
        date_key = str(trade.get("trading_date", trade.get("entry_date", "")))
        by_date[date_key].append(trade)

    accepted = []
    for date_key in sorted(by_date.keys()):
        budget_remaining = daily_budget
        day_trades = by_date[date_key]

        for trade in day_trades:
            max_loss = abs(trade.get("max_loss", 0))
            num_contracts = trade.get("num_contracts", 1)
            risk = max_loss * num_contracts * 100

            if risk <= budget_remaining:
                accepted.append(trade)
                budget_remaining -= risk

    from scripts.backtesting.results.metrics import StandardMetrics
    metrics = StandardMetrics.compute(accepted) if accepted else {}

    return {
        "metrics": metrics,
        "total_trades": len(accepted),
        "source": "equal_weight",
    }


def load_v3_baseline(
    results_dir: str = "results/tiered_portfolio_v3",
) -> Dict[str, Any]:
    """Load v3 cross-ticker baseline results for comparison."""
    import pandas as pd
    from scripts.backtesting.results.metrics import StandardMetrics

    trades_path = os.path.join(results_dir, "portfolio_trades.csv")
    if not os.path.exists(trades_path):
        return {"error": f"v3 baseline not found at {trades_path}"}

    df = pd.read_csv(trades_path)
    results = []
    for _, row in df.iterrows():
        results.append({
            "pnl": row.get("adjusted_pnl", row.get("pnl", 0)),
            "credit": abs(row.get("credit", 0)),
            "max_loss": abs(row.get("max_loss", 0)),
            "trading_date": row.get("entry_date", ""),
        })

    return {
        "metrics": StandardMetrics.compute(results),
        "total_trades": len(results),
        "source": trades_path,
    }


def build_comparison_table(
    orchestrated_metrics: Dict[str, Any],
    orchestrated_trades: int,
    baselines: Dict[str, Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Build a comparison table across all baselines.

    Returns list of dicts suitable for DataFrame or table display.
    """
    rows = [{
        "system": "ORCHESTRATED",
        "trades": orchestrated_trades,
        **orchestrated_metrics,
    }]

    for name, data in baselines.items():
        if "error" in data:
            continue
        metrics = data.get("metrics", {})
        rows.append({
            "system": name,
            "trades": data.get("total_trades", 0),
            **metrics,
        })

    return rows
