"""StandardMetrics -- compute aggregate trading metrics.

Wraps the pattern from credit_spread_utils/metrics.py compute_metrics().
"""

import math
from typing import Any, Dict, List


class StandardMetrics:
    """Compute standard trading performance metrics."""

    @staticmethod
    def compute(results: List[Dict[str, Any]]) -> Dict[str, Any]:
        if not results:
            return {
                "total_trades": 0,
                "wins": 0,
                "losses": 0,
                "win_rate": 0.0,
                "total_credits": 0.0,
                "total_gains": 0.0,
                "total_losses": 0.0,
                "net_pnl": 0.0,
                "roi": 0.0,
                "profit_factor": 0.0,
                "sharpe": 0.0,
                "max_drawdown": 0.0,
                "avg_pnl": 0.0,
            }

        wins = 0
        losses = 0
        total_credits = 0.0
        total_gains = 0.0
        total_losses = 0.0
        total_risk = 0.0
        pnl_series = []

        for r in results:
            pnl = r.get("pnl", 0.0)
            credit = r.get("credit", 0.0)
            max_loss = r.get("max_loss", 0.0)

            total_credits += credit
            total_risk += max_loss
            pnl_series.append(pnl)

            if pnl >= 0:
                wins += 1
                total_gains += pnl
            else:
                losses += 1
                total_losses += abs(pnl)

        total_trades = len(results)
        testable = wins + losses
        win_rate = (wins / testable * 100) if testable > 0 else 0.0
        net_pnl = total_gains - total_losses
        roi = (net_pnl / total_risk * 100) if total_risk > 0 else 0.0

        if total_losses > 0:
            profit_factor = total_gains / total_losses
        elif total_gains > 0:
            profit_factor = float("inf")
        else:
            profit_factor = 0.0

        avg_pnl = net_pnl / total_trades if total_trades > 0 else 0.0

        # Sharpe ratio (annualized, assuming daily)
        sharpe = 0.0
        if len(pnl_series) > 1:
            mean_pnl = sum(pnl_series) / len(pnl_series)
            variance = sum((p - mean_pnl) ** 2 for p in pnl_series) / (len(pnl_series) - 1)
            std_pnl = math.sqrt(variance) if variance > 0 else 0
            if std_pnl > 0:
                sharpe = (mean_pnl / std_pnl) * math.sqrt(252)

        # Max drawdown
        max_drawdown = StandardMetrics._compute_max_drawdown(pnl_series)

        return {
            "total_trades": total_trades,
            "wins": wins,
            "losses": losses,
            "win_rate": round(win_rate, 2),
            "total_credits": round(total_credits, 2),
            "total_gains": round(total_gains, 2),
            "total_losses": round(total_losses, 2),
            "net_pnl": round(net_pnl, 2),
            "roi": round(roi, 2),
            "profit_factor": round(profit_factor, 4),
            "sharpe": round(sharpe, 4),
            "max_drawdown": round(max_drawdown, 2),
            "avg_pnl": round(avg_pnl, 2),
        }

    @staticmethod
    def _compute_max_drawdown(pnl_series: List[float]) -> float:
        if not pnl_series:
            return 0.0

        cumulative = 0.0
        peak = 0.0
        max_dd = 0.0

        for pnl in pnl_series:
            cumulative += pnl
            if cumulative > peak:
                peak = cumulative
            drawdown = peak - cumulative
            if drawdown > max_dd:
                max_dd = drawdown

        return max_dd
