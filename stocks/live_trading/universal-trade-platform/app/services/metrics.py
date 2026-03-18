"""Performance metrics computation for closed positions."""

from __future__ import annotations

import math


def compute_metrics(results: list[dict]) -> dict:
    """Compute performance metrics from a list of closed position dicts.

    Each dict should have at minimum: pnl, credit (entry price), max_loss.
    """
    if not results:
        return {
            "total_trades": 0,
            "wins": 0,
            "losses": 0,
            "win_rate": 0.0,
            "net_pnl": 0.0,
            "roi": 0.0,
            "profit_factor": 0.0,
            "sharpe": 0.0,
            "max_drawdown": 0.0,
            "avg_pnl": 0.0,
        }

    pnls = [r.get("pnl", 0) for r in results]
    total_trades = len(pnls)
    wins = sum(1 for p in pnls if p > 0)
    losses = sum(1 for p in pnls if p <= 0)
    net_pnl = sum(pnls)
    win_rate = wins / total_trades if total_trades else 0.0
    avg_pnl = net_pnl / total_trades if total_trades else 0.0

    # ROI: net P&L / total capital deployed
    total_deployed = sum(abs(r.get("max_loss", 0)) for r in results)
    roi = (net_pnl / total_deployed * 100) if total_deployed else 0.0

    # Profit factor: gross profit / gross loss
    gross_profit = sum(p for p in pnls if p > 0)
    gross_loss = abs(sum(p for p in pnls if p < 0))
    profit_factor = (gross_profit / gross_loss) if gross_loss else float("inf") if gross_profit else 0.0

    # Sharpe ratio (simplified: mean/std of P&L series)
    if len(pnls) > 1:
        mean_pnl = sum(pnls) / len(pnls)
        variance = sum((p - mean_pnl) ** 2 for p in pnls) / (len(pnls) - 1)
        std_pnl = math.sqrt(variance)
        sharpe = (mean_pnl / std_pnl) if std_pnl else 0.0
    else:
        sharpe = 0.0

    # Max drawdown (peak-to-trough of cumulative P&L)
    cumulative = 0.0
    peak = 0.0
    max_dd = 0.0
    for p in pnls:
        cumulative += p
        if cumulative > peak:
            peak = cumulative
        dd = peak - cumulative
        if dd > max_dd:
            max_dd = dd

    return {
        "total_trades": total_trades,
        "wins": wins,
        "losses": losses,
        "win_rate": round(win_rate, 4),
        "net_pnl": round(net_pnl, 2),
        "roi": round(roi, 2),
        "profit_factor": round(profit_factor, 2) if profit_factor != float("inf") else 999.99,
        "sharpe": round(sharpe, 4),
        "max_drawdown": round(max_dd, 2),
        "avg_pnl": round(avg_pnl, 2),
    }
