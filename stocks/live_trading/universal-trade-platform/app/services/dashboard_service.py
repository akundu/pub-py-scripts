"""Dashboard aggregation service for positions and performance."""

from __future__ import annotations

from collections import defaultdict
from datetime import UTC, date, datetime
from typing import Optional

from app.models import (
    DailyPnL,
    DashboardSummary,
    PerformanceMetrics,
    PositionSource,
    ReconciliationEntry,
    StatusReport,
    TrackedPosition,
)
from app.services.metrics import compute_metrics
from app.services.position_store import PlatformPositionStore


class DashboardService:
    """Aggregates position store data into dashboard views."""

    def __init__(self, position_store: PlatformPositionStore) -> None:
        self._store = position_store

    def get_summary(self) -> DashboardSummary:
        """Build a dashboard summary of current state."""
        open_positions = self._store.get_open_positions()
        closed_positions = self._store.get_closed_positions()
        account = self._store.get_account_summary()

        # Build TrackedPosition objects for response
        active = [TrackedPosition.model_validate(p) for p in open_positions]

        # Count positions by source
        by_source: dict[str, int] = defaultdict(int)
        for p in open_positions:
            src = p.get("source", PositionSource.LIVE_API.value)
            by_source[src] += 1

        return DashboardSummary(
            active_positions=active,
            cash_deployed=account["cash_deployed"],
            cash_available=0,  # would come from broker account balance
            total_pnl=account["total_pnl"],
            unrealized_pnl=account["unrealized_pnl"],
            realized_pnl=account["realized_pnl"],
            positions_by_source=dict(by_source),
        )

    def get_performance(
        self,
        start_date: Optional[date] = None,
        end_date: Optional[date] = None,
    ) -> PerformanceMetrics:
        """Compute performance metrics for closed positions."""
        results = self._store.export_results()

        # Filter by date if requested
        if start_date or end_date:
            filtered = []
            for r in results:
                exit_str = r.get("exit_time", "")
                if exit_str:
                    try:
                        exit_d = datetime.fromisoformat(str(exit_str)).date()
                        if start_date and exit_d < start_date:
                            continue
                        if end_date and exit_d > end_date:
                            continue
                    except (ValueError, TypeError):
                        continue
                filtered.append(r)
            results = filtered

        m = compute_metrics(results)
        return PerformanceMetrics(**m)

    def get_daily_pnl(self, days: int = 30) -> list[DailyPnL]:
        """Group closed positions by exit date for daily P&L."""
        closed = self._store.get_closed_positions()
        daily: dict[date, dict] = defaultdict(
            lambda: {"realized_pnl": 0, "trades_closed": 0, "trades_opened": 0}
        )

        for pos in closed:
            exit_str = pos.get("exit_time")
            if exit_str:
                try:
                    d = datetime.fromisoformat(str(exit_str)).date()
                    daily[d]["realized_pnl"] += pos.get("pnl", 0)
                    daily[d]["trades_closed"] += 1
                except (ValueError, TypeError):
                    continue

            entry_str = pos.get("entry_time")
            if entry_str:
                try:
                    d = datetime.fromisoformat(str(entry_str)).date()
                    daily[d]["trades_opened"] += 1
                except (ValueError, TypeError):
                    continue

        # Sort by date, limit to last N days
        sorted_dates = sorted(daily.keys(), reverse=True)[:days]
        return [
            DailyPnL(
                date=d,
                realized_pnl=round(daily[d]["realized_pnl"], 2),
                total_pnl=round(daily[d]["realized_pnl"], 2),
                trades_opened=daily[d]["trades_opened"],
                trades_closed=daily[d]["trades_closed"],
            )
            for d in sorted(sorted_dates)
        ]

    def get_status(self) -> StatusReport:
        """Build a comprehensive status report of the trading system.

        Aggregates active positions, pending orders, recent closed trades,
        and connection status into a single view.
        """
        from app.core.provider import ProviderRegistry
        from app.services.trade_service import get_pending_orders

        # Active positions
        open_positions = self._store.get_open_positions()
        active = [TrackedPosition.model_validate(p) for p in open_positions]

        # Pending orders (in-transit)
        pending = get_pending_orders()
        in_transit = []
        for order_id, request in pending.items():
            if request.equity_order:
                in_transit.append({
                    "order_id": order_id,
                    "type": "equity",
                    "symbol": request.equity_order.symbol,
                    "side": request.equity_order.side.value,
                    "quantity": request.equity_order.quantity,
                    "broker": request.equity_order.broker.value,
                })
            elif request.multi_leg_order:
                order = request.multi_leg_order
                in_transit.append({
                    "order_id": order_id,
                    "type": "multi_leg",
                    "symbol": order.legs[0].symbol if order.legs else "UNKNOWN",
                    "legs": len(order.legs),
                    "quantity": order.quantity,
                    "broker": order.broker.value,
                })

        # Recent closed positions (last 10)
        closed = self._store.get_closed_positions()
        # Sort by exit_time descending
        closed_sorted = sorted(
            closed,
            key=lambda p: p.get("exit_time", ""),
            reverse=True,
        )
        recent_closed = closed_sorted[:10]

        # Connection status
        connection_status = {}
        try:
            providers = ProviderRegistry.all()
            for p in providers:
                connected = getattr(p, "_connected", None)
                connection_status[p.broker.value] = {
                    "connected": connected if connected is not None else True,
                }
        except Exception:
            pass

        # Cache stats (from IBKR provider if available)
        cache_stats = {}
        try:
            from app.models import Broker
            ibkr = ProviderRegistry.get(Broker.IBKR)
            if hasattr(ibkr, "cache_stats"):
                cache_stats = ibkr.cache_stats
        except Exception:
            pass

        return StatusReport(
            active_positions=active,
            in_transit_orders=in_transit,
            recent_closed=recent_closed,
            cache_stats=cache_stats,
            connection_status=connection_status,
        )

    def get_cumulative_pnl(self) -> list[dict]:
        """Running total of P&L over time."""
        closed = self._store.get_closed_positions()
        entries = []
        for pos in closed:
            exit_str = pos.get("exit_time")
            if exit_str:
                try:
                    dt = datetime.fromisoformat(str(exit_str))
                    entries.append((dt, pos.get("pnl", 0)))
                except (ValueError, TypeError):
                    continue

        entries.sort(key=lambda x: x[0])
        cumulative = 0.0
        result = []
        for dt, pnl in entries:
            cumulative += pnl
            result.append({
                "timestamp": dt.isoformat(),
                "cumulative_pnl": round(cumulative, 2),
                "trade_pnl": round(pnl, 2),
            })
        return result
