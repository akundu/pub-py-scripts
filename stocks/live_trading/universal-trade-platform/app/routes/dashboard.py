"""Dashboard and performance endpoints."""

from __future__ import annotations

from datetime import date
from typing import Annotated, Optional

from fastapi import APIRouter, HTTPException, Security
from fastapi.responses import PlainTextResponse

from app.auth import TokenData, require_auth
from app.models import DailyPnL, DashboardSummary, PerformanceMetrics, StatusReport
from app.services.dashboard_service import DashboardService
from app.services.live_data_service import get_live_data_service
from app.services.position_store import get_position_store
from app.services.terminal_display import TerminalRenderer

router = APIRouter(prefix="/dashboard", tags=["dashboard"])


def _get_dashboard_service() -> DashboardService:
    store = get_position_store()
    if not store:
        raise HTTPException(status_code=503, detail="Position store not initialized")
    return DashboardService(store)


@router.get("/summary", response_model=DashboardSummary)
async def dashboard_summary(
    _user: Annotated[TokenData, Security(require_auth, scopes=["account:read"])],
) -> DashboardSummary:
    """Aggregated dashboard summary of positions and P&L."""
    svc = get_live_data_service()
    if svc:
        return await svc.get_summary()
    return _get_dashboard_service().get_summary()


@router.get("/portfolio")
async def portfolio_view(
    _user: Annotated[TokenData, Security(require_auth, scopes=["account:read"])],
    recent_count: int = 5,
    include_quotes: bool = False,
) -> dict:
    """Full portfolio view with broker-authoritative cost basis and marks.

    Returns positions enriched with AvgCost, Mark, and MktVal from broker
    when available. This is what the CLI 'portfolio' command displays.

    Query params:
        include_quotes: If true, fetches current quotes for each underlying
            and adds current_price + breach_status to each position.
            Adds ~1-3s latency. Default false for CLI speed.
    """
    svc = get_live_data_service()
    if svc:
        return await svc.get_portfolio(
            recent_count=recent_count, include_quotes=include_quotes,
        )

    # Fallback: local-only view
    summary = _get_dashboard_service().get_summary()
    result = {
        "positions": [pos.model_dump() for pos in summary.active_positions],
        "balances": {},
        "realized_pnl": summary.realized_pnl,
        "unrealized_pnl": summary.unrealized_pnl,
        "total_pnl": summary.total_pnl,
        "positions_by_source": summary.positions_by_source,
    }
    store = get_position_store()
    if store:
        closed = store.get_closed_positions()
        recent = sorted(closed, key=lambda p: p.get("exit_time", ""), reverse=True)[:5]
        result["recent_closed"] = recent
    return result


@router.get("/performance", response_model=PerformanceMetrics)
async def performance(
    _user: Annotated[TokenData, Security(require_auth, scopes=["account:read"])],
    start_date: Optional[date] = None,
    end_date: Optional[date] = None,
) -> PerformanceMetrics:
    """Compute performance metrics for closed positions."""
    return _get_dashboard_service().get_performance(start_date, end_date)


@router.get("/pnl/daily", response_model=list[DailyPnL])
async def daily_pnl(
    _user: Annotated[TokenData, Security(require_auth, scopes=["account:read"])],
    days: int = 30,
) -> list[DailyPnL]:
    """Daily P&L breakdown."""
    return _get_dashboard_service().get_daily_pnl(days)


@router.get("/status", response_model=StatusReport)
async def status_dashboard(
    _user: Annotated[TokenData, Security(require_auth, scopes=["account:read"])],
) -> StatusReport:
    """Full status: active positions, in-transit orders, recent closed, cache stats."""
    svc = get_live_data_service()
    if svc:
        return await svc.get_status()
    return _get_dashboard_service().get_status()


@router.get("/terminal", response_class=PlainTextResponse)
async def terminal_view(
    _user: Annotated[TokenData, Security(require_auth, scopes=["account:read"])],
) -> str:
    """Render dashboard as ANSI terminal text."""
    svc = _get_dashboard_service()
    summary = svc.get_summary()
    metrics = svc.get_performance()
    return TerminalRenderer.render(summary, metrics)


@router.get("/advisor/recommendations")
async def advisor_recommendations(
    _user: Annotated[TokenData, Security(require_auth, scopes=["account:read"])],
) -> dict:
    """Get current advisor entry/exit recommendations."""
    try:
        from utp import _daemon_state
    except ImportError:
        return {"entries": [], "exits": [], "profile": None, "last_eval": None}

    return {
        "entries": _daemon_state.get("advisor_entries", []),
        "exits": _daemon_state.get("advisor_exits", []),
        "profile": _daemon_state.get("advisor_profile"),
        "last_eval": _daemon_state.get("advisor_last_eval"),
    }


@router.get("/advisor/status")
async def advisor_status(
    _user: Annotated[TokenData, Security(require_auth, scopes=["account:read"])],
) -> dict:
    """Get advisor status."""
    try:
        from utp import _daemon_state
    except ImportError:
        return {"active": False}

    return {
        "active": _daemon_state.get("advisor_profile") is not None,
        "profile": _daemon_state.get("advisor_profile"),
        "last_eval": _daemon_state.get("advisor_last_eval"),
        "pending_entries": len(_daemon_state.get("advisor_entries", [])),
        "pending_exits": len(_daemon_state.get("advisor_exits", [])),
    }
