"""POST /trade/execute and POST /trade/close — trade execution endpoints."""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, BackgroundTasks, Header, HTTPException, Security
from pydantic import BaseModel, Field

from app.auth import TokenData, require_auth
from app.config import settings
from app.models import OrderResult, OrderStatus, TradeRequest
from app.services.trade_service import (
    await_order_fill,
    execute_trade,
    poll_order_status,
)

router = APIRouter(prefix="/trade", tags=["trading"])


class ClosePositionRequest(BaseModel):
    position_id: str
    quantity: int | None = None
    net_price: float = Field(default=0.05, description="Debit price for closing")


@router.post("/execute", response_model=OrderResult)
async def trade_execute(
    request: TradeRequest,
    background_tasks: BackgroundTasks,
    _user: Annotated[TokenData, Security(require_auth, scopes=["trades:write"])],
    x_dry_run: Annotated[str | None, Header()] = None,
    x_async: Annotated[str | None, Header()] = None,
) -> OrderResult:
    """Execute an equity or multi-leg options trade.

    Headers:
        X-Dry-Run: true — simulate without sending to broker (paper trade).
        X-Async: true — return immediately after submission; fill status
            is broadcast via WebSocket at /ws/orders. Without this header,
            the endpoint blocks until the order reaches a terminal state
            (FILLED, CANCELLED, REJECTED, FAILED) or the poll timeout
            expires (default 30s, configurable via ORDER_POLL_TIMEOUT_SECONDS).
    """
    dry_run = (x_dry_run or "").lower() == "true"
    async_mode = (x_async or "").lower() == "true"

    result = await execute_trade(request, dry_run=dry_run)

    if dry_run:
        return result

    broker = (
        request.equity_order.broker
        if request.equity_order
        else request.multi_leg_order.broker  # type: ignore[union-attr]
    )

    if async_mode:
        # Fire-and-forget: poll in background, return immediately
        background_tasks.add_task(poll_order_status, broker, result.order_id)
        return result

    # Synchronous mode (default): wait for fill or timeout
    if result.status not in {OrderStatus.FILLED, OrderStatus.CANCELLED,
                              OrderStatus.REJECTED, OrderStatus.FAILED}:
        result = await await_order_fill(
            broker=broker,
            order_id=result.order_id,
            poll_interval=settings.order_poll_interval_seconds,
            timeout=settings.order_poll_timeout_seconds,
        )

    return result


@router.post("/close")
async def close_position(
    request: ClosePositionRequest,
    _user: Annotated[TokenData, Security(require_auth, scopes=["trades:write"])],
) -> dict:
    """Close an open position by ID."""
    from app.services.position_store import get_position_store

    store = get_position_store()
    if not store:
        raise HTTPException(status_code=503, detail="Position store not initialized")

    pos = store.get_position(request.position_id)
    if not pos:
        # Try prefix match
        matches = [pid for pid in store._positions if pid.startswith(request.position_id)]
        if len(matches) == 1:
            pos = store.get_position(matches[0])
            request.position_id = matches[0]
        elif len(matches) > 1:
            raise HTTPException(status_code=400, detail=f"Ambiguous prefix: {len(matches)} matches")
        else:
            raise HTTPException(status_code=404, detail="Position not found")

    if pos.get("status") != "open":
        raise HTTPException(status_code=400, detail="Position is not open")

    if request.quantity:
        result = store.reduce_quantity(request.position_id, request.quantity)
    else:
        result = store.close_position(request.position_id, request.net_price, "api_close")

    return {"status": "ok", "position": result}


@router.post("/advisor/confirm")
async def confirm_advisor_trade(
    _user: Annotated[TokenData, Security(require_auth, scopes=["trades:write"])],
    priority: int = 0,
) -> dict:
    """Confirm and execute an advisor recommendation by priority number."""
    try:
        from utp import _daemon_state
    except ImportError:
        raise HTTPException(status_code=503, detail="Advisor not available")

    entries = _daemon_state.get("advisor_entries", [])
    match = None
    for entry in entries:
        if entry.get("priority") == priority:
            match = entry
            break

    if not match:
        raise HTTPException(status_code=404, detail=f"No recommendation with priority {priority}")

    return {
        "status": "confirmed",
        "recommendation": match,
        "message": f"Recommendation {priority} ({match.get('tier_label', '?')}) confirmed for execution",
    }
