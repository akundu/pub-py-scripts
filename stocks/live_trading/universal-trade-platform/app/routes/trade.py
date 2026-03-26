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


async def _close_by_con_id(position: dict, quantity: int | None, net_price: float | None) -> OrderResult | None:
    """Close a multi-leg position using stored conIds — bypasses contract qualification.

    This is more reliable than qualifying from scratch since we already know the exact
    contracts from when the position was synced.
    """
    from app.core.provider import ProviderRegistry
    from app.models import Broker, OrderType

    try:
        provider = ProviderRegistry.get(Broker.IBKR)
    except Exception:
        return None

    # Check if this is a CPG REST provider — delegate to execute_multi_leg_order instead
    if hasattr(provider, "_gateway_url"):
        return await _close_by_con_id_rest(provider, position, quantity, net_price)

    ib = getattr(provider, "_ib", None)
    if not ib or not getattr(provider, "_connected", False):
        return None

    from ib_insync import ComboLeg, Contract, LimitOrder, MarketOrder

    legs = position.get("legs") or []
    symbol = position.get("symbol", "")
    close_qty = quantity or int(abs(position.get("quantity", 1)))
    exchange = getattr(provider, "_exchange", "SMART")

    combo_legs = []
    first_close_action = None
    for leg in legs:
        con_id = leg.get("con_id")
        if not con_id:
            return None  # Fall back to qualification path

        action_str = leg.get("action", "")
        # Reverse: SELL→BUY (close), BUY→SELL (close)
        close_action = "BUY" if "SELL" in action_str else "SELL"
        if first_close_action is None:
            first_close_action = close_action

        combo_legs.append(ComboLeg(
            conId=con_id,
            ratio=1,  # always 1 per leg — close_qty on the order controls spread count
            action=close_action,
            exchange=exchange,
        ))

    combo = Contract(
        symbol=symbol,
        secType="BAG",
        exchange=exchange,
        currency="USD",
        comboLegs=combo_legs,
    )

    # IBKR combo pricing convention (order action always "BUY"):
    # - First combo leg BUY + positive price = debit (you pay) → closing a credit spread
    # - First combo leg SELL + negative price = credit (you receive) → closing a debit spread
    #
    # Determine from the original position: if there's a SELL leg (short),
    # closing means BUY-ing it back (debit). Sort BUY first for credit spread close.
    has_short_leg = any(leg.get("action", "") == "SELL" for leg in legs)
    if has_short_leg:
        combo_legs.sort(key=lambda cl: (0 if cl.action == "BUY" else 1))
    else:
        combo_legs.sort(key=lambda cl: (0 if cl.action == "SELL" else 1))

    import logging as _log
    _logger = _log.getLogger("utp.close")

    if net_price is not None:
        ibkr_price = abs(net_price) if has_short_leg else -abs(net_price)
        ib_order = LimitOrder("BUY", close_qty, ibkr_price)
    else:
        ibkr_price = 0.0
        ib_order = MarketOrder("BUY", close_qty)
    ib_order.tif = "DAY"

    price_label = f"LIMIT {ibkr_price:+.4f}" if net_price is not None else "MARKET"
    _logger.info(
        "Close order: %s %d legs x%d, %s (net=%s, avg_cost=%.4f), combo_legs=%s",
        symbol, len(combo_legs), close_qty, price_label,
        f"{net_price:.4f}" if net_price is not None else "MARKET",
        position.get("avg_cost", 0),
        [(cl.conId, cl.action, cl.ratio) for cl in combo_legs],
    )

    trade = ib.placeOrder(combo, ib_order)

    # Wait briefly for IBKR to process and check for immediate rejection
    import asyncio
    await asyncio.sleep(0.5)
    ibkr_status = trade.orderStatus.status if trade.orderStatus else "Unknown"
    ibkr_message = ""
    if trade.log:
        ibkr_message = "; ".join(entry.message for entry in trade.log if hasattr(entry, "message") and entry.message)

    _logger.info("Close order result: orderId=%s, status=%s, log=%s",
                 trade.order.orderId, ibkr_status, ibkr_message)

    status_map = {
        "Submitted": OrderStatus.SUBMITTED,
        "PreSubmitted": OrderStatus.PENDING,
        "Filled": OrderStatus.FILLED,
        "Cancelled": OrderStatus.CANCELLED,
        "Inactive": OrderStatus.FAILED,
    }

    return OrderResult(
        order_id=str(trade.order.orderId),
        broker=Broker.IBKR,
        status=status_map.get(ibkr_status, OrderStatus.SUBMITTED),
        message=f"Close {symbol}: {len(combo_legs)} legs x{close_qty} @ ibkr_price={ibkr_price:+.2f} "
                f"({ibkr_status}){(' — ' + ibkr_message) if ibkr_message else ''}",
        filled_price=trade.orderStatus.avgFillPrice if trade.orderStatus and trade.orderStatus.avgFillPrice else None,
    )


async def _close_by_con_id_rest(provider, position: dict, quantity: int | None, net_price: float | None) -> OrderResult | None:
    """Close a multi-leg position via CPG REST using stored conIds."""
    from app.models import Broker, MultiLegOrder, OptionLeg, OptionType, OptionAction, OrderType, OrderResult, OrderStatus
    import logging as _log
    _logger = _log.getLogger("utp.close")

    legs = position.get("legs") or []
    symbol = position.get("symbol", "")
    close_qty = quantity or int(abs(position.get("quantity", 1)))
    exp = position.get("expiration", "")

    # Build closing MultiLegOrder using the stored conIds
    close_legs = []
    for leg in legs:
        action_str = leg.get("action", "")
        close_action = "BUY_TO_CLOSE" if "SELL" in action_str else "SELL_TO_CLOSE"
        ot = leg.get("option_type", "PUT")
        close_legs.append(OptionLeg(
            symbol=symbol,
            expiration=exp,
            strike=float(leg.get("strike", 0)),
            option_type=OptionType(ot),
            action=OptionAction(close_action),
            quantity=close_qty,
        ))

    if not close_legs:
        return None

    order = MultiLegOrder(
        broker=Broker.IBKR,
        legs=close_legs,
        order_type=OrderType.LIMIT if net_price else OrderType.MARKET,
        quantity=close_qty,
        net_price=net_price,
    )

    _logger.info("CPG close order: %s %d legs x%d, net_price=%s", symbol, len(close_legs), close_qty,
                 f"${net_price:.2f}" if net_price else "MARKET")

    try:
        result = await provider.execute_multi_leg_order(order)
        return result
    except Exception as e:
        _logger.error("CPG close order failed: %s", e)
        return OrderResult(
            broker=Broker.IBKR,
            status=OrderStatus.FAILED,
            message=f"CPG close failed: {e}",
        )


class ClosePositionRequest(BaseModel):
    position_id: str
    quantity: int | None = None
    net_price: float | None = Field(default=None, description="Debit price for closing (default: current mark)")


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
    background_tasks: BackgroundTasks,
    _user: Annotated[TokenData, Security(require_auth, scopes=["trades:write"])],
    x_dry_run: Annotated[str | None, Header()] = None,
) -> dict:
    """Close an open position by ID — submits a closing order to IBKR.

    For multi-leg positions: builds a reversed combo order (SELL→BUY_TO_CLOSE, etc.)
    For equity positions: submits a market sell order.
    """
    from app.services.position_store import get_position_store
    from app.services.trade_service import build_closing_trade_request

    dry_run = (x_dry_run or "").lower() == "true"
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

    # net_price: if provided → LIMIT order at that price
    # if None → MARKET order (IBKR fills at best available)
    net_price = request.net_price

    # Build closing trade request using shared function
    trade_request = build_closing_trade_request(pos, request.quantity, net_price)

    # Determine broker from the trade request
    from app.models import Broker
    broker = (
        trade_request.equity_order.broker if trade_request.equity_order
        else trade_request.multi_leg_order.broker  # type: ignore[union-attr]
    )

    # If legs have con_ids, use direct conId-based closing (bypasses qualification)
    legs = pos.get("legs") or []
    has_con_ids = all(leg.get("con_id") for leg in legs) if legs else False
    if has_con_ids and trade_request.multi_leg_order and not dry_run:
        import logging as _log
        _close_logger = _log.getLogger("utp.close")
        try:
            result = await _close_by_con_id(pos, request.quantity, net_price)
        except Exception as e:
            _close_logger.error("_close_by_con_id failed: %s", e, exc_info=True)
            raise HTTPException(status_code=500, detail=f"Close order failed: {e}")
        if result:
            # Update local position store
            if result.status == OrderStatus.FILLED:
                exit_price = result.filled_price or net_price or 0
                if request.quantity and request.quantity < abs(pos.get("quantity", 0)):
                    updated = store.reduce_quantity(request.position_id, request.quantity)
                else:
                    updated = store.close_position(request.position_id, exit_price, "api_close")
                return {"status": "ok", "order_result": result.model_dump(), "position": updated}
            if result.status not in {OrderStatus.FILLED, OrderStatus.CANCELLED,
                                      OrderStatus.REJECTED, OrderStatus.FAILED}:
                # Wait for fill
                result = await await_order_fill(
                    broker=broker,
                    order_id=result.order_id,
                    poll_interval=settings.order_poll_interval_seconds,
                    timeout=settings.order_poll_timeout_seconds,
                )
                if result.status == OrderStatus.FILLED:
                    exit_price = result.filled_price or net_price or 0
                    if request.quantity and request.quantity < abs(pos.get("quantity", 0)):
                        updated = store.reduce_quantity(request.position_id, request.quantity)
                    else:
                        updated = store.close_position(request.position_id, exit_price, "api_close")
                    return {"status": "ok", "order_result": result.model_dump(), "position": updated}
            return {
                "status": "order_not_filled",
                "order_result": result.model_dump(),
                "position": pos,
                "message": f"Order status: {result.status.value} — {result.message}",
            }

    # Fallback: submit via execute_trade (requires contract qualification)
    result = await execute_trade(trade_request, dry_run=dry_run)

    if dry_run:
        return {"status": "ok", "order_result": result.model_dump(), "position": pos, "dry_run": True}

    # Wait for fill
    if result.status not in {OrderStatus.FILLED, OrderStatus.CANCELLED,
                              OrderStatus.REJECTED, OrderStatus.FAILED}:
        result = await await_order_fill(
            broker=broker,
            order_id=result.order_id,
            poll_interval=settings.order_poll_interval_seconds,
            timeout=settings.order_poll_timeout_seconds,
        )

    # Update local position store based on fill result
    if result.status == OrderStatus.FILLED:
        exit_price = result.filled_price or net_price or 0
        if request.quantity and request.quantity < abs(pos.get("quantity", 0)):
            # Partial close
            updated = store.reduce_quantity(request.position_id, request.quantity)
        else:
            # Full close
            updated = store.close_position(request.position_id, exit_price, "api_close")
        return {"status": "ok", "order_result": result.model_dump(), "position": updated}

    # Order didn't fill — return the order result without modifying local store
    return {
        "status": "order_not_filled",
        "order_result": result.model_dump(),
        "position": pos,
        "message": f"Order status: {result.status.value} — {result.message}",
    }


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
