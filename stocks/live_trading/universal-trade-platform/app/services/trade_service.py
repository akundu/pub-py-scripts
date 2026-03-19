"""Trade execution service with order fill tracking."""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Awaitable, Callable
from typing import Optional

from app.core.provider import ProviderRegistry
from app.models import (
    Broker,
    EquityOrder,
    MultiLegOrder,
    OrderResult,
    OrderStatus,
    PositionSource,
    TradeRequest,
)
from app.websocket import ws_manager

logger = logging.getLogger(__name__)

# Terminal states — no further status changes expected
TERMINAL_STATUSES = frozenset({
    OrderStatus.FILLED,
    OrderStatus.CANCELLED,
    OrderStatus.REJECTED,
    OrderStatus.FAILED,
})

# Track pending orders for position creation on fill
_pending_orders: dict[str, TradeRequest] = {}


def get_pending_orders() -> dict[str, TradeRequest]:
    """Return a copy of pending orders awaiting fill confirmation."""
    return dict(_pending_orders)


def _handle_close_position(request: TradeRequest, exit_price: float) -> str:
    """Close or partially close a position based on request fields.

    Returns the position_id of the affected position.
    """
    from app.services.position_store import get_position_store
    store = get_position_store()
    if not store:
        return request.closing_position_id or ""
    pos_id = request.closing_position_id
    close_qty = request.closing_quantity
    try:
        pos = store._positions.get(pos_id, {})
        total_qty = int(pos.get("quantity", 0))
        if close_qty and close_qty < total_qty:
            # Partial close: reduce quantity
            store.reduce_quantity(pos_id, close_qty)
        else:
            # Full close
            store.close_position(pos_id, exit_price=exit_price, reason="closed_via_cli")
    except KeyError:
        pass
    return pos_id


async def execute_trade(request: TradeRequest, dry_run: bool = False) -> OrderResult:
    """Execute a trade through the appropriate broker provider."""
    from app.services.ledger import get_ledger
    from app.services.position_store import get_position_store

    if request.equity_order:
        broker = request.equity_order.broker
        if dry_run:
            result = OrderResult(
                broker=broker,
                status=OrderStatus.PENDING,
                message=f"DRY RUN: Would {request.equity_order.side.value} "
                        f"{request.equity_order.quantity} {request.equity_order.symbol}",
                dry_run=True,
            )
            # Paper trade: track in position store
            store = get_position_store()
            ledger = get_ledger()
            if store:
                if request.closing_position_id:
                    pos_id = _handle_close_position(request, exit_price=0)
                else:
                    pos_id = store.add_position(request, result, is_paper=True)
                if ledger:
                    await ledger.log_order_submitted(
                        broker=broker, order_id=result.order_id,
                        source=PositionSource.PAPER, dry_run=True,
                        data={"symbol": request.equity_order.symbol},
                    )
                    if request.closing_position_id:
                        await ledger.log_position_closed(
                            broker=broker, position_id=pos_id,
                            source=PositionSource.PAPER,
                            data={"symbol": request.equity_order.symbol, "order_id": result.order_id},
                        )
                    else:
                        await ledger.log_position_opened(
                            broker=broker, position_id=pos_id,
                            source=PositionSource.PAPER, dry_run=True,
                            data={"symbol": request.equity_order.symbol, "order_id": result.order_id},
                        )
            return result
        provider = ProviderRegistry.get(broker)
        result = await provider.execute_equity_order(request.equity_order)

        # Log live order submission
        ledger = get_ledger()
        if ledger:
            await ledger.log_order_submitted(
                broker=broker, order_id=result.order_id,
                source=PositionSource.LIVE_API,
                data={"symbol": request.equity_order.symbol},
            )
        _pending_orders[result.order_id] = request
        return result

    order = request.multi_leg_order
    assert order is not None
    broker = order.broker
    if dry_run:
        legs_desc = ", ".join(
            f"{l.action.value} {l.quantity} {l.symbol} {l.strike}{l.option_type.value[0]} {l.expiration}"
            for l in order.legs
        )
        result = OrderResult(
            broker=broker,
            status=OrderStatus.PENDING,
            message=f"DRY RUN: Multi-leg [{legs_desc}]",
            dry_run=True,
        )
        # Paper trade: track in position store
        store = get_position_store()
        ledger = get_ledger()
        if store:
            if request.closing_position_id:
                pos_id = _handle_close_position(request, exit_price=order.net_price or 0)
            else:
                pos_id = store.add_position(request, result, is_paper=True)
            if ledger:
                await ledger.log_order_submitted(
                    broker=broker, order_id=result.order_id,
                    source=PositionSource.PAPER, dry_run=True,
                    data={"symbol": order.legs[0].symbol, "legs": len(order.legs)},
                )
                if request.closing_position_id:
                    await ledger.log_position_closed(
                        broker=broker, position_id=pos_id,
                        source=PositionSource.PAPER,
                        data={"symbol": order.legs[0].symbol, "order_id": result.order_id},
                    )
                else:
                    await ledger.log_position_opened(
                        broker=broker, position_id=pos_id,
                        source=PositionSource.PAPER, dry_run=True,
                        data={"symbol": order.legs[0].symbol, "order_id": result.order_id},
                    )
        return result
    provider = ProviderRegistry.get(broker)
    result = await provider.execute_multi_leg_order(order)

    # Log live order submission
    ledger = get_ledger()
    if ledger:
        await ledger.log_order_submitted(
            broker=broker, order_id=result.order_id,
            source=PositionSource.LIVE_API,
            data={"symbol": order.legs[0].symbol, "legs": len(order.legs)},
        )
    _pending_orders[result.order_id] = request
    return result


async def await_order_fill(
    broker: Broker,
    order_id: str,
    poll_interval: float = 1.0,
    timeout: float = 30.0,
    on_status_update: Optional[Callable[[OrderResult, float], Awaitable[None]]] = None,
) -> OrderResult:
    """Poll order status until terminal state or timeout.

    Args:
        broker: The broker that received the order.
        order_id: The order ID to track.
        poll_interval: Seconds between status checks.
        timeout: Maximum seconds to wait before giving up.
        on_status_update: Optional async callback invoked on each poll with
            (order_result, elapsed_seconds). Used for WebSocket broadcast,
            terminal display, etc.

    Returns:
        The final OrderResult. If the order reached a terminal state (FILLED,
        CANCELLED, REJECTED, FAILED), the result reflects that. On timeout,
        returns the last known status with a timeout note in the message.
    """
    from app.services.ledger import get_ledger
    from app.services.position_store import get_position_store

    provider = ProviderRegistry.get(broker)
    start = time.monotonic()
    last_result: Optional[OrderResult] = None
    last_logged_status: Optional[str] = None

    while True:
        elapsed = time.monotonic() - start
        if elapsed >= timeout:
            break

        await asyncio.sleep(poll_interval)
        elapsed = time.monotonic() - start

        result = await provider.get_order_status(order_id)
        last_result = result

        # Notify listener
        if on_status_update is not None:
            await on_status_update(result, elapsed)

        # Log status change only when status actually changes
        current_status = result.status.value
        if current_status != last_logged_status:
            ledger = get_ledger()
            if ledger:
                await ledger.log_status_change(
                    broker=broker, order_id=order_id,
                    status=current_status,
                )
            last_logged_status = current_status

        if result.status in TERMINAL_STATUSES:
            # On fill, create position from pending request (or close existing)
            if result.status == OrderStatus.FILLED and order_id in _pending_orders:
                request = _pending_orders.pop(order_id)
                store = get_position_store()
                if store:
                    if request.closing_position_id:
                        exit_price = result.filled_price or 0
                        pos_id = _handle_close_position(request, exit_price=exit_price)
                        if ledger:
                            await ledger.log_position_closed(
                                broker=broker, position_id=pos_id,
                                source=PositionSource.LIVE_API,
                                data={"order_id": order_id},
                            )
                    else:
                        pos_id = store.add_position(request, result, is_paper=False)
                        if ledger:
                            await ledger.log_position_opened(
                                broker=broker, position_id=pos_id,
                                source=PositionSource.LIVE_API,
                                data={"order_id": order_id},
                            )
            else:
                _pending_orders.pop(order_id, None)

            logger.info("Order %s reached terminal state: %s", order_id, result.status.value)
            return result

    # Timeout — order still pending
    logger.warning("Order %s polling timed out after %.1fs", order_id, timeout)

    if last_result is None:
        last_result = OrderResult(
            order_id=order_id,
            broker=broker,
            status=OrderStatus.SUBMITTED,
            message=f"Polling timed out after {timeout:.0f}s — order may still fill",
        )
    else:
        last_result.message = (
            f"{last_result.message} "
            f"[polling timed out after {timeout:.0f}s — order may still fill]"
        ).strip()

    # Don't remove from _pending_orders on timeout — position sync may pick it up
    return last_result


async def poll_order_status(broker: Broker, order_id: str) -> None:
    """Background task: poll order status and broadcast updates via WebSocket.

    Thin wrapper around await_order_fill() for backward compatibility with
    code that fires this as a background task.
    """
    from app.config import settings

    async def _broadcast(result: OrderResult, elapsed: float) -> None:
        await ws_manager.broadcast_order_update(result)

    await await_order_fill(
        broker=broker,
        order_id=order_id,
        poll_interval=settings.order_poll_interval_seconds,
        timeout=settings.order_poll_timeout_seconds,
        on_status_update=_broadcast,
    )


def build_closing_trade_request(position: dict, quantity: int | None = None,
                                 net_price: float = 0.05) -> TradeRequest:
    """Build a TradeRequest that closes (or partially closes) an open position.

    Used by both POST /trade/close and the CLI close command.
    Reverses option legs (SELL→BUY_TO_CLOSE, BUY→SELL_TO_CLOSE).
    """
    from app.models import (
        Broker, EquityOrder, MultiLegOrder, OptionAction, OptionLeg,
        OptionType, OrderSide,
    )

    order_type = position.get("order_type", "equity")
    symbol = position.get("symbol", "")
    legs = position.get("legs") or []
    close_qty = quantity or int(abs(position.get("quantity", 1)))
    broker = Broker(position.get("broker", "ibkr"))

    if order_type in ("multi_leg", "option") and legs:
        close_legs = []
        exp = position.get("expiration", "")
        exp_raw = exp.replace("-", "") if exp else ""
        for leg in legs:
            action = leg.get("action", "")
            if "SELL" in action:
                close_action = OptionAction.BUY_TO_CLOSE
            else:
                close_action = OptionAction.SELL_TO_CLOSE

            close_legs.append(OptionLeg(
                symbol=symbol,
                expiration=exp_raw,
                strike=float(leg.get("strike", 0)),
                option_type=OptionType(leg.get("option_type", "PUT")),
                action=close_action,
                quantity=int(leg.get("quantity", 1)),
            ))

        return TradeRequest(
            multi_leg_order=MultiLegOrder(
                broker=broker,
                legs=close_legs,
                quantity=close_qty,
                net_price=net_price,
            )
        )
    else:
        return TradeRequest(
            equity_order=EquityOrder(
                broker=broker,
                symbol=symbol,
                side=OrderSide.SELL if position.get("quantity", 0) > 0 else OrderSide.BUY,
                quantity=close_qty,
            )
        )
