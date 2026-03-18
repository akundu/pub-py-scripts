"""WebSocket endpoints — order status and real-time quote streaming."""

from __future__ import annotations

import json

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from app.websocket import quote_manager, ws_manager

router = APIRouter(tags=["websocket"])


@router.websocket("/ws/orders")
async def ws_orders(websocket: WebSocket) -> None:
    """Stream order status changes (PENDING -> FILLED etc.) in real-time."""
    await ws_manager.accept(websocket)
    try:
        while True:
            # Keep connection alive; client can send pings or commands
            await websocket.receive_text()
    except WebSocketDisconnect:
        await ws_manager.disconnect(websocket)


@router.websocket("/ws/quotes")
async def ws_quotes(websocket: WebSocket) -> None:
    """Stream real-time quote updates from IBKR market data subscriptions.

    Client sends JSON messages to subscribe/unsubscribe:
        {"action": "subscribe", "symbols": ["SPX", "NDX"]}
        {"action": "unsubscribe", "symbols": ["SPX"]}

    Server broadcasts on each tick batch:
        {"type": "quote_update", "symbol": "SPX", "bid": 5683.25, "ask": 5683.50, ...}

    If no symbols specified, client receives ALL streaming quotes.
    """
    await quote_manager.accept(websocket)
    try:
        while True:
            raw = await websocket.receive_text()
            try:
                msg = json.loads(raw)
                action = msg.get("action", "")
                symbols = msg.get("symbols", [])
                if action == "subscribe" and symbols:
                    await quote_manager.subscribe_client(websocket, symbols)
                elif action == "unsubscribe" and symbols:
                    await quote_manager.unsubscribe_client(websocket, symbols)
            except (json.JSONDecodeError, AttributeError):
                pass  # Ignore malformed messages
    except WebSocketDisconnect:
        await quote_manager.disconnect(websocket)
