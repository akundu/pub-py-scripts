"""WebSocket connection manager for real-time order status streaming."""

from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass, field

from fastapi import WebSocket

from app.models import OrderResult

logger = logging.getLogger(__name__)


@dataclass
class ConnectionManager:
    """Manages active WebSocket connections and broadcasts order updates."""

    _connections: list[WebSocket] = field(default_factory=list)
    _lock: asyncio.Lock = field(default_factory=asyncio.Lock)

    async def accept(self, ws: WebSocket) -> None:
        await ws.accept()
        async with self._lock:
            self._connections.append(ws)
        logger.info("WebSocket client connected (%d total)", len(self._connections))

    async def disconnect(self, ws: WebSocket) -> None:
        async with self._lock:
            if ws in self._connections:
                self._connections.remove(ws)
        logger.info("WebSocket client disconnected (%d remaining)", len(self._connections))

    async def broadcast_order_update(self, result: OrderResult) -> None:
        """Send order status update to all connected clients."""
        payload = json.dumps(
            {
                "type": "order_update",
                "order_id": result.order_id,
                "broker": result.broker.value,
                "status": result.status.value,
                "message": result.message,
                "timestamp": result.created_at.isoformat(),
            }
        )
        async with self._lock:
            stale: list[WebSocket] = []
            for ws in self._connections:
                try:
                    await ws.send_text(payload)
                except Exception:
                    stale.append(ws)
            for ws in stale:
                self._connections.remove(ws)


ws_manager = ConnectionManager()


@dataclass
class QuoteStreamManager:
    """Manages WebSocket connections for real-time quote streaming.

    Clients can subscribe to specific symbols. Ticks are broadcast only
    to clients subscribed to that symbol.
    """

    # client -> set of subscribed symbols
    _subscriptions: dict[WebSocket, set[str]] = field(default_factory=dict)
    _lock: asyncio.Lock = field(default_factory=asyncio.Lock)

    async def accept(self, ws: WebSocket) -> None:
        await ws.accept()
        async with self._lock:
            self._subscriptions[ws] = set()
        logger.info("Quote WS client connected (%d total)", len(self._subscriptions))

    async def disconnect(self, ws: WebSocket) -> None:
        async with self._lock:
            self._subscriptions.pop(ws, None)
        logger.info("Quote WS client disconnected (%d remaining)", len(self._subscriptions))

    async def subscribe_client(self, ws: WebSocket, symbols: list[str]) -> None:
        """Subscribe a client to specific symbols."""
        async with self._lock:
            if ws in self._subscriptions:
                self._subscriptions[ws].update(s.upper() for s in symbols)

    async def unsubscribe_client(self, ws: WebSocket, symbols: list[str]) -> None:
        """Unsubscribe a client from specific symbols."""
        async with self._lock:
            if ws in self._subscriptions:
                self._subscriptions[ws] -= {s.upper() for s in symbols}

    async def broadcast_quote(self, tick: dict) -> None:
        """Broadcast a tick update to all clients subscribed to that symbol."""
        symbol = tick.get("symbol", "").upper()
        if not symbol:
            return

        payload = json.dumps({"type": "quote_update", **tick})

        async with self._lock:
            stale: list[WebSocket] = []
            for ws, subs in self._subscriptions.items():
                # Send if client is subscribed to this symbol, or has no subscriptions (= all)
                if not subs or symbol in subs:
                    try:
                        await ws.send_text(payload)
                    except Exception:
                        stale.append(ws)
            for ws in stale:
                self._subscriptions.pop(ws, None)


quote_manager = QuoteStreamManager()
