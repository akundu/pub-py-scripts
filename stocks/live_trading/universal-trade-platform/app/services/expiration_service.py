"""Expiration detection and EOD auto-close service."""

from __future__ import annotations

import logging
from datetime import UTC, date, datetime, time
from typing import Optional

from app.models import Broker, LedgerEventType, PositionSource
from app.services.ledger import TransactionLedger
from app.services.position_store import PlatformPositionStore
from app.websocket import ConnectionManager

logger = logging.getLogger(__name__)

MARKET_CLOSE_UTC = time(20, 0)  # 4 PM ET


class ExpirationService:
    """Detects expired options and auto-closes positions."""

    def __init__(
        self,
        position_store: PlatformPositionStore,
        ledger: TransactionLedger,
        ws_manager: Optional[ConnectionManager] = None,
    ) -> None:
        self._store = position_store
        self._ledger = ledger
        self._ws = ws_manager

    async def check_expirations(self, today: date) -> list[str]:
        """Find and auto-close expired positions."""
        expired = self._store.get_expired_positions(today)
        closed_ids: list[str] = []

        for pos in expired:
            position_id = pos.get("position_id", "")
            try:
                await self._auto_close(position_id, "expired", exit_price=0.0)
                closed_ids.append(position_id)
            except Exception as e:
                logger.error("Failed to close expired position %s: %s", position_id, e)

        if closed_ids:
            logger.info("Auto-closed %d expired positions", len(closed_ids))
        return closed_ids

    async def check_eod_exits(self, now: datetime) -> list[str]:
        """Close remaining 0DTE positions at end of day."""
        if now.time() < MARKET_CLOSE_UTC:
            return []

        today = now.date()
        open_positions = self._store.get_open_positions()
        closed_ids: list[str] = []

        for pos in open_positions:
            exp = pos.get("expiration")
            if exp:
                try:
                    exp_date = date.fromisoformat(exp)
                    if exp_date == today:
                        position_id = pos.get("position_id", "")
                        await self._auto_close(position_id, "eod_exit", exit_price=0.0)
                        closed_ids.append(position_id)
                except (ValueError, TypeError):
                    continue

        if closed_ids:
            logger.info("EOD auto-closed %d positions", len(closed_ids))
        return closed_ids

    def get_expiring_positions(self, target_date: date) -> list[dict]:
        """Preview positions that will expire on the target date."""
        results = []
        for pos in self._store.get_open_positions():
            exp = pos.get("expiration")
            if exp:
                try:
                    if date.fromisoformat(exp) == target_date:
                        results.append(pos)
                except (ValueError, TypeError):
                    continue
        return results

    async def _auto_close(
        self, position_id: str, reason: str, exit_price: float
    ) -> None:
        """Close a position in the store and log to ledger."""
        pos = self._store.get_position(position_id)
        if not pos:
            return

        self._store.close_position(position_id, exit_price, reason)
        broker = Broker(pos.get("broker", "robinhood"))
        source = PositionSource(pos.get("source", "live_api"))

        await self._ledger.log_position_closed(
            broker=broker,
            position_id=position_id,
            source=source,
            data={"reason": reason, "exit_price": exit_price},
        )

        if self._ws:
            from app.websocket import ws_manager
            # Broadcast position close event
            from app.models import OrderResult, OrderStatus
            await ws_manager.broadcast_order_update(
                OrderResult(
                    order_id=position_id,
                    broker=broker,
                    status=OrderStatus.FILLED,
                    message=f"Position auto-closed: {reason}",
                )
            )

        logger.info("Auto-closed position %s (reason=%s)", position_id, reason)
