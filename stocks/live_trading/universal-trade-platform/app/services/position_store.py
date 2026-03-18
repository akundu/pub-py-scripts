"""JSON-backed position tracking for live and paper trades."""

from __future__ import annotations

import json
import logging
import os
import tempfile
import threading
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Optional

from app.models import (
    Broker,
    OrderResult,
    PositionSource,
    TradeRequest,
    TrackedPosition,
)

logger = logging.getLogger(__name__)


class PlatformPositionStore:
    """Persistent position store backed by a JSON file."""

    def __init__(self, store_path: Path) -> None:
        self._path = store_path
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._positions: dict[str, dict] = {}
        self._lock = threading.Lock()
        self._load()

    def _load(self) -> None:
        if self._path.exists():
            try:
                with open(self._path, "r") as f:
                    self._positions = json.load(f)
            except (json.JSONDecodeError, OSError) as e:
                logger.debug("Failed to load positions: %s", e)
                self._positions = {}

    def _save(self) -> None:
        fd, tmp = tempfile.mkstemp(dir=str(self._path.parent), suffix=".tmp")
        try:
            with os.fdopen(fd, "w") as f:
                json.dump(self._positions, f, indent=2, default=str)
            os.replace(tmp, str(self._path))
        except BaseException:
            try:
                os.unlink(tmp)
            except OSError:
                pass
            raise

    def add_position(
        self,
        request: TradeRequest,
        result: OrderResult,
        is_paper: bool = False,
    ) -> str:
        """Create a tracked position from a trade request and result."""
        with self._lock:
            source = PositionSource.PAPER if is_paper else PositionSource.LIVE_API

            if request.equity_order:
                order = request.equity_order
                pos = TrackedPosition(
                    source=source,
                    broker=order.broker,
                    order_type="equity",
                    symbol=order.symbol,
                    side=order.side.value,
                    quantity=order.quantity,
                    entry_price=order.limit_price or 0,
                )
            else:
                order = request.multi_leg_order
                assert order is not None
                legs_data = [
                    {
                        "symbol": leg.symbol,
                        "expiration": leg.expiration,
                        "strike": leg.strike,
                        "option_type": leg.option_type.value,
                        "action": leg.action.value,
                        "quantity": leg.quantity,
                    }
                    for leg in order.legs
                ]
                expirations = [leg.expiration for leg in order.legs]
                nearest_exp = min(expirations) if expirations else None

                pos = TrackedPosition(
                    source=source,
                    broker=order.broker,
                    order_type="multi_leg",
                    symbol=order.legs[0].symbol if order.legs else "UNKNOWN",
                    quantity=order.quantity,
                    entry_price=order.net_price or 0,
                    legs=legs_data,
                    expiration=nearest_exp,
                )

            if result.filled_price is not None:
                pos.entry_price = result.filled_price
            pos.order_id = result.order_id

            self._positions[pos.position_id] = json.loads(pos.model_dump_json())
            self._save()
            return pos.position_id

    def add_position_from_sync(
        self,
        broker: Broker,
        symbol: str,
        quantity: float,
        avg_cost: float,
        market_value: float,
        unrealized_pnl: float,
        source: PositionSource = PositionSource.EXTERNAL_SYNC,
    ) -> str:
        """Add a position discovered via broker sync."""
        with self._lock:
            # Use abs(avg_cost) — IBKR may return negative for short positions
            # but entry_price should be the absolute cost basis per share
            entry = abs(avg_cost) if avg_cost else 0
            pos = TrackedPosition(
                source=source,
                broker=broker,
                order_type="equity",
                symbol=symbol,
                quantity=quantity,
                entry_price=entry,
                current_mark=market_value / quantity if quantity else 0,
                unrealized_pnl=unrealized_pnl,
                last_synced_at=datetime.now(UTC),
            )
            self._positions[pos.position_id] = json.loads(pos.model_dump_json())
            self._save()
            return pos.position_id

    def add_position_from_csv(
        self,
        broker: Broker,
        symbol: str,
        side: str,
        quantity: float,
        price: float,
        trade_date: datetime,
        status: str = "open",
    ) -> str:
        """Add a position imported from CSV."""
        with self._lock:
            pos = TrackedPosition(
                source=PositionSource.CSV_IMPORT,
                broker=broker,
                order_type="equity",
                symbol=symbol,
                side=side,
                quantity=quantity,
                entry_price=price,
                entry_time=trade_date,
                status=status,
            )
            self._positions[pos.position_id] = json.loads(pos.model_dump_json())
            self._save()
            return pos.position_id

    def close_position(
        self, position_id: str, exit_price: float, reason: str
    ) -> dict:
        """Close a position and compute P&L."""
        with self._lock:
            if position_id not in self._positions:
                raise KeyError(f"Position {position_id} not found")

            pos = self._positions[position_id]
            pos["status"] = "closed"
            pos["exit_price"] = exit_price
            pos["exit_reason"] = reason
            pos["exit_time"] = datetime.now(UTC).isoformat()

            entry_price = pos.get("entry_price", 0)
            quantity = pos.get("quantity", 0)
            order_type = pos.get("order_type", "equity")

            if order_type == "multi_leg":
                # Detect credit vs debit from leg actions.
                # Credit spread: first leg is SELL_TO_OPEN → you received premium.
                # Debit spread: first leg is BUY_TO_OPEN → you paid premium.
                # IBKR fills: entry_price is negative for credits, positive for debits.
                # Dry-run/stub: entry_price is always positive (from net_price).
                legs = pos.get("legs") or []
                first_action = legs[0].get("action", "") if legs else ""
                is_credit = "SELL" in first_action

                if is_credit:
                    # Credit spread: P&L = credit_received - cost_to_close
                    # Use abs() because entry may be negative (IBKR) or positive (dry-run)
                    pnl = (abs(entry_price) - exit_price) * quantity * 100
                else:
                    # Debit spread: P&L = value_on_close - debit_paid
                    pnl = (exit_price - abs(entry_price)) * quantity * 100
            else:
                side = pos.get("side", "BUY")
                if side == "SELL":
                    pnl = (entry_price - exit_price) * quantity
                else:
                    pnl = (exit_price - entry_price) * quantity

            pos["pnl"] = round(pnl, 2)
            self._save()
            return pos

    def reduce_quantity(self, position_id: str, amount: int) -> dict:
        """Reduce position quantity by *amount* contracts.

        If quantity drops to zero or below, the position is auto-closed
        with exit_reason='fully_reduced'.
        """
        with self._lock:
            if position_id not in self._positions:
                raise KeyError(f"Position {position_id} not found")
            pos = self._positions[position_id]
            current = pos.get("quantity", 0)
            new_qty = max(0, current - amount)
            pos["quantity"] = new_qty
            if new_qty <= 0:
                pos["status"] = "closed"
                pos["exit_reason"] = "fully_reduced"
                pos["exit_time"] = datetime.now(UTC).isoformat()
            self._save()
            return pos

    def update_mark(
        self, position_id: str, current_price: float, timestamp: Optional[datetime] = None
    ) -> None:
        """Update mark-to-market for a position."""
        with self._lock:
            if position_id not in self._positions:
                return

            pos = self._positions[position_id]
            pos["current_mark"] = current_price
            pos["last_synced_at"] = (timestamp or datetime.now(UTC)).isoformat()

            entry_price = pos.get("entry_price", 0)
            quantity = pos.get("quantity", 0)
            order_type = pos.get("order_type", "equity")

            if order_type == "multi_leg":
                pos["unrealized_pnl"] = round((entry_price - current_price) * quantity * 100, 2)
            else:
                side = pos.get("side", "BUY")
                if side == "SELL":
                    pos["unrealized_pnl"] = round((entry_price - current_price) * quantity, 2)
                else:
                    pos["unrealized_pnl"] = round((current_price - entry_price) * quantity, 2)

            self._save()

    def get_open_positions(self, include_paper: bool = True) -> list[dict]:
        """Return all open positions."""
        results = []
        for pos in self._positions.values():
            if pos.get("status") != "open":
                continue
            if not include_paper and pos.get("source") == PositionSource.PAPER.value:
                continue
            results.append(pos)
        return results

    def get_closed_positions(
        self, start_date: Optional[date] = None, end_date: Optional[date] = None
    ) -> list[dict]:
        """Return closed positions, optionally filtered by date range."""
        results = []
        for pos in self._positions.values():
            if pos.get("status") != "closed":
                continue
            if start_date or end_date:
                exit_time_str = pos.get("exit_time")
                if exit_time_str:
                    try:
                        exit_date = datetime.fromisoformat(exit_time_str).date()
                        if start_date and exit_date < start_date:
                            continue
                        if end_date and exit_date > end_date:
                            continue
                    except (ValueError, TypeError):
                        continue
            results.append(pos)
        return results

    def get_expired_positions(self, today: date) -> list[dict]:
        """Return open positions with expiration before today.

        Same-day expirations (0DTE) are NOT returned — they are still live
        until market close. Use check_eod_exits() for same-day closure.
        Only positions expiring strictly before today are considered expired.
        """
        results = []
        for pos in self._positions.values():
            if pos.get("status") != "open":
                continue
            exp = pos.get("expiration")
            if exp:
                try:
                    exp_date = date.fromisoformat(exp)
                    if exp_date < today:  # strictly before, not on
                        results.append(pos)
                except (ValueError, TypeError):
                    continue
        return results

    def get_position(self, position_id: str) -> Optional[dict]:
        """Get a single position by ID."""
        return self._positions.get(position_id)

    def get_account_summary(self) -> dict:
        """Compute account summary from positions."""
        open_positions = self.get_open_positions()
        closed_positions = self.get_closed_positions()

        cash_deployed = sum(
            abs(p.get("entry_price", 0)) * abs(p.get("quantity", 0))
            for p in open_positions
        )
        realized_pnl = sum(p.get("pnl", 0) for p in closed_positions)
        unrealized_pnl = sum(p.get("unrealized_pnl") or 0 for p in open_positions)

        return {
            "open_count": len(open_positions),
            "closed_count": len(closed_positions),
            "cash_deployed": round(cash_deployed, 2),
            "realized_pnl": round(realized_pnl, 2),
            "unrealized_pnl": round(unrealized_pnl, 2),
            "total_pnl": round(realized_pnl + unrealized_pnl, 2),
        }

    def find_by_broker_symbol(self, broker: Broker, symbol: str) -> Optional[dict]:
        """Find an open position by broker and symbol."""
        for pos in self._positions.values():
            if (
                pos.get("status") == "open"
                and pos.get("broker") == broker.value
                and pos.get("symbol") == symbol
            ):
                return pos
        return None

    def export_results(self) -> list[dict]:
        """Export closed positions in a format compatible with metrics computation."""
        closed = self.get_closed_positions()
        results = []
        for pos in closed:
            results.append({
                "pnl": pos.get("pnl", 0),
                "credit": abs(pos.get("entry_price", 0)),
                "max_loss": abs(pos.get("entry_price", 0)) * pos.get("quantity", 1),
                "symbol": pos.get("symbol", ""),
                "entry_time": pos.get("entry_time", ""),
                "exit_time": pos.get("exit_time", ""),
                "exit_reason": pos.get("exit_reason", ""),
            })
        return results


# ── Module-level accessor ─────────────────────────────────────────────────────

_position_store: Optional[PlatformPositionStore] = None


def get_position_store() -> Optional[PlatformPositionStore]:
    return _position_store


def init_position_store(data_dir: Path) -> PlatformPositionStore:
    global _position_store
    store_path = Path(data_dir) / "positions.json"
    _position_store = PlatformPositionStore(store_path)
    logger.info("Position store initialized at %s", store_path)
    return _position_store


def reset_position_store() -> None:
    """Reset the module-level store (for tests)."""
    global _position_store
    _position_store = None
