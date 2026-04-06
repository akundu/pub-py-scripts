"""Profit target monitoring service — auto-closes positions at profit target.

Follows the same pattern as ExpirationService: background loop checks positions
every 30s during market hours and closes when profit target is reached.

Targets are stored in a JSON file alongside positions.json so they survive
daemon restarts.
"""

from __future__ import annotations

import json
import logging
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Optional

from app.models import Broker, LedgerEventType, OrderResult, OrderStatus, PositionSource
from app.services.ledger import TransactionLedger
from app.services.position_store import PlatformPositionStore

logger = logging.getLogger(__name__)

# Module-level singleton
_service: Optional["ProfitTargetService"] = None


def init_profit_target_service(
    data_dir: Path | str,
    position_store: PlatformPositionStore,
    ledger: TransactionLedger,
) -> "ProfitTargetService":
    global _service
    _service = ProfitTargetService(data_dir, position_store, ledger)
    return _service


def get_profit_target_service() -> Optional["ProfitTargetService"]:
    return _service


def reset_profit_target_service() -> None:
    global _service
    _service = None


class ProfitTargetService:
    """Monitors open positions and auto-closes when profit target is reached."""

    def __init__(
        self,
        data_dir: Path | str,
        position_store: PlatformPositionStore,
        ledger: TransactionLedger,
    ) -> None:
        self._store = position_store
        self._ledger = ledger
        self._data_dir = Path(data_dir)
        self._targets_path = self._data_dir / "profit_targets.json"
        self._close_in_progress: set[str] = set()  # Per-position lock
        self._targets: dict[str, dict] = {}
        self._load()

    def _load(self) -> None:
        """Load targets from JSON file."""
        if self._targets_path.exists():
            try:
                with open(self._targets_path) as f:
                    data = json.load(f)
                self._targets = data.get("positions", {})
                logger.info("Loaded %d profit targets", len(self._targets))
            except Exception as e:
                logger.warning("Failed to load profit targets: %s", e)
                self._targets = {}
        else:
            self._targets = {}

    def _save(self) -> None:
        """Save targets to JSON file."""
        try:
            self._data_dir.mkdir(parents=True, exist_ok=True)
            with open(self._targets_path, "w") as f:
                json.dump({"positions": self._targets}, f, indent=2, default=str)
        except Exception as e:
            logger.warning("Failed to save profit targets: %s", e)

    def set_target(
        self,
        position_id: str,
        entry_credit: float,
        profit_target_pct: float,
        symbol: str = "",
        short_strike: float = 0,
        long_strike: float = 0,
        quantity: int = 1,
    ) -> None:
        """Register a profit target for a position."""
        if profit_target_pct <= 0:
            return  # 0 = no target

        self._targets[position_id] = {
            "entry_credit": entry_credit,
            "profit_target_pct": profit_target_pct,
            "symbol": symbol,
            "short_strike": short_strike,
            "long_strike": long_strike,
            "quantity": quantity,
            "created_at": datetime.now(UTC).isoformat(),
        }
        self._save()
        logger.info(
            "Profit target set: %s %s %.0f%% (credit=%.2f)",
            position_id[:8], symbol, profit_target_pct, entry_credit,
        )

    def remove_target(self, position_id: str) -> bool:
        """Remove a profit target. Returns True if found and removed."""
        if position_id in self._targets:
            del self._targets[position_id]
            self._save()
            return True
        # Try prefix match
        for pid in list(self._targets.keys()):
            if pid.startswith(position_id):
                del self._targets[pid]
                self._save()
                return True
        return False

    def update_target(self, position_id: str, profit_target_pct: float) -> bool:
        """Update a position's profit target percentage."""
        # Find by exact or prefix match
        matched = None
        for pid in self._targets:
            if pid == position_id or pid.startswith(position_id):
                matched = pid
                break
        if not matched:
            return False

        if profit_target_pct <= 0:
            del self._targets[matched]
        else:
            self._targets[matched]["profit_target_pct"] = profit_target_pct
        self._save()
        return True

    def get_targets(self) -> dict[str, dict]:
        """Return all active targets."""
        return dict(self._targets)

    async def check_targets(self) -> list[str]:
        """Check all targets against current marks. Close positions that hit target.

        Returns list of position IDs that were closed.
        """
        if not self._targets:
            return []

        closed_ids: list[str] = []
        open_positions = self._store.get_open_positions()
        pos_by_id = {p.get("position_id", ""): p for p in open_positions}

        for pid, target in list(self._targets.items()):
            # Skip if already being closed
            if pid in self._close_in_progress:
                continue

            pos = pos_by_id.get(pid)
            if not pos or pos.get("status") != "open":
                # Position no longer open — remove target
                self.remove_target(pid)
                continue

            entry_credit = target.get("entry_credit", 0)
            target_pct = target.get("profit_target_pct", 50)
            if not entry_credit or entry_credit <= 0:
                continue

            # Current mark (what it costs to buy back the spread)
            mark = pos.get("market_price")
            if mark is None:
                # Try avg_cost as fallback
                mark = pos.get("avg_cost")
            if mark is None:
                continue

            btc_cost = abs(mark)
            if entry_credit <= btc_cost:
                continue  # No profit yet

            profit_pct = ((entry_credit - btc_cost) / entry_credit) * 100

            if profit_pct >= target_pct:
                self._close_in_progress.add(pid)
                try:
                    logger.info(
                        "Profit target HIT: %s %s profit=%.1f%% (target=%.0f%%), closing at MARKET",
                        pid[:8], target.get("symbol"), profit_pct, target_pct,
                    )

                    # Close via position store + ledger (same as ExpirationService)
                    self._store.close_position(pid, btc_cost, "profit_target")
                    broker = Broker(pos.get("broker", "ibkr"))
                    source = PositionSource(pos.get("source", "live_api"))

                    await self._ledger.log_position_closed(
                        broker=broker,
                        position_id=pid,
                        source=source,
                        data={
                            "reason": "profit_target",
                            "exit_price": btc_cost,
                            "profit_pct": round(profit_pct, 1),
                            "target_pct": target_pct,
                            "entry_credit": entry_credit,
                        },
                    )

                    self.remove_target(pid)
                    closed_ids.append(pid)

                except Exception as e:
                    logger.error("Profit target close failed %s: %s", pid[:8], e)
                finally:
                    self._close_in_progress.discard(pid)

        if closed_ids:
            logger.info("Profit target: closed %d positions", len(closed_ids))

        return closed_ids
