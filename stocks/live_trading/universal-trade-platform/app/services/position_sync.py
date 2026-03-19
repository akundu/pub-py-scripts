"""Background position sync loop — polls brokers for out-of-band positions."""

from __future__ import annotations

import logging
from datetime import UTC, datetime, time
from typing import Optional

from app.core.provider import ProviderRegistry
from app.models import (
    Broker,
    LedgerEventType,
    PositionSource,
    ReconciliationEntry,
    ReconciliationReport,
    SyncResult,
)
from app.services.ledger import TransactionLedger
from app.services.position_store import PlatformPositionStore
from app.websocket import ConnectionManager

logger = logging.getLogger(__name__)

MARKET_OPEN_UTC = time(13, 30)   # 9:30 AM ET
MARKET_CLOSE_UTC = time(20, 0)   # 4:00 PM ET


class PositionSyncService:
    """Polls all connected brokers for positions, diffs against store."""

    def __init__(
        self,
        position_store: PlatformPositionStore,
        ledger: TransactionLedger,
        ws_manager: Optional[ConnectionManager] = None,
    ) -> None:
        self._store = position_store
        self._ledger = ledger
        self._ws = ws_manager

    async def sync_all_brokers(self) -> SyncResult:
        """Sync positions from all registered providers.

        Groups option legs with the same symbol+expiration into multi_leg
        positions (credit/debit spreads, iron condors) instead of importing
        each leg separately.
        """
        result = SyncResult()
        providers = ProviderRegistry.all()

        for provider in providers:
            try:
                broker_positions = await provider.get_positions()
                broker = provider.broker

                # Separate equities from options, group options by (symbol, expiration)
                equities = []
                option_groups: dict[tuple[str, str], list] = {}  # (symbol, exp) -> [positions]

                for bp in broker_positions:
                    sec_type = getattr(bp, "sec_type", None) or ""
                    if sec_type in ("OPT", "FOP"):
                        exp = getattr(bp, "expiration", "") or ""
                        key = (bp.symbol, exp)
                        option_groups.setdefault(key, []).append(bp)
                    else:
                        equities.append(bp)

                # Import equities (one position per con_id)
                for bp in equities:
                    con_id = getattr(bp, "con_id", None)
                    if con_id:
                        existing = self._store.find_by_con_id(con_id)
                    else:
                        existing = self._store.find_by_broker_symbol(broker, bp.symbol)

                    if existing is None:
                        pos_id = self._store.add_position_from_sync(
                            broker=broker, symbol=bp.symbol,
                            quantity=bp.quantity, avg_cost=bp.avg_cost,
                            market_value=bp.market_value,
                            unrealized_pnl=bp.unrealized_pnl,
                            con_id=con_id,
                            sec_type=getattr(bp, "sec_type", None),
                        )
                        await self._log_sync(broker, pos_id, bp.symbol, bp.quantity, bp.avg_cost, con_id)
                        result.new_positions += 1
                    else:
                        self._update_existing(existing, bp, con_id)
                        result.updated_positions += 1

                # Import option groups as multi_leg positions
                for (symbol, exp), legs in option_groups.items():
                    # Check if we already have a multi_leg position for this symbol+exp
                    existing = self._find_multi_leg(broker, symbol, exp, legs)

                    if existing is None:
                        # Build a combined multi_leg position
                        con_ids = [getattr(l, "con_id", None) for l in legs]
                        total_mv = sum(l.market_value for l in legs)
                        total_upnl = sum(l.unrealized_pnl for l in legs)
                        # Net avg_cost across legs
                        total_avg = sum(
                            (-1 if l.quantity < 0 else 1) * l.avg_cost
                            for l in legs
                        )
                        # Use the absolute quantity of the first leg as spread quantity
                        # (e.g., 5 short + 5 long = qty 5)
                        spread_qty = min(abs(l.quantity) for l in legs) if legs else 1
                        # Determine if it's a spread (2 legs) or condor (4 legs)
                        order_type = "multi_leg"

                        # Build legs list for display
                        legs_data = []
                        for l in sorted(legs, key=lambda x: getattr(x, "strike", 0) or 0):
                            action = "SELL" if l.quantity < 0 else "BUY"
                            right = getattr(l, "right", "") or ""
                            strike = getattr(l, "strike", 0) or 0
                            legs_data.append({
                                "action": action,
                                "option_type": "PUT" if right == "P" else "CALL",
                                "strike": strike,
                                "quantity": int(abs(l.quantity)),
                                "con_id": getattr(l, "con_id", None),
                            })

                        # Format expiration
                        fmt_exp = None
                        if exp and len(exp) == 8:
                            fmt_exp = f"{exp[:4]}-{exp[4:6]}-{exp[6:8]}"
                        elif exp:
                            fmt_exp = exp

                        pos_id = self._store.add_position_from_sync(
                            broker=broker, symbol=symbol,
                            quantity=spread_qty,
                            avg_cost=total_avg,
                            market_value=total_mv,
                            unrealized_pnl=total_upnl,
                            con_id=con_ids[0] if con_ids else None,
                            sec_type="OPT",
                            expiration=exp,
                            strike=getattr(legs[0], "strike", None),
                            right=getattr(legs[0], "right", None),
                        )
                        # Override order_type and add legs
                        self._store.update_field(pos_id, "order_type", order_type)
                        self._store.update_field(pos_id, "legs", legs_data)
                        self._store.update_field(pos_id, "con_ids", con_ids)

                        await self._log_sync(broker, pos_id, symbol, spread_qty, total_avg,
                                           con_ids[0] if con_ids else None)
                        result.new_positions += 1
                    else:
                        # Update existing multi_leg
                        pos_id = existing.get("position_id", "")
                        total_mv = sum(l.market_value for l in legs)
                        total_upnl = sum(l.unrealized_pnl for l in legs)
                        mark = total_mv / abs(existing.get("quantity", 1)) if existing.get("quantity") else 0
                        self._store.update_mark(pos_id, mark, datetime.now(UTC))
                        self._store.update_field(pos_id, "unrealized_pnl", total_upnl)
                        result.updated_positions += 1

                result.brokers_synced.append(broker.value)
            except Exception as e:
                logger.error("Failed to sync broker %s: %s", provider.broker.value, e)

        if result.new_positions > 0:
            logger.info(
                "Position sync: %d new, %d updated across %s",
                result.new_positions,
                result.updated_positions,
                result.brokers_synced,
            )

        return result

    def _find_multi_leg(self, broker, symbol, exp, legs) -> Optional[dict]:
        """Find an existing multi_leg position matching these option legs."""
        # First check by con_ids
        for leg in legs:
            con_id = getattr(leg, "con_id", None)
            if con_id:
                existing = self._store.find_by_con_id(con_id)
                if existing:
                    return existing
                # Check if con_id is in a position's con_ids list
                for pos in self._store.get_open_positions():
                    if con_id in (pos.get("con_ids") or []):
                        return pos

        # Fall back: match by symbol + expiration + order_type
        fmt_exp = f"{exp[:4]}-{exp[4:6]}-{exp[6:8]}" if exp and len(exp) == 8 else exp
        for pos in self._store.get_open_positions():
            if (pos.get("broker") == broker.value
                    and pos.get("symbol") == symbol
                    and pos.get("order_type") in ("multi_leg", "option")
                    and pos.get("expiration") == fmt_exp):
                return pos
        return None

    def _update_existing(self, existing, bp, con_id):
        """Update mark and con_id on an existing position."""
        pos_id = existing.get("position_id", "")
        current_mark = bp.market_value / bp.quantity if bp.quantity else 0
        self._store.update_mark(pos_id, current_mark, datetime.now(UTC))
        if con_id and not existing.get("con_id"):
            self._store.update_field(pos_id, "con_id", con_id)

    async def _log_sync(self, broker, pos_id, symbol, quantity, avg_cost, con_id):
        """Log a position sync event to the ledger."""
        await self._ledger.append(
            __import__("app.models", fromlist=["LedgerEntry"]).LedgerEntry(
                event_type=LedgerEventType.POSITION_SYNCED,
                broker=broker,
                position_id=pos_id,
                source=PositionSource.EXTERNAL_SYNC,
                data={
                    "symbol": symbol,
                    "quantity": quantity,
                    "avg_cost": avg_cost,
                    "con_id": con_id,
                    "action": "new",
                },
            )
        )

    async def reconcile(self, broker: Broker) -> ReconciliationReport:
        """Compare system positions against broker-reported positions.

        Args:
            broker: Which broker to reconcile against.

        Returns:
            ReconciliationReport with matched/mismatched entries.
        """
        provider = ProviderRegistry.get(broker)
        broker_positions = await provider.get_positions()

        # System open positions for this broker
        system_positions = [
            p for p in self._store.get_open_positions()
            if p.get("broker") == broker.value
        ]

        report = ReconciliationReport(
            broker=broker.value,
            total_system_positions=len(system_positions),
            total_broker_positions=len(broker_positions),
        )

        # Index broker positions by symbol
        broker_by_symbol: dict[str, float] = {}
        for bp in broker_positions:
            broker_by_symbol[bp.symbol] = bp.quantity

        # Index system positions by symbol
        system_by_symbol: dict[str, float] = {}
        for sp in system_positions:
            sym = sp.get("symbol", "")
            system_by_symbol[sym] = system_by_symbol.get(sym, 0) + sp.get("quantity", 0)

        all_symbols = set(broker_by_symbol.keys()) | set(system_by_symbol.keys())

        for sym in sorted(all_symbols):
            sys_qty = system_by_symbol.get(sym)
            brk_qty = broker_by_symbol.get(sym)

            if sys_qty is not None and brk_qty is not None:
                if abs(sys_qty - brk_qty) < 0.001:
                    report.matched += 1
                    report.discrepancies.append(ReconciliationEntry(
                        symbol=sym,
                        broker=broker.value,
                        system_quantity=sys_qty,
                        broker_quantity=brk_qty,
                        discrepancy_type="matched",
                        details=f"Quantities match: {sys_qty}",
                    ))
                else:
                    report.discrepancies.append(ReconciliationEntry(
                        symbol=sym,
                        broker=broker.value,
                        system_quantity=sys_qty,
                        broker_quantity=brk_qty,
                        discrepancy_type="quantity_mismatch",
                        details=f"System={sys_qty}, Broker={brk_qty}",
                    ))
            elif sys_qty is not None and brk_qty is None:
                report.discrepancies.append(ReconciliationEntry(
                    symbol=sym,
                    broker=broker.value,
                    system_quantity=sys_qty,
                    broker_quantity=None,
                    discrepancy_type="missing_at_broker",
                    details=f"System has {sys_qty} but broker reports nothing",
                ))
            else:
                report.discrepancies.append(ReconciliationEntry(
                    symbol=sym,
                    broker=broker.value,
                    system_quantity=None,
                    broker_quantity=brk_qty,
                    discrepancy_type="missing_in_system",
                    details=f"Broker has {brk_qty} but system has no record",
                ))

        return report

    @staticmethod
    def is_trading_hours(now: Optional[datetime] = None) -> bool:
        """Check if current time is within market hours (9:30 AM - 4:00 PM ET)."""
        if now is None:
            now = datetime.now(UTC)
        t = now.time()
        return MARKET_OPEN_UTC <= t <= MARKET_CLOSE_UTC
