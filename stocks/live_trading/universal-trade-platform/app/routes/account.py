"""Account endpoints — positions, sync, and expiration."""

from __future__ import annotations

import os
from datetime import UTC, date, datetime, timedelta
from typing import Annotated, Optional

from fastapi import APIRouter, HTTPException, Security

from app.auth import TokenData, require_auth
from app.core.provider import ProviderRegistry
from app.models import AggregatedPositions, Broker, ReconciliationReport, SyncResult

router = APIRouter(prefix="/account", tags=["account"])


def _is_worker() -> bool:
    """True if running as a uvicorn worker (not the IBKR process)."""
    return os.environ.get("_UTP_DAEMON_WORKER") == "1"


async def _proxy_to_ibkr(method: str, path: str, json_data: dict | None = None) -> dict:
    """Forward a request to the IBKR process (used by workers for state-mutating ops)."""
    import httpx
    port = int(os.environ.get("_UTP_DAEMON_PORT", "8001"))
    url = f"http://127.0.0.1:{port}{path}"
    async with httpx.AsyncClient(timeout=60.0) as client:
        if method == "POST":
            resp = await client.post(url, json=json_data)
        else:
            resp = await client.get(url)
        resp.raise_for_status()
        return resp.json()


@router.get("/positions", response_model=AggregatedPositions)
async def get_positions(
    _user: Annotated[TokenData, Security(require_auth, scopes=["account:read"])],
) -> AggregatedPositions:
    """Aggregate positions across all active broker providers."""
    return await ProviderRegistry.aggregate_positions()


@router.post("/sync", response_model=SyncResult)
async def sync_positions(
    _user: Annotated[TokenData, Security(require_auth, scopes=["account:read"])],
) -> SyncResult:
    """Manually trigger a position sync across all brokers."""
    if _is_worker():
        data = await _proxy_to_ibkr("POST", "/account/sync")
        return SyncResult(**data)

    from app.services.ledger import get_ledger
    from app.services.position_store import get_position_store
    from app.services.position_sync import PositionSyncService

    store = get_position_store()
    ledger = get_ledger()
    if not store or not ledger:
        raise HTTPException(status_code=503, detail="Services not initialized")

    sync_service = PositionSyncService(store, ledger)
    return await sync_service.sync_all_brokers()


@router.post("/check-expirations")
async def check_expirations(
    _user: Annotated[TokenData, Security(require_auth, scopes=["trades:write"])],
) -> dict:
    """Manually trigger expiration check and auto-close."""
    if _is_worker():
        return await _proxy_to_ibkr("POST", "/account/check-expirations")

    from app.services.expiration_service import ExpirationService
    from app.services.ledger import get_ledger
    from app.services.position_store import get_position_store

    store = get_position_store()
    ledger = get_ledger()
    if not store or not ledger:
        raise HTTPException(status_code=503, detail="Services not initialized")

    exp_service = ExpirationService(store, ledger)
    today = datetime.now(UTC).date()
    closed_ids = await exp_service.check_expirations(today)
    return {"closed_positions": closed_ids, "count": len(closed_ids)}


@router.get("/expiring")
async def expiring_positions(
    _user: Annotated[TokenData, Security(require_auth, scopes=["account:read"])],
    target_date: Optional[date] = None,
) -> list[dict]:
    """List positions expiring on a given date."""
    from app.services.expiration_service import ExpirationService
    from app.services.ledger import get_ledger
    from app.services.position_store import get_position_store

    store = get_position_store()
    ledger = get_ledger()
    if not store or not ledger:
        raise HTTPException(status_code=503, detail="Services not initialized")

    exp_service = ExpirationService(store, ledger)
    target = target_date or datetime.now(UTC).date()
    return exp_service.get_expiring_positions(target)


@router.get("/reconciliation", response_model=ReconciliationReport)
async def reconcile_positions(
    _user: Annotated[TokenData, Security(require_auth, scopes=["account:read"])],
    broker: str = "ibkr",
) -> ReconciliationReport:
    """Reconcile system positions against broker-reported positions."""
    from app.services.ledger import get_ledger
    from app.services.position_store import get_position_store
    from app.services.position_sync import PositionSyncService

    store = get_position_store()
    ledger = get_ledger()
    if not store or not ledger:
        raise HTTPException(status_code=503, detail="Services not initialized")

    try:
        broker_enum = Broker(broker)
    except ValueError:
        raise HTTPException(status_code=400, detail=f"Unknown broker: {broker}")

    sync_service = PositionSyncService(store, ledger)
    return await sync_service.reconcile(broker_enum)


@router.post("/flush")
async def flush_positions(
    _user: Annotated[TokenData, Security(require_auth, scopes=["trades:write"])],
) -> dict:
    """Flush open positions from daemon memory and re-sync from broker.

    Preserves closed positions (P&L history). Only clears open positions
    so they can be re-imported fresh from IBKR.
    """
    # Workers proxy state-mutating ops to the IBKR process which owns the data
    if _is_worker():
        return await _proxy_to_ibkr("POST", "/account/flush")

    from app.services.ledger import get_ledger
    from app.services.position_store import get_position_store
    from app.services.position_sync import PositionSyncService

    store = get_position_store()
    ledger = get_ledger()
    if not store or not ledger:
        raise HTTPException(status_code=503, detail="Services not initialized")

    # Clear only open positions, preserve closed
    open_cleared = 0
    closed_kept = 0
    with store._lock:
        to_remove = []
        for pid, pos in store._positions.items():
            if pos.get("status") == "open":
                to_remove.append(pid)
            else:
                closed_kept += 1
        for pid in to_remove:
            del store._positions[pid]
            open_cleared += 1
        store._save()

    # Re-sync from broker
    sync_service = PositionSyncService(store, ledger)
    sync_result = await sync_service.sync_all_brokers()

    return {
        "open_cleared": open_cleared,
        "closed_preserved": closed_kept,
        "synced_new": sync_result.new_positions,
        "synced_updated": sync_result.updated_positions,
        "open_positions": len(store.get_open_positions()),
    }


@router.post("/hard-reset")
async def hard_reset(
    _user: Annotated[TokenData, Security(require_auth, scopes=["trades:write"])],
) -> dict:
    """Hard reset: clear ALL positions (open + closed) in memory and on disk, then re-sync from broker.

    This is the only reliable way to reset when the daemon is running,
    since the daemon holds positions in memory.
    """
    if _is_worker():
        return await _proxy_to_ibkr("POST", "/account/hard-reset")

    from app.services.ledger import get_ledger
    from app.services.position_store import get_position_store
    from app.services.position_sync import PositionSyncService
    from app.services.execution_store import get_execution_store

    store = get_position_store()
    ledger = get_ledger()
    if not store or not ledger:
        raise HTTPException(status_code=503, detail="Services not initialized")

    # Clear in-memory positions (open + closed)
    old_count = len(store._positions)
    with store._lock:
        store._positions.clear()
        store._save()

    # Clear execution cache
    exec_store = get_execution_store()
    exec_cleared = 0
    if exec_store:
        exec_cleared = exec_store.flush()

    # Re-sync from broker
    sync_service = PositionSyncService(store, ledger)
    sync_result = await sync_service.sync_all_brokers()

    return {
        "cleared": old_count,
        "executions_cleared": exec_cleared,
        "synced_new": sync_result.new_positions,
        "synced_updated": sync_result.updated_positions,
        "open_positions": len(store.get_open_positions()),
    }


@router.get("/trades")
async def get_trades(
    _user: Annotated[TokenData, Security(require_auth, scopes=["account:read"])],
    days: int = 0,
    include_all: bool = False,
) -> list[dict]:
    """Return recent closed positions as trade history."""
    from app.services.live_data_service import get_live_data_service

    svc = get_live_data_service()
    if svc:
        return await svc.get_trades(days=days, include_all=include_all)

    from app.services.position_store import get_position_store

    store = get_position_store()
    if not store:
        raise HTTPException(status_code=503, detail="Position store not initialized")

    if include_all:
        return store.get_open_positions() + store.get_closed_positions()

    if days > 0:
        start = date.today() - timedelta(days=days)
        return store.get_closed_positions(start_date=start)

    return store.get_closed_positions(start_date=date.today())


@router.get("/orders")
async def get_orders(
    _user: Annotated[TokenData, Security(require_auth, scopes=["account:read"])],
    broker: str = "ibkr",
) -> list[dict]:
    """Return open/working orders from the broker."""
    try:
        broker_enum = Broker(broker)
    except ValueError:
        raise HTTPException(status_code=400, detail=f"Unknown broker: {broker}")

    provider = ProviderRegistry.get(broker_enum)
    orders = await provider.get_open_orders()
    return [o.model_dump() for o in orders]


@router.post("/cancel")
async def cancel_order(
    _user: Annotated[TokenData, Security(require_auth, scopes=["trades:write"])],
    order_id: str = "",
    broker: str = "ibkr",
) -> dict:
    """Cancel a working order by ID."""
    if not order_id:
        raise HTTPException(status_code=400, detail="order_id required")

    try:
        broker_enum = Broker(broker)
    except ValueError:
        raise HTTPException(status_code=400, detail=f"Unknown broker: {broker}")

    provider = ProviderRegistry.get(broker_enum)
    result = await provider.cancel_order(order_id)
    return result.model_dump()


@router.get("/executions")
async def get_executions(
    _user: Annotated[TokenData, Security(require_auth, scopes=["account:read"])],
    symbol: Optional[str] = None,
    flush: bool = False,
) -> dict:
    """Fetch IBKR executions (last ~7 days), merge into local cache, return grouped by order."""
    from app.services.execution_store import get_execution_store, init_execution_store
    from pathlib import Path
    from app.config import settings

    # Get or init execution store
    exec_store = get_execution_store()
    if not exec_store:
        data_dir = Path(settings.data_dir)
        exec_store = init_execution_store(data_dir)

    if flush:
        exec_store.flush()

    # Fetch from IBKR
    new_count = 0
    fetch_count = 0
    try:
        provider = ProviderRegistry.get(Broker.IBKR)
        if hasattr(provider, "get_executions"):
            raw = await provider.get_executions()
            fetch_count = len(raw)
            if raw:
                new_count = exec_store.merge_executions(raw)
    except Exception:
        pass

    groups = exec_store.get_grouped_by_order()
    if symbol:
        groups = [g for g in groups if g["symbol"] == symbol.upper()]

    return {
        "fetched": fetch_count,
        "new": new_count,
        "total_cached": exec_store.count,
        "orders": groups,
    }


# ── Profit Targets ────────────────────────────────────────────────────────────


@router.get("/profit-targets")
async def get_profit_targets(
    _user: Annotated[TokenData, Security(require_auth, scopes=["account:read"])],
) -> dict:
    """List all active profit targets."""
    from app.services.profit_target_service import get_profit_target_service
    svc = get_profit_target_service()
    if not svc:
        return {"positions": {}}
    return {"positions": svc.get_targets()}


@router.post("/profit-targets")
async def set_profit_target(
    _user: Annotated[TokenData, Security(require_auth, scopes=["trades:write"])],
    position_id: str = "",
    entry_credit: float = 0,
    profit_target_pct: float = 50,
    symbol: str = "",
    short_strike: float = 0,
    long_strike: float = 0,
    quantity: int = 1,
) -> dict:
    """Set a profit target for a position."""
    from app.services.profit_target_service import get_profit_target_service
    svc = get_profit_target_service()
    if not svc:
        raise HTTPException(status_code=503, detail="Profit target service not initialized")
    if not position_id:
        raise HTTPException(status_code=400, detail="position_id required")
    svc.set_target(position_id, entry_credit, profit_target_pct, symbol, short_strike, long_strike, quantity)
    return {"status": "ok", "position_id": position_id, "profit_target_pct": profit_target_pct}


@router.put("/profit-targets/{position_id}")
async def update_profit_target(
    position_id: str,
    _user: Annotated[TokenData, Security(require_auth, scopes=["trades:write"])],
    profit_target_pct: float = 50,
) -> dict:
    """Update a position's profit target percentage. Set to 0 to remove."""
    from app.services.profit_target_service import get_profit_target_service
    svc = get_profit_target_service()
    if not svc:
        raise HTTPException(status_code=503, detail="Service not initialized")
    if not svc.update_target(position_id, profit_target_pct):
        raise HTTPException(status_code=404, detail="Position not found in profit targets")
    return {"status": "ok", "position_id": position_id, "profit_target_pct": profit_target_pct}


@router.delete("/profit-targets/{position_id}")
async def delete_profit_target(
    position_id: str,
    _user: Annotated[TokenData, Security(require_auth, scopes=["trades:write"])],
) -> dict:
    """Remove a profit target."""
    from app.services.profit_target_service import get_profit_target_service
    svc = get_profit_target_service()
    if not svc:
        raise HTTPException(status_code=503, detail="Service not initialized")
    if not svc.remove_target(position_id):
        raise HTTPException(status_code=404, detail="Position not found")
    return {"status": "removed", "position_id": position_id}
