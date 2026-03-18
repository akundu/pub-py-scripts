"""Account endpoints — positions, sync, and expiration."""

from __future__ import annotations

from datetime import UTC, date, datetime, timedelta
from typing import Annotated, Optional

from fastapi import APIRouter, HTTPException, Security

from app.auth import TokenData, require_auth
from app.core.provider import ProviderRegistry
from app.models import AggregatedPositions, Broker, ReconciliationReport, SyncResult

router = APIRouter(prefix="/account", tags=["account"])


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
