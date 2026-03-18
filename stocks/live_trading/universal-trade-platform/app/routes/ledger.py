"""Ledger query and replay endpoints."""

from __future__ import annotations

from typing import Annotated, Optional

from fastapi import APIRouter, HTTPException, Security

from app.auth import TokenData, require_auth
from app.models import Broker, LedgerEntry, LedgerEventType, LedgerQuery, PositionSource
from app.services.ledger import get_ledger

router = APIRouter(prefix="/ledger", tags=["ledger"])


@router.get("/entries", response_model=list[LedgerEntry])
async def query_entries(
    _user: Annotated[TokenData, Security(require_auth, scopes=["trades:read"])],
    broker: Optional[Broker] = None,
    event_type: Optional[LedgerEventType] = None,
    source: Optional[PositionSource] = None,
    order_id: Optional[str] = None,
    limit: int = 100,
    offset: int = 0,
) -> list[LedgerEntry]:
    """Query ledger entries with optional filters."""
    ledger = get_ledger()
    if not ledger:
        raise HTTPException(status_code=503, detail="Ledger not initialized")

    q = LedgerQuery(
        broker=broker,
        event_type=event_type,
        source=source,
        order_id=order_id,
        limit=limit,
        offset=offset,
    )
    return await ledger.query(q)


@router.get("/entries/recent", response_model=list[LedgerEntry])
async def recent_entries(
    _user: Annotated[TokenData, Security(require_auth, scopes=["trades:read"])],
    n: int = 50,
) -> list[LedgerEntry]:
    """Return the last N ledger entries."""
    ledger = get_ledger()
    if not ledger:
        raise HTTPException(status_code=503, detail="Ledger not initialized")
    return await ledger.get_recent(n)


@router.post("/snapshot")
async def create_snapshot(
    _user: Annotated[TokenData, Security(require_auth, scopes=["account:read"])],
) -> dict:
    """Trigger a manual ledger snapshot."""
    ledger = get_ledger()
    if not ledger:
        raise HTTPException(status_code=503, detail="Ledger not initialized")

    from app.services.position_store import get_position_store

    store = get_position_store()
    positions = store.get_open_positions() if store else []
    account = store.get_account_summary() if store else {}
    filename = await ledger.save_snapshot(positions, account)
    return {"snapshot": filename}


@router.get("/snapshots")
async def list_snapshots(
    _user: Annotated[TokenData, Security(require_auth, scopes=["account:read"])],
) -> list[str]:
    """List available snapshots."""
    ledger = get_ledger()
    if not ledger:
        raise HTTPException(status_code=503, detail="Ledger not initialized")
    return ledger.list_snapshots()


@router.get("/replay")
async def replay(
    _user: Annotated[TokenData, Security(require_auth, scopes=["account:read"])],
    from_snapshot: Optional[str] = None,
) -> dict:
    """Replay ledger from a snapshot for state reconstruction."""
    ledger = get_ledger()
    if not ledger:
        raise HTTPException(status_code=503, detail="Ledger not initialized")

    state, entries = await ledger.replay(from_snapshot)
    return {
        "state": state,
        "entries_count": len(entries),
        "entries": [e.model_dump() for e in entries[:100]],
    }
