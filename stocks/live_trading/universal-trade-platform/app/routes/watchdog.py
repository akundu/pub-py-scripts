"""Watchdog REST endpoints — portfolio advisory suggestions."""

from __future__ import annotations

from typing import Annotated, Optional

from fastapi import APIRouter, Security

from app.auth import TokenData, require_auth

router = APIRouter(prefix="/watchdog", tags=["watchdog"])


def _get_svc():
    from app.services.watchdog_service import get_watchdog_service
    return get_watchdog_service()


@router.get("/suggestions")
async def list_suggestions(
    _user: Annotated[TokenData, Security(require_auth, scopes=["account:read"])],
    position_id: Optional[str] = None,
) -> list[dict]:
    """Return pending watchdog suggestions, optionally filtered by position_id."""
    svc = _get_svc()
    if not svc:
        return []
    return svc.get_suggestions(position_id=position_id)


@router.post("/dismiss/{suggestion_id}")
async def dismiss_suggestion(
    suggestion_id: str,
    _user: Annotated[TokenData, Security(require_auth, scopes=["trade:write"])],
) -> dict:
    """Dismiss a pending suggestion by ID or prefix."""
    svc = _get_svc()
    if not svc:
        return {"error": "Watchdog service not initialized"}
    ok = svc.dismiss(suggestion_id)
    if not ok:
        return {"error": f"Suggestion {suggestion_id!r} not found or not pending"}
    return {"status": "dismissed", "suggestion_id": suggestion_id}


@router.get("/config")
async def get_config(
    _user: Annotated[TokenData, Security(require_auth, scopes=["account:read"])],
) -> dict:
    """Return current watchdog configuration."""
    svc = _get_svc()
    if not svc:
        return {"error": "Watchdog service not initialized"}
    return svc.config.to_dict()


@router.post("/config")
async def update_config(
    updates: dict,
    _user: Annotated[TokenData, Security(require_auth, scopes=["trade:write"])],
) -> dict:
    """Partial update to watchdog configuration.

    Settable fields: enabled, action_mode, default_interval_seconds,
    suggestion_ttl_seconds, module_overrides.

    module_overrides example:
      {"close_advisor": {"interval_seconds": 60, "enabled": true}}
    """
    svc = _get_svc()
    if not svc:
        return {"error": "Watchdog service not initialized"}
    cfg = svc.update_config(updates)
    return cfg.to_dict()


@router.get("/status")
async def get_status(
    _user: Annotated[TokenData, Security(require_auth, scopes=["account:read"])],
) -> dict:
    """Return watchdog service status: modules, last-run times, suggestion counts."""
    svc = _get_svc()
    if not svc:
        return {"enabled": False, "error": "Watchdog service not initialized"}
    return svc.get_status()


@router.get("/close-eligible")
async def list_close_eligible(
    _user: Annotated[TokenData, Security(require_auth, scopes=["account:read"])],
    min_pct: float = 0.50,
    dte: Optional[int] = None,
    min_dollars: Optional[float] = None,
) -> list[dict]:
    """List open credit spread positions that have captured >= min_pct of premium.

    Query params:
      min_pct:      Fraction captured (0.0–1.0). Default 0.50 = 50%.
      dte:          Filter to positions expiring in exactly this many days.
      min_dollars:  Filter to positions that free up >= this much margin when closed.
    """
    svc = _get_svc()
    if not svc:
        return []
    return await svc.get_close_eligible(min_pct=min_pct, dte=dte, min_dollars=min_dollars)


@router.post("/close-eligible")
async def execute_close_eligible(
    body: dict,
    _user: Annotated[TokenData, Security(require_auth, scopes=["trade:write"])],
) -> dict:
    """Close all eligible positions that have captured >= min_pct of premium.

    Body fields (all optional):
      min_pct:      Fraction captured (default 0.50).
      dte:          Filter to one DTE.
      min_dollars:  Filter by minimum dollars freed.
      qty:          Contracts to close per position (default: full position quantity).
      execute:      bool — if false (default) returns preview only; if true, closes all eligible.
    """
    svc = _get_svc()
    if not svc:
        return {"error": "Watchdog service not initialized"}
    min_pct = float(body.get("min_pct", 0.50))
    dte = body.get("dte")
    min_dollars = body.get("min_dollars")
    execute = bool(body.get("execute", False))
    qty: Optional[int] = int(body["qty"]) if body.get("qty") is not None else None
    eligible = await svc.get_close_eligible(min_pct=min_pct, dte=dte, min_dollars=min_dollars)
    if not execute:
        return {"preview": True, "count": len(eligible), "positions": eligible}
    results = []
    for pos in eligible:
        from app.services.watchdog_service import WatchdogSuggestion
        from datetime import datetime, UTC
        import uuid
        s = WatchdogSuggestion(
            suggestion_id=str(uuid.uuid4())[:8],
            position_id=pos["position_id"],
            symbol=pos.get("symbol", ""),
            module="close_advisor",
            suggestion_type="close_profit",
            severity="warning",
            title=f"Batch close {pos['pct_captured'] * 100:.0f}%",
            description=f"Batch close: {pos['pct_captured'] * 100:.1f}% captured",
            action={"type": "close", "position_id": pos["position_id"], "close_reason": "batch_close"},
            created_at=datetime.now(UTC),
        )
        svc._suggestions[s.suggestion_id] = s
        await svc._auto_execute(s, qty=qty)
        results.append({
            "position_id": pos["position_id"],
            "symbol": pos.get("symbol"),
            "pct_captured": pos["pct_captured"],
            "qty": qty,
            "status": s.status,
        })
    return {"executed": True, "count": len(results), "results": results}


@router.post("/execute/{suggestion_id}")
async def execute_suggestion(
    suggestion_id: str,
    _user: Annotated[TokenData, Security(require_auth, scopes=["trade:write"])],
    body: dict = {},
) -> dict:
    """Execute a specific pending watchdog suggestion by ID or prefix.

    For close suggestions: submits a MARKET closing order.
    For roll suggestions: executes via RollService.
    Breach alert suggestions cannot be executed (monitor-only).

    Optional body fields:
      qty: int — number of contracts to close/roll (default: full position quantity)
    """
    svc = _get_svc()
    if not svc:
        return {"error": "Watchdog service not initialized"}
    qty: Optional[int] = body.get("qty") if body else None
    if qty is not None:
        qty = int(qty)
    return await svc.execute_suggestion(suggestion_id, qty=qty)
