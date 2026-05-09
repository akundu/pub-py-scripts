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
