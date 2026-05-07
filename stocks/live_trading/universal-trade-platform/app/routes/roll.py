"""Roll management endpoints — suggestions, execution, configuration."""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Header, HTTPException, Security

from app.auth import TokenData, require_auth
from app.services.roll_service import get_roll_service

router = APIRouter(prefix="/roll", tags=["roll"])

# Keys accepted as per-execute overrides in request body
_OVERRIDE_KEYS = {"dte", "otm_pct", "width", "quantity", "close_quantity"}


def _get_service():
    svc = get_roll_service()
    if not svc:
        raise HTTPException(status_code=503, detail="Roll service not initialized")
    return svc


def _extract_overrides(body: dict) -> dict | None:
    """Pull override keys from request body; return None if empty."""
    overrides = {k: v for k, v in body.items() if k in _OVERRIDE_KEYS and v is not None}
    return overrides if overrides else None


@router.get("/suggestions")
async def get_suggestions(
    _user: Annotated[TokenData, Security(require_auth, scopes=["account:read"])],
) -> list[dict]:
    """Return current pending roll suggestions."""
    svc = _get_service()
    return svc.get_suggestions()


@router.post("/execute/{suggestion_id}")
async def execute_suggestion(
    suggestion_id: str,
    body: dict = {},
    _user: Annotated[TokenData, Security(require_auth, scopes=["trade:write"])] = None,
    x_dry_run: Annotated[str | None, Header()] = None,
) -> dict:
    """Execute a roll suggestion with optional per-execute overrides.

    For FORWARD rolls: closes current position, then opens new spread.
    For MIRROR rolls: opens new spread only (keeps original).

    Set X-Dry-Run: true header to preview without executing.

    Optional body keys (overrides for this execution only):
      dte (int)             — target DTE for new position
      otm_pct (float)       — short strike OTM% from current price
      width (float)         — spread width for new position
      quantity (int)        — contracts to open in new position
      close_quantity (int)  — contracts to close from original (partial roll)
    """
    svc = _get_service()
    overrides = _extract_overrides(body)
    dry_run = (x_dry_run or "").lower() == "true"

    if dry_run:
        suggestion = svc.get_suggestion(suggestion_id)
        if not suggestion:
            raise HTTPException(status_code=404, detail="Suggestion not found")
        preview = suggestion
        if overrides:
            preview = svc._apply_overrides(suggestion, overrides)
        return {"status": "dry_run", "suggestion": preview.to_dict()}

    result = await svc.execute_roll(suggestion_id, overrides=overrides)
    if "error" in result:
        if "not found" in result["error"].lower():
            raise HTTPException(status_code=404, detail=result["error"])
        raise HTTPException(status_code=400, detail=result["error"])
    return result


@router.post("/dismiss/{suggestion_id}")
async def dismiss_suggestion(
    suggestion_id: str,
    _user: Annotated[TokenData, Security(require_auth, scopes=["trade:write"])],
) -> dict:
    """Dismiss (reject) a roll suggestion."""
    svc = _get_service()
    if svc.dismiss_suggestion(suggestion_id):
        return {"status": "dismissed", "suggestion_id": suggestion_id}
    raise HTTPException(
        status_code=404,
        detail=f"Suggestion {suggestion_id} not found or not pending",
    )


@router.post("/forward/{position_id}")
async def manual_forward_roll(
    position_id: str,
    body: dict = {},
    _user: Annotated[TokenData, Security(require_auth, scopes=["trade:write"])] = None,
) -> dict:
    """Force-build (and optionally execute) a forward roll for any open position.

    Works regardless of current breach severity — useful for manual roll commands.

    Body keys:
      confirm (bool)        — if true, execute immediately; default false (preview only)
      dte (int)             — target DTE
      otm_pct (float)       — short strike OTM% from current price
      width (float)         — spread width
      quantity (int)        — contracts to open
      close_quantity (int)  — contracts to close from original
    """
    svc = _get_service()
    overrides = _extract_overrides(body)
    confirm = bool(body.get("confirm", False))

    suggestion = await svc.build_manual_forward(position_id, overrides=overrides)
    if not suggestion:
        raise HTTPException(
            status_code=404,
            detail=f"Position {position_id} not found or could not build forward suggestion",
        )

    if not confirm:
        return {"status": "preview", "suggestion": suggestion.to_dict()}

    result = await svc.execute_roll(suggestion.suggestion_id)
    if "error" in result:
        raise HTTPException(status_code=400, detail=result["error"])
    return result


@router.post("/mirror/{position_id}")
async def manual_mirror_roll(
    position_id: str,
    body: dict = {},
    _user: Annotated[TokenData, Security(require_auth, scopes=["trade:write"])] = None,
) -> dict:
    """Force-build (and optionally execute) a mirror roll for any open position.

    Body keys:
      confirm (bool)        — if true, execute immediately; default false (preview only)
      otm_pct (float)       — short strike OTM% from current price (default: ATM)
      width (float)         — spread width
      quantity (int)        — contracts to open
    """
    svc = _get_service()
    overrides = _extract_overrides(body)
    confirm = bool(body.get("confirm", False))

    suggestion = await svc.build_manual_mirror(position_id, overrides=overrides)
    if not suggestion:
        raise HTTPException(
            status_code=404,
            detail=f"Position {position_id} not found or could not build mirror suggestion",
        )

    if not confirm:
        return {"status": "preview", "suggestion": suggestion.to_dict()}

    result = await svc.execute_roll(suggestion.suggestion_id)
    if "error" in result:
        raise HTTPException(status_code=400, detail=result["error"])
    return result


@router.get("/config")
async def get_config(
    _user: Annotated[TokenData, Security(require_auth, scopes=["account:read"])],
) -> dict:
    """Return current roll service configuration."""
    svc = _get_service()
    return svc.config.to_dict()


@router.post("/config")
async def update_config(
    body: dict,
    _user: Annotated[TokenData, Security(require_auth, scopes=["trade:write"])],
) -> dict:
    """Update roll service configuration. Accepts partial updates.

    Updatable keys include all RollConfig fields:
      forward_default_otm_pct, forward_default_width, forward_default_quantity,
      forward_partial_close_pct, notify_on_severity, notify_channel,
      notify_cooldown_minutes, forward_min_dte, mirror_trigger_severity, etc.
    """
    svc = _get_service()
    new_config = svc.update_config(body)
    return new_config.to_dict()
