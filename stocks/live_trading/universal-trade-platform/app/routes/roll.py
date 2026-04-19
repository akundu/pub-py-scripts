"""Roll management endpoints — suggestions, execution, configuration."""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Header, HTTPException, Security

from app.auth import TokenData, require_auth
from app.services.roll_service import get_roll_service

router = APIRouter(prefix="/roll", tags=["roll"])


def _get_service():
    svc = get_roll_service()
    if not svc:
        raise HTTPException(status_code=503, detail="Roll service not initialized")
    return svc


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
    _user: Annotated[TokenData, Security(require_auth, scopes=["trade:write"])],
    x_dry_run: Annotated[str | None, Header()] = None,
) -> dict:
    """Execute a roll suggestion.

    For FORWARD rolls: closes current position, then opens new spread.
    For MIRROR rolls: opens new spread only (keeps original).

    Set X-Dry-Run: true header to preview without executing.
    """
    svc = _get_service()

    dry_run = (x_dry_run or "").lower() == "true"
    if dry_run:
        suggestion = svc.get_suggestion(suggestion_id)
        if not suggestion:
            raise HTTPException(status_code=404, detail="Suggestion not found")
        return {"status": "dry_run", "suggestion": suggestion.to_dict()}

    result = await svc.execute_roll(suggestion_id)
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
    raise HTTPException(status_code=404, detail=f"Suggestion {suggestion_id} not found or not pending")


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
    """Update roll service configuration. Accepts partial updates."""
    svc = _get_service()
    new_config = svc.update_config(body)
    return new_config.to_dict()
