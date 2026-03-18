"""Playbook endpoints — execute and validate YAML instruction files."""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, HTTPException, Security, UploadFile, File, Header

from app.auth import TokenData, require_auth
from app.models import PlaybookResult
from app.services.playbook_service import PlaybookService, PlaybookValidationError

router = APIRouter(prefix="/playbook", tags=["playbook"])


@router.post("/execute", response_model=PlaybookResult)
async def execute_playbook(
    file: UploadFile = File(...),
    _user: Annotated[TokenData, Security(require_auth, scopes=["trades:write"])] = None,
    x_dry_run: Annotated[str | None, Header()] = None,
) -> PlaybookResult:
    """Execute a playbook from uploaded YAML file.

    Pass `X-Dry-Run: true` header to simulate without sending to the broker.
    """
    content = await file.read()
    try:
        yaml_str = content.decode("utf-8")
    except UnicodeDecodeError:
        raise HTTPException(status_code=400, detail="Invalid UTF-8 in uploaded file")

    service = PlaybookService()
    try:
        playbook = service.load(yaml_str)
    except PlaybookValidationError as e:
        raise HTTPException(status_code=422, detail=str(e))

    dry_run = (x_dry_run or "").lower() == "true"
    return await service.execute(playbook, dry_run=dry_run)


@router.post("/validate")
async def validate_playbook(
    file: UploadFile = File(...),
    _user: Annotated[TokenData, Security(require_auth, scopes=["trades:write"])] = None,
) -> list[dict]:
    """Validate a playbook without executing — check instruction structure."""
    content = await file.read()
    try:
        yaml_str = content.decode("utf-8")
    except UnicodeDecodeError:
        raise HTTPException(status_code=400, detail="Invalid UTF-8 in uploaded file")

    service = PlaybookService()
    try:
        playbook = service.load(yaml_str)
    except PlaybookValidationError as e:
        raise HTTPException(status_code=422, detail=str(e))

    return await service.validate(playbook)
