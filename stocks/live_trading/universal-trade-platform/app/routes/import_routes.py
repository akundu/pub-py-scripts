"""CSV import endpoints."""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Annotated

from fastapi import APIRouter, File, HTTPException, Query, Security, UploadFile

from app.auth import TokenData, require_auth
from app.config import settings
from app.models import Broker, CSVImportResult
from app.services.csv_importer import CSVTransactionImporter
from app.services.ledger import get_ledger
from app.services.position_store import get_position_store

router = APIRouter(prefix="/import", tags=["import"])


def _get_importer() -> CSVTransactionImporter:
    store = get_position_store()
    ledger = get_ledger()
    if not store or not ledger:
        raise HTTPException(status_code=503, detail="Services not initialized")
    return CSVTransactionImporter(store, ledger)


@router.post("/csv", response_model=CSVImportResult)
async def import_csv(
    _user: Annotated[TokenData, Security(require_auth, scopes=["trades:write"])],
    file: UploadFile = File(...),
    broker: Broker = Query(..., description="Source broker for the CSV"),
) -> CSVImportResult:
    """Upload and import a CSV transaction file."""
    importer = _get_importer()

    content = await file.read()
    text = content.decode("utf-8-sig")

    # Save file for audit trail
    import_dir = Path(settings.csv_import_dir) / broker.value
    import_dir.mkdir(parents=True, exist_ok=True)
    save_path = import_dir / (file.filename or "upload.csv")
    with open(save_path, "w") as f:
        f.write(text)

    return await importer.import_content(
        content=text,
        broker=broker,
        filename=file.filename or "upload.csv",
    )


@router.get("/formats")
async def supported_formats(
    _user: Annotated[TokenData, Security(require_auth, scopes=["trades:read"])],
) -> dict:
    """Return supported CSV formats and example column headers."""
    return CSVTransactionImporter.supported_formats()


@router.post("/preview")
async def preview_csv(
    _user: Annotated[TokenData, Security(require_auth, scopes=["trades:read"])],
    file: UploadFile = File(...),
    broker: Broker = Query(..., description="Source broker for the CSV"),
) -> list[dict]:
    """Preview first 10 rows of a CSV without importing."""
    importer = _get_importer()
    content = await file.read()
    text = content.decode("utf-8-sig")
    return importer.preview(text, broker)
