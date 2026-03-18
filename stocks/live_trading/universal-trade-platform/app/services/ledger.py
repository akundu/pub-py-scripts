"""Append-only JSONL transaction ledger with snapshot support."""

from __future__ import annotations

import asyncio
import json
import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import Optional

from app.models import (
    Broker,
    LedgerEntry,
    LedgerEventType,
    LedgerQuery,
    PositionSource,
)

logger = logging.getLogger(__name__)


class TransactionLedger:
    """Append-only JSONL ledger for all trade events."""

    def __init__(self, ledger_dir: Path) -> None:
        self._dir = ledger_dir
        self._dir.mkdir(parents=True, exist_ok=True)
        self._file = self._dir / "ledger.jsonl"
        self._snapshot_dir = self._dir / "snapshots"
        self._snapshot_dir.mkdir(parents=True, exist_ok=True)
        self._lock = asyncio.Lock()
        self._sequence = self._recover_sequence()

    def _recover_sequence(self) -> int:
        """Recover sequence counter from last line of ledger file."""
        if not self._file.exists():
            return 0
        try:
            last_line = ""
            with open(self._file, "r") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        last_line = line
            if last_line:
                data = json.loads(last_line)
                return data.get("sequence_number", 0) + 1
        except Exception as e:
            logger.debug("Failed to recover ledger sequence: %s", e)
        return 0

    async def append(self, entry: LedgerEntry) -> LedgerEntry:
        """Append a ledger entry to the JSONL file."""
        async with self._lock:
            entry.sequence_number = self._sequence
            self._sequence += 1
            line = entry.model_dump_json() + "\n"
            with open(self._file, "a") as f:
                f.write(line)
        return entry

    async def log_order_submitted(
        self,
        broker: Broker,
        order_id: str,
        source: PositionSource,
        dry_run: bool = False,
        data: Optional[dict] = None,
    ) -> LedgerEntry:
        entry = LedgerEntry(
            event_type=LedgerEventType.ORDER_SUBMITTED,
            broker=broker,
            order_id=order_id,
            source=source,
            dry_run=dry_run,
            data=data or {},
        )
        return await self.append(entry)

    async def log_status_change(
        self,
        broker: Broker,
        order_id: str,
        status: str,
        data: Optional[dict] = None,
    ) -> LedgerEntry:
        entry = LedgerEntry(
            event_type=LedgerEventType.ORDER_STATUS_CHANGE,
            broker=broker,
            order_id=order_id,
            data={"status": status, **(data or {})},
        )
        return await self.append(entry)

    async def log_position_opened(
        self,
        broker: Broker,
        position_id: str,
        source: PositionSource,
        dry_run: bool = False,
        data: Optional[dict] = None,
    ) -> LedgerEntry:
        entry = LedgerEntry(
            event_type=LedgerEventType.POSITION_OPENED,
            broker=broker,
            position_id=position_id,
            source=source,
            dry_run=dry_run,
            data=data or {},
        )
        return await self.append(entry)

    async def log_position_closed(
        self,
        broker: Broker,
        position_id: str,
        source: PositionSource,
        data: Optional[dict] = None,
    ) -> LedgerEntry:
        entry = LedgerEntry(
            event_type=LedgerEventType.POSITION_CLOSED,
            broker=broker,
            position_id=position_id,
            source=source,
            data=data or {},
        )
        return await self.append(entry)

    async def save_snapshot(self, positions: list[dict], account_state: dict) -> str:
        """Save a point-in-time snapshot for state reconstruction."""
        snapshot = {
            "sequence_number": self._sequence,
            "timestamp": datetime.now(UTC).isoformat(),
            "positions": positions,
            "account_state": account_state,
        }
        filename = f"snapshot_{self._sequence}.json"
        path = self._snapshot_dir / filename
        with open(path, "w") as f:
            json.dump(snapshot, f, indent=2, default=str)

        await self.append(LedgerEntry(
            event_type=LedgerEventType.SNAPSHOT,
            data={"snapshot_file": filename},
        ))
        return filename

    async def query(self, q: LedgerQuery) -> list[LedgerEntry]:
        """Query ledger entries with filters and pagination."""
        if not self._file.exists():
            return []

        results: list[LedgerEntry] = []
        skipped = 0

        with open(self._file, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = LedgerEntry.model_validate_json(line)
                except Exception:
                    continue

                if q.start_date and entry.timestamp < q.start_date:
                    continue
                if q.end_date and entry.timestamp > q.end_date:
                    continue
                if q.broker and entry.broker != q.broker:
                    continue
                if q.event_type and entry.event_type != q.event_type:
                    continue
                if q.source and entry.source != q.source:
                    continue
                if q.order_id and entry.order_id != q.order_id:
                    continue

                if skipped < q.offset:
                    skipped += 1
                    continue

                results.append(entry)
                if len(results) >= q.limit:
                    break

        return results

    async def get_recent(self, n: int = 50) -> list[LedgerEntry]:
        """Return the last N ledger entries."""
        if not self._file.exists():
            return []

        all_lines: list[str] = []
        with open(self._file, "r") as f:
            for line in f:
                line = line.strip()
                if line:
                    all_lines.append(line)

        entries: list[LedgerEntry] = []
        for line in all_lines[-n:]:
            try:
                entries.append(LedgerEntry.model_validate_json(line))
            except Exception:
                continue
        return entries

    async def replay(self, from_snapshot: Optional[str] = None) -> tuple[dict, list[LedgerEntry]]:
        """Replay ledger from a snapshot (or beginning) for state reconstruction."""
        state: dict = {"positions": [], "account_state": {}}
        start_seq = 0

        if from_snapshot:
            snap_path = self._snapshot_dir / from_snapshot
            if snap_path.exists():
                with open(snap_path, "r") as f:
                    state = json.load(f)
                start_seq = state.get("sequence_number", 0)

        entries: list[LedgerEntry] = []
        if self._file.exists():
            with open(self._file, "r") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        entry = LedgerEntry.model_validate_json(line)
                        if entry.sequence_number >= start_seq:
                            entries.append(entry)
                    except Exception:
                        continue

        return state, entries

    def list_snapshots(self) -> list[str]:
        """List available snapshot filenames."""
        if not self._snapshot_dir.exists():
            return []
        return sorted(f.name for f in self._snapshot_dir.glob("snapshot_*.json"))


# ── Module-level accessor ─────────────────────────────────────────────────────

_ledger: Optional[TransactionLedger] = None


def get_ledger() -> Optional[TransactionLedger]:
    return _ledger


def init_ledger(data_dir: Path) -> TransactionLedger:
    global _ledger
    ledger_dir = Path(data_dir) / "ledger"
    _ledger = TransactionLedger(ledger_dir)
    logger.info("Transaction ledger initialized at %s", ledger_dir)
    return _ledger


def reset_ledger() -> None:
    """Reset the module-level ledger (for tests)."""
    global _ledger
    _ledger = None
