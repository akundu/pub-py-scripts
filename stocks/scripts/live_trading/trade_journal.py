"""JSONL append-only trade journal for logging all decisions.

Every signal generation, entry, exit, skip, and market event is logged with
timestamp, event type, ticker, details, and reasoning. Supports filtering
by date range and event type for review.
"""

import json
import logging
from dataclasses import asdict, dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


def _json_serial(obj):
    """JSON serializer for datetime/date objects."""
    if isinstance(obj, (datetime, date)):
        return obj.isoformat()
    raise TypeError(f"Type {type(obj)} not serializable")


@dataclass
class JournalEntry:
    """A single journal entry representing a trading event."""
    timestamp: datetime
    event_type: str       # signal_generated, signal_rejected, position_opened,
                          # position_closed, exit_triggered, market_open, market_close
    ticker: str
    details: Dict[str, Any] = field(default_factory=dict)
    reasoning: str = ""

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["timestamp"] = self.timestamp.isoformat()
        return d

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "JournalEntry":
        data = dict(data)
        ts = data.get("timestamp", "")
        if isinstance(ts, str):
            data["timestamp"] = datetime.fromisoformat(ts)
        return cls(**data)


class TradeJournal:
    """Append-only JSONL trade journal with filtering support."""

    def __init__(self, journal_path: str = "data/live_trading/journal.jsonl"):
        self._path = Path(journal_path)
        self._path.parent.mkdir(parents=True, exist_ok=True)

    def log(self, entry: JournalEntry) -> None:
        """Append a journal entry."""
        with open(self._path, "a") as f:
            f.write(json.dumps(entry.to_dict(), default=_json_serial) + "\n")

    def log_entry(self, position_id: str, position: Dict, signal: Dict) -> None:
        """Log a position entry."""
        self.log(JournalEntry(
            timestamp=datetime.now(),
            event_type="position_opened",
            ticker=signal.get("ticker", position.get("option_type", "")),
            details={
                "position_id": position_id,
                "option_type": position.get("option_type"),
                "short_strike": position.get("short_strike"),
                "long_strike": position.get("long_strike"),
                "num_contracts": position.get("num_contracts"),
                "initial_credit": position.get("initial_credit"),
                "dte": position.get("dte"),
                "signal": {k: v for k, v in signal.items()
                           if k not in ("timestamp",) and not isinstance(v, (datetime, date))},
            },
        ))

    def log_exit(self, position_id: str, reason: str, pnl: float, ticker: str = "") -> None:
        """Log a position exit."""
        self.log(JournalEntry(
            timestamp=datetime.now(),
            event_type="position_closed",
            ticker=ticker,
            details={
                "position_id": position_id,
                "exit_reason": reason,
                "pnl": pnl,
            },
        ))

    def log_skip(self, signal: Dict, reason: str, ticker: str = "") -> None:
        """Log a skipped signal."""
        self.log(JournalEntry(
            timestamp=datetime.now(),
            event_type="signal_rejected",
            ticker=ticker,
            details={
                "option_type": signal.get("option_type"),
                "target_strike": signal.get("percentile_target_strike"),
            },
            reasoning=reason,
        ))

    def log_event(self, event_type: str, ticker: str, details: Dict = None, reasoning: str = "") -> None:
        """Log a generic event (market_open, market_close, etc.)."""
        self.log(JournalEntry(
            timestamp=datetime.now(),
            event_type=event_type,
            ticker=ticker,
            details=details or {},
            reasoning=reasoning,
        ))

    def get_entries(
        self,
        start_date: Optional[date] = None,
        end_date: Optional[date] = None,
        event_type: Optional[str] = None,
    ) -> List[JournalEntry]:
        """Read and filter journal entries."""
        if not self._path.exists():
            return []

        entries = []
        with open(self._path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    entry = JournalEntry.from_dict(data)
                except (json.JSONDecodeError, TypeError, KeyError):
                    continue

                # Filter by event type
                if event_type and entry.event_type != event_type:
                    continue

                # Filter by date
                entry_date = entry.timestamp.date()
                if start_date and entry_date < start_date:
                    continue
                if end_date and entry_date > end_date:
                    continue

                entries.append(entry)

        return entries

    def get_recent(self, n: int = 20) -> List[JournalEntry]:
        """Get the last N journal entries."""
        if not self._path.exists():
            return []

        lines = []
        with open(self._path) as f:
            for line in f:
                lines.append(line.strip())

        entries = []
        for line in lines[-n:]:
            if not line:
                continue
            try:
                data = json.loads(line)
                entries.append(JournalEntry.from_dict(data))
            except (json.JSONDecodeError, TypeError, KeyError):
                continue
        return entries
