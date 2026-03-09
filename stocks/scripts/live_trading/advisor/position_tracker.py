"""Lightweight JSON-backed position tracker for the live advisor.

Tracks open and closed positions, daily budget usage, and trade rate limiting.
Persists to data/live_advisor_v2/positions.json for cross-session continuity.
"""

import json
import logging
import uuid
from dataclasses import asdict, dataclass, field
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class TrackedPosition:
    """A position being tracked by the advisor."""
    pos_id: str
    tier_label: str
    priority: int
    direction: str          # "put" or "call"
    short_strike: float
    long_strike: float
    credit: float           # per-share
    num_contracts: int
    total_credit: float     # credit * contracts * 100
    max_loss: float
    dte: int
    entry_time: str         # ISO format
    entry_price: float      # underlying price at entry
    status: str = "open"    # "open" | "closed"
    close_reason: str = ""
    close_time: str = ""
    exit_price: float = 0.0
    realized_pnl: float = 0.0

    def width(self) -> float:
        return abs(self.short_strike - self.long_strike)


@dataclass
class EODState:
    """Persisted end-of-day direction state for pursuit_eod tiers."""
    direction: Optional[str] = None  # "put" | "call" | None
    skip_next_day: bool = True
    computed_date: str = ""          # ISO date when this was computed


class PositionTracker:
    """JSON-backed position and budget tracker."""

    def __init__(self, data_dir: Optional[Path] = None, profile_name: Optional[str] = None):
        if data_dir is None:
            base = Path(__file__).resolve().parents[3] / "data" / "live_advisor"
            if profile_name:
                data_dir = base / profile_name
            else:
                data_dir = base / "default"
        self._data_dir = data_dir
        self._data_dir.mkdir(parents=True, exist_ok=True)
        self._positions_file = self._data_dir / "positions.json"
        self._positions: List[TrackedPosition] = []
        self._eod_state: EODState = EODState()
        self._trade_timestamps: List[datetime] = []  # for rate limiting
        self._load()

    def _load(self) -> None:
        """Load positions and EOD state from JSON."""
        if not self._positions_file.exists():
            return
        try:
            data = json.loads(self._positions_file.read_text())
            for p in data.get("positions", []):
                self._positions.append(TrackedPosition(**p))
            eod = data.get("eod_state", {})
            if eod:
                self._eod_state = EODState(**eod)
            for ts_str in data.get("trade_timestamps", []):
                self._trade_timestamps.append(datetime.fromisoformat(ts_str))
        except Exception as e:
            logger.warning(f"Error loading positions: {e}")

    def _save(self) -> None:
        """Persist positions and EOD state to JSON."""
        data = {
            "positions": [asdict(p) for p in self._positions],
            "eod_state": asdict(self._eod_state),
            "trade_timestamps": [ts.isoformat() for ts in self._trade_timestamps],
        }
        self._positions_file.write_text(json.dumps(data, indent=2))

    def add_position(
        self,
        tier_label: str,
        priority: int,
        direction: str,
        short_strike: float,
        long_strike: float,
        credit: float,
        num_contracts: int,
        dte: int,
        entry_price: float,
    ) -> TrackedPosition:
        """Record a new confirmed entry."""
        pos_id = uuid.uuid4().hex[:8]
        total_credit = credit * num_contracts * 100
        max_loss = abs(short_strike - long_strike) * num_contracts * 100 - total_credit
        pos = TrackedPosition(
            pos_id=pos_id,
            tier_label=tier_label,
            priority=priority,
            direction=direction,
            short_strike=short_strike,
            long_strike=long_strike,
            credit=credit,
            num_contracts=num_contracts,
            total_credit=total_credit,
            max_loss=max_loss,
            dte=dte,
            entry_time=datetime.now(timezone.utc).isoformat(),
            entry_price=entry_price,
        )
        self._positions.append(pos)
        self._trade_timestamps.append(datetime.now(timezone.utc))
        self._save()
        logger.info(f"Added position {pos_id}: {tier_label} {direction} "
                     f"{short_strike}/{long_strike} x{num_contracts}")
        return pos

    def close_position(
        self,
        pos_id: str,
        reason: str = "manual",
        exit_price: float = 0.0,
    ) -> Optional[TrackedPosition]:
        """Close an open position."""
        for pos in self._positions:
            if pos.pos_id == pos_id and pos.status == "open":
                pos.status = "closed"
                pos.close_reason = reason
                pos.close_time = datetime.now(timezone.utc).isoformat()
                pos.exit_price = exit_price
                if exit_price > 0:
                    pos.realized_pnl = self._compute_pnl(pos, exit_price)
                self._save()
                logger.info(f"Closed position {pos_id}: {reason}")
                return pos
        return None

    def get_open_positions(self) -> List[TrackedPosition]:
        """Return all open positions."""
        return [p for p in self._positions if p.status == "open"]

    def get_today_positions(self) -> List[TrackedPosition]:
        """Return positions opened today (UTC date)."""
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        return [p for p in self._positions if p.entry_time[:10] == today]

    def get_daily_budget_used(self) -> float:
        """Total max_loss committed today."""
        return sum(p.max_loss for p in self.get_today_positions() if p.status == "open")

    def get_daily_trade_count(self) -> int:
        """Number of trades opened today."""
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        return sum(1 for p in self._positions if p.entry_time[:10] == today)

    def check_rate_limit(self, window_minutes: int = 10, max_trades: int = 2) -> int:
        """Return how many more trades are allowed in the rolling window."""
        cutoff = datetime.now(timezone.utc) - timedelta(minutes=window_minutes)
        recent = sum(1 for ts in self._trade_timestamps if ts >= cutoff)
        return max(0, max_trades - recent)

    def get_eod_state(self) -> EODState:
        return self._eod_state

    def set_eod_state(
        self, direction: Optional[str], skip_next_day: bool, computed_date: date
    ) -> None:
        self._eod_state = EODState(
            direction=direction,
            skip_next_day=skip_next_day,
            computed_date=computed_date.isoformat(),
        )
        self._save()

    def get_daily_summary(self) -> Dict[str, Any]:
        """Summary stats for today."""
        today_positions = self.get_today_positions()
        open_positions = self.get_open_positions()
        closed_today = [p for p in today_positions if p.status == "closed"]
        return {
            "trades_today": len(today_positions),
            "open_count": len(open_positions),
            "closed_today": len(closed_today),
            "budget_used": self.get_daily_budget_used(),
            "realized_pnl": sum(p.realized_pnl for p in closed_today),
            "total_credit": sum(p.total_credit for p in today_positions),
        }

    @staticmethod
    def _compute_pnl(pos: TrackedPosition, exit_price: float) -> float:
        """Compute realized P&L given underlying exit price."""
        width = pos.width()
        if pos.direction == "put":
            if exit_price >= pos.short_strike:
                pnl_per_share = pos.credit  # full win
            elif exit_price <= pos.long_strike:
                pnl_per_share = pos.credit - width  # max loss
            else:
                intrinsic = pos.short_strike - exit_price
                pnl_per_share = pos.credit - intrinsic
        else:  # call
            if exit_price <= pos.short_strike:
                pnl_per_share = pos.credit  # full win
            elif exit_price >= pos.long_strike:
                pnl_per_share = pos.credit - width  # max loss
            else:
                intrinsic = exit_price - pos.short_strike
                pnl_per_share = pos.credit - intrinsic
        return pnl_per_share * pos.num_contracts * 100
