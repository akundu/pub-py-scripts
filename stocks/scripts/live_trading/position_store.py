"""JSON-backed persistent position store for live/paper trading.

Stores open and closed positions in a JSON file, with daily mark-to-market
snapshots in a separate file. Compatible with StandardMetrics.compute() for
performance reporting.
"""

import json
import logging
import os
import uuid
from datetime import date, datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from scripts.backtesting.instruments.base import InstrumentPosition
from scripts.backtesting.results.metrics import StandardMetrics

logger = logging.getLogger(__name__)


def _json_serial(obj):
    """JSON serializer for objects not serializable by default."""
    if isinstance(obj, (datetime, date)):
        return obj.isoformat()
    raise TypeError(f"Type {type(obj)} not serializable")


class PositionStore:
    """JSON-file-backed position store with mark-to-market tracking.

    Persists positions across restarts. Compatible with StandardMetrics
    for performance reporting.
    """

    def __init__(self, db_path: str = "data/live_trading/positions.json"):
        self._db_path = Path(db_path)
        self._snapshot_path = self._db_path.parent / "daily_snapshots.json"
        self._positions: Dict[str, Dict] = {}  # position_id -> position dict
        self._snapshots: Dict[str, List[Dict]] = {}  # date_str -> [snapshot dicts]
        self._load()

    def _load(self) -> None:
        """Load positions from JSON file."""
        if self._db_path.exists():
            try:
                with open(self._db_path) as f:
                    self._positions = json.load(f)
            except (json.JSONDecodeError, IOError):
                logger.warning(f"Could not load positions from {self._db_path}, starting fresh")
                self._positions = {}
        if self._snapshot_path.exists():
            try:
                with open(self._snapshot_path) as f:
                    self._snapshots = json.load(f)
            except (json.JSONDecodeError, IOError):
                self._snapshots = {}

    def _save(self) -> None:
        """Persist positions to JSON file."""
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self._db_path, "w") as f:
            json.dump(self._positions, f, indent=2, default=_json_serial)

    def _save_snapshots(self) -> None:
        """Persist daily snapshots."""
        self._snapshot_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self._snapshot_path, "w") as f:
            json.dump(self._snapshots, f, indent=2, default=_json_serial)

    def add_position(
        self,
        position: InstrumentPosition,
        signal: Dict[str, Any],
        dte: int,
        expiration_date: date,
    ) -> str:
        """Add a new open position. Returns the position_id."""
        position_id = str(uuid.uuid4())[:8]

        pos_dict = {
            "position_id": position_id,
            "status": "open",
            "instrument_type": position.instrument_type,
            "option_type": position.option_type,
            "short_strike": position.short_strike,
            "long_strike": position.long_strike,
            "initial_credit": position.initial_credit,
            "max_loss": position.max_loss,
            "num_contracts": position.num_contracts,
            "entry_time": position.entry_time.isoformat() if isinstance(position.entry_time, datetime) else str(position.entry_time),
            "dte": dte,
            "expiration_date": expiration_date.isoformat() if isinstance(expiration_date, date) else str(expiration_date),
            "entry_signal": signal,
            "metadata": position.metadata,
            # Exit fields (populated on close)
            "exit_time": None,
            "exit_price": None,
            "exit_reason": None,
            "pnl": None,
            "pnl_per_contract": None,
            # Roll tracking
            "roll_count": 0,
            "roll_chain_id": None,
            # Mark-to-market
            "current_mark": None,
            "current_pnl": None,
            "last_mark_time": None,
        }

        self._positions[position_id] = pos_dict
        self._save()
        logger.info(
            f"Position {position_id}: {position.option_type} "
            f"{position.short_strike}/{position.long_strike} "
            f"x{position.num_contracts} @ {position.initial_credit:.4f} "
            f"DTE={dte}"
        )
        return position_id

    def close_position(
        self,
        position_id: str,
        exit_price: float,
        exit_time: datetime,
        exit_reason: str,
        pnl: Optional[float] = None,
        pnl_per_contract: Optional[float] = None,
    ) -> Optional[Dict]:
        """Close a position. Returns the result dict compatible with StandardMetrics."""
        pos = self._positions.get(position_id)
        if pos is None:
            logger.warning(f"Position {position_id} not found")
            return None
        if pos["status"] != "open":
            logger.warning(f"Position {position_id} already {pos['status']}")
            return None

        pos["status"] = "closed"
        pos["exit_time"] = exit_time.isoformat() if isinstance(exit_time, datetime) else str(exit_time)
        pos["exit_price"] = exit_price
        pos["exit_reason"] = exit_reason
        pos["pnl"] = pnl
        pos["pnl_per_contract"] = pnl_per_contract
        self._save()

        logger.info(
            f"Closed {position_id}: {exit_reason} @ {exit_price:.2f}, "
            f"P&L=${pnl:.2f}" if pnl is not None else f"Closed {position_id}: {exit_reason}"
        )

        return self._to_result_dict(pos)

    def update_mark_to_market(
        self, position_id: str, current_price: float, timestamp: datetime
    ) -> None:
        """Update mark-to-market for an open position."""
        pos = self._positions.get(position_id)
        if pos is None or pos["status"] != "open":
            return

        pos["current_mark"] = current_price
        pos["last_mark_time"] = timestamp.isoformat() if isinstance(timestamp, datetime) else str(timestamp)

        # Estimate current P&L using credit spread logic
        from scripts.backtesting.instruments.pnl import calculate_spread_pnl
        pnl_per_share = calculate_spread_pnl(
            initial_credit=pos["initial_credit"],
            short_strike=pos["short_strike"],
            long_strike=pos["long_strike"],
            underlying_price=current_price,
            option_type=pos["option_type"],
        )
        pos["current_pnl"] = round(pnl_per_share * 100 * pos["num_contracts"], 2)
        # Don't save on every tick — caller can batch
        return

    def save(self) -> None:
        """Explicit save (for batching mark-to-market updates)."""
        self._save()

    def get_open_positions(self) -> List[Dict]:
        """Return all open positions."""
        return [p for p in self._positions.values() if p["status"] == "open"]

    def get_closed_positions(self) -> List[Dict]:
        """Return all closed positions."""
        return [p for p in self._positions.values() if p["status"] == "closed"]

    def get_position(self, position_id: str) -> Optional[Dict]:
        """Return a specific position."""
        return self._positions.get(position_id)

    def get_expired_positions(self, today: date) -> List[Dict]:
        """Return open positions whose expiration_date <= today."""
        today_str = today.isoformat()
        return [
            p for p in self._positions.values()
            if p["status"] == "open" and p.get("expiration_date", "9999-12-31") <= today_str
        ]

    def increment_roll_count(self, position_id: str, chain_id: Optional[str] = None) -> None:
        """Increment roll count and optionally set chain ID."""
        pos = self._positions.get(position_id)
        if pos:
            pos["roll_count"] = pos.get("roll_count", 0) + 1
            if chain_id:
                pos["roll_chain_id"] = chain_id
            self._save()

    def get_performance(
        self,
        start_date: Optional[date] = None,
        end_date: Optional[date] = None,
    ) -> Dict[str, Any]:
        """Compute performance metrics using StandardMetrics."""
        results = self.export_results(start_date, end_date)
        return StandardMetrics.compute(results)

    def get_daily_summary(self, target_date: date) -> Dict[str, Any]:
        """Get summary for a specific date."""
        date_str = target_date.isoformat()
        open_positions = self.get_open_positions()
        closed_today = [
            p for p in self._positions.values()
            if p["status"] == "closed"
            and p.get("exit_time", "").startswith(date_str)
        ]
        opened_today = [
            p for p in self._positions.values()
            if p.get("entry_time", "").startswith(date_str)
        ]

        total_pnl = sum(p.get("pnl", 0) or 0 for p in closed_today)
        unrealized_pnl = sum(p.get("current_pnl", 0) or 0 for p in open_positions)

        return {
            "date": date_str,
            "positions_opened": len(opened_today),
            "positions_closed": len(closed_today),
            "positions_open": len(open_positions),
            "realized_pnl": round(total_pnl, 2),
            "unrealized_pnl": round(unrealized_pnl, 2),
            "total_pnl": round(total_pnl + unrealized_pnl, 2),
        }

    def save_daily_snapshot(self, target_date: date) -> None:
        """Save a mark-to-market snapshot for the given date."""
        summary = self.get_daily_summary(target_date)
        open_positions = [
            {
                "position_id": p["position_id"],
                "option_type": p["option_type"],
                "short_strike": p["short_strike"],
                "long_strike": p["long_strike"],
                "current_pnl": p.get("current_pnl"),
                "current_mark": p.get("current_mark"),
            }
            for p in self.get_open_positions()
        ]
        summary["open_positions_detail"] = open_positions
        date_str = target_date.isoformat()
        if date_str not in self._snapshots:
            self._snapshots[date_str] = []
        self._snapshots[date_str].append(summary)
        self._save_snapshots()

    def export_results(
        self,
        start_date: Optional[date] = None,
        end_date: Optional[date] = None,
    ) -> List[Dict[str, Any]]:
        """Export closed positions as result dicts compatible with StandardMetrics.compute()."""
        results = []
        for p in self._positions.values():
            if p["status"] != "closed":
                continue
            result = self._to_result_dict(p)
            if result is None:
                continue
            # Date filter
            entry_date_str = p.get("entry_time", "")[:10]
            if start_date and entry_date_str < start_date.isoformat():
                continue
            if end_date and entry_date_str > end_date.isoformat():
                continue
            results.append(result)
        return results

    def clear_all(self) -> None:
        """Clear all positions (for fresh start)."""
        self._positions.clear()
        self._save()

    def _to_result_dict(self, pos: Dict) -> Optional[Dict[str, Any]]:
        """Convert a position dict to StandardMetrics-compatible result dict."""
        pnl = pos.get("pnl")
        if pnl is None:
            return None
        return {
            "pnl": pnl,
            "credit": pos["initial_credit"] * pos["num_contracts"] * 100,
            "max_loss": pos["max_loss"],
            "instrument_type": pos["instrument_type"],
            "option_type": pos["option_type"],
            "entry_time": pos["entry_time"],
            "exit_time": pos["exit_time"],
            "short_strike": pos["short_strike"],
            "long_strike": pos["long_strike"],
            "initial_credit": pos["initial_credit"],
            "num_contracts": pos["num_contracts"],
            "exit_price": pos["exit_price"],
            "exit_reason": pos["exit_reason"],
            "pnl_per_contract": pos.get("pnl_per_contract"),
            "roll_count": pos.get("roll_count", 0),
            "trading_date": pos.get("entry_time", "")[:10],
        }
