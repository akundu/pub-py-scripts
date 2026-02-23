#!/usr/bin/env python3
"""
Position Tracker for Continuous Mode

Tracks manually entered positions and monitors exit conditions.
"""

import sys
import json
from pathlib import Path
from datetime import datetime, date
from typing import List, Dict, Optional
from dataclasses import dataclass, asdict
import uuid

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from scripts.continuous.config import CONFIG
from scripts.continuous.alert_manager import ALERTS, AlertLevel


@dataclass
class Position:
    """Represents an open position."""
    position_id: str
    entry_time: str
    ticker: str

    # Config details
    dte: int
    expiration_date: str
    band: str
    spread_type: str
    flow_mode: str

    # Trade details
    strikes: Dict[str, float]  # {short_call, long_call, short_put, long_put}
    n_contracts: int
    credit_received: float
    max_risk: float

    # Exit thresholds
    profit_target: float  # Dollar amount
    stop_loss: float  # Dollar amount

    # Current status
    current_pnl: float = 0.0
    status: str = 'open'  # open, closed
    notes: str = ''

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict) -> 'Position':
        """Create from dictionary."""
        return cls(**data)

    def check_exit_conditions(self) -> List[str]:
        """
        Check if any exit conditions are met.

        Returns:
            List of exit reasons (empty if no exit needed)
        """
        reasons = []

        # Profit target
        if self.current_pnl >= self.profit_target:
            reasons.append(f'Profit target hit (${self.current_pnl:.2f} >= ${self.profit_target:.2f})')

        # Stop loss
        if self.current_pnl <= -self.stop_loss:
            reasons.append(f'Stop loss hit (${self.current_pnl:.2f} <= -${self.stop_loss:.2f})')

        # Time-based exit (1 day before expiration)
        try:
            exp_date = datetime.strptime(self.expiration_date, '%Y-%m-%d').date()
            days_to_exp = (exp_date - date.today()).days

            if days_to_exp <= CONFIG.time_exit_dte:
                reasons.append(f'Time exit ({days_to_exp} days to expiration)')
        except ValueError:
            pass

        return reasons


class PositionTracker:
    """Manages position tracking."""

    def __init__(self, positions_file: Path = None):
        """Initialize position tracker."""
        self.positions_file = positions_file or CONFIG.positions_file
        self.positions: List[Position] = []
        self.load_positions()

    def load_positions(self):
        """Load positions from JSON file."""
        if not self.positions_file.exists():
            self.positions = []
            return

        try:
            with open(self.positions_file, 'r') as f:
                data = json.load(f)
                self.positions = [Position.from_dict(p) for p in data]
        except Exception as e:
            print(f"Error loading positions: {e}")
            self.positions = []

    def save_positions(self):
        """Save positions to JSON file."""
        try:
            self.positions_file.parent.mkdir(parents=True, exist_ok=True)
            with open(self.positions_file, 'w') as f:
                data = [p.to_dict() for p in self.positions]
                json.dump(data, f, indent=2)
        except Exception as e:
            print(f"Error saving positions: {e}")

    def add_position(
        self,
        ticker: str,
        dte: int,
        expiration_date: str,
        band: str,
        spread_type: str,
        flow_mode: str,
        strikes: Dict[str, float],
        n_contracts: int,
        credit_received: float,
        max_risk: float,
        profit_target_pct: float = 0.50,
        stop_loss_mult: float = 2.0,
        notes: str = ''
    ) -> Position:
        """
        Add a new position.

        Args:
            ticker: Ticker symbol
            dte: Days to expiration
            expiration_date: Expiration date (YYYY-MM-DD)
            band: Percentile band (P95, P97, etc.)
            spread_type: put_spread, call_spread, iron_condor
            flow_mode: with_flow, against_flow, neutral
            strikes: Strike prices dict
            n_contracts: Number of contracts
            credit_received: Total credit received
            max_risk: Maximum risk (width * contracts - credit)
            profit_target_pct: Profit target as % of credit (default 50%)
            stop_loss_mult: Stop loss as multiple of credit (default 2x)
            notes: Optional notes

        Returns:
            Created position
        """
        position = Position(
            position_id=str(uuid.uuid4())[:8],
            entry_time=datetime.now().isoformat(),
            ticker=ticker,
            dte=dte,
            expiration_date=expiration_date,
            band=band,
            spread_type=spread_type,
            flow_mode=flow_mode,
            strikes=strikes,
            n_contracts=n_contracts,
            credit_received=credit_received,
            max_risk=max_risk,
            profit_target=credit_received * profit_target_pct,
            stop_loss=credit_received * stop_loss_mult,
            current_pnl=0.0,
            status='open',
            notes=notes,
        )

        self.positions.append(position)
        self.save_positions()

        ALERTS.send_alert(
            f"New position added: {position.position_id} | "
            f"{dte}DTE {band} {spread_type} | "
            f"Credit: ${credit_received:.2f}",
            AlertLevel.INFO
        )

        return position

    def update_pnl(self, position_id: str, current_pnl: float):
        """Update position P&L."""
        for pos in self.positions:
            if pos.position_id == position_id and pos.status == 'open':
                pos.current_pnl = current_pnl
                self.save_positions()
                return True
        return False

    def close_position(self, position_id: str, final_pnl: float, notes: str = ''):
        """Close a position."""
        for pos in self.positions:
            if pos.position_id == position_id and pos.status == 'open':
                pos.current_pnl = final_pnl
                pos.status = 'closed'
                if notes:
                    pos.notes += f" | Closed: {notes}"
                self.save_positions()

                ALERTS.send_alert(
                    f"Position closed: {position_id} | "
                    f"P&L: ${final_pnl:.2f} ({(final_pnl/pos.credit_received*100):+.1f}%)",
                    AlertLevel.INFO
                )
                return True
        return False

    def get_open_positions(self) -> List[Position]:
        """Get all open positions."""
        return [p for p in self.positions if p.status == 'open']

    def get_total_risk(self) -> float:
        """Calculate total capital at risk."""
        return sum(p.max_risk for p in self.get_open_positions())

    def check_exit_signals(self):
        """Check all open positions for exit signals."""
        for pos in self.get_open_positions():
            reasons = pos.check_exit_conditions()
            if reasons:
                for reason in reasons:
                    ALERTS.alert_position_exit(
                        pos.position_id,
                        reason,
                        pos.current_pnl,
                        pos.credit_received
                    )

    def get_summary(self) -> Dict:
        """Get portfolio summary."""
        open_positions = self.get_open_positions()
        closed_positions = [p for p in self.positions if p.status == 'closed']

        total_open_pnl = sum(p.current_pnl for p in open_positions)
        total_closed_pnl = sum(p.current_pnl for p in closed_positions)

        return {
            'total_positions': len(self.positions),
            'open_positions': len(open_positions),
            'closed_positions': len(closed_positions),
            'total_risk': self.get_total_risk(),
            'unrealized_pnl': total_open_pnl,
            'realized_pnl': total_closed_pnl,
            'total_pnl': total_open_pnl + total_closed_pnl,
        }


if __name__ == '__main__':
    """Test position tracker."""
    print("Testing position tracker...")

    tracker = PositionTracker()

    # Add a test position
    pos = tracker.add_position(
        ticker='NDX',
        dte=3,
        expiration_date='2026-02-24',
        band='P98',
        spread_type='iron_condor',
        flow_mode='with_flow',
        strikes={
            'short_call': 20500,
            'long_call': 20600,
            'short_put': 19500,
            'long_put': 19400,
        },
        n_contracts=2,
        credit_received=285.0,
        max_risk=1715.0,
        notes='Test position'
    )

    print(f"\nAdded position: {pos.position_id}")

    # Update P&L
    tracker.update_pnl(pos.position_id, 142.50)
    print(f"Updated P&L: ${pos.current_pnl:.2f}")

    # Check exit conditions
    tracker.check_exit_signals()

    # Summary
    summary = tracker.get_summary()
    print(f"\nPortfolio Summary:")
    print(f"  Open: {summary['open_positions']}")
    print(f"  Total Risk: ${summary['total_risk']:,.2f}")
    print(f"  Unrealized P&L: ${summary['unrealized_pnl']:,.2f}")

    print(f"\nPositions saved to: {tracker.positions_file}")
