"""IntervalBudget, PositionTracker, and IntervalSelector for 5-minute interval orchestration.

IntervalBudget: Decaying allocation model — each interval gets remaining_daily / remaining_intervals.
PositionTracker: Tracks open positions across intervals, checks exit conditions at each tick.
IntervalSelector: Combines exits + selection + position tracking per interval.
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from .evaluator import Proposal
from .selector import SlotSelector

logger = logging.getLogger(__name__)


@dataclass
class IntervalBudget:
    """Decaying allocation: each interval gets remaining / intervals_left.

    This naturally front-loads capital when more intervals remain (early in the day).
    """
    daily_budget: float
    daily_used: float = 0.0
    total_intervals: int = 78          # 6.5h / 5min
    intervals_elapsed: int = 0
    interval_budget_cap: Optional[float] = None

    @property
    def remaining(self) -> float:
        return max(0.0, self.daily_budget - self.daily_used)

    @property
    def intervals_left(self) -> int:
        return max(1, self.total_intervals - self.intervals_elapsed)

    @property
    def effective_interval_budget(self) -> float:
        """Budget available for this interval (decaying allocation)."""
        return self.remaining / self.intervals_left

    def consume(self, amount: float) -> None:
        """Consume budget (positive = use, negative = free)."""
        self.daily_used += amount

    def tick(self) -> None:
        """Advance to the next interval."""
        self.intervals_elapsed += 1

    def reset_day(self) -> None:
        """Reset for a new trading day."""
        self.daily_used = 0.0
        self.intervals_elapsed = 0


class PositionTracker:
    """Tracks orchestrator-accepted positions and checks exits at each interval.

    Uses the existing ExitRule classes from the backtesting framework. At each
    interval, all open positions are checked for exit signals. Closed positions
    are moved to a separate list and their capital is freed.
    """

    def __init__(self, exit_rules=None):
        """Initialize with a list of ExitRule instances or a CompositeExit."""
        self.open_positions: List[Dict[str, Any]] = []
        self.closed_positions: List[Dict[str, Any]] = []
        self._exit_rules = exit_rules

    def open(self, proposal: Proposal, interval_key: str) -> Dict[str, Any]:
        """Record a new position from an accepted proposal. Returns the position dict."""
        original_trade = proposal.metadata.get("original_trade", {})
        pos = {
            "instance_id": proposal.instance_id,
            "algo_name": proposal.algo_name,
            "ticker": proposal.ticker,
            "option_type": proposal.option_type,
            "short_strike": original_trade.get("short_strike", 0),
            "long_strike": original_trade.get("long_strike", 0),
            "initial_credit": proposal.expected_credit,
            "max_loss": proposal.max_loss,
            "num_contracts": proposal.num_contracts,
            "entry_interval": interval_key,
            "entry_time": proposal.entry_time,
            "trading_date": str(proposal.trading_date),
            "roll_count": 0,
            "dte": original_trade.get("dte", 0),
            "original_trade": original_trade,
        }
        self.open_positions.append(pos)
        return pos

    def check_exits(
        self,
        current_price: float,
        current_time: datetime,
        day_context: Any = None,
    ) -> List[Tuple[Dict[str, Any], Any]]:
        """Check all open positions for exit signals.

        Returns list of (position, exit_signal) tuples for exited positions.
        """
        if not self._exit_rules or not self.open_positions:
            return []

        exits = []
        still_open = []

        for pos in self.open_positions:
            signal = self._exit_rules.should_exit(
                pos, current_price, current_time, day_context
            )
            if signal and signal.triggered:
                pos["exit_reason"] = signal.reason
                pos["exit_time"] = str(signal.exit_time)
                pos["exit_price"] = signal.exit_price
                self.closed_positions.append(pos)
                exits.append((pos, signal))
            else:
                still_open.append(pos)

        self.open_positions = still_open
        return exits

    def capital_in_use(self) -> float:
        """Total max_loss across all open positions (in dollars)."""
        return sum(
            p.get("max_loss", 0) * p.get("num_contracts", 1) * 100
            for p in self.open_positions
        )

    def force_close_eod(self, current_price: float, eod_time: datetime) -> List[Dict]:
        """Force-close all remaining open positions at EOD."""
        closed = []
        for pos in self.open_positions:
            pos["exit_reason"] = "eod_force_close"
            pos["exit_time"] = str(eod_time)
            pos["exit_price"] = current_price
            self.closed_positions.append(pos)
            closed.append(pos)

        self.open_positions = []
        return closed

    def reset_day(self, current_price: float = 0.0, eod_time: datetime = None) -> List[Dict]:
        """Reset for new day. Force-closes remaining positions first."""
        forced = []
        if self.open_positions and eod_time:
            forced = self.force_close_eod(current_price, eod_time)
        self.open_positions = []
        return forced


class IntervalSelector:
    """Orchestrates selection + position tracking per interval.

    Combines:
    1. Exit checking on open positions (frees capital)
    2. Budget computation (decaying allocation minus capital in use)
    3. New entry selection from proposals
    4. Position tracking for accepted entries
    """

    def __init__(
        self,
        slot_selector: SlotSelector,
        budget: IntervalBudget,
        position_tracker: PositionTracker,
        max_risk_per_transaction: Optional[float] = None,
    ):
        self.slot_selector = slot_selector
        self.budget = budget
        self.positions = position_tracker
        self.max_risk_per_transaction = max_risk_per_transaction

    def evaluate_interval(
        self,
        proposals: List[Proposal],
        current_price: float,
        current_time: datetime,
        interval_key: str,
        day_context: Any = None,
        trigger_context: Any = None,
    ) -> Tuple[List[Proposal], List[Tuple[Dict, Any]]]:
        """Evaluate a single interval: check exits, select new entries.

        Args:
            trigger_context: Optional TriggerContext for adaptive budget mechanisms.
                Only used when self.budget is AdaptiveIntervalBudget.

        Returns:
            (accepted_proposals, exit_events)
        """
        from .adaptive_budget import AdaptiveIntervalBudget

        # 1. Check exits on all open positions FIRST
        exits = self.positions.check_exits(current_price, current_time, day_context)

        # Free capital from closed positions
        for pos, signal in exits:
            freed = pos.get("max_loss", 0) * pos.get("num_contracts", 1) * 100
            self.budget.consume(-freed)

        # 2. Compute available budget
        capital_used = self.positions.capital_in_use()

        # 3. Enforce per-transaction risk cap on original proposals (before scaling)
        if self.max_risk_per_transaction is not None:
            proposals = [
                p for p in proposals
                if p.total_max_loss <= self.max_risk_per_transaction
            ]

        if isinstance(self.budget, AdaptiveIntervalBudget) and trigger_context is not None:
            # Adaptive path: filter 0DTE after cutoff, compute adaptive budget
            proposals = self.budget.filter_proposals_by_dte_cutoff(
                proposals, current_time
            )
            budget_for_interval, contract_mult = self.budget.compute_interval_budget(
                proposals, trigger_context
            )
            # Adaptive per-interval budget is the boosted amount (already >= flat cap).
            # Also constrained by daily remaining minus capital in use.
            daily_available = max(0.0, self.budget.remaining - capital_used)
            available = min(budget_for_interval, daily_available)

            # Scale contracts on proposals if contract_mult > 1.0
            # (per-transaction cap already applied to originals above)
            if contract_mult > 1.0:
                proposals = self._scale_contracts(proposals, contract_mult)
        else:
            # Original decaying path (backward compatible)
            available = max(0.0, self.budget.remaining - capital_used)
            if self.budget.interval_budget_cap is not None:
                available = min(available, self.budget.interval_budget_cap)

        # 4. Select new entries from proposals
        accepted = self.slot_selector.select(proposals, available)

        # 4. Open new positions and consume budget
        for p in accepted:
            self.positions.open(p, interval_key)
            self.budget.consume(p.total_max_loss)

        # 5. Advance interval counter
        self.budget.tick()

        return accepted, exits

    @staticmethod
    def _scale_contracts(proposals: List[Proposal], multiplier: float) -> List[Proposal]:
        """Scale num_contracts on proposals by multiplier (rounds down)."""
        scaled = []
        for p in proposals:
            new_contracts = max(1, int(p.num_contracts * multiplier))
            if new_contracts != p.num_contracts:
                # Create a new Proposal with scaled contracts
                scaled.append(Proposal(
                    instance_id=p.instance_id,
                    algo_name=p.algo_name,
                    ticker=p.ticker,
                    can_activate=p.can_activate,
                    expected_credit=p.expected_credit,
                    max_loss=p.max_loss,
                    score=p.score,
                    num_contracts=new_contracts,
                    option_type=p.option_type,
                    trading_date=p.trading_date,
                    entry_time=p.entry_time,
                    metadata=p.metadata,
                ))
            else:
                scaled.append(p)
        return scaled
