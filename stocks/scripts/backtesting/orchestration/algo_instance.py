"""AlgoInstance and SubOrchestrator -- pollable algo wrappers.

AlgoInstance wraps a strategy config + triggers into a pollable unit.
SubOrchestrator groups instances and picks the best internally.
"""

import copy
import logging
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

from .evaluator import Proposal, score_proposal
from .triggers.base import Trigger, TriggerContext

logger = logging.getLogger(__name__)


@dataclass
class AlgoInstanceConfig:
    """Static configuration for an algo instance."""
    algo_name: str
    instance_id: str
    config_path: str
    overrides: Dict[str, Any] = field(default_factory=dict)
    triggers: List[Trigger] = field(default_factory=list)
    trigger_mode: str = "any"     # "any" (OR) or "all" (AND)
    priority: int = 5
    budget_share: float = 1.0
    enabled: bool = True


class AlgoInstance:
    """Wraps a strategy config + triggers into a pollable unit.

    In backtest mode (Phase 1), each instance runs independently via
    BacktestEngine and produces a trade CSV. In Phase 2, those results
    are replayed and scored.

    The poll() method is used during Phase 2 replay -- it checks triggers
    against the slot context and returns the best trade from this instance's
    results for that slot.

    Supports both daily mode (one poll per date) and interval mode
    (poll with interval_key to match trades to specific 5-min intervals).
    """

    def __init__(self, config: AlgoInstanceConfig):
        self.config = config
        self.instance_id = config.instance_id
        self.algo_name = config.algo_name
        self.triggers = config.triggers
        self.trigger_mode = config.trigger_mode
        self.priority = config.priority
        self.budget_share = config.budget_share

        # Populated after Phase 1 backtest
        self._trades: List[Dict[str, Any]] = []
        self._trades_by_date: Dict[str, List[Dict]] = {}
        self._trades_by_interval: Dict[str, List[Dict]] = {}
        self._backtest_results: Optional[Dict[str, Any]] = None

    @staticmethod
    def make_interval_key(dt: datetime, interval_minutes: int = 5) -> str:
        """Create interval key from datetime: 'YYYY-MM-DD_HHMM'."""
        # Floor to interval boundary
        minute = (dt.minute // interval_minutes) * interval_minutes
        return f"{dt.strftime('%Y-%m-%d')}_{dt.hour:02d}{minute:02d}"

    @staticmethod
    def make_interval_key_from_parts(date_str: str, time_str: str,
                                     interval_minutes: int = 5) -> str:
        """Create interval key from date and time strings.

        Handles various entry_time formats:
        - "14:00" (time only)
        - "2025-09-15 14:00:00" (tz-naive datetime)
        - "2025-09-15 14:00:00+00:00" (tz-aware datetime)
        - "2025-09-15T14:00:00+00:00" (ISO format)
        """
        time_val = str(time_str).strip()
        try:
            # Try parsing as full datetime first (handles all ISO variants)
            if len(time_val) > 8:
                # Full datetime string — use pd.Timestamp for robust parsing
                import pandas as pd
                ts = pd.Timestamp(time_val)
                minute = (ts.minute // interval_minutes) * interval_minutes
                return f"{date_str}_{ts.hour:02d}{minute:02d}"
            else:
                # Short time string like "14:00" or "14:30"
                parts = time_val.split(":")
                hour = int(parts[0])
                min_val = int(parts[1]) if len(parts) > 1 else 0
                minute = (min_val // interval_minutes) * interval_minutes
                return f"{date_str}_{hour:02d}{minute:02d}"
        except (ValueError, TypeError, IndexError):
            return f"{date_str}_0000"

    def check_triggers(self, context: TriggerContext) -> bool:
        """Check if this instance's triggers allow activation."""
        if not self.triggers:
            return True

        if self.trigger_mode == "all":
            return all(t.evaluate(context) for t in self.triggers)
        else:  # "any"
            return any(t.evaluate(context) for t in self.triggers)

    def load_trades(self, trades: List[Dict[str, Any]],
                    interval_minutes: int = 5) -> None:
        """Load Phase 1 backtest trades for Phase 2 replay.

        Builds both daily and interval-level indexes.
        """
        self._trades = trades
        self._trades_by_date.clear()
        self._trades_by_interval.clear()

        for t in trades:
            date_key = str(t.get("trading_date", t.get("entry_date", "")))
            self._trades_by_date.setdefault(date_key, []).append(t)

            # Build interval index from entry_time
            entry_time = t.get("entry_time", "")
            if entry_time:
                ik = self.make_interval_key_from_parts(
                    date_key, str(entry_time), interval_minutes
                )
                self._trades_by_interval.setdefault(ik, []).append(t)

    def set_backtest_results(self, results: Dict[str, Any]) -> None:
        """Store the full backtest results (metrics, summary)."""
        self._backtest_results = results

    @property
    def backtest_results(self) -> Optional[Dict[str, Any]]:
        return self._backtest_results

    @property
    def trades(self) -> List[Dict[str, Any]]:
        return self._trades

    def poll(self, context: TriggerContext,
             interval_key: Optional[str] = None,
             scoring_weights: Optional[tuple] = None) -> List[Proposal]:
        """Get proposals for a trading slot.

        Args:
            context: Trigger context for the slot.
            interval_key: If provided, match trades to this specific interval.
                If None, returns all trades for the date (daily mode).
            scoring_weights: Optional (w_credit, w_volume, w_bidask) tuple.

        Returns all matching trades converted to Proposals with scores.
        Returns empty list if triggers don't fire or no trades exist.
        """
        if not self.check_triggers(context):
            return []

        if interval_key is not None:
            day_trades = self._trades_by_interval.get(interval_key, [])
        else:
            date_key = str(context.trading_date)
            day_trades = self._trades_by_date.get(date_key, [])

        if not day_trades:
            return []

        proposals = []
        score_kwargs = {}
        if scoring_weights is not None:
            score_kwargs["weights"] = scoring_weights

        for trade in day_trades:
            credit = abs(trade.get("credit", 0.0))
            max_loss = abs(trade.get("max_loss", 0.0))
            num_contracts = trade.get("num_contracts", 1)
            min_leg_volume = trade.get("min_leg_volume", 0)
            avg_ba_pct = trade.get("avg_bid_ask_pct", 0.5)

            score = score_proposal(
                credit=credit,
                max_loss=max_loss,
                num_contracts=num_contracts,
                min_leg_volume=min_leg_volume,
                avg_bid_ask_pct=avg_ba_pct,
                **score_kwargs,
            )

            proposal = Proposal(
                instance_id=self.instance_id,
                algo_name=self.algo_name,
                ticker=trade.get("ticker", ""),
                can_activate=True,
                expected_credit=credit,
                max_loss=max_loss,
                score=score,
                num_contracts=num_contracts,
                option_type=trade.get("option_type", ""),
                trading_date=context.trading_date,
                entry_time=trade.get("entry_time", ""),
                metadata={
                    "original_trade": trade,
                    "priority": self.priority,
                    "budget_share": self.budget_share,
                },
            )
            proposals.append(proposal)

        return proposals

    def build_backtest_config(self):
        """Build a BacktestConfig from config_path + overrides.

        Must be called inside subprocess (imports are per-process).
        """
        from scripts.backtesting.config import BacktestConfig

        config = BacktestConfig.load(self.config.config_path)

        # Apply overrides
        for key, value in self.config.overrides.items():
            if key == "ticker":
                config.infra.ticker = value
            elif key == "output_dir":
                config.infra.output_dir = value
            elif hasattr(config.infra, key):
                setattr(config.infra, key, value)
            else:
                # Strategy param override
                config.strategy.params[key] = value

        return config

    def __repr__(self):
        return (
            f"AlgoInstance(id={self.instance_id!r}, "
            f"algo={self.algo_name!r}, priority={self.priority})"
        )


class SubOrchestrator(AlgoInstance):
    """An orchestrator that acts as an AlgoInstance in a parent orchestrator.

    Groups child instances, picks the best internally, and surfaces
    the winning Proposal upward.
    """

    def __init__(self, config: AlgoInstanceConfig, children: List[AlgoInstance],
                 selection_mode: str = "best_score"):
        super().__init__(config)
        self.children = children
        self.selection_mode = selection_mode

    def poll(self, context: TriggerContext,
             interval_key: Optional[str] = None,
             scoring_weights: Optional[tuple] = None) -> List[Proposal]:
        """Poll all children, select the best, return as this group's proposals."""
        if not self.check_triggers(context):
            return []

        all_proposals = []
        for child in self.children:
            child_proposals = child.poll(
                context, interval_key=interval_key,
                scoring_weights=scoring_weights,
            )
            all_proposals.extend(child_proposals)

        if not all_proposals:
            return []

        # Select the best within this group (uses child-level priorities)
        if self.selection_mode == "priority":
            all_proposals.sort(
                key=lambda p: (p.metadata.get("priority", 999), -p.score)
            )
            selected = [all_proposals[0]]
        elif self.selection_mode == "top_n":
            n = self.config.overrides.get("top_n", 1)
            all_proposals.sort(key=lambda p: -p.score)
            selected = all_proposals[:n]
        else:  # best_score
            all_proposals.sort(key=lambda p: -p.score)
            selected = [all_proposals[0]]

        # Stamp group priority onto selected proposals so the top-level
        # priority_fallback selector can tier them correctly.
        for p in selected:
            p.metadata["priority"] = self.priority

        return selected

    def load_trades(self, trades: List[Dict[str, Any]],
                    interval_minutes: int = 5) -> None:
        """SubOrchestrator doesn't hold trades directly -- children do."""
        pass

    @property
    def all_children(self) -> List[AlgoInstance]:
        """Recursively get all leaf AlgoInstances."""
        result = []
        for child in self.children:
            if isinstance(child, SubOrchestrator):
                result.extend(child.all_children)
            else:
                result.append(child)
        return result

    def __repr__(self):
        return (
            f"SubOrchestrator(id={self.instance_id!r}, "
            f"children={len(self.children)}, mode={self.selection_mode!r})"
        )
