"""SlotSelector -- per-slot algo selection from proposals."""

import logging
from collections import defaultdict
from typing import List, Optional

from .evaluator import Proposal

logger = logging.getLogger(__name__)


class SlotSelector:
    """Selects the best proposal(s) per trading slot.

    Selection modes:
    - "best_score": Highest composite score wins (default)
    - "priority": Highest priority (lowest number) wins, score breaks ties
    - "top_n": Top N proposals by score (portfolio mode)
    - "priority_fallback": Try highest-priority tier first; fall through to
      lower tiers only if the higher tier produces zero accepted trades
    """

    def __init__(self, mode: str = "best_score", top_n: int = 1):
        self.mode = mode
        self.top_n = top_n

    def select(
        self,
        proposals: List[Proposal],
        budget_remaining: float = float("inf"),
    ) -> List[Proposal]:
        """Pick winning proposal(s) from candidates.

        Args:
            proposals: All proposals for this slot (already trigger-filtered).
            budget_remaining: Remaining daily budget.

        Returns:
            List of accepted proposals (may be empty).
        """
        if not proposals:
            return []

        # Filter to only activated proposals
        active = [p for p in proposals if p.can_activate]
        if not active:
            return []

        # Priority-fallback: try tiers in order, stop at first tier with accepts
        if self.mode == "priority_fallback":
            return self._select_priority_fallback(active, budget_remaining)

        # Sort by selection mode
        if self.mode == "priority":
            active.sort(key=lambda p: (
                p.metadata.get("priority", 999),
                -p.score,
            ))
        elif self.mode == "top_n":
            active.sort(key=lambda p: -p.score)
        else:  # best_score
            active.sort(key=lambda p: -p.score)

        # Select with budget enforcement
        accepted = []
        budget_used = 0.0
        limit = self.top_n if self.mode == "top_n" else 1

        for proposal in active:
            risk = proposal.total_max_loss
            if budget_used + risk > budget_remaining:
                continue
            accepted.append(proposal)
            budget_used += risk
            if len(accepted) >= limit:
                break

        return accepted

    def _select_priority_fallback(
        self,
        active: List[Proposal],
        budget_remaining: float,
    ) -> List[Proposal]:
        """Try highest-priority tier first; fall through only if zero accepted."""
        tiers: dict = defaultdict(list)
        for p in active:
            tiers[p.metadata.get("priority", 999)].append(p)

        limit = self.top_n
        accepted = []
        budget_used = 0.0

        for tier_key in sorted(tiers.keys()):
            tier_proposals = sorted(tiers[tier_key], key=lambda p: -p.score)
            for proposal in tier_proposals:
                risk = proposal.total_max_loss
                if budget_used + risk <= budget_remaining:
                    accepted.append(proposal)
                    budget_used += risk
                    if len(accepted) >= limit:
                        return accepted
            if accepted:
                return accepted  # got something from this tier, don't fall through

        return accepted
