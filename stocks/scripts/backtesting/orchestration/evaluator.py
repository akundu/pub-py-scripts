"""Proposal dataclass and scoring function for orchestration.

Scoring logic ported from run_tiered_backtest_v3.py:score_trade().
"""

import math
from dataclasses import dataclass, field
from typing import Any, Dict, Optional


@dataclass
class Proposal:
    """Returned by an AlgoInstance when polled.

    Contains everything needed for the orchestrator to compare and select
    among competing proposals from different algo instances.
    """
    instance_id: str
    algo_name: str
    ticker: str
    can_activate: bool
    expected_credit: float        # Per-contract credit
    max_loss: float               # Per-contract max loss
    score: float                  # Composite score (0.0-1.0)
    num_contracts: int
    option_type: str              # "put", "call", "iron_condor"
    trading_date: Optional[Any] = None
    entry_time: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def total_credit(self) -> float:
        return self.expected_credit * self.num_contracts * 100

    @property
    def total_max_loss(self) -> float:
        return self.max_loss * self.num_contracts * 100

    @property
    def credit_risk_ratio(self) -> float:
        if self.max_loss <= 0:
            return 0.0
        return self.expected_credit / self.max_loss


def score_proposal(
    credit: float,
    max_loss: float,
    num_contracts: int = 1,
    min_leg_volume: int = 0,
    avg_bid_ask_pct: float = 0.5,
    weights: tuple = (0.40, 0.30, 0.30),
) -> float:
    """Score a proposal for cross-algo comparison.  Higher = better.

    Ported from run_tiered_backtest_v3.py:score_trade().

    Components:
      1. Credit/Risk ratio (w=0.40): credit / max_loss
      2. Volume adequacy (w=0.30): min_leg_volume relative to contracts
      3. Bid-ask tightness (w=0.30): tighter spread = higher score

    Returns float in [0, 1].
    """
    w_credit, w_volume, w_bidask = weights

    # Credit/Risk component
    num_contracts = max(num_contracts, 1)
    credit = abs(credit)
    max_loss = abs(max_loss)
    credit_risk = (credit / max_loss) if max_loss > 0 else 0
    cr_score = min(1.0, max(0.0, (credit_risk - 0.05) / 0.45))

    # Volume component
    vol_ratio = min_leg_volume / num_contracts if num_contracts > 0 else 0
    if vol_ratio <= 0:
        vol_score = 0.0
    else:
        vol_score = min(1.0, math.log(1 + vol_ratio) / math.log(6))

    # Bid-ask component
    ba_score = max(0.0, min(1.0, (0.25 - avg_bid_ask_pct) / 0.24))

    return w_credit * cr_score + w_volume * vol_score + w_bidask * ba_score
