"""Multi-algorithm orchestration system.

Coordinates multiple strategy instances, evaluates triggers, and selects
the best algo per trading slot based on scoring and budget constraints.
Supports both daily and 5-minute interval modes.
"""

from .adaptive_budget import AdaptiveBudgetConfig, AdaptiveIntervalBudget
from .evaluator import Proposal, score_proposal
from .algo_instance import AlgoInstance, SubOrchestrator
from .manifest import OrchestrationManifest
from .selector import SlotSelector
from .engine import OrchestratorEngine
from .interval_selector import IntervalBudget, IntervalSelector, PositionTracker
from .baseline import (
    compute_standalone_baseline,
    compute_equal_weight_baseline,
    load_v3_baseline,
    build_comparison_table,
)

__all__ = [
    "AdaptiveBudgetConfig",
    "AdaptiveIntervalBudget",
    "Proposal",
    "score_proposal",
    "AlgoInstance",
    "SubOrchestrator",
    "OrchestrationManifest",
    "SlotSelector",
    "OrchestratorEngine",
    "IntervalBudget",
    "IntervalSelector",
    "PositionTracker",
    "compute_standalone_baseline",
    "compute_equal_weight_baseline",
    "load_v3_baseline",
    "build_comparison_table",
]
