"""Adaptive interval budget allocation for orchestration.

Replaces the flat decaying IntervalBudget with six composable mechanisms:
1. Reserve Bonus — set aside N% of daily budget as a bonus pool released in last M intervals
2. ROI Percentile Tiers — per-DTE percentile-based budget scaling (replaces simple opportunity scaling)
3. Momentum Boost — boost when intraday move exceeds threshold
4. Time-Weight Curve — weight later intervals higher (configurable curve)
5. Contract Scaling — scale num_contracts on best proposals
6. VIX Integration — regime-based budget multipliers

Core principle: **adaptive budget >= flat budget, always.**
  flat_budget  = remaining / intervals_left   (the decaying baseline)
  boost        = max(1.0, roi_tier_mult × momentum × time_weight)
  reserve_bonus = additive $ released in last M intervals
  final_budget = flat_budget × boost + reserve_bonus

The adaptive mechanisms can only INCREASE allocation above what flat would give.
Below-median proposals get exactly the flat budget. Above-median get more.
The reserve is an additive bonus pool — it doesn't reduce early intervals.

Also enforces the 0DTE cutoff rule: no new 0DTE trades after max_trading_hour_utc
on the day of expiration. Capital that would go to 0DTE is diverted to 1DTE+.
"""

import logging
import math
import statistics
from dataclasses import dataclass, field
from datetime import datetime, time
from typing import Any, Dict, List, Optional, Tuple

from .evaluator import Proposal

logger = logging.getLogger(__name__)


@dataclass
class AdaptiveBudgetConfig:
    """All YAML-configurable parameters for adaptive budget allocation."""

    # Reserve floor — hold back reserve_pct until last reserve_release_intervals
    reserve_enabled: bool = True
    reserve_pct: float = 0.30
    reserve_release_intervals: int = 24  # last 2 hours at 5min intervals

    # Opportunity scaling — deploy more when credit/risk is above historical median
    opportunity_scaling_enabled: bool = True
    opportunity_max_multiplier: float = 3.0
    opportunity_min_multiplier: float = 0.5

    # Momentum — boost when intraday move exceeds threshold
    momentum_enabled: bool = True
    momentum_threshold: float = 0.01  # 1% intraday move
    momentum_boost: float = 1.5

    # Time weight — favor later intervals for 0DTE
    time_weight_enabled: bool = True
    time_weight_curve: str = "linear"  # "linear", "exponential", "step"
    time_weight_early_factor: float = 0.7
    time_weight_late_factor: float = 1.5

    # Contract scaling — scale num_contracts instead of budget
    contract_scaling_enabled: bool = False
    contract_max_multiplier: float = 2.0

    # VIX integration — multipliers by regime
    vix_budget_multipliers: Dict[str, float] = field(default_factory=lambda: {
        "low": 1.2,
        "normal": 1.0,
        "high": 0.6,
        "extreme": 0.25,
    })

    # ROI-based dynamic allocation
    # roi_mode: "fixed_roi" uses fixed ROI% thresholds (no historical data needed)
    #           "percentile" uses per-DTE CR percentile tiers (legacy)
    roi_tier_enabled: bool = True
    roi_mode: str = "fixed_roi"
    # fixed_roi mode: ROI = credit/(width-credit)*100, normalized by DTE+1
    roi_thresholds: List[float] = field(default_factory=lambda: [6.0, 9.0])
    roi_multipliers: List[float] = field(default_factory=lambda: [1.0, 2.0, 4.0])
    roi_max_multiplier: float = 4.0
    roi_normalize_dte: bool = True
    # Legacy percentile mode fields
    roi_tier_percentiles: List[float] = field(default_factory=lambda: [50, 75, 90, 95])
    roi_tier_multipliers: List[float] = field(default_factory=lambda: [1.0, 1.5, 2.0, 3.0, 4.0])
    roi_tier_min_trades: int = 30

    # Safety limits
    absolute_interval_cap: Optional[float] = None  # hard dollar cap per interval
    max_daily_utilization: float = 0.95
    combined_max_multiplier: float = 5.0

    # 0DTE cutoff — no new 0DTE trades after this time on expiration day
    # 18:30 UTC = 11:30 AM PST = 2:30 PM ET
    dte0_cutoff_utc: str = "18:30"

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "AdaptiveBudgetConfig":
        """Build config from a parsed YAML dict."""
        if not d:
            return cls()
        kwargs = {}
        for fld in cls.__dataclass_fields__:
            if fld in d:
                kwargs[fld] = d[fld]
        return cls(**kwargs)


class AdaptiveIntervalBudget:
    """Adaptive interval budget that composes five mechanisms on top of decaying base.

    Drop-in replacement for IntervalBudget when interval_budget_mode == "adaptive".
    Maintains the same interface: remaining, intervals_left, consume(), tick(), reset_day().
    """

    def __init__(
        self,
        daily_budget: float,
        total_intervals: int = 78,
        interval_budget_cap: Optional[float] = None,
        config: Optional[AdaptiveBudgetConfig] = None,
    ):
        self.daily_budget = daily_budget
        self.daily_used: float = 0.0
        self.total_intervals = total_intervals
        self.intervals_elapsed: int = 0
        self.interval_budget_cap = interval_budget_cap
        self.config = config or AdaptiveBudgetConfig()

        # Historical baseline (populated from Phase 1 trades)
        self._median_cr: Optional[float] = None
        self._cr_values: List[float] = []

        # Per-DTE percentile distributions for ROI tier scaling
        self._dte_percentiles: Dict[str, Dict[float, float]] = {}
        self._global_percentiles: Dict[float, float] = {}

        # VIX multiplier (set per day)
        self._vix_multiplier: float = 1.0

        # Per-interval analytics log
        self._interval_log: List[Dict[str, Any]] = []

        # Parse 0DTE cutoff time
        parts = self.config.dte0_cutoff_utc.split(":")
        self._dte0_cutoff = time(int(parts[0]), int(parts[1]))

    # --- Same interface as IntervalBudget ---

    @property
    def remaining(self) -> float:
        return max(0.0, self.daily_budget - self.daily_used)

    @property
    def intervals_left(self) -> int:
        return max(1, self.total_intervals - self.intervals_elapsed)

    @property
    def effective_interval_budget(self) -> float:
        """Base decaying allocation (before adaptive multipliers)."""
        return self.remaining / self.intervals_left

    def consume(self, amount: float) -> None:
        self.daily_used += amount

    def tick(self) -> None:
        self.intervals_elapsed += 1

    def reset_day(self) -> None:
        self.daily_used = 0.0
        self.intervals_elapsed = 0
        self._vix_multiplier = 1.0

    # --- Adaptive-specific methods ---

    def load_historical_stats(self, phase1_trades: List[Dict[str, Any]]) -> None:
        """Precompute median credit/risk and per-DTE percentile distributions.

        Called once after Phase 1 completes. Builds:
        - _median_cr: global median for legacy opportunity_scaling fallback
        - _global_percentiles: global CR percentile breakpoints
        - _dte_percentiles: per-DTE-bucket CR percentile breakpoints
        """
        cr_values = []
        dte_bucket_crs: Dict[str, List[float]] = {
            "0": [], "1": [], "2": [], "3-5": [], "6-10": [],
        }

        for trade in phase1_trades:
            credit = abs(trade.get("credit", trade.get("initial_credit", 0)))
            max_loss = abs(trade.get("max_loss", trade.get("spread_width", 0)))
            if max_loss > 0 and credit > 0:
                cr = credit / max_loss
                cr_values.append(cr)
                dte = trade.get("dte", 0)
                bucket = _dte_to_bucket(dte)
                dte_bucket_crs[bucket].append(cr)

        self._cr_values = cr_values
        if cr_values:
            self._median_cr = statistics.median(cr_values)

            # Compute global percentile breakpoints
            self._global_percentiles = {
                p: _percentile(cr_values, p)
                for p in self.config.roi_tier_percentiles
            }

            # Compute per-DTE-bucket percentile breakpoints
            self._dte_percentiles = {}
            for bucket, crs in dte_bucket_crs.items():
                if len(crs) >= self.config.roi_tier_min_trades:
                    self._dte_percentiles[bucket] = {
                        p: _percentile(crs, p)
                        for p in self.config.roi_tier_percentiles
                    }

            logger.debug(
                f"Adaptive budget: loaded {len(cr_values)} historical trades, "
                f"median CR={self._median_cr:.4f}, "
                f"DTE buckets with enough data: {list(self._dte_percentiles.keys())}"
            )
        else:
            self._median_cr = None
            self._global_percentiles = {}
            self._dte_percentiles = {}
            logger.debug("Adaptive budget: no historical trades for baseline, "
                         "opportunity scaling disabled")

    def set_vix_multiplier(self, regime: Optional[str]) -> None:
        """Set VIX multiplier from regime string. Called at day start."""
        if regime and regime in self.config.vix_budget_multipliers:
            self._vix_multiplier = self.config.vix_budget_multipliers[regime]
        else:
            self._vix_multiplier = 1.0

    def is_past_dte0_cutoff(self, current_time: Optional[datetime]) -> bool:
        """Check if current time is past the 0DTE cutoff."""
        if current_time is None:
            return False
        return current_time.time() >= self._dte0_cutoff

    def filter_proposals_by_dte_cutoff(
        self,
        proposals: List[Proposal],
        current_time: Optional[datetime],
    ) -> List[Proposal]:
        """Filter out 0DTE proposals after cutoff time.

        After dte0_cutoff_utc on expiration day, only 1DTE+ trades are allowed.
        This diverts capital from 0DTE to longer-dated trades.
        """
        if not self.is_past_dte0_cutoff(current_time):
            return proposals

        filtered = []
        for p in proposals:
            dte = p.metadata.get("original_trade", {}).get("dte", 0)
            if dte >= 1:
                filtered.append(p)
            else:
                logger.debug(
                    f"Adaptive budget: filtered 0DTE proposal {p.instance_id} "
                    f"after cutoff {self.config.dte0_cutoff_utc}"
                )
        return filtered

    def compute_interval_budget(
        self,
        proposals: List[Proposal],
        trigger_context: Any = None,
    ) -> tuple:
        """Compute adaptive budget for this interval.

        The flat baseline per interval is interval_budget_cap (the amount flat
        mode would give). The adaptive boost multiplies that base so more trades
        fit in a single interval when opportunity is good. Reserve bonus adds
        extra $ in the last M intervals.

        Core guarantee: adaptive >= flat, always.

        Formula:
          flat_base    = interval_budget_cap (or remaining/intervals_left if no cap)
          boost        = max(1.0, opportunity × momentum × time_weight)
          reserve_bonus = additive $ per interval in the release zone
          budget       = flat_base × boost + reserve_bonus
          ...capped by daily remaining × max_daily_utilization

        Returns:
            (budget_dollars, contract_multiplier)
        """
        cfg = self.config

        # Flat base = what the flat mode gives per interval.
        # When interval_budget_cap is set, that IS the flat per-interval budget.
        # Otherwise fall back to the decaying allocation.
        if self.interval_budget_cap is not None:
            flat_base = self.interval_budget_cap
        else:
            flat_base = self.effective_interval_budget

        # --- VIX adjustment (high VIX is a real risk signal) ---
        flat_base *= self._vix_multiplier

        # --- 1. Boost from opportunity/ROI tier, momentum, time weight ---
        # Each mechanism returns >= 1.0 (boost-only, never reduces below flat)
        opportunity_mult, roi_tier_info = self._compute_opportunity_or_roi_multiplier(proposals)
        momentum_mult = self._compute_momentum_multiplier(trigger_context)
        time_weight = self._compute_time_weight()

        # Boost is the product, floored at 1.0 (can never go below flat)
        raw_boost = opportunity_mult * momentum_mult * time_weight
        boost = max(1.0, min(raw_boost, cfg.combined_max_multiplier))

        # --- 2. Reserve bonus (additive, released late in the day) ---
        reserve_bonus = self._compute_reserve_bonus()

        # Final budget = flat × boost + reserve bonus
        budget = flat_base * boost + reserve_bonus

        # Apply absolute cap
        if cfg.absolute_interval_cap is not None:
            budget = min(budget, cfg.absolute_interval_cap)

        # Apply max daily utilization (don't exceed daily_budget × 95%)
        max_available = self.daily_budget * cfg.max_daily_utilization - self.daily_used
        budget = min(budget, max(0.0, max_available))

        # --- 3. Contract Scaling ---
        contract_mult = 1.0
        if cfg.contract_scaling_enabled:
            contract_mult = self._compute_contract_multiplier(proposals)

        # Log this interval's analytics
        log_entry = {
            "interval_index": self.intervals_elapsed,
            "intervals_left": self.intervals_left,
            "flat_base": flat_base,
            "vix_multiplier": self._vix_multiplier,
            "reserve_bonus": reserve_bonus,
            "opportunity_mult": opportunity_mult,
            "momentum_mult": momentum_mult,
            "time_weight": time_weight,
            "boost": boost,
            "contract_mult": contract_mult,
            "final_budget": budget,
            "remaining": self.remaining,
            "daily_used": self.daily_used,
            "num_proposals": len(proposals),
        }
        log_entry.update(roi_tier_info)
        self._interval_log.append(log_entry)

        return (budget, contract_mult)

    @property
    def interval_log(self) -> List[Dict[str, Any]]:
        return list(self._interval_log)

    # --- Private mechanism implementations ---

    def _compute_reserve_bonus(self) -> float:
        """Reserve bonus: additive $ per interval, released in the last M intervals.

        The reserve pool = daily_budget × reserve_pct. Before the release zone,
        this bonus is $0. In the release zone, the remaining reserve pool is
        spread evenly across the remaining intervals (decaying within the zone).
        """
        cfg = self.config
        if not cfg.reserve_enabled:
            return 0.0

        intervals_left = self.intervals_left
        release_start = cfg.reserve_release_intervals

        if intervals_left > release_start:
            # Before release zone — no bonus
            return 0.0

        # In the release zone — spread reserve pool across remaining intervals
        reserve_pool = self.daily_budget * cfg.reserve_pct
        return reserve_pool / intervals_left

    def _compute_opportunity_or_roi_multiplier(
        self, proposals: List[Proposal]
    ) -> Tuple[float, Dict[str, Any]]:
        """Compute opportunity multiplier using ROI tiers or legacy scaling.

        Supports three modes:
        - fixed_roi: fixed ROI% thresholds (no historical data needed)
        - percentile: per-DTE CR percentile tiers (legacy, needs history)
        - legacy: simple best_cr / median_cr ratio

        Returns (multiplier, info_dict) where info_dict has observability fields.
        """
        cfg = self.config
        empty_info = {"roi_tier": "none", "proposal_cr": 0.0, "proposal_dte": 0,
                      "bucket_used": "", "percentile_rank": "", "norm_roi": 0.0}

        if not proposals:
            return 1.0, empty_info

        # Fixed ROI mode — no historical data needed
        if cfg.roi_tier_enabled and cfg.roi_mode == "fixed_roi":
            return self._compute_fixed_roi_multiplier(proposals)

        # Percentile mode — needs historical data
        if cfg.roi_tier_enabled and self._global_percentiles:
            return self._compute_roi_tier_multiplier(proposals)

        # Legacy fallback: simple best_cr / median_cr
        return self._compute_opportunity_multiplier_legacy(proposals), empty_info

    def _compute_fixed_roi_multiplier(
        self, proposals: List[Proposal]
    ) -> Tuple[float, Dict[str, Any]]:
        """Fixed ROI threshold multiplier (no historical data needed).

        ROI = credit / (width - credit) * 100, normalized by DTE+1.
        Thresholds: <6% → 1x, 6-9% → 2x, >9% → 4x (configurable).
        """
        cfg = self.config
        best = max(proposals, key=lambda p: p.credit_risk_ratio)
        cr = best.credit_risk_ratio  # credit / max_loss
        dte = best.metadata.get("original_trade", {}).get("dte", 0)

        # ROI = credit / max_loss * 100 (credit_risk_ratio is already credit/max_loss)
        roi = cr * 100
        norm_roi = roi / (dte + 1) if cfg.roi_normalize_dte else roi

        # Find tier
        tier_labels = ["flat", "good", "strong", "exceptional", "spike"]
        mult = cfg.roi_multipliers[0]
        tier_label = tier_labels[0]

        for i in range(len(cfg.roi_thresholds) - 1, -1, -1):
            if norm_roi >= cfg.roi_thresholds[i]:
                mult = cfg.roi_multipliers[i + 1]
                tier_label = tier_labels[min(i + 1, len(tier_labels) - 1)]
                break

        mult = min(mult, cfg.roi_max_multiplier)

        info = {
            "roi_tier": tier_label,
            "proposal_cr": round(cr, 6),
            "proposal_dte": dte,
            "norm_roi": round(norm_roi, 2),
            "bucket_used": "fixed_roi",
            "percentile_rank": f"roi={norm_roi:.1f}%",
        }
        return mult, info

    def _compute_roi_tier_multiplier(
        self, proposals: List[Proposal]
    ) -> Tuple[float, Dict[str, Any]]:
        """Per-DTE percentile-tiered budget multiplier.

        Looks up the best proposal's CR against its DTE bucket's distribution.
        Falls back to global distribution if the DTE bucket has sparse data.
        """
        cfg = self.config
        best_proposal = max(proposals, key=lambda p: p.credit_risk_ratio)
        cr = best_proposal.credit_risk_ratio
        dte = best_proposal.metadata.get("original_trade", {}).get("dte", 0)
        bucket = _dte_to_bucket(dte)

        # Use per-DTE percentiles if available, else global
        percentiles = self._dte_percentiles.get(bucket, self._global_percentiles)
        bucket_used = bucket if bucket in self._dte_percentiles else "global"

        # Find which tier this CR falls in
        tier_label = "flat"
        tier_labels = ["flat", "good", "strong", "exceptional", "spike"]
        mult = cfg.roi_tier_multipliers[0]

        for i, pct in enumerate(cfg.roi_tier_percentiles):
            threshold = percentiles.get(pct, 0)
            if cr < threshold:
                mult = cfg.roi_tier_multipliers[i]
                tier_label = tier_labels[i] if i < len(tier_labels) else f"tier_{i}"
                break
        else:
            # Above highest percentile
            mult = cfg.roi_tier_multipliers[-1]
            tier_label = tier_labels[-1] if len(tier_labels) > len(cfg.roi_tier_percentiles) else f"tier_{len(cfg.roi_tier_percentiles)}"

        # Floor at 1.0: never reduce below flat
        mult = max(1.0, mult)

        info = {
            "roi_tier": tier_label,
            "proposal_cr": round(cr, 6),
            "proposal_dte": dte,
            "bucket_used": bucket_used,
            "percentile_rank": _find_percentile_rank(cr, percentiles, cfg.roi_tier_percentiles),
        }
        return mult, info

    def _compute_opportunity_multiplier_legacy(self, proposals: List[Proposal]) -> float:
        """Legacy opportunity scaling: simple best_cr / median_cr ratio.

        Used when roi_tier_enabled=False.
        """
        cfg = self.config
        if not cfg.opportunity_scaling_enabled or not proposals or self._median_cr is None:
            return 1.0

        best_cr = max(p.credit_risk_ratio for p in proposals)
        if self._median_cr <= 0:
            return 1.0

        ratio = best_cr / self._median_cr
        return max(1.0, min(cfg.opportunity_max_multiplier, ratio))

    def _compute_momentum_multiplier(self, trigger_context: Any) -> float:
        """Boost budget when intraday move exceeds threshold."""
        cfg = self.config
        if not cfg.momentum_enabled or trigger_context is None:
            return 1.0

        intraday_return = getattr(trigger_context, "intraday_return", None)
        if intraday_return is None:
            return 1.0

        if abs(intraday_return) > cfg.momentum_threshold:
            return cfg.momentum_boost
        return 1.0

    def _compute_time_weight(self) -> float:
        """Boost budget later in the day (for 0DTE, less time for reversal = safer).

        Returns >= 1.0 always. Early intervals get 1.0 (flat budget).
        Late intervals get up to time_weight_late_factor boost.
        """
        cfg = self.config
        if not cfg.time_weight_enabled:
            return 1.0

        # Progress through the day: 0.0 = first interval, 1.0 = last
        progress = self.intervals_elapsed / max(1, self.total_intervals - 1)

        if cfg.time_weight_curve == "linear":
            # Linear ramp: 1.0 at start → late_factor at end
            return 1.0 + (cfg.time_weight_late_factor - 1.0) * progress

        elif cfg.time_weight_curve == "exponential":
            # Exponential: slow growth early, fast late. 1.0 → late_factor
            return 1.0 * (cfg.time_weight_late_factor ** progress)

        elif cfg.time_weight_curve == "step":
            # Step: 1.0 for first half, late_factor for second half
            if progress < 0.5:
                return 1.0
            return cfg.time_weight_late_factor

        return 1.0

    def _compute_contract_multiplier(self, proposals: List[Proposal]) -> float:
        """Compute contract scaling multiplier based on best proposal quality.

        When ROI tiers are enabled, uses the same per-DTE percentile logic.
        Otherwise falls back to legacy median ratio.
        """
        cfg = self.config
        if not cfg.contract_scaling_enabled or not proposals:
            return 1.0

        if cfg.roi_tier_enabled and cfg.roi_mode == "fixed_roi":
            mult, _ = self._compute_fixed_roi_multiplier(proposals)
            return max(1.0, min(cfg.contract_max_multiplier, mult))

        if cfg.roi_tier_enabled and self._global_percentiles:
            mult, _ = self._compute_roi_tier_multiplier(proposals)
            return max(1.0, min(cfg.contract_max_multiplier, mult))

        # Legacy: simple ratio (needs historical data)
        if self._median_cr is None or self._median_cr <= 0:
            return 1.0
        best_cr = max(p.credit_risk_ratio for p in proposals)

        ratio = best_cr / self._median_cr
        mult = max(1.0, min(cfg.contract_max_multiplier, ratio))
        return mult


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------

_DTE_BUCKET_MAP = {0: "0", 1: "1", 2: "2"}


def _dte_to_bucket(dte: int) -> str:
    """Map a DTE value to its bucket key."""
    if dte in _DTE_BUCKET_MAP:
        return _DTE_BUCKET_MAP[dte]
    if 3 <= dte <= 5:
        return "3-5"
    if 6 <= dte <= 10:
        return "6-10"
    # Out of range — closest bucket
    if dte < 0:
        return "0"
    return "6-10"


def _percentile(values: List[float], pct: float) -> float:
    """Compute percentile using linear interpolation (matches numpy)."""
    if not values:
        return 0.0
    sorted_vals = sorted(values)
    n = len(sorted_vals)
    if n == 1:
        return sorted_vals[0]
    # Linear interpolation (numpy default method)
    k = (pct / 100.0) * (n - 1)
    f = math.floor(k)
    c = min(f + 1, n - 1)
    d = k - f
    return sorted_vals[f] + d * (sorted_vals[c] - sorted_vals[f])


def _find_percentile_rank(
    cr: float,
    percentiles: Dict[float, float],
    breakpoints: List[float],
) -> str:
    """Return a human-readable percentile rank string like '<P50' or 'P75-P90'."""
    if not percentiles or not breakpoints:
        return ""
    for i, pct in enumerate(breakpoints):
        threshold = percentiles.get(pct, 0)
        if cr < threshold:
            if i == 0:
                return f"<P{int(pct)}"
            return f"P{int(breakpoints[i-1])}-P{int(pct)}"
    return f">P{int(breakpoints[-1])}"
