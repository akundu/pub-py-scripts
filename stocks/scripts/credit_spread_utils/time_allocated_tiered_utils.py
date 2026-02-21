"""
Time-Allocated Tiered Strategy utilities.

Extends the tiered strategy with time-based capital allocation across hourly
windows (6am-9:30am PST), tier priority deployment (T3->T2->T1 based on ROI),
directional bias (put/call weighting based on day direction), and slope-based
entry timing (wait for momentum to flatten before deploying).

Key differences from standard Tiered:
- Capital allocated across hourly windows, not all at once
- Tiers deploy in priority order (safest first: T3 -> T2 -> T1)
- Slope detection prevents entry while price is trending strongly
- Direction bias allocates more capital to the favored side
- Unused budget carries forward with configurable decay
"""

from dataclasses import dataclass, field
from datetime import datetime
from math import floor
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import json
import logging

import pandas as pd

from .scale_in_utils import calculate_layer_pnl, check_breach


@dataclass
class TierConfig:
    """Configuration for a single investment tier."""
    level: int                        # 1, 2, or 3
    percent_beyond: float             # e.g., 0.025 for P97
    spread_width: float               # e.g., 40 for T3
    roi_threshold: float              # e.g., 0.035 for T3
    max_cumulative_budget_pct: float  # 0.65 for T3, 0.95 for T2, 1.0 for T1
    daily_capital_limit: Optional[float] = None  # Max capital per day for this tier


@dataclass
class HourlyWindowConfig:
    """Configuration for a single hourly deployment window."""
    label: str                    # "6am", "7am", etc.
    start_hour_pst: int
    end_hour_pst: int
    end_minute_pst: int           # 0 for full hours, 30 for 9:00-9:30
    budget_pct: float             # Fraction of total_capital (computed for 9am)
    start_minute_pst: int = 0     # Minute offset within start_hour (e.g., 30 for 6:30am)


@dataclass
class SlopeConfig:
    """Configuration for slope-based entry timing."""
    lookback_bars: int = 5                  # Number of 5-min bars for average slope
    flatten_ratio_threshold: float = 0.4    # |instant_slope / avg_slope| must be < this
    min_directional_move_pct: float = 0.0005  # Below this, consider market "flat"
    require_double_flatten: bool = False    # Require 2 consecutive flattened bars
    skip_slope: bool = False               # Skip slope detection entirely (deploy immediately)


@dataclass
class ClosePredictorIntegrationConfig:
    """Configuration for integrating close predictor confidence into tier deployment.

    When enabled, the close predictor's confidence level and band width
    dynamically adjust ROI thresholds and budget scaling:
    - HIGH confidence (narrow bands, later in day) -> lower ROI thresholds, more aggressive
    - LOW confidence (wide bands, early morning) -> higher ROI thresholds, more conservative
    """
    enabled: bool = False
    band_level: str = "P95"                    # Which band to check (P95/P97/P98/P99)
    confidence_roi_multipliers: Dict[str, float] = field(default_factory=lambda: {
        "HIGH": 0.70, "MEDIUM": 0.90, "LOW": 1.10, "VERY_LOW": 1.40
    })
    budget_scale_clamp: Tuple[float, float] = field(default_factory=lambda: (0.5, 1.5))
    time_of_day_penalties: Dict[str, float] = field(default_factory=lambda: {
        "6am": 0.15, "7am": 0.10, "8am": 0.05, "9am": 0.0
    })
    band_strike_validation: bool = True        # Require short strike outside predicted band
    per_window_band_levels: Dict[str, str] = field(default_factory=dict)
    # Per-window band level overrides, e.g. {"6am": "P99", "7am": "P99", "8am": "P98", "9am": "P97"}
    # When empty, falls back to `band_level` for all windows.
    dynamic_percent_beyond: bool = False
    # When True, derive tier percent_beyond dynamically from predicted band width
    # instead of using static values from config. Tier level scaling:
    #   T3 (safest): 100% of band half-width (default)
    #   T2: 85% of band half-width
    #   T1 (riskiest): 70% of band half-width
    tier_level_scale: Dict[str, float] = field(default_factory=lambda: {
        "3": 1.0, "2": 0.85, "1": 0.70
    })
    # Configurable scaling of band half-width per tier level.
    # Lower values = strikes closer to ATM = more premium = more deployments.
    # E.g. {"3": 0.60, "2": 0.50, "1": 0.40} for more aggressive strike placement.


@dataclass
class TimeAllocatedTieredConfig:
    """Full configuration for the time-allocated tiered strategy."""
    enabled: bool = True
    total_capital: float = 500000.0
    ticker: str = "NDX"
    equities_dir: str = "equities_output"
    hourly_windows: List[HourlyWindowConfig] = field(default_factory=list)
    put_tiers: List[TierConfig] = field(default_factory=list)
    call_tiers: List[TierConfig] = field(default_factory=list)
    slope_config: SlopeConfig = field(default_factory=SlopeConfig)
    carry_forward_decay: float = 0.5
    direction_priority_split: float = 0.70
    max_concurrent_exposure: Optional[float] = None
    profit_targets: Dict[str, float] = field(default_factory=dict)
    close_predictor_config: Optional[ClosePredictorIntegrationConfig] = None

    @classmethod
    def from_file(cls, path: str) -> 'TimeAllocatedTieredConfig':
        """Load config from JSON file."""
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"Time-allocated tiered config file not found: {path}")
        with open(p, 'r') as f:
            config = json.load(f)
        return cls.from_dict(config)

    @classmethod
    def from_dict(cls, d: dict) -> 'TimeAllocatedTieredConfig':
        """Create config from dictionary (parsed JSON)."""
        # Parse hourly windows
        hourly_windows = []
        for w in d.get('hourly_windows', []):
            budget_pct = w.get('budget_pct', 0.0)
            if isinstance(budget_pct, str) and budget_pct == 'remainder':
                budget_pct = -1.0  # sentinel: computed later in validate()
            hourly_windows.append(HourlyWindowConfig(
                label=w['label'],
                start_hour_pst=w['start_hour_pst'],
                end_hour_pst=w['end_hour_pst'],
                end_minute_pst=w.get('end_minute_pst', 0),
                budget_pct=float(budget_pct),
                start_minute_pst=w.get('start_minute_pst', 0),
            ))

        # Parse tiers
        tiers_config = d.get('tiers', {})
        put_tiers = []
        for t in tiers_config.get('put', []):
            put_tiers.append(TierConfig(
                level=t['level'],
                percent_beyond=t['percent_beyond'],
                spread_width=t['spread_width'],
                roi_threshold=t['roi_threshold'],
                max_cumulative_budget_pct=t['max_cumulative_budget_pct'],
                daily_capital_limit=t.get('daily_capital_limit'),
            ))

        call_tiers = []
        for t in tiers_config.get('call', []):
            call_tiers.append(TierConfig(
                level=t['level'],
                percent_beyond=t['percent_beyond'],
                spread_width=t['spread_width'],
                roi_threshold=t['roi_threshold'],
                max_cumulative_budget_pct=t['max_cumulative_budget_pct'],
                daily_capital_limit=t.get('daily_capital_limit'),
            ))

        # Parse slope config
        slope_dict = d.get('slope_detection', {})
        slope_config = SlopeConfig(
            lookback_bars=slope_dict.get('lookback_bars', 5),
            flatten_ratio_threshold=slope_dict.get('flatten_ratio_threshold', 0.4),
            min_directional_move_pct=slope_dict.get('min_directional_move_pct', 0.0005),
            require_double_flatten=slope_dict.get('require_double_flatten', False),
            skip_slope=slope_dict.get('skip_slope', False),
        )

        # Parse close predictor integration config
        cp_dict = d.get('close_predictor_integration')
        close_predictor_config = None
        if cp_dict is not None:
            budget_clamp = cp_dict.get('budget_scale_clamp', [0.5, 1.5])
            if isinstance(budget_clamp, (list, tuple)) and len(budget_clamp) == 2:
                budget_clamp = (float(budget_clamp[0]), float(budget_clamp[1]))
            else:
                budget_clamp = (0.5, 1.5)
            close_predictor_config = ClosePredictorIntegrationConfig(
                enabled=cp_dict.get('enabled', False),
                band_level=cp_dict.get('band_level', 'P95'),
                confidence_roi_multipliers=cp_dict.get('confidence_roi_multipliers', {
                    "HIGH": 0.70, "MEDIUM": 0.90, "LOW": 1.10, "VERY_LOW": 1.40
                }),
                budget_scale_clamp=budget_clamp,
                time_of_day_penalties=cp_dict.get('time_of_day_penalties', {
                    "6am": 0.15, "7am": 0.10, "8am": 0.05, "9am": 0.0
                }),
                band_strike_validation=cp_dict.get('band_strike_validation', True),
                per_window_band_levels=cp_dict.get('per_window_band_levels', {}),
                dynamic_percent_beyond=cp_dict.get('dynamic_percent_beyond', False),
                tier_level_scale=cp_dict.get('tier_level_scale', {"3": 1.0, "2": 0.85, "1": 0.70}),
            )

        config = cls(
            enabled=d.get('enabled', True),
            total_capital=float(d.get('total_capital', 500000.0)),
            ticker=d.get('ticker', 'NDX'),
            equities_dir=d.get('equities_dir', 'equities_output'),
            hourly_windows=hourly_windows,
            put_tiers=put_tiers,
            call_tiers=call_tiers,
            slope_config=slope_config,
            carry_forward_decay=float(d.get('carry_forward_decay', 0.5)),
            direction_priority_split=float(d.get('direction_priority_split', 0.70)),
            max_concurrent_exposure=d.get('max_concurrent_exposure'),
            profit_targets=d.get('profit_targets', {}),
            close_predictor_config=close_predictor_config,
        )

        # Resolve remainder budget
        config._resolve_remainder_budget()
        return config

    def _resolve_remainder_budget(self):
        """Compute the remainder window's budget_pct."""
        explicit_sum = 0.0
        remainder_idx = None
        for i, w in enumerate(self.hourly_windows):
            if w.budget_pct < 0:
                remainder_idx = i
            else:
                explicit_sum += w.budget_pct
        if remainder_idx is not None:
            self.hourly_windows[remainder_idx].budget_pct = max(0.0, 1.0 - explicit_sum)

    def get_tiers(self, option_type: str) -> List[TierConfig]:
        """Get tiers for the specified option type, sorted by level descending (T3 first)."""
        if option_type.lower() == 'put':
            tiers = list(self.put_tiers)
        elif option_type.lower() == 'call':
            tiers = list(self.call_tiers)
        else:
            return []
        return sorted(tiers, key=lambda t: t.level, reverse=True)

    def validate(self) -> bool:
        """Check that budget percentages of all windows sum to <= 1.0."""
        total = sum(w.budget_pct for w in self.hourly_windows)
        if total > 1.001:
            raise ValueError(
                f"Window budget percentages sum to {total:.4f}, must be <= 1.0"
            )
        return True


@dataclass
class TierPosition:
    """A single deployed position within a window."""
    tier_level: int
    option_type: str
    short_strike: float
    long_strike: float
    spread_width: float
    num_contracts: int
    capital_at_risk: float        # num_contracts * spread_width * 100
    initial_credit_per_share: float
    initial_credit_total: float
    roi: float                    # credit_total / capital_at_risk
    entry_timestamp: Optional[datetime] = None
    activated: bool = True
    actual_pnl_per_share: Optional[float] = None
    actual_pnl_total: Optional[float] = None


@dataclass
class WindowDeployment:
    """Tracks deployment for a single hourly window."""
    window_label: str
    budget_dollars: float         # Original budget + any carry-forward
    deployed_positions: List[TierPosition] = field(default_factory=list)
    remaining_budget: float = 0.0
    slope_blocked: bool = False
    direction_blocked: bool = False

    @property
    def total_deployed(self) -> float:
        return sum(p.capital_at_risk for p in self.deployed_positions)

    @property
    def total_credit(self) -> float:
        return sum(p.initial_credit_total for p in self.deployed_positions)


@dataclass
class TimeAllocatedTradeState:
    """Full state for a day's trading."""
    trading_date: Optional[datetime] = None
    option_type: str = 'put'
    prev_close: float = 0.0
    total_capital: float = 0.0
    window_deployments: List[WindowDeployment] = field(default_factory=list)

    @property
    def total_deployed(self) -> float:
        return sum(wd.total_deployed for wd in self.window_deployments)

    @property
    def total_credit(self) -> float:
        return sum(wd.total_credit for wd in self.window_deployments)

    @property
    def total_capital_at_risk(self) -> float:
        return self.total_deployed

    @property
    def total_pnl(self) -> Optional[float]:
        all_positions = []
        for wd in self.window_deployments:
            all_positions.extend(wd.deployed_positions)
        if not all_positions:
            return None
        if not any(p.actual_pnl_total is not None for p in all_positions):
            return None
        return sum(p.actual_pnl_total or 0 for p in all_positions
                   if p.actual_pnl_total is not None)


# ============================================================================
# Core Algorithm Functions
# ============================================================================

def check_slope_flattened(
    intraday_df: pd.DataFrame,
    target_timestamp: datetime,
    option_type: str,
    slope_config: SlopeConfig,
) -> Tuple[bool, Dict[str, float]]:
    """
    Check if price momentum has flattened near target_timestamp.

    Uses 5-min bars from equities_output:
    - instant_slope = most recent bar's return
    - avg_slope = mean of last N bars' returns
    - flatten_ratio = |instant_slope| / |avg_slope|

    For PUTS: price was falling -> flattened when instant_slope >= 0 or flatten_ratio < threshold
    For CALLS: price was rising -> flattened when instant_slope <= 0 or flatten_ratio < threshold

    Returns:
        Tuple of (is_flattened, info_dict with slope details)
    """
    info = {
        'instant_slope': 0.0,
        'avg_slope': 0.0,
        'flatten_ratio': 0.0,
        'flattened': False,
    }

    if intraday_df is None or intraday_df.empty:
        # No data -> consider flattened (don't block)
        info['flattened'] = True
        return True, info

    # Get bars up to target_timestamp
    ts = pd.Timestamp(target_timestamp)
    if ts.tzinfo is None:
        ts = ts.tz_localize('UTC')
    else:
        ts = ts.tz_convert('UTC')

    mask = intraday_df['timestamp'] <= ts
    bars_before = intraday_df[mask]

    needed = slope_config.lookback_bars + 1
    if len(bars_before) < needed:
        # Have data but not enough bars yet -> not flattened (wait for more data)
        info['flattened'] = False
        return False, info

    # Take the last (lookback_bars + 1) bars to compute lookback_bars returns
    recent = bars_before.tail(needed)
    closes = recent['close'].values

    # Compute per-bar returns
    returns = []
    for i in range(1, len(closes)):
        if closes[i - 1] != 0:
            returns.append((closes[i] - closes[i - 1]) / closes[i - 1])
        else:
            returns.append(0.0)

    if not returns:
        info['flattened'] = True
        return True, info

    instant_slope = returns[-1]
    avg_slope = sum(returns) / len(returns)
    info['instant_slope'] = instant_slope
    info['avg_slope'] = avg_slope

    # If average movement is negligible, market is flat
    if abs(avg_slope) < slope_config.min_directional_move_pct:
        info['flatten_ratio'] = 0.0
        info['flattened'] = True
        return True, info

    flatten_ratio = abs(instant_slope) / abs(avg_slope) if abs(avg_slope) > 0 else 0.0
    info['flatten_ratio'] = flatten_ratio

    # Direction-specific logic
    if option_type.lower() == 'put':
        if avg_slope < 0:
            # Price was falling: flattened if instant slope turned positive or ratio low
            flattened = (instant_slope >= 0) or (flatten_ratio < slope_config.flatten_ratio_threshold)
        else:
            # Price was rising: no downward pressure -> always flattened for puts
            flattened = True
    else:  # call
        if avg_slope > 0:
            # Price was rising: flattened if instant slope turned negative or ratio low
            flattened = (instant_slope <= 0) or (flatten_ratio < slope_config.flatten_ratio_threshold)
        else:
            # Price was falling: no upward pressure -> always flattened for calls
            flattened = True

    # Double-flatten: check at t and t-1
    if slope_config.require_double_flatten and flattened and len(returns) >= 2:
        prev_instant = returns[-2]
        prev_ratio = abs(prev_instant) / abs(avg_slope) if abs(avg_slope) > 0 else 0.0

        if option_type.lower() == 'put':
            if avg_slope < 0:
                prev_flat = (prev_instant >= 0) or (prev_ratio < slope_config.flatten_ratio_threshold)
            else:
                prev_flat = True
        else:
            if avg_slope > 0:
                prev_flat = (prev_instant <= 0) or (prev_ratio < slope_config.flatten_ratio_threshold)
            else:
                prev_flat = True

        flattened = flattened and prev_flat

    info['flattened'] = flattened
    return flattened, info


def check_direction_bias(
    current_price: float,
    prev_close: float,
    option_type: str,
) -> bool:
    """
    Returns True if option_type aligns with day direction.

    PUT -> True when current_price < prev_close (down day)
    CALL -> True when current_price > prev_close (up day)
    If prices are equal, both are considered aligned.
    """
    if option_type.lower() == 'put':
        return current_price <= prev_close
    else:
        return current_price >= prev_close


def calculate_contracts_for_budget(
    available_budget: float,
    spread_width: float,
    credit_per_share: float,
) -> Tuple[int, float, float]:
    """
    Convert dollar budget to contract count.

    Args:
        available_budget: Dollar amount available for deployment
        spread_width: Spread width in points
        credit_per_share: Credit received per share

    Returns:
        (num_contracts, actual_capital_used, total_credit)
    """
    capital_per_contract = spread_width * 100  # gross collateral
    if capital_per_contract <= 0:
        return (0, 0.0, 0.0)

    num_contracts = floor(available_budget / capital_per_contract)
    if num_contracts <= 0:
        return (0, 0.0, 0.0)

    actual_capital = num_contracts * capital_per_contract
    total_credit = credit_per_share * num_contracts * 100
    return (num_contracts, actual_capital, total_credit)


def find_best_spread_for_tier(
    interval_results: List[Dict],
    tier: TierConfig,
    option_type: str,
    prev_close: float,
) -> Optional[Dict]:
    """
    From the interval results in a window, find the best spread
    matching this tier's percent_beyond.

    Returns the interval result with highest credit that passes
    the tier's constraints, or None if no matching spread found.
    """
    best_result = None
    best_credit = 0.0

    for result in interval_results:
        best_spread = result.get('best_spread')
        if not best_spread:
            continue

        net_credit = best_spread.get('net_credit', 0)
        width = best_spread.get('width', 0)

        # Check that spread width is within tier's range
        if width > 0 and width > tier.spread_width * 1.5:
            continue

        # Check credit is positive
        if net_credit <= 0:
            continue

        if net_credit > best_credit:
            best_credit = net_credit
            best_result = result

    return best_result


def allocate_single_window(
    window_config: HourlyWindowConfig,
    window_budget: float,
    window_intervals: List[Dict],
    tiers: List[TierConfig],
    option_type: str,
    prev_close: float,
    intraday_df: Optional[pd.DataFrame],
    slope_config: SlopeConfig,
    max_exposure_remaining: Optional[float],
    logger: Optional[logging.Logger] = None,
    tier_deployed: Optional[Dict[int, float]] = None,
) -> WindowDeployment:
    """
    Core allocation for one hourly window.

    1. Slope check (find first interval where slope is flattened)
    2. Deploy T3 -> T2 -> T1 with budget ceilings and ROI thresholds
    """
    deployment = WindowDeployment(
        window_label=window_config.label,
        budget_dollars=window_budget,
    )

    if logger:
        logger.info(f"  Window {window_config.label}: Processing with budget=${window_budget:,.2f}, {len(window_intervals)} intervals")

    if window_budget <= 0:
        deployment.remaining_budget = 0.0
        if logger:
            logger.info(f"  Window {window_config.label}: Skipping - no budget")
        return deployment

    # Cap by max exposure remaining
    effective_budget = window_budget
    if max_exposure_remaining is not None:
        effective_budget = min(effective_budget, max(0.0, max_exposure_remaining))
        if logger:
            logger.info(f"  Window {window_config.label}: Effective budget capped at ${effective_budget:,.2f} (max exposure)")

    if effective_budget <= 0:
        deployment.remaining_budget = window_budget
        if logger:
            logger.info(f"  Window {window_config.label}: Skipping - effective budget is 0")
        return deployment

    if not window_intervals:
        deployment.remaining_budget = window_budget
        deployment.slope_blocked = True
        if logger:
            logger.info(f"  Window {window_config.label}: Skipping - no intervals available")
        return deployment

    # Slope check: find first interval where slope has flattened
    # When skip_slope is True, bypass slope detection and deploy immediately
    if slope_config.skip_slope:
        deploy_results = window_intervals
        if logger:
            logger.info(f"  Window {window_config.label}: Slope check SKIPPED (skip_slope=True)")
    else:
        deploy_results = None
        slope_checks = 0
        for idx, result in enumerate(window_intervals):
            ts = result.get('timestamp')
            if ts is None:
                continue

            slope_checks += 1
            if intraday_df is not None and not intraday_df.empty:
                flattened, slope_info = check_slope_flattened(
                    intraday_df, ts, option_type, slope_config
                )
                if not flattened:
                    if logger:
                        logger.debug(
                            f"    Window {window_config.label}: slope not flattened at {ts}, "
                            f"instant={slope_info.get('instant_slope', 0):.6f}, "
                            f"avg={slope_info.get('avg_slope', 0):.6f}, "
                            f"ratio={slope_info.get('flatten_ratio', 0):.4f}"
                        )
                    continue
                else:
                    if logger:
                        logger.info(
                            f"  Window {window_config.label}: Slope FLATTENED at {ts} after {slope_checks} checks, "
                            f"instant={slope_info.get('instant_slope', 0):.6f}, "
                            f"avg={slope_info.get('avg_slope', 0):.6f}, "
                            f"ratio={slope_info.get('flatten_ratio', 0):.4f}"
                        )

            # Only use intervals from the flattened point onward for spread selection
            deploy_results = window_intervals[idx:]
            break

    if deploy_results is None:
        deployment.slope_blocked = True
        deployment.remaining_budget = window_budget
        if logger:
            logger.info(f"  Window {window_config.label}: BLOCKED - slope never flattened (checked {slope_checks} intervals)")
        return deployment

    # Deploy tiers: T3 -> T2 -> T1 (tiers already sorted by level descending)
    cumulative_deployed = 0.0

    if logger:
        logger.info(f"  Window {window_config.label}: Attempting tier deployment (T3→T2→T1)...")

    for tier in tiers:
        ceiling = effective_budget * tier.max_cumulative_budget_pct
        available = ceiling - cumulative_deployed

        # Apply per-tier daily capital limit
        if tier.daily_capital_limit is not None and tier_deployed is not None:
            already_deployed = tier_deployed.get(tier.level, 0.0)
            tier_remaining = tier.daily_capital_limit - already_deployed
            if tier_remaining <= 0:
                if logger:
                    logger.info(
                        f"    T{tier.level}: Skipping - daily capital limit "
                        f"${tier.daily_capital_limit:,.2f} reached "
                        f"(deployed=${already_deployed:,.2f})"
                    )
                continue
            available = min(available, tier_remaining)

        if logger:
            logger.debug(
                f"    T{tier.level}: ceiling=${ceiling:,.2f} "
                f"({tier.max_cumulative_budget_pct*100:.0f}% of ${effective_budget:,.2f}), "
                f"available=${available:,.2f}, "
                f"width={tier.spread_width}pts, "
                f"roi_threshold={tier.roi_threshold*100:.2f}%"
            )

        if available <= 0:
            if logger:
                logger.debug(f"    T{tier.level}: Skipping - no budget available")
            continue

        # Find spread matching this tier
        spread_result = find_best_spread_for_tier(
            deploy_results, tier, option_type, prev_close
        )
        if spread_result is None:
            if logger:
                logger.debug(f"    T{tier.level}: No matching spread found")
            continue

        best_spread = spread_result.get('best_spread', {})
        credit_per_share = best_spread.get('net_credit', 0)
        spread_width_actual = best_spread.get('width', 0)

        if logger:
            logger.debug(
                f"    T{tier.level}: Found spread with credit=${credit_per_share:.2f}/share, "
                f"width={spread_width_actual:.0f}pts"
            )

        if credit_per_share <= 0:
            if logger:
                logger.debug(f"    T{tier.level}: Skipping - no credit")
            continue

        # ROI check
        capital_per_contract = tier.spread_width * 100
        if capital_per_contract <= 0:
            if logger:
                logger.debug(f"    T{tier.level}: Skipping - invalid spread width")
            continue
        credit_per_contract = credit_per_share * 100
        roi = credit_per_contract / capital_per_contract

        if roi < tier.roi_threshold:
            if logger:
                logger.info(
                    f"    T{tier.level}: ROI check FAILED - "
                    f"ROI={roi*100:.2f}% < threshold={tier.roi_threshold*100:.2f}% "
                    f"(credit=${credit_per_contract:,.2f}, capital=${capital_per_contract:,.2f})"
                )
            continue

        if logger:
            logger.info(
                f"    T{tier.level}: ROI check PASSED - "
                f"ROI={roi*100:.2f}% >= threshold={tier.roi_threshold*100:.2f}%"
            )

        # Contract sizing
        num_contracts = floor(available / capital_per_contract)
        if num_contracts <= 0:
            if logger:
                logger.debug(
                    f"    T{tier.level}: Skipping - not enough budget for even 1 contract "
                    f"(need ${capital_per_contract:,.2f}, have ${available:,.2f})"
                )
            continue

        actual_capital = num_contracts * capital_per_contract
        total_credit = credit_per_share * num_contracts * 100
        cumulative_deployed += actual_capital

        # Update cross-window per-tier tracking
        if tier_deployed is not None:
            tier_deployed[tier.level] = tier_deployed.get(tier.level, 0.0) + actual_capital

        if logger:
            logger.info(
                f"    T{tier.level}: ✓ DEPLOYING {num_contracts} contracts, "
                f"capital=${actual_capital:,.2f}, credit=${total_credit:,.2f}"
            )

        # Calculate strikes
        if option_type.lower() == 'put':
            short_strike = prev_close * (1 - tier.percent_beyond)
            long_strike = short_strike - tier.spread_width
        else:
            short_strike = prev_close * (1 + tier.percent_beyond)
            long_strike = short_strike + tier.spread_width

        position = TierPosition(
            tier_level=tier.level,
            option_type=option_type,
            short_strike=round(short_strike, 2),
            long_strike=round(long_strike, 2),
            spread_width=tier.spread_width,
            num_contracts=num_contracts,
            capital_at_risk=actual_capital,
            initial_credit_per_share=credit_per_share,
            initial_credit_total=total_credit,
            roi=roi,
            entry_timestamp=spread_result.get('timestamp'),
            activated=True,
        )
        deployment.deployed_positions.append(position)

    deployment.remaining_budget = effective_budget - cumulative_deployed

    if logger:
        logger.info(
            f"  Window {window_config.label}: COMPLETE - "
            f"Deployed ${cumulative_deployed:,.2f}, "
            f"Remaining ${deployment.remaining_budget:,.2f}, "
            f"Positions: {len(deployment.deployed_positions)}"
        )

    return deployment


def allocate_across_windows(
    trading_date: Optional[datetime],
    option_type: str,
    prev_close: float,
    config: TimeAllocatedTieredConfig,
    window_intervals: Dict[str, List[Dict]],
    intraday_df: Optional[pd.DataFrame],
    logger: Optional[logging.Logger] = None,
) -> TimeAllocatedTradeState:
    """
    Walk through windows chronologically, allocating capital.

    Args:
        trading_date: Trading date
        option_type: 'put' or 'call'
        prev_close: Previous day's close
        config: Strategy configuration
        window_intervals: label -> interval results
        intraday_df: Intraday 5-min bars for slope detection
        logger: Optional logger

    Returns:
        TimeAllocatedTradeState with all window deployments
    """
    if logger:
        logger.info(f"\n{'='*80}")
        logger.info(f"TIME-ALLOCATED TIERED ALLOCATION")
        logger.info(f"Date: {trading_date}, Type: {option_type.upper()}, Prev Close: ${prev_close:.2f}")
        logger.info(f"Total Capital: ${config.total_capital:,.2f}")
        logger.info(f"Windows: {[w.label for w in config.hourly_windows]}")
        logger.info(f"Slope Detection: {'DISABLED' if config.slope_config.skip_slope else 'ENABLED'}")
        logger.info(f"{'='*80}")

    trade_state = TimeAllocatedTradeState(
        trading_date=trading_date,
        option_type=option_type,
        prev_close=prev_close,
        total_capital=config.total_capital,
    )

    # Determine direction weight
    # Use the latest price from the first non-empty window's intervals
    latest_price = prev_close
    for w in config.hourly_windows:
        intervals = window_intervals.get(w.label, [])
        if intervals:
            # Try to get current price from first interval
            first = intervals[0]
            best_spread = first.get('best_spread', {})
            if best_spread:
                # Use the underlying price from the interval if available
                price = first.get('current_close', first.get('prev_close', prev_close))
                if price:
                    latest_price = price
            break

    is_down_day = latest_price < prev_close
    if option_type.lower() == 'put':
        direction_weight = config.direction_priority_split if is_down_day else (1.0 - config.direction_priority_split)
    else:
        direction_weight = config.direction_priority_split if not is_down_day else (1.0 - config.direction_priority_split)

    if logger:
        logger.info(
            f"\nDirection Bias: Latest=${latest_price:.2f}, Prev=${prev_close:.2f}, "
            f"{'DOWN' if is_down_day else 'UP'} day, "
            f"{option_type.upper()} weight={direction_weight*100:.0f}%"
        )

    carry_forward = 0.0
    tiers = config.get_tiers(option_type)
    cumulative_exposure = 0.0
    tier_deployed: Dict[int, float] = {}  # Track per-tier capital across windows

    if logger:
        logger.info(f"\nTier Configuration ({len(tiers)} tiers):")
        for t in tiers:
            logger.info(
                f"  T{t.level}: width={t.spread_width}pts, "
                f"roi>={t.roi_threshold*100:.2f}%, "
                f"budget<={t.max_cumulative_budget_pct*100:.0f}%"
            )

    for window_config in config.hourly_windows:
        base_budget = config.total_capital * window_config.budget_pct * direction_weight
        window_budget = base_budget + carry_forward

        # Cap by max_concurrent_exposure if set
        max_exposure_remaining = None
        if config.max_concurrent_exposure is not None:
            max_exposure_remaining = config.max_concurrent_exposure - cumulative_exposure

        intervals = window_intervals.get(window_config.label, [])

        deployment = allocate_single_window(
            window_config=window_config,
            window_budget=window_budget,
            window_intervals=intervals,
            tiers=tiers,
            option_type=option_type,
            prev_close=prev_close,
            intraday_df=intraday_df,
            slope_config=config.slope_config,
            max_exposure_remaining=max_exposure_remaining,
            logger=logger,
            tier_deployed=tier_deployed,
        )

        trade_state.window_deployments.append(deployment)

        # Track cumulative exposure
        cumulative_exposure += deployment.total_deployed

        # Carry forward: unused budget * decay factor
        unused = deployment.remaining_budget
        carry_forward = unused * config.carry_forward_decay

        if logger and carry_forward > 0:
            logger.info(
                f"  → Carry-forward to next window: ${carry_forward:,.2f} "
                f"(unused=${unused:,.2f}, decay={config.carry_forward_decay})"
            )

    if logger:
        logger.info(f"\n{'='*80}")
        logger.info(f"ALLOCATION COMPLETE")
        logger.info(f"Total Deployed: ${cumulative_exposure:,.2f}")
        logger.info(f"Total Positions: {sum(len(wd.deployed_positions) for wd in trade_state.window_deployments)}")
        logger.info(f"{'='*80}\n")

    return trade_state


def calculate_time_allocated_pnl(
    trade_state: TimeAllocatedTradeState,
    close_price: float,
) -> TimeAllocatedTradeState:
    """
    Calculate P&L for all deployed positions using intrinsic value at close.

    Reuses calculate_layer_pnl() from scale_in_utils.py.
    """
    for deployment in trade_state.window_deployments:
        for position in deployment.deployed_positions:
            if not position.activated:
                continue

            position.actual_pnl_per_share = calculate_layer_pnl(
                position.initial_credit_per_share,
                position.short_strike,
                position.long_strike,
                close_price,
                position.option_type,
            )
            position.actual_pnl_total = (
                position.actual_pnl_per_share * position.num_contracts * 100
            )

    return trade_state


def generate_time_allocated_summary(
    trade_state: TimeAllocatedTradeState,
) -> Dict[str, Any]:
    """
    Summary dict with per-window and per-tier breakdowns.
    """
    total_credit = trade_state.total_credit
    total_pnl = trade_state.total_pnl
    total_capital_at_risk = trade_state.total_capital_at_risk

    roi = (total_pnl / total_capital_at_risk * 100) if (
        total_capital_at_risk > 0 and total_pnl is not None
    ) else 0.0

    # Per-window stats
    window_stats = []
    for wd in trade_state.window_deployments:
        w_pnl = None
        if wd.deployed_positions:
            pnls = [p.actual_pnl_total for p in wd.deployed_positions if p.actual_pnl_total is not None]
            if pnls:
                w_pnl = sum(pnls)

        window_stats.append({
            'label': wd.window_label,
            'budget': wd.budget_dollars,
            'deployed': wd.total_deployed,
            'remaining': wd.remaining_budget,
            'credit': wd.total_credit,
            'pnl': w_pnl,
            'slope_blocked': wd.slope_blocked,
            'direction_blocked': wd.direction_blocked,
            'num_positions': len(wd.deployed_positions),
        })

    # Per-tier stats
    tier_stats: Dict[int, Dict[str, Any]] = {}
    for wd in trade_state.window_deployments:
        for pos in wd.deployed_positions:
            level = pos.tier_level
            if level not in tier_stats:
                tier_stats[level] = {
                    'activation_count': 0,
                    'total_capital': 0.0,
                    'total_credit': 0.0,
                    'total_pnl': 0.0,
                }
            tier_stats[level]['activation_count'] += 1
            tier_stats[level]['total_capital'] += pos.capital_at_risk
            tier_stats[level]['total_credit'] += pos.initial_credit_total
            if pos.actual_pnl_total is not None:
                tier_stats[level]['total_pnl'] += pos.actual_pnl_total

    summary: Dict[str, Any] = {
        'trading_date': trade_state.trading_date,
        'option_type': trade_state.option_type,
        'prev_close': trade_state.prev_close,
        'total_capital': trade_state.total_capital,
        'total_capital_at_risk': total_capital_at_risk,
        'total_credit': total_credit,
        'total_pnl': total_pnl,
        'total_max_loss': total_capital_at_risk,
        'roi': roi,
        'window_stats': window_stats,
        'tier_stats': tier_stats,
    }

    return summary


def load_time_allocated_tiered_config(
    config_path: Optional[str],
) -> Optional[TimeAllocatedTieredConfig]:
    """
    Load time-allocated tiered config from file path.

    Args:
        config_path: Path to JSON config file

    Returns:
        TimeAllocatedTieredConfig or None if path is None
    """
    if not config_path:
        return None
    try:
        return TimeAllocatedTieredConfig.from_file(config_path)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        raise ValueError(f"Failed to load time-allocated tiered config: {e}")
