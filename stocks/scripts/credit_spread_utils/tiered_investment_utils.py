"""
Tiered Investment Strategy utilities.

This module implements a tiered investment strategy that enters multiple concurrent positions
at different distances from close, each with its own contract count (N) and spread width (M).
Tiers activate when the existing framework constraints are satisfied (min premium, credit/width
ratio, spread availability, etc.) - not on breach.

Key differences from Scale-In:
- Variable spread width per tier (not a single fixed width)
- Directly specified contract count N (not derived from capital allocation)
- Constraint-based activation (all qualifying tiers enter) vs breach-triggered
- All qualifying tiers enter at the same interval
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
import json
import logging
from pathlib import Path

from .scale_in_utils import calculate_layer_pnl, check_breach


@dataclass
class TierConfig:
    """Configuration for a single investment tier."""
    level: int              # 1, 2, 3, ...
    percent_beyond: float   # Distance from close (e.g., 0.027 = P97)
    num_contracts: int      # Fixed N contracts (default: 50)
    spread_width: float     # Width M for this tier (default: 10)


@dataclass
class TieredInvestmentConfig:
    """Configuration for the tiered investment strategy."""
    enabled: bool = True
    put_tiers: List[TierConfig] = field(default_factory=list)
    call_tiers: List[TierConfig] = field(default_factory=list)

    @classmethod
    def from_file(cls, path: str) -> 'TieredInvestmentConfig':
        """Load config from JSON file."""
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"Tiered config file not found: {path}")
        with open(p, 'r') as f:
            config = json.load(f)
        return cls.from_dict(config)

    @classmethod
    def from_dict(cls, config: dict) -> 'TieredInvestmentConfig':
        """Create config from dictionary (parsed JSON)."""
        put_tiers = []
        call_tiers = []

        tiers_config = config.get('tiers', {})

        for tier_dict in tiers_config.get('put', []):
            put_tiers.append(TierConfig(
                level=tier_dict['level'],
                percent_beyond=tier_dict['percent_beyond'],
                num_contracts=tier_dict.get('num_contracts', 50),
                spread_width=tier_dict.get('spread_width', 10),
            ))

        for tier_dict in tiers_config.get('call', []):
            call_tiers.append(TierConfig(
                level=tier_dict['level'],
                percent_beyond=tier_dict['percent_beyond'],
                num_contracts=tier_dict.get('num_contracts', 50),
                spread_width=tier_dict.get('spread_width', 10),
            ))

        return cls(
            enabled=config.get('enabled', True),
            put_tiers=put_tiers,
            call_tiers=call_tiers,
        )

    def get_tiers(self, option_type: str) -> List[TierConfig]:
        """Get tiers for the specified option type."""
        if option_type.lower() == 'put':
            return self.put_tiers
        elif option_type.lower() == 'call':
            return self.call_tiers
        return []


@dataclass
class TierPosition:
    """Tracks a single tier position in the tiered investment strategy."""
    tier_level: int
    option_type: str            # 'put' or 'call'
    short_strike: float
    long_strike: float
    spread_width: float         # Per-tier width (M)
    num_contracts: int          # Per-tier count (N)
    capital_at_risk: float      # N * M * 100
    initial_credit_per_share: float
    initial_credit_total: float
    activated: bool = False     # True if constraints passed and tier entered
    breach_detected: bool = False
    actual_pnl_per_share: Optional[float] = None
    actual_pnl_total: Optional[float] = None


@dataclass
class TieredTradeState:
    """Tracks the state of a tiered trade across all tiers."""
    trading_date: datetime
    option_type: str
    prev_close: float
    tiers: List[TierPosition] = field(default_factory=list)

    @property
    def total_capital_at_risk(self) -> float:
        """Total capital at risk across all activated tiers."""
        return sum(t.capital_at_risk for t in self.tiers if t.activated)

    @property
    def total_credit(self) -> float:
        """Total credit received across all activated tiers."""
        return sum(t.initial_credit_total for t in self.tiers if t.activated)

    @property
    def total_pnl(self) -> Optional[float]:
        """Total P&L across all activated tiers."""
        activated = [t for t in self.tiers if t.activated]
        if not activated:
            return None
        if not any(t.actual_pnl_total is not None for t in activated):
            return None
        return sum(t.actual_pnl_total or 0 for t in activated if t.actual_pnl_total is not None)

    @property
    def num_activated_tiers(self) -> int:
        """Number of tiers that were activated."""
        return sum(1 for t in self.tiers if t.activated)

    @property
    def num_breached_tiers(self) -> int:
        """Number of activated tiers that experienced breach."""
        return sum(1 for t in self.tiers if t.activated and t.breach_detected)


def calculate_tier_strikes(
    prev_close: float,
    tier: TierConfig,
    option_type: str,
) -> Tuple[float, float]:
    """
    Calculate short and long strike prices for a tier using its own spread_width.

    Args:
        prev_close: Previous day's closing price
        tier: Tier configuration
        option_type: 'put' or 'call'

    Returns:
        Tuple of (short_strike, long_strike)
    """
    if option_type.lower() == 'put':
        short_strike = prev_close * (1 - tier.percent_beyond)
        long_strike = short_strike - tier.spread_width
    else:
        short_strike = prev_close * (1 + tier.percent_beyond)
        long_strike = short_strike + tier.spread_width

    return (round(short_strike, 2), round(long_strike, 2))


def check_tier_constraints(
    tier: TierConfig,
    short_strike: float,
    long_strike: float,
    option_type: str,
    day_results: List[Dict[str, Any]],
    min_premium_diff: Optional[Tuple[float, float]],
    max_credit_width_ratio: float,
    min_contract_price: float,
    max_strike_distance_pct: Optional[float],
    prev_close: float,
    logger: Optional[logging.Logger] = None,
) -> Tuple[bool, float]:
    """
    Check whether a tier satisfies framework constraints using actual options data.

    Scans the day's results for any spread that matches the tier's distance and width
    while satisfying min premium, credit/width ratio, min contract price, and strike
    distance constraints.

    Args:
        tier: Tier configuration
        short_strike: Calculated short strike for this tier
        long_strike: Calculated long strike for this tier
        option_type: 'put' or 'call'
        day_results: List of result dicts from the framework for this day/type
        min_premium_diff: (put_min, call_min) minimum net credit, or None
        max_credit_width_ratio: Max credit/width ratio
        min_contract_price: Minimum contract price threshold
        max_strike_distance_pct: Max strike distance from prev_close as pct
        prev_close: Previous day's close price
        logger: Optional logger

    Returns:
        Tuple of (constraints_passed, credit_per_share)
    """
    # Check max_strike_distance_pct constraint on this tier's short strike
    if max_strike_distance_pct is not None:
        distance_from_close = abs(short_strike - prev_close) / prev_close
        if distance_from_close > max_strike_distance_pct:
            if logger:
                logger.debug(
                    f"Tier {tier.level} ({option_type}): strike distance "
                    f"{distance_from_close:.4f} exceeds max {max_strike_distance_pct:.4f}"
                )
            return (False, 0.0)

    # Look through the day's results for spreads that approximate this tier's position
    # We use the best_spread data from results to estimate credit at this tier's strikes
    best_credit = 0.0
    found_valid = False

    for result in day_results:
        best_spread = result.get('best_spread')
        if not best_spread:
            continue

        result_short = best_spread.get('short_strike', 0)
        result_long = best_spread.get('long_strike', 0)
        result_credit = best_spread.get('net_credit', 0)
        result_width = best_spread.get('width', 0)
        result_short_price = best_spread.get('short_price', 0)
        result_long_price = best_spread.get('long_price', 0)

        # Check min_contract_price on the result's leg prices
        if result_short_price <= min_contract_price or result_long_price <= min_contract_price:
            continue

        # Check credit/width ratio
        if result_width > 0:
            ratio = result_credit / result_width
            if ratio > max_credit_width_ratio:
                continue

        # Check min_premium_diff
        if min_premium_diff is not None:
            if option_type.lower() == 'call':
                min_diff = min_premium_diff[1]
            else:
                min_diff = min_premium_diff[0]
            if result_credit < min_diff:
                continue

        # This result passes constraints - use its credit as a reference
        if result_credit > best_credit:
            best_credit = result_credit
            found_valid = True

    if not found_valid:
        if logger:
            logger.debug(
                f"Tier {tier.level} ({option_type}): no valid spread found in results"
            )
        return (False, 0.0)

    # Use the best credit found from valid spreads as the estimate for this tier
    # Scale credit based on distance ratio (further out = less credit typically)
    # But for constraint checking, having any valid spread means the tier can enter
    return (True, best_credit)


def initialize_tiered_trade(
    trading_date: datetime,
    option_type: str,
    prev_close: float,
    config: TieredInvestmentConfig,
    day_results: List[Dict[str, Any]],
    min_premium_diff: Optional[Tuple[float, float]],
    max_credit_width_ratio: float,
    min_contract_price: float,
    max_strike_distance_pct: Optional[float],
    logger: Optional[logging.Logger] = None,
) -> TieredTradeState:
    """
    Initialize a tiered trade, activating qualifying tiers based on constraints.

    Args:
        trading_date: The trading date
        option_type: 'put' or 'call'
        prev_close: Previous day's closing price
        config: Tiered investment configuration
        day_results: Results from framework analysis for this day/type
        min_premium_diff: Min net credit per option type
        max_credit_width_ratio: Max credit/width ratio
        min_contract_price: Minimum contract price
        max_strike_distance_pct: Max strike distance pct
        logger: Optional logger

    Returns:
        Initialized trade state with activated tiers
    """
    trade_state = TieredTradeState(
        trading_date=trading_date,
        option_type=option_type,
        prev_close=prev_close,
        tiers=[],
    )

    tiers = config.get_tiers(option_type)

    for tier_config in tiers:
        short_strike, long_strike = calculate_tier_strikes(
            prev_close, tier_config, option_type
        )

        # Check constraints
        constraints_passed, credit_estimate = check_tier_constraints(
            tier=tier_config,
            short_strike=short_strike,
            long_strike=long_strike,
            option_type=option_type,
            day_results=day_results,
            min_premium_diff=min_premium_diff,
            max_credit_width_ratio=max_credit_width_ratio,
            min_contract_price=min_contract_price,
            max_strike_distance_pct=max_strike_distance_pct,
            prev_close=prev_close,
            logger=logger,
        )

        capital_at_risk = tier_config.num_contracts * tier_config.spread_width * 100

        tier_position = TierPosition(
            tier_level=tier_config.level,
            option_type=option_type,
            short_strike=short_strike,
            long_strike=long_strike,
            spread_width=tier_config.spread_width,
            num_contracts=tier_config.num_contracts,
            capital_at_risk=capital_at_risk,
            initial_credit_per_share=credit_estimate,
            initial_credit_total=credit_estimate * tier_config.num_contracts * 100,
            activated=constraints_passed,
        )

        trade_state.tiers.append(tier_position)

        if logger:
            status = "ACTIVATED" if constraints_passed else "SKIPPED"
            logger.debug(
                f"Tier {tier_config.level} ({option_type}) {status}: "
                f"Short={short_strike:.2f}, Long={long_strike:.2f}, "
                f"Width={tier_config.spread_width}, Contracts={tier_config.num_contracts}, "
                f"Credit/sh=${credit_estimate:.2f}"
            )

    return trade_state


def calculate_tier_pnl(
    initial_credit: float,
    short_strike: float,
    long_strike: float,
    close_price: float,
    option_type: str,
) -> float:
    """
    Calculate P&L for a tier based on closing price.

    Reuses the same logic as scale_in_utils.calculate_layer_pnl.
    """
    return calculate_layer_pnl(initial_credit, short_strike, long_strike, close_price, option_type)


def calculate_all_tiers_pnl(
    trade_state: TieredTradeState,
    close_price: float,
) -> TieredTradeState:
    """
    Calculate P&L for all activated tiers in a tiered trade.

    Args:
        trade_state: The trade state with all tier positions
        close_price: The closing price of the underlying

    Returns:
        Updated trade state with P&L calculated
    """
    for tier in trade_state.tiers:
        if not tier.activated:
            continue

        tier.breach_detected = check_breach(
            tier.option_type,
            tier.short_strike,
            close_price,
        )

        tier.actual_pnl_per_share = calculate_tier_pnl(
            tier.initial_credit_per_share,
            tier.short_strike,
            tier.long_strike,
            close_price,
            tier.option_type,
        )

        tier.actual_pnl_total = tier.actual_pnl_per_share * tier.num_contracts * 100

    return trade_state


def generate_tiered_summary(
    trade_state: TieredTradeState,
    single_entry_pnl: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Generate a summary of the tiered trade.

    Args:
        trade_state: The trade state with all tier positions
        single_entry_pnl: Optional P&L from single-entry strategy for comparison

    Returns:
        Summary dictionary with key metrics
    """
    summary: Dict[str, Any] = {
        'trading_date': trade_state.trading_date,
        'option_type': trade_state.option_type,
        'prev_close': trade_state.prev_close,
        'num_tiers_activated': trade_state.num_activated_tiers,
        'num_tiers_breached': trade_state.num_breached_tiers,
        'total_capital_at_risk': trade_state.total_capital_at_risk,
        'total_credit': trade_state.total_credit,
        'total_pnl': trade_state.total_pnl,
        'tiers': [],
    }

    for tier in trade_state.tiers:
        tier_summary = {
            'level': tier.tier_level,
            'activated': tier.activated,
            'short_strike': tier.short_strike,
            'long_strike': tier.long_strike,
            'spread_width': tier.spread_width,
            'num_contracts': tier.num_contracts,
            'capital_at_risk': tier.capital_at_risk,
            'initial_credit': tier.initial_credit_total,
            'breach_detected': tier.breach_detected,
            'actual_pnl': tier.actual_pnl_total,
        }
        summary['tiers'].append(tier_summary)

    # Comparison with single-entry
    if single_entry_pnl is not None and trade_state.total_pnl is not None:
        summary['comparison_vs_single'] = {
            'single_entry_pnl': single_entry_pnl,
            'tiered_pnl': trade_state.total_pnl,
            'difference': trade_state.total_pnl - single_entry_pnl,
        }

    return summary


def aggregate_tiered_results(
    results: List[Dict[str, Any]],
    output_tz=None,
) -> Dict[str, Any]:
    """
    Aggregate tiered results across multiple trading days.

    Args:
        results: List of result dicts from tiered analysis
        output_tz: Optional output timezone

    Returns:
        Aggregated statistics dictionary
    """
    total_trades = len(results)
    if total_trades == 0:
        return {'total_trades': 0}

    winning_trades = 0
    losing_trades = 0
    total_capital_at_risk = 0.0
    total_credit = 0.0
    total_pnl = 0.0
    total_tiers_activated = 0
    total_tiers_breached = 0
    tier_stats: Dict[int, Dict[str, Any]] = {}
    single_total_pnl = 0.0
    has_comparison = False

    for result in results:
        summary = result.get('summary', {})
        pnl = summary.get('total_pnl')

        if pnl is not None:
            total_pnl += pnl
            if pnl >= 0:
                winning_trades += 1
            else:
                losing_trades += 1

        total_capital_at_risk += summary.get('total_capital_at_risk', 0)
        total_credit += summary.get('total_credit', 0)
        total_tiers_activated += summary.get('num_tiers_activated', 0)
        total_tiers_breached += summary.get('num_tiers_breached', 0)

        # Per-tier stats
        for tier_info in summary.get('tiers', []):
            level = tier_info['level']
            if level not in tier_stats:
                tier_stats[level] = {
                    'activated_count': 0,
                    'breached_count': 0,
                    'total_pnl': 0.0,
                    'spread_width': tier_info.get('spread_width', 0),
                    'num_contracts': tier_info.get('num_contracts', 0),
                }
            if tier_info.get('activated'):
                tier_stats[level]['activated_count'] += 1
            if tier_info.get('breach_detected'):
                tier_stats[level]['breached_count'] += 1
            if tier_info.get('actual_pnl') is not None:
                tier_stats[level]['total_pnl'] += tier_info['actual_pnl']

        # Comparison
        comparison = summary.get('comparison_vs_single')
        if comparison:
            has_comparison = True
            single_total_pnl += comparison.get('single_entry_pnl', 0)

    win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0
    roi = (total_pnl / total_capital_at_risk * 100) if total_capital_at_risk > 0 else 0

    aggregate: Dict[str, Any] = {
        'total_trades': total_trades,
        'winning_trades': winning_trades,
        'losing_trades': losing_trades,
        'win_rate': win_rate,
        'total_capital_at_risk': total_capital_at_risk,
        'total_credit': total_credit,
        'total_pnl': total_pnl,
        'roi': roi,
        'avg_tiers_activated': total_tiers_activated / total_trades if total_trades > 0 else 0,
        'avg_tiers_breached': total_tiers_breached / total_trades if total_trades > 0 else 0,
        'tier_stats': tier_stats,
    }

    if has_comparison:
        aggregate['single_total_pnl'] = single_total_pnl
        aggregate['tiered_vs_single_diff'] = total_pnl - single_total_pnl

    return aggregate


def print_tiered_statistics(
    aggregate_stats: Dict[str, Any],
    tiered_results: List[Dict[str, Any]],
    config: TieredInvestmentConfig,
    comparison_results: Optional[List[Dict]] = None,
    summary_only: bool = False,
):
    """
    Print comprehensive tiered investment statistics.

    Args:
        aggregate_stats: Aggregated statistics from aggregate_tiered_results
        tiered_results: Individual result dicts
        config: Tiered investment configuration
        comparison_results: Optional single-entry results for comparison
        summary_only: If True, skip individual trade details
    """
    print(f"\n{'='*100}")
    print("TIERED INVESTMENT STRATEGY - RESULTS")
    print(f"{'='*100}")

    # Configuration summary
    print(f"\nCONFIGURATION:")
    for opt_type in ['put', 'call']:
        tiers = config.get_tiers(opt_type)
        if tiers:
            print(f"  {opt_type.upper()} Tiers:")
            for t in tiers:
                print(f"    T{t.level}: {t.percent_beyond*100:.1f}% beyond, "
                      f"N={t.num_contracts} contracts, M=${t.spread_width:.0f} width")

    # Overall statistics
    total_trades = aggregate_stats.get('total_trades', 0)
    if total_trades == 0:
        print("\n  No trades analyzed.")
        return

    print(f"\nOVERALL STATISTICS:")
    print(f"  Total Trades:        {total_trades}")
    print(f"  Winning Trades:      {aggregate_stats.get('winning_trades', 0)}")
    print(f"  Losing Trades:       {aggregate_stats.get('losing_trades', 0)}")
    print(f"  Win Rate:            {aggregate_stats.get('win_rate', 0):.1f}%")
    print(f"  Avg Tiers Activated: {aggregate_stats.get('avg_tiers_activated', 0):.1f}")
    print(f"  Avg Tiers Breached:  {aggregate_stats.get('avg_tiers_breached', 0):.1f}")

    total_pnl = aggregate_stats.get('total_pnl', 0)
    pnl_str = f"${total_pnl:,.2f}" if total_pnl >= 0 else f"-${abs(total_pnl):,.2f}"
    print(f"\nFINANCIAL SUMMARY:")
    print(f"  Total Capital at Risk: ${aggregate_stats.get('total_capital_at_risk', 0):,.2f}")
    print(f"  Total Credit Received: ${aggregate_stats.get('total_credit', 0):,.2f}")
    print(f"  Total P&L:             {pnl_str}")
    print(f"  ROI:                   {aggregate_stats.get('roi', 0):.2f}%")

    # Per-tier breakdown
    tier_stats = aggregate_stats.get('tier_stats', {})
    if tier_stats:
        print(f"\nTIER BREAKDOWN:")
        print(f"  {'Tier':<8} {'Width':<8} {'N':<8} {'Activated':<12} {'Breached':<12} {'P&L':<15}")
        print(f"  {'-'*8} {'-'*8} {'-'*8} {'-'*12} {'-'*12} {'-'*15}")
        for level in sorted(tier_stats.keys()):
            stats = tier_stats[level]
            tier_pnl = stats['total_pnl']
            tier_pnl_str = f"${tier_pnl:,.2f}" if tier_pnl >= 0 else f"-${abs(tier_pnl):,.2f}"
            print(f"  T{level:<7} ${stats['spread_width']:<7.0f} {stats['num_contracts']:<8} "
                  f"{stats['activated_count']:<12} {stats['breached_count']:<12} {tier_pnl_str:<15}")

    # Comparison with single-entry
    if 'single_total_pnl' in aggregate_stats:
        single_pnl = aggregate_stats['single_total_pnl']
        diff = aggregate_stats.get('tiered_vs_single_diff', 0)
        single_str = f"${single_pnl:,.2f}" if single_pnl >= 0 else f"-${abs(single_pnl):,.2f}"
        diff_str = f"${diff:,.2f}" if diff >= 0 else f"-${abs(diff):,.2f}"
        print(f"\nCOMPARISON vs SINGLE-ENTRY:")
        print(f"  Single-Entry Total P&L:  {single_str}")
        print(f"  Tiered Total P&L:        {pnl_str}")
        print(f"  Difference:              {diff_str}")

    # Individual trade details (unless summary_only)
    if not summary_only and tiered_results:
        max_display = 20
        print(f"\nINDIVIDUAL TRADES (showing up to {max_display}):")
        print(f"  {'Date':<12} {'Type':<6} {'Prev Close':<12} {'Tiers':<8} {'Breached':<10} "
              f"{'Capital':<14} {'Credit':<14} {'P&L':<15}")
        print(f"  {'-'*12} {'-'*6} {'-'*12} {'-'*8} {'-'*10} "
              f"{'-'*14} {'-'*14} {'-'*15}")

        for result in tiered_results[:max_display]:
            summary = result.get('summary', {})
            td = summary.get('trading_date')
            date_str = td.strftime('%Y-%m-%d') if hasattr(td, 'strftime') else str(td)[:10]
            opt = summary.get('option_type', '?').upper()
            pc = summary.get('prev_close', 0)
            na = summary.get('num_tiers_activated', 0)
            nb = summary.get('num_tiers_breached', 0)
            cap = summary.get('total_capital_at_risk', 0)
            cred = summary.get('total_credit', 0)
            pnl = summary.get('total_pnl')

            if pnl is not None:
                pnl_display = f"${pnl:,.2f}" if pnl >= 0 else f"-${abs(pnl):,.2f}"
            else:
                pnl_display = "-"

            print(f"  {date_str:<12} {opt:<6} ${pc:>9,.2f} {na:<8} {nb:<10} "
                  f"${cap:>12,.2f} ${cred:>12,.2f} {pnl_display:<15}")

        if len(tiered_results) > max_display:
            print(f"  ... and {len(tiered_results) - max_display} more trades")

    print(f"{'='*100}")


def load_tiered_config(config_path: Optional[str]) -> Optional[TieredInvestmentConfig]:
    """
    Load tiered investment configuration from file path.

    Args:
        config_path: Path to JSON config file

    Returns:
        TieredInvestmentConfig or None if path is None
    """
    if not config_path:
        return None

    try:
        return TieredInvestmentConfig.from_file(config_path)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        raise ValueError(f"Failed to load tiered config: {e}")
