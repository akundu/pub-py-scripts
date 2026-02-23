#!/usr/bin/env python3
"""
Opportunity Scanner for Continuous Mode

Finds option spreads matching grid-validated configurations.
"""

import sys
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Dict, Optional
from dataclasses import dataclass, asdict
import pandas as pd

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from scripts.regime_strategy_selector import filter_configs_by_regime
from scripts.continuous.market_data import MarketContext
from scripts.continuous.config import CONFIG
from scripts.continuous.strike_resolver import (
    resolve_trade, find_option_chain_file, load_option_chain,
    get_expiration_date, pst_to_et, ResolvedTrade,
)


@dataclass
class TradeOpportunity:
    """Represents a trade opportunity."""
    timestamp: str
    config_rank: int  # Rank among filtered configs

    # Config details
    dte: int
    band: str
    spread_type: str
    flow_mode: str
    entry_time_pst: str

    # Expected metrics (from backtest)
    expected_win_pct: float
    expected_roi_pct: float
    sharpe: float
    trade_score: float

    # Estimated trade details (would come from option chain in real implementation)
    estimated_credit: float
    estimated_max_risk: float
    estimated_strikes: Optional[Dict[str, float]] = None

    # Alerts
    is_in_entry_window: bool = False
    meets_quality_threshold: bool = False

    # Resolved trade details (specific strikes and prices)
    resolved_trade: Optional[Dict] = None  # ResolvedTrade.to_dict()
    trade_instruction: Optional[str] = None  # Human-readable instruction

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return asdict(self)

    def __str__(self) -> str:
        """Human-readable format."""
        return (
            f"#{self.config_rank} | {self.dte}DTE {self.band} {self.spread_type.upper()} "
            f"({self.flow_mode}) @ {self.entry_time_pst} PST | "
            f"Win:{self.expected_win_pct:.1f}% ROI:{self.expected_roi_pct:.1f}% "
            f"Sharpe:{self.sharpe:.2f} | "
            f"Credit:${self.estimated_credit:.0f} Risk:${self.estimated_max_risk:.0f} | "
            f"Score:{self.trade_score:.1f}"
        )


def scan_opportunities(
    market_context: MarketContext,
    grid_file: Path = None,
    top_n: int = 20,
    option_chain: Dict = None,
    target_date: 'date' = None,
) -> List[TradeOpportunity]:
    """
    Scan for trade opportunities based on current market regime.

    Args:
        market_context: Current market context
        grid_file: Path to grid CSV (default: from config)
        top_n: Number of configs to load from regime filter
        option_chain: Pre-loaded option chain dict (from load_option_chain)
        target_date: Date for option chain lookup (if chain not provided)

    Returns:
        List of TradeOpportunity objects
    """
    from datetime import date as date_type

    if grid_file is None:
        grid_file = CONFIG.grid_file

    if not grid_file.exists():
        print(f"Error: Grid file not found: {grid_file}")
        return []

    # Load grid
    df = pd.read_csv(grid_file)

    # Filter by regime (with VIX dynamics if available)
    vix_direction = getattr(market_context, 'vix_direction', 'stable')
    vix_velocity = getattr(market_context, 'vix_velocity', 0.0)
    vix_level = getattr(market_context, 'vix_level', 15.0)
    vix_term_spread = getattr(market_context, 'vix_term_spread', None)

    filtered = filter_configs_by_regime(
        df,
        vix_regime=market_context.vix_regime,
        trend=market_context.trend,
        top_n=top_n,
        vix_direction=vix_direction,
        vix_velocity=vix_velocity,
        vix_level=vix_level,
        vix_term_spread=vix_term_spread,
    )

    if len(filtered) == 0:
        print("Warning: No configs match current regime")
        return []

    # Load option chain if not provided
    if option_chain is None and target_date is not None:
        chain_file = find_option_chain_file(
            market_context.ticker, target_date
        )
        if chain_file:
            option_chain = load_option_chain(chain_file)

    # Convert to opportunities
    opportunities = []
    timestamp = datetime.now().isoformat()

    for rank, (idx, config) in enumerate(filtered.iterrows(), 1):
        # Parse entry time
        entry_time = config.get('entry_time_pst', config.get('time_pst', '07:30'))

        # Check if in entry window
        if ':' in entry_time:
            entry_hour = int(entry_time.split(':')[0])
        else:
            entry_hour = 7  # Default

        is_in_window = (
            entry_hour in CONFIG.preferred_entry_hours and
            market_context.is_market_hours
        )

        # Actual ROI for quality check (avg_pnl / max_risk)
        _risk = config.get('max_risk_per_contract',
                config.get('avg_max_risk',
                config.get('expected_max_risk', 0)))
        _pnl = config.get('avg_pnl', 0)
        _actual_roi = (_pnl / _risk * 100) if (_pnl and _risk > 0) else config.get('expected_roi_pct', config.get('roi_pct', 0))

        # Check quality thresholds
        meets_quality = (
            config.get('expected_win_pct', config.get('win_rate_pct', 0)) >= CONFIG.min_win_rate and
            _actual_roi >= CONFIG.min_roi and
            config.get('sharpe', 0) >= CONFIG.min_sharpe
        )

        # Get trade score
        trade_score = config.get('trade_score', 0)
        if trade_score == 0 and all(k in config for k in ['expected_roi_pct', 'sharpe', 'expected_win_pct']):
            # Calculate if not present
            trade_score = (
                config['expected_roi_pct'] * 0.25 +
                config['sharpe'] * 8 +
                config['expected_win_pct'] * 0.6
            )

        # Estimate credit and risk (from backtest averages)
        estimated_credit = config.get('avg_credit', config.get('expected_credit', 250.0))
        estimated_max_risk = config.get('max_risk_per_contract',
                            config.get('avg_max_risk',
                            config.get('expected_max_risk', 2000.0)))

        # Actual ROI: avg_pnl / max_risk (not credit/risk which overstates returns)
        avg_pnl = config.get('avg_pnl', 0)
        if avg_pnl and estimated_max_risk > 0:
            actual_roi_pct = (avg_pnl / estimated_max_risk) * 100
        else:
            # Fall back to the grid's ROI (credit/risk) if avg_pnl not available
            actual_roi_pct = config.get('expected_roi_pct', config.get('roi_pct', 0))

        # Resolve actual strikes and prices
        resolved = None
        trade_instruction = None
        resolved_dict = None
        try:
            dte_val = int(config['dte'])
            spread_width = int(config.get('spread_width', 10))
            vix_val = getattr(market_context, 'vix_level', 15.0)

            # Compute expiration date
            if target_date is not None:
                exp_date = get_expiration_date(target_date, dte_val)
                exp_str = exp_date.strftime('%Y-%m-%d')
            else:
                exp_date = get_expiration_date(date_type.today(), dte_val)
                exp_str = exp_date.strftime('%Y-%m-%d')

            resolved = resolve_trade(
                current_price=market_context.current_price,
                band=config['band'],
                dte=dte_val,
                spread_type=config['spread_type'],
                spread_width=spread_width,
                vix=vix_val,
                chain=option_chain,
                expiration=exp_str,
            )
            if resolved:
                resolved_dict = resolved.to_dict()
                trade_instruction = resolved.instruction_text()
        except Exception as e:
            pass  # Strike resolution is best-effort

        opportunity = TradeOpportunity(
            timestamp=timestamp,
            config_rank=rank,
            dte=int(config['dte']),
            band=config['band'],
            spread_type=config['spread_type'],
            flow_mode=config['flow_mode'],
            entry_time_pst=entry_time,
            expected_win_pct=config.get('expected_win_pct', config.get('win_rate_pct', 0)),
            expected_roi_pct=actual_roi_pct,
            sharpe=config.get('sharpe', 0),
            trade_score=trade_score,
            estimated_credit=estimated_credit,
            estimated_max_risk=estimated_max_risk,
            estimated_strikes=None,
            is_in_entry_window=is_in_window,
            meets_quality_threshold=meets_quality,
            resolved_trade=resolved_dict,
            trade_instruction=trade_instruction,
        )

        opportunities.append(opportunity)

    # Sort by trade score
    opportunities.sort(key=lambda x: x.trade_score, reverse=True)

    return opportunities


def filter_actionable_opportunities(
    opportunities: List[TradeOpportunity],
    require_entry_window: bool = True,
    require_quality: bool = True,
    top_n: int = 5
) -> List[TradeOpportunity]:
    """
    Filter to actionable opportunities.

    Args:
        opportunities: All opportunities
        require_entry_window: Only return if in entry window
        require_quality: Only return if meets quality thresholds
        top_n: Maximum number to return

    Returns:
        Filtered list of opportunities
    """
    filtered = opportunities

    if require_entry_window:
        filtered = [opp for opp in filtered if opp.is_in_entry_window]

    if require_quality:
        filtered = [opp for opp in filtered if opp.meets_quality_threshold]

    return filtered[:top_n]


if __name__ == '__main__':
    """Test opportunity scanner."""
    from scripts.continuous.market_data import get_current_market_context

    print("=" * 80)
    print("OPPORTUNITY SCANNER TEST")
    print("=" * 80)

    # Get market context
    print("\nFetching market context...")
    context = get_current_market_context('NDX', trend='sideways')

    print(f"VIX: {context.vix_level:.2f} (Regime: {context.vix_regime.upper()})")
    print(f"Trend: {context.trend.upper()}")
    print(f"Market Hours: {'YES' if context.is_market_hours else 'NO'}")

    # Scan opportunities
    print("\nScanning for opportunities...")
    opportunities = scan_opportunities(context, top_n=20)

    print(f"\nFound {len(opportunities)} opportunities")

    # Show top 10
    print("\n" + "=" * 80)
    print("TOP 10 OPPORTUNITIES")
    print("=" * 80)
    for opp in opportunities[:10]:
        entry_flag = "✓ ENTRY WINDOW" if opp.is_in_entry_window else ""
        quality_flag = "✓ QUALITY" if opp.meets_quality_threshold else ""
        print(f"{opp} {entry_flag} {quality_flag}")

    # Show actionable only
    actionable = filter_actionable_opportunities(
        opportunities,
        require_entry_window=False,  # Don't require for testing
        require_quality=True,
        top_n=5
    )

    print("\n" + "=" * 80)
    print("TOP 5 ACTIONABLE NOW")
    print("=" * 80)
    for opp in actionable:
        print(opp)
