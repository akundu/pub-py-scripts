#!/usr/bin/env python3
"""
Regime-Based Strategy Selector

Selects optimal spread configurations based on current market regime:
1. VIX level (low/medium/high/extreme)
2. Trend direction (up/down/sideways)
3. Volatility regime (stable/expanding/contracting)

Usage:
    python scripts/regime_strategy_selector.py --current-vix 18.5 --trend up
"""

import sys
import argparse
from pathlib import Path
from datetime import datetime, date
import pandas as pd
import numpy as np
from typing import Dict, List, Optional

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def detect_vix_regime(vix_level: float) -> str:
    """Classify VIX level into regime."""
    if vix_level < 12:
        return 'very_low'
    elif vix_level < 16:
        return 'low'
    elif vix_level < 20:
        return 'medium'
    elif vix_level < 30:
        return 'high'
    else:
        return 'extreme'


def get_regime_recommendations(
    vix_regime: str,
    trend: str = 'sideways'
) -> Dict:
    """
    Get trading recommendations based on regime.

    Args:
        vix_regime: VIX regime classification
        trend: Market trend (up/down/sideways)

    Returns:
        Dict with recommended parameters
    """
    recommendations = {
        'very_low': {
            # VIX < 12: Complacent market, risk of vol expansion
            'preferred_spreads': ['iron_condor'],
            'preferred_bands': ['P99', 'P100'],  # Use wider bands
            'preferred_dtes': [3, 5],
            'flow_mode': 'neutral',
            'risk_level': 'aggressive',
            'rationale': 'Low vol environment - sell premium aggressively with wide bands',
        },
        'low': {
            # VIX 12-16: Normal low vol
            'preferred_spreads': ['iron_condor', 'put_spread'],
            'preferred_bands': ['P97', 'P98', 'P99'],
            'preferred_dtes': [1, 3, 5],
            'flow_mode': 'with_flow',
            'risk_level': 'balanced',
            'rationale': 'Favorable environment - standard credit spread strategies work well',
        },
        'medium': {
            # VIX 16-20: Average volatility
            'preferred_spreads': ['put_spread', 'iron_condor'],
            'preferred_bands': ['P97', 'P98'],
            'preferred_dtes': [1, 3],
            'flow_mode': 'with_flow',
            'risk_level': 'balanced',
            'rationale': 'Normal volatility - stick to proven configs',
        },
        'high': {
            # VIX 20-30: Elevated volatility
            'preferred_spreads': ['put_spread'],
            'preferred_bands': ['P95', 'P97'],  # Tighter bands for safety
            'preferred_dtes': [1],  # Shorter DTE to reduce exposure
            'flow_mode': 'with_flow',
            'risk_level': 'conservative',
            'rationale': 'High vol - reduce risk with tighter bands and shorter DTE',
        },
        'extreme': {
            # VIX > 30: Crisis mode
            'preferred_spreads': ['put_spread'],  # Only directional
            'preferred_bands': ['P95'],  # Very tight
            'preferred_dtes': [1],  # 1 day only
            'flow_mode': 'with_flow',
            'risk_level': 'very_conservative',
            'rationale': 'Extreme vol - minimal exposure, very tight bands, or sit out',
        },
    }

    base_rec = recommendations.get(vix_regime, recommendations['medium'])

    # Adjust for trend
    if trend == 'up':
        # Bullish trend - prefer call spreads
        base_rec['preferred_spreads'] = ['call_spread', 'iron_condor']
        base_rec['trend_adjustment'] = 'Bullish - favor call spreads'
    elif trend == 'down':
        # Bearish trend - prefer put spreads
        base_rec['preferred_spreads'] = ['put_spread']
        base_rec['trend_adjustment'] = 'Bearish - focus on put spreads'
    else:
        base_rec['trend_adjustment'] = 'Sideways - iron condors optimal'

    return base_rec


def _get_adjacent_regime(vix_regime: str, direction: str) -> str:
    """Get the next regime in the given direction."""
    order = ['very_low', 'low', 'medium', 'high', 'extreme']
    idx = order.index(vix_regime) if vix_regime in order else 2
    if direction == 'rising' and idx < len(order) - 1:
        return order[idx + 1]
    elif direction == 'falling' and idx > 0:
        return order[idx - 1]
    return vix_regime


def _distance_to_boundary(vix_level: float, vix_regime: str) -> float:
    """How far VIX is from the nearest regime boundary (as fraction of band width).
    Returns 0.0-1.0 where 0.0 = right at boundary, 1.0 = center of regime."""
    boundaries = {
        'very_low': (0, 12),
        'low': (12, 16),
        'medium': (16, 20),
        'high': (20, 30),
        'extreme': (30, 60),
    }
    low, high = boundaries.get(vix_regime, (12, 20))
    band_width = high - low
    if band_width == 0:
        return 1.0
    dist_to_low = abs(vix_level - low)
    dist_to_high = abs(vix_level - high)
    nearest_dist = min(dist_to_low, dist_to_high)
    return min(1.0, nearest_dist / (band_width / 2))


def filter_configs_by_regime(
    df: pd.DataFrame,
    vix_regime: str,
    trend: str = 'sideways',
    top_n: int = 20,
    vix_direction: str = 'stable',
    vix_velocity: float = 0.0,
    vix_level: float = 15.0,
    vix_term_spread: Optional[float] = None,
) -> pd.DataFrame:
    """Filter grid configs to those suitable for current regime.

    Now accounts for VIX direction/velocity to blend in adjacent regime
    configs when VIX is moving toward a boundary.

    Args:
        df: Full grid DataFrame
        vix_regime: Current VIX regime classification
        trend: Market trend (up/down/sideways)
        top_n: Number of configs to return
        vix_direction: 'rising', 'falling', or 'stable'
        vix_velocity: VIX change rate (points per 5-min interval)
        vix_level: Current VIX absolute level
        vix_term_spread: VIX - VIX1D (positive = near-term stress)
    """
    recommendations = get_regime_recommendations(vix_regime, trend)

    # Filter by regime recommendations
    filtered = df[
        (df['spread_type'].isin(recommendations['preferred_spreads'])) &
        (df['band'].isin(recommendations['preferred_bands'])) &
        (df['dte'].isin(recommendations['preferred_dtes']))
    ].copy()

    if len(filtered) == 0:
        print(f"Warning: No configs match regime criteria, using all configs")
        filtered = df.copy()

    # --- VIX Direction Blending ---
    # When VIX is approaching a regime boundary and moving toward it,
    # blend in configs from the adjacent (more conservative) regime.
    boundary_distance = _distance_to_boundary(vix_level, vix_regime)
    approaching_boundary = (
        boundary_distance < 0.4 and  # within 40% of boundary
        abs(vix_velocity) > 0.1 and  # moving meaningfully
        (
            (vix_direction == 'rising' and vix_level > 0) or
            (vix_direction == 'falling' and vix_level > 0)
        )
    )

    if approaching_boundary:
        adjacent_regime = _get_adjacent_regime(vix_regime, vix_direction)
        if adjacent_regime != vix_regime:
            adj_recs = get_regime_recommendations(adjacent_regime, trend)
            adj_filtered = df[
                (df['spread_type'].isin(adj_recs['preferred_spreads'])) &
                (df['band'].isin(adj_recs['preferred_bands'])) &
                (df['dte'].isin(adj_recs['preferred_dtes']))
            ].copy()

            if len(adj_filtered) > 0:
                # Blend: take some configs from adjacent regime proportional to proximity
                blend_ratio = max(0.1, min(0.5, (0.4 - boundary_distance) / 0.4))
                n_adjacent = max(1, int(top_n * blend_ratio))
                n_current = top_n - n_adjacent

                # Mark source for debugging
                filtered['_regime_source'] = vix_regime
                adj_filtered['_regime_source'] = adjacent_regime

                # Sort both by score
                score_col = 'trade_score' if 'trade_score' in filtered.columns else None
                if score_col:
                    filtered = filtered.sort_values(score_col, ascending=False)
                    adj_filtered = adj_filtered.sort_values(score_col, ascending=False)

                filtered = pd.concat([
                    filtered.head(n_current),
                    adj_filtered.head(n_adjacent)
                ], ignore_index=True)

                # Clean up
                if '_regime_source' in filtered.columns:
                    filtered = filtered.drop(columns=['_regime_source'])

    # --- VIX Term Spread Adjustment ---
    # When VIX1D > VIX (inverted term structure = near-term stress),
    # prefer tighter bands and shorter DTEs even within current regime.
    if vix_term_spread is not None and vix_term_spread < -1.0:
        # VIX1D significantly below VIX: near-term complacency — widen bands OK
        pass
    elif vix_term_spread is not None and vix_term_spread > 1.0:
        # Near-term stress: tighten by preferring lower bands and shorter DTEs
        if 'band' in filtered.columns:
            tighter = filtered[filtered['band'].isin(['P95', 'P97'])]
            if len(tighter) >= top_n // 2:
                filtered = tighter
        if 'dte' in filtered.columns:
            shorter = filtered[filtered['dte'] <= 3]
            if len(shorter) >= top_n // 3:
                filtered = shorter

    # --- VIX Velocity Score Adjustment ---
    # When VIX is moving fast (high velocity), penalize high-risk configs
    if abs(vix_velocity) > 0.3 and 'trade_score' in filtered.columns:
        # High VIX velocity: boost configs with higher win rate (safer)
        win_col = 'expected_win_pct' if 'expected_win_pct' in filtered.columns else 'win_rate_pct'
        if win_col in filtered.columns:
            velocity_penalty = min(0.2, abs(vix_velocity) * 0.3)
            # Adjust score: boost high-win-rate configs, penalize lower ones
            median_win = filtered[win_col].median()
            filtered['_adj_score'] = filtered['trade_score'] * (
                1.0 + velocity_penalty * ((filtered[win_col] - median_win) / 10.0)
            )
            filtered = filtered.sort_values('_adj_score', ascending=False)
            filtered = filtered.drop(columns=['_adj_score'])

    # Prefer flow mode if specified
    if 'flow_mode' in filtered.columns:
        flow_filtered = filtered[filtered['flow_mode'] == recommendations['flow_mode']]
        if len(flow_filtered) > 0:
            filtered = flow_filtered

    # Sort by composite score
    if 'trade_score' in filtered.columns:
        filtered = filtered.sort_values('trade_score', ascending=False)
    elif all(col in filtered.columns for col in ['roi_pct', 'sharpe', 'win_rate_pct']):
        filtered['temp_score'] = (
            filtered['roi_pct'] * 0.3 +
            filtered['sharpe'] * 10 +
            filtered['win_rate_pct'] * 0.5
        )
        filtered = filtered.sort_values('temp_score', ascending=False)

    return filtered.head(top_n)


def main():
    parser = argparse.ArgumentParser(description='Select strategies based on market regime')
    parser.add_argument('--current-vix', type=float, default=None,
                        help='Current VIX level (if not provided, will fetch from DB)')
    parser.add_argument('--trend', type=str, default='sideways',
                        choices=['up', 'down', 'sideways'],
                        help='Current market trend')
    parser.add_argument('--input', type=Path,
                        default=Path('results/backtest_tight/grid_trading_ready.csv'),
                        help='Input grid CSV')
    parser.add_argument('--output', type=Path, default=None,
                        help='Output CSV for regime-filtered configs')
    parser.add_argument('--top-n', type=int, default=20,
                        help='Number of configs to return')
    args = parser.parse_args()

    print("=" * 80)
    print("REGIME-BASED STRATEGY SELECTOR")
    print("=" * 80)

    # Get current VIX
    if args.current_vix is None:
        # Try to fetch from DB
        try:
            from common.questdb_db import get_latest_price
            vix_data = get_latest_price('I:VIX')
            if vix_data:
                args.current_vix = vix_data['close']
                print(f"Fetched current VIX from DB: {args.current_vix:.2f}")
        except:
            args.current_vix = 18.0  # Default fallback
            print(f"Could not fetch VIX, using default: {args.current_vix:.2f}")
    else:
        print(f"Using provided VIX: {args.current_vix:.2f}")

    # Detect regime
    vix_regime = detect_vix_regime(args.current_vix)
    print(f"VIX Regime: {vix_regime.upper().replace('_', ' ')}")
    print(f"Trend: {args.trend.upper()}")
    print()

    # Get recommendations
    recommendations = get_regime_recommendations(vix_regime, args.trend)

    print("=" * 80)
    print("REGIME RECOMMENDATIONS")
    print("=" * 80)
    print(f"\nRisk Level: {recommendations['risk_level'].upper()}")
    print(f"Rationale: {recommendations['rationale']}")
    if 'trend_adjustment' in recommendations:
        print(f"Trend Adjustment: {recommendations['trend_adjustment']}")

    print(f"\nPreferred Configuration:")
    print(f"  Spread Types: {', '.join(recommendations['preferred_spreads'])}")
    print(f"  Bands: {', '.join(recommendations['preferred_bands'])}")
    print(f"  DTEs: {', '.join(map(str, recommendations['preferred_dtes']))}")
    print(f"  Flow Mode: {recommendations['flow_mode']}")

    # Load grid configs
    if not args.input.exists():
        print(f"\nError: {args.input} not found")
        return

    df = pd.read_csv(args.input)
    print(f"\nLoaded {len(df)} configurations from grid")

    # Filter by regime
    filtered = filter_configs_by_regime(df, vix_regime, args.trend, args.top_n)

    print(f"\n" + "=" * 80)
    print(f"TOP {len(filtered)} CONFIGS FOR CURRENT REGIME")
    print("=" * 80)

    # Display key columns
    display_cols = ['dte', 'band', 'spread_type', 'flow_mode']

    # Add entry time if available
    if 'entry_time_pst' in filtered.columns:
        display_cols.append('entry_time_pst')
    elif 'time_pst' in filtered.columns:
        display_cols.append('time_pst')

    # Add performance metrics
    if 'expected_win_pct' in filtered.columns:
        display_cols.extend(['expected_win_pct', 'expected_roi_pct', 'sharpe'])
    elif 'win_rate_pct' in filtered.columns:
        display_cols.extend(['win_rate_pct', 'roi_pct', 'sharpe'])

    # Add P&L if available
    if 'avg_pnl' in filtered.columns:
        display_cols.append('avg_pnl')

    print("\n" + filtered[display_cols].to_string(index=False))

    # Save output
    if args.output:
        filtered.to_csv(args.output, index=False)
        print(f"\n✓ Saved {len(filtered)} regime-filtered configs to {args.output}")
    else:
        # Auto-generate output filename
        output_file = Path('results/regime_strategies') / f'regime_{vix_regime}_{args.trend}_{date.today()}.csv'
        output_file.parent.mkdir(parents=True, exist_ok=True)
        filtered.to_csv(output_file, index=False)
        print(f"\n✓ Saved to {output_file}")

    # Comparison to all configs
    print("\n" + "=" * 80)
    print("REGIME-FILTERED VS ALL CONFIGS")
    print("=" * 80)

    roi_col = 'expected_roi_pct' if 'expected_roi_pct' in df.columns else 'roi_pct'
    win_col = 'expected_win_pct' if 'expected_win_pct' in df.columns else 'win_rate_pct'

    if all(col in df.columns for col in [roi_col, win_col, 'sharpe']):
        print(f"\nAverage Metrics:")
        print(f"                    All Configs    Regime-Filtered    Difference")
        print(f"  Win Rate:        {df[win_col].mean():7.1f}%     {filtered[win_col].mean():7.1f}%        {filtered[win_col].mean() - df[win_col].mean():+6.1f}%")
        print(f"  ROI:             {df[roi_col].mean():7.1f}%     {filtered[roi_col].mean():7.1f}%        {filtered[roi_col].mean() - df[roi_col].mean():+6.1f}%")
        print(f"  Sharpe:          {df['sharpe'].mean():7.2f}      {filtered['sharpe'].mean():7.2f}         {filtered['sharpe'].mean() - df['sharpe'].mean():+6.2f}")

    print("\n" + "=" * 80)
    print("READY TO TRADE")
    print("=" * 80)
    print(f"\nUse the {len(filtered)} configs above for today's market conditions.")
    print(f"Review again tomorrow or when VIX/trend changes significantly.")


if __name__ == '__main__':
    main()
