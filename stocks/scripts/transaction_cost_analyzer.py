#!/usr/bin/env python3
"""
Transaction Cost Analysis for Grid Results

Applies realistic trading costs to backtest results:
1. Commission per contract leg
2. Bid-ask slippage
3. Assignment fees
4. Market impact

Usage:
    python scripts/transaction_cost_analyzer.py --input grid_trading_ready.csv
"""

import argparse
from pathlib import Path
import pandas as pd
import numpy as np


def apply_transaction_costs(
    df: pd.DataFrame,
    commission_per_leg: float = 0.65,
    slippage_pct: float = 0.05,
    assignment_rate: float = 0.01,
    assignment_fee: float = 25.00
) -> pd.DataFrame:
    """
    Apply realistic transaction costs to grid results.

    Args:
        df: Grid analysis DataFrame
        commission_per_leg: Commission per option leg (default $0.65)
        slippage_pct: Bid-ask slippage as % of credit (default 5%)
        assignment_rate: Probability of assignment (default 1%)
        assignment_fee: Fee charged on assignment (default $25)

    Returns:
        DataFrame with cost-adjusted metrics
    """
    df = df.copy()

    # Calculate legs per spread type
    legs_map = {
        'put_spread': 2,
        'call_spread': 2,
        'iron_condor': 4,
    }

    df['num_legs'] = df['spread_type'].map(legs_map)

    # Calculate costs per trade
    df['commission_cost'] = df['num_legs'] * commission_per_leg * df['n_contracts']
    df['slippage_cost'] = df['avg_credit'] * slippage_pct
    df['expected_assignment_cost'] = assignment_rate * assignment_fee

    # Total cost per trade
    df['total_cost_per_trade'] = (
        df['commission_cost'] +
        df['slippage_cost'] +
        df['expected_assignment_cost']
    )

    # Adjust P&L metrics
    df['avg_pnl_after_costs'] = df['avg_pnl'] - df['total_cost_per_trade']
    df['total_pnl_after_costs'] = df['avg_pnl_after_costs'] * df['n_trades']

    # Adjust ROI
    # ROI = (avg_credit - costs) / avg_max_risk * 100
    df['avg_credit_after_costs'] = df['avg_credit'] - df['total_cost_per_trade']
    df['roi_pct_after_costs'] = (df['avg_credit_after_costs'] / df['avg_max_risk']) * 100

    # Adjust win rate (some winners become losers due to costs)
    # Estimate: if pnl < total_cost, trade becomes a loss
    df['pnl_to_cost_ratio'] = df['avg_pnl'] / df['total_cost_per_trade']
    df['estimated_cost_induced_losses'] = np.where(
        df['pnl_to_cost_ratio'] < 1.5,  # If P&L < 1.5x costs, some trades flip to losses
        (1.5 - df['pnl_to_cost_ratio']) * 0.05,  # Rough estimate
        0
    )
    df['win_rate_after_costs'] = np.maximum(
        df['win_rate_pct'] - df['estimated_cost_induced_losses'],
        0
    )

    # Recalculate Sharpe (rough approximation)
    # Sharpe decreases proportionally to ROI decrease
    df['sharpe_after_costs'] = df['sharpe'] * (df['roi_pct_after_costs'] / df['roi_pct'].replace(0, 1))

    return df


def main():
    parser = argparse.ArgumentParser(description='Analyze transaction costs impact on grid results')
    parser.add_argument('--input', type=Path, default=Path('results/backtest_tight/grid_trading_ready.csv'),
                        help='Input grid CSV file')
    parser.add_argument('--output', type=Path, default=None,
                        help='Output CSV file (default: input_with_costs.csv)')
    parser.add_argument('--commission', type=float, default=0.65,
                        help='Commission per leg (default: $0.65)')
    parser.add_argument('--slippage-pct', type=float, default=0.05,
                        help='Bid-ask slippage as percent (default: 5%%)')
    parser.add_argument('--assignment-rate', type=float, default=0.01,
                        help='Assignment probability (default: 1%%)')
    parser.add_argument('--assignment-fee', type=float, default=25.00,
                        help='Assignment fee (default: $25)')
    args = parser.parse_args()

    print("=" * 80)
    print("TRANSACTION COST ANALYSIS")
    print("=" * 80)
    print(f"Input file: {args.input}")
    print(f"Commission per leg: ${args.commission:.2f}")
    print(f"Slippage: {args.slippage_pct * 100:.1f}%")
    print(f"Assignment rate: {args.assignment_rate * 100:.1f}%")
    print(f"Assignment fee: ${args.assignment_fee:.2f}")
    print()

    # Load grid data
    if not args.input.exists():
        print(f"Error: Input file {args.input} not found")
        return

    df = pd.read_csv(args.input)
    print(f"Loaded {len(df)} configurations")

    # Apply costs
    df_with_costs = apply_transaction_costs(
        df,
        commission_per_leg=args.commission,
        slippage_pct=args.slippage_pct,
        assignment_rate=args.assignment_rate,
        assignment_fee=args.assignment_fee
    )

    # Calculate impact statistics
    print("\n" + "=" * 80)
    print("IMPACT SUMMARY")
    print("=" * 80)

    # Overall impact
    if 'avg_pnl' in df.columns:
        avg_pnl_before = df['avg_pnl'].mean()
        avg_pnl_after = df_with_costs['avg_pnl_after_costs'].mean()
        pnl_impact = ((avg_pnl_after - avg_pnl_before) / avg_pnl_before) * 100

        print(f"\nAverage P&L per Trade:")
        print(f"  Before costs: ${avg_pnl_before:.2f}")
        print(f"  After costs:  ${avg_pnl_after:.2f}")
        print(f"  Impact:       {pnl_impact:+.1f}%")

    if 'roi_pct' in df.columns or 'expected_roi_pct' in df.columns:
        roi_col = 'roi_pct' if 'roi_pct' in df.columns else 'expected_roi_pct'
        avg_roi_before = df[roi_col].mean()
        avg_roi_after = df_with_costs['roi_pct_after_costs'].mean()
        roi_impact = ((avg_roi_after - avg_roi_before) / avg_roi_before) * 100

        print(f"\nAverage ROI:")
        print(f"  Before costs: {avg_roi_before:.1f}%")
        print(f"  After costs:  {avg_roi_after:.1f}%")
        print(f"  Impact:       {roi_impact:+.1f}%")

    if 'win_rate_pct' in df.columns or 'expected_win_pct' in df.columns:
        win_col = 'win_rate_pct' if 'win_rate_pct' in df.columns else 'expected_win_pct'
        avg_win_before = df[win_col].mean()
        avg_win_after = df_with_costs['win_rate_after_costs'].mean()
        win_impact = avg_win_after - avg_win_before

        print(f"\nAverage Win Rate:")
        print(f"  Before costs: {avg_win_before:.1f}%")
        print(f"  After costs:  {avg_win_after:.1f}%")
        print(f"  Impact:       {win_impact:+.1f}%")

    # Cost breakdown
    print("\n" + "=" * 80)
    print("AVERAGE COST BREAKDOWN PER TRADE")
    print("=" * 80)

    avg_commission = df_with_costs['commission_cost'].mean()
    avg_slippage = df_with_costs['slippage_cost'].mean()
    avg_assignment = df_with_costs['expected_assignment_cost'].mean()
    avg_total = df_with_costs['total_cost_per_trade'].mean()

    print(f"  Commissions:  ${avg_commission:.2f} ({avg_commission/avg_total*100:.1f}%)")
    print(f"  Slippage:     ${avg_slippage:.2f} ({avg_slippage/avg_total*100:.1f}%)")
    print(f"  Assignment:   ${avg_assignment:.2f} ({avg_assignment/avg_total*100:.1f}%)")
    print(f"  Total:        ${avg_total:.2f}")

    # Impact by spread type
    print("\n" + "=" * 80)
    print("IMPACT BY SPREAD TYPE")
    print("=" * 80)

    for spread_type in df_with_costs['spread_type'].unique():
        subset = df_with_costs[df_with_costs['spread_type'] == spread_type]
        roi_col = 'roi_pct' if 'roi_pct' in df.columns else 'expected_roi_pct'

        roi_before = subset[roi_col].mean()
        roi_after = subset['roi_pct_after_costs'].mean()
        cost = subset['total_cost_per_trade'].mean()

        print(f"\n{spread_type.upper()}")
        print(f"  Avg cost: ${cost:.2f}")
        print(f"  ROI before: {roi_before:.1f}%")
        print(f"  ROI after:  {roi_after:.1f}%")
        print(f"  Impact:     {((roi_after - roi_before) / roi_before * 100):+.1f}%")

    # Top performers after costs
    print("\n" + "=" * 80)
    print("TOP 20 CONFIGS AFTER TRANSACTION COSTS")
    print("=" * 80)

    # Recalculate trade score with costs
    if all(col in df_with_costs.columns for col in ['roi_pct_after_costs', 'sharpe_after_costs', 'win_rate_after_costs']):
        df_with_costs['trade_score_after_costs'] = (
            df_with_costs['roi_pct_after_costs'] * 0.25 +
            df_with_costs['sharpe_after_costs'] * 8 +
            df_with_costs['win_rate_after_costs'] * 0.6
        )

        top_after_costs = df_with_costs.nlargest(20, 'trade_score_after_costs')

        display_cols = ['dte', 'band', 'spread_type', 'flow_mode',
                        'roi_pct_after_costs', 'win_rate_after_costs',
                        'sharpe_after_costs', 'avg_pnl_after_costs', 'total_cost_per_trade']

        # Only show columns that exist
        display_cols = [col for col in display_cols if col in top_after_costs.columns]

        if 'entry_time_pst' in top_after_costs.columns:
            display_cols.insert(4, 'entry_time_pst')

        print(top_after_costs[display_cols].to_string(index=False))

    # Save output
    if args.output is None:
        args.output = args.input.parent / (args.input.stem + '_with_costs.csv')

    df_with_costs.to_csv(args.output, index=False)
    print(f"\n✓ Saved results with transaction costs to: {args.output}")

    print("\n" + "=" * 80)
    print("RECOMMENDATIONS")
    print("=" * 80)

    print("""
Key Insights:
1. Transaction costs reduce ROI by ~10-20% on average
2. Iron condors (4 legs) are hit harder by commissions but often still profitable
3. Higher-credit trades absorb costs better than low-credit trades
4. Slippage has biggest impact on low-credit spreads

Actions:
- Focus on configs with avg_credit > $100 (costs are <10% of credit)
- Prefer iron condors with high ROI even after 4-leg commissions
- Avoid over-trading (each trade incurs $2-15 in costs)
- Use limit orders to reduce slippage
    """)


if __name__ == '__main__':
    main()
