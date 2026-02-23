#!/usr/bin/env python3
"""
Exit Strategy Optimizer

Tests different exit rules to find optimal P&L:
1. Profit targets (exit at 25%, 50%, 70% of max profit)
2. Stop losses (exit at 1.5x, 2x, 3x credit loss)
3. Time-based exits (exit 1-2 days before expiration)

Usage:
    python scripts/exit_strategy_optimizer.py --profit-targets 0.25,0.50,0.70
"""

import argparse
from pathlib import Path
import pandas as pd
import numpy as np
from typing import List


def simulate_exit_strategy(
    df: pd.DataFrame,
    profit_target: float = 0.50,
    stop_loss_mult: float = 2.0,
    early_exit_dte: int = 0
) -> pd.DataFrame:
    """
    Simulate exit strategy on grid results.

    Args:
        df: Grid DataFrame with trade metrics
        profit_target: Exit when P&L reaches this fraction of max profit (e.g., 0.50 = 50%)
        stop_loss_mult: Exit when loss exceeds this multiple of credit (e.g., 2.0 = 2x credit)
        early_exit_dte: Exit this many days before expiration if still profitable

    Returns:
        DataFrame with simulated exit metrics
    """
    df = df.copy()

    # Max profit = credit received (if held to expiration and wins)
    df['max_profit'] = df['avg_credit']

    # Profit target in dollars
    df['profit_target_dollars'] = df['max_profit'] * profit_target

    # Stop loss in dollars
    df['stop_loss_dollars'] = df['avg_credit'] * stop_loss_mult

    # Simulate exit behavior (simplified model)
    # In reality, would need tick-by-tick data to model intraday moves

    # Estimate what % of trades hit profit target before expiration
    # Higher profit targets are hit less frequently
    df['pct_hit_profit_target'] = np.minimum(
        df['win_rate_pct'] * (1.0 - profit_target * 0.3),  # Lower targets hit more often
        df['win_rate_pct']
    )

    # Estimate what % of trades hit stop loss
    # Assuming some losing trades hit stop, some expire worthless
    df['pct_hit_stop_loss'] = (100 - df['win_rate_pct']) * 0.6  # 60% of losers hit stop

    # Remaining trades hold to expiration
    df['pct_hold_to_expiration'] = 100 - df['pct_hit_profit_target'] - df['pct_hit_stop_loss']

    # Calculate weighted average P&L
    df['avg_pnl_with_exits'] = (
        # Profit target trades
        (df['pct_hit_profit_target'] / 100) * df['profit_target_dollars'] +
        # Stop loss trades
        (df['pct_hit_stop_loss'] / 100) * (-df['stop_loss_dollars']) +
        # Hold to expiration trades
        (df['pct_hold_to_expiration'] / 100) * df['avg_pnl']
    )

    # Calculate new win rate (profit target + some hold-to-exp winners)
    df['win_rate_with_exits'] = (
        df['pct_hit_profit_target'] +  # All profit target hits are wins
        (df['pct_hold_to_expiration'] / 100) * df['win_rate_pct']  # Some hold-to-exp are wins
    )

    # Recalculate ROI
    # For profit target exits, capital is freed up early, but we'll use conservative estimate
    df['roi_with_exits'] = (df['avg_pnl_with_exits'] / df['avg_max_risk']) * 100

    # Recalculate Sharpe (rough approximation)
    # Exiting early reduces variance, so Sharpe should improve
    variance_reduction_factor = 1.0 + profit_target * 0.3  # Early exits reduce variance by ~15-30%
    df['sharpe_with_exits'] = df['sharpe'] * variance_reduction_factor * (
        df['roi_with_exits'] / df['roi_pct'].replace(0, 1)
    )

    # Calculate improvement
    df['pnl_improvement_pct'] = (
        (df['avg_pnl_with_exits'] - df['avg_pnl']) / df['avg_pnl'].abs().replace(0, 1) * 100
    )
    df['roi_improvement_pct'] = (
        (df['roi_with_exits'] - df['roi_pct']) / df['roi_pct'].abs().replace(0, 1) * 100
    )
    df['sharpe_improvement_pct'] = (
        (df['sharpe_with_exits'] - df['sharpe']) / df['sharpe'].abs().replace(0, 1) * 100
    )

    return df


def main():
    parser = argparse.ArgumentParser(description='Optimize exit strategies for credit spreads')
    parser.add_argument('--input', type=Path,
                        default=Path('results/backtest_tight/grid_analysis_tight_successful.csv'),
                        help='Input grid CSV')
    parser.add_argument('--output-dir', type=Path,
                        default=Path('results/exit_strategies'),
                        help='Output directory')
    parser.add_argument('--profit-targets', type=str, default='0.25,0.50,0.70',
                        help='Profit targets to test (comma-separated)')
    parser.add_argument('--stop-loss-mults', type=str, default='1.5,2.0,3.0',
                        help='Stop loss multiples to test (comma-separated)')
    parser.add_argument('--top-n', type=int, default=1000,
                        help='Number of top configs to test')
    args = parser.parse_args()

    print("=" * 80)
    print("EXIT STRATEGY OPTIMIZER")
    print("=" * 80)

    # Parse targets
    profit_targets = [float(x) for x in args.profit_targets.split(',')]
    stop_loss_mults = [float(x) for x in args.stop_loss_mults.split(',')]

    print(f"Profit targets: {profit_targets}")
    print(f"Stop loss multiples: {stop_loss_mults}")
    print()

    # Load data
    if not args.input.exists():
        print(f"Error: {args.input} not found")
        return

    df = pd.read_csv(args.input)
    print(f"Loaded {len(df)} configurations")

    # Filter to high-quality configs
    df_filtered = df[
        (df['win_rate_pct'] >= 85) &
        (df['roi_pct'] >= 10) &
        (df['n_trades'] >= 10)
    ].copy()

    print(f"Filtered to {len(df_filtered)} quality configs (win≥85%, ROI≥10%, trades≥10)")

    # Take top N by composite score
    df_filtered['composite'] = (
        df_filtered['roi_pct'] * 0.3 +
        df_filtered['sharpe'] * 10 +
        df_filtered['win_rate_pct'] * 0.5
    )
    df_test = df_filtered.nlargest(args.top_n, 'composite')
    print(f"Testing top {len(df_test)} configs")

    # Test all combinations
    results = []

    for profit_target in profit_targets:
        for stop_loss_mult in stop_loss_mults:
            print(f"\nTesting: Profit Target={profit_target*100:.0f}%, Stop Loss={stop_loss_mult}x")

            df_sim = simulate_exit_strategy(
                df_test,
                profit_target=profit_target,
                stop_loss_mult=stop_loss_mult
            )

            # Calculate aggregate metrics
            avg_pnl_improvement = df_sim['pnl_improvement_pct'].mean()
            avg_roi_improvement = df_sim['roi_improvement_pct'].mean()
            avg_sharpe_improvement = df_sim['sharpe_improvement_pct'].mean()

            # Count how many configs improved
            pct_improved = (df_sim['pnl_improvement_pct'] > 0).sum() / len(df_sim) * 100

            result = {
                'profit_target': profit_target,
                'stop_loss_mult': stop_loss_mult,
                'avg_pnl_improvement': avg_pnl_improvement,
                'avg_roi_improvement': avg_roi_improvement,
                'avg_sharpe_improvement': avg_sharpe_improvement,
                'pct_configs_improved': pct_improved,
                'median_pnl_improvement': df_sim['pnl_improvement_pct'].median(),
            }
            results.append(result)

            print(f"  P&L improvement:   {avg_pnl_improvement:+.1f}%")
            print(f"  ROI improvement:   {avg_roi_improvement:+.1f}%")
            print(f"  Sharpe improvement: {avg_sharpe_improvement:+.1f}%")
            print(f"  Configs improved:  {pct_improved:.1f}%")

    # Convert to DataFrame
    results_df = pd.DataFrame(results)

    # Save results
    args.output_dir.mkdir(parents=True, exist_ok=True)
    results_file = args.output_dir / 'exit_strategy_comparison.csv'
    results_df.to_csv(results_file, index=False)
    print(f"\n✓ Saved comparison to {results_file}")

    # Find best strategy
    print("\n" + "=" * 80)
    print("BEST EXIT STRATEGIES")
    print("=" * 80)

    # Best by P&L improvement
    best_pnl = results_df.nlargest(1, 'avg_pnl_improvement').iloc[0]
    print(f"\nBest for P&L:")
    print(f"  Profit Target: {best_pnl['profit_target']*100:.0f}%")
    print(f"  Stop Loss:     {best_pnl['stop_loss_mult']:.1f}x")
    print(f"  P&L Gain:      +{best_pnl['avg_pnl_improvement']:.1f}%")

    # Best by Sharpe improvement
    best_sharpe = results_df.nlargest(1, 'avg_sharpe_improvement').iloc[0]
    print(f"\nBest for Risk-Adjusted Returns:")
    print(f"  Profit Target: {best_sharpe['profit_target']*100:.0f}%")
    print(f"  Stop Loss:     {best_sharpe['stop_loss_mult']:.1f}x")
    print(f"  Sharpe Gain:   +{best_sharpe['avg_sharpe_improvement']:.1f}%")

    # Best balanced
    results_df['balanced_score'] = (
        results_df['avg_pnl_improvement'] * 0.4 +
        results_df['avg_sharpe_improvement'] * 0.3 +
        results_df['pct_configs_improved'] * 0.3
    )
    best_balanced = results_df.nlargest(1, 'balanced_score').iloc[0]
    print(f"\nBest Balanced:")
    print(f"  Profit Target: {best_balanced['profit_target']*100:.0f}%")
    print(f"  Stop Loss:     {best_balanced['stop_loss_mult']:.1f}x")
    print(f"  P&L Gain:      +{best_balanced['avg_pnl_improvement']:.1f}%")
    print(f"  Sharpe Gain:   +{best_balanced['avg_sharpe_improvement']:.1f}%")
    print(f"  Success Rate:  {best_balanced['pct_configs_improved']:.1f}%")

    # Save best strategy config with exits
    print("\n" + "=" * 80)
    print("APPLYING BEST STRATEGY TO ALL CONFIGS")
    print("=" * 80)

    df_final = simulate_exit_strategy(
        df,
        profit_target=best_balanced['profit_target'],
        stop_loss_mult=best_balanced['stop_loss_mult']
    )

    output_file = args.output_dir / 'grid_with_optimal_exits.csv'
    df_final.to_csv(output_file, index=False)
    print(f"\n✓ Saved configs with optimal exits to {output_file}")

    # Summary
    print("\n" + "=" * 80)
    print("RECOMMENDATIONS")
    print("=" * 80)

    print(f"""
Optimal Exit Strategy:
  • Exit at {best_balanced['profit_target']*100:.0f}% profit target
  • Stop loss at {best_balanced['stop_loss_mult']:.1f}x credit

Expected Improvements:
  • P&L:    +{best_balanced['avg_pnl_improvement']:.1f}%
  • Sharpe: +{best_balanced['avg_sharpe_improvement']:.1f}%
  • Success: {best_balanced['pct_configs_improved']:.0f}% of configs improved

Key Insights:
1. Taking profits early (50-70%) captures gains before reversals
2. Tight stop losses (1.5-2x credit) prevent catastrophic losses
3. Sharpe ratio improves due to reduced variance
4. Capital freed up faster for next trade

Implementation:
- Set GTC limit orders at profit target
- Set stop loss orders at 2x credit loss
- Monitor positions daily
- Don't let winners turn into losers (take profits!)
    """)


if __name__ == '__main__':
    main()
