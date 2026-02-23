#!/usr/bin/env python3
"""
Position Sizing Optimizer

Determines optimal position sizes for credit spread strategies using multiple methods:
1. Fixed Position Size
2. Kelly Criterion
3. Risk-Based Sizing
4. Volatility-Adjusted Sizing

Calculates expected returns, max drawdown, and risk metrics for different capital levels.
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Tuple
import pandas as pd
import numpy as np
import json

# Add project root to path
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from common.logging_utils import get_logger

logger = get_logger("position_sizing", level="INFO")


class PositionSizingOptimizer:
    """Optimizes position sizing for credit spread strategies."""

    def __init__(
        self,
        strategy_results: pd.DataFrame,
        capital_levels: List[float] = None,
        risk_free_rate: float = 0.05
    ):
        """
        Initialize optimizer.

        Args:
            strategy_results: DataFrame with backtest results
            capital_levels: List of capital amounts to test
            risk_free_rate: Annual risk-free rate (default 5%)
        """
        self.results = strategy_results
        self.capital_levels = capital_levels or [25000, 50000, 100000, 250000, 500000]
        self.risk_free_rate = risk_free_rate

    def fixed_position_sizing(
        self,
        config_row: pd.Series,
        capital: float,
        position_sizes: List[float] = None
    ) -> Dict:
        """
        Calculate returns for fixed position sizes.

        Args:
            config_row: Single row from strategy results
            capital: Total capital
            position_sizes: List of fixed position sizes to test

        Returns:
            Dict with optimal sizing and metrics
        """
        position_sizes = position_sizes or [1000, 2000, 5000, 10000, 20000]

        results = []

        for size in position_sizes:
            # Calculate how many positions we can have
            max_positions = int(capital / size)

            if max_positions == 0:
                continue

            # Daily opportunities from config
            daily_opps = config_row['avg_spreads_per_day']

            # Positions per day (limited by capital)
            positions_per_day = min(daily_opps, max_positions)

            # Expected daily profit
            avg_profit_per_position = config_row['avg_credit'] * 0.5  # 50% profit target
            expected_daily_profit = positions_per_day * avg_profit_per_position

            # Expected daily loss (assume win rate)
            win_rate = config_row.get('win_rate', 95) / 100
            loss_rate = 1 - win_rate
            avg_loss_per_position = config_row['avg_max_loss'] * loss_rate

            # Net expected daily P&L
            expected_daily_pnl = (positions_per_day * avg_profit_per_position * win_rate) - \
                                 (positions_per_day * avg_loss_per_position * loss_rate)

            # Daily return %
            daily_return_pct = (expected_daily_pnl / capital) * 100

            # Estimate max drawdown (simplified)
            # Assume worst case: consecutive losses
            consecutive_losses = int(1 / (1 - win_rate)) if win_rate < 1 else 10
            max_drawdown = min(positions_per_day * avg_loss_per_position * consecutive_losses, capital * 0.5)
            max_drawdown_pct = (max_drawdown / capital) * 100

            # Calculate Sharpe ratio
            # Simplified: assuming trading ~250 days/year
            annual_return = expected_daily_pnl * 250
            annual_return_pct = (annual_return / capital) * 100

            # Estimate volatility (std of daily returns)
            daily_std = daily_return_pct * 0.5  # Simplified estimate
            annual_std = daily_std * np.sqrt(250)

            sharpe = (annual_return_pct - self.risk_free_rate) / annual_std if annual_std > 0 else 0

            results.append({
                'position_size': size,
                'max_positions': max_positions,
                'positions_per_day': positions_per_day,
                'expected_daily_pnl': expected_daily_pnl,
                'daily_return_pct': daily_return_pct,
                'annual_return_pct': annual_return_pct,
                'max_drawdown': max_drawdown,
                'max_drawdown_pct': max_drawdown_pct,
                'sharpe_ratio': sharpe,
                'capital_utilization': (positions_per_day * size / capital) * 100
            })

        if not results:
            return None

        # Find optimal by Sharpe ratio
        results_df = pd.DataFrame(results)
        optimal_idx = results_df['sharpe_ratio'].idxmax()
        optimal = results_df.loc[optimal_idx].to_dict()

        return {
            'method': 'fixed',
            'capital': capital,
            'optimal': optimal,
            'all_results': results_df
        }

    def kelly_criterion_sizing(
        self,
        config_row: pd.Series,
        capital: float,
        kelly_fraction: float = 0.25
    ) -> Dict:
        """
        Calculate position sizing using Kelly Criterion.

        Kelly formula: f = (bp - q) / b
        Where: p = win probability, q = loss probability, b = odds

        Args:
            config_row: Single row from strategy results
            capital: Total capital
            kelly_fraction: Fraction of Kelly to use (0.25 = quarter-Kelly)

        Returns:
            Dict with Kelly sizing and metrics
        """
        # Extract metrics
        win_rate = config_row.get('win_rate', 95) / 100
        loss_rate = 1 - win_rate
        avg_credit = config_row['avg_credit']
        avg_max_loss = config_row['avg_max_loss']

        # Calculate odds (profit/loss ratio)
        avg_profit = avg_credit * 0.5  # 50% profit target
        odds = avg_profit / avg_max_loss if avg_max_loss > 0 else 0

        # Kelly fraction
        kelly_f = ((odds * win_rate) - loss_rate) / odds if odds > 0 else 0

        # Apply fractional Kelly for safety
        fractional_kelly = kelly_f * kelly_fraction

        # Position size based on Kelly
        if fractional_kelly <= 0:
            return None

        position_capital = capital * fractional_kelly
        position_size = min(position_capital, avg_max_loss * 2)  # Cap at 2x avg loss

        # Daily opportunities
        daily_opps = config_row['avg_spreads_per_day']

        # Max positions
        max_positions = int(capital / position_size)
        positions_per_day = min(daily_opps, max_positions)

        # Expected daily P&L
        expected_daily_pnl = positions_per_day * (avg_profit * win_rate - avg_max_loss * loss_rate)
        daily_return_pct = (expected_daily_pnl / capital) * 100

        # Annual metrics
        annual_return = expected_daily_pnl * 250
        annual_return_pct = (annual_return / capital) * 100

        # Estimate drawdown
        consecutive_losses = int(1 / (1 - win_rate)) if win_rate < 1 else 10
        max_drawdown = min(positions_per_day * avg_max_loss * consecutive_losses, capital * 0.5)
        max_drawdown_pct = (max_drawdown / capital) * 100

        # Sharpe
        daily_std = daily_return_pct * 0.5
        annual_std = daily_std * np.sqrt(250)
        sharpe = (annual_return_pct - self.risk_free_rate) / annual_std if annual_std > 0 else 0

        return {
            'method': 'kelly',
            'capital': capital,
            'kelly_fraction': kelly_fraction,
            'kelly_f': kelly_f,
            'fractional_kelly': fractional_kelly,
            'position_size': position_size,
            'max_positions': max_positions,
            'positions_per_day': positions_per_day,
            'expected_daily_pnl': expected_daily_pnl,
            'daily_return_pct': daily_return_pct,
            'annual_return_pct': annual_return_pct,
            'max_drawdown': max_drawdown,
            'max_drawdown_pct': max_drawdown_pct,
            'sharpe_ratio': sharpe,
            'capital_utilization': (positions_per_day * position_size / capital) * 100
        }

    def risk_based_sizing(
        self,
        config_row: pd.Series,
        capital: float,
        risk_pct: float = 2.0
    ) -> Dict:
        """
        Calculate position sizing based on risk percentage.

        Position size = (Risk% * Capital) / Max Loss

        Args:
            config_row: Single row from strategy results
            capital: Total capital
            risk_pct: Percentage of capital to risk per position

        Returns:
            Dict with risk-based sizing and metrics
        """
        # Extract metrics
        win_rate = config_row.get('win_rate', 95) / 100
        loss_rate = 1 - win_rate
        avg_credit = config_row['avg_credit']
        avg_max_loss = config_row['avg_max_loss']

        # Risk amount per position
        risk_amount = capital * (risk_pct / 100)

        # Number of contracts
        contracts = int(risk_amount / avg_max_loss)

        if contracts == 0:
            return None

        # Position size
        position_size = contracts * avg_max_loss

        # Daily opportunities
        daily_opps = config_row['avg_spreads_per_day']

        # Max positions based on capital
        max_positions = int(capital / position_size)
        positions_per_day = min(daily_opps, max_positions)

        # Expected daily P&L
        avg_profit = avg_credit * 0.5
        expected_daily_pnl = positions_per_day * contracts * (avg_profit * win_rate - avg_max_loss * loss_rate)
        daily_return_pct = (expected_daily_pnl / capital) * 100

        # Annual metrics
        annual_return = expected_daily_pnl * 250
        annual_return_pct = (annual_return / capital) * 100

        # Drawdown
        consecutive_losses = int(1 / (1 - win_rate)) if win_rate < 1 else 10
        max_drawdown = min(positions_per_day * position_size * consecutive_losses, capital * 0.5)
        max_drawdown_pct = (max_drawdown / capital) * 100

        # Sharpe
        daily_std = daily_return_pct * 0.5
        annual_std = daily_std * np.sqrt(250)
        sharpe = (annual_return_pct - self.risk_free_rate) / annual_std if annual_std > 0 else 0

        return {
            'method': 'risk_based',
            'capital': capital,
            'risk_pct': risk_pct,
            'contracts': contracts,
            'position_size': position_size,
            'max_positions': max_positions,
            'positions_per_day': positions_per_day,
            'expected_daily_pnl': expected_daily_pnl,
            'daily_return_pct': daily_return_pct,
            'annual_return_pct': annual_return_pct,
            'max_drawdown': max_drawdown,
            'max_drawdown_pct': max_drawdown_pct,
            'sharpe_ratio': sharpe,
            'capital_utilization': (positions_per_day * position_size / capital) * 100
        }

    def analyze_strategy(
        self,
        config_name: str,
        risk_tolerance: str = 'moderate'
    ) -> pd.DataFrame:
        """
        Analyze a single strategy across all capital levels and methods.

        Args:
            config_name: Strategy configuration name
            risk_tolerance: 'conservative', 'moderate', or 'aggressive'

        Returns:
            DataFrame with all sizing recommendations
        """
        # Get strategy row
        strategy_row = self.results[self.results['config'] == config_name]

        if strategy_row.empty:
            logger.error(f"Strategy {config_name} not found")
            return pd.DataFrame()

        config_row = strategy_row.iloc[0]

        # Set parameters based on risk tolerance
        if risk_tolerance == 'conservative':
            kelly_fractions = [0.125, 0.25]  # 1/8 Kelly, 1/4 Kelly
            risk_pcts = [1.0, 1.5]
        elif risk_tolerance == 'moderate':
            kelly_fractions = [0.25, 0.5]  # 1/4 Kelly, 1/2 Kelly
            risk_pcts = [2.0, 3.0]
        else:  # aggressive
            kelly_fractions = [0.5, 1.0]  # 1/2 Kelly, Full Kelly
            risk_pcts = [5.0, 7.5]

        all_results = []

        for capital in self.capital_levels:
            # Fixed position sizing
            fixed_result = self.fixed_position_sizing(config_row, capital)
            if fixed_result:
                optimal = fixed_result['optimal']
                optimal['config'] = config_name
                optimal['capital'] = capital
                optimal['method'] = 'fixed'
                all_results.append(optimal)

            # Kelly criterion
            for kelly_frac in kelly_fractions:
                kelly_result = self.kelly_criterion_sizing(config_row, capital, kelly_frac)
                if kelly_result:
                    kelly_result['config'] = config_name
                    all_results.append(kelly_result)

            # Risk-based
            for risk_pct in risk_pcts:
                risk_result = self.risk_based_sizing(config_row, capital, risk_pct)
                if risk_result:
                    risk_result['config'] = config_name
                    all_results.append(risk_result)

        return pd.DataFrame(all_results)

    def optimize_all_strategies(
        self,
        top_n: int = 20,
        risk_tolerance: str = 'moderate'
    ) -> pd.DataFrame:
        """
        Optimize position sizing for top N strategies.

        Args:
            top_n: Number of top strategies to analyze
            risk_tolerance: Risk tolerance level

        Returns:
            DataFrame with all recommendations
        """
        # Get top strategies by composite score or ROI
        if 'composite_score' in self.results.columns:
            top_strategies = self.results.nlargest(top_n, 'composite_score')
        else:
            top_strategies = self.results.nlargest(top_n, 'avg_roi')

        all_results = []

        for idx, row in top_strategies.iterrows():
            config_name = row['config']
            logger.info(f"Analyzing {config_name}...")

            results_df = self.analyze_strategy(config_name, risk_tolerance)
            if not results_df.empty:
                all_results.append(results_df)

        if not all_results:
            return pd.DataFrame()

        return pd.concat(all_results, ignore_index=True)


def main():
    parser = argparse.ArgumentParser(description='Position Sizing Optimizer')
    parser.add_argument('--results', required=True, help='Path to backtest results CSV')
    parser.add_argument('--capital-levels', nargs='+', type=float,
                        help='Capital levels to test (default: 25k, 50k, 100k, 250k, 500k)')
    parser.add_argument('--risk-tolerance', default='moderate',
                        choices=['conservative', 'moderate', 'aggressive'],
                        help='Risk tolerance level')
    parser.add_argument('--top-n', type=int, default=20, help='Analyze top N strategies')
    parser.add_argument('--output', default='results/position_sizing_recommendations.csv',
                        help='Output file')
    parser.add_argument('--strategy', help='Analyze specific strategy (config name)')

    args = parser.parse_args()

    # Load backtest results
    logger.info(f"Loading results from {args.results}...")
    results_df = pd.read_csv(args.results)

    logger.info(f"Loaded {len(results_df)} configurations")

    # Initialize optimizer
    capital_levels = args.capital_levels or [25000, 50000, 100000, 250000, 500000]
    optimizer = PositionSizingOptimizer(results_df, capital_levels)

    logger.info(f"\n{'='*80}")
    logger.info(f"POSITION SIZING OPTIMIZER")
    logger.info(f"{'='*80}")
    logger.info(f"Risk Tolerance: {args.risk_tolerance}")
    logger.info(f"Capital Levels: {', '.join([f'${x:,.0f}' for x in capital_levels])}")
    logger.info(f"{'='*80}\n")

    # Analyze
    if args.strategy:
        # Single strategy
        logger.info(f"Analyzing strategy: {args.strategy}")
        recommendations = optimizer.analyze_strategy(args.strategy, args.risk_tolerance)
    else:
        # Top N strategies
        logger.info(f"Analyzing top {args.top_n} strategies...")
        recommendations = optimizer.optimize_all_strategies(args.top_n, args.risk_tolerance)

    if recommendations.empty:
        print("No recommendations generated")
        return 1

    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    recommendations.to_csv(output_path, index=False)

    logger.info(f"\n{'='*80}")
    logger.info(f"OPTIMIZATION COMPLETE")
    logger.info(f"{'='*80}")
    logger.info(f"Results saved to: {output_path}")
    logger.info(f"Total recommendations: {len(recommendations)}")

    # Show top recommendations
    logger.info(f"\n{'='*80}")
    logger.info(f"TOP 10 BY SHARPE RATIO")
    logger.info(f"{'='*80}\n")

    top_sharpe = recommendations.nlargest(10, 'sharpe_ratio')[[
        'config', 'method', 'capital', 'positions_per_day',
        'annual_return_pct', 'max_drawdown_pct', 'sharpe_ratio'
    ]]
    logger.info(f"\n{top_sharpe.to_string(index=False)}")

    logger.info(f"\n{'='*80}")
    logger.info(f"TOP 10 BY ANNUAL RETURN %")
    logger.info(f"{'='*80}\n")

    top_return = recommendations.nlargest(10, 'annual_return_pct')[[
        'config', 'method', 'capital', 'positions_per_day',
        'annual_return_pct', 'max_drawdown_pct', 'sharpe_ratio'
    ]]
    logger.info(f"\n{top_return.to_string(index=False)}")

    # Summary by method
    logger.info(f"\n{'='*80}")
    logger.info(f"SUMMARY BY METHOD")
    logger.info(f"{'='*80}\n")

    method_summary = recommendations.groupby('method').agg({
        'annual_return_pct': 'mean',
        'max_drawdown_pct': 'mean',
        'sharpe_ratio': 'mean',
        'capital_utilization': 'mean'
    }).round(2)
    method_summary.columns = ['Avg Annual Return %', 'Avg Max DD %', 'Avg Sharpe', 'Avg Capital Util %']
    logger.info(f"\n{method_summary.to_string()}")

    logger.info(f"\n{'='*80}\n")

    return 0


if __name__ == '__main__':
    sys.exit(main())
