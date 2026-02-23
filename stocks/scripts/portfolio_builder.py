#!/usr/bin/env python3
"""
Portfolio Builder

Builds diversified portfolios of credit spread strategies by:
- Analyzing correlations between strategies
- Calculating risk contributions
- Optimizing allocations for risk-adjusted returns
- Ensuring diversification across DTEs and flow modes

Outputs optimal portfolio allocations with expected returns and risk metrics.
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Tuple
import pandas as pd
import numpy as np
import json
from itertools import combinations

# Add project root to path
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from common.logging_utils import get_logger

logger = get_logger("portfolio_builder", level="INFO")


class PortfolioBuilder:
    """Builds and optimizes portfolios of credit spread strategies."""

    def __init__(
        self,
        strategy_results: pd.DataFrame,
        position_sizing_results: pd.DataFrame = None,
        max_correlation: float = 0.7,
        min_strategies: int = 3,
        max_strategies: int = 5
    ):
        """
        Initialize portfolio builder.

        Args:
            strategy_results: DataFrame with backtest results
            position_sizing_results: DataFrame with position sizing recommendations
            max_correlation: Maximum allowed correlation between strategies
            min_strategies: Minimum strategies in portfolio
            max_strategies: Maximum strategies in portfolio
        """
        self.results = strategy_results
        self.sizing_results = position_sizing_results
        self.max_correlation = max_correlation
        self.min_strategies = min_strategies
        self.max_strategies = max_strategies

    def estimate_correlation(self, strategy1: pd.Series, strategy2: pd.Series) -> float:
        """
        Estimate correlation between two strategies.

        Since we don't have daily returns, we estimate based on:
        - DTE similarity (same DTE = higher correlation)
        - Flow mode similarity (same mode = higher correlation)
        - Percentile similarity

        Args:
            strategy1: First strategy row
            strategy2: Second strategy row

        Returns:
            Estimated correlation (0-1)
        """
        correlation = 0.0

        # DTE correlation (0 vs 1 DTE = low, 1 vs 2 DTE = high)
        dte_diff = abs(strategy1['dte'] - strategy2['dte'])
        if dte_diff == 0:
            correlation += 0.5
        elif dte_diff == 1:
            correlation += 0.3
        elif dte_diff <= 3:
            correlation += 0.2
        else:
            correlation += 0.1

        # Flow mode correlation
        if strategy1['flow_mode'] == strategy2['flow_mode']:
            correlation += 0.3
        else:
            correlation += 0.1

        # Percentile correlation
        percentile_diff = abs(strategy1['percentile'] - strategy2['percentile'])
        if percentile_diff == 0:
            correlation += 0.2
        elif percentile_diff <= 2:
            correlation += 0.1

        return min(correlation, 1.0)

    def calculate_diversification_score(self, strategies: List[pd.Series]) -> float:
        """
        Calculate diversification score for a group of strategies.

        Higher score = better diversification

        Args:
            strategies: List of strategy rows

        Returns:
            Diversification score (0-1)
        """
        if len(strategies) <= 1:
            return 0.0

        # Calculate average pairwise correlation
        total_correlation = 0.0
        pairs = 0

        for i, strat1 in enumerate(strategies):
            for strat2 in strategies[i+1:]:
                total_correlation += self.estimate_correlation(strat1, strat2)
                pairs += 1

        avg_correlation = total_correlation / pairs if pairs > 0 else 0

        # Diversification score = 1 - avg_correlation
        diversification = 1 - avg_correlation

        # Bonus for DTE diversity
        unique_dtes = len(set(s['dte'] for s in strategies))
        dte_bonus = min(unique_dtes / len(strategies), 0.5)

        # Bonus for flow mode diversity
        unique_modes = len(set(s['flow_mode'] for s in strategies))
        mode_bonus = min(unique_modes / 3, 0.3)  # Max 3 modes

        # Total score
        total_score = diversification + dte_bonus + mode_bonus
        return min(total_score, 1.0)

    def build_portfolio(
        self,
        strategies: List[str],
        capital: float,
        allocation_method: str = 'equal_weight'
    ) -> Dict:
        """
        Build a portfolio from selected strategies.

        Args:
            strategies: List of config names
            capital: Total portfolio capital
            allocation_method: 'equal_weight', 'risk_parity', or 'sharpe_weighted'

        Returns:
            Portfolio dict with allocations and metrics
        """
        # Get strategy rows
        strategy_rows = []
        for config in strategies:
            row = self.results[self.results['config'] == config]
            if not row.empty:
                strategy_rows.append(row.iloc[0])

        if not strategy_rows:
            return None

        # Calculate allocations
        allocations = {}

        if allocation_method == 'equal_weight':
            # Equal weight allocation
            weight = 1.0 / len(strategies)
            for config in strategies:
                allocations[config] = weight

        elif allocation_method == 'risk_parity':
            # Risk parity: allocate inversely to volatility/drawdown
            max_dds = []
            for row in strategy_rows:
                # Estimate max drawdown from ROI volatility
                roi_std = row.get('avg_roi', 10) * 0.2  # Simplified estimate
                max_dds.append(max(roi_std, 1.0))

            # Inverse volatility weights
            inv_vols = [1.0 / dd for dd in max_dds]
            total_inv_vol = sum(inv_vols)
            weights = [iv / total_inv_vol for iv in inv_vols]

            for config, weight in zip(strategies, weights):
                allocations[config] = weight

        elif allocation_method == 'sharpe_weighted':
            # Sharpe-weighted allocation
            sharpes = []
            for row in strategy_rows:
                sharpe = row.get('sharpe_estimate', 1.0)
                sharpes.append(max(sharpe, 0.1))

            total_sharpe = sum(sharpes)
            weights = [s / total_sharpe for s in sharpes]

            for config, weight in zip(strategies, weights):
                allocations[config] = weight

        # Calculate portfolio metrics
        portfolio_roi = 0.0
        portfolio_spreads = 0.0
        portfolio_consistency = 0.0

        for config, weight in allocations.items():
            row = self.results[self.results['config'] == config].iloc[0]
            portfolio_roi += weight * row['avg_roi']
            portfolio_spreads += weight * row['total_spreads']
            portfolio_consistency += weight * row.get('consistency_pct', 0)

        # Diversification score
        diversification = self.calculate_diversification_score(strategy_rows)

        # Estimate portfolio Sharpe (simplified)
        # Portfolio variance is reduced by diversification
        avg_sharpe = np.mean([row.get('sharpe_estimate', 1.0) for row in strategy_rows])
        portfolio_sharpe = avg_sharpe * (1 + diversification * 0.5)

        # Check correlation constraint
        max_correlation = 0.0
        for i, row1 in enumerate(strategy_rows):
            for row2 in strategy_rows[i+1:]:
                corr = self.estimate_correlation(row1, row2)
                max_correlation = max(max_correlation, corr)

        return {
            'strategies': strategies,
            'allocations': allocations,
            'capital': capital,
            'method': allocation_method,
            'portfolio_roi': portfolio_roi,
            'portfolio_sharpe': portfolio_sharpe,
            'portfolio_spreads': portfolio_spreads,
            'portfolio_consistency': portfolio_consistency,
            'diversification_score': diversification,
            'max_correlation': max_correlation,
            'num_strategies': len(strategies),
            'passes_correlation_filter': max_correlation <= self.max_correlation
        }

    def find_optimal_portfolios(
        self,
        capital: float,
        top_n_strategies: int = 20,
        num_portfolios: int = 10
    ) -> pd.DataFrame:
        """
        Find optimal portfolios from top strategies.

        Args:
            capital: Portfolio capital
            top_n_strategies: Consider top N strategies
            num_portfolios: Number of portfolios to generate

        Returns:
            DataFrame with portfolio recommendations
        """
        # Get top strategies
        if 'composite_score' in self.results.columns:
            top_strategies = self.results.nlargest(top_n_strategies, 'composite_score')
        else:
            top_strategies = self.results.nlargest(top_n_strategies, 'avg_roi')

        candidate_configs = top_strategies['config'].tolist()

        all_portfolios = []

        # Generate portfolios of different sizes
        for n_strats in range(self.min_strategies, self.max_strategies + 1):
            # Test all combinations of size n_strats
            for combo in combinations(candidate_configs[:min(15, len(candidate_configs))], n_strats):
                # Build portfolio
                for method in ['equal_weight', 'risk_parity', 'sharpe_weighted']:
                    portfolio = self.build_portfolio(list(combo), capital, method)

                    if portfolio and portfolio['passes_correlation_filter']:
                        all_portfolios.append(portfolio)

        if not all_portfolios:
            logger.warning("No portfolios pass correlation filter")
            return pd.DataFrame()

        # Convert to DataFrame
        portfolio_records = []
        for p in all_portfolios:
            record = {
                'portfolio_name': f"{p['num_strategies']}_strat_{p['method'][:4]}",
                'num_strategies': p['num_strategies'],
                'allocation_method': p['method'],
                'capital': p['capital'],
                'expected_roi': p['portfolio_roi'],
                'estimated_sharpe': p['portfolio_sharpe'],
                'total_spreads': p['portfolio_spreads'],
                'consistency': p['portfolio_consistency'],
                'diversification': p['diversification_score'],
                'max_correlation': p['max_correlation'],
                'strategies': ', '.join(p['strategies']),
            }

            # Add individual allocations
            for config, weight in p['allocations'].items():
                record[f'alloc_{config}'] = weight

            portfolio_records.append(record)

        portfolios_df = pd.DataFrame(portfolio_records)

        # Sort by Sharpe ratio
        portfolios_df = portfolios_df.sort_values('estimated_sharpe', ascending=False)

        # Return top portfolios
        return portfolios_df.head(num_portfolios)

    def build_risk_tiered_portfolios(
        self,
        capital: float
    ) -> Dict[str, Dict]:
        """
        Build portfolios for different risk tolerance levels.

        Args:
            capital: Portfolio capital

        Returns:
            Dict with 'conservative', 'moderate', 'aggressive' portfolios
        """
        portfolios = {}

        # Conservative: High percentile, longer DTE, lower Sharpe
        conservative_strategies = self.results[
            (self.results['percentile'] >= 99) &
            (self.results['dte'] >= 2)
        ].nlargest(5, 'consistency_pct')

        if not conservative_strategies.empty:
            configs = conservative_strategies['config'].tolist()[:3]
            portfolios['conservative'] = self.build_portfolio(
                configs, capital, 'risk_parity'
            )

        # Moderate: Balanced approach
        moderate_strategies = self.results[
            (self.results['percentile'] >= 98) &
            (self.results['dte'].isin([1, 2, 3]))
        ].nlargest(5, 'composite_score' if 'composite_score' in self.results.columns else 'avg_roi')

        if not moderate_strategies.empty:
            configs = moderate_strategies['config'].tolist()[:4]
            portfolios['moderate'] = self.build_portfolio(
                configs, capital, 'sharpe_weighted'
            )

        # Aggressive: High ROI, lower percentile, 0-1 DTE
        aggressive_strategies = self.results[
            (self.results['dte'] <= 1)
        ].nlargest(5, 'avg_roi')

        if not aggressive_strategies.empty:
            configs = aggressive_strategies['config'].tolist()[:5]
            portfolios['aggressive'] = self.build_portfolio(
                configs, capital, 'sharpe_weighted'
            )

        return portfolios


def main():
    parser = argparse.ArgumentParser(description='Portfolio Builder')
    parser.add_argument('--results', required=True, help='Path to backtest results CSV')
    parser.add_argument('--sizing', help='Path to position sizing results CSV (optional)')
    parser.add_argument('--capital', type=float, default=100000, help='Portfolio capital')
    parser.add_argument('--max-correlation', type=float, default=0.7,
                        help='Max correlation between strategies')
    parser.add_argument('--min-strategies', type=int, default=3, help='Min strategies in portfolio')
    parser.add_argument('--max-strategies', type=int, default=5, help='Max strategies in portfolio')
    parser.add_argument('--top-n', type=int, default=20, help='Consider top N strategies')
    parser.add_argument('--num-portfolios', type=int, default=10, help='Generate N portfolios')
    parser.add_argument('--output', default='results/optimal_portfolios.csv', help='Output file')
    parser.add_argument('--build-tiered', action='store_true',
                        help='Build risk-tiered portfolios')

    args = parser.parse_args()

    # Load results
    logger.info(f"Loading results from {args.results}...")
    results_df = pd.read_csv(args.results)

    sizing_df = None
    if args.sizing:
        logger.info(f"Loading position sizing from {args.sizing}...")
        sizing_df = pd.read_csv(args.sizing)

    # Initialize builder
    builder = PortfolioBuilder(
        results_df,
        sizing_df,
        max_correlation=args.max_correlation,
        min_strategies=args.min_strategies,
        max_strategies=args.max_strategies
    )

    logger.info(f"\n{'='*80}")
    logger.info(f"PORTFOLIO BUILDER")
    logger.info(f"{'='*80}")
    logger.info(f"Capital: ${args.capital:,.0f}")
    logger.info(f"Max Correlation: {args.max_correlation}")
    logger.info(f"Strategy Range: {args.min_strategies}-{args.max_strategies}")
    logger.info(f"{'='*80}\n")

    if args.build_tiered:
        # Build risk-tiered portfolios
        logger.info("Building risk-tiered portfolios...")
        tiered_portfolios = builder.build_risk_tiered_portfolios(args.capital)

        # Save to JSON
        output_path = Path(args.output).parent / 'risk_tiered_portfolios.json'
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Convert to serializable format
        serializable = {}
        for risk_level, portfolio in tiered_portfolios.items():
            if portfolio:
                serializable[risk_level] = {
                    'strategies': portfolio['strategies'],
                    'allocations': {k: float(v) for k, v in portfolio['allocations'].items()},
                    'expected_roi': float(portfolio['portfolio_roi']),
                    'estimated_sharpe': float(portfolio['portfolio_sharpe']),
                    'diversification': float(portfolio['diversification_score'])
                }

        with open(output_path, 'w') as f:
            json.dump(serializable, f, indent=2)

        logger.info(f"Risk-tiered portfolios saved to: {output_path}")

        # Print summary
        for risk_level, portfolio in tiered_portfolios.items():
            if portfolio:
                logger.info(f"\n{risk_level.upper()} PORTFOLIO:")
                logger.info(f"  Strategies: {len(portfolio['strategies'])}")
                logger.info(f"  Expected ROI: {portfolio['portfolio_roi']:.1f}%")
                logger.info(f"  Estimated Sharpe: {portfolio['portfolio_sharpe']:.2f}")
                logger.info(f"  Diversification: {portfolio['diversification_score']:.2f}")
                logger.info(f"  Allocations:")
                for config, weight in portfolio['allocations'].items():
                    logger.info(f"    {config}: {weight*100:.1f}%")

    else:
        # Find optimal portfolios
        logger.info(f"Finding optimal portfolios from top {args.top_n} strategies...")
        portfolios_df = builder.find_optimal_portfolios(
            capital=args.capital,
            top_n_strategies=args.top_n,
            num_portfolios=args.num_portfolios
        )

        if portfolios_df.empty:
            print("No portfolios generated")
            return 1

        # Save results
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        portfolios_df.to_csv(output_path, index=False)

        logger.info(f"\n{'='*80}")
        logger.info(f"PORTFOLIO OPTIMIZATION COMPLETE")
        logger.info(f"{'='*80}")
        logger.info(f"Results saved to: {output_path}")
        logger.info(f"Portfolios generated: {len(portfolios_df)}")

        # Show top portfolios
        logger.info(f"\n{'='*80}")
        logger.info(f"TOP {min(5, len(portfolios_df))} PORTFOLIOS BY SHARPE RATIO")
        logger.info(f"{'='*80}\n")

        top_portfolios = portfolios_df.head(5)[[
            'portfolio_name', 'num_strategies', 'expected_roi',
            'estimated_sharpe', 'diversification', 'max_correlation'
        ]]
        logger.info(f"\n{top_portfolios.to_string(index=False)}")

        # Show best portfolio details
        if not portfolios_df.empty:
            best = portfolios_df.iloc[0]
            logger.info(f"\n{'='*80}")
            logger.info(f"BEST PORTFOLIO DETAILS")
            logger.info(f"{'='*80}")
            logger.info(f"Name: {best['portfolio_name']}")
            logger.info(f"Capital: ${best['capital']:,.0f}")
            logger.info(f"Expected ROI: {best['expected_roi']:.1f}%")
            logger.info(f"Estimated Sharpe: {best['estimated_sharpe']:.2f}")
            logger.info(f"Diversification: {best['diversification']:.2f}")
            logger.info(f"Consistency: {best['consistency']:.1f}%")
            logger.info(f"\nStrategies:")
            for strat in best['strategies'].split(', '):
                logger.info(f"  - {strat}")

    logger.info(f"\n{'='*80}\n")

    return 0


if __name__ == '__main__':
    sys.exit(main())
