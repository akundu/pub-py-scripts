#!/usr/bin/env python3
"""
Analyze Phase 1 backtest results and identify top configurations.

Filters out problematic configs and ranks by multiple metrics:
- ROI
- Win Rate (consistency)
- Total Spreads (opportunity volume)
- Risk-Adjusted Returns (Sharpe-like ratio)
"""

import pandas as pd
import argparse
import sys
from pathlib import Path

def analyze_results(csv_path: str, top_n: int = 20):
    """Analyze backtest results and identify top performers."""

    # Load results
    df = pd.read_csv(csv_path)

    print(f"\n{'='*80}")
    print(f"PHASE 1 BACKTEST ANALYSIS")
    print(f"{'='*80}")
    print(f"Total Configurations: {len(df)}")
    print(f"Configs with Spreads: {len(df[df['total_spreads'] > 0])}")
    print(f"{'='*80}\n")

    # Filter to configs with spreads
    df_valid = df[df['total_spreads'] > 0].copy()

    # Calculate additional metrics
    df_valid['consistency_pct'] = (df_valid['days_with_spreads'] / df_valid['days_processed'] * 100)
    df_valid['avg_spreads_per_day'] = df_valid['total_spreads'] / df_valid['days_with_spreads']

    # Filter out configs with unrealistic ROI (likely data issues)
    df_valid = df_valid[df_valid['avg_roi'] < 10000].copy()

    # Calculate risk-adjusted score (simple version)
    # Higher score = better (high ROI, high consistency, reasonable volume)
    df_valid['risk_score'] = (
        (df_valid['avg_roi'] / 100) *  # Normalize ROI
        (df_valid['consistency_pct'] / 100) *  # Consistency weight
        (1 + (df_valid['total_spreads'] / df_valid['total_spreads'].max()))  # Volume bonus
    )

    print("\n" + "="*80)
    print(f"TOP {top_n} CONFIGURATIONS BY ROI")
    print("="*80)
    top_roi = df_valid.nlargest(top_n, 'avg_roi')[[
        'config', 'dte', 'percentile', 'spread_width', 'flow_mode',
        'total_spreads', 'avg_roi', 'avg_credit', 'consistency_pct'
    ]]
    print(top_roi.to_string(index=False))

    print("\n" + "="*80)
    print(f"TOP {top_n} CONFIGURATIONS BY VOLUME (Total Spreads)")
    print("="*80)
    top_volume = df_valid.nlargest(top_n, 'total_spreads')[[
        'config', 'dte', 'percentile', 'spread_width', 'flow_mode',
        'total_spreads', 'avg_roi', 'avg_credit', 'consistency_pct'
    ]]
    print(top_volume.to_string(index=False))

    print("\n" + "="*80)
    print(f"TOP {top_n} CONFIGURATIONS BY CONSISTENCY")
    print("="*80)
    top_consistency = df_valid.nlargest(top_n, 'consistency_pct')[[
        'config', 'dte', 'percentile', 'spread_width', 'flow_mode',
        'total_spreads', 'avg_roi', 'consistency_pct', 'days_with_spreads', 'days_processed'
    ]]
    print(top_consistency.to_string(index=False))

    print("\n" + "="*80)
    print(f"TOP {top_n} CONFIGURATIONS BY RISK-ADJUSTED SCORE")
    print("="*80)
    top_risk_adj = df_valid.nlargest(top_n, 'risk_score')[[
        'config', 'dte', 'percentile', 'spread_width', 'flow_mode',
        'total_spreads', 'avg_roi', 'consistency_pct', 'risk_score'
    ]]
    print(top_risk_adj.to_string(index=False))

    # Analysis by parameter
    print("\n" + "="*80)
    print("PARAMETER ANALYSIS")
    print("="*80)

    print("\n--- By DTE ---")
    dte_analysis = df_valid.groupby('dte').agg({
        'avg_roi': 'mean',
        'total_spreads': 'sum',
        'consistency_pct': 'mean',
        'config': 'count'
    }).round(2)
    dte_analysis.columns = ['Avg ROI', 'Total Spreads', 'Avg Consistency %', 'Configs']
    print(dte_analysis)

    print("\n--- By Percentile ---")
    pct_analysis = df_valid.groupby('percentile').agg({
        'avg_roi': 'mean',
        'total_spreads': 'sum',
        'consistency_pct': 'mean',
        'config': 'count'
    }).round(2)
    pct_analysis.columns = ['Avg ROI', 'Total Spreads', 'Avg Consistency %', 'Configs']
    print(pct_analysis)

    print("\n--- By Spread Width ---")
    width_analysis = df_valid.groupby('spread_width').agg({
        'avg_roi': 'mean',
        'total_spreads': 'sum',
        'consistency_pct': 'mean',
        'config': 'count'
    }).round(2)
    width_analysis.columns = ['Avg ROI', 'Total Spreads', 'Avg Consistency %', 'Configs']
    print(width_analysis)

    print("\n--- By Flow Mode ---")
    flow_analysis = df_valid.groupby('flow_mode').agg({
        'avg_roi': 'mean',
        'total_spreads': 'sum',
        'consistency_pct': 'mean',
        'config': 'count'
    }).round(2)
    flow_analysis.columns = ['Avg ROI', 'Total Spreads', 'Avg Consistency %', 'Configs']
    print(flow_analysis)

    # Best combinations
    print("\n" + "="*80)
    print("BEST PARAMETER COMBINATIONS")
    print("="*80)

    print("\nBest DTE + Percentile Combos (by avg ROI):")
    combo_analysis = df_valid.groupby(['dte', 'percentile']).agg({
        'avg_roi': 'mean',
        'total_spreads': 'sum',
        'consistency_pct': 'mean'
    }).round(2).nlargest(10, 'avg_roi')
    print(combo_analysis)

    # Summary stats
    print("\n" + "="*80)
    print("SUMMARY STATISTICS")
    print("="*80)
    print(f"Total Spreads Found: {df_valid['total_spreads'].sum():,.0f}")
    print(f"Average ROI: {df_valid['avg_roi'].mean():.2f}%")
    print(f"Median ROI: {df_valid['avg_roi'].median():.2f}%")
    print(f"Best ROI: {df_valid['avg_roi'].max():.2f}%")
    print(f"Average Consistency: {df_valid['consistency_pct'].mean():.1f}%")
    print(f"Average Credit: ${df_valid['avg_credit'].mean():.2f}")
    print(f"Average Max Loss: ${df_valid['avg_max_loss'].mean():.2f}")

    # Save top configs for Phase 2
    output_path = Path(csv_path).parent / 'phase1_top_configs.csv'
    top_configs = df_valid.nlargest(top_n, 'risk_score')
    top_configs.to_csv(output_path, index=False)
    print(f"\nTop {top_n} configs saved to: {output_path}")

    print(f"\n{'='*80}\n")

    return df_valid


def main():
    parser = argparse.ArgumentParser(description='Analyze Phase 1 backtest results')
    parser.add_argument('--input', required=True, help='Path to backtest results CSV')
    parser.add_argument('--top-n', type=int, default=20, help='Number of top configs to show')

    args = parser.parse_args()

    if not Path(args.input).exists():
        print(f"ERROR: File not found: {args.input}")
        return 1

    analyze_results(args.input, args.top_n)
    return 0


if __name__ == '__main__':
    sys.exit(main())
