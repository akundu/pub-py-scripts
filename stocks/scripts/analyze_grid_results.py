#!/usr/bin/env python3
"""
Analyze grid search results and extract top configurations.

This script filters and ranks grid search results by performance metrics
and exports top configurations for the next phase of testing.
"""

import argparse
import json
import pandas as pd
from pathlib import Path
from typing import List, Dict, Any
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s'
)
logger = logging.getLogger(__name__)


def calculate_sharpe_ratio(df: pd.DataFrame, config_col: str = 'config_id') -> pd.DataFrame:
    """
    Calculate Sharpe ratio for each configuration.

    Args:
        df: DataFrame with trade results
        config_col: Column name for configuration ID

    Returns:
        DataFrame with Sharpe ratios per config
    """
    sharpe_results = []

    for config_id, group in df.groupby(config_col):
        if 'roi_pct' not in group.columns or len(group) < 2:
            continue

        mean_roi = group['roi_pct'].mean()
        std_roi = group['roi_pct'].std()

        if std_roi > 0:
            sharpe = mean_roi / std_roi
        else:
            sharpe = 0.0

        sharpe_results.append({
            'config_id': config_id,
            'sharpe_ratio': sharpe,
            'mean_roi': mean_roi,
            'std_roi': std_roi
        })

    return pd.DataFrame(sharpe_results)


def aggregate_results(df: pd.DataFrame, config_col: str = 'config_id') -> pd.DataFrame:
    """
    Aggregate results by configuration.

    Args:
        df: DataFrame with trade results
        config_col: Column name for configuration ID

    Returns:
        Aggregated DataFrame with one row per config
    """
    agg_dict = {
        'win': ['sum', 'mean'],
        'roi_pct': ['mean', 'std'],
        'pnl': ['sum', 'mean', 'std'],
    }

    # Add optional columns if they exist
    if 'max_drawdown' in df.columns:
        agg_dict['max_drawdown'] = 'min'
    if 'total_trades' in df.columns:
        agg_dict['total_trades'] = 'first'

    agg_df = df.groupby(config_col).agg(agg_dict).reset_index()

    # Flatten column names
    agg_df.columns = ['_'.join(col).strip('_') if col[1] else col[0]
                      for col in agg_df.columns.values]

    # Rename for clarity
    rename_map = {
        'win_sum': 'total_wins',
        'win_mean': 'win_rate',
        'roi_pct_mean': 'avg_roi',
        'roi_pct_std': 'std_roi',
        'pnl_sum': 'total_pnl',
        'pnl_mean': 'avg_pnl',
        'pnl_std': 'std_pnl'
    }

    agg_df.rename(columns=rename_map, inplace=True)

    return agg_df


def extract_config_params(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract configuration parameters from config_id or separate columns.

    Args:
        df: DataFrame with results

    Returns:
        DataFrame with extracted parameters
    """
    # If parameters are already in separate columns, use them
    param_cols = ['dte', 'percentile', 'spread_width', 'strategy_type',
                  'direction_window_minutes', 'flow_mode', 'entry_time',
                  'profit_target_pct']

    existing_cols = [col for col in param_cols if col in df.columns]

    if existing_cols:
        return df

    # Otherwise, try to extract from config_id
    # (This would require knowing the config_id format)
    logger.warning("Configuration parameters not found in columns")
    return df


def rank_configs(
    df: pd.DataFrame,
    sort_by: List[str] = ['sharpe_ratio', 'win_rate', 'avg_roi'],
    top_n: int = 10
) -> pd.DataFrame:
    """
    Rank configurations by multiple metrics.

    Args:
        df: Aggregated DataFrame
        sort_by: List of columns to sort by (descending)
        top_n: Number of top configs to return

    Returns:
        Top N configurations
    """
    # Ensure all sort columns exist
    sort_cols = [col for col in sort_by if col in df.columns]

    if not sort_cols:
        logger.warning(f"None of the sort columns {sort_by} found in DataFrame")
        sort_cols = ['win_rate']  # Fallback

    # Sort descending by all metrics
    sorted_df = df.sort_values(sort_cols, ascending=False)

    return sorted_df.head(top_n)


def export_top_configs(
    df: pd.DataFrame,
    output_file: Path,
    config_params: List[str]
) -> None:
    """
    Export top configurations to JSON for next phase.

    Args:
        df: Top configurations DataFrame
        output_file: Output JSON file path
        config_params: List of parameter columns to export
    """
    configs = []

    for _, row in df.iterrows():
        config = {}
        for param in config_params:
            if param in row:
                config[param] = row[param]

        # Add performance metrics for reference
        config['performance'] = {
            'win_rate': float(row.get('win_rate', 0)),
            'avg_roi': float(row.get('avg_roi', 0)),
            'sharpe_ratio': float(row.get('sharpe_ratio', 0)),
            'total_trades': int(row.get('total_trades', 0))
        }

        configs.append(config)

    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, 'w') as f:
        json.dump(configs, f, indent=2)

    logger.info(f"Exported {len(configs)} configurations to {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description='Analyze grid search results and extract top configurations'
    )
    parser.add_argument('--input', required=True, help='Input CSV file')
    parser.add_argument('--output', required=True, help='Output JSON file for top configs')
    parser.add_argument('--top-n', type=int, default=10, help='Number of top configs to export')
    parser.add_argument('--sort-by', nargs='+', default=['sharpe_ratio', 'win_rate'],
                        help='Metrics to sort by')
    parser.add_argument('--summary', help='Optional summary text file output')
    parser.add_argument('--config-col', default='config_id',
                        help='Column name for configuration ID')

    args = parser.parse_args()

    # Load results
    logger.info(f"Loading results from {args.input}")
    df = pd.read_csv(args.input)
    logger.info(f"Loaded {len(df)} rows")

    # Extract configuration parameters
    df = extract_config_params(df)

    # Calculate Sharpe ratios
    sharpe_df = calculate_sharpe_ratio(df, args.config_col)

    # Aggregate by configuration
    agg_df = aggregate_results(df, args.config_col)

    # Merge Sharpe ratios
    if not sharpe_df.empty:
        agg_df = agg_df.merge(sharpe_df, on=args.config_col, how='left')

    # Merge config parameters (get first occurrence)
    param_cols = ['dte', 'percentile', 'spread_width', 'strategy_type',
                  'direction_window_minutes', 'flow_mode', 'entry_time',
                  'profit_target_pct']
    existing_params = [col for col in param_cols if col in df.columns]

    if existing_params:
        config_params_df = df.groupby(args.config_col)[existing_params].first().reset_index()
        agg_df = agg_df.merge(config_params_df, on=args.config_col, how='left')

    # Rank configurations
    top_configs = rank_configs(agg_df, args.sort_by, args.top_n)

    logger.info(f"\nTop {args.top_n} configurations:")
    print(top_configs.to_string())

    # Export to JSON
    output_path = Path(args.output)
    export_top_configs(top_configs, output_path, existing_params)

    # Generate summary if requested
    if args.summary:
        summary_path = Path(args.summary)
        summary_path.parent.mkdir(parents=True, exist_ok=True)

        with open(summary_path, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write(f"GRID SEARCH RESULTS SUMMARY\n")
            f.write("=" * 80 + "\n\n")

            f.write(f"Total configurations tested: {agg_df[args.config_col].nunique()}\n")
            f.write(f"Total trades: {len(df)}\n")
            f.write(f"Date range: {df['date'].min()} to {df['date'].max()}\n\n")

            f.write(f"Top {args.top_n} configurations by {', '.join(args.sort_by)}:\n")
            f.write("-" * 80 + "\n")
            f.write(top_configs.to_string(index=False))
            f.write("\n\n")

            f.write("Overall Statistics:\n")
            f.write("-" * 80 + "\n")
            f.write(f"Average win rate: {agg_df['win_rate'].mean():.2%}\n")
            f.write(f"Average ROI: {agg_df['avg_roi'].mean():.2f}%\n")
            if 'sharpe_ratio' in agg_df.columns:
                f.write(f"Average Sharpe: {agg_df['sharpe_ratio'].mean():.2f}\n")

        logger.info(f"Summary written to {summary_path}")

    logger.info("Analysis complete!")


if __name__ == '__main__':
    main()
