#!/usr/bin/env python3
"""
Generate comprehensive backtest reports from grid search results.

This script creates markdown reports with performance analysis,
strategy comparisons, and production recommendations.
"""

import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s'
)
logger = logging.getLogger(__name__)


class BacktestReportGenerator:
    """Generate comprehensive backtest reports."""

    def __init__(self, df: pd.DataFrame):
        """
        Initialize report generator.

        Args:
            df: DataFrame with backtest results
        """
        self.df = df
        self.config_params = self._identify_config_params()

    def _identify_config_params(self) -> List[str]:
        """Identify configuration parameter columns."""
        possible_params = [
            'strategy_type', 'dte', 'percentile', 'spread_width',
            'direction_window_minutes', 'flow_mode', 'entry_time',
            'profit_target_pct', 'config_id'
        ]
        return [col for col in possible_params if col in self.df.columns]

    def generate_report(self, output_file: Path) -> None:
        """
        Generate full markdown report.

        Args:
            output_file: Output markdown file path
        """
        output_file.parent.mkdir(parents=True, exist_ok=True)

        with open(output_file, 'w') as f:
            self._write_header(f)
            self._write_dataset_summary(f)
            self._write_overall_stats(f)
            self._write_top_by_win_rate(f)
            self._write_top_by_roi(f)
            self._write_top_by_sharpe(f)
            self._write_strategy_comparison(f)
            self._write_percentile_analysis(f)
            self._write_dte_analysis(f)
            self._write_flow_mode_analysis(f)
            self._write_profit_target_analysis(f)
            self._write_production_recommendations(f)

        logger.info(f"Report generated: {output_file}")

    def _write_header(self, f):
        """Write report header."""
        f.write("# Percentile-Based Strike Selection Backtest Results\n\n")
        f.write(f"**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write("---\n\n")

    def _write_dataset_summary(self, f):
        """Write dataset summary section."""
        f.write("## Dataset Summary\n\n")

        if 'date' in self.df.columns:
            start_date = self.df['date'].min()
            end_date = self.df['date'].max()
            num_days = self.df['date'].nunique()
            f.write(f"- **Date Range**: {start_date} to {end_date}\n")
            f.write(f"- **Trading Days**: {num_days}\n")

        if 'config_id' in self.df.columns:
            num_configs = self.df['config_id'].nunique()
            f.write(f"- **Configurations Tested**: {num_configs}\n")

        f.write(f"- **Total Trades**: {len(self.df)}\n\n")

    def _write_overall_stats(self, f):
        """Write overall statistics section."""
        f.write("## Overall Statistics\n\n")

        if 'win' in self.df.columns:
            win_rate = self.df['win'].mean() * 100
            f.write(f"- **Overall Win Rate**: {win_rate:.2f}%\n")

        if 'roi_pct' in self.df.columns:
            avg_roi = self.df['roi_pct'].mean()
            f.write(f"- **Average ROI**: {avg_roi:.2f}%\n")

        if 'pnl' in self.df.columns:
            total_pnl = self.df['pnl'].sum()
            avg_pnl = self.df['pnl'].mean()
            f.write(f"- **Total P&L**: ${total_pnl:,.2f}\n")
            f.write(f"- **Average P&L per Trade**: ${avg_pnl:,.2f}\n")

        f.write("\n")

    def _write_top_by_win_rate(self, f, top_n: int = 10):
        """Write top configurations by win rate."""
        f.write(f"## Top {top_n} Configurations by Win Rate\n\n")

        if 'config_id' not in self.df.columns or 'win' not in self.df.columns:
            f.write("*Configuration data not available*\n\n")
            return

        grouped = self.df.groupby('config_id').agg({
            'win': ['sum', 'mean', 'count'],
            'roi_pct': 'mean',
            'pnl': ['mean', 'sum']
        }).reset_index()

        grouped.columns = ['_'.join(col).strip('_') if col[1] else col[0]
                          for col in grouped.columns.values]

        top = grouped.nlargest(top_n, 'win_mean')

        self._write_config_table(f, top, [
            ('Config', 'config_id'),
            ('Trades', 'win_count'),
            ('Win Rate', 'win_mean', '.2%'),
            ('Avg ROI', 'roi_pct_mean', '.2f'),
            ('Avg P&L', 'pnl_mean', ',.2f'),
            ('Total P&L', 'pnl_sum', ',.2f')
        ])

    def _write_top_by_roi(self, f, top_n: int = 10):
        """Write top configurations by ROI."""
        f.write(f"## Top {top_n} Configurations by ROI\n\n")

        if 'config_id' not in self.df.columns or 'roi_pct' not in self.df.columns:
            f.write("*Configuration data not available*\n\n")
            return

        grouped = self.df.groupby('config_id').agg({
            'win': ['sum', 'mean', 'count'],
            'roi_pct': 'mean',
            'pnl': ['mean', 'sum']
        }).reset_index()

        grouped.columns = ['_'.join(col).strip('_') if col[1] else col[0]
                          for col in grouped.columns.values]

        top = grouped.nlargest(top_n, 'roi_pct_mean')

        self._write_config_table(f, top, [
            ('Config', 'config_id'),
            ('Trades', 'win_count'),
            ('Win Rate', 'win_mean', '.2%'),
            ('Avg ROI', 'roi_pct_mean', '.2f'),
            ('Avg P&L', 'pnl_mean', ',.2f'),
            ('Total P&L', 'pnl_sum', ',.2f')
        ])

    def _write_top_by_sharpe(self, f, top_n: int = 10):
        """Write top configurations by Sharpe ratio."""
        f.write(f"## Top {top_n} Configurations by Sharpe Ratio\n\n")

        if 'config_id' not in self.df.columns or 'roi_pct' not in self.df.columns:
            f.write("*Configuration data not available*\n\n")
            return

        # Calculate Sharpe ratio
        sharpe_data = []
        for config_id, group in self.df.groupby('config_id'):
            if len(group) < 2:
                continue

            mean_roi = group['roi_pct'].mean()
            std_roi = group['roi_pct'].std()
            sharpe = mean_roi / std_roi if std_roi > 0 else 0

            sharpe_data.append({
                'config_id': config_id,
                'sharpe_ratio': sharpe,
                'trades': len(group),
                'win_rate': group['win'].mean(),
                'avg_roi': mean_roi,
                'total_pnl': group['pnl'].sum()
            })

        sharpe_df = pd.DataFrame(sharpe_data)
        top = sharpe_df.nlargest(top_n, 'sharpe_ratio')

        self._write_config_table(f, top, [
            ('Config', 'config_id'),
            ('Trades', 'trades'),
            ('Sharpe', 'sharpe_ratio', '.2f'),
            ('Win Rate', 'win_rate', '.2%'),
            ('Avg ROI', 'avg_roi', '.2f'),
            ('Total P&L', 'total_pnl', ',.2f')
        ])

    def _write_strategy_comparison(self, f):
        """Write strategy type comparison."""
        f.write("## Strategy Type Comparison\n\n")

        if 'strategy_type' not in self.df.columns:
            f.write("*Strategy type data not available*\n\n")
            return

        comparison = self.df.groupby('strategy_type').agg({
            'win': ['count', 'mean'],
            'roi_pct': 'mean',
            'pnl': ['mean', 'sum']
        }).reset_index()

        comparison.columns = ['Strategy', 'Trades', 'Win Rate', 'Avg ROI', 'Avg P&L', 'Total P&L']

        self._write_simple_table(f, comparison, {
            'Win Rate': '.2%',
            'Avg ROI': '.2f',
            'Avg P&L': ',.2f',
            'Total P&L': ',.2f'
        })

    def _write_percentile_analysis(self, f):
        """Write percentile analysis."""
        f.write("## Percentile Analysis\n\n")

        if 'percentile' not in self.df.columns:
            f.write("*Percentile data not available*\n\n")
            return

        analysis = self.df.groupby('percentile').agg({
            'win': ['count', 'mean'],
            'roi_pct': 'mean',
            'pnl': 'mean'
        }).reset_index()

        analysis.columns = ['Percentile', 'Trades', 'Win Rate', 'Avg ROI', 'Avg P&L']

        self._write_simple_table(f, analysis, {
            'Win Rate': '.2%',
            'Avg ROI': '.2f',
            'Avg P&L': ',.2f'
        })

    def _write_dte_analysis(self, f):
        """Write DTE analysis."""
        f.write("## DTE Analysis\n\n")

        if 'dte' not in self.df.columns:
            f.write("*DTE data not available*\n\n")
            return

        analysis = self.df.groupby('dte').agg({
            'win': ['count', 'mean'],
            'roi_pct': 'mean',
            'pnl': 'mean'
        }).reset_index()

        analysis.columns = ['DTE', 'Trades', 'Win Rate', 'Avg ROI', 'Avg P&L']

        self._write_simple_table(f, analysis, {
            'Win Rate': '.2%',
            'Avg ROI': '.2f',
            'Avg P&L': ',.2f'
        })

    def _write_flow_mode_analysis(self, f):
        """Write flow mode analysis."""
        f.write("## Flow Mode Analysis\n\n")

        if 'flow_mode' not in self.df.columns:
            f.write("*Flow mode data not available*\n\n")
            return

        analysis = self.df.groupby('flow_mode').agg({
            'win': ['count', 'mean'],
            'roi_pct': 'mean',
            'pnl': 'mean'
        }).reset_index()

        analysis.columns = ['Flow Mode', 'Trades', 'Win Rate', 'Avg ROI', 'Avg P&L']

        self._write_simple_table(f, analysis, {
            'Win Rate': '.2%',
            'Avg ROI': '.2f',
            'Avg P&L': ',.2f'
        })

    def _write_profit_target_analysis(self, f):
        """Write profit target analysis."""
        f.write("## Profit Target Analysis\n\n")

        if 'profit_target_pct' not in self.df.columns:
            f.write("*Profit target data not available*\n\n")
            return

        analysis = self.df.groupby('profit_target_pct').agg({
            'win': ['count', 'mean'],
            'roi_pct': 'mean',
            'pnl': 'mean'
        }).reset_index()

        analysis.columns = ['Profit Target', 'Trades', 'Win Rate', 'Avg ROI', 'Avg P&L']

        self._write_simple_table(f, analysis, {
            'Profit Target': '.0%',
            'Win Rate': '.2%',
            'Avg ROI': '.2f',
            'Avg P&L': ',.2f'
        })

    def _write_production_recommendations(self, f):
        """Write production strategy recommendations."""
        f.write("## Production Strategy Recommendations\n\n")

        f.write("Based on the backtest results, the following strategies are recommended:\n\n")

        # Find best by different criteria
        if 'config_id' in self.df.columns:
            # Best win rate
            win_rate_config = self.df.groupby('config_id')['win'].mean().idxmax()
            win_rate = self.df.groupby('config_id')['win'].mean().max()

            # Best ROI
            roi_config = self.df.groupby('config_id')['roi_pct'].mean().idxmax()
            roi = self.df.groupby('config_id')['roi_pct'].mean().max()

            f.write(f"### Conservative Strategy (Highest Win Rate)\n\n")
            f.write(f"- **Config**: {win_rate_config}\n")
            f.write(f"- **Win Rate**: {win_rate:.2%}\n")
            f.write(f"- **Use Case**: Capital preservation, steady income\n\n")

            f.write(f"### Aggressive Strategy (Highest ROI)\n\n")
            f.write(f"- **Config**: {roi_config}\n")
            f.write(f"- **ROI**: {roi:.2f}%\n")
            f.write(f"- **Use Case**: Maximum returns, higher risk tolerance\n\n")

    def _write_config_table(self, f, df: pd.DataFrame, columns: List[tuple]):
        """Write a formatted configuration table."""
        header = "| " + " | ".join([col[0] for col in columns]) + " |\n"
        separator = "| " + " | ".join(['---' for _ in columns]) + " |\n"

        f.write(header)
        f.write(separator)

        for _, row in df.iterrows():
            values = []
            for col_spec in columns:
                col_name = col_spec[1]
                fmt = col_spec[2] if len(col_spec) > 2 else ''

                value = row.get(col_name, '')

                if fmt and pd.notna(value):
                    if fmt.endswith('%'):
                        value = f"{value:{fmt}}"
                    else:
                        value = f"{value:{fmt}}"

                values.append(str(value))

            f.write("| " + " | ".join(values) + " |\n")

        f.write("\n")

    def _write_simple_table(self, f, df: pd.DataFrame, format_map: Dict[str, str]):
        """Write a simple table from DataFrame."""
        # Create header
        header = "| " + " | ".join(df.columns) + " |\n"
        separator = "| " + " | ".join(['---' for _ in df.columns]) + " |\n"

        f.write(header)
        f.write(separator)

        # Write rows
        for _, row in df.iterrows():
            values = []
            for col in df.columns:
                value = row[col]
                fmt = format_map.get(col, '')

                if fmt and pd.notna(value):
                    if fmt.endswith('%'):
                        value = f"{value:{fmt}}"
                    else:
                        value = f"{value:{fmt}}"

                values.append(str(value))

            f.write("| " + " | ".join(values) + " |\n")

        f.write("\n")


def main():
    parser = argparse.ArgumentParser(
        description='Generate backtest report from grid search results'
    )
    parser.add_argument('--input', required=True, help='Input CSV file')
    parser.add_argument('--output', required=True, help='Output markdown file')

    args = parser.parse_args()

    # Load data
    logger.info(f"Loading results from {args.input}")
    df = pd.read_csv(args.input)
    logger.info(f"Loaded {len(df)} rows")

    # Generate report
    generator = BacktestReportGenerator(df)
    output_path = Path(args.output)
    generator.generate_report(output_path)

    logger.info("Report generation complete!")


if __name__ == '__main__':
    main()
