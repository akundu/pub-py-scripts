"""
Visualization Engine

Comprehensive visualization capabilities for backtesting results including
equity curves, performance comparisons, risk analysis, and statistical plots.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any, List, Optional, Tuple
import logging
import warnings
from datetime import datetime
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.offline as pyo

from ..analysis.comparison import ComparisonResult
from ..backtesting.metrics import PerformanceMetrics


class VisualizationEngine:
    """
    Comprehensive visualization engine for backtesting results.
    
    Features:
    - Equity curve overlays
    - Performance comparison charts
    - Risk analysis plots
    - Statistical visualizations
    - Interactive Plotly charts
    - Export capabilities
    """
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger(__name__)
        
        # Set style
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
        
        # Suppress warnings
        warnings.filterwarnings('ignore')
    
    def plot_equity_curves(
        self,
        results: Dict[str, Dict[str, Any]],
        title: str = "Equity Curves Comparison",
        normalize: bool = True,
        log_scale: bool = False,
        show_drawdown: bool = True,
        figsize: Tuple[int, int] = (12, 8)
    ) -> plt.Figure:
        """
        Plot equity curves for multiple strategies.
        
        Args:
            results: Dictionary mapping strategy names to results
            title: Chart title
            normalize: Whether to normalize all curves to start at 100
            log_scale: Whether to use log scale for y-axis
            show_drawdown: Whether to show drawdown shading
            figsize: Figure size
            
        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        colors = plt.cm.Set1(np.linspace(0, 1, len(results)))
        
        for i, (strategy_name, result) in enumerate(results.items()):
            try:
                # Extract equity curve
                equity_data = result.get('equity_curve', {})
                if not equity_data:
                    continue
                
                # Convert to DataFrame
                if isinstance(equity_data, dict):
                    df = pd.DataFrame.from_dict(equity_data, orient='index')
                    df.index = pd.to_datetime(df.index)
                    df.columns = ['equity']
                else:
                    df = equity_data
                
                if df.empty:
                    continue
                
                # Normalize if requested
                if normalize:
                    df['equity'] = (df['equity'] / df['equity'].iloc[0]) * 100
                
                # Plot equity curve
                ax.plot(df.index, df['equity'], 
                       label=strategy_name, 
                       color=colors[i], 
                       linewidth=2)
                
                # Add drawdown shading if requested
                if show_drawdown:
                    peak = df['equity'].expanding().max()
                    drawdown = (df['equity'] - peak) / peak * 100
                    ax.fill_between(df.index, df['equity'], peak, 
                                  alpha=0.1, color=colors[i])
                
            except Exception as e:
                self.logger.warning(f"Error plotting equity curve for {strategy_name}: {str(e)}")
                continue
        
        # Formatting
        ax.set_title(title, fontsize=16, fontweight='bold')
        ax.set_xlabel('Date', fontsize=12)
        ax.set_ylabel('Equity' + (' (Normalized)' if normalize else ''), fontsize=12)
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, alpha=0.3)
        
        if log_scale:
            ax.set_yscale('log')
        
        plt.tight_layout()
        return fig
    
    def plot_performance_comparison(
        self,
        comparison_result: ComparisonResult,
        metrics: List[str] = None,
        title: str = "Performance Comparison",
        figsize: Tuple[int, int] = (15, 10)
    ) -> plt.Figure:
        """
        Plot performance comparison charts.
        
        Args:
            comparison_result: Comparison result object
            metrics: List of metrics to plot
            title: Chart title
            figsize: Figure size
            
        Returns:
            Matplotlib figure
        """
        if metrics is None:
            metrics = ['total_return', 'sharpe_ratio', 'max_drawdown', 'win_rate']
        
        # Filter metrics that exist in comparison table
        available_metrics = [m for m in metrics if m in comparison_result.comparison_table.columns]
        
        if not available_metrics:
            raise ValueError("No valid metrics found for plotting")
        
        # Create subplots
        n_metrics = len(available_metrics)
        n_cols = min(2, n_metrics)
        n_rows = (n_metrics + 1) // 2
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
        if n_metrics == 1:
            axes = [axes]
        elif n_rows == 1:
            axes = [axes]
        else:
            axes = axes.flatten()
        
        # Plot each metric
        for i, metric in enumerate(available_metrics):
            ax = axes[i]
            
            # Get data for this metric
            data = comparison_result.comparison_table[metric].dropna()
            
            # Create bar plot
            bars = ax.bar(range(len(data)), data.values, 
                         color=plt.cm.Set1(np.linspace(0, 1, len(data))))
            
            # Formatting
            ax.set_title(f'{metric.replace("_", " ").title()}', fontweight='bold')
            ax.set_xticks(range(len(data)))
            ax.set_xticklabels(data.index, rotation=45, ha='right')
            ax.grid(True, alpha=0.3)
            
            # Add value labels on bars
            for bar, value in zip(bars, data.values):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{value:.2f}', ha='center', va='bottom')
        
        # Hide unused subplots
        for i in range(n_metrics, len(axes)):
            axes[i].set_visible(False)
        
        plt.suptitle(title, fontsize=16, fontweight='bold')
        plt.tight_layout()
        return fig
    
    def plot_risk_return_scatter(
        self,
        comparison_result: ComparisonResult,
        title: str = "Risk vs Return Analysis",
        figsize: Tuple[int, int] = (10, 8)
    ) -> plt.Figure:
        """
        Plot risk vs return scatter plot.
        
        Args:
            comparison_result: Comparison result object
            title: Chart title
            figsize: Figure size
            
        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        # Extract risk and return data
        data = comparison_result.comparison_table
        
        if 'volatility' not in data.columns or 'annualized_return' not in data.columns:
            raise ValueError("Required metrics (volatility, annualized_return) not found")
        
        # Create scatter plot
        scatter = ax.scatter(data['volatility'], data['annualized_return'], 
                           s=100, alpha=0.7, c=range(len(data)), 
                           cmap='viridis')
        
        # Add labels for each point
        for i, strategy in enumerate(data.index):
            ax.annotate(strategy, 
                       (data.loc[strategy, 'volatility'], data.loc[strategy, 'annualized_return']),
                       xytext=(5, 5), textcoords='offset points',
                       fontsize=9)
        
        # Add efficient frontier line (simplified)
        if len(data) > 2:
            # Sort by volatility
            sorted_data = data.sort_values('volatility')
            ax.plot(sorted_data['volatility'], sorted_data['annualized_return'], 
                   '--', alpha=0.5, color='gray', label='Efficient Frontier')
        
        # Formatting
        ax.set_xlabel('Volatility (%)', fontsize=12)
        ax.set_ylabel('Annualized Return (%)', fontsize=12)
        ax.set_title(title, fontsize=16, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        plt.tight_layout()
        return fig
    
    def plot_correlation_heatmap(
        self,
        comparison_result: ComparisonResult,
        title: str = "Strategy Correlation Matrix",
        figsize: Tuple[int, int] = (8, 6)
    ) -> plt.Figure:
        """
        Plot correlation heatmap between strategies.
        
        Args:
            comparison_result: Comparison result object
            title: Chart title
            figsize: Figure size
            
        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        if comparison_result.correlation_matrix.empty:
            ax.text(0.5, 0.5, 'No correlation data available', 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title(title, fontsize=16, fontweight='bold')
            return fig
        
        # Create heatmap
        sns.heatmap(comparison_result.correlation_matrix, 
                   annot=True, cmap='coolwarm', center=0,
                   square=True, ax=ax)
        
        ax.set_title(title, fontsize=16, fontweight='bold')
        plt.tight_layout()
        return fig
    
    def plot_drawdown_analysis(
        self,
        results: Dict[str, Dict[str, Any]],
        title: str = "Drawdown Analysis",
        figsize: Tuple[int, int] = (12, 8)
    ) -> plt.Figure:
        """
        Plot drawdown analysis for strategies.
        
        Args:
            results: Dictionary mapping strategy names to results
            title: Chart title
            figsize: Figure size
            
        Returns:
            Matplotlib figure
        """
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize, sharex=True)
        
        colors = plt.cm.Set1(np.linspace(0, 1, len(results)))
        
        for i, (strategy_name, result) in enumerate(results.items()):
            try:
                # Extract equity curve
                equity_data = result.get('equity_curve', {})
                if not equity_data:
                    continue
                
                # Convert to DataFrame
                if isinstance(equity_data, dict):
                    df = pd.DataFrame.from_dict(equity_data, orient='index')
                    df.index = pd.to_datetime(df.index)
                    df.columns = ['equity']
                else:
                    df = equity_data
                
                if df.empty:
                    continue
                
                # Calculate drawdown
                peak = df['equity'].expanding().max()
                drawdown = (df['equity'] - peak) / peak * 100
                
                # Plot equity curve
                ax1.plot(df.index, df['equity'], 
                        label=strategy_name, 
                        color=colors[i], 
                        linewidth=2)
                
                # Plot drawdown
                ax2.fill_between(df.index, drawdown, 0, 
                               alpha=0.3, color=colors[i], 
                               label=strategy_name)
                
            except Exception as e:
                self.logger.warning(f"Error plotting drawdown for {strategy_name}: {str(e)}")
                continue
        
        # Formatting
        ax1.set_title(title, fontsize=16, fontweight='bold')
        ax1.set_ylabel('Equity', fontsize=12)
        ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax1.grid(True, alpha=0.3)
        
        ax2.set_xlabel('Date', fontsize=12)
        ax2.set_ylabel('Drawdown (%)', fontsize=12)
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim(bottom=0)
        
        plt.tight_layout()
        return fig
    
    def create_interactive_dashboard(
        self,
        comparison_result: ComparisonResult,
        title: str = "Backtesting Results Dashboard"
    ) -> go.Figure:
        """
        Create interactive Plotly dashboard.
        
        Args:
            comparison_result: Comparison result object
            title: Dashboard title
            
        Returns:
            Plotly figure
        """
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Performance Metrics', 'Risk vs Return', 
                           'Correlation Matrix', 'Drawdown Analysis'),
            specs=[[{"type": "bar"}, {"type": "scatter"}],
                   [{"type": "heatmap"}, {"type": "scatter"}]]
        )
        
        # Performance metrics bar chart
        metrics_to_plot = ['total_return', 'sharpe_ratio', 'max_drawdown']
        for metric in metrics_to_plot:
            if metric in comparison_result.comparison_table.columns:
                fig.add_trace(
                    go.Bar(
                        x=comparison_result.comparison_table.index,
                        y=comparison_result.comparison_table[metric],
                        name=metric.replace('_', ' ').title(),
                        showlegend=True
                    ),
                    row=1, col=1
                )
        
        # Risk vs Return scatter
        if 'volatility' in comparison_result.comparison_table.columns and 'annualized_return' in comparison_result.comparison_table.columns:
            fig.add_trace(
                go.Scatter(
                    x=comparison_result.comparison_table['volatility'],
                    y=comparison_result.comparison_table['annualized_return'],
                    mode='markers+text',
                    text=comparison_result.comparison_table.index,
                    textposition="top center",
                    name='Strategies',
                    marker=dict(size=10, color='blue')
                ),
                row=1, col=2
            )
        
        # Correlation heatmap
        if not comparison_result.correlation_matrix.empty:
            fig.add_trace(
                go.Heatmap(
                    z=comparison_result.correlation_matrix.values,
                    x=comparison_result.correlation_matrix.columns,
                    y=comparison_result.correlation_matrix.index,
                    colorscale='RdBu',
                    zmid=0
                ),
                row=2, col=1
            )
        
        # Update layout
        fig.update_layout(
            title=title,
            height=800,
            showlegend=True
        )
        
        return fig
    
    def export_plots(
        self,
        comparison_result: ComparisonResult,
        results: Dict[str, Dict[str, Any]],
        output_dir: str = "./plots"
    ) -> None:
        """
        Export all plots to files.
        
        Args:
            comparison_result: Comparison result object
            results: Strategy results dictionary
            output_dir: Output directory
        """
        import os
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        try:
            # Equity curves
            fig1 = self.plot_equity_curves(results)
            fig1.savefig(f"{output_dir}/equity_curves.png", dpi=300, bbox_inches='tight')
            plt.close(fig1)
            
            # Performance comparison
            fig2 = self.plot_performance_comparison(comparison_result)
            fig2.savefig(f"{output_dir}/performance_comparison.png", dpi=300, bbox_inches='tight')
            plt.close(fig2)
            
            # Risk vs Return
            fig3 = self.plot_risk_return_scatter(comparison_result)
            fig3.savefig(f"{output_dir}/risk_return_scatter.png", dpi=300, bbox_inches='tight')
            plt.close(fig3)
            
            # Correlation heatmap
            fig4 = self.plot_correlation_heatmap(comparison_result)
            fig4.savefig(f"{output_dir}/correlation_heatmap.png", dpi=300, bbox_inches='tight')
            plt.close(fig4)
            
            # Drawdown analysis
            fig5 = self.plot_drawdown_analysis(results)
            fig5.savefig(f"{output_dir}/drawdown_analysis.png", dpi=300, bbox_inches='tight')
            plt.close(fig5)
            
            # Interactive dashboard
            interactive_fig = self.create_interactive_dashboard(comparison_result)
            interactive_fig.write_html(f"{output_dir}/interactive_dashboard.html")
            
            self.logger.info(f"All plots exported to {output_dir}")
            
        except Exception as e:
            self.logger.error(f"Error exporting plots: {str(e)}")
    
    def plot_trade_analysis(
        self,
        results: Dict[str, Dict[str, Any]],
        title: str = "Trade Analysis",
        figsize: Tuple[int, int] = (15, 10)
    ) -> plt.Figure:
        """
        Plot trade analysis charts.
        
        Args:
            results: Dictionary mapping strategy names to results
            title: Chart title
            figsize: Figure size
            
        Returns:
            Matplotlib figure
        """
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        
        # Extract trade data
        trade_data = {}
        for strategy_name, result in results.items():
            trades = result.get('trades', [])
            if trades:
                trade_data[strategy_name] = trades
        
        if not trade_data:
            fig.suptitle("No trade data available", fontsize=16)
            return fig
        
        # Plot 1: Trade P&L distribution
        ax1 = axes[0, 0]
        for strategy_name, trades in trade_data.items():
            pnl_values = [trade['pnl'] for trade in trades]
            ax1.hist(pnl_values, alpha=0.7, label=strategy_name, bins=20)
        ax1.set_title('Trade P&L Distribution')
        ax1.set_xlabel('P&L ($)')
        ax1.set_ylabel('Frequency')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Win rate comparison
        ax2 = axes[0, 1]
        win_rates = []
        strategy_names = []
        for strategy_name, trades in trade_data.items():
            winning_trades = [t for t in trades if t['pnl'] > 0]
            win_rate = len(winning_trades) / len(trades) * 100 if trades else 0
            win_rates.append(win_rate)
            strategy_names.append(strategy_name)
        
        bars = ax2.bar(strategy_names, win_rates, color=plt.cm.Set1(np.linspace(0, 1, len(strategy_names))))
        ax2.set_title('Win Rate Comparison')
        ax2.set_ylabel('Win Rate (%)')
        ax2.set_ylim(0, 100)
        
        # Add value labels on bars
        for bar, value in zip(bars, win_rates):
            ax2.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
                    f'{value:.1f}%', ha='center', va='bottom')
        
        # Plot 3: Trade duration analysis
        ax3 = axes[1, 0]
        for strategy_name, trades in trade_data.items():
            durations = [trade['duration_days'] for trade in trades]
            ax3.scatter(range(len(durations)), durations, alpha=0.7, label=strategy_name)
        ax3.set_title('Trade Duration Analysis')
        ax3.set_xlabel('Trade Number')
        ax3.set_ylabel('Duration (Days)')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: Monthly returns
        ax4 = axes[1, 1]
        for strategy_name, result in trade_data.items():
            # This would require more complex data processing
            # For now, show a placeholder
            ax4.text(0.5, 0.5, f'Monthly returns for {strategy_name}', 
                    ha='center', va='center', transform=ax4.transAxes)
        ax4.set_title('Monthly Returns')
        
        plt.suptitle(title, fontsize=16, fontweight='bold')
        plt.tight_layout()
        return fig
