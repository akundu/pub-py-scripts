"""
Visualization module for streak analysis.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)

# Set style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")


def plot_streak_histogram(streak_stats: Dict, figsize: Tuple[int, int] = (12, 8)) -> plt.Figure:
    """
    Plot histogram of streak lengths.
    
    Args:
        streak_stats: Streak statistics from compute_streak_stats
        figsize: Figure size (width, height)
        
    Returns:
        Matplotlib figure object
    """
    if not streak_stats or 'streaks' not in streak_stats:
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(0.5, 0.5, 'No streak data available', ha='center', va='center', transform=ax.transAxes)
        return fig
    
    streaks_df = streak_stats['streaks']
    if streaks_df.empty:
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(0.5, 0.5, 'No streaks found', ha='center', va='center', transform=ax.transAxes)
        return fig
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize)
    
    # Separate positive and negative streaks
    positive_streaks = streaks_df[streaks_df['direction'] > 0]
    negative_streaks = streaks_df[streaks_df['direction'] < 0]
    
    # Plot positive streaks
    if len(positive_streaks) > 0:
        ax1.hist(positive_streaks['length'], bins=20, alpha=0.7, color='green', 
                label=f'Positive Streaks (N={len(positive_streaks)})')
        ax1.axvline(positive_streaks['length'].mean(), color='darkgreen', linestyle='--', 
                   label=f'Mean: {positive_streaks["length"].mean():.1f}')
    
    ax1.set_title('Positive Streak Length Distribution')
    ax1.set_xlabel('Streak Length (days)')
    ax1.set_ylabel('Frequency')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot negative streaks
    if len(negative_streaks) > 0:
        ax2.hist(negative_streaks['length'], bins=20, alpha=0.7, color='red',
                label=f'Negative Streaks (N={len(negative_streaks)})')
        ax2.axvline(negative_streaks['length'].mean(), color='darkred', linestyle='--',
                   label=f'Mean: {negative_streaks["length"].mean():.1f}')
    
    ax2.set_title('Negative Streak Length Distribution')
    ax2.set_xlabel('Streak Length (days)')
    ax2.set_ylabel('Frequency')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


def plot_forward_returns(streak_stats: Dict, figsize: Tuple[int, int] = (12, 8)) -> plt.Figure:
    """
    Plot forward return distributions conditional on streak length.
    
    Args:
        streak_stats: Streak statistics from compute_streak_stats
        figsize: Figure size (width, height)
        
    Returns:
        Matplotlib figure object
    """
    if not streak_stats or 'streaks' not in streak_stats:
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(0.5, 0.5, 'No streak data available', ha='center', va='center', transform=ax.transAxes)
        return fig
    
    streaks_df = streak_stats['streaks']
    if streaks_df.empty:
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(0.5, 0.5, 'No streaks found', ha='center', va='center', transform=ax.transAxes)
        return fig
    
    # Get forward return columns
    forward_cols = [col for col in streaks_df.columns if col.startswith('forward_return_')]
    if not forward_cols:
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(0.5, 0.5, 'No forward return data available', ha='center', va='center', transform=ax.transAxes)
        return fig
    
    n_horizons = len(forward_cols)
    fig, axes = plt.subplots(2, (n_horizons + 1) // 2, figsize=figsize)
    if n_horizons == 1:
        axes = [axes]
    elif n_horizons == 2:
        axes = axes.flatten()
    
    for i, col in enumerate(forward_cols):
        horizon = int(col.split('_')[-1])
        ax = axes[i] if n_horizons > 1 else axes
        
        # Separate positive and negative streaks
        positive_data = streaks_df[streaks_df['direction'] > 0][col].dropna()
        negative_data = streaks_df[streaks_df['direction'] < 0][col].dropna()
        
        if len(positive_data) > 0:
            ax.hist(positive_data, bins=15, alpha=0.7, color='green', 
                   label=f'After +ve streaks (N={len(positive_data)})')
        
        if len(negative_data) > 0:
            ax.hist(negative_data, bins=15, alpha=0.7, color='red',
                   label=f'After -ve streaks (N={len(negative_data)})')
        
        ax.set_title(f'{horizon}-Day Forward Returns')
        ax.set_xlabel('Return')
        ax.set_ylabel('Frequency')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Add mean lines
        if len(positive_data) > 0:
            ax.axvline(positive_data.mean(), color='darkgreen', linestyle='--', alpha=0.8)
        if len(negative_data) > 0:
            ax.axvline(negative_data.mean(), color='darkred', linestyle='--', alpha=0.8)
    
    plt.tight_layout()
    return fig


def plot_momentum_analysis(streak_stats: Dict, figsize: Tuple[int, int] = (12, 8)) -> plt.Figure:
    """
    Plot momentum vs reversal probability analysis.
    
    Args:
        streak_stats: Streak statistics from compute_streak_stats
        figsize: Figure size (width, height)
        
    Returns:
        Matplotlib figure object
    """
    if not streak_stats or 'momentum_metrics' not in streak_stats:
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(0.5, 0.5, 'No momentum data available', ha='center', va='center', transform=ax.transAxes)
        return fig
    
    momentum_metrics = streak_stats['momentum_metrics']
    if not momentum_metrics:
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(0.5, 0.5, 'No momentum metrics available', ha='center', va='center', transform=ax.transAxes)
        return fig
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    # Extract data for plotting
    horizons = []
    win_rates = []
    mean_returns = []
    sample_sizes = []
    
    for key, metrics in momentum_metrics.items():
        if 'win_rate' in metrics:
            horizon = int(key.split('_')[1])
            horizons.append(horizon)
            win_rates.append(metrics['win_rate'])
            mean_returns.append(metrics.get('mean_return', 0))
            sample_sizes.append(metrics.get('sample_size', 0))
    
    if not horizons:
        ax1.text(0.5, 0.5, 'No win rate data available', ha='center', va='center', transform=ax1.transAxes)
        ax2.text(0.5, 0.5, 'No return data available', ha='center', va='center', transform=ax2.transAxes)
        return fig
    
    # Sort by horizon
    sorted_data = sorted(zip(horizons, win_rates, mean_returns, sample_sizes))
    horizons, win_rates, mean_returns, sample_sizes = zip(*sorted_data)
    
    # Plot win rates
    ax1.plot(horizons, win_rates, 'bo-', linewidth=2, markersize=8)
    ax1.axhline(y=0.5, color='red', linestyle='--', alpha=0.7, label='50% threshold')
    ax1.set_title('Win Rate vs Forward Horizon')
    ax1.set_xlabel('Forward Horizon (days)')
    ax1.set_ylabel('Win Rate')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, 1)
    
    # Plot mean returns
    colors = ['green' if r > 0 else 'red' for r in mean_returns]
    ax2.bar(horizons, mean_returns, color=colors, alpha=0.7)
    ax2.axhline(y=0, color='black', linestyle='-', alpha=0.5)
    ax2.set_title('Mean Forward Returns vs Horizon')
    ax2.set_xlabel('Forward Horizon (days)')
    ax2.set_ylabel('Mean Return')
    ax2.grid(True, alpha=0.3)
    
    # Add sample size annotations
    for i, (h, s) in enumerate(zip(horizons, sample_sizes)):
        ax2.annotate(f'N={s}', (h, mean_returns[i]), 
                    ha='center', va='bottom' if mean_returns[i] > 0 else 'top')
    
    plt.tight_layout()
    return fig


def plot_volatility_regime_analysis(streak_stats: Dict, figsize: Tuple[int, int] = (12, 8)) -> plt.Figure:
    """
    Plot volatility regime analysis.
    
    Args:
        streak_stats: Streak statistics from compute_streak_stats
        figsize: Figure size (width, height)
        
    Returns:
        Matplotlib figure object
    """
    if not streak_stats or 'statistics' not in streak_stats:
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(0.5, 0.5, 'No statistics data available', ha='center', va='center', transform=ax.transAxes)
        return fig
    
    statistics = streak_stats['statistics']
    if 'volatility_regime_stats' not in statistics:
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(0.5, 0.5, 'No volatility regime data available', ha='center', va='center', transform=ax.transAxes)
        return fig
    
    regime_stats = statistics['volatility_regime_stats']
    if not regime_stats:
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(0.5, 0.5, 'No regime data available', ha='center', va='center', transform=ax.transAxes)
        return fig
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=figsize)
    
    regimes = list(regime_stats.keys())
    counts = [regime_stats[r]['count'] for r in regimes]
    avg_lengths = [regime_stats[r]['avg_length'] for r in regimes]
    avg_returns = [regime_stats[r]['avg_return'] for r in regimes]
    volatilities = [regime_stats[r]['volatility'] for r in regimes]
    
    # Plot 1: Streak counts by regime
    bars1 = ax1.bar(regimes, counts, color=['lightblue', 'orange', 'lightcoral'])
    ax1.set_title('Streak Count by Volatility Regime')
    ax1.set_ylabel('Number of Streaks')
    ax1.grid(True, alpha=0.3)
    
    # Add count labels on bars
    for bar, count in zip(bars1, counts):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                str(count), ha='center', va='bottom')
    
    # Plot 2: Average streak length by regime
    bars2 = ax2.bar(regimes, avg_lengths, color=['lightgreen', 'gold', 'lightpink'])
    ax2.set_title('Average Streak Length by Regime')
    ax2.set_ylabel('Average Length (days)')
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Average returns by regime
    colors3 = ['green' if r > 0 else 'red' for r in avg_returns]
    bars3 = ax3.bar(regimes, avg_returns, color=colors3, alpha=0.7)
    ax3.axhline(y=0, color='black', linestyle='-', alpha=0.5)
    ax3.set_title('Average Returns by Regime')
    ax3.set_ylabel('Average Return')
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Volatility by regime
    bars4 = ax4.bar(regimes, volatilities, color=['lightsteelblue', 'wheat', 'mistyrose'])
    ax4.set_title('Volatility by Regime')
    ax4.set_ylabel('Volatility')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


def plot_intervaled_evaluation(eval_results: Dict, figsize: Tuple[int, int] = (15, 10)) -> plt.Figure:
    """
    Plot intervaled evaluation results.
    
    Args:
        eval_results: Results from evaluate_intervals
        figsize: Figure size (width, height)
        
    Returns:
        Matplotlib figure object
    """
    if not eval_results or 'intervals' not in eval_results:
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(0.5, 0.5, 'No interval data available', ha='center', va='center', transform=ax.transAxes)
        return fig
    
    intervals = eval_results['intervals']
    if not intervals:
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(0.5, 0.5, 'No intervals found', ha='center', va='center', transform=ax.transAxes)
        return fig
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=figsize)
    
    # Extract data
    interval_ids = [i['interval_id'] for i in intervals]
    streak_counts = [i['streak_count'] for i in intervals]
    data_points = [i['data_points'] for i in intervals]
    
    # Plot 1: Streak count by interval
    ax1.plot(interval_ids, streak_counts, 'bo-', linewidth=2, markersize=6)
    ax1.set_title('Streak Count by Interval')
    ax1.set_xlabel('Interval ID')
    ax1.set_ylabel('Number of Streaks')
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Data points by interval
    ax2.plot(interval_ids, data_points, 'go-', linewidth=2, markersize=6)
    ax2.set_title('Data Points by Interval')
    ax2.set_xlabel('Interval ID')
    ax2.set_ylabel('Number of Data Points')
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Win rates across intervals (if available)
    win_rates_by_horizon = {}
    for interval in intervals:
        momentum_metrics = interval.get('momentum_metrics', {})
        for key, metrics in momentum_metrics.items():
            if 'win_rate' in metrics:
                horizon = key
                if horizon not in win_rates_by_horizon:
                    win_rates_by_horizon[horizon] = []
                win_rates_by_horizon[horizon].append(metrics['win_rate'])
    
    if win_rates_by_horizon:
        for horizon, win_rates in win_rates_by_horizon.items():
            if len(win_rates) == len(interval_ids):
                ax3.plot(interval_ids, win_rates, 'o-', linewidth=2, markersize=6, label=horizon)
        
        ax3.axhline(y=0.5, color='red', linestyle='--', alpha=0.7, label='50% threshold')
        ax3.set_title('Win Rates by Interval')
        ax3.set_xlabel('Interval ID')
        ax3.set_ylabel('Win Rate')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        ax3.set_ylim(0, 1)
    else:
        ax3.text(0.5, 0.5, 'No win rate data available', ha='center', va='center', transform=ax3.transAxes)
    
    # Plot 4: Stability metrics (if available)
    aggregated = eval_results.get('aggregated', {})
    if 'stability_metrics' in aggregated:
        stability = aggregated['stability_metrics']
        metrics_names = list(stability.keys())
        metrics_values = list(stability.values())
        
        bars = ax4.bar(metrics_names, metrics_values, color=['lightblue', 'lightgreen', 'lightcoral'])
        ax4.set_title('Stability Metrics')
        ax4.set_ylabel('Value')
        ax4.grid(True, alpha=0.3)
        
        # Rotate x-axis labels for better readability
        ax4.tick_params(axis='x', rotation=45)
    else:
        ax4.text(0.5, 0.5, 'No stability metrics available', ha='center', va='center', transform=ax4.transAxes)
    
    plt.tight_layout()
    return fig


def create_summary_dashboard(streak_stats: Dict, suggestions: Dict, 
                           eval_results: Optional[Dict] = None,
                           figsize: Tuple[int, int] = (20, 15)) -> plt.Figure:
    """
    Create a comprehensive summary dashboard.
    
    Args:
        streak_stats: Streak statistics from compute_streak_stats
        suggestions: Signal suggestions from suggest_thresholds
        eval_results: Optional intervaled evaluation results
        figsize: Figure size (width, height)
        
    Returns:
        Matplotlib figure object
    """
    fig = plt.figure(figsize=figsize)
    
    # Create grid layout
    gs = fig.add_gridspec(3, 4, hspace=0.3, wspace=0.3)
    
    # Title
    fig.suptitle('Streak Analysis Summary Dashboard', fontsize=16, fontweight='bold')
    
    # Plot 1: Streak histogram (top left)
    ax1 = fig.add_subplot(gs[0, :2])
    if streak_stats and 'streaks' in streak_stats:
        streaks_df = streak_stats['streaks']
        if not streaks_df.empty:
            positive_streaks = streaks_df[streaks_df['direction'] > 0]
            negative_streaks = streaks_df[streaks_df['direction'] < 0]
            
            if len(positive_streaks) > 0:
                ax1.hist(positive_streaks['length'], bins=15, alpha=0.7, color='green', 
                        label=f'Positive (N={len(positive_streaks)})')
            if len(negative_streaks) > 0:
                ax1.hist(negative_streaks['length'], bins=15, alpha=0.7, color='red',
                        label=f'Negative (N={len(negative_streaks)})')
            
            ax1.set_title('Streak Length Distribution')
            ax1.set_xlabel('Streak Length (days)')
            ax1.set_ylabel('Frequency')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
        else:
            ax1.text(0.5, 0.5, 'No streaks found', ha='center', va='center', transform=ax1.transAxes)
    else:
        ax1.text(0.5, 0.5, 'No streak data available', ha='center', va='center', transform=ax1.transAxes)
    
    # Plot 2: Momentum analysis (top right)
    ax2 = fig.add_subplot(gs[0, 2:])
    if streak_stats and 'momentum_metrics' in streak_stats:
        momentum_metrics = streak_stats['momentum_metrics']
        if momentum_metrics:
            horizons = []
            win_rates = []
            for key, metrics in momentum_metrics.items():
                if 'win_rate' in metrics:
                    horizon = int(key.split('_')[1])
                    horizons.append(horizon)
                    win_rates.append(metrics['win_rate'])
            
            if horizons:
                sorted_data = sorted(zip(horizons, win_rates))
                horizons, win_rates = zip(*sorted_data)
                
                ax2.plot(horizons, win_rates, 'bo-', linewidth=2, markersize=8)
                ax2.axhline(y=0.5, color='red', linestyle='--', alpha=0.7, label='50% threshold')
                ax2.set_title('Win Rate vs Forward Horizon')
                ax2.set_xlabel('Forward Horizon (days)')
                ax2.set_ylabel('Win Rate')
                ax2.legend()
                ax2.grid(True, alpha=0.3)
                ax2.set_ylim(0, 1)
            else:
                ax2.text(0.5, 0.5, 'No win rate data available', ha='center', va='center', transform=ax2.transAxes)
        else:
            ax2.text(0.5, 0.5, 'No momentum metrics available', ha='center', va='center', transform=ax2.transAxes)
    else:
        ax2.text(0.5, 0.5, 'No momentum data available', ha='center', va='center', transform=ax2.transAxes)
    
    # Plot 3: Signal suggestions (middle left)
    ax3 = fig.add_subplot(gs[1, :2])
    if suggestions:
        buy_signals = suggestions.get('buy_thresholds', [])
        short_signals = suggestions.get('short_thresholds', [])
        
        if buy_signals or short_signals:
            # Create summary table
            table_data = []
            for signal in buy_signals[:3]:  # Top 3 buy signals
                table_data.append([
                    f"Buy after {signal['streak_length']} up days",
                    f"{signal['horizon']}d horizon",
                    f"{signal['win_rate']:.1%}",
                    f"{signal['sample_size']}"
                ])
            
            for signal in short_signals[:3]:  # Top 3 short signals
                table_data.append([
                    f"Short after {signal['streak_length']} down days",
                    f"{signal['horizon']}d horizon",
                    f"{signal['win_rate']:.1%}",
                    f"{signal['sample_size']}"
                ])
            
            if table_data:
                table = ax3.table(cellText=table_data,
                                colLabels=['Signal', 'Horizon', 'Win Rate', 'N'],
                                cellLoc='center',
                                loc='center')
                table.auto_set_font_size(False)
                table.set_fontsize(9)
                table.scale(1, 2)
                ax3.set_title('Top Signal Suggestions')
                ax3.axis('off')
            else:
                ax3.text(0.5, 0.5, 'No signal suggestions available', ha='center', va='center', transform=ax3.transAxes)
        else:
            ax3.text(0.5, 0.5, 'No signal suggestions available', ha='center', va='center', transform=ax3.transAxes)
    else:
        ax3.text(0.5, 0.5, 'No suggestions data available', ha='center', va='center', transform=ax3.transAxes)
    
    # Plot 4: Volatility regime analysis (middle right)
    ax4 = fig.add_subplot(gs[1, 2:])
    if streak_stats and 'statistics' in streak_stats:
        statistics = streak_stats['statistics']
        if 'volatility_regime_stats' in statistics:
            regime_stats = statistics['volatility_regime_stats']
            if regime_stats:
                regimes = list(regime_stats.keys())
                counts = [regime_stats[r]['count'] for r in regimes]
                
                bars = ax4.bar(regimes, counts, color=['lightblue', 'orange', 'lightcoral'])
                ax4.set_title('Streak Count by Volatility Regime')
                ax4.set_ylabel('Number of Streaks')
                ax4.grid(True, alpha=0.3)
                
                # Add count labels on bars
                for bar, count in zip(bars, counts):
                    ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                            str(count), ha='center', va='bottom')
            else:
                ax4.text(0.5, 0.5, 'No regime data available', ha='center', va='center', transform=ax4.transAxes)
        else:
            ax4.text(0.5, 0.5, 'No volatility regime data available', ha='center', va='center', transform=ax4.transAxes)
    else:
        ax4.text(0.5, 0.5, 'No statistics data available', ha='center', va='center', transform=ax4.transAxes)
    
    # Plot 5: Intervaled evaluation (bottom row)
    ax5 = fig.add_subplot(gs[2, :])
    if eval_results and 'intervals' in eval_results:
        intervals = eval_results['intervals']
        if intervals:
            interval_ids = [i['interval_id'] for i in intervals]
            streak_counts = [i['streak_count'] for i in intervals]
            
            ax5.plot(interval_ids, streak_counts, 'mo-', linewidth=2, markersize=6)
            ax5.set_title('Streak Count by Interval')
            ax5.set_xlabel('Interval ID')
            ax5.set_ylabel('Number of Streaks')
            ax5.grid(True, alpha=0.3)
            
            # Add aggregated stats if available
            aggregated = eval_results.get('aggregated', {})
            if 'streak_count_stats' in aggregated:
                stats = aggregated['streak_count_stats']
                mean_count = stats.get('mean', 0)
                ax5.axhline(y=mean_count, color='red', linestyle='--', alpha=0.7, 
                           label=f'Mean: {mean_count:.1f}')
                ax5.legend()
        else:
            ax5.text(0.5, 0.5, 'No intervals found', ha='center', va='center', transform=ax5.transAxes)
    else:
        ax5.text(0.5, 0.5, 'No interval data available', ha='center', va='center', transform=ax5.transAxes)
    
    return fig
