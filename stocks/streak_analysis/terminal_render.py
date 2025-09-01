"""
Terminal rendering module for ASCII charts and tables.
"""

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.text import Text
from rich.layout import Layout
from rich.columns import Columns
from rich import box
from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np

console = Console()


def render_streak_summary(streak_stats: Dict) -> None:
    """
    Render streak analysis summary in terminal.
    
    Args:
        streak_stats: Streak statistics from compute_streak_stats
    """
    if not streak_stats or 'streaks' not in streak_stats:
        console.print("[red]No streak data available[/red]")
        return
    
    streaks_df = streak_stats['streaks']
    if streaks_df.empty:
        console.print("[yellow]No streaks found[/yellow]")
        return
    
    # Create summary table
    table = Table(title="Streak Analysis Summary", box=box.ROUNDED)
    table.add_column("Metric", style="cyan", no_wrap=True)
    table.add_column("Value", style="magenta")
    
    # Overall statistics
    statistics = streak_stats.get('statistics', {})
    table.add_row("Total Streaks", str(statistics.get('total_streaks', 0)))
    table.add_row("Positive Streaks", str(statistics.get('positive_streaks', 0)))
    table.add_row("Negative Streaks", str(statistics.get('negative_streaks', 0)))
    
    if 'avg_length' in statistics:
        table.add_row("Average Length", f"{statistics['avg_length']:.1f} days")
    if 'max_length' in statistics:
        table.add_row("Maximum Length", f"{statistics['max_length']} days")
    
    console.print(table)
    
    # Data summary
    data_summary = streak_stats.get('data_summary', {})
    if data_summary:
        console.print(f"\n[bold]Data Coverage:[/bold] {data_summary.get('total_rows', 0)} rows")
        if 'date_range' in data_summary:
            date_range = data_summary['date_range']
            console.print(f"Date Range: {date_range['start']} to {date_range['end']}")


def render_streak_histogram(streak_stats: Dict, max_width: int = 80) -> None:
    """
    Render ASCII histogram of streak lengths.
    
    Args:
        streak_stats: Streak statistics from compute_streak_stats
        max_width: Maximum width for the histogram
    """
    if not streak_stats or 'streaks' not in streak_stats:
        console.print("[red]No streak data available for histogram[/red]")
        return
    
    streaks_df = streak_stats['streaks']
    if streaks_df.empty:
        console.print("[yellow]No streaks found for histogram[/yellow]")
        return
    
    # Separate positive and negative streaks
    positive_streaks = streaks_df[streaks_df['direction'] > 0]
    negative_streaks = streaks_df[streaks_df['direction'] < 0]
    
    console.print("\n[bold green]Positive Streak Length Distribution[/bold green]")
    if len(positive_streaks) > 0:
        render_histogram(positive_streaks['length'], max_width, color="green")
    else:
        console.print("[yellow]No positive streaks found[/yellow]")
    
    console.print("\n[bold red]Negative Streak Length Distribution[/bold red]")
    if len(negative_streaks) > 0:
        render_histogram(negative_streaks['length'], max_width, color="red")
    else:
        console.print("[yellow]No negative streaks found[/yellow]")


def render_histogram(data: pd.Series, max_width: int, color: str = "white") -> None:
    """
    Render ASCII histogram for a series of data.
    
    Args:
        data: Series of numeric data
        max_width: Maximum width for the histogram
        color: Color for the histogram bars
    """
    if data.empty:
        return
    
    # Calculate histogram
    bins = min(20, len(data.unique()))
    hist, bin_edges = np.histogram(data, bins=bins)
    
    # Normalize to fit max_width
    max_count = hist.max() if len(hist) > 0 else 1
    scale_factor = (max_width - 20) / max_count if max_count > 0 else 1
    
    # Print histogram
    for i, count in enumerate(hist):
        if count > 0:
            bar_length = int(count * scale_factor)
            bar = "█" * bar_length
            bin_start = bin_edges[i]
            bin_end = bin_edges[i + 1]
            
            console.print(f"{bin_start:6.1f}-{bin_end:6.1f}: {bar} {count}")


def render_momentum_metrics(streak_stats: Dict) -> None:
    """
    Render momentum metrics in terminal.
    
    Args:
        streak_stats: Streak statistics from compute_streak_stats
    """
    if not streak_stats or 'momentum_metrics' not in streak_stats:
        console.print("[red]No momentum metrics available[/red]")
        return
    
    momentum_metrics = streak_stats['momentum_metrics']
    if not momentum_metrics:
        console.print("[yellow]No momentum metrics found[/yellow]")
        return
    
    # Create momentum table
    table = Table(title="Momentum Analysis", box=box.ROUNDED)
    table.add_column("Horizon", style="cyan", no_wrap=True)
    table.add_column("Win Rate", style="green")
    table.add_column("Mean Return", style="yellow")
    table.add_column("Sample Size", style="blue")
    
    for key, metrics in momentum_metrics.items():
        if 'win_rate' in metrics:
            horizon = key.replace('horizon_', '') + ' days'
            win_rate = f"{metrics['win_rate']:.1%}"
            mean_return = f"{metrics.get('mean_return', 0):.2%}"
            sample_size = str(metrics.get('sample_size', 0))
            
            # Color code win rate
            if metrics['win_rate'] > 0.6:
                win_rate = f"[green]{win_rate}[/green]"
            elif metrics['win_rate'] < 0.4:
                win_rate = f"[red]{win_rate}[/red]"
            else:
                win_rate = f"[yellow]{win_rate}[/yellow]"
            
            table.add_row(horizon, win_rate, mean_return, sample_size)
    
    console.print(table)


def render_signal_suggestions(suggestions: Dict) -> None:
    """
    Render signal suggestions in terminal.
    
    Args:
        suggestions: Signal suggestions from suggest_thresholds
    """
    if not suggestions:
        console.print("[red]No signal suggestions available[/red]")
        return
    
    # Buy signals
    buy_signals = suggestions.get('buy_thresholds', [])
    if buy_signals:
        console.print("\n[bold green]Buy Signal Suggestions[/bold green]")
        table = Table(box=box.ROUNDED)
        table.add_column("Streak Length", style="cyan")
        table.add_column("Horizon", style="blue")
        table.add_column("Win Rate", style="green")
        table.add_column("Mean Return", style="yellow")
        table.add_column("Sample Size", style="white")
        
        for signal in buy_signals[:5]:  # Top 5
            table.add_row(
                str(signal['streak_length']),
                f"{signal['horizon']}d",
                f"{signal['win_rate']:.1%}",
                f"{signal['mean_return']:.2%}",
                str(signal['sample_size'])
            )
        
        console.print(table)
    
    # Short signals
    short_signals = suggestions.get('short_thresholds', [])
    if short_signals:
        console.print("\n[bold red]Short Signal Suggestions[/bold red]")
        table = Table(box=box.ROUNDED)
        table.add_column("Streak Length", style="cyan")
        table.add_column("Horizon", style="blue")
        table.add_column("Win Rate", style="green")
        table.add_column("Mean Return", style="yellow")
        table.add_column("Sample Size", style="white")
        
        for signal in short_signals[:5]:  # Top 5
            table.add_row(
                str(signal['streak_length']),
                f"{signal['horizon']}d",
                f"{signal['win_rate']:.1%}",
                f"{signal['mean_return']:.2%}",
                str(signal['sample_size'])
            )
        
        console.print(table)
    
    # Confidence metrics
    confidence_metrics = suggestions.get('confidence_metrics', {})
    if confidence_metrics:
        console.print(f"\n[bold]Confidence Metrics:[/bold]")
        console.print(f"Total streaks analyzed: {confidence_metrics.get('total_streaks_analyzed', 0)}")
        console.print(f"Confidence level: {confidence_metrics.get('confidence_level', 0.95):.1%}")


def render_volatility_regime_analysis(streak_stats: Dict) -> None:
    """
    Render volatility regime analysis in terminal.
    
    Args:
        streak_stats: Streak statistics from compute_streak_stats
    """
    if not streak_stats or 'statistics' not in streak_stats:
        console.print("[red]No volatility regime data available[/red]")
        return
    
    statistics = streak_stats['statistics']
    if 'volatility_regime_stats' not in statistics:
        console.print("[yellow]No volatility regime statistics found[/yellow]")
        return
    
    regime_stats = statistics['volatility_regime_stats']
    if not regime_stats:
        console.print("[yellow]No regime data available[/yellow]")
        return
    
    console.print("\n[bold]Volatility Regime Analysis[/bold]")
    table = Table(box=box.ROUNDED)
    table.add_column("Regime", style="cyan")
    table.add_column("Streak Count", style="blue")
    table.add_column("Avg Length", style="green")
    table.add_column("Avg Return", style="yellow")
    table.add_column("Volatility", style="red")
    
    for regime, stats in regime_stats.items():
        table.add_row(
            regime.capitalize(),
            str(stats['count']),
            f"{stats['avg_length']:.1f}",
            f"{stats['avg_return']:.2%}",
            f"{stats['volatility']:.2%}"
        )
    
    console.print(table)


def render_intervaled_evaluation(eval_results: Dict) -> None:
    """
    Render intervaled evaluation results in terminal.
    
    Args:
        eval_results: Results from evaluate_intervals
    """
    if not eval_results or 'intervals' not in eval_results:
        console.print("[red]No intervaled evaluation data available[/red]")
        return
    
    intervals = eval_results['intervals']
    if not intervals:
        console.print("[yellow]No intervals found[/yellow]")
        return
    
    # Summary table
    aggregated = eval_results.get('aggregated', {})
    if aggregated:
        console.print("\n[bold]Intervaled Evaluation Summary[/bold]")
        table = Table(box=box.ROUNDED)
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="magenta")
        
        table.add_row("Total Intervals", str(aggregated.get('total_intervals', 0)))
        table.add_row("Total Data Points", str(aggregated.get('total_data_points', 0)))
        table.add_row("Total Streaks", str(aggregated.get('total_streaks', 0)))
        
        if 'streak_count_stats' in aggregated:
            stats = aggregated['streak_count_stats']
            table.add_row("Mean Streaks/Interval", f"{stats.get('mean', 0):.1f}")
            table.add_row("Std Streaks/Interval", f"{stats.get('std', 0):.1f}")
        
        console.print(table)
    
    # Individual interval results
    console.print(f"\n[bold]Individual Interval Results[/bold]")
    table = Table(box=box.ROUNDED)
    table.add_column("Interval", style="cyan")
    table.add_column("Date Range", style="blue")
    table.add_column("Data Points", style="green")
    table.add_column("Streaks", style="yellow")
    
    for interval in intervals[:10]:  # Show first 10 intervals
        start_date = interval['start_date'].strftime('%Y-%m-%d')
        end_date = interval['end_date'].strftime('%Y-%m-%d')
        date_range = f"{start_date} to {end_date}"
        
        table.add_row(
            str(interval['interval_id']),
            date_range,
            str(interval['data_points']),
            str(interval['streak_count'])
        )
    
    if len(intervals) > 10:
        table.add_row("...", "...", "...", "...")
    
    console.print(table)


def render_comprehensive_report(streak_stats: Dict, suggestions: Dict, 
                              eval_results: Optional[Dict] = None) -> None:
    """
    Render a comprehensive report in terminal.
    
    Args:
        streak_stats: Streak statistics from compute_streak_stats
        suggestions: Signal suggestions from suggest_thresholds
        eval_results: Optional intervaled evaluation results
    """
    console.print(Panel.fit("[bold blue]Streak Analysis Report[/bold blue]", box=box.DOUBLE))
    
    # Basic summary
    render_streak_summary(streak_stats)
    
    # Streak histogram
    render_streak_histogram(streak_stats)
    
    # Momentum metrics
    render_momentum_metrics(streak_stats)
    
    # Signal suggestions
    render_signal_suggestions(suggestions)
    
    # Volatility regime analysis
    render_volatility_regime_analysis(streak_stats)
    
    # Intervaled evaluation (if available)
    if eval_results:
        render_intervaled_evaluation(eval_results)
    
    console.print("\n[bold green]Report Complete![/bold green]")


def render_ascii_chart(data: List[Tuple[str, float]], title: str, 
                       max_width: int = 60, color: str = "white") -> None:
    """
    Render a simple ASCII bar chart.
    
    Args:
        data: List of (label, value) tuples
        title: Chart title
        max_width: Maximum width for the chart
        color: Color for the bars
    """
    if not data:
        return
    
    console.print(f"\n[bold]{title}[/bold]")
    
    # Find maximum value for scaling
    max_value = max(value for _, value in data) if data else 1
    scale_factor = (max_width - 20) / max_value if max_value > 0 else 1
    
    for label, value in data:
        bar_length = int(value * scale_factor)
        bar = "█" * bar_length
        console.print(f"{label:15s}: {bar} {value:.2f}")
