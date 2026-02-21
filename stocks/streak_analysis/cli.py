"""
Command-line interface for streak analysis.
"""

import asyncio
import typer
from typing import Optional, List
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from .config import StreakConfig, EXAMPLE_CONFIGS
from .data_provider import DbServerProvider, MockDataProvider
from .preprocess import prepare_data, validate_data_coverage
from .streaks import compute_streak_stats
from .signals import suggest_thresholds, generate_signal_summary
from .evaluation import evaluate_intervals
from .viz import (
    plot_streak_histogram, 
    plot_forward_returns, 
    plot_momentum_analysis,
    plot_volatility_regime_analysis,
    plot_intervaled_evaluation,
    create_summary_dashboard
)
from .terminal_render import render_comprehensive_report

app = typer.Typer(help="Streak Analysis System - Analyze stock price streaks using db_server.py")


@app.command()
def analyze(
    symbol: str = typer.Argument(..., help="Stock symbol to analyze"),
    timeframe: str = typer.Option("daily", "--timeframe", "-t", 
                                 help="Data timeframe: realtime, hourly, daily"),
    lookback_days: int = typer.Option(90, "--lookback-days", "-l", 
                                     help="Number of days to look back"),
    min_streak_threshold: int = typer.Option(0, "--min-streak-threshold", "-m",
                                            help="Minimum streak length to count"),
    aggregation_level: str = typer.Option("day", "--aggregation-level", "-a",
                                         help="Evaluation granularity: day, week, month"),
    evaluation_mode: str = typer.Option("close_to_close", "--evaluation-mode", "-e",
                                       help="Return calculation method"),
    db_host: str = typer.Option("localhost", "--db-host", help="Database server host"),
    db_port: int = typer.Option(9002, "--db-port", help="Database server port"),
    ascii_only: bool = typer.Option(False, "--ascii", help="Force ASCII-only output"),
    no_plots: bool = typer.Option(False, "--no-plots", help="Skip generating plots"),
    export_csv: Optional[str] = typer.Option(None, "--export-csv", help="Path to export CSV files"),
    mock_data: bool = typer.Option(False, "--mock", help="Use mock data for testing"),
    random_seed: Optional[int] = typer.Option(42, "--seed", help="Random seed for reproducibility")
):
    """
    Analyze streaks for a given stock symbol.
    """
    # Set random seed
    if random_seed is not None:
        np.random.seed(random_seed)
        typer.echo(f"Set random seed to {random_seed}")
    
    # Create configuration
    config = StreakConfig(
        symbol=symbol,
        timeframe=timeframe,
        lookback_days=lookback_days,
        min_streak_threshold=min_streak_threshold,
        aggregation_level=aggregation_level,
        evaluation_mode=evaluation_mode,
        db_host=db_host,
        db_port=db_port,
        ascii_only=ascii_only,
        no_plots=no_plots,
        export_csv=export_csv,
        random_seed=random_seed
    )
    
    # Run analysis
    asyncio.run(run_analysis(config))


@app.command()
def config_examples():
    """Show example configurations."""
    typer.echo("Example Configurations:")
    typer.echo("=" * 50)
    
    for name, config in EXAMPLE_CONFIGS.items():
        typer.echo(f"\n{name.upper()}:")
        for key, value in config.items():
            typer.echo(f"  {key}: {value}")
    
    typer.echo("\nTo use these examples, create a config file:")
    typer.echo("  python -m streak_analysis.cli analyze TQQQ --timeframe daily --lookback-days 365")


@app.command()
def test_connection(
    db_host: str = typer.Option("localhost", "--db-host", help="Database server host"),
    db_port: int = typer.Option(9002, "--db-port", help="Database server port")
):
    """Test connection to db_server.py."""
    asyncio.run(test_db_connection(db_host, db_port))


async def test_db_connection(host: str, port: int):
    """Test database connection."""
    typer.echo(f"Testing connection to {host}:{port}...")
    
    try:
        async with DbServerProvider(host=host, port=port) as provider:
            is_connected = await provider.test_connection()
            if is_connected:
                typer.echo("✅ Connection successful!")
            else:
                typer.echo("❌ Connection failed!")
    except Exception as e:
        typer.echo(f"❌ Connection error: {e}")
        typer.echo("\nMake sure db_server.py is running on the specified port.")


async def run_analysis(config: StreakConfig):
    """Run the complete streak analysis."""
    typer.echo(f"🔍 Analyzing {config.symbol} using {config.timeframe} data...")
    
    # Initialize data provider
    if config.mock_data:
        provider = MockDataProvider()
        typer.echo("🧪 Using mock data for testing")
    else:
        provider = DbServerProvider(host=config.db_host, port=config.db_port)
        typer.echo(f"📡 Connecting to {config.db_host}:{config.db_port}")
    
    try:
        # Fetch data
        typer.echo("📊 Fetching data...")
        if config.timeframe == "daily":
            df = await provider.get_daily(config.symbol, config.lookback_days)
        elif config.timeframe == "hourly":
            df = await provider.get_hourly(config.symbol, config.lookback_days)
        elif config.timeframe == "realtime":
            df = await provider.get_realtime_window(config.symbol, config.realtime_window_days)
        else:
            raise ValueError(f"Unknown timeframe: {config.timeframe}")
        
        if df.empty:
            typer.echo(f"❌ No data found for {config.symbol}")
            return
        
        typer.echo(f"✅ Fetched {len(df)} data points")
        
        # Validate data
        validation = validate_data_coverage(df)
        if not validation['is_valid']:
            typer.echo("⚠️  Data validation warnings:")
            for issue in validation['quality_issues']:
                typer.echo(f"  - {issue}")
        
        # Prepare data
        typer.echo("🔧 Preparing data...")
        df_prepared = prepare_data(
            df, 
            aggregation_level=config.aggregation_level,
            evaluation_mode=config.evaluation_mode
        )
        
        if df_prepared.empty:
            typer.echo("❌ No data after preparation")
            return
        
        typer.echo(f"✅ Prepared {len(df_prepared)} data points")
        
        # Compute streak statistics
        typer.echo("📈 Computing streak statistics...")
        streak_stats = compute_streak_stats(
            df_prepared, 
            min_streak_threshold=config.min_streak_threshold
        )
        
        if not streak_stats or 'streaks' not in streak_stats:
            typer.echo("❌ No streaks found")
            return
        
        typer.echo(f"✅ Found {len(streak_stats['streaks'])} streaks")
        
        # Generate signal suggestions
        typer.echo("💡 Generating signal suggestions...")
        suggestions = suggest_thresholds(streak_stats)
        
        # Run intervaled evaluation
        typer.echo("🔄 Running intervaled evaluation...")
        eval_results = evaluate_intervals(
            df_prepared,
            n_days=config.evaluation_intervals.n_days,
            m_days=config.evaluation_intervals.m_days,
            min_streak_threshold=config.min_streak_threshold,
            evaluation_mode=config.evaluation_mode,
            aggregation_level=config.aggregation_level
        )
        
        # Terminal output
        if config.ascii_only or not config.no_plots:
            typer.echo("\n" + "="*60)
            typer.echo("📊 STREAK ANALYSIS RESULTS")
            typer.echo("="*60)
            
            render_comprehensive_report(streak_stats, suggestions, eval_results)
        
        # Generate plots (if not disabled)
        if not config.no_plots:
            typer.echo("\n🎨 Generating plots...")
            
            # Create plots directory
            plots_dir = Path("plots")
            plots_dir.mkdir(exist_ok=True)
            
            # Generate individual plots
            fig1 = plot_streak_histogram(streak_stats)
            fig1.savefig(plots_dir / f"{config.symbol}_streak_histogram.png", dpi=300, bbox_inches='tight')
            
            fig2 = plot_momentum_analysis(streak_stats)
            fig2.savefig(plots_dir / f"{config.symbol}_momentum_analysis.png", dpi=300, bbox_inches='tight')
            
            if 'volatility_regime_stats' in streak_stats.get('statistics', {}):
                fig3 = plot_volatility_regime_analysis(streak_stats)
                fig3.savefig(plots_dir / f"{config.symbol}_volatility_regimes.png", dpi=300, bbox_inches='tight')
            
            if eval_results:
                fig4 = plot_intervaled_evaluation(eval_results)
                fig4.savefig(plots_dir / f"{config.symbol}_intervaled_evaluation.png", dpi=300, bbox_inches='tight')
            
            # Create summary dashboard
            fig5 = create_summary_dashboard(streak_stats, suggestions, eval_results)
            fig5.savefig(plots_dir / f"{config.symbol}_summary_dashboard.png", dpi=300, bbox_inches='tight')
            
            typer.echo(f"✅ Plots saved to {plots_dir}/")
        
        # Export CSV (if requested)
        if config.export_csv:
            typer.echo(f"💾 Exporting CSV files to {config.export_csv}...")
            export_path = Path(config.export_csv)
            export_path.mkdir(exist_ok=True)
            
            # Export streaks data
            if 'streaks' in streak_stats and not streak_stats['streaks'].empty:
                streak_stats['streaks'].to_csv(export_path / f"{config.symbol}_streaks.csv")
            
            # Export suggestions
            if suggestions:
                suggestions_df = pd.DataFrame(suggestions.get('buy_thresholds', []) + 
                                           suggestions.get('short_thresholds', []))
                if not suggestions_df.empty:
                    suggestions_df.to_csv(export_path / f"{config.symbol}_signals.csv", index=False)
            
            # Export intervaled evaluation results
            if eval_results:
                intervals_df = pd.DataFrame(eval_results['intervals'])
                intervals_df.to_csv(export_path / f"{config.symbol}_intervals.csv", index=False)
            
            typer.echo(f"✅ CSV files exported to {export_path}/")
        
        typer.echo("\n🎉 Analysis complete!")
        
    except Exception as e:
        typer.echo(f"❌ Error during analysis: {e}")
        raise
    finally:
        if hasattr(provider, 'session') and provider.session:
            await provider.session.close()


@app.command()
def batch_analyze(
    symbols: List[str] = typer.Argument(..., help="List of stock symbols to analyze"),
    config_file: Optional[Path] = typer.Option(None, "--config", "-c", 
                                              help="Configuration file (YAML/JSON)"),
    **kwargs
):
    """Analyze multiple symbols in batch."""
    if config_file and config_file.exists():
        if config_file.suffix.lower() == '.yaml':
            config = StreakConfig.from_yaml(config_file)
        elif config_file.suffix.lower() == '.json':
            config = StreakConfig.from_json(config_file)
        else:
            typer.echo("❌ Config file must be YAML or JSON")
            return
    else:
        # Use default config with overrides
        config = StreakConfig(**kwargs)
    
    typer.echo(f"🔄 Batch analyzing {len(symbols)} symbols...")
    
    for symbol in symbols:
        typer.echo(f"\n{'='*50}")
        typer.echo(f"📊 Analyzing {symbol}")
        typer.echo(f"{'='*50}")
        
        # Update config for this symbol
        config.symbol = symbol
        
        try:
            asyncio.run(run_analysis(config))
        except Exception as e:
            typer.echo(f"❌ Error analyzing {symbol}: {e}")
            continue
    
    typer.echo(f"\n🎉 Batch analysis complete for {len(symbols)} symbols!")


if __name__ == "__main__":
    app()
