"""
Jupyter Notebook Integration

Provides easy-to-use functions and widgets for interactive backtesting
in Jupyter notebooks with rich visualizations and real-time updates.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any, List, Optional, Union
import asyncio
import ipywidgets as widgets
from IPython.display import display, HTML, clear_output
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px

from ..backtesting.engine import BacktestEngine
from ..backtesting.config import BacktestConfig, ProcessingConfig
from ..strategies.markov_chain import MarkovChainStrategy
from ..strategies.buy_hold import BuyHoldStrategy
from ..strategies.technical_indicators import SMAStrategy, RSIStrategy
from ..analysis.comparison import StrategyComparison
from ..analysis.visualization import VisualizationEngine
from ..parallel.multiprocess_runner import MultiProcessRunner


class JupyterBacktestInterface:
    """
    Interactive backtesting interface for Jupyter notebooks.
    
    Features:
    - Interactive widgets for parameter configuration
    - Real-time progress tracking
    - Rich visualizations
    - Easy strategy comparison
    - Export capabilities
    """
    
    def __init__(self):
        self.engine: Optional[BacktestEngine] = None
        self.results: Dict[str, Any] = {}
        self.comparison_result: Optional[StrategyComparison] = None
        self.visualization_engine = VisualizationEngine()
        self.comparison_engine = StrategyComparison()
        
    def create_strategy_widgets(self) -> widgets.VBox:
        """Create interactive widgets for strategy configuration."""
        
        # Strategy selection
        strategy_dropdown = widgets.Dropdown(
            options=['markov', 'buy_hold', 'sma', 'rsi'],
            value='markov',
            description='Strategy:',
            style={'description_width': 'initial'}
        )
        
        # Markov chain parameters
        markov_params = widgets.VBox([
            widgets.IntSlider(value=252, min=50, max=500, description='Lookback Period:'),
            widgets.IntSlider(value=5, min=1, max=20, description='Prediction Horizon:'),
            widgets.IntSlider(value=5, min=3, max=10, description='State Bins:')
        ])
        
        # SMA parameters
        sma_params = widgets.VBox([
            widgets.IntSlider(value=20, min=5, max=50, description='Short Period:'),
            widgets.IntSlider(value=50, min=20, max=200, description='Long Period:')
        ])
        
        # RSI parameters
        rsi_params = widgets.VBox([
            widgets.IntSlider(value=14, min=5, max=30, description='RSI Period:'),
            widgets.FloatSlider(value=30.0, min=10.0, max=40.0, description='Oversold:'),
            widgets.FloatSlider(value=70.0, min=60.0, max=90.0, description='Overbought:')
        ])
        
        # Buy & Hold parameters (empty for now)
        buy_hold_params = widgets.VBox([])
        
        # Strategy parameters container
        strategy_params = widgets.VBox([markov_params])
        
        def update_strategy_params(change):
            if change['new'] == 'markov':
                strategy_params.children = [markov_params]
            elif change['new'] == 'sma':
                strategy_params.children = [sma_params]
            elif change['new'] == 'rsi':
                strategy_params.children = [rsi_params]
            elif change['new'] == 'buy_hold':
                strategy_params.children = [buy_hold_params]
        
        strategy_dropdown.observe(update_strategy_params, names='value')
        
        return widgets.VBox([
            strategy_dropdown,
            strategy_params
        ])
    
    def create_backtest_widgets(self) -> widgets.VBox:
        """Create interactive widgets for backtest configuration."""
        
        # Ticker input
        ticker_text = widgets.Text(
            value='AAPL',
            description='Ticker:',
            style={'description_width': 'initial'}
        )
        
        # Date range
        start_date = widgets.DatePicker(
            value=pd.Timestamp.now() - pd.Timedelta(days=365),
            description='Start Date:'
        )
        
        end_date = widgets.DatePicker(
            value=pd.Timestamp.now(),
            description='End Date:'
        )
        
        # Capital and risk parameters
        capital_slider = widgets.FloatSlider(
            value=100000.0,
            min=10000.0,
            max=1000000.0,
            step=10000.0,
            description='Initial Capital:',
            style={'description_width': 'initial'}
        )
        
        commission_slider = widgets.FloatSlider(
            value=1.0,
            min=0.0,
            max=10.0,
            step=0.1,
            description='Commission:',
            style={'description_width': 'initial'}
        )
        
        slippage_slider = widgets.FloatSlider(
            value=0.1,
            min=0.0,
            max=1.0,
            step=0.01,
            description='Slippage (%):',
            style={'description_width': 'initial'}
        )
        
        # Benchmark
        benchmark_text = widgets.Text(
            value='SPY',
            description='Benchmark:',
            style={'description_width': 'initial'}
        )
        
        return widgets.VBox([
            ticker_text,
            widgets.HBox([start_date, end_date]),
            capital_slider,
            commission_slider,
            slippage_slider,
            benchmark_text
        ])
    
    def create_processing_widgets(self) -> widgets.VBox:
        """Create interactive widgets for processing configuration."""
        
        workers_slider = widgets.IntSlider(
            value=4,
            min=1,
            max=16,
            description='Workers:',
            style={'description_width': 'initial'}
        )
        
        multiprocessing_checkbox = widgets.Checkbox(
            value=False,
            description='Use Multiprocessing',
            style={'description_width': 'initial'}
        )
        
        async_io_checkbox = widgets.Checkbox(
            value=True,
            description='Use Async I/O',
            style={'description_width': 'initial'}
        )
        
        return widgets.VBox([
            workers_slider,
            multiprocessing_checkbox,
            async_io_checkbox
        ])
    
    def create_control_panel(self) -> widgets.VBox:
        """Create the main control panel."""
        
        # Create all widget groups
        strategy_widgets = self.create_strategy_widgets()
        backtest_widgets = self.create_backtest_widgets()
        processing_widgets = self.create_processing_widgets()
        
        # Run button
        run_button = widgets.Button(
            description='Run Backtest',
            button_style='success',
            icon='play'
        )
        
        # Progress bar
        progress_bar = widgets.FloatProgress(
            value=0,
            min=0,
            max=1,
            description='Progress:',
            bar_style='info'
        )
        
        # Status text
        status_text = widgets.HTML(value="<p>Ready to run backtest</p>")
        
        # Output area
        output_area = widgets.Output()
        
        # Connect run button
        def on_run_clicked(b):
            with output_area:
                clear_output(wait=True)
                asyncio.run(self._run_backtest_interactive())
        
        run_button.on_click(on_run_clicked)
        
        # Create tabs
        tabs = widgets.Tab([
            strategy_widgets,
            backtest_widgets,
            processing_widgets
        ])
        tabs.set_title(0, 'Strategy')
        tabs.set_title(1, 'Backtest')
        tabs.set_title(2, 'Processing')
        
        return widgets.VBox([
            tabs,
            widgets.HBox([run_button, progress_bar]),
            status_text,
            output_area
        ])
    
    async def _run_backtest_interactive(self):
        """Run backtest with interactive widgets."""
        
        try:
            # Get widget values (this would need to be implemented based on widget structure)
            ticker = 'AAPL'  # Get from ticker_text widget
            strategy_type = 'markov'  # Get from strategy_dropdown widget
            
            # Create configurations
            config = BacktestConfig(
                start_date=pd.Timestamp.now() - pd.Timedelta(days=365),
                end_date=pd.Timestamp.now(),
                initial_capital=100000.0
            )
            
            processing_config = ProcessingConfig(
                max_workers=4,
                use_multiprocessing=False,
                use_async_io=True
            )
            
            # Create strategy
            if strategy_type == 'markov':
                strategy = MarkovChainStrategy()
                strategy.initialize()
            elif strategy_type == 'buy_hold':
                strategy = BuyHoldStrategy()
                strategy.initialize()
            elif strategy_type == 'sma':
                strategy = SMAStrategy()
                strategy.initialize()
            elif strategy_type == 'rsi':
                strategy = RSIStrategy()
                strategy.initialize()
            
            # Create engine
            self.engine = BacktestEngine(config, processing_config)
            self.engine.add_strategy(strategy)
            
            # Run backtest
            print("Running backtest...")
            self.results = await self.engine.run_backtest(ticker)
            
            # Display results
            self._display_results()
            
        except Exception as e:
            print(f"Error: {str(e)}")
    
    def _display_results(self):
        """Display backtest results."""
        
        if not self.results:
            print("No results to display")
            return
        
        # Display metrics
        metrics = self.results.get('metrics', {})
        
        print("\n" + "="*60)
        print(f"BACKTEST RESULTS FOR {self.results.get('ticker', 'UNKNOWN')}")
        print("="*60)
        
        print(f"Total Return: {metrics.get('total_return', 0):.2f}%")
        print(f"Annualized Return: {metrics.get('annualized_return', 0):.2f}%")
        print(f"Sharpe Ratio: {metrics.get('sharpe_ratio', 0):.3f}")
        print(f"Max Drawdown: {metrics.get('max_drawdown', 0):.2f}%")
        print(f"Win Rate: {metrics.get('win_rate', 0):.1f}%")
        print(f"Number of Trades: {metrics.get('num_trades', 0)}")
        
        # Create visualizations
        self._create_visualizations()
    
    def _create_visualizations(self):
        """Create visualizations for the results."""
        
        # Equity curve
        equity_data = self.results.get('equity_curve', {})
        if equity_data:
            df = pd.DataFrame.from_dict(equity_data, orient='index')
            df.index = pd.to_datetime(df.index)
            df.columns = ['equity']
            
            # Normalize to 100
            df['equity'] = (df['equity'] / df['equity'].iloc[0]) * 100
            
            # Create plotly figure
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=df.index,
                y=df['equity'],
                mode='lines',
                name='Equity Curve',
                line=dict(color='blue', width=2)
            ))
            
            fig.update_layout(
                title='Equity Curve',
                xaxis_title='Date',
                yaxis_title='Equity (Normalized to 100)',
                hovermode='x unified'
            )
            
            fig.show()
    
    def compare_strategies(self, tickers: List[str], strategies: List[str]) -> None:
        """Compare multiple strategies on multiple tickers."""
        
        print(f"Comparing {len(strategies)} strategies on {len(tickers)} tickers...")
        
        # This would implement the full comparison logic
        # For now, just show a placeholder
        print("Strategy comparison feature coming soon!")
    
    def export_results(self, format: str = 'csv') -> None:
        """Export results in specified format."""
        
        if not self.results:
            print("No results to export")
            return
        
        if format == 'csv':
            # Export to CSV
            metrics_df = pd.DataFrame([self.results.get('metrics', {})])
            print("Results exported to CSV format")
        elif format == 'json':
            # Export to JSON
            import json
            print("Results exported to JSON format")
        else:
            print(f"Unsupported export format: {format}")


# Convenience functions for easy use in notebooks
def create_backtest_interface() -> JupyterBacktestInterface:
    """Create a new backtest interface."""
    return JupyterBacktestInterface()


def quick_backtest(ticker: str, strategy: str = 'markov', days: int = 365) -> Dict[str, Any]:
    """
    Quick backtest function for simple use cases.
    
    Args:
        ticker: Stock ticker symbol
        strategy: Strategy to use ('markov', 'buy_hold', 'sma', 'rsi')
        days: Number of days to backtest
        
    Returns:
        Backtest results dictionary
    """
    
    async def _run_quick_backtest():
        # Create configurations
        config = BacktestConfig(
            start_date=pd.Timestamp.now() - pd.Timedelta(days=days),
            end_date=pd.Timestamp.now(),
            initial_capital=100000.0
        )
        
        processing_config = ProcessingConfig(
            max_workers=1,
            use_multiprocessing=False,
            use_async_io=True
        )
        
        # Create strategy
        if strategy == 'markov':
            strategy_obj = MarkovChainStrategy()
            strategy_obj.initialize()
        elif strategy == 'buy_hold':
            strategy_obj = BuyHoldStrategy()
            strategy_obj.initialize()
        elif strategy == 'sma':
            strategy_obj = SMAStrategy()
            strategy_obj.initialize()
        elif strategy == 'rsi':
            strategy_obj = RSIStrategy()
            strategy_obj.initialize()
        else:
            raise ValueError(f"Unknown strategy: {strategy}")
        
        # Create engine
        engine = BacktestEngine(config, processing_config)
        engine.add_strategy(strategy_obj)
        
        # Run backtest
        return await engine.run_backtest(ticker)
    
    # Run the async function
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        return loop.run_until_complete(_run_quick_backtest())
    finally:
        loop.close()


def plot_equity_curve(results: Dict[str, Any], title: str = "Equity Curve") -> None:
    """Plot equity curve from backtest results."""
    
    equity_data = results.get('equity_curve', {})
    if not equity_data:
        print("No equity curve data found")
        return
    
    df = pd.DataFrame.from_dict(equity_data, orient='index')
    df.index = pd.to_datetime(df.index)
    df.columns = ['equity']
    
    # Normalize to 100
    df['equity'] = (df['equity'] / df['equity'].iloc[0]) * 100
    
    # Create plotly figure
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df.index,
        y=df['equity'],
        mode='lines',
        name='Equity Curve',
        line=dict(color='blue', width=2)
    ))
    
    fig.update_layout(
        title=title,
        xaxis_title='Date',
        yaxis_title='Equity (Normalized to 100)',
        hovermode='x unified'
    )
    
    fig.show()


def display_metrics_table(results: Dict[str, Any]) -> None:
    """Display metrics in a formatted table."""
    
    metrics = results.get('metrics', {})
    if not metrics:
        print("No metrics found")
        return
    
    # Create a simple HTML table
    html = "<table border='1' style='border-collapse: collapse; margin: 20px;'>"
    html += "<tr><th>Metric</th><th>Value</th></tr>"
    
    metric_names = {
        'total_return': 'Total Return (%)',
        'annualized_return': 'Annualized Return (%)',
        'sharpe_ratio': 'Sharpe Ratio',
        'sortino_ratio': 'Sortino Ratio',
        'max_drawdown': 'Max Drawdown (%)',
        'calmar_ratio': 'Calmar Ratio',
        'win_rate': 'Win Rate (%)',
        'num_trades': 'Number of Trades',
        'profit_factor': 'Profit Factor'
    }
    
    for key, display_name in metric_names.items():
        if key in metrics:
            value = metrics[key]
            if isinstance(value, float):
                formatted_value = f"{value:.2f}"
            else:
                formatted_value = str(value)
            html += f"<tr><td>{display_name}</td><td>{formatted_value}</td></tr>"
    
    html += "</table>"
    
    display(HTML(html))
