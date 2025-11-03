"""
Stock Investment Prediction & Backtesting Framework using Markov Chains

A comprehensive framework for stock analysis and backtesting with:
- Markov chain prediction models
- Multiple strategy implementations
- Parallel processing capabilities
- Rich visualization and reporting
- CLI and Jupyter notebook interfaces
"""

__version__ = "1.0.0"
__author__ = "Stock Analysis Team"

from .strategies.base import AbstractStrategy
from .strategies.markov_chain import MarkovChainStrategy
from .strategies.buy_hold import BuyHoldStrategy
from .strategies.technical_indicators import SMAStrategy, RSIStrategy
from .backtesting.engine import BacktestEngine
from .backtesting.config import BacktestConfig
from .backtesting.portfolio import Portfolio
from .backtesting.metrics import PerformanceMetrics
from .data.fetcher import DataFetcher
from .analysis.comparison import StrategyComparison
from .analysis.visualization import VisualizationEngine
from .parallel.multiprocess_runner import MultiProcessRunner
from .cli.main import main as cli_main

__all__ = [
    "AbstractStrategy",
    "MarkovChainStrategy", 
    "BuyHoldStrategy",
    "SMAStrategy",
    "RSIStrategy",
    "BacktestEngine",
    "BacktestConfig",
    "Portfolio",
    "PerformanceMetrics",
    "DataFetcher",
    "StrategyComparison",
    "VisualizationEngine",
    "MultiProcessRunner",
    "cli_main"
]
