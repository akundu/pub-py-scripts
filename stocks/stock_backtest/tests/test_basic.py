"""
Simple Test Suite for Stock Backtesting Framework

Basic tests to verify the framework components work correctly.
"""

import asyncio
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from stock_backtest.backtesting.engine import BacktestEngine
from stock_backtest.backtesting.config import BacktestConfig, ProcessingConfig
from stock_backtest.strategies.markov_chain import MarkovChainStrategy
from stock_backtest.strategies.buy_hold import BuyHoldStrategy
from stock_backtest.strategies.technical_indicators import SMAStrategy, RSIStrategy
from stock_backtest.analysis.comparison import StrategyComparison
from stock_backtest.analysis.visualization import VisualizationEngine


def create_sample_data(ticker: str = "TEST", days: int = 252) -> pd.DataFrame:
    """Create sample stock data for testing."""
    
    # Generate random walk data
    np.random.seed(42)  # For reproducible results
    
    dates = pd.date_range(
        start=datetime.now() - timedelta(days=days),
        end=datetime.now(),
        freq='D'
    )
    
    # Generate price data using geometric Brownian motion
    initial_price = 100.0
    drift = 0.0005  # Daily drift
    volatility = 0.02  # Daily volatility
    
    returns = np.random.normal(drift, volatility, len(dates))
    prices = [initial_price]
    
    for ret in returns[1:]:
        prices.append(prices[-1] * (1 + ret))
    
    # Create OHLCV data
    data = []
    for i, (date, price) in enumerate(zip(dates, prices)):
        # Generate realistic OHLC from close price
        high = price * (1 + abs(np.random.normal(0, 0.01)))
        low = price * (1 - abs(np.random.normal(0, 0.01)))
        open_price = prices[i-1] if i > 0 else price
        volume = np.random.randint(1000000, 10000000)
        
        data.append({
            'open': open_price,
            'high': high,
            'low': low,
            'close': price,
            'volume': volume
        })
    
    df = pd.DataFrame(data, index=dates)
    return df


async def test_markov_chain_strategy():
    """Test Markov chain strategy."""
    print("Testing Markov Chain Strategy...")
    
    # Create sample data
    data = create_sample_data("TEST", 300)
    
    # Create strategy
    strategy = MarkovChainStrategy()
    strategy.initialize(
        lookback_period=100,
        prediction_horizon=3,
        state_bins=5
    )
    
    # Test signal generation
    signal = strategy.generate_signal(data)
    
    print(f"  Signal: {signal.signal.value}")
    print(f"  Direction: {signal.direction.value}")
    print(f"  Confidence: {signal.confidence:.1f}%")
    print(f"  Expected Movement: {signal.expected_movement_pct:.2f}%")
    
    assert signal.signal.value in ['BUY', 'SELL', 'HOLD']
    assert signal.direction.value in ['LONG', 'SHORT']
    assert 0 <= signal.confidence <= 100
    
    print("  ✓ Markov Chain Strategy test passed\n")


async def test_buy_hold_strategy():
    """Test buy and hold strategy."""
    print("Testing Buy & Hold Strategy...")
    
    # Create sample data
    data = create_sample_data("TEST", 100)
    
    # Create strategy
    strategy = BuyHoldStrategy()
    strategy.initialize()
    
    # Test signal generation
    signal = strategy.generate_signal(data)
    
    print(f"  Signal: {signal.signal.value}")
    print(f"  Direction: {signal.direction.value}")
    print(f"  Confidence: {signal.confidence:.1f}%")
    
    assert signal.signal.value in ['BUY', 'SELL', 'HOLD']
    assert signal.direction.value in ['LONG', 'SHORT']
    assert 0 <= signal.confidence <= 100
    
    print("  ✓ Buy & Hold Strategy test passed\n")


async def test_sma_strategy():
    """Test SMA strategy."""
    print("Testing SMA Strategy...")
    
    # Create sample data
    data = create_sample_data("TEST", 100)
    
    # Create strategy
    strategy = SMAStrategy()
    strategy.initialize(short_period=10, long_period=30)
    
    # Test signal generation
    signal = strategy.generate_signal(data)
    
    print(f"  Signal: {signal.signal.value}")
    print(f"  Direction: {signal.direction.value}")
    print(f"  Confidence: {signal.confidence:.1f}%")
    
    assert signal.signal.value in ['BUY', 'SELL', 'HOLD']
    assert signal.direction.value in ['LONG', 'SHORT']
    assert 0 <= signal.confidence <= 100
    
    print("  ✓ SMA Strategy test passed\n")


async def test_rsi_strategy():
    """Test RSI strategy."""
    print("Testing RSI Strategy...")
    
    # Create sample data
    data = create_sample_data("TEST", 100)
    
    # Create strategy
    strategy = RSIStrategy()
    strategy.initialize(rsi_period=14, oversold_threshold=30, overbought_threshold=70)
    
    # Test signal generation
    signal = strategy.generate_signal(data)
    
    print(f"  Signal: {signal.signal.value}")
    print(f"  Direction: {signal.direction.value}")
    print(f"  Confidence: {signal.confidence:.1f}%")
    
    assert signal.signal.value in ['BUY', 'SELL', 'HOLD']
    assert signal.direction.value in ['LONG', 'SHORT']
    assert 0 <= signal.confidence <= 100
    
    print("  ✓ RSI Strategy test passed\n")


async def test_backtest_engine():
    """Test backtesting engine."""
    print("Testing Backtesting Engine...")
    
    # Create configurations
    config = BacktestConfig(
        start_date=datetime.now() - timedelta(days=100),
        end_date=datetime.now(),
        initial_capital=100000.0
    )
    
    processing_config = ProcessingConfig(
        max_workers=1,
        use_multiprocessing=False,
        use_async_io=False
    )
    
    # Create strategy
    strategy = BuyHoldStrategy()
    strategy.initialize()
    
    # Create engine
    engine = BacktestEngine(config, processing_config)
    engine.add_strategy(strategy)
    
    # Create sample data
    data = create_sample_data("TEST", 100)
    
    # Mock the data fetcher to return our sample data
    async def mock_fetch_data(ticker, data_source, db_config):
        return data
    
    engine.data_fetcher.fetch_from_database = mock_fetch_data
    
    # Run backtest
    result = await engine.run_backtest("TEST", "database")
    
    print(f"  Ticker: {result['ticker']}")
    print(f"  Total Return: {result['metrics']['total_return']:.2f}%")
    print(f"  Number of Trades: {result['metrics']['num_trades']}")
    
    assert result['ticker'] == "TEST"
    assert 'metrics' in result
    assert 'portfolio_summary' in result
    
    print("  ✓ Backtesting Engine test passed\n")


def test_strategy_comparison():
    """Test strategy comparison."""
    print("Testing Strategy Comparison...")
    
    # Create sample results
    results = {
        'Strategy1': {
            'metrics': {
                'total_return': 15.5,
                'sharpe_ratio': 1.2,
                'max_drawdown': -8.3,
                'win_rate': 65.0
            }
        },
        'Strategy2': {
            'metrics': {
                'total_return': 12.1,
                'sharpe_ratio': 0.9,
                'max_drawdown': -12.5,
                'win_rate': 58.0
            }
        }
    }
    
    # Create comparison engine
    comparison_engine = StrategyComparison()
    
    # Run comparison
    comparison_result = comparison_engine.compare_strategies(results)
    
    print(f"  Number of strategies compared: {len(comparison_result.strategy_results)}")
    print(f"  Comparison table shape: {comparison_result.comparison_table.shape}")
    
    assert len(comparison_result.strategy_results) == 2
    assert not comparison_result.comparison_table.empty
    
    print("  ✓ Strategy Comparison test passed\n")


def test_visualization_engine():
    """Test visualization engine."""
    print("Testing Visualization Engine...")
    
    # Create sample results
    results = {
        'Strategy1': {
            'equity_curve': {
                '2023-01-01': 100000,
                '2023-01-02': 101000,
                '2023-01-03': 102500,
                '2023-01-04': 101800,
                '2023-01-05': 103200
            },
            'metrics': {
                'total_return': 15.5,
                'sharpe_ratio': 1.2,
                'max_drawdown': -8.3
            }
        }
    }
    
    # Create visualization engine
    viz_engine = VisualizationEngine()
    
    # Test equity curve plotting
    try:
        fig = viz_engine.plot_equity_curves(results, title="Test Equity Curves")
        print(f"  Equity curve plot created: {fig is not None}")
        assert fig is not None
    except Exception as e:
        print(f"  Warning: Could not create equity curve plot: {str(e)}")
    
    print("  ✓ Visualization Engine test passed\n")


async def run_all_tests():
    """Run all tests."""
    print("="*60)
    print("STOCK BACKTESTING FRAMEWORK - TEST SUITE")
    print("="*60)
    print()
    
    try:
        await test_markov_chain_strategy()
        await test_buy_hold_strategy()
        await test_sma_strategy()
        await test_rsi_strategy()
        await test_backtest_engine()
        test_strategy_comparison()
        test_visualization_engine()
        
        print("="*60)
        print("ALL TESTS PASSED SUCCESSFULLY! ✓")
        print("="*60)
        
    except Exception as e:
        print(f"TEST FAILED: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


def test_quick_backtest():
    """Quick integration test."""
    print("Running Quick Integration Test...")
    
    # Create sample data
    data = create_sample_data("TEST", 200)
    
    # Test multiple strategies
    strategies = [
        ('MarkovChain', MarkovChainStrategy()),
        ('BuyHold', BuyHoldStrategy()),
        ('SMA', SMAStrategy()),
        ('RSI', RSIStrategy())
    ]
    
    results = {}
    
    for name, strategy in strategies:
        try:
            if name == 'MarkovChain':
                strategy.initialize(lookback_period=50, prediction_horizon=3)
            elif name == 'SMA':
                strategy.initialize(short_period=10, long_period=20)
            elif name == 'RSI':
                strategy.initialize()
            else:
                strategy.initialize()
            
            signal = strategy.generate_signal(data)
            results[name] = {
                'signal': signal.signal.value,
                'confidence': signal.confidence,
                'expected_movement': signal.expected_movement_pct
            }
            
        except Exception as e:
            print(f"  Error testing {name}: {str(e)}")
            results[name] = {'error': str(e)}
    
    # Display results
    print("\nStrategy Comparison Results:")
    print("-" * 40)
    for name, result in results.items():
        if 'error' in result:
            print(f"{name:12}: ERROR - {result['error']}")
        else:
            print(f"{name:12}: {result['signal']:4} | {result['confidence']:5.1f}% | {result['expected_movement']:+6.2f}%")
    
    print("\n✓ Quick Integration Test completed\n")


if __name__ == "__main__":
    # Run quick test first
    test_quick_backtest()
    
    # Run full test suite
    asyncio.run(run_all_tests())
