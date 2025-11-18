"""
Examples demonstrating how to use the Markov Chain Strategy parameters.

This file shows how to configure:
1. state_bins - for discretization granularity
2. momentum_periods - for momentum calculation timeframes  
3. volatility_period - for volatility calculation window
4. exponential_decay - for data weighting
5. no_decay_periods - for equal weighting of recent data
"""

import pandas as pd
import numpy as np
from stock_backtest.strategies.markov_chain import MarkovChainStrategy

def create_sample_data():
    """Create sample stock data for demonstration."""
    dates = pd.date_range('2023-01-01', periods=300, freq='D')
    np.random.seed(42)
    
    # Generate realistic price data
    returns = np.random.normal(0.001, 0.02, 300)  # 0.1% daily return, 2% volatility
    prices = 100 * np.exp(np.cumsum(returns))
    
    # Generate volume data
    volumes = np.random.lognormal(10, 0.5, 300)
    
    data = pd.DataFrame({
        'date': dates,
        'open': prices * (1 + np.random.normal(0, 0.005, 300)),
        'high': prices * (1 + np.abs(np.random.normal(0, 0.01, 300))),
        'low': prices * (1 - np.abs(np.random.normal(0, 0.01, 300))),
        'close': prices,
        'volume': volumes
    })
    
    return data

def example_state_bins():
    """Demonstrate different state_bins configurations."""
    print("=== STATE_BINS EXAMPLES ===")
    
    data = create_sample_data()
    
    # Fine-grained states (more sensitive)
    strategy_fine = MarkovChainStrategy("FineGrained")
    strategy_fine.initialize(
        state_bins=10,  # 10 bins per feature
        lookback_period=100
    )
    
    # Coarse states (more robust)
    strategy_coarse = MarkovChainStrategy("CoarseGrained") 
    strategy_coarse.initialize(
        state_bins=3,   # 3 bins per feature
        lookback_period=100
    )
    
    print(f"Fine-grained strategy: {strategy_fine.parameters['state_bins']} bins")
    print(f"Coarse-grained strategy: {strategy_coarse.parameters['state_bins']} bins")
    
    # Generate signals
    signal_fine = strategy_fine.generate_signal(data)
    signal_coarse = strategy_coarse.generate_signal(data)
    
    print(f"Fine-grained signal: {signal_fine.signal}, confidence: {signal_fine.confidence:.1f}%")
    print(f"Coarse-grained signal: {signal_coarse.signal}, confidence: {signal_coarse.confidence:.1f}%")

def example_momentum_periods():
    """Demonstrate different momentum_periods configurations."""
    print("\n=== MOMENTUM_PERIODS EXAMPLES ===")
    
    data = create_sample_data()
    
    # Short-term momentum focus
    strategy_short = MarkovChainStrategy("ShortTerm")
    strategy_short.initialize(
        momentum_periods=[3, 7, 14],  # Short-term momentum
        lookback_period=100
    )
    
    # Long-term momentum focus
    strategy_long = MarkovChainStrategy("LongTerm")
    strategy_long.initialize(
        momentum_periods=[20, 50, 100],  # Long-term momentum
        lookback_period=100
    )
    
    # Mixed timeframes
    strategy_mixed = MarkovChainStrategy("MixedTerm")
    strategy_mixed.initialize(
        momentum_periods=[5, 15, 30, 60],  # Multiple timeframes
        lookback_period=100
    )
    
    print(f"Short-term momentum periods: {strategy_short.parameters['momentum_periods']}")
    print(f"Long-term momentum periods: {strategy_long.parameters['momentum_periods']}")
    print(f"Mixed momentum periods: {strategy_mixed.parameters['momentum_periods']}")

def example_volatility_period():
    """Demonstrate different volatility_period configurations."""
    print("\n=== VOLATILITY_PERIOD EXAMPLES ===")
    
    data = create_sample_data()
    
    # Responsive volatility
    strategy_responsive = MarkovChainStrategy("ResponsiveVol")
    strategy_responsive.initialize(
        volatility_period=10,  # More responsive
        lookback_period=100
    )
    
    # Smooth volatility
    strategy_smooth = MarkovChainStrategy("SmoothVol")
    strategy_smooth.initialize(
        volatility_period=30,  # Smoother
        lookback_period=100
    )
    
    print(f"Responsive volatility period: {strategy_responsive.parameters['volatility_period']} days")
    print(f"Smooth volatility period: {strategy_smooth.parameters['volatility_period']} days")

def example_exponential_decay():
    """Demonstrate exponential decay effects."""
    print("\n=== EXPONENTIAL_DECAY EXAMPLES ===")
    
    data = create_sample_data()
    
    # Fast decay (more emphasis on recent data)
    strategy_fast = MarkovChainStrategy("FastDecay")
    strategy_fast.initialize(
        exponential_decay=0.90,  # Fast decay
        lookback_period=100
    )
    
    # Slow decay (more balanced weighting)
    strategy_slow = MarkovChainStrategy("SlowDecay")
    strategy_slow.initialize(
        exponential_decay=0.98,  # Slow decay
        lookback_period=100
    )
    
    # No decay for last 5 days
    strategy_no_decay = MarkovChainStrategy("NoDecayRecent")
    strategy_no_decay.initialize(
        exponential_decay=0.95,
        no_decay_periods=5,  # Last 5 days get equal weight
        lookback_period=100
    )
    
    print(f"Fast decay factor: {strategy_fast.parameters['exponential_decay']}")
    print(f"Slow decay factor: {strategy_slow.parameters['exponential_decay']}")
    print(f"No decay periods: {strategy_no_decay.parameters['no_decay_periods']}")
    
    # Show weight distribution
    weights_fast = strategy_fast._apply_exponential_weights(20)
    weights_slow = strategy_slow._apply_exponential_weights(20)
    weights_no_decay = strategy_no_decay._apply_exponential_weights(20)
    
    print(f"\nWeight distribution (last 10 periods):")
    print(f"Fast decay: {weights_fast[-10:]}")
    print(f"Slow decay: {weights_slow[-10:]}")
    print(f"No decay (last 5): {weights_no_decay[-10:]}")

def example_comprehensive_config():
    """Demonstrate a comprehensive configuration."""
    print("\n=== COMPREHENSIVE CONFIGURATION EXAMPLE ===")
    
    data = create_sample_data()
    
    # Comprehensive strategy configuration
    strategy = MarkovChainStrategy("Comprehensive")
    strategy.initialize(
        # State discretization
        state_bins=7,  # Moderate granularity
        
        # Momentum analysis
        momentum_periods=[5, 10, 20, 50],  # Multiple timeframes
        
        # Volatility analysis
        volatility_period=15,  # Moderate responsiveness
        
        # Data weighting
        exponential_decay=0.96,  # Moderate decay
        no_decay_periods=7,      # Last week gets equal weight
        
        # Other parameters
        lookback_period=200,     # 200 days of history
        prediction_horizon=3,    # Predict 3 days ahead
        min_state_frequency=3    # Minimum state frequency
    )
    
    print("Comprehensive strategy parameters:")
    for key, value in strategy.parameters.items():
        print(f"  {key}: {value}")
    
    # Generate signal
    signal = strategy.generate_signal(data)
    print(f"\nGenerated signal:")
    print(f"  Signal: {signal.signal}")
    print(f"  Confidence: {signal.confidence:.1f}%")
    print(f"  Expected movement: {signal.expected_movement_pct:.2f}%")
    print(f"  Reasoning: {signal.reasoning}")

if __name__ == "__main__":
    example_state_bins()
    example_momentum_periods()
    example_volatility_period()
    example_exponential_decay()
    example_comprehensive_config()
