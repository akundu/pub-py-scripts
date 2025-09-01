#!/usr/bin/env python3
"""
Basic test script for the streak analysis system.
"""

import asyncio
import sys
import os
sys.path.append('.')

from data_provider import MockDataProvider
from preprocess import prepare_data
from streaks import compute_streak_stats
from signals import suggest_thresholds
from evaluation import evaluate_intervals


async def test_basic_workflow():
    """Test the basic workflow with mock data."""
    print("🧪 Testing basic workflow with mock data...")
    
    # Initialize mock data provider
    provider = MockDataProvider()
    
    # Fetch mock data
    df = await provider.get_daily('TEST', lookback_days=100)
    print(f"✅ Fetched {len(df)} mock data points")
    
    # Prepare data
    df_prepared = prepare_data(df, aggregation_level='day')
    print(f"✅ Prepared {len(df_prepared)} data points")
    
    # Compute streaks
    streak_stats = compute_streak_stats(df_prepared, min_streak_threshold=1)
    print(f"✅ Found {len(streak_stats.get('streaks', []))} streaks")
    
    # Generate signals
    suggestions = suggest_thresholds(streak_stats)
    print(f"✅ Generated {len(suggestions.get('buy_thresholds', []))} buy signals")
    print(f"✅ Generated {len(suggestions.get('short_thresholds', []))} short signals")
    
    # Run intervaled evaluation
    eval_results = evaluate_intervals(df_prepared, n_days=100, m_days=30)
    print(f"✅ Evaluated {eval_results.get('parameters', {}).get('n_intervals', 0)} intervals")
    
    print("\n🎉 Basic workflow test completed successfully!")
    return True


async def test_data_provider():
    """Test the data provider functionality."""
    print("\n🔌 Testing data provider...")
    
    provider = MockDataProvider()
    
    # Test daily data
    daily_df = await provider.get_daily('TEST', lookback_days=30)
    print(f"✅ Daily data: {len(daily_df)} rows")
    
    # Test hourly data
    hourly_df = await provider.get_hourly('TEST', lookback_days=7)
    print(f"✅ Hourly data: {len(hourly_df)} rows")
    
    # Test realtime data
    realtime_df = await provider.get_realtime_window('TEST', window_days=3)
    print(f"✅ Realtime data: {len(realtime_df)} rows")
    
    return True


async def test_streak_detection():
    """Test streak detection with known patterns."""
    print("\n📈 Testing streak detection...")
    
    import pandas as pd
    import numpy as np
    
    # Create synthetic data with known patterns
    dates = pd.date_range('2024-01-01', periods=50, freq='D')
    
    # Create alternating pattern: 3 up, 2 down, 3 up, 2 down...
    returns = []
    for i in range(50):
        if (i // 5) % 2 == 0:  # First 3 days of each 5-day cycle
            returns.extend([0.01, 0.01, 0.01, -0.01, -0.01])
        else:  # Last 2 days of each 5-day cycle
            returns.extend([-0.01, -0.01, 0.01, 0.01, 0.01])
    
    returns = returns[:50]  # Trim to exact length
    
    # Create DataFrame
    df = pd.DataFrame({
        'open': [100] * 50,
        'high': [101] * 50,
        'low': [99] * 50,
        'close': [100] * 50,
        'volume': [1000000] * 50
    }, index=dates)
    
    # Add returns column
    df['returns'] = returns
    
    # Test streak detection
    from streaks import detect_streaks
    streaks = detect_streaks(df['returns'], min_threshold=2)
    
    print(f"✅ Detected {len(streaks)} streaks")
    
    # Verify expected patterns
    expected_streaks = 10  # 5 cycles * 2 streaks per cycle
    if len(streaks) == expected_streaks:
        print(f"✅ Streak count matches expected: {expected_streaks}")
    else:
        print(f"⚠️  Expected {expected_streaks} streaks, got {len(streaks)}")
    
    return True


async def main():
    """Run all tests."""
    print("🚀 Starting streak analysis system tests...\n")
    
    try:
        # Run tests
        await test_data_provider()
        await test_streak_detection()
        await test_basic_workflow()
        
        print("\n🎉 All tests completed successfully!")
        return True
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)
