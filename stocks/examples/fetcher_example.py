#!/usr/bin/env python3
"""
Example usage of the data fetcher module.

Demonstrates how to use the new fetcher architecture to fetch
financial data from various sources.
"""

import asyncio
import os
from datetime import datetime, timedelta
from common.fetcher import FetcherFactory


async def example_yahoo_finance():
    """Example: Fetch data using Yahoo Finance."""
    print("\n" + "="*60)
    print("Example 1: Yahoo Finance - Daily Data")
    print("="*60)
    
    # Create Yahoo Finance fetcher
    fetcher = FetcherFactory.create_fetcher('yahoo', log_level='INFO')
    
    # Fetch daily data for AAPL
    result = await fetcher.fetch_historical_data(
        symbol='AAPL',
        timeframe='daily',
        start_date='2024-01-01',
        end_date='2024-01-31'
    )
    
    if result.success:
        print(f"✓ Fetched {result.records_fetched} records from {result.source}")
        print(f"  Date range: {result.start_date} to {result.end_date}")
        print(f"  Data shape: {result.data.shape}")
        print(f"\n  First few rows:")
        print(result.data.head())
    else:
        print(f"✗ Error: {result.error}")


async def example_yahoo_hourly_with_limit():
    """Example: Yahoo Finance hourly data with automatic limit handling."""
    print("\n" + "="*60)
    print("Example 2: Yahoo Finance - Hourly Data (729-day limit)")
    print("="*60)
    
    fetcher = FetcherFactory.create_fetcher('yahoo', log_level='INFO')
    
    # Try to fetch 1000 days of hourly data (will be limited to 729)
    today = datetime.now()
    start_date = (today - timedelta(days=1000)).strftime('%Y-%m-%d')
    end_date = today.strftime('%Y-%m-%d')
    
    result = await fetcher.fetch_historical_data(
        symbol='AAPL',
        timeframe='hourly',
        start_date=start_date,
        end_date=end_date
    )
    
    if result.success:
        print(f"✓ Fetched {result.records_fetched} records")
        print(f"  Requested start: {start_date}")
        print(f"  Actual start: {result.start_date} (adjusted to 729-day limit)")
        print(f"  End date: {result.end_date}")
    else:
        print(f"✗ Error: {result.error}")


async def example_index_symbol():
    """Example: Index symbol automatically routed to Yahoo Finance."""
    print("\n" + "="*60)
    print("Example 3: Index Symbol (Auto-routed to Yahoo Finance)")
    print("="*60)
    
    # Use factory with index symbol - will auto-detect and use Yahoo Finance
    fetcher = FetcherFactory.get_fetcher_for_symbol(
        symbol='I:SPX',  # S&P 500 index
        default_source='polygon'  # Would normally use Polygon, but index forces Yahoo
    )
    
    print(f"Fetcher type: {type(fetcher).__name__}")
    print(f"Fetcher name: {fetcher.name}")
    
    result = await fetcher.fetch_historical_data(
        symbol='^GSPC',  # Yahoo Finance symbol for S&P 500
        timeframe='daily',
        start_date='2024-01-01',
        end_date='2024-01-10'
    )
    
    if result.success:
        print(f"✓ Fetched {result.records_fetched} records for S&P 500")
        print(f"  Latest close: ${result.data['close'].iloc[-1]:.2f}")
    else:
        print(f"✗ Error: {result.error}")


async def example_polygon():
    """Example: Fetch data using Polygon (requires API key)."""
    print("\n" + "="*60)
    print("Example 4: Polygon.io - Daily Data with Chunking")
    print("="*60)
    
    # Check if API key is available
    api_key = os.getenv('POLYGON_API_KEY')
    if not api_key:
        print("⚠ Skipping - POLYGON_API_KEY not set")
        return
    
    # Create Polygon fetcher
    fetcher = FetcherFactory.create_fetcher(
        'polygon',
        api_key=api_key,
        log_level='INFO'
    )
    
    # Fetch data with chunking
    result = await fetcher.fetch_historical_data(
        symbol='AAPL',
        timeframe='daily',
        start_date='2023-01-01',
        end_date='2024-01-31',
        chunk_size='monthly'  # Fetch in monthly chunks
    )
    
    if result.success:
        print(f"✓ Fetched {result.records_fetched} records")
        print(f"  Chunk size: {result.metadata.get('chunk_size', 'N/A')}")
        print(f"  Data columns: {list(result.data.columns)}")
    else:
        print(f"✗ Error: {result.error}")


async def example_current_price():
    """Example: Fetch current price."""
    print("\n" + "="*60)
    print("Example 5: Current Price")
    print("="*60)
    
    fetcher = FetcherFactory.create_fetcher('yahoo')
    
    try:
        price_data = await fetcher.fetch_current_price('AAPL')
        print(f"✓ Current price for AAPL:")
        print(f"  Price: ${price_data['price']:.2f}")
        print(f"  Timestamp: {price_data['timestamp']}")
        print(f"  Source: {price_data['source']}")
        if price_data.get('bid_price'):
            print(f"  Bid: ${price_data['bid_price']:.2f}")
        if price_data.get('ask_price'):
            print(f"  Ask: ${price_data['ask_price']:.2f}")
    except Exception as e:
        print(f"✗ Error: {e}")


async def example_error_handling():
    """Example: Error handling."""
    print("\n" + "="*60)
    print("Example 6: Error Handling")
    print("="*60)
    
    fetcher = FetcherFactory.create_fetcher('yahoo', log_level='ERROR')
    
    # Try to fetch invalid symbol
    result = await fetcher.fetch_historical_data(
        symbol='INVALID_SYMBOL_XYZ',
        timeframe='daily',
        start_date='2024-01-01',
        end_date='2024-01-31'
    )
    
    if not result.success:
        print(f"✓ Error handled gracefully:")
        print(f"  Success: {result.success}")
        print(f"  Error message: {result.error}")
        print(f"  Data is empty: {result.data.empty}")
    else:
        print(f"  Unexpectedly succeeded")


async def main():
    """Run all examples."""
    print("\n" + "="*60)
    print("Data Fetcher Module Examples")
    print("="*60)
    
    # Run examples
    await example_yahoo_finance()
    await example_yahoo_hourly_with_limit()
    await example_index_symbol()
    await example_polygon()
    await example_current_price()
    await example_error_handling()
    
    print("\n" + "="*60)
    print("Examples complete!")
    print("="*60)
    print("\nTo use in your code:")
    print("  from common.fetcher import FetcherFactory")
    print("  fetcher = FetcherFactory.create_fetcher('yahoo')")
    print("  result = await fetcher.fetch_historical_data(...)")
    print()


if __name__ == '__main__':
    asyncio.run(main())
