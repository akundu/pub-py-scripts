#!/usr/bin/env python3
"""
Test script to fetch NDX (NASDAQ 100) data directly from Polygon.io
"""

import os
import sys
import argparse
from datetime import datetime, timedelta
from polygon.rest import RESTClient
import pandas as pd

def test_polygon_ndx(symbol="NDX"):
    """Test fetching data from Polygon for a given symbol
    
    Args:
        symbol: Symbol to fetch (default: "NDX"). For indices, use "NDX" or "I:NDX".
                For stocks, use the ticker symbol directly (e.g., "AAPL").
    """
    
    # Get API key from environment
    api_key = os.getenv('POLYGON_API_KEY')
    if not api_key:
        print("ERROR: POLYGON_API_KEY environment variable not set", file=sys.stderr)
        sys.exit(1)
    
    print(f"Testing Polygon API for symbol: {symbol}")
    print(f"API Key: {api_key[:10]}...{api_key[-4:] if len(api_key) > 14 else '***'}")
    print()
    
    # Create Polygon client
    client = RESTClient(api_key)
    
    # Calculate date range (last 5 days)
    end_date = datetime.now()
    start_date = end_date - timedelta(days=5)
    
    start_str = start_date.strftime('%Y-%m-%d')
    end_str = end_date.strftime('%Y-%m-%d')
    
    print(f"Fetching {symbol} data from {start_str} to {end_str}")
    print()
    
    # Polygon requires I: prefix for indices
    # If symbol doesn't start with I: or O:, assume it's an index and add I: prefix
    if not symbol.startswith(('I:', 'O:')):
        polygon_symbol = f"I:{symbol}"
    else:
        polygon_symbol = symbol
    
    try:
        # Fetch daily aggregates
        print(f"Attempting to fetch daily aggregates for symbol: {polygon_symbol}")
        aggs = []
        
        for agg in client.list_aggs(
            ticker=polygon_symbol,
            multiplier=1,
            timespan="day",
            from_=start_str,
            to=end_str,
            limit=50000
        ):
            aggs.append(agg)
        
        if not aggs:
            print(f"❌ No daily data returned from Polygon for {polygon_symbol}")
            return
        
        print(f"✓ Successfully fetched {len(aggs)} records")
        print()
        
        # Convert to DataFrame
        df_data = []
        for agg in aggs:
            df_data.append({
                'timestamp': pd.to_datetime(agg.timestamp, unit='ms', utc=True),
                'open': agg.open,
                'high': agg.high,
                'low': agg.low,
                'close': agg.close,
                'volume': agg.volume,
                'vwap': getattr(agg, 'vwap', None),
                'transactions': getattr(agg, 'transactions', None)
            })
        
        df = pd.DataFrame(df_data)
        df.set_index('timestamp', inplace=True)
        df.sort_index(inplace=True)
        
        print("Data retrieved:")
        print("=" * 80)
        print(df.to_string())
        print("=" * 80)
        print()
        print(f"Date range: {df.index.min()} to {df.index.max()}")
        print(f"Total records: {len(df)}")
        print()
        
        # Check if today's data is included
        today = datetime.now().date()
        today_data = df[df.index.date == today]
        if not today_data.empty:
            print(f"✓ Today's data ({today}) is included:")
            print(today_data.to_string())
        else:
            print(f"⚠ Today's data ({today}) is NOT included")
            print(f"  Latest data: {df.index.max().date()}")
        
        # Now fetch hourly data
        print()
        print("=" * 80)
        print("Fetching HOURLY data...")
        print("=" * 80)
        print()
        
        # For hourly, use a shorter range (last 2 days)
        hourly_start = end_date - timedelta(days=2)
        hourly_start_str = hourly_start.strftime('%Y-%m-%d')
        
        print(f"Fetching hourly {symbol} data from {hourly_start_str} to {end_str}")
        print()
        
        hourly_aggs = []
        for agg in client.list_aggs(
            ticker=polygon_symbol,
            multiplier=1,
            timespan="hour",
            from_=hourly_start_str,
            to=end_str,
            limit=50000
        ):
            hourly_aggs.append(agg)
        
        if not hourly_aggs:
            print("❌ No hourly data returned from Polygon")
        else:
            print(f"✓ Successfully fetched {len(hourly_aggs)} hourly records")
            print()
            
            # Convert to DataFrame
            hourly_df_data = []
            for agg in hourly_aggs:
                hourly_df_data.append({
                    'timestamp': pd.to_datetime(agg.timestamp, unit='ms', utc=True),
                    'open': agg.open,
                    'high': agg.high,
                    'low': agg.low,
                    'close': agg.close,
                    'volume': agg.volume,
                    'vwap': getattr(agg, 'vwap', None),
                    'transactions': getattr(agg, 'transactions', None)
                })
            
            hourly_df = pd.DataFrame(hourly_df_data)
            hourly_df.set_index('timestamp', inplace=True)
            hourly_df.sort_index(inplace=True)
            
            print("Hourly data retrieved:")
            print("=" * 80)
            print(hourly_df.to_string())
            print("=" * 80)
            print()
            print(f"Hourly date range: {hourly_df.index.min()} to {hourly_df.index.max()}")
            print(f"Total hourly records: {len(hourly_df)}")
            print()
            
            # Check if today's hourly data is included
            today_hourly = hourly_df[hourly_df.index.date == today]
            if not today_hourly.empty:
                print(f"✓ Today's hourly data ({today}) is included ({len(today_hourly)} hours):")
                print(today_hourly.to_string())
            else:
                print(f"⚠ Today's hourly data ({today}) is NOT included")
                print(f"  Latest hourly data: {hourly_df.index.max()}")
        
    except Exception as e:
        print(f"❌ Error fetching data from Polygon: {e}")
        import traceback
        traceback.print_exc()
        return

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Test script to fetch data from Polygon.io for a given symbol",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s                    # Default: fetch NDX (NASDAQ 100)
  %(prog)s -s NDX             # Fetch NDX index
  %(prog)s --symbol SPX       # Fetch S&P 500 index
  %(prog)s -s AAPL            # Fetch Apple stock (will use I:AAPL, may need adjustment)
  %(prog)s -s I:NDX           # Explicitly specify index format
        """
    )
    parser.add_argument(
        '-s', '--symbol',
        type=str,
        default='NDX',
        help='Symbol to fetch (default: NDX). For indices, use symbol name (e.g., NDX, SPX). '
             'For explicit index format, use I: prefix (e.g., I:NDX). '
             'For stocks, use ticker symbol directly.'
    )
    
    args = parser.parse_args()
    test_polygon_ndx(args.symbol)
