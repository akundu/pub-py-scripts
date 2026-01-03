#!/usr/bin/env python3
"""
Test script to verify financial_data module functionality.

Tests:
1. Basic financial data fetching (async)
2. IV analysis integration when syncing
3. Caching behavior
4. Multiprocessing support
5. Database saving
"""

import os
import sys
import asyncio
import json
from pathlib import Path
from datetime import datetime

# Ensure project root is on sys.path
CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from common.stock_db import get_stock_db
from common.financial_data import get_financial_info, get_financial_info_worker


async def test_async_financial_fetch(symbol: str = "AAPL", force_fetch: bool = False):
    """Test async financial data fetching."""
    print(f"\n{'='*60}")
    print(f"TEST 1: Async Financial Data Fetch (force_fetch={force_fetch})")
    print(f"{'='*60}")
    
    # Get database config
    db_config = os.getenv("QUESTDB_URL") or "questdb://user:password@localhost:8812/stock_data"
    
    # Initialize database
    db = get_stock_db('questdb', db_config=db_config, enable_cache=True, 
                     redis_url=os.getenv("REDIS_URL"), log_level="INFO")
    await db._init_db()
    
    try:
        result = await get_financial_info(
            symbol=symbol,
            db_instance=db,
            force_fetch=force_fetch
        )
        
        print(f"\nResult for {symbol}:")
        print(f"  Source: {result.get('source')}")
        print(f"  Fetch time: {result.get('fetch_time_ms', 0):.2f}ms")
        print(f"  Error: {result.get('error', 'None')}")
        
        if result.get('financial_data'):
            financial = result['financial_data']
            print(f"\nFinancial Data Keys: {list(financial.keys())[:10]}...")  # Show first 10 keys
            
            # Check for key financial ratios
            key_ratios = ['price_to_earnings', 'price_to_book', 'price_to_sales', 'dividend_yield']
            print("\nKey Ratios:")
            for ratio in key_ratios:
                if ratio in financial:
                    print(f"  {ratio}: {financial[ratio]}")
            
            # Check for IV analysis data
            if 'iv_analysis_json' in financial:
                print("\n✓ IV Analysis JSON found in financial data")
                try:
                    iv_data = json.loads(financial['iv_analysis_json'])
                    print(f"  IV Metrics: {iv_data.get('metrics', {}).keys()}")
                    print(f"  IV Strategy: {iv_data.get('strategy', {}).get('recommendation', 'N/A')}")
                except:
                    print("  (Could not parse IV JSON)")
            else:
                print("\n- No IV Analysis JSON (expected if not syncing with include_iv_analysis=True)")
            
            if 'iv_30d' in financial:
                print(f"\n✓ IV Columns found: iv_30d={financial.get('iv_30d')}, iv_rank={financial.get('iv_rank')}")
        else:
            print("  No financial data returned")
        
        return result
        
    finally:
        await db.close()


async def test_iv_analysis_integration(symbol: str = "AAPL"):
    """Test IV analysis integration when syncing."""
    print(f"\n{'='*60}")
    print(f"TEST 2: IV Analysis Integration (force_fetch=True, include_iv_analysis=True)")
    print(f"{'='*60}")
    
    # Get database config
    db_config = os.getenv("QUESTDB_URL") or "questdb://user:password@localhost:8812/stock_data"
    
    # Initialize database
    db = get_stock_db('questdb', db_config=db_config, enable_cache=True, 
                     redis_url=os.getenv("REDIS_URL"), log_level="INFO")
    await db._init_db()
    
    try:
        print(f"\nFetching financial data with IV analysis for {symbol}...")
        print("(This may take 30-60 seconds as it fetches from API and calculates IV)")
        
        result = await get_financial_info(
            symbol=symbol,
            db_instance=db,
            force_fetch=True,  # Force API fetch
            include_iv_analysis=True,  # Include IV analysis
            iv_calendar_days=90,
            iv_server_url=os.getenv("DB_SERVER_URL", "http://localhost:9100"),
            iv_use_polygon=False,
            iv_data_dir="data"
        )
        
        print(f"\nResult for {symbol}:")
        print(f"  Source: {result.get('source')}")
        print(f"  Fetch time: {result.get('fetch_time_ms', 0):.2f}ms")
        print(f"  Error: {result.get('error', 'None')}")
        
        if result.get('financial_data'):
            financial = result['financial_data']
            
            # Check IV analysis data
            print("\n" + "="*60)
            print("IV Analysis Integration Check:")
            print("="*60)
            
            has_iv_columns = any(key in financial for key in ['iv_30d', 'iv_rank', 'relative_rank'])
            has_iv_json = 'iv_analysis_json' in financial
            
            if has_iv_columns:
                print("✓ IV columns found in financial data:")
                print(f"  iv_30d: {financial.get('iv_30d')}")
                print(f"  iv_rank: {financial.get('iv_rank')}")
                print(f"  relative_rank: {financial.get('relative_rank')}")
            else:
                print("✗ IV columns NOT found in financial data")
            
            if has_iv_json:
                print("✓ IV analysis JSON found")
                try:
                    iv_data = json.loads(financial['iv_analysis_json'])
                    print(f"  Ticker: {iv_data.get('ticker')}")
                    print(f"  Metrics: {list(iv_data.get('metrics', {}).keys())}")
                    print(f"  Strategy: {iv_data.get('strategy', {}).get('recommendation', 'N/A')}")
                except Exception as e:
                    print(f"  (Could not parse: {e})")
            else:
                print("✗ IV analysis JSON NOT found")
            
            # Verify data was saved to DB
            print("\n" + "="*60)
            print("Database Verification:")
            print("="*60)
            try:
                db_result = await db.get_financial_info(symbol)
                if not db_result.empty:
                    latest = db_result.iloc[-1].to_dict()
                    print("✓ Data found in database")
                    if 'iv_30d' in latest:
                        print(f"  iv_30d in DB: {latest.get('iv_30d')}")
                    if 'iv_analysis_json' in latest:
                        print(f"  iv_analysis_json in DB: {bool(latest.get('iv_analysis_json'))}")
                else:
                    print("✗ Data NOT found in database")
            except Exception as e:
                print(f"✗ Error checking database: {e}")
        else:
            print("✗ No financial data returned")
        
        return result
        
    finally:
        await db.close()


def test_multiprocessing_worker(symbol: str = "MSFT"):
    """Test multiprocessing worker function using actual ProcessPoolExecutor."""
    print(f"\n{'='*60}")
    print(f"TEST 3: Multiprocessing Worker (symbol={symbol})")
    print(f"{'='*60}")
    
    # Get database config
    db_config = os.getenv("QUESTDB_URL") or "questdb://user:password@localhost:8812/stock_data"
    
    print(f"\nRunning worker for {symbol} using ProcessPoolExecutor...")
    print("Note: This creates a separate process with its own event loop.")
    
    try:
        from concurrent.futures import ProcessPoolExecutor
        
        # Use ProcessPoolExecutor to actually test multiprocessing
        with ProcessPoolExecutor(max_workers=1) as executor:
            future = executor.submit(
                get_financial_info_worker,
                symbol=symbol,
                db_config=db_config,
                db_type="questdb",
                force_fetch=False,  # Use cache if available
                include_iv_analysis=False,  # Skip IV for speed
                redis_url=os.getenv("REDIS_URL"),
                log_level="INFO"
            )
            result = future.result(timeout=30)  # 30 second timeout
        
        print(f"\nResult for {symbol}:")
        print(f"  Source: {result.get('source', 'None')}")
        fetch_time = result.get('fetch_time_ms')
        if fetch_time is not None:
            try:
                print(f"  Fetch time: {fetch_time:.2f}ms")
            except (TypeError, ValueError):
                print(f"  Fetch time: {fetch_time}")
        else:
            print(f"  Fetch time: N/A")
        print(f"  Error: {result.get('error', 'None')}")
        
        if result.get('financial_data'):
            financial = result['financial_data']
            print(f"\nFinancial Data Keys: {list(financial.keys())[:10]}...")
            print("✓ Worker function executed successfully in separate process")
        else:
            print("✗ No financial data returned")
            if result.get('error'):
                print(f"  Error details: {result.get('error')}")
        
        return result
    except Exception as e:
        print(f"\n✗ Worker function failed with exception: {e}")
        import traceback
        traceback.print_exc()
        return {
            "symbol": symbol,
            "financial_data": None,
            "error": str(e),
            "source": None,
            "fetch_time_ms": 0
        }


async def test_caching_behavior(symbol: str = "AAPL"):
    """Test caching behavior."""
    print(f"\n{'='*60}")
    print(f"TEST 4: Caching Behavior")
    print(f"{'='*60}")
    
    # Get database config
    db_config = os.getenv("QUESTDB_URL") or "questdb://user:password@localhost:8812/stock_data"
    
    # Initialize database
    db = get_stock_db('questdb', db_config=db_config, enable_cache=True, 
                     redis_url=os.getenv("REDIS_URL"), log_level="INFO")
    await db._init_db()
    
    try:
        # First fetch - should hit API or DB
        print(f"\nFirst fetch for {symbol} (force_fetch=False)...")
        start1 = datetime.now()
        result1 = await get_financial_info(symbol, db, force_fetch=False)
        time1 = (datetime.now() - start1).total_seconds() * 1000
        
        print(f"  Source: {result1.get('source')}")
        print(f"  Time: {time1:.2f}ms")
        
        # Second fetch - should hit cache
        print(f"\nSecond fetch for {symbol} (force_fetch=False, should use cache)...")
        start2 = datetime.now()
        result2 = await get_financial_info(symbol, db, force_fetch=False)
        time2 = (datetime.now() - start2).total_seconds() * 1000
        
        print(f"  Source: {result2.get('source')}")
        print(f"  Time: {time2:.2f}ms")
        
        if result2.get('source') == 'cache' and time2 < time1:
            print(f"\n✓ Caching working: Cache hit was {time1/time2:.1f}x faster")
        elif result2.get('source') == 'cache':
            print(f"\n✓ Cache hit confirmed (source: cache)")
        else:
            print(f"\n⚠ Cache may not be working (source: {result2.get('source')})")
        
    finally:
        await db.close()


async def main():
    """Run all tests."""
    print("\n" + "="*60)
    print("Financial Data Module Verification Tests")
    print("="*60)
    print("\nPrerequisites:")
    print("  - POLYGON_API_KEY environment variable set")
    print("  - REDIS_URL environment variable set (optional, for caching)")
    print("  - Database accessible (QUESTDB_URL or default)")
    
    # Check prerequisites
    if not os.getenv("POLYGON_API_KEY"):
        print("\n✗ ERROR: POLYGON_API_KEY not set!")
        return
    
    print("\n✓ Prerequisites check passed")
    
    # Test 1: Basic async fetch
    try:
        await test_async_financial_fetch("AAPL", force_fetch=False)
    except Exception as e:
        print(f"\n✗ Test 1 failed: {e}")
        import traceback
        traceback.print_exc()
    
    # Test 2: IV analysis integration
    try:
        await test_iv_analysis_integration("AAPL")
    except Exception as e:
        print(f"\n✗ Test 2 failed: {e}")
        import traceback
        traceback.print_exc()
    
    # Test 3: Multiprocessing (ProcessPoolExecutor creates separate processes with clean event loops)
    try:
        # Run in executor to avoid blocking async event loop
        import asyncio
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(None, test_multiprocessing_worker, "MSFT")
    except Exception as e:
        print(f"\n✗ Test 3 failed: {e}")
        import traceback
        traceback.print_exc()
    
    # Test 4: Caching
    try:
        await test_caching_behavior("AAPL")
    except Exception as e:
        print(f"\n✗ Test 4 failed: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "="*60)
    print("All tests completed!")
    print("="*60)


if __name__ == "__main__":
    asyncio.run(main())
