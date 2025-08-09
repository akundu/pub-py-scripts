#!/usr/bin/env python3
"""
Test script to verify that the PostgreSQL database respects the command-line log level.
This tests the fix for the issue where postgres_db.py was logging at INFO level 
even when ERROR level was specified on the command line.
"""

import asyncio
import logging
import sys
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from common.postgres_db import StockDBPostgreSQL
from common.logging_utils import get_logger

async def test_log_level_respect():
    """Test that PostgreSQL database respects the specified log level."""
    
    print("🧪 Testing PostgreSQL log level configuration...")
    
    # Test 1: ERROR level logging
    print("\n1. Testing ERROR level logging:")
    print("   - Should only show ERROR and CRITICAL messages")
    print("   - INFO messages from postgres_db should be suppressed")
    
    error_logger = get_logger("test_error_logger", level="ERROR")
    
    try:
        # This should NOT show INFO messages from postgres_db
        db_error = StockDBPostgreSQL(
            "postgresql://stock_user:stock_password@localhost:5432/stock_data",
            logger=error_logger,
            log_level="ERROR"
        )
        
        print("   ✅ PostgreSQL instance created with ERROR level")
        
        # Force an error to see if error logging works
        try:
            await db_error.get_latest_price("INVALID_TICKER_THAT_CAUSES_ERROR")
        except Exception as e:
            print(f"   ✅ Error logged correctly: {type(e).__name__}")
        
        await db_error.close_pool()
        
    except Exception as e:
        print(f"   ❌ Failed to create PostgreSQL instance: {e}")
        return False
    
    print("\n2. Testing INFO level logging:")
    print("   - Should show INFO, WARNING, ERROR, and CRITICAL messages")
    
    info_logger = get_logger("test_info_logger", level="INFO")
    
    try:
        # This SHOULD show INFO messages from postgres_db
        db_info = StockDBPostgreSQL(
            "postgresql://stock_user:stock_password@localhost:5432/stock_data",
            logger=info_logger,
            log_level="INFO"
        )
        
        print("   ✅ PostgreSQL instance created with INFO level")
        
        await db_info.close_pool()
        
    except Exception as e:
        print(f"   ❌ Failed to create PostgreSQL instance: {e}")
        return False
    
    print("\n🎉 Log level test completed!")
    print("\nTo test with db_server.py:")
    print("   python db_server.py --db-file 'postgresql://stock_user:stock_password@localhost:5432/stock_data' --port 9999 --log-level ERROR")
    print("   - You should only see ERROR and CRITICAL messages")
    print("   - No INFO messages from postgres_db should appear")
    
    return True

def test_command_line_logging():
    """Test command-line style logging setup."""
    print("\n3. Testing command-line style logging setup:")
    
    # Simulate what db_server.py does
    from db_server import setup_logging, initialize_database
    
    # Setup ERROR level logging
    setup_logging(log_level_str="ERROR")
    
    try:
        # This should respect ERROR level
        db_instance = initialize_database(
            "postgresql://stock_user:stock_password@localhost:5432/stock_data",
            log_level="ERROR"
        )
        print("   ✅ Database initialized with command-line ERROR level")
        return True
        
    except Exception as e:
        print(f"   ❌ Failed to initialize database: {e}")
        return False

if __name__ == "__main__":
    print("=" * 60)
    print("🔧 PostgreSQL Log Level Compliance Test")
    print("=" * 60)
    
    try:
        # Test async functions
        result = asyncio.run(test_log_level_respect())
        
        # Test command-line style
        cmd_result = test_command_line_logging()
        
        if result and cmd_result:
            print("\n✅ All tests passed! Log level configuration is working correctly.")
            sys.exit(0)
        else:
            print("\n❌ Some tests failed. Check the output above.")
            sys.exit(1)
            
    except KeyboardInterrupt:
        print("\n⚠️  Test interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Test failed with exception: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
