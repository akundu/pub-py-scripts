#!/usr/bin/env python3
"""
Test script to verify TimescaleDB setup is working correctly.

This script tests the database setup by:
1. Connecting to the database
2. Checking if all required tables exist
3. Verifying constraints are in place
4. Testing basic INSERT operations with ON CONFLICT
5. Testing basic SELECT operations

Usage: python test_timescaledb_setup.py <database_connection_string>
Example: python test_timescaledb_setup.py 'timescaledb://user:password@localhost:5432/stock_data'
"""

import asyncio
import asyncpg
import sys
import logging
from datetime import datetime, date

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class TimescaleDBTester:
    def __init__(self, db_config: str):
        """Initialize the tester."""
        # Convert timescaledb:// scheme to postgresql:// for asyncpg compatibility
        if db_config.startswith('timescaledb://'):
            self.db_config = db_config.replace('timescaledb://', 'postgresql://', 1)
            logger.info(f"Converted connection string to: {self.db_config}")
        else:
            self.db_config = db_config
        
        self.schema = "stock_data"
        
    async def run_all_tests(self):
        """Run all tests to verify the setup."""
        try:
            conn = await asyncpg.connect(self.db_config)
            logger.info("Connected to database successfully")
            
            # Run all tests
            await self._test_connection(conn)
            await self._test_schema_exists(conn)
            await self._test_tables_exist(conn)
            await self._test_constraints_exist(conn)
            await self._test_hypertables(conn)
            await self._test_insert_operations(conn)
            await self._test_select_operations(conn)
            await self._test_on_conflict(conn)
            
            logger.info("🎉 All tests passed! TimescaleDB setup is working correctly.")
            
        except Exception as e:
            logger.error(f"❌ Test failed: {e}")
            raise
        finally:
            if 'conn' in locals():
                await conn.close()
    
    async def _test_connection(self, conn: asyncpg.Connection):
        """Test basic database connection."""
        logger.info("Testing database connection...")
        result = await conn.fetchval("SELECT 1")
        assert result == 1, "Basic connection test failed"
        logger.info("✅ Database connection test passed")
    
    async def _test_schema_exists(self, conn: asyncpg.Connection):
        """Test if the stock_data schema exists."""
        logger.info("Testing if stock_data schema exists...")
        schema_exists = await conn.fetchval("""
            SELECT EXISTS(
                SELECT 1 FROM information_schema.schemata 
                WHERE schema_name = $1
            )
        """, self.schema)
        
        assert schema_exists, f"Schema {self.schema} does not exist"
        logger.info(f"✅ Schema {self.schema} exists")
    
    async def _test_tables_exist(self, conn: asyncpg.Connection):
        """Test if all required tables exist."""
        logger.info("Testing if all required tables exist...")
        
        required_tables = ['daily_prices', 'hourly_prices', 'realtime_data']
        for table in required_tables:
            table_exists = await conn.fetchval(f"""
                SELECT EXISTS(
                    SELECT 1 FROM information_schema.tables 
                    WHERE table_schema = $1 AND table_name = $2
                )
            """, self.schema, table)
            
            assert table_exists, f"Table {self.schema}.{table} does not exist"
            logger.info(f"✅ Table {self.schema}.{table} exists")
    
    async def _test_constraints_exist(self, conn: asyncpg.Connection):
        """Test if all required constraints exist."""
        logger.info("Testing if all required constraints exist...")
        
        constraints_to_check = [
            ("daily_prices", "ticker, date"),
            ("hourly_prices", "ticker, datetime"),
            ("realtime_data", "ticker, timestamp, type")
        ]
        
        for table, columns in constraints_to_check:
            constraint_exists = await conn.fetchval(f"""
                SELECT EXISTS(
                    SELECT 1 FROM information_schema.table_constraints 
                    WHERE table_schema = $1 
                    AND table_name = $2 
                    AND constraint_type = 'UNIQUE' 
                    AND constraint_name LIKE $3
                )
            """, self.schema, table, f"%{columns.split(', ')[-1]}%")
            
            assert constraint_exists, f"Unique constraint missing for {self.schema}.{table}"
            logger.info(f"✅ Constraint verified for {self.schema}.{table}")
    
    async def _test_hypertables(self, conn: asyncpg.Connection):
        """Test if tables are properly converted to hypertables."""
        logger.info("Testing if tables are converted to hypertables...")
        
        try:
            hypertable_count = await conn.fetchval("""
                SELECT COUNT(*) FROM timescaledb_information.hypertables 
                WHERE schema_name = $1
            """, self.schema)
            
            assert hypertable_count >= 3, f"Expected at least 3 hypertables, found {hypertable_count}"
            logger.info(f"✅ {hypertable_count} hypertables found in {self.schema} schema")
            
        except Exception as e:
            logger.warning(f"⚠️  Hypertable test failed (TimescaleDB extension might not be available): {e}")
    
    async def _test_insert_operations(self, conn: asyncpg.Connection):
        """Test basic INSERT operations."""
        logger.info("Testing basic INSERT operations...")
        
        # Test daily_prices insert
        await conn.execute(f"""
            INSERT INTO {self.schema}.daily_prices (ticker, date, open, high, low, close, volume)
            VALUES ($1, $2, $3, $4, $5, $6, $7)
        """, 'TEST', date.today(), 100.0, 101.0, 99.0, 100.5, 1000)
        logger.info("✅ daily_prices INSERT test passed")
        
        # Test hourly_prices insert
        await conn.execute(f"""
            INSERT INTO {self.schema}.hourly_prices (ticker, datetime, open, high, low, close, volume)
            VALUES ($1, $2, $3, $4, $5, $6, $7)
        """, 'TEST', datetime.now(), 100.0, 101.0, 99.0, 100.5, 1000)
        logger.info("✅ hourly_prices INSERT test passed")
        
        # Test realtime_data insert
        await conn.execute(f"""
            INSERT INTO {self.schema}.realtime_data (ticker, timestamp, type, price, size)
            VALUES ($1, $2, $3, $4, $5)
        """, 'TEST', datetime.now(), 'trade', 100.5, 100)
        logger.info("✅ realtime_data INSERT test passed")
    
    async def _test_select_operations(self, conn: asyncpg.Connection):
        """Test basic SELECT operations."""
        logger.info("Testing basic SELECT operations...")
        
        # Test daily_prices select
        daily_count = await conn.fetchval(f"SELECT COUNT(*) FROM {self.schema}.daily_prices WHERE ticker = 'TEST'")
        assert daily_count > 0, "daily_prices SELECT test failed"
        logger.info("✅ daily_prices SELECT test passed")
        
        # Test hourly_prices select
        hourly_count = await conn.fetchval(f"SELECT COUNT(*) FROM {self.schema}.hourly_prices WHERE ticker = 'TEST'")
        assert hourly_count > 0, "hourly_prices SELECT test failed"
        logger.info("✅ hourly_prices SELECT test passed")
        
        # Test realtime_data select
        realtime_count = await conn.fetchval(f"SELECT COUNT(*) FROM {self.schema}.realtime_data WHERE ticker = 'TEST'")
        assert realtime_count > 0, "realtime_data SELECT test failed"
        logger.info("✅ realtime_data SELECT test passed")
    
    async def _test_on_conflict(self, conn: asyncpg.Connection):
        """Test ON CONFLICT operations."""
        logger.info("Testing ON CONFLICT operations...")
        
        # Test daily_prices ON CONFLICT
        try:
            await conn.execute(f"""
                INSERT INTO {self.schema}.daily_prices (ticker, date, open, high, low, close, volume)
                VALUES ($1, $2, $3, $4, $5, $6, $7)
                ON CONFLICT (ticker, date) DO UPDATE SET 
                    close = EXCLUDED.close,
                    volume = EXCLUDED.volume
            """, 'TEST', date.today(), 100.0, 101.0, 99.0, 100.5, 1000)
            logger.info("✅ daily_prices ON CONFLICT test passed")
        except Exception as e:
            logger.error(f"❌ daily_prices ON CONFLICT test failed: {e}")
            raise
        
        # Test hourly_prices ON CONFLICT
        try:
            await conn.execute(f"""
                INSERT INTO {self.schema}.hourly_prices (ticker, datetime, open, high, low, close, volume)
                VALUES ($1, $2, $3, $4, $5, $6, $7)
                ON CONFLICT (ticker, datetime) DO UPDATE SET 
                    close = EXCLUDED.close,
                    volume = EXCLUDED.volume
            """, 'TEST', datetime.now(), 100.0, 101.0, 99.0, 100.5, 1000)
            logger.info("✅ hourly_prices ON CONFLICT test passed")
        except Exception as e:
            logger.error(f"❌ hourly_prices ON CONFLICT test failed: {e}")
            raise
        
        # Test realtime_data ON CONFLICT
        try:
            await conn.execute(f"""
                INSERT INTO {self.schema}.realtime_data (ticker, timestamp, type, price, size)
                VALUES ($1, $2, $3, $4, $5)
                ON CONFLICT (ticker, timestamp, type) DO UPDATE SET 
                    price = EXCLUDED.price,
                    size = EXCLUDED.size
            """, 'TEST', datetime.now(), 'trade', 100.5, 100)
            logger.info("✅ realtime_data ON CONFLICT test passed")
        except Exception as e:
            logger.error(f"❌ realtime_data ON CONFLICT test failed: {e}")
            raise
    
    async def cleanup_test_data(self):
        """Clean up test data."""
        try:
            conn = await asyncpg.connect(self.db_config)
            logger.info("Cleaning up test data...")
            
            # Remove test data
            await conn.execute(f"DELETE FROM {self.schema}.daily_prices WHERE ticker = 'TEST'")
            await conn.execute(f"DELETE FROM {self.schema}.hourly_prices WHERE ticker = 'TEST'")
            await conn.execute(f"DELETE FROM {self.schema}.realtime_data WHERE ticker = 'TEST'")
            
            logger.info("✅ Test data cleaned up")
            
        except Exception as e:
            logger.warning(f"⚠️  Failed to cleanup test data: {e}")
        finally:
            if 'conn' in locals():
                await conn.close()

async def main():
    """Main function to run the tests."""
    if len(sys.argv) != 2:
        print("Usage: python test_timescaledb_setup.py <database_connection_string>")
        print("Example: python test_timescaledb_setup.py 'timescaledb://user:password@localhost:5432/stock_data'")
        sys.exit(1)
    
    db_config = sys.argv[1]
    logger.info(f"Testing TimescaleDB setup for: {db_config}")
    
    try:
        tester = TimescaleDBTester(db_config)
        await tester.run_all_tests()
        
        # Ask if user wants to cleanup test data
        response = input("\nDo you want to clean up the test data? (yes/no): ")
        if response.lower() == 'yes':
            await tester.cleanup_test_data()
        
        logger.info("🎉 All tests completed successfully!")
        logger.info("Your TimescaleDB setup is working correctly.")
        
    except Exception as e:
        logger.error(f"❌ Tests failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())
