#!/usr/bin/env python3
"""
Clean up duplicate tables in the public schema.

This script removes duplicate tables from the public schema that might be causing
conflicts with the stock_data schema tables.

WARNING: This will DELETE data from the public schema tables!
Only use this if you're sure you want to remove the duplicate tables.

Usage: python cleanup_duplicate_tables.py <database_connection_string>
Example: python cleanup_duplicate_tables.py 'timescaledb://user:password@localhost:5432/stock_data'
"""

import asyncio
import asyncpg
import sys
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DuplicateTableCleanup:
    def __init__(self, db_config: str):
        """Initialize the cleanup process."""
        # Convert timescaledb:// scheme to postgresql:// for asyncpg compatibility
        if db_config.startswith('timescaledb://'):
            self.db_config = db_config.replace('timescaledb://', 'postgresql://', 1)
            logger.info(f"Converted connection string to: {self.db_config}")
        else:
            self.db_config = db_config
        
        self.schema = "stock_data"
        
    async def cleanup_duplicates(self):
        """Clean up duplicate tables in the public schema."""
        try:
            conn = await asyncpg.connect(self.db_config)
            logger.info("Connected to database successfully")
            
            # Check for duplicate tables
            duplicates = await self._find_duplicate_tables(conn)
            
            if not duplicates:
                logger.info("No duplicate tables found. Nothing to clean up.")
                return
            
            # Show what will be cleaned up
            logger.info(f"Found {len(duplicates)} duplicate tables in public schema:")
            for table in duplicates:
                logger.info(f"  - public.{table}")
            
            # Ask for confirmation
            response = input("\n⚠️  WARNING: This will DELETE data from public schema tables!\n"
                           "Are you sure you want to continue? (yes/no): ")
            
            if response.lower() != 'yes':
                logger.info("Cleanup cancelled by user.")
                return
            
            # Perform cleanup
            await self._remove_duplicate_tables(conn, duplicates)
            
            logger.info("✅ Duplicate table cleanup completed successfully!")
            
        except Exception as e:
            logger.error(f"Failed to cleanup duplicate tables: {e}")
            raise
        finally:
            if 'conn' in locals():
                await conn.close()
    
    async def _find_duplicate_tables(self, conn: asyncpg.Connection):
        """Find tables that exist in both public and stock_data schemas."""
        logger.info("Checking for duplicate tables...")
        
        # Get tables in public schema
        public_tables = await conn.fetch("""
            SELECT tablename FROM pg_tables 
            WHERE schemaname = 'public' 
            AND tablename IN ('daily_prices', 'hourly_prices', 'realtime_data')
        """)
        
        # Get tables in stock_data schema
        stock_data_tables = await conn.fetch(f"""
            SELECT tablename FROM pg_tables 
            WHERE schemaname = '{self.schema}' 
            AND tablename IN ('daily_prices', 'hourly_prices', 'realtime_data')
        """)
        
        public_table_names = {row['tablename'] for row in public_tables}
        stock_data_table_names = {row['tablename'] for row in stock_data_tables}
        
        # Find duplicates
        duplicates = public_table_names.intersection(stock_data_table_names)
        
        logger.info(f"Tables in public schema: {list(public_table_names)}")
        logger.info(f"Tables in {self.schema} schema: {list(stock_data_table_names)}")
        logger.info(f"Duplicate tables: {list(duplicates)}")
        
        return list(duplicates)
    
    async def _remove_duplicate_tables(self, conn: asyncpg.Connection, tables: list):
        """Remove duplicate tables from the public schema."""
        logger.info("Removing duplicate tables from public schema...")
        
        for table in tables:
            try:
                # Check if table has data
                row_count = await conn.fetchval(f"SELECT COUNT(*) FROM public.{table}")
                logger.info(f"Table public.{table} has {row_count} rows")
                
                if row_count > 0:
                    logger.warning(f"⚠️  Table public.{table} contains {row_count} rows of data!")
                    response = input(f"Delete public.{table} with {row_count} rows? (yes/no): ")
                    if response.lower() != 'yes':
                        logger.info(f"Skipping deletion of public.{table}")
                        continue
                
                # Drop the table
                await conn.execute(f"DROP TABLE IF EXISTS public.{table} CASCADE")
                logger.info(f"✅ Dropped table public.{table}")
                
            except Exception as e:
                logger.error(f"Failed to drop table public.{table}: {e}")
    
    async def verify_cleanup(self):
        """Verify that the cleanup was successful."""
        try:
            conn = await asyncpg.connect(self.db_config)
            logger.info("Verifying cleanup...")
            
            # Check if duplicate tables still exist
            duplicates = await self._find_duplicate_tables(conn)
            
            if not duplicates:
                logger.info("✅ Cleanup verification successful - no duplicate tables found")
            else:
                logger.warning(f"⚠️  Cleanup verification failed - {len(duplicates)} duplicate tables still exist")
                for table in duplicates:
                    logger.warning(f"  - public.{table}")
            
        except Exception as e:
            logger.error(f"Failed to verify cleanup: {e}")
        finally:
            if 'conn' in locals():
                await conn.close()

async def main():
    """Main function to run the duplicate table cleanup."""
    if len(sys.argv) != 2:
        print("Usage: python cleanup_duplicate_tables.py <database_connection_string>")
        print("Example: python cleanup_duplicate_tables.py 'timescaledb://user:password@localhost:5432/stock_data'")
        sys.exit(1)
    
    db_config = sys.argv[1]
    logger.info(f"Starting duplicate table cleanup for: {db_config}")
    
    try:
        cleanup = DuplicateTableCleanup(db_config)
        await cleanup.cleanup_duplicates()
        
        # Verify cleanup
        await cleanup.verify_cleanup()
        
        logger.info("🎉 Duplicate table cleanup completed successfully!")
        logger.info("Your application should now work with the stock_data schema tables.")
        
    except Exception as e:
        logger.error(f"❌ Duplicate table cleanup failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())
