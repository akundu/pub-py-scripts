#!/usr/bin/env python3
"""
Fix TimescaleDB constraints for ON CONFLICT support.

This script adds the missing unique constraints to the TimescaleDB tables
that are needed for the ON CONFLICT clause to work properly.
"""

import asyncio
import asyncpg
import sys
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

async def fix_constraints(db_config: str):
    """Fix missing unique constraints in TimescaleDB tables."""
    
    # Convert timescaledb:// scheme to postgresql:// for asyncpg compatibility
    if db_config.startswith('timescaledb://'):
        db_config = db_config.replace('timescaledb://', 'postgresql://', 1)
    
    try:
        conn = await asyncpg.connect(db_config)
        logger.info("Connected to database successfully")
        
        # Define constraints to add in stock_data schema
        constraints_to_add = [
            {
                'table': 'stock_data.daily_prices',
                'columns': 'ticker, date',
                'constraint_name': 'daily_prices_ticker_date_unique'
            },
            {
                'table': 'stock_data.hourly_prices', 
                'columns': 'ticker, datetime',
                'constraint_name': 'hourly_prices_ticker_datetime_unique'
            },
            {
                'table': 'stock_data.realtime_data',
                'columns': 'ticker, timestamp, type', 
                'constraint_name': 'realtime_data_ticker_timestamp_type_unique'
            }
        ]
        
        for constraint in constraints_to_add:
            table = constraint['table']
            columns = constraint['columns']
            constraint_name = constraint['constraint_name']
            
            try:
                # Extract table name without schema for constraint checking
                table_name = table.split('.')[-1] if '.' in table else table
                # Check if constraint already exists
                constraint_exists = await conn.fetchval(f"""
                    SELECT EXISTS(
                        SELECT 1 FROM information_schema.table_constraints 
                        WHERE table_name = $1 
                        AND constraint_type = 'UNIQUE' 
                        AND constraint_name LIKE $2
                    )
                """, table_name, f"%{constraint_name.split('_')[-2]}_{constraint_name.split('_')[-1]}%")
                
                if not constraint_exists:
                    logger.info(f"Adding unique constraint to {table} table...")
                    await conn.execute(f"""
                        ALTER TABLE {table} 
                        ADD CONSTRAINT {constraint_name} 
                        UNIQUE({columns})
                    """)
                    logger.info(f"Successfully added unique constraint to {table} table")
                else:
                    logger.info(f"Unique constraint already exists on {table} table")
                    
            except Exception as e:
                if "already exists" in str(e).lower():
                    logger.info(f"Unique constraint already exists on {table} table")
                else:
                    logger.error(f"Could not add constraint to {table}: {e}")
        
        logger.info("Constraint check completed successfully")
        
    except Exception as e:
        logger.error(f"Failed to fix constraints: {e}")
        raise
    finally:
        if 'conn' in locals():
            await conn.close()

async def main():
    """Main function to run the constraint fix."""
    if len(sys.argv) != 2:
        print("Usage: python fix_timescaledb_constraints.py <database_connection_string>")
        print("Example: python fix_timescaledb_constraints.py 'timescaledb://user:password@localhost:5432/stock_data'")
        sys.exit(1)
    
    db_config = sys.argv[1]
    logger.info(f"Fixing constraints for database: {db_config}")
    
    try:
        await fix_constraints(db_config)
        logger.info("All constraints have been fixed successfully!")
    except Exception as e:
        logger.error(f"Failed to fix constraints: {e}")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())
