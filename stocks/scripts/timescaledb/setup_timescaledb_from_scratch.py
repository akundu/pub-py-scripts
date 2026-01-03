#!/usr/bin/env python3
"""
Setup TimescaleDB from scratch with proper schemas, tables, and constraints.

This script creates a complete TimescaleDB setup with:
1. stock_data schema
2. All required tables with proper constraints
3. Hypertables for time-series optimization
4. Materialized views for performance
5. Proper unique constraints for ON CONFLICT support

Usage: python setup_timescaledb_from_scratch.py <database_connection_string>
Example: python setup_timescaledb_from_scratch.py 'timescaledb://user:password@localhost:5432/stock_data'
"""

import asyncio
import asyncpg
import sys
import logging
from datetime import datetime

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class TimescaleDBSetup:
    def __init__(self, db_config: str):
        """Initialize the TimescaleDB setup."""
        # Convert timescaledb:// scheme to postgresql:// for asyncpg compatibility
        if db_config.startswith('timescaledb://'):
            self.db_config = db_config.replace('timescaledb://', 'postgresql://', 1)
            logger.info(f"Converted connection string to: {self.db_config}")
        else:
            self.db_config = db_config
        
        self.schema = "stock_data"
        self.chunk_time_interval = "1 day"
        
    async def setup_database(self):
        """Set up the complete TimescaleDB database from scratch."""
        try:
            conn = await asyncpg.connect(self.db_config)
            logger.info("Connected to database successfully")
            
            # 1. Create schema
            await self._create_schema(conn)
            
            # 2. Create base tables with constraints
            await self._create_base_tables(conn)
            
            # 3. Enable TimescaleDB extension
            await self._enable_timescaledb(conn)
            
            # 4. Convert tables to hypertables
            await self._create_hypertables(conn)
            
            # 5. Create materialized views
            await self._create_materialized_views(conn)
            
            # 6. Create indexes for performance
            await self._create_indexes(conn)
            
            # 7. Verify setup
            await self._verify_setup(conn)
            
            logger.info("✅ TimescaleDB setup completed successfully!")
            
        except Exception as e:
            logger.error(f"Failed to setup TimescaleDB: {e}")
            raise
        finally:
            if 'conn' in locals():
                await conn.close()
    
    async def _create_schema(self, conn: asyncpg.Connection):
        """Create the stock_data schema."""
        logger.info("Creating stock_data schema...")
        try:
            await conn.execute(f"CREATE SCHEMA IF NOT EXISTS {self.schema}")
            logger.info(f"Schema {self.schema} created/verified successfully")
        except Exception as e:
            logger.warning(f"Schema creation issue (might already exist): {e}")
    
    async def _create_base_tables(self, conn: asyncpg.Connection):
        """Create all base tables with proper constraints."""
        logger.info("Creating base tables with constraints...")
        
        # Create daily_prices table
        await conn.execute(f"""
            CREATE TABLE IF NOT EXISTS {self.schema}.daily_prices (
                ticker VARCHAR(255) NOT NULL,
                date DATE NOT NULL,
                open DOUBLE PRECISION,
                high DOUBLE PRECISION,
                low DOUBLE PRECISION,
                close DOUBLE PRECISION,
                volume BIGINT,
                ma_10 DOUBLE PRECISION,
                ma_50 DOUBLE PRECISION,
                ma_100 DOUBLE PRECISION,
                ma_200 DOUBLE PRECISION,
                ema_8 DOUBLE PRECISION,
                ema_21 DOUBLE PRECISION,
                ema_34 DOUBLE PRECISION,
                ema_55 DOUBLE PRECISION,
                ema_89 DOUBLE PRECISION,
                UNIQUE(ticker, date)
            )
        """)
        logger.info("daily_prices table created with constraints")
        
        # Create hourly_prices table
        await conn.execute(f"""
            CREATE TABLE IF NOT EXISTS {self.schema}.hourly_prices (
                ticker VARCHAR(255) NOT NULL,
                datetime TIMESTAMP NOT NULL,
                open DOUBLE PRECISION,
                high DOUBLE PRECISION,
                low DOUBLE PRECISION,
                close DOUBLE PRECISION,
                volume BIGINT,
                UNIQUE(ticker, datetime)
            )
        """)
        logger.info("hourly_prices table created with constraints")
        
        # Create realtime_data table
        await conn.execute(f"""
            CREATE TABLE IF NOT EXISTS {self.schema}.realtime_data (
                ticker VARCHAR(255) NOT NULL,
                timestamp TIMESTAMP NOT NULL,
                type VARCHAR(50) NOT NULL,
                price DOUBLE PRECISION,
                size BIGINT,
                ask_price DOUBLE PRECISION,
                ask_size BIGINT,
                write_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(ticker, timestamp, type)
            )
        """)
        logger.info("realtime_data table created with constraints")
        
        # Create table_counts table for optimization
        await conn.execute(f"""
            CREATE TABLE IF NOT EXISTS {self.schema}.table_counts (
                table_name VARCHAR(255) PRIMARY KEY,
                row_count BIGINT DEFAULT 0,
                last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        logger.info("table_counts table created")
        
        # Create fast count views
        await conn.execute(f"""
            CREATE OR REPLACE VIEW {self.schema}.hourly_prices_count AS
            SELECT COUNT(*) as count FROM {self.schema}.hourly_prices
        """)
        
        await conn.execute(f"""
            CREATE OR REPLACE VIEW {self.schema}.daily_prices_count AS
            SELECT COUNT(*) as count FROM {self.schema}.daily_prices
        """)
        
        await conn.execute(f"""
            CREATE OR REPLACE VIEW {self.schema}.realtime_data_count AS
            SELECT COUNT(*) as count FROM {self.schema}.realtime_data
        """)
        logger.info("Fast count views created")
    
    async def _enable_timescaledb(self, conn: asyncpg.Connection):
        """Enable TimescaleDB extension."""
        logger.info("Enabling TimescaleDB extension...")
        try:
            await conn.execute("CREATE EXTENSION IF NOT EXISTS timescaledb")
            logger.info("TimescaleDB extension enabled successfully")
        except Exception as e:
            logger.error(f"Failed to enable TimescaleDB extension: {e}")
            raise RuntimeError("TimescaleDB extension not available. Ensure TimescaleDB is installed.")
    
    async def _create_hypertables(self, conn: asyncpg.Connection):
        """Convert regular tables to TimescaleDB hypertables."""
        logger.info("Converting tables to TimescaleDB hypertables...")
        
        # Convert daily_prices to hypertable
        try:
            await conn.execute(f"""
                SELECT create_hypertable(
                    '{self.schema}.daily_prices', 
                    'date',
                    chunk_time_interval => INTERVAL '{self.chunk_time_interval}',
                    if_not_exists => TRUE
                )
            """)
            logger.info("daily_prices converted to hypertable")
        except Exception as e:
            if "already a hypertable" not in str(e):
                logger.warning(f"Could not create daily_prices hypertable: {e}")
        
        # Convert hourly_prices to hypertable
        try:
            await conn.execute(f"""
                SELECT create_hypertable(
                    '{self.schema}.hourly_prices', 
                    'datetime',
                    chunk_time_interval => INTERVAL '{self.chunk_time_interval}',
                    if_not_exists => TRUE
                )
            """)
            logger.info("hourly_prices converted to hypertable")
        except Exception as e:
            if "already a hypertable" not in str(e):
                logger.warning(f"Could not create hourly_prices hypertable: {e}")
        
        # Convert realtime_data to hypertable
        try:
            await conn.execute(f"""
                SELECT create_hypertable(
                    '{self.schema}.realtime_data', 
                    'timestamp',
                    chunk_time_interval => INTERVAL '{self.chunk_time_interval}',
                    if_not_exists => TRUE
                )
            """)
            logger.info("realtime_data converted to hypertable")
        except Exception as e:
            if "already a hypertable" not in str(e):
                logger.warning(f"Could not create realtime_data hypertable: {e}")
    
    async def _create_materialized_views(self, conn: asyncpg.Connection):
        """Create materialized views for performance optimization."""
        logger.info("Creating materialized views...")
        
        # Daily OHLCV aggregate from hourly data
        try:
            await conn.execute(f"""
                CREATE MATERIALIZED VIEW IF NOT EXISTS {self.schema}.daily_ohlcv_agg AS
                SELECT 
                    date_trunc('day', datetime) as day,
                    ticker,
                    (array_agg(open ORDER BY datetime ASC) FILTER (WHERE open IS NOT NULL))[1] as open,
                    max(high) as high,
                    min(low) as low,
                    (array_agg(close ORDER BY datetime DESC) FILTER (WHERE close IS NOT NULL))[1] as close,
                    sum(volume) as volume,
                    count(*) as hour_count
                FROM {self.schema}.hourly_prices
                GROUP BY date_trunc('day', datetime), ticker
            """)
            logger.info("Daily OHLCV materialized view created")
        except Exception as e:
            if "already exists" not in str(e):
                logger.warning(f"Could not create daily OHLCV materialized view: {e}")
        
        # Hourly volume aggregate from realtime data
        try:
            await conn.execute(f"""
                CREATE MATERIALIZED VIEW IF NOT EXISTS {self.schema}.hourly_volume_agg AS
                SELECT 
                    date_trunc('hour', timestamp) as hour,
                    ticker,
                    sum(size) as total_volume,
                    count(*) as tick_count,
                    avg(price) as avg_price,
                    (array_agg(price ORDER BY timestamp ASC) FILTER (WHERE price IS NOT NULL))[1] as first_price,
                    (array_agg(price ORDER BY timestamp DESC) FILTER (WHERE price IS NOT NULL))[1] as last_price,
                    max(price) as high_price,
                    min(price) as low_price
                FROM {self.schema}.realtime_data
                WHERE type = 'trade'
                GROUP BY date_trunc('hour', timestamp), ticker
            """)
            logger.info("Hourly volume materialized view created")
        except Exception as e:
            if "already exists" not in str(e):
                logger.warning(f"Could not create hourly volume materialized view: {e}")
    
    async def _create_indexes(self, conn: asyncpg.Connection):
        """Create performance indexes."""
        logger.info("Creating performance indexes...")
        
        # Space dimension indexes (time is handled automatically by TimescaleDB)
        indexes = [
            (f"{self.schema}.daily_prices", "ticker"),
            (f"{self.schema}.hourly_prices", "ticker"),
            (f"{self.schema}.realtime_data", "ticker"),
            (f"{self.schema}.realtime_data", "type"),
            (f"{self.schema}.realtime_data", "ticker, type"),
        ]
        
        for table, columns in indexes:
            try:
                index_name = f"idx_{table.split('.')[-1]}_{columns.replace(', ', '_').replace(' ', '')}"
                await conn.execute(f"""
                    CREATE INDEX IF NOT EXISTS {index_name} 
                    ON {table}({columns})
                """)
                logger.debug(f"Created index {index_name}")
            except Exception as e:
                logger.warning(f"Could not create index on {table}({columns}): {e}")
        
        logger.info("Performance indexes created")
    
    async def _verify_setup(self, conn: asyncpg.Connection):
        """Verify that the setup is complete and working."""
        logger.info("Verifying setup...")
        
        # Check if all tables exist
        tables_to_check = ['daily_prices', 'hourly_prices', 'realtime_data']
        for table in tables_to_check:
            try:
                await conn.fetchval(f"SELECT 1 FROM {self.schema}.{table} LIMIT 1")
                logger.info(f"✅ Table {table} verified")
            except Exception as e:
                logger.error(f"❌ Table {table} verification failed: {e}")
                raise
        
        # Check if constraints exist
        constraints_to_check = [
            ("daily_prices", "ticker, date"),
            ("hourly_prices", "ticker, datetime"),
            ("realtime_data", "ticker, timestamp, type")
        ]
        
        for table, columns in constraints_to_check:
            try:
                constraint_exists = await conn.fetchval(f"""
                    SELECT EXISTS(
                        SELECT 1 FROM information_schema.table_constraints 
                        WHERE table_schema = '{self.schema}'
                        AND table_name = '{table}' 
                        AND constraint_type = 'UNIQUE' 
                        AND constraint_name LIKE '%{columns.split(', ')[-1]}%'
                    )
                """)
                
                if constraint_exists:
                    logger.info(f"✅ Constraint verified for {table}")
                else:
                    logger.warning(f"⚠️  Constraint missing for {table}")
            except Exception as e:
                logger.warning(f"Could not verify constraint for {table}: {e}")
        
        # Check if hypertables are working
        try:
            hypertable_count = await conn.fetchval("""
                SELECT COUNT(*) FROM timescaledb_information.hypertables 
                WHERE schema_name = $1
            """, self.schema)
            logger.info(f"✅ {hypertable_count} hypertables created in {self.schema} schema")
        except Exception as e:
            logger.warning(f"Could not verify hypertables: {e}")
        
        logger.info("Setup verification completed")

async def main():
    """Main function to run the TimescaleDB setup."""
    if len(sys.argv) != 2:
        print("Usage: python setup_timescaledb_from_scratch.py <database_connection_string>")
        print("Example: python setup_timescaledb_from_scratch.py 'timescaledb://user:password@localhost:5432/stock_data'")
        sys.exit(1)
    
    db_config = sys.argv[1]
    logger.info(f"Setting up TimescaleDB from scratch for: {db_config}")
    
    try:
        setup = TimescaleDBSetup(db_config)
        await setup.setup_database()
        logger.info("🎉 TimescaleDB setup completed successfully!")
        logger.info("You can now run your stock data application with proper schema support.")
    except Exception as e:
        logger.error(f"❌ TimescaleDB setup failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())
