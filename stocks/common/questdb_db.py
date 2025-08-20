"""
QuestDB implementation for stock data storage and retrieval.
Provides high-performance time-series storage with automatic partitioning,
columnar storage, and QuestDB-specific optimizations.
"""

import pandas as pd
from datetime import datetime, timezone, timedelta
import asyncio
import asyncpg
from typing import List, Dict, Any, Optional
import logging
from contextlib import asynccontextmanager
from .stock_db import StockDBBase
from .logging_utils import get_logger
import pytz
from dateutil import parser as date_parser


class StockQuestDB(StockDBBase):
    """
    QuestDB implementation optimized for high-performance time-series stock data.
    Inherits from StockDBBase and implements QuestDB-specific features:
    - Automatic time-based partitioning (monthly for daily/hourly, daily for realtime)
    - Columnar storage optimized for analytical queries
    - SYMBOL columns for efficient categorical data storage
    - High-throughput data ingestion via PostgreSQL wire protocol
    """
    
    def __init__(self, 
                 db_config: str,
                 pool_max_size: int = 10,
                 pool_connection_timeout_minutes: int = 30,
                 logger: Optional[logging.Logger] = None,
                 log_level: str = "INFO"):
        """
        Initialize QuestDB connection with time-series specific settings.
        
        Args:
            db_config: QuestDB connection string (questdb:// or postgresql:// format)
            pool_max_size: Maximum number of connections in the pool
            pool_connection_timeout_minutes: Connection timeout in minutes
            logger: Optional logger instance
            log_level: Logging level
        """
        # Convert questdb:// URLs to postgresql:// for QuestDB's PostgreSQL wire protocol
        original_config = db_config
        if db_config.startswith('questdb://'):
            db_config = db_config.replace('questdb://', 'postgresql://', 1)
            
        super().__init__(db_config, logger)
        
        if original_config.startswith('questdb://'):
            self.logger.info(f"Converted questdb:// URL to postgresql:// for QuestDB compatibility")
        
        # QuestDB-specific configuration
        self.pool_max_size = pool_max_size
        self.pool_connection_timeout_minutes = pool_connection_timeout_minutes
        self._connection_pool = None
        self._tables_ensured = False
        self._tables_ensured_at = None
        
        self.logger.info(f"QuestDB initialized with pool size: {pool_max_size}, "
                        f"connection timeout: {pool_connection_timeout_minutes} minutes")

    async def _init_db(self) -> None:
        """Initialize the QuestDB database with required tables."""
        await self._ensure_tables_exist()

    async def _ensure_tables_exist(self) -> None:
        """Create tables if they don't exist in QuestDB."""
        # Check cache first
        if self._tables_ensured and self._tables_ensured_at:
            cache_age = datetime.now() - self._tables_ensured_at
            if cache_age.total_seconds() < 600:  # 10 minutes cache
                self.logger.debug("Tables existence already verified (cached)")
                return

        self.logger.info("Creating QuestDB tables and optimizations...")
        
        async with self.get_connection() as conn:
            # Create tables with QuestDB optimizations
            await self._create_daily_prices_table(conn)
            await self._create_hourly_prices_table(conn)
            await self._create_realtime_data_table(conn)
            
            # Create indexes for optimal performance
            await self._create_questdb_indexes(conn)
            
        # Update cache
        self._tables_ensured = True
        self._tables_ensured_at = datetime.now()

    async def _create_daily_prices_table(self, conn: asyncpg.Connection) -> None:
        """Create daily_prices table with QuestDB optimizations."""
        create_table_sql = """
        CREATE TABLE IF NOT EXISTS daily_prices (
            ticker SYMBOL,
            date TIMESTAMP,
            open DOUBLE,
            high DOUBLE,
            low DOUBLE,
            close DOUBLE,
            volume LONG,
            ma_10 DOUBLE,
            ma_50 DOUBLE,
            ma_100 DOUBLE,
            ma_200 DOUBLE,
            ema_8 DOUBLE,
            ema_21 DOUBLE,
            ema_34 DOUBLE,
            ema_55 DOUBLE,
            ema_89 DOUBLE
        ) TIMESTAMP(date) PARTITION BY MONTH;
        """
        await conn.execute(create_table_sql)

    async def _create_hourly_prices_table(self, conn: asyncpg.Connection) -> None:
        """Create hourly_prices table with QuestDB optimizations."""
        create_table_sql = """
        CREATE TABLE IF NOT EXISTS hourly_prices (
            ticker SYMBOL,
            datetime TIMESTAMP,
            open DOUBLE,
            high DOUBLE,
            low DOUBLE,
            close DOUBLE,
            volume LONG
        ) TIMESTAMP(datetime) PARTITION BY MONTH;
        """
        await conn.execute(create_table_sql)

    async def _create_realtime_data_table(self, conn: asyncpg.Connection) -> None:
        """Create realtime_data table with QuestDB optimizations."""
        create_table_sql = """
        CREATE TABLE IF NOT EXISTS realtime_data (
            ticker SYMBOL,
            timestamp TIMESTAMP,
            type SYMBOL,
            price DOUBLE,
            size LONG,
            ask_price DOUBLE,
            ask_size LONG,
            write_timestamp TIMESTAMP
        ) TIMESTAMP(timestamp) PARTITION BY DAY;
        """
        await conn.execute(create_table_sql)

    async def _create_questdb_indexes(self, conn: asyncpg.Connection) -> None:
        """Create QuestDB-specific indexes for optimal performance."""
        indexes = [
            "CREATE INDEX IF NOT EXISTS idx_daily_prices_ticker ON daily_prices(ticker);",
            "CREATE INDEX IF NOT EXISTS idx_hourly_prices_ticker ON hourly_prices(ticker);",
            "CREATE INDEX IF NOT EXISTS idx_realtime_data_ticker ON realtime_data(ticker);",
            "CREATE INDEX IF NOT EXISTS idx_realtime_data_type ON realtime_data(type);",
            "CREATE INDEX IF NOT EXISTS idx_daily_prices_ticker_date ON daily_prices(ticker, date);",
            "CREATE INDEX IF NOT EXISTS idx_hourly_prices_ticker_datetime ON hourly_prices(ticker, datetime);",
            "CREATE INDEX IF NOT EXISTS idx_realtime_data_ticker_timestamp ON realtime_data(ticker, timestamp);"
        ]
        
        for index_sql in indexes:
            try:
                await conn.execute(index_sql)
            except Exception as e:
                self.logger.warning(f"Could not create index: {e}")

    @asynccontextmanager
    async def get_connection(self):
        """Context manager for safely getting and returning connections."""
        if self._connection_pool is None:
            self._connection_pool = await asyncpg.create_pool(
                self.db_config,
                min_size=1,
                max_size=self.pool_max_size,
                command_timeout=self.pool_connection_timeout_minutes * 60
            )
        
        async with self._connection_pool.acquire() as conn:
            yield conn

    async def save_stock_data(
        self,
        df: pd.DataFrame,
        ticker: str,
        interval: str = "daily",
        ma_periods: List[int] = None,
        ema_periods: List[int] = None,
        on_duplicate: str = "ignore",
    ) -> None:
        """Save aggregated (daily/hourly) stock data to QuestDB."""
        # Set default periods
        if ma_periods is None:
            ma_periods = [10, 50, 100, 200]
        if ema_periods is None:
            ema_periods = [8, 21, 34, 55, 89]

        async with self.get_connection() as conn:
            df_copy = df.copy()
            if df_copy.empty:
                return
                
            df_copy.reset_index(inplace=True)
            df_copy.columns = [col.lower() for col in df_copy.columns]
            df_copy['ticker'] = ticker

            table = 'daily_prices' if interval == 'daily' else 'hourly_prices'
            date_col = 'date' if interval == 'daily' else 'datetime'
            
            if 'index' in df_copy.columns and date_col not in df_copy.columns:
                df_copy.rename(columns={'index': date_col}, inplace=True)
                
            required_cols = ['ticker', date_col, 'open', 'high', 'low', 'close', 'volume']
            df_copy = df_copy[[col for col in required_cols if col in df_copy.columns]]

            if date_col not in df_copy.columns:
                self.logger.warning(f"Date column '{date_col}' not found for {ticker} ({interval})")
                return

            df_copy[date_col] = pd.to_datetime(df_copy[date_col])
            df_copy = df_copy.dropna(subset=[date_col])
            
            if df_copy.empty:
                self.logger.warning(f"No valid data for {ticker} ({interval})")
                return

            # Calculate MA and EMA for daily data only
            if interval == "daily":
                df_copy = await self._get_historical_data(
                    ticker,
                    df_copy[date_col].min(),
                    df_copy[date_col].max(),
                    df_copy,
                    interval,
                    ma_periods,
                    ema_periods,
                    date_col,
                    conn,
                )

            # Convert to QuestDB-optimized format
            df_copy[date_col] = df_copy[date_col].dt.strftime('%Y-%m-%d %H:%M:%S')
            
            # Prepare data for bulk insert
            records = []
            for _, row in df_copy.iterrows():
                record = {
                    'ticker': row['ticker'],
                    date_col: row[date_col],
                    'open': row.get('open', 0.0),
                    'high': row.get('high', 0.0),
                    'low': row.get('low', 0.0),
                    'close': row.get('close', 0.0),
                    'volume': int(row.get('volume', 0))
                }
                
                # Add MA values for daily data
                if interval == 'daily':
                    for period in ma_periods:
                        ma_key = f'ma_{period}'
                        if ma_key in row:
                            record[ma_key] = row[ma_key]
                    
                    # Add EMA values
                    for period in ema_periods:
                        ema_key = f'ema_{period}'
                        if ema_key in row:
                            record[ema_key] = row[ema_key]
                
                records.append(record)

            if records:
                # Use QuestDB's optimized bulk insert
                await self._bulk_insert_stock_data(conn, table, records, interval)

    async def _bulk_insert_stock_data(self, conn: asyncpg.Connection, table: str, records: List[Dict], interval: str):
        """Bulk insert stock data using QuestDB's optimized insert."""
        if not records:
            return
            
        # Build the INSERT statement dynamically based on available columns
        first_record = records[0]
        columns = list(first_record.keys())
        placeholders = [f'${i+1}' for i in range(len(columns))]
        
        insert_sql = f"""
        INSERT INTO {table} ({', '.join(columns)})
        VALUES ({', '.join(placeholders)})
        """
        
        # Prepare the data for insertion
        values = []
        for record in records:
            row_values = []
            for col in columns:
                value = record.get(col)
                if col in ['date', 'datetime', 'timestamp', 'write_timestamp'] and isinstance(value, str):
                    # Convert string timestamps to datetime objects for asyncpg
                    try:
                        if value:
                            row_values.append(date_parser.parse(value))
                        else:
                            row_values.append(None)
                    except Exception as e:
                        self.logger.warning(f"Failed to parse date '{value}' for column '{col}': {e}")
                        row_values.append(None)
                else:
                    row_values.append(value)
            values.append(tuple(row_values))
        
        # Execute bulk insert
        try:
            await conn.executemany(insert_sql, values)
            self.logger.info(f"Inserted {len(records)} {interval} records for {records[0]['ticker']}")
        except Exception as e:
            self.logger.error(f"Error inserting {interval} data: {e}")
            raise

    async def get_stock_data(
        self,
        ticker: str,
        start_date: str | None = None,
        end_date: str | None = None,
        interval: str = "daily",
        conn=None,
    ) -> pd.DataFrame:
        """Retrieve aggregated (daily/hourly) stock data from QuestDB."""
        if conn is None:
            async with self.get_connection() as conn:
                return await self._get_stock_data(conn, ticker, start_date, end_date, interval)
        else:
            return await self._get_stock_data(conn, ticker, start_date, end_date, interval)

    async def _get_stock_data(
        self,
        conn: asyncpg.Connection,
        ticker: str,
        start_date: str | None = None,
        end_date: str | None = None,
        interval: str = "daily",
    ) -> pd.DataFrame:
        """Internal method to retrieve stock data from QuestDB."""
        table = 'daily_prices' if interval == 'daily' else 'hourly_prices'
        date_col = 'date' if interval == 'daily' else 'datetime'

        query = f"SELECT * FROM {table} WHERE ticker = $1"
        params = [ticker]

        if start_date:
            query += f" AND {date_col} >= ${len(params) + 1}"
            # Convert string date to datetime object for asyncpg
            if isinstance(start_date, str):
                params.append(date_parser.parse(start_date))
            else:
                params.append(start_date)
        if end_date:
            query += f" AND {date_col} <= ${len(params) + 1}"
            # Convert string date to datetime object for asyncpg
            if isinstance(end_date, str):
                params.append(date_parser.parse(end_date))
            else:
                params.append(end_date)

        query += f" ORDER BY {date_col}"

        try:
            rows = await conn.fetch(query, *params)
            if rows:
                # Convert asyncpg records to DataFrame with proper column names
                columns = list(rows[0].keys()) if rows else []
                data = [dict(row) for row in rows]
                df = pd.DataFrame(data, columns=columns)
                
                # Convert date column to datetime and set as index
                if date_col in df.columns:
                    df[date_col] = pd.to_datetime(df[date_col])
                    df.set_index(date_col, inplace=True)
                else:
                    self.logger.warning(f"Expected date column '{date_col}' not found in results. Available columns: {list(df.columns)}")
                return df
            else:
                return pd.DataFrame()
        except Exception as e:
            self.logger.error(f"Error retrieving {interval} data for {ticker}: {e}")
            return pd.DataFrame()

    async def save_realtime_data(self, df: pd.DataFrame, ticker: str, data_type: str = "quote", on_duplicate: str = "ignore") -> None:
        """Save realtime (tick) stock data to QuestDB."""
        async with self.get_connection() as conn:
            df_copy = df.copy()
            if df_copy.empty:
                return
                
            df_copy.reset_index(inplace=True)
            df_copy.columns = [col.lower() for col in df_copy.columns]
            df_copy['ticker'] = ticker
            df_copy['type'] = data_type

            if 'timestamp' not in df_copy.columns and 'index' in df_copy.columns:
                df_copy.rename(columns={'index': 'timestamp'}, inplace=True)

            required_cols = ['ticker', 'timestamp', 'type', 'price', 'size']
            df_copy = df_copy[[col for col in required_cols if col in df_copy.columns]]

            if 'timestamp' not in df_copy.columns or 'price' not in df_copy.columns or 'size' not in df_copy.columns:
                self.logger.warning(f"Missing required columns for realtime data of {ticker}")
                return

            df_copy['timestamp'] = pd.to_datetime(df_copy['timestamp'])
            df_copy = df_copy.dropna(subset=['timestamp'])
            
            if df_copy.empty:
                return

            # Add write_timestamp
            current_time = datetime.now(timezone.utc)
            df_copy['write_timestamp'] = current_time

            # Convert to QuestDB format
            df_copy['timestamp'] = df_copy['timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S.%f')
            df_copy['write_timestamp'] = df_copy['write_timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S.%f')

            # Prepare records for bulk insert
            records = []
            for _, row in df_copy.iterrows():
                record = {
                    'ticker': row['ticker'],
                    'timestamp': row['timestamp'],
                    'type': row['type'],
                    'price': float(row.get('price', 0.0)),
                    'size': int(row.get('size', 0)),
                    'ask_price': float(row.get('ask_price', 0.0)) if 'ask_price' in row else None,
                    'ask_size': int(row.get('ask_size', 0)) if 'ask_size' in row else None,
                    'write_timestamp': row['write_timestamp']
                }
                records.append(record)

            if records:
                await self._bulk_insert_realtime_data(conn, records)

    async def _bulk_insert_realtime_data(self, conn: asyncpg.Connection, records: List[Dict]):
        """Bulk insert realtime data using QuestDB's optimized insert."""
        if not records:
            return
            
        # Build the INSERT statement
        columns = ['ticker', 'timestamp', 'type', 'price', 'size', 'ask_price', 'ask_size', 'write_timestamp']
        placeholders = [f'${i+1}' for i in range(len(columns))]
        
        insert_sql = f"""
        INSERT INTO realtime_data ({', '.join(columns)})
        VALUES ({', '.join(placeholders)})
        """
        
        # Prepare the data for insertion
        values = []
        for record in records:
            row_values = []
            for col in columns:
                value = record.get(col)
                if col in ['timestamp', 'write_timestamp'] and isinstance(value, str):
                    row_values.append(value)
                else:
                    row_values.append(value)
            values.append(tuple(row_values))
        
        # Execute bulk insert
        try:
            await conn.executemany(insert_sql, values)
            self.logger.info(f"Inserted {len(records)} realtime records for {records[0]['ticker']}")
        except Exception as e:
            self.logger.error(f"Error inserting realtime data: {e}")
            raise

    async def get_realtime_data(self, ticker: str, start_datetime: str | None = None, end_datetime: str | None = None, data_type: str = "quote") -> pd.DataFrame:
        """Retrieve realtime (tick) stock data from QuestDB."""
        async with self.get_connection() as conn:
            query = "SELECT * FROM realtime_data WHERE ticker = $1 AND type = $2"
            params = [ticker, data_type]

            if start_datetime:
                query += " AND timestamp >= $3"
                # Convert string datetime to datetime object for asyncpg
                if isinstance(start_datetime, str):
                    params.append(date_parser.parse(start_datetime))
                else:
                    params.append(start_datetime)
            if end_datetime:
                query += " AND timestamp <= $4"
                # Convert string datetime to datetime object for asyncpg
                if isinstance(end_datetime, str):
                    params.append(date_parser.parse(end_datetime))
                else:
                    params.append(end_datetime)

            query += " ORDER BY timestamp"

            try:
                rows = await conn.fetch(query, *params)
                if rows:
                    # Convert asyncpg records to DataFrame with proper column names
                    columns = list(rows[0].keys()) if rows else []
                    data = [dict(row) for row in rows]
                    df = pd.DataFrame(data, columns=columns)
                    
                    # Convert timestamp columns to datetime
                    if 'timestamp' in df.columns:
                        df['timestamp'] = pd.to_datetime(df['timestamp'])
                        df.set_index('timestamp', inplace=True)
                    if 'write_timestamp' in df.columns:
                        df['write_timestamp'] = pd.to_datetime(df['write_timestamp'])
                    return df
                else:
                    return pd.DataFrame()
            except Exception as e:
                self.logger.error(f"Error retrieving realtime data for {ticker}: {e}")
                return pd.DataFrame()

    async def get_latest_price(self, ticker: str) -> float | None:
        """Get the most recent price for a ticker from QuestDB."""
        async with self.get_connection() as conn:
            try:
                # Try realtime_data first
                row = await conn.fetchrow(
                    "SELECT price FROM realtime_data WHERE ticker = $1 AND type = 'quote' ORDER BY timestamp DESC LIMIT 1",
                    ticker
                )
                if row:
                    return row['price']

                # Try hourly_prices
                row = await conn.fetchrow(
                    "SELECT close FROM hourly_prices WHERE ticker = $1 ORDER BY datetime DESC LIMIT 1",
                    ticker
                )
                if row:
                    return row['close']

                # Try daily_prices
                row = await conn.fetchrow(
                    "SELECT close FROM daily_prices WHERE ticker = $1 ORDER BY date DESC LIMIT 1",
                    ticker
                )
                if row:
                    return row['close']

            except Exception as e:
                self.logger.error(f"Error getting latest price for {ticker}: {e}")

        return None

    async def get_latest_prices(self, tickers: List[str]) -> Dict[str, float | None]:
        """Get the most recent prices for multiple tickers from QuestDB."""
        result = {}
        
        async with self.get_connection() as conn:
            for ticker in tickers:
                try:
                    # Try realtime_data first
                    row = await conn.fetchrow(
                        "SELECT price FROM realtime_data WHERE ticker = $1 AND type = 'quote' ORDER BY timestamp DESC LIMIT 1",
                        ticker
                    )
                    if row:
                        result[ticker] = row['price']
                        continue

                    # Try hourly_prices
                    row = await conn.fetchrow(
                        "SELECT close FROM hourly_prices WHERE ticker = $1 ORDER BY datetime DESC LIMIT 1",
                        ticker
                    )
                    if row:
                        result[ticker] = row['close']
                        continue

                    # Try daily_prices
                    row = await conn.fetchrow(
                        "SELECT close FROM daily_prices WHERE ticker = $1 ORDER BY date DESC LIMIT 1",
                        ticker
                    )
                    if row:
                        result[ticker] = row['close']
                    else:
                        result[ticker] = None

                except Exception as e:
                    self.logger.error(f"Error getting latest price for {ticker}: {e}")
                    result[ticker] = None
        
        return result

    async def get_previous_close_prices(self, tickers: List[str]) -> Dict[str, float | None]:
        """Get the most recent daily close prices for multiple tickers, excluding today's data."""
        result = {}
        
        async with self.get_connection() as conn:
            # Get today's date in EST timezone (market timezone)
            today = datetime.now(pytz.timezone('US/Eastern')).date()
            
            for ticker in tickers:
                try:
                    # Get the most recent close price that is NOT from today
                    row = await conn.fetchrow(
                        "SELECT close FROM daily_prices WHERE ticker = $1 AND date < $2 ORDER BY date DESC LIMIT 1",
                        ticker, today
                    )
                    if row:
                        result[ticker] = row['close']
                    else:
                        # Fallback: if no previous day data, get the most recent available
                        row = await conn.fetchrow(
                            "SELECT close FROM daily_prices WHERE ticker = $1 ORDER BY date DESC LIMIT 1",
                            ticker
                        )
                        if row:
                            result[ticker] = row['close']
                        else:
                            result[ticker] = None
                except Exception as e:
                    self.logger.error(f"Error getting previous close for {ticker}: {e}")
                    result[ticker] = None
        
        return result

    async def get_today_opening_prices(self, tickers: List[str]) -> Dict[str, float | None]:
        """Get today's opening prices for multiple tickers from QuestDB."""
        result = {}
        
        async with self.get_connection() as conn:
            # Get today's date in EST timezone (market timezone)
            today = datetime.now(pytz.timezone('US/Eastern')).date()
            
            for ticker in tickers:
                try:
                    # Get today's opening price
                    row = await conn.fetchrow(
                        "SELECT open FROM daily_prices WHERE ticker = $1 AND date = $2 ORDER BY date DESC LIMIT 1",
                        ticker, today
                    )
                    if row:
                        result[ticker] = row['open']
                    else:
                        result[ticker] = None
                except Exception as e:
                    self.logger.error(f"Error getting today's opening for {ticker}: {e}")
                    result[ticker] = None
        
        return result

    async def execute_select_sql(self, sql_query: str, params: tuple = ()) -> pd.DataFrame:
        """Execute a direct SELECT SQL query on QuestDB and return results as a DataFrame."""
        async with self.get_connection() as conn:
            try:
                # Convert tuple params to list for asyncpg
                param_list = list(params)
                rows = await conn.fetch(sql_query, *param_list)
                if rows:
                    df = pd.DataFrame(rows)
                    return df
                else:
                    return pd.DataFrame()
            except Exception as e:
                self.logger.error(f"Error executing SELECT query: {e}")
                return pd.DataFrame()

    async def execute_raw_sql(self, sql_query: str, params: tuple = ()) -> list[dict[str, Any]]:
        """Execute a raw SQL query on QuestDB."""
        async with self.get_connection() as conn:
            try:
                # Convert tuple params to list for asyncpg
                param_list = list(params)
                
                if sql_query.strip().upper().startswith('SELECT'):
                    # SELECT query - fetch results
                    rows = await conn.fetch(sql_query, *param_list)
                    results = []
                    for row in rows:
                        record = {}
                        for key, value in row.items():
                            if isinstance(value, bytes):
                                record[key] = value.hex()  # Convert bytes to hex string
                            else:
                                record[key] = value
                        results.append(record)
                    return results
                else:
                    # Non-SELECT query - execute without returning data
                    await conn.execute(sql_query, *param_list)
                    return []
                    
            except Exception as e:
                self.logger.error(f"Error executing raw SQL query: {e}")
                raise

    async def _get_historical_data(
        self,
        ticker: str,
        min_date_val: pd.Timestamp,
        max_date_val: pd.Timestamp,
        df_copy: pd.DataFrame,
        interval: str,
        ma_periods: List[int] = None,
        ema_periods: List[int] = None,
        date_col: str = "datetime",
        conn=None,
    ) -> pd.DataFrame:
        """Get historical data to ensure we have enough data for MA/EMA calculations."""
        # Get historical data to ensure we have enough data for MA/EMA calculations
        max_period = max(
            max(ma_periods) if ma_periods else [0],
            max(ema_periods) if ema_periods else [0],
        )
        if max_period > 0:
            # Get historical data from before the new data range
            historical_start = (
                min_date_val - pd.Timedelta(days=max_period * 2)
            ).strftime("%Y-%m-%d")
            historical_end = (min_date_val - pd.Timedelta(days=1)).strftime("%Y-%m-%d")

            historical_df = await self.get_stock_data(
                ticker, historical_start, historical_end, interval, conn
            )

            if not historical_df.empty:
                # Combine historical data with new data for calculation
                historical_df.reset_index(inplace=True)
                historical_df.columns = [col.lower() for col in historical_df.columns]

                # Ensure consistent timezone handling
                if isinstance(historical_df[date_col].dtype, pd.DatetimeTZDtype):
                    historical_df[date_col] = historical_df[date_col].dt.tz_localize(None)
                if isinstance(df_copy[date_col].dtype, pd.DatetimeTZDtype):
                    df_copy[date_col] = df_copy[date_col].dt.tz_localize(None)

                # Ensure both DataFrames have the same columns and data types
                common_cols = list(set(historical_df.columns) & set(df_copy.columns))
                
                # Convert both DataFrames to have the same dtypes for common columns
                for col in common_cols:
                    if col in historical_df.columns and col in df_copy.columns:
                        # Use the dtype from df_copy as the target
                        target_dtype = df_copy[col].dtype
                        historical_df[col] = historical_df[col].astype(target_dtype)
                
                # Select only common columns
                historical_df = historical_df[common_cols].copy()
                df_copy = df_copy[common_cols].copy()

                # Ensure both DataFrames are not empty
                if not historical_df.empty and not df_copy.empty:
                    try:
                        # Combine datasets
                        combined_df = pd.concat([historical_df, df_copy], ignore_index=True)
                        combined_df = combined_df.sort_values(date_col)
                        combined_df = combined_df.drop_duplicates(
                            subset=[date_col], keep="last"
                        )
                    except Exception as e:
                        self.logger.warning(f"Error concatenating DataFrames for {ticker}: {str(e)}")
                        # Fall back to using just the new data
                        combined_df = df_copy.copy()
                else:
                    combined_df = df_copy.copy()
            else:
                combined_df = df_copy.copy()
        else:
            combined_df = df_copy.copy()

        # Prepare data for MA/EMA calculation
        records_for_calculation = []
        for _, row in combined_df.iterrows():
            # Check if the date is NaT before calling strftime
            if pd.isna(row[date_col]):
                self.logger.warning(f"Warning: Skipping row with NaT date for {ticker}")
                continue
                
            record = {
                "date": row[date_col].strftime("%Y-%m-%d"),
                "price": row.get("close", 0),
            }
            records_for_calculation.append(record)

        # Calculate moving averages
        for period in ma_periods:
            records_for_calculation = self._calculate_moving_average(
                ticker, records_for_calculation, period, "price"
            )

        # Calculate exponential moving averages
        for period in ema_periods:
            records_for_calculation = self._calculate_exponential_moving_average(
                ticker, records_for_calculation, period, "price"
            )

        # Create a new DataFrame for the results
        result_df = df_copy.copy()
        
        # Add MA and EMA values back to result_df for the new data only
        for i, row in result_df.iterrows():
            # Check if the date is NaT before calling strftime
            if pd.isna(row[date_col]):
                self.logger.warning(f"Warning: Skipping row with NaT date for {ticker} in result processing")
                continue
                
            row_date = row[date_col].strftime("%Y-%m-%d")
            for calc_record in records_for_calculation:
                if calc_record["date"] == row_date:
                    # Add MA values
                    for period in ma_periods:
                        ma_key = f"ma_{period}"
                        if ma_key in calc_record:
                            result_df.loc[i, ma_key] = calc_record[ma_key]

                    # Add EMA values
                    for period in ema_periods:
                        ema_key = f"ema_{period}"
                        if ema_key in calc_record:
                            result_df.loc[i, ema_key] = calc_record[ema_key]
                    break
        return result_df

    def _calculate_moving_average(self, ticker: str, records: List[Dict], period: int, price_col: str) -> List[Dict]:
        """Calculate moving average for a given period."""
        if len(records) < period:
            return records
            
        for i in range(period - 1, len(records)):
            prices = [records[j][price_col] for j in range(i - period + 1, i + 1)]
            records[i][f"ma_{period}"] = sum(prices) / len(prices)
            
        return records

    def _calculate_exponential_moving_average(self, ticker: str, records: List[Dict], period: int, price_col: str) -> List[Dict]:
        """Calculate exponential moving average for a given period."""
        if len(records) < period:
            return records
            
        # Calculate SMA for the first period values
        sma = sum(records[period - 1][price_col] for i in range(period)) / period
        records[period - 1][f"ema_{period}"] = sma
        
        # Calculate EMA for remaining values
        multiplier = 2 / (period + 1)
        for i in range(period, len(records)):
            ema = (records[i][price_col] * multiplier) + (records[i - 1][f"ema_{period}"] * (1 - multiplier))
            records[i][f"ema_{period}"] = ema
            
        return records

    async def close_session(self):
        """Close any active sessions and connection pool (compatible with base class interface)."""
        await self.close()
    
    async def close(self):
        """Close the connection pool."""
        if self._connection_pool:
            await self._connection_pool.close()
            self._connection_pool = None
