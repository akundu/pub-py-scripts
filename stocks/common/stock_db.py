import sqlite3
import pandas as pd
from datetime import datetime
import os
from abc import ABCMeta, abstractmethod
import duckdb
import asyncio
import aiohttp
import json
import base64
from typing import Any, List
from .common_strategies import (
    calculate_moving_average,
    calculate_exponential_moving_average,
)

DEFAULT_DATA_DIR = './data'
def get_default_db_path(db_type: str) -> str:   
    if db_type.lower() == "remote":
        return "localhost:8080"
    return os.path.join(
        DEFAULT_DATA_DIR, f"stock_data.{"db" if db_type == "sqlite" else db_type}"
    )

class StockDBBase(metaclass=ABCMeta):
    """
    Abstract base class for stock database operations.
    """
    def __init__(self, db_config: str):
        self.db_config = db_config

    @abstractmethod
    def _init_db(self) -> None:
        """Initialize the database with required tables. (No-op for client)"""
        pass

    @abstractmethod
    async def save_stock_data(
        self,
        df: pd.DataFrame,
        ticker: str,
        interval: str = "daily",
        ma_periods: List[int] = None,
        ema_periods: List[int] = None,
    ) -> None:
        """Save aggregated (daily/hourly) stock data to the database."""
        pass

    @abstractmethod
    async def get_stock_data(
        self,
        ticker: str,
        start_date: str | None = None,
        end_date: str | None = None,
        interval: str = "daily",
        conn=None,
    ) -> pd.DataFrame:
        """Retrieve aggregated (daily/hourly) stock data from the database."""
        pass

    @abstractmethod
    async def save_realtime_data(self, df: pd.DataFrame, ticker: str, data_type: str = "quote") -> None:
        """Save realtime (tick/quote/trade) stock data to the database."""
        pass

    @abstractmethod
    async def get_realtime_data(self, ticker: str, start_datetime: str | None = None, end_datetime: str | None = None, data_type: str = "quote") -> pd.DataFrame:
        """Retrieve realtime (tick/quote/trade) stock data from the database."""
        pass

    @abstractmethod
    async def get_latest_price(self, ticker: str) -> float | None:
        """Get the most recent price for a ticker from any available data (realtime, hourly, daily)."""
        pass

    @abstractmethod
    async def execute_select_sql(self, sql_query: str, params: tuple = ()) -> pd.DataFrame:
        """Execute a direct SELECT SQL query and return results as a DataFrame."""
        pass

    @abstractmethod
    async def execute_raw_sql(self, sql_query: str, params: tuple = ()) -> list[dict[str, Any]]:
        """Execute a raw SQL query (INSERT, UPDATE, DELETE, DDL potentially with RETURNING) 
           and return resulting rows if any, with binary data Base64 encoded."""
        pass

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

                # Combine datasets
                combined_df = pd.concat([historical_df, df_copy], ignore_index=True)
                combined_df = combined_df.sort_values(date_col)
                combined_df = combined_df.drop_duplicates(
                    subset=[date_col], keep="last"
                )
            else:
                combined_df = df_copy.copy()

            # Prepare data for MA/EMA calculation
            records_for_calculation = []
            for _, row in combined_df.iterrows():
                record = {
                    "date": row[date_col].strftime("%Y-%m-%d"),
                    "price": row.get("close", 0),
                }
                records_for_calculation.append(record)

            # Calculate moving averages
            for period in ma_periods:
                records_for_calculation = calculate_moving_average(
                    ticker, records_for_calculation, period, "price"
                )

            # Calculate exponential moving averages
            for period in ema_periods:
                records_for_calculation = calculate_exponential_moving_average(
                    ticker, records_for_calculation, period, "price"
                )

            # Add MA and EMA values back to df_copy for the new data only
            for i, row in df_copy.iterrows():
                row_date = row[date_col].strftime("%Y-%m-%d")
                for calc_record in records_for_calculation:
                    if calc_record["date"] == row_date:
                        # Add MA values
                        for period in ma_periods:
                            ma_key = f"ma_{period}"
                            if ma_key in calc_record:
                                df_copy.at[i, ma_key] = calc_record[ma_key]

                        # Add EMA values
                        for period in ema_periods:
                            ema_key = f"ema_{period}"
                            if ema_key in calc_record:
                                df_copy.at[i, ema_key] = calc_record[ema_key]
                        break
        return df_copy


class StockDBSQLite(StockDBBase):
    """
    A class to manage stock data storage and retrieval in an SQLite database.
    """
    def __init__(self, db_path: str = get_default_db_path("db")) -> None:
        super().__init__(db_path)
        self.db_path = db_path
        self._init_db()

    def _init_db(self) -> None:
        """Initialize the SQLite database with required tables if they don't exist."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS daily_prices (
                    ticker TEXT,
                    date DATE,
                    open REAL,
                    high REAL,
                    low REAL,
                    close REAL,
                    volume INTEGER,
                    ma_10 REAL,
                    ma_50 REAL,
                    ma_100 REAL,
                    ma_200 REAL,
                    ema_8 REAL,
                    ema_21 REAL,
                    ema_34 REAL,
                    ema_55 REAL,
                    ema_89 REAL,
                    PRIMARY KEY (ticker, date)
                )
            """
            )

            cursor.execute('''
                CREATE TABLE IF NOT EXISTS hourly_prices (
                    ticker TEXT,
                    datetime DATETIME, -- Stored as YYYY-MM-DD HH:MM:SS
                    open REAL,
                    high REAL,
                    low REAL,
                    close REAL,
                    volume INTEGER,
                    PRIMARY KEY (ticker, datetime)
                )
            ''')

            cursor.execute('''
                CREATE TABLE IF NOT EXISTS realtime_data (
                    ticker TEXT,
                    timestamp DATETIME, -- Stored as YYYY-MM-DD HH:MM:SS.ffffff
                    type TEXT, -- 'quote' or 'trade'
                    price REAL, 
                    size INTEGER,
                    ask_price REAL, -- Nullable, for quotes
                    ask_size INTEGER, -- Nullable, for quotes
                    PRIMARY KEY (ticker, timestamp, type)
                )
            ''')

            # Add MA and EMA columns to existing daily_prices table if they don't exist
            self._add_ma_ema_columns_if_needed(cursor)
            conn.commit()

    def _add_ma_ema_columns_if_needed(self, cursor) -> None:
        """Add MA and EMA columns to daily_prices table if they don't exist."""
        # Get existing columns
        cursor.execute("PRAGMA table_info(daily_prices)")
        existing_columns = {row[1] for row in cursor.fetchall()}

        # Default MA and EMA columns to add
        ma_columns = ["ma_10", "ma_50", "ma_100", "ma_200"]
        ema_columns = ["ema_8", "ema_21", "ema_34", "ema_55", "ema_89"]

        # Add missing MA columns
        for col in ma_columns:
            if col not in existing_columns:
                cursor.execute(f"ALTER TABLE daily_prices ADD COLUMN {col} REAL")

        # Add missing EMA columns
        for col in ema_columns:
            if col not in existing_columns:
                cursor.execute(f"ALTER TABLE daily_prices ADD COLUMN {col} REAL")

    async def save_stock_data(
        self,
        df: pd.DataFrame,
        ticker: str,
        interval: str = "daily",
        ma_periods: List[int] = None,
        ema_periods: List[int] = None,
    ) -> None:
        """Save aggregated (daily/hourly) stock data to the SQLite database."""
        # Set default periods
        if ma_periods is None:
            ma_periods = [10, 50, 100, 200]
        if ema_periods is None:
            ema_periods = [8, 21, 34, 55, 89]

        with sqlite3.connect(self.db_path) as conn:
            df_copy = df.copy()
            if df_copy.empty:
                # print(f"Empty DataFrame provided for {ticker} ({interval}). Skipping DB save.")
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
                print(f"Warning: Date column '{date_col}' not found in DataFrame for {ticker} ({interval}). Skipping DB save.")
                return

            df_copy[date_col] = pd.to_datetime(df_copy[date_col])
            min_date_val = df_copy[date_col].min()
            max_date_val = df_copy[date_col].max()

            if pd.isna(min_date_val) or pd.isna(max_date_val):
                print(f"Warning: Min/max date is NaT for {ticker} ({interval}). Skipping DB save for this batch.")
                return

            # Calculate MA and EMA for daily data only
            if interval == "daily":
                df_copy = await self._get_historical_data(
                    ticker,
                    min_date_val,
                    max_date_val,
                    df_copy,
                    interval,
                    ma_periods,
                    ema_periods,
                    date_col,
                    conn,
                )

            date_format_str = '%Y-%m-%d' if interval == 'daily' else '%Y-%m-%d %H:%M:%S'
            min_date_str = min_date_val.strftime(date_format_str)
            max_date_str = max_date_val.strftime(date_format_str)

            cursor = conn.cursor()
            cursor.execute(f'''
                DELETE FROM {table} 
                WHERE ticker = ? 
                AND {date_col} BETWEEN ? AND ?
            ''', (ticker, min_date_str, max_date_str))

            df_copy[date_col] = df_copy[date_col].dt.strftime(date_format_str)
            df_copy.to_sql(table, conn, if_exists='append', index=False)

    async def _get_stock_data(
        self,
        conn,
        ticker: str,
        start_date: str | None = None,
        end_date: str | None = None,
        interval: str = "daily",
    ) -> pd.DataFrame:
        if conn is None:
            return None
        else:
            table = 'daily_prices' if interval == 'daily' else 'hourly_prices'
            date_col_name = 'date' if interval == 'daily' else 'datetime' # Name of column in DB

            query = f"SELECT * FROM {table} WHERE ticker = ?"
            params: list[str | None] = [ticker]

            if start_date:
                query += f" AND {date_col_name} >= ?"
                params.append(start_date)
            if end_date:
                query += f" AND {date_col_name} <= ?"
                params.append(end_date)

            query += f" ORDER BY {date_col_name}"

            df = pd.read_sql_query(query, conn, params=params) # type: ignore

        if not df.empty:
            df[date_col_name] = pd.to_datetime(df[date_col_name], errors='coerce')
            df.set_index(date_col_name, inplace=True)
            df = df[df.index.notna()]
        return df

    async def get_stock_data(
        self,
        ticker: str,
        start_date: str | None = None,
        end_date: str | None = None,
        interval: str = "daily",
        conn=None,
    ) -> pd.DataFrame:
        df = None
        if conn is None:
            with sqlite3.connect(self.db_path) as conn:
                df = await self._get_stock_data(
                    conn, ticker, start_date, end_date, interval
                )
        else:
            df = await self._get_stock_data(
                conn, ticker, start_date, end_date, interval
            )
        return df

    async def save_realtime_data(self, df: pd.DataFrame, ticker: str, data_type: str = "quote") -> None:
        """Save realtime (tick) stock data to the SQLite database."""
        with sqlite3.connect(self.db_path) as conn:
            df_copy = df.copy()
            if df_copy.empty:
                # print(f"Empty DataFrame provided for realtime data of {ticker}. Skipping DB save.")
                return
            df_copy.reset_index(inplace=True)
            df_copy.columns = [col.lower() for col in df_copy.columns]
            df_copy['ticker'] = ticker
            df_copy['type'] = data_type

            # Standardize column names; expecting 'timestamp', 'price', 'volume' from input df index or columns
            # If index was named, it's now a column. If unnamed, it was 'index'.
            if 'timestamp' not in df_copy.columns and 'index' in df_copy.columns:
                df_copy.rename(columns={'index': 'timestamp'}, inplace=True)

            required_cols = ['ticker', 'timestamp', 'type', 'price', 'size']
            df_copy = df_copy[[col for col in required_cols if col in df_copy.columns]]

            if 'timestamp' not in df_copy.columns or 'price' not in df_copy.columns or 'size' not in df_copy.columns:
                print(f"Warning: Missing required columns (timestamp, price, size) for realtime data of {ticker}. Skipping DB save.")
                return

            df_copy['timestamp'] = pd.to_datetime(df_copy['timestamp'])
            min_ts_val = df_copy['timestamp'].min()
            max_ts_val = df_copy['timestamp'].max()

            if pd.isna(min_ts_val) or pd.isna(max_ts_val):
                print(f"Warning: Min/max timestamp is NaT for realtime data of {ticker}. Skipping DB save.")
                return

            # SQLite DATETIME stores up to milliseconds if string is formatted that way
            ts_format_str = '%Y-%m-%d %H:%M:%S.%f' 
            min_ts_str = min_ts_val.strftime(ts_format_str)
            max_ts_str = max_ts_val.strftime(ts_format_str)

            cursor = conn.cursor()
            cursor.execute('''
                DELETE FROM realtime_data 
                WHERE ticker = ? 
                AND type = ?
                AND timestamp BETWEEN ? AND ?
            ''', (ticker, data_type, min_ts_str, max_ts_str))

            df_copy['timestamp'] = df_copy['timestamp'].dt.strftime(ts_format_str)
            df_copy.to_sql('realtime_data', conn, if_exists='append', index=False)

    async def get_realtime_data(self, ticker: str, start_datetime: str | None = None, end_datetime: str | None = None, data_type: str = "quote") -> pd.DataFrame:
        """Retrieve realtime (tick) stock data from the SQLite database."""
        with sqlite3.connect(self.db_path) as conn:
            query = "SELECT * FROM realtime_data WHERE ticker = ? AND type = ?"
            params: list[str | None] = [ticker, data_type]

            if start_datetime:
                query += " AND timestamp >= ?"
                params.append(start_datetime) # Expects 'YYYY-MM-DD HH:MM:SS.ffffff'
            if end_datetime:
                query += " AND timestamp <= ?"
                params.append(end_datetime) # Expects 'YYYY-MM-DD HH:MM:SS.ffffff'

            query += " ORDER BY timestamp"
            df = pd.read_sql_query(query, conn, params=params) # type: ignore

        if not df.empty:
            df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
            df.set_index('timestamp', inplace=True)
            df = df[df.index.notna()]
        return df

    async def get_latest_price(self, ticker: str) -> float | None:
        """Get the most recent price for a ticker (realtime -> hourly -> daily)."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            latest_price = None

            # 1. Try realtime_data
            try:
                cursor.execute("SELECT price FROM realtime_data WHERE ticker = ? AND type = 'quote' ORDER BY timestamp DESC LIMIT 1", (ticker,))
                row = cursor.fetchone()
                if row: latest_price = row[0]
            except sqlite3.Error as e:
                print(f"SQLite error (realtime_data for {ticker}): {e}")

            # 2. Try hourly_prices if no realtime found
            if latest_price is None:
                try:
                    cursor.execute("SELECT close FROM hourly_prices WHERE ticker = ? ORDER BY datetime DESC LIMIT 1", (ticker,))
                    row = cursor.fetchone()
                    if row: latest_price = row[0]
                except sqlite3.Error as e:
                    print(f"SQLite error (hourly_prices for {ticker}): {e}")

            # 3. Try daily_prices if no hourly found
            if latest_price is None:
                try:
                    cursor.execute("SELECT close FROM daily_prices WHERE ticker = ? ORDER BY date DESC LIMIT 1", (ticker,))
                    row = cursor.fetchone()
                    if row: latest_price = row[0]
                except sqlite3.Error as e:
                    print(f"SQLite error (daily_prices for {ticker}): {e}")

        return latest_price

    async def execute_select_sql(self, sql_query: str, params: tuple = ()) -> pd.DataFrame:
        """Execute a direct SELECT SQL query on SQLite and return results as a DataFrame."""        
        def _run_sync():
            with sqlite3.connect(self.db_path) as conn:
                df = pd.read_sql_query(sql_query, conn, params=params)
            return df
        return await asyncio.get_event_loop().run_in_executor(None, _run_sync)

    async def execute_raw_sql(self, sql_query: str, params: tuple = ()) -> list[dict[str, Any]]:
        """Execute a raw SQL query on SQLite. If it returns data (e.g., RETURNING clause),
           that data is returned with binary fields Base64 encoded. Otherwise, an empty list is returned."""
        def _run_sync() -> list[dict[str, Any]]:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute(sql_query, params)

                results = []
                if cursor.description: # Check if the query was meant to return data
                    column_names = [desc[0] for desc in cursor.description]
                    rows = cursor.fetchall()
                    for row in rows:
                        record = {}
                        for i, value in enumerate(row):
                            if isinstance(value, bytes):
                                record[column_names[i]] = base64.b64encode(value).decode('utf-8')
                            else:
                                record[column_names[i]] = value
                        results.append(record)
                conn.commit() # Commit changes regardless of RETURNING clause
                # Note: cursor.rowcount would give affected rows for DML without RETURNING.
                # Current design returns data rows if available, otherwise empty list for this path.
                return results
        return await asyncio.get_event_loop().run_in_executor(None, _run_sync)

class StockDBDuckDB(StockDBBase):
    """
    A class to manage stock data storage and retrieval in a DuckDB database.
    """
    def __init__(self, db_path: str = get_default_db_path("duckdb")) -> None:
        super().__init__(db_path)
        self.db_path = db_path
        self._init_db()

    def _init_db(self) -> None:
        """Initialize the DuckDB database with required tables if they don't exist."""
        with duckdb.connect(database=self.db_path, read_only=False) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS daily_prices (
                    ticker VARCHAR,
                    date DATE,
                    open DOUBLE,
                    high DOUBLE,
                    low DOUBLE,
                    close DOUBLE,
                    volume BIGINT,
                    ma_10 DOUBLE,
                    ma_50 DOUBLE,
                    ma_100 DOUBLE,
                    ma_200 DOUBLE,
                    ema_8 DOUBLE,
                    ema_21 DOUBLE,
                    ema_34 DOUBLE,
                    ema_55 DOUBLE,
                    ema_89 DOUBLE,
                    PRIMARY KEY (ticker, date)
                )
            """
            )

            conn.execute('''
                CREATE TABLE IF NOT EXISTS hourly_prices (
                    ticker VARCHAR,
                    datetime TIMESTAMP,
                    open DOUBLE,
                    high DOUBLE,
                    low DOUBLE,
                    close DOUBLE,
                    volume BIGINT,
                    PRIMARY KEY (ticker, datetime)
                )
            ''')
            conn.execute('''
                CREATE TABLE IF NOT EXISTS realtime_data (
                    ticker VARCHAR, timestamp TIMESTAMP, type VARCHAR, price DOUBLE, size BIGINT,
                    ask_price DOUBLE, ask_size BIGINT,
                    PRIMARY KEY (ticker, timestamp, type)
                )
            ''')

            # Add MA and EMA columns to existing daily_prices table if they don't exist
            self._add_ma_ema_columns_if_needed_duckdb(conn)

    def _add_ma_ema_columns_if_needed_duckdb(self, conn) -> None:
        """Add MA and EMA columns to daily_prices table if they don't exist in DuckDB."""
        try:
            # Get existing columns
            result = conn.execute("DESCRIBE daily_prices").fetchall()
            existing_columns = {row[0] for row in result}

            # Default MA and EMA columns to add
            ma_columns = ["ma_10", "ma_50", "ma_100", "ma_200"]
            ema_columns = ["ema_8", "ema_21", "ema_34", "ema_55", "ema_89"]

            # Add missing MA columns
            for col in ma_columns:
                if col not in existing_columns:
                    conn.execute(f"ALTER TABLE daily_prices ADD COLUMN {col} DOUBLE")

            # Add missing EMA columns
            for col in ema_columns:
                if col not in existing_columns:
                    conn.execute(f"ALTER TABLE daily_prices ADD COLUMN {col} DOUBLE")
        except Exception as e:
            # Table might not exist yet, which is fine
            pass

    async def save_stock_data(
        self,
        df: pd.DataFrame,
        ticker: str,
        interval: str = "daily",
        ma_periods: List[int] = None,
        ema_periods: List[int] = None,
    ) -> None:
        """Save aggregated (daily/hourly) stock data to the DuckDB database."""
        # Set default periods
        if ma_periods is None:
            ma_periods = [10, 50, 100, 200]
        if ema_periods is None:
            ema_periods = [8, 21, 34, 55, 89]

        with duckdb.connect(database=self.db_path, read_only=False) as conn:
            df_copy = df.copy()
            if df_copy.empty:
                # print(f"Empty DataFrame provided for {ticker} ({interval}). Skipping DB save.")
                return
            df_copy.reset_index(inplace=True)
            df_copy.columns = [col.lower() for col in df_copy.columns]
            df_copy['ticker'] = ticker

            date_col = 'date' if interval == 'daily' else 'datetime'
            table = 'daily_prices' if interval == 'daily' else 'hourly_prices'

            if 'index' in df_copy.columns and date_col not in df_copy.columns:
                df_copy.rename(columns={"index": date_col}, inplace=True)

            required_cols = ['ticker', date_col, 'open', 'high', 'low', 'close', 'volume']
            df_copy = df_copy[[col for col in required_cols if col in df_copy.columns]]

            if date_col not in df_copy.columns:
                print(f"Warning: Date column '{date_col}' not found in DataFrame for {ticker} ({interval}). Skipping.")
                return

            df_copy[date_col] = pd.to_datetime(df_copy[date_col])
            min_date_val = df_copy[date_col].min()
            max_date_val = df_copy[date_col].max()

            if pd.isna(min_date_val) or pd.isna(max_date_val):
                print(f"Warning: Min/max date is NaT for {ticker} ({interval}). Skipping DB save.")
                return

            # Calculate MA and EMA for daily data only
            if interval == "daily":
                df_copy = await self._get_historical_data(
                    ticker,
                    min_date_val,
                    max_date_val,
                    df_copy,
                    interval,
                    ma_periods,
                    ema_periods,
                    date_col,
                    conn,
                )

            conn.execute(f"DELETE FROM {table} WHERE ticker = ? AND {date_col} BETWEEN ? AND ?", 
                         (ticker, min_date_val, max_date_val))
            try:
                conn.register('df_to_insert', df_copy)
                cols_str = ", ".join([f'"{col}"' for col in df_copy.columns]) # Ensure quoting for safety
                conn.execute(f"INSERT INTO {table} ({cols_str}) SELECT {cols_str} FROM df_to_insert")
            except Exception as e:
                print(f"Error saving aggregated to DuckDB for {ticker} ({interval}): {e}")

    async def get_stock_data(
        self,
        ticker: str,
        start_date: str | None = None,
        end_date: str | None = None,
        interval: str = "daily",
        conn=None,
    ) -> pd.DataFrame:
        def _execute_query(connection):
            table = 'daily_prices' if interval == 'daily' else 'hourly_prices'
            date_col_name = 'date' if interval == 'daily' else 'datetime'

            query = f"SELECT * FROM {table} WHERE ticker = ?"
            params: list[str | pd.Timestamp | None] = [ticker]

            if start_date:
                query += f" AND {date_col_name} >= ?"
                params.append(pd.to_datetime(start_date).strftime('%Y-%m-%d' if interval == 'daily' else '%Y-%m-%d %H:%M:%S'))
            if end_date:
                query += f" AND {date_col_name} <= ?"
                params.append(pd.to_datetime(end_date).strftime('%Y-%m-%d' if interval == 'daily' else '%Y-%m-%d %H:%M:%S'))

            query += f" ORDER BY {date_col_name}"
            df = connection.execute(query, parameters=params).fetchdf()  # type: ignore

            if not df.empty:
                df[date_col_name] = pd.to_datetime(df[date_col_name], errors="coerce")
                df.set_index(date_col_name, inplace=True)
                df = df[df.index.notna()]
            return df

        if conn is not None:
            # Use the provided connection
            return _execute_query(conn)
        else:
            # Create a new connection
            with duckdb.connect(database=self.db_path, read_only=True) as new_conn:
                return _execute_query(new_conn)

    async def save_realtime_data(self, df: pd.DataFrame, ticker: str, data_type: str = "quote") -> None:
        """Save realtime (tick) stock data to the DuckDB database."""
        with duckdb.connect(database=self.db_path, read_only=False) as conn:
            df_copy = df.copy()
            if df_copy.empty:
                # print(f"Empty DataFrame for realtime data of {ticker}. Skipping DuckDB save.")
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
                print(f"Warning: Missing required columns for realtime data (DuckDB) of {ticker}. Skipping.")
                return

            df_copy['timestamp'] = pd.to_datetime(df_copy['timestamp']) # Ensure it's datetime64[ns]
            min_ts_val = df_copy['timestamp'].min()
            max_ts_val = df_copy['timestamp'].max()

            if pd.isna(min_ts_val) or pd.isna(max_ts_val):
                print(f"Warning: Min/max timestamp is NaT for realtime data (DuckDB) of {ticker}. Skipping.")
                return

            conn.execute("DELETE FROM realtime_data WHERE ticker = ? AND type = ? AND timestamp BETWEEN ? AND ?",
                         (ticker, data_type, min_ts_val, max_ts_val))
            try:
                conn.register('df_rt_to_insert', df_copy)
                cols_str = ", ".join([f'"{col}"' for col in df_copy.columns])
                conn.execute(f"INSERT INTO realtime_data ({cols_str}) SELECT {cols_str} FROM df_rt_to_insert")
            except Exception as e:
                print(f"Error saving realtime data to DuckDB for {ticker}: {e}")

    async def get_realtime_data(self, ticker: str, start_datetime: str | None = None, end_datetime: str | None = None, data_type: str = "quote") -> pd.DataFrame:
        """Retrieve realtime (tick) stock data from the DuckDB database."""
        with duckdb.connect(database=self.db_path, read_only=True) as conn:
            query = "SELECT * FROM realtime_data WHERE ticker = ? AND type = ?"
            params: list[str | pd.Timestamp | None] = [ticker, data_type]

            if start_datetime:
                query += " AND timestamp >= ?"
                params.append(pd.to_datetime(start_datetime)) # DuckDB handles datetime objects
            if end_datetime:
                query += " AND timestamp <= ?"
                params.append(pd.to_datetime(end_datetime))

            query += " ORDER BY timestamp"
            df = conn.execute(query, parameters=params).fetchdf() # type: ignore

        if not df.empty:
            df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
            df.set_index('timestamp', inplace=True)
            df = df[df.index.notna()]
        return df

    async def get_latest_price(self, ticker: str) -> float | None:
        """Get the most recent price for a ticker (realtime -> hourly -> daily) from DuckDB."""
        with duckdb.connect(database=self.db_path, read_only=True) as conn:
            latest_price = None
            # 1. Try realtime_data
            try:
                res_rt = conn.execute("SELECT price FROM realtime_data WHERE ticker = ? AND type = 'quote' ORDER BY timestamp DESC LIMIT 1", (ticker,)).fetchone()
                if res_rt: latest_price = res_rt[0]
            except duckdb.Error as e:
                print(f"DuckDB error (realtime_data for {ticker}): {e}")

            # 2. Try hourly_prices
            if latest_price is None:
                try:
                    res_h = conn.execute("SELECT close FROM hourly_prices WHERE ticker = ? ORDER BY datetime DESC LIMIT 1", (ticker,)).fetchone()
                    if res_h: latest_price = res_h[0]
                except duckdb.Error as e:
                    print(f"DuckDB error (hourly_prices for {ticker}): {e}")

            # 3. Try daily_prices
            if latest_price is None:
                try:
                    res_d = conn.execute("SELECT close FROM daily_prices WHERE ticker = ? ORDER BY date DESC LIMIT 1", (ticker,)).fetchone()
                    if res_d: latest_price = res_d[0]
                except duckdb.Error as e:
                    print(f"DuckDB error (daily_prices for {ticker}): {e}")

        return latest_price

    async def execute_select_sql(self, sql_query: str, params: tuple = ()) -> pd.DataFrame:
        """Execute a direct SELECT SQL query on DuckDB and return results as a DataFrame."""
        def _run_sync():
            with duckdb.connect(database=self.db_path, read_only=True) as conn:
                df = conn.execute(sql_query, parameters=list(params) if params else []).fetchdf()
            return df
        return await asyncio.get_event_loop().run_in_executor(None, _run_sync)

    async def execute_raw_sql(self, sql_query: str, params: tuple = ()) -> list[dict[str, Any]]:
        """Execute a raw SQL query on DuckDB. If it returns data (e.g., RETURNING clause),
           that data is returned with binary fields Base64 encoded. Otherwise, an empty list is returned."""
        def _run_sync() -> list[dict[str, Any]]:
            with duckdb.connect(database=self.db_path, read_only=False) as conn:
                rel = conn.execute(sql_query, parameters=list(params) if params else [])
                results = []
                # DuckDB's relation object (rel) might not have a .description like sqlite cursor before fetching.
                # We fetch data first, then check if any rows were returned.
                try:
                    data_rows = rel.fetchall() # Returns list of tuples
                    if data_rows: # If there are rows, there must be columns
                        column_names = [desc[0] for desc in rel.description] # Now description should be available
                        for row in data_rows:
                            record = {}
                            for i, value in enumerate(row):
                                if isinstance(value, bytes):
                                    record[column_names[i]] = base64.b64encode(value).decode('utf-8')
                                else:
                                    record[column_names[i]] = value
                            results.append(record)
                except Exception as e:
                    # This might happen if the query is pure DDL or doesn't produce a result set that can be fetched.
                    # In such cases, we expect no data, so `results` remains empty.
                    # print(f"DuckDB: Note during raw SQL execution for query '{sql_query[:50]}...': {e}")
                    pass # `results` will be empty, which is the intended return for no data output
                return results
        return await asyncio.get_event_loop().run_in_executor(None, _run_sync)

class StockDBClient(StockDBBase):
    """
    A client class that implements the StockDBBase interface by sending
    requests to a StockDBServer over a network.
    """
    def __init__(self, server_address: str):
        super().__init__(server_address)
        self.server_url = f"http://{server_address}"
        self._session: aiohttp.ClientSession | None = None
        print(f"StockDBClient initialized, configured for server at {self.server_url}")

    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession()
        return self._session

    async def close_session(self):
        if self._session and not self._session.closed:
            await self._session.close()
            self._session = None

    def _init_db(self) -> None:
        # Client does not initialize DB; server handles it.
        pass

    async def _make_request(self, command: str, params: dict) -> dict:
        session = await self._get_session()
        payload = {"command": command, "params": params}
        try:
            async with session.post(f"{self.server_url}/db_command", json=payload) as response:
                response.raise_for_status()
                return await response.json()
        except aiohttp.ClientError as e:
            print(f"StockDBClient network error for command '{command}': {e}")
            raise ConnectionError(f"Failed to connect or communicate with server for command '{command}': {e}") from e
        except json.JSONDecodeError as e:
            print(f"StockDBClient JSON decode error for command '{command}': {e}")
            raise ValueError(f"Invalid JSON response from server for command '{command}': {e}") from e

    def _parse_df_from_response(self, response_data: list[dict], index_col: str) -> pd.DataFrame:
        if not response_data:
            return pd.DataFrame()
        df = pd.DataFrame.from_records(response_data)
        if index_col in df.columns:
            df[index_col] = pd.to_datetime(df[index_col], errors='coerce')
            df.set_index(index_col, inplace=True)
            df = df[df.index.notna()]
        return df

    async def save_stock_data(
        self,
        df: pd.DataFrame,
        ticker: str,
        interval: str = "daily",
        ma_periods: List[int] = None,
        ema_periods: List[int] = None,
    ) -> None:
        if df.empty: return
        # Server expects list of records and index_col hint
        df_reset = df.reset_index()
        index_col_name = df.index.name or ('timestamp' if 'timestamp' in df.columns else 'date') # Default index name

        # Ensure datetime columns are stringified in ISO format for JSON
        for col_name in df_reset.select_dtypes(include=['datetime64[ns]', 'datetime64[ns, UTC]']).columns:
            df_reset[col_name] = df_reset[col_name].dt.strftime('%Y-%m-%dT%H:%M:%S.%fZ')

        params = {
            "ticker": ticker,
            "interval": interval,
            "data": df_reset.to_dict(orient="records"),
            "index_col": index_col_name,
            "ma_periods": ma_periods,
            "ema_periods": ema_periods,
        }
        response = await self._make_request("save_stock_data", params)
        if response.get("error"):
            raise Exception(f"Server error saving stock data: {response['error']}")

    async def get_stock_data(
        self,
        ticker: str,
        start_date: str | None = None,
        end_date: str | None = None,
        interval: str = "daily",
        conn=None,
    ) -> pd.DataFrame:
        params = {"ticker": ticker, "start_date": start_date, "end_date": end_date, "interval": interval}
        response = await self._make_request("get_stock_data", params)
        if response.get("error"):
            raise Exception(f"Server error getting stock data: {response['error']}")

        index_col = 'date' if interval == 'daily' else 'datetime'
        return self._parse_df_from_response(response.get("data", []), index_col)

    async def save_realtime_data(self, df: pd.DataFrame, ticker: str, data_type: str = "quote") -> None:
        if df.empty: return
        df_reset = df.reset_index()
        index_col_name = df.index.name or 'timestamp'

        for col_name in df_reset.select_dtypes(include=['datetime64[ns]', 'datetime64[ns, UTC]']).columns:
            df_reset[col_name] = df_reset[col_name].dt.strftime('%Y-%m-%dT%H:%M:%S.%fZ')

        # Add type to each record
        records = df_reset.to_dict(orient='records')
        # for record in records: # No, type is a top-level param for the command
        #    record['type'] = data_type

        params = {
            "ticker": ticker,
            "data_type": data_type, # Pass data_type to server
            "data": records,
            "index_col": index_col_name
        }
        response = await self._make_request("save_realtime_data", params)
        if response.get("error"):
            raise Exception(f"Server error saving realtime data: {response['error']}")

    async def get_realtime_data(self, ticker: str, start_datetime: str | None = None, end_datetime: str | None = None, data_type: str = "quote") -> pd.DataFrame:
        params = {"ticker": ticker, "start_datetime": start_datetime, "end_datetime": end_datetime, "data_type": data_type}
        response = await self._make_request("get_realtime_data", params)
        if response.get("error"):
            raise Exception(f"Server error getting realtime data: {response['error']}")
        return self._parse_df_from_response(response.get("data", []), 'timestamp')

    async def get_latest_price(self, ticker: str) -> float | None:
        params = {"ticker": ticker}
        response = await self._make_request("get_latest_price", params)
        if response.get("error"):
            raise Exception(f"Server error getting latest price: {response['error']}")
        return response.get("latest_price")

    async def execute_select_sql(self, sql_query: str, params: tuple = ()) -> pd.DataFrame:
        server_params = {
            "sql_query": sql_query,
            "query_type": "select",
            "query_params": list(params) # Server expects a list
        }
        response = await self._make_request("execute_sql", server_params)
        if response.get("error"):
            raise Exception(f"Server error executing select SQL: {response.get('details', response['error'])}")
        return self._parse_df_from_response(response.get("data", []), 'timestamp') # Assuming timestamp or date will be handled by _parse

    async def execute_raw_sql(self, sql_query: str, params: tuple = ()) -> list[dict[str, Any]]:
        server_params = {
            "sql_query": sql_query,
            "query_type": "raw",
            "query_params": list(params) # Server expects a list
        }
        response = await self._make_request("execute_sql", server_params)
        if response.get("error"):
            raise Exception(f"Server error executing raw SQL: {response.get('details', response['error'])}")
        # Expects server to return a list of dicts under "data" key now
        return response.get("data", []) 


def get_stock_db(db_type: str, db_config: str | None = None) -> StockDBBase:
    actual_db_config = db_config
    if actual_db_config is None:
        actual_db_config = get_default_db_path(db_type)
    
    db_type_lower = db_type.lower()
    if db_type_lower == "sqlite":
        return StockDBSQLite(actual_db_config)
    elif db_type_lower == "duckdb":
        return StockDBDuckDB(actual_db_config)
    elif db_type_lower == "remote":
        return StockDBClient(actual_db_config)
    else:
        raise ValueError(f"Unsupported database type: {db_type}. Choose 'sqlite', 'duckdb', or 'remote'.")
