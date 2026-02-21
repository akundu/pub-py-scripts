import sqlite3
import pandas as pd
from datetime import datetime, timezone
import os
from abc import ABCMeta, abstractmethod
import duckdb
import asyncio
import aiohttp
import json
import base64
from typing import Any, List, Dict, Optional, Union
from .common_strategies import (
    calculate_moving_average,
    calculate_exponential_moving_average,
)
import sys
import logging
from .logging_utils import get_logger, log_info, log_warning, log_error, log_debug
from aiohttp import ClientTimeout, TCPConnector
import time
import atexit

DEFAULT_DATA_DIR = './data'
def get_default_db_path(db_type: str) -> str:   
    if db_type.lower() == "remote":
        return "localhost:8080"
    elif db_type.lower() == "remote":
        return f"stock_data.db"
    return os.path.join(
        DEFAULT_DATA_DIR, f"stock_data.{db_type}"
    )

class StockDBBase(metaclass=ABCMeta):
    """
    Abstract base class for stock database operations.
    """
    def __init__(self, db_config: str, logger: logging.Logger = None):
        self.db_config = db_config
        self.logger = get_logger("stock_db", logger)

    @abstractmethod
    def _init_db(self) -> None:
        "Initialize the database with required tables. (No-op for client)"""
        pass

    @abstractmethod
    async def save_stock_data(
        self,
        df: pd.DataFrame,
        ticker: str,
        interval: str = "daily",
        ma_periods: List[int] = None,
        ema_periods: List[int] = None,
        on_duplicate: str = "ignore",
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
    async def save_realtime_data(self, df: pd.DataFrame, ticker: str, data_type: str = "quote", on_duplicate: str = "ignore") -> None:
        """Save realtime (tick/quote/trade) stock data to the database."""
        pass

    @abstractmethod
    async def get_realtime_data(self, ticker: str, start_datetime: str | None = None, end_datetime: str | None = None, data_type: str = "quote") -> pd.DataFrame:
        """Retrieve realtime (tick/quote/trade) stock data from the database."""
        pass

    @abstractmethod
    async def get_latest_price(self, ticker: str, use_market_time: bool = True) -> float | None:
        """Get the most recent price for a ticker from any available data (realtime, hourly, daily)."""
        pass

    @abstractmethod
    async def get_latest_prices(self, tickers: List[str], use_market_time: bool = True) -> Dict[str, float | None]:
        """Get the most recent prices for multiple tickers from any available data (realtime, hourly, daily)."""
        pass

    @abstractmethod
    async def get_previous_close_prices(self, tickers: List[str]) -> Dict[str, float | None]:
        """Get the most recent daily close prices for multiple tickers."""
        pass

    @abstractmethod
    async def get_today_opening_prices(self, tickers: List[str]) -> Dict[str, float | None]:
        """Get today's opening prices for multiple tickers."""
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

    # ---- Options API (abstract) ----
    @abstractmethod
    async def save_options_data(self, df: pd.DataFrame, ticker: str) -> None:
        """Save options snapshot rows. df expected columns: option_ticker, expiration, strike, type,
        bid, ask, day_close, fmv, price, delta, gamma, theta, vega, implied_volatility, volume, open_interest,
        last_quote_timestamp. The DB layer will add ticker, write_timestamp, and bucketed timestamp for dedup."""
        pass

    @abstractmethod
    async def get_options_data(
        self,
        ticker: str,
        expiration_date: str | None = None,
        start_datetime: str | None = None,
        end_datetime: str | None = None,
        option_tickers: List[str] | None = None,
    ) -> pd.DataFrame:
        """Retrieve options rows filtered by ticker/expiration/date range/option tickers."""
        pass

    @abstractmethod
    async def get_latest_options_data(
        self,
        ticker: str,
        expiration_date: str | None = None,
        option_tickers: List[str] | None = None,
    ) -> pd.DataFrame:
        """Retrieve latest bucketed snapshot per option_ticker for a ticker (and optional expiration filter)."""
        pass

    @abstractmethod
    async def get_option_price_feature(self, ticker: str, option_ticker: str) -> dict[str, Any] | None:
        """Return latest record with price,bid,ask,day_close,fmv for a specific option ticker."""
        pass


    # ---- Financial Info Storage API (abstract) ----
    @abstractmethod
    async def save_financial_info(self, ticker: str, financial_data: dict) -> None:
        """Save financial ratios data to the database."""
        pass

    @abstractmethod
    async def get_financial_info(self, ticker: str, start_date: str | None = None, end_date: str | None = None) -> pd.DataFrame:
        """Retrieve financial info data from the database."""
        pass

    # ---- Combined price-series helper (optional, can be overridden by backends) ----
    async def get_merged_price_series(
        self,
        ticker: str,
        lookback_days: int = 365,
        hourly_days: int = 7,
        realtime_hours: int = 24,
    ) -> pd.DataFrame:
        """
        Return a merged time series for a ticker composed of:
        - Last `realtime_hours` hours from realtime quotes (highest resolution)
        - Previous `hourly_days` days from hourly bars
        - Up to `lookback_days` days of daily bars before that
        
        The default implementation raises NotImplementedError and should be
        overridden by backends that support realtime + aggregated data
        (e.g. QuestDB). Callers should be prepared for NotImplementedError.
        """
        raise NotImplementedError(
            "get_merged_price_series is not implemented for this StockDB backend"
        )

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
                        print(f"Error concatenating DataFrames for {ticker}: {str(e)}")
                        print(f"Historical DataFrame shape: {historical_df.shape}")
                        print(f"New DataFrame shape: {df_copy.shape}")
                        print(f"Historical DataFrame columns: {historical_df.columns.tolist()}")
                        print(f"New DataFrame columns: {df_copy.columns.tolist()}")
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
                print(f"Warning: Skipping row with NaT date for {ticker}")
                continue
                
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

        # Create a new DataFrame for the results
        result_df = df_copy.copy()
        
        # Add MA and EMA values back to result_df for the new data only
        for i, row in result_df.iterrows():
            # Check if the date is NaT before calling strftime
            if pd.isna(row[date_col]):
                print(f"Warning: Skipping row with NaT date for {ticker} in result processing")
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


class StockDBSQLite(StockDBBase):
    """
    A class to manage stock data storage and retrieval in an SQLite database.
    """
    def __init__(self, db_path: str = get_default_db_path("db"), logger: logging.Logger = None) -> None:
        super().__init__(db_path, logger)
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
                    write_timestamp DATETIME DEFAULT (DATETIME('now')),
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
                    write_timestamp DATETIME DEFAULT (DATETIME('now')),
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
                    write_timestamp DATETIME DEFAULT (DATETIME('now')), -- When the data was written to DB
                    PRIMARY KEY (ticker, timestamp, type)
                )
            ''')
            
            # Add write_timestamp column to existing table if it doesn't exist
            cursor.execute("PRAGMA table_info(realtime_data)")
            existing_columns = {row[1] for row in cursor.fetchall()}
            if 'write_timestamp' not in existing_columns:
                # SQLite doesn't allow non-constant defaults in ALTER TABLE, so add without default first
                cursor.execute("ALTER TABLE realtime_data ADD COLUMN write_timestamp DATETIME")
                # Update existing rows to have write_timestamp = timestamp
                cursor.execute("UPDATE realtime_data SET write_timestamp = timestamp WHERE write_timestamp IS NULL")

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
        on_duplicate: str = "ignore",
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
            # Add write_timestamp column with current UTC time
            from datetime import datetime, timezone as _tz
            df_copy['write_timestamp'] = datetime.now(_tz.utc)

            # Ensure table has write_timestamp column
            try:
                with conn:
                    conn.execute(f"ALTER TABLE {table} ADD COLUMN write_timestamp TIMESTAMP")
            except Exception:
                pass

            required_cols = ['ticker', date_col, 'open', 'high', 'low', 'close', 'volume', 'write_timestamp']
            df_copy = df_copy[[col for col in required_cols if col in df_copy.columns]]

            if date_col not in df_copy.columns:
                print(f"Warning: Date column '{date_col}' not found in DataFrame for {ticker} ({interval}). Skipping DB save.")
                return

            df_copy[date_col] = pd.to_datetime(df_copy[date_col])
            
            # Remove rows with NaT dates
            initial_count = len(df_copy)
            df_copy = df_copy.dropna(subset=[date_col])
            if len(df_copy) < initial_count:
                print(f"Warning: Removed {initial_count - len(df_copy)} rows with NaT dates for {ticker} ({interval})")
            
            if df_copy.empty:
                print(f"Warning: No valid data remaining for {ticker} ({interval}) after removing NaT dates. Skipping DB save.")
                return
            
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
            
            # For hourly data, we need to be more careful with the datetime format
            # Delete existing data in the range to avoid UNIQUE constraint violations
            if interval == 'hourly':
                # First, let's see what's actually in the database for this range
                cursor.execute(f'''
                    SELECT COUNT(*) FROM {table} 
                    WHERE ticker = ? 
                    AND {date_col} >= ? 
                    AND {date_col} <= ?
                ''', (ticker, min_date_str, max_date_str))
                existing_count = cursor.fetchone()[0]
                print(f"Found {existing_count} existing {interval} records for {ticker} in range {min_date_str} to {max_date_str}")
                
                # Delete existing data in the range
                cursor.execute(f'''
                    DELETE FROM {table} 
                    WHERE ticker = ? 
                    AND {date_col} >= ? 
                    AND {date_col} <= ?
                ''', (ticker, min_date_str, max_date_str))
                deleted_count = cursor.rowcount
                print(f"Deleted {deleted_count} existing {interval} records for {ticker}")
                
            else:
                # Daily data uses simple date format
                cursor.execute(f'''
                    DELETE FROM {table} 
                    WHERE ticker = ? 
                    AND {date_col} BETWEEN ? AND ?
                ''', (ticker, min_date_str, max_date_str))

            df_copy[date_col] = df_copy[date_col].dt.strftime(date_format_str)
            
            # Check for duplicates in the DataFrame before insertion
            duplicate_check = df_copy.duplicated(subset=['ticker', date_col], keep=False)
            if duplicate_check.any():
                print(f"Warning: Found {duplicate_check.sum()} duplicate rows in {ticker} {interval} data. Removing duplicates...")
                df_copy = df_copy.drop_duplicates(subset=['ticker', date_col], keep='last')
                print(f"After removing duplicates: {len(df_copy)} rows remaining")
            
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

    async def save_realtime_data(self, df: pd.DataFrame, ticker: str, data_type: str = "quote", on_duplicate: str = "ignore") -> None:
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

            # Add write_timestamp column if not already present
            if 'write_timestamp' not in df_copy.columns:
                current_time = datetime.now(timezone.utc)
                df_copy['write_timestamp'] = current_time

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
            df_copy['write_timestamp'] = df_copy['write_timestamp'].dt.strftime(ts_format_str)
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
            # Ensure write_timestamp is also converted to datetime
            if 'write_timestamp' in df.columns:
                df['write_timestamp'] = pd.to_datetime(df['write_timestamp'], errors='coerce')
            df.set_index('timestamp', inplace=True)
            df = df[df.index.notna()]
        return df

    async def get_latest_price(self, ticker: str, use_market_time: bool = True) -> float | None:
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

    async def get_latest_prices(self, tickers: List[str], use_market_time: bool = True) -> Dict[str, float | None]:
        """Get the most recent prices for multiple tickers (realtime -> hourly -> daily)."""
        result = {}
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            for ticker in tickers:
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

                result[ticker] = latest_price
        
        return result

    async def get_previous_close_prices(self, tickers: List[str]) -> Dict[str, float | None]:
        """Get previous close prices for multiple tickers.
        
        Returns the second-to-last close price from the database (if available).
        This ensures comparisons are made against actual data that exists, not assumptions
        about today's close. If only one close exists, uses that one.
        """
        result = {}
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Get today's date in EST timezone (market timezone)
            import pytz
            today = datetime.now(pytz.timezone('US/Eastern')).date()
            
            for ticker in tickers:
                latest_close = None
                try:
                    # Get the last 2 closes that exist in the database (ordered by date DESC)
                    # Use the second-to-last one as the reference for comparison
                    # This ensures we compare against actual data, not assumptions about today's close
                    cursor.execute("SELECT date, close FROM daily_prices WHERE ticker = ? ORDER BY date DESC LIMIT 2", (ticker,))
                    rows = cursor.fetchall()
                    if rows and len(rows) >= 2:
                        # We have at least 2 closes - use the second-to-last one
                        latest_close = rows[1][1]  # rows[1] is second row, [1] is close column
                    elif rows and len(rows) == 1:
                        # Only one close available - use it
                        latest_close = rows[0][1]  # rows[0] is first row, [1] is close column
                except sqlite3.Error as e:
                    print(f"SQLite error (daily_prices for {ticker}): {e}")
                
                result[ticker] = latest_close
        
        return result

    async def get_today_opening_prices(self, tickers: List[str]) -> Dict[str, float | None]:
        """Get today's opening prices for multiple tickers from daily_prices."""
        result = {}
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Get today's date in EST timezone (market timezone)
            import pytz
            today = datetime.now(pytz.timezone('US/Eastern')).date()
            
            for ticker in tickers:
                opening_price = None
                try:
                    # Get today's opening price
                    cursor.execute("SELECT open FROM daily_prices WHERE ticker = ? AND date = ? ORDER BY date DESC LIMIT 1", (ticker, today))
                    row = cursor.fetchone()
                    if row: 
                        opening_price = row[0]
                    else:
                        # If no today data found, might not have market open data yet
                        pass
                except sqlite3.Error as e:
                    print(f"SQLite error (daily_prices for {ticker}): {e}")
                
                result[ticker] = opening_price
        
        return result

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

    # ---- Options API (stub implementations) ----
    async def save_options_data(self, df: pd.DataFrame, ticker: str) -> None:
        """Save options snapshot rows to SQLite (not implemented)."""
        self.logger.warning("Options data storage not implemented for SQLite")
        pass

    async def get_options_data(
        self,
        ticker: str,
        expiration_date: str | None = None,
        start_datetime: str | None = None,
        end_datetime: str | None = None,
        option_tickers: List[str] | None = None,
    ) -> pd.DataFrame:
        """Retrieve options data from SQLite (not implemented)."""
        self.logger.warning("Options data retrieval not implemented for SQLite")
        return pd.DataFrame()

    async def get_latest_options_data(
        self,
        ticker: str,
        expiration_date: str | None = None,
        option_tickers: List[str] | None = None,
    ) -> pd.DataFrame:
        """Retrieve latest options data from SQLite (not implemented)."""
        self.logger.warning("Latest options data retrieval not implemented for SQLite")
        return pd.DataFrame()

    async def get_option_price_feature(self, ticker: str, option_ticker: str) -> dict[str, Any] | None:
        """Get option price feature from SQLite (not implemented)."""
        self.logger.warning("Option price feature retrieval not implemented for SQLite")
        return None

    # ---- Financial Info API ----
    async def save_financial_info(self, ticker: str, financial_data: dict) -> None:
        """Save financial ratios data to SQLite (not implemented)."""
        self.logger.warning("Financial info storage not implemented for SQLite")
        pass

    async def get_financial_info(self, ticker: str, start_date: str | None = None, end_date: str | None = None) -> pd.DataFrame:
        """Retrieve financial info data from SQLite (not implemented)."""
        self.logger.warning("Financial info retrieval not implemented for SQLite")
        return pd.DataFrame()

class StockDBDuckDB(StockDBBase):
    """
    A class to manage stock data storage and retrieval in a DuckDB database.
    """
    def __init__(self, db_path: str = get_default_db_path("duckdb"), logger: logging.Logger = None) -> None:
        super().__init__(db_path, logger)
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
                    write_timestamp TIMESTAMP DEFAULT NOW(),
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
                    write_timestamp TIMESTAMP DEFAULT NOW(),
                    PRIMARY KEY (ticker, datetime)
                )
            ''')
            conn.execute('''
                CREATE TABLE IF NOT EXISTS realtime_data (
                    ticker VARCHAR, timestamp TIMESTAMP, type VARCHAR, price DOUBLE, size BIGINT,
                    ask_price DOUBLE, ask_size BIGINT, write_timestamp TIMESTAMP DEFAULT NOW(),
                    PRIMARY KEY (ticker, timestamp, type)
                )
            ''')
            
            # Add write_timestamp column to existing table if it doesn't exist
            try:
                result = conn.execute("DESCRIBE realtime_data").fetchall()
                existing_columns = {row[0] for row in result}
                if 'write_timestamp' not in existing_columns:
                    conn.execute("ALTER TABLE realtime_data ADD COLUMN write_timestamp TIMESTAMP DEFAULT NOW()")
                    # Update existing rows to have write_timestamp = timestamp
                    conn.execute("UPDATE realtime_data SET write_timestamp = timestamp WHERE write_timestamp IS NULL")
            except Exception as e:
                # Table might not exist yet, which is fine
                pass

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
        on_duplicate: str = "ignore",
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

            # Add write_timestamp column with current UTC time
            from datetime import datetime, timezone as _tz
            df_copy['write_timestamp'] = datetime.now(_tz.utc)

            # Ensure table has write_timestamp column
            try:
                conn.execute(f"ALTER TABLE {table} ADD COLUMN write_timestamp TIMESTAMP")
            except Exception:
                pass

            required_cols = ['ticker', date_col, 'open', 'high', 'low', 'close', 'volume', 'write_timestamp']
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

    async def save_realtime_data(self, df: pd.DataFrame, ticker: str, data_type: str = "quote", on_duplicate: str = "ignore") -> None:
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

            # Add write_timestamp column if not already present
            if 'write_timestamp' not in df_copy.columns:
                current_time = datetime.now(timezone.utc)
                df_copy['write_timestamp'] = current_time

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
            # Ensure write_timestamp is also converted to datetime
            if 'write_timestamp' in df.columns:
                df['write_timestamp'] = pd.to_datetime(df['write_timestamp'], errors='coerce')
            df.set_index('timestamp', inplace=True)
            df = df[df.index.notna()]
        return df

    async def get_latest_price(self, ticker: str, use_market_time: bool = True) -> float | None:
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

    async def get_latest_prices(self, tickers: List[str], use_market_time: bool = True) -> Dict[str, float | None]:
        """Get the most recent prices for multiple tickers (realtime -> hourly -> daily) from DuckDB."""
        result = {}
        
        with duckdb.connect(database=self.db_path, read_only=True) as conn:
            for ticker in tickers:
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

                result[ticker] = latest_price
        
        return result

    async def get_previous_close_prices(self, tickers: List[str]) -> Dict[str, float | None]:
        """Get previous close prices for multiple tickers.
        
        Returns the second-to-last close price from the database (if available).
        This ensures comparisons are made against actual data that exists, not assumptions
        about today's close. If only one close exists, uses that one.
        """
        result = {}
        
        with duckdb.connect(database=self.db_path, read_only=True) as conn:
            # Get today's date in EST timezone (market timezone)
            import pytz
            today = datetime.now(pytz.timezone('US/Eastern')).date()
            
            for ticker in tickers:
                latest_close = None
                try:
                    # Get the last 2 closes that exist in the database (ordered by date DESC)
                    # Use the second-to-last one as the reference for comparison
                    # This ensures we compare against actual data, not assumptions about today's close
                    res = conn.execute("SELECT date, close FROM daily_prices WHERE ticker = ? ORDER BY date DESC LIMIT 2", (ticker,)).fetchall()
                    if res and len(res) >= 2:
                        # We have at least 2 closes - use the second-to-last one
                        latest_close = res[1][1]  # res[1] is second row, [1] is close column
                    elif res and len(res) == 1:
                        # Only one close available - use it
                        latest_close = res[0][1]  # res[0] is first row, [1] is close column
                except duckdb.Error as e:
                    print(f"DuckDB error (daily_prices for {ticker}): {e}")
                
                result[ticker] = latest_close
        
        return result

    async def get_today_opening_prices(self, tickers: List[str]) -> Dict[str, float | None]:
        """Get today's opening prices for multiple tickers from daily_prices."""
        result = {}
        
        with duckdb.connect(database=self.db_path, read_only=True) as conn:
            # Get today's date in EST timezone (market timezone)
            import pytz
            today = datetime.now(pytz.timezone('US/Eastern')).date()
            
            for ticker in tickers:
                opening_price = None
                try:
                    # Get today's opening price
                    res = conn.execute("SELECT open FROM daily_prices WHERE ticker = ? AND date = ? ORDER BY date DESC LIMIT 1", (ticker, today)).fetchone()
                    if res: 
                        opening_price = res[0]
                    else:
                        # If no today data found, might not have market open data yet
                        pass
                except duckdb.Error as e:
                    print(f"DuckDB error (daily_prices for {ticker}): {e}")
                
                result[ticker] = opening_price
        
        return result

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

    async def save_financial_info(self, ticker: str, financial_data: dict) -> None:
        """Save financial ratios data to DuckDB (not implemented)."""
        self.logger.warning("Financial info storage not implemented for DuckDB")
        pass

    async def get_financial_info(self, ticker: str, start_date: str | None = None, end_date: str | None = None) -> pd.DataFrame:
        """Retrieve financial info data from DuckDB (not implemented)."""
        self.logger.warning("Financial info retrieval not implemented for DuckDB")
        return pd.DataFrame()

class StockDBClient(StockDBBase):
    """
    Network client that sends database requests to a StockDBServer over a network.
    
    Features:
    - Configurable response timeout (default: 10 seconds)
    - Configurable retry mechanism (default: 10 retries)
    - Random retry intervals to prevent thundering herd (default: 1 second base)
    - Automatic session management with aiohttp
    - Comprehensive error handling and logging
    
    Args:
        server_address: Server address in format "host:port"
        logger: Optional logger instance
        response_timeout: Response timeout in seconds (default: 10.0)
        max_retries: Maximum number of retry attempts (default: 10)
        retry_interval: Base retry interval in seconds (default: 1.0)
    
    Example:
        # Create client with default settings
        client = StockDBClient("localhost:8080")
        
        # Create client with custom timeout and retry settings
        client = StockDBClient("localhost:8080", response_timeout=30.0, max_retries=5, retry_interval=2.0)
        
        # Update settings after initialization
        client.update_connection_settings(response_timeout=15.0, max_retries=8)
    """
    def __init__(self, server_address: str, logger: logging.Logger = None, 
                 response_timeout: float = 10.0, max_retries: int = 10, retry_interval: float = 1.0):
        super().__init__(server_address, logger)
        self.server_url = f"http://{server_address}"
        self.session_manager = StockDBSessionManager()
        self.response_timeout = response_timeout
        self.max_retries = max_retries
        self.retry_interval = retry_interval
        self._session_closed = False  # Track if session has been explicitly closed
        # print(f"StockDBClient initialized, configured for server at {self.server_url}", file=sys.stderr)
        # Timing metrics
        self._timing_enabled = True
        self._timing_start_ts = time.time()
        self._timing_last_report_ts = self._timing_start_ts
        self._timing_lock = asyncio.Lock()
        self._cmd_stats: Dict[str, Dict[str, float | int]] = {}
        # Ensure summary on exit
        atexit.register(self._print_timing_summary_sync)

    def update_connection_settings(self, response_timeout: float = None, max_retries: int = None, retry_interval: float = None) -> None:
        """
        Update connection settings after initialization.
        
        Args:
            response_timeout: New response timeout in seconds
            max_retries: New maximum number of retries
            retry_interval: New retry interval in seconds
        """
        if response_timeout is not None:
            self.response_timeout = response_timeout
        if max_retries is not None:
            self.max_retries = max_retries
        if retry_interval is not None:
            self.retry_interval = retry_interval
        
        self.logger.info(f"Updated connection settings: timeout={self.response_timeout}s, max_retries={self.max_retries}, retry_interval={self.retry_interval}s")

    def get_connection_settings(self) -> Dict[str, Any]:
        """
        Get current connection settings.
        
        Returns:
            Dictionary with current timeout, max_retries, and retry_interval values
        """
        return {
            "response_timeout": self.response_timeout,
            "max_retries": self.max_retries,
            "retry_interval": self.retry_interval
        }

    async def _get_session(self) -> aiohttp.ClientSession:
        return await self.session_manager.get_session()

    async def close_session(self):
        await self.session_manager.close()
        self._session_closed = True
    
    def __del__(self):
        """Cleanup when the object is destroyed."""
        if hasattr(self, 'session_manager') and hasattr(self, '_session_closed'):
            # Only attempt cleanup if session hasn't been explicitly closed
            if not self._session_closed:
                try:
                    import asyncio
                    loop = asyncio.get_event_loop()
                    if loop.is_running():
                        # Schedule cleanup if loop is running
                        loop.create_task(self.session_manager.close())
                    else:
                        # Run cleanup if loop is not running
                        loop.run_until_complete(self.session_manager.close())
                except:
                    # Ignore cleanup errors during destruction
                    pass

    async def __aenter__(self):
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit with proper cleanup."""
        await self.close_session()

    def _init_db(self) -> None:
        # Client does not initialize DB; server handles it.
        pass

    async def _make_request(self, command: str, params: dict) -> dict:
        session = await self._get_session()
        payload = {"command": command, "params": params}
        
        # Retry logic with random backoff
        for attempt in range(self.max_retries):
            try:
                t0 = time.time()
                async with session.post(f"{self.server_url}/db_command", json=payload, timeout=self.response_timeout) as response:
                    response.raise_for_status()
                    data = await response.json()
                    await self._record_timing(command, time.time() - t0, success=True)
                    await self._maybe_periodic_report()
                    return data
                    
            except (aiohttp.ClientError, asyncio.TimeoutError) as e:
                if attempt < self.max_retries - 1:
                    # Random delay between retry_interval/2 and retry_interval*1.5 seconds
                    import random
                    delay = random.uniform(self.retry_interval * 0.5, self.retry_interval * 1.5)
                    
                    # Provide specific error messages for different error types
                    if isinstance(e, asyncio.TimeoutError):
                        error_msg = f"Request timeout after {self.response_timeout}s"
                    else:
                        error_msg = str(e)
                    
                    self.logger.warning(f"StockDBClient attempt {attempt + 1}/{self.max_retries} failed for command '{command}': {error_msg}. Retrying in {delay:.2f}s...")
                    await self._record_timing(command, time.time() - t0, success=False)
                    await self._maybe_periodic_report()
                    await asyncio.sleep(delay)
                    continue
                else:
                    # Final attempt failed
                    if isinstance(e, asyncio.TimeoutError):
                        error_msg = f"Request timeout after {self.response_timeout}s"
                    else:
                        error_msg = str(e)
                    
                    self.logger.error(f"StockDBClient network error for command '{command}' after {self.max_retries} attempts: {error_msg}", exc_info=True)
                    await self._record_timing(command, time.time() - t0, success=False)
                    await self._maybe_periodic_report(force=True)
                    raise ConnectionError(f"Failed to connect or communicate with server for command '{command}' after {self.max_retries} attempts: {error_msg}") from e
                    
            except json.JSONDecodeError as e:
                self.logger.error(f"StockDBClient JSON decode error for command '{command}': {e}", exc_info=True)
                raise ValueError(f"Invalid JSON response from server for command '{command}': {e}") from e

    async def _record_timing(self, command: str, duration_s: float, success: bool) -> None:
        if not self._timing_enabled:
            return
        async with self._timing_lock:
            stats = self._cmd_stats.setdefault(command, {
                "count": 0,
                "success": 0,
                "fail": 0,
                "total_s": 0.0,
                "max_s": 0.0
            })
            stats["count"] += 1
            stats["total_s"] += duration_s
            stats["max_s"] = max(stats["max_s"], duration_s)
            if success:
                stats["success"] += 1
            else:
                stats["fail"] += 1

    def _format_summary(self) -> str:
        lines = []
        elapsed = time.time() - self._timing_start_ts
        lines.append(f"Client timing summary (elapsed {elapsed:.1f}s):")
        for cmd, s in sorted(self._cmd_stats.items()):
            count = int(s.get("count", 0))
            total_s = float(s.get("total_s", 0.0))
            max_s = float(s.get("max_s", 0.0))
            success = int(s.get("success", 0))
            fail = int(s.get("fail", 0))
            avg = (total_s / count) if count else 0.0
            lines.append(f"- {cmd}: count={count}, success={success}, fail={fail}, avg={avg:.3f}s, max={max_s:.3f}s, total={total_s:.1f}s")
        return "\n".join(lines)

    async def _maybe_periodic_report(self, force: bool = False) -> None:
        if not self._timing_enabled:
            return
        now = time.time()
        if force or (now - self._timing_last_report_ts) >= 10.0:
            self._timing_last_report_ts = now
            summary = self._format_summary()
            self.logger.info(summary)

    def _print_timing_summary_sync(self) -> None:
        try:
            summary = self._format_summary()
            print(summary, file=sys.stderr)
        except Exception:
            pass

    def _parse_df_from_response(self, response_data: list[dict], index_col: str) -> pd.DataFrame:
        if not response_data:
            return pd.DataFrame()
        # Handle case where response_data might be a list of lists instead of list of dicts
        if response_data and isinstance(response_data[0], list):
            # If it's a list of lists, we can't preserve column names - this shouldn't happen
            # but we'll handle it gracefully
            df = pd.DataFrame(response_data)
        else:
            df = pd.DataFrame.from_records(response_data)
        
        # Parse all timestamp columns (not just the index column)
        # Look for columns with 'timestamp' in the name
        for col in df.columns:
            if 'timestamp' in col.lower() and not pd.api.types.is_datetime64_any_dtype(df[col]):
                # Check if the column is already an integer (Unix timestamp)
                if pd.api.types.is_integer_dtype(df[col]):
                    # Try to convert integer to datetime (Unix timestamp)
                    first_val = df[col].iloc[0] if len(df) > 0 and not df[col].isna().all() else 0
                    if first_val > 1e10:  # Likely milliseconds
                        df[col] = pd.to_datetime(df[col], unit='ms', errors='coerce')
                    elif first_val > 1e9:  # Likely seconds
                        df[col] = pd.to_datetime(df[col], unit='s', errors='coerce')
                    else:
                        # Try general conversion
                        df[col] = pd.to_datetime(df[col], errors='coerce')
                else:
                    # Try to parse with explicit format first, then fall back to flexible parsing
                    try:
                        # Try ISO format first (most common for our timestamps)
                        df[col] = pd.to_datetime(df[col], format='ISO8601', errors='coerce')
                    except (ValueError, TypeError):
                        try:
                            # Try specific ISO format patterns
                            df[col] = pd.to_datetime(df[col], format='%Y-%m-%dT%H:%M:%S.%fZ', errors='coerce')
                        except (ValueError, TypeError):
                            try:
                                # Try without microseconds
                                df[col] = pd.to_datetime(df[col], format='%Y-%m-%dT%H:%M:%SZ', errors='coerce')
                            except (ValueError, TypeError):
                                # Fall back to flexible parsing but suppress the warning
                                import warnings
                                with warnings.catch_warnings():
                                    warnings.filterwarnings('ignore', category=UserWarning, message='Could not infer format')
                                    df[col] = pd.to_datetime(df[col], errors='coerce')
        
        if index_col in df.columns:
            # Check if the index column is already an integer (Unix timestamp)
            if pd.api.types.is_integer_dtype(df[index_col]):
                # Try to convert integer to datetime (Unix timestamp)
                first_val = df[index_col].iloc[0] if len(df) > 0 else 0
                # Check if it's a date in YYYYMMDD format (8 digits, between 19000101 and 99991231)
                if 19000101 <= first_val <= 99991231 and len(str(int(first_val))) == 8:
                    # It's a date in YYYYMMDD format - convert it properly
                    df[index_col] = df[index_col].astype(str).apply(
                        lambda x: pd.to_datetime(x, format='%Y%m%d', errors='coerce') if len(x) == 8 else pd.NaT
                    )
                elif first_val > 1e10:  # Likely milliseconds
                    df[index_col] = pd.to_datetime(df[index_col], unit='ms', errors='coerce')
                elif first_val > 1e9:  # Likely seconds
                    df[index_col] = pd.to_datetime(df[index_col], unit='s', errors='coerce')
                else:
                    # Try general conversion, but validate the result
                    df[index_col] = pd.to_datetime(df[index_col], errors='coerce')
                    # If the result is before 1900, it's likely wrong (probably epoch time misinterpretation)
                    if not df[index_col].isna().all() and df[index_col].min().year < 1900:
                        # Try treating as YYYYMMDD format instead
                        df[index_col] = df[index_col].astype(str).apply(
                            lambda x: pd.to_datetime(x, format='%Y%m%d', errors='coerce') if len(x) == 8 else pd.NaT
                        )
            else:
                # Try to parse with explicit format first, then fall back to flexible parsing
                try:
                    # Try ISO format first (most common for our timestamps)
                    df[index_col] = pd.to_datetime(df[index_col], format='ISO8601', errors='coerce')
                except (ValueError, TypeError):
                    try:
                        # Try specific ISO format patterns
                        df[index_col] = pd.to_datetime(df[index_col], format='%Y-%m-%dT%H:%M:%S.%fZ', errors='coerce')
                    except (ValueError, TypeError):
                        try:
                            # Try without microseconds
                            df[index_col] = pd.to_datetime(df[index_col], format='%Y-%m-%dT%H:%M:%SZ', errors='coerce')
                        except (ValueError, TypeError):
                            # Fall back to flexible parsing but suppress the warning
                            import warnings
                            with warnings.catch_warnings():
                                warnings.filterwarnings('ignore', category=UserWarning, message='Could not infer format')
                                df[index_col] = pd.to_datetime(df[index_col], errors='coerce')
            
            df.set_index(index_col, inplace=True)
            # Ensure index is datetime type after setting as index
            if not pd.api.types.is_datetime64_any_dtype(df.index):
                try:
                    if pd.api.types.is_integer_dtype(df.index):
                        first_val = df.index[0] if len(df) > 0 else 0
                        if first_val > 1e10:  # Likely milliseconds
                            df.index = pd.to_datetime(df.index, unit='ms', errors='coerce')
                        else:  # Likely seconds
                            df.index = pd.to_datetime(df.index, unit='s', errors='coerce')
                    else:
                        df.index = pd.to_datetime(df.index, errors='coerce')
                except Exception:
                    pass
            df = df[df.index.notna()]
        return df

    async def save_stock_data(
        self,
        df: pd.DataFrame,
        ticker: str,
        interval: str = "daily",
        ma_periods: List[int] = None,
        ema_periods: List[int] = None,
        on_duplicate: str = "ignore",
    ) -> None:
        if df.empty: return
        # Server expects list of records and index_col hint
        df_reset = df.reset_index()
        index_col_name = df.index.name or ('timestamp' if 'timestamp' in df.columns else 'date') # Default index name

        # Add write_timestamp column with current UTC time
        from datetime import datetime, timezone as _tz
        df_reset['write_timestamp'] = datetime.now(_tz.utc)

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
            "on_duplicate": on_duplicate,
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

    async def save_realtime_data(self, df: pd.DataFrame, ticker: str, data_type: str = "quote", on_duplicate: str = "ignore") -> None:
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
            "index_col": index_col_name,
            "on_duplicate": on_duplicate
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

    async def get_latest_price(self, ticker: str, use_market_time: bool = True) -> float | None:
        params = {"ticker": ticker, "use_market_time": use_market_time}
        response = await self._make_request("get_latest_price", params)
        if response.get("error"):
            raise Exception(f"Server error getting latest price: {response['error']}")
        return response.get("latest_price")

    async def get_latest_prices(self, tickers: List[str], use_market_time: bool = True) -> Dict[str, float | None]:
        """Get the most recent prices for multiple tickers from the server."""
        params = {"tickers": tickers, "use_market_time": use_market_time}
        response = await self._make_request("get_latest_prices", params)
        if response.get("error"):
            raise Exception(f"Server error getting latest prices: {response['error']}")
        return response.get("prices", {})

    async def get_previous_close_prices(self, tickers: List[str]) -> Dict[str, float | None]:
        """Get the most recent daily close prices for multiple tickers from the server."""
        params = {"tickers": tickers}
        response = await self._make_request("get_previous_close_prices", params)
        if response.get("error"):
            raise Exception(f"Server error getting previous close prices: {response['error']}")
        return response.get("prices", {})

    async def get_today_opening_prices(self, tickers: List[str]) -> Dict[str, float | None]:
        """Get today's opening prices for multiple tickers from the server."""
        params = {"tickers": tickers}
        response = await self._make_request("get_today_opening_prices", params)
        if response.get("error"):
            raise Exception(f"Server error getting today's opening prices: {response['error']}")
        return response.get("prices", {})

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


    # ---- Options API (client → server passthrough) ----
    async def save_options_data(self, df: pd.DataFrame, ticker: str) -> None:
        """Send options snapshot rows to the server for persistence.
        Accepts a DataFrame with expected option columns; serializes and posts.
        """
        if df is None or df.empty:
            return
        df_reset = df.reset_index()
        # Convert datetime-like columns to ISO strings for JSON
        for col_name in df_reset.select_dtypes(include=['datetime64[ns]', 'datetime64[ns, UTC]']).columns:
            df_reset[col_name] = df_reset[col_name].dt.strftime('%Y-%m-%dT%H:%M:%S.%fZ')
        params = {
            "ticker": ticker,
            "data": df_reset.to_dict(orient="records"),
        }
        response = await self._make_request("save_options_data", params)
        if response.get("error"):
            raise Exception(f"Server error saving options data: {response['error']}")

    async def get_options_data(
        self,
        ticker: str,
        expiration_date: str | None = None,
        start_datetime: str | None = None,
        end_datetime: str | None = None,
        option_tickers: List[str] | None = None,
    ) -> pd.DataFrame:
        params = {
            "ticker": ticker,
            "expiration_date": expiration_date,
            "start_datetime": start_datetime,
            "end_datetime": end_datetime,
            "option_tickers": option_tickers,
        }
        response = await self._make_request("get_options_data", params)
        if response.get("error"):
            raise Exception(f"Server error getting options data: {response['error']}")
        records = response.get("data", [])
        return pd.DataFrame.from_records(records)

    async def get_latest_options_data(
        self,
        ticker: str,
        expiration_date: str | None = None,
        option_tickers: List[str] | None = None,
    ) -> pd.DataFrame:
        params = {
            "ticker": ticker,
            "expiration_date": expiration_date,
            "option_tickers": option_tickers,
        }
        response = await self._make_request("get_latest_options_data", params)
        if response.get("error"):
            raise Exception(f"Server error getting latest options data: {response['error']}")
        records = response.get("data", [])
        return pd.DataFrame.from_records(records)

    async def get_option_price_feature(self, ticker: str, option_ticker: str) -> dict[str, Any] | None:
        params = {"ticker": ticker, "option_ticker": option_ticker}
        response = await self._make_request("get_option_price_feature", params)
        if response.get("error"):
            raise Exception(f"Server error getting option price feature: {response['error']}")
        return response.get("data")

    async def save_financial_info(self, ticker: str, financial_data: dict) -> None:
        """Save financial ratios data via remote server."""
        if not financial_data:
            return
            
        # Prepare the financial data for transmission
        # Convert datetime objects to ISO strings for JSON serialization
        prepared_data = financial_data.copy()
        
        # Handle date conversion if present
        if 'date' in prepared_data and prepared_data['date']:
            if hasattr(prepared_data['date'], 'isoformat'):
                prepared_data['date'] = prepared_data['date'].isoformat()
            elif hasattr(prepared_data['date'], 'strftime'):
                prepared_data['date'] = prepared_data['date'].strftime('%Y-%m-%d')
        
        params = {
            "ticker": ticker,
            "financial_data": prepared_data
        }
        
        response = await self._make_request("save_financial_info", params)
        if response.get("error"):
            raise Exception(f"Server error saving financial info: {response['error']}")

    async def get_financial_info(self, ticker: str, start_date: str | None = None, end_date: str | None = None) -> pd.DataFrame:
        """Retrieve financial info data via remote server."""
        params = {
            "ticker": ticker,
            "start_date": start_date,
            "end_date": end_date
        }
        
        response = await self._make_request("get_financial_info", params)
        if response.get("error"):
            raise Exception(f"Server error getting financial info: {response['error']}")
        
        # Parse the response data into a DataFrame
        records = response.get("data", [])
        if not records:
            return pd.DataFrame()
            
        df = pd.DataFrame.from_records(records)
        
        # Convert date column to datetime and set as index if present
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
            df.set_index('date', inplace=True)
            df = df[df.index.notna()]
        
        return df

class StockDBSessionManager:
    """Manages aiohttp client sessions with connection pooling and retry logic."""
    
    def __init__(self, max_connections: int = 200, max_per_host: int = 100):
        self.max_connections = max_connections
        self.max_per_host = max_per_host
        self._session: Optional[aiohttp.ClientSession] = None
        self._connector: Optional[TCPConnector] = None
        
    async def get_session(self) -> aiohttp.ClientSession:
        """Get or create a client session with proper configuration."""
        if self._session is None or self._session.closed:
            await self._create_session()
        return self._session
    
    async def _create_session(self):
        """Create a new session with optimized settings."""
        # Create connector with connection pooling
        self._connector = TCPConnector(
            limit=self.max_connections,
            limit_per_host=self.max_per_host,
            keepalive_timeout=60,  # Increased from 30
            enable_cleanup_closed=True,
            ttl_dns_cache=600,     # Increased from 300
            use_dns_cache=True,
            force_close=False,
            ssl=False,             # Disable SSL for local connections
        )
        
        # Configure timeouts
        timeout = ClientTimeout(
            total=120,  # Increased from 60 for database operations
            connect=20,  # Increased from 10 
            sock_read=60,  # Increased from 30
            sock_connect=20,  # Increased from 10
        )
        
        # Create session
        self._session = aiohttp.ClientSession(
            connector=self._connector,
            timeout=timeout,
            headers={'Connection': 'keep-alive'},
        )
    
    async def close(self):
        """Close the session and connector."""
        if self._session and not self._session.closed:
            await self._session.close()
            self._session = None
        
        if self._connector and not self._connector.closed:
            await self._connector.close()
            self._connector = None


def get_stock_db(db_type: str, db_config: str | None = None, logger: logging.Logger = None, log_level: str = "INFO", timeout: float = None, historical_batch_size: int = 25, questdb_connection_timeout_seconds: int = 180, enable_cache: bool = True, redis_url: str | None = None, auto_init: bool = True) -> StockDBBase:
    actual_db_config = db_config
    if actual_db_config is None:
        actual_db_config = get_default_db_path(db_type)
    
    db_type_lower = db_type.lower()
    if db_type_lower == "sqlite":
        return StockDBSQLite(actual_db_config, logger)
    elif db_type_lower == "duckdb":
        return StockDBDuckDB(actual_db_config, logger)
    elif db_type_lower == "postgresql":
        from .postgres_db import StockDBPostgreSQL
        return StockDBPostgreSQL(actual_db_config, logger=logger, log_level=log_level, historical_batch_size=historical_batch_size)
    elif db_type_lower == "timescaledb":
        from .timescale_db import StockDBTimescale
        return StockDBTimescale(actual_db_config, logger=logger, log_level=log_level, historical_batch_size=historical_batch_size)
    elif db_type_lower == "questdb":
        from .questdb_db import StockQuestDB
        return StockQuestDB(actual_db_config, logger=logger, log_level=log_level, 
                           connection_timeout_seconds=questdb_connection_timeout_seconds,
                           enable_cache=enable_cache, redis_url=redis_url, auto_init=auto_init)
    elif db_type_lower == "remote":
        if timeout is not None:
            return StockDBClient(actual_db_config, logger, response_timeout=timeout)
        else:
            return StockDBClient(actual_db_config, logger)
    else:
        raise ValueError(f"Unsupported database type: {db_type}. Choose 'sqlite', 'duckdb', 'postgresql', 'timescaledb', 'questdb', or 'remote'.")
