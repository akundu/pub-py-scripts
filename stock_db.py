import sqlite3
import pandas as pd
from datetime import datetime
import os
from abc import ABCMeta, abstractmethod
import duckdb

DEFAULT_DATA_DIR = './data'
def get_default_db_path(db_type: str) -> str:   
    return os.path.join(DEFAULT_DATA_DIR, f"stock_data.{db_type}")

class StockDBBase(metaclass=ABCMeta):
    """
    Abstract base class for stock database operations.
    """
    def __init__(self, db_path: str):
        self.db_path = db_path
        self._init_db()

    @abstractmethod
    def _init_db(self) -> None:
        """Initialize the database with required tables."""
        pass

    @abstractmethod
    def save_stock_data(self, df: pd.DataFrame, ticker: str, interval: str = 'daily') -> None:
        """Save aggregated (daily/hourly) stock data to the database."""
        pass

    @abstractmethod
    def get_stock_data(self, ticker: str, start_date: str | None = None, end_date: str | None = None, 
                       interval: str = 'daily') -> pd.DataFrame:
        """Retrieve aggregated (daily/hourly) stock data from the database."""
        pass
    
    @abstractmethod
    def save_realtime_data(self, df: pd.DataFrame, ticker: str) -> None:
        """Save realtime (tick) stock data to the database."""
        pass

    @abstractmethod
    def get_realtime_data(self, ticker: str, start_datetime: str | None = None, end_datetime: str | None = None) -> pd.DataFrame:
        """Retrieve realtime (tick) stock data from the database."""
        pass

    @abstractmethod
    def get_latest_price(self, ticker: str) -> float | None:
        """Get the most recent price for a ticker from any available data (realtime, hourly, daily)."""
        pass

class StockDBSQLite(StockDBBase):
    """
    A class to manage stock data storage and retrieval in an SQLite database.
    """
    def __init__(self, db_path: str = get_default_db_path("db")) -> None:
        super().__init__(db_path)

    def _init_db(self) -> None:
        """Initialize the SQLite database with required tables if they don't exist."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS daily_prices (
                    ticker TEXT,
                    date DATE,
                    open REAL,
                    high REAL,
                    low REAL,
                    close REAL,
                    volume INTEGER,
                    PRIMARY KEY (ticker, date)
                )
            ''')
            
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
                CREATE TABLE IF NOT EXISTS realtime_prices (
                    ticker TEXT,
                    timestamp DATETIME, -- Stored as YYYY-MM-DD HH:MM:SS.ffffff
                    price REAL,
                    volume INTEGER, -- Trade volume for this tick
                    PRIMARY KEY (ticker, timestamp)
                )
            ''')
            conn.commit()

    def save_stock_data(self, df: pd.DataFrame, ticker: str, interval: str = 'daily') -> None:
        """Save aggregated (daily/hourly) stock data to the SQLite database."""
        with sqlite3.connect(self.db_path) as conn:
            df_copy = df.copy()
            if df_copy.empty:
                # print(f"Empty DataFrame provided for {ticker} ({interval}). Skipping DB save.")
                return
            df_copy.reset_index(inplace=True)
            
            df_copy.columns = [col.lower() for col in df_copy.columns]
            df_copy['ticker'] = ticker
            
            if interval == 'daily':
                table = 'daily_prices'
                date_col = 'date'
                if 'index' in df_copy.columns and date_col not in df_copy.columns:
                     df_copy.rename(columns={'index': date_col}, inplace=True)
                required_cols = ['ticker', 'date', 'open', 'high', 'low', 'close', 'volume']
            else:  # hourly
                table = 'hourly_prices'
                date_col = 'datetime'
                if 'index' in df_copy.columns and date_col not in df_copy.columns:
                     df_copy.rename(columns={'index': date_col}, inplace=True)
                required_cols = ['ticker', 'datetime', 'open', 'high', 'low', 'close', 'volume']
            
            # Filter only for columns that exist in df_copy to avoid KeyErrors
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

    def get_stock_data(self, ticker: str, start_date: str | None = None, end_date: str | None = None, 
                       interval: str = 'daily') -> pd.DataFrame:
        with sqlite3.connect(self.db_path) as conn:
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

    def save_realtime_data(self, df: pd.DataFrame, ticker: str) -> None:
        """Save realtime (tick) stock data to the SQLite database."""
        with sqlite3.connect(self.db_path) as conn:
            df_copy = df.copy()
            if df_copy.empty:
                # print(f"Empty DataFrame provided for realtime data of {ticker}. Skipping DB save.")
                return
            df_copy.reset_index(inplace=True)
            df_copy.columns = [col.lower() for col in df_copy.columns]
            df_copy['ticker'] = ticker

            # Standardize column names; expecting 'timestamp', 'price', 'volume' from input df index or columns
            # If index was named, it's now a column. If unnamed, it was 'index'.
            if 'timestamp' not in df_copy.columns and 'index' in df_copy.columns:
                df_copy.rename(columns={'index': 'timestamp'}, inplace=True)
            
            required_cols = ['ticker', 'timestamp', 'price', 'volume']
            df_copy = df_copy[[col for col in required_cols if col in df_copy.columns]]

            if 'timestamp' not in df_copy.columns or 'price' not in df_copy.columns or 'volume' not in df_copy.columns:
                print(f"Warning: Missing required columns (timestamp, price, volume) for realtime data of {ticker}. Skipping DB save.")
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
                DELETE FROM realtime_prices 
                WHERE ticker = ? 
                AND timestamp BETWEEN ? AND ?
            ''', (ticker, min_ts_str, max_ts_str))

            df_copy['timestamp'] = df_copy['timestamp'].dt.strftime(ts_format_str)
            df_copy.to_sql('realtime_prices', conn, if_exists='append', index=False)

    def get_realtime_data(self, ticker: str, start_datetime: str | None = None, end_datetime: str | None = None) -> pd.DataFrame:
        """Retrieve realtime (tick) stock data from the SQLite database."""
        with sqlite3.connect(self.db_path) as conn:
            query = "SELECT * FROM realtime_prices WHERE ticker = ?"
            params: list[str | None] = [ticker]

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

    def get_latest_price(self, ticker: str) -> float | None:
        """Get the most recent price for a ticker (realtime -> hourly -> daily)."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            latest_price = None

            # 1. Try realtime_prices
            try:
                cursor.execute("SELECT price FROM realtime_prices WHERE ticker = ? ORDER BY timestamp DESC LIMIT 1", (ticker,))
                row = cursor.fetchone()
                if row: latest_price = row[0]
            except sqlite3.Error as e: print(f"SQLite error (realtime_prices for {ticker}): {e}")

            # 2. Try hourly_prices if no realtime found
            if latest_price is None:
                try:
                    cursor.execute("SELECT close FROM hourly_prices WHERE ticker = ? ORDER BY datetime DESC LIMIT 1", (ticker,))
                    row = cursor.fetchone()
                    if row: latest_price = row[0]
                except sqlite3.Error as e: print(f"SQLite error (hourly_prices for {ticker}): {e}")

            # 3. Try daily_prices if no hourly found
            if latest_price is None:
                try:
                    cursor.execute("SELECT close FROM daily_prices WHERE ticker = ? ORDER BY date DESC LIMIT 1", (ticker,))
                    row = cursor.fetchone()
                    if row: latest_price = row[0]
                except sqlite3.Error as e: print(f"SQLite error (daily_prices for {ticker}): {e}")
            
        return latest_price

class StockDBDuckDB(StockDBBase):
    """
    A class to manage stock data storage and retrieval in a DuckDB database.
    """
    def __init__(self, db_path: str = get_default_db_path("duckdb")) -> None:
        super().__init__(db_path)

    def _init_db(self) -> None:
        """Initialize the DuckDB database with required tables if they don't exist."""
        with duckdb.connect(database=self.db_path, read_only=False) as conn:
            conn.execute('''
                CREATE TABLE IF NOT EXISTS daily_prices (
                    ticker VARCHAR,
                    date DATE,
                    open DOUBLE,
                    high DOUBLE,
                    low DOUBLE,
                    close DOUBLE,
                    volume BIGINT,
                    PRIMARY KEY (ticker, date)
                )
            ''')
            
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
                CREATE TABLE IF NOT EXISTS realtime_prices (
                    ticker VARCHAR,
                    timestamp TIMESTAMP, -- DuckDB TIMESTAMP has microsecond precision
                    price DOUBLE,
                    volume BIGINT,
                    PRIMARY KEY (ticker, timestamp)
                )
            ''')

    def save_stock_data(self, df: pd.DataFrame, ticker: str, interval: str = 'daily') -> None:
        """Save aggregated (daily/hourly) stock data to the DuckDB database."""
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
                 df_copy.rename(columns={'index': date_col}, inplace=True)
            
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
            
            conn.execute(f"DELETE FROM {table} WHERE ticker = ? AND {date_col} BETWEEN ? AND ?", 
                         (ticker, min_date_val, max_date_val))
            try:
                conn.register('df_to_insert', df_copy)
                cols_str = ", ".join([f'\"{col}\"'.lower() for col in df_copy.columns]) # Ensure quoting for safety
                conn.execute(f"INSERT INTO {table} ({cols_str}) SELECT {cols_str} FROM df_to_insert")
            except Exception as e:
                print(f"Error saving aggregated to DuckDB for {ticker} ({interval}): {e}")

    def get_stock_data(self, ticker: str, start_date: str | None = None, end_date: str | None = None, 
                       interval: str = 'daily') -> pd.DataFrame:
        with duckdb.connect(database=self.db_path, read_only=True) as conn:
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
            df = conn.execute(query, parameters=params).fetchdf() # type: ignore
        
        if not df.empty:
            df[date_col_name] = pd.to_datetime(df[date_col_name], errors='coerce')
            df.set_index(date_col_name, inplace=True)
            df = df[df.index.notna()]
        return df

    def save_realtime_data(self, df: pd.DataFrame, ticker: str) -> None:
        """Save realtime (tick) stock data to the DuckDB database."""
        with duckdb.connect(database=self.db_path, read_only=False) as conn:
            df_copy = df.copy()
            if df_copy.empty:
                # print(f"Empty DataFrame for realtime data of {ticker}. Skipping DuckDB save.")
                return
            df_copy.reset_index(inplace=True)
            df_copy.columns = [col.lower() for col in df_copy.columns]
            df_copy['ticker'] = ticker

            if 'timestamp' not in df_copy.columns and 'index' in df_copy.columns:
                df_copy.rename(columns={'index': 'timestamp'}, inplace=True)
            
            required_cols = ['ticker', 'timestamp', 'price', 'volume']
            df_copy = df_copy[[col for col in required_cols if col in df_copy.columns]]

            if 'timestamp' not in df_copy.columns or 'price' not in df_copy.columns or 'volume' not in df_copy.columns:
                print(f"Warning: Missing required columns for realtime data (DuckDB) of {ticker}. Skipping.")
                return

            df_copy['timestamp'] = pd.to_datetime(df_copy['timestamp']) # Ensure it's datetime64[ns]
            min_ts_val = df_copy['timestamp'].min()
            max_ts_val = df_copy['timestamp'].max()

            if pd.isna(min_ts_val) or pd.isna(max_ts_val):
                print(f"Warning: Min/max timestamp is NaT for realtime data (DuckDB) of {ticker}. Skipping.")
                return
            
            conn.execute("DELETE FROM realtime_prices WHERE ticker = ? AND timestamp BETWEEN ? AND ?",
                         (ticker, min_ts_val, max_ts_val))
            try:
                conn.register('df_rt_to_insert', df_copy)
                cols_str = ", ".join([f'\"{col}\"'.lower() for col in df_copy.columns])
                conn.execute(f"INSERT INTO realtime_prices ({cols_str}) SELECT {cols_str} FROM df_rt_to_insert")
            except Exception as e:
                print(f"Error saving realtime data to DuckDB for {ticker}: {e}")

    def get_realtime_data(self, ticker: str, start_datetime: str | None = None, end_datetime: str | None = None) -> pd.DataFrame:
        """Retrieve realtime (tick) stock data from the DuckDB database."""
        with duckdb.connect(database=self.db_path, read_only=True) as conn:
            query = "SELECT * FROM realtime_prices WHERE ticker = ?"
            params: list[str | pd.Timestamp | None] = [ticker]

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

    def get_latest_price(self, ticker: str) -> float | None:
        """Get the most recent price for a ticker (realtime -> hourly -> daily) from DuckDB."""
        with duckdb.connect(database=self.db_path, read_only=True) as conn:
            latest_price = None
            # 1. Try realtime_prices
            try:
                res_rt = conn.execute("SELECT price FROM realtime_prices WHERE ticker = ? ORDER BY timestamp DESC LIMIT 1", (ticker,)).fetchone()
                if res_rt: latest_price = res_rt[0]
            except duckdb.Error as e: print(f"DuckDB error (realtime_prices for {ticker}): {e}")

            # 2. Try hourly_prices
            if latest_price is None:
                try:
                    res_h = conn.execute("SELECT close FROM hourly_prices WHERE ticker = ? ORDER BY datetime DESC LIMIT 1", (ticker,)).fetchone()
                    if res_h: latest_price = res_h[0]
                except duckdb.Error as e: print(f"DuckDB error (hourly_prices for {ticker}): {e}")

            # 3. Try daily_prices
            if latest_price is None:
                try:
                    res_d = conn.execute("SELECT close FROM daily_prices WHERE ticker = ? ORDER BY date DESC LIMIT 1", (ticker,)).fetchone()
                    if res_d: latest_price = res_d[0]
                except duckdb.Error as e: print(f"DuckDB error (daily_prices for {ticker}): {e}")
            
        return latest_price


def get_stock_db(db_type: str, db_path: str | None = None) -> StockDBBase:
    """
    Factory function to get an instance of a stock database.

    Args:
        db_type: The type of database ('sqlite' or 'duckdb').
        db_path: The path to the database file. 
                 If None, uses default for the chosen db_type.

    Returns:
        An instance of StockDBBase.
    
    Raises:
        ValueError: If an unsupported db_type is provided.
    """
    actual_db_path = db_path
    if actual_db_path is None:
        actual_db_path = get_default_db_path("duckdb" if db_type.lower() == "duckdb" else "db")
    
    if db_type.lower() == "sqlite":
        return StockDBSQLite(actual_db_path)
    elif db_type.lower() == "duckdb":
        return StockDBDuckDB(actual_db_path)
    else:
        raise ValueError(f"Unsupported database type: {db_type}. Choose 'sqlite' or 'duckdb'.")

# Removed old standalone functions as all operations should go through class instances. 