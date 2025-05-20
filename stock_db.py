import sqlite3
import pandas as pd
from datetime import datetime
import os
from abc import ABCMeta, abstractmethod
import duckdb

DEFAULT_SQLITE_PATH = "stock_data.db"
DEFAULT_DUCKDB_PATH = "stock_data.duckdb" # Can also be ":memory:"

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
        """Save stock data to the database."""
        pass

    @abstractmethod
    def get_stock_data(self, ticker: str, start_date: str | None = None, end_date: str | None = None, 
                       interval: str = 'daily') -> pd.DataFrame:
        """Retrieve stock data from the database."""
        pass

    @abstractmethod
    def get_latest_price(self, ticker: str) -> float | None:
        """Get the most recent price for a ticker."""
        pass

class StockDBSQLite(StockDBBase):
    """
    A class to manage stock data storage and retrieval in an SQLite database.
    """
    def __init__(self, db_path: str = DEFAULT_SQLITE_PATH) -> None:
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
                    datetime DATETIME,
                    open REAL,
                    high REAL,
                    low REAL,
                    close REAL,
                    volume INTEGER,
                    PRIMARY KEY (ticker, datetime)
                )
            ''')
            conn.commit()

    def save_stock_data(self, df: pd.DataFrame, ticker: str, interval: str = 'daily') -> None:
        """Save stock data to the SQLite database."""
        with sqlite3.connect(self.db_path) as conn:
            df_copy = df.copy()
            df_copy.reset_index(inplace=True)
            
            df_copy.columns = [col.lower() for col in df_copy.columns]
            df_copy['ticker'] = ticker
            
            if interval == 'daily':
                table = 'daily_prices'
                date_col = 'date'
                # Ensure 'date' is the name of the index after reset_index if it was the original index name
                if 'index' in df_copy.columns and date_col not in df_copy.columns: # Common if index was unnamed
                     df_copy.rename(columns={'index': date_col}, inplace=True)
                required_cols = ['ticker', 'date', 'open', 'high', 'low', 'close', 'volume']
                # Filter only for columns that exist in df_copy to avoid KeyErrors
                df_copy = df_copy[[col for col in required_cols if col in df_copy.columns]]
            else:  # hourly
                table = 'hourly_prices'
                date_col = 'datetime'
                if 'index' in df_copy.columns and date_col not in df_copy.columns:
                     df_copy.rename(columns={'index': date_col}, inplace=True)
                required_cols = ['ticker', 'datetime', 'open', 'high', 'low', 'close', 'volume']
                df_copy = df_copy[[col for col in required_cols if col in df_copy.columns]]

            if date_col not in df_copy.columns:
                print(f"Warning: Date column '{date_col}' not found in DataFrame for {ticker} ({interval}). Skipping DB save.")
                return

            # Convert to datetime objects first for min/max, then to string for SQL
            df_copy[date_col] = pd.to_datetime(df_copy[date_col])
            min_date_val = df_copy[date_col].min()
            max_date_val = df_copy[date_col].max()

            if pd.isna(min_date_val) or pd.isna(max_date_val):
                print(f"Warning: Min/max date is NaT for {ticker} ({interval}). Skipping DB save for this batch.")
                return

            if interval == 'daily':
                min_date_str = min_date_val.strftime('%Y-%m-%d')
                max_date_str = max_date_val.strftime('%Y-%m-%d')
            else: # hourly
                min_date_str = min_date_val.strftime('%Y-%m-%d %H:%M:%S')
                max_date_str = max_date_val.strftime('%Y-%m-%d %H:%M:%S')
            
            cursor = conn.cursor()
            cursor.execute(f'''
                DELETE FROM {table} 
                WHERE ticker = ? 
                AND {date_col} BETWEEN ? AND ?
            ''', (ticker, min_date_str, max_date_str))

            if interval == 'daily':
                df_copy[date_col] = df_copy[date_col].dt.strftime('%Y-%m-%d')
            else: # hourly
                df_copy[date_col] = df_copy[date_col].dt.strftime('%Y-%m-%d %H:%M:%S')
            
            df_copy.to_sql(table, conn, if_exists='append', index=False)

    def get_stock_data(self, ticker: str, start_date: str | None = None, end_date: str | None = None, 
                       interval: str = 'daily') -> pd.DataFrame:
        with sqlite3.connect(self.db_path) as conn:
            table = 'daily_prices' if interval == 'daily' else 'hourly_prices'
            date_col = 'date' if interval == 'daily' else 'datetime'
            
            query = f"SELECT * FROM {table} WHERE ticker = ?"
            params: list[str | None] = [ticker]
            
            if start_date:
                query += f" AND {date_col} >= ?"
                params.append(start_date)
            if end_date:
                query += f" AND {date_col} <= ?"
                params.append(end_date)
            
            query += f" ORDER BY {date_col}"
            
            df = pd.read_sql_query(query, conn, params=params) # type: ignore
        
        if not df.empty:
            # Convert the date/datetime column to datetime objects and set as index
            # The format might not be strictly necessary if ISO 8601 is used by SQLite,
            # but explicit is often safer.
            if interval == 'daily':
                df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
            else: # hourly
                df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
            
            df.set_index(date_col, inplace=True)
            df = df[df.index.notna()] # Remove rows where index conversion (to NaT) failed
        
        return df

    def get_latest_price(self, ticker: str) -> float | None:
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            result = None
            
            try:
                cursor.execute('''
                    SELECT close FROM hourly_prices 
                    WHERE ticker = ? 
                    ORDER BY datetime DESC LIMIT 1
                ''', (ticker,))
                result = cursor.fetchone()
            except sqlite3.Error as e:
                print(f"SQLite error querying hourly_prices for {ticker}: {e}")

            if not result:
                try:
                    cursor.execute('''
                        SELECT close FROM daily_prices 
                        WHERE ticker = ? 
                        ORDER BY date DESC LIMIT 1
                    ''', (ticker,))
                    result = cursor.fetchone()
                except sqlite3.Error as e:
                    print(f"SQLite error querying daily_prices for {ticker}: {e}")
            
        return result[0] if result else None

class StockDBDuckDB(StockDBBase):
    """
    A class to manage stock data storage and retrieval in a DuckDB database.
    """
    def __init__(self, db_path: str = DEFAULT_DUCKDB_PATH) -> None:
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

    def save_stock_data(self, df: pd.DataFrame, ticker: str, interval: str = 'daily') -> None:
        """Save stock data to the DuckDB database."""
        with duckdb.connect(database=self.db_path, read_only=False) as conn:
            df_copy = df.copy()
            df_copy.reset_index(inplace=True)
            
            df_copy.columns = [col.lower() for col in df_copy.columns]
            df_copy['ticker'] = ticker
            
            if interval == 'daily':
                table = 'daily_prices'
                date_col = 'date'
                if 'index' in df_copy.columns and date_col not in df_copy.columns:
                     df_copy.rename(columns={'index': date_col}, inplace=True)
                required_cols = ['ticker', 'date', 'open', 'high', 'low', 'close', 'volume']
                df_copy = df_copy[[col for col in required_cols if col in df_copy.columns]]
            else:  # hourly
                table = 'hourly_prices'
                date_col = 'datetime'
                if 'index' in df_copy.columns and date_col not in df_copy.columns:
                     df_copy.rename(columns={'index': date_col}, inplace=True)
                required_cols = ['ticker', 'datetime', 'open', 'high', 'low', 'close', 'volume']
                df_copy = df_copy[[col for col in required_cols if col in df_copy.columns]]

            if date_col not in df_copy.columns:
                print(f"Warning: Date column '{date_col}' not found in DataFrame for {ticker} ({interval}). Skipping DB save.")
                return

            # Ensure date_col is datetime for min/max operations
            df_copy[date_col] = pd.to_datetime(df_copy[date_col])

            min_date_val = df_copy[date_col].min()
            max_date_val = df_copy[date_col].max()

            if pd.isna(min_date_val) or pd.isna(max_date_val):
                print(f"Warning: Min/max date is NaT for {ticker} ({interval}). Skipping DB save for this batch.")
                return
            
            # DuckDB can often handle datetime objects directly in comparisons
            # No need to convert to string for the DELETE query if types are compatible
            # For DuckDB, DATE and TIMESTAMP types are used.
            
            conn.execute(f'''
                DELETE FROM {table} 
                WHERE ticker = ? 
                AND {date_col} BETWEEN ? AND ?
            ''', (ticker, min_date_val, max_date_val))

            # DuckDB can directly ingest pandas DataFrame with appropriate types
            # including datetime64[ns] for TIMESTAMP and DATE columns
            # No explicit string conversion of date_col needed before insert if it's already datetime
            try:
                # Ensure columns in df_copy match table schema if using direct insert from DataFrame object
                # For hourly, df_copy['datetime'] must be datetime64[ns]
                # For daily, df_copy['date'] must be datetime64[ns] (DuckDB will handle date part)
                conn.register('df_to_insert', df_copy)
                # Explicitly list columns to ensure order and match
                cols_str = ", ".join(df_copy.columns)
                conn.execute(f"INSERT INTO {table} ({cols_str}) SELECT {cols_str} FROM df_to_insert")
            except Exception as e:
                print(f"Error saving to DuckDB for {ticker} ({interval}): {e}")
                print("DataFrame columns:", df_copy.columns)
                print("DataFrame dtypes:", df_copy.dtypes)


    def get_stock_data(self, ticker: str, start_date: str | None = None, end_date: str | None = None, 
                       interval: str = 'daily') -> pd.DataFrame:
        with duckdb.connect(database=self.db_path, read_only=True) as conn:
            table = 'daily_prices' if interval == 'daily' else 'hourly_prices'
            date_col = 'date' if interval == 'daily' else 'datetime'
            
            query = f"SELECT * FROM {table} WHERE ticker = ?"
            params: list[str | pd.Timestamp | None] = [ticker]
            
            # Convert string dates to datetime objects for DuckDB query if needed, or pass as strings
            # DuckDB is good at casting standard date/timestamp strings
            if start_date:
                query += f" AND {date_col} >= ?"
                params.append(pd.to_datetime(start_date).strftime('%Y-%m-%d' if interval == 'daily' else '%Y-%m-%d %H:%M:%S'))
            if end_date:
                query += f" AND {date_col} <= ?"
                params.append(pd.to_datetime(end_date).strftime('%Y-%m-%d' if interval == 'daily' else '%Y-%m-%d %H:%M:%S'))
            
            query += f" ORDER BY {date_col}"
            
            df = conn.execute(query, parameters=params).fetchdf() # type: ignore
        
        if not df.empty:
            # DuckDB usually returns datetime types correctly for DATE and TIMESTAMP
            # but ensure it's set as index
            df[date_col] = pd.to_datetime(df[date_col], errors='coerce') # Ensure it's datetime
            df.set_index(date_col, inplace=True)
            df = df[df.index.notna()]
        
        return df

    def get_latest_price(self, ticker: str) -> float | None:
        with duckdb.connect(database=self.db_path, read_only=True) as conn:
            result = None
            
            try:
                res_hourly = conn.execute('''
                    SELECT close FROM hourly_prices 
                    WHERE ticker = ? 
                    ORDER BY datetime DESC LIMIT 1
                ''', (ticker,)).fetchone()
                if res_hourly:
                    result = res_hourly[0]
            except duckdb.Error as e:
                 print(f"DuckDB error querying hourly_prices for {ticker}: {e}")


            if result is None: # Check if result is still None, not just if res_hourly was None
                try:
                    res_daily = conn.execute('''
                        SELECT close FROM daily_prices 
                        WHERE ticker = ? 
                        ORDER BY date DESC LIMIT 1
                    ''', (ticker,)).fetchone()
                    if res_daily:
                        result = res_daily[0]
                except duckdb.Error as e:
                    print(f"DuckDB error querying daily_prices for {ticker}: {e}")

        return result if result is not None else None


def get_stock_db(db_type: str, db_path: str | None = None) -> StockDBBase:
    """
    Factory function to get an instance of a stock database.

    Args:
        db_type: The type of database ('sqlite' or 'duckdb').
        db_path: The path to the database file. 
                 If None, uses default for the chosen db_type.
                 For DuckDB, can be ":memory:" for an in-memory database.

    Returns:
        An instance of StockDBBase.
    
    Raises:
        ValueError: If an unsupported db_type is provided.
    """
    if db_type.lower() == "sqlite":
        return StockDBSQLite(db_path or DEFAULT_SQLITE_PATH)
    elif db_type.lower() == "duckdb":
        return StockDBDuckDB(db_path or DEFAULT_DUCKDB_PATH)
    else:
        raise ValueError(f"Unsupported database type: {db_type}. Choose 'sqlite' or 'duckdb'.")

# Removed old standalone functions as all operations should go through class instances. 