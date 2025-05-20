import sqlite3
import pandas as pd
from datetime import datetime
import os

DEFAULT_DB_PATH = "stock_data.db"

class StockDB:
    """
    A class to manage stock data storage and retrieval in an SQLite database.
    """
    def __init__(self, db_path: str = DEFAULT_DB_PATH) -> None:
        """
        Initializes the StockDB instance.

        Args:
            db_path: The path to the SQLite database file.
        """
        self.db_path = db_path
        self._init_db() # Initialize database and tables upon instantiation

    def _init_db(self) -> None:
        """Initialize the database with required tables if they don't exist."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Create tables for daily and hourly data
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
        """Save stock data to the database."""
        with sqlite3.connect(self.db_path) as conn:
            # Reset index to make date/datetime a column
            df_copy = df.copy() # Operate on a copy to avoid SettingWithCopyWarning if df is a slice
            df_copy.reset_index(inplace=True)
            
            # Rename columns to match database schema
            df_copy.columns = [col.lower() for col in df_copy.columns]
            
            # Add ticker column
            df_copy['ticker'] = ticker
            
            # Select and reorder columns to match database schema
            if interval == 'daily':
                table = 'daily_prices'
                date_col = 'date'
                # Ensure columns exist before selection to avoid KeyError
                required_cols = ['ticker', 'date', 'open', 'high', 'low', 'close', 'volume']
                df_copy = df_copy[[col for col in required_cols if col in df_copy.columns]]

            else:  # hourly
                table = 'hourly_prices'
                date_col = 'datetime'
                required_cols = ['ticker', 'datetime', 'open', 'high', 'low', 'close', 'volume']
                df_copy = df_copy[[col for col in required_cols if col in df_copy.columns]]

            if date_col not in df_copy.columns:
                print(f"Warning: Date column '{date_col}' not found in DataFrame for {ticker} ({interval}). Skipping DB save for this batch.")
                return

            # Convert timestamps to strings for BETWEEN clause and ensure they are datetime objects first
            df_copy[date_col] = pd.to_datetime(df_copy[date_col])
            if interval == 'daily':
                min_date_str = df_copy[date_col].min().strftime('%Y-%m-%d')
                max_date_str = df_copy[date_col].max().strftime('%Y-%m-%d')
            else: # hourly
                min_date_str = df_copy[date_col].min().strftime('%Y-%m-%d %H:%M:%S')
                max_date_str = df_copy[date_col].max().strftime('%Y-%m-%d %H:%M:%S')
            
            # Delete existing data for the date range
            cursor = conn.cursor()
            cursor.execute(f'''
                DELETE FROM {table} 
                WHERE ticker = ? 
                AND {date_col} BETWEEN ? AND ?
            ''', (ticker, min_date_str, max_date_str))
            # conn.commit() # Commit is handled by context manager upon exiting 'with' if no errors

            # Convert datetime column to string format before saving
            if interval == 'daily':
                df_copy[date_col] = df_copy[date_col].dt.strftime('%Y-%m-%d')
            else: # hourly
                df_copy[date_col] = df_copy[date_col].dt.strftime('%Y-%m-%d %H:%M:%S')
            
            # Save to database
            df_copy.to_sql(table, conn, if_exists='append', index=False)
            # conn.commit() # Commit handled by context manager

    def get_stock_data(self, ticker: str, start_date: str | None = None, end_date: str | None = None, 
                       interval: str = 'daily') -> pd.DataFrame:
        """Retrieve stock data from the database."""
        with sqlite3.connect(self.db_path) as conn:
            table = 'daily_prices' if interval == 'daily' else 'hourly_prices'
            date_col = 'date' if interval == 'daily' else 'datetime'
            
            query = f"SELECT * FROM {table} WHERE ticker = ?"
            params: list[str | None] = [ticker] # Ensure params list type
            
            if start_date:
                query += f" AND {date_col} >= ?"
                params.append(start_date)
            if end_date:
                query += f" AND {date_col} <= ?"
                params.append(end_date)
            
            query += f" ORDER BY {date_col}"
            
            df = pd.read_sql_query(query, conn, params=params) # type: ignore
        
        if not df.empty:
            # Convert the date/datetime column to datetime index
            if interval == 'daily':
                df[date_col] = pd.to_datetime(df[date_col], format='%Y-%m-%d', errors='coerce')
            else: # hourly
                df[date_col] = pd.to_datetime(df[date_col], format='%Y-%m-%d %H:%M:%S', errors='coerce')
            df.set_index(date_col, inplace=True)
            # df.dropna(subset=[df.index.name], inplace=True) # Remove rows where index conversion failed
            df = df[df.index.notna()] # Remove rows where index conversion (to NaT) failed
        
        return df

    def get_latest_price(self, ticker: str) -> float | None:
        """Get the most recent price for a ticker."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            result = None
            
            # Try hourly first
            cursor.execute('''
                SELECT close FROM hourly_prices 
                WHERE ticker = ? 
                ORDER BY datetime DESC LIMIT 1
            ''', (ticker,))
            result = cursor.fetchone()
            
            if not result:
                # Then try daily
                cursor.execute('''
                    SELECT close FROM daily_prices 
                    WHERE ticker = ? 
                    ORDER BY date DESC LIMIT 1
                ''', (ticker,))
                result = cursor.fetchone()
            
        return result[0] if result else None

# For potential direct use or testing, keep standalone functions as wrappers (optional)
# Or remove them if all usage will be through the class.
# For this refactor, we assume direct class usage elsewhere.

# def init_db(db_path: str = DEFAULT_DB_PATH) -> None:
#     db = StockDB(db_path)
#     # Initialization is now handled in __init__

# def save_stock_data(df: pd.DataFrame, ticker: str, interval: str = 'daily', db_path: str = DEFAULT_DB_PATH) -> None:
#     db = StockDB(db_path)
#     db.save_stock_data(df, ticker, interval)

# def get_stock_data(ticker: str, start_date: str | None = None, end_date: str | None = None, 
#                    interval: str = 'daily', db_path: str = DEFAULT_DB_PATH) -> pd.DataFrame:
#     db = StockDB(db_path)
#     return db.get_stock_data(ticker, start_date, end_date, interval)

# def get_latest_price(ticker: str, db_path: str = DEFAULT_DB_PATH) -> float | None:
#     db = StockDB(db_path)
#     return db.get_latest_price(ticker) 