import sqlite3
import pandas as pd
from datetime import datetime
import os

DEFAULT_DB_PATH = "stock_data.db"

def init_db(db_path: str = DEFAULT_DB_PATH) -> None:
    """Initialize the database with required tables if they don't exist."""
    conn = sqlite3.connect(db_path)
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
    conn.close()

def save_stock_data(df: pd.DataFrame, ticker: str, interval: str = 'daily', db_path: str = DEFAULT_DB_PATH) -> None:
    """Save stock data to the database."""
    conn = sqlite3.connect(db_path)
    
    # Reset index to make date/datetime a column
    df = df.reset_index()
    
    # Rename columns to match database schema
    df.columns = [col.lower() for col in df.columns]
    
    # Add ticker column
    df['ticker'] = ticker
    
    # Select and reorder columns to match database schema
    if interval == 'daily':
        table = 'daily_prices'
        date_col = 'date'
        df = df[['ticker', 'date', 'open', 'high', 'low', 'close', 'volume']]
    else:  # hourly
        table = 'hourly_prices'
        date_col = 'datetime'
        df = df[['ticker', 'datetime', 'open', 'high', 'low', 'close', 'volume']]
    
    # Convert timestamps to strings
    min_date = df[date_col].min().strftime('%Y-%m-%d %H:%M:%S')
    max_date = df[date_col].max().strftime('%Y-%m-%d %H:%M:%S')
    
    # Delete existing data for the date range
    cursor = conn.cursor()
    cursor.execute(f'''
        DELETE FROM {table} 
        WHERE ticker = ? 
        AND {date_col} BETWEEN ? AND ?
    ''', (ticker, min_date, max_date))
    conn.commit()
    
    # Convert datetime column to string format before saving
    df[date_col] = df[date_col].dt.strftime('%Y-%m-%d %H:%M:%S')
    
    # Save to database
    df.to_sql(table, conn, if_exists='append', index=False)
    conn.close()

def get_stock_data(ticker: str, start_date: str | None = None, end_date: str | None = None, 
                   interval: str = 'daily', db_path: str = DEFAULT_DB_PATH) -> pd.DataFrame:
    """Retrieve stock data from the database."""
    conn = sqlite3.connect(db_path)
    
    table = 'daily_prices' if interval == 'daily' else 'hourly_prices'
    date_col = 'date' if interval == 'daily' else 'datetime'
    
    query = f"SELECT * FROM {table} WHERE ticker = ?"
    params = [ticker]
    
    if start_date:
        query += f" AND {date_col} >= ?"
        params.append(start_date)
    if end_date:
        query += f" AND {date_col} <= ?"
        params.append(end_date)
    
    query += f" ORDER BY {date_col}"
    
    df = pd.read_sql_query(query, conn, params=params)
    conn.close()
    
    if not df.empty:
        # Convert the date/datetime column to datetime index using ISO8601 format
        df[date_col] = pd.to_datetime(df[date_col], format='ISO8601')
        df.set_index(date_col, inplace=True)
    
    return df

def get_latest_price(ticker: str, db_path: str = DEFAULT_DB_PATH) -> float | None:
    """Get the most recent price for a ticker."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Try hourly first, then daily
    cursor.execute('''
        SELECT close, datetime FROM hourly_prices 
        WHERE ticker = ? 
        ORDER BY datetime DESC LIMIT 1
    ''', (ticker,))
    
    result = cursor.fetchone()
    
    if not result:
        cursor.execute('''
            SELECT close, date FROM daily_prices 
            WHERE ticker = ? 
            ORDER BY date DESC LIMIT 1
        ''', (ticker,))
        result = cursor.fetchone()
    
    conn.close()
    return result[0] if result else None 