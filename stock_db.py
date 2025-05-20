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

# ---- New Code for Client-Server Architecture ----

class StockDBServer:
    """
    A conceptual server class that wraps a StockDBBase instance and handles
    requests to perform database operations.
    """
    def __init__(self, db_instance: StockDBBase):
        if not isinstance(db_instance, StockDBBase):
            raise ValueError("db_instance must be an instance of StockDBBase.")
        self.db = db_instance
        print(f"StockDBServer initialized with {type(db_instance).__name__} at {db_instance.db_path}")

    def handle_save_stock_data(self, data: dict) -> dict:
        """Handles request to save aggregated stock data."""
        try:
            df = pd.read_json(data['df_json'], orient='split')
            ticker = data['ticker']
            interval = data['interval']
            self.db.save_stock_data(df, ticker, interval)
            return {"status": "success", "message": f"Data saved for {ticker} ({interval})."}
        except Exception as e:
            return {"status": "error", "message": str(e)}

    def handle_get_stock_data(self, data: dict) -> dict:
        """Handles request to retrieve aggregated stock data."""
        try:
            ticker = data['ticker']
            start_date = data.get('start_date')
            end_date = data.get('end_date')
            interval = data.get('interval', 'daily')
            df = self.db.get_stock_data(ticker, start_date, end_date, interval)
            return {"status": "success", "df_json": df.to_json(orient='split', date_format='iso') if not df.empty else None}
        except Exception as e:
            return {"status": "error", "message": str(e)}

    def handle_save_realtime_data(self, data: dict) -> dict:
        """Handles request to save realtime stock data."""
        try:
            df = pd.read_json(data['df_json'], orient='split')
            ticker = data['ticker']
            # Ensure the index is a DatetimeIndex after deserialization from JSON
            if isinstance(df.index, pd.RangeIndex) and 'timestamp' in df.columns: # common if index was reset before to_json
                 df['timestamp'] = pd.to_datetime(df['timestamp'])
                 df = df.set_index('timestamp')
            elif not isinstance(df.index, pd.DatetimeIndex):
                 # If index is not DatetimeIndex and 'timestamp' col doesn't exist or didn't fix it,
                 # this might indicate an issue or a different DataFrame structure than expected.
                 # For now, we'll attempt to convert the existing index if it's not already a DatetimeIndex.
                 df.index = pd.to_datetime(df.index)

            self.db.save_realtime_data(df, ticker)
            return {"status": "success", "message": f"Realtime data saved for {ticker}."}
        except Exception as e:
            return {"status": "error", "message": str(e)}

    def handle_get_realtime_data(self, data: dict) -> dict:
        """Handles request to retrieve realtime stock data."""
        try:
            ticker = data['ticker']
            start_datetime = data.get('start_datetime')
            end_datetime = data.get('end_datetime')
            df = self.db.get_realtime_data(ticker, start_datetime, end_datetime)
            return {"status": "success", "df_json": df.to_json(orient='split', date_format='iso') if not df.empty else None}
        except Exception as e:
            return {"status": "error", "message": str(e)}

    def handle_get_latest_price(self, data: dict) -> dict:
        """Handles request to get the latest price for a ticker."""
        try:
            ticker = data['ticker']
            price = self.db.get_latest_price(ticker)
            return {"status": "success", "price": price}
        except Exception as e:
            return {"status": "error", "message": str(e)}

    # Note: _init_db is an internal method of StockDBBase instances and
    # typically wouldn't be exposed or controlled via a server API directly.
    # The server assumes the underlying DB is already initialized when the StockDBServer is created.

class StockDBClient(StockDBBase):
    """
    A client class that implements the StockDBBase interface by sending
    requests to a StockDBServer over a network.
    """
    def __init__(self, host: str, port: int):
        self.host = host
        self.port = port
        # db_path is not directly used by client as it relies on server's DB,
        # but we need to satisfy StockDBBase constructor.
        super().__init__(db_path=f"remote_db_at_{host}:{port}")
        print(f"StockDBClient initialized, configured to connect to server at {self.host}:{self.port}")

    def _init_db(self) -> None:
        """
        Initialization is handled by the server-side database instance.
        Client typically doesn't initialize the DB directly.
        """
        # print("Client-side _init_db called: DB initialization is managed by the server.")
        pass # Server handles DB initialization

    def _make_network_request(self, endpoint: str, payload: dict) -> dict:
        """
        Placeholder for making a network request to the StockDBServer.
        In a real implementation, this would use libraries like 'requests' for HTTP
        or 'socket' for direct TCP communication.

        Args:
            endpoint: The API endpoint path on the server (e.g., "/save_realtime_data").
            payload: A dictionary containing the data to send to the server.

        Returns:
            A dictionary representing the server's JSON response.

        Raises:
            NotImplementedError: As this is a placeholder.
            # In a real implementation, this would also handle network errors,
            # connection issues, timeouts, and non-success HTTP status codes.
        """
        # Example structure if using 'requests' library:
        # import requests
        # import json
        # url = f"http://{self.host}:{self.port}{endpoint}"
        # try:
        #     response = requests.post(url, json=payload, timeout=10) # 10-second timeout
        #     response.raise_for_status() # Raises an HTTPError for bad responses (4XX or 5XX)
        #     return response.json()
        # except requests.exceptions.RequestException as e:
        #     # Handle various network errors (ConnectionError, Timeout, TooManyRedirects, etc.)
        #     print(f"Network request to {url} failed: {e}")
        #     # Re-raise as a custom exception or return an error structure
        #     raise ConnectionError(f"Failed to connect or communicate with server at {url}: {e}") from e
        # except json.JSONDecodeError as e:
        #     print(f"Failed to decode JSON response from {url}: {e}")
        #     raise ValueError(f"Invalid JSON response from server: {e}") from e

        raise NotImplementedError(
            "_make_network_request is not implemented. "
            "This method should handle the actual network communication (e.g., HTTP POST) "
            "to the StockDBServer using a library like 'requests' or 'socket'."
        )

    def _handle_server_response_df(self, response: dict) -> pd.DataFrame:
        """Helper to process server response for DataFrame-returning methods."""
        if response["status"] == "success":
            if response["df_json"]:
                df = pd.read_json(response["df_json"], orient='split')
                # Ensure DatetimeIndex for consistency, especially for time-series data
                if 'date' in df.columns and df['date'].dtype == 'object': # For get_stock_data
                    df['date'] = pd.to_datetime(df['date'])
                    df = df.set_index('date')
                elif 'datetime' in df.columns and df['datetime'].dtype == 'object': # For get_stock_data (hourly)
                     df['datetime'] = pd.to_datetime(df['datetime'])
                     df = df.set_index('datetime')
                elif 'timestamp' in df.columns and df['timestamp'].dtype == 'object': # For get_realtime_data
                     df['timestamp'] = pd.to_datetime(df['timestamp'])
                     df = df.set_index('timestamp')
                elif df.index.dtype == 'int64' and isinstance(df.index, pd.RangeIndex) : # Default from to_json if index was simple range
                    # This case means the original df had a simple integer index.
                    # If specific index handling is needed (like a named index that became a column),
                    # it should be handled by ensuring 'orient' and 'date_format' in to_json/read_json
                    # preserve it, or by specific logic here if it's standardized.
                    # For now, we assume if it's a RangeIndex, it might not need to be a DatetimeIndex
                    # unless columns like 'date', 'datetime', 'timestamp' are present (handled above).
                    pass
                elif not isinstance(df.index, pd.DatetimeIndex) and df.index.dtype != 'int64':
                    # If it's not a DatetimeIndex and not a simple integer RangeIndex, try to convert.
                    # This handles cases where the index was a date/time string.
                    try:
                        df.index = pd.to_datetime(df.index)
                    except Exception:
                        print(f"Client: Could not convert index to DatetimeIndex for df: {df.head()}")

                return df
            return pd.DataFrame() # Return empty DataFrame if df_json is None
        else:
            raise Exception(f"Server error: {response['message']}")

    def _handle_server_response_generic(self, response: dict, success_key: str | None = None):
        """Helper to process server response for non-DataFrame methods."""
        if response["status"] == "success":
            return response.get(success_key) if success_key else True
        else:
            raise Exception(f"Server error: {response['message']}")

    def save_stock_data(self, df: pd.DataFrame, ticker: str, interval: str = 'daily') -> None:
        request_data = {
            "df_json": df.to_json(orient='split', date_format='iso'),
            "ticker": ticker,
            "interval": interval
        }
        # response = self.server.handle_save_stock_data(request_data)
        response = self._make_network_request("/save_stock_data", request_data)
        self._handle_server_response_generic(response)

    def get_stock_data(self, ticker: str, start_date: str | None = None, end_date: str | None = None, 
                       interval: str = 'daily') -> pd.DataFrame:
        request_data = {
            "ticker": ticker,
            "start_date": start_date,
            "end_date": end_date,
            "interval": interval
        }
        # response = self.server.handle_get_stock_data(request_data)
        response = self._make_network_request("/get_stock_data", request_data)
        return self._handle_server_response_df(response)

    def save_realtime_data(self, df: pd.DataFrame, ticker: str) -> None:
        # For DataFrames with DatetimeIndex, to_json with orient='split' handles it well.
        # Make sure the DataFrame is structured as expected by the server's handler.
        # If df.index is DatetimeIndex, it should be fine.
        # If it's in a column 'timestamp', that also needs to be handled.
        # The server-side handler for save_realtime_data has logic to set_index if 'timestamp' column exists.
        df_copy = df.copy()
        if isinstance(df_copy.index, pd.DatetimeIndex) and df_copy.index.name == 'timestamp':
            df_copy = df_copy.reset_index() # ensure 'timestamp' is a column for to_json if it was the index
        
        request_data = {
            "df_json": df_copy.to_json(orient='split', date_format='iso'), # Sending index=True by default with split
            "ticker": ticker
        }
        # response = self.server.handle_save_realtime_data(request_data)
        response = self._make_network_request("/save_realtime_data", request_data)
        self._handle_server_response_generic(response)

    def get_realtime_data(self, ticker: str, start_datetime: str | None = None, end_datetime: str | None = None) -> pd.DataFrame:
        request_data = {
            "ticker": ticker,
            "start_datetime": start_datetime,
            "end_datetime": end_datetime
        }
        # response = self.server.handle_get_realtime_data(request_data)
        response = self._make_network_request("/get_realtime_data", request_data)
        return self._handle_server_response_df(response)

    def get_latest_price(self, ticker: str) -> float | None:
        request_data = {"ticker": ticker}
        # response = self.server.handle_get_latest_price(request_data)
        response = self._make_network_request("/get_latest_price", request_data)
        return self._handle_server_response_generic(response, success_key="price")


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
