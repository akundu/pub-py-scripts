from datetime import datetime, timedelta
import sqlite3
import duckdb
import pandas as pd
import re
import json # For LLM API interaction
import aiohttp # For async HTTP requests
import os
import asyncio 
import argparse
import sys
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.ticker import FuncFormatter
from typing import List, Tuple, Optional, Union
import numpy as np
from zoneinfo import ZoneInfo

# --- Configuration ---
# DATABASE_FILE = "your_stock_data.db" # Replace with your actual database file path

# Market hours constants
MARKET_OPEN_HOUR = 9
MARKET_OPEN_MINUTE = 30
MARKET_CLOSE_HOUR = 16
MARKET_CLOSE_MINUTE = 0
MARKET_TIMEZONE = 'America/New_York'

# --- Database Interaction ---
def get_db_type_from_file(db_file: str) -> str:
    """
    Determine database type from file extension.
    Returns 'duckdb' or 'sqlite' or raises ValueError if extension is not recognized.
    """
    ext = Path(db_file).suffix.lower()
    if ext == '.duckdb':
        return 'duckdb'
    elif ext in ['.db', '.sqlite']:
        return 'sqlite'
    else:
        raise ValueError(f"Unsupported database file extension: {ext}. Use .duckdb for DuckDB or .db/.sqlite for SQLite.")

def connect_db(db_file: str):
    """
    Connects to either SQLite or DuckDB database in read-only mode.
    Database type is determined by file extension.
    Returns a database connection object.
    """
    try:
        db_type = get_db_type_from_file(db_file)
        if db_type == 'duckdb':
            conn = duckdb.connect(db_file, read_only=True)
            print(f"Successfully connected to DuckDB database: {db_file}", file=sys.stderr)
        else:  # sqlite
            db_uri = f"file:{db_file}?mode=ro"
            conn = sqlite3.connect(db_uri, uri=True)
            print(f"Successfully connected to SQLite database: {db_file}", file=sys.stderr)
        return conn
    except Exception as e:
        print(f"Error connecting to database {db_file}: {e}")
        raise e

def fetch_data(conn: Union[sqlite3.Connection, duckdb.DuckDBPyConnection], query: str, params: Optional[dict] = None):
    """
    Fetches data from either SQLite or DuckDB database using a given query and parameters.
    Returns a pandas DataFrame.
    """
    try:
        if isinstance(conn, duckdb.DuckDBPyConnection):
            if params:
                # DuckDB uses ? for parameters, so we need to convert named parameters
                for key, value in params.items():
                    query = query.replace(f":{key}", f"'{value}'" if isinstance(value, str) else str(value))
            return conn.execute(query).df()
        else:  # SQLite
            return pd.read_sql_query(query, conn, params=params)
    except Exception as e:
        print(f"Error fetching data with query '{query}': {e}")
        return pd.DataFrame()

def get_daily_prices(conn: Union[sqlite3.Connection, duckdb.DuckDBPyConnection], 
                     ticker: str, 
                     start_date: Optional[str] = None, 
                     end_date: Optional[str] = None):
    """
    Fetches daily prices for a given ticker.
    Optionally filters by date range.
    """
    query = "SELECT * FROM daily_prices WHERE ticker = :ticker"
    params = {'ticker': ticker}
    if start_date:
        query += " AND date >= :start_date"
        params['start_date'] = start_date
    if end_date:
        query += " AND date <= :end_date"
        params['end_date'] = end_date
    query += " ORDER BY date ASC"
    df = fetch_data(conn, query, params)
    if not df.empty:
        df['date'] = pd.to_datetime(df['date'])
    return df

def get_hourly_prices(conn: Union[sqlite3.Connection, duckdb.DuckDBPyConnection], 
                      ticker: str, 
                      start_datetime: Optional[str] = None, 
                      end_datetime: Optional[str] = None):
    """
    Fetches hourly prices for a given ticker.
    Optionally filters by datetime range.
    """
    query = "SELECT * FROM hourly_prices WHERE ticker = :ticker"
    params = {'ticker': ticker}
    if start_datetime:
        query += " AND datetime >= :start_datetime"
        params['start_datetime'] = start_datetime
    if end_datetime:
        query += " AND datetime <= :end_datetime"
        params['end_datetime'] = end_datetime
    query += " ORDER BY datetime ASC"
    df = fetch_data(conn, query, params)
    if not df.empty:
        df['datetime'] = pd.to_datetime(df['datetime'])
    return df

def get_table_schema(conn: Union[sqlite3.Connection, duckdb.DuckDBPyConnection], db_type: str) -> str:
    """
    Fetch the schema of tables from the database.
    Returns a formatted string describing the schema.
    """
    schema_info = []
    
    if db_type == 'sqlite':
        # Get list of tables
        tables_query = "SELECT name FROM sqlite_master WHERE type='table'"
        tables = fetch_data(conn, tables_query)
        
        for table in tables['name']:
            # Get table schema
            schema_query = f"PRAGMA table_info({table})"
            columns = fetch_data(conn, schema_query)
            
            # Format column info
            columns_info = []
            for _, col in columns.iterrows():
                nullable = "Nullable" if col['notnull'] == 0 else "NOT NULL"
                pk = "PRIMARY KEY" if col['pk'] == 1 else ""
                col_info = f"   - {col['name']} {col['type']}: {col['dflt_value'] if col['dflt_value'] is not None else ''} {nullable} {pk}".strip()
                columns_info.append(col_info)
            
            schema_info.append(f"{table}:\n" + "\n".join(columns_info))
    
    else:  # duckdb
        # Get list of tables
        tables_query = "SELECT table_name FROM information_schema.tables WHERE table_schema = 'main'"
        tables = fetch_data(conn, tables_query)
        
        for table in tables['table_name']:
            # Get table schema
            schema_query = f"""
                SELECT 
                    column_name,
                    data_type,
                    is_nullable,
                    column_default
                FROM information_schema.columns 
                WHERE table_name = '{table}'
                ORDER BY ordinal_position
            """
            columns = fetch_data(conn, schema_query)
            
            # Format column info
            columns_info = []
            for _, col in columns.iterrows():
                nullable = "Nullable" if col['is_nullable'] == 'YES' else "NOT NULL"
                default = f"DEFAULT {col['column_default']}" if col['column_default'] is not None else ""
                col_info = f"   - {col['column_name']} {col['data_type']}: {default} {nullable}".strip()
                columns_info.append(col_info)
            
            schema_info.append(f"{table}:\n" + "\n".join(columns_info))
    
    return "\n\n".join(schema_info)

# --- LLM Based Strategy to SQL ---
async def generate_sql_from_strategy_llm(strategy_string: str, model_name: str = "gemini-2.0-flash", db_type: str = 'sqlite', db_connection: Union[sqlite3.Connection, duckdb.DuckDBPyConnection] = None):
    """
    Takes an English strategy string and uses an LLM to generate a SQL query.
    """
    print(f"\n--- Attempting to generate SQL for: '{strategy_string}' using LLM ---", file=sys.stderr )

    # Get date range from database
    if db_connection is None:
        raise ValueError("Database connection is required to fetch schema")
    
    # Get date range for both daily and hourly data
    date_ranges = {}
    for table in ['daily_prices', 'hourly_prices']:
        try:
            if db_type == 'sqlite':
                date_col = 'date' if table == 'daily_prices' else 'datetime'
                query = f"SELECT MIN({date_col}) as min_date, MAX({date_col}) as max_date FROM {table}"
            else:  # duckdb
                date_col = 'date' if table == 'daily_prices' else 'datetime'
                query = f"SELECT MIN({date_col}) as min_date, MAX({date_col}) as max_date FROM {table}"
            
            date_range = fetch_data(db_connection, query)
            if not date_range.empty:
                date_ranges[table] = {
                    'min_date': date_range['min_date'].iloc[0],
                    'max_date': date_range['max_date'].iloc[0]
                }
        except Exception as e:
            print(f"Warning: Could not fetch date range for {table}: {str(e)}")

    # Database-specific instructions
    db_specific_instructions = {
        'sqlite': """
        SQLite-specific instructions:
        - Use named parameters with :param_name syntax
        - Use SQLite date functions (e.g., date('now', '-N days'))
        - Use SQLite window functions (e.g., LAG, LEAD)
        - Use SQLite's CASE WHEN syntax
        - Available date range:
          Daily data: {daily_min} to {daily_max}
          Hourly data: {hourly_min} to {hourly_max}
        """.format(
            daily_min=date_ranges.get('daily_prices', {}).get('min_date', 'unknown'),
            daily_max=date_ranges.get('daily_prices', {}).get('max_date', 'unknown'),
            hourly_min=date_ranges.get('hourly_prices', {}).get('min_date', 'unknown'),
            hourly_max=date_ranges.get('hourly_prices', {}).get('max_date', 'unknown')
        ),
        'duckdb': """
        DuckDB-specific instructions:
        - Use positional parameters with ? syntax
        - Use DuckDB date functions (e.g., date_add('day', -N, current_date))
        - Use DuckDB window functions (e.g., LAG, LEAD)
        - Use DuckDB's CASE WHEN syntax
        - DuckDB supports more advanced analytics functions if needed
        - IMPORTANT: DuckDB does not allow window functions in WHERE clauses
          - Use CTEs (WITH clause) or subqueries to handle window functions
          - Example for MA crossover:
            WITH lagged_data AS (
              SELECT *,
                LAG(ma_10, 1) OVER (PARTITION BY ticker ORDER BY date) as prev_ma_10,
                LAG(ma_50, 1) OVER (PARTITION BY ticker ORDER BY date) as prev_ma_50
              FROM daily_prices
            )
            SELECT * FROM lagged_data
            WHERE prev_ma_10 <= prev_ma_50 AND ma_10 > ma_50
        - Available date range:
          Daily data: {daily_min} to {daily_max}
          Hourly data: {hourly_min} to {hourly_max}
        """.format(
            daily_min=date_ranges.get('daily_prices', {}).get('min_date', 'unknown'),
            daily_max=date_ranges.get('daily_prices', {}).get('max_date', 'unknown'),
            hourly_min=date_ranges.get('hourly_prices', {}).get('min_date', 'unknown'),
            hourly_max=date_ranges.get('hourly_prices', {}).get('max_date', 'unknown')
        )
    }

    # Get schema from database
    schema_info = get_table_schema(db_connection, db_type)

    # Detailed instructions for the LLM
    prompt = f"""
    You are an expert financial analyst and SQL query writer.
    Your task is to convert an English stock trading strategy into a single, executable SQL query for {db_type.upper()}.
    The query should be read-only (i.e., only SELECT statements) and should be made for {db_type.upper()} and should explicitly use the schema of the tables and explicit only provide the select statement.
    Assume today's date is {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}.

    {db_specific_instructions[db_type]}

    Available tables and their schemas:

    {schema_info}

    Instructions for query generation:
    - The query MUST be a single SELECT statement.
    - Identify the ticker, timeframe (daily/hourly), and conditions from the strategy.
    - For MA crossovers (e.g., "MA_10 crosses above MA_50"):
        - You'll need to compare the MAs on the current period and the previous period.
        - Use window functions (LAG) if appropriate or select consecutive rows.
        - Example logic for MA10 crossing above MA50:
          (prev.ma_10 <= prev.ma_50 AND curr.ma_10 > curr.ma_50)
    - For price level conditions (e.g., "close drops below $150"):
        - Compare the relevant price (usually 'close') with the target value.
    - Handle date/time references:
        - "latest" or "current" usually means the most recent record.
        - "today" refers to {datetime.now().strftime('%Y-%m-%d')}.
        - "last N days/hours" requires date arithmetic (e.g., date('now', '-N days')).
    - If the strategy implies an action (BUY/SELL), the query should select data points WHERE the conditions for that action are met.
    - IMPORTANT: Always include an 'action' column in your SELECT statement that indicates the action to take (e.g., 'BUY' or 'SELL').
    - Select relevant columns that would help verify the condition, including ticker, date/datetime, prices, and involved indicators.
    - If a strategy is too complex to be represented by a single SELECT query or is ambiguous, return "Error: Strategy too complex or ambiguous for a single SQL query."
    - IMPORTANT: Make sure to use dates within the available date range shown above. Do not use future dates or dates outside the available range.

    REQUIRED COLUMNS FOR PERFORMANCE CALCULATION:
    Your query MUST return exactly these columns in this order:
    1. date: The date and time of the signal (DATETIME type, format: YYYY-MM-DD HH:MM:SS)
    2. ticker: The stock symbol (TEXT type)
    3. close: The closing price at the time of the signal (REAL type)
    4. action: The action to take ('BUY' or 'SELL' as TEXT)
    5. buying_price: The price at which the position was bought (REAL type)
    6. buying_datetime: The exact date and time when the position was bought (DATETIME type, format: YYYY-MM-DD HH:MM:SS)

    IMPORTANT: For buying_price and buying_datetime:
    - These should be the price and time of the most recent BUY signal
    - Use window functions to track the last BUY signal
    - Example for SQLite:
      LAG(CASE WHEN action = 'BUY' THEN close ELSE NULL END, 1) OVER (
        PARTITION BY ticker 
        ORDER BY datetime
      ) as buying_price
    - Example for DuckDB:
      LAG(CASE WHEN action = 'BUY' THEN close ELSE NULL END, 1) OVER (
        PARTITION BY ticker 
        ORDER BY datetime
      ) as buying_price

    Example of correct column selection:
    WITH signals AS (
      SELECT 
        datetime as date,
        ticker,
        close,
        CASE WHEN condition THEN 'BUY' ELSE 'SELL' END as action
      FROM ...
    )
    SELECT 
      date,
      ticker,
      close,
      action,
      LAG(CASE WHEN action = 'BUY' THEN close ELSE NULL END, 1) OVER (
        PARTITION BY ticker 
        ORDER BY date
      ) as buying_price,
      LAG(CASE WHEN action = 'BUY' THEN date ELSE NULL END, 1) OVER (
        PARTITION BY ticker 
        ORDER BY date
      ) as buying_datetime
    FROM signals

    Note: 
    - For the first row of each ticker, buying_price and buying_datetime will be NULL, which is expected.
    - Always use the full datetime (YYYY-MM-DD HH:MM:SS) for both date and buying_datetime columns.
    - For hourly data, use the datetime column directly.
    - For daily data, convert the date to datetime by appending ' 00:00:00' or use the appropriate datetime function.

    User's Strategy: "{strategy_string}"

    Generated SQL Query:
    """

    print("\n--- LLM Prompt ---", file=sys.stderr)
    print(prompt, file=sys.stderr)
    print("--- End Prompt ---\n", file=sys.stderr)

    chat_history = [{"role": "user", "parts": [{"text": prompt}]}]
    payload = {"contents": chat_history}
    api_key = os.getenv('GEMINI_KEY')
    api_url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent?key={api_key}"

    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(api_url, headers={'Content-Type': 'application/json'}, data=json.dumps(payload)) as response:
                if response.status != 200:
                    error_text = await response.text()
                    print(f"LLM API request failed with status {response.status}: {error_text}")
                    return f"Error: LLM API request failed ({response.status})."
                result = await response.json()

        if (result is not None and 
            result.get('candidates') and result['candidates'][0].get('content') and
            result['candidates'][0]['content'].get('parts') and
            result['candidates'][0]['content']['parts'][0].get('text')):
            generated_text = result['candidates'][0]['content']['parts'][0]['text']
            # Clean up the response, sometimes LLMs wrap SQL in ```sql ... ```
            sql_query = generated_text.replace("```sql", "").replace("```", "").strip()
            
            # Remove any leading/trailing whitespace and newlines
            sql_query = sql_query.strip()
            
            # Remove any leading text that's not SQL (like "ite" in your case)
            if not sql_query.upper().startswith(("SELECT", "WITH")):
                # Find the first occurrence of SELECT or WITH
                select_pos = sql_query.upper().find("SELECT")
                with_pos = sql_query.upper().find("WITH")
                if select_pos != -1 and (with_pos == -1 or select_pos < with_pos):
                    sql_query = sql_query[select_pos:]
                elif with_pos != -1:
                    sql_query = sql_query[with_pos:]
                else:
                    print("LLM did not return a valid SELECT or WITH query.")
                    return "Error: LLM did not return a valid SELECT or WITH query."

            print(f"LLM Generated SQL: \n{sql_query}")

            # Basic validation: ensure it's a SELECT query
            if not (sql_query.upper().startswith("SELECT") or 
                    sql_query.upper().startswith("WITH")):
                print("LLM did not return a valid SELECT or WITH query.")
                return "Error: LLM did not return a valid SELECT or WITH query."
            return sql_query
        else:
            print("LLM response structure unexpected or content missing.")
            print(f"Full LLM response: {result}")
            return "Error: LLM response structure unexpected."
    except Exception as e:
        print(f"Error in LLM interaction: {str(e)}")
        return f"Error: {str(e)}"


# --- Strategy Parsing (Simplified Regex-based) ---
def parse_strategy_string(strategy_string):
    """
    Parses a simplified English strategy string using Regex.
    Returns a dictionary with parsed components or None if parsing fails.
    """
    parsed_strategy = {
        'action': None, 'ticker': None, 'conditions': [], 'timeframe': 'daily',
        'target_value': None, 'indicator_1': None, 'indicator_2': None, 'relationship': None
    }
    ticker_match = re.search(r"\b([A-Z]{1,5})\b", strategy_string)
    if ticker_match: parsed_strategy['ticker'] = ticker_match.group(1)

    if re.search(r"\b(buy|long)\b", strategy_string, re.IGNORECASE): parsed_strategy['action'] = "BUY"
    elif re.search(r"\b(sell|short)\b", strategy_string, re.IGNORECASE): parsed_strategy['action'] = "SELL"

    if re.search(r"\bhourly\b", strategy_string, re.IGNORECASE): parsed_strategy['timeframe'] = "hourly"
    elif re.search(r"\bdaily\b", strategy_string, re.IGNORECASE): parsed_strategy['timeframe'] = "daily"

    ma_cross_match = re.search(r"(ma_(\d+))\s+(crosses above|crosses below)\s+(ma_(\d+))", strategy_string, re.IGNORECASE)
    if ma_cross_match:
        parsed_strategy['indicator_1'] = ma_cross_match.group(1).lower()
        parsed_strategy['relationship'] = ma_cross_match.group(3).lower()
        parsed_strategy['indicator_2'] = ma_cross_match.group(4).lower()
        parsed_strategy['conditions'].append({
            'type': 'ma_cross', 'indicator1': parsed_strategy['indicator_1'],
            'indicator2': parsed_strategy['indicator_2'], 'relationship': parsed_strategy['relationship']
        })

    price_level_match = re.search(r"(price|close)\s+(drops below|is below|is above|exceeds|is less than|is greater than)\s+\$?(\d+\.?\d*)", strategy_string, re.IGNORECASE)
    if price_level_match:
        parsed_strategy['relationship'] = price_level_match.group(2).lower()
        parsed_strategy['target_value'] = float(price_level_match.group(3))
        parsed_strategy['conditions'].append({
            'type': 'price_level', 'metric': 'close',
            'relationship': parsed_strategy['relationship'], 'value': parsed_strategy['target_value']
        })

    if not parsed_strategy['action'] or not parsed_strategy['ticker']:
        print(f"Could not parse essential action or ticker from: {strategy_string}")
        return None
    # if not parsed_strategy['conditions']: # Conditions might be implicit in LLM approach
    #     print(f"No conditions identified in strategy: {strategy_string}")
    #     return None
    return parsed_strategy


# --- Strategy Execution Logic (Placeholder for Regex-based) ---
def execute_strategy_regex(conn, parsed_strategy):
    """
    Executes the parsed strategy (from regex parser).
    """
    if not parsed_strategy:
        print("Cannot execute: Strategy not parsed (regex).")
        return

    print(f"\n--- Executing REGEX Parsed Strategy ---")
    print(f"Parsed Strategy: {parsed_strategy}")

    ticker = parsed_strategy['ticker']
    timeframe = parsed_strategy['timeframe']
    
    if timeframe == 'daily':
        end_date = datetime.today().strftime('%Y-%m-%d')
        start_date = (datetime.today() - timedelta(days=365)).strftime('%Y-%m-%d')
        data_df = get_daily_prices(conn, ticker, start_date, end_date)
        price_col, date_col = 'close', 'date'
    elif timeframe == 'hourly':
        end_dt = datetime.now()
        start_dt = end_dt - timedelta(days=30)
        data_df = get_hourly_prices(conn, ticker, 
                                    start_dt.strftime('%Y-%m-%d %H:%M:%S'), 
                                    end_dt.strftime('%Y-%m-%d %H:%M:%S'))
        price_col, date_col = 'close', 'datetime'
    else:
        print(f"Unsupported timeframe: {timeframe}"); return

    if data_df.empty:
        print(f"No data found for {ticker} in {timeframe} timeframe."); return
    print(f"Fetched {len(data_df)} records for {ticker} ({timeframe}). Latest: {data_df[date_col].iloc[-1]}")

    signals = []
    for condition in parsed_strategy.get('conditions', []):
        if condition['type'] == 'ma_cross':
            if len(data_df) < 2: print("Not enough data for MA cross check."); continue
            ind1_col, ind2_col = condition['indicator1'], condition['indicator2']
            relationship = condition['relationship']

            for col in [ind1_col, ind2_col]:
                if col not in data_df.columns:
                    try:
                        window = int(col.split('_')[1])
                        data_df[col] = data_df[price_col].rolling(window=window).mean()
                        print(f"Calculated {col} on the fly.")
                    except Exception as e: print(f"Could not find/calculate {col}: {e}"); continue
            
            temp_df = data_df[[date_col, ind1_col, ind2_col]].dropna().tail(2)
            if len(temp_df) < 2: print("Not enough non-NaN data for MA cross."); continue
            
            prev_ind1, prev_ind2 = temp_df.iloc[0][ind1_col], temp_df.iloc[0][ind2_col]
            curr_ind1, curr_ind2 = temp_df.iloc[1][ind1_col], temp_df.iloc[1][ind2_col]
            curr_date = temp_df.iloc[1][date_col]
            
            signal_reason = ""
            if relationship == 'crosses above' and prev_ind1 <= prev_ind2 and curr_ind1 > curr_ind2:
                signal_reason = f"{ind1_col} ({curr_ind1:.2f}) X> {ind2_col} ({curr_ind2:.2f})"
            elif relationship == 'crosses below' and prev_ind1 >= prev_ind2 and curr_ind1 < curr_ind2:
                signal_reason = f"{ind1_col} ({curr_ind1:.2f}) X< {ind2_col} ({curr_ind2:.2f})"
            
            if signal_reason:
                print(f"Signal: {signal_reason} on {curr_date}")
                signals.append((curr_date, parsed_strategy['action'], ticker, signal_reason))
            # else: print(f"No '{relationship}' signal for {ind1_col} vs {ind2_col} on {curr_date}")

        elif condition['type'] == 'price_level':
            if data_df.empty: continue
            latest_price = data_df[price_col].iloc[-1]
            target_value, relationship = condition['value'], condition['relationship']
            latest_date = data_df[date_col].iloc[-1]
            signal_reason = ""
            if relationship in ['drops below', 'is below', 'is less than'] and latest_price < target_value:
                signal_reason = f"{price_col} ({latest_price:.2f}) {relationship} {target_value:.2f}"
            elif relationship in ['is above', 'exceeds', 'is greater than'] and latest_price > target_value:
                signal_reason = f"{price_col} ({latest_price:.2f}) {relationship} {target_value:.2f}"

            if signal_reason:
                print(f"Signal: {signal_reason} on {latest_date}")
                signals.append((latest_date, parsed_strategy['action'], ticker, signal_reason))
            # else: print(f"No '{relationship}' signal for {price_col} vs {target_value} on {latest_date}")

    if signals:
        print("\n--- REGEX Generated Signals ---")
        for sig_date, sig_action, sig_ticker, sig_reason in signals:
            print(f"{sig_date}: {sig_action} {sig_ticker} - Reason: {sig_reason}")
    else:
        print("No signals generated by REGEX strategy based on the latest data.")

async def execute_test_strategy(db_connection):
    """
    Executes the test strategy.
    """
    strategies_to_test = [
        "Buy AAPL when MA_10 crosses above MA_50 on daily chart",
        "Sell AAPL if hourly close drops below 150.5",
        "Show AAPL daily prices where close is greater than 150 and volume is above 110000 for the last 3 days", # More complex for LLM
        "Find days for MSFT where the 8-day EMA is above the 21-day EMA and the close price increased from the previous day" # Needs LAG
    ]

    for strategy_text in strategies_to_test:
        print(f"\n\n>>> Testing Strategy (Regex Parser): {strategy_text}", file=sys.stderr)
        parsed_regex = parse_strategy_string(strategy_text) # Regex parser
        if parsed_regex:
            execute_strategy_regex(db_connection, parsed_regex)
        else:
            print("Failed to parse strategy with Regex parser.")

        print(f"\n>>> Testing Strategy (LLM SQL Generation): {strategy_text}", file=sys.stderr)
        generated_sql = await generate_sql_from_strategy_llm(strategy_text)
        
        if generated_sql and not generated_sql.startswith("Error:"):
            print(f"--- Executing LLM Generated SQL ---")
            # IMPORTANT: Always review LLM-generated SQL before execution on a live database.
            # For this example, we'll try to execute it.
            llm_data_df = fetch_data(db_connection, generated_sql)
            if not llm_data_df.empty:
                print(f"LLM Query returned {len(llm_data_df)} rows:")
                print(llm_data_df.head())
            else:
                print("LLM Query returned no data or an error occurred during fetch.")
        else:
            print(f"Failed to generate or execute SQL from LLM: {generated_sql}")

def read_strategies_from_file(file_path: str) -> List[str]:
    """Read strategies from a file, where each paragraph is a separate strategy."""
    try:
        with open(file_path, 'r') as f:
            content = f.read()
            # Split by blank lines and filter out empty strings
            strategies = [s.strip() for s in content.split('\n\n') if s.strip()]
        return strategies
    except Exception as e:
        print(f"Error reading strategies file: {e}")
        return []

def is_market_hours(dt: datetime) -> bool:
    """Check if the given datetime is during market hours."""
    # Convert to market timezone
    market_dt = dt.astimezone(ZoneInfo(MARKET_TIMEZONE))
    
    # Check if it's a weekday
    if market_dt.weekday() >= 5:  # 5 = Saturday, 6 = Sunday
        return False
    
    # Check if it's during market hours
    market_time = market_dt.time()
    market_open = datetime.strptime(f"{MARKET_OPEN_HOUR}:{MARKET_OPEN_MINUTE}", "%H:%M").time()
    market_close = datetime.strptime(f"{MARKET_CLOSE_HOUR}:{MARKET_CLOSE_MINUTE}", "%H:%M").time()
    
    return market_open <= market_time <= market_close

def adjust_to_market_hours(start_dt: datetime, end_dt: datetime) -> Tuple[datetime, datetime]:
    """Adjust start and end times to market hours."""
    market_tz = ZoneInfo(MARKET_TIMEZONE)
    
    # Convert to market timezone
    start_dt = start_dt.astimezone(market_tz)
    end_dt = end_dt.astimezone(market_tz)
    
    # Adjust start time to market open if before
    if start_dt.time() < datetime.strptime(f"{MARKET_OPEN_HOUR}:{MARKET_OPEN_MINUTE}", "%H:%M").time():
        start_dt = start_dt.replace(hour=MARKET_OPEN_HOUR, minute=MARKET_OPEN_MINUTE)
    
    # Adjust end time to market close if after
    if end_dt.time() > datetime.strptime(f"{MARKET_CLOSE_HOUR}:{MARKET_CLOSE_MINUTE}", "%H:%M").time():
        end_dt = end_dt.replace(hour=MARKET_CLOSE_HOUR, minute=MARKET_CLOSE_MINUTE)
    
    return start_dt, end_dt

async def calculate_strategy_performance(
    db_connection: Union[sqlite3.Connection, duckdb.DuckDBPyConnection],
    strategy: str,
    investment_amount: float,
    start_date: datetime,
    end_date: datetime,
    timeframe: str = 'daily',
    signals_df: pd.DataFrame = None,
    db_type: str = 'sqlite',
    sql_query: Optional[str] = None
) -> Tuple[pd.DataFrame, float]:
    """
    Calculate the performance of a trading strategy.
    Can take a natural language strategy to convert to SQL or a direct SQL query.
    """
    if signals_df is None:
        # If an explicit SQL query is not provided, generate one from the NL strategy.
        if sql_query is None:
            sql_query = await generate_sql_from_strategy_llm(strategy, db_type=db_type, db_connection=db_connection)
            if sql_query.startswith("Error:"):
                print(f"Error generating SQL: {sql_query}")
                return pd.DataFrame(), 0.0
        
        # Execute the query (either provided directly or generated by LLM)
        print(f"\nExecuting Query:\n---\n{sql_query}\n---\n", file=sys.stderr)
        signals_df = fetch_data(db_connection, sql_query)
        if signals_df.empty:
            print("The query returned no signals.")
            # Keep the diagnostic info for when the query fails
            print("\nDiagnostic Information:")
            print(f"SQL Query:\n{sql_query}")
            print("\nChecking data availability:")
            for table in ['daily_prices', 'hourly_prices']:
                try:
                    date_col = 'date' if table == 'daily_prices' else 'datetime'
                    check_query = f"SELECT COUNT(*) as count FROM {table}"
                    count = fetch_data(db_connection, check_query)
                    print(f"- {table}: {count['count'].iloc[0]} rows")
                    
                    range_query = f"SELECT MIN({date_col}) as min_date, MAX({date_col}) as max_date FROM {table}"
                    date_range = fetch_data(db_connection, range_query)
                    if not date_range.empty:
                        min_d = date_range['min_date'].iloc[0]
                        max_d = date_range['max_date'].iloc[0]
                        print(f"  Date range: {min_d} to {max_d}")
                except Exception as e:
                    print(f"  Error checking {table}: {str(e)}")
            return pd.DataFrame(), 0.0
        
        # Print the SQL query results
        print("\n--- SQL Query Results ---")
        print(f"Number of signals found: {len(signals_df)}")
        print("\nFirst 5 rows of results:")
        print(signals_df.head().to_string())
        print("\nColumns in results:", signals_df.columns.tolist())
        print("--- End SQL Query Results ---\n")

    # Initialize performance tracking
    performance_data = []
    current_position = 0
    current_cash = investment_amount
    shares_held = 0

    # Sort by date to ensure chronological processing
    signals_df = signals_df.sort_values('date')

    for _, signal in signals_df.iterrows():
        if signal['action'] == 'BUY' and current_position == 0:
            # Calculate shares to buy
            shares_to_buy = current_cash / signal['close']
            shares_held = shares_to_buy
            current_cash = 0
            current_position = 1
            performance_data.append({
                'date': signal['date'],
                'action': 'BUY',
                'price': signal['close'],
                'shares': shares_held,
                'cash': current_cash,
                'position_value': shares_held * signal['close'],
                'total_value': current_cash + (shares_held * signal['close'])
            })
        elif signal['action'] == 'SELL' and current_position == 1:
            # Sell all shares
            current_cash = shares_held * signal['close']
            shares_held = 0
            current_position = 0
            performance_data.append({
                'date': signal['date'],
                'action': 'SELL',
                'price': signal['close'],
                'shares': shares_held,
                'cash': current_cash,
                'position_value': 0,
                'total_value': current_cash
            })

    # Create performance DataFrame
    performance_df = pd.DataFrame(performance_data)
    if not performance_df.empty:
        final_value = performance_df['total_value'].iloc[-1]
        final_return = ((final_value - investment_amount) / investment_amount) * 100
    else:
        final_value = investment_amount
        final_return = 0.0

    return performance_df, final_return

def plot_strategy_performance(
    performance_df: pd.DataFrame,
    investment_amount: float,
    start_date: datetime,
    end_date: datetime,
    timeframe: str = 'daily'
):
    """
    Plot the strategy performance over time.
    """
    if performance_df.empty:
        print("No performance data to plot.")
        return

    # Calculate returns if not present
    if 'returns' not in performance_df.columns:
        performance_df['returns'] = (performance_df['total_value'] - investment_amount) / investment_amount

    plt.figure(figsize=(15, 10))
    
    # Plot portfolio value
    plt.subplot(2, 1, 1)
    plt.plot(performance_df['date'], performance_df['total_value'], label='Portfolio Value', color='blue')
    plt.title('Strategy Performance Over Time')
    plt.ylabel('Portfolio Value ($)')
    plt.grid(True)
    
    # Format y-axis as currency
    plt.gca().yaxis.set_major_formatter(FuncFormatter(lambda x, p: f'${x:,.2f}'))
    
    # Format x-axis dates
    plt.gca().xaxis.set_major_locator(mdates.MonthLocator())
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    plt.xticks(rotation=45)
    
    # Plot returns
    plt.subplot(2, 1, 2)
    plt.plot(performance_df['date'], performance_df['returns'] * 100, label='Returns', color='green')
    plt.title('Strategy Returns Over Time')
    plt.ylabel('Returns (%)')
    plt.xlabel('Date')
    plt.grid(True)
    
    # Format y-axis as percentage
    plt.gca().yaxis.set_major_formatter(FuncFormatter(lambda x, p: f'{x:.1f}%'))
    
    # Format x-axis dates
    plt.gca().xaxis.set_major_locator(mdates.MonthLocator())
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.show()

def read_sql_query(sql_query_arg: Optional[str], sql_file_arg: Optional[str]) -> Optional[str]:
    """Reads SQL query from direct arg, file, or stdin."""
    if sql_query_arg:
        return sql_query_arg
    
    if sql_file_arg:
        if sql_file_arg == '-':
            print("Reading SQL query from stdin. Press Ctrl+D (or Ctrl+Z on Windows) to end.", file=sys.stderr)
            return sys.stdin.read()
        else:
            try:
                with open(sql_file_arg, 'r') as f:
                    return f.read()
            except FileNotFoundError:
                print(f"Error: SQL file not found at {sql_file_arg}", file=sys.stderr)
                sys.exit(1)
            except Exception as e:
                print(f"Error reading SQL file: {e}", file=sys.stderr)
                sys.exit(1)
    
    return None

def report_performance(
    performance_df: pd.DataFrame, 
    final_return: float, 
    stock: str, 
    strategy_text: str,
    args: argparse.Namespace, 
    start_date: Optional[datetime], 
    end_date: Optional[datetime]
):
    """Prints, exports, and plots the performance results."""
    if not performance_df.empty:
        print(f"\n>>> Strategy Performance Summary for {stock}:")
        print(f"Strategy: {strategy_text}")
        print(f"Investment Amount: ${args.investment_amount:,.2f}")
        start_str = start_date.strftime('%Y-%m-%d') if start_date else "N/A"
        end_str = end_date.strftime('%Y-%m-%d') if end_date else "N/A"
        print(f"Period: {start_str} to {end_str}")
        print(f"Timeframe: {args.timeframe}")
        print(f"Final Portfolio Value: ${performance_df['total_value'].iloc[-1]:,.2f}")
        print(f"Return: {final_return:.2f}%")
        
        if args.export_csv:
            strategy_summary = re.sub(r'[^\w-]', '', strategy_text.replace(" ", "_"))[:30]
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            csv_filename = f"/tmp/{stock}_{strategy_summary}_{timestamp}.csv"
            
            try:
                performance_df.to_csv(csv_filename, index=False)
                print(f"\nPerformance data exported to: {csv_filename}", file=sys.stderr)
            except Exception as e:
                print(f"\nError exporting to CSV: {str(e)}", file=sys.stderr)
        
        if args.plot:
            plot_strategy_performance(
                performance_df,
                args.investment_amount,
                start_date,
                end_date,
                args.timeframe
            )
    else:
        print(f"No performance data could be generated for {stock} with strategy: '{strategy_text}'")

    
def process_args():
    parser = argparse.ArgumentParser(description='Test trading strategies using LLM-generated or user-provided SQL queries.')
    parser.add_argument('--db-file', required=True, help='Path to the database file (.duckdb for DuckDB, .db/.sqlite for SQLite)')

    # Strategy input group - one and only one of these is required
    strategy_group = parser.add_mutually_exclusive_group(required=True)
    strategy_group.add_argument('--strategy', help='Trading strategy in natural language to test. Use {STOCK}, {START_DATE}, and {END_DATE} as placeholders.')
    strategy_group.add_argument('--strategy-file', help='File containing trading strategies to test. Use {STOCK}, {START_DATE}, and {END_DATE} as placeholders.')
    strategy_group.add_argument('--sql-query', help='Direct SQL query to execute. Use {STOCK}, {START_DATE}, and {END_DATE} as placeholders.')
    strategy_group.add_argument('--sql-file', help="File containing SQL query to execute. Use '-' to read from stdin. Use {STOCK}, {START_DATE}, and {END_DATE} as placeholders.")

    # Stock input - one of these is required
    stock_group = parser.add_mutually_exclusive_group(required=True)
    stock_group.add_argument('--stock', help='Single stock symbol to test the strategy on')
    stock_group.add_argument('--stock-list', help='YAML file containing a list of stock symbols to test the strategy on')

    parser.add_argument('--investment-amount', type=float, default=10000.0, help='Initial investment amount (default: 10000.0)')
    parser.add_argument('--start-date', help='Start date for backtesting (YYYY-MM-DD)')
    parser.add_argument('--end-date', help='End date for backtesting (YYYY-MM-DD)')
    parser.add_argument('--timeframe', choices=['daily', 'hourly'], default='daily',
                      help='Timeframe for analysis (default: daily)')
    parser.add_argument('--model', default='gemini-2.0-flash',
                      help='LLM model to use for SQL generation (default: gemini-2.0-flash)')
    parser.add_argument('--plot', action='store_true',
                      help='Plot strategy performance')
    parser.add_argument('--export-csv', 
                      help='Path to save the CSV file. If not provided, a file will be created in /tmp with a generated name.')
    parser.add_argument('--execute-only', action='store_true',
                      help='Only execute the SQL query and show results, skip performance calculation')
    parser.add_argument('--rows', type=int, default=5,
                      help='Number of rows to display in results (default: 5, use -1 for all rows)')

    args = parser.parse_args()
    
    # Convert dates to datetime objects
    start_date = datetime.strptime(args.start_date, '%Y-%m-%d') if args.start_date else None
    end_date = datetime.strptime(args.end_date, '%Y-%m-%d') if args.end_date else None
    
    return args, start_date, end_date

def get_output_filename(args: argparse.Namespace, stock: str, strategy_text: str = None) -> str:
    """Generate an output filename for CSV export."""
    if args.export_csv:
        # Use the specified filename
        output_path = args.export_csv
        # Create parent directory if it doesn't exist
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    else:
        # Generate a filename in /tmp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if strategy_text:
            # Clean up strategy text for filename
            strategy_part = re.sub(r'[^\w-]', '', strategy_text.replace(" ", "_"))[:30]
            base_name = f"{stock}_{strategy_part}_{timestamp}.csv"
        else:
            base_name = f"{stock}_sql_results_{timestamp}.csv"
        output_path = os.path.join('/tmp', base_name)
    
    return output_path

def export_to_csv(df: pd.DataFrame, filename: str) -> bool:
    """Export DataFrame to CSV file."""
    try:
        df.to_csv(filename, index=False)
        print(f"\nResults exported to: {filename}", file=sys.stderr)
        return True
    except Exception as e:
        print(f"\nError exporting to CSV: {str(e)}", file=sys.stderr)
        return False

def read_stock_list(file_path: str) -> List[str]:
    """Read stock symbols from a YAML file."""
    try:
        import yaml
        with open(file_path, 'r') as f:
            data = yaml.safe_load(f)
            if isinstance(data, dict) and 'symbols' in data:
                return data['symbols']
            else:
                print(f"Warning: Invalid YAML format in {file_path}. Expected a dictionary with 'symbols' key.")
                return []
    except Exception as e:
        print(f"Error reading stock list file: {e}")
        return []

def format_date_for_sql(date: Optional[datetime], db_type: str) -> str:
    """Format a datetime object for SQL based on database type."""
    if date is None:
        return "NULL"
    return f"{date.strftime('%Y-%m-%d')}"
    # if db_type == 'sqlite':
    #     return f"{date.strftime('%Y-%m-%d')}"
    # else:  # duckdb
    #     return f"DATE '{date.strftime('%Y-%m-%d')}'"

def replace_placeholders(text: str, stock: str, start_date: Optional[datetime], end_date: Optional[datetime], db_type: str) -> str:
    """Replace placeholders in text with actual values."""
    replacements = {
        "{STOCK}": stock,
        "{START_DATE}": format_date_for_sql(start_date, db_type),
        "{END_DATE}": format_date_for_sql(end_date, db_type)
    }
    
    result = text
    for placeholder, value in replacements.items():
        if placeholder in result:
            result = result.replace(placeholder, value)
    
    return result

def display_sql_results(df: pd.DataFrame, stock: str, query: str, num_rows: int = 5):
    """Display SQL query results in a formatted way."""
    print(f"\n=== SQL Results for {stock} ===")
    print(f"Query:\n{query}\n")
    print(f"Number of rows: {len(df)}")
    if not df.empty:
        if num_rows == -1:
            print("\nAll rows:")
            print(df.to_string())
        else:
            print(f"\nFirst {num_rows} rows:")
            print(df.head(num_rows).to_string())
        print("\nColumns:", df.columns.tolist())
    print("=" * 50)

async def execute_strategies(db_connection: Union[sqlite3.Connection, duckdb.DuckDBPyConnection], 
                           args: argparse.Namespace,
                           start_date: Optional[datetime],
                           end_date: Optional[datetime]):
    # Get stocks to test
    if args.stock:
        stocks = [args.stock]
    elif args.stock_list:
        stocks = read_stock_list(args.stock_list)
        if not stocks:
            print("No stocks found in the stock list file.", file=sys.stderr)
            return
    else:
        # Should not be reached due to arg group
        print("No stocks provided. Use --stock or --stock-list.", file=sys.stderr)
        return

    db_type = 'duckdb' if isinstance(db_connection, duckdb.DuckDBPyConnection) else 'sqlite'

    # Check for direct SQL input first
    sql_query = read_sql_query(args.sql_query, args.sql_file)

    if sql_query:
        print("\n" + "="*80)
        print("Executing provided SQL Query for specified stocks", file=sys.stderr)
        print("="*80 + "\n")

        for stock in stocks:
            print(f"\n--- Processing stock: {stock} ---")
            stock_specific_sql = replace_placeholders(sql_query, stock, start_date, end_date, db_type)
            
            # Check if any placeholders weren't replaced
            missing_placeholders = [p for p in ["{STOCK}", "{START_DATE}", "{END_DATE}"] if p in stock_specific_sql]
            if missing_placeholders:
                print(f"Warning: The following placeholders were not replaced: {', '.join(missing_placeholders)}", file=sys.stderr)
            print(f"query to execute: {stock_specific_sql}", file=sys.stderr)
            
            if args.execute_only:
                # Just execute and display results
                results_df = fetch_data(db_connection, stock_specific_sql)
                display_sql_results(results_df, stock, stock_specific_sql, args.rows)
                
                # Export to CSV if requested
                if args.export_csv:
                    output_file = get_output_filename(args, stock)
                    export_to_csv(results_df, output_file)
            else:
                # Calculate performance
                performance_df, final_return = await calculate_strategy_performance(
                    db_connection=db_connection,
                    strategy=f"Custom SQL for {stock}",
                    investment_amount=args.investment_amount,
                    start_date=start_date,
                    end_date=end_date,
                    timeframe=args.timeframe,
                    db_type=db_type,
                    sql_query=stock_specific_sql
                )
                
                report_performance(
                    performance_df, final_return, stock, f"Custom SQL for {stock}",
                    args, start_date, end_date
                )

    else: # Fallback to natural language strategy
        if args.strategy:
            strategies = [args.strategy]
        elif args.strategy_file:
            strategies = read_strategies_from_file(args.strategy_file)
        else:
            # Should not be reached
            print("No strategy or SQL query provided.", file=sys.stderr)
            return

        for strategy in strategies:
            print("\n" + "="*80)
            print(f"Testing Strategy: {strategy}", file=sys.stderr)
            print("="*80 + "\n")

            for stock in stocks:
                print(f"\n--- Processing stock: {stock} ---")
                
                stock_specific_strategy = replace_placeholders(strategy, stock, start_date, end_date, db_type)
                
                # Check if any placeholders weren't replaced
                missing_placeholders = [p for p in ["{STOCK}", "{START_DATE}", "{END_DATE}"] if p in stock_specific_strategy]
                if missing_placeholders:
                    print(f"Warning: The following placeholders were not replaced: {', '.join(missing_placeholders)}", file=sys.stderr)
                
                if args.execute_only:
                    # Generate SQL and execute
                    sql_query = await generate_sql_from_strategy_llm(stock_specific_strategy, db_type=db_type, db_connection=db_connection)
                    if sql_query.startswith("Error:"):
                        print(f"Error generating SQL: {sql_query}")
                        continue
                    
                    results_df = fetch_data(db_connection, sql_query)
                    display_sql_results(results_df, stock, sql_query, args.rows)
                    
                    # Export to CSV if requested
                    if args.export_csv:
                        output_file = get_output_filename(args, stock, stock_specific_strategy)
                        export_to_csv(results_df, output_file)
                else:
                    # Calculate performance
                    performance_df, final_return = await calculate_strategy_performance(
                        db_connection,
                        stock_specific_strategy,
                        args.investment_amount,
                        start_date,
                        end_date,
                        args.timeframe,
                        db_type=db_type
                    )
                    
                    report_performance(
                        performance_df, final_return, stock, stock_specific_strategy,
                        args, start_date, end_date
                    )

async def main():
    args, start_date, end_date = process_args()
    
    try:
        db_connection = connect_db(args.db_file)
        try:
            await execute_strategies(db_connection, args, start_date, end_date)
        finally:
            db_connection.close()
    except ValueError as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())