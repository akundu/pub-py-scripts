import sqlite3
import pandas as pd
import re
from datetime import datetime, timedelta
import json # For LLM API interaction

# --- Configuration ---
# DATABASE_FILE = "your_stock_data.db" # Replace with your actual database file path

# --- Database Interaction ---
def connect_db(db_file):
    """
    Connects to the SQLite database in read-only mode.
    Returns a database connection object.
    """
    try:
        db_uri = f"file:{db_file}?mode=ro"
        conn = sqlite3.connect(db_uri, uri=True)
        print(f"Successfully connected to {db_file} in read-only mode.")
        return conn
    except sqlite3.Error as e:
        print(f"Error connecting to database: {e}")
        return None

def fetch_data(conn, query, params=None):
    """
    Fetches data from the database using a given query and parameters.
    Returns a pandas DataFrame.
    """
    try:
        return pd.read_sql_query(query, conn, params=params)
    except Exception as e:
        print(f"Error fetching data with query '{query}': {e}")
        return pd.DataFrame()

def get_daily_prices(conn, ticker, start_date=None, end_date=None):
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

def get_hourly_prices(conn, ticker, start_datetime=None, end_datetime=None):
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

# --- LLM Based Strategy to SQL ---
async def generate_sql_from_strategy_llm(strategy_string):
    """
    Takes an English strategy string and uses an LLM to generate an SQLite query.
    """
    print(f"\n--- Attempting to generate SQL for: '{strategy_string}' using LLM ---")

    # Detailed instructions for the LLM
    # It's crucial to provide the schema and desired output format.
    prompt = f"""
    You are an expert financial analyst and SQLite query writer.
    Your task is to convert an English stock trading strategy into a single, executable SQLite query.
    The query should be read-only (i.e., only SELECT statements).
    Assume today's date is {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}.

    Available SQLite tables and their schemas:

    1. daily_prices:
       - ticker TEXT: Stock symbol (e.g., 'AAPL')
       - date DATE: Trading date (YYYY-MM-DD)
       - open REAL: Opening price
       - high REAL: Highest price
       - low REAL: Lowest price
       - close REAL: Closing price
       - volume INTEGER: Trading volume
       - ma_10 REAL: 10-day simple moving average of close price
       - ma_50 REAL: 50-day simple moving average of close price
       - ma_100 REAL: 100-day simple moving average of close price
       - ma_200 REAL: 200-day simple moving average of close price
       - ema_8 REAL: 8-day exponential moving average
       - ema_21 REAL: 21-day exponential moving average
       (Other EMAs: ema_34, ema_55, ema_89)
       PRIMARY KEY (ticker, date)

    2. hourly_prices:
       - ticker TEXT: Stock symbol
       - datetime DATETIME: Trading date and time (YYYY-MM-DD HH:MM:SS)
       - open REAL
       - high REAL
       - low REAL
       - close REAL
       - volume INTEGER
       PRIMARY KEY (ticker, datetime)

    3. realtime_data: (Less likely needed for historical strategy analysis, but available)
       - ticker TEXT
       - timestamp DATETIME (YYYY-MM-DD HH:MM:SS.ffffff)
       - type TEXT ('quote' or 'trade')
       - price REAL
       - size INTEGER
       - ask_price REAL (Nullable)
       - ask_size INTEGER (Nullable)
       PRIMARY KEY (ticker, timestamp, type)

    Instructions for query generation:
    - The query MUST be a single SQLite SELECT statement.
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
    - Select relevant columns that would help verify the condition, including ticker, date/datetime, prices, and involved indicators.
    - If a strategy is too complex to be represented by a single SELECT query or is ambiguous, return "Error: Strategy too complex or ambiguous for a single SQL query."

    User's Strategy: "{strategy_string}"

    Generated SQLite Query:
    """

    chat_history = [{"role": "user", "parts": [{"text": prompt}]}]
    payload = {"contents": chat_history}
    api_key = "" # Provided by the environment
    api_url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={api_key}"

    try:
        response = await fetch(api_url, { # Assuming 'fetch' is available in the environment for async calls
            'method': 'POST',
            'headers': {'Content-Type': 'application/json'},
            'body': json.dumps(payload)
        })
        
        if not response.ok:
            error_text = await response.text()
            print(f"LLM API request failed with status {response.status}: {error_text}")
            return f"Error: LLM API request failed ({response.status})."

        result = await response.json()

        if (result.get('candidates') and result['candidates'][0].get('content') and
            result['candidates'][0]['content'].get('parts') and
            result['candidates'][0]['content']['parts'][0].get('text')):
            generated_text = result['candidates'][0]['content']['parts'][0]['text']
            # Clean up the response, sometimes LLMs wrap SQL in ```sql ... ```
            sql_query = generated_text.replace("```sql", "").replace("```", "").strip()
            print(f"LLM Generated SQL: \n{sql_query}")
            
            # Basic validation: ensure it's a SELECT query
            if not sql_query.upper().startswith("SELECT"):
                print("LLM did not return a valid SELECT query.")
                return "Error: LLM did not return a valid SELECT query."
            return sql_query
        else:
            print("LLM response structure unexpected or content missing.")
            print(f"Full LLM response: {result}")
            return "Error: LLM response structure unexpected."
    except Exception as e:
        print(f"Error during LLM API call or processing: {e}")
        return f"Error: Exception during LLM call - {e}"


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

# --- Main Program Flow ---
# Note: The `generate_sql_from_strategy_llm` is an async function.
# To run it in a synchronous script like this, you'd typically use asyncio.run()
# For simplicity in this example, we'll call it and it will print,
# but true async integration needs an event loop.
import asyncio 

async def main(): # Make main async to use await
    # Dummy DB setup (run once separately if needed)
    # ... (dummy DB setup code from previous version can be placed here if you run this file directly)
    # conn_setup = sqlite3.connect('test_stock_data.db'); cursor_setup = conn_setup.cursor()
    # cursor_setup.execute("CREATE TABLE IF NOT EXISTS daily_prices (ticker TEXT, date DATE, open REAL, high REAL, low REAL, close REAL, volume INTEGER, ma_10 REAL, ma_50 REAL, ma_100 REAL, ma_200 REAL, ema_8 REAL, ema_21 REAL, ema_34 REAL, ema_55 REAL, ema_89 REAL, PRIMARY KEY (ticker, date))")
    # cursor_setup.execute("CREATE TABLE IF NOT EXISTS hourly_prices (ticker TEXT, datetime DATETIME, open REAL, high REAL, low REAL, close REAL, volume INTEGER, PRIMARY KEY (ticker, datetime))")
    # daily_data = [('AAPL', '2023-01-01', 150, 152, 149, 151, 100000, 150.5, 148.0, None, None, None, None, None, None, None), ('AAPL', '2023-01-02', 151.2, 153, 150.5, 152.5, 120000, 150.8, 148.5, None, None, None, None, None, None, None), ('AAPL', '2023-01-03', 152, 152.5, 148, 149, 120000, 150.9, 148.8, None, None, None, None, None, None, None), ('AAPL', '2023-01-04', 149.5, 151, 148.5, 150.5, 110000, 150.7, 149.2, None, None, None, None, None, None, None), ('AAPL', '2023-01-05', 150, 155, 150, 154.5, 130000, 151.5, 150.0, None, None, None, None, None, None, None)]
    # hourly_data = [('AAPL', '2023-01-05 09:30:00', 150, 150.5, 149.8, 150.2, 10000), ('AAPL', '2023-01-05 10:30:00', 150.2, 151, 150.1, 150.8, 12000), ('AAPL', '2023-01-05 11:30:00', 150.8, 150.9, 150, 150.1, 12000)]
    # cursor_setup.executemany("INSERT OR IGNORE INTO daily_prices VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", daily_data)
    # cursor_setup.executemany("INSERT OR IGNORE INTO hourly_prices VALUES (?, ?, ?, ?, ?, ?, ?)", hourly_data)
    # conn_setup.commit(); conn_setup.close(); print("Dummy DB potentially updated.")

    DATABASE_FILE = 'test_stock_data.db' 
    db_connection = connect_db(DATABASE_FILE)

    if db_connection:
        strategies_to_test = [
            "Buy AAPL when MA_10 crosses above MA_50 on daily chart",
            "Sell AAPL if hourly close drops below 150.5",
            "Show AAPL daily prices where close is greater than 150 and volume is above 110000 for the last 3 days", # More complex for LLM
            "Find days for MSFT where the 8-day EMA is above the 21-day EMA and the close price increased from the previous day" # Needs LAG
        ]

        for strategy_text in strategies_to_test:
            print(f"\n\n>>> Testing Strategy (Regex Parser): {strategy_text}")
            parsed_regex = parse_strategy_string(strategy_text) # Regex parser
            if parsed_regex:
                execute_strategy_regex(db_connection, parsed_regex)
            else:
                print("Failed to parse strategy with Regex parser.")

            print(f"\n>>> Testing Strategy (LLM SQL Generation): {strategy_text}")
            # This is an async function.
            # In a real application, you'd await this properly.
            # For this script, it will initiate the call.
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
        
        db_connection.close()
        print("\nDatabase connection closed.")

if __name__ == "__main__":
    # If running this script directly, setup asyncio event loop
    # This is a simplified way to run the async main function.
    # In a larger application (e.g., web server, GUI),
    # the event loop would be managed by that application's framework.
    try:
        asyncio.run(main())
    except RuntimeError as e:
        if "Cannot run the event loop while another loop is running" in str(e):
            print("Asyncio loop already running. This might happen in certain environments (e.g. Jupyter).")
            # If you are in an environment like Jupyter that already has an event loop:
            # loop = asyncio.get_event_loop()
            # loop.create_task(main())
            # Or, you might need to use `await main()` directly if in an async context.
        else:
            raise e
