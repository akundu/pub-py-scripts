from datetime import datetime
import sys

sql_creation_instructions = """
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
  2. ticker: The stock symbol (TEXT type) Note: use the placeholder {{STOCK}} to indicate the stock symbol in the sql query.
  3. close: The closing price at the time of the signal (REAL type)
  4. action: The action to take ('BUY' or 'SELL' as TEXT) Note: because you can have sell short through the action "SELL" before the action "BUY" happens
  5. buying_price: The price at which the position was bought (REAL type)
  6. buying_datetime: The exact date and time when the position was bought (DATETIME type, format: YYYY-MM-DD HH:MM:SS). 
      Use the placeholder {{START_DATE}} to indicate the buying datetime in the sql query and the {{END_DATE}} to indicate the end date.

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

  """

def generate_prompt(strategy_string: str, db_type: str, db_specific_instructions: dict, schema_info: str = None):
    """
    Generates a prompt for the LLM to generate a SQL query.
    """
    # Detailed instructions for the LLM
    prompt = f"""
    You are an expert financial analyst and SQL query writer.
    Your task is to convert an English stock trading strategy into a single, executable SQL query for {db_type.upper()}.
    The query should be read-only (i.e., only SELECT statements) and should be made for {db_type.upper()} and should explicitly use the schema of the tables and explicit only provide the select statement.
    Assume today's date is {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}.

    {db_specific_instructions}

    Available tables and their schemas:
    {schema_info}

    User's Strategy: "{strategy_string}"

    Generated SQL Query:
    """
    # TODO: add sql creation instructions
    # prompt += sql_creation_instructions

    # print("\n--- LLM Prompt ---", file=sys.stderr)
    # print(prompt, file=sys.stderr)
    # print("--- End Prompt ---\n", file=sys.stderr)
    return prompt
