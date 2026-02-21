"""
Gemini-powered natural language to SQL query generator for QuestDB stock data.

This module provides functionality to convert natural language queries into SQL
queries using Google's Gemini AI, specifically for querying StockQuestDB tables.
"""

import sys
import re
from typing import Optional, Dict, Any, Tuple
from datetime import datetime, timezone, timedelta
import google.genai as genai
from google.genai.errors import APIError

# Model aliases matching perm_application_analyzer.py
MODEL_ALIASES = {
    "flash": "gemini-flash-latest",
    "flash-3": "gemini-3-flash-preview",
    "pro": "gemini-pro-latest",
    "pro-3": "gemini-3-pro-preview",
    "flash-lite": "gemini-2.5-flash-lite",
    "gemini-3": "gemini-3-pro-preview"
}


def list_available_models() -> Dict[str, Any]:
    """
    List all available Gemini models by querying the Gemini API.
    
    Returns:
        Dictionary with model information:
        - models: List of model objects with name, display_name, description, etc.
        - model_names: List of model name strings
        - model_dict: Dictionary mapping model names to model info
        
    Raises:
        Exception: If API call fails or GEMINI_API_KEY is not set
        
    Example:
        >>> models = list_available_models()
        >>> print([m.name for m in models['models']])
        ['gemini-3-flash', 'gemini-3-pro', ...]
    """
    try:
        client = genai.Client()
    except Exception as e:
        raise Exception(f"Failed to initialize Gemini client. Make sure GEMINI_API_KEY environment variable is set.\nDetails: {e}")
    
    try:
        # Query Gemini API for available models
        model_list = list(client.models.list())
        
        # Extract model information
        models = []
        model_names = []
        model_dict = {}
        
        for model in model_list:
            model_info = {
                "name": model.name,
                "display_name": getattr(model, 'display_name', model.name),
                "description": getattr(model, 'description', ''),
                "version": getattr(model, 'version', ''),
                "supported_generation_methods": getattr(model, 'supported_generation_methods', []),
            }
            models.append(model_info)
            model_names.append(model.name)
            model_dict[model.name] = model_info
        
        return {
            "models": models,
            "model_names": model_names,
            "model_dict": model_dict
        }
        
    except APIError as e:
        raise Exception(f"Gemini API Error while listing models: {e}")
    except Exception as e:
        raise Exception(f"Unexpected error listing models: {e}")


def get_model_info() -> Dict[str, Any]:
    """
    Get detailed information about available models from Gemini API.
    Also includes alias mappings to known model IDs.
    
    Returns:
        Dictionary with model information including:
        - models: List of all available models from API
        - model_names: List of model name strings
        - aliases: List of known model aliases (from MODEL_ALIASES)
        - alias_mappings: Dictionary mapping aliases to model IDs
        - default: The default model alias
        
    Example:
        >>> info = get_model_info()
        >>> print(info['model_names'])
        ['gemini-3-flash', 'gemini-3-pro', ...]
    """
    # Get models from API
    api_models = list_available_models()
    
    # Build alias mappings (map aliases to actual model names if they exist)
    alias_mappings = {}
    available_model_names = set(api_models['model_names'])
    
    for alias, model_id in MODEL_ALIASES.items():
        # Check if the model ID exists in the API results
        if model_id in available_model_names:
            alias_mappings[alias] = model_id
        else:
            # Try to find a similar model (e.g., if model_id is "gemini-3-flash" but API has "models/gemini-3-flash")
            for model_name in available_model_names:
                if model_name.endswith(model_id) or model_id in model_name:
                    alias_mappings[alias] = model_name
                    break
    
    return {
        "models": api_models['models'],
        "model_names": api_models['model_names'],
        "model_dict": api_models['model_dict'],
        "aliases": list(MODEL_ALIASES.keys()),
        "alias_mappings": alias_mappings,
        "default": "flash",
        "descriptions": {
            "flash": "Fast and efficient model, good for most queries (default)",
            "pro": "More powerful model for complex queries",
            "flash-lite": "Lightweight version of flash model",
            "gemini-3": "Preview of Gemini 3 Pro model"
        }
    }

# StockQuestDB table schemas (static, from questdb_db.py)
TABLE_SCHEMAS = """
## Database Schema: StockQuestDB (QuestDB)

### Table: daily_prices
- ticker (SYMBOL) - Stock ticker symbol
- date (TIMESTAMP) - Trading date (designated timestamp)
- open (DOUBLE) - Opening price
- high (DOUBLE) - Highest price
- low (DOUBLE) - Lowest price
- close (DOUBLE) - Closing price
- volume (LONG) - Trading volume
- ma_10, ma_50, ma_100, ma_200 (DOUBLE) - Moving averages
- ema_8, ema_21, ema_34, ema_55, ema_89 (DOUBLE) - Exponential moving averages
- write_timestamp (TIMESTAMP) - When the record was written

### Table: hourly_prices
- ticker (SYMBOL) - Stock ticker symbol
- datetime (TIMESTAMP) - Hour timestamp (designated timestamp)
- open (DOUBLE) - Opening price for the hour
- high (DOUBLE) - Highest price for the hour
- low (DOUBLE) - Lowest price for the hour
- close (DOUBLE) - Closing price for the hour
- volume (LONG) - Trading volume for the hour
- write_timestamp (TIMESTAMP) - When the record was written

### Table: realtime_data
- ticker (SYMBOL) - Stock ticker symbol
- timestamp (TIMESTAMP) - Quote timestamp (designated timestamp)
- type (SYMBOL) - Data type (e.g., 'quote', 'trade')
- price (DOUBLE) - Current price
- size (LONG) - Trade size
- ask_price (DOUBLE) - Ask price
- ask_size (LONG) - Ask size
- write_timestamp (TIMESTAMP) - When the record was written

### Table: options_data
- ticker (SYMBOL) - Underlying stock ticker symbol
- option_ticker (SYMBOL) - Option contract ticker (e.g., 'AAPL240315C00150000')
- expiration_date (TIMESTAMP) - Option expiration date
- strike_price (DOUBLE) - Strike price
- option_type (SYMBOL) - 'CALL' or 'PUT'
- timestamp (TIMESTAMP) - Quote timestamp (designated timestamp)
- write_timestamp (TIMESTAMP) - When the record was written
- last_quote_timestamp (TIMESTAMP) - Last quote update time
- price (DOUBLE) - Option price
- bid (DOUBLE) - Bid price
- ask (DOUBLE) - Ask price
- day_close (DOUBLE) - Day's closing price
- fmv (DOUBLE) - Fair market value
- delta, gamma, theta, vega, rho (DOUBLE) - Greeks
- implied_volatility (DOUBLE) - Implied volatility
- volume (LONG) - Trading volume
- open_interest (LONG) - Open interest

### Table: financial_info
- ticker (SYMBOL) - Stock ticker symbol
- date (TIMESTAMP) - Financial data date (designated timestamp)
- price (DOUBLE) - Stock price
- market_cap (LONG) - Market capitalization
- earnings_per_share (DOUBLE) - EPS
- price_to_earnings (DOUBLE) - P/E ratio
- price_to_book (DOUBLE) - P/B ratio
- price_to_sales (DOUBLE) - P/S ratio
- price_to_cash_flow (DOUBLE) - P/CF ratio
- price_to_free_cash_flow (DOUBLE) - P/FCF ratio
- dividend_yield (DOUBLE) - Dividend yield
- return_on_assets (DOUBLE) - ROA
- return_on_equity (DOUBLE) - ROE
- debt_to_equity (DOUBLE) - D/E ratio
- current_ratio (DOUBLE) - Current ratio
- quick_ratio (DOUBLE) - Quick ratio
- cash_ratio (DOUBLE) - Cash ratio
- ev_to_sales (DOUBLE) - EV/Sales
- ev_to_ebitda (DOUBLE) - EV/EBITDA
- enterprise_value (LONG) - Enterprise value
- free_cash_flow (LONG) - Free cash flow
- iv_30d (DOUBLE) - 30-day implied volatility
- iv_90d (DOUBLE) - 90-day implied volatility
- iv_rank (DOUBLE) - 30-day IV rank (percentile within 1-year historical range)
- iv_90d_rank (DOUBLE) - 90-day IV rank (percentile within 1-year historical range)
- iv_rank_diff (DOUBLE) - Rank ratio (30-day rank / 90-day rank). Shows 30-day IV rank in context of 90-day IV rank. > 1.0 = front month more expensive relative to history, < 1.0 = back month more expensive, = 1.0 = equal
- relative_rank (DOUBLE) - Relative IV rank vs benchmark (VOO)
- iv_analysis_json (STRING) - Full IV analysis as JSON
- iv_analysis_spare (STRING) - Spare column for future use
- write_timestamp (TIMESTAMP) - When the record was written

## Important Notes:
- QuestDB uses PostgreSQL wire protocol, so SQL syntax is PostgreSQL-compatible
- TIMESTAMP columns store UTC timezone-naive datetime values
- SYMBOL columns are indexed string types
- Use WHERE clauses to filter by ticker, date ranges, etc.
- Always use LIMIT to restrict result size (default max: 1000 rows)
- Only SELECT queries are allowed (read-only)
"""


def validate_sql_readonly(sql: str) -> Tuple[bool, Optional[str]]:
    """
    Validate that SQL query is read-only (SELECT only).
    
    Args:
        sql: SQL query string
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    sql_stripped = sql.lstrip()
    sql_upper = sql_stripped.upper()
    
    # Check for forbidden write/DDL keywords as whole words
    forbidden_keywords = [
        'INSERT', 'UPDATE', 'DELETE', 'DROP', 'CREATE', 'ALTER',
        'TRUNCATE', 'GRANT', 'REVOKE', 'EXEC', 'EXECUTE', 'CALL'
    ]
    
    for keyword in forbidden_keywords:
        # Match keyword as a whole word to avoid false positives
        if re.search(rf"\\b{keyword}\\b", sql_upper, flags=re.IGNORECASE):
            return False, f"Forbidden keyword '{keyword}' found. Only read-only queries are allowed."
    
    # Must start with SELECT or WITH (for CTEs)
    if not (sql_upper.startswith('SELECT') or sql_upper.startswith('WITH')):
        return False, "Query must start with SELECT or WITH (CTE). Only read-only queries are allowed."
    
    return True, None


def validate_sql_tables(sql: str) -> Tuple[bool, Optional[str]]:
    """
    Validate that SQL query only references allowed StockQuestDB tables.
    
    Args:
        sql: SQL query string
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    allowed_tables = {
        'daily_prices', 'hourly_prices', 'realtime_data',
        'options_data', 'financial_info'
    }
    
    sql_upper = sql.upper()
    
    # Extract table names from FROM/JOIN clauses
    # Simple pattern matching - look for FROM/JOIN followed by table names
    table_pattern = r'(?:FROM|JOIN)\s+(\w+)'
    matches = re.findall(table_pattern, sql_upper, re.IGNORECASE)
    
    for match in matches:
        table_name = match.lower()
        if table_name not in allowed_tables:
            return False, f"Table '{table_name}' is not allowed. Allowed tables: {', '.join(sorted(allowed_tables))}"
    
    return True, None


def ensure_limit_clause(sql: str, max_rows: int = 1000) -> str:
    """
    Ensure SQL query has a LIMIT clause, adding one if missing.
    
    Args:
        sql: SQL query string
        max_rows: Maximum number of rows to return
        
    Returns:
        SQL query with LIMIT clause
    """
    sql_upper = sql.strip().upper()
    
    # Check if LIMIT already exists
    if 'LIMIT' in sql_upper:
        # Extract existing limit value and ensure it's not too high
        limit_match = re.search(r'LIMIT\s+(\d+)', sql_upper, re.IGNORECASE)
        if limit_match:
            existing_limit = int(limit_match.group(1))
            if existing_limit > max_rows:
                # Replace with max_rows
                sql = re.sub(r'LIMIT\s+\d+', f'LIMIT {max_rows}', sql, flags=re.IGNORECASE)
        # If LIMIT exists but no number found, add max_rows
        elif not re.search(r'LIMIT\s+\d+', sql_upper, re.IGNORECASE):
            sql = sql.rstrip(';').rstrip() + f' LIMIT {max_rows}'
    else:
        # Add LIMIT clause before semicolon or at end
        sql = sql.rstrip(';').rstrip() + f' LIMIT {max_rows}'
    
    return sql


def normalize_sql_for_questdb(sql: str) -> str:
    """
    Apply small rewrites so Gemini-generated SQL works against QuestDB.

    Currently:
    - Replace CURRENT_DATE with proper TIMESTAMP comparisons
    - Fix date = 'YYYY-MM-DD' comparisons to use date ranges for TIMESTAMP columns
    """
    # Compute today's date range in UTC
    today_utc = datetime.now(timezone.utc).date()
    today_start = datetime.combine(today_utc, datetime.min.time()).replace(tzinfo=timezone.utc).replace(tzinfo=None)
    tomorrow_start = today_start + timedelta(days=1)
    
    today_literal = today_utc.strftime("%Y-%m-%d")
    
    # Replace CURRENT_DATE with date range for TIMESTAMP comparisons
    # Pattern: WHERE date = CURRENT_DATE -> WHERE date >= 'YYYY-MM-DD 00:00:00' AND date < 'YYYY-MM-DD+1 00:00:00'
    sql = re.sub(
        r"(\w+)\s*=\s*CURRENT_DATE",
        rf"\1 >= '{today_start}' AND \1 < '{tomorrow_start}'",
        sql,
        flags=re.IGNORECASE
    )
    
    # Fix date = 'YYYY-MM-DD' patterns to use date ranges for TIMESTAMP columns
    # This handles cases where Gemini generates: WHERE date = '2025-12-15'
    # Convert to: WHERE date >= '2025-12-15 00:00:00' AND date < '2025-12-16 00:00:00'
    def fix_date_equality(match):
        col_name = match.group(1)
        date_str = match.group(2)
        try:
            # Parse the date string
            date_obj = datetime.strptime(date_str, '%Y-%m-%d')
            start = date_obj.replace(tzinfo=None)
            end = start + timedelta(days=1)
            return f"{col_name} >= '{start}' AND {col_name} < '{end}'"
        except ValueError:
            # If parsing fails, return original
            return match.group(0)
    
    # Match patterns like: date = 'YYYY-MM-DD' or date='YYYY-MM-DD'
    sql = re.sub(
        r"(\w+)\s*=\s*['\"](\d{4}-\d{2}-\d{2})['\"]",
        fix_date_equality,
        sql,
        flags=re.IGNORECASE
    )

    # Normalize COUNT(*) to COUNT() to avoid '*' parsing issues in QuestDB
    # This keeps the semantics (COUNT of all rows) but removes the '*' token
    sql = re.sub(r"COUNT\(\s*\*\s*\)", "COUNT()", sql, flags=re.IGNORECASE)

    return sql


def query_gemini_for_sql(natural_query: str, model_alias: str = "flash", max_rows: int = 1000) -> str:
    """
    Query Gemini API to generate SQL from natural language.
    
    Args:
        natural_query: Natural language query description
        model_alias: Model alias (flash, pro, flash-lite, gemini-3)
        max_rows: Maximum number of rows to return (default: 1000)
        
    Returns:
        Generated SQL query string
        
    Raises:
        ValueError: If model alias is invalid
        Exception: If API call fails or returns invalid SQL
    """
    # Validate model alias
    if model_alias not in MODEL_ALIASES:
        raise ValueError(f"Invalid model alias '{model_alias}'. Choose from: {list(MODEL_ALIASES.keys())}")
    
    model_id = MODEL_ALIASES[model_alias]
    
    # Build prompt
    prompt = f"""You are a SQL query expert specializing in QuestDB (PostgreSQL-compatible) queries for stock market data.

## Database Schema:
{TABLE_SCHEMAS}

## User Query:
{natural_query}

## Requirements:
1. Generate a valid PostgreSQL-compatible SELECT query
2. **CRITICAL: Choose the appropriate table based on query context:**
   - **realtime_data**: Use for "today", "current", "latest", "now", "live" queries about CURRENT prices. This table has the most up-to-date live prices with `price` column. Use for queries like "current price", "latest price", "top stocks by price today"
   - **daily_prices**: Use for historical daily data, "yesterday", specific dates, multi-day ranges, daily aggregations, and daily performance calculations (open vs close). Has `open`, `close`, `high`, `low`, `volume` columns. Use for queries like "daily performance", "yesterday's data", "last 30 days", "top performers by daily change"
   - **hourly_prices**: Use for hourly-level analysis within recent days (last few days/weeks). Has hourly `open`, `close`, `high`, `low`, `volume` columns
   - **options_data**: Use for options-related queries
   - **financial_info**: Use for financial metrics (P/E, market cap, etc.)
3. Use appropriate WHERE clauses to filter data
4. Include a LIMIT clause with maximum {max_rows} rows
5. Use proper column names from the schema
6. **IMPORTANT: Date/Timestamp Filtering:**
   - For **realtime_data**: Use `timestamp` column, filter by recent time (e.g., WHERE timestamp >= CURRENT_DATE or ORDER BY timestamp DESC)
   - For **daily_prices**: The `date` column is a TIMESTAMP. For "today" use: WHERE date = (SELECT MAX(date) FROM daily_prices). For specific dates use ranges: WHERE date >= '2024-01-15 00:00:00' AND date < '2024-01-16 00:00:00'
   - For **hourly_prices**: Use `datetime` column with timestamp ranges
   - For "most recent" or "latest" data, use: ORDER BY timestamp/date/datetime DESC LIMIT N
7. For ticker filtering, use WHERE ticker = 'SYMBOL' (case-sensitive)
8. Return ONLY the SQL query, no explanations, no markdown formatting, just the raw SQL

## Example Queries:
- "latest price for AAPL" -> SELECT price, timestamp FROM realtime_data WHERE ticker = 'AAPL' ORDER BY timestamp DESC LIMIT 1
- "current price for AAPL and MSFT" -> SELECT ticker, price, timestamp FROM realtime_data WHERE ticker IN ('AAPL', 'MSFT') ORDER BY timestamp DESC LIMIT 2
- "top 10 most traded tickers today" -> SELECT ticker, SUM(size) as total_volume FROM realtime_data WHERE timestamp >= CURRENT_DATE GROUP BY ticker ORDER BY total_volume DESC LIMIT 10
- "top performing stocks today" -> SELECT ticker, price, timestamp FROM realtime_data WHERE timestamp >= CURRENT_DATE ORDER BY price DESC LIMIT 10
- "top stocks by current price" -> SELECT ticker, price, timestamp FROM realtime_data WHERE timestamp >= CURRENT_DATE ORDER BY price DESC LIMIT 10
- "AAPL daily prices last 30 days" -> SELECT * FROM daily_prices WHERE ticker = 'AAPL' AND date >= CURRENT_DATE - INTERVAL '30 days' ORDER BY date DESC LIMIT 1000
- "AAPL daily performance yesterday" -> SELECT ticker, ((close - open) / open) * 100 AS performance_pct FROM daily_prices WHERE ticker = 'AAPL' AND date = (SELECT MAX(date) FROM daily_prices WHERE date < (SELECT MAX(date) FROM daily_prices)) LIMIT 1
- "most recent daily prices for AAPL" -> SELECT * FROM daily_prices WHERE ticker = 'AAPL' ORDER BY date DESC LIMIT 10
- "AAPL hourly prices last 7 days" -> SELECT * FROM hourly_prices WHERE ticker = 'AAPL' AND datetime >= CURRENT_DATE - INTERVAL '7 days' ORDER BY datetime DESC LIMIT 1000
- "AAPL options expiring in March 2024" -> SELECT * FROM options_data WHERE ticker = 'AAPL' AND expiration_date >= '2024-03-01 00:00:00' AND expiration_date < '2024-04-01 00:00:00' LIMIT {max_rows}
- "SPY movement % over a day for the last 9 months as a percentage percentile" -> SELECT 
    -- Volatility as % of Opening Price
    avg(volatility_pct) as mean_volatility,
    stddev(volatility_pct) as std_dev_volatility,
    min(volatility_pct) as min_volatility,
    max(volatility_pct) as max_volatility,
    -- Percentile breakdown of the daily % range
    approx_percentile(volatility_pct, 0.25, 5) as p25,
    approx_percentile(volatility_pct, 0.50, 5) as p50,
    approx_percentile(volatility_pct, 0.75, 5) as p75,
    approx_percentile(volatility_pct, 0.90, 5) as p90,
    approx_percentile(volatility_pct, 0.95, 5) as p95,
    approx_percentile(volatility_pct, 0.99, 5) as p99
FROM (
    SELECT 
        ((high - low) / open) * 100 as volatility_pct
    FROM daily_prices
    WHERE ticker = 'AAPL' 
      AND date > dateadd('M', -9, now())
)
- "SPY day over day close prices as a percentage percentile" -> SELECT 
    avg(return_pct) as mean,
    stddev(return_pct) as std_dev,
    min(return_pct) as min_val,
    max(return_pct) as max_val,
    -- Added 3rd argument '5' for maximum precision (significant digits)
    approx_percentile(return_pct + 100, 0.25, 5) - 100 as p25,
    approx_percentile(return_pct + 100, 0.50, 5) - 100 as p50,
    approx_percentile(return_pct + 100, 0.75, 5) - 100 as p75,
    approx_percentile(return_pct + 100, 0.90, 5) - 100 as p90,
    approx_percentile(return_pct + 100, 0.95, 5) - 100 as p95,
    approx_percentile(return_pct + 100, 0.99, 5) - 100 as p99
FROM (
    SELECT 
        ((close - prev_close) / prev_close) * 100 as return_pct
    FROM (
        SELECT 
            close, 
            lag(close) OVER (PARTITION BY ticker ORDER BY date) as prev_close
        FROM daily_prices
        WHERE ticker = 'SPY' 
          AND date > dateadd('y', -1, now())
    )
    WHERE prev_close IS NOT NULL
)



Generate the SQL query now:
"""
    
    # Initialize Gemini client
    try:
        client = genai.Client()
    except Exception as e:
        raise Exception(f"Failed to initialize Gemini client. Make sure GEMINI_API_KEY environment variable is set.\nDetails: {e}")
    
    # Call Gemini API
    try:
        response = client.models.generate_content(
            model=model_id,
            contents=[prompt]
        )
        
        if not response.candidates or not response.candidates[0].content or not response.candidates[0].content.parts:
            finish_reason = response.candidates[0].finish_reason if response.candidates else 'No candidates'
            raise Exception(f"Gemini returned empty response. Finish reason: {finish_reason}")
        
        sql = response.text.strip()
        
        # Clean up SQL - remove markdown code blocks if present
        sql = re.sub(r'^```sql\s*', '', sql, flags=re.IGNORECASE | re.MULTILINE)
        sql = re.sub(r'^```\s*', '', sql, flags=re.IGNORECASE | re.MULTILINE)
        sql = re.sub(r'```\s*$', '', sql, flags=re.IGNORECASE | re.MULTILINE)
        sql = sql.strip()

        # Apply QuestDB-specific normalization (e.g., CURRENT_DATE -> 'YYYY-MM-DD')
        sql = normalize_sql_for_questdb(sql)
        
        # Validate SQL is read-only
        is_valid, error_msg = validate_sql_readonly(sql)
        if not is_valid:
            raise ValueError(f"Generated SQL is not read-only: {error_msg}\nGenerated SQL: {sql}")
        
        # Validate tables
        is_valid, error_msg = validate_sql_tables(sql)
        if not is_valid:
            raise ValueError(f"Generated SQL references invalid tables: {error_msg}\nGenerated SQL: {sql}")
        
        # Ensure LIMIT clause
        sql = ensure_limit_clause(sql, max_rows)
        
        return sql
        
    except APIError as e:
        raise Exception(f"Gemini API Error: {e}")
    except ValueError:
        raise  # Re-raise validation errors
    except Exception as e:
        raise Exception(f"Unexpected error generating SQL: {e}")


def generate_and_validate_sql(natural_query: str, model_alias: str = "flash", max_rows: int = 1000) -> str:
    """
    Generate SQL from natural language and validate it.
    
    This is a convenience wrapper that handles validation and error messages.
    
    Args:
        natural_query: Natural language query description
        model_alias: Model alias (flash, pro, flash-lite, gemini-3)
        max_rows: Maximum number of rows to return (default: 1000)
        
    Returns:
        Validated SQL query string
        
    Raises:
        ValueError: If query generation or validation fails
    """
    try:
        sql = query_gemini_for_sql(natural_query, model_alias, max_rows)
        return sql
    except Exception as e:
        raise ValueError(f"Failed to generate valid SQL: {e}")

