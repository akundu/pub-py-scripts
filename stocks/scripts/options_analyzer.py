#!/usr/bin/env python3
"""
Options Analyzer - Covered Call Premium Analysis Tool

This program analyzes covered call opportunities across all strike prices and tickers
in a QuestDB options database. It calculates potential premiums for $1M stock positions
and provides flexible filtering, sorting, and output options.

Usage:
    export POLYGON_API_KEY=YOUR_API_KEY  # Optional, for symbol lists
    python options_analyzer.py --db-conn questdb://user:pass@host:8812/db
    python options_analyzer.py --symbols AAPL,MSFT,GOOGL --days 14 --output csv
    python options_analyzer.py --types sp-500 --sort daily_premium --group-by ticker
    python options_analyzer.py --min-volume 1000 --max-days 30 --output results.csv
"""

import os
import sys
import argparse
import asyncio
import pandas as pd
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Any, Optional, Tuple, Union
from tabulate import tabulate
from pathlib import Path
import json
import re
import math

# Ensure project root is on sys.path so `common` can be imported when running from any cwd
CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Import common symbol loading functions
from common.symbol_loader import add_symbol_arguments, fetch_lists_data
from common.stock_db import get_stock_db


class FilterExpression:
    """Represents a single filter expression like 'pe_ratio > 20' or 'num_contracts * 0.1 > volume'."""
    
    def __init__(self, field: str, operator: str, value: Union[float, int, str, None], is_field_comparison: bool = False, field_expression: str = None):
        self.field = field
        self.operator = operator
        self.value = value
        self.is_field_comparison = is_field_comparison  # True if comparing against another field
        self.field_expression = field_expression  # Mathematical expression for the field (e.g., "num_contracts * 0.1")
    
    def _validate_expression_fields(self, df: pd.DataFrame, expression: str) -> bool:
        """Validate that all field names in the expression exist in the DataFrame."""
        import re
        # Extract field names from the expression (simple regex for field names)
        field_pattern = r'\b[a-zA-Z_][a-zA-Z0-9_]*\b'
        potential_fields = re.findall(field_pattern, expression)
        
        # Check if any of these are actual DataFrame columns
        for field in potential_fields:
            if field in df.columns:
                return True
        return False
    
    def _evaluate_expression(self, df: pd.DataFrame, expression: str) -> pd.Series:
        """Evaluate a mathematical or boolean expression like 'num_contracts * 0.1' or 'num_contracts < volume*2' against DataFrame columns."""
        try:
            # Replace field names with their column references
            eval_expression = expression
            for field_name in df.columns:
                if field_name in eval_expression:
                    eval_expression = eval_expression.replace(field_name, f"df['{field_name}']")
            
            # Evaluate the expression safely
            result = eval(eval_expression)
            
            # If the result is a boolean Series (from comparison operations), return it directly
            if isinstance(result, pd.Series) and result.dtype == bool:
                return result
            
            # If the result is a scalar boolean, convert to Series
            if isinstance(result, bool):
                return pd.Series([result] * len(df), index=df.index)
            
            # Otherwise, return the result as-is (for mathematical expressions)
            return result
        except Exception as e:
            # If evaluation fails, return all False
            return pd.Series([False] * len(df), index=df.index)
    
    def apply(self, df: pd.DataFrame) -> pd.Series:
        """Apply this filter to a DataFrame and return a boolean Series."""
        # Handle complete mathematical expressions (e.g., "num_contracts < volume*2")
        if self.field_expression and self.is_field_comparison and not self.value:
            # This is a complete expression like "num_contracts < volume*2"
            if not self._validate_expression_fields(df, self.field_expression):
                return pd.Series([False] * len(df), index=df.index)
            return self._evaluate_expression(df, self.field_expression)
        
        # Handle mathematical expressions in the field
        if self.field_expression:
            if not self._validate_expression_fields(df, self.field_expression):
                return pd.Series([False] * len(df), index=df.index)
            column = self._evaluate_expression(df, self.field_expression)
        else:
            if self.field not in df.columns:
                # If field doesn't exist, return all False (no matches)
                return pd.Series([False] * len(df), index=df.index)
            column = df[self.field]
        
        # Handle field-to-field comparisons
        if self.is_field_comparison:
            if self.value not in df.columns:
                # If comparison field doesn't exist, return all False (no matches)
                return pd.Series([False] * len(df), index=df.index)
            
            comparison_column = df[self.value]
            
            # Handle None/NaN values for both columns
            if self.operator == '>':
                return (pd.notna(column)) & (pd.notna(comparison_column)) & (column > comparison_column)
            elif self.operator == '>=':
                return (pd.notna(column)) & (pd.notna(comparison_column)) & (column >= comparison_column)
            elif self.operator == '<':
                return (pd.notna(column)) & (pd.notna(comparison_column)) & (column < comparison_column)
            elif self.operator == '<=':
                return (pd.notna(column)) & (pd.notna(comparison_column)) & (column <= comparison_column)
            elif self.operator == '==':
                return (pd.notna(column)) & (pd.notna(comparison_column)) & (column == comparison_column)
            elif self.operator == '!=':
                return (pd.notna(column)) & (pd.notna(comparison_column)) & (column != comparison_column)
            else:
                raise ValueError(f"Unsupported operator for field comparison: {self.operator}")
        
        # Handle value comparisons (original logic)
        if self.operator == 'exists':
            return pd.notna(column)
        elif self.operator == 'not_exists':
            return pd.isna(column)
        elif self.operator == '>':
            return (pd.notna(column)) & (column > self.value)
        elif self.operator == '>=':
            return (pd.notna(column)) & (column >= self.value)
        elif self.operator == '<':
            return (pd.notna(column)) & (column < self.value)
        elif self.operator == '<=':
            return (pd.notna(column)) & (column <= self.value)
        elif self.operator == '==':
            return (pd.notna(column)) & (column == self.value)
        elif self.operator == '!=':
            return (pd.notna(column)) & (column != self.value)
        else:
            raise ValueError(f"Unsupported operator: {self.operator}")
    
    def __str__(self):
        field_display = self.field_expression if self.field_expression else self.field
        if self.is_field_comparison:
            return f"{field_display} {self.operator} {self.value}"
        else:
            return f"{field_display} {self.operator} {self.value}"


class FilterParser:
    """Parses filter expressions and applies them to DataFrames."""
    
    # Supported fields and their types
    SUPPORTED_FIELDS = {
        'pe_ratio': float,
        'market_cap': float,
        'volume': int,
        'num_contracts': int,
        'potential_premium': float,
        'daily_premium': float,
        'current_price': float,
        'strike_price': float,
        'price_above_current': float,
        'option_premium': float,
        'option_premium_percentage': float,
        'premium_above_diff_percentage': float,
        'days_to_expiry': int,
        'delta': float,
        'theta': float
    }
    
    # Supported operators
    SUPPORTED_OPERATORS = ['>', '>=', '<', '<=', '==', '!=', 'exists', 'not_exists']
    
    @classmethod
    def resolve_field_name(cls, name: str) -> str:
        """Resolve a field name by exact or case-insensitive substring match against SUPPORTED_FIELDS.
        Raises ValueError on zero or multiple matches."""
        if name in cls.SUPPORTED_FIELDS:
            return name
        lowered = name.lower()
        matches = [f for f in cls.SUPPORTED_FIELDS.keys() if lowered in f.lower()]
        if len(matches) == 1:
            return matches[0]
        if len(matches) == 0:
            raise ValueError(f"Unknown field: {name}. Supported fields: {list(cls.SUPPORTED_FIELDS.keys())}")
        raise ValueError(f"Ambiguous field '{name}' matches: {matches}. Please be more specific.")

    @classmethod
    def _normalize_expression(cls, expression: str) -> str:
        """Replace identifiers in an expression with resolved full field names using substring matching."""
        token_pattern = r"\b([A-Za-z_][A-Za-z0-9_]*)\b"
        tokens = set(re.findall(token_pattern, expression))
        normalized = expression
        # Sort tokens by length descending to avoid partial replacement issues
        for tok in sorted(tokens, key=len, reverse=True):
            try:
                resolved = cls.resolve_field_name(tok)
            except ValueError:
                # Leave non-field tokens (e.g., functions) as-is
                continue
            # Replace whole-word occurrences only
            normalized = re.sub(rf"\b{re.escape(tok)}\b", resolved, normalized)
        return normalized

    @classmethod
    def parse_expression(cls, expression: str) -> FilterExpression:
        """Parse a single filter expression like 'pe_ratio > 20'."""
        expression = expression.strip()

        
        # Handle special cases
        # Support generic '<field> exists' and '<field> not_exists' with substring field resolution
        m = re.match(r"^([A-Za-z_][A-Za-z0-9_]*)\s+(exists|not_exists)$", expression, re.IGNORECASE)
        if m:
            raw_field, op = m.group(1), m.group(2).lower()
            field = cls.resolve_field_name(raw_field)
            return FilterExpression(field, 'exists' if op == 'exists' else 'not_exists', None)
        
        # Parse comparison expressions
        for op in ['>=', '<=', '==', '!=', '>', '<']:
            if op in expression:
                parts = expression.split(op, 1)
                if len(parts) == 2:
                    field_expr = parts[0].strip()
                    value_str = parts[1].strip()
                    
                    if op not in cls.SUPPORTED_OPERATORS:
                        raise ValueError(f"Unsupported operator: {op}. Supported operators: {cls.SUPPORTED_OPERATORS}")
                    
                    # Check if this is a field-to-field comparison
                    # Try resolving value_str as a field name (with substring support)
                    value_field_resolved = None
                    try:
                        value_field_resolved = cls.resolve_field_name(value_str)
                    except ValueError:
                        value_field_resolved = None
                    
                    if value_field_resolved is not None:
                        # This is a field-to-field comparison
                        # Check if field_expr contains mathematical operations
                        if cls._has_math_operations(field_expr):
                            return FilterExpression('', op, value_field_resolved, is_field_comparison=True, field_expression=field_expr)
                        else:
                            # Resolve left-hand field name as well
                            field_left = cls.resolve_field_name(field_expr)
                            return FilterExpression(field_left, op, value_field_resolved, is_field_comparison=True)
                    elif cls._has_math_operations(value_str):
                        # This is a field-to-mathematical-expression comparison
                        # Check if field_expr contains mathematical operations
                        if cls._has_math_operations(field_expr):
                            expr_norm = cls._normalize_expression(f"{field_expr} {op} {value_str}")
                            return FilterExpression('', op, '', is_field_comparison=True, field_expression=expr_norm)
                        else:
                            # Resolve left-hand field name
                            field_left = cls.resolve_field_name(field_expr)
                            expr_norm = cls._normalize_expression(f"{field_left} {op} {value_str}")
                            return FilterExpression('', op, '', is_field_comparison=True, field_expression=expr_norm)
                    else:
                        # This is a value comparison
                        # Check if field_expr contains mathematical operations
                        if cls._has_math_operations(field_expr):
                            # Try to determine the primary field for type checking
                            primary_field = cls._extract_primary_field(field_expr)
                            if primary_field and primary_field in cls.SUPPORTED_FIELDS:
                                value_type = cls.SUPPORTED_FIELDS[primary_field]
                            else:
                                value_type = float  # Default to float for mathematical expressions
                            
                            try:
                                # Handle market_cap with B/M/T suffixes
                                if primary_field == 'market_cap':
                                    value = cls._parse_market_cap_value(value_str)
                                else:
                                    value = value_type(value_str)
                            except ValueError:
                                raise ValueError(f"Invalid value for {field_expr}: {value_str}. Expected {value_type.__name__}")
                            
                            expr_norm = cls._normalize_expression(field_expr)
                            return FilterExpression('', op, value, is_field_comparison=False, field_expression=expr_norm)
                        else:
                            # Simple field comparison
                            # Resolve left-hand field name (with substring support)
                            field_left = cls.resolve_field_name(field_expr)
                            
                            value_type = cls.SUPPORTED_FIELDS[field_left]
                            try:
                                # Handle market_cap with B/M/T suffixes
                                if field_left == 'market_cap':
                                    value = cls._parse_market_cap_value(value_str)
                                else:
                                    value = value_type(value_str)
                            except ValueError:
                                raise ValueError(f"Invalid value for {field_left}: {value_str}. Expected {value_type.__name__}")
                            
                            return FilterExpression(field_left, op, value, is_field_comparison=False)
        
        raise ValueError(f"Could not parse expression: {expression}")
    
    @classmethod
    def _has_math_operations(cls, expression: str) -> bool:
        """Check if an expression contains mathematical operations."""
        import re
        # Look for mathematical operators
        math_pattern = r'[\+\-\*/]'
        return bool(re.search(math_pattern, expression))
    
    @classmethod
    def _extract_primary_field(cls, expression: str) -> str:
        """Extract the primary field name from a mathematical expression."""
        import re
        # Extract field names from the expression
        field_pattern = r'\b[a-zA-Z_][a-zA-Z0-9_]*\b'
        fields = re.findall(field_pattern, expression)
        
        # Return the first field that's in our supported fields
        for field in fields:
            if field in cls.SUPPORTED_FIELDS:
                return field
        return fields[0] if fields else None
    
    @classmethod
    def _parse_market_cap_value(cls, value_str: str) -> float:
        """Parse market cap value with optional B/M/T suffixes."""
        value_str = value_str.strip().upper()
        
        # Handle T (trillion) suffix
        if value_str.endswith('T'):
            number_str = value_str[:-1]
            try:
                return float(number_str) * 1_000_000_000_000
            except ValueError:
                raise ValueError(f"Invalid market cap value: {value_str}. Expected format like '3.5T' or '1000T'")
        
        # Handle B (billion) suffix
        elif value_str.endswith('B'):
            number_str = value_str[:-1]
            try:
                return float(number_str) * 1_000_000_000
            except ValueError:
                raise ValueError(f"Invalid market cap value: {value_str}. Expected format like '3.5B' or '1000B'")
        
        # Handle M (million) suffix
        elif value_str.endswith('M'):
            number_str = value_str[:-1]
            try:
                return float(number_str) * 1_000_000
            except ValueError:
                raise ValueError(f"Invalid market cap value: {value_str}. Expected format like '500M' or '1000M'")
        
        # No suffix - treat as raw number
        else:
            try:
                return float(value_str)
            except ValueError:
                raise ValueError(f"Invalid market cap value: {value_str}. Expected format like '3000000000', '3.5B', '500M', or '1T'")
    
    @classmethod
    def parse_filters(cls, filter_strings: List[str]) -> List[FilterExpression]:
        """Parse multiple filter expressions from a list of strings."""
        filters = []
        for filter_str in filter_strings:
            # Remove all extra whitespace within the string for robust parsing
            compact = ' '.join(filter_str.split())
            # Split by comma and parse each expression
            expressions = [expr.strip() for expr in compact.split(',') if expr.strip()]
            for expr in expressions:
                filters.append(cls.parse_expression(expr))
        return filters
    
    @classmethod
    def apply_filters(cls, df: pd.DataFrame, filters: List[FilterExpression], logic: str = 'AND') -> pd.DataFrame:
        """Apply a list of filters to a DataFrame using AND or OR logic."""
        if not filters:
            return df
        
        if len(filters) == 1:
            mask = filters[0].apply(df)
        else:
            # Apply first filter
            mask = filters[0].apply(df)
            
            # Apply remaining filters with specified logic
            for filter_expr in filters[1:]:
                filter_mask = filter_expr.apply(df)
                if logic.upper() == 'AND':
                    mask = mask & filter_mask
                elif logic.upper() == 'OR':
                    mask = mask | filter_mask
                else:
                    raise ValueError(f"Unsupported logic: {logic}. Use 'AND' or 'OR'")
        
        return df[mask]


class OptionsAnalyzer:
    """Analyzes covered call opportunities across all strike prices and tickers."""
    
    def __init__(self, db_conn: str, quiet: bool = False):
        """Initialize the options analyzer with database connection."""
        self.db_conn = db_conn
        self.quiet = quiet
        self.db = None
    
    def _create_compact_headers(self, df: pd.DataFrame) -> Dict[str, str]:
        """Create compact headers that are at most 4 characters longer than the data width."""
        compact_headers = {}
        
        # Define mapping for common columns to shorter names
        header_mapping = {
            'ticker': 'TKR',
            'current_price': 'PRC',
            'pe_ratio': 'P/E',
            'market_cap': 'MKT_CAP',
            'market_cap_b': 'MKT_B',
            'strike_price': 'STRK',
            'price_above_current': 'ABOVE',
            'option_premium': 'PREM',
            'option_premium_percentage': 'PREM%',
            'premium_above_diff_percentage': 'DIFF%',
            'delta': 'DEL',
            'theta': 'TH',
            'volume': 'VOL',
            'num_contracts': 'CNT',
            'potential_premium': 'POT_PREM',
            'daily_premium': 'DAY_PREM',
            'expiration_date': 'EXP (UTC)',
            'days_to_expiry': 'DAYS',
            'last_quote_timestamp': 'LQUOTE_TS',
            'write_timestamp': 'WRITE_TS (EST)',
            'option_ticker': 'OPT_TKR'
        }
        
        for col in df.columns:
            if col in header_mapping:
                compact_headers[col] = header_mapping[col]
            else:
                # For unknown columns, use the original name but truncate if too long
                compact_headers[col] = col[:8] if len(col) > 8 else col
        
        return compact_headers
        
    async def initialize(self):
        """Initialize database connection."""
        try:
            self.db = get_stock_db('questdb', db_config=self.db_conn)
            if not self.quiet:
                print("Database connection established successfully.", file=sys.stderr)
        except Exception as e:
            print(f"Error connecting to database: {e}", file=sys.stderr)
            sys.exit(1)
    
    async def get_available_tickers(self) -> List[str]:
        """Get all available tickers from the daily_prices table."""
        try:
            # First, let's check if the table exists and has data
            check_query = """
            SELECT COUNT(*) as row_count 
            FROM daily_prices
            """
            count_df = await self.db.execute_select_sql(check_query)
            
            if not count_df.empty:
                # QuestDB might return the count in different column names
                row_count = None
                for col in count_df.columns:
                    col_str = str(col).lower()
                    if 'count' in col_str or col == 0:  # QuestDB sometimes returns as column 0
                        row_count = count_df.iloc[0][col]
                        break
                
                if row_count is not None:
                    if not self.quiet:
                        print(f"Found {row_count} rows in daily_prices table")
                    
                    if row_count == 0:
                        if not self.quiet:
                            print("daily_prices table is empty", file=sys.stderr)
                        return []
                else:
                    if not self.quiet:
                        print(f"Could not determine row count. Columns: {list(count_df.columns)}")
            
            # Now get the distinct tickers
            query = """
            SELECT DISTINCT ticker 
            FROM daily_prices 
            ORDER BY ticker
            """
            df = await self.db.execute_select_sql(query)
            
            if df.empty:
                if not self.quiet:
                    print("No tickers found in daily_prices table", file=sys.stderr)
                return []
            
            # QuestDB might return columns as integers (0, 1, 2, etc.)
            # The first column should be the ticker
            if len(df.columns) > 0:
                ticker_col = df.columns[0]  # First column should be ticker
                tickers = df[ticker_col].tolist()
                if not self.quiet:
                    print(f"Found {len(tickers)} unique tickers")
                return tickers
            else:
                if not self.quiet:
                    print(f"Unexpected column structure. Available columns: {list(df.columns)}", file=sys.stderr)
                return []
        except Exception as e:
            if not self.quiet:
                print(f"Error fetching available tickers: {e}", file=sys.stderr)
            return []
    
    async def get_financial_info(self, tickers: List[str]) -> Dict[str, Dict[str, Any]]:
        """Get financial information (P/E, market_cap) for the given tickers."""
        financial_data = {}
        
        for ticker in tickers:
            try:
                # Get latest financial info for each ticker
                query = """
                SELECT 
                    price_to_earnings as pe_ratio,
                    market_cap,
                    price
                FROM financial_info 
                WHERE ticker = $1 
                ORDER BY date DESC 
                LIMIT 1
                """
                df = await self.db.execute_select_sql(query, (ticker,))
                
                if not df.empty:
                    # QuestDB returns columns as integers, so we need to map them
                    # Based on debug output: Column 5 = price_to_earnings, Column 3 = market_cap, Column 2 = price
                    row = df.iloc[0]
                    financial_data[ticker] = {
                        'pe_ratio': row.iloc[0] if len(row) > 0 else None,  # First column should be pe_ratio
                        'market_cap': row.iloc[1] if len(row) > 1 else None,  # Second column should be market_cap
                        'price': row.iloc[2] if len(row) > 2 else None  # Third column should be price
                    }
                else:
                    financial_data[ticker] = {
                        'pe_ratio': None,
                        'market_cap': None,
                        'price': None
                    }
            except Exception as e:
                if not self.quiet:
                    print(f"Warning: Could not fetch financial info for {ticker}: {e}")
                financial_data[ticker] = {
                    'pe_ratio': None,
                    'market_cap': None,
                    'price': None
                }
        
        return financial_data
    
    async def analyze_options(
        self,
        tickers: List[str],
        days_to_expiry: Optional[int] = None,
        min_volume: int = 0,
        max_days: Optional[int] = None,
        min_premium: float = 0.0,
        position_size: float = 1000000.0,
        filters: Optional[List[FilterExpression]] = None,
        filter_logic: str = 'AND'
    ) -> pd.DataFrame:
        """
        Analyze covered call opportunities for the given tickers.
        
        Args:
            tickers: List of ticker symbols to analyze
            days_to_expiry: Number of days to expiry (if None, analyze all available)
            min_volume: Minimum volume filter
            max_days: Maximum days to expiry filter
            min_premium: Minimum potential premium filter
            position_size: Position size in dollars for calculations
            filters: List of FilterExpression objects to apply
            filter_logic: Logic to combine filters ('AND' or 'OR')
            
        Returns:
            DataFrame with analysis results
        """
        if not tickers:
            return pd.DataFrame()
        
        # Convert tickers to uppercase for database compatibility
        tickers_upper = [ticker.upper() for ticker in tickers]
        
        # Build the main analysis query based on q8d.sql
        ticker_placeholders = ','.join([f'${i+1}' for i in range(len(tickers_upper))])
        # With no SQL-side volume or OTM filtering, only min_premium follows tickers
        premium_param = len(tickers_upper) + 1
        
        # Base query structure - using column indices since QuestDB returns them as integers
        query = f"""
        WITH latest_stock_prices AS (
            SELECT
                ticker,
                ROUND(close, 2) AS current_price
            FROM (
                SELECT ticker, close, date
                FROM daily_prices
                WHERE ticker IN ({ticker_placeholders})
            ) LATEST ON date PARTITION BY ticker
        ),
        filtered_options AS (
            SELECT
                o.ticker,
                o.option_ticker,
                ROUND(o.strike_price, 2) AS strike_price,
                ROUND(o.bid, 2) AS bid,
                ROUND(o.ask, 2) AS ask,
                ROUND(o.delta, 2) AS delta,
                ROUND(o.theta, 2) AS theta,
                COALESCE(ROUND(o.implied_volatility, 2), 0.00) AS implied_volatility,
                o.volume,
                o.expiration_date,
                o.write_timestamp,
                o.last_quote_timestamp,
                ROUND(o.strike_price - l.current_price, 2) AS price_above_current,
                CAST((o.expiration_date - cast(date_trunc('day', now()) as timestamp)) / 86400000000L AS INT) AS days_to_expiry,
                l.current_price
            FROM (
                (SELECT * FROM options_data
                 WHERE option_type = 'call')
                LATEST ON timestamp PARTITION BY option_ticker
            ) o
            JOIN latest_stock_prices l ON o.ticker = l.ticker
            WHERE 1=1
        """
        
        # Only ticker params + min_premium are passed to the query now
        params = list(tickers_upper) + [min_premium]
        
        # Add expiration date filters
        if days_to_expiry is not None:
            query += f"""
              AND o.expiration_date BETWEEN 
                  date_trunc('day', cast(now() as timestamp)) + {days_to_expiry - 1} * 24 * 60 * 60 * 1000000L
                  AND date_trunc('day', cast(now() as timestamp)) + {days_to_expiry + 1} * 24 * 60 * 60 * 1000000L
            """
        
        if max_days is not None:
            query += f"""
              AND CAST((o.expiration_date - cast(date_trunc('day', now()) as timestamp)) / 86400000000L AS INT) <= {max_days}
            """
        
        query += f"""
        ),
        with_premium AS (
            SELECT
                ticker,
                current_price,
                strike_price,
                price_above_current,
                -- fallback logic: ask -> bid -> 0.01 (minimal fallback)
                ROUND(COALESCE(ask, bid, 0.01), 2) AS option_premium,
                volume,
                ask,
                bid,
                delta,
                theta,
                implied_volatility,
                expiration_date,
                days_to_expiry,
                write_timestamp,
                last_quote_timestamp,
                option_ticker,
                -- contracts possible for specified position size
                ROUND(floor({position_size} / (current_price * 100)), 0) AS num_contracts,
                -- potential premium earned using fallback option_premium
                ROUND(floor({position_size} / (current_price * 100)) * (COALESCE(ask, bid, 0.01) * 100), 2) AS potential_premium,
                -- daily premium (amortized)
                CASE 
                    WHEN CAST((expiration_date - cast(date_trunc('day', now()) as timestamp)) / 86400000000L AS INT) > 0 
                    THEN ROUND(floor({position_size} / (current_price * 100)) * (COALESCE(ask, bid, 0.01) * 100) / CAST((expiration_date - cast(date_trunc('day', now()) as timestamp)) / 86400000000L AS INT), 2)
                    ELSE 0
                END AS daily_premium
            FROM filtered_options
        ),
        deduplicated AS (
            SELECT
                ticker,
                current_price,
                strike_price,
                price_above_current,
                option_premium,
                MAX(volume) AS volume,
                MAX(delta) AS delta,
                MAX(theta) AS theta,
                num_contracts,
                potential_premium,
                daily_premium,
                expiration_date,
                days_to_expiry,
                MAX(write_timestamp) AS write_timestamp,
                MAX(last_quote_timestamp) AS last_quote_timestamp,
                option_ticker
            FROM with_premium
            GROUP BY 
                ticker,
                current_price,
                strike_price,
                price_above_current,
                option_premium,
                num_contracts,
                potential_premium,
                daily_premium,
                expiration_date,
                days_to_expiry,
                option_ticker
        )
        SELECT
            ticker,
            ROUND(current_price, 2) AS current_price,
            strike_price,
            price_above_current,
            option_premium,
            delta,
            theta,
            volume,
            num_contracts,
            potential_premium,
            daily_premium,
            expiration_date,
            days_to_expiry,
            last_quote_timestamp,
            write_timestamp,
            option_ticker
        FROM deduplicated
        WHERE potential_premium >= ${premium_param}
        """
        
        try:
            df = await self.db.execute_select_sql(query, tuple(params))
            
            if not df.empty:
                # Convert timestamp columns
                if 'expiration_date' in df.columns:
                    df['expiration_date'] = pd.to_datetime(df['expiration_date'])
                if 'last_quote_timestamp' in df.columns:
                    df['last_quote_timestamp'] = pd.to_datetime(df['last_quote_timestamp'])
                if 'write_timestamp' in df.columns:
                    df['write_timestamp'] = pd.to_datetime(df['write_timestamp'])
                
                # Replace current_price with latest from DB and recompute dependent fields
                try:
                    ticker_col = df.columns[0]
                    current_price_col = 1
                    strike_price_col = 2
                    price_above_current_col = 3
                    option_premium_col = 4
                    num_contracts_col = 6
                    potential_premium_col = 7
                    daily_premium_col = 8
                    days_to_expiry_col = 10

                    unique_tickers = list(pd.unique(df[ticker_col]))
                    latest_prices: Dict[str, Optional[float]] = {}
                    for t in unique_tickers:
                        try:
                            latest_prices[t] = await self.db.get_latest_price(t)
                        except Exception:
                            latest_prices[t] = None

                    # Update current_price where we have a latest value
                    mapped_prices = df[ticker_col].map(lambda x: latest_prices.get(x))
                    # Preserve original where latest is None
                    df[current_price_col] = mapped_prices.combine_first(df[current_price_col])

                    # Recompute dependent fields
                    df[price_above_current_col] = (df[strike_price_col] - df[current_price_col]).round(2)

                    # num_contracts = floor(position_size / (current_price * 100))
                    df[num_contracts_col] = df[current_price_col].apply(lambda cp: 0 if pd.isna(cp) or cp <= 0 else math.floor(position_size / (cp * 100)))

                    # potential_premium = num_contracts * (option_premium * 100)
                    df[potential_premium_col] = (df[num_contracts_col] * (df[option_premium_col] * 100)).round(2)

                    # daily_premium = potential_premium / days_to_expiry (if > 0)
                    def _daily(row):
                        days = row[days_to_expiry_col]
                        return 0 if pd.isna(days) or days <= 0 else round(row[potential_premium_col] / days, 2)
                    df[daily_premium_col] = df.apply(_daily, axis=1)
                except Exception as _:
                    # If any recompute step fails, continue with original values
                    pass
                
                # Note: Filters will be applied after financial data is added in format_output
            
            return df
        except Exception as e:
            if not self.quiet:
                print(f"Error analyzing options: {e}", file=sys.stderr)
            return pd.DataFrame()
    
    def format_output(
        self,
        df: pd.DataFrame,
        financial_data: Dict[str, Dict[str, Any]],
        output_format: str = 'table',
        group_by: str = 'overall',
        output_file: Optional[str] = None,
        sort_by: Optional[str] = None,
        filters: Optional[List[FilterExpression]] = None,
        filter_logic: str = 'AND'
    ) -> str:
        """Format the analysis results for output."""
        if df.empty:
            return "No options data found matching the criteria."
        
        # Add financial information to the dataframe
        # QuestDB returns columns as integers, so we need to map them
        # Based on the query: Column 0 = ticker, Column 1 = current_price, etc.
        ticker_col = df.columns[0]  # First column should be ticker
        df['pe_ratio'] = df[ticker_col].map(lambda x: financial_data.get(x, {}).get('pe_ratio'))
        df['market_cap'] = df[ticker_col].map(lambda x: financial_data.get(x, {}).get('market_cap'))
        df['market_cap_b'] = df['market_cap'].apply(lambda x: round(x / 1e9, 2) if pd.notna(x) and x is not None else None)
        
        # Map integer columns to expected names based on the query structure
        # Query columns: ticker, current_price, strike_price, price_above_current, option_premium, 
        # delta, theta, volume, num_contracts, potential_premium, daily_premium, expiration_date, days_to_expiry, 
        # last_quote_timestamp, write_timestamp, option_ticker
        column_mapping = {
            0: 'ticker',
            1: 'current_price', 
            2: 'strike_price',
            3: 'price_above_current',
            4: 'option_premium',
            5: 'delta',
            6: 'theta',
            7: 'volume',
            8: 'num_contracts',
            9: 'potential_premium',
            10: 'daily_premium',
            11: 'expiration_date',
            12: 'days_to_expiry',
            13: 'last_quote_timestamp',
            14: 'write_timestamp',
            15: 'option_ticker'
        }
        
        # Rename columns
        df_renamed = df.rename(columns=column_mapping)
        
        # Add option premium percentage calculation
        df_renamed['option_premium_percentage'] = (df_renamed['option_premium'] / df_renamed['current_price'] * 100).round(2)
        # Add premium vs above difference percentage relative to price_above_current
        df_renamed['premium_above_diff_percentage'] = (
            (
                (df_renamed['option_premium'] - df_renamed['price_above_current']) / df_renamed['price_above_current']
            ).where(df_renamed['price_above_current'] != 0)
            * 100
        ).round(2)
        
        # Convert timestamps to EST/EDT (America/New_York) for display, except expiration_date which should be UTC
        try:
            # Handle expiration_date (UTC)
            if 'expiration_date' in df_renamed.columns:
                ser = pd.to_datetime(df_renamed['expiration_date'], errors='coerce')
                # Ensure timezone-aware UTC
                if getattr(ser.dt, 'tz', None) is None:
                    ser = ser.dt.tz_localize('UTC')
                else:
                    ser = ser.dt.tz_convert('UTC')
                df_renamed['expiration_date'] = ser.dt.strftime('%Y-%m-%d %H:%M:%S')

            # Handle other timestamps in America/New_York
            for ts_col in ['last_quote_timestamp', 'write_timestamp']:
                if ts_col in df_renamed.columns:
                    ser = pd.to_datetime(df_renamed[ts_col], errors='coerce')
                    # Localize naive timestamps to UTC, then convert to America/New_York
                    if getattr(ser.dt, 'tz', None) is None:
                        ser = ser.dt.tz_localize('UTC')
                    ser = ser.dt.tz_convert('America/New_York')
                    # Format as string for clean table output
                    df_renamed[ts_col] = ser.dt.strftime('%Y-%m-%d %H:%M:%S')
        except Exception:
            # If conversion fails, leave as-is
            pass
        
        # Reorder columns for better display
        display_columns = [
            'ticker', 'pe_ratio', 'market_cap_b', 'current_price', 'strike_price',
            'price_above_current', 'option_premium', 'option_premium_percentage', 'premium_above_diff_percentage',
            'delta', 'theta',
            'potential_premium', 'daily_premium', 'expiration_date', 'days_to_expiry',
            #'volume', 'num_contracts', 'last_quote_timestamp', 'write_timestamp', 'option_ticker'
            'volume', 'num_contracts', 'write_timestamp', 'option_ticker'
        ]
        
        # Only include columns that exist in the dataframe
        available_columns = [col for col in display_columns if col in df_renamed.columns]
        df_display = df_renamed[available_columns].copy()
        
        # Apply filters if provided (after financial data is added)
        if filters:
            df_display = FilterParser.apply_filters(df_display, filters, filter_logic)
        
        # Apply sorting if specified (supports full names and abbreviated headers)
        if sort_by:
            # Map abbreviated headers back to full column names
            header_reverse_map = {v: k for k, v in self._create_compact_headers(df_display).items()}
            sort_key = sort_by
            if sort_by in header_reverse_map:
                sort_key = header_reverse_map[sort_by]
            # Also support substring resolution for field names
            if sort_key not in df_display.columns:
                # try case-insensitive substring match
                candidates = [c for c in df_display.columns if sort_key.lower() in str(c).lower()]
                if len(candidates) == 1:
                    sort_key = candidates[0]
            if sort_key in df_display.columns:
                df_display = df_display.sort_values(by=sort_key, ascending=False)
        
        if group_by == 'ticker':
            # Group by ticker and show results per ticker
            output_lines = []
            for ticker in sorted(df_display['ticker'].unique()):
                ticker_data = df_display[df_display['ticker'] == ticker]
                output_lines.append(f"\n--- {ticker} ---")
                
                if output_format == 'table':
                    # Format numeric columns for better display
                    ticker_data_formatted = ticker_data.copy()
                    for col in ['current_price', 'strike_price', 'price_above_current', 'option_premium', 'potential_premium', 'daily_premium']:
                        if col in ticker_data_formatted.columns:
                            ticker_data_formatted[col] = ticker_data_formatted[col].apply(lambda x: f"${x:.2f}" if pd.notna(x) else "N/A")
                    
                    for col in ['pe_ratio', 'market_cap_b']:
                        if col in ticker_data_formatted.columns:
                            ticker_data_formatted[col] = ticker_data_formatted[col].apply(lambda x: f"{x:.2f}" if pd.notna(x) else "N/A")
                    
                    for col in ['option_premium_percentage']:
                        if col in ticker_data_formatted.columns:
                            ticker_data_formatted[col] = ticker_data_formatted[col].apply(lambda x: f"{x:.2f}%" if pd.notna(x) else "N/A")
                    
                    for col in ['premium_above_diff_percentage']:
                        if col in ticker_data_formatted.columns:
                            ticker_data_formatted[col] = ticker_data_formatted[col].apply(lambda x: f"{x:.2f}%" if pd.notna(x) else "N/A")
                    
                    # Create compact headers and rename columns to ensure alignment
                    compact_headers = self._create_compact_headers(ticker_data_formatted)
                    ticker_data_formatted = ticker_data_formatted.rename(columns=compact_headers)
                    
                    table = tabulate(
                        ticker_data_formatted,
                        headers='keys',
                        tablefmt='grid',
                        showindex=False
                    )
                    output_lines.append(table)
                else:
                    # CSV format
                    output_lines.append(ticker_data.to_csv(index=False))
        else:
            # Overall ranking
            if output_format == 'table':
                # Format numeric columns for better display
                df_formatted = df_display.copy()
                for col in ['current_price', 'strike_price', 'price_above_current', 'option_premium', 'potential_premium', 'daily_premium']:
                    if col in df_formatted.columns:
                        df_formatted[col] = df_formatted[col].apply(lambda x: f"${x:.2f}" if pd.notna(x) else "N/A")
                
                for col in ['pe_ratio', 'market_cap_b']:
                    if col in df_formatted.columns:
                        df_formatted[col] = df_formatted[col].apply(lambda x: f"{x:.2f}" if pd.notna(x) else "N/A")
                
                for col in ['option_premium_percentage']:
                    if col in df_formatted.columns:
                        df_formatted[col] = df_formatted[col].apply(lambda x: f"{x:.2f}%" if pd.notna(x) else "N/A")
                
                for col in ['premium_above_diff_percentage']:
                    if col in df_formatted.columns:
                        df_formatted[col] = df_formatted[col].apply(lambda x: f"{x:.2f}%" if pd.notna(x) else "N/A")
                
                # Create compact headers and rename columns to ensure alignment
                compact_headers = self._create_compact_headers(df_formatted)
                df_formatted = df_formatted.rename(columns=compact_headers)
                
                output_lines = [tabulate(
                    df_formatted,
                    headers='keys',
                    tablefmt='grid',
                    showindex=False
                )]
            else:
                # CSV format
                output_lines = [df_display.to_csv(index=False)]
        
        result = "\n".join(output_lines)
        
        # Save to file if specified
        if output_file:
            with open(output_file, 'w') as f:
                f.write(result)
            if not self.quiet:
                print(f"Results saved to {output_file}")
        
        return result


async def main():
    """Main function to run the options analyzer."""
    parser = argparse.ArgumentParser(
        description="Analyze covered call opportunities across all strike prices and tickers.",
        epilog="""
Examples:
  # Analyze all available tickers
  python options_analyzer.py --db-conn questdb://user:pass@host:8812/db

  # Analyze specific symbols with 14-day expiry window
  python options_analyzer.py --symbols AAPL MSFT GOOGL --days 14 --output csv

  # Analyze S&P 500 stocks sorted by daily premium
  python options_analyzer.py --types sp-500 --sort daily_premium --group-by ticker

  # Filter by volume and max days save to file
  python options_analyzer.py --min-volume 1000 --max-days 30 --output results.csv

  # Show only high-premium opportunities
  python options_analyzer.py --min-premium 5000 --sort potential_premium

  # Filter by P/E ratio and market cap (using B/M suffixes)
  python options_analyzer.py --filter "pe_ratio > 20" --filter "market_cap < 1B"

  # Filter with OR logic
  python options_analyzer.py --filter "pe_ratio > 30" --filter "market_cap > 5B" --filter-logic OR

  # Filter for options with volume data
  python options_analyzer.py --filter "volume exists" --filter "pe_ratio exists"

  # Market cap filtering with different formats
  python options_analyzer.py --filter "market_cap > 500M" --filter "market_cap < 3.5B"

  # Field-to-field comparisons
  python options_analyzer.py --filter "num_contracts > volume" --filter "potential_premium > daily_premium"

  # Mathematical expressions in filters
  python options_analyzer.py --filter "num_contracts*0.1 > volume" --filter "potential_premium+1000 > daily_premium"

  # Percentage fields (derived)
  # option_premium_percentage = (option_premium / current_price) * 100
  # premium_above_diff_percentage = ((option_premium - price_above_current) / price_above_current) * 100
  python options_analyzer.py --filter "option_premium_percentage >= 10" --filter "premium_above_diff_percentage > 0"
"""
    )
    
    # Add symbol input arguments using common library
    add_symbol_arguments(parser, required=False)
    
    # Database connection
    parser.add_argument(
        '--db-conn',
        type=str,
        required=True,
        help="QuestDB connection string (e.g. questdb://user:pass@host:8812/db)."
    )
    
    # Analysis parameters
    parser.add_argument(
        '--days',
        type=int,
        default=None,
        help="Number of days to expiry window (e.g. 14 for ±14 days around target). If not specified analyze all available expirations."
    )
    parser.add_argument(
        '--max-days',
        type=int,
        default=None,
        help="Maximum days to expiry filter."
    )
    parser.add_argument(
        '--min-volume',
        type=int,
        default=0,
        help="Minimum volume filter for options (default: 0)."
    )
    parser.add_argument(
        '--min-premium',
        type=float,
        default=0.0,
        help="Minimum potential premium filter (default: 0.0)."
    )
    parser.add_argument(
        '--position-size',
        type=float,
        default=1000000.0,
        help="Position size in dollars for premium calculations (default: 1000000.0 = $1M)."
    )
    parser.add_argument(
        '--data-dir',
        default='data',
        help="Directory to store data files (default: data)."
    )
    
    # Filter options
    parser.add_argument(
        '--filter',
        action='append',
        type=str,
        help="Filter expressions (can be used multiple times). Format: 'field operator value' or 'field operator field' or 'field exists/not_exists' or 'field*multiplier operator value'. "
             "Supported fields: pe_ratio market_cap volume num_contracts potential_premium daily_premium current_price strike_price price_above_current option_premium option_premium_percentage premium_above_diff_percentage days_to_expiry. "
             "Supported operators: > >= < <= == != exists not_exists. "
             "Mathematical operations: + - * / (e.g. 'num_contracts*0.1 > volume' 'potential_premium+1000 > daily_premium'). "
             "Derived fields: option_premium_percentage = (option_premium / current_price) * 100; "
             "premium_above_diff_percentage = ((option_premium - price_above_current) / price_above_current) * 100. "
             "Market cap values support T (trillion) B (billion) and M (million) suffixes. "
             "Examples: 'pe_ratio > 20' or 'market_cap < 3.5T' or 'num_contracts > volume' or 'num_contracts*0.1 > volume' or 'potential_premium > daily_premium' or 'volume exists' or "
             "'option_premium_percentage >= 10' or 'premium_above_diff_percentage > 0'. "
             "Multiple expressions in one --filter can be comma-separated."
    )
    parser.add_argument(
        '--filter-logic',
        choices=['AND', 'OR'],
        default='AND',
        help="Logic to combine multiple filter expressions (default: AND)."
    )
    
    # Output options
    parser.add_argument(
        '--output',
        type=str,
        default='table',
        help="Output format: 'table' 'csv' or filename (e.g. 'results.csv')."
    )
    parser.add_argument(
        '--group-by',
        choices=['ticker', 'overall'],
        default='overall',
        help="Group results by ticker or show overall ranking (default: overall)."
    )
    parser.add_argument(
        '--sort',
        type=str,
        default='daily_premium',
        help=(
            "Sort by any displayed field (full or abbreviated header). "
            "Examples: daily_premium potential_premium ticker days_to_expiry option_premium_percentage premium_above_diff_percentage "
            "or abbreviations like TKR PRC STRK PREM%% DIFF%% POT_PREM DAY_PREM."
        )
    )
    parser.add_argument(
        '--quiet',
        action='store_true',
        help="Suppress progress output."
    )
    
    # If help is requested, print and exit early to avoid running any analysis code
    if any(flag in sys.argv for flag in ("-h", "--help")):
        parser.print_help()
        sys.exit(0)

    args = parser.parse_args()
    
    # Initialize analyzer
    analyzer = OptionsAnalyzer(args.db_conn, args.quiet)
    await analyzer.initialize()
    
    # Get symbols list using common library
    symbols_list = await fetch_lists_data(args, args.quiet)
    if not symbols_list:
        print("No symbols specified or found. Exiting.", file=sys.stderr)
        sys.exit(1)
    
    if not args.quiet:
        print(f"Analyzing {len(symbols_list)} tickers...")
    
    # Get financial information
    financial_data = await analyzer.get_financial_info(symbols_list)
    
    # Parse filters
    filters = []
    if hasattr(args, 'filter') and args.filter:
        try:
            # Normalize whitespace in each filter input (collapse internal spaces)
            normalized_filters = [' '.join(f.split()) for f in args.filter]
            filters = FilterParser.parse_filters(normalized_filters)
            if not args.quiet and filters:
                print(f"Applied {len(filters)} filter(s) with {args.filter_logic} logic:")
                for i, f in enumerate(filters, 1):
                    print(f"  {i}. {f}")
        except Exception as e:
            print(f"Error parsing filters: {e}", file=sys.stderr)
            sys.exit(1)
    
    # Analyze options
    df = await analyzer.analyze_options(
        tickers=symbols_list,
        days_to_expiry=args.days,
        min_volume=args.min_volume,
        max_days=args.max_days,
        min_premium=args.min_premium,
        position_size=args.position_size,
        filters=filters,
        filter_logic=args.filter_logic
    )
    
    if df.empty:
        print("No options data found matching the criteria.")
        return
    
    # Determine output format and file
    output_format = 'table'
    output_file = None
    
    if args.output.lower() == 'csv':
        output_format = 'csv'
    elif args.output.lower() != 'table':
        output_file = args.output
        if args.output.endswith('.csv'):
            output_format = 'csv'
        else:
            output_format = 'table'
    
    # Normalize sort input by stripping all whitespace characters
    import re as _re
    sort_arg = _re.sub(r"\s+", "", args.sort) if hasattr(args, 'sort') and args.sort else None

    # Format and display results
    result = analyzer.format_output(
        df=df,
        financial_data=financial_data,
        output_format=output_format,
        group_by=args.group_by,
        output_file=output_file,
        sort_by=sort_arg,
        filters=filters,
        filter_logic=args.filter_logic
    )
    
    if not args.quiet or output_file is None:
        print(result)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nOperation cancelled by user.")
    except Exception as e:
        print(f"\nAn unexpected error occurred: {e}", file=sys.stderr)
        sys.exit(1)
