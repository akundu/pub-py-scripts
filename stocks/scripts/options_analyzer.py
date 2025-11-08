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
        'theta': float,
        # Spread-related fields
        'long_strike_price': float,
        'long_option_premium': float,
        'long_days_to_expiry': int,
        'long_delta': float,
        'long_theta': float,
        'premium_diff': float,
        'short_premium_total': float,
        'short_daily_premium': float,
        'long_premium_total': float,
        'net_premium': float,
        'net_daily_premium': float
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
    
    def __init__(self, db_conn: str, quiet: bool = False, debug: bool = False, enable_cache: bool = True):
        """Initialize the options analyzer with database connection."""
        self.db_conn = db_conn
        self.quiet = quiet
        self.debug = debug
        self.enable_cache = enable_cache
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
            'option_ticker': 'OPT_TKR',
            # Spread-related columns
            'long_strike_price': 'L_STRK',
            'long_option_premium': 'L_PREM',
            'long_expiration_date': 'L_EXP',
            'long_days_to_expiry': 'L_DAYS',
            'long_option_ticker': 'L_OPT_TKR',
            'long_delta': 'L_DEL',
            'long_theta': 'L_TH',
            'long_volume': 'L_VOL',
            'premium_diff': 'PREM_DIFF',
            'short_premium_total': 'S_PREM_TOT',
            'short_daily_premium': 'S_DAY_PREM',
            'long_premium_total': 'L_PREM_TOT',
            'net_premium': 'NET_PREM',
            'net_daily_premium': 'NET_DAY'
        }
        
        for col in df.columns:
            if col in header_mapping:
                compact_headers[col] = header_mapping[col]
            else:
                # For unknown columns, use the original name but truncate if too long
                compact_headers[col] = col[:8] if len(col) > 8 else col
        
        return compact_headers
    
    def _format_csv_output(
        self, 
        df: pd.DataFrame, 
        delimiter: str = ',', 
        quoting: str = 'minimal', 
        group_by: str = 'overall',
        output_file: Optional[str] = None
    ) -> str:
        """Format DataFrame as CSV with proper formatting."""
        import csv
        
        # Convert quoting string to csv module constant
        quoting_map = {
            'minimal': csv.QUOTE_MINIMAL,
            'all': csv.QUOTE_ALL,
            'none': csv.QUOTE_NONE,
            'nonnumeric': csv.QUOTE_NONNUMERIC
        }
        csv_quoting = quoting_map.get(quoting, csv.QUOTE_MINIMAL)
        
        # Create a copy for CSV formatting
        df_csv = df.copy()
        
        # Format numeric columns for CSV (remove $ symbols and % symbols for cleaner data)
        for col in ['current_price', 'strike_price', 'price_above_current', 'option_premium', 'potential_premium', 'daily_premium',
                    'long_strike_price', 'long_option_premium', 'premium_diff', 'short_premium_total', 'short_daily_premium', 'long_premium_total', 'net_premium', 'net_daily_premium']:
            if col in df_csv.columns:
                df_csv[col] = df_csv[col].apply(lambda x: float(x.replace('$', '').replace(',', '')) if isinstance(x, str) and '$' in str(x) else x)
        
        for col in ['pe_ratio', 'market_cap_b']:
            if col in df_csv.columns:
                df_csv[col] = df_csv[col].apply(lambda x: float(x.replace(',', '')) if isinstance(x, str) and ',' in str(x) else x)
        
        for col in ['option_premium_percentage', 'premium_above_diff_percentage']:
            if col in df_csv.columns:
                df_csv[col] = df_csv[col].apply(lambda x: float(x.replace('%', '').replace(',', '')) if isinstance(x, str) and '%' in str(x) else x)
        
        # Handle grouping
        if group_by == 'ticker':
            # For CSV, we'll create a single CSV with all data but add a grouping column
            df_csv['group'] = df_csv['ticker']
            # Sort by ticker first, then by the original sort order
            df_csv = df_csv.sort_values(['ticker'])
        
        # Generate CSV content
        csv_content = df_csv.to_csv(
            index=False, 
            sep=delimiter, 
            quoting=csv_quoting,
            na_rep='',
            float_format='%.2f'
        )
        
        # Save to file if specified
        if output_file:
            with open(output_file, 'w', newline='', encoding='utf-8') as f:
                f.write(csv_content)
            if not self.quiet:
                print(f"CSV results saved to {output_file}")
        
        return csv_content
        
    async def initialize(self):
        """Initialize database connection."""
        try:
            # Set log level based on quiet mode - INFO for non-quiet, WARNING for quiet
            log_level = "WARNING" if self.quiet else "INFO"
            self.db = get_stock_db('questdb', db_config=self.db_conn, enable_cache=self.enable_cache, log_level=log_level)
            if not self.quiet:
                cache_status = "enabled" if self.enable_cache else "disabled"
                print(f"Database connection established successfully (cache: {cache_status}).", file=sys.stderr)
        except Exception as e:
            print(f"Error connecting to database: {e}", file=sys.stderr)
            sys.exit(1)
    
    async def get_available_tickers(self) -> List[str]:
        """Get all available tickers from the daily_prices table."""
        try:
            if self.debug or not self.quiet:
                print("DEBUG: Checking daily_prices table for available tickers", file=sys.stderr)
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
        
        if self.debug or not self.quiet:
            print(f"DEBUG: Fetching financial info for {len(tickers)} tickers", file=sys.stderr)
        
        for ticker in tickers:
            try:
                # Use the cached get_financial_info method instead of direct SQL
                df = await self.db.get_financial_info(ticker)
                
                if not df.empty:
                    # Map column names to expected fields
                    row = df.iloc[0]
                    # Try different possible column names
                    pe_ratio = None
                    if 'price_to_earnings' in df.columns:
                        pe_ratio = row['price_to_earnings']
                    elif 'pe_ratio' in df.columns:
                        pe_ratio = row['pe_ratio']
                    
                    market_cap = row['market_cap'] if 'market_cap' in df.columns else None
                    price = row['price'] if 'price' in df.columns else None
                    
                    financial_data[ticker] = {
                        'pe_ratio': pe_ratio,
                        'market_cap': market_cap,
                        'price': price
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
        filter_logic: str = 'AND',
        use_market_time: bool = True,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        max_concurrent: int = 10,
        batch_size: int = 50,
        timestamp_lookback_days: int = 7,
        max_workers: int = 4,
        spread_mode: bool = False,
        spread_strike_tolerance: float = 0.0,
        spread_long_days: int = 90,
        spread_long_days_tolerance: int = 14,
        spread_long_min_days: Optional[int] = None,
        min_write_timestamp: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Analyze covered call opportunities for the given tickers.
        
        Args:
            tickers: List of ticker symbols to analyze
            days_to_expiry: Number of days to expiry (if None, analyze all available)
            min_volume: Minimum volume filter
            max_days: Maximum days from today for expiration (convenience param that sets end_date, overrides end_date if both provided)
            min_premium: Minimum potential premium filter
            position_size: Position size in dollars for calculations
            filters: List of FilterExpression objects to apply
            filter_logic: Logic to combine filters ('AND' or 'OR')
            use_market_time: Whether to use market hours logic for price fetching
            start_date: Start date for option expiration filtering (YYYY-MM-DD format, defaults to today)
            end_date: End date for option expiration filtering (YYYY-MM-DD format, defaults to None, overridden by max_days)
            max_concurrent: Maximum concurrent queries per batch (lower = less memory)
            batch_size: Number of tickers per batch (lower = less memory)
            timestamp_lookback_days: Days to look back for option timestamp data (default: 7, lower = less memory but may miss data)
            max_workers: Number of worker processes for multiprocessing (default: 4, typically CPU count)
            spread_mode: Enable calendar spread analysis (sell short-term, buy long-term)
            spread_strike_tolerance: Percentage tolerance for matching strike prices (e.g., 5.0 for ±5%)
            spread_long_days: Maximum/target days to expiry for long-term options to buy
            spread_long_days_tolerance: Days tolerance for long option expiration window (default: 14, searches ±14 days around target, ignored if spread_long_min_days is set)
            spread_long_min_days: Minimum days to expiry for long options (if set, searches from min to spread_long_days instead of using tolerance)
            min_write_timestamp: Minimum write timestamp for options (EST format like '2025-11-05 10:00:00', filters out older options)
            
        Returns:
            DataFrame with analysis results
        """
        if not tickers:
            return pd.DataFrame()
        
        # Convert tickers to uppercase for database compatibility
        tickers_upper = [ticker.upper() for ticker in tickers]
        
        # Default start_date to today if not specified
        if start_date is None:
            from datetime import date
            start_date = date.today().strftime('%Y-%m-%d')
        
        # If max_days is set, calculate end_date from today (overrides explicit end_date)
        if max_days is not None:
            from datetime import date, timedelta
            end_date = (date.today() + timedelta(days=max_days)).strftime('%Y-%m-%d')
            if not self.quiet:
                print(f"Using max_days={max_days}: filtering options expiring through {end_date}")
        
        # Use memory-efficient batch fetching instead of single large query
        try:
            if self.debug or not self.quiet:
                print(f"DEBUG: Starting options fetch for {len(tickers_upper)} tickers", file=sys.stderr)
                print(f"DEBUG: Date range: {start_date} to {end_date}", file=sys.stderr)
                print(f"DEBUG: Tickers: {tickers_upper[:10]}{'...' if len(tickers_upper) > 10 else ''}", file=sys.stderr)
            
            # Choose between asyncio-only or hybrid asyncio+multiprocess based on max_workers
            if max_workers > 1:
                if not self.quiet:
                    print(f"Using multiprocess mode with {max_workers} workers")
                # Use hybrid asyncio + multiprocessing
                options_df = await self.db.get_latest_options_data_batch_multiprocess(
                    tickers=tickers_upper,
                    start_datetime=start_date,
                    end_datetime=end_date,
                    batch_size=batch_size,
                    max_workers=max_workers,
                    timestamp_lookback_days=timestamp_lookback_days
                )
            else:
                # Use asyncio-only (single process)
                if not self.quiet:
                    print("Using single-process mode")
                options_df = await self.db.get_latest_options_data_batch(
                    tickers=tickers_upper,
                    start_datetime=start_date,
                    end_datetime=end_date,
                    max_concurrent=max_concurrent,
                    batch_size=batch_size,
                    timestamp_lookback_days=timestamp_lookback_days
                )
            
            if self.debug or not self.quiet:
                print(f"DEBUG: Fetched {len(options_df)} total options from database", file=sys.stderr)
                if not options_df.empty:
                    print(f"DEBUG: Options columns: {list(options_df.columns)}", file=sys.stderr)
                    if 'ticker' in options_df.columns:
                        unique_tickers = options_df['ticker'].unique().tolist()
                        print(f"DEBUG: Options tickers: {unique_tickers[:10]}{'...' if len(unique_tickers) > 10 else ''}", file=sys.stderr)
                    if 'option_type' in options_df.columns:
                        option_types = options_df['option_type'].unique().tolist()
                        print(f"DEBUG: Option types found: {option_types}", file=sys.stderr)
                else:
                    print("DEBUG: No options data returned from database", file=sys.stderr)
            
            if options_df.empty:
                if not self.quiet:
                    print("DEBUG: Empty options DataFrame after fetch. Possible reasons:", file=sys.stderr)
                    print("  1. No options data in database for these tickers", file=sys.stderr)
                    print(f"  2. No options expiring between {start_date} and {end_date}", file=sys.stderr)
                    print(f"  3. Options exist but outside timestamp_lookback_days={timestamp_lookback_days} window", file=sys.stderr)
                return pd.DataFrame()
            
            # Filter for call options only
            before_call_filter = len(options_df)
            if 'option_type' in options_df.columns:
                options_df = options_df[options_df['option_type'] == 'call']
                if self.debug or not self.quiet:
                    print(f"DEBUG: After call filter: {len(options_df)} options (was {before_call_filter})", file=sys.stderr)
            else:
                if self.debug or not self.quiet:
                    print("DEBUG: Warning: 'option_type' column not found in options DataFrame", file=sys.stderr)
            
            if options_df.empty:
                if not self.quiet:
                    print("DEBUG: No call options found after filtering", file=sys.stderr)
                return pd.DataFrame()
            
            # Get latest stock prices for all tickers
            stock_prices = {}
            for ticker in tickers_upper:
                try:
                    price = await self.db.get_latest_price(ticker, use_market_time=use_market_time)
                    if price:
                        stock_prices[ticker] = price
                except Exception as e:
                    if not self.quiet:
                        print(f"Warning: Could not fetch price for {ticker}: {e}", file=sys.stderr)
            
            if self.debug or not self.quiet:
                print(f"DEBUG: Fetched stock prices for {len(stock_prices)}/{len(tickers_upper)} tickers", file=sys.stderr)
                if stock_prices:
                    sample_prices = list(stock_prices.items())[:5]
                    print(f"DEBUG: Sample prices: {sample_prices}", file=sys.stderr)
            
            if not stock_prices:
                if not self.quiet:
                    print("DEBUG: No stock prices fetched. Cannot calculate option metrics.", file=sys.stderr)
                return pd.DataFrame()
            
            # Build the analysis DataFrame
            df = options_df.copy()
            
            # Map stock prices to each option row
            df['current_price'] = df['ticker'].map(stock_prices)
            
            # Filter out rows where we don't have a stock price
            before_price_filter = len(df)
            df = df[df['current_price'].notna()]
            
            if self.debug or not self.quiet:
                print(f"DEBUG: After price mapping: {len(df)} options (was {before_price_filter})", file=sys.stderr)
            
            if df.empty:
                if not self.quiet:
                    print("DEBUG: No options remaining after price mapping", file=sys.stderr)
                return pd.DataFrame()
            
            # Calculate derived fields
            df['strike_price'] = df['strike_price'].round(2)
            df['price_above_current'] = (df['strike_price'] - df['current_price']).round(2)
            
            # Option premium: use ask, fallback to bid, fallback to 0.01
            df['option_premium'] = df.apply(
                lambda row: round(row['ask'] if pd.notna(row.get('ask')) else (row['bid'] if pd.notna(row.get('bid')) else 0.01), 2),
                axis=1
            )
            
            # Calculate days to expiry
            df['expiration_date'] = pd.to_datetime(df['expiration_date'])
            # Ensure expiration_date is timezone-aware UTC
            if df['expiration_date'].dt.tz is None:
                df['expiration_date'] = df['expiration_date'].dt.tz_localize('UTC')
            today = pd.Timestamp.now(tz='UTC').normalize()
            df['days_to_expiry'] = ((df['expiration_date'] - today).dt.total_seconds() / 86400).astype(int)
            
            # Apply days_to_expiry filter if specified
            if days_to_expiry is not None:
                before_days_filter = len(df)
                df = df[
                    (df['days_to_expiry'] >= days_to_expiry - 1) &
                    (df['days_to_expiry'] <= days_to_expiry + 1)
                ]
                if self.debug or not self.quiet:
                    print(f"DEBUG: After days_to_expiry filter ({days_to_expiry}): {len(df)} options (was {before_days_filter})", file=sys.stderr)
            
            # Calculate position metrics
            df['num_contracts'] = df['current_price'].apply(
                lambda cp: 0 if pd.isna(cp) or cp <= 0 else math.floor(position_size / (cp * 100))
            )
            
            # Potential premium = num_contracts * (option_premium * 100)
            df['potential_premium'] = (df['num_contracts'] * (df['option_premium'] * 100)).round(2)
            
            # Daily premium = potential_premium / days_to_expiry
            df['daily_premium'] = df.apply(
                lambda row: 0 if row['days_to_expiry'] <= 0 else round(row['potential_premium'] / row['days_to_expiry'], 2),
                axis=1
            )
            
            # Apply filters
            if min_volume > 0:
                before_volume_filter = len(df)
                df = df[df['volume'] >= min_volume]
                if self.debug or not self.quiet:
                    print(f"DEBUG: After min_volume filter ({min_volume}): {len(df)} options (was {before_volume_filter})", file=sys.stderr)
            
            if min_premium > 0.0:
                before_premium_filter = len(df)
                df = df[df['potential_premium'] >= min_premium]
                if self.debug or not self.quiet:
                    print(f"DEBUG: After min_premium filter ({min_premium}): {len(df)} options (was {before_premium_filter})", file=sys.stderr)
            
            # Apply write timestamp filter if specified
            if min_write_timestamp:
                try:
                    # Parse EST timestamp and convert to UTC for comparison
                    import pytz
                    est = pytz.timezone('America/New_York')
                    # Parse the input timestamp (assume EST)
                    min_ts = pd.to_datetime(min_write_timestamp)
                    # If timezone-naive, localize to EST
                    if min_ts.tz is None:
                        min_ts = est.localize(min_ts)
                    # Convert to UTC for comparison
                    min_ts_utc = min_ts.astimezone(pytz.UTC)
                    
                    # Ensure write_timestamp column is datetime and timezone-aware
                    if 'write_timestamp' in df.columns:
                        df['write_timestamp'] = pd.to_datetime(df['write_timestamp'])
                        # If timezone-naive, assume UTC
                        if df['write_timestamp'].dt.tz is None:
                            df['write_timestamp'] = df['write_timestamp'].dt.tz_localize('UTC')
                        # Filter
                        df = df[df['write_timestamp'] >= min_ts_utc]
                        if not self.quiet:
                            print(f"Filtered options to those written after {min_write_timestamp} EST ({min_ts_utc} UTC)")
                except Exception as e:
                    if not self.quiet:
                        print(f"Warning: Could not apply write timestamp filter: {e}", file=sys.stderr)

            
            # Round numeric columns
            for col in ['bid', 'ask', 'delta', 'theta']:
                if col in df.columns:
                    df[col] = df[col].round(2)
            
            # Convert timestamps
            for ts_col in ['last_quote_timestamp', 'write_timestamp']:
                if ts_col in df.columns:
                    df[ts_col] = pd.to_datetime(df[ts_col])
            
            # Select and order columns for output
            output_cols = [
                'ticker', 'current_price', 'strike_price', 'price_above_current',
                'option_premium', 'delta', 'theta', 'volume', 'num_contracts',
                'potential_premium', 'daily_premium', 'expiration_date', 'days_to_expiry',
                'last_quote_timestamp', 'write_timestamp', 'option_ticker'
            ]
            
            # Only include columns that exist
            available_cols = [col for col in output_cols if col in df.columns]
            df = df[available_cols]
            
            # If spread mode is enabled, match short-term options with long-term options
            if spread_mode:
                if not self.quiet:
                    print(f"\n=== Starting Spread Analysis ===")
                    print(f"Short-term options found: {len(df)}")
                    if not df.empty:
                        print(f"Short-term tickers: {df['ticker'].unique().tolist()}")
                        print(f"Short-term strike range: ${df['strike_price'].min():.2f} to ${df['strike_price'].max():.2f}")
                        print(f"Short-term days to expiry: {df['days_to_expiry'].min()} to {df['days_to_expiry'].max()}")
                if self.debug:
                    print(f"DEBUG: Spread mode parameters:", file=sys.stderr)
                    print(f"  spread_strike_tolerance: {spread_strike_tolerance}%", file=sys.stderr)
                    print(f"  spread_long_days: {spread_long_days}", file=sys.stderr)
                    print(f"  spread_long_min_days: {spread_long_min_days}", file=sys.stderr)
                    print(f"  spread_long_days_tolerance: {spread_long_days_tolerance}", file=sys.stderr)
                
                df = await self._create_spread_analysis(
                    df_short=df,
                    tickers=tickers_upper,
                    spread_strike_tolerance=spread_strike_tolerance,
                    spread_long_days=spread_long_days,
                    spread_long_days_tolerance=spread_long_days_tolerance,
                    spread_long_min_days=spread_long_min_days,
                    start_date=start_date,
                    end_date=end_date,
                    max_concurrent=max_concurrent,
                    batch_size=batch_size,
                timestamp_lookback_days=timestamp_lookback_days,
                max_workers=max_workers,
                position_size=position_size,
                min_write_timestamp=min_write_timestamp
            )
            
            if self.debug or not self.quiet:
                print(f"DEBUG: Final options count before spread/filtering: {len(df)}", file=sys.stderr)
            
            return df
        except Exception as e:
            if not self.quiet:
                print(f"Error analyzing options: {e}", file=sys.stderr)
            import traceback
            if self.debug:
                traceback.print_exc()
            return pd.DataFrame()
    
    async def _create_spread_analysis(
        self,
        df_short: pd.DataFrame,
        tickers: List[str],
        spread_strike_tolerance: float,
        spread_long_days: int,
        spread_long_days_tolerance: int,
            spread_long_min_days: Optional[int],
            start_date: Optional[str],
            end_date: Optional[str],
            max_concurrent: int,
            batch_size: int,
            timestamp_lookback_days: int,
            max_workers: int,
            position_size: float,
            min_write_timestamp: Optional[str]
        ) -> pd.DataFrame:
        """
        Match short-term options with long-term options to create calendar spread analysis.
        
        Args:
            df_short: DataFrame with short-term option analysis
            tickers: List of tickers to analyze
            spread_strike_tolerance: Percentage tolerance for strike matching
            spread_long_days: Maximum/target days to expiry for long options
            spread_long_days_tolerance: Days tolerance for long option expiration window (e.g., ±14 days, ignored if spread_long_min_days is set)
            spread_long_min_days: Minimum days to expiry for long options (if set, searches from min to max instead of using tolerance)
            Other args: Same as analyze_options
            
        Returns:
            DataFrame with spread analysis including long option details and net calculations
        """
        if df_short.empty:
            return df_short
        
        # Calculate the target date range for long options (around spread_long_days from today)
        from datetime import date, timedelta
        today = date.today()
        
        # If spread_long_min_days is set, use it as the lower bound; otherwise use tolerance-based approach
        if spread_long_min_days is not None:
            long_start_date = (today + timedelta(days=spread_long_min_days)).strftime('%Y-%m-%d')
            long_end_date = (today + timedelta(days=spread_long_days)).strftime('%Y-%m-%d')
            if not self.quiet:
                print(f"Fetching long-term options expiring between {spread_long_min_days} and {spread_long_days} days")
                print(f"  Date range: {long_start_date} to {long_end_date}")
            if self.debug:
                print(f"DEBUG: Calculated long-term date range: {long_start_date} to {long_end_date} (from today {today})", file=sys.stderr)
        else:
            # Allow ±N days window around the target long expiry (configurable)
            long_start_date = (today + timedelta(days=spread_long_days - spread_long_days_tolerance)).strftime('%Y-%m-%d')
            long_end_date = (today + timedelta(days=spread_long_days + spread_long_days_tolerance)).strftime('%Y-%m-%d')
            if not self.quiet:
                print(f"Fetching long-term options expiring around {spread_long_days} days (±{spread_long_days_tolerance} days)")
                print(f"  Date range: {long_start_date} to {long_end_date}")
            if self.debug:
                print(f"DEBUG: Calculated long-term date range: {long_start_date} to {long_end_date} (from today {today})", file=sys.stderr)
        
        if self.debug:
            print(f"DEBUG: Tickers to fetch: {tickers}")
            print(f"DEBUG: Short-term options count: {len(df_short)}")
            print(f"DEBUG: Short-term date range in df_short:")
            if not df_short.empty and 'expiration_date' in df_short.columns:
                print(f"  Min expiration: {df_short['expiration_date'].min()}")
                print(f"  Max expiration: {df_short['expiration_date'].max()}")
        
        # Fetch long-term options data
        # Use a much larger timestamp lookback for long-term options since they may have been
        # written weeks or months ago but are still valid. Use at least 180 days to catch options
        # that were written when they were first listed (which could be months before expiration)
        long_timestamp_lookback_days = max(timestamp_lookback_days, 180)  # At least 180 days for long-term options
        if self.debug or not self.quiet:
            print(f"DEBUG: Using timestamp_lookback_days={long_timestamp_lookback_days} for long-term options (vs {timestamp_lookback_days} for short-term)", file=sys.stderr)
        
        try:
            if self.debug:
                print(f"DEBUG: Calling get_latest_options_data_batch with:", file=sys.stderr)
                print(f"  tickers: {tickers}", file=sys.stderr)
                print(f"  start_datetime: {long_start_date}", file=sys.stderr)
                print(f"  end_datetime: {long_end_date}", file=sys.stderr)
                print(f"  timestamp_lookback_days: {long_timestamp_lookback_days}", file=sys.stderr)
            
            if max_workers > 1:
                long_options_df = await self.db.get_latest_options_data_batch_multiprocess(
                    tickers=tickers,
                    start_datetime=long_start_date,
                    end_datetime=long_end_date,
                    batch_size=batch_size,
                    max_workers=max_workers,
                    timestamp_lookback_days=long_timestamp_lookback_days
                )
            else:
                long_options_df = await self.db.get_latest_options_data_batch(
                    tickers=tickers,
                    start_datetime=long_start_date,
                    end_datetime=long_end_date,
                    max_concurrent=max_concurrent,
                    batch_size=batch_size,
                    timestamp_lookback_days=long_timestamp_lookback_days
                )
            
            if self.debug:
                print(f"DEBUG: Batch fetch returned {len(long_options_df)} rows", file=sys.stderr)
            
            if self.debug:
                print(f"DEBUG: Fetched {len(long_options_df)} total options from database")
                if not long_options_df.empty:
                    print(f"DEBUG: Long options columns: {list(long_options_df.columns)}")
                    if 'ticker' in long_options_df.columns:
                        unique_tickers = long_options_df['ticker'].unique().tolist()
                        print(f"DEBUG: Long options tickers: {unique_tickers}")
                        # Show count per ticker
                        for ticker in unique_tickers:
                            count = len(long_options_df[long_options_df['ticker'] == ticker])
                            print(f"DEBUG:   {ticker}: {count} options")
                    if 'option_type' in long_options_df.columns:
                        print(f"DEBUG: Option types: {long_options_df['option_type'].unique().tolist()}")
                else:
                    print(f"DEBUG: No long-term options found in database for date range {long_start_date} to {long_end_date}")
            
            # Apply write timestamp filter to long options if specified
            if min_write_timestamp and not long_options_df.empty:
                try:
                    import pytz
                    est = pytz.timezone('America/New_York')
                    min_ts = pd.to_datetime(min_write_timestamp)
                    if min_ts.tz is None:
                        min_ts = est.localize(min_ts)
                    min_ts_utc = min_ts.astimezone(pytz.UTC)
                    
                    if 'write_timestamp' in long_options_df.columns:
                        long_options_df['write_timestamp'] = pd.to_datetime(long_options_df['write_timestamp'])
                        if long_options_df['write_timestamp'].dt.tz is None:
                            long_options_df['write_timestamp'] = long_options_df['write_timestamp'].dt.tz_localize('UTC')
                        
                        before_count = len(long_options_df)
                        long_options_df = long_options_df[long_options_df['write_timestamp'] >= min_ts_utc]
                        after_count = len(long_options_df)
                        
                        if not self.quiet:
                            print(f"Filtered long options by write timestamp: {before_count} -> {after_count} options")
                        if self.debug:
                            print(f"DEBUG: Applied write timestamp filter >= {min_ts_utc} UTC")
                except Exception as e:
                    if not self.quiet:
                        print(f"Warning: Could not apply write timestamp filter to long options: {e}", file=sys.stderr)
            
            if long_options_df.empty:
                if not self.quiet:
                    print("Warning: No long-term options found for spread analysis.", file=sys.stderr)
                if self.debug:
                    print("DEBUG: This could mean:", file=sys.stderr)
                    print("  1. No options data in database for the specified date range", file=sys.stderr)
                    print("  2. Options exist but not in the target expiration window", file=sys.stderr)
                    print(f"  3. Try increasing --spread-long-days-tolerance (currently {spread_long_days_tolerance} days)", file=sys.stderr)
                    print(f"  4. Check database for tickers: {tickers}", file=sys.stderr)
                return pd.DataFrame()
            
            # Filter for call options only
            if 'option_type' in long_options_df.columns:
                long_options_df = long_options_df[long_options_df['option_type'] == 'call']
            
            if self.debug:
                print(f"DEBUG: After filtering for calls: {len(long_options_df)} call options")
            
            if long_options_df.empty:
                if not self.quiet:
                    print("Warning: No long-term call options found for spread analysis.", file=sys.stderr)
                return pd.DataFrame()
            
            # Calculate days to expiry for long options
            long_options_df['expiration_date'] = pd.to_datetime(long_options_df['expiration_date'])
            if long_options_df['expiration_date'].dt.tz is None:
                long_options_df['expiration_date'] = long_options_df['expiration_date'].dt.tz_localize('UTC')
            today_ts = pd.Timestamp.now(tz='UTC').normalize()
            long_options_df['days_to_expiry'] = ((long_options_df['expiration_date'] - today_ts).dt.total_seconds() / 86400).astype(int)
            
            if self.debug:
                print(f"DEBUG: Long options days to expiry range: {long_options_df['days_to_expiry'].min()} to {long_options_df['days_to_expiry'].max()}")
                print(f"DEBUG: Long options strike price range: {long_options_df['strike_price'].min()} to {long_options_df['strike_price'].max()}")
            
            # Match short options with long options
            spread_results = []
            
            # Reset index to avoid issues with duplicate indices that could cause infinite loops
            df_short = df_short.reset_index(drop=True)
            
            # Deduplicate by option_ticker to avoid processing the same option multiple times
            if 'option_ticker' in df_short.columns:
                before_dedup = len(df_short)
                df_short = df_short.drop_duplicates(subset=['option_ticker'], keep='first')
                if self.debug and len(df_short) < before_dedup:
                    print(f"DEBUG: Deduplicated df_short: {before_dedup} -> {len(df_short)} rows (removed {before_dedup - len(df_short)} duplicates)", file=sys.stderr)
            
            for idx, short_row in df_short.iterrows():
                ticker = short_row['ticker']
                short_strike = short_row['strike_price']
                
                # Filter long options for this ticker
                ticker_long_options = long_options_df[long_options_df['ticker'] == ticker].copy()
                
                if ticker_long_options.empty:
                    if self.debug:
                        print(f"DEBUG: No long options found for ticker {ticker}")
                    continue
                
                if self.debug:
                    print(f"DEBUG: Processing {ticker} - short strike: ${short_strike:.2f}, {len(ticker_long_options)} long options available")
                
                # Calculate strike tolerance range
                tolerance_multiplier = spread_strike_tolerance / 100.0
                strike_min = short_strike * (1 - tolerance_multiplier)
                strike_max = short_strike * (1 + tolerance_multiplier)
                
                if self.debug:
                    print(f"DEBUG:   Strike tolerance range: ${strike_min:.2f} to ${strike_max:.2f} ({spread_strike_tolerance}%)")
                
                # Find matching long options within strike tolerance
                matching_long = ticker_long_options[
                    (ticker_long_options['strike_price'] >= strike_min) &
                    (ticker_long_options['strike_price'] <= strike_max)
                ].copy()  # Explicit copy to avoid SettingWithCopyWarning
                
                if self.debug:
                    print(f"DEBUG:   Found {len(matching_long)} matching long options within strike tolerance")
                
                if matching_long.empty:
                    if self.debug:
                        if not ticker_long_options.empty:
                            available_strikes = ticker_long_options['strike_price'].unique()
                            print(f"DEBUG:   No matches. Available strikes for {ticker}: {sorted(available_strikes)[:10]}..." if len(available_strikes) > 10 else f"DEBUG:   No matches. Available strikes for {ticker}: {sorted(available_strikes)}")
                        else:
                            print(f"DEBUG:   No long options available for ticker {ticker} in the database")
                    continue
                
                # Pick the best matching long option (closest strike, then closest to target days)
                matching_long['strike_diff'] = abs(matching_long['strike_price'] - short_strike)
                matching_long['days_diff'] = abs(matching_long['days_to_expiry'] - spread_long_days)
                matching_long = matching_long.sort_values(['strike_diff', 'days_diff'])
                
                best_long = matching_long.iloc[0]
                
                # Calculate long option premium (use ask, fallback to bid, fallback to 0.01)
                long_premium = best_long.get('ask', best_long.get('bid', 0.01))
                if pd.isna(long_premium):
                    long_premium = best_long.get('bid', 0.01)
                if pd.isna(long_premium):
                    long_premium = 0.01
                long_premium = round(float(long_premium), 2)
                
                # SPREAD MODE: Calculate num_contracts based on long option premium (investment in long options)
                # This represents how many spreads you can afford with your position_size
                num_contracts = math.floor(position_size / (long_premium * 100)) if long_premium > 0 else 0
                
                # Calculate premiums based on spread position sizing
                short_premium = short_row['option_premium']
                premium_diff = round(long_premium - short_premium, 2)  # Long - Short
                short_premium_total = round(num_contracts * short_premium * 100, 2)
                long_premium_total = round(num_contracts * long_premium * 100, 2)
                net_premium = round(short_premium_total - long_premium_total, 2)
                
                # Daily premium calculations (until short expiration - exit point)
                short_days = short_row['days_to_expiry']
                short_daily_premium = round(short_premium_total / short_days, 2) if short_days > 0 else 0
                net_daily_premium = round(net_premium / short_days, 2) if short_days > 0 else 0
                
                # Create spread result row
                spread_row = short_row.to_dict()
                spread_row.update({
                    'num_contracts': num_contracts,  # Override with spread-based calculation
                    'long_strike_price': round(float(best_long['strike_price']), 2),
                    'long_option_premium': long_premium,
                    'long_expiration_date': best_long['expiration_date'],
                    'long_days_to_expiry': int(best_long['days_to_expiry']),
                    'long_option_ticker': best_long.get('option_ticker', ''),
                    'long_delta': round(float(best_long.get('delta', 0)), 2) if pd.notna(best_long.get('delta')) else None,
                    'long_theta': round(float(best_long.get('theta', 0)), 2) if pd.notna(best_long.get('theta')) else None,
                    'long_volume': int(best_long.get('volume', 0)) if pd.notna(best_long.get('volume')) else 0,
                    'premium_diff': premium_diff,
                    'short_premium_total': short_premium_total,
                    'short_daily_premium': short_daily_premium,
                    'long_premium_total': long_premium_total,
                    'net_premium': net_premium,
                    'net_daily_premium': net_daily_premium
                })
                
                spread_results.append(spread_row)
            
            if not spread_results:
                if not self.quiet:
                    print("Warning: No matching spread opportunities found within strike tolerance.", file=sys.stderr)
                if self.debug:
                    print(f"DEBUG: Summary - Processed {len(df_short)} short options, but none matched with long options", file=sys.stderr)
                    print("DEBUG: Possible reasons:", file=sys.stderr)
                    print("  1. Strike prices don't overlap between short and long options", file=sys.stderr)
                    print("  2. Strike tolerance is too strict (try increasing --spread-strike-tolerance)", file=sys.stderr)
                    print(f"  3. Long options not available in the {spread_long_days}±{spread_long_days_tolerance} day window", file=sys.stderr)
                    print(f"  4. Try increasing --spread-long-days-tolerance to widen the search window", file=sys.stderr)
                return pd.DataFrame()
            
            # Create spread DataFrame
            df_spread = pd.DataFrame(spread_results)
            
            if not self.quiet:
                print(f"✓ Found {len(df_spread)} spread opportunities (matched short and long options).")
            
            return df_spread
            
        except Exception as e:
            if not self.quiet:
                print(f"Error creating spread analysis: {e}", file=sys.stderr)
            import traceback
            traceback.print_exc()
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
        filter_logic: str = 'AND',
        csv_delimiter: str = ',',
        csv_quoting: str = 'minimal',
        csv_columns: Optional[List[str]] = None,
        top_n: Optional[int] = None
    ) -> str:
        """Format the analysis results for output."""
        if self.debug or not self.quiet:
            print(f"DEBUG: format_output called with {len(df)} rows", file=sys.stderr)
        if df.empty:
            if not self.quiet:
                print("DEBUG: DataFrame is empty in format_output", file=sys.stderr)
            return "No options data found matching the criteria."
        
        # Add financial information to the dataframe
        # DataFrame already has named columns from the new batch method
        if 'ticker' not in df.columns:
            return "Error: DataFrame missing 'ticker' column"
        
        df_renamed = df.copy()
        df_renamed['pe_ratio'] = df_renamed['ticker'].map(lambda x: financial_data.get(x, {}).get('pe_ratio'))
        df_renamed['market_cap'] = df_renamed['ticker'].map(lambda x: financial_data.get(x, {}).get('market_cap'))
        df_renamed['market_cap_b'] = df_renamed['market_cap'].apply(lambda x: round(x / 1e9, 2) if pd.notna(x) and x is not None else None)
        
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
        # Check if this is spread mode by looking for spread-specific columns
        is_spread_mode = 'net_premium' in df_renamed.columns
        
        if is_spread_mode:
            display_columns = [
                'ticker', 'pe_ratio', 'market_cap_b', 'current_price',
                # Short option details
                'strike_price', 'price_above_current', 'option_premium', 'option_premium_percentage',
                'delta', 'theta', 'expiration_date', 'days_to_expiry',
                'short_premium_total', 'short_daily_premium',
                # Long option details
                'long_strike_price', 'long_option_premium', 'long_delta', 'long_theta',
                'long_expiration_date', 'long_days_to_expiry', 'long_premium_total',
                # Premium comparison
                'premium_diff',
                # Net spread calculations
                'net_premium', 'net_daily_premium',
                # Additional details
                'volume', 'num_contracts', 'write_timestamp', 'option_ticker', 'long_option_ticker'
            ]
        else:
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
            before_filter_count = len(df_display)
            df_display = FilterParser.apply_filters(df_display, filters, filter_logic)
            if self.debug or not self.quiet:
                print(f"DEBUG: After filter application: {len(df_display)} rows (was {before_filter_count})", file=sys.stderr)
        
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
                # Sort premium_diff ascending (lower is better for spreads), others descending
                ascending = (sort_key == 'premium_diff')
                df_display = df_display.sort_values(by=sort_key, ascending=ascending)
        
        # Apply top-n limit per ticker if specified
        if top_n is not None and top_n > 0:
            before_topn = len(df_display)
            if 'ticker' in df_display.columns:
                # Keep top N per ticker (after sorting)
                df_display = df_display.groupby('ticker', group_keys=False).head(top_n)
                if not self.quiet:
                    print(f"Limiting to top {top_n} options per ticker (after sorting)")
            else:
                # If no ticker column, just take top N overall
                df_display = df_display.head(top_n)
            if self.debug or not self.quiet:
                print(f"DEBUG: After top-n filter ({top_n}): {len(df_display)} rows (was {before_topn})", file=sys.stderr)
        
        # Handle CSV column selection
        if csv_columns and output_format == 'csv':
            # Filter to only include specified columns
            available_csv_columns = [col for col in csv_columns if col in df_display.columns]
            if available_csv_columns:
                df_display = df_display[available_csv_columns]
        
        # Handle CSV formatting
        if output_format == 'csv':
            return self._format_csv_output(df_display, csv_delimiter, csv_quoting, group_by, output_file)
        
        if group_by == 'ticker':
            # Group by ticker and show results per ticker
            output_lines = []
            for ticker in sorted(df_display['ticker'].unique()):
                ticker_data = df_display[df_display['ticker'] == ticker]
                output_lines.append(f"\n--- {ticker} ---")
                
                if output_format == 'table':
                    # Format numeric columns for better display
                    ticker_data_formatted = ticker_data.copy()
                    for col in ['current_price', 'strike_price', 'price_above_current', 'option_premium', 'potential_premium', 'daily_premium', 
                                'long_strike_price', 'long_option_premium', 'premium_diff', 'short_premium_total', 'short_daily_premium', 'long_premium_total', 'net_premium', 'net_daily_premium']:
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
                    # CSV format - this should not be reached with new logic
                    output_lines.append(ticker_data.to_csv(index=False))
        else:
            # Overall ranking
            if output_format == 'table':
                # Format numeric columns for better display
                df_formatted = df_display.copy()
                for col in ['current_price', 'strike_price', 'price_above_current', 'option_premium', 'potential_premium', 'daily_premium',
                            'long_strike_price', 'long_option_premium', 'premium_diff', 'short_premium_total', 'short_daily_premium', 'long_premium_total', 'net_premium', 'net_daily_premium']:
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
                # CSV format - this should not be reached with new logic
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

  # Filter by volume and max days (options expiring within 30 days), save to file
  python options_analyzer.py --symbols AAPL MSFT --min-volume 1000 --max-days 30 --output results.csv

  # CSV with custom formatting
  python options_analyzer.py --symbols AAPL --output results.csv --csv-delimiter ";" --csv-quoting all

  # CSV with specific columns only
  python options_analyzer.py --symbols AAPL --output results.csv --csv-columns "ticker,current_price,strike_price,potential_premium,daily_premium"

  # Show only high-premium opportunities
  python options_analyzer.py --min-premium 5000 --sort potential_premium

  # Filter by expiration date range (show only options expiring in January 2024)
  python options_analyzer.py --start-date 2024-01-01 --end-date 2024-01-31
  
  # Show only options expiring within 30 days from today
  python options_analyzer.py --symbols AAPL --max-days 30
  
  # Show only options expiring today or later (default behavior)
  python options_analyzer.py --symbols AAPL
  
  # Show options expiring from a specific date onwards
  python options_analyzer.py --start-date 2024-02-15
  
  # Show all options including those already expired
  python options_analyzer.py --start-date 2020-01-01
  
  # max-days overrides end-date (shows options expiring within 60 days, not through 2024-12-31)
  python options_analyzer.py --symbols AAPL --end-date 2024-12-31 --max-days 60
  
  # Filter options by write timestamp (only show options written after specified time in EST)
  python options_analyzer.py --symbols AAPL --min-write-timestamp "2025-11-05 10:00:00"
  
  # Use multiprocessing with 8 workers (automatically enabled when max-workers > 1)
  python options_analyzer.py --symbols AAPL --max-workers 8

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

  # Calendar spread analysis (sell short-term, buy long-term)
  # Exact strike match, 90-day long options
  python options_analyzer.py --symbols AAPL --spread --max-days 30 --spread-long-days 90

  # Spread with 5% strike tolerance (allows ±5% difference in strikes)
  python options_analyzer.py --symbols AAPL MSFT --spread --spread-strike-tolerance 5.0 --spread-long-days 120

  # Spread with explicit min/max days range
  python options_analyzer.py --symbols AAPL --spread --spread-long-min-days 60 --spread-long-days 120

  # Spread with filters on net premium
  python options_analyzer.py --symbols AAPL --spread --filter "net_daily_premium > 100" --filter "net_premium > 1000"

  # Spread sorted by net daily premium
  python options_analyzer.py --symbols AAPL GOOGL --spread --sort net_daily_premium --max-days 14

  # Limit to top 5 options per ticker
  python options_analyzer.py --symbols AAPL MSFT GOOGL --top-n 5 --sort daily_premium

  # Spread mode with top 3 spreads per ticker
  python options_analyzer.py --symbols AAPL MSFT --spread --top-n 3 --sort net_daily_premium
"""
    )
    
    # Add symbol input arguments using common library
    add_symbol_arguments(parser, required=True)
    
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
        help="Maximum days from today for option expiration (convenience parameter that sets end-date to today + max-days, overrides --end-date if both are provided)."
    )
    parser.add_argument(
        '--start-date',
        type=str,
        default=None,
        help="Start date for option expiration filtering in YYYY-MM-DD format (defaults to today to show only options expiring today or later)."
    )
    parser.add_argument(
        '--end-date',
        type=str,
        default=None,
        help="End date for option expiration filtering in YYYY-MM-DD format (defaults to None for no upper bound, overridden by --max-days if both are provided)."
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
    
    parser.add_argument(
        '--no-market-time',
        action='store_true',
        help="Disable market-hours logic (gets latest stock price from any source regardless of market open/closed).",
    )
    
    parser.add_argument(
        '--no-cache',
        action='store_true',
        help="Disable Redis cache (all queries will go directly to the database). Useful for performance comparison.",
    )
    
    parser.add_argument(
        '--min-write-timestamp',
        type=str,
        default=None,
        help="Minimum write timestamp for options in EST format (e.g., '2025-11-05 10:00:00'). Filters out options written before this time. Useful for getting only fresh options data.",
    )
    
    # Spread analysis options
    parser.add_argument(
        '--spread',
        action='store_true',
        help="Enable calendar spread analysis mode (sell short-term calls, buy long-term calls at similar strikes).",
    )
    parser.add_argument(
        '--spread-strike-tolerance',
        type=float,
        default=0.0,
        help="Percentage tolerance for matching strike prices in spread mode (e.g., 5.0 for ±5%%). Default: 0.0 (exact match).",
    )
    parser.add_argument(
        '--spread-long-days',
        type=int,
        default=90,
        help="Target days to expiry for long-term options to buy in spread mode (default: 90).",
    )
    parser.add_argument(
        '--spread-long-days-tolerance',
        type=int,
        default=14,
        help="Days tolerance for long option expiration window in spread mode (default: 14, searches ±14 days around target). Ignored if --spread-long-min-days is specified.",
    )
    parser.add_argument(
        '--spread-long-min-days',
        type=int,
        default=None,
        help="Minimum days to expiry for long options in spread mode. If set, searches from this min to --spread-long-days (ignores tolerance). Example: --spread-long-min-days 60 --spread-long-days 120 searches 60-120 day window.",
    )
    
    # Performance tuning options
    parser.add_argument(
        '--timestamp-lookback-days',
        type=int,
        default=7,
        help="Number of days to look back for option timestamp data (default: 7). Lower values use less memory but may miss older data. Increase if you see missing options."
    )
    
    parser.add_argument(
        '--max-workers',
        type=int,
        default=4,
        help="Number of worker processes for multiprocessing (default: 4, typically set to CPU count). Multiprocessing is automatically enabled when max-workers > 1."
    )
    
    # Filter options
    parser.add_argument(
        '--filter',
        action='append',
        type=str,
        help="Filter expressions (can be used multiple times). Format: 'field operator value' or 'field operator field' or 'field exists/not_exists' or 'field*multiplier operator value'. "
             "Supported fields: pe_ratio market_cap volume num_contracts potential_premium daily_premium current_price strike_price price_above_current option_premium option_premium_percentage premium_above_diff_percentage days_to_expiry delta theta "
             "long_strike_price long_option_premium long_days_to_expiry long_delta long_theta premium_diff short_premium_total short_daily_premium long_premium_total net_premium net_daily_premium. "
             "Supported operators: > >= < <= == != exists not_exists. "
             "Mathematical operations: + - * / (e.g. 'num_contracts*0.1 > volume' 'potential_premium+1000 > daily_premium'). "
             "Derived fields: option_premium_percentage = (option_premium / current_price) * 100; "
             "premium_above_diff_percentage = ((option_premium - price_above_current) / price_above_current) * 100. "
             "Spread fields (when --spread is enabled): premium_diff = long_premium - short_premium; num_contracts = position_size / (long_premium * 100); "
             "short_premium_total = short_premium * num_contracts * 100; short_daily_premium = short_premium_total / short_days_to_expiry; "
             "long_premium_total = long_premium * num_contracts * 100; net_premium = short_premium_total - long_premium_total; net_daily_premium = net_premium / short_days_to_expiry. "
             "Market cap values support T (trillion) B (billion) and M (million) suffixes. "
             "Examples: 'pe_ratio > 20' or 'market_cap < 3.5T' or 'num_contracts > volume' or 'num_contracts*0.1 > volume' or 'potential_premium > daily_premium' or 'volume exists' or "
             "'option_premium_percentage >= 10' or 'premium_above_diff_percentage > 0' or 'net_daily_premium > 100'. "
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
            "net_daily_premium net_premium premium_diff short_premium_total short_daily_premium long_premium_total long_strike_price long_days_to_expiry "
            "or abbreviations like TKR PRC STRK PREM%% DIFF%% POT_PREM DAY_PREM NET_DAY NET_PREM PREM_DIFF S_PREM_TOT S_DAY_PREM L_PREM_TOT L_STRK L_DAYS."
        )
    )
    parser.add_argument(
        '--quiet',
        action='store_true',
        help="Suppress progress output."
    )
    parser.add_argument(
        '--debug',
        action='store_true',
        help="Enable debug output with detailed information about data fetching and matching."
    )
    parser.add_argument(
        '--top-n',
        type=int,
        default=None,
        help="Limit results to top N options per ticker (based on sort order). Example: --top-n 10 shows only the best 10 options for each ticker. Applied after sorting and filtering."
    )
    
    # CSV formatting options
    parser.add_argument(
        '--csv-delimiter',
        type=str,
        default=',',
        help="CSV delimiter character (default: ',')."
    )
    parser.add_argument(
        '--csv-quoting',
        choices=['minimal', 'all', 'none', 'nonnumeric'],
        default='minimal',
        help="CSV quoting style: minimal (default), all, none, nonnumeric."
    )
    parser.add_argument(
        '--csv-columns',
        type=str,
        help="Comma-separated list of columns to include in CSV output. If not specified, all columns are included."
    )
    
    parser.add_argument(
        '--stats',
        action='store_true',
        help="Display cache statistics at the end of the output (cache hit rate, etc.)."
    )
    
    # If help is requested, print and exit early to avoid running any analysis code
    if any(flag in sys.argv for flag in ("-h", "--help")):
        parser.print_help()
        sys.exit(0)

    args = parser.parse_args()
    
    # Log parsed arguments for debugging
    if args.debug or not args.quiet:
        print("DEBUG: Parsed arguments:", file=sys.stderr)
        print(f"  symbols: {getattr(args, 'symbols', None)}", file=sys.stderr)
        print(f"  types: {getattr(args, 'types', None)}", file=sys.stderr)
        print(f"  max_days: {args.max_days}", file=sys.stderr)
        print(f"  start_date: {args.start_date}", file=sys.stderr)
        print(f"  end_date: {args.end_date}", file=sys.stderr)
        print(f"  spread: {args.spread}", file=sys.stderr)
        print(f"  spread_strike_tolerance: {args.spread_strike_tolerance}", file=sys.stderr)
        print(f"  spread_long_days: {args.spread_long_days}", file=sys.stderr)
        print(f"  spread_long_min_days: {args.spread_long_min_days}", file=sys.stderr)
        print(f"  spread_long_days_tolerance: {args.spread_long_days_tolerance}", file=sys.stderr)
        print(f"  position_size: {args.position_size}", file=sys.stderr)
        print(f"  top_n: {args.top_n}", file=sys.stderr)
        print(f"  sort: {args.sort}", file=sys.stderr)
        print(f"  timestamp_lookback_days: {args.timestamp_lookback_days}", file=sys.stderr)
        print(f"  max_workers: {args.max_workers}", file=sys.stderr)
    
    # Initialize analyzer
    enable_cache = not args.no_cache
    analyzer = OptionsAnalyzer(args.db_conn, args.quiet, args.debug, enable_cache=enable_cache)
    await analyzer.initialize()
    
    # Get symbols list using common library
    symbols_list = await fetch_lists_data(args, args.quiet)
    if not symbols_list:
        print("No symbols specified or found. Exiting.", file=sys.stderr)
        sys.exit(1)
    
    if not args.quiet:
        print(f"Analyzing {len(symbols_list)} tickers...")
    
    if args.debug or not args.quiet:
        print(f"DEBUG: Symbols list: {symbols_list[:10]}{'...' if len(symbols_list) > 10 else ''}", file=sys.stderr)
    
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
        filter_logic=args.filter_logic,
        use_market_time=not args.no_market_time,
        start_date=args.start_date,
        end_date=args.end_date,
        timestamp_lookback_days=args.timestamp_lookback_days,
        max_workers=args.max_workers,
        spread_mode=args.spread,
        spread_strike_tolerance=args.spread_strike_tolerance,
        spread_long_days=args.spread_long_days,
        spread_long_days_tolerance=args.spread_long_days_tolerance,
        spread_long_min_days=args.spread_long_min_days,
        min_write_timestamp=args.min_write_timestamp
    )
    
    if df.empty:
        if not args.quiet:
            print("DEBUG: DataFrame is empty after analysis. Check debug output above for details.", file=sys.stderr)
        print("No options data found matching the criteria.")
        return
    
    # Print multiprocess statistics if using multiprocessing
    if args.max_workers > 1 and hasattr(analyzer.db, 'print_process_statistics'):
        analyzer.db.print_process_statistics(quiet=args.quiet)
    
    # Print cache statistics if requested
    if args.stats and hasattr(analyzer.db, 'get_cache_statistics'):
        cache_stats = analyzer.db.get_cache_statistics()
        if not args.quiet:
            print("\n=== Cache Statistics ===", file=sys.stderr)
            if cache_stats.get('enabled', False):
                print(f"Cache Status: ENABLED", file=sys.stderr)
                print(f"Total Requests: {cache_stats.get('total_requests', 0)}", file=sys.stderr)
                print(f"Cache Hits: {cache_stats.get('hits', 0)}", file=sys.stderr)
                print(f"Cache Misses: {cache_stats.get('misses', 0)}", file=sys.stderr)
                hit_rate = cache_stats.get('hit_rate', 0.0)
                print(f"Hit Rate: {hit_rate:.2%}", file=sys.stderr)
                print(f"Cache Sets: {cache_stats.get('sets', 0)}", file=sys.stderr)
                print(f"Cache Invalidations: {cache_stats.get('invalidations', 0)}", file=sys.stderr)
                print(f"Cache Errors: {cache_stats.get('errors', 0)}", file=sys.stderr)
            else:
                print(f"Cache Status: DISABLED", file=sys.stderr)
            # Database query statistics
            db_query_count = cache_stats.get('db_query_count', 0)
            print(f"\n=== Database Query Statistics ===", file=sys.stderr)
            print(f"Total Database Queries: {db_query_count}", file=sys.stderr)
            print("===================================\n", file=sys.stderr)
    
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
    
    # If in spread mode and user didn't specify a sort, default to net_daily_premium
    if args.spread and args.sort == 'daily_premium':  # daily_premium is the default
        sort_arg = 'net_daily_premium'

    # Parse CSV columns if specified
    csv_columns = None
    if hasattr(args, 'csv_columns') and args.csv_columns:
        csv_columns = [col.strip() for col in args.csv_columns.split(',')]

    # Format and display results
    result = analyzer.format_output(
        df=df,
        financial_data=financial_data,
        output_format=output_format,
        group_by=args.group_by,
        output_file=output_file,
        sort_by=sort_arg,
        filters=filters,
        filter_logic=args.filter_logic,
        csv_delimiter=args.csv_delimiter,
        csv_quoting=args.csv_quoting,
        csv_columns=csv_columns,
        top_n=args.top_n
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
