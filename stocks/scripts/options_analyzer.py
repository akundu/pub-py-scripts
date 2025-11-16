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
from concurrent.futures import ProcessPoolExecutor
import functools
import subprocess
import multiprocessing

# Ensure project root is on sys.path so `common` can be imported when running from any cwd
CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Import common functions
from common.common import (
    black_scholes_call,
    check_tickers_for_refresh as common_check_tickers_for_refresh,
    fetch_latest_option_timestamp_standalone as common_fetch_latest_option_timestamp_standalone,
    get_redis_client_for_refresh,
    check_redis_refresh_pending,
    set_redis_refresh_pending,
    clear_redis_refresh_pending,
    set_redis_last_write_timestamp,
    REDIS_AVAILABLE
)

# Import common symbol loading functions
from common.symbol_loader import add_symbol_arguments, fetch_lists_data
from common.stock_db import get_stock_db
from common.market_hours import is_market_hours as common_is_market_hours

# Import fetch_options for refresh functionality
try:
    from scripts.fetch_options import HistoricalDataFetcher
    POLYGON_AVAILABLE = True
except ImportError:
    # If fetch_options is not available, refresh feature will be disabled
    HistoricalDataFetcher = None
    POLYGON_AVAILABLE = False


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
        'implied_volatility': float,
        # Spread-related fields
        'long_strike_price': float,
        'long_option_premium': float,
        'long_days_to_expiry': int,
        'long_delta': float,
        'long_theta': float,
        'long_expiration_date': str,
        'long_option_ticker': str,
        'long_volume': int,
        'long_implied_volatility': float,
        'long_contracts_available': int,
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


# Module-level cache for latest option timestamps (per process)
# This ensures timestamps are only fetched once per ticker per process
_timestamp_cache_per_process: Dict[str, pd.Timestamp] = {}


# ============================================================================
# Common Utility Functions
# ============================================================================

def _extract_ticker_from_option_ticker(option_ticker: Any) -> Optional[str]:
    """Extract ticker symbol from option_ticker (e.g., 'AAPL250117C00150000' -> 'AAPL')."""
    if pd.isna(option_ticker):
        return None
    option_ticker_str = str(option_ticker)
    # Remove "O:" prefix if present
    if option_ticker_str.startswith("O:"):
        option_ticker_str = option_ticker_str[2:]
    # Extract ticker (first 1-5 uppercase letters before the date)
    match = re.match(r'^([A-Z]{1,5})', option_ticker_str)
    if match:
        return match.group(1)
    return None


def _calculate_option_premium(row: pd.Series) -> float:
    """Calculate option premium using mid-price (bid+ask)/2 if both available, else fallback to ask, bid, last price, or 0.01."""
    bid = row.get('bid')
    ask = row.get('ask')
    last_price = row.get('price')
    
    if pd.notna(bid) and pd.notna(ask):
        return round((float(bid) + float(ask)) / 2.0, 2)
    elif pd.notna(ask):
        return round(float(ask), 2)
    elif pd.notna(bid):
        return round(float(bid), 2)
    elif pd.notna(last_price):
        return round(float(last_price), 2)
    else:
        return 0.01


def _format_bid_ask(row: pd.Series) -> str:
    """Format bid and ask prices as 'bid:ask' string."""
    bid_val = row.get('bid', 0) if pd.notna(row.get('bid')) else 0
    ask_val = row.get('ask', 0) if pd.notna(row.get('ask')) else 0
    if pd.notna(row.get('bid')) or pd.notna(row.get('ask')):
        return f"{bid_val:.2f}:{ask_val:.2f}"
    return "N/A:N/A"


def _normalize_to_utc(x: Any) -> pd.Timestamp:
    """Normalize a datetime value to UTC timezone-aware Timestamp."""
    if pd.isna(x):
        return pd.NaT
    if isinstance(x, pd.Timestamp):
        dt = x
    else:
        dt = pd.to_datetime(x, errors='coerce', utc=True)
        if pd.isna(dt):
            return pd.NaT
    if dt.tz is None:
        dt = dt.tz_localize('UTC')
    else:
        dt = dt.tz_convert('UTC')
    return dt


def _normalize_timestamp_to_utc(x: Any) -> pd.Timestamp:
    """Normalize a timestamp value to UTC timezone-aware Timestamp."""
    if pd.isna(x):
        return pd.NaT
    if isinstance(x, pd.Timestamp):
        dt = x
    else:
        dt = pd.to_datetime(x, errors='coerce', utc=True)
        if pd.isna(dt):
            return pd.NaT
    if dt.tz is None:
        dt = dt.tz_localize('UTC')
    else:
        dt = dt.tz_convert('UTC')
    return dt


def _safe_days_calc(exp_date: Any, today_ref: pd.Timestamp) -> int:
    """Calculate days to expiry from expiration date, handling timezone-aware timestamps."""
    if pd.isna(exp_date):
        return 0
    if isinstance(exp_date, pd.Timestamp):
        if exp_date.tz is None:
            exp_date = exp_date.tz_localize('UTC')
        else:
            exp_date = exp_date.tz_convert('UTC')
    else:
        exp_date = pd.to_datetime(exp_date, utc=True)
    return int((exp_date - today_ref).total_seconds() / 86400)


def _format_price_with_change(current_price: float, prev_close: Optional[float]) -> Tuple[str, Optional[float]]:
    """Format price with change percentage. Returns (formatted_string, change_percentage)."""
    if pd.isna(current_price) or current_price is None:
        return None, None
    
    if prev_close is None or pd.isna(prev_close) or prev_close <= 0:
        return f"${current_price:.2f}", None
    
    change = current_price - prev_close
    change_pct = (change / prev_close) * 100
    
    if change >= 0:
        return f"+${change:.2f} (+{change_pct:.2f}%)", change_pct
    else:
        return f"-${abs(change):.2f} ({change_pct:.2f}%)", change_pct


def _normalize_write_timestamp(x: Any) -> pd.Timestamp:
    """Normalize write timestamp to UTC."""
    if pd.isna(x):
        return pd.NaT
    if isinstance(x, pd.Timestamp):
        dt = x
    else:
        dt = pd.to_datetime(x, errors='coerce')
        if pd.isna(dt):
            return pd.NaT
    if dt.tz is None:
        return dt.tz_localize('UTC')
    return dt.tz_convert('UTC')


def _import_filter_classes():
    """Import FilterParser and FilterExpression classes for use in worker processes."""
    try:
        from scripts.options_analyzer import FilterParser, FilterExpression
        return FilterParser, FilterExpression
    except ImportError:
        import sys
        if 'scripts.options_analyzer' in sys.modules:
            mod = sys.modules['scripts.options_analyzer']
            return mod.FilterParser, mod.FilterExpression
        else:
            import importlib
            mod = importlib.import_module('scripts.options_analyzer')
            return mod.FilterParser, mod.FilterExpression


def _setup_worker_imports():
    """Set up sys.path for worker processes."""
    from pathlib import Path
    CURRENT_DIR = Path(__file__).resolve().parent
    PROJECT_ROOT = CURRENT_DIR.parent
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))


# ============================================================================
# Options Processing Helper Functions
# ============================================================================

def _ensure_ticker_column(options_df: pd.DataFrame) -> pd.DataFrame:
    """Ensure ticker column exists in options DataFrame, extracting from option_ticker if needed."""
    if 'ticker' not in options_df.columns and 'option_ticker' in options_df.columns:
        options_df = options_df.copy()
        options_df['ticker'] = options_df['option_ticker'].apply(_extract_ticker_from_option_ticker)
    return options_df


def _calculate_option_metrics(
    df: pd.DataFrame,
    position_size: float,
    days_to_expiry: Optional[int] = None
) -> pd.DataFrame:
    """Calculate option metrics: premium, contracts, potential premium, daily premium."""
    df = df.copy()
    df['strike_price'] = df['strike_price'].round(2)
    df['price_above_current'] = (df['strike_price'] - df['current_price']).round(2)
    df['option_premium'] = df.apply(_calculate_option_premium, axis=1)
    df['bid_ask'] = df.apply(_format_bid_ask, axis=1)
    
    # Normalize expiration_date and calculate days_to_expiry
    df['expiration_date'] = df['expiration_date'].apply(_normalize_to_utc)
    today = pd.Timestamp.now(tz='UTC').normalize()
    df['days_to_expiry'] = df['expiration_date'].apply(lambda x: _safe_days_calc(x, today))
    
    # Normalize implied_volatility
    if 'implied_volatility' in df.columns:
        df['implied_volatility'] = pd.to_numeric(df['implied_volatility'], errors='coerce').round(4)
    else:
        df['implied_volatility'] = pd.Series([float('nan')] * len(df), index=df.index)
    
    # Apply days_to_expiry filter
    if days_to_expiry is not None:
        df = df[
            (df['days_to_expiry'] >= days_to_expiry - 1) &
            (df['days_to_expiry'] <= days_to_expiry + 1)
        ].copy()
    
    # Calculate num_contracts and premiums
    df['num_contracts'] = df.apply(
        lambda row: 0 if pd.isna(row.get('current_price')) or row.get('current_price') <= 0 
                  else math.floor(position_size / (row.get('current_price') * 100)),
        axis=1
    )
    df['potential_premium'] = (df['num_contracts'] * (df['option_premium'] * 100)).round(2)
    df['daily_premium'] = df.apply(
        lambda row: 0 if row['days_to_expiry'] <= 0 
                  else round(row['potential_premium'] / row['days_to_expiry'], 2),
        axis=1
    )
    
    return df


def _apply_basic_filters(
    df: pd.DataFrame,
    min_volume: int = 0,
    min_premium: float = 0.0,
    min_write_timestamp: Optional[str] = None
) -> pd.DataFrame:
    """Apply basic filters: volume, premium, write timestamp."""
    if min_volume > 0:
        df = df[df['volume'] >= min_volume].copy()
    
    if min_premium > 0.0:
        df = df[df['potential_premium'] >= min_premium].copy()
    
    if min_write_timestamp:
        try:
            import pytz
            est = pytz.timezone('America/New_York')
            min_ts = pd.to_datetime(min_write_timestamp)
            if min_ts.tz is None:
                min_ts = est.localize(min_ts)
            min_ts_utc = min_ts.astimezone(pytz.UTC)
            if 'write_timestamp' in df.columns:
                df['write_timestamp'] = df['write_timestamp'].apply(_normalize_write_timestamp)
                df = df[df['write_timestamp'] >= min_ts_utc].copy()
        except Exception:
            pass  # Ignore timestamp filter errors
    
    return df


async def _get_previous_close_for_date(db, ticker: str, reference_date: pd.Timestamp, debug: bool = False) -> Optional[float]:
    """
    Get the previous close price for a ticker based on a reference date.
    Returns the close price from the day before the reference date.
    
    Uses StockDataService.get() to fetch daily prices for a date range and picks the relevant price.
    
    Args:
        db: Database connection object (must have get_stock_data method)
        ticker: Ticker symbol
        reference_date: The date to use as reference (will find close price before this date)
        debug: Whether to print debug messages
        
    Returns:
        Previous close price (float) or None if not found
    """
    try:
        # Convert reference_date to date in EST timezone
        import pytz
        from datetime import datetime, timedelta
        
        if reference_date.tz is None:
            reference_date = reference_date.tz_localize('UTC')
        else:
            reference_date = reference_date.tz_convert('UTC')
        
        # Convert to EST for date comparison
        est = pytz.timezone('America/New_York')
        reference_date_est = reference_date.tz_convert(est)
        reference_date_only = reference_date_est.date()
        
        # Calculate date range: fetch last 10 days before reference date to ensure we get the previous close
        # This handles weekends/holidays where there might be gaps
        end_date = reference_date_only - timedelta(days=1)  # Day before reference date
        start_date = end_date - timedelta(days=10)  # Go back 10 days to ensure we get data
        
        # Use get_stock_data to fetch daily prices for the date range
        if hasattr(db, 'get_stock_data'):
            df = await db.get_stock_data(
                ticker=ticker,
                start_date=start_date.strftime('%Y-%m-%d'),
                end_date=end_date.strftime('%Y-%m-%d'),
                interval='daily'
            )
            
            if df is not None and not df.empty:
                # Get the most recent close price (last row, since data is ordered by date)
                # The date index should be the date column
                if isinstance(df.index, pd.DatetimeIndex):
                    # Filter to only dates before the reference date
                    df_filtered = df[df.index.date < reference_date_only]
                    if not df_filtered.empty:
                        # Get the last row (most recent date before reference)
                        latest_row = df_filtered.iloc[-1]
                        prev_close = float(latest_row['close'])
                        prev_date = df_filtered.index[-1].date() if hasattr(df_filtered.index[-1], 'date') else df_filtered.index[-1]
                        
                        if debug:
                            print(f"DEBUG: {ticker} - Found previous close ${prev_close:.2f} for date {prev_date} (before reference date {reference_date_only})", file=sys.stderr)
                        return prev_close
                else:
                    # If index is not DatetimeIndex, check if there's a 'date' column
                    if 'date' in df.columns:
                        df['date'] = pd.to_datetime(df['date'])
                        df_filtered = df[df['date'].dt.date < reference_date_only]
                        if not df_filtered.empty:
                            latest_row = df_filtered.iloc[-1]
                            prev_close = float(latest_row['close'])
                            prev_date = latest_row['date'].date() if hasattr(latest_row['date'], 'date') else latest_row['date']
                            
                            if debug:
                                print(f"DEBUG: {ticker} - Found previous close ${prev_close:.2f} for date {prev_date} (before reference date {reference_date_only})", file=sys.stderr)
                            return prev_close
        
        if debug:
            print(f"DEBUG: {ticker} - No previous close found before {reference_date_only}", file=sys.stderr)
        return None
    except Exception as e:
        if debug:
            print(f"DEBUG: {ticker} - Error getting previous close for date {reference_date}: {e}", file=sys.stderr)
        return None


async def _attach_price_data(
    options_df: pd.DataFrame,
    db,
    ticker: str,
    use_market_time: bool,
    redis_client: Optional[Any] = None,
    debug: bool = False
) -> pd.DataFrame:
    """Attach current price and price change data to options DataFrame."""
    price_data = await db.get_latest_price_with_data(ticker, use_market_time=use_market_time)
    if not price_data or not price_data.get('price'):
        return pd.DataFrame()
    
    current_price = price_data['price']
    price_timestamp = price_data.get('timestamp')
    price_source = price_data.get('source', 'unknown')
    options_df = options_df.copy()
    options_df['current_price'] = current_price
    
    # Get previous close price based on the current price's date
    # If market is open: compare realtime price to previous close
    # If market is closed: compare current close price to previous close (day before current price's date)
    prev_close = None
    if price_timestamp:
        # Use the current price's timestamp to get the previous close
        if isinstance(price_timestamp, pd.Timestamp):
            prev_close = await _get_previous_close_for_date(db, ticker, price_timestamp, debug=debug)
        else:
            # Convert to Timestamp if needed
            try:
                ts = pd.to_datetime(price_timestamp, utc=True)
                prev_close = await _get_previous_close_for_date(db, ticker, ts, debug=debug)
            except Exception as e:
                if debug:
                    print(f"DEBUG: {ticker} - Could not parse timestamp {price_timestamp}: {e}", file=sys.stderr)
    
    # Fallback to standard method if we couldn't get previous close based on date
    if prev_close is None:
        prev_close_prices = await db.get_previous_close_prices([ticker])
        prev_close = prev_close_prices.get(ticker) if prev_close_prices else None
    
    if debug:
        prev_close_str = f"{prev_close:.2f}" if prev_close is not None else "None"
        print(f"DEBUG: {ticker} - current_price=${current_price:.2f}, prev_close=${prev_close_str}, source={price_source}, timestamp={price_timestamp}", file=sys.stderr)
        if prev_close is None:
            print(f"DEBUG: {ticker} - WARNING: prev_close is None, cannot calculate price change", file=sys.stderr)
        elif prev_close <= 0:
            print(f"DEBUG: {ticker} - WARNING: prev_close={prev_close} is <= 0, cannot calculate price change", file=sys.stderr)
        elif abs(current_price - prev_close) < 0.01:
            print(f"DEBUG: {ticker} - WARNING: current_price ({current_price:.2f}) equals prev_close ({prev_close:.2f}), change will be 0", file=sys.stderr)
    
    # Format price with change
    price_with_change, price_change_pct = _format_price_with_change(current_price, prev_close)
    options_df['price_with_change'] = price_with_change
    options_df['price_change_pct'] = price_change_pct
    
    # Fetch latest option timestamp (with Redis caching)
    latest_opt_ts = await _fetch_latest_option_timestamp_standalone(db, ticker, redis_client=redis_client, debug=debug)
    options_df['latest_opt_ts'] = latest_opt_ts
    
    # Filter out rows without price
    options_df = options_df[options_df['current_price'].notna()].copy()
    
    return options_df


def _format_age_seconds(x: Any) -> Optional[str]:
    """Format age in seconds as a readable string."""
    if pd.isna(x) or x is None:
        return None
    try:
        # If it's already a Timestamp (shouldn't happen, but handle it), convert to age
        if isinstance(x, pd.Timestamp):
            from datetime import datetime, timezone
            now_utc = datetime.now(timezone.utc)
            if x.tz is None:
                x = x.tz_localize('UTC')
            else:
                x = x.tz_convert('UTC')
            x_dt = x.to_pydatetime()
            age_sec = (now_utc - x_dt).total_seconds()
        elif isinstance(x, str):
            # If it's a string, try to parse as float
            age_sec = float(x)
        else:
            # It should be a float (age in seconds)
            age_sec = float(x)
        
        if age_sec < 0:
            return None
        # Format as seconds with 1 decimal place
        return f"{age_sec:.1f}"
    except (ValueError, TypeError, AttributeError):
        # If conversion fails, return as string representation
        return str(x) if x is not None else None


def _normalize_expiration_date_for_display(x: Any) -> pd.Timestamp:
    """Normalize expiration date to UTC for display."""
    if pd.isna(x):
        return pd.NaT
    if isinstance(x, pd.Timestamp):
        dt = x
    else:
        dt = pd.to_datetime(x, errors='coerce')
        if pd.isna(dt):
            return pd.NaT
    if dt.tz is None:
        dt = dt.tz_localize('UTC')
    else:
        dt = dt.tz_convert('UTC')
    return dt


def _normalize_timestamp_for_display(x: Any) -> pd.Timestamp:
    """Normalize timestamp to America/New_York for display."""
    if pd.isna(x):
        return pd.NaT
    if isinstance(x, pd.Timestamp):
        dt = x
    else:
        dt = pd.to_datetime(x, errors='coerce')
        if pd.isna(dt):
            return pd.NaT
    if dt.tz is None:
        dt = dt.tz_localize('UTC')
    else:
        dt = dt.tz_convert('UTC')
    return dt.tz_convert('America/New_York')


def _format_dataframe_for_display(df: pd.DataFrame) -> pd.DataFrame:
    """Format DataFrame columns for table display (currency, percentages, etc.)."""
    df = df.copy()
    
    # Format currency columns
    currency_cols = ['current_price', 'strike_price', 'price_above_current', 'option_premium', 
                     'potential_premium', 'daily_premium', 'long_strike_price', 'long_option_premium', 
                     'premium_diff', 'short_premium_total', 'short_daily_premium', 'long_premium_total', 
                     'net_premium', 'net_daily_premium']
    for col in currency_cols:
        if col in df.columns:
            df[col] = df[col].apply(lambda x: f"${x:.2f}" if pd.notna(x) else "N/A")
    
    # Format numeric columns
    numeric_cols = ['pe_ratio', 'market_cap_b']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = df[col].apply(lambda x: f"{x:.2f}" if pd.notna(x) else "N/A")
    
    # Format percentage columns
    pct_cols = ['option_premium_percentage', 'premium_above_diff_percentage']
    for col in pct_cols:
        if col in df.columns:
            df[col] = df[col].apply(lambda x: f"{x:.2f}%" if pd.notna(x) else "N/A")
    
    # Format implied volatility columns (convert from decimal to percentage)
    iv_cols = ['implied_volatility', 'long_implied_volatility']
    for col in iv_cols:
        if col in df.columns:
            df[col] = df[col].apply(lambda x: f"{x * 100:.2f}%" if pd.notna(x) else "N/A")
    
    # Format integer columns
    if 'long_contracts_available' in df.columns:
        df['long_contracts_available'] = df['long_contracts_available'].apply(
            lambda x: f"{int(x)}" if pd.notna(x) else "N/A"
        )
    
    return df


def _normalize_and_select_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize timestamps and select output columns."""
    df = df.copy()
    
    # Round numeric columns
    for col in ['bid', 'ask', 'delta', 'theta']:
        if col in df.columns:
            df[col] = df[col].round(2)
    
    if 'implied_volatility' in df.columns:
        df['implied_volatility'] = df['implied_volatility'].round(4)
    
    # Normalize timestamp columns
    for ts_col in ['last_quote_timestamp', 'write_timestamp']:
        if ts_col in df.columns:
            df[ts_col] = df[ts_col].apply(_normalize_timestamp_to_utc)
    
    output_cols = [
        'ticker', 'current_price', 'price_with_change', 'price_change_pct', 'strike_price', 'price_above_current',
        'option_premium', 'bid_ask', 'implied_volatility', 'delta', 'theta', 'volume', 'num_contracts',
        'potential_premium', 'daily_premium', 'expiration_date', 'days_to_expiry',
        'last_quote_timestamp', 'option_ticker', 'latest_opt_ts'
    ]
    available_cols = [c for c in output_cols if c in df.columns]
    return df[available_cols].copy()


# Wrapper to use module-level cache
async def _fetch_latest_option_timestamp_standalone(
    db,
    ticker: str,
    cache: Optional[Dict[str, pd.Timestamp]] = None,
    redis_client: Optional[Any] = None,
    debug: bool = False
) -> Optional[float]:
    """
    Standalone function to fetch latest option write timestamp for a single ticker.
    Returns the age in seconds (difference between now and the timestamp).
    Can be used in multiprocessing workers or regular code paths.
    
    Args:
        db: Database connection object (must have execute_select_sql method)
        ticker: Ticker symbol to fetch timestamp for
        cache: Optional dictionary to use/update as cache (defaults to module-level cache)
        redis_client: Optional Redis client for timestamp caching
        debug: Whether to print debug messages
        
    Returns:
        Age in seconds since the latest write timestamp (float), or None if no timestamp found
    """
    # Use provided cache or module-level cache
    if cache is None:
        cache = _timestamp_cache_per_process
    
    return await common_fetch_latest_option_timestamp_standalone(db, ticker, cache, redis_client=redis_client, debug=debug)


# Alias for backward compatibility
_black_scholes_call_standalone = black_scholes_call


def _process_ticker_analysis(args_tuple):
    """
    Process a single ticker's options analysis end-to-end in a separate process.
    This includes: fetching options, filtering, price attachment, metric calculation, and filter application.
    
    Args:
        args_tuple: Tuple containing all parameters needed for analysis:
            - ticker: Ticker symbol
            - db_config: Database connection string
            - start_date: Start date for expiration filtering
            - end_date: End date for expiration filtering
            - timestamp_lookback_days: Days to look back for timestamp data
            - position_size: Position size in dollars
            - days_to_expiry: Optional days to expiry filter
            - min_volume: Minimum volume filter
            - min_premium: Minimum premium filter
            - min_write_timestamp: Optional minimum write timestamp
            - use_market_time: Whether to use market hours logic
            - filters: List of FilterExpression objects (must be picklable)
            - filter_logic: 'AND' or 'OR'
            - enable_cache: Whether caching is enabled
            - redis_url: Redis URL for caching
            - log_level: Logging level
            - debug: Whether debug output is enabled
    
    Returns:
        Tuple of (DataFrame with processed results, error_message or None)
    """
    import asyncio
    import sys
    import os
    import re
    import math
    import pandas as pd
    from pathlib import Path
    
    # Unpack arguments
    (ticker, db_config, start_date, end_date, timestamp_lookback_days,
     position_size, days_to_expiry, min_volume, min_premium, min_write_timestamp,
     use_market_time, filters, filter_logic, enable_cache, redis_url, log_level, debug) = args_tuple
    
    # Get Redis client for timestamp caching in worker process
    redis_client = None
    if enable_cache and redis_url and REDIS_AVAILABLE:
        redis_client = get_redis_client_for_refresh(redis_url)
    
    # Re-import needed modules in worker process
    _setup_worker_imports()
    from common.stock_db import get_stock_db
    FilterParser, FilterExpression = _import_filter_classes()
    
    async def _async_process():
        # Create database connection in worker process
        db = get_stock_db('questdb', db_config=db_config, enable_cache=enable_cache, 
                        redis_url=redis_url, log_level=log_level)
        
        try:
            # Use database as async context manager
            async with db:
                # Fetch options data for this ticker
                if debug:
                    print(f"DEBUG [PID {os.getpid()}]: Processing ticker {ticker}", file=sys.stderr)
                
                # Use single-process batch method with just one ticker
                options_df = await db.get_latest_options_data_batch(
                    tickers=[ticker],
                    start_datetime=start_date,
                    end_datetime=end_date,
                    max_concurrent=1,
                    batch_size=1,
                    timestamp_lookback_days=timestamp_lookback_days
                )
                
                if options_df.empty:
                    if debug:
                        print(f"DEBUG [PID {os.getpid()}]: No options data for {ticker}", file=sys.stderr)
                    return pd.DataFrame(), None
                
                # Ensure ticker column exists
                options_df = _ensure_ticker_column(options_df)
                
                # Filter for call options
                if 'option_type' in options_df.columns:
                    options_df = options_df[options_df['option_type'] == 'call'].copy()
                
                if options_df.empty:
                    return pd.DataFrame(), None
                
                # Attach price data
                options_df = await _attach_price_data(options_df, db, ticker, use_market_time, redis_client=redis_client, debug=debug)
                if options_df.empty:
                    if debug:
                        print(f"DEBUG [PID {os.getpid()}]: No price data for {ticker}", file=sys.stderr)
                    return pd.DataFrame(), None
                
                # Calculate metrics
                df = _calculate_option_metrics(options_df, position_size, days_to_expiry)
                if df.empty:
                    return pd.DataFrame(), None
                
                # Apply basic filters
                df = _apply_basic_filters(df, min_volume, min_premium, min_write_timestamp)
                
                # Apply custom filters if provided
                if filters:
                    df = FilterParser.apply_filters(df, filters, filter_logic)
                
                if df.empty:
                    return pd.DataFrame(), None
                
                # Normalize and select columns
                df = _normalize_and_select_columns(df)
                
                if debug:
                    print(f"DEBUG [PID {os.getpid()}]: {ticker} processed - {len(df)} options", file=sys.stderr)
                
                return df, None
                
        except Exception as e:
            error_msg = f"Error processing {ticker}: {str(e)}"
            if debug:
                import traceback
                print(f"DEBUG [PID {os.getpid()}]: {error_msg}", file=sys.stderr)
                traceback.print_exc(file=sys.stderr)
            return pd.DataFrame(), error_msg
    
    # Run async function in worker process
    return asyncio.run(_async_process())


def _process_ticker_spread_analysis(args_tuple):
    """
    Process a single ticker's spread analysis end-to-end in a separate process.
    This includes: fetching short-term options, processing them, fetching long-term options,
    matching them, and calculating spread metrics - all in one worker process.
    
    Args:
        args_tuple: Tuple containing all parameters needed for spread analysis:
            - ticker: Ticker symbol
            - db_config: Database connection string
            - start_date: Start date for short-term expiration filtering
            - end_date: End date for short-term expiration filtering
            - long_start_date: Start date for long-term expiration filtering
            - long_end_date: End date for long-term expiration filtering
            - timestamp_lookback_days: Days to look back for timestamp data
            - position_size: Position size in dollars
            - days_to_expiry: Optional days to expiry filter for short-term
            - min_volume: Minimum volume filter
            - min_premium: Minimum premium filter
            - min_write_timestamp: Optional minimum write timestamp
            - use_market_time: Whether to use market hours logic
            - filters: List of FilterExpression objects
            - filter_logic: 'AND' or 'OR'
            - spread_strike_tolerance: Percentage tolerance for strike matching
            - spread_long_days: Target days to expiry for long options
            - risk_free_rate: Risk-free rate for Black-Scholes
            - enable_cache: Whether caching is enabled
            - redis_url: Redis URL for caching
            - log_level: Logging level
            - debug: Whether debug output is enabled
    
    Returns:
        Tuple of (DataFrame with spread results, error_message or None)
    """
    import asyncio
    import sys
    import os
    import re
    import math
    import pandas as pd
    from pathlib import Path
    
    # Unpack arguments
    (ticker, db_config, start_date, end_date, long_start_date, long_end_date,
     timestamp_lookback_days, position_size, days_to_expiry, min_volume, min_premium,
     min_write_timestamp, use_market_time, filters, filter_logic, spread_strike_tolerance,
     spread_long_days, risk_free_rate, enable_cache, redis_url, log_level, debug) = args_tuple
    
    # Get Redis client for timestamp caching in worker process
    redis_client = None
    if enable_cache and redis_url and REDIS_AVAILABLE:
        redis_client = get_redis_client_for_refresh(redis_url)
    
    # Re-import needed modules in worker process
    _setup_worker_imports()
    from common.stock_db import get_stock_db
    FilterParser, FilterExpression = _import_filter_classes()
    
    async def _async_process():
        # Create database connection in worker process
        db = get_stock_db('questdb', db_config=db_config, enable_cache=enable_cache, 
                        redis_url=redis_url, log_level=log_level)
        
        try:
            # Use database as async context manager
            async with db:
                if debug:
                    print(f"DEBUG [PID {os.getpid()}]: Processing spread analysis for ticker {ticker}", file=sys.stderr)
                
                # ===== STEP 1: Fetch and process short-term options =====
                if debug:
                    print(f"DEBUG [PID {os.getpid()}]: {ticker} - Fetching short-term options (date range: {start_date} to {end_date})", file=sys.stderr)
                short_options_df = await db.get_latest_options_data_batch(
                    tickers=[ticker],
                    start_datetime=start_date,
                    end_datetime=end_date,
                    max_concurrent=1,
                    batch_size=1,
                    timestamp_lookback_days=timestamp_lookback_days
                )
                
                if debug:
                    print(f"DEBUG [PID {os.getpid()}]: {ticker} - Fetched {len(short_options_df)} short-term options from DB", file=sys.stderr)
                
                if short_options_df.empty:
                    if debug:
                        print(f"DEBUG [PID {os.getpid()}]: {ticker} - No short-term options found in DB", file=sys.stderr)
                    return pd.DataFrame(), None
                
                # Ensure ticker column exists
                short_options_df = _ensure_ticker_column(short_options_df)
                
                # Filter for call options
                before_call_filter = len(short_options_df)
                if 'option_type' in short_options_df.columns:
                    short_options_df = short_options_df[short_options_df['option_type'] == 'call'].copy()
                    if debug:
                        print(f"DEBUG [PID {os.getpid()}]: {ticker} - After call filter: {len(short_options_df)} options (was {before_call_filter})", file=sys.stderr)
                
                if short_options_df.empty:
                    if debug:
                        print(f"DEBUG [PID {os.getpid()}]: {ticker} - No call options after filtering", file=sys.stderr)
                    return pd.DataFrame(), None
                
                # Attach price data
                before_price_filter = len(short_options_df)
                short_options_df = await _attach_price_data(short_options_df, db, ticker, use_market_time, redis_client=redis_client, debug=debug)
                if debug and not short_options_df.empty:
                    current_price = short_options_df['current_price'].iloc[0] if 'current_price' in short_options_df.columns else None
                    print(f"DEBUG [PID {os.getpid()}]: {ticker} - After price filter: {len(short_options_df)} options (was {before_price_filter}), current_price=${current_price:.2f}" if current_price else f"DEBUG [PID {os.getpid()}]: {ticker} - After price filter: {len(short_options_df)} options (was {before_price_filter})", file=sys.stderr)
                
                if short_options_df.empty:
                    if debug:
                        print(f"DEBUG [PID {os.getpid()}]: {ticker} - No options after price filter", file=sys.stderr)
                    return pd.DataFrame(), None
                
                # Process short-term options
                if debug:
                    print(f"DEBUG [PID {os.getpid()}]: {ticker} - Processing {len(short_options_df)} short-term options", file=sys.stderr)
                df_short = _calculate_option_metrics(short_options_df, position_size, days_to_expiry)
                
                if days_to_expiry is not None and not df_short.empty:
                    before_days_filter = len(df_short)
                    if debug:
                        print(f"DEBUG [PID {os.getpid()}]: {ticker} - After days_to_expiry filter ({days_to_expiry}): {len(df_short)} options (was {before_days_filter})", file=sys.stderr)
                        if not df_short.empty:
                            print(f"DEBUG [PID {os.getpid()}]: {ticker} - Short-term days_to_expiry range: {df_short['days_to_expiry'].min()} to {df_short['days_to_expiry'].max()}", file=sys.stderr)
                
                if df_short.empty:
                    if debug:
                        print(f"DEBUG [PID {os.getpid()}]: {ticker} - No options after days_to_expiry filter", file=sys.stderr)
                    return pd.DataFrame(), None
                
                # Apply basic filters with debug logging
                before_volume_filter = len(df_short)
                df_short = _apply_basic_filters(df_short, min_volume, min_premium, min_write_timestamp)
                if debug and min_volume > 0 and before_volume_filter != len(df_short):
                    print(f"DEBUG [PID {os.getpid()}]: {ticker} - After min_volume filter ({min_volume}): {len(df_short)} options (was {before_volume_filter})", file=sys.stderr)
                if debug and min_premium > 0.0 and before_volume_filter != len(df_short):
                    print(f"DEBUG [PID {os.getpid()}]: {ticker} - After min_premium filter (${min_premium}): {len(df_short)} options", file=sys.stderr)
                
                # Split filters into short-term filters (apply before matching) and spread filters (apply after matching)
                # Spread-only columns: long_*, premium_diff, net_premium, net_daily_premium, short_premium_total, long_premium_total
                spread_only_columns = {'long_strike_price', 'long_option_premium', 'long_days_to_expiry', 'long_delta', 
                                      'long_theta', 'long_expiration_date', 'long_option_ticker', 'long_volume', 
                                      'long_implied_volatility', 'premium_diff', 'net_premium', 'net_daily_premium',
                                      'short_premium_total', 'long_premium_total', 'short_daily_premium'}
                
                short_term_filters = []
                spread_filters = []
                
                if filters:
                    import re
                    for f in filters:
                        # Extract field names from filter
                        field_expr = f.field_expression if f.field_expression else f.field
                        # Check if filter references any spread-only columns
                        field_names = re.findall(r'\b[a-zA-Z_][a-zA-Z0-9_]*\b', str(field_expr))
                        if f.is_field_comparison and f.value:
                            field_names.append(str(f.value))
                        
                        is_spread_filter = any(col in spread_only_columns for col in field_names)
                        if is_spread_filter:
                            spread_filters.append(f)
                        else:
                            short_term_filters.append(f)
                
                # Apply short-term filters before matching
                if short_term_filters:
                    before_custom_filters = len(df_short)
                    df_short = FilterParser.apply_filters(df_short, short_term_filters, filter_logic)
                    if debug:
                        print(f"DEBUG [PID {os.getpid()}]: {ticker} - After short-term filters ({len(short_term_filters)} filters, logic={filter_logic}): {len(df_short)} options (was {before_custom_filters})", file=sys.stderr)
                        if short_term_filters:
                            filter_strs = [str(f) for f in short_term_filters]
                            print(f"DEBUG [PID {os.getpid()}]: {ticker} - Short-term filters: {filter_strs}", file=sys.stderr)
                
                if spread_filters and debug:
                    filter_strs = [str(f) for f in spread_filters]
                    print(f"DEBUG [PID {os.getpid()}]: {ticker} - Spread filters (will apply after matching): {filter_strs}", file=sys.stderr)
                
                if df_short.empty:
                    if debug:
                        print(f"DEBUG [PID {os.getpid()}]: {ticker} - No short-term options remaining after all filters", file=sys.stderr)
                    return pd.DataFrame(), None
                
                if debug:
                    print(f"DEBUG [PID {os.getpid()}]: {ticker} - Short-term options summary: {len(df_short)} options", file=sys.stderr)
                    if not df_short.empty:
                        print(f"DEBUG [PID {os.getpid()}]: {ticker} - Short-term strike range: ${df_short['strike_price'].min():.2f} to ${df_short['strike_price'].max():.2f}", file=sys.stderr)
                        print(f"DEBUG [PID {os.getpid()}]: {ticker} - Short-term days_to_expiry range: {df_short['days_to_expiry'].min()} to {df_short['days_to_expiry'].max()}", file=sys.stderr)
                
                # ===== STEP 2: Fetch and prepare long-term options =====
                # Use larger timestamp lookback for long-term options
                long_timestamp_lookback = max(timestamp_lookback_days, 180)
                
                if debug:
                    print(f"DEBUG [PID {os.getpid()}]: {ticker} - Fetching long-term options (date range: {long_start_date} to {long_end_date})", file=sys.stderr)
                long_options_df = await db.get_latest_options_data_batch(
                    tickers=[ticker],
                    start_datetime=long_start_date,
                    end_datetime=long_end_date,
                    max_concurrent=1,
                    batch_size=1,
                    timestamp_lookback_days=long_timestamp_lookback
                )
                
                if debug:
                    print(f"DEBUG [PID {os.getpid()}]: {ticker} - Fetched {len(long_options_df)} long-term options from DB", file=sys.stderr)
                
                if long_options_df.empty:
                    if debug:
                        print(f"DEBUG [PID {os.getpid()}]: {ticker} - No long-term options found in DB", file=sys.stderr)
                    return pd.DataFrame(), None
                
                # Ensure ticker column exists
                long_options_df = _ensure_ticker_column(long_options_df)
                
                # Filter for call options
                before_long_call_filter = len(long_options_df)
                if 'option_type' in long_options_df.columns:
                    long_options_df = long_options_df[long_options_df['option_type'] == 'call'].copy()
                    if debug:
                        print(f"DEBUG [PID {os.getpid()}]: {ticker} - After long call filter: {len(long_options_df)} options (was {before_long_call_filter})", file=sys.stderr)
                
                if long_options_df.empty:
                    if debug:
                        print(f"DEBUG [PID {os.getpid()}]: {ticker} - No long-term call options after filtering", file=sys.stderr)
                    return pd.DataFrame(), None
                
                # Apply write timestamp filter
                if min_write_timestamp:
                    before_long_write_ts_filter = len(long_options_df)
                    long_options_df = _apply_basic_filters(long_options_df, 0, 0.0, min_write_timestamp)
                    if debug and before_long_write_ts_filter != len(long_options_df):
                        print(f"DEBUG [PID {os.getpid()}]: {ticker} - After long min_write_timestamp filter: {len(long_options_df)} options (was {before_long_write_ts_filter})", file=sys.stderr)
                
                # Normalize implied_volatility
                if 'implied_volatility' in long_options_df.columns:
                    long_options_df['implied_volatility'] = pd.to_numeric(long_options_df['implied_volatility'], errors='coerce').round(4)
                else:
                    long_options_df['implied_volatility'] = pd.Series([float('nan')] * len(long_options_df), index=long_options_df.index)
                
                # Calculate days to expiry for long options
                long_options_df['expiration_date'] = long_options_df['expiration_date'].apply(_normalize_to_utc)
                today = pd.Timestamp.now(tz='UTC').normalize()
                long_options_df['days_to_expiry'] = long_options_df['expiration_date'].apply(lambda x: _safe_days_calc(x, today))
                
                if debug:
                    print(f"DEBUG [PID {os.getpid()}]: {ticker} - Long-term options summary: {len(long_options_df)} options", file=sys.stderr)
                    if not long_options_df.empty:
                        print(f"DEBUG [PID {os.getpid()}]: {ticker} - Long-term strike range: ${long_options_df['strike_price'].min():.2f} to ${long_options_df['strike_price'].max():.2f}", file=sys.stderr)
                        print(f"DEBUG [PID {os.getpid()}]: {ticker} - Long-term days_to_expiry range: {long_options_df['days_to_expiry'].min()} to {long_options_df['days_to_expiry'].max()}", file=sys.stderr)
                
                if long_options_df.empty:
                    if debug:
                        print(f"DEBUG [PID {os.getpid()}]: {ticker} - No long-term options remaining after filters", file=sys.stderr)
                    return pd.DataFrame(), None
                
                # ===== STEP 3: Match short-term with long-term options =====
                if debug:
                    print(f"DEBUG [PID {os.getpid()}]: {ticker} - Starting spread matching: {len(df_short)} short options vs {len(long_options_df)} long options", file=sys.stderr)
                    print(f"DEBUG [PID {os.getpid()}]: {ticker} - Spread parameters: strike_tolerance={spread_strike_tolerance}%, long_days={spread_long_days}", file=sys.stderr)
                
                # Convert to dictionaries for matching
                short_rows_list = [row.to_dict() for _, row in df_short.iterrows()]
                long_options_dict = {ticker: [row.to_dict() for _, row in long_options_df.iterrows()]}
                
                if debug:
                    print(f"DEBUG [PID {os.getpid()}]: {ticker} - Converted to dicts: {len(short_rows_list)} short rows, {len(long_options_dict.get(ticker, []))} long rows", file=sys.stderr)
                
                # Process each short option match
                spread_results = []
                matches_checked = 0
                for short_row_dict in short_rows_list:
                    matches_checked += 1
                    result = _process_spread_match((
                        short_row_dict,
                        long_options_dict,
                        spread_strike_tolerance,
                        spread_long_days,
                        position_size,
                        risk_free_rate,
                        debug
                    ))
                    if result is not None:
                        spread_results.append(result)
                        if debug and len(spread_results) <= 5:  # Log first 5 matches
                            print(f"DEBUG [PID {os.getpid()}]: {ticker} - Found match #{len(spread_results)}: short_strike=${short_row_dict.get('strike_price', 'N/A')}, long_strike=${result.get('long_strike_price', 'N/A')}", file=sys.stderr)
                
                if debug:
                    print(f"DEBUG [PID {os.getpid()}]: {ticker} - Spread matching complete: checked {matches_checked} short options, found {len(spread_results)} matches", file=sys.stderr)
                
                if not spread_results:
                    if debug:
                        print(f"DEBUG [PID {os.getpid()}]: {ticker} - No spread matches found. Possible reasons:", file=sys.stderr)
                        print(f"DEBUG [PID {os.getpid()}]:   - Strike prices don't overlap (short range: ${df_short['strike_price'].min():.2f}-${df_short['strike_price'].max():.2f}, long range: ${long_options_df['strike_price'].min():.2f}-${long_options_df['strike_price'].max():.2f})", file=sys.stderr)
                        print(f"DEBUG [PID {os.getpid()}]:   - Strike tolerance ({spread_strike_tolerance}%) too strict", file=sys.stderr)
                        print(f"DEBUG [PID {os.getpid()}]:   - Long options not in target days range ({spread_long_days} days)", file=sys.stderr)
                    return pd.DataFrame(), None
                
                # Convert to DataFrame
                df_spread = pd.DataFrame(spread_results)
                
                if debug:
                    print(f"DEBUG [PID {os.getpid()}]: {ticker} - Created spread DataFrame with {len(df_spread)} matches", file=sys.stderr)
                
                # Apply spread filters after matching
                if not df_spread.empty and spread_filters:
                    before_spread_filters = len(df_spread)
                    df_spread = FilterParser.apply_filters(df_spread, spread_filters, filter_logic)
                    if debug:
                        print(f"DEBUG [PID {os.getpid()}]: {ticker} - After spread filters ({len(spread_filters)} filters, logic={filter_logic}): {len(df_spread)} matches (was {before_spread_filters})", file=sys.stderr)
                        if df_spread.empty:
                            print(f"DEBUG [PID {os.getpid()}]: {ticker} - All spread matches filtered out by spread filters", file=sys.stderr)
                
                if df_spread.empty:
                    if debug:
                        print(f"DEBUG [PID {os.getpid()}]: {ticker} - No spread results after all filters", file=sys.stderr)
                    return pd.DataFrame(), None
                
                if debug:
                    print(f"DEBUG [PID {os.getpid()}]: {ticker} processed - {len(df_spread)} spread matches", file=sys.stderr)
                
                return df_spread, None
                
        except Exception as e:
            error_msg = f"Error processing spread analysis for {ticker}: {str(e)}"
            if debug:
                import traceback
                print(f"DEBUG [PID {os.getpid()}]: {error_msg}", file=sys.stderr)
                traceback.print_exc(file=sys.stderr)
            return pd.DataFrame(), error_msg
    
    # Run async function in worker process
    return asyncio.run(_async_process())


def _process_spread_match(args_tuple):
    """
    Process a single spread match (standalone function for multiprocessing).
    
    Args:
        args_tuple: Tuple containing:
            - short_row_dict: Dictionary representation of short option row
            - long_options_dict: Dictionary of long options data (ticker -> list of option dicts)
            - spread_strike_tolerance: Percentage tolerance for strike matching
            - spread_long_days: Target days to expiry for long options
            - position_size: Position size in dollars
            - risk_free_rate: Annual risk-free rate
            - debug: Whether to print debug messages
    
    Returns:
        Dictionary with spread result row, or None if no match found
    """
    # Import common function in worker process
    from common.common import black_scholes_call
    
    short_row_dict, long_options_dict, spread_strike_tolerance, spread_long_days, position_size, risk_free_rate, debug = args_tuple
    
    ticker = short_row_dict['ticker']
    short_strike = short_row_dict['strike_price']
    
    # Get long options for this ticker
    ticker_long_options_list = long_options_dict.get(ticker, [])
    
    if not ticker_long_options_list:
        if debug:
            print(f"DEBUG: No long options found for ticker {ticker}")
        return None
    
    # Convert list of dicts back to DataFrame for easier filtering
    import pandas as pd
    ticker_long_options = pd.DataFrame(ticker_long_options_list)
    # Ensure expiration_date is properly typed as Timestamp if present
    if 'expiration_date' in ticker_long_options.columns:
        ticker_long_options['expiration_date'] = pd.to_datetime(ticker_long_options['expiration_date'], errors='coerce')
    
    if ticker_long_options.empty:
        if debug:
            print(f"DEBUG: No long options found for ticker {ticker}")
        return None
    
    if debug:
        print(f"DEBUG: Processing {ticker} - short strike: ${short_strike:.2f}, {len(ticker_long_options)} long options available")
    
    # Calculate strike tolerance range
    tolerance_multiplier = spread_strike_tolerance / 100.0
    strike_min = short_strike * (1 - tolerance_multiplier)
    strike_max = short_strike * (1 + tolerance_multiplier)
    
    if debug:
        print(f"DEBUG:   Strike tolerance range: ${strike_min:.2f} to ${strike_max:.2f} ({spread_strike_tolerance}%)")
    
    # Find matching long options within strike tolerance
    matching_long = ticker_long_options[
        (ticker_long_options['strike_price'] >= strike_min) &
        (ticker_long_options['strike_price'] <= strike_max)
    ].copy()
    
    if debug:
        print(f"DEBUG:   Found {len(matching_long)} matching long options within strike tolerance")
    
    if matching_long.empty:
        if debug:
            if not ticker_long_options.empty:
                available_strikes = ticker_long_options['strike_price'].unique()
                print(f"DEBUG:   No matches. Available strikes for {ticker}: {sorted(available_strikes)[:10]}..." if len(available_strikes) > 10 else f"DEBUG:   No matches. Available strikes for {ticker}: {sorted(available_strikes)}")
            else:
                print(f"DEBUG:   No long options available for ticker {ticker} in the database")
        return None
    
    # Pick the best matching long option (closest strike, then closest to target days)
    matching_long['strike_diff'] = abs(matching_long['strike_price'] - short_strike)
    matching_long['days_diff'] = abs(matching_long['days_to_expiry'] - spread_long_days)
    matching_long = matching_long.sort_values(['strike_diff', 'days_diff'])
    
    best_long = matching_long.iloc[0].to_dict()
    
    # Calculate long option premium using utility function
    long_premium = _calculate_option_premium(pd.Series(best_long))
    
    long_iv = best_long.get('implied_volatility')
    if pd.notna(long_iv):
        long_iv = float(long_iv)
    else:
        long_iv = None

    # SPREAD MODE: Calculate num_contracts based on long option premium (investment in long options)
    # This represents how many spreads you can afford with your position_size
    num_contracts = math.floor(position_size / (long_premium * 100)) if long_premium > 0 else 0
    
    # Calculate premiums based on spread position sizing
    short_premium = short_row_dict['option_premium']
    premium_diff = round(long_premium - short_premium, 2)  # Long - Short (per contract)
    short_premium_total = round(num_contracts * short_premium * 100, 2)  # Total received from selling short-term options
    long_premium_total = round(num_contracts * long_premium * 100, 2)  # Total paid for buying long-term options
    
    # Debug output for spread calculations
    if debug:
        ticker = short_row_dict.get('ticker', 'UNKNOWN')
        short_strike = short_row_dict.get('strike_price', 'N/A')
        short_days = short_row_dict.get('days_to_expiry', 0)
        long_days = int(best_long.get('days_to_expiry', 0))
        long_strike_val = best_long.get('strike_price')
        long_strike_str = f"{long_strike_val:.2f}" if pd.notna(long_strike_val) else "N/A"
        
        print(f"DEBUG: {ticker} SPREAD premium calculation:", file=sys.stderr)
        print(f"  position_size = ${position_size:,.2f}", file=sys.stderr)
        print(f"  SHORT option:", file=sys.stderr)
        print(f"    strike_price = ${short_strike:.2f}", file=sys.stderr)
        print(f"    days_to_expiry = {short_days}", file=sys.stderr)
        print(f"    option_premium = ${short_premium:.2f}", file=sys.stderr)
        print(f"  LONG option:", file=sys.stderr)
        print(f"    strike_price = ${long_strike_str}", file=sys.stderr)
        print(f"    days_to_expiry = {long_days}", file=sys.stderr)
        print(f"    option_premium = ${long_premium:.2f}", file=sys.stderr)
        print(f"  num_contracts = floor({position_size:,.2f} / (${long_premium:.2f} * 100)) = floor({position_size:,.2f} / ${long_premium * 100:.2f}) = {num_contracts}", file=sys.stderr)
        print(f"  premium_diff (per contract) = ${long_premium:.2f} - ${short_premium:.2f} = ${premium_diff:.2f}", file=sys.stderr)
        print(f"  short_premium_total = {num_contracts} * ${short_premium:.2f} * 100 = ${short_premium_total:,.2f}", file=sys.stderr)
        print(f"  long_premium_total = {num_contracts} * ${long_premium:.2f} * 100 = ${long_premium_total:,.2f}", file=sys.stderr)
    
    # Calculate Black-Scholes based long option value at short expiration
    # This estimates what the long option will be worth when the short expires
    current_price = short_row_dict.get('current_price')
    long_strike = float(best_long['strike_price'])
    short_days = short_row_dict['days_to_expiry']
    long_days = int(best_long['days_to_expiry'])
    
    # Time remaining for long option at short expiration (in years)
    time_to_long_expiry_at_short_expiry = (long_days - short_days) / 365.0
    
    # Get implied volatility from long option data, or use a default
    implied_vol = best_long.get('implied_volatility')
    if pd.isna(implied_vol) or implied_vol is None or implied_vol <= 0:
        # Default to 0.3 (30% annual volatility) if not available
        implied_vol = 0.3
    else:
        implied_vol = float(implied_vol)
        # Ensure it's in decimal form (not percentage)
        if implied_vol > 1.0:
            implied_vol = implied_vol / 100.0
    
    # Calculate Black-Scholes long option value at short expiration
    # Assume stock price stays the same (or use current price as estimate)
    # NOTE: This is an estimate - actual value will depend on stock movement and volatility changes
    estimated_stock_price_at_short_expiry = current_price if current_price else long_strike
    
    long_option_value_at_short_expiry = 0.0
    if time_to_long_expiry_at_short_expiry > 0 and current_price:
        try:
            long_option_value_at_short_expiry = black_scholes_call(
                S=estimated_stock_price_at_short_expiry,
                K=long_strike,
                T=time_to_long_expiry_at_short_expiry,
                r=risk_free_rate,
                sigma=implied_vol
            )
        except Exception as e:
            if debug:
                print(f"DEBUG: Black-Scholes calculation error: {e}")
            # Fallback: use current long premium as estimate
            long_option_value_at_short_expiry = long_premium
    
    # Net premium calculation for calendar spread:
    # Formula: net_premium = short_premium_total - (long_premium_total - long_premium_at_short_expiry_total)
    # Expanded: net_premium = short_premium_total - long_premium_total + long_premium_at_short_expiry_total
    #
    # This represents the net cash flow if you:
    # 1. Receive short_premium_total from selling short-term calls
    # 2. Pay long_premium_total to buy long-term calls (now)
    # 3. Could recover long_premium_at_short_expiry_total by selling long calls when short expires
    #
    # NOTE: net_premium can exceed short_premium_total if the long option appreciates significantly
    # (i.e., when long_premium_at_short_expiry_total > long_premium_total)
    # This happens when the long option gains value faster than it decays, which is possible
    # if volatility increases or the stock moves favorably, but the estimate assumes stock price stays flat.
    long_premium_at_short_expiry_total = round(num_contracts * long_option_value_at_short_expiry * 100, 2)
    net_premium = round(short_premium_total - (long_premium_total - long_premium_at_short_expiry_total), 2)
    
    # Debug output for net premium calculations
    if debug:
        print(f"  Black-Scholes estimate (long option value at short expiry):", file=sys.stderr)
        print(f"    estimated_stock_price_at_short_expiry = ${estimated_stock_price_at_short_expiry:.2f}", file=sys.stderr)
        print(f"    long_strike = ${long_strike:.2f}", file=sys.stderr)
        print(f"    time_to_long_expiry_at_short_expiry = {time_to_long_expiry_at_short_expiry:.4f} years ({long_days - short_days} days)", file=sys.stderr)
        print(f"    implied_volatility = {implied_vol:.4f}", file=sys.stderr)
        print(f"    long_option_value_at_short_expiry (per contract) = ${long_option_value_at_short_expiry:.2f}", file=sys.stderr)
        print(f"    long_premium_at_short_expiry_total = {num_contracts} * ${long_option_value_at_short_expiry:.2f} * 100 = ${long_premium_at_short_expiry_total:,.2f}", file=sys.stderr)
        print(f"  net_premium calculation:", file=sys.stderr)
        print(f"    net_premium = short_premium_total - (long_premium_total - long_premium_at_short_expiry_total)", file=sys.stderr)
        print(f"    net_premium = ${short_premium_total:,.2f} - (${long_premium_total:,.2f} - ${long_premium_at_short_expiry_total:,.2f})", file=sys.stderr)
        print(f"    net_premium = ${short_premium_total:,.2f} - ${long_premium_total - long_premium_at_short_expiry_total:,.2f} = ${net_premium:,.2f}", file=sys.stderr)
    
    # Daily premium calculations (until short expiration - exit point)
    short_daily_premium = round(short_premium_total / short_days, 2) if short_days > 0 else 0
    # Net daily premium = net premium / days (amortized over short days)
    net_daily_premium = round(net_premium / short_days, 2) if short_days > 0 else 0
    
    if debug:
        print(f"  daily_premium calculations:", file=sys.stderr)
        if short_days > 0:
            print(f"    short_daily_premium = ${short_premium_total:,.2f} / {short_days} days = ${short_daily_premium:.2f}", file=sys.stderr)
            print(f"    net_daily_premium = ${net_premium:,.2f} / {short_days} days = ${net_daily_premium:.2f}", file=sys.stderr)
        else:
            print(f"    short_daily_premium = $0.00 (short_days <= 0)", file=sys.stderr)
            print(f"    net_daily_premium = $0.00 (short_days <= 0)", file=sys.stderr)
        print("", file=sys.stderr)  # Empty line for readability

    long_contracts_available = best_long.get('open_interest')
    if pd.notna(long_contracts_available):
        try:
            long_contracts_available = int(float(long_contracts_available))
        except (TypeError, ValueError):
            long_contracts_available = None
    else:
        long_contracts_available = None
    
    # Format long bid:ask using utility function
    long_bid_ask = _format_bid_ask(pd.Series(best_long))
    
    # Create spread result row
    spread_row = short_row_dict.copy()
    # Ensure long_expiration_date is properly normalized using utility function
    long_exp_dt = _normalize_to_utc(best_long.get('expiration_date'))
    
    spread_row.update({
        'num_contracts': num_contracts,  # Override with spread-based calculation
        'long_strike_price': round(float(best_long['strike_price']), 2),
        'long_option_premium': long_premium,
        'long_bid_ask': long_bid_ask,
        'long_expiration_date': long_exp_dt,
        'long_days_to_expiry': int(best_long['days_to_expiry']),
        'long_option_ticker': best_long.get('option_ticker', ''),
        'long_delta': round(float(best_long.get('delta', 0)), 2) if pd.notna(best_long.get('delta')) else None,
        'long_theta': round(float(best_long.get('theta', 0)), 2) if pd.notna(best_long.get('theta')) else None,
        'long_volume': int(best_long.get('volume', 0)) if pd.notna(best_long.get('volume')) else 0,
        'long_implied_volatility': long_iv,
        'long_contracts_available': long_contracts_available,
        'premium_diff': premium_diff,
        'short_premium_total': short_premium_total,
        'short_daily_premium': short_daily_premium,
        'long_premium_total': long_premium_total,
        'net_premium': net_premium,
        'net_daily_premium': net_daily_premium
    })
    
    return spread_row


class OptionsAnalyzer:
    """Analyzes covered call opportunities across all strike prices and tickers."""
    
    def __init__(self, db_conn: str, log_level: str = "INFO", debug: bool = False, enable_cache: bool = True, redis_url: str | None = None, risk_free_rate: float = 0.05):
        """Initialize the options analyzer with database connection."""
        self.db_conn = db_conn
        self.log_level = log_level.upper()  # Normalize to uppercase
        self.debug = debug or (self.log_level == "DEBUG")
        self.enable_cache = enable_cache
        self.redis_url = redis_url
        self.db = None
        self.risk_free_rate = risk_free_rate  # Annual risk-free rate (default 5%)
        self._timestamp_cache: Dict[str, pd.Timestamp] = {}  # Cache for latest option timestamps per ticker
        self._redis_client = None  # Redis client for timestamp caching
    
    def _should_log(self, level: str) -> bool:
        """Check if a log level should be printed based on current log_level setting."""
        levels = {"DEBUG": 0, "INFO": 1, "WARNING": 2, "ERROR": 3}
        current_level = levels.get(self.log_level, 1)
        message_level = levels.get(level.upper(), 1)
        return message_level >= current_level
    
    def _log(self, level: str, message: str, file=sys.stderr):
        """Log a message if the level is appropriate."""
        if self._should_log(level):
            print(message, file=file)
    
    async def _fetch_latest_option_timestamps(
        self,
        tickers: List[str],
        cache: Optional[Dict[str, pd.Timestamp]] = None
    ) -> Dict[str, Optional[float]]:
        """
        Fetch latest option write timestamps for multiple tickers.
        Returns ages in seconds (difference between now and the timestamp).
        
        Args:
            tickers: List of ticker symbols to fetch timestamps for
            cache: Optional dictionary to use/update as cache (avoids duplicate fetches)
            
        Returns:
            Dictionary mapping ticker -> age in seconds (float) or None if no timestamp found
        """
        # Use instance cache if no cache provided
        if cache is None:
            cache = self._timestamp_cache
        
        # Fetch ages for all tickers using the standalone function (it handles caching internally)
        result: Dict[str, Optional[float]] = {}
        for ticker in tickers:
            age_seconds = await _fetch_latest_option_timestamp_standalone(
                self.db, ticker, cache=cache, redis_client=self._redis_client, debug=self.debug
            )
            result[ticker] = age_seconds
        
        return result
    
    @staticmethod
    def _extract_ticker_from_option_ticker(option_ticker):
        """Extract ticker symbol from option_ticker (e.g., 'AAPL250117C00150000' -> 'AAPL')."""
        return _extract_ticker_from_option_ticker(option_ticker)
    
    def _black_scholes_call(self, S: float, K: float, T: float, r: float, sigma: float) -> float:
        """Calculate Black-Scholes call option price (delegates to common function)."""
        return black_scholes_call(S, K, T, r, sigma)
    
    def _create_compact_headers(self, df: pd.DataFrame) -> Dict[str, str]:
        """Create compact headers that are at most 4 characters longer than the data width."""
        compact_headers = {}

        'current_price', 'strike_price', 'volume', 'pe_ratio', 'market_cap_b'
        
        # Define mapping for common columns to shorter names
        header_mapping = {
            # 'ticker': 'ticker',
            'current_price': 'curr_price',
            # 'pe_ratio': 'pe_ratio',
            # 'market_cap': 'market_cap',
            # 'market_cap_b': 'market_cap_b',
            # 'strike_price': 'strike_price',
            'price_above_current': 'price_above_curr',
            'option_premium': 'opt_prem.',
            'bid_ask': 'bid:ask',
            'option_premium_percentage': 'opt_prem.%',
            # 'premium_above_diff_percentage': 'DIFF%',
            'implied_volatility': 'iv',
            # 'delta': 'delta',
            # 'theta': 'theta',
            # 'volume': 'volume',
            # 'num_contracts': 'CNT',
            # 'potential_premium': 'POT_PREM',
            # 'daily_premium': 'DAILY_PREM',
            # 'expiration_date': 'EXP (UTC)',
            # 'days_to_expiry': 'DAYS',
            # 'last_quote_timestamp': 'LQUOTE_TS',
            # 'write_timestamp': 'WRITE_TS (EST)',
            # 'option_ticker': 'OPT_TKR',
            # # Spread-related columns
            'long_strike_price': 'l_strike',
            'long_option_premium': 'l_prem',
            'long_bid_ask': 'l_bid:ask',
            'long_expiration_date': 'l_expiration_date',
            'long_days_to_expiry': 'l_days_to_expiry',
            'long_option_ticker': 'l_option_ticker',
            'long_delta': 'l_delta',
            'long_theta': 'l_theta',
            'long_implied_volatility': 'liv',
            'long_volume': 'l_volume',
            'long_contracts_available': 'l_cnt_avl',
            'premium_diff': 'prem_diff',
            'short_premium_total': 's_prem_tot',
            'short_daily_premium': 's_day_prem',
            'long_premium_total': 'l_prem_tot',
            # 'net_premium': 'net_premium',
            # 'net_daily_premium': 'net_daily_premium'
        }
        
        for col in df.columns:
            if col in header_mapping:
                compact_headers[col] = header_mapping[col]
            else:
                # For unknown columns, use the original name but truncate if too long
                compact_headers[col] = col[:15] if len(col) > 8 else col
        
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
            self._log("INFO", f"CSV results saved to {output_file}")
        
        return csv_content
        
    async def initialize(self):
        """Initialize database connection."""
        try:
            self.db = get_stock_db('questdb', db_config=self.db_conn, enable_cache=self.enable_cache, redis_url=self.redis_url, log_level=self.log_level)
            cache_status = "enabled" if self.enable_cache else "disabled"
            self._log("INFO", f"Database connection established successfully (cache: {cache_status}).")
            
            # Initialize Redis client for timestamp caching
            if self.enable_cache and self.redis_url and REDIS_AVAILABLE:
                self._redis_client = get_redis_client_for_refresh(self.redis_url)
        except Exception as e:
            print(f"Error connecting to database: {e}", file=sys.stderr)
            sys.exit(1)
    
    async def get_financial_info(self, tickers: List[str]) -> Dict[str, Dict[str, Any]]:
        """Get financial information (P/E, market_cap) for the given tickers."""
        financial_data = {}
        
        if self.debug:
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
                self._log("WARNING", f"Could not fetch financial info for {ticker}: {e}")
                financial_data[ticker] = {
                    'pe_ratio': None,
                    'market_cap': None,
                    'price': None
                }
        
        return financial_data
    
    async def _analyze_options_multiprocess(
        self,
        tickers_upper: List[str],
        start_date: str,
        end_date: Optional[str],
        timestamp_lookback_days: int,
        position_size: float,
        days_to_expiry: Optional[int],
        min_volume: int,
        min_premium: float,
        min_write_timestamp: Optional[str],
        use_market_time: bool,
        filters: Optional[List[FilterExpression]],
        filter_logic: str,
        max_workers: int
    ) -> pd.DataFrame:
        """
        Analyze options using multiprocessing where each ticker is processed end-to-end in a separate process.
        All processing (fetching, filtering, metrics, filters) happens in worker processes.
        Main process only aggregates, sorts, and presents results.
        """
        from concurrent.futures import ProcessPoolExecutor
        
        self._log("INFO", f"Using multiprocessing mode: processing {len(tickers_upper)} tickers with {max_workers} workers")
        self._log("INFO", "Each ticker will be processed end-to-end in a separate process (fetch, filter, metrics, filters)")
        
        # Prepare arguments for each ticker
        process_args = []
        for ticker in tickers_upper:
            args = (
                ticker,
                self.db_conn,
                start_date,
                end_date,
                timestamp_lookback_days,
                position_size,
                days_to_expiry,
                min_volume,
                min_premium,
                min_write_timestamp,
                use_market_time,
                filters,  # FilterExpression objects should be picklable
                filter_logic,
                self.enable_cache,
                self.redis_url,
                self.log_level,
                self.debug
            )
            process_args.append(args)
        
        # Execute in parallel using ProcessPoolExecutor
        loop = asyncio.get_event_loop()
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = [
                loop.run_in_executor(executor, _process_ticker_analysis, args)
                for args in process_args
            ]
            results = await asyncio.gather(*futures, return_exceptions=True)
        
        # Collect results and handle errors
        dfs = []
        errors = []
        for i, result in enumerate(results):
            ticker = tickers_upper[i]
            if isinstance(result, Exception):
                error_msg = f"Error processing {ticker}: {result}"
                errors.append(error_msg)
                self._log("ERROR", error_msg)
                if self.debug:
                    import traceback
                    traceback.print_exc()
            elif isinstance(result, tuple) and len(result) == 2:
                df, error = result
                if error:
                    errors.append(error)
                    self._log("WARNING", error)
                if not df.empty:
                    dfs.append(df)
            else:
                self._log("WARNING", f"Unexpected result type for {ticker}: {type(result)}")
        
        if errors and self.debug:
            self._log("INFO", f"Encountered {len(errors)} error(s) during processing")
        
        # Concatenate all results
        if not dfs:
            self._log("INFO", "No results from any ticker")
            return pd.DataFrame()
        
        combined_df = pd.concat(dfs, ignore_index=True)
        self._log("INFO", f"Combined results from {len(dfs)} tickers: {len(combined_df)} total options")
        
        return combined_df
    
    async def _analyze_spread_multiprocess(
        self,
        tickers_upper: List[str],
        start_date: str,
        end_date: Optional[str],
        long_start_date: str,
        long_end_date: str,
        timestamp_lookback_days: int,
        position_size: float,
        days_to_expiry: Optional[int],
        min_volume: int,
        min_premium: float,
        min_write_timestamp: Optional[str],
        use_market_time: bool,
        filters: Optional[List[FilterExpression]],
        filter_logic: str,
        spread_strike_tolerance: float,
        spread_long_days: int,
        max_workers: int
    ) -> pd.DataFrame:
        """
        Analyze spread options using multiprocessing where each ticker is processed end-to-end in a separate process.
        All processing (fetching short-term, processing short-term, fetching long-term, matching, spread calculations)
        happens in worker processes. Main process only aggregates results.
        """
        from concurrent.futures import ProcessPoolExecutor
        
        self._log("INFO", f"Using multiprocessing mode for spread analysis: processing {len(tickers_upper)} tickers with {max_workers} workers")
        self._log("INFO", "Each ticker will be processed end-to-end in a separate process (short-term fetch/process, long-term fetch, matching, spread calculations)")
        
        # Prepare arguments for each ticker
        process_args = []
        for ticker in tickers_upper:
            args = (
                ticker,
                self.db_conn,
                start_date,
                end_date,
                long_start_date,
                long_end_date,
                timestamp_lookback_days,
                position_size,
                days_to_expiry,
                min_volume,
                min_premium,
                min_write_timestamp,
                use_market_time,
                filters,
                filter_logic,
                spread_strike_tolerance,
                spread_long_days,
                self.risk_free_rate,
                self.enable_cache,
                self.redis_url,
                self.log_level,
                self.debug
            )
            process_args.append(args)
        
        # Execute in parallel using ProcessPoolExecutor
        loop = asyncio.get_event_loop()
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = [
                loop.run_in_executor(executor, _process_ticker_spread_analysis, args)
                for args in process_args
            ]
            results = await asyncio.gather(*futures, return_exceptions=True)
        
        # Collect results and handle errors
        dfs = []
        errors = []
        for i, result in enumerate(results):
            ticker = tickers_upper[i]
            if isinstance(result, Exception):
                error_msg = f"Error processing spread analysis for {ticker}: {result}"
                errors.append(error_msg)
                self._log("ERROR", error_msg)
                if self.debug:
                    import traceback
                    traceback.print_exc()
            elif isinstance(result, tuple) and len(result) == 2:
                df, error = result
                if error:
                    errors.append(error)
                    self._log("WARNING", error)
                if not df.empty:
                    dfs.append(df)
            else:
                self._log("WARNING", f"Unexpected result type for {ticker}: {type(result)}")
        
        if errors and self.debug:
            self._log("INFO", f"Encountered {len(errors)} error(s) during spread processing")
        
        # Concatenate all results
        if not dfs:
            self._log("INFO", "No spread results from any ticker")
            return pd.DataFrame()
        
        combined_df = pd.concat(dfs, ignore_index=True)
        self._log("INFO", f"Combined spread results from {len(dfs)} tickers: {len(combined_df)} total spread opportunities")
        
        return combined_df
    
    async def analyze_options(
        self,
        tickers: List[str],
        days_to_expiry: Optional[int] = None,
        min_volume: int = 0,
        max_days: Optional[int] = None,
        min_premium: float = 0.0,
        position_size: float = 100000.0,
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
            if self.debug:
                print(f"DEBUG: Using max_days={max_days}: filtering options expiring through {end_date}", file=sys.stderr)
        
        # Use memory-efficient batch fetching instead of single large query
        try:
            # If using multiprocessing, process each ticker in a separate process
            if max_workers > 1:
                if spread_mode:
                    # Calculate long-term date range for spread mode
                    long_start_date, long_end_date = self._calculate_long_options_date_range(
                        spread_long_days, spread_long_days_tolerance, spread_long_min_days
                    )
                    return await self._analyze_spread_multiprocess(
                        tickers_upper=tickers_upper,
                        start_date=start_date,
                        end_date=end_date,
                        long_start_date=long_start_date,
                        long_end_date=long_end_date,
                        timestamp_lookback_days=timestamp_lookback_days,
                        position_size=position_size,
                        days_to_expiry=days_to_expiry,
                        min_volume=min_volume,
                        min_premium=min_premium,
                        min_write_timestamp=min_write_timestamp,
                        use_market_time=use_market_time,
                        filters=filters,
                        filter_logic=filter_logic,
                        spread_strike_tolerance=spread_strike_tolerance,
                        spread_long_days=spread_long_days,
                        max_workers=max_workers
                    )
                else:
                    return await self._analyze_options_multiprocess(
                        tickers_upper=tickers_upper,
                        start_date=start_date,
                        end_date=end_date,
                        timestamp_lookback_days=timestamp_lookback_days,
                        position_size=position_size,
                        days_to_expiry=days_to_expiry,
                        min_volume=min_volume,
                        min_premium=min_premium,
                        min_write_timestamp=min_write_timestamp,
                        use_market_time=use_market_time,
                        filters=filters,
                        filter_logic=filter_logic,
                        max_workers=max_workers
                    )
            
            # 1) Fetch options universe (combined short and long if spread mode, otherwise just short)
            long_options_df = None
            if spread_mode:
                # Compute long-term date window upfront
                long_start_date, long_end_date = self._calculate_long_options_date_range(
                    spread_long_days, spread_long_days_tolerance, spread_long_min_days
                )
                
                # Calculate combined date range for single fetch
                combined_start_date, combined_end_date = self._calculate_combined_date_range(
                    start_date, end_date, long_start_date, long_end_date
                )
                
                # Use larger timestamp lookback for combined fetch (long-term options may be older)
                combined_timestamp_lookback = max(timestamp_lookback_days, 180)
                
                self._log("INFO", f"Fetching options for {len(tickers_upper)} tickers (combined short and long-term range: {combined_start_date} to {combined_end_date})...")
                
                # Single fetch covering both short and long term ranges
                options_df = await self._fetch_combined_options_window(
                    tickers_upper=tickers_upper,
                    start_date=combined_start_date,
                    end_date=combined_end_date,
                    timestamp_lookback_days=combined_timestamp_lookback,
                    max_workers=max_workers,
                    max_concurrent=max_concurrent,
                    batch_size=batch_size,
                )
                
                self._log("INFO", f"Fetched {len(options_df)} total options (combined short and long-term)")
                
                # Split the combined results into short and long term
                if not options_df.empty and 'expiration_date' in options_df.columns:
                    # Normalize expiration_date for comparison
                    from datetime import date as date_type
                    today_date = date_type.today()
                    
                    def get_days_to_expiry(exp_date):
                        if pd.isna(exp_date):
                            return None
                        try:
                            if isinstance(exp_date, pd.Timestamp):
                                exp_dt = exp_date.date() if hasattr(exp_date, 'date') else pd.to_datetime(exp_date).date()
                            else:
                                exp_dt = pd.to_datetime(exp_date).date()
                            return (exp_dt - today_date).days
                        except:
                            return None
                    
                    options_df['_temp_days_to_expiry'] = options_df['expiration_date'].apply(get_days_to_expiry)
                    
                    # Short-term options: within the original short-term date range
                    short_mask = options_df['_temp_days_to_expiry'].notna()
                    if end_date:
                        end_dt = pd.to_datetime(end_date).date()
                        short_mask = short_mask & (options_df['expiration_date'].apply(lambda x: pd.to_datetime(x).date() if pd.notna(x) else None) <= end_dt)
                    if start_date:
                        start_dt = pd.to_datetime(start_date).date()
                        short_mask = short_mask & (options_df['expiration_date'].apply(lambda x: pd.to_datetime(x).date() if pd.notna(x) else None) >= start_dt)
                    
                    # Long-term options: within the long-term date range
                    long_mask = options_df['_temp_days_to_expiry'].notna()
                    if long_start_date:
                        long_start_dt = pd.to_datetime(long_start_date).date()
                        long_mask = long_mask & (options_df['expiration_date'].apply(lambda x: pd.to_datetime(x).date() if pd.notna(x) else None) >= long_start_dt)
                    if long_end_date:
                        long_end_dt = pd.to_datetime(long_end_date).date()
                        long_mask = long_mask & (options_df['expiration_date'].apply(lambda x: pd.to_datetime(x).date() if pd.notna(x) else None) <= long_end_dt)
                    
                    # Extract long options before filtering short options
                    long_options_df = options_df[long_mask].copy()
                    if not long_options_df.empty:
                        long_options_df = self._filter_and_prepare_long_options(
                            long_options_df=long_options_df,
                            min_write_timestamp=min_write_timestamp,
                            long_start_date=long_start_date,
                            long_end_date=long_end_date,
                            tickers=tickers_upper
                        )
                    
                    # Keep only short-term options in main options_df
                    options_df = options_df[short_mask].copy()
                    options_df = options_df.drop(columns=['_temp_days_to_expiry'], errors='ignore')
                    
                    if not long_options_df.empty:
                        self._log("INFO", f"Split into {len(options_df)} short-term and {len(long_options_df)} long-term options")
            else:
                self._log("INFO", f"Fetching options for {len(tickers_upper)} tickers (date range: {start_date} to {end_date or 'unlimited'})...")
                options_df = await self._fetch_short_options_window(
                    tickers_upper=tickers_upper,
                    start_date=start_date,
                    end_date=end_date,
                    timestamp_lookback_days=timestamp_lookback_days,
                    max_workers=max_workers,
                    max_concurrent=max_concurrent,
                    batch_size=batch_size,
                )
                self._log("INFO", f"Fetched {len(options_df)} options")
            if options_df.empty:
                return pd.DataFrame()

            # 2) Filter to calls
            options_df = self._filter_call_options(options_df)
            if options_df.empty:
                self._log("INFO", "No call options found after filtering")
                return pd.DataFrame()

            # 3) Attach latest prices (market-time aware)
            df = await self._attach_latest_prices(options_df, tickers_upper, use_market_time)
            if df.empty:
                return pd.DataFrame()

            # 4) Derive metrics and apply filters
            df = self._derive_and_filter_short_metrics(
                df=df,
                position_size=position_size,
                days_to_expiry=days_to_expiry,
                min_volume=min_volume,
                min_premium=min_premium,
                min_write_timestamp=min_write_timestamp,
            )
            if df.empty:
                return pd.DataFrame()

            # 5) Normalize timestamps and select columns
            df = self._normalize_short_timestamps_and_select(df)
            
            # If spread mode is enabled, match short-term options with long-term options
            if spread_mode:
                self._log("INFO", f"\n=== Starting Spread Analysis ===")
                self._log("INFO", f"Short-term options found: {len(df)}")
                if not df.empty:
                    self._log("INFO", f"Short-term tickers: {df['ticker'].unique().tolist()}")
                    self._log("INFO", f"Short-term strike range: ${df['strike_price'].min():.2f} to ${df['strike_price'].max():.2f}")
                    self._log("INFO", f"Short-term days to expiry: {df['days_to_expiry'].min()} to {df['days_to_expiry'].max()}")
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
                    min_write_timestamp=min_write_timestamp,
                    long_options_df=long_options_df  # reuse concurrently fetched long options if available
                )
            
            if self.debug:
                print(f"DEBUG: Final options count before spread/filtering: {len(df)}", file=sys.stderr)
            
            return df
        except Exception as e:
            self._log("ERROR", f"Error analyzing options: {e}")
            import traceback
            if self.debug:
                traceback.print_exc()
            return pd.DataFrame()

    # ===== Helper methods for analyze_options =====
    async def _fetch_short_options_window(
        self,
        tickers_upper: List[str],
        start_date: str,
        end_date: Optional[str],
        timestamp_lookback_days: int,
        max_workers: int,
        max_concurrent: int,
        batch_size: int
    ) -> pd.DataFrame:
        self._log("INFO", f"Starting options fetch for {len(tickers_upper)} tickers (date range: {start_date} to {end_date or 'unlimited'})...")
        if self.debug:
            print(f"DEBUG: Starting options fetch for {len(tickers_upper)} tickers", file=sys.stderr)
            print(f"DEBUG: Date range: {start_date} to {end_date}", file=sys.stderr)
            print(f"DEBUG: Tickers: {tickers_upper[:10]}{'...' if len(tickers_upper) > 10 else ''}", file=sys.stderr)

        if max_workers > 1:
            if self.debug:
                print(f"DEBUG: Using multiprocess mode with {max_workers} workers", file=sys.stderr)
            options_df = await self.db.get_latest_options_data_batch_multiprocess(
                tickers=tickers_upper,
                start_datetime=start_date,
                end_datetime=end_date,
                batch_size=batch_size,
                max_workers=max_workers,
                timestamp_lookback_days=timestamp_lookback_days
            )
        else:
            if self.debug:
                print("DEBUG: Using single-process mode", file=sys.stderr)
            options_df = await self.db.get_latest_options_data_batch(
                tickers=tickers_upper,
                start_datetime=start_date,
                end_datetime=end_date,
                max_concurrent=max_concurrent,
                batch_size=batch_size,
                timestamp_lookback_days=timestamp_lookback_days
            )

        if self.debug:
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
        
        # Ensure ticker column exists - extract from option_ticker if missing
        if not options_df.empty and 'ticker' not in options_df.columns:
            if 'option_ticker' in options_df.columns:
                if self.debug:
                    print("DEBUG: ticker column missing, extracting from option_ticker", file=sys.stderr)
                options_df['ticker'] = options_df['option_ticker'].apply(self._extract_ticker_from_option_ticker)
                
                if self.debug:
                    extracted_tickers = options_df['ticker'].dropna().unique().tolist()
                    print(f"DEBUG: Extracted tickers from option_ticker: {extracted_tickers[:10]}{'...' if len(extracted_tickers) > 10 else ''}", file=sys.stderr)
            else:
                # If we have the list of tickers being queried, we could try to match, but it's safer to fail
                if self.debug:
                    print("DEBUG: Warning: Neither 'ticker' nor 'option_ticker' column found in options DataFrame", file=sys.stderr)
        
        self._log("INFO", f"Finished options fetch: {len(options_df)} options retrieved for {len(tickers_upper)} tickers")
        return options_df

    def _filter_call_options(self, options_df: pd.DataFrame) -> pd.DataFrame:
        before_call_filter = len(options_df)
        if 'option_type' in options_df.columns:
            options_df = options_df[options_df['option_type'] == 'call']
            self._log("INFO", f"After call filter: {len(options_df)} options (was {before_call_filter})")
            if self.debug:
                print(f"DEBUG: After call filter: {len(options_df)} options (was {before_call_filter})", file=sys.stderr)
        else:
            self._log("WARNING", "'option_type' column not found in options DataFrame")
            if self.debug:
                print("DEBUG: Warning: 'option_type' column not found in options DataFrame", file=sys.stderr)
        return options_df

    async def _attach_latest_prices(
        self,
        options_df: pd.DataFrame,
        tickers_upper: List[str],
        use_market_time: bool,
        timestamp_cache: Optional[Dict[str, pd.Timestamp]] = None
    ) -> pd.DataFrame:
        stock_prices: Dict[str, float] = {}
        price_sources: Dict[str, str] = {}
        price_timestamps: Dict[str, Any] = {}

        async def fetch_price_for_ticker(ticker):
            try:
                price_data = await self.db.get_latest_price_with_data(ticker, use_market_time=use_market_time)
                if price_data and price_data.get('price'):
                    price = price_data['price']
                    source = price_data.get('source', 'unknown')
                    timestamp = price_data.get('timestamp')
                    if self.debug:
                        if source == 'realtime':
                            market_status = "OPEN (using latest realtime price)"
                        elif source == 'daily':
                            market_status = "CLOSED (using last close price from daily data)"
                        elif source == 'hourly':
                            market_status = "CLOSED (using hourly close price as fallback)"
                        else:
                            market_status = f"UNKNOWN (source: {source})"
                        print(f"DEBUG: {ticker}: ${price:.2f} from {source} - Market {market_status}", file=sys.stderr)
                    return ticker, price, source, timestamp
                return ticker, None, None, None
            except Exception as e:
                self._log("WARNING", f"Could not fetch price for {ticker}: {e}")
                return ticker, None, None, None

        price_tasks = [fetch_price_for_ticker(ticker) for ticker in tickers_upper]
        price_results = await asyncio.gather(*price_tasks)
        for ticker, price, source, timestamp in price_results:
            if price is not None:
                stock_prices[ticker] = price
                price_sources[ticker] = source
                if timestamp is not None:
                    price_timestamps[ticker] = timestamp

        # Fetch previous close prices based on each ticker's current price date
        # If market is open: compare realtime price to previous close
        # If market is closed: compare current close price to previous close (day before current price's date)
        prev_close_prices: Dict[str, float] = {}
        try:
            # First, try to get previous close based on each ticker's price timestamp
            for ticker in stock_prices.keys():
                timestamp = price_timestamps.get(ticker)
                if timestamp:
                    try:
                        if isinstance(timestamp, pd.Timestamp):
                            prev_close = await _get_previous_close_for_date(self.db, ticker, timestamp, debug=self.debug)
                        else:
                            ts = pd.to_datetime(timestamp, utc=True)
                            prev_close = await _get_previous_close_for_date(self.db, ticker, ts, debug=self.debug)
                        if prev_close is not None:
                            prev_close_prices[ticker] = prev_close
                    except Exception as e:
                        if self.debug:
                            print(f"DEBUG: {ticker} - Error getting previous close for date: {e}", file=sys.stderr)
            
            # Fallback: for any tickers we couldn't get previous close for, use standard method
            missing_tickers = [t for t in stock_prices.keys() if t not in prev_close_prices]
            if missing_tickers:
                fallback_prev_closes = await self.db.get_previous_close_prices(missing_tickers)
                prev_close_prices.update(fallback_prev_closes)
            
            if self.debug:
                print(f"DEBUG: Fetched previous close prices for {len(prev_close_prices)} tickers", file=sys.stderr)
                # Debug each ticker's prices
                for ticker in list(stock_prices.keys())[:5]:  # Show first 5 for debugging
                    current = stock_prices.get(ticker)
                    prev = prev_close_prices.get(ticker)
                    timestamp = price_timestamps.get(ticker)
                    if current and prev:
                        change = current - prev
                        change_pct = (change / prev) * 100 if prev > 0 else 0
                        print(f"DEBUG: {ticker} - current=${current:.2f}, prev_close=${prev:.2f}, change=${change:.2f} ({change_pct:.2f}%), timestamp={timestamp}", file=sys.stderr)
                    elif current:
                        print(f"DEBUG: {ticker} - current=${current:.2f}, prev_close=None (cannot calculate change), timestamp={timestamp}", file=sys.stderr)
        except Exception as e:
            if self.debug:
                print(f"DEBUG: Could not fetch previous close prices: {e}", file=sys.stderr)

        # Fetch latest option timestamps for each ticker (same as refresh check)
        # Use the reusable method with caching (use instance cache if no cache provided)
        cache_to_use = timestamp_cache if timestamp_cache is not None else self._timestamp_cache
        latest_opt_timestamps = await self._fetch_latest_option_timestamps(list(stock_prices.keys()), cache=cache_to_use)

        self._log("INFO", f"Fetched stock prices for {len(stock_prices)}/{len(tickers_upper)} tickers")
        if self.debug:
            print(f"DEBUG: Fetched stock prices for {len(stock_prices)}/{len(tickers_upper)} tickers", file=sys.stderr)
            if stock_prices and price_sources:
                source_counts: Dict[str, int] = {}
                for _, source in price_sources.items():
                    source_counts[source] = source_counts.get(source, 0) + 1
                print(f"DEBUG: Price sources: {source_counts}", file=sys.stderr)
                if self.debug:
                    sample_prices = [(t, stock_prices[t], price_sources.get(t, 'unknown')) for t in list(stock_prices.keys())[:5]]
                    print(f"DEBUG: Sample prices (ticker, price, source): {sample_prices}", file=sys.stderr)

        if not stock_prices:
            self._log("WARNING", "No stock prices fetched. Cannot calculate option metrics.")
            return pd.DataFrame()

        df = options_df.copy()
        if 'ticker' not in df.columns:
            self._log("ERROR", f"DataFrame missing 'ticker' column. Available columns: {list(df.columns)}")
            return pd.DataFrame()
        df['current_price'] = df['ticker'].map(stock_prices)
        df['latest_opt_ts'] = df['ticker'].map(latest_opt_timestamps)
        
        # Calculate price change and format as single column using helper function
        # Store percentage change separately for sorting
        def format_price_with_change_wrapper(row):
            ticker = row['ticker']
            current = row['current_price']
            prev_close = prev_close_prices.get(ticker) if prev_close_prices else None
            return _format_price_with_change(current, prev_close)
        
        # Apply formatting and store both display value and sort value
        result = df.apply(format_price_with_change_wrapper, axis=1, result_type='expand')
        df['price_with_change'] = result[0]
        df['price_change_pct'] = result[1]  # Store percentage for sorting
        
        before_price_filter = len(df)
        df = df[df['current_price'].notna()]
        self._log("INFO", f"After price mapping: {len(df)} options (was {before_price_filter})")
        if self.debug:
            print(f"DEBUG: After price mapping: {len(df)} options (was {before_price_filter})", file=sys.stderr)
        return df

    def _derive_and_filter_short_metrics(
        self,
        df: pd.DataFrame,
        position_size: float,
        days_to_expiry: Optional[int],
        min_volume: int,
        min_premium: float,
        min_write_timestamp: Optional[str]
    ) -> pd.DataFrame:
        """Derive metrics and apply filters for short options (uses helper functions)."""
        # Use helper function to calculate metrics
        df = _calculate_option_metrics(df, position_size, days_to_expiry)
        
        if days_to_expiry is not None and not df.empty:
            before_days_filter = len(df)
            self._log("INFO", f"After days_to_expiry filter ({days_to_expiry}): {len(df)} options (was {before_days_filter})")
            if self.debug:
                print(f"DEBUG: After days_to_expiry filter ({days_to_expiry}): {len(df)} options (was {before_days_filter})", file=sys.stderr)

        # Apply basic filters using helper function
            before_volume_filter = len(df)
        df = _apply_basic_filters(df, min_volume, min_premium, min_write_timestamp)
        
        if min_volume > 0 and before_volume_filter != len(df):
            self._log("INFO", f"After min_volume filter ({min_volume}): {len(df)} options (was {before_volume_filter})")
            if self.debug:
                print(f"DEBUG: After min_volume filter ({min_volume}): {len(df)} options (was {before_volume_filter})", file=sys.stderr)

        if min_premium > 0.0 and before_volume_filter != len(df):
            self._log("INFO", f"After min_premium filter ({min_premium}): {len(df)} options")
            if self.debug:
                print(f"DEBUG: After min_premium filter ({min_premium}): {len(df)} options", file=sys.stderr)

        if min_write_timestamp:
            try:
                import pytz
                est = pytz.timezone('America/New_York')
                min_ts = pd.to_datetime(min_write_timestamp)
                if min_ts.tz is None:
                    min_ts = est.localize(min_ts)
                min_ts_utc = min_ts.astimezone(pytz.UTC)
                self._log("INFO", f"Filtered options to those written after {min_write_timestamp} EST ({min_ts_utc} UTC)")
            except Exception as e:
                self._log("WARNING", f"Could not apply write timestamp filter: {e}")
        
        return df

    def _normalize_short_timestamps_and_select(self, df: pd.DataFrame) -> pd.DataFrame:
        """Normalize timestamps and select output columns (uses helper function)."""
        return _normalize_and_select_columns(df)
    
    def _calculate_long_options_date_range(
        self,
        spread_long_days: int,
        spread_long_days_tolerance: int,
        spread_long_min_days: Optional[int]
    ) -> Tuple[str, str]:
        """
        Calculate the date range for fetching long-term options.
        
        Returns:
            Tuple of (start_date, end_date) as strings in YYYY-MM-DD format
        """
        from datetime import date, timedelta
        today = date.today()
        
        if spread_long_min_days is not None:
            long_start_date = (today + timedelta(days=spread_long_min_days)).strftime('%Y-%m-%d')
            long_end_date = (today + timedelta(days=spread_long_days)).strftime('%Y-%m-%d')
            self._log("INFO", f"Long-term options expiring between {spread_long_min_days} and {spread_long_days} days")
            self._log("INFO", f"  Date range: {long_start_date} to {long_end_date}")
            if self.debug:
                print(f"DEBUG: Calculated long-term date range: {long_start_date} to {long_end_date} (from today {today})", file=sys.stderr)
        else:
            long_start_date = (today + timedelta(days=spread_long_days - spread_long_days_tolerance)).strftime('%Y-%m-%d')
            long_end_date = (today + timedelta(days=spread_long_days + spread_long_days_tolerance)).strftime('%Y-%m-%d')
            self._log("INFO", f"Long-term options expiring around {spread_long_days} days (±{spread_long_days_tolerance} days)")
            self._log("INFO", f"  Date range: {long_start_date} to {long_end_date}")
            if self.debug:
                print(f"DEBUG: Calculated long-term date range: {long_start_date} to {long_end_date} (from today {today})", file=sys.stderr)
        
        return long_start_date, long_end_date
    
    def _calculate_combined_date_range(
        self,
        short_start_date: str,
        short_end_date: Optional[str],
        long_start_date: str,
        long_end_date: str
    ) -> Tuple[str, Optional[str]]:
        """
        Calculate the combined date range covering both short and long term options.
        
        Returns:
            Tuple of (combined_start_date, combined_end_date) as strings in YYYY-MM-DD format
        """
        from datetime import datetime
        
        # Convert all dates to date objects for comparison
        short_start_dt = datetime.strptime(short_start_date, '%Y-%m-%d').date()
        short_end_dt = datetime.strptime(short_end_date, '%Y-%m-%d').date() if short_end_date else None
        long_start_dt = datetime.strptime(long_start_date, '%Y-%m-%d').date()
        long_end_dt = datetime.strptime(long_end_date, '%Y-%m-%d').date()
        
        # Find the overall min and max dates
        all_dates = [short_start_dt, long_start_dt, long_end_dt]
        if short_end_dt:
            all_dates.append(short_end_dt)
        
        combined_start = min(all_dates)
        combined_end = max(all_dates)
        
        return combined_start.strftime('%Y-%m-%d'), combined_end.strftime('%Y-%m-%d')
    
    async def _fetch_combined_options_window(
        self,
        tickers_upper: List[str],
        start_date: str,
        end_date: Optional[str],
        timestamp_lookback_days: int,
        max_workers: int,
        max_concurrent: int,
        batch_size: int
    ) -> pd.DataFrame:
        """Fetch options for combined date range (used in spread mode)."""
        self._log("INFO", f"Starting combined options fetch for {len(tickers_upper)} tickers (combined date range: {start_date} to {end_date or 'unlimited'})...")
        if self.debug:
            print(f"DEBUG: Starting combined options fetch for {len(tickers_upper)} tickers", file=sys.stderr)
            print(f"DEBUG: Combined date range: {start_date} to {end_date}", file=sys.stderr)
            print(f"DEBUG: Tickers: {tickers_upper[:10]}{'...' if len(tickers_upper) > 10 else ''}", file=sys.stderr)

        if max_workers > 1:
            if self.debug:
                print(f"DEBUG: Using multiprocess mode with {max_workers} workers", file=sys.stderr)
            options_df = await self.db.get_latest_options_data_batch_multiprocess(
                tickers=tickers_upper,
                start_datetime=start_date,
                end_datetime=end_date,
                batch_size=batch_size,
                max_workers=max_workers,
                timestamp_lookback_days=timestamp_lookback_days
            )
        else:
            if self.debug:
                print("DEBUG: Using single-process mode", file=sys.stderr)
            options_df = await self.db.get_latest_options_data_batch(
                tickers=tickers_upper,
                start_datetime=start_date,
                end_datetime=end_date,
                max_concurrent=max_concurrent,
                batch_size=batch_size,
                timestamp_lookback_days=timestamp_lookback_days
            )

        if self.debug:
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
        
        # Ensure ticker column exists - extract from option_ticker if missing
        if not options_df.empty and 'ticker' not in options_df.columns:
            if 'option_ticker' in options_df.columns:
                if self.debug:
                    print("DEBUG: ticker column missing, extracting from option_ticker", file=sys.stderr)
                options_df['ticker'] = options_df['option_ticker'].apply(self._extract_ticker_from_option_ticker)
                
                if self.debug:
                    extracted_tickers = options_df['ticker'].dropna().unique().tolist()
                    print(f"DEBUG: Extracted tickers from option_ticker: {extracted_tickers[:10]}{'...' if len(extracted_tickers) > 10 else ''}", file=sys.stderr)
            else:
                # If we have the list of tickers being queried, we could try to match, but it's safer to fail
                if self.debug:
                    print("DEBUG: Warning: Neither 'ticker' nor 'option_ticker' column found in options DataFrame", file=sys.stderr)
        
        self._log("INFO", f"Finished combined options fetch: {len(options_df)} options retrieved for {len(tickers_upper)} tickers")
        return options_df
    
    async def _fetch_long_term_options(
        self,
        tickers: List[str],
        long_start_date: str,
        long_end_date: str,
        timestamp_lookback_days: int,
        max_workers: int,
        max_concurrent: int,
        batch_size: int
    ) -> pd.DataFrame:
        """
        Fetch long-term options from the database.
        
        Returns:
            DataFrame with long-term options data
        """
        # Use a much larger timestamp lookback for long-term options since they may have been
        # written weeks or months ago but are still valid. Use at least 180 days to catch options
        # that were written when they were first listed (which could be months before expiration)
        long_timestamp_lookback_days = max(timestamp_lookback_days, 180)
        if self.debug:
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
            
            # Ensure ticker column exists - extract from option_ticker if missing
            if not long_options_df.empty and 'ticker' not in long_options_df.columns:
                if 'option_ticker' in long_options_df.columns:
                    if self.debug:
                        print("DEBUG: ticker column missing in long options, extracting from option_ticker", file=sys.stderr)
                    long_options_df['ticker'] = long_options_df['option_ticker'].apply(self._extract_ticker_from_option_ticker)
                    
                    if self.debug:
                        extracted_tickers = long_options_df['ticker'].dropna().unique().tolist()
                        print(f"DEBUG: Extracted tickers from long option_ticker: {extracted_tickers[:10]}{'...' if len(extracted_tickers) > 10 else ''}", file=sys.stderr)
                else:
                    if self.debug:
                        print("DEBUG: Warning: Neither 'ticker' nor 'option_ticker' column found in long options DataFrame", file=sys.stderr)
            
            return long_options_df
        except Exception as e:
            self._log("ERROR", f"Error fetching long-term options: {e}")
            import traceback
            if self.debug:
                traceback.print_exc()
            raise
    
    def _filter_and_prepare_long_options(
        self,
        long_options_df: pd.DataFrame,
        min_write_timestamp: Optional[str],
        long_start_date: str,
        long_end_date: str,
        tickers: List[str]
    ) -> pd.DataFrame:
        """
        Filter and prepare long options DataFrame (call options only, write timestamp filter, etc.).
        
        Returns:
            Filtered and prepared DataFrame
        """
        if self.debug:
            print(f"DEBUG: Fetched {len(long_options_df)} total options from database")
            if not long_options_df.empty:
                print(f"DEBUG: Long options columns: {list(long_options_df.columns)}")
                if 'ticker' in long_options_df.columns:
                    unique_tickers = long_options_df['ticker'].unique().tolist()
                    print(f"DEBUG: Long options tickers: {unique_tickers}")
                    for ticker in unique_tickers:
                        count = len(long_options_df[long_options_df['ticker'] == ticker])
                        print(f"DEBUG:   {ticker}: {count} options")
                if 'option_type' in long_options_df.columns:
                    print(f"DEBUG: Option types: {long_options_df['option_type'].unique().tolist()}")
            else:
                print(f"DEBUG: No long-term options found in database for date range {long_start_date} to {long_end_date}")
        
        # Apply write timestamp filter to long options if specified
        if min_write_timestamp and not long_options_df.empty:
            before_count = len(long_options_df)
            long_options_df = _apply_basic_filters(long_options_df, 0, 0.0, min_write_timestamp)
            after_count = len(long_options_df)
            if before_count != after_count:
                self._log("INFO", f"Filtered long options by write timestamp: {before_count} -> {after_count} options")
                if self.debug:
                    import pytz
                    est = pytz.timezone('America/New_York')
                    min_ts = pd.to_datetime(min_write_timestamp)
                    if min_ts.tz is None:
                        min_ts = est.localize(min_ts)
                    min_ts_utc = min_ts.astimezone(pytz.UTC)
                    print(f"DEBUG: Applied write timestamp filter >= {min_ts_utc} UTC")
        
        if long_options_df.empty:
            self._log("WARNING", "No long-term options found for spread analysis.")
            if self.debug:
                print("DEBUG: This could mean:", file=sys.stderr)
                print("  1. No options data in database for the specified date range", file=sys.stderr)
                print("  2. Options exist but not in the target expiration window", file=sys.stderr)
                print(f"  3. Check database for tickers: {tickers}", file=sys.stderr)
            return pd.DataFrame()
        
        # Filter for call options only
        if 'option_type' in long_options_df.columns:
            long_options_df = long_options_df[long_options_df['option_type'] == 'call'].copy()
        
        if self.debug:
            print(f"DEBUG: After filtering for calls: {len(long_options_df)} call options")
        
        if long_options_df.empty:
            self._log("WARNING", "No long-term call options found for spread analysis.")
            return pd.DataFrame()
        
        # Ensure we have a copy before modifying
        long_options_df = long_options_df.copy()
        if 'implied_volatility' in long_options_df.columns:
            long_options_df['implied_volatility'] = pd.to_numeric(long_options_df['implied_volatility'], errors='coerce').round(4)
        else:
            long_options_df['implied_volatility'] = pd.Series([float('nan')] * len(long_options_df), index=long_options_df.index)
        
        # Calculate days to expiry for long options using utility functions
        long_options_df['expiration_date'] = long_options_df['expiration_date'].apply(_normalize_to_utc)
        today_ts = pd.Timestamp.now(tz='UTC').normalize()
        long_options_df['days_to_expiry'] = long_options_df['expiration_date'].apply(lambda x: _safe_days_calc(x, today_ts))
        
        if self.debug:
            print(f"DEBUG: Long options days to expiry range: {long_options_df['days_to_expiry'].min()} to {long_options_df['days_to_expiry'].max()}")
            print(f"DEBUG: Long options strike price range: {long_options_df['strike_price'].min()} to {long_options_df['strike_price'].max()}")
        
        return long_options_df
    
    def _prepare_spread_matching_data(
        self,
        df_short: pd.DataFrame,
        long_options_df: pd.DataFrame
    ) -> Tuple[List[Dict], Dict[str, List[Dict]]]:
        """
        Prepare data structures for multiprocessing spread matching.
        
        Returns:
            Tuple of (short_rows_list, long_options_dict)
        """
        # Reset index and deduplicate
        df_short = df_short.reset_index(drop=True)
        if 'option_ticker' in df_short.columns:
            before_dedup = len(df_short)
            df_short = df_short.drop_duplicates(subset=['option_ticker'], keep='first')
            if self.debug and len(df_short) < before_dedup:
                print(f"DEBUG: Deduplicated df_short: {before_dedup} -> {len(df_short)} rows (removed {before_dedup - len(df_short)} duplicates)", file=sys.stderr)
        
        # Convert df_short rows to dictionaries (serializable)
        short_rows_list = [row.to_dict() for _, row in df_short.iterrows()]
        
        # Convert long_options_df to a dictionary structure (ticker -> list of option dicts)
        long_options_dict = {}
        for ticker in long_options_df['ticker'].unique():
            ticker_options = long_options_df[long_options_df['ticker'] == ticker]
            long_options_dict[ticker] = []
            for _, row in ticker_options.iterrows():
                row_dict = row.to_dict()
                # Timestamps are preserved as-is (they're pickleable)
                if 'expiration_date' in row_dict and isinstance(row_dict['expiration_date'], pd.Timestamp):
                    row_dict['expiration_date'] = row_dict['expiration_date']
                long_options_dict[ticker].append(row_dict)
        
        return short_rows_list, long_options_dict
    
    async def _execute_spread_matching(
        self,
        short_rows_list: List[Dict],
        long_options_dict: Dict[str, List[Dict]],
        spread_strike_tolerance: float,
        spread_long_days: int,
        position_size: float,
        max_workers: int
    ) -> List[Dict]:
        """
        Execute spread matching using multiprocessing or sequential processing.
        
        Returns:
            List of spread result dictionaries
        """
        # Prepare arguments for multiprocessing
        process_args = [
            (
                short_row_dict,
                long_options_dict,
                spread_strike_tolerance,
                spread_long_days,
                position_size,
                self.risk_free_rate,
                self.debug
            )
            for short_row_dict in short_rows_list
        ]
        
        # Use multiprocessing to process spread matches in parallel
        if max_workers > 1 and len(short_rows_list) > 0:
            if self.debug:
                print(f"DEBUG: Processing {len(short_rows_list)} spread matches using {max_workers} CPU workers", file=sys.stderr)
            
            loop = asyncio.get_event_loop()
            with ProcessPoolExecutor(max_workers=max_workers) as executor:
                futures = [loop.run_in_executor(executor, _process_spread_match, args) for args in process_args]
                results = await asyncio.gather(*futures)
            
            # Filter out None results (no matches)
            return [r for r in results if r is not None]
        else:
            # Fallback to sequential processing
            if self.debug and max_workers <= 1:
                print(f"DEBUG: Using sequential processing (max_workers={max_workers})", file=sys.stderr)
            spread_results = []
            for args in process_args:
                result = _process_spread_match(args)
                if result is not None:
                    spread_results.append(result)
            return spread_results
    
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
            min_write_timestamp: Optional[str],
            long_options_df: Optional[pd.DataFrame] = None
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
            long_options_df: Optional pre-fetched long-term options DataFrame. If provided, skips fetching.
            Other args: Same as analyze_options
            
        Returns:
            DataFrame with spread analysis including long option details and net calculations
        """
        if df_short.empty:
            return df_short
        
        try:
            # Calculate date range for long options
            long_start_date, long_end_date = self._calculate_long_options_date_range(
                spread_long_days, spread_long_days_tolerance, spread_long_min_days
            )
            
            if self.debug:
                print(f"DEBUG: Tickers to fetch: {tickers}", file=sys.stderr)
                print(f"DEBUG: Short-term options count: {len(df_short)}", file=sys.stderr)
                print(f"DEBUG: Short-term date range in df_short:", file=sys.stderr)
                if not df_short.empty and 'expiration_date' in df_short.columns:
                    print(f"  Min expiration: {df_short['expiration_date'].min()}", file=sys.stderr)
                    print(f"  Max expiration: {df_short['expiration_date'].max()}", file=sys.stderr)
            
            # Fetch long-term options if not already provided
            if long_options_df is None:
                self._log("INFO", f"Fetching long-term options...")
                long_options_df = await self._fetch_long_term_options(
                    tickers=tickers,
                    long_start_date=long_start_date,
                    long_end_date=long_end_date,
                    timestamp_lookback_days=timestamp_lookback_days,
                    max_workers=max_workers,
                    max_concurrent=max_concurrent,
                    batch_size=batch_size
                )
            else:
                self._log("INFO", f"Using pre-fetched long-term options ({len(long_options_df)} rows)")
                if self.debug:
                    print(f"DEBUG: Using pre-fetched long-term options DataFrame with {len(long_options_df)} rows", file=sys.stderr)
                long_options_df = long_options_df.copy()
            
            # Filter and prepare long options
            long_options_df = self._filter_and_prepare_long_options(
                long_options_df=long_options_df,
                min_write_timestamp=min_write_timestamp,
                long_start_date=long_start_date,
                long_end_date=long_end_date,
                tickers=tickers
            )
            
            if long_options_df.empty:
                return pd.DataFrame()
            
            # Prepare data for spread matching
            short_rows_list, long_options_dict = self._prepare_spread_matching_data(
                df_short=df_short,
                long_options_df=long_options_df
            )
            
            # Execute spread matching
            spread_results = await self._execute_spread_matching(
                short_rows_list=short_rows_list,
                long_options_dict=long_options_dict,
                spread_strike_tolerance=spread_strike_tolerance,
                spread_long_days=spread_long_days,
                position_size=position_size,
                max_workers=max_workers
            )
            
            if not spread_results:
                self._log("WARNING", "No matching spread opportunities found within strike tolerance.")
                if self.debug:
                    print(f"DEBUG: Summary - Processed {len(short_rows_list)} short options, but none matched with long options", file=sys.stderr)
                    print("DEBUG: Possible reasons:", file=sys.stderr)
                    print("  1. Strike prices don't overlap between short and long options", file=sys.stderr)
                    print("  2. Strike tolerance is too strict (try increasing --spread-strike-tolerance)", file=sys.stderr)
                    print(f"  3. Long options not available in the {spread_long_days}±{spread_long_days_tolerance} day window", file=sys.stderr)
                    print(f"  4. Try increasing --spread-long-days-tolerance to widen the search window", file=sys.stderr)
                return pd.DataFrame()
            
            # Create spread DataFrame
            df_spread = pd.DataFrame(spread_results)
            
            self._log("INFO", f"✓ Found {len(df_spread)} spread opportunities (matched short and long options).")
            
            return df_spread
            
        except Exception as e:
            self._log("ERROR", f"Error creating spread analysis: {e}")
            import traceback
            if self.debug:
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
        top_n: Optional[int] = 1
    ) -> str:
        """Format the analysis results for output."""
        if self.debug:
            print(f"DEBUG: format_output called with {len(df)} rows", file=sys.stderr)
        if df.empty:
            self._log("INFO", "DataFrame is empty in format_output")
            return "No options data found matching the criteria."
        
        # Add financial information to the dataframe
        # DataFrame already has named columns from the new batch method
        if 'ticker' not in df.columns:
            return "Error: DataFrame missing 'ticker' column"
        
        df_renamed = df.copy()
        df_renamed['pe_ratio'] = df_renamed['ticker'].map(lambda x: financial_data.get(x, {}).get('pe_ratio'))
        df_renamed['market_cap'] = df_renamed['ticker'].map(lambda x: financial_data.get(x, {}).get('market_cap'))
        df_renamed['market_cap_b'] = df_renamed['market_cap'].apply(lambda x: round(x / 1e9, 2) if pd.notna(x) and x is not None else None)
        if 'implied_volatility' in df_renamed.columns:
            df_renamed['implied_volatility'] = pd.to_numeric(df_renamed['implied_volatility'], errors='coerce')
        if 'long_implied_volatility' in df_renamed.columns:
            df_renamed['long_implied_volatility'] = pd.to_numeric(df_renamed['long_implied_volatility'], errors='coerce')
        if 'long_contracts_available' in df_renamed.columns:
            df_renamed['long_contracts_available'] = pd.to_numeric(df_renamed['long_contracts_available'], errors='coerce')
        
        # Add option premium percentage calculation
        df_renamed['option_premium_percentage'] = (df_renamed['option_premium'] / df_renamed['current_price'] * 100).round(2)
        # Add premium vs above difference percentage relative to price_above_current
        df_renamed['premium_above_diff_percentage'] = (
            (
                (df_renamed['option_premium'] - df_renamed['price_above_current']) / df_renamed['price_above_current']
            ).where(df_renamed['price_above_current'] != 0)
            * 100
        ).round(2)
        
        # Format latest_opt_ts FIRST as age in seconds (it's a float, not a timestamp)
        # This must happen BEFORE any timestamp normalization to prevent conversion
        if 'latest_opt_ts' in df_renamed.columns:
            df_renamed['latest_opt_ts'] = df_renamed['latest_opt_ts'].apply(_format_age_seconds).astype(str)
        
        # Convert timestamps to EST/EDT (America/New_York) for display, except expiration_date which should be UTC
        try:
            if 'expiration_date' in df_renamed.columns:
                ser = df_renamed['expiration_date'].apply(_normalize_expiration_date_for_display)
                df_renamed['expiration_date'] = ser.dt.strftime('%Y-%m-%d %H:%M:%S')
            
            for ts_col in ['last_quote_timestamp', 'write_timestamp']:
                if ts_col in df_renamed.columns:
                    ser = df_renamed[ts_col].apply(_normalize_timestamp_for_display)
                    df_renamed[ts_col] = ser.dt.strftime('%Y-%m-%d %H:%M:%S')
        except Exception:
            # If conversion fails, leave as-is
            pass
        
        # Reorder columns for better display
        # Check if this is spread mode by looking for spread-specific columns
        is_spread_mode = 'net_premium' in df_renamed.columns
        
        if is_spread_mode:
            display_columns = [
                'ticker', 'pe_ratio', 'market_cap_b', 'current_price', 'price_with_change', 'price_change_pct',
                # Short option details
                'strike_price', 'price_above_current', 'option_premium', 'bid_ask',
                'implied_volatility', 'delta', 'theta', 'expiration_date', 'days_to_expiry',
                'short_premium_total', 'short_daily_premium',
                # Long option details
                'long_strike_price', 'long_option_premium', 'long_bid_ask', 'long_implied_volatility', 'long_delta', 'long_theta',
                'long_expiration_date', 'long_days_to_expiry', 'long_premium_total', 'long_contracts_available',
                # Premium comparison
                'premium_diff',
                # Net spread calculations
                'net_premium', 'net_daily_premium',
                # Additional details
                'volume', 'num_contracts', 'option_ticker', 'long_option_ticker', 'latest_opt_ts'
            ]
        else:
            display_columns = [
                'ticker', 'pe_ratio', 'market_cap_b', 'current_price', 'price_with_change', 'price_change_pct', 'strike_price',
                'price_above_current', 'option_premium', 'bid_ask', 'premium_above_diff_percentage',
                'implied_volatility', 'delta', 'theta',
                'potential_premium', 'daily_premium', 'expiration_date', 'days_to_expiry',
                #'volume', 'num_contracts', 'last_quote_timestamp', 'option_ticker'
                'volume', 'num_contracts', 'option_ticker', 'latest_opt_ts'
            ]
        
        # Only include columns that exist in the dataframe
        available_columns = [col for col in display_columns if col in df_renamed.columns]
        df_display = df_renamed[available_columns].copy()
        
        # Apply filters if provided (after financial data is added)
        if filters:
            before_filter_count = len(df_display)
            df_display = FilterParser.apply_filters(df_display, filters, filter_logic)
            self._log("INFO", f"After filter application: {len(df_display)} rows (was {before_filter_count})")
            if self.debug:
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
                self._log("INFO", f"Limiting to top {top_n} options per ticker (after sorting)")
            else:
                # If no ticker column, just take top N overall
                df_display = df_display.head(top_n)
            self._log("INFO", f"After top-n filter ({top_n}): {len(df_display)} rows (was {before_topn})")
            if self.debug:
                print(f"DEBUG: After top-n filter ({top_n}): {len(df_display)} rows (was {before_topn})", file=sys.stderr)
        
        # Handle CSV formatting
        if output_format == 'csv':
            df_csv = df_display.copy()
            compact_headers = self._create_compact_headers(df_csv)
            header_reverse_map = {v: k for k, v in compact_headers.items()}
            
            if csv_columns:
                resolved_columns = []
                for requested in csv_columns:
                    if requested in df_csv.columns:
                        resolved_columns.append(requested)
                        continue
                    # Allow compact header names
                    if requested in header_reverse_map:
                        resolved_columns.append(header_reverse_map[requested])
                        continue
                    # Case-insensitive match against compact headers
                    matches = [
                        header_reverse_map[h]
                        for h in header_reverse_map
                        if h.lower() == requested.lower()
                    ]
                    if len(matches) == 1:
                        resolved_columns.append(matches[0])
                        continue
                    # Case-insensitive substring match on original columns
                    substring_matches = [
                        col for col in df_csv.columns
                        if requested.lower() in str(col).lower()
                    ]
                    if len(substring_matches) == 1:
                        resolved_columns.append(substring_matches[0])
                if resolved_columns:
                    df_csv = df_csv[resolved_columns]
                    compact_headers = {
                        col: compact_headers[col] for col in resolved_columns if col in compact_headers
                    }
            
            df_csv = df_csv.rename(columns=compact_headers)
            return self._format_csv_output(df_csv, csv_delimiter, csv_quoting, group_by, output_file)
        
        if group_by == 'ticker':
            # Group by ticker and show results per ticker
            output_lines = []
            for ticker in sorted(df_display['ticker'].unique()):
                ticker_data = df_display[df_display['ticker'] == ticker]
                output_lines.append(f"\n--- {ticker} ---")
                
                if output_format == 'table':
                    # Format numeric columns for better display
                    ticker_data_formatted = _format_dataframe_for_display(ticker_data)
                    
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
                df_formatted = _format_dataframe_for_display(df_display)
                
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
            self._log("INFO", f"Results saved to {output_file}")
        
        return result


# ============================================================================
# Helper functions for main() - modularized for reusability
# ============================================================================

# Alias for backward compatibility
_get_redis_client_for_refresh = get_redis_client_for_refresh
_check_redis_refresh_pending = check_redis_refresh_pending
_set_redis_refresh_pending = set_redis_refresh_pending
_clear_redis_refresh_pending = clear_redis_refresh_pending


def _build_analysis_args(args, tickers: List[str], filters: List[FilterExpression]) -> Dict[str, Any]:
    """Build arguments dictionary for analyze_options call. Reusable for both initial and refresh analysis."""
    return {
        'tickers': tickers,
        'days_to_expiry': args.days,
        'min_volume': args.min_volume,
        'max_days': args.max_days,
        'batch_size': args.batch_size,
        'min_premium': args.min_premium,
        'position_size': args.position_size,
        'filters': filters,
        'filter_logic': args.filter_logic,
        'use_market_time': not args.no_market_time,
        'start_date': args.start_date,
        'end_date': args.end_date,
        'timestamp_lookback_days': args.timestamp_lookback_days,
        'max_workers': args.max_workers,
        'spread_mode': args.spread,
        'spread_strike_tolerance': args.spread_strike_tolerance,
        'spread_long_days': args.spread_long_days,
        'spread_long_days_tolerance': args.spread_long_days_tolerance,
        'spread_long_min_days': args.spread_long_min_days,
        'min_write_timestamp': args.min_write_timestamp
    }


async def _check_tickers_for_refresh(
    analyzer: OptionsAnalyzer,
    tickers: List[str],
    refresh_threshold_seconds: int,
    redis_client: Optional[Any] = None,
    timestamp_cache: Optional[Dict[str, pd.Timestamp]] = None,
    min_write_timestamp: Optional[str] = None
) -> List[str]:
    """
    Check which tickers need refresh based on their latest write_timestamp.
    Also includes tickers that don't meet the min_write_timestamp criteria.
    
    Args:
        analyzer: OptionsAnalyzer instance
        tickers: List of ticker symbols to check
        refresh_threshold_seconds: Age threshold in seconds for refresh
        redis_client: Optional Redis client for deduplication
        timestamp_cache: Optional cache dictionary to reuse previously fetched timestamps
        min_write_timestamp: Optional minimum write timestamp (EST format) - tickers with data older than this will be refreshed
        
    Returns:
        List of ticker symbols that need refresh
    """
    # Use analyzer's method to fetch timestamps
    async def fetch_timestamps(tickers_list: List[str], cache: Optional[Dict]) -> Dict[str, Optional[float]]:
        """Wrapper to use analyzer's timestamp fetching method."""
        return await analyzer._fetch_latest_option_timestamps(tickers_list, cache=cache)
    
    return await common_check_tickers_for_refresh(
        db=analyzer.db,
        tickers=tickers,
        refresh_threshold_seconds=refresh_threshold_seconds,
        fetch_timestamp_func=fetch_timestamps,
        redis_client=redis_client,
        timestamp_cache=timestamp_cache,
        min_write_timestamp=min_write_timestamp,
        debug=analyzer.debug
    )


def _calculate_refresh_date_ranges(
    analyzer: OptionsAnalyzer,
    args,
    today_str: str,
    today_date
) -> Tuple[str, Optional[str], int]:
    """
    Calculate date ranges and max_days for refresh fetch.
    For refresh, we always use 30 days max expiration.
    
    Returns:
        Tuple of (short_start_date, max_end_date, combined_max_days)
    """
    from datetime import date
    
    # Short-term date range (from original analysis)
    short_start_date = args.start_date if args.start_date else today_str
    short_end_date = args.end_date
    
    # Calculate long-term date range if in spread mode
    long_start_date = None
    long_end_date = None
    if args.spread:
        long_start_date, long_end_date = analyzer._calculate_long_options_date_range(
            args.spread_long_days,
            args.spread_long_days_tolerance,
            args.spread_long_min_days
        )
    
    # Determine the maximum end date for combined fetch (if spread mode) or display
    max_end_date = short_end_date
    if args.spread and long_end_date:
        # Convert to date objects for comparison
        short_end_dt = datetime.strptime(short_end_date, '%Y-%m-%d').date() if short_end_date else None
        long_end_dt = datetime.strptime(long_end_date, '%Y-%m-%d').date()
        
        if short_end_dt:
            max_end_date = max(short_end_dt, long_end_dt).strftime('%Y-%m-%d')
        else:
            max_end_date = long_end_date
    
    # For refresh, always use 30 days max expiration
    combined_max_days = 30
    
    return short_start_date, max_end_date, combined_max_days


def _process_refresh_batch(args_tuple):
    """
    Process a batch of tickers for refresh in a separate process.
    
    Args:
        args_tuple: Tuple containing:
            - tickers: List of ticker symbols to refresh
            - db_conn: Database connection string
            - api_key: Polygon API key
            - data_dir: Data directory
            - today_str: Today's date string
            - enable_cache: Whether caching is enabled
            - redis_url: Redis URL for caching
            - log_level: Logging level
            - debug: Whether debug output is enabled
    
    Returns:
        List of result dictionaries, one per ticker
    """
    import asyncio
    import sys
    import os
    import pandas as pd
    from pathlib import Path
    
    # Unpack arguments
    (tickers, db_conn, api_key, data_dir, today_str, enable_cache, redis_url, log_level, debug) = args_tuple
    
    # Re-import needed modules in worker process
    CURRENT_DIR = Path(__file__).resolve().parent
    PROJECT_ROOT = CURRENT_DIR.parent
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    
    from common.stock_db import get_stock_db
    from common.common import get_redis_client_for_refresh, set_redis_last_write_timestamp, REDIS_AVAILABLE
    from scripts.fetch_options import HistoricalDataFetcher
    
    async def _async_process():
        # Get process ID for logging
        import multiprocessing
        process_id = os.getpid()
        try:
            process_name = multiprocessing.current_process().name
        except:
            process_name = 'unknown'
        ticker_list = ', '.join(tickers)
        print(f"INFO [PID {process_id}, Process: {process_name}]: Starting refresh batch - Processing {len(tickers)} ticker(s): {ticker_list}", file=sys.stderr)
        
        # Create database connection in worker process
        db = get_stock_db('questdb', db_config=db_conn, enable_cache=enable_cache, 
                        redis_url=redis_url, log_level=log_level)
        
        # Create fetcher instance
        fetcher = HistoricalDataFetcher(
            api_key,
            data_dir,
            quiet=True,  # Suppress fetch_options progress in workers
            snapshot_max_concurrent=0
        )
        
        results = []
        
        try:
            async with db:
                for ticker in tickers:
                    try:
                        # Get stock price for the ticker
                        stock_result = await fetcher.get_stock_price_for_date(ticker, today_str)
                        stock_close_price = stock_result['data'].get('close') if stock_result.get('success') else None
                        
                        # Fetch options - always use 30 days max expiration for refresh
                        options_result = await fetcher.get_active_options_for_date(
                            symbol=ticker,
                            target_date_str=today_str,
                            option_type='call',
                            stock_close_price=stock_close_price,
                            strike_range_percent=None,
                            max_days_to_expiry=30,  # Always use 30 days for refresh
                            include_expired=False,
                            use_cache=False,
                            save_to_csv=False,
                            use_db=False,
                            db_conn=None,
                            force_fresh=True,
                            enable_cache=enable_cache,
                            redis_url=redis_url
                        )
                        
                        if options_result.get('success'):
                            contracts = options_result['data'].get('contracts', [])
                            if contracts:
                                # Convert contracts to DataFrame and save to database
                                contracts_df = pd.DataFrame.from_records(contracts)
                                if not contracts_df.empty:
                                    # Map columns to match DB schema
                                    if 'ticker' in contracts_df.columns and 'option_ticker' not in contracts_df.columns:
                                        contracts_df = contracts_df.rename(columns={'ticker': 'option_ticker'})
                                    
                                    column_mapping = {
                                        'expiration': 'expiration_date',
                                        'strike': 'strike_price',
                                        'type': 'option_type',
                                    }
                                    for old_name, new_name in column_mapping.items():
                                        if old_name in contracts_df.columns:
                                            contracts_df = contracts_df.rename(columns={old_name: new_name})
                                    
                                    # Save to database
                                    await db.save_options_data(df=contracts_df, ticker=ticker)
                                    
                                    # Update Redis cache with the current timestamp
                                    if enable_cache and redis_url:
                                        redis_client = get_redis_client_for_refresh(redis_url) if redis_url else None
                                        if redis_client:
                                            from datetime import datetime, timezone
                                            now_utc = datetime.now(timezone.utc)
                                            set_redis_last_write_timestamp(redis_client, ticker, now_utc, ttl_seconds=86400)
                            
                            results.append({'ticker': ticker, 'success': True, 'contracts': len(contracts)})
                        else:
                            results.append({'ticker': ticker, 'success': False, 'error': options_result.get('error', 'Unknown error')})
                    except Exception as e:
                        results.append({'ticker': ticker, 'success': False, 'error': str(e)})
        except Exception as e:
            # If database connection fails, return errors for all tickers
            for ticker in tickers:
                results.append({'ticker': ticker, 'success': False, 'error': f"Database error: {str(e)}"})
        
        # Log completion with process ID
        successful_count = sum(1 for r in results if r.get('success'))
        print(f"INFO [PID {process_id}]: Completed refresh batch - {successful_count}/{len(tickers)} ticker(s) successful", file=sys.stderr)
        
        return results
    
    # Run async function in worker process
    return asyncio.run(_async_process())


async def _fetch_and_save_refresh_options(
    fetcher: Any,
    ticker: str,
    today_str: str,
    combined_max_days: int,
    analyzer: OptionsAnalyzer,
    enable_cache: bool,
    redis_client: Optional[Any] = None
) -> Dict[str, Any]:
    """Fetch and save options data for a single ticker during refresh."""
    try:
        # Get stock price for the ticker (needed for fetch_options)
        stock_result = await fetcher.get_stock_price_for_date(ticker, today_str)
        stock_close_price = stock_result['data'].get('close') if stock_result.get('success') else None
        
        # Fetch options - always use 30 days max expiration for refresh
        redis_url = os.getenv('REDIS_URL', 'redis://localhost:6379/0') if enable_cache else None
        
        options_result = await fetcher.get_active_options_for_date(
            symbol=ticker,
            target_date_str=today_str,
            option_type='call',  # Only calls for covered call analysis
            stock_close_price=stock_close_price,
            strike_range_percent=None,  # Fetch all strikes
            max_days_to_expiry=30,  # Always use 30 days for refresh
            include_expired=False,
            use_cache=False,  # Don't use CSV cache
            save_to_csv=False,  # Don't save to CSV
            use_db=False,  # We'll save manually
            db_conn=None,
            force_fresh=True,  # Force fresh fetch
            enable_cache=enable_cache,
            redis_url=redis_url
        )
        
        if options_result.get('success'):
            contracts = options_result['data'].get('contracts', [])
            if contracts:
                # Convert contracts to DataFrame and save to database
                contracts_df = pd.DataFrame.from_records(contracts)
                if not contracts_df.empty:
                    # Map columns to match DB schema
                    if 'ticker' in contracts_df.columns and 'option_ticker' not in contracts_df.columns:
                        contracts_df = contracts_df.rename(columns={'ticker': 'option_ticker'})
                    
                    column_mapping = {
                        'expiration': 'expiration_date',
                        'strike': 'strike_price',
                        'type': 'option_type',
                    }
                    for old_name, new_name in column_mapping.items():
                        if old_name in contracts_df.columns:
                            contracts_df = contracts_df.rename(columns={old_name: new_name})
                    
                    # Save to database using analyzer's db connection
                    await analyzer.db.save_options_data(df=contracts_df, ticker=ticker)
                    
                    # Update Redis cache with the current timestamp
                    if redis_client:
                        from datetime import datetime, timezone
                        now_utc = datetime.now(timezone.utc)
                        set_redis_last_write_timestamp(redis_client, ticker, now_utc, ttl_seconds=86400)
                        _clear_redis_refresh_pending(redis_client, ticker)
            
            analyzer._log("INFO", f"  ✓ Fetched and saved {len(contracts)} options contracts for {ticker}")
            return {'ticker': ticker, 'success': True, 'contracts': len(contracts)}
        else:
            # Clear Redis flag on failure
            if redis_client:
                _clear_redis_refresh_pending(redis_client, ticker)
            analyzer._log("WARNING", f"  ✗ Failed to fetch options for {ticker}: {options_result.get('error', 'Unknown error')}")
            return {'ticker': ticker, 'success': False, 'error': options_result.get('error')}
    except Exception as e:
        # Clear Redis flag on error
        if redis_client:
            _clear_redis_refresh_pending(redis_client, ticker)
        analyzer._log("ERROR", f"  ✗ Error fetching options for {ticker}: {e}")
        return {'ticker': ticker, 'success': False, 'error': str(e)}


async def _run_refresh_analysis(
    analyzer: OptionsAnalyzer,
    args,
    df: pd.DataFrame,
    filters: List[FilterExpression],
    refresh_threshold_seconds: int,
    redis_client: Optional[Any] = None,
    timestamp_cache: Optional[Dict[str, pd.Timestamp]] = None
) -> pd.DataFrame:
    """
    Run the refresh analysis: check timestamps, fetch fresh data, and re-analyze.
    
    Returns:
        Updated DataFrame with refreshed results, or original if refresh fails/skipped
    """
    original_df = df.copy()
    
    if not POLYGON_AVAILABLE or HistoricalDataFetcher is None:
        analyzer._log("WARNING", "Warning: --refresh-results requires fetch_options module. Refresh skipped.")
        return original_df
    
    api_key = os.getenv('POLYGON_API_KEY')
    if not api_key:
        analyzer._log("WARNING", "Warning: POLYGON_API_KEY environment variable not set. Refresh skipped.")
        return original_df
    
    # Check if market is open
    now_utc = datetime.now(timezone.utc)
    is_market_open = common_is_market_hours(now_utc, "America/New_York")
    
    if not is_market_open:
        analyzer._log("INFO", "Market is closed. Skipping refresh (option prices don't change during non-market hours).")
        return original_df
    
    # Extract unique tickers from results
    result_tickers = df['ticker'].unique().tolist() if 'ticker' in df.columns else []
    
    if not result_tickers:
        analyzer._log("INFO", "No tickers found in results. Skipping refresh.")
        return original_df
    
    # Check which tickers need refresh (reuse timestamp cache if provided)
    min_write_timestamp = getattr(args, 'min_write_timestamp', None)
    tickers_to_refresh = await _check_tickers_for_refresh(
        analyzer, result_tickers, refresh_threshold_seconds, redis_client, timestamp_cache, min_write_timestamp
    )
    
    if not tickers_to_refresh:
        analyzer._log("INFO", f"\nAll tickers have fresh data (within {refresh_threshold_seconds}s threshold). No refresh needed.")
        return original_df
    
    # Calculate percentage
    refresh_percentage = (len(tickers_to_refresh) / len(result_tickers) * 100) if result_tickers else 0
    
    # Print summary to stderr so it's always visible
    print(f"\n=== Refresh Summary ===", file=sys.stderr)
    print(f"Total tickers in results: {len(result_tickers)}", file=sys.stderr)
    print(f"Tickers being refreshed: {len(tickers_to_refresh)} ({refresh_percentage:.1f}%)", file=sys.stderr)
    print(f"\nAll tickers in results: {', '.join(sorted(result_tickers))}", file=sys.stderr)
    print(f"\nTickers being refreshed: {', '.join(sorted(tickers_to_refresh))}", file=sys.stderr)
    print("", file=sys.stderr)
    
    analyzer._log("INFO", f"\n=== Refreshing options data for {len(tickers_to_refresh)} ticker(s) ===")
    analyzer._log("INFO", f"Tickers to refresh: {', '.join(tickers_to_refresh)}")
    
    try:
        # Calculate date ranges
        today_str = datetime.now().strftime('%Y-%m-%d')
        from datetime import date
        today_date = date.today()
        
        short_start_date, max_end_date, combined_max_days = _calculate_refresh_date_ranges(
            analyzer, args, today_str, today_date
        )
        
        # Set Redis flags for pending refreshes (only during market hours)
        if redis_client:
            for ticker in tickers_to_refresh:
                _set_redis_refresh_pending(redis_client, ticker, ttl_seconds=900)
        
        # Use multiprocessing with max_workers/2 processes
        enable_cache = not args.no_cache
        redis_url = os.getenv('REDIS_URL', 'redis://localhost:6379/0') if enable_cache else None
        max_workers = getattr(args, 'max_workers', 4)
        refresh_workers = max(1, max_workers // 2)  # Use half of max_workers, minimum 1
        
        analyzer._log("INFO", f"Using {refresh_workers} processes for refresh (max_workers={max_workers})")
        
        # Split tickers across processes
        tickers_per_process = max(1, len(tickers_to_refresh) // refresh_workers)
        ticker_batches = []
        for i in range(0, len(tickers_to_refresh), tickers_per_process):
            batch = tickers_to_refresh[i:i + tickers_per_process]
            if batch:
                ticker_batches.append(batch)
        
        # Ensure we don't have more batches than workers
        if len(ticker_batches) > refresh_workers:
            ticker_batches = ticker_batches[:refresh_workers]
        
        analyzer._log("INFO", f"Split {len(tickers_to_refresh)} tickers into {len(ticker_batches)} batches")
        
        # Log ticker distribution per batch (print to stderr so it's always visible)
        print(f"\nRefresh multiprocess ticker distribution ({refresh_workers} processes):", file=sys.stderr)
        for i, batch in enumerate(ticker_batches, 1):
            ticker_list = ', '.join(batch)
            print(f"  Process {i}: {len(batch)} ticker(s) - {ticker_list}", file=sys.stderr)
        print("", file=sys.stderr)  # Empty line for readability
        
        # Prepare arguments for each batch
        process_args = []
        for batch in ticker_batches:
            args_tuple = (
                batch,
                analyzer.db_conn,
                api_key,
                args.data_dir,
                today_str,
                enable_cache,
                redis_url,
                analyzer.log_level,
                analyzer.debug
            )
            process_args.append(args_tuple)
        
        # Execute in parallel using ProcessPoolExecutor
        from concurrent.futures import ProcessPoolExecutor
        loop = asyncio.get_event_loop()
        start_time = datetime.now()
        
        with ProcessPoolExecutor(max_workers=refresh_workers) as executor:
            futures = [
                loop.run_in_executor(executor, _process_refresh_batch, args)
                for args in process_args
            ]
            batch_results = await asyncio.gather(*futures, return_exceptions=True)
        
        # Flatten results from all batches
        refresh_results = []
        for i, batch_result in enumerate(batch_results):
            if isinstance(batch_result, Exception):
                # If a batch failed, mark all tickers in that batch as failed
                batch = ticker_batches[i]
                for ticker in batch:
                    refresh_results.append({'ticker': ticker, 'success': False, 'error': str(batch_result)})
            else:
                refresh_results.extend(batch_result)
        
        # Clear Redis flags for successful refreshes
        if redis_client:
            for result in refresh_results:
                if result.get('success'):
                    _clear_redis_refresh_pending(redis_client, result['ticker'])
        
        # Log progress
            elapsed = (datetime.now() - start_time).total_seconds()
        successful = sum(1 for r in refresh_results if r.get('success'))
        analyzer._log("INFO", f"\nRefresh complete: {successful}/{len(tickers_to_refresh)} tickers successful ({elapsed:.1f}s elapsed)")
        
        # Log individual results
        for result in refresh_results:
            ticker = result.get('ticker', 'unknown')
            if result.get('success'):
                contracts = result.get('contracts', 0)
                analyzer._log("INFO", f"  ✓ {ticker}: {contracts} contracts")
            else:
                error = result.get('error', 'Unknown error')
                analyzer._log("WARNING", f"  ✗ {ticker}: {error}")
        
        # Small delay to ensure database commits are visible before re-analysis
        if successful > 0:
            await asyncio.sleep(0.5)  # 500ms delay to ensure DB commits are visible
        
        analyzer._log("INFO", f"Re-analyzing options for refreshed tickers...")
        
        # Re-run analysis on refreshed tickers only
        analysis_args = _build_analysis_args(args, tickers_to_refresh, filters)
        df = await analyzer.analyze_options(**analysis_args)
        
        if df.empty:
            analyzer._log("WARNING", "Warning: No results after refresh. Using original results.")
            return original_df
        else:
            analyzer._log("INFO", f"✓ Re-analysis complete: {len(df)} options found after refresh")
            return df
    
    except Exception as e:
        analyzer._log("ERROR", f"Error during refresh: {e}")
        if args.debug:
            import traceback
            traceback.print_exc()
        analyzer._log("WARNING", "Using original analysis results due to refresh error.")
        return original_df


def _run_background_refresh_worker_subprocess(
    db_conn: str,
    tickers: List[str],
    refresh_threshold_seconds: int,
    redis_url: Optional[str],
    log_level: str,
    debug: bool,
    enable_cache: bool,
    args_dict: Dict[str, Any]
) -> None:
    """
    Worker function for subprocess-based background refresh.
    This version doesn't use multiprocessing.Queue since it's called via subprocess.
    """
    _run_background_refresh_worker_internal(
        db_conn, tickers, refresh_threshold_seconds, redis_url,
        log_level, debug, enable_cache, args_dict, status_queue=None
    )


def _run_background_refresh_worker(
    db_conn: str,
    tickers: List[str],
    refresh_threshold_seconds: int,
    redis_url: Optional[str],
    log_level: str,
    debug: bool,
    enable_cache: bool,
    args_dict: Dict[str, Any],
    status_queue: Optional[Any] = None
) -> None:
    """
    Worker function to run refresh in background process (multiprocessing version).
    """
    _run_background_refresh_worker_internal(
        db_conn, tickers, refresh_threshold_seconds, redis_url,
        log_level, debug, enable_cache, args_dict, status_queue
    )


def _run_background_refresh_worker_internal(
    db_conn: str,
    tickers: List[str],
    refresh_threshold_seconds: int,
    redis_url: Optional[str],
    log_level: str,
    debug: bool,
    enable_cache: bool,
    args_dict: Dict[str, Any],
    status_queue: Optional[Any] = None
) -> None:
    """
    Worker function to run refresh in background process.
    
    Args:
        status_queue: Optional multiprocessing.Queue to send status updates to parent process
    """
    import asyncio
    import sys
    import os
    from pathlib import Path
    
    # Immediately send status to parent if queue is provided
    if status_queue is not None:
        try:
            import multiprocessing
            worker_pid = os.getpid()
            try:
                worker_name = multiprocessing.current_process().name
            except:
                worker_name = 'unknown'
            status_queue.put({
                'pid': worker_pid,
                'name': worker_name,
                'tickers': tickers,
                'ticker_count': len(tickers)
            })
        except Exception as e:
            # If queue communication fails, continue anyway
            print(f"Warning: Failed to send status to parent: {e}", file=sys.stderr)
    
    # Re-import needed modules in the worker process
    CURRENT_DIR = Path(__file__).resolve().parent
    PROJECT_ROOT = CURRENT_DIR.parent
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    
    async def _worker():
        try:
            # Log process info
            worker_pid = os.getpid()
            try:
                import multiprocessing
                worker_name = multiprocessing.current_process().name
            except:
                worker_name = 'unknown'
            print(f"Background refresh worker started [PID {worker_pid}, Process: {worker_name}]", file=sys.stderr)
            
            # Create analyzer in worker process
            analyzer = OptionsAnalyzer(db_conn, log_level=log_level, debug=debug, enable_cache=enable_cache, redis_url=redis_url)
            await analyzer.initialize()
            
            # Get Redis client
            redis_client = None
            if redis_url and REDIS_AVAILABLE:
                redis_client = _get_redis_client_for_refresh(redis_url)
            
            # Check market hours
            from common.market_hours import is_market_hours
            now_utc = datetime.now(timezone.utc)
            is_market_open = is_market_hours(now_utc, "America/New_York")
            
            if not is_market_open:
                analyzer._log("INFO", "Background refresh: Market is closed. Skipping refresh.")
                return
            
            # Check which tickers need refresh
            min_write_timestamp = args_dict.get('min_write_timestamp', None)
            tickers_to_refresh = await _check_tickers_for_refresh(
                analyzer, tickers, refresh_threshold_seconds, redis_client, None, min_write_timestamp
            )
            
            if not tickers_to_refresh:
                analyzer._log("INFO", "Background refresh: No tickers need refresh.")
                return
            
            analyzer._log("INFO", f"Background refresh: Refreshing {len(tickers_to_refresh)} ticker(s): {', '.join(tickers_to_refresh)}")
            
            # Import fetch_options in worker
            try:
                from scripts.fetch_options import HistoricalDataFetcher
            except ImportError:
                analyzer._log("ERROR", "Background refresh: fetch_options module not available.")
                return
            
            api_key = os.getenv('POLYGON_API_KEY')
            if not api_key:
                analyzer._log("ERROR", "Background refresh: POLYGON_API_KEY not set.")
                return
            
            # Calculate date ranges
            from datetime import date
            today_str = datetime.now().strftime('%Y-%m-%d')
            today_date = date.today()
            
            short_start_date, max_end_date, combined_max_days = _calculate_refresh_date_ranges(
                analyzer, args_dict, today_str, today_date
            )
            
            # Set Redis flags
            if redis_client:
                for ticker in tickers_to_refresh:
                    _set_redis_refresh_pending(redis_client, ticker, ttl_seconds=1800)
            
            # Use multiprocessing with max_workers/2 processes
            max_workers = args_dict.get('max_workers', 4)
            refresh_workers = max(1, max_workers // 2)  # Use half of max_workers, minimum 1
            
            analyzer._log("INFO", f"Background refresh: Using {refresh_workers} processes (max_workers={max_workers})")
            
            # Split tickers across processes
            tickers_per_process = max(1, len(tickers_to_refresh) // refresh_workers)
            ticker_batches = []
            for i in range(0, len(tickers_to_refresh), tickers_per_process):
                batch = tickers_to_refresh[i:i + tickers_per_process]
                if batch:
                    ticker_batches.append(batch)
            
            # Ensure we don't have more batches than workers
            if len(ticker_batches) > refresh_workers:
                ticker_batches = ticker_batches[:refresh_workers]
            
            analyzer._log("INFO", f"Background refresh: Split {len(tickers_to_refresh)} tickers into {len(ticker_batches)} batches")
            
            # Log ticker distribution per batch (print to stderr so it's always visible)
            print(f"\nBackground refresh multiprocess ticker distribution ({refresh_workers} processes):", file=sys.stderr)
            for i, batch in enumerate(ticker_batches, 1):
                ticker_list = ', '.join(batch)
                print(f"  Process {i}: {len(batch)} ticker(s) - {ticker_list}", file=sys.stderr)
            print("", file=sys.stderr)  # Empty line for readability
            
            # Prepare arguments for each batch
            process_args = []
            for batch in ticker_batches:
                args_tuple = (
                    batch,
                    db_conn,
                    api_key,
                    args_dict.get('data_dir', './data'),
                    today_str,
                    enable_cache,
                    redis_url,
                    log_level,
                    debug
                )
                process_args.append(args_tuple)
            
            # Execute in parallel using ProcessPoolExecutor
            from concurrent.futures import ProcessPoolExecutor
            loop = asyncio.get_event_loop()
            
            with ProcessPoolExecutor(max_workers=refresh_workers) as executor:
                futures = [
                    loop.run_in_executor(executor, _process_refresh_batch, args)
                    for args in process_args
                ]
                batch_results = await asyncio.gather(*futures, return_exceptions=True)
            
            # Flatten results from all batches
            refresh_results = []
            for i, batch_result in enumerate(batch_results):
                if isinstance(batch_result, Exception):
                    # If a batch failed, mark all tickers in that batch as failed
                    batch = ticker_batches[i]
                    for ticker in batch:
                        refresh_results.append({'ticker': ticker, 'success': False, 'error': str(batch_result)})
                else:
                    refresh_results.extend(batch_result)
            
            # Clear Redis flags and log results
            for result in refresh_results:
                ticker = result.get('ticker', 'unknown')
                if result.get('success'):
                    if redis_client:
                        _clear_redis_refresh_pending(redis_client, ticker)
                    contracts = result.get('contracts', 0)
                    analyzer._log("INFO", f"Background refresh: ✓ {ticker} - {contracts} contracts")
                else:
                    if redis_client:
                        _clear_redis_refresh_pending(redis_client, ticker)
                    error = result.get('error', 'Unknown error')
                    analyzer._log("WARNING", f"Background refresh: ✗ {ticker} - {error}")
            
            analyzer._log("INFO", "Background refresh: Complete")
            
        except Exception as e:
            print(f"Background refresh worker error: {e}", file=sys.stderr)
            import traceback
            traceback.print_exc()
        finally:
            if analyzer and analyzer.db:
                await analyzer.db.close()
    
    # Run the async worker
    asyncio.run(_worker())


async def _run_background_refresh(
    analyzer: OptionsAnalyzer,
    args,
    df: pd.DataFrame,
    filters: List[FilterExpression],
    refresh_threshold_seconds: int,
    redis_client: Optional[Any],
    timestamp_cache: Optional[Dict[str, pd.Timestamp]] = None
) -> None:
    """
    Run refresh in a background process without waiting.
    Main process continues and shows existing results.
    """
    if not POLYGON_AVAILABLE or HistoricalDataFetcher is None:
        analyzer._log("WARNING", "Background refresh: fetch_options module not available. Skipping.")
        return
    
    api_key = os.getenv('POLYGON_API_KEY')
    if not api_key:
        analyzer._log("WARNING", "Background refresh: POLYGON_API_KEY not set. Skipping.")
        return
    
    # Check if market is open
    now_utc = datetime.now(timezone.utc)
    is_market_open = common_is_market_hours(now_utc, "America/New_York")
    
    if not is_market_open:
        analyzer._log("INFO", "Background refresh: Market is closed. Skipping.")
        return
    
    # Extract unique tickers from results
    result_tickers = df['ticker'].unique().tolist() if 'ticker' in df.columns else []
    
    if not result_tickers:
        analyzer._log("INFO", "Background refresh: No tickers found in results. Skipping.")
        return
    
    # Check which tickers need refresh (with Redis deduplication, reuse timestamp cache if provided)
    min_write_timestamp = getattr(args, 'min_write_timestamp', None)
    tickers_to_refresh = await _check_tickers_for_refresh(
        analyzer, result_tickers, refresh_threshold_seconds, redis_client, timestamp_cache, min_write_timestamp
    )
    
    if not tickers_to_refresh:
        analyzer._log("INFO", "Background refresh: No tickers need refresh.")
        return
    
    # Calculate percentage
    refresh_percentage = (len(tickers_to_refresh) / len(result_tickers) * 100) if result_tickers else 0
    
    # Print summary to stderr so it's always visible
    print(f"\n=== Background Refresh Summary ===", file=sys.stderr)
    print(f"Total tickers in results: {len(result_tickers)}", file=sys.stderr)
    print(f"Tickers being refreshed: {len(tickers_to_refresh)} ({refresh_percentage:.1f}%)", file=sys.stderr)
    print(f"\nAll tickers in results: {', '.join(sorted(result_tickers))}", file=sys.stderr)
    print(f"\nTickers being refreshed: {', '.join(sorted(tickers_to_refresh))}", file=sys.stderr)
    print("", file=sys.stderr)
    
    analyzer._log("WARNING", f"Background refresh initiated for {len(tickers_to_refresh)} ticker(s): {', '.join(tickers_to_refresh)}")
    analyzer._log("INFO", f"Starting background refresh for {len(tickers_to_refresh)} ticker(s): {', '.join(tickers_to_refresh)}")
    
    # Calculate refresh workers and ticker distribution in main process for visibility
    max_workers = getattr(args, 'max_workers', 4)
    refresh_workers = max(1, max_workers // 2)  # Use half of max_workers, minimum 1
    
    # Split tickers across processes to show distribution
    tickers_per_process = max(1, len(tickers_to_refresh) // refresh_workers)
    ticker_batches = []
    for i in range(0, len(tickers_to_refresh), tickers_per_process):
        batch = tickers_to_refresh[i:i + tickers_per_process]
        if batch:
            ticker_batches.append(batch)
    
    # Ensure we don't have more batches than workers
    if len(ticker_batches) > refresh_workers:
        ticker_batches = ticker_batches[:refresh_workers]
    
    # Print ticker distribution to stderr so it's always visible
    print(f"\nBackground refresh multiprocess ticker distribution ({refresh_workers} processes):", file=sys.stderr)
    for i, batch in enumerate(ticker_batches, 1):
        ticker_list = ', '.join(batch)
        print(f"  Process {i}: {len(batch)} ticker(s) - {ticker_list}", file=sys.stderr)
    print("", file=sys.stderr)  # Empty line for readability
    
    # Prepare arguments for worker process
    args_dict = {
        'data_dir': getattr(args, 'data_dir', './data'),
        'spread': getattr(args, 'spread', False),
        'start_date': getattr(args, 'start_date', None),
        'end_date': getattr(args, 'end_date', None),
        'max_days': getattr(args, 'max_days', None),
        'spread_long_days': getattr(args, 'spread_long_days', 90),
        'spread_long_days_tolerance': getattr(args, 'spread_long_days_tolerance', 10),
        'spread_long_min_days': getattr(args, 'spread_long_min_days', None),
        'min_write_timestamp': min_write_timestamp,
        'max_workers': max_workers,
    }
    
    # Spawn background process using subprocess so it can survive parent exit
    try:
        import subprocess
        import json
        import tempfile
        
        # Create a temporary file to pass arguments (since subprocess needs serializable args)
        # We'll pass the arguments as JSON in environment variables and a temp file
        temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False)
        worker_args = {
            'db_conn': analyzer.db_conn,
            'tickers': tickers_to_refresh,
            'refresh_threshold_seconds': refresh_threshold_seconds,
            'redis_url': analyzer.redis_url,
            'log_level': analyzer.log_level,
            'debug': analyzer.debug,
            'enable_cache': analyzer.enable_cache,
            'args_dict': args_dict
        }
        json.dump(worker_args, temp_file)
        temp_file.close()
        args_file = temp_file.name
        
        # Get the script directory for proper path resolution
        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(script_dir)
        
        # Create a wrapper script that will run the worker
        # We need to import the module and call the worker function
        script_content = f"""
import sys
import os
import json
import asyncio
from pathlib import Path

# Add project to path
PROJECT_ROOT = r'{project_root}'
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Import the worker function
from scripts.options_analyzer import _run_background_refresh_worker_subprocess

# Read arguments from file
with open(r'{args_file}', 'r') as f:
    args = json.load(f)

# Run the worker
_run_background_refresh_worker_subprocess(**args)
"""
        
        # Write wrapper script to temp file
        script_file = tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False)
        script_file.write(script_content)
        script_file.close()
        script_path = script_file.name
        
        # Make script executable
        os.chmod(script_path, 0o755)
        
        # Start subprocess with proper detachment
        # Use start_new_session=True to create a new process group
        # Redirect stdout/stderr to files so we can see output
        log_file = tempfile.NamedTemporaryFile(mode='w', suffix='.log', delete=False)
        log_file.close()
        
        process = subprocess.Popen(
            [sys.executable, script_path],
            stdout=open(log_file.name, 'w'),
            stderr=subprocess.STDOUT,
            start_new_session=True,  # Create new session so process survives parent exit
            cwd=os.getcwd()
        )
        
        # Give it a moment to start and send initial status
        import time
        time.sleep(0.5)
        
        # Check if process is still running
        if process.poll() is None:
            print(f"\nBackground refresh process started and running:", file=sys.stderr)
            print(f"  Worker PID: {process.pid}", file=sys.stderr)
            print(f"  Processing {len(tickers_to_refresh)} ticker(s)", file=sys.stderr)
            print(f"  Main process PID: {os.getpid()}", file=sys.stderr)
            print(f"  Tickers: {', '.join(tickers_to_refresh[:10])}{'...' if len(tickers_to_refresh) > 10 else ''}", file=sys.stderr)
            print(f"  Log file: {log_file.name}", file=sys.stderr)
            print(f"\nTo check background processes: ps auxww | grep -E 'options_analyzer|{process.pid}'", file=sys.stderr)
            print(f"To view logs: tail -f {log_file.name}", file=sys.stderr)
            print("", file=sys.stderr)
            analyzer._log("INFO", f"Background refresh process started (PID: {process.pid}). Main process continuing with existing results.")
        else:
            # Process exited immediately, check the log
            print(f"\nBackground refresh process exited immediately (exit code: {process.returncode})", file=sys.stderr)
            print(f"Check log file for details: {log_file.name}", file=sys.stderr)
            try:
                with open(log_file.name, 'r') as f:
                    log_content = f.read()
                    if log_content:
                        print(f"Log content:\n{log_content}", file=sys.stderr)
            except:
                pass
            analyzer._log("WARNING", f"Background refresh process exited immediately (PID: {process.pid}, exit code: {process.returncode})")
    except Exception as e:
        print(f"ERROR: Failed to start background refresh process: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc(file=sys.stderr)
        analyzer._log("ERROR", f"Failed to start background refresh process: {e}")


def _print_statistics(analyzer: OptionsAnalyzer, args):
    """Print multiprocess and cache statistics."""
    # Print multiprocess statistics if using multiprocessing
    if args.max_workers > 1 and hasattr(analyzer.db, 'print_process_statistics'):
        # Use log_level to determine if we should print (INFO or lower)
        quiet = not analyzer._should_log("INFO")
        analyzer.db.print_process_statistics(quiet=quiet)
    
    # Print cache statistics (always print at INFO level or lower)
    if analyzer._should_log("INFO") and hasattr(analyzer.db, 'get_cache_statistics'):
        cache_stats = analyzer.db.get_cache_statistics()
        print("\n=== Cache Statistics ===", file=sys.stderr)
        if cache_stats.get('enabled', False):
            print(f"Cache Status: ENABLED", file=sys.stderr)
            print(f"Total Requests: {cache_stats.get('total_requests', 0)}", file=sys.stderr)
            print(f"Cache Hits: {cache_stats.get('hits', 0)}", file=sys.stderr)
            print(f"Cache Misses: {cache_stats.get('misses', 0)}", file=sys.stderr)
            hit_rate = cache_stats.get('hit_rate', 0.0)
            print(f"Hit Rate: {hit_rate:.2%}", file=sys.stderr)
            negative_hits = cache_stats.get('negative_hits', 0)
            negative_sets = cache_stats.get('negative_sets', 0)
            print(f"Negative Cache Hits: {negative_hits}", file=sys.stderr)
            print(f"Negative Cache Sets: {negative_sets}", file=sys.stderr)
            print(f"Cache Sets: {cache_stats.get('sets', 0)}", file=sys.stderr)
            print(f"Cache Invalidations: {cache_stats.get('invalidations', 0)}", file=sys.stderr)
            print(f"Cache Errors: {cache_stats.get('errors', 0)}", file=sys.stderr)
        else:
            print(f"Cache Status: DISABLED", file=sys.stderr)
        # Database query statistics (if available)
        db_query_count = cache_stats.get('db_query_count')
        if db_query_count is not None:
            print(f"\n=== Database Query Statistics ===", file=sys.stderr)
            print(f"Total Database Queries: {db_query_count}", file=sys.stderr)
            print("===================================\n", file=sys.stderr)
        else:
            print("===================================\n", file=sys.stderr)


def _parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments for the options analyzer."""
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
        '--batch-size',
        type=int,
        default=300,
        help="Number of tickers per batch when fetching options in multiprocessing mode (default: 300). Lower uses less memory."
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
        default=100000.0,
        help="Position size in dollars for premium calculations (default: 100000.0 = $100K)."
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
        help="Disable Redis caching for QuestDB operations (default: cache enabled)"
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
             "Supported operators: > >= < <= == != exists not_exists. "
             "Mathematical operations: + - * / (e.g. 'num_contracts*0.1 > volume' 'potential_premium+1000 > daily_premium'). "
             "Field-to-field comparisons: 'num_contracts > volume' or 'potential_premium > daily_premium'. "
             "Market cap values support T (trillion) B (billion) and M (million) suffixes (e.g. 'market_cap < 3.5T'). "
             "Multiple expressions in one --filter can be comma-separated. "
             "STANDARD FIELDS (always available): "
             "Financial: pe_ratio (float), market_cap (float, supports T/B/M suffixes). "
             "Pricing: current_price (float), strike_price (float), price_above_current (float), option_premium (float). "
             "Derived percentages: option_premium_percentage (float, = option_premium/current_price*100), "
             "premium_above_diff_percentage (float, = (option_premium-price_above_current)/price_above_current*100). "
             "Option Greeks: delta (float), theta (float), implied_volatility (float). "
             "Volume/Contracts: volume (int), num_contracts (int). "
             "Premium calculations: potential_premium (float, = num_contracts*option_premium*100), "
             "daily_premium (float, = potential_premium/days_to_expiry). "
             "Time: days_to_expiry (int). "
             "SPREAD MODE FIELDS (only when --spread is enabled): "
             "Long option details: long_strike_price (float), long_option_premium (float), long_days_to_expiry (int), "
             "long_delta (float), long_theta (float), long_implied_volatility (float), long_expiration_date (str), "
             "long_option_ticker (str), long_volume (int), long_contracts_available (int, open interest). "
             "Spread calculations: premium_diff (float, = long_premium - short_premium per contract), "
             "short_premium_total (float, = num_contracts*short_premium*100), "
             "short_daily_premium (float, = short_premium_total/short_days_to_expiry), "
             "long_premium_total (float, = num_contracts*long_premium*100), "
             "net_premium (float, = short_premium_total - (long_premium_total - estimated_long_premium_at_short_expiry_total), "
             "uses Black-Scholes to estimate long option value at short expiration), "
             "net_daily_premium (float, = net_premium/short_days_to_expiry). "
             "Note: In spread mode, num_contracts = floor(position_size / (long_premium * 100)). "
             "Examples: 'pe_ratio > 20', 'market_cap < 3.5T', 'num_contracts > volume', 'num_contracts*0.1 > volume', "
             "'potential_premium > daily_premium', 'volume exists', 'option_premium_percentage >= 10', "
             "'premium_above_diff_percentage > 0', 'net_daily_premium > 100', 'net_premium > 1000'."
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
        '--log-level',
        type=str,
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        default='INFO',
        help="Set logging level: DEBUG (most verbose), INFO (default), WARNING, ERROR (least verbose)."
    )
    parser.add_argument(
        '--debug',
        action='store_true',
        help="Enable debug output with detailed information about data fetching and matching."
    )
    parser.add_argument(
        '--top-n',
        type=int,
        default=1,
        help="Limit results to top N options per ticker (based on sort order). Example: --top-n 10 shows only the best 10 options for each ticker. Applied after sorting and filtering. Default: 1."
    )
    parser.add_argument(
        '--refresh-results',
        type=int,
        nargs='?',
        const=300,
        default=None,
        help="If market is open, refresh options data for tickers in results and re-analyze if the most recent write_timestamp is older than the specified threshold (in seconds). Default: 300 seconds. Requires POLYGON_API_KEY environment variable. Example: --refresh-results 300 or --refresh-results (uses default 300)."
    )
    parser.add_argument(
        '--refresh-results-background',
        type=int,
        nargs='?',
        const=300,
        default=None,
        help="Like --refresh-results, but runs refresh in a background process without waiting. Main process shows existing analysis results immediately. Requires POLYGON_API_KEY and Redis (for deduplication). Example: --refresh-results-background 300"
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
    if args.debug:
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
    
    return args


def _build_analysis_args(args: argparse.Namespace, symbols_list: List[str], filters: List) -> dict:
    """Build arguments dictionary for analyze_options from parsed args."""
    return {
        'tickers': symbols_list,
        'days_to_expiry': args.days,
        'min_volume': args.min_volume,
        'max_days': args.max_days,
        'min_premium': args.min_premium,
        'position_size': args.position_size,
        'filters': filters,
        'filter_logic': args.filter_logic,
        'use_market_time': not args.no_market_time,
        'start_date': args.start_date,
        'end_date': args.end_date,
        'max_concurrent': getattr(args, 'max_concurrent', 10),  # Default to 10 if not specified
        'batch_size': args.batch_size,
        'timestamp_lookback_days': args.timestamp_lookback_days,
        'max_workers': args.max_workers,
        'spread_mode': args.spread,
        'spread_strike_tolerance': args.spread_strike_tolerance,
        'spread_long_days': args.spread_long_days,
        'spread_long_days_tolerance': args.spread_long_days_tolerance,
        'spread_long_min_days': args.spread_long_min_days,
        'min_write_timestamp': args.min_write_timestamp
    }


def _print_statistics(analyzer: 'OptionsAnalyzer', args: argparse.Namespace) -> None:
    """Print multiprocess and cache statistics."""
    # Print multiprocess statistics if using multiprocessing
    if args.max_workers > 1 and hasattr(analyzer.db, 'print_process_statistics'):
        quiet = not analyzer._should_log("INFO")
        analyzer.db.print_process_statistics(quiet=quiet)
    
    # Print cache statistics (always print at INFO level or lower)
    if analyzer._should_log("INFO") and hasattr(analyzer.db, 'get_cache_statistics'):
        cache_stats = analyzer.db.get_cache_statistics()
        print("\n=== Cache Statistics ===", file=sys.stderr)
        if cache_stats.get('enabled', False):
            print(f"Cache Status: ENABLED", file=sys.stderr)
            print(f"Total Requests: {cache_stats.get('total_requests', 0)}", file=sys.stderr)
            print(f"Cache Hits: {cache_stats.get('hits', 0)}", file=sys.stderr)
            print(f"Cache Misses: {cache_stats.get('misses', 0)}", file=sys.stderr)
            hit_rate = cache_stats.get('hit_rate', 0.0)
            print(f"Hit Rate: {hit_rate:.2%}", file=sys.stderr)
            negative_hits = cache_stats.get('negative_hits', 0)
            negative_sets = cache_stats.get('negative_sets', 0)
            print(f"Negative Cache Hits: {negative_hits}", file=sys.stderr)
            print(f"Negative Cache Sets: {negative_sets}", file=sys.stderr)
            print(f"Cache Sets: {cache_stats.get('sets', 0)}", file=sys.stderr)
            print(f"Cache Invalidations: {cache_stats.get('invalidations', 0)}", file=sys.stderr)
            print(f"Cache Errors: {cache_stats.get('errors', 0)}", file=sys.stderr)
        else:
            print(f"Cache Status: DISABLED", file=sys.stderr)
        # Database query statistics (if available)
        db_query_count = cache_stats.get('db_query_count')
        if db_query_count is not None:
            print(f"\n=== Database Query Statistics ===", file=sys.stderr)
            print(f"Total Database Queries: {db_query_count}", file=sys.stderr)
            print("===================================\n", file=sys.stderr)
        else:
            print("===================================\n", file=sys.stderr)
    
    # Also check for args.stats flag (legacy support)
    if args.stats and hasattr(analyzer.db, 'get_cache_stats'):
        stats = analyzer.db.get_cache_stats()
        if stats:
            print("\n===================================", file=sys.stderr)
            print("Cache Statistics:", file=sys.stderr)
            print("===================================", file=sys.stderr)
            for key, value in stats.items():
                print(f"{key}: {value}", file=sys.stderr)
            print("===================================\n", file=sys.stderr)


async def main():
    """Main function to run the options analyzer."""
    args = _parse_arguments()
    
    # Initialize analyzer
    enable_cache = not args.no_cache
    redis_url = os.getenv('REDIS_URL', 'redis://localhost:6379/0') if enable_cache else None
    log_level = args.log_level if hasattr(args, 'log_level') else ("DEBUG" if args.debug else "INFO")
    
    analyzer = OptionsAnalyzer(args.db_conn, log_level=log_level, debug=args.debug, enable_cache=enable_cache, redis_url=redis_url)
    await analyzer.initialize()
    
    # Use async context manager to ensure close() is always called
    async with analyzer.db:
        # Get symbols list using common library
        quiet = not analyzer._should_log("INFO")
        symbols_list = await fetch_lists_data(args, quiet)
        if not symbols_list:
            print("No symbols specified or found. Exiting.", file=sys.stderr)
            sys.exit(1)
        
        analyzer._log("INFO", f"Analyzing {len(symbols_list)} tickers...")
        
        if args.debug:
            print(f"DEBUG: Symbols list: {symbols_list[:10]}{'...' if len(symbols_list) > 10 else ''}", file=sys.stderr)
        
        # Get financial information
        financial_data = await analyzer.get_financial_info(symbols_list)
        
        # Parse filters
        filters = []
        if hasattr(args, 'filter') and args.filter:
            try:
                normalized_filters = [' '.join(f.split()) for f in args.filter]
                filters = FilterParser.parse_filters(normalized_filters)
                if filters:
                    analyzer._log("INFO", f"Applied {len(filters)} filter(s) with {args.filter_logic} logic:")
                    for i, f in enumerate(filters, 1):
                        analyzer._log("INFO", f"  {i}. {f}")
            except Exception as e:
                print(f"Error parsing filters: {e}", file=sys.stderr)
                sys.exit(1)
        
        # Analyze options
        analysis_args = _build_analysis_args(args, symbols_list, filters)
        df = await analyzer.analyze_options(**analysis_args)
        
        if df.empty:
            analyzer._log("INFO", "DataFrame is empty after analysis. Check debug output above for details.")
            print("No options data found matching the criteria.")
            return
        
        # Get Redis client for refresh deduplication (if available)
        redis_url = os.getenv('REDIS_URL', 'redis://localhost:6379/0') if enable_cache else None
        redis_client = None
        if redis_url and REDIS_AVAILABLE:
            redis_client = _get_redis_client_for_refresh(redis_url)
        
        # Use the analyzer's instance-level timestamp cache
        timestamp_cache = analyzer._timestamp_cache
        
        # Refresh results if requested and market is open
        if args.refresh_results is not None:
            refresh_threshold_seconds = args.refresh_results
            df = await _run_refresh_analysis(
                analyzer, args, df, filters, refresh_threshold_seconds, redis_client, timestamp_cache
            )
        elif args.refresh_results_background is not None:
            refresh_threshold_seconds = args.refresh_results_background
            await _run_background_refresh(
                analyzer, args, df, filters, refresh_threshold_seconds, redis_client, timestamp_cache
            )
        
        # Print statistics
        _print_statistics(analyzer, args)
        
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
        
        # Normalize sort input
        import re as _re
        sort_arg = _re.sub(r"\s+", "", args.sort) if hasattr(args, 'sort') and args.sort else None
        
        # If in spread mode and user didn't specify a sort, default to net_daily_premium
        if args.spread and args.sort == 'daily_premium':
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
        
        if analyzer._should_log("INFO") or output_file is None:
            print(result)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nOperation cancelled by user.")
    except Exception as e:
        print(f"\nAn unexpected error occurred: {e}", file=sys.stderr)
        sys.exit(1)

