"""
Filter system for options analysis - parses and applies filter expressions to DataFrames.
"""

import re
import pandas as pd
from typing import Dict, List, Any, Optional, Union


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
        # Look for mathematical operators
        math_pattern = r'[\+\-\*/]'
        return bool(re.search(math_pattern, expression))
    
    @classmethod
    def _extract_primary_field(cls, expression: str) -> str:
        """Extract the primary field name from a mathematical expression."""
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

