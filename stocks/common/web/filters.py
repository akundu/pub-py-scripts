"""
DataFrame filtering utilities for parsing and applying filter expressions.
"""

import pandas as pd
import re
import logging
from typing import List, Dict, Any

logger = logging.getLogger("db_server_logger")


def parse_filter_strings(filter_str: str) -> List[Dict[str, Any]]:
    """
    Parse pipe-separated filter strings into filter objects.
    
    Supports:
    - Comparison operators: >, <, >=, <=, ==, !=
    - Existence checks: exists, not_exists
    - Field-to-field comparisons
    - Mathematical expressions
    - Percentage-based filters
    
    Args:
        filter_str: Pipe-separated filter expressions (e.g., "delta < 0.35|l_delta > 0.48|spread < 20%")
        
    Returns:
        List of filter dictionaries compatible with apply_filters
        
    Examples:
        >>> filters = parse_filter_strings("price > 100|volume < 1000000")
        >>> len(filters)
        2
        >>> filters[0]['field']
        'price'
        >>> filters[0]['operator']
        '>'
        >>> filters[0]['value']
        100
    """
    filters = []
    if not filter_str:
        return filters
    
    # Split by pipe
    filter_expressions = filter_str.split('|')
    
    for expr in filter_expressions:
        expr = expr.strip()
        if not expr:
            continue
        
        # Debug: log the original expression
        logger.debug(f"Parsing filter expression: '{expr}'")
        
        # Handle exists/not_exists
        exists_match = re.match(r'^([a-zA-Z_][a-zA-Z0-9_]*)\s+(exists|not_exists)$', expr, re.IGNORECASE)
        if exists_match:
            filters.append({
                'field': exists_match.group(1),
                'operator': exists_match.group(2).lower(),
                'value': None,
                'isFieldComparison': False
            })
            continue
        
        # Parse comparison operators (check longer ones first)
        operators = ['>=', '<=', '==', '!=', '>', '<']
        for op in operators:
            if op in expr:
                parts = expr.split(op, 1)
                if len(parts) == 2:
                    field_expr = parts[0].strip()
                    value_str = parts[1].strip()
                    
                    # Check for mathematical expressions in field
                    has_math = bool(re.search(r'[+\-*/]', field_expr))
                    
                    # Check if value is a field name (field-to-field comparison)
                    # We'll check this later when we have the DataFrame
                    is_field_comparison = False
                    value = value_str
                    
                    # Try to parse value as number
                    if not has_math:
                        # Check if value is a date string (YYYY-MM-DD)
                        date_match = re.match(r'^(\d{4}-\d{2}-\d{2})', value_str)
                        if date_match:
                            value = value_str
                        else:
                            # Try parsing as number - handle negative numbers correctly
                            # Remove any extra whitespace that might separate the minus sign
                            # This handles cases like " -0.1" (space before minus) from URL decoding
                            value_str_clean = value_str.replace(' ', '').replace('\t', '')
                            try:
                                num_value = float(value_str_clean)
                                value = num_value
                                logger.debug(f"Parsed numeric value: {num_value} from '{value_str}' (cleaned: '{value_str_clean}')")
                            except ValueError:
                                # If that fails, try the original string (in case of special formats)
                                try:
                                    num_value = float(value_str)
                                    value = num_value
                                    logger.debug(f"Parsed numeric value: {num_value} from '{value_str}' (original)")
                                except ValueError:
                                    # Keep as string
                                    value = value_str
                                    logger.debug(f"Keeping as string: '{value_str}'")
                    
                    filter_obj = {
                        'field': field_expr,
                        'operator': op,
                        'value': value,
                        'valueStr': value_str,  # Preserve original for percentage detection
                        'isFieldComparison': is_field_comparison,
                        'hasMath': has_math
                    }
                    logger.debug(f"Created filter: {field_expr} {op} {value} (type: {type(value).__name__})")
                    filters.append(filter_obj)
                    break
        
    return filters


def apply_filters(df: pd.DataFrame, filters: List[Dict[str, Any]], filter_logic: str = 'AND') -> pd.DataFrame:
    """
    Apply filters to DataFrame.
    
    Supports:
    - Standard comparison operators: >, <, >=, <=, ==, !=
    - Existence checks: exists, not_exists
    - Field-to-field comparisons
    - Mathematical expressions in field names
    - Percentage-based filtering for spread columns
    
    Args:
        df: DataFrame to filter
        filters: List of filter dictionaries with keys: field, operator, value
        filter_logic: 'AND' or 'OR' logic for combining filters
        
    Returns:
        Filtered DataFrame
        
    Examples:
        >>> df = pd.DataFrame({'price': [100, 150, 200], 'volume': [1000, 2000, 3000]})
        >>> filters = [{'field': 'price', 'operator': '>', 'value': 120}]
        >>> result = apply_filters(df, filters)
        >>> len(result)
        2
        >>> all(result['price'] > 120)
        True
    """
    if not filters:
        return df
    
    # Evaluate each filter separately
    filter_masks = []
    for filter_obj in filters:
        field = filter_obj.get('field', '')
        operator = filter_obj.get('operator', '==')
        value = filter_obj.get('value')
        is_field_comparison = filter_obj.get('isFieldComparison', False)
        has_math = filter_obj.get('hasMath', False)
        
        if not field:
            continue
        
        filter_mask = pd.Series([True] * len(df), index=df.index)
        
        # Handle exists/not_exists
        if operator.lower() in ('exists', 'not_exists'):
            if operator.lower() == 'exists':
                filter_mask = df[field].notna() if field in df.columns else pd.Series([False] * len(df), index=df.index)
            else:
                filter_mask = df[field].isna() if field in df.columns else pd.Series([False] * len(df), index=df.index)
            filter_masks.append(filter_mask)
            continue
        
        # Check if field exists
        if field not in df.columns:
            # Try to find similar column (case-insensitive)
            field_lower = field.lower()
            matching_cols = [col for col in df.columns if col.lower() == field_lower]
            if not matching_cols:
                # Field doesn't exist, skip this filter
                continue
            field = matching_cols[0]
        
        # Check if value is actually a field name (field-to-field comparison)
        # This needs to be checked after we know the DataFrame columns
        if not is_field_comparison and isinstance(value, str) and not has_math:
            # Check if value string matches a column name (case-insensitive)
            value_lower = value.lower()
            matching_value_cols = [col for col in df.columns if col.lower() == value_lower]
            if matching_value_cols:
                is_field_comparison = True
                value = matching_value_cols[0]
        
        # Handle field-to-field comparison
        if is_field_comparison:
            if value not in df.columns:
                # Try to find similar column
                value_lower = value.lower()
                matching_cols = [col for col in df.columns if col.lower() == value_lower]
                if not matching_cols:
                    continue
                value_col = matching_cols[0]
            else:
                value_col = value
            
            if operator == '>':
                filter_mask = (df[field] > df[value_col])
            elif operator == '<':
                filter_mask = (df[field] < df[value_col])
            elif operator == '>=':
                filter_mask = (df[field] >= df[value_col])
            elif operator == '<=':
                filter_mask = (df[field] <= df[value_col])
            elif operator == '==':
                filter_mask = (df[field] == df[value_col])
            elif operator == '!=':
                filter_mask = (df[field] != df[value_col])
            filter_masks.append(filter_mask)
            continue
        
        # Handle math expressions (e.g., "curr_price*1.05")
        if has_math:
            try:
                # Evaluate math expression for each row
                field_values = []
                for idx, row in df.iterrows():
                    expr = field
                    # Replace field names with values
                    for col in df.columns:
                        if col in expr:
                            expr = expr.replace(col, str(row[col]) if pd.notna(row[col]) else '0')
                    try:
                        result = eval(expr)
                        field_values.append(float(result) if pd.notna(result) else None)
                    except:
                        field_values.append(None)
                
                field_series = pd.Series(field_values, index=df.index)
                
                # Apply operator
                if operator == '>':
                    filter_mask = (field_series > value)
                elif operator == '<':
                    filter_mask = (field_series < value)
                elif operator == '>=':
                    filter_mask = (field_series >= value)
                elif operator == '<=':
                    filter_mask = (field_series <= value)
                elif operator == '==':
                    filter_mask = (field_series == value)
                elif operator == '!=':
                    filter_mask = (field_series != value)
                filter_masks.append(filter_mask)
            except Exception as e:
                logger.warning(f"Error evaluating math expression '{field}': {e}")
                continue
        else:
            # Regular field comparison
            # Handle percentage-based filtering for spread fields
            value_to_compare = value
            value_str = filter_obj.get('valueStr', str(value) if value is not None else '')
            
            # Check if this is a percentage-based filter (e.g., "spread < 20%")
            if isinstance(value_str, str) and value_str.endswith('%'):
                field_lower = field.lower()
                if field_lower in ('spread', 'l_spread', 'long_spread'):
                    try:
                        percent_value = float(value_str[:-1])
                        # Find the option premium column
                        prem_col = None
                        if field_lower == 'spread':
                            # Try to find short option premium column
                            for col in ['opt_prem.', 'opt_prem', 'option_premium', 'premium']:
                                if col in df.columns:
                                    prem_col = col
                                    break
                        elif field_lower in ('l_spread', 'long_spread'):
                            # Try to find long option premium column
                            for col in ['l_prem', 'l_opt_prem', 'long_option_premium', 'long_premium']:
                                if col in df.columns:
                                    prem_col = col
                                    break
                        
                        if prem_col and prem_col in df.columns:
                            # Calculate percentage of premium for each row
                            value_to_compare = (percent_value / 100.0) * df[prem_col]
                        else:
                            # Can't calculate percentage without premium column, skip filter
                            logger.warning(f"Cannot apply percentage filter '{field} {operator} {value_str}': premium column not found")
                            continue
                    except (ValueError, TypeError):
                        # Invalid percentage value, use original value
                        pass
            
            if operator == '>':
                filter_mask = (df[field] > value_to_compare)
            elif operator == '<':
                filter_mask = (df[field] < value_to_compare)
                # Debug logging for negative number comparisons
                if isinstance(value_to_compare, (int, float)) and value_to_compare < 0:
                    logger.debug(f"Applying filter: {field} < {value_to_compare} (negative value)")
                    logger.debug(f"Sample values: {df[field].head(10).tolist()}")
                    logger.debug(f"Filter mask (first 10): {filter_mask.head(10).tolist()}")
                    logger.debug(f"Rows matching filter: {filter_mask.sum()} out of {len(df)}")
            elif operator == '>=':
                filter_mask = (df[field] >= value_to_compare)
            elif operator == '<=':
                filter_mask = (df[field] <= value_to_compare)
            elif operator == '==':
                filter_mask = (df[field] == value_to_compare)
            elif operator == '!=':
                filter_mask = (df[field] != value_to_compare)
            filter_masks.append(filter_mask)
    
    # Combine filter masks based on logic
    if not filter_masks:
        return df
    
    if filter_logic == 'OR':
        # OR logic: any filter matches
        combined_mask = filter_masks[0]
        for fm in filter_masks[1:]:
            combined_mask = combined_mask | fm
    else:
        # AND logic: all filters must match (default)
        combined_mask = filter_masks[0]
        for fm in filter_masks[1:]:
            combined_mask = combined_mask & fm
    
    return df[combined_mask].copy()

