"""
Data formatting utilities for HTML report generation.
"""

import pandas as pd
import re
from datetime import datetime


def format_age_seconds(age_seconds):
    """Format age in seconds to human-readable format.
    
    Args:
        age_seconds: Age in seconds (float or numeric string)
        
    Returns:
        Human-readable string like "5 secs", "2 mins", "3 hrs", "1.2 days"
    """
    if pd.isna(age_seconds) or age_seconds == '' or age_seconds is None:
        return ''
    
    try:
        # Convert to float
        if isinstance(age_seconds, str):
            # Try to extract first valid number from string
            match = re.search(r'-?\d+\.?\d*', age_seconds)
            if match:
                age_sec = float(match.group())
            else:
                return str(age_seconds)
        else:
            age_sec = float(age_seconds)
        
        # Handle negative or zero values
        if age_sec < 0:
            return 'N/A'
        if age_sec == 0:
            return '0 secs'
        
        # Convert to appropriate unit
        if age_sec < 60:
            # Less than 1 minute - show seconds
            return f"{age_sec:.1f} secs" if age_sec < 10 else f"{int(age_sec)} secs"
        elif age_sec < 3600:
            # Less than 1 hour - show minutes
            mins = age_sec / 60
            return f"{mins:.1f} mins" if mins < 10 else f"{int(mins)} mins"
        elif age_sec < 86400:
            # Less than 1 day - show hours
            hrs = age_sec / 3600
            return f"{hrs:.1f} hrs" if hrs < 10 else f"{int(hrs)} hrs"
        else:
            # 1 day or more - show days
            days = age_sec / 86400
            return f"{days:.1f} days" if days < 10 else f"{int(days)} days"
    except (ValueError, TypeError, AttributeError):
        return str(age_seconds) if age_seconds != '' else ''


def format_numeric_value(x, col_name):
    """Format numeric value based on column type.
    
    Args:
        x: Value to format
        col_name: Name of the column
        
    Returns:
        Formatted string value
    """
    if pd.isna(x) or x == '' or x is None:
        return ''
    try:
        # Handle malformed strings like '0.120.260.110.210.36'
        val_str = str(x)
        # Try direct conversion first
        try:
            val = float(val_str)
        except (ValueError, TypeError):
            # Extract first valid number from string if direct conversion fails
            match = re.search(r'-?\d+\.?\d*', val_str)
            if match:
                val = float(match.group())
            else:
                # If no number found, return the string as-is
                return str(x) if x != '' else ''
        
        col_lower = col_name.lower()
        if 'premium' in col_lower or 'price' in col_lower or 'cap' in col_lower:
            return f"${val:,.2f}"
        elif 'ratio' in col_lower or 'delta' in col_lower or 'theta' in col_lower:
            return f"{val:.2f}"
        elif 'days' in col_lower or 'volume' in col_lower or 'contracts' in col_lower or 'options' in col_lower or 'purchase' in col_lower:
            return f"{int(val):,}"
        elif 'percentage' in col_lower:
            return f"{val:.2f}%"
        else:
            return f"{val:.2f}"
    except (ValueError, TypeError, AttributeError):
        return str(x) if x != '' else ''


def truncate_header(text, max_length=15):
    """Wrap header text so each line is at most max_length characters.
    
    Args:
        text: Header text to wrap
        max_length: Maximum length per line
        
    Returns:
        Wrapped text with <br> tags for line breaks
    """
    import textwrap
    text = text.replace("_", " ")
    wrapped_lines = textwrap.wrap(
        text,
        width=max_length,
        break_long_words=True,
        break_on_hyphens=False
    )
    return "<br>".join(wrapped_lines) if wrapped_lines else text


def normalize_col_name(name: str) -> str:
    """Normalize column name for consistent matching.
    
    Args:
        name: Column name to normalize
        
    Returns:
        Normalized column name (lowercase, underscores, no spaces)
    """
    return str(name).strip().lower().replace(' ', '_')


def extract_numeric_value(value):
    """Extract numeric value from potentially malformed string.
    
    Args:
        value: Value that might be a number, string, or malformed string
        
    Returns:
        Float value or None if extraction fails
    """
    if pd.isna(value) or value == '' or value is None:
        return None
    
    try:
        if isinstance(value, (int, float)):
            return float(value)
        
        val_str = str(value)
        # Try direct conversion first
        try:
            return float(val_str)
        except (ValueError, TypeError):
            # Extract first valid number from string
            match = re.search(r'-?\d+\.?\d*', val_str)
            if match:
                return float(match.group())
            return None
    except (ValueError, TypeError, AttributeError):
        return None

