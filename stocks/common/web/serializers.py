"""
Data serialization utilities for converting DataFrames and datetime objects to JSON-serializable formats.
"""

import pandas as pd
from datetime import datetime
from typing import Dict, Any, List

# Try to import numpy for type handling
try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    np = None


def dataframe_to_json_records(df: pd.DataFrame) -> List[Dict[str, Any]]:
    """
    Convert a pandas DataFrame to JSON-serializable records with ISO timestamps.
    
    Args:
        df: Pandas DataFrame to convert
        
    Returns:
        List of dictionaries with JSON-serializable values
        
    Example:
        >>> df = pd.DataFrame({'date': pd.date_range('2024-01-01', periods=2), 'price': [100, 101]})
        >>> records = dataframe_to_json_records(df)
        >>> print(records[0]['date'])  # '2024-01-01T00:00:00.000000'
    """
    if df is None or df.empty:
        return []
    
    df_serializable = df.copy()
    
    # Convert datetime columns to ISO strings
    datetime_columns = df_serializable.select_dtypes(include=['datetime64[ns]', 'datetime64[ns, UTC]']).columns
    for col in datetime_columns:
        df_serializable[col] = df_serializable[col].dt.strftime('%Y-%m-%dT%H:%M:%S.%fZ')
    
    # Handle object columns that may contain Timestamp/datetime objects
    object_columns = df_serializable.select_dtypes(include=['object']).columns
    for col in object_columns:
        series = df_serializable[col].dropna()
        if series.empty:
            continue
        sample_val = series.iloc[0]
        if isinstance(sample_val, (pd.Timestamp, datetime)):
            df_serializable[col] = df_serializable[col].apply(
                lambda x: x.isoformat() if isinstance(x, (pd.Timestamp, datetime)) else x
            )
    
    # Convert to records first
    records = df_serializable.to_dict(orient='records')
    
    # Recursively convert any remaining Timestamp/datetime objects in the records
    records = [convert_timestamps_recursive(record) for record in records]
    
    return records


def serialize_mapping_datetime(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convert datetime-like values within a dict to ISO strings (recursively).
    
    This function recursively processes nested dictionaries and lists to convert
    all datetime-like values to ISO strings.
    
    Args:
        data: Dictionary with potential datetime values
        
    Returns:
        Dictionary with datetime values converted to ISO strings
        
    Example:
        >>> data = {'time': datetime(2024, 1, 1), 'value': 100}
        >>> result = serialize_mapping_datetime(data)
        >>> isinstance(result['time'], str)  # True
        
        >>> nested = {'outer': {'inner': {'time': datetime(2024, 1, 1)}}}
        >>> result = serialize_mapping_datetime(nested)
        >>> isinstance(result['outer']['inner']['time'], str)  # True
    """
    if not data:
        return data
    
    # Use the recursive conversion function for consistent handling
    return convert_timestamps_recursive(data)


def convert_timestamps_recursive(obj: Any) -> Any:
    """
    Recursively convert Timestamp/datetime objects to ISO strings throughout nested structures.
    
    Handles:
    - Pandas Timestamps
    - Python datetime objects
    - Nested dictionaries
    - Lists and tuples
    - Numpy types (int64, float64, nan, etc.)
    
    Args:
        obj: Object to convert (can be dict, list, datetime, or primitive)
        
    Returns:
        Object with all datetime-like values converted to ISO strings
        
    Example:
        >>> data = {'nested': {'time': pd.Timestamp('2024-01-01'), 'list': [datetime(2024, 1, 2)]}}
        >>> result = convert_timestamps_recursive(data)
        >>> isinstance(result['nested']['time'], str)  # True
        >>> isinstance(result['nested']['list'][0], str)  # True
    """
    if isinstance(obj, (pd.Timestamp, datetime)):
        return obj.isoformat()
    elif isinstance(obj, dict):
        return {key: convert_timestamps_recursive(value) for key, value in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [convert_timestamps_recursive(item) for item in obj]
    elif NUMPY_AVAILABLE and np is not None:
        # Handle numpy types
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            if np.isnan(obj):
                return None
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return [convert_timestamps_recursive(item) for item in obj.tolist()]
    
    # Check for None
    if obj is None:
        return None
    
    # Check for regular Python float NaN (from pandas DataFrame.to_dict())
    if isinstance(obj, float):
        try:
            import math
            if math.isnan(obj):
                return None
        except (ValueError, TypeError):
            pass
        return obj
    
    # Only check pd.isna for types that could be NaN (not already handled types)
    if not isinstance(obj, (int, str, bool, list, dict, tuple)):
        try:
            if pd.isna(obj):
                return None
        except (TypeError, ValueError):
            pass
    
    return obj

