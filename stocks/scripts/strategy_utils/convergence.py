"""
Parameter convergence analysis across multiple timeframes.

Analyzes how consistent optimal parameters are across different time windows
to determine confidence levels for recommendations.
"""

import pandas as pd
from collections import Counter
from typing import Any, Dict, List, Optional, Tuple

# Parameters to track for convergence analysis
TRACKED_PARAMS = [
    'percent_beyond_put',
    'percent_beyond_call',
    'max_spread_width_put',
    'max_spread_width_call',
    'min_trading_hour',
    'max_trading_hour',
]

# Confidence thresholds
CONFIDENCE_THRESHOLDS = {
    'HIGH': 0.80,    # 80%+ agreement
    'MEDIUM': 0.60,  # 60%+ agreement
    'LOW': 0.40,     # 40%+ agreement
    'NONE': 0.0,     # Below 40%
}


def extract_param_value(row: pd.Series, param: str) -> Any:
    """
    Extract parameter value from a results row.

    Handles both combined params (max_spread_width) and split params
    (max_spread_width_put, max_spread_width_call).
    """
    # Direct match
    if param in row.index:
        return row[param]

    # Handle percent_beyond split
    if param == 'percent_beyond_put' and 'percent_beyond' in row.index:
        return row['percent_beyond']
    if param == 'percent_beyond_call' and 'percent_beyond' in row.index:
        return row['percent_beyond']

    # Handle max_spread_width split
    if param == 'max_spread_width_put' and 'max_spread_width' in row.index:
        return row['max_spread_width']
    if param == 'max_spread_width_call' and 'max_spread_width' in row.index:
        return row['max_spread_width']

    return None


def get_modal_value(df: pd.DataFrame, param: str, top_n: int = 10) -> Tuple[Any, int]:
    """
    Get the most frequent (modal) value for a parameter in top N results.

    Args:
        df: Results DataFrame
        param: Parameter name to analyze
        top_n: Number of top results to consider

    Returns:
        Tuple of (modal_value, count_of_mode)
    """
    top_results = df.head(top_n)
    values = []

    for _, row in top_results.iterrows():
        value = extract_param_value(row, param)
        if value is not None:
            values.append(value)

    if not values:
        return None, 0

    counter = Counter(values)
    mode, count = counter.most_common(1)[0]
    return mode, count


def calculate_param_confidence(
    values_by_window: Dict[str, Any],
    weights: Dict[str, float]
) -> Tuple[float, str]:
    """
    Calculate confidence score for a parameter based on cross-window agreement.

    Args:
        values_by_window: Dict mapping window names to parameter values
        weights: Dict mapping window names to their weights

    Returns:
        Tuple of (confidence_score, confidence_level)
    """
    if not values_by_window:
        return 0.0, 'NONE'

    # Count value occurrences
    counter = Counter(values_by_window.values())

    if not counter:
        return 0.0, 'NONE'

    # Get most common value
    most_common_value, _ = counter.most_common(1)[0]

    # Calculate weighted agreement
    weighted_agreement = 0.0
    total_weight = 0.0

    for window, value in values_by_window.items():
        weight = weights.get(window, 0.0)
        total_weight += weight
        if value == most_common_value:
            weighted_agreement += weight

    if total_weight == 0:
        return 0.0, 'NONE'

    confidence_score = weighted_agreement / total_weight

    # Determine level
    for level, threshold in sorted(CONFIDENCE_THRESHOLDS.items(),
                                   key=lambda x: -x[1]):
        if confidence_score >= threshold:
            return confidence_score, level

    return confidence_score, 'NONE'


def analyze_convergence(
    results: Dict[str, pd.DataFrame],
    weights: Dict[str, float],
    top_n: int = 10,
    params: Optional[List[str]] = None
) -> Dict[str, Dict]:
    """
    Compare top parameters across timeframes.

    Args:
        results: Dict mapping window names to results DataFrames
        weights: Dict mapping window names to their weights
        top_n: Number of top results to consider per window
        params: List of parameters to analyze (defaults to TRACKED_PARAMS)

    Returns:
        Dict with convergence analysis for each parameter:
        {
            'param_name': {
                'values': {window: value, ...},
                'confidence_score': float,
                'confidence_level': str,
                'recommended': value
            },
            ...
        }
    """
    if params is None:
        params = TRACKED_PARAMS

    convergence = {}

    for param in params:
        values_by_window = {}

        for window, df in results.items():
            if df is None or df.empty:
                continue

            modal_value, _ = get_modal_value(df, param, top_n)
            if modal_value is not None:
                values_by_window[window] = modal_value

        confidence_score, confidence_level = calculate_param_confidence(
            values_by_window, weights
        )

        # Calculate weighted recommendation
        recommended = calculate_weighted_recommendation(
            values_by_window, weights, param
        )

        convergence[param] = {
            'values': values_by_window,
            'confidence_score': confidence_score,
            'confidence_level': confidence_level,
            'recommended': recommended,
        }

    return convergence


def calculate_weighted_recommendation(
    values_by_window: Dict[str, Any],
    weights: Dict[str, float],
    param: str
) -> Any:
    """
    Calculate weighted recommendation for a parameter.

    For numeric parameters, returns weighted average.
    For categorical parameters, returns weighted mode.

    Args:
        values_by_window: Dict mapping window names to parameter values
        weights: Dict mapping window names to their weights
        param: Parameter name (used to determine if numeric)

    Returns:
        Recommended value
    """
    if not values_by_window:
        return None

    # Check if numeric
    sample_value = next(iter(values_by_window.values()))
    is_numeric = isinstance(sample_value, (int, float))

    if is_numeric:
        # Weighted average for numeric params
        weighted_sum = 0.0
        total_weight = 0.0

        for window, value in values_by_window.items():
            weight = weights.get(window, 0.0)
            weighted_sum += value * weight
            total_weight += weight

        if total_weight == 0:
            return sample_value

        result = weighted_sum / total_weight

        # Round to appropriate precision
        if isinstance(sample_value, int):
            return int(round(result))
        elif param.startswith('percent_'):
            return round(result, 4)  # 4 decimal places for percentages
        else:
            return round(result, 2)
    else:
        # Weighted mode for categorical params
        weighted_counts = Counter()

        for window, value in values_by_window.items():
            weight = weights.get(window, 0.0)
            weighted_counts[value] += weight

        return weighted_counts.most_common(1)[0][0]


def detect_trend(values_by_window: Dict[str, Any], ordered_windows: List[str]) -> str:
    """
    Detect trend direction for a parameter across time windows.

    Args:
        values_by_window: Dict mapping window names to parameter values
        ordered_windows: Windows in chronological order (oldest to newest)

    Returns:
        'INCREASING', 'DECREASING', 'STABLE', or 'MIXED'
    """
    ordered_values = []
    for window in ordered_windows:
        if window in values_by_window:
            value = values_by_window[window]
            if isinstance(value, (int, float)):
                ordered_values.append(value)

    if len(ordered_values) < 2:
        return 'STABLE'

    # Calculate direction changes
    increases = 0
    decreases = 0

    for i in range(1, len(ordered_values)):
        diff = ordered_values[i] - ordered_values[i-1]
        if diff > 0:
            increases += 1
        elif diff < 0:
            decreases += 1

    total_changes = increases + decreases
    if total_changes == 0:
        return 'STABLE'

    # Strong trend if 80%+ in one direction
    if increases / total_changes >= 0.8:
        return 'INCREASING'
    elif decreases / total_changes >= 0.8:
        return 'DECREASING'
    else:
        return 'MIXED'


def get_convergence_summary(convergence: Dict[str, Dict]) -> Dict[str, Any]:
    """
    Generate summary statistics for convergence analysis.

    Args:
        convergence: Output from analyze_convergence()

    Returns:
        Summary dict with counts and overall confidence
    """
    high_confidence = sum(1 for p in convergence.values()
                         if p['confidence_level'] == 'HIGH')
    medium_confidence = sum(1 for p in convergence.values()
                           if p['confidence_level'] == 'MEDIUM')
    low_confidence = sum(1 for p in convergence.values()
                        if p['confidence_level'] in ('LOW', 'NONE'))

    # Overall confidence is weighted average
    total_params = len(convergence)
    avg_confidence = sum(p['confidence_score'] for p in convergence.values()) / total_params if total_params > 0 else 0

    # Determine overall level
    if avg_confidence >= 0.75:
        overall_level = 'HIGH'
    elif avg_confidence >= 0.55:
        overall_level = 'MEDIUM-HIGH'
    elif avg_confidence >= 0.40:
        overall_level = 'MEDIUM'
    else:
        overall_level = 'LOW'

    return {
        'total_params': total_params,
        'high_confidence_count': high_confidence,
        'medium_confidence_count': medium_confidence,
        'low_confidence_count': low_confidence,
        'average_confidence': round(avg_confidence * 100, 1),
        'overall_level': overall_level,
    }
