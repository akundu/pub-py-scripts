"""
Web utilities package for serialization, filtering, and HTML generation.
"""

from .serializers import (
    dataframe_to_json_records,
    serialize_mapping_datetime,
    convert_timestamps_recursive,
    clean_for_json
)
from .filters import (
    parse_filter_strings,
    apply_filters
)
from .html_generators import (
    format_options_html,
    generate_stock_info_html
)

__all__ = [
    'dataframe_to_json_records',
    'serialize_mapping_datetime',
    'convert_timestamps_recursive',
    'clean_for_json',
    'parse_filter_strings',
    'apply_filters',
    'format_options_html',
    'generate_stock_info_html',
]

