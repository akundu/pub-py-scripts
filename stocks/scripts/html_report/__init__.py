"""
HTML Report Generator - Modular package for generating HTML reports with sortable tables.

This package provides a modular structure for generating HTML reports with:
- Sortable tables
- Filtering capabilities
- Mobile-responsive card layouts
- Comprehensive analysis sections
"""

from .generator import generate_html_output

# Export main function and commonly used utilities
__all__ = ['generate_html_output']

