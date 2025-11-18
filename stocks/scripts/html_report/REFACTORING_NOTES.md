# HTML Report Generator Refactoring

## Overview
The `html_report_generator.py` file (4062 lines) has been refactored into a modular structure for better maintainability, readability, and reusability.

## New Structure

```
scripts/html_report/
├── __init__.py              # Main entry point
├── constants.py             # Configuration and constants
├── formatters.py            # Data formatting utilities
├── css_generator.py         # CSS styles generation
├── js_generator.py          # JavaScript code generation
├── data_preparer.py         # DataFrame preparation and column handling
├── html_templates.py        # HTML template generation
├── table_generator.py       # Table HTML generation
├── card_generator.py        # Mobile card HTML generation
└── generator.py             # Main orchestrator
```

## Module Responsibilities

### constants.py
- Column name mappings
- Hidden column definitions
- Column group definitions
- Display configuration

### formatters.py
- `format_age_seconds()` - Format age in seconds
- `format_numeric_value()` - Format numeric values
- `truncate_header()` - Wrap header text
- `normalize_col_name()` - Normalize column names
- `extract_numeric_value()` - Extract numeric values from strings

### css_generator.py
- `get_css_styles()` - Returns CSS styles as string

### js_generator.py
- `get_javascript()` - Returns JavaScript code as string
- Note: JavaScript is kept as a single string for now. Further modularization can be done by splitting into separate JS files if needed.

### data_preparer.py
- `prepare_dataframe_for_display()` - Prepare DataFrame for HTML display
- Column normalization and ordering
- Date formatting
- Raw value preservation for filtering

### html_templates.py
- `generate_header_html()` - Generate HTML header
- `generate_tabs_html()` - Generate tab navigation
- `generate_summary_statistics_html()` - Generate summary statistics
- `generate_detailed_analysis_html()` - Generate detailed analysis section

### table_generator.py
- `generate_table_html()` - Generate table HTML
- `generate_table_headers()` - Generate table headers with grouping
- `generate_table_rows()` - Generate table rows with data attributes

### card_generator.py
- `generate_cards_html()` - Generate mobile card HTML
- `generate_card_html()` - Generate individual card HTML

### generator.py
- `generate_html_output()` - Main entry point that orchestrates all modules

## Benefits

1. **Modularity**: Each module has a single, clear responsibility
2. **Reusability**: Functions can be imported and used independently
3. **Testability**: Smaller functions are easier to test
4. **Maintainability**: Changes to one area don't affect others
5. **Readability**: Smaller files are easier to understand

## Migration Path

The original `html_report_generator.py` can be updated to import from the new modules:

```python
from scripts.html_report import generate_html_output
```

This maintains backward compatibility while using the new modular structure.

