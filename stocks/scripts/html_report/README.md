# HTML Report Generator - Modular Structure

## Overview

The `html_report_generator.py` file has been refactored into a modular, maintainable structure. The code is now organized into focused modules, each with a single responsibility.

## Module Structure

```
scripts/html_report/
├── __init__.py              # Package initialization, exports main function
├── constants.py             # Configuration and constants (column mappings, hidden columns, etc.)
├── formatters.py            # Data formatting utilities (age, numeric values, headers)
├── css_generator.py         # CSS styles generation (~800 lines)
├── js_generator.py          # JavaScript code generation (~1400 lines)
├── data_preparer.py         # DataFrame preparation and column handling (~350 lines)
├── html_templates.py        # HTML template generation (summary stats, detailed analysis)
├── table_generator.py       # Table and card HTML generation (~800 lines)
└── generator.py             # Main orchestrator (~150 lines)
```

## Usage

### New Code (Recommended)

```python
from scripts.html_report import generate_html_output

# Use exactly as before
generate_html_output(df, output_dir)
```

### Backward Compatibility

The original `html_report_generator.py` file still works and will automatically use the new modules if available, falling back to the original implementation if modules are missing.

```python
# This still works
from scripts.html_report_generator import generate_html_output

generate_html_output(df, output_dir)
```

## Module Responsibilities

### constants.py
- Column name mappings (`COMPACT_HEADER_MAP`)
- Hidden column definitions
- Column group definitions
- Display configuration

### formatters.py
- `format_age_seconds()` - Format age in seconds to human-readable
- `format_numeric_value()` - Format numeric values based on column type
- `truncate_header()` - Wrap header text
- `normalize_col_name()` - Normalize column names for matching
- `extract_numeric_value()` - Extract numeric values from strings

### css_generator.py
- `get_css_styles()` - Returns complete CSS stylesheet as string

### js_generator.py
- `get_javascript()` - Returns complete JavaScript code as string
- Handles sorting, filtering, URL state management, etc.

### data_preparer.py
- `prepare_dataframe_for_display()` - Prepares DataFrame for HTML display
- Column normalization and ordering
- Date formatting
- Raw value preservation for filtering

### html_templates.py
- `generate_summary_statistics_html()` - Summary statistics section
- `generate_detailed_analysis_html()` - Detailed analysis section

### table_generator.py
- `generate_table_and_cards_html()` - Generates table and mobile card HTML
- Filter UI generation
- Column grouping logic
- Data attribute generation for filtering

### generator.py
- `generate_html_output()` - Main entry point
- Orchestrates all modules
- Handles data splitting (calls/puts)
- Generates final HTML document

## Benefits

1. **Modularity**: Each module has a single, clear responsibility
2. **Reusability**: Functions can be imported and used independently
3. **Testability**: Smaller functions are easier to test
4. **Maintainability**: Changes to one area don't affect others
5. **Readability**: Smaller files are easier to understand and navigate
6. **Backward Compatibility**: Original file still works

## Migration Notes

- The original `html_report_generator.py` file is preserved with fallback code
- All existing imports continue to work
- New code should import from `scripts.html_report` package
- The modular structure is ready for further enhancements

## File Sizes

- **Original file**: 4,084 lines (with fallback code)
- **New modules**: 4,306 lines total across 9 focused modules
- **Largest modules**: 
  - `js_generator.py`: ~1,400 lines (JavaScript code)
  - `table_generator.py`: ~800 lines (table/card generation)
  - `css_generator.py`: ~800 lines (CSS styles)

## Next Steps

Future improvements could include:
- Further splitting JavaScript into logical sections
- Extracting card generation to separate module
- Adding unit tests for individual modules
- Creating configuration files for column definitions

