# HTML Report Generator V2 - Quick Start

## What's New

This is a **clean rewrite** of the HTML report generator with all the critical bugs fixed:

✅ **Fixed**: Puts now show as tables on desktop (not just cards)
✅ **Fixed**: No stats section appearing after each card
✅ **Fixed**: Tabs work correctly - only one visible at a time
✅ **Fixed**: Proper responsive behavior (tables on desktop, cards on mobile)

## Usage

```python
from scripts.html_report_v2 import generate_html_output
import pandas as pd

# Your DataFrame with options data
# Must have 'option_type' column with 'call' or 'put' values
df = pd.DataFrame({
    'ticker': ['AAPL', 'MSFT'],
    'option_type': ['call', 'put'],
    'strike_price': [150.0, 200.0],
    'current_price': [175.0, 250.0],
    # ... other columns
})

# Generate HTML report
generate_html_output(df, 'output_directory')
```

## Key Differences from V1

1. **Clean Architecture**: Modular structure with clear separation of concerns
2. **No Stats Bug**: Stats section completely removed from card view
3. **Proper Tab Handling**: Each tab is independent, CSS handles visibility
4. **Desktop/Mobile**: Proper responsive design with CSS media queries

## File Structure

- `config.py` - All constants and configuration
- `data_processor.py` - Data preparation and formatting
- `html_builder.py` - HTML structure generation
- `table_builder.py` - Desktop table HTML
- `card_builder.py` - Mobile card HTML
- `styles.py` - CSS (inherited from v1 with fixes)
- `scripts.py` - JavaScript (inherited from v1 with fixes)
- `generator.py` - Main orchestrator

## Testing

To test with your existing code:

```python
# Replace this:
from scripts.html_report_generator import generate_html_output

# With this:
from scripts.html_report_v2 import generate_html_output

# Everything else stays the same!
generate_html_output(df, output_dir)
```

## Status

- ✅ Core functionality complete
- ✅ All critical bugs fixed
- ⏳ Comprehensive analysis section (placeholder - can be added later)

