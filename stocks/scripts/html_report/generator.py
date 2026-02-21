"""
Main HTML report generator - orchestrates all modules.
"""

import pandas as pd
import sys
from pathlib import Path
from datetime import datetime
from .css_generator import get_css_styles
from .js_generator import get_javascript
from .data_preparer import prepare_dataframe_for_display
from .html_templates import generate_summary_statistics_html, generate_detailed_analysis_html
from .table_generator import generate_table_and_cards_html
from .formatters import normalize_col_name

def generate_html_output(df: pd.DataFrame, output_dir: str) -> None:
    """Generate HTML output with sortable table.
    
    Args:
        df: DataFrame with the results
        output_dir: Directory path where to create the HTML output
    """
    # Create output directory if it doesn't exist
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Split data into calls and puts if option_type column exists
    df_calls = pd.DataFrame()
    df_puts = pd.DataFrame()
    has_calls = False
    has_puts = False
    
    if 'option_type' in df.columns:
        df_calls = df[df['option_type'].str.lower() == 'call'].copy() if 'option_type' in df.columns else pd.DataFrame()
        df_puts = df[df['option_type'].str.lower() == 'put'].copy() if 'option_type' in df.columns else pd.DataFrame()
        has_calls = len(df_calls) > 0
        has_puts = len(df_puts) > 0
    else:
        # If no option_type column, treat all as calls (backward compatibility)
        df_calls = df.copy()
        has_calls = True
    
    # Prepare DataFrames for display
    df_calls_display, df_calls_raw = prepare_dataframe_for_display(df_calls) if has_calls else (pd.DataFrame(), pd.DataFrame())
    df_puts_display, df_puts_raw = prepare_dataframe_for_display(df_puts) if has_puts else (pd.DataFrame(), pd.DataFrame())
    df_display, df_raw = prepare_dataframe_for_display(df)  # For comprehensive analysis
    
    # Get current timestamp for display
    now = datetime.now()
    timestamp = now.strftime("%Y-%m-%d %H:%M:%S")
    # Get ISO timestamp with timezone for JavaScript parsing (ensure it's parseable)
    # Format as ISO string with 'Z' suffix for UTC to ensure JavaScript can parse it
    if now.tzinfo is None:
        # If no timezone info, format as UTC
        iso_timestamp = now.strftime("%Y-%m-%dT%H:%M:%SZ")
    else:
        iso_timestamp = now.isoformat()
    
    # Generate HTML - build it piece by piece
    html_parts = []
    
    # HTML head and styles
    html_parts.append("""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Options Analysis Results</title>
    <style>
""")
    html_parts.append(get_css_styles())
    html_parts.append("""    </style>
</head>
<body>
    <div class="container">
        <div class="header">
""")
    
    # Detect option type from data for title
    option_type_detected = 'call'  # Default
    if has_calls and has_puts:
        option_type_detected = 'mixed'
    elif has_puts:
        option_type_detected = 'put'
    
    # Set title based on option type
    if option_type_detected == 'put':
        title = "📊 Cash-Secured Puts Analysis Results"
    elif option_type_detected == 'mixed':
        title = "📊 Options Analysis Results (Calls & Puts)"
    else:
        title = "📊 Covered Calls Analysis Results"
    
    html_parts.append(f'            <h1>{title}</h1>\n')
    html_parts.append("""            <div class="tabs">\n""")
    
    # Dynamically create tab buttons based on available data
    tab_index = 0
    if has_calls:
        html_parts.append(f'                <button class="tab-button {"active" if tab_index == 0 else ""}" onclick="switchTab({tab_index})">📞 Calls</button>\n')
        tab_index += 1
    if has_puts:
        html_parts.append(f'                <button class="tab-button {"active" if tab_index == 0 else ""}" onclick="switchTab({tab_index})">📉 Puts</button>\n')
        tab_index += 1
    # Always add comprehensive analysis tab
    html_parts.append(f'                <button class="tab-button" onclick="switchTab({tab_index})">📊 Comprehensive Analysis</button>\n')
    html_parts.append("""            </div>
            <p id="generatedTime" data-generated="""" + iso_timestamp + """">Data updated: <span id="dataTimestamp">""" + timestamp + """</span> <span id="timeAgo"></span></p>
            <p class="desktop-only">Click column headers to sort • """ + str(len(df)) + """ total results</p>
            <p class="mobile-only">Tap cards to expand details • """ + str(len(df)) + """ total results</p>
        </div>
""")
    
    # Generate tabs for calls and puts
    tab_index = 0
    if has_calls:
        html_parts.append(generate_table_and_cards_html(df_calls_display, df_calls_raw, 'calls', tab_index == 0, normalize_col_name))
        tab_index += 1
    if has_puts:
        html_parts.append(generate_table_and_cards_html(df_puts_display, df_puts_raw, 'puts', tab_index == 0, normalize_col_name))
        tab_index += 1
    
    # Comprehensive analysis tab
    html_parts.append(f"""        <div class="tab-content">
""")
    html_parts.append(generate_detailed_analysis_html(df_display))
    html_parts.append("""        </div>
    </div>
    
    <script>
""")
    html_parts.append(get_javascript())
    html_parts.append("""    </script>
</body>
</html>
""")
    
    html_content = ''.join(html_parts)
    
    # Write HTML file
    html_file = output_path / 'index.html'
    with open(html_file, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    print(f"HTML output generated successfully!", file=sys.stderr)
    print(f"Output directory: {output_path.absolute()}", file=sys.stderr)
    print(f"Open: {html_file.absolute()}", file=sys.stderr)

