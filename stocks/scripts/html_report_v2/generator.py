"""
Main HTML report generator - orchestrates all modules.
"""

import pandas as pd
import sys
from pathlib import Path
from .data_processor import prepare_dataframe_for_display, split_calls_and_puts
from .html_builder import (
    build_html_document, build_header, build_tab_button,
    build_tab_content, get_timestamp_strings, get_title
)
from .table_builder import build_table_html
from .card_builder import build_cards_html
from .analysis_builder import generate_detailed_analysis_html
from .styles import get_css_styles
from .scripts import get_javascript


def generate_html_output(df: pd.DataFrame, output_dir: str) -> None:
    """Generate HTML output with sortable table.
    
    Args:
        df: DataFrame with the results
        output_dir: Directory path where to create the HTML output
    """
    # Create output directory if it doesn't exist
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Split data into calls and puts
    df_calls, df_puts, has_calls, has_puts = split_calls_and_puts(df)
    
    # Prepare DataFrames for display
    df_calls_display, df_calls_raw = prepare_dataframe_for_display(df_calls) if has_calls else (pd.DataFrame(), pd.DataFrame())
    df_puts_display, df_puts_raw = prepare_dataframe_for_display(df_puts) if has_puts else (pd.DataFrame(), pd.DataFrame())
    df_display, df_raw = prepare_dataframe_for_display(df)  # For comprehensive analysis
    
    # Get timestamps
    timestamp, iso_timestamp = get_timestamp_strings()
    
    # Get title
    title = get_title(has_calls, has_puts)
    
    # Build tab buttons
    tab_buttons = []
    tab_index = 0
    
    if has_calls:
        tab_buttons.append(build_tab_button(tab_index, 'Calls', '📞', is_active=True))
        tab_index += 1
    
    if has_puts:
        tab_buttons.append(build_tab_button(tab_index, 'Puts', '📉', is_active=(not has_calls)))
        tab_index += 1
    
    tab_buttons.append(build_tab_button(tab_index, 'Comprehensive Analysis', '📊', is_active=(not has_calls and not has_puts)))
    
    # Build header
    header_html = build_header(title, timestamp, iso_timestamp, len(df), tab_buttons)
    
    # Build tab contents
    tab_contents = []
    tab_index = 0
    
    if has_calls:
        # Build calls tab content
        table_html = build_table_html(df_calls_display, df_calls_raw, 'calls')
        cards_html = build_cards_html(df_calls_display, df_calls_raw, 'calls')
        # Add stats section at bottom
        stats_html = f"""        <div class="stats">
            <div class="stat-item">
                <div class="stat-value" id="callstotalCount">{len(df_calls_display)}</div>
                <div class="stat-label">Total Results</div>
            </div>
            <div class="stat-item">
                <div class="stat-value" id="callsvisibleCount">{len(df_calls_display)}</div>
                <div class="stat-label">Visible Rows</div>
            </div>
        </div>
"""
        calls_content = table_html + '\n' + cards_html + '\n' + stats_html
        tab_contents.append(build_tab_content(calls_content, 'calls', is_active=True))
        tab_index += 1
    
    if has_puts:
        # Build puts tab content
        table_html = build_table_html(df_puts_display, df_puts_raw, 'puts')
        cards_html = build_cards_html(df_puts_display, df_puts_raw, 'puts')
        # Add stats section at bottom
        stats_html = f"""        <div class="stats">
            <div class="stat-item">
                <div class="stat-value" id="putstotalCount">{len(df_puts_display)}</div>
                <div class="stat-label">Total Results</div>
            </div>
            <div class="stat-item">
                <div class="stat-value" id="putsvisibleCount">{len(df_puts_display)}</div>
                <div class="stat-label">Visible Rows</div>
            </div>
        </div>
"""
        puts_content = table_html + '\n' + cards_html + '\n' + stats_html
        tab_contents.append(build_tab_content(puts_content, 'puts', is_active=(not has_calls)))
        tab_index += 1
    
    # Build comprehensive analysis tab
    analysis_content = generate_detailed_analysis_html(df_display)
    tab_contents.append(build_tab_content(analysis_content, 'analysis', is_active=(not has_calls and not has_puts)))
    
    # Combine all content
    body_content = header_html + '\n'.join(tab_contents) + '\n    </div>\n'
    
    # Get CSS and JavaScript
    css_content = get_css_styles()
    js_content = get_javascript()
    
    # Build complete HTML document
    html_content = build_html_document(title, css_content, js_content, body_content)
    
    # Write HTML file
    html_file = output_path / 'index.html'
    with open(html_file, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    print(f"HTML output generated successfully!", file=sys.stderr)
    print(f"Output directory: {output_path.absolute()}", file=sys.stderr)
    print(f"Open: {html_file.absolute()}", file=sys.stderr)

