"""
Main HTML report generator - orchestrates all modules.
"""

import logging
import sys
import pandas as pd
from pathlib import Path

logger = logging.getLogger(__name__)
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

def generate_html_output(df: pd.DataFrame, output_dir: str, csv_source: str = None) -> None:
    """Generate HTML output with sortable table.
    
    Args:
        df: DataFrame with the results (used only for structure/metadata, not embedded in HTML)
        output_dir: Directory path where to create the HTML output
        csv_source: Path to CSV file or URL (default: None, will use default location)
    """
    # Create output directory if it doesn't exist
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Split data into calls and puts (for metadata only)
    df_calls, df_puts, has_calls, has_puts = split_calls_and_puts(df)
    
    # Prepare DataFrames for display (only for structure/metadata)
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
    
    # Build tab contents (with empty table structures - data will be loaded via API)
    tab_contents = []
    tab_index = 0
    
    if has_calls:
        # Build calls tab content with empty table structure
        table_html = build_table_html(df_calls_display, df_calls_raw, 'calls', empty=True)
        cards_html = '<div id="callscardsContainer" class="cards-container"></div>'
        # Add stats section at bottom
        stats_html = f"""        <div class="stats">
            <div class="stat-item">
                <div class="stat-value" id="callstotalCount">0</div>
                <div class="stat-label">Total Results</div>
            </div>
            <div class="stat-item">
                <div class="stat-value" id="callsvisibleCount">0</div>
                <div class="stat-label">Visible Rows</div>
            </div>
        </div>
"""
        calls_content = table_html + '\n' + cards_html + '\n' + stats_html
        tab_contents.append(build_tab_content(calls_content, 'calls', is_active=True))
        tab_index += 1
    
    if has_puts:
        # Build puts tab content with empty table structure
        table_html = build_table_html(df_puts_display, df_puts_raw, 'puts', empty=True)
        cards_html = '<div id="putscardsContainer" class="cards-container"></div>'
        # Add stats section at bottom
        stats_html = f"""        <div class="stats">
            <div class="stat-item">
                <div class="stat-value" id="putstotalCount">0</div>
                <div class="stat-label">Total Results</div>
            </div>
            <div class="stat-item">
                <div class="stat-value" id="putsvisibleCount">0</div>
                <div class="stat-label">Visible Rows</div>
            </div>
        </div>
"""
        puts_content = table_html + '\n' + cards_html + '\n' + stats_html
        tab_contents.append(build_tab_content(puts_content, 'puts', is_active=(not has_calls)))
        tab_index += 1
    
    # Build comprehensive analysis tab (empty initially, will be loaded dynamically)
    analysis_content = '''        <div class="analysis-controls" style="margin-bottom: 20px; padding: 15px; background: #f5f5f5; border-radius: 8px;">
            <div style="display: flex; align-items: center; gap: 15px; flex-wrap: wrap;">
                <label style="display: flex; align-items: center; gap: 8px; cursor: pointer;">
                    <input type="checkbox" id="useGeminiAnalysis" style="width: 18px; height: 18px; cursor: pointer;">
                    <span style="font-weight: 500;">Use Gemini AI Analysis</span>
                </label>
                <button id="loadAnalysisBtn" onclick="loadComprehensiveAnalysis()" style="padding: 8px 16px; background: #667eea; color: white; border: none; border-radius: 4px; cursor: pointer; font-weight: 500;">
                    🔄 Load Analysis
                </button>
                <span id="analysisStatus" style="color: #666; font-size: 14px;"></span>
            </div>
            <div style="margin-top: 10px; font-size: 12px; color: #888;">
                <strong>Rule-based:</strong> Fast, deterministic analysis based on scoring algorithm (loads automatically)<br>
                <strong>Gemini AI:</strong> AI-powered analysis with risk/aggressive/conservative recommendations (check box and click button, slower, requires API key)
            </div>
        </div>
        <div id="analysisContent">
            <div class="loading-indicator"><div class="spinner"></div><div class="loading-text">Loading rule-based analysis...</div></div>
        </div>'''
    tab_contents.append(build_tab_content(analysis_content, 'analysis', is_active=(not has_calls and not has_puts)))
    
    # Combine all content
    body_content = header_html + '\n'.join(tab_contents) + '\n    </div>\n'
    
    # Get CSS and JavaScript
    css_content = get_css_styles()
    js_content = get_javascript()
    
    # Determine CSV source (default to a standard location if not provided)
    if csv_source is None:
        # Default to a standard location - can be overridden by client
        csv_source = "/tmp/results.csv"
    
    # Build API configuration (only CSV source needed - API uses same host)
    api_config = {
        "csv_source": csv_source
    }
    
    # Build complete HTML document with external JS
    html_content = build_html_document(
        title, 
        css_content, 
        js_content, 
        body_content,
        external_js=True,
        js_file="app.js",
        api_config=api_config
    )
    
    # Write HTML file
    html_file = output_path / 'index.html'
    with open(html_file, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    # Write JavaScript file
    js_file = output_path / 'app.js'
    with open(js_file, 'w', encoding='utf-8') as f:
        f.write(js_content)
    
    print(f"HTML output generated successfully!", file=sys.stderr)
    print(f"Output directory: {output_path.absolute()}", file=sys.stderr)
    print(f"Open: {html_file.absolute()}", file=sys.stderr)

