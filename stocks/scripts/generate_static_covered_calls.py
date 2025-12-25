#!/usr/bin/env python3
"""
Generate static HTML/CSS/JS files for covered calls analysis.
This creates static files in common/web/covered_call/ that can be served directly
or deployed to a CDN.
"""

import sys
from pathlib import Path

# Add scripts directory to path
scripts_dir = Path(__file__).parent
if str(scripts_dir) not in sys.path:
    sys.path.insert(0, str(scripts_dir))

from html_report_v2.styles import get_css_styles
from html_report_v2.scripts import get_javascript
from html_report_v2.html_builder import build_html_document, build_header, build_tab_button, build_tab_content, get_title
from html_report_v2.table_builder import build_table_html
from html_report_v2.data_processor import split_calls_and_puts
import pandas as pd

def modify_javascript(js_content: str) -> str:
    """Modify JavaScript to remove CSV source dependency.
    
    The API endpoint doesn't need the source parameter anymore since it
    fetches data directly from the database.
    """
    # Remove CSV source from API_CONFIG
    js_content = js_content.replace(
        "const API_CONFIG = window.API_CONFIG || {\n            csv_source: '/tmp/results.csv'\n        };",
        "const API_CONFIG = window.API_CONFIG || {};"
    )
    
    # Remove source parameter from API calls
    js_content = js_content.replace(
        "const params = new URLSearchParams({\n                    source: API_CONFIG.csv_source,\n                    option_type: optionType\n                });",
        "const params = new URLSearchParams({\n                    option_type: optionType\n                });"
    )
    
    # Remove source from analysis API call
    js_content = js_content.replace(
        "const params = new URLSearchParams({\n                    source: API_CONFIG.csv_source,\n                    option_type: 'all',  // Get all data for comprehensive analysis\n                    use_gemini: useGemini ? 'true' : 'false'  // Use checkbox value\n                });",
        "const params = new URLSearchParams({\n                    option_type: 'all',  // Get all data for comprehensive analysis\n                    use_gemini: useGemini ? 'true' : 'false'  // Use checkbox value\n                });"
    )
    
    return js_content

def generate_static_files():
    """Generate static HTML, CSS, and JS files."""
    # Output directory
    output_dir = Path(__file__).parent.parent / "common" / "web" / "covered_call"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Generating static files in: {output_dir}")
    
    # Get CSS
    css_content = get_css_styles()
    
    # Get JavaScript and modify it
    js_content = get_javascript()
    js_content = modify_javascript(js_content)
    
    # We always show both Calls and Puts tabs in the static shell.
    # The actual data (and which side has rows) is determined at runtime by the API.
    has_calls = True
    has_puts = True
    
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
    
    # Build header (with placeholder counts - will be updated by JS)
    header_html = build_header(title, "Loading...", "", 0, tab_buttons)
    
    # Build tab contents (with empty table structures - data will be loaded via API)
    tab_contents = []
    tab_index = 0
    
    if has_calls:
        # Build calls tab content with empty table structure
        # We need to prepare a DataFrame with the expected columns for table building
        calls_df = pd.DataFrame({
            'ticker': [],
            'option_type': ['call'] * 0,  # Empty but with option_type
            'current_price': [],
            'strike_price': [],
            'l_strike': [],
            'opt_prem.': [],
            'l_prem': [],
            'expiration_date': [],
            'l_expiration_date': [],
            'bid:ask': [],  # Short bid/ask prices
            'l_bid:ask': [],  # Long bid/ask prices
            'delta': [],
            'l_delta': [],
            'theta': [],
            'l_theta': [],
            'net_daily_premi': [],
            'volume': [],
            'num_contracts': [],
            'pe_ratio': [],
            'market_cap_b': [],
            # Include premium_ratio_pct so the static header has this column
            'premium_ratio_pct': [],
            # Include hidden columns so they appear in headers (can be toggled visible)
            's_prem_tot': [],  # Short premium total
            's_day_prem': [],  # Short daily premium
            'l_prem_tot': [],  # Long premium total
            'latest_option_writets': [],  # Last update timestamp (renamed from latest_opt_ts)
            'net_premium': [],  # Net premium
            'spread_slippage': [],  # Spread slippage
            'net_premium_after_spread': [],  # Net premium after spread
            'net_daily_premium_after_spread': [],  # Net daily premium after spread
            'spread_impact_pct': [],  # Spread impact percentage
            'liquidity_score': [],  # Liquidity score
            'assignment_risk': [],  # Assignment risk
            'trade_quality': [],  # Trade quality score
            'option_ticker': [],  # Option ticker
            'l_option_ticker': [],  # Long option ticker
            'days_to_expiry': [],  # Days to expiry
            'l_days_to_expiry': [],  # Long days to expiry
            'price_change_pct': [],  # Price change percentage
        })
        from html_report_v2.data_processor import prepare_dataframe_for_display
        calls_display, calls_raw = prepare_dataframe_for_display(calls_df)
        table_html = build_table_html(calls_display, calls_raw, 'calls', empty=True)
        cards_html = '<div id="callscardsContainer" class="cards-container"></div>'
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
        puts_df = pd.DataFrame({
            'ticker': [],
            'option_type': ['put'] * 0,  # Empty but with option_type
            'current_price': [],
            'strike_price': [],
            'l_strike': [],
            'opt_prem.': [],
            'l_prem': [],
            'expiration_date': [],
            'l_expiration_date': [],
            'bid:ask': [],  # Short bid/ask prices
            'l_bid:ask': [],  # Long bid/ask prices
            'delta': [],
            'l_delta': [],
            'theta': [],
            'l_theta': [],
            'net_daily_premi': [],
            'volume': [],
            'num_contracts': [],
            'pe_ratio': [],
            'market_cap_b': [],
            # Include premium_ratio_pct so the static header has this column
            'premium_ratio_pct': [],
            # Include hidden columns so they appear in headers (can be toggled visible)
            's_prem_tot': [],  # Short premium total
            's_day_prem': [],  # Short daily premium
            'l_prem_tot': [],  # Long premium total
            'latest_option_writets': [],  # Last update timestamp (renamed from latest_opt_ts)
            'net_premium': [],  # Net premium
            'spread_slippage': [],  # Spread slippage
            'net_premium_after_spread': [],  # Net premium after spread
            'net_daily_premium_after_spread': [],  # Net daily premium after spread
            'spread_impact_pct': [],  # Spread impact percentage
            'liquidity_score': [],  # Liquidity score
            'assignment_risk': [],  # Assignment risk
            'trade_quality': [],  # Trade quality score
            'option_ticker': [],  # Option ticker
            'l_option_ticker': [],  # Long option ticker
            'days_to_expiry': [],  # Days to expiry
            'l_days_to_expiry': [],  # Long days to expiry
            'price_change_pct': [],  # Price change percentage
        })
        from html_report_v2.data_processor import prepare_dataframe_for_display
        puts_display, puts_raw = prepare_dataframe_for_display(puts_df)
        table_html = build_table_html(puts_display, puts_raw, 'puts', empty=True)
        cards_html = '<div id="putscardsContainer" class="cards-container"></div>'
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
    
    # Build complete HTML document with external JS
    html_content = build_html_document(
        title, 
        css_content, 
        js_content, 
        body_content,
        external_js=True,
        js_file="app.js",
        api_config={}  # Empty config - no CSV source needed
    )
    
    # Write HTML file
    html_file = output_dir / 'index.html'
    with open(html_file, 'w', encoding='utf-8') as f:
        f.write(html_content)
    print(f"✓ Created {html_file}")
    
    # Write CSS file
    css_file = output_dir / 'styles.css'
    with open(css_file, 'w', encoding='utf-8') as f:
        f.write(css_content)
    print(f"✓ Created {css_file}")
    
    # Write JavaScript file
    js_file = output_dir / 'app.js'
    with open(js_file, 'w', encoding='utf-8') as f:
        f.write(js_content)
    print(f"✓ Created {js_file}")
    
    print(f"\n✅ Static files generated successfully in {output_dir}")
    print(f"   Files: index.html, styles.css, app.js")
    print(f"\n   To serve these files:")
    print(f"   - Access via: /stock_info/api/covered_calls/view")
    print(f"   - Or deploy to CDN and update the route")

if __name__ == "__main__":
    generate_static_files()


