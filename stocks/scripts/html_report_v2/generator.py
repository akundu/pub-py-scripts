"""
Main HTML report generator - orchestrates all modules.
"""

import logging
import pandas as pd
import sys
import threading
import time
import urllib.request
import urllib.error
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

# Module-level set to track tickers that are being/have been fetched
# This ensures we only fetch each ticker once, even across multiple calls
_warmup_tickers_lock = threading.Lock()
_warmup_tickers_in_progress = set()


def _warmup_cache_for_ticker(ticker: str, host: str = "mm.kundu.dev", port: int = 9100) -> None:
    """Fire-and-forget cache warmup for a single ticker."""
    url = f"http://{host}:{port}/api/stock_info/{ticker}?show_iv=true&show_news=true"
    
    logger.debug(f"[CACHE WARMUP] Sending HTTP request for {ticker} to {url}")
    
    try:
        # Make request with timeout - fire and forget, errors are expected
        req = urllib.request.Request(url)
        logger.debug(f"[CACHE WARMUP] Making HTTP GET request to: {url}")
        urllib.request.urlopen(req, timeout=5.0)
        success_msg = f"[CACHE WARMUP] Successfully sent request for {ticker} to {host}:{port}"
        logger.debug(success_msg)
    except TimeoutError:
        # Timeout errors are expected and ignored (fire-and-forget)
        logger.debug(f"[CACHE WARMUP] Timeout for {ticker} (expected, ignoring)")
        pass
    except (urllib.error.URLError, urllib.error.HTTPError, OSError) as e:
        # Other errors are also ignored (fire-and-forget), but log at debug level
        logger.debug(f"[CACHE WARMUP] Error sending request for {ticker} to {url}: {type(e).__name__}: {e}")
        pass


def _warmup_stock_info_cache(df: pd.DataFrame, host: str = "mm.kundu.dev", port: int = 9100) -> None:
    """Background cache warmup: Extract unique tickers and fetch stock_info in background.
    
    This function ensures each ticker is only fetched once, even if called multiple times.
    Uses a module-level set with thread-safe locking to prevent duplicate fetches.
    """
    if df.empty:
        return
    
    # Extract unique ticker symbols from DataFrame
    ticker_col = None
    for col in ['ticker', 'TICKER', 'Ticker']:
        if col in df.columns:
            ticker_col = col
            break
    
    if not ticker_col:
        return
    
    # Get unique tickers (excluding NaN/None/empty values)
    tickers = df[ticker_col].dropna().astype(str).str.strip()
    tickers = tickers[tickers != '']
    tickers = tickers[tickers != 'N/A']
    unique_tickers = tickers.unique()
    
    if len(unique_tickers) == 0:
        return
    
    # Thread-safe check: Only fetch tickers that haven't been fetched yet
    with _warmup_tickers_lock:
        # Filter out tickers that are already being/have been fetched
        new_tickers = [t for t in unique_tickers if t not in _warmup_tickers_in_progress]
        
        if len(new_tickers) == 0:
            # All tickers are already being fetched
            return
        
        # Mark new tickers as in-progress
        for ticker in new_tickers:
            _warmup_tickers_in_progress.add(ticker)
    
    # Fire-and-forget: Start background threads only for new tickers
    def warmup_worker(ticker: str) -> None:
        try:
            _warmup_cache_for_ticker(ticker, host, port)
        finally:
            # Remove from in-progress set after fetch completes (or fails)
            # Note: We keep it in the set to prevent re-fetching, even if it failed
            # This ensures we only try once per ticker per process
            pass
    
    threads = []
    for ticker in new_tickers:
        # Start a non-daemon thread for each new ticker so we can wait for HTTP requests to be sent
        thread = threading.Thread(target=warmup_worker, args=(ticker,), daemon=False)
        thread.start()
        threads.append(thread)
    
    if len(new_tickers) > 0:
        print(f"Background cache warmup: Fetching stock info for {len(new_tickers)} new tickers (skipped {len(unique_tickers) - len(new_tickers)} already in progress)", file=sys.stderr)
        logger.debug(f"[CACHE WARMUP] Starting warmup for {len(new_tickers)} tickers via {host}:{port}")
        logger.debug(f"[CACHE WARMUP] Tickers: {', '.join(new_tickers[:10])}{'...' if len(new_tickers) > 10 else ''}")
        print(f"[CACHE WARMUP] Starting warmup for {len(new_tickers)} tickers", file=sys.stderr)
        print(f"[CACHE WARMUP] Database server: {host}:{port}", file=sys.stderr)
        print(f"[CACHE WARMUP] HTTP endpoint base: http://{host}:{port}/api/stock_info/", file=sys.stderr)
        print(f"[CACHE WARMUP] Tickers to warmup: {', '.join(new_tickers[:20])}{'...' if len(new_tickers) > 20 else ''}", file=sys.stderr)
        
        # Wait for threads to send HTTP requests (with timeout)
        # Wait at least 5 seconds to give jobs a chance to run (fire-and-forget)
        # This ensures the HTTP requests are actually sent before the script exits
        min_wait = 5.0
        max_wait = max(min_wait, 2.0 * len(new_tickers))
        start_time = time.time()
        
        for thread in threads:
            remaining_time = max_wait - (time.time() - start_time)
            if remaining_time <= 0:
                break
            thread.join(timeout=remaining_time)
        
        # Ensure we've waited at least min_wait seconds
        elapsed = time.time() - start_time
        if elapsed < min_wait:
            time.sleep(min_wait - elapsed)
            elapsed = time.time() - start_time
        
        logger.debug(f"[CACHE WARMUP] Waited {elapsed:.2f}s for HTTP requests to be sent")
        print(f"[CACHE WARMUP] Waited {elapsed:.2f}s for HTTP requests to be sent", file=sys.stderr)


def generate_html_output(df: pd.DataFrame, output_dir: str, db_server_host: str = "mm.kundu.dev", db_server_port: int = 9100, csv_source: str = None) -> None:
    """Generate HTML output with sortable table.
    
    Args:
        df: DataFrame with the results (used only for structure/metadata, not embedded in HTML)
        output_dir: Directory path where to create the HTML output
        db_server_host: Database server hostname (default: "mm.kundu.dev")
        db_server_port: Database server port (default: 9100)
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
    
    # Background cache warmup: Start fetching stock_info for all tickers (fire-and-forget)
    _warmup_stock_info_cache(df_display, host=db_server_host, port=db_server_port)
    
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

