"""
Main HTML report generator - orchestrates all modules.
"""

import logging
import pandas as pd
import sys
import threading
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


def _warmup_cache_for_ticker(ticker: str, host: str = "mm.kundu.dev", port: int = 9001) -> None:
    """Fire-and-forget cache warmup for a single ticker."""
    url = f"http://{host}:{port}/api/stock_info/{ticker}?show_iv=true&show_news=false"
    
    # Log the HTTP request destination
    import traceback
    try:
        # Get call stack to show where this is being called from
        stack = traceback.extract_stack()[-3:-1]
        caller_info = []
        for frame in stack:
            caller_info.append(f"  {frame.filename}:{frame.lineno} in {frame.name}")
        call_stack_str = '\n'.join(caller_info) if caller_info else "  (call stack unavailable)"
    except Exception:
        call_stack_str = "  (call stack unavailable)"
    
    logger.debug(f"[CACHE WARMUP] Sending HTTP request for {ticker}")
    print(f"[CACHE WARMUP] Sending HTTP request for {ticker}", file=sys.stderr)
    print(f"[CACHE WARMUP] HTTP endpoint URL: {url}", file=sys.stderr)
    print(f"[CACHE WARMUP] Called from:\n{call_stack_str}", file=sys.stderr)
    
    try:
        # Make request with short timeout, don't wait for response
        req = urllib.request.Request(url)
        print(f"[CACHE WARMUP] Making HTTP GET request to: {url}", file=sys.stderr)
        urllib.request.urlopen(req, timeout=1.0)
        success_msg = f"[CACHE WARMUP] Successfully sent request for {ticker} to {host}:{port}"
        logger.debug(success_msg)
        print(success_msg, file=sys.stderr)
    except (urllib.error.URLError, urllib.error.HTTPError, OSError, TimeoutError) as e:
        # Log errors in debug mode, but still ignore them (fire-and-forget)
        error_msg = f"[CACHE WARMUP] Error sending request for {ticker} to {url}: {type(e).__name__}: {e}"
        logger.debug(error_msg)
        print(error_msg, file=sys.stderr)
        pass


def _warmup_stock_info_cache(df: pd.DataFrame, host: str = "mm.kundu.dev", port: int = 9001) -> None:
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
    
    for ticker in new_tickers:
        # Start a daemon thread for each new ticker (fire-and-forget)
        thread = threading.Thread(target=warmup_worker, args=(ticker,), daemon=True)
        thread.start()
    
    if len(new_tickers) > 0:
        print(f"Background cache warmup: Fetching stock info for {len(new_tickers)} new tickers (skipped {len(unique_tickers) - len(new_tickers)} already in progress)", file=sys.stderr)
        logger.debug(f"[CACHE WARMUP] Starting warmup for {len(new_tickers)} tickers via {host}:{port}")
        logger.debug(f"[CACHE WARMUP] Tickers: {', '.join(new_tickers[:10])}{'...' if len(new_tickers) > 10 else ''}")
        print(f"[CACHE WARMUP] Starting warmup for {len(new_tickers)} tickers", file=sys.stderr)
        print(f"[CACHE WARMUP] Database server: {host}:{port}", file=sys.stderr)
        print(f"[CACHE WARMUP] HTTP endpoint base: http://{host}:{port}/api/stock_info/", file=sys.stderr)
        print(f"[CACHE WARMUP] Tickers to warmup: {', '.join(new_tickers[:20])}{'...' if len(new_tickers) > 20 else ''}", file=sys.stderr)


def generate_html_output(df: pd.DataFrame, output_dir: str, db_server_host: str = "mm.kundu.dev", db_server_port: int = 9001) -> None:
    """Generate HTML output with sortable table.
    
    Args:
        df: DataFrame with the results
        output_dir: Directory path where to create the HTML output
        db_server_host: Database server hostname (default: "mm.kundu.dev")
        db_server_port: Database server port (default: 9001)
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
    
    # Background cache warmup: Start fetching stock_info for all tickers (fire-and-forget)
    _warmup_stock_info_cache(df_display, host=db_server_host, port=db_server_port)
    
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

