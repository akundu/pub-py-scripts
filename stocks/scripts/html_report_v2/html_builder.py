"""
HTML structure generation.
"""

from datetime import datetime
from typing import List
from zoneinfo import ZoneInfo
import json


def build_html_document(
    title: str,
    css_content: str,
    js_content: str,
    body_content: str,
    external_js: bool = False,
    js_file: str = "app.js",
    api_config: dict = None
) -> str:
    """Build complete HTML document.
    
    Args:
        title: Page title
        css_content: CSS styles as string
        js_content: JavaScript code as string (used if external_js=False)
        body_content: HTML body content
        external_js: If True, use external JS file instead of inline
        js_file: Name of external JS file (if external_js=True)
        api_config: Dictionary with API configuration (server_host, server_port, csv_source)
        
    Returns:
        Complete HTML document as string
    """
    if external_js:
        # Use external JS file
        js_tag = f'    <script src="{js_file}"></script>'
        if api_config:
            # Inject API config as a script tag before the external JS
            api_config_json = json.dumps(api_config)
            js_tag = f"""    <script>
        window.API_CONFIG = {api_config_json};
    </script>
    <script src="{js_file}"></script>"""
    else:
        # Inline JS
        js_tag = f"""    <script>
{js_content}
    </script>"""
    
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title}</title>
    <style>
{css_content}
    </style>
</head>
<body>
{body_content}
{js_tag}
</body>
</html>
"""


def build_header(
    title: str,
    timestamp: str,
    iso_timestamp: str,
    total_results: int,
    tab_buttons: List[str]
) -> str:
    """Build HTML header section.
    
    Args:
        title: Report title
        timestamp: Formatted timestamp string (placeholder, will be updated by JS)
        iso_timestamp: ISO format timestamp for JavaScript (placeholder, will be updated by JS)
        total_results: Total number of results
        tab_buttons: List of tab button HTML strings
        
    Returns:
        Header HTML as string
    """
    tabs_html = '\n'.join(tab_buttons)
    
    return f"""    <div class="container">
        <div class="header">
            <h1>{title}</h1>
            <div class="tabs">
{tabs_html}
            </div>
            <p id="generatedTime" data-generated="">Data updated: <span id="dataTimestamp">Loading...</span> <span id="timeAgo"></span></p>
            <p class="desktop-only">Click column headers to sort • {total_results} total results</p>
            <p class="mobile-only">Tap cards to expand details • {total_results} total results</p>
        </div>
"""


def build_tab_button(tab_index: int, label: str, icon: str, is_active: bool = False) -> str:
    """Build a single tab button.
    
    Args:
        tab_index: Tab index (0-based)
        label: Button label text
        icon: Emoji icon
        is_active: Whether this tab is active by default
        
    Returns:
        Tab button HTML as string
    """
    active_class = ' active' if is_active else ''
    return f'                <button class="tab-button{active_class}" onclick="switchTab({tab_index})">{icon} {label}</button>'


def build_tab_content(content: str, prefix: str, is_active: bool = False) -> str:
    """Build tab content wrapper.
    
    Args:
        content: Tab content HTML
        prefix: Tab prefix ('calls', 'puts', or 'analysis')
        is_active: Whether this tab is active by default
        
    Returns:
        Tab content HTML as string
    """
    active_class = ' active' if is_active else ''
    return f"""        <div class="tab-content{active_class}" id="{prefix}Tab">
{content}
        </div>
"""


def get_timestamp_strings() -> tuple:
    """Get formatted and ISO timestamp strings.
    
    Returns:
        Tuple of (formatted_timestamp, iso_timestamp)
    """
    pacific = ZoneInfo("America/Los_Angeles")
    now = datetime.now(tz=pacific)
    formatted = now.strftime("%Y-%m-%d %H:%M:%S %Z")
    iso = now.isoformat()
    
    return formatted, iso


def get_title(has_calls: bool, has_puts: bool) -> str:
    """Get report title based on option types.
    
    Args:
        has_calls: Whether calls data exists
        has_puts: Whether puts data exists
        
    Returns:
        Title string
    """
    if has_calls and has_puts:
        return "📊 Options Analysis Results (Calls & Puts)"
    elif has_puts:
        return "📊 Cash-Secured Puts Analysis Results"
    else:
        return "📊 Covered Calls Analysis Results"

