#!/usr/bin/env python3
"""
HTML and text formatters for range percentiles data.
"""

TABLE_CELL_WIDTH = 14


def _table_row(cells: list[str], width: int) -> str:
    """Format a row of cells with given width per column."""
    return " | ".join(str(c).rjust(width) for c in cells)


def format_as_text_table(out: dict) -> str:
    """Format a single-ticker result dict as compact ASCII tables."""
    display_ticker = out["ticker"]
    last_date_str = out["last_trading_day"]
    previous_close = out["previous_close"]
    days = out["lookback_calendar_days"]
    n_data = out["lookback_days"]
    percentiles = out["percentiles"]
    when_up = out.get("when_up")
    when_down = out.get("when_down")
    n_up = out["when_up_day_count"]
    n_down = out["when_down_day_count"]
    min_direction_days = out.get("min_direction_days", 5)
    window = out.get("window", 1)

    w = TABLE_CELL_WIDTH
    header_cells = [f"p{p}" for p in percentiles]
    sep = "-+-".join("-" * w for _ in percentiles)

    try:
        prev_close_short = f"{float(previous_close):,.2f}"
    except (TypeError, ValueError):
        prev_close_short = str(previous_close)
    close_note = " (manual)" if out.get("close_override") else ""
    window_note = f"  window: {window}d" if window > 1 else ""
    lines = [
        f"{display_ticker}  last: {last_date_str}  close: {prev_close_short}{close_note}  ({days} calendar days, {n_data} trading days){window_note}",
        "",
    ]

    def emit_block(block: dict | None, n: int, direction_label: str) -> None:
        if block is None:
            lines.append(f"  {direction_label}: n={n} (need >={min_direction_days})")
            lines.append("")
            return
        lines.append(f"  {direction_label} (n={n})")
        lines.append("  " + _table_row(header_cells, w))
        lines.append("  " + sep)
        pcts = [f"{block['pct'][f'p{p}']}%" for p in percentiles]
        lines.append("  " + _table_row(pcts, w))
        prices = [str(round(float(block["price"][f"p{p}"]), 2)) for p in percentiles]
        lines.append("  " + _table_row(prices, w))
        lines.append("")

    emit_block(when_down, n_down, "DOWN")
    emit_block(when_up, n_up, "UP")
    return "\n".join(lines).strip()


def format_multi_window_as_text_table(out: dict) -> str:
    """
    Format multi-window result as transposed ASCII table.

    Layout:
    - Header: ticker metadata
    - Separate tables for DOWN and UP
    - Rows: percentiles (p75, p90, p95, ...)
    - Columns: window sizes (1d, 3d, 5d, ...) with % and price sub-columns

    Args:
        out: Multi-window result dict from compute_range_percentiles_multi_window

    Returns:
        Formatted ASCII table as string
    """
    display_ticker = out["ticker"]
    metadata = out["metadata"]
    last_date_str = metadata["last_trading_day"]
    previous_close = metadata["previous_close"]
    days = metadata["lookback_calendar_days"]
    n_data = metadata.get("lookback_days", "N/A")
    percentiles = metadata["percentiles"]
    window_list = metadata["window_list"]
    skipped_windows = metadata.get("skipped_windows", [])
    windows_data = out["windows"]

    try:
        prev_close_short = f"{float(previous_close):,.2f}"
    except (TypeError, ValueError):
        prev_close_short = str(previous_close)

    close_note = " (manual)" if metadata.get("close_override") else ""

    lines = [
        f"{display_ticker}  last: {last_date_str}  close: {prev_close_short}{close_note}  ({days} calendar days, {n_data} trading days)",
        f"Multi-window analysis: {len(window_list)} windows",
        "",
    ]

    if skipped_windows:
        lines.append(f"⚠ Skipped windows (insufficient data): {', '.join(map(str, skipped_windows))}")
        lines.append("")

    # Table formatting
    w_label = 6
    w_col = 11

    # DOWN MOVES table
    lines.append("DOWN MOVES")
    lines.append("")

    # Header row - window labels
    header_parts = ["".ljust(w_label)]
    for window in window_list:
        header_parts.append(f"{window}d".center(w_col * 2 + 3))
    lines.append(" | ".join(header_parts))

    # Sub-header row - % and $ for each window
    subheader_parts = ["".ljust(w_label)]
    for _ in window_list:
        subheader_parts.append(f"%".rjust(w_col))
        subheader_parts.append(f"$".rjust(w_col))
    lines.append(" | ".join(subheader_parts))

    # Separator
    sep_parts = ["-" * w_label]
    for _ in window_list:
        sep_parts.append("-" * w_col)
        sep_parts.append("-" * w_col)
    lines.append("-+-".join(sep_parts))

    # Data rows - one per percentile
    for p in percentiles:
        row_parts = [f"p{p}".ljust(w_label)]

        for window in window_list:
            win_data = windows_data.get(str(window))
            if win_data and win_data.get("when_down"):
                when_down = win_data["when_down"]
                pct = when_down["pct"][f"p{p}"]
                price = when_down["price"][f"p{p}"]
                row_parts.append(f"{pct}%".rjust(w_col))
                row_parts.append(f"{price:,.2f}".rjust(w_col))
            else:
                row_parts.append("--".rjust(w_col))
                row_parts.append("--".rjust(w_col))

        lines.append(" | ".join(row_parts))

    lines.append("")
    lines.append("")

    # UP MOVES table
    lines.append("UP MOVES")
    lines.append("")

    # Header row - window labels
    header_parts = ["".ljust(w_label)]
    for window in window_list:
        header_parts.append(f"{window}d".center(w_col * 2 + 3))
    lines.append(" | ".join(header_parts))

    # Sub-header row - % and $ for each window
    subheader_parts = ["".ljust(w_label)]
    for _ in window_list:
        subheader_parts.append(f"%".rjust(w_col))
        subheader_parts.append(f"$".rjust(w_col))
    lines.append(" | ".join(subheader_parts))

    # Separator
    lines.append("-+-".join(sep_parts))

    # Data rows - one per percentile
    for p in percentiles:
        row_parts = [f"p{p}".ljust(w_label)]

        for window in window_list:
            win_data = windows_data.get(str(window))
            if win_data and win_data.get("when_up"):
                when_up = win_data["when_up"]
                pct = when_up["pct"][f"p{p}"]
                price = when_up["price"][f"p{p}"]
                row_parts.append(f"+{pct}%".rjust(w_col))
                row_parts.append(f"{price:,.2f}".rjust(w_col))
            else:
                row_parts.append("--".rjust(w_col))
                row_parts.append("--".rjust(w_col))

        lines.append(" | ".join(row_parts))

    return "\n".join(lines).strip()


def format_as_html(results: list[dict], params: dict = None) -> str:
    from datetime import date as _date
    """
    Format range percentiles results as a styled HTML page.

    Args:
        results: List of result dicts from compute_range_percentiles_multi
        params: Optional dict of parameters used (for display in header)

    Returns:
        Complete HTML page as string
    """
    if not results:
        return "<html><body><h1>No results</h1></body></html>"

    params = params or {}
    window = results[0].get("window", 1)
    days = results[0].get("lookback_calendar_days", "N/A")
    n_data = results[0].get("lookback_days", "N/A")

    # Build HTML
    html_parts = [
        """<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Range Percentiles Analysis</title>
    <style>
        :root {
            --bg-primary: #f5f5f5;
            --bg-secondary: white;
            --text-primary: #333;
            --text-secondary: #666;
            --text-tertiary: #7f8c8d;
            --text-quaternary: #999;
            --border-color: #ecf0f1;
            --table-header-bg: #34495e;
            --table-header-text: white;
            --table-hover: #f8f9fa;
            --shadow: rgba(0,0,0,0.1);
            --down-bg: #ffe5e5;
            --down-text: #c0392b;
            --up-bg: #e5ffe5;
            --up-text: #27ae60;
            --insufficient-bg: #f0f0f0;
            --insufficient-text: #999;
            --heading-color: #2c3e50;
        }

        /* Dark mode */
        @media (prefers-color-scheme: dark) {
            :root {
                --bg-primary: #1a1a1a;
                --bg-secondary: #2d2d2d;
                --text-primary: #e0e0e0;
                --text-secondary: #b0b0b0;
                --text-tertiary: #888;
                --text-quaternary: #666;
                --border-color: #444;
                --table-header-bg: #1e3a5f;
                --table-header-text: #e0e0e0;
                --table-hover: #3a3a3a;
                --shadow: rgba(0,0,0,0.5);
                --down-bg: #4a2020;
                --down-text: #ff6b6b;
                --up-bg: #204a20;
                --up-text: #6bff6b;
                --insufficient-bg: #3a3a3a;
                --insufficient-text: #888;
                --heading-color: #e0e0e0;
            }
        }

        /* Manual dark mode override */
        [data-theme="dark"] {
            --bg-primary: #1a1a1a;
            --bg-secondary: #2d2d2d;
            --text-primary: #e0e0e0;
            --text-secondary: #b0b0b0;
            --text-tertiary: #888;
            --text-quaternary: #666;
            --border-color: #444;
            --table-header-bg: #1e3a5f;
            --table-header-text: #e0e0e0;
            --table-hover: #3a3a3a;
            --shadow: rgba(0,0,0,0.5);
            --down-bg: #4a2020;
            --down-text: #ff6b6b;
            --up-bg: #204a20;
            --up-text: #6bff6b;
            --insufficient-bg: #3a3a3a;
            --insufficient-text: #888;
            --heading-color: #e0e0e0;
        }

        /* Manual light mode override */
        [data-theme="light"] {
            --bg-primary: #f5f5f5;
            --bg-secondary: white;
            --text-primary: #333;
            --text-secondary: #666;
            --text-tertiary: #7f8c8d;
            --text-quaternary: #999;
            --border-color: #ecf0f1;
            --table-header-bg: #34495e;
            --table-header-text: white;
            --table-hover: #f8f9fa;
            --shadow: rgba(0,0,0,0.1);
            --down-bg: #ffe5e5;
            --down-text: #c0392b;
            --up-bg: #e5ffe5;
            --up-text: #27ae60;
            --insufficient-bg: #f0f0f0;
            --insufficient-text: #999;
            --heading-color: #2c3e50;
        }

        body {
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
            max-width: 1600px;
            margin: 0 auto;
            padding: 20px;
            background: var(--bg-primary);
            color: var(--text-primary);
            font-size: 16px;
        }
        .header {
            background: var(--bg-secondary);
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 4px var(--shadow);
            margin-bottom: 20px;
            position: relative;
        }
        h1 {
            margin: 0 0 10px 0;
            color: var(--heading-color);
            font-size: 32px;
        }
        .params {
            color: var(--text-secondary);
            font-size: 15px;
            margin-top: 10px;
        }
        .params span {
            display: inline-block;
            margin-right: 20px;
            padding: 6px 12px;
            background: var(--border-color);
            border-radius: 4px;
        }
        .theme-toggle {
            position: absolute;
            top: 20px;
            right: 20px;
            padding: 10px 18px;
            background: var(--bg-primary);
            border: 1px solid var(--border-color);
            border-radius: 6px;
            cursor: pointer;
            font-size: 15px;
            color: var(--text-primary);
            transition: all 0.2s;
        }
        .theme-toggle:hover {
            background: var(--table-hover);
        }
        .ticker-section {
            background: var(--bg-secondary);
            padding: 25px;
            border-radius: 8px;
            box-shadow: 0 2px 4px var(--shadow);
            margin-bottom: 25px;
        }
        .ticker-header {
            font-size: 28px;
            font-weight: bold;
            color: var(--heading-color);
            margin-bottom: 8px;
        }
        .ticker-info {
            color: var(--text-secondary);
            font-size: 15px;
            margin-bottom: 20px;
        }
        .ticker-info span {
            margin-right: 20px;
        }
        .direction-section {
            margin-bottom: 40px;
        }
        .direction-header {
            font-size: 18px;
            font-weight: 700;
            margin-bottom: 25px;
            padding: 14px 20px;
            border-radius: 8px;
            display: inline-block;
            box-shadow: 0 2px 8px var(--shadow);
            letter-spacing: 0.5px;
            text-transform: uppercase;
        }
        .direction-header.down {
            background: linear-gradient(135deg, var(--down-bg) 0%, var(--down-bg) 100%);
            color: var(--down-text);
            border-left: 4px solid var(--down-text);
        }
        .direction-header.up {
            background: linear-gradient(135deg, var(--up-bg) 0%, var(--up-bg) 100%);
            color: var(--up-text);
            border-left: 4px solid var(--up-text);
        }
        .direction-header.insufficient {
            background: var(--insufficient-bg);
            color: var(--insufficient-text);
        }
        table {
            width: 100%;
            border-collapse: separate;
            border-spacing: 0;
            margin-top: 0;
            font-size: 16px;
            border: none;
            border-radius: 10px;
            overflow: hidden;
            box-shadow: 0 4px 12px var(--shadow);
        }
        th {
            background: linear-gradient(180deg, #3d5a80 0%, #2c4259 100%);
            color: #ffffff;
            padding: 16px 18px;
            text-align: center;
            font-weight: 700;
            font-size: 16px;
            border-right: 1px solid rgba(255,255,255,0.15);
            text-transform: uppercase;
            letter-spacing: 0.5px;
            position: relative;
        }
        th::after {
            content: '';
            position: absolute;
            bottom: 0;
            left: 10%;
            right: 10%;
            height: 2px;
            background: linear-gradient(90deg, transparent, rgba(255,255,255,0.3), transparent);
        }
        th:last-child {
            border-right: none;
        }
        td {
            padding: 14px 16px;
            text-align: center;
            border-bottom: 1px solid var(--border-color);
            border-right: 2px solid var(--border-color);
        }
        td:last-child {
            border-right: none;
        }
        tr:last-child td {
            border-bottom: none;
        }
        tr:hover td {
            background: var(--table-hover);
        }
        .percentile-cell {
            display: flex;
            flex-direction: column;
            align-items: center;
            gap: 6px;
        }
        .percentile-pct {
            font-weight: 700;
            font-size: 17px;
            color: var(--heading-color);
        }
        .percentile-price {
            font-size: 15px;
            color: var(--text-secondary);
            font-weight: 500;
        }
        .note {
            color: var(--text-quaternary);
            font-size: 14px;
            font-style: italic;
            margin-top: 8px;
        }
    </style>
</head>
<body>
    <div class="header">
        <button class="theme-toggle" onclick="toggleTheme()" id="themeToggle">🌙 Dark</button>
        <h1>📊 Range Percentiles Analysis</h1>
        <div class="params">
"""
    ]

    today_str = _date.today().strftime('%Y-%m-%d')
    html_parts.append(f'            <span><strong>Window:</strong> {window} trading day{"s" if window != 1 else ""}</span>')
    html_parts.append(f'            <span><strong>Lookback:</strong> {days} calendar days ({n_data} trading days)</span>')
    html_parts.append(f'            <span><strong>Updated:</strong> {today_str}</span>')

    if params.get("tickers"):
        html_parts.append(f'            <span><strong>Tickers:</strong> {", ".join(params["tickers"])}</span>')

    html_parts.append("""
        </div>
    </div>
""")

    # Add each ticker section
    for result in results:
        ticker = result["ticker"]
        last_date = result["last_trading_day"]
        prev_close = result["previous_close"]
        n_data = result["lookback_days"]
        percentiles = result["percentiles"]
        when_up = result.get("when_up")
        when_down = result.get("when_down")
        n_up = result["when_up_day_count"]
        n_down = result["when_down_day_count"]
        min_direction_days = result.get("min_direction_days", 5)

        try:
            prev_close_fmt = f"${float(prev_close):,.2f}"
        except (TypeError, ValueError):
            prev_close_fmt = str(prev_close)

        html_parts.append(f"""
    <div class="ticker-section">
        <div class="ticker-header">{ticker}</div>
        <div class="ticker-info">
            <span><strong>Last Trading Day:</strong> {last_date}</span>
            <span><strong>Close:</strong> {prev_close_fmt}</span>
            <span><strong>Data Points:</strong> {n_data} days</span>
        </div>
""")

        # DOWN section
        if when_down is None:
            html_parts.append(f"""
        <div class="direction-section">
            <div class="direction-header insufficient">⬇️ DOWN MOVES (n={n_down})</div>
            <div class="note">Insufficient data (need ≥{min_direction_days} days)</div>
        </div>
""")
        else:
            html_parts.append(f"""
        <div class="direction-section">
            <div class="direction-header down">⬇️ DOWN MOVES (n={n_down})</div>
            <table>
                <thead>
                    <tr>
""")
            for p in percentiles:
                html_parts.append(f'                        <th>p{p}</th>\n')
            html_parts.append("""                    </tr>
                </thead>
                <tbody>
                    <tr>
""")
            for p in percentiles:
                pct = when_down["pct"][f"p{p}"]
                price = when_down["price"][f"p{p}"]
                html_parts.append(f'''                        <td>
                            <div class="percentile-cell">
                                <div class="percentile-pct">{pct}%</div>
                                <div class="percentile-price">${price:,.2f}</div>
                            </div>
                        </td>
''')
            html_parts.append("""                    </tr>
                </tbody>
            </table>
        </div>
""")

        # UP section
        if when_up is None:
            html_parts.append(f"""
        <div class="direction-section">
            <div class="direction-header insufficient">⬆️ UP MOVES (n={n_up})</div>
            <div class="note">Insufficient data (need ≥{min_direction_days} days)</div>
        </div>
""")
        else:
            html_parts.append(f"""
        <div class="direction-section">
            <div class="direction-header up">⬆️ UP MOVES (n={n_up})</div>
            <table>
                <thead>
                    <tr>
""")
            for p in percentiles:
                html_parts.append(f'                        <th>p{p}</th>\n')
            html_parts.append("""                    </tr>
                </thead>
                <tbody>
                    <tr>
""")
            for p in percentiles:
                pct = when_up["pct"][f"p{p}"]
                price = when_up["price"][f"p{p}"]
                html_parts.append(f'''                        <td>
                            <div class="percentile-cell">
                                <div class="percentile-pct">+{pct}%</div>
                                <div class="percentile-price">${price:,.2f}</div>
                            </div>
                        </td>
''')
            html_parts.append("""                    </tr>
                </tbody>
            </table>
        </div>
""")

        html_parts.append("""    </div>

    <script>
        // Theme switching functions
        function toggleTheme() {
            const html = document.documentElement;
            const currentTheme = html.getAttribute('data-theme');
            const newTheme = currentTheme === 'dark' ? 'light' : 'dark';
            html.setAttribute('data-theme', newTheme);
            updateThemeButton(newTheme);
            localStorage.setItem('theme', newTheme);
        }

        function updateThemeButton(theme) {
            const button = document.getElementById('themeToggle');
            if (button) {
                button.textContent = theme === 'dark' ? '☀️ Light' : '🌙 Dark';
            }
        }

        function initTheme() {
            // Check URL parameter first
            const urlParams = new URLSearchParams(window.location.search);
            const urlTheme = urlParams.get('theme');

            if (urlTheme === 'dark' || urlTheme === 'light') {
                document.documentElement.setAttribute('data-theme', urlTheme);
                updateThemeButton(urlTheme);
                localStorage.setItem('theme', urlTheme);
                return;
            }

            // Check localStorage
            const savedTheme = localStorage.getItem('theme');
            if (savedTheme) {
                document.documentElement.setAttribute('data-theme', savedTheme);
                updateThemeButton(savedTheme);
                return;
            }

            // Check system preference
            if (window.matchMedia && window.matchMedia('(prefers-color-scheme: dark)').matches) {
                document.documentElement.setAttribute('data-theme', 'dark');
                updateThemeButton('dark');
            } else {
                document.documentElement.setAttribute('data-theme', 'light');
                updateThemeButton('light');
            }
        }

        // Initialize theme on page load
        initTheme();

        // Listen for system theme changes
        if (window.matchMedia) {
            window.matchMedia('(prefers-color-scheme: dark)').addEventListener('change', function(e) {
                // Only auto-switch if user hasn't manually set a preference
                if (!localStorage.getItem('theme')) {
                    const newTheme = e.matches ? 'dark' : 'light';
                    document.documentElement.setAttribute('data-theme', newTheme);
                    updateThemeButton(newTheme);
                }
            });
        }
    </script>
</body>
</html>""")

    return "".join(html_parts)


# Percentile colors for Chart.js
PERCENTILE_COLORS = {
    75: "rgb(54, 162, 235)",   # Blue
    90: "rgb(255, 206, 86)",   # Yellow
    95: "rgb(255, 159, 64)",   # Orange
    98: "rgb(255, 99, 132)",   # Red
    99: "rgb(153, 102, 255)",  # Purple
    100: "rgb(201, 203, 207)", # Gray
}


def _generate_ticker_content_html(result: dict, ticker_id: str) -> tuple[str, dict, dict]:
    """
    Generate HTML content for a single ticker's multi-window analysis.

    Returns:
        Tuple of (html_content, chart_data_down, chart_data_up)
    """
    from datetime import date as _date, timedelta as _timedelta
    today = _date.today()
    today_str = today.strftime('%Y-%m-%d')

    ticker = result["ticker"]
    metadata = result["metadata"]
    last_date = metadata["last_trading_day"]
    prev_close = metadata["previous_close"]
    percentiles = metadata["percentiles"]
    window_list = metadata["window_list"]
    skipped_windows = metadata.get("skipped_windows", [])
    windows_data = result["windows"]

    try:
        prev_close_fmt = f"${float(prev_close):,.2f}"
    except (TypeError, ValueError):
        prev_close_fmt = str(prev_close)

    # Generate chart data
    chart_data_down = _generate_chart_data(windows_data, window_list, percentiles, "down", prev_close)
    chart_data_up = _generate_chart_data(windows_data, window_list, percentiles, "up", prev_close)

    html_parts = []

    # Ticker info header
    html_parts.append(f"""
        <div class="ticker-info-header">
            <span><strong>Last Trading Day:</strong> {last_date}</span>
            <span><strong>Close:</strong> {prev_close_fmt}</span>
        </div>
""")

    if skipped_windows:
        html_parts.append(f"""
        <div class="warning">
            ⚠️ <strong>Warning:</strong> Some windows were skipped due to insufficient data: {', '.join(map(str, skipped_windows))}
        </div>
""")

    # DOWN MOVES table
    html_parts.append("""
        <div class="section">
            <div class="direction-header down">⬇️ DOWN MOVES</div>
            <div class="table-section">
                <table>
                    <thead>
                        <tr>
                            <th></th>
""")

    for i, window in enumerate(window_list):
        # Window represents N days ahead: window=0 → 0DTE (today), window=1 → 1DTE (tomorrow), etc.
        dte = window
        if dte == 0:
            date_str = today_str
            label = "0d"
        else:
            # Approximate calendar date (trading days → calendar days with weekends)
            calendar_days_approx = int(dte * 1.4)
            date_str = (today + _timedelta(days=calendar_days_approx)).strftime('%Y-%m-%d')
            label = f"{dte}d"
        html_parts.append(f'                            <th colspan="2">{label}<br><small style="font-weight:normal;font-size:11px;opacity:0.8">{date_str}</small></th>\n')

    html_parts.append("""                        </tr>
                        <tr>
                            <th></th>
""")

    for _ in window_list:
        html_parts.append('                            <th>%</th>\n')
        html_parts.append('                            <th>$</th>\n')

    html_parts.append("""                        </tr>
                    </thead>
                    <tbody>
""")

    for p in percentiles:
        html_parts.append(f'                        <tr>\n')
        html_parts.append(f'                            <td>p{p}</td>\n')

        for window in window_list:
            win_data = windows_data.get(str(window))
            if win_data and win_data.get("when_down"):
                when_down = win_data["when_down"]
                pct = when_down["pct"][f"p{p}"]
                price = when_down["price"][f"p{p}"]
                html_parts.append(f'                            <td>{pct}%</td>\n')
                html_parts.append(f'                            <td>${price:,.2f}</td>\n')
            else:
                html_parts.append(f'                            <td class="insufficient">--</td>\n')
                html_parts.append(f'                            <td class="insufficient">--</td>\n')

        html_parts.append(f'                        </tr>\n')

    html_parts.append(f"""                    </tbody>
                </table>
            </div>

            <h2>DOWN Moves - Percentile Evolution</h2>
            <div class="chart-container">
                <canvas id="chartDown_{ticker_id}"></canvas>
            </div>
        </div>
""")

    # UP MOVES table
    html_parts.append("""
        <div class="section">
            <div class="direction-header up">⬆️ UP MOVES</div>
            <div class="table-section">
                <table>
                    <thead>
                        <tr>
                            <th></th>
""")

    for i, window in enumerate(window_list):
        # Window represents N days ahead: window=0 → 0DTE (today), window=1 → 1DTE (tomorrow), etc.
        dte = window
        if dte == 0:
            date_str = today_str
            label = "0d"
        else:
            # Approximate calendar date (trading days → calendar days with weekends)
            calendar_days_approx = int(dte * 1.4)
            date_str = (today + _timedelta(days=calendar_days_approx)).strftime('%Y-%m-%d')
            label = f"{dte}d"
        html_parts.append(f'                            <th colspan="2">{label}<br><small style="font-weight:normal;font-size:11px;opacity:0.8">{date_str}</small></th>\n')

    html_parts.append("""                        </tr>
                        <tr>
                            <th></th>
""")

    for _ in window_list:
        html_parts.append('                            <th>%</th>\n')
        html_parts.append('                            <th>$</th>\n')

    html_parts.append("""                        </tr>
                    </thead>
                    <tbody>
""")

    for p in percentiles:
        html_parts.append(f'                        <tr>\n')
        html_parts.append(f'                            <td>p{p}</td>\n')

        for window in window_list:
            win_data = windows_data.get(str(window))
            if win_data and win_data.get("when_up"):
                when_up = win_data["when_up"]
                pct = when_up["pct"][f"p{p}"]
                price = when_up["price"][f"p{p}"]
                html_parts.append(f'                            <td>+{pct}%</td>\n')
                html_parts.append(f'                            <td>${price:,.2f}</td>\n')
            else:
                html_parts.append(f'                            <td class="insufficient">--</td>\n')
                html_parts.append(f'                            <td class="insufficient">--</td>\n')

        html_parts.append(f'                        </tr>\n')

    html_parts.append(f"""                    </tbody>
                </table>
            </div>

            <h2>UP Moves - Percentile Evolution</h2>
            <div class="chart-container">
                <canvas id="chartUp_{ticker_id}"></canvas>
            </div>
        </div>
""")

    return ("".join(html_parts), chart_data_down, chart_data_up)


def _generate_chart_data(windows_dict: dict, window_list: list[int],
                         percentiles: list[int], direction: str, reference_close: float) -> dict:
    """
    Convert multi-window data to Chart.js format with dual y-axes support.

    Args:
        windows_dict: The 'windows' dict from result
        window_list: List of window sizes [1, 3, 5, ...]
        percentiles: List of percentiles [75, 90, 95, ...]
        direction: 'up' or 'down'
        reference_close: Reference close price for price calculations

    Returns:
        Chart.js data structure with labels, datasets, and price range info
    """
    labels = window_list
    datasets = []
    direction_key = f"when_{direction}"

    # Track min/max for price scale calculation
    all_pcts = []

    for p in percentiles:
        data_points = []
        for window in window_list:
            win_data = windows_dict.get(str(window))
            if win_data and win_data.get(direction_key):
                pct = win_data[direction_key]["pct"][f"p{p}"]
                data_points.append(pct)
                all_pcts.append(pct)
            else:
                # No data for this window/percentile combo
                data_points.append(None)

        color = PERCENTILE_COLORS.get(p, "rgb(150, 150, 150)")
        datasets.append({
            "label": f"p{p}",
            "data": data_points,
            "borderColor": color,
            "backgroundColor": color.replace("rgb", "rgba").replace(")", ", 0.1)"),
            "tension": 0.1,
            "pointRadius": 4,
            "pointHoverRadius": 6,
            "yAxisID": "yPercent",
        })

    # Calculate price range for right y-axis
    if all_pcts:
        min_pct = min(all_pcts)
        max_pct = max(all_pcts)
        # Add 10% padding
        pct_range = max_pct - min_pct
        min_pct_padded = min_pct - pct_range * 0.1
        max_pct_padded = max_pct + pct_range * 0.1

        # Convert to prices
        min_price = reference_close * (1 + min_pct_padded / 100)
        max_price = reference_close * (1 + max_pct_padded / 100)
    else:
        min_price = reference_close * 0.95
        max_price = reference_close * 1.05

    return {
        "labels": labels,
        "datasets": datasets,
        "priceRange": {
            "min": min_price,
            "max": max_price,
            "referenceClose": reference_close
        }
    }


def format_multi_window_as_html(result: dict | list[dict], params: dict = None, multi_ticker: bool = False) -> str:
    """
    Format multi-window analysis as styled HTML with tables and charts.

    Includes:
    - Metadata header
    - Transposed tables (same layout as CLI)
    - Interactive Chart.js graphs (one per direction: UP/DOWN)
    - Charts show percentile evolution across windows
    - Multi-ticker support with tabbed interface

    Args:
        result: Multi-window result dict or list of dicts from compute_range_percentiles_multi_window_batch
        params: Optional dict of parameters used (for display in header)
        multi_ticker: If True, render tabbed interface for multiple tickers

    Returns:
        Complete HTML page as string
    """
    import json
    from datetime import date as _date
    params = params or {}

    # Handle both single result and list of results
    if isinstance(result, list):
        results = result
    else:
        results = [result]
        multi_ticker = False

    if not results:
        return "<html><body><h1>No results</h1></body></html>"

    # Get common metadata from first result
    first_result = results[0]
    days = first_result["metadata"]["lookback_calendar_days"]
    n_data = first_result["metadata"].get("lookback_days", "N/A")
    window_list = first_result["metadata"]["window_list"]
    percentiles = first_result["metadata"]["percentiles"]

    # Generate ticker list for title
    tickers_str = ", ".join([r["ticker"] for r in results])

    # Build HTML header with styles
    html_parts = [
        """<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Multi-Window Range Analysis - """ + tickers_str + """</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0"></script>
    <style>
        /* Light mode (default) */
        :root {
            --bg-primary: #f5f5f5;
            --bg-secondary: white;
            --text-primary: #333;
            --text-secondary: #666;
            --text-tertiary: #7f8c8d;
            --border-color: #ecf0f1;
            --table-header-bg: #34495e;
            --table-header-text: white;
            --table-hover: #f8f9fa;
            --shadow: rgba(0,0,0,0.1);
            --warning-bg: #fff3cd;
            --warning-border: #ffc107;
            --warning-text: #856404;
            --down-bg: #ffe5e5;
            --down-text: #c0392b;
            --up-bg: #e5ffe5;
            --up-text: #27ae60;
            --tab-inactive: #ecf0f1;
            --tab-inactive-hover: #bdc3c7;
            --tab-active: #3498db;
        }

        /* Dark mode */
        @media (prefers-color-scheme: dark) {
            :root {
                --bg-primary: #1a1a1a;
                --bg-secondary: #2d2d2d;
                --text-primary: #e0e0e0;
                --text-secondary: #b0b0b0;
                --text-tertiary: #888;
                --border-color: #444;
                --table-header-bg: #1e3a5f;
                --table-header-text: #e0e0e0;
                --table-hover: #3a3a3a;
                --shadow: rgba(0,0,0,0.5);
                --warning-bg: #4a3f1f;
                --warning-border: #806520;
                --warning-text: #f0d090;
                --down-bg: #4a2020;
                --down-text: #ff6b6b;
                --up-bg: #204a20;
                --up-text: #6bff6b;
                --tab-inactive: #3a3a3a;
                --tab-inactive-hover: #4a4a4a;
                --tab-active: #2874a6;
            }
        }

        /* Manual dark mode override */
        [data-theme="dark"] {
            --bg-primary: #1a1a1a;
            --bg-secondary: #2d2d2d;
            --text-primary: #e0e0e0;
            --text-secondary: #b0b0b0;
            --text-tertiary: #888;
            --border-color: #444;
            --table-header-bg: #1e3a5f;
            --table-header-text: #e0e0e0;
            --table-hover: #3a3a3a;
            --shadow: rgba(0,0,0,0.5);
            --warning-bg: #4a3f1f;
            --warning-border: #806520;
            --warning-text: #f0d090;
            --down-bg: #4a2020;
            --down-text: #ff6b6b;
            --up-bg: #204a20;
            --up-text: #6bff6b;
            --tab-inactive: #3a3a3a;
            --tab-inactive-hover: #4a4a4a;
            --tab-active: #2874a6;
        }

        /* Manual light mode override */
        [data-theme="light"] {
            --bg-primary: #f5f5f5;
            --bg-secondary: white;
            --text-primary: #333;
            --text-secondary: #666;
            --text-tertiary: #7f8c8d;
            --border-color: #ecf0f1;
            --table-header-bg: #34495e;
            --table-header-text: white;
            --table-hover: #f8f9fa;
            --shadow: rgba(0,0,0,0.1);
            --warning-bg: #fff3cd;
            --warning-border: #ffc107;
            --warning-text: #856404;
            --down-bg: #ffe5e5;
            --down-text: #c0392b;
            --up-bg: #e5ffe5;
            --up-text: #27ae60;
            --tab-inactive: #ecf0f1;
            --tab-inactive-hover: #bdc3c7;
            --tab-active: #3498db;
        }

        body {
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
            max-width: 1800px;
            margin: 0 auto;
            padding: 20px;
            background: var(--bg-primary);
            color: var(--text-primary);
            transition: background-color 0.3s, color 0.3s;
            font-size: 16px;
        }
        .header {
            background: var(--bg-secondary);
            padding: 25px;
            border-radius: 8px;
            box-shadow: 0 2px 4px var(--shadow);
            margin-bottom: 25px;
            position: relative;
        }
        h1 {
            margin: 0 0 12px 0;
            color: var(--text-primary);
            font-size: 32px;
        }
        h2 {
            color: var(--text-primary);
            font-size: 26px;
            margin-top: 0;
        }
        .params {
            color: var(--text-secondary);
            font-size: 15px;
            margin-top: 12px;
        }
        .params span {
            display: inline-block;
            margin-right: 20px;
            padding: 6px 12px;
            background: var(--tab-inactive);
            border-radius: 4px;
        }
        .ticker-info-header {
            color: var(--text-secondary);
            font-size: 15px;
            margin-bottom: 18px;
        }
        .ticker-info-header span {
            margin-right: 25px;
        }
        .warning {
            background: var(--warning-bg);
            border: 1px solid var(--warning-border);
            color: var(--warning-text);
            padding: 10px;
            border-radius: 4px;
            margin-bottom: 20px;
        }
        .section {
            background: var(--bg-secondary);
            padding: 25px;
            border-radius: 8px;
            box-shadow: 0 2px 4px var(--shadow);
            margin-bottom: 25px;
        }
        .theme-toggle {
            position: absolute;
            top: 20px;
            right: 20px;
            padding: 10px 18px;
            background: var(--tab-inactive);
            border: none;
            border-radius: 6px;
            cursor: pointer;
            font-size: 15px;
            color: var(--text-primary);
            transition: background 0.3s;
        }
        .theme-toggle:hover {
            background: var(--tab-inactive-hover);
        }
        .table-section {
            overflow-x: auto;
        }
        table {
            width: 100%;
            border-collapse: separate;
            border-spacing: 0;
            font-size: 15px;
            border: none;
            border-radius: 10px;
            overflow: hidden;
            box-shadow: 0 4px 12px var(--shadow);
        }
        th {
            background: linear-gradient(180deg, #3d5a80 0%, #2c4259 100%);
            color: #ffffff;
            padding: 14px 12px;
            text-align: center;
            font-weight: 700;
            position: sticky;
            top: 0;
            font-size: 15px;
            border-right: 1px solid rgba(255,255,255,0.15);
            text-transform: uppercase;
            letter-spacing: 0.5px;
            position: relative;
        }
        th::after {
            content: '';
            position: absolute;
            bottom: 0;
            left: 10%;
            right: 10%;
            height: 2px;
            background: linear-gradient(90deg, transparent, rgba(255,255,255,0.3), transparent);
        }
        th:last-child {
            border-right: none;
        }
        th:first-child {
            text-align: left;
            padding-left: 15px;
        }
        td {
            padding: 12px 10px;
            text-align: center;
            border-bottom: 1px solid var(--border-color);
            border-right: 1px solid var(--border-color);
            font-size: 15px;
        }
        td:last-child {
            border-right: none;
        }
        tr:last-child td {
            border-bottom: none;
        }
        td:first-child {
            text-align: left;
            font-weight: 700;
            color: var(--text-tertiary);
            padding-left: 15px;
            font-size: 15px;
        }
        tr:hover td {
            background: var(--table-hover);
        }
        .chart-container {
            position: relative;
            width: 100%;
            height: 400px;
            margin: 20px 0;
        }
        .direction-header {
            font-size: 18px;
            font-weight: 700;
            margin-bottom: 25px;
            padding: 14px 20px;
            border-radius: 8px;
            display: inline-block;
            box-shadow: 0 2px 8px var(--shadow);
            letter-spacing: 0.5px;
            text-transform: uppercase;
        }
        .direction-header.down {
            background: linear-gradient(135deg, var(--down-bg) 0%, var(--down-bg) 100%);
            color: var(--down-text);
            border-left: 4px solid var(--down-text);
        }
        .direction-header.up {
            background: linear-gradient(135deg, var(--up-bg) 0%, var(--up-bg) 100%);
            color: var(--up-text);
            border-left: 4px solid var(--up-text);
        }
        .insufficient {
            color: var(--text-secondary);
            font-style: italic;
            font-size: 14px;
        }
        /* Tab styles */
        .tabs {
            display: flex;
            gap: 5px;
            border-bottom: 2px solid var(--table-header-bg);
            margin-bottom: 20px;
        }
        .tab-button {
            padding: 12px 24px;
            background: var(--tab-inactive);
            border: none;
            border-radius: 8px 8px 0 0;
            cursor: pointer;
            font-size: 16px;
            font-weight: 600;
            color: var(--text-tertiary);
            transition: all 0.3s;
        }
        .tab-button:hover {
            background: var(--tab-inactive-hover);
        }
        .tab-button.active {
            background: var(--tab-active);
            color: var(--table-header-text);
        }
        .tab-content {
            display: none;
        }
        .tab-content.active {
            display: block;
        }
    </style>
</head>
<body>
    <div class="header">
        <button class="theme-toggle" onclick="toggleTheme()" id="themeToggle">🌙 Dark</button>
        <h1>📊 Multi-Window Range Percentiles</h1>
        <div class="params">
            <span><strong>Windows:</strong> """ + ", ".join([f"{w}d" for w in window_list]) + """</span>
            <span><strong>Lookback:</strong> """ + str(days) + """ calendar days (""" + str(n_data) + """ trading days)</span>
            <span><strong>Updated:</strong> """ + _date.today().strftime('%Y-%m-%d') + """</span>
        </div>
    </div>
"""
    ]

    # Add tabs if multiple tickers
    if multi_ticker:
        html_parts.append('<div class="tabs">\n')
        for i, res in enumerate(results):
            active_class = "active" if i == 0 else ""
            ticker = res["ticker"]
            html_parts.append(f'        <button class="tab-button {active_class}" onclick="showTab(\'{ticker}\')">{ticker}</button>\n')
        html_parts.append('    </div>\n')

    # Generate content for each ticker
    chart_data_all = {}
    for result_item in results:
        ticker = result_item["ticker"]
        ticker_id = ticker.replace(":", "_").replace(".", "_")
        active_class = "active" if result_item == results[0] else ""

        # Generate content using helper function
        content_html, chart_down, chart_up = _generate_ticker_content_html(result_item, ticker_id)

        # Store chart data
        chart_data_all[ticker_id] = {
            "down": chart_down,
            "up": chart_up
        }

        # Wrap in tab content if multi-ticker
        if multi_ticker:
            html_parts.append(f'    <div class="tab-content {active_class}" id="tab_{ticker}">\n')

        html_parts.append(content_html)

        if multi_ticker:
            html_parts.append('    </div>\n')

    # Add Chart.js initialization scripts
    chart_data_json = json.dumps(chart_data_all)

    html_parts.append(f"""
    <script>
        // Chart data for all tickers
        const chartDataAll = {chart_data_json};

        // Helper function to convert percent to price
        function percentToPrice(pct, referenceClose) {{
            return referenceClose * (1 + pct / 100);
        }}

        // Initialize all charts
        Object.keys(chartDataAll).forEach(function(tickerId) {{
            const chartDataDown = chartDataAll[tickerId].down;
            const chartDataUp = chartDataAll[tickerId].up;

            // Initialize DOWN chart
            new Chart(document.getElementById('chartDown_' + tickerId), {{
                type: 'line',
                data: chartDataDown,
                options: {{
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: {{
                        legend: {{
                            position: 'top',
                        }},
                        tooltip: {{
                            mode: 'index',
                            intersect: false,
                            callbacks: {{
                                label: function(context) {{
                                    const pct = context.parsed.y;
                                    const price = percentToPrice(pct, chartDataDown.priceRange.referenceClose);
                                    return context.dataset.label + ': ' + pct.toFixed(2) + '% ($' + price.toFixed(2) + ')';
                                }}
                            }}
                        }}
                    }},
                    scales: {{
                        x: {{
                            title: {{
                                display: true,
                                text: 'Window Size (trading days)'
                            }}
                        }},
                        yPercent: {{
                            type: 'linear',
                            position: 'left',
                            title: {{
                                display: true,
                                text: 'Percentile Value (%)'
                            }},
                            ticks: {{
                                callback: function(value) {{
                                    return value.toFixed(2) + '%';
                                }}
                            }}
                        }},
                        yPrice: {{
                            type: 'linear',
                            position: 'right',
                            title: {{
                                display: true,
                                text: 'Price ($)'
                            }},
                            min: chartDataDown.priceRange.min,
                            max: chartDataDown.priceRange.max,
                            grid: {{
                                drawOnChartArea: false
                            }},
                            ticks: {{
                                callback: function(value) {{
                                    return '$' + value.toFixed(2);
                                }}
                            }}
                        }}
                    }},
                    interaction: {{
                        mode: 'nearest',
                        axis: 'x',
                        intersect: false
                    }}
                }}
            }});

            // Initialize UP chart
            new Chart(document.getElementById('chartUp_' + tickerId), {{
                type: 'line',
                data: chartDataUp,
                options: {{
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: {{
                        legend: {{
                            position: 'top',
                        }},
                        tooltip: {{
                            mode: 'index',
                            intersect: false,
                            callbacks: {{
                                label: function(context) {{
                                    const pct = context.parsed.y;
                                    const price = percentToPrice(pct, chartDataUp.priceRange.referenceClose);
                                    return context.dataset.label + ': ' + pct.toFixed(2) + '% ($' + price.toFixed(2) + ')';
                                }}
                            }}
                        }}
                    }},
                    scales: {{
                        x: {{
                            title: {{
                                display: true,
                                text: 'Window Size (trading days)'
                            }}
                        }},
                        yPercent: {{
                            type: 'linear',
                            position: 'left',
                            title: {{
                                display: true,
                                text: 'Percentile Value (%)'
                            }},
                            ticks: {{
                                callback: function(value) {{
                                    return value.toFixed(2) + '%';
                                }}
                            }}
                        }},
                        yPrice: {{
                            type: 'linear',
                            position: 'right',
                            title: {{
                                display: true,
                                text: 'Price ($)'
                            }},
                            min: chartDataUp.priceRange.min,
                            max: chartDataUp.priceRange.max,
                            grid: {{
                                drawOnChartArea: false
                            }},
                            ticks: {{
                                callback: function(value) {{
                                    return '$' + value.toFixed(2);
                                }}
                            }}
                        }}
                    }},
                    interaction: {{
                        mode: 'nearest',
                        axis: 'x',
                        intersect: false
                    }}
                }}
            }});
        }});

        // Tab switching function
        function showTab(ticker) {{
            // Hide all tab contents
            const allContents = document.querySelectorAll('.tab-content');
            allContents.forEach(function(content) {{
                content.classList.remove('active');
            }});

            // Deactivate all tab buttons
            const allButtons = document.querySelectorAll('.tab-button');
            allButtons.forEach(function(button) {{
                button.classList.remove('active');
            }});

            // Show selected tab
            const selectedTab = document.getElementById('tab_' + ticker);
            if (selectedTab) {{
                selectedTab.classList.add('active');
            }}

            // Activate selected button
            const buttons = document.querySelectorAll('.tab-button');
            buttons.forEach(function(button) {{
                if (button.textContent === ticker) {{
                    button.classList.add('active');
                }}
            }});
        }}

        // Theme switching function
        function toggleTheme() {{
            const html = document.documentElement;
            const currentTheme = html.getAttribute('data-theme');
            const newTheme = currentTheme === 'dark' ? 'light' : 'dark';
            html.setAttribute('data-theme', newTheme);
            updateThemeButton(newTheme);

            // Save preference
            localStorage.setItem('theme', newTheme);
        }}

        function updateThemeButton(theme) {{
            const button = document.getElementById('themeToggle');
            if (theme === 'dark') {{
                button.textContent = '☀️ Light';
            }} else {{
                button.textContent = '🌙 Dark';
            }}
        }}

        function initTheme() {{
            // Check URL parameter first
            const urlParams = new URLSearchParams(window.location.search);
            const urlTheme = urlParams.get('theme');

            if (urlTheme === 'dark' || urlTheme === 'light') {{
                document.documentElement.setAttribute('data-theme', urlTheme);
                updateThemeButton(urlTheme);
                localStorage.setItem('theme', urlTheme);
                return;
            }}

            // Check localStorage
            const savedTheme = localStorage.getItem('theme');
            if (savedTheme) {{
                document.documentElement.setAttribute('data-theme', savedTheme);
                updateThemeButton(savedTheme);
                return;
            }}

            // Check system preference
            if (window.matchMedia && window.matchMedia('(prefers-color-scheme: dark)').matches) {{
                document.documentElement.setAttribute('data-theme', 'dark');
                updateThemeButton('dark');
            }} else {{
                document.documentElement.setAttribute('data-theme', 'light');
                updateThemeButton('light');
            }}
        }}

        // Initialize theme on page load
        initTheme();

        // Listen for system theme changes
        if (window.matchMedia) {{
            window.matchMedia('(prefers-color-scheme: dark)').addEventListener('change', function(e) {{
                // Only auto-switch if user hasn't manually set a preference
                if (!localStorage.getItem('theme')) {{
                    const newTheme = e.matches ? 'dark' : 'light';
                    document.documentElement.setAttribute('data-theme', newTheme);
                    updateThemeButton(newTheme);
                }}
            }});
        }}
    </script>
</body>
</html>""")

    return "".join(html_parts)
