#!/usr/bin/env python3
"""
Build HTML versions of all markdown documentation files.

Converts every .md file in the docs/ directory to a styled HTML file in docs/html/.
Uses a dark theme consistent with the project's backtest report styling.

Usage:
  python docs/build_html.py              # Build all HTML docs
  python docs/build_html.py --watch      # Rebuild on file changes (requires watchdog)
  python docs/build_html.py --clean      # Remove all generated HTML files
  python docs/build_html.py --list       # List all docs and their HTML status

Examples:
  python docs/build_html.py
      Convert all .md files in docs/ to docs/html/*.html

  python docs/build_html.py --clean
      Remove docs/html/ directory

  python docs/build_html.py --list
      Show which .md files have corresponding .html
"""

from __future__ import annotations

import argparse
import re
import shutil
import sys
from pathlib import Path

import markdown

DOCS_DIR = Path(__file__).parent
HTML_DIR = DOCS_DIR / "html"

# Dark-themed CSS matching the project's backtest report style
CSS = """
:root {
    --bg-primary: #0d1117;
    --bg-secondary: #161b22;
    --bg-tertiary: #21262d;
    --text-primary: #e6edf3;
    --text-secondary: #8b949e;
    --text-muted: #6e7681;
    --border: #30363d;
    --accent-blue: #58a6ff;
    --accent-green: #3fb950;
    --accent-red: #f85149;
    --accent-orange: #d29922;
    --accent-purple: #bc8cff;
    --code-bg: #161b22;
    --link: #58a6ff;
}

* { margin: 0; padding: 0; box-sizing: border-box; }

body {
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Noto Sans', Helvetica, Arial, sans-serif;
    background: var(--bg-primary);
    color: var(--text-primary);
    line-height: 1.7;
    padding: 0;
    margin: 0;
}

.container {
    max-width: 960px;
    margin: 0 auto;
    padding: 40px 32px;
}

/* Navigation bar */
.nav-bar {
    background: var(--bg-secondary);
    border-bottom: 1px solid var(--border);
    padding: 12px 32px;
    position: sticky;
    top: 0;
    z-index: 100;
    display: flex;
    align-items: center;
    gap: 24px;
    flex-wrap: wrap;
}

.nav-bar a {
    color: var(--text-secondary);
    text-decoration: none;
    font-size: 14px;
    padding: 4px 8px;
    border-radius: 6px;
    transition: all 0.2s;
}

.nav-bar a:hover {
    color: var(--text-primary);
    background: var(--bg-tertiary);
}

.nav-bar a.active {
    color: var(--accent-blue);
    background: rgba(88, 166, 255, 0.1);
}

.nav-bar .nav-title {
    color: var(--text-primary);
    font-weight: 600;
    font-size: 15px;
    margin-right: 8px;
}

/* Headings */
h1 {
    font-size: 2em;
    font-weight: 600;
    margin: 0 0 16px 0;
    padding-bottom: 12px;
    border-bottom: 1px solid var(--border);
    color: var(--text-primary);
}

h2 {
    font-size: 1.5em;
    font-weight: 600;
    margin: 32px 0 16px 0;
    padding-bottom: 8px;
    border-bottom: 1px solid var(--border);
    color: var(--text-primary);
}

h3 {
    font-size: 1.25em;
    font-weight: 600;
    margin: 24px 0 12px 0;
    color: var(--text-primary);
}

h4 {
    font-size: 1.1em;
    font-weight: 600;
    margin: 20px 0 8px 0;
    color: var(--text-secondary);
}

/* Paragraphs and text */
p {
    margin: 0 0 16px 0;
    color: var(--text-primary);
}

a {
    color: var(--link);
    text-decoration: none;
}

a:hover {
    text-decoration: underline;
}

strong {
    color: var(--text-primary);
    font-weight: 600;
}

em {
    color: var(--text-secondary);
}

hr {
    border: none;
    border-top: 1px solid var(--border);
    margin: 32px 0;
}

/* Lists */
ul, ol {
    margin: 0 0 16px 0;
    padding-left: 28px;
}

li {
    margin: 4px 0;
    color: var(--text-primary);
}

li > ul, li > ol {
    margin: 4px 0 4px 0;
}

/* Code */
code {
    font-family: 'SF Mono', 'Fira Code', 'Fira Mono', Menlo, Consolas, monospace;
    font-size: 0.88em;
    background: var(--code-bg);
    border: 1px solid var(--border);
    border-radius: 6px;
    padding: 2px 6px;
    color: var(--accent-blue);
}

pre {
    background: var(--code-bg);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 16px 20px;
    margin: 0 0 16px 0;
    overflow-x: auto;
    line-height: 1.5;
}

pre code {
    background: none;
    border: none;
    padding: 0;
    font-size: 0.88em;
    color: var(--text-primary);
}

/* Tables */
table {
    width: 100%;
    border-collapse: collapse;
    margin: 0 0 16px 0;
    font-size: 0.95em;
}

thead {
    background: var(--bg-secondary);
}

th {
    text-align: left;
    padding: 10px 14px;
    font-weight: 600;
    color: var(--text-primary);
    border-bottom: 2px solid var(--border);
    white-space: nowrap;
}

td {
    padding: 8px 14px;
    border-bottom: 1px solid var(--border);
    color: var(--text-primary);
}

tr:hover {
    background: var(--bg-secondary);
}

/* Blockquotes */
blockquote {
    border-left: 4px solid var(--accent-blue);
    padding: 8px 16px;
    margin: 0 0 16px 0;
    background: var(--bg-secondary);
    border-radius: 0 8px 8px 0;
    color: var(--text-secondary);
}

blockquote p {
    margin: 0;
    color: var(--text-secondary);
}

/* Definition lists */
dt {
    font-weight: 600;
    color: var(--text-primary);
    margin-top: 12px;
}

dd {
    margin-left: 20px;
    margin-bottom: 8px;
    color: var(--text-secondary);
}

/* Footer */
.footer {
    margin-top: 48px;
    padding-top: 16px;
    border-top: 1px solid var(--border);
    color: var(--text-muted);
    font-size: 0.85em;
    text-align: center;
}

/* Responsive */
@media (max-width: 768px) {
    .container { padding: 20px 16px; }
    .nav-bar { padding: 10px 16px; }
    table { font-size: 0.85em; }
    th, td { padding: 6px 8px; }
}
"""

# Navigation links for all docs
NAV_ITEMS = [
    ("index.html", "Home"),
    ("usage_guide.html", "Usage Guide"),
    ("architecture.html", "Architecture"),
    ("api_reference.html", "API Reference"),
    ("playbook.html", "Playbook"),
    ("providers.html", "Providers"),
    ("configuration.html", "Configuration"),
    ("authentication.html", "Authentication"),
    ("testing.html", "Testing"),
    ("ibkr_setup_guide.html", "IBKR Setup"),
    ("etrade_setup_guide.html", "E*TRADE Setup"),
    ("symbology.html", "Symbology"),
    ("websockets.html", "WebSockets"),
]


def _build_nav(current_file: str) -> str:
    """Build the navigation bar HTML."""
    links = ['<span class="nav-title">UTP Docs</span>']
    for href, label in NAV_ITEMS:
        active = ' class="active"' if href == current_file else ""
        links.append(f'<a href="{href}"{active}>{label}</a>')
    return "\n    ".join(links)


def _md_to_html(md_content: str) -> str:
    """Convert markdown to HTML using the markdown library."""
    extensions = ["tables", "fenced_code", "codehilite", "toc", "attr_list"]
    return markdown.markdown(md_content, extensions=extensions)


def _fix_md_links(html: str) -> str:
    """Convert .md links to .html links for cross-doc navigation."""
    return re.sub(r'href="([^"]+)\.md"', r'href="\1.html"', html)


def _wrap_html(body: str, title: str, current_file: str) -> str:
    """Wrap converted HTML in a full page with nav, CSS, and footer."""
    nav = _build_nav(current_file)
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title} — Universal Trade Platform</title>
    <style>{CSS}</style>
</head>
<body>
    <nav class="nav-bar">
    {nav}
    </nav>
    <div class="container">
    {body}
    <div class="footer">
        Universal Trade Platform Documentation — Generated from Markdown sources in <code>docs/</code>
    </div>
    </div>
</body>
</html>"""


def _title_from_filename(filename: str) -> str:
    """Derive a human-readable title from a filename."""
    name = filename.replace(".md", "").replace("_", " ")
    # Special casing
    titles = {
        "api reference": "API Reference",
        "ibkr setup guide": "IBKR Setup Guide",
        "usage guide": "Usage Guide",
        "websockets": "WebSockets",
    }
    return titles.get(name, name.title())


def build_all() -> list[tuple[str, str]]:
    """Convert all .md files to .html. Returns list of (md_path, html_path)."""
    HTML_DIR.mkdir(exist_ok=True)
    results = []

    md_files = sorted(DOCS_DIR.glob("*.md"))
    if not md_files:
        print("No .md files found in", DOCS_DIR)
        return results

    for md_path in md_files:
        html_filename = md_path.stem + ".html"
        html_path = HTML_DIR / html_filename
        title = _title_from_filename(md_path.name)

        md_content = md_path.read_text(encoding="utf-8")
        body = _md_to_html(md_content)
        body = _fix_md_links(body)
        full_html = _wrap_html(body, title, html_filename)

        html_path.write_text(full_html, encoding="utf-8")
        results.append((str(md_path.name), str(html_path.name)))
        print(f"  [OK] {md_path.name} → html/{html_filename}")

    # Build index page
    _build_index(md_files)
    results.append(("(generated)", "html/index.html"))

    return results


def _build_index(md_files: list[Path]) -> None:
    """Build an index.html landing page linking to all docs."""
    cards = []
    descriptions = {
        "api_reference": "All REST endpoints with request/response schemas, authentication scopes, and example payloads. Includes execution store and trade simulation.",
        "architecture": "System design, persistence model, background tasks, request flows, and extension patterns.",
        "authentication": "API key and OAuth2/JWT authentication flows, scopes, and security best practices.",
        "configuration": "All environment variables with defaults, .env template, and persistence directory layout.",
        "etrade_setup_guide": "Step-by-step E*TRADE API setup: OAuth authorization, token management, sandbox testing, and production deployment.",
        "ibkr_setup_guide": "Step-by-step TWS/IB Gateway connection setup, troubleshooting, and market data configuration.",
        "playbook": "Trade playbook system: YAML instruction format, CLI usage, reconciliation (flush/hard-reset), status dashboard, readiness test.",
        "providers": "BrokerProvider interface, stub and live implementations (IBKR + E*TRADE), ProviderRegistry, and how to add new brokers.",
        "symbology": "Symbol mapping across brokers (UUID, conId, OSI format), OptionContract dataclass.",
        "testing": "498 tests in a single file: fixtures, test class descriptions, testing patterns, and how to add new tests.",
        "usage_guide": "15 common workflows: quotes, option chains, credit spreads, iron condors, equity trades, daemon mode, playbooks, execution history, trade simulation, system reset, and more.",
        "websockets": "Real-time order status streaming via WebSocket, message format, and client examples.",
    }

    for md_path in md_files:
        name = md_path.stem
        title = _title_from_filename(md_path.name)
        desc = descriptions.get(name, "")
        html_file = name + ".html"
        cards.append(f"""
        <div class="card">
            <h3><a href="{html_file}">{title}</a></h3>
            <p>{desc}</p>
        </div>""")

    cards_html = "\n".join(cards)

    index_body = f"""
    <h1>Universal Trade Platform — Documentation</h1>
    <p>Unified multi-broker trading API (FastAPI) supporting Robinhood, E*TRADE, and IBKR.
    Features daemon mode, execution store, trade simulation, conId-based deduplication, and market data streaming.</p>

    <h2>Quick Start</h2>
    <pre><code># Start daemon (recommended)
python utp.py daemon --live

# Run all tests (359 tests)
python -m pytest tests/ -v

# CLI auto-detects daemon
python utp.py portfolio
python utp.py quote SPX NDX
python utp.py executions --live

# Execute a playbook (paper)
python utp.py playbook execute playbooks/example_mixed.yaml --paper</code></pre>

    <h2>Documentation</h2>
    <div class="card-grid">
    {cards_html}
    </div>
    """

    # Extra card-grid CSS
    extra_css = """
    .card-grid {
        display: grid;
        grid-template-columns: repeat(auto-fill, minmax(380px, 1fr));
        gap: 16px;
        margin-top: 16px;
    }
    .card {
        background: var(--bg-secondary);
        border: 1px solid var(--border);
        border-radius: 8px;
        padding: 20px;
        transition: border-color 0.2s;
    }
    .card:hover {
        border-color: var(--accent-blue);
    }
    .card h3 {
        margin: 0 0 8px 0;
        font-size: 1.1em;
    }
    .card h3 a {
        color: var(--accent-blue);
    }
    .card p {
        margin: 0;
        color: var(--text-secondary);
        font-size: 0.92em;
        line-height: 1.5;
    }
    """

    nav = _build_nav("index.html")
    full_html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Documentation — Universal Trade Platform</title>
    <style>{CSS}
    {extra_css}</style>
</head>
<body>
    <nav class="nav-bar">
    {nav}
    </nav>
    <div class="container">
    {index_body}
    <div class="footer">
        Universal Trade Platform Documentation — Generated from Markdown sources in <code>docs/</code><br>
        Rebuild: <code>python docs/build_html.py</code>
    </div>
    </div>
</body>
</html>"""

    (HTML_DIR / "index.html").write_text(full_html, encoding="utf-8")
    print(f"  [OK] index.html (landing page)")


def clean() -> None:
    """Remove all generated HTML files."""
    if HTML_DIR.exists():
        shutil.rmtree(HTML_DIR)
        print(f"  Removed {HTML_DIR}")
    else:
        print(f"  {HTML_DIR} does not exist")


def list_docs() -> None:
    """List all docs and their HTML build status."""
    for md_path in sorted(DOCS_DIR.glob("*.md")):
        html_path = HTML_DIR / (md_path.stem + ".html")
        status = "✓" if html_path.exists() else "✗"
        html_age = ""
        if html_path.exists():
            md_mtime = md_path.stat().st_mtime
            html_mtime = html_path.stat().st_mtime
            if md_mtime > html_mtime:
                html_age = " (STALE — md is newer)"
        print(f"  [{status}] {md_path.name:30s} → html/{md_path.stem}.html{html_age}")


def main():
    parser = argparse.ArgumentParser(
        description="""Build HTML versions of all markdown documentation files.
Converts docs/*.md → docs/html/*.html with dark-themed styling and cross-doc navigation.""",
        epilog="""
Examples:
  %(prog)s              Build all HTML docs
  %(prog)s --clean      Remove docs/html/ directory
  %(prog)s --list       Show build status of all docs
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--clean", action="store_true", help="Remove all generated HTML files")
    parser.add_argument("--list", action="store_true", help="List docs and their HTML build status")

    args = parser.parse_args()

    if args.clean:
        clean()
    elif args.list:
        list_docs()
    else:
        print(f"\nBuilding HTML docs from {DOCS_DIR}/ → {HTML_DIR}/\n")
        results = build_all()
        print(f"\n  Done: {len(results)} files generated in {HTML_DIR}/\n")


if __name__ == "__main__":
    main()
