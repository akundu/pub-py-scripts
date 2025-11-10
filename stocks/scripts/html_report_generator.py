#!/usr/bin/env python3
"""
HTML Report Generator - Generate HTML reports with sortable tables for covered calls analysis.

This module handles the generation of HTML output with embedded CSS and JavaScript
for displaying and sorting tabular data.
"""

import pandas as pd
import sys
from pathlib import Path
from datetime import datetime


def format_numeric_value(x, col_name):
    """Format numeric value based on column type.
    
    Args:
        x: Value to format
        col_name: Name of the column
        
    Returns:
        Formatted string value
    """
    if pd.isna(x) or x == '' or x is None:
        return ''
    try:
        val = float(x)
        if 'premium' in col_name.lower() or 'price' in col_name.lower() or 'cap' in col_name.lower():
            return f"${val:,.2f}"
        elif 'ratio' in col_name.lower() or 'delta' in col_name.lower() or 'theta' in col_name.lower():
            return f"{val:.2f}"
        elif 'days' in col_name.lower() or 'volume' in col_name.lower() or 'contracts' in col_name.lower():
            return f"{int(val):,}"
        elif 'percentage' in col_name.lower():
            return f"{val:.2f}%"
        else:
            return f"{val:.2f}"
    except (ValueError, TypeError):
        return str(x) if x != '' else ''


def truncate_header(text, max_length=15):
    """Truncate header text to max_length characters, breaking at word boundaries when possible.
    
    Args:
        text: Header text to truncate
        max_length: Maximum length per line
        
    Returns:
        Truncated text with <br> tags for line breaks
    """
    if len(text) <= max_length:
        return text
    
    # Try to break at word boundaries
    words = text.split()
    if len(words) > 1:
        result = []
        current_line = ""
        for word in words:
            if len(current_line) + len(word) + 1 <= max_length:
                if current_line:
                    current_line += " " + word
                else:
                    current_line = word
            else:
                if current_line:
                    result.append(current_line)
                current_line = word
        if current_line:
            result.append(current_line)
        return "<br>".join(result)
    else:
        # Single word, just truncate
        return text[:max_length]


def get_css_styles():
    """Get CSS styles for the HTML report.
    
    Returns:
        String containing CSS styles
    """
    return """        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 20px;
            min-height: 100vh;
        }
        
        .container {
            max-width: 95%;
            margin: 0 auto;
            background: white;
            border-radius: 12px;
            box-shadow: 0 20px 60px rgba(0,0,0,0.3);
            overflow: hidden;
        }
        
        .header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 30px;
            text-align: center;
        }
        
        .header h1 {
            font-size: 2.5em;
            margin-bottom: 10px;
        }
        
        .header p {
            font-size: 1.1em;
            opacity: 0.9;
        }
        
        .table-wrapper {
            overflow-x: auto;
            padding: 20px;
        }
        
        table {
            width: 100%;
            border-collapse: collapse;
            font-size: 0.9em;
        }
        
        thead {
            background: #f8f9fa;
            position: sticky;
            top: 0;
            z-index: 10;
        }
        
        th {
            padding: 15px 12px;
            text-align: left;
            font-weight: 600;
            color: #333;
            border-bottom: 2px solid #dee2e6;
            cursor: pointer;
            user-select: none;
            white-space: normal;
            word-wrap: break-word;
            word-break: break-word;
            max-width: 120px;
            line-height: 1.3;
            position: relative;
        }
        
        th:hover {
            background: #e9ecef;
        }
        
        th.sortable::after {
            content: ' ↕';
            opacity: 0.5;
            font-size: 0.8em;
        }
        
        th.sort-asc::after {
            content: ' ↑';
            opacity: 1;
            color: #667eea;
        }
        
        th.sort-desc::after {
            content: ' ↓';
            opacity: 1;
            color: #667eea;
        }
        
        td {
            padding: 12px;
            border-bottom: 1px solid #dee2e6;
            color: #495057;
        }
        
        tbody tr:hover {
            background: #f8f9fa;
        }
        
        tbody tr:nth-child(even) {
            background: #f8f9fa;
        }
        
        tbody tr:nth-child(even):hover {
            background: #e9ecef;
        }
        
        .stats {
            padding: 20px;
            background: #f8f9fa;
            border-top: 1px solid #dee2e6;
            display: flex;
            justify-content: space-around;
            flex-wrap: wrap;
        }
        
        .stat-item {
            text-align: center;
            padding: 10px;
        }
        
        .stat-value {
            font-size: 1.5em;
            font-weight: bold;
            color: #667eea;
        }
        
        .stat-label {
            font-size: 0.9em;
            color: #6c757d;
            margin-top: 5px;
        }
        
        @media (max-width: 768px) {
            .header h1 {
                font-size: 1.8em;
            }
            
            table {
                font-size: 0.8em;
            }
            
            th, td {
                padding: 8px 6px;
            }
        }
"""


def get_javascript():
    """Get JavaScript code for table sorting functionality.
    
    Returns:
        String containing JavaScript code
    """
    return """        let sortDirection = {};
        let currentSortColumn = -1;
        
        function sortTable(columnIndex) {
            const table = document.getElementById('resultsTable');
            const tbody = table.querySelector('tbody');
            const rows = Array.from(tbody.querySelectorAll('tr'));
            const headers = table.querySelectorAll('th');
            
            // Remove sort classes from all headers
            headers.forEach(h => {
                h.classList.remove('sort-asc', 'sort-desc');
            });
            
            // Determine sort direction
            if (currentSortColumn === columnIndex) {
                sortDirection[columnIndex] = sortDirection[columnIndex] === 'asc' ? 'desc' : 'asc';
            } else {
                sortDirection[columnIndex] = 'asc';
                currentSortColumn = columnIndex;
            }
            
            // Add sort class to current header
            headers[columnIndex].classList.add(sortDirection[columnIndex] === 'asc' ? 'sort-asc' : 'sort-desc');
            
            // Sort rows
            rows.sort((a, b) => {
                const aText = a.cells[columnIndex].textContent.trim();
                const bText = b.cells[columnIndex].textContent.trim();
                
                // Try to parse as number (handles currency, percentages, etc.)
                const aNum = parseFloat(aText.replace(/[^0-9.-]/g, ''));
                const bNum = parseFloat(bText.replace(/[^0-9.-]/g, ''));
                
                let comparison = 0;
                
                if (!isNaN(aNum) && !isNaN(bNum)) {
                    // Both are numbers
                    comparison = aNum - bNum;
                } else {
                    // String comparison
                    comparison = aText.localeCompare(bText);
                }
                
                return sortDirection[columnIndex] === 'asc' ? comparison : -comparison;
            });
            
            // Re-append sorted rows
            rows.forEach(row => tbody.appendChild(row));
            
            // Update visible count
            document.getElementById('visibleCount').textContent = rows.length;
        }
        
        // Initialize - sort by net_daily_premium if available
        document.addEventListener('DOMContentLoaded', function() {
            const headers = document.querySelectorAll('th');
            headers.forEach((header, index) => {
                const headerText = header.textContent.toLowerCase().replace(/\\s+/g, ' ');
                if (headerText.includes('net') && headerText.includes('daily') && headerText.includes('premium')) {
                    sortTable(index);
                }
            });
        });
"""


def generate_html_output(df: pd.DataFrame, output_dir: str) -> None:
    """Generate HTML output with sortable table.
    
    Args:
        df: DataFrame with the results
        output_dir: Directory path where to create the HTML output
    """
    # Create output directory if it doesn't exist
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Prepare data for HTML table
    df_display = df.copy()
    
    # Format numeric columns for better display
    numeric_cols = [
        'pe_ratio', 'market_cap_b', 'current_price', 'strike_price', 'price_above_current',
        'option_premium', 'option_premium_percentage', 'delta', 'theta', 'days_to_expiry',
        'short_premium_total', 'short_daily_premium', 'long_strike_price', 'long_option_premium',
        'long_delta', 'long_theta', 'long_days_to_expiry', 'long_premium_total', 'premium_diff',
        'net_premium', 'net_daily_premium', 'volume', 'num_contracts'
    ]
    
    for col in numeric_cols:
        if col in df_display.columns:
            df_display[col] = df_display[col].apply(lambda x: format_numeric_value(x, col))
    
    # Replace remaining NaN with empty strings for display
    df_display = df_display.fillna('')
    
    # Get current timestamp for display
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Generate HTML - build it piece by piece
    html_parts = []
    
    # HTML head and styles
    html_parts.append("""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Covered Calls Analysis Results</title>
    <style>
""")
    html_parts.append(get_css_styles())
    html_parts.append("""    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>📊 Covered Calls Analysis Results</h1>
            <p>Generated: """ + timestamp + """</p>
            <p>Click column headers to sort • """ + str(len(df)) + """ total results</p>
        </div>
        
        <div class="table-wrapper">
            <table id="resultsTable">
                <thead>
                    <tr>
""")
    
    # Generate table headers
    for col in df_display.columns:
        col_index = df_display.columns.get_loc(col)
        col_title = col.replace("_", " ").title()
        truncated_title = truncate_header(col_title, 15)
        html_parts.append(f'                        <th class="sortable" onclick="sortTable({col_index})">{truncated_title}</th>\n')
    
    html_parts.append("""                    </tr>
                </thead>
                <tbody>
""")
    
    # Generate table rows
    for _, row in df_display.iterrows():
        html_parts.append("                    <tr>\n")
        for col in df_display.columns:
            cell_value = str(row[col]) if pd.notna(row[col]) else ''
            # Escape HTML special characters
            cell_value = cell_value.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
            html_parts.append(f'                        <td>{cell_value}</td>\n')
        html_parts.append("                    </tr>\n")
    
    html_parts.append("""                </tbody>
            </table>
        </div>
        
        <div class="stats">
            <div class="stat-item">
                <div class="stat-value" id="totalCount">""" + str(len(df)) + """</div>
                <div class="stat-label">Total Results</div>
            </div>
            <div class="stat-item">
                <div class="stat-value" id="visibleCount">""" + str(len(df)) + """</div>
                <div class="stat-label">Visible Rows</div>
            </div>
        </div>
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

