#!/usr/bin/env python3
"""
HTML Report Generator - Generate HTML reports with sortable tables for covered calls analysis.

This module handles the generation of HTML output with embedded CSS and JavaScript
for displaying and sorting tabular data.
"""

import pandas as pd
import sys
import textwrap
import html
from pathlib import Path
from datetime import datetime


COMPACT_HEADER_MAP = {
    'ticker': 'ticker',
    'price': 'price',
    'P/E': 'P/E',
    'MKT_CAP': 'MKT_CAP',
    'MKT_B': 'MKT_B',
    'STRK': 'STRK',
    'price_above_current': 'current_strike_diff',
    'option_premium': 'opt_premium',
    'bid_ask': 'bid:ask',
    'option_premium_percentage': 'opt_premium%',
    'premium_above_diff_percentage': 'DIFF%',
    'implied_volatility': 'IV',
    'delta': 'DEL',
    'theta': 'TH',
    'volume': 'VOL',
    'num_contracts': 'CNT',
    'potential_premium': 'POT_PREM',
    'daily_premium': 'DAILY_PREM',
    'expiration_date': 'EXP (UTC)',
    'days_to_expiry': 'DAYS',
    'last_quote_timestamp': 'LQUOTE_TS',
    'write_timestamp': 'WRITE_TS (EST)',
    'option_ticker': 'OPT_TKR',
    'long_strike_price': 'L_STRK',
    'long_option_premium': 'L_PREM',
    'long_bid_ask': 'l_bid:ask',
    'long_expiration_date': 'L_EXP',
    'long_days_to_expiry': 'L_DAYS',
    'long_option_ticker': 'L_OPT_TKR',
    'long_delta': 'L_DEL',
    'long_theta': 'L_TH',
    'long_implied_volatility': 'LIV',
    'long_volume': 'L_VOL',
    'long_contracts_available': 'L_CNT_AVL',
    'premium_diff': 'PREM_DIFF',
    'short_premium_total': 'S_PREM_TOT',
    'short_daily_premium': 'S_DAY_PREM',
    'long_premium_total': 'L_PREM_TOT',
    'NET_PREM': 'NET_PREM',
    'NET_DAY': 'NET_DAY'
}


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
        elif 'days' in col_name.lower() or 'volume' in col_name.lower() or 'contracts' in col_name.lower() or 'options' in col_name.lower() or 'purchase' in col_name.lower():
            return f"{int(val):,}"
        elif 'percentage' in col_name.lower():
            return f"{val:.2f}%"
        else:
            return f"{val:.2f}"
    except (ValueError, TypeError):
        return str(x) if x != '' else ''


def truncate_header(text, max_length=15):
    """Wrap header text so each line is at most max_length characters.
    
    Args:
        text: Header text to wrap
        max_length: Maximum length per line
        
    Returns:
        Wrapped text with <br> tags for line breaks
    """
    text = text.replace("_", " ")
    wrapped_lines = textwrap.wrap(
        text,
        width=max_length,
        break_long_words=True,
        break_on_hyphens=False
    )
    return "<br>".join(wrapped_lines) if wrapped_lines else text


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
        
        .tabs {
            display: flex;
            justify-content: center;
            gap: 10px;
            margin-bottom: 20px;
            flex-wrap: wrap;
        }
        
        .tab-button {
            padding: 12px 24px;
            background: rgba(255, 255, 255, 0.2);
            color: white;
            border: 2px solid rgba(255, 255, 255, 0.3);
            border-radius: 8px;
            cursor: pointer;
            font-size: 1em;
            font-weight: 600;
            transition: all 0.3s ease;
            user-select: none;
        }
        
        .tab-button:hover {
            background: rgba(255, 255, 255, 0.3);
            border-color: rgba(255, 255, 255, 0.5);
        }
        
        .tab-button.active {
            background: white;
            color: #667eea;
            border-color: white;
        }
        
        .tab-content {
            display: none;
        }
        
        .tab-content.active {
            display: block;
        }
        
        .table-wrapper {
            overflow-x: auto;
            padding: 20px;
        }
        
        table {
            width: 100%;
            border-collapse: collapse;
            font-size: 0.9em;
            table-layout: fixed;
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
            max-width: 15ch;
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
            max-width: 15ch;
            white-space: normal;
            word-wrap: break-word;
            word-break: break-word;
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
        
        .detailed-analysis {
            padding: 30px;
            background: #f8f9fa;
            border-top: 2px solid #dee2e6;
        }
        
        .detailed-analysis h2 {
            color: #667eea;
            font-size: 2em;
            margin-bottom: 20px;
            text-align: center;
            border-bottom: 3px solid #667eea;
            padding-bottom: 10px;
        }
        
        .analysis-item {
            background: white;
            border-radius: 8px;
            padding: 25px;
            margin-bottom: 25px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
            border-left: 4px solid #667eea;
        }
        
        .analysis-item h3 {
            color: #667eea;
            font-size: 1.5em;
            margin-bottom: 15px;
            border-bottom: 2px solid #e9ecef;
            padding-bottom: 10px;
        }
        
        .analysis-section {
            margin-bottom: 20px;
        }
        
        .analysis-section h4 {
            color: #495057;
            font-size: 1.2em;
            margin-bottom: 10px;
            margin-top: 15px;
        }
        
        .analysis-section p {
            margin: 8px 0;
            line-height: 1.6;
            color: #495057;
        }
        
        .analysis-section .label {
            font-weight: 600;
            color: #333;
        }
        
        .option-tickers {
            background: #f8f9fa;
            padding: 15px;
            border-radius: 5px;
            margin: 10px 0;
            font-family: 'Courier New', monospace;
        }
        
        .option-tickers .short {
            color: #dc3545;
        }
        
        .option-tickers .long {
            color: #28a745;
        }
        
        .risk-badge {
            display: inline-block;
            padding: 5px 12px;
            border-radius: 15px;
            font-weight: 600;
            font-size: 0.9em;
        }
        
        .risk-low {
            background: #d4edda;
            color: #155724;
        }
        
        .risk-moderate {
            background: #fff3cd;
            color: #856404;
        }
        
        .risk-high {
            background: #f8d7da;
            color: #721c24;
        }
        
        .score-badge {
            display: inline-block;
            padding: 8px 15px;
            border-radius: 20px;
            font-weight: bold;
            font-size: 1.1em;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
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
        
        // Initialize - preserve original CSV order (no auto-sort)
        // Users can click column headers to sort if desired
        
        // Tab switching functionality
        function switchTab(tabIndex) {
            // Hide all tab contents
            const tabContents = document.querySelectorAll('.tab-content');
            tabContents.forEach(content => content.classList.remove('active'));
            
            // Remove active class from all tab buttons
            const tabButtons = document.querySelectorAll('.tab-button');
            tabButtons.forEach(button => button.classList.remove('active'));
            
            // Show selected tab content
            tabContents[tabIndex].classList.add('active');
            tabButtons[tabIndex].classList.add('active');
        }
        
        // Initialize first tab as active
        document.addEventListener('DOMContentLoaded', function() {
            switchTab(0);
        });
"""


def generate_summary_statistics_html(df: pd.DataFrame) -> str:
    """Generate HTML for summary statistics section.
    
    Args:
        df: DataFrame with the results
        
    Returns:
        String containing HTML for summary statistics
    """
    # Check if required columns exist
    if 'spread_impact_pct' not in df.columns:
        return ''
    
    html_parts = []
    html_parts.append('            <div class="analysis-item">\n')
    html_parts.append('                <h3>📊 SUMMARY STATISTICS</h3>\n')
    
    # Spread Impact Statistics
    spread_impact_mean = df['spread_impact_pct'].mean()
    spread_impact_median = df['spread_impact_pct'].median()
    low_impact_count = (df['spread_impact_pct'] < 5).sum()
    high_impact_count = (df['spread_impact_pct'] > 10).sum()
    
    html_parts.append('                <div class="analysis-section">\n')
    html_parts.append('                    <h4>Spread Impact Statistics</h4>\n')
    html_parts.append(f'                    <p><span class="label">Average spread impact:</span> {spread_impact_mean:.2f}%</p>\n')
    html_parts.append(f'                    <p><span class="label">Median spread impact:</span> {spread_impact_median:.2f}%</p>\n')
    html_parts.append(f'                    <p><span class="label">Trades with &lt;5% impact:</span> {low_impact_count}/{len(df)}</p>\n')
    html_parts.append(f'                    <p><span class="label">Trades with &gt;10% impact:</span> <span class="risk-badge risk-high">{high_impact_count}/{len(df)} ⚠️</span></p>\n')
    html_parts.append('                </div>\n')
    
    # Liquidity Distribution
    if 'liquidity_score' in df.columns:
        high_liquidity = (df['liquidity_score'] >= 7).sum()
        medium_liquidity = ((df['liquidity_score'] >= 4) & (df['liquidity_score'] < 7)).sum()
        low_liquidity = (df['liquidity_score'] < 4).sum()
        
        html_parts.append('                <div class="analysis-section">\n')
        html_parts.append('                    <h4>Liquidity Distribution</h4>\n')
        html_parts.append(f'                    <p><span class="label">High liquidity (7-10):</span> {high_liquidity} trades</p>\n')
        html_parts.append(f'                    <p><span class="label">Medium liquidity (4-6):</span> {medium_liquidity} trades</p>\n')
        html_parts.append(f'                    <p><span class="label">Low liquidity (&lt;4):</span> <span class="risk-badge risk-high">{low_liquidity} trades ⚠️</span></p>\n')
        html_parts.append('                </div>\n')
    
    html_parts.append('            </div>\n')
    
    return ''.join(html_parts)


def generate_detailed_analysis_html(df: pd.DataFrame) -> str:
    """Generate HTML for unified comprehensive analysis section.
    
    Args:
        df: DataFrame with the results
        
    Returns:
        String containing HTML for comprehensive analysis
    """
    top_10 = df.nlargest(10, 'net_daily_premi')
    
    html_parts = []
    html_parts.append('        <div class="detailed-analysis">\n')
    html_parts.append('            <h2>📊 COMPREHENSIVE ANALYSIS: TOP 10 PICKS WITH OPTION TICKERS</h2>\n')
    
    # Add summary statistics at the top
    html_parts.append(generate_summary_statistics_html(df))
    
    for idx, (row_idx, row) in enumerate(top_10.iterrows(), 1):
        ticker = row['ticker']
        
        # Calculate values
        moneyness = ((row['strike_price'] - row['curr_price']) / row['curr_price'] * 100) if pd.notna(row['strike_price']) and pd.notna(row['curr_price']) else 0
        spread_width = row['l_strike'] - row['strike_price'] if pd.notna(row['l_strike']) and pd.notna(row['strike_price']) else 0
        delta_diff = row['l_delta'] - row['delta'] if pd.notna(row['l_delta']) and pd.notna(row['delta']) else 0
        roi = (row['net_premium'] / 100000 * 100) if pd.notna(row['net_premium']) else 0
        
        # Calculate score
        score = 0
        if pd.notna(row['net_daily_premi']):
            if row['net_daily_premi'] > 10000: score += 3
            elif row['net_daily_premi'] > 7000: score += 2
            else: score += 1
        
        if pd.notna(row['volume']):
            if row['volume'] > 1000: score += 3
            elif row['volume'] > 300: score += 2
            elif row['volume'] > 100: score += 1
        
        if pd.notna(row['delta']):
            if row['delta'] < 0.35: score += 3
            elif row['delta'] < 0.50: score += 2
            else: score += 1
        
        if pd.notna(row['pe_ratio']):
            if row['pe_ratio'] < 25: score += 2
            elif row['pe_ratio'] < 50: score += 1
        
        # Determine recommendation
        if score >= 9:
            recommendation = "STRONG BUY - Excellent risk/reward"
        elif score >= 7:
            recommendation = "BUY - Good opportunity"
        elif score >= 5:
            recommendation = "HOLD - Acceptable but monitor"
        else:
            recommendation = "PASS - Better opportunities available"
        
        # Assignment risk
        if pd.notna(row['delta']):
            if row['delta'] < 0.35:
                assignment_risk = "LOW - Strike is well OTM"
                risk_class = "risk-low"
            elif row['delta'] < 0.50:
                assignment_risk = "MODERATE - Near ATM, watch closely"
                risk_class = "risk-moderate"
            else:
                assignment_risk = "HIGH - ITM or very close, likely assignment"
                risk_class = "risk-high"
        else:
            assignment_risk = "UNKNOWN"
            risk_class = "risk-moderate"
        
        # Liquidity
        if pd.notna(row['volume']):
            if row['volume'] > 1000:
                liquidity = "EXCELLENT - Very liquid"
            elif row['volume'] > 300:
                liquidity = "GOOD - Adequate liquidity"
            elif row['volume'] > 100:
                liquidity = "FAIR - May have wider spreads"
            else:
                liquidity = "POOR - Low liquidity, watch bid-ask"
        else:
            liquidity = "UNKNOWN"
        
        # Valuation
        if pd.notna(row['pe_ratio']):
            if row['pe_ratio'] < 15:
                valuation = "ATTRACTIVE - Trading at discount"
            elif row['pe_ratio'] < 25:
                valuation = "FAIR - Reasonably valued"
            elif row['pe_ratio'] < 50:
                valuation = "ELEVATED - Premium valuation"
            else:
                valuation = "EXPENSIVE - Very high P/E"
        else:
            valuation = "UNKNOWN"
        
        # Format dates
        exp_date = str(row['expiration_date'])[:10] if pd.notna(row['expiration_date']) else 'N/A'
        l_exp_date = str(row['l_expiration_date'])[:10] if pd.notna(row['l_expiration_date']) else 'N/A'
        
        html_parts.append(f'            <div class="analysis-item">\n')
        html_parts.append(f'                <h3>#{idx}: {ticker}</h3>\n')
        
        # Position Structure
        html_parts.append('                <div class="analysis-section">\n')
        html_parts.append('                    <h4>📊 POSITION STRUCTURE</h4>\n')
        html_parts.append(f'                    <p><span class="label">Current Price:</span> ${row["curr_price"]:.2f}</p>\n')
        html_parts.append(f'                    <p><span class="label">Short Strike:</span> ${row["strike_price"]:.2f} ({moneyness:.2f}% OTM)</p>\n')
        html_parts.append(f'                    <p><span class="label">Long Strike:</span> ${row["l_strike"]:.2f}</p>\n')
        html_parts.append(f'                    <p><span class="label">Spread Width:</span> ${spread_width:.2f}</p>\n')
        html_parts.append('                </div>\n')
        
        # Option Tickers
        html_parts.append('                <div class="analysis-section">\n')
        html_parts.append('                    <h4>🎯 OPTION TICKERS</h4>\n')
        html_parts.append('                    <div class="option-tickers">\n')
        option_ticker_short = html.escape(str(row["option_ticker"])) if pd.notna(row["option_ticker"]) else "N/A"
        option_ticker_long = html.escape(str(row["l_option_ticker"])) if pd.notna(row["l_option_ticker"]) else "N/A"
        bid_ask_short = html.escape(str(row.get("bid:ask", "N/A:N/A")))
        bid_ask_long = html.escape(str(row.get("l_bid:ask", "N/A:N/A")))
        html_parts.append(f'                        <p><span class="short">┌─ SHORT (SELL):</span> {option_ticker_short}</p>\n')
        html_parts.append(f'                        <p>│  Strike: ${row["strike_price"]:.2f} | Expiry: {html.escape(exp_date)} ({int(row["days_to_expiry"]) if pd.notna(row["days_to_expiry"]) else 0} DTE)</p>\n')
        html_parts.append(f'                        <p>│  Premium: ${row["opt_prem."]:.2f} per contract | Bid:Ask: {bid_ask_short}</p>\n')
        html_parts.append(f'                        <p>│  Total Credit: ${row["s_prem_tot"]:,.0f} ({int(row["num_contracts"]) if pd.notna(row["num_contracts"]) else 0} contracts)</p>\n')
        html_parts.append('                        <p>│</p>\n')
        html_parts.append(f'                        <p><span class="long">└─ LONG (BUY):</span> {option_ticker_long}</p>\n')
        html_parts.append(f'                        <p>   Strike: ${row["l_strike"]:.2f} | Expiry: {html.escape(l_exp_date)} ({int(row["l_days_to_expiry"]) if pd.notna(row["l_days_to_expiry"]) else 0} DTE)</p>\n')
        html_parts.append(f'                        <p>   Premium: ${row["l_prem"]:.2f} per contract | Bid:Ask: {bid_ask_long}</p>\n')
        html_parts.append(f'                        <p>   Total Debit: ${row["l_prem_tot"]:,.0f} ({int(row["num_contracts"]) if pd.notna(row["num_contracts"]) else 0} contracts)</p>\n')
        html_parts.append('                    </div>\n')
        html_parts.append('                </div>\n')
        
        # Premium Breakdown
        html_parts.append('                <div class="analysis-section">\n')
        html_parts.append('                    <h4>💰 PREMIUM BREAKDOWN</h4>\n')
        html_parts.append(f'                    <p><span class="label">Short Premium:</span> ${row["s_prem_tot"]:,.0f} (${row["s_day_prem"]:,.0f}/day)</p>\n')
        html_parts.append(f'                    <p><span class="label">Long Premium:</span> ${row["l_prem_tot"]:,.0f}</p>\n')
        html_parts.append(f'                    <p><span class="label">Net Credit:</span> ${row["net_premium"]:,.0f}</p>\n')
        html_parts.append(f'                    <p><span class="label">Daily Income:</span> ${row["net_daily_premi"]:,.2f}</p>\n')
        html_parts.append(f'                    <p><span class="label">ROI on $100k:</span> {roi:.2f}%</p>\n')
        if 'long_options_to_purchase' in row and pd.notna(row.get('long_options_to_purchase')):
            long_options = int(row.get('long_options_to_purchase', 0))
            html_parts.append(f'                    <p><span class="label">Long Options to Purchase:</span> {long_options:,} contracts (based on net premium)</p>\n')
        html_parts.append('                </div>\n')
        
        # Spread & Liquidity Analysis (if available)
        if 'spread_impact_pct' in row and pd.notna(row.get('spread_impact_pct')):
            spread_slippage = row.get('spread_slippage', 0)
            net_after_spread = row.get('net_premium_after_spread', row['net_premium'])
            net_daily_after = row.get('net_daily_premium_after_spread', row['net_daily_premi'])
            spread_impact = row.get('spread_impact_pct', 0)
            liquidity_score = row.get('liquidity_score', 0)
            assignment_risk = row.get('assignment_risk', 0)
            trade_quality = row.get('trade_quality', 0)
            
            html_parts.append('                <div class="analysis-section">\n')
            html_parts.append('                    <h4>💱 SPREAD & LIQUIDITY ANALYSIS</h4>\n')
            html_parts.append(f'                    <p><span class="label">Spread Slippage:</span> ${spread_slippage:,.0f}</p>\n')
            html_parts.append(f'                    <p><span class="label">Net Premium After Spread:</span> ${net_after_spread:,.0f}</p>\n')
            html_parts.append(f'                    <p><span class="label">Daily Income After Spread:</span> ${net_daily_after:,.2f}</p>\n')
            html_parts.append(f'                    <p><span class="label">Spread Impact:</span> {spread_impact:.2f}%</p>\n')
            html_parts.append(f'                    <p><span class="label">Liquidity Score:</span> {liquidity_score:.0f}/10</p>\n')
            html_parts.append(f'                    <p><span class="label">Assignment Risk:</span> {assignment_risk:.0f}/6</p>\n')
            html_parts.append(f'                    <p><span class="label">Trade Quality Score:</span> {trade_quality:.1f}</p>\n')
            html_parts.append('                </div>\n')
        
        # Greeks & Risk
        html_parts.append('                <div class="analysis-section">\n')
        html_parts.append('                    <h4>📈 GREEKS & RISK</h4>\n')
        html_parts.append(f'                    <p><span class="label">Short Delta:</span> {row["delta"]:.2f} | <span class="label">Long Delta:</span> {row["l_delta"]:.2f} | <span class="label">Net:</span> {delta_diff:.3f}</p>\n')
        html_parts.append(f'                    <p><span class="label">Short Theta:</span> {row["theta"]:.2f} | <span class="label">Long Theta:</span> {row["l_theta"]:.2f}</p>\n')
        html_parts.append('                </div>\n')
        
        # Liquidity & Fundamentals
        html_parts.append('                <div class="analysis-section">\n')
        html_parts.append('                    <h4>🔄 LIQUIDITY & FUNDAMENTALS</h4>\n')
        html_parts.append(f'                    <p><span class="label">Volume:</span> {row["volume"]:,.0f} contracts</p>\n')
        html_parts.append(f'                    <p><span class="label">Num Contracts:</span> {row["num_contracts"]:.0f}</p>\n')
        html_parts.append(f'                    <p><span class="label">P/E Ratio:</span> {row["pe_ratio"]:.2f}</p>\n')
        html_parts.append(f'                    <p><span class="label">Market Cap:</span> ${row["market_cap_b"]:.2f}B</p>\n')
        html_parts.append('                </div>\n')
        
        # Risk Assessment
        html_parts.append('                <div class="analysis-section">\n')
        html_parts.append('                    <h4>⚠️ RISK ASSESSMENT</h4>\n')
        html_parts.append(f'                    <p><span class="label">Assignment Risk:</span> <span class="risk-badge {risk_class}">{assignment_risk}</span></p>\n')
        html_parts.append(f'                    <p><span class="label">Liquidity:</span> {liquidity}</p>\n')
        html_parts.append(f'                    <p><span class="label">Valuation:</span> {valuation}</p>\n')
        html_parts.append('                </div>\n')
        
        # Overall Score
        html_parts.append('                <div class="analysis-section">\n')
        html_parts.append(f'                    <p><span class="label">⭐ OVERALL SCORE:</span> <span class="score-badge">{score}/11</span></p>\n')
        html_parts.append(f'                    <p><span class="label">Recommendation:</span> {recommendation}</p>\n')
        html_parts.append('                </div>\n')
        
        html_parts.append('            </div>\n')
    
    html_parts.append('        </div>\n')
    
    return ''.join(html_parts)


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
        'ticker','pe_ratio','market_cap_b','curr_price','strike_price','price_above_curr','opt_prem.','IV','delta','theta','expiration_date','days_to_expiry','s_prem_tot','s_day_prem','l_strike','l_prem','liv','l_delta','l_theta','l_expiration_date','l_days_to_expiry','l_prem_tot','l_cnt_avl','prem_diff','net_premium','net_daily_premi','volume','num_contracts','option_ticker','l_option_ticker',
        'spread_slippage','net_premium_after_spread','net_daily_premium_after_spread','spread_impact_pct','liquidity_score','assignment_risk','trade_quality','long_options_to_purchase'
    ]
    
    for col in numeric_cols:
        if col in df_display.columns:
            df_display[col] = df_display[col].apply(lambda x: format_numeric_value(x, col))
    
    # Replace remaining NaN with empty strings for display
    df_display = df_display.fillna('')
    
    # Apply compact headers to keep column names concise
    # compact_headers = {}
    # for col in df_display.columns:
    #     if col in COMPACT_HEADER_MAP:
    #         compact_headers[col] = COMPACT_HEADER_MAP[col]
    #     else:
    #         compact_headers[col] = col
    # df_display = df_display.rename(columns=compact_headers)
    
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
            <div class="tabs">
                <button class="tab-button active" onclick="switchTab(0)">📋 Data Table</button>
                <button class="tab-button" onclick="switchTab(1)">📊 Comprehensive Analysis</button>
            </div>
            <p>Generated: """ + timestamp + """</p>
            <p>Click column headers to sort • """ + str(len(df)) + """ total results</p>
        </div>
        
        <div class="tab-content active">
        <div class="table-wrapper">
            <table id="resultsTable">
                <thead>
                    <tr>
""")
    
    # Generate table headers
    for col in df_display.columns:
        col_index = df_display.columns.get_loc(col)
        truncated_title = truncate_header(str(col), 15)
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
        
        <div class="tab-content">
""")
    
    # Add comprehensive analysis section in second tab
    html_parts.append(generate_detailed_analysis_html(df))
    
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


